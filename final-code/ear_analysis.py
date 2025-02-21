import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from instance_normalization import InstanceNormalization
from pyts.classification import (
    BOSSVS,
    SAXVSM,
    TSBF,
    KNeighborsClassifier,
    LearningShapelets,
    TimeSeriesForest,
)
from tensorflow.keras import Layer, Model  # type: ignore
from tensorflow.keras.callbacks import ReduceLROnPlateau  # type: ignore
from tensorflow.keras.config import enable_unsafe_deserialization  # type: ignore
from tensorflow.keras.layers import (  # type: ignore
    LSTM,
    Activation,
    Attention,
    AveragePooling1D,
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling1D,
    Input,
    Lambda,
    MaxPooling1D,
    Multiply,
    PReLU,
    Reshape,
    Softmax,
    add,
    concatenate,
)
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.saving import register_keras_serializable  # type: ignore
from tensorflow.keras.utils import pad_sequences, to_categorical  # type: ignore


class KerasTimeSeriesClassifier:
    def __init__(self, model: Model) -> None:
        self.model = model
        self.callbacks = [
            ReduceLROnPlateau(monitor="loss", factor=0.5, patience=50, min_lr=0.0001)
        ]

    def fit(
        self, X: np.ndarray, y: np.ndarray, epochs: int, batch_size: int  # noqa: N803
    ) -> None:
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size)

    def predict(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        return self.model.predict(X)

    def save(self, path: str) -> None:
        self.model.save(path)


# MARK: LSTM
# https://ieeexplore.ieee.org/document/8141873


class LongShortTermMemory(KerasTimeSeriesClassifier):
    def __init__(self, path: str | None = None, attention: bool = False) -> None:
        self.attention = attention

        self.model = load_model(path) if path else self.build_model()
        super().__init__(self.model)

    def build_model(self) -> Model:
        input = Input(shape=(256, 1))

        x1 = Conv1D(128, 8, padding="same")(input)
        x1 = BatchNormalization()(x1)
        x1 = Activation("relu")(x1)

        x1 = Conv1D(256, 5, padding="same")(x1)
        x1 = BatchNormalization()(x1)
        x1 = Activation("relu")(x1)

        x1 = Conv1D(128, 3, padding="same")(x1)
        x1 = BatchNormalization()(x1)
        x1 = Activation("relu")(x1)

        x1 = GlobalAveragePooling1D()(x1)
        x1 = Reshape((1, 128))(x1)

        @register_keras_serializable(package="Custom", name="TransposeLayer")
        class TransposeLayer(Layer):
            def __init__(self, **kwargs: dict) -> None:
                super(TransposeLayer, self).__init__(**kwargs)

            def call(self, inputs: tf.Tensor) -> tf.Tensor:
                return tf.transpose(inputs, perm=[0, 2, 1])

        x2 = TransposeLayer()(input)

        x2 = LSTM(128, return_sequences=True)(x2)
        if self.attention:
            x2 = Attention()([x2, x2])
        x2 = Dropout(0.2)(x2)

        x = concatenate([x1, x2])
        x = Flatten()(x)
        x = Dense(2, activation="softmax")(x)

        model = Model(inputs=input, outputs=x)
        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )

        return model


# MARK: Deep Neural Network Ensembles for Time Series Classification
# https://arxiv.org/pdf/1903.06602 and https://link.springer.com/article/10.1007/s10618-019-00619-1
# code from https://github.com/hfawaz/dl-4-tsc and https://github.com/hfawaz/ijcnn19ensemble


class MultiLayerPerceptron(KerasTimeSeriesClassifier):
    def __init__(self, path: str | None = None) -> None:
        self.model = load_model(path) if path else self.build_model()
        super().__init__(self.model)

    def build_model(self) -> Model:
        input = Input(shape=(256, 1))

        x = Dropout(0.1)(input)
        x = Dense(500, activation="relu")(x)

        x = Dropout(0.2)(x)
        x = Dense(500, activation="relu")(x)

        x = Dropout(0.3)(x)
        x = GlobalAveragePooling1D()(x)
        x = Dense(2, activation="softmax")(x)

        model = Model(inputs=input, outputs=x)
        model.compile(
            optimizer="adadelta", loss="binary_crossentropy", metrics=["accuracy"]
        )

        return model


class FullyConvolutionalNeuralNetwork(KerasTimeSeriesClassifier):
    def __init__(self, path: str | None = None) -> None:
        self.model = load_model(path) if path else self.build_model()
        super().__init__(self.model)

    def build_model(self) -> Model:
        input = Input(shape=(256, 1))

        x = Conv1D(128, 8, padding="same")(input)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Conv1D(256, 5, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Conv1D(128, 3, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = GlobalAveragePooling1D()(x)
        x = Dense(2, activation="softmax")(x)

        model = Model(inputs=input, outputs=x)
        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )

        return model


class ConvolutionalNeuralNetwork(KerasTimeSeriesClassifier):
    def __init__(self, path: str | None = None) -> None:
        self.model = load_model(path) if path else self.build_model()
        super().__init__(self.model)

    def build_model(self) -> Model:
        input = Input(shape=(256, 1))

        x = Conv1D(6, 7, padding="same", activation="sigmoid")(input)
        x = AveragePooling1D(3)(x)

        x = Conv1D(12, 7, padding="same", activation="sigmoid")(x)
        x = AveragePooling1D(3)(x)

        x = Flatten()(x)
        x = Dense(2, activation="softmax")(x)

        model = Model(inputs=input, outputs=x)
        model.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])

        return model


class ResNet(KerasTimeSeriesClassifier):
    def __init__(self, path: str | None = None) -> None:
        self.model = load_model(path) if path else self.build_model()
        super().__init__(self.model)

    def build_model(self) -> Model:
        n_feature_maps = 64

        input = Input(shape=(256, 1))

        x = Conv1D(n_feature_maps, 8, padding="same")(input)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Conv1D(n_feature_maps, 5, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Conv1D(n_feature_maps, 3, padding="same")(x)
        x = BatchNormalization()(x)

        y = Conv1D(n_feature_maps, 1, padding="same")(input)
        y = BatchNormalization()(y)

        block1 = add([x, y])
        block1 = Activation("relu")(block1)

        x = Conv1D(n_feature_maps * 2, 8, padding="same")(block1)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Conv1D(n_feature_maps * 2, 5, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Conv1D(n_feature_maps * 2, 3, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        y = Conv1D(n_feature_maps * 2, 1, padding="same")(block1)
        y = BatchNormalization()(y)

        block2 = add([x, y])
        block2 = Activation("relu")(block2)

        x = Conv1D(n_feature_maps * 2, 8, padding="same")(block2)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Conv1D(n_feature_maps * 2, 5, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Conv1D(n_feature_maps * 2, 3, padding="same")(x)
        x = BatchNormalization()(x)

        y = BatchNormalization()(block2)

        block3 = add([x, y])
        block3 = Activation("relu")(block3)

        x = GlobalAveragePooling1D()(block3)
        x = Dense(2, activation="softmax")(x)

        model = Model(inputs=input, outputs=x)
        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )

        return model


class Encoder(KerasTimeSeriesClassifier):
    def __init__(self, path: str | None = None) -> None:
        self.model = load_model(path) if path else self.build_model()
        super().__init__(self.model)

    def build_model(self) -> Model:
        input = Input(shape=(256, 1))

        x = Conv1D(128, 5, padding="same")(input)
        x = InstanceNormalization()(x)
        x = PReLU(shared_axes=[1])(x)
        x = Dropout(0.2)(x)
        x = MaxPooling1D(2)(x)

        x = Conv1D(256, 11, padding="same")(x)
        x = InstanceNormalization()(x)
        x = PReLU(shared_axes=[1])(x)
        x = Dropout(0.2)(x)
        x = MaxPooling1D(2)(x)

        x = Conv1D(512, 21, padding="same")(x)
        x = InstanceNormalization()(x)
        x = PReLU(shared_axes=[1])(x)
        x = Dropout(0.2)(x)

        def attention_data_slice(x: tf.Tensor) -> tf.Tensor:
            return x[:, :, :256]  # type: ignore

        def attention_softmax_slice(x: tf.Tensor) -> tf.Tensor:
            return x[:, :, 256:]  # type: ignore

        attention_data = Lambda(
            attention_data_slice, output_shape=lambda s: (s[1], 256)
        )(x)
        attention_softmax = Lambda(
            attention_softmax_slice, output_shape=lambda s: (s[1], 256)
        )(x)

        atttention_softmax = Softmax()(attention_softmax)
        x = Multiply()([attention_data, atttention_softmax])

        x = Dense(256, activation="sigmoid")(x)
        x = InstanceNormalization()(x)

        x = Flatten()(x)
        x = Dense(2, activation="softmax")(x)

        model = Model(inputs=input, outputs=x)
        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )

        return model


# MARK: Classical Methods


def generate_datasets() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path_to_faceforensics = "/dcs/large/u2204489/faceforensics/"
    path_to_train = path_to_faceforensics + "train/"
    path_to_train_real = path_to_train + "real/"
    path_to_train_fake = path_to_train + "fake/"
    path_to_test = path_to_faceforensics + "test/"
    path_to_test_real = path_to_test + "real/"
    path_to_test_fake = path_to_test + "fake/"

    X_train, y_train, X_test, y_test = [], [], [], []  # noqa: N806

    for path, label in [
        (path_to_train_real, 0),
        (path_to_train_fake, 1),
        (path_to_test_real, 0),
        (path_to_test_fake, 1),
    ]:
        for file in Path(path).glob("*.npy"):
            data = np.load(file)
            if "train" in str(path):
                X_train.append(data)
                y_train.append(label)
            else:
                X_test.append(data)
                y_test.append(label)

    # pad or truncate time series to 256
    X_train = pad_sequences(  # noqa: N806
        X_train, maxlen=256, dtype="float32", padding="post"
    )
    X_test = pad_sequences(  # noqa: N806
        X_test, maxlen=256, dtype="float32", padding="post"
    )

    X_train = np.expand_dims(X_train, axis=-1)  # noqa: N806
    X_test = np.expand_dims(X_test, axis=-1)  # noqa: N806

    y_train = to_categorical(y_train, num_classes=2)
    y_test = to_categorical(y_test, num_classes=2)

    return X_train, y_train, X_test, y_test


def compare_keras_models() -> None:  # noqa: PLR0912
    enable_unsafe_deserialization()

    X_train, y_train, X_test, y_test = generate_datasets()  # noqa: N806

    models = [
        LongShortTermMemory(),
        LongShortTermMemory(attention=True),
        MultiLayerPerceptron(),
        FullyConvolutionalNeuralNetwork(),
        ConvolutionalNeuralNetwork(),
        ResNet(),
        # Encoder(),
    ]

    epochs = [500, 500, 5000, 2000, 2000, 1500, 100]
    batch_size = 16
    resnet_batch_size = 64

    results = []

    for i, (model, epoch) in enumerate(zip(models, epochs)):
        model_path = "/dcs/large/u2204489/" + model.__class__.__name__.lower()
        if i == 1:
            model_path += "_attention"
        model_path += ".keras"

        if Path(model_path).exists():
            model = load_model(model_path)  # noqa: PLW2901
        else:
            print(f"Training {model.__class__.__name__}...")
            if isinstance(model, ResNet):
                model.fit(X_train, y_train, epoch, resnet_batch_size)
            else:
                model.fit(X_train, y_train, epoch, batch_size)

            model.save(model_path)

        false_positives = 0
        false_negatives = 0
        true_positives = 0
        true_negatives = 0

        y_pred = model.predict(X_test)

        for pred, actual in zip(y_pred, y_test):
            if pred[0] > pred[1]:
                if np.argmax(actual) == 0:
                    true_negatives += 1
                else:
                    false_negatives += 1
            else:  # noqa: PLR5501
                if np.argmax(actual) == 1:
                    true_positives += 1
                else:
                    false_positives += 1

        results.append(
            {
                "model": model.__class__.__name__,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "true_positives": true_positives,
                "true_negatives": true_negatives,
                "accuracy": (true_positives + true_negatives) / len(y_test),
            }
        )

    for result in results:
        print(result)

    print("done :)")


if __name__ == "__main__":
    arg = None
    try:
        arg = sys.argv[1]
    except IndexError:
        print("Please provide an argument")
        sys.exit(1)

    if arg == "keras":
        compare_keras_models()
    else:
        print("Please provide a valid argument")
        sys.exit(1)
