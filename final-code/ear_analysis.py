from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from pyts.classification import (
    TSBF,
    KNeighborsClassifier,
    LearningShapelets,
    TimeSeriesForest,
)
from sklearn.base import BaseEstimator
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
    Reshape,
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

# MARK: Main external class


class EarAnalysis:
    """Analyses ear data to determine if it is real or fake."""

    def __init__(
        self,
        path: str | None, dataset: list[list[tuple[int, np.ndarray]]] | None,
        base_path: str,
    ) -> None:
        if path is not None:
            if path.endswith(".joblib"):
                self.model = joblib.load(path)
                self.tensorflow = False
            else:
                self.model = load_model(path)
                self.tensorflow = True
        elif dataset is not None:
            self.model, self.tensorflow = self._get_model(dataset, base_path)
        else:
            raise ValueError("Either path or dataset must be provided.")

    def _get_model(
        self, dataset: list[list[tuple[int, np.ndarray]]], base_path: str
    ) -> tuple[Model | BaseEstimator, bool]:
        trainset, testset = dataset
        y_train, X_train = zip(*trainset)  # noqa: N806
        y_test, X_test = zip(*testset) # noqa: N806

        # Define models
        keras_models = [
            LongShortTermMemory(),
            LongShortTermMemory(attention=True),
            MultiLayerPerceptron(),
            FullyConvolutionalNeuralNetwork(),
            ConvolutionalNeuralNetwork(),
            ResNet(),
        ]

        classical_models = [
            KNeighborsClassifier(metric="dtw", n_jobs=-1),
            KNeighborsClassifier(metric="dtw_sakoechiba", n_jobs=-1),
            KNeighborsClassifier(metric="dtw_itakura", n_jobs=-1),
            KNeighborsClassifier(metric="dtw_fast", n_jobs=-1),
            LearningShapelets(random_state=42, n_jobs=-1),
            TimeSeriesForest(n_jobs=-1, random_state=42),
            TSBF(n_jobs=-1, random_state=42)
        ]

        keras_epochs = [500, 500, 5000, 2000, 2000, 1500]
        batch_size = 16
        resnet_batch_size = 64

        models = [(model, True) for model in keras_models] + [
            (model, False) for model in classical_models
        ]
        epochs = keras_epochs + [None] * len(classical_models)

        best_model, best_accuracy, best_tensorflow = None, 0, False

        for (model, is_tensorflow), epoch in zip(models, epochs):
            model_name = model.__class__.__name__.lower()
            model_path = Path(
                f"{base_path}/{model_name}.{"keras" if is_tensorflow else "joblib"}"
            )
            # Load model if it exists
            if model_path.exists():
                model = ( # noqa: PLW2901
                    load_model(model_path) if is_tensorflow else joblib.load(model_path)
                )
            else:
                print(f"Training {model_name}...")
                if is_tensorflow:
                    model.fit(
                        np.expand_dims(X_train, axis=-1),
                        to_categorical(y_train, num_classes=2),
                        epoch,
                        resnet_batch_size if isinstance(model, ResNet) else batch_size
                    )
                    model.save(str(model_path))
                else:
                    model.fit(X_train, y_train)
                    joblib.dump(model, str(model_path))

            # Evaluate model
            y_pred = (
                model.predict(np.expand_dims(X_test, axis=-1))
                if is_tensorflow else model.predict(np.array(X_test))
            )
            y_pred_labels = np.argmax(y_pred, axis=1) if is_tensorflow else y_pred
            accuracy = np.mean(y_pred_labels == y_test)

            # Track best model
            if accuracy > best_accuracy:
                best_model = model
                best_accuracy = accuracy
                best_tensorflow = is_tensorflow

        print(f"Best model: {best_model.__class__.__name__} with accuracy {best_accuracy}")  # noqa: E501
        return best_model, best_tensorflow


    def predict(self, data: np.ndarray) -> int:
        data = pad_sequences([data], maxlen=256, dtype="float32", padding="post")
        if self.tensorflow:
            data = np.expand_dims(data, axis=-1)
            return int(np.argmax(self.model.predict(data))) # type: ignore
        else:  # noqa: RET505
            return self.model.predict(np.array(data))[0] # type: ignore
