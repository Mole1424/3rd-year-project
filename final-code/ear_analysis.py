import random
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
from sklearn.model_selection import train_test_split
from tensorflow.keras import Layer, Model  # type: ignore
from tensorflow.keras.callbacks import ReduceLROnPlateau  # type: ignore
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
from tensorflow.keras.utils import to_categorical  # type: ignore

INPUT_LENGTH = 512

class KerasTimeSeriesClassifier:
    """Interface for time series classifiers using Keras"""
    def __init__(self, model: Model) -> None:
        self.model = model
        self.callbacks = [
            ReduceLROnPlateau(monitor="loss", factor=0.5, patience=50, min_lr=0.0001)
        ]

    def fit(
        self, train_data: tf.data.Dataset, validation_data: tf.data.Dataset, epochs: int
    ) -> None:
        self.model.fit(
            train_data,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=self.callbacks,
            verbose=2,
        )

    def predict(self, X: np.ndarray, verbose: int) -> np.ndarray:  # noqa: N803
        return self.model.predict(X, verbose=verbose)

    def save(self, path: str) -> None:
        self.model.save(path)


# MARK: LSTM
# https://ieeexplore.ieee.org/document/8141873

# custom layer to transpose input
@register_keras_serializable(package="Custom", name="TransposeLayer")
class TransposeLayer(Layer):
    def __init__(self, **kwargs: dict) -> None:
        super(TransposeLayer, self).__init__(**kwargs)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return tf.transpose(inputs, perm=[0, 2, 1])


class LongShortTermMemory(KerasTimeSeriesClassifier):
    def __init__(self, path: str | None = None, attention: bool = False) -> None:
        self.attention = attention

        self.model = load_model(path) if path else self.build_model()
        super().__init__(self.model)

    def build_model(self) -> Model:
        input = Input(shape=(INPUT_LENGTH, 1))

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
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
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
        input = Input(shape=(INPUT_LENGTH, 1))

        x = Dropout(0.1)(input)
        x = Dense(500, activation="relu")(x)

        x = Dropout(0.2)(x)
        x = Dense(500, activation="relu")(x)

        x = Dropout(0.3)(x)
        x = GlobalAveragePooling1D()(x)
        x = Dense(2, activation="softmax")(x)

        model = Model(inputs=input, outputs=x)
        model.compile(
            optimizer="adadelta", loss="categorical_crossentropy", metrics=["accuracy"]
        )

        return model


class FullyConvolutionalNeuralNetwork(KerasTimeSeriesClassifier):
    def __init__(self, path: str | None = None) -> None:
        self.model = load_model(path) if path else self.build_model()
        super().__init__(self.model)

    def build_model(self) -> Model:
        input = Input(shape=(INPUT_LENGTH, 1))

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
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )

        return model


class ConvolutionalNeuralNetwork(KerasTimeSeriesClassifier):
    def __init__(self, path: str | None = None) -> None:
        self.model = load_model(path) if path else self.build_model()
        super().__init__(self.model)

    def build_model(self) -> Model:
        input = Input(shape=(INPUT_LENGTH, 1))

        x = Conv1D(6, 7, padding="same", activation="sigmoid")(input)
        x = AveragePooling1D(3, padding="same")(x)

        x = Conv1D(12, 7, padding="same", activation="sigmoid")(x)
        x = AveragePooling1D(3, padding="same")(x)

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

        input = Input(shape=(INPUT_LENGTH, 1))

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
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )

        return model

# MARK: Main external class


class EarAnalysis:
    """Analyses EAR data to determine if it is real or fake."""

    def __init__(
        self,
        path: str | None,
        dataset: list[list[tuple[np.ndarray, int]]] | None,
        path_to_models: str,
        dataset_name: str,
        eye_landmarker: str,
    ) -> None:
        if path is not None:
            # load model from path if given
            if path.endswith(".joblib"):
                self.model = joblib.load(path)
                self.tensorflow = False
            else:
                self.model = load_model(
                    path, custom_objects={"TransposeLayer": TransposeLayer}
                )
                self.tensorflow = True
            self.best_path = path
        elif dataset is not None:
            # otherwise train a new models and get best one
            self.model, self.tensorflow, self.best_path = self._get_model(
                dataset, path_to_models, dataset_name, eye_landmarker
            )
        else:
            raise ValueError("Either path or dataset must be provided.")

    def get_best_path(self) -> str:
        return self.best_path

    def _get_model(
        self,
        dataset: list[list[tuple[np.ndarray, int]]],
        path_to_models: str,
        dataset_name: str,
        eye_landmarker: str,
    ) -> tuple[Model | BaseEstimator, bool, str]:
        """train models on dataset and return best one"""

        # extract train and test sets
        trainset, testset = dataset

        # shuffle the data
        random.seed(42)
        random.shuffle(trainset)
        random.shuffle(testset)

        # split into X and y
        X_train, y_train = zip(*trainset)  # noqa: N806
        X_test, y_test = zip(*testset) # noqa: N806

        # further split train set into train and validation sets
        tf_X_train, tf_X_val, tf_y_train, tf_y_val = train_test_split(  # noqa: N806
            X_train, y_train, train_size=0.8, random_state=42
        )

        # define models
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
        names = [
            "lstm",
            "lstm_attention",
            "mlp",
            "fcn",
            "cnn",
            "resnet",
            "knn_dtw",
            "knn_dtw_sakoechiba",
            "knn_dtw_itakura",
            "knn_dtw_fast",
            "learning_shapelets",
            "time_series_forest",
            "tsbf"
        ]

        # hyperparams for training
        keras_epochs = [500, 500, 5000, 2000, 2000, 1500]
        batch_size = 16
        resnet_batch_size = 64

        # add models to central list for tracking
        models = list(zip(keras_models, [True] * len(keras_models), names))
        models += list(
            zip(classical_models, [False] * len(classical_models), names[6:])
        )
        epochs = keras_epochs + [None] * len(classical_models)

        best_model, best_model_name, best_accuracy, best_tensorflow = None, "", 0, False

        # foreach model
        for (model, is_tensorflow, name), epoch in zip(models, epochs):

            # load model if it exists, otherwise train it
            model_path = Path(
                f"{path_to_models}{dataset_name}_{eye_landmarker}_{name}."
                f"{"keras" if is_tensorflow else "joblib"}"
            )
            print(f"Checking {model_path!s}...")
            if model_path.exists():
                model = ( # noqa: PLW2901
                    load_model(model_path) if is_tensorflow else joblib.load(model_path)
                )
            else:
                print(f"Training {name}...")
                if is_tensorflow:
                    train_dataset = tf.data.Dataset.from_tensor_slices(
                        (np.expand_dims(np.array(tf_X_train), axis=-1),
                        to_categorical(tf_y_train, num_classes=2)),
                    ).batch(
                        resnet_batch_size if isinstance(model, ResNet) else batch_size
                    ).prefetch(tf.data.AUTOTUNE)
                    val_dataset = tf.data.Dataset.from_tensor_slices(
                        (np.expand_dims(np.array(tf_X_val), axis=-1),
                        to_categorical(tf_y_val, num_classes=2)),
                    ).batch(
                        resnet_batch_size if isinstance(model, ResNet) else batch_size
                    ).prefetch(tf.data.AUTOTUNE)

                    model.fit(train_dataset, val_dataset, epoch) # type: ignore
                    model.save(str(model_path))
                else:
                    model.fit(X_train, y_train)
                    joblib.dump(model, str(model_path))

            # evaluate model on accuracy
            y_pred = (
                model.predict(np.expand_dims(X_test, axis=-1), verbose=0) # type: ignore
                if is_tensorflow else model.predict(np.array(X_test)) # type: ignore
            )
            y_pred_labels = np.argmax(y_pred, axis=1) if is_tensorflow else y_pred
            accuracy = np.mean(y_pred_labels == y_test)

            # promote model if it is the best
            if accuracy > best_accuracy:
                best_model = model
                best_model_name = name
                best_accuracy = accuracy
                best_tensorflow = is_tensorflow

        # save and return best model
        print(f"Best model: {best_model_name} with accuracy {best_accuracy}")
        best_path = (
            f"{path_to_models}/{dataset_name}_{eye_landmarker}_{best_model_name}."
        )
        best_path += "keras" if best_tensorflow else "joblib"
        return best_model, best_tensorflow, best_path


    def predict(self, data: np.ndarray) -> int:
        """predict if ear time series is real or fake"""

        # pad or truncate data to desired length
        if len(data) < INPUT_LENGTH:
            data = np.pad(
                data, (0, INPUT_LENGTH - len(data)), "constant", constant_values=-1
            )
        elif len(data) > INPUT_LENGTH:
            data = data[:INPUT_LENGTH]

        # predict using model
        if self.tensorflow:
            data = np.expand_dims(np.array(data), axis=-1)
            return int(np.argmax(self.model.predict(data, verbose=0))) # type: ignore
        else:  # noqa: RET505
            return self.model.predict(np.array([data]))[0] # type: ignore
