import numpy as np
from fastdtw import fastdtw
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras import Model, Sequential  # type: ignore
from tensorflow.keras.layers import (  # type: ignore
    LSTM,
    Bidirectional,
    Conv1D,
    Dense,
    Input,
)

# MARK: Neural Networks


def dense_model(nun_layers: int, num_units: int) -> Model:
    """A model of <num_layers> Dense layers"""
    model = Sequential()
    model.add(Input(shape=(None, 1)))
    for _ in range(nun_layers):
        model.add(Dense(num_units, activation="relu"))
    model.add(Dense(1), activation="sigmoid")
    return model


def cnn_model(num_filters: int, kernel_size: int, num_layers: int) -> Model:
    """Model of <num_layers> Conv1D layers"""
    model = Sequential()
    model.add(Input(shape=(None, 1)))
    for _ in range(num_layers):
        model.add(Conv1D(num_filters, kernel_size, activation="relu"))
    model.add(Dense(num_filters, activation="relu"))
    model.add(Dense(1), activation="sigmoid")
    return model


def lstm_model(num_units: int, bidirectional: bool) -> Model:
    """LSTM model"""
    model = Sequential()
    model.add(Input(shape=(None, 1)))
    lstm = LSTM(num_units, activation="relu", return_sequences=True)
    model.add(Bidirectional(lstm) if bidirectional else lstm)
    model.add(Dense(1), activation="sigmoid")
    return model


def compile_and_train_model(  # noqa: PLR0913
    model: Model,
    model_path: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> Model:
    """compile and train a given model (saves the model to model_path)"""
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=32)
    model.save(model_path)
    return model


# https://www.intechopen.com/chapters/1185930
# MARK: Nearest Neighbour with Dynamic Time Warping


# https://medium.com/@quantclubiitkgp/time-series-classification-using-dynamic-time-warping-k-nearest-neighbour-e683896e0861
class NearestNeighbour:
    """Nearest Neighbour with Dynamic Time Warping"""

    def __init__(self) -> None:
        self.knn = KNeighborsClassifier(n_neighbors=1, metric=self._dtw)

    def _dtw(self, ears1: np.ndarray, ears2: np.ndarray) -> float:
        """Dynamic Time Warping distance between two ears"""
        return float(fastdtw(ears1, ears2)[0])

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit the model"""
        self.knn.fit(x_train, y_train)

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        """Predict the model"""
        return self.knn.predict(x_test).astype(bool)


# MARK: Global Alignment Kernel
