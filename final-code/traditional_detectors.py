from os import listdir
from pathlib import Path

import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input  # type: ignore
from tensorflow.keras.applications import EfficientNetB4, Xception  # type: ignore
from tensorflow.keras.applications.efficientnet import preprocess_input  # type: ignore
from tensorflow.keras.layers import (  # type: ignore
    BatchNormalization,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
)
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array, load_img  # type: ignore
from tensorflow.keras.utils import to_categorical  # type: ignore

path_to_train_data = "/dcs/large/u2204489/faceforensics/train"


def create_image_dataset() -> None:
    # loop over the training videos
    for type in ["fake", "real"]:
        videos = [
            f for f in listdir(f"{path_to_train_data}/{type}") if f.endswith(".mp4")
        ]
        for video in videos:
            count = 0
            # open the video and get the frame rate
            cap = cv.VideoCapture(str(Path(path_to_train_data) / type / video))
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                # save the cropped face as a png file
                cv.imwrite(
                    str(Path(path_to_train_data) / type / f"{video}{count}.png"),
                    cv.resize(frame, (256, 256)),
                )
                count += 1
            cap.release()


def preprocess_data() -> tuple:
    # X is images, Y is real(0)/fake(1)
    X, Y = [], []  # noqa: N806

    # load the images and labels
    real_data = [f for f in listdir(f"{path_to_train_data}/real") if f.endswith(".png")]
    fake_data = [f for f in listdir(f"{path_to_train_data}/fake") if f.endswith(".png")]

    for img in real_data:
        X.append(
            img_to_array(load_img(f"{path_to_train_data}/real/{img}")).flatten() / 255.0
        )
        Y.append(0)
    for img in fake_data:
        X.append(
            img_to_array(load_img(f"{path_to_train_data}/fake/{img}")).flatten() / 255.0
        )
        Y.append(1)

    # preprocess the data
    X = np.array(X)  # noqa: N806
    Y = to_categorical(Y, 2)  # noqa: N806

    X = X.reshape(-1, 256, 256, 3)  # noqa: N806

    return X, Y


# Video Face Manipulation Detection Through Ensemble of CNNs
# https://arxiv.org/pdf/2004.07676v1


def efficientnet_b4() -> Model:
    backbone = EfficientNetB4(include_top=False, input_shape=(256, 256, 3))

    inputs = Input(shape=(256, 256, 3))
    x = backbone(preprocess_input(inputs))
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(2, activation="softmax")(x)

    return Model(inputs=inputs, outputs=x)


# FaceForensics++: Learning to Detect Manipulated Facial Images
# https://arxiv.org/pdf/1901.08971v3


def xception() -> Model:
    backbone = Xception(include_top=False, input_shape=(256, 256, 3))

    inputs = Input(shape=(256, 256, 3))
    x = backbone(preprocess_input(inputs))
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(2, activation="softmax")(x)

    return Model(inputs=inputs, outputs=x)


def main() -> None:
    create_image_dataset()
    X, Y = preprocess_data()  # noqa: N806

    model = efficientnet_b4()
    model.compile(
        optimizer=Adam(learning_rate=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-07),
        loss=tf.compat.v1.losses.log_loss,
    )
    model.fit(X, Y, batch_size=32, epochs=10, validation_split=0.2)
    model.save("/dcs/large/u2204489/efficientnet_b4.keras")

    model = xception()
    model.compile(
        optimizer=Adam(learning_rate=0.0002, bera_1=0.9, beta_2=0.999, epsilon=1e-07),
        loss="categorical_crossentropy",
    )
    model.fit(X, Y, batch_size=32, epochs=10, validation_split=0.2)
    model.save("/dcs/large/u2204489/xception.keras")

    print("Done :)")


if __name__ == "__main__":
    main()
