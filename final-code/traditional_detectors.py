from os import listdir
from pathlib import Path
from random import sample

import cv2 as cv
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input  # type: ignore
from tensorflow.keras.applications import EfficientNetB4, Xception  # type: ignore
from tensorflow.keras.layers import (  # type: ignore
    BatchNormalization,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
)
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array, load_img  # type: ignore
from tensorflow.keras.utils import Sequence, to_categorical  # type: ignore

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


class ImageDataGenerator(Sequence):
    def __init__(  # noqa: PLR0913
        self,
        reals: list[str],
        fakes: list[str],
        batch_size: int,
        image_size: tuple[int, int],
        train: bool = True,
        split: float = 0.8,
    ) -> None:
        super().__init__()

        # split real and fake images into train/test sets
        reals_train, reals_test = train_test_split(
            reals, train_size=split, random_state=42
        )
        fakes_train, fakes_test = train_test_split(
            fakes, train_size=split, random_state=42
        )

        # select train or test set based on flag
        self.reals = reals_train if train else reals_test
        self.fakes = fakes_train if train else fakes_test

        self.batch_size = batch_size
        self.real_batch_size = batch_size // 2
        self.fake_batch_size = batch_size // 2
        self.image_size = image_size
        self.indices = np.arange(len(self.reals) + len(self.fakes))
        self.on_epoch_end()

    def __len__(self) -> int:
        return len(self.indices) // self.batch_size

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        real_batch = sample(self.reals, self.real_batch_size)
        fake_batch = sample(self.fakes, self.fake_batch_size)

        return self.__data_generation(
            real_batch + fake_batch,
            [1] * self.real_batch_size + [0] * self.fake_batch_size,
        )

    def __data_generation(
        self, X: list[str], Y: list[int]  # noqa: N803
    ) -> tuple[np.ndarray, np.ndarray]:
        return np.array(
            [
                img_to_array(load_img(img_path, target_size=self.image_size)).flatten()
                / 255.0
                for img_path in X
            ]
        ).reshape(-1, self.image_size[0], self.image_size[1], 3), to_categorical(Y, num_classes=2)



# Video Face Manipulation Detection Through Ensemble of CNNs
# https://arxiv.org/pdf/2004.07676v1


def efficientnet_b4() -> Model:
    backbone = EfficientNetB4(include_top=False, input_shape=(256, 256, 3))

    inputs = Input(shape=(256, 256, 3))
    x = backbone(inputs)
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
    x = backbone(inputs)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(2, activation="softmax")(x)

    return Model(inputs=inputs, outputs=x)


def main() -> None:
    # create_image_dataset()

    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    fakes = [
        str(f) for f in listdir(f"{path_to_train_data}/fake") if f.endswith(".png")
    ]
    reals = [
        str(f) for f in listdir(f"{path_to_train_data}/real") if f.endswith(".png")
    ]
    fakes = [f"{path_to_train_data}/fake/{f}" for f in fakes]
    reals = [f"{path_to_train_data}/real/{r}" for r in reals]

    train_generator = ImageDataGenerator(reals, fakes, 32, (256, 256), True)
    test_generator = ImageDataGenerator(reals, fakes, 32, (256, 256), False)

    model = efficientnet_b4()
    model.compile(
        optimizer=Adam(learning_rate=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-07),
        loss="binary_crossentropy",
    )
    model.fit(train_generator, epochs=60, validation_data=test_generator)
    model.save("/dcs/large/u2204489/efficientnet_b4_1.keras")

    model = xception()
    model.compile(
        optimizer=Adam(learning_rate=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-07),
        loss="categorical_crossentropy",
    )
    model.fit(train_generator, epochs=60, validation_data=test_generator)
    model.save("/dcs/large/u2204489/xception_1.keras")

    print("Done :)")


if __name__ == "__main__":
    main()
