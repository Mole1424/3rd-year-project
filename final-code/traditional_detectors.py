from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input  # type: ignore
from tensorflow.keras.applications import (  # type: ignore
    VGG19,
    EfficientNetB4,
    ResNet50,
    Xception,
)
from tensorflow.keras.layers import (  # type: ignore
    BatchNormalization,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
)
from tensorflow.keras.models import Model, load_model  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array, load_img  # type: ignore
from tensorflow.keras.utils import Sequence, to_categorical  # type: ignore


class ImageDataGenerator(Sequence):
    def __init__(
        self,
        path_to_dataset: str,
        batch_size: int,
        image_size: tuple[int, int],
        train: bool = True
    ) -> None:
        super().__init__()

        reals = list(Path(f"{path_to_dataset}real").rglob("*.jpg"))
        fakes = list(Path(f"{path_to_dataset}fake").rglob("*.jpg"))

        # split real and fake images into train/test sets
        reals_train, reals_test = train_test_split(
            reals, train_size=0.8, random_state=42
        )
        fakes_train, fakes_test = train_test_split(
            fakes, train_size=0.8, random_state=42
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
        start = index * self.batch_size
        end = start + self.batch_size
        indices = self.indices[start:end]

        real_batch = [self.reals[i] for i in indices if i < len(self.reals)]
        fake_batch = [
            self.fakes[i - len(self.reals)] for i in indices if i >= len(self.reals)
        ]

        return self.__data_generation(
            real_batch + fake_batch, [1] * len(real_batch) + [0] * len(fake_batch)
        )

    def __data_generation(
        self, X: list[str], Y: list[int]  # noqa: N803
    ) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.array(
                [
                    # load the image, resize it, and flatten it
                    img_to_array(
                        load_img(img_path, target_size=self.image_size)
                    ).flatten() / 255.0
                    for img_path in X
                ]
            ).reshape(-1, self.image_size[0], self.image_size[1], 3) # reshape the image
        , to_categorical(Y, num_classes=2))

    def on_epoch_end(self) -> None:
        np.random.shuffle(self.indices)



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

# resnet50 and vgg19 models

def vgg19() -> Model:
    backbone = VGG19(include_top=False, input_shape=(256, 256, 3))

    inputs = Input(shape=(256, 256, 3))
    x = backbone(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dense(2, activation="softmax")(x)

    return Model(inputs=inputs, outputs=x)

def resnet50() -> Model:
    backbone = ResNet50(include_top=False, input_shape=(256, 256, 3))

    inputs = Input(shape=(256, 256, 3))
    x = backbone(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(2, activation="softmax")(x)

    return Model(inputs=inputs, outputs=x)


def train_detectors(
    path_to_models: str, name: str, path_to_dataset: str
) -> tuple[Model, Model, Model, Model]:

    # create the image data generator
    train_generator = ImageDataGenerator(path_to_dataset, 32, (256, 256), True)
    test_generator = ImageDataGenerator(path_to_dataset, 32, (256, 256), False)

    # train the models

    # check if the models already exist
    if not Path(f"{path_to_models}efficientnet{name}.keras").exists():
        # if not comile, train, and save with appropriate specs
        efficientnet = efficientnet_b4()
        efficientnet.compile(
            optimizer=Adam(
                learning_rate=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-07
            ),
            loss="binary_crossentropy",
        )
        efficientnet.fit(train_generator, epochs=60, validation_data=test_generator)
        efficientnet.save(f"{path_to_models}efficientnet{name}.keras")
    else:
        efficientnet = load_model(f"{path_to_models}efficientnet{name}.keras")

    if not Path(f"{path_to_models}resnet{name}.keras").exists():
        x_ception = xception()
        x_ception.compile(
            optimizer=Adam(
                learning_rate=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-07
            ),
            loss="categorical_crossentropy",
        )
        x_ception.fit(train_generator, epochs=60, validation_data=test_generator)
        x_ception.save(f"{path_to_models}xception{name}.keras")
    else:
        x_ception = load_model(f"{path_to_models}xception{name}.keras")

    if not Path(f"{path_to_models}vgg19{name}.keras").exists():
        vgg = vgg19()
        vgg.compile(
            optimizer=Adam(
                learning_rate=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-7
            ),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        vgg.fit(train_generator, epochs=20, validation_data=test_generator)
        vgg.save(f"{path_to_models}vgg19{name}.keras")
    else:
        vgg = load_model(f"{path_to_models}vgg19{name}.keras")

    if not Path(f"{path_to_models}resnet50{name}.keras").exists():
        resnet = resnet50()
        resnet.compile(
            optimizer=Adam(),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        resnet.fit(train_generator, epochs=20, validation_data=test_generator)
        resnet.save(f"{path_to_models}resnet50{name}.keras")
    else:
        resnet = load_model(f"{path_to_models}resnet50{name}.keras")

    print("Training Done :)")

    return efficientnet, x_ception, vgg, resnet
