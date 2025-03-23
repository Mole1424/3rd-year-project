from pathlib import Path
from random import shuffle
from typing import Generator

import numpy as np
import tensorflow as tf
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
from tensorflow.keras.utils import to_categorical  # type: ignore


def load_image(path: str, target_size: tuple[int, int]) -> np.ndarray:
    """Load an image from path"""
    image = load_img(path, target_size=target_size)
    image = img_to_array(image)
    return image / 255.0 # normalise

def image_generator(
    paths: list, labels: list, target_size: tuple[int, int]
) -> Generator:
    """Generate images from a list of paths and labels."""
    for path, label in zip(paths, labels):
        yield load_image(path, target_size), to_categorical(label, 2) # type: ignore

def get_dataset(
    paths: list, labels: list, batch_size: int, target_size: tuple[int, int]
) -> tf.data.Dataset:
    """Create a tf.data.Dataset from a list of paths and labels."""
    return tf.data.Dataset.from_generator(
        lambda: image_generator(paths, labels, target_size),
        output_signature=(
            tf.TensorSpec(shape=(target_size[0], target_size[1], 3), dtype=tf.float32), # type: ignore
            tf.TensorSpec(shape=(2,), dtype=tf.float32), # type: ignore
        )
    ).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)


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


# MARK: Xception
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

# MARK: VGG19
# based on https://github.com/rahul9903/Deepfake/blob/main/Deepfake_detection.ipynb and https://www.kaggle.com/code/navneethkrishna23/deepfake-detection-vgg16
# thanks to Pradyumna Yadav, Priyansh Sharma, and Sakshi Verma
# and Navneeth Krishna, Darshan V Prasad, Haxrsxha, and Sanjay Tc

def vgg19() -> Model:
    backbone = VGG19(include_top=False, input_shape=(256, 256, 3))

    inputs = Input(shape=(256, 256, 3))
    x = backbone(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dense(2, activation="softmax")(x)

    return Model(inputs=inputs, outputs=x)

# MARK: ResNet50
# based on https://www.kaggle.com/code/lightningblunt/deepfake-image-detection-using-resnet50
# thanks to Manas Tiwari

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


def train_detectors(  # noqa: PLR0915
    path_to_models: str,
    name: str,
    path_to_dataset: str,
) -> tuple[Model, Model, Model, Model]:
    """Train the detectors and return the models."""
    # load psths to images
    reals = list(Path(f"{path_to_dataset}real").rglob("*.jpg"))
    fakes = list(Path(f"{path_to_dataset}fake").rglob("*.jpg"))

    # split into real and fake
    real_train, real_test = train_test_split(reals, train_size=0.8, random_state=42)
    fake_train, fake_test = train_test_split(fakes, train_size=0.8, random_state=42)

    # set batch size (related to #GPUs to avoid errors)
    batch_size = 32 * len(tf.config.list_physical_devices("GPU"))

    # ensure train sizes are equal for fair training
    train_size = min(len(real_train), len(fake_train)) // batch_size * batch_size
    test_size = min(len(real_test), len(fake_test)) // batch_size * batch_size
    real_train = real_train[:train_size]
    fake_train = fake_train[:train_size]
    real_test = real_test[:test_size]
    fake_test = fake_test[:test_size]

    # create labels
    real_labels = [0] * len(real_train + real_test)
    fake_labels = [1] * len(fake_train + fake_test)

    # convert paths to strings
    train_paths = [str(path) for path in real_train + fake_train]
    test_paths = [str(path) for path in real_test + fake_test]
    train_labels = real_labels[: len(real_train)] + fake_labels[: len(fake_train)]
    test_labels = real_labels[len(real_train) :] + fake_labels[len(fake_train) :]

    # shuffle data
    zipped_train = list(zip(train_paths, train_labels))
    shuffle(zipped_train)
    train_paths, train_labels = zip(*zipped_train)
    zipped_test = list(zip(test_paths, test_labels))
    shuffle(zipped_test)
    test_paths, test_labels = zip(*zipped_test)

    # get datasets
    train_generator = get_dataset(train_paths, train_labels, batch_size, (256, 256)) # type: ignore
    test_generator = get_dataset(test_paths, test_labels, batch_size, (256, 256)) # type: ignore

    # train the models

    # check if the models already exist (have been trained)
    if not Path(f"{path_to_models}efficientnet_{name}.keras").exists():
        # if not comile, train, and save with appropriate specs
        efficientnet = efficientnet_b4()
        efficientnet.compile(
            optimizer=Adam(
                learning_rate=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-07
            ),
            loss="binary_crossentropy",
        )
        efficientnet.fit(train_generator, epochs=60, validation_data=test_generator)
        efficientnet.save(f"{path_to_models}efficientnet_{name}.keras")
    else:
        efficientnet = load_model(f"{path_to_models}efficientnet_{name}.keras")

    if not Path(f"{path_to_models}resnet_{name}.keras").exists():
        x_ception = xception()
        x_ception.compile(
            optimizer=Adam(
                learning_rate=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-07
            ),
            loss="categorical_crossentropy",
        )
        x_ception.fit(train_generator, epochs=60, validation_data=test_generator)
        x_ception.save(f"{path_to_models}xception_{name}.keras")
    else:
        x_ception = load_model(f"{path_to_models}xception_{name}.keras")

    if not Path(f"{path_to_models}vgg19_{name}.keras").exists():
        vgg = vgg19()
        vgg.compile(
            optimizer=Adam(
                learning_rate=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-7
            ),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        vgg.fit(train_generator, epochs=20, validation_data=test_generator)
        vgg.save(f"{path_to_models}vgg19_{name}.keras")
    else:
        vgg = load_model(f"{path_to_models}vgg19_{name}.keras")

    if not Path(f"{path_to_models}resnet50_{name}.keras").exists():
        resnet = resnet50()
        resnet.compile(
            optimizer=Adam(),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        resnet.fit(train_generator, epochs=20, validation_data=test_generator)
        resnet.save(f"{path_to_models}resnet50_{name}.keras")
    else:
        resnet = load_model(f"{path_to_models}resnet50_{name}.keras")

    print("Training Done :)")

    return efficientnet, x_ception, vgg, resnet
