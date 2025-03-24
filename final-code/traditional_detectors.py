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
    path_to_models: str, name: str, path_to_dataset: str
) -> tuple[Model, Model, Model, Model]:
    """Train the detectors and return the models."""
    # this is cursed python code but i love it
    batch_size = 32 * (len(tf.config.list_physical_devices("GPU")) or 1)

    # load paths to images
    reals = list(map(str, Path(f"{path_to_dataset}real").rglob("*.jpg")))
    fakes = list(map(str, Path(f"{path_to_dataset}fake").rglob("*.jpg")))

    # split the data
    train_reals, test_reals = train_test_split(reals, train_size=0.8, random_state=42)
    train_fakes, test_fakes = train_test_split(fakes, train_size=0.8, random_state=42)

    # combine into train and test data
    train_data = train_reals + train_fakes
    train_labels = [1] * len(train_reals) + [0] * len(train_fakes)
    test_data = test_reals + test_fakes
    test_labels = [1] * len(test_reals) + [0] * len(test_fakes)

    # shuffle the data
    train_set = list(zip(train_data, train_labels))
    shuffle(train_set)
    train_data, train_labels = zip(*train_set)
    test_set = list(zip(test_data, test_labels))
    shuffle(test_set)
    test_data, test_labels = zip(*test_set)

    # create the datasets
    train_generator = get_dataset(train_data, train_labels, batch_size, (256, 256)) # type: ignore
    test_generator = get_dataset(test_data, test_labels, batch_size, (256, 256)) # type: ignore

    # train the models

    # check if the models already exist (have been trained)
    print("Checking efficientnet")
    if not Path(f"{path_to_models}efficientnet_{name}.keras").exists():
        # if not comile, train, and save with appropriate specs
        print("Training efficientnet")
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
        print("Loading efficientnet")
        efficientnet = load_model(f"{path_to_models}efficientnet_{name}.keras")
    print("Got efficientnet")

    print("Checking xception")
    if not Path(f"{path_to_models}xception_{name}.keras").exists():
        print("Training xception")
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
        print("Loading xception")
        x_ception = load_model(f"{path_to_models}xception_{name}.keras")
    print("Got xception")

    print("Checking vgg19")
    if not Path(f"{path_to_models}vgg19_{name}.keras").exists():
        print("Training vgg19")
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
        print("Loading vgg19")
        vgg = load_model(f"{path_to_models}vgg19_{name}.keras")
    print("Got vgg19")

    print("Checking resnet50")
    if not Path(f"{path_to_models}resnet50_{name}.keras").exists():
        print("Training resnet50")
        resnet = resnet50()
        resnet.compile(
            optimizer=Adam(),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        resnet.fit(train_generator, epochs=20, validation_data=test_generator)
        resnet.save(f"{path_to_models}resnet50_{name}.keras")
    else:
        print("Loading resnet50")
        resnet = load_model(f"{path_to_models}resnet50_{name}.keras")
    print("Got resnet50")

    return efficientnet, x_ception, vgg, resnet
