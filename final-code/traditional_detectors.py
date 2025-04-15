import random
from pathlib import Path
from typing import Generator

import cv2 as cv
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input, Sequential  # type: ignore
from tensorflow.keras.applications import (  # type: ignore
    VGG19,
    EfficientNetB4,
    ResNet50,
    Xception,
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # type: ignore
from tensorflow.keras.layers import (  # type: ignore
    BatchNormalization,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    RandomBrightness,
    RandomContrast,
    RandomFlip,
    RandomRotation,
    RandomZoom,
)
from tensorflow.keras.models import Model, load_model  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.utils import to_categorical  # type: ignore


def image_generator(
    paths: list, labels: list, target_size: tuple[int, int]
) -> Generator:
    """Generate images from a list of paths and labels."""
    for path, label in zip(paths, labels):
        image = cv.resize(cv.imread(path), target_size).astype(np.float32) / 255.0
        label_one_hot = to_categorical(label, 2)
        yield image, label_one_hot

def get_dataset(
    paths: list, labels: list, batch_size: int, target_size: tuple[int, int]
) -> tf.data.Dataset:
    """Create a tf.data.Dataset from a list of paths and labels."""
    augmentations = Sequential([
        RandomFlip("horizontal"),
        RandomBrightness(0.1),
        RandomContrast(0.1),
        RandomRotation(0.1),
        RandomZoom(0.1),
    ])

    def augment(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """Apply augmentations to an image."""
        image = augmentations(image)
        return image, label

    return tf.data.Dataset.from_generator(
        lambda: image_generator(paths, labels, target_size),
        output_signature=(
            tf.TensorSpec(shape=(target_size[0], target_size[1], 3), dtype=tf.float32), # type: ignore
            tf.TensorSpec(shape=(2,), dtype=tf.float32), # type: ignore
        )
    ).map(augment).batch(batch_size).prefetch(tf.data.AUTOTUNE)


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
    path_to_models: str, name: str, path_to_dataset: str,
) -> tuple[Model, Model, Model, Model]:
    """Train the detectors and return the models."""

    batch_size = 32

    # load paths to images
    reals = sorted(map(str, Path(f"{path_to_dataset}real").rglob("*.png")))
    fakes = sorted(map(str, Path(f"{path_to_dataset}fake").rglob("*.png")))

    # group frames by video (32 frames per video)
    reals = [reals[i : i + 32] for i in range(0, len(reals), 32)]
    fakes = [fakes[i : i + 32] for i in range(0, len(fakes), 32)]

    # split videos into train and test sets
    train_reals, test_reals = train_test_split(reals, train_size=0.8, random_state=42)
    train_fakes, test_fakes = train_test_split(fakes, train_size=0.8, random_state=42)

    # flatten the lists
    train_reals = [frame for video in train_reals for frame in video]
    test_reals = [frame for video in test_reals for frame in video]
    train_fakes = [frame for video in train_fakes for frame in video]
    test_fakes = [frame for video in test_fakes for frame in video]

    # combine into train and test data and flatten
    train_data = train_reals + train_fakes
    train_labels = [1] * len(train_reals) + [0] * len(train_fakes)
    test_data = test_reals + test_fakes
    test_labels = [1] * len(test_reals) + [0] * len(test_fakes)

    # shuffle the data
    random.seed(42)
    train_set = list(zip(train_data, train_labels))
    random.shuffle(train_set)
    train_data, train_labels = zip(*train_set)
    test_set = list(zip(test_data, test_labels))
    random.shuffle(test_set)
    test_data, test_labels = zip(*test_set)

    # create the datasets
    train_generator = get_dataset(train_data, train_labels, batch_size, (256, 256)) # type: ignore
    test_generator = get_dataset(test_data, test_labels, batch_size, (256, 256)) # type: ignore

    # train the models
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3),
    ]

    # check if the models already exist (have been trained)
    print("Checking efficientnet")
    efficientnet_path = f"{path_to_models}{name}_efficientnet.keras"
    if not Path(efficientnet_path).exists():
        # if not comile, train, and save with appropriate specs
        print("Training efficientnet")
        efficientnet = efficientnet_b4()
        efficientnet.compile(
            optimizer=Adam(
                learning_rate=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-07
            ),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        efficientnet.fit(
            train_generator,
            epochs=100,
            validation_data=test_generator,
            verbose=2,
            callbacks=callbacks,
        )
        efficientnet.save(efficientnet_path)
    else:
        print("Loading efficientnet")
        efficientnet = load_model(efficientnet_path)
    print("Got efficientnet")

    print("Checking xception")
    xception_path = f"{path_to_models}{name}_xception.keras"
    if not Path(xception_path).exists():
        print("Training xception")
        x_ception = xception()
        x_ception.compile(
            optimizer=Adam(
                learning_rate=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-07
            ),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        x_ception.fit(
            train_generator,
            epochs=20,
            validation_data=test_generator,
            verbose=2,
            callbacks=callbacks,
        )
        x_ception.save(xception_path)
    else:
        print("Loading xception")
        x_ception = load_model(xception_path)
    print("Got xception")

    print("Checking vgg19")
    vgg19_path = f"{path_to_models}{name}_vgg19.keras"
    if not Path(vgg19_path).exists():
        print("Training vgg19")
        vgg = vgg19()
        vgg.compile(
            optimizer=Adam(
                learning_rate=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-7
            ),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        vgg.fit(
            train_generator,
            epochs=20,
            validation_data=test_generator,
            verbose=2,
            callbacks=callbacks,
        )
        vgg.save(vgg19_path)
    else:
        print("Loading vgg19")
        vgg = load_model(vgg19_path)
    print("Got vgg19")

    print("Checking resnet50")
    resnet_path = f"{path_to_models}{name}_resnet50.keras"
    if not Path(resnet_path).exists():
        print("Training resnet50")
        resnet = resnet50()
        resnet.compile(
            optimizer=Adam(),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        resnet.fit(
            train_generator,
            epochs=20,
            validation_data=test_generator,
            verbose=2,
            callbacks=callbacks,
        )
        resnet.save(resnet_path)
    else:
        print("Loading resnet50")
        resnet = load_model(resnet_path)
    print("Got resnet50")

    return efficientnet, x_ception, vgg, resnet


def test() -> None:
    # set up paths
    path_to_models = "/dcs/large/u2204489/"
    path_to_dataset = "/dcs/large/u2204489/faceforensics/"
    name="faceforensics"

    # load the models
    resnet50 = load_model(f"{path_to_models}{name}_resnet50.keras")
    xception = load_model(f"{path_to_models}{name}_xception.keras")
    efficientnet = load_model(f"{path_to_models}{name}_efficientnet.keras")
    vgg19 = load_model(f"{path_to_models}{name}_vgg19.keras")

    # load videos
    videos = list(Path(path_to_dataset).rglob("*.mp4"))

    # get random sample
    random.seed(42)
    video_sample = random.sample(videos, 10)

    # trackers
    total_frames = 0
    resnet_correct, xception_correct, efficientnet_correct, vgg19_correct = 0, 0, 0, 0

    for video in video_sample:
        cap = cv.VideoCapture(str(video))
        # sample 100 random frames
        num_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        random.seed(42)
        frames = sorted(random.sample(range(num_frames), 100))

        for frame in frames:
            cap.set(cv.CAP_PROP_POS_FRAMES, frame)
            success, image = cap.read()
            if not success:
                continue
            # preproess the image
            image = cv.resize(image, (256, 256)).astype(np.float32) / 255.0
            image = np.expand_dims(image, axis=0)
            # predict with each model
            pred_resnet = np.argmax(resnet50.predict(image, verbose=0))
            pred_xception = np.argmax(xception.predict(image, verbose=0))
            pred_efficientnet = np.argmax(efficientnet.predict(image, verbose=0))
            pred_vgg19 = np.argmax(vgg19.predict(image, verbose=0))

            truth = 1 if "real" in str(video) else 0
            if pred_resnet == truth:
                resnet_correct += 1
            if pred_xception == truth:
                xception_correct += 1
            if pred_efficientnet == truth:
                efficientnet_correct += 1
            if pred_vgg19 == truth:
                vgg19_correct += 1
            total_frames += 1
        cap.release()

    print(f"ResNet50: {resnet_correct / total_frames * 100:.2f}%")
    print(f"Xception: {xception_correct / total_frames * 100:.2f}%")
    print(f"EfficientNet: {efficientnet_correct / total_frames * 100:.2f}%")
    print(f"VGG19: {vgg19_correct / total_frames * 100:.2f}%")
    print("Done :)")

if __name__ == "__main__":
    test()
