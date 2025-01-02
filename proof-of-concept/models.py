# vgg19 model based on https://github.com/rahul9903/Deepfake/blob/main/Deepfake_detection.ipynb
# thanks to Pradyumna Yadav, Priyansh Sharma, and Sakshi Verma
# also influenced by https://www.kaggle.com/code/navneethkrishna23/deepfake-detection-vgg16
# thanks to Navneeth Krishna, Darshan V Prasad, Haxrsxha, and Sanjay Tc

# resnet model based on https://www.kaggle.com/code/lightningblunt/deepfake-image-detection-using-resnet50
# thanks to Manas Tiwari

from os import listdir
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.config import list_physical_devices  # type: ignore  # noqa: F401
from tensorflow.keras import optimizers  # type: ignore
from tensorflow.keras.applications import VGG19, ResNet50  # type: ignore
from tensorflow.keras.callbacks import History  # type: ignore
from tensorflow.keras.layers import (  # type: ignore
    BatchNormalization,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
)
from tensorflow.keras.models import Sequential, load_model  # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array, load_img  # type: ignore
from tensorflow.keras.utils import to_categorical  # type: ignore

path_to_train_data = "/dcs/large/u2204489/faceforensics/train"
path_to_test_data = "/dcs/large/u2204489/faceforensics"
vgg_model_path = "/dcs/large/u2204489/vgg19.keras"
resnet_model_path = "/dcs/large/u2204489/resnet50.keras"
input_shape = (256, 256, 3)


def create_image_dataset() -> None:
    # loop over the training videos
    for type in ["fake", "real"]:
        videos = [
            f for f in listdir(f"{path_to_train_data}/{type}") if f.endswith(".mp4")
        ]
        for video in videos:
            count = 0
            # open the video and get the frame rate
            cap = cv2.VideoCapture(str(Path(path_to_train_data) / type / video))
            frame_rate = cap.get(5)
            while cap.isOpened():
                # attempt to read the video frame by frame
                frame_id = cap.get(1)
                ret, frame = cap.read()
                if not ret:
                    break
                # only process every nth frame
                if frame_id % (int(frame_rate) + 1) == 0:
                    # save the cropped face as a png file
                    cv2.imwrite(
                        str(Path(path_to_train_data) / type / f"{video}{count}.png"),
                        cv2.resize(frame, (256, 256)),
                    )
                    count += 1
            cap.release()


def save_graphs(history: History, epochs: int, model_type: str) -> None:
    # plot the accuracy and loss graphs
    plt.figure()
    epoch_list = list(range(1, epochs + 1))
    plt.plot(epoch_list, history.history["accuracy"], label="Train Accuracy")
    plt.plot(epoch_list, history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.xticks(epoch_list)
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title(f"Accuracy for {model_type}")
    plt.savefig(f"{model_type}-accuracy.png")

    plt.figure()
    plt.plot(epoch_list, history.history["loss"], label="Train Loss")
    plt.plot(epoch_list, history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.xticks(epoch_list)
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"Loss for {model_type}")
    plt.savefig(f"{model_type}-loss.png")


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

    X_train, X_val, Y_train, Y_val = train_test_split(  # noqa: N806
        X, Y, test_size=0.2, random_state=5
    )
    return X_train, X_val, Y_train, Y_val


def train_vgg_model() -> None:
    X_train, X_val, Y_train, Y_val = preprocess_data()  # noqa: N806

    # create the VGG19 model
    vgg19 = VGG19(include_top=False, weights="imagenet", input_shape=input_shape)
    vgg19.trainable = True

    # create the final model
    model = Sequential()
    model.add(vgg19)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(2, activation="softmax"))
    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizers.Adam(
            learning_rate=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False
        ),
        metrics=["accuracy"],
    )

    # train the model
    epochs = 20
    batch_size = 100
    history = model.fit(
        X_train,
        Y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, Y_val),
        verbose=1,
    )
    # save the model and graphs
    model.save(vgg_model_path)
    save_graphs(history, epochs, "vgg19")


def train_resnet_model() -> None:
    X_train, X_val, Y_train, Y_val = preprocess_data()  # noqa: N806

    resnet = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    model = Sequential()
    model.add(resnet)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(64, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(2, activation="softmax"))
    model.compile(
        optimizer=optimizers.Adam(),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    epochs = 8
    steps_per_epoch = 256
    validation_steps = 256
    history = model.fit(
        X_train,
        Y_train,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=(X_val, Y_val),
        validation_steps=validation_steps,
    )
    model.save(resnet_model_path)
    save_graphs(history, epochs, "resnet50")


def process_video(
    video_path: str,
    is_real: bool,
    vgg_model: Sequential,
    resnet_model: Sequential,
) -> tuple:
    # open the video and get the frame rate
    cap = cv2.VideoCapture(video_path)
    vgg_count = 0
    resnet_count = 0

    while cap.isOpened():
        # attempt to read the video frame by frame
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.resize(frame, (256, 256))
        # classify the frame as real or fake
        img = img_to_array(img).flatten() / 255.0
        img = img.reshape(-1, 256, 256, 3)
        vgg_prediction = vgg_model.predict(img, verbose=0)
        resnet_prediction = resnet_model.predict(img, verbose=0)
        if np.argmax(vgg_prediction) == 1:
            vgg_count += 1
        if np.argmax(resnet_prediction) == 1:
            resnet_count += 1

    cap.release()
    # a video is classified as fake if >10 frames are fake
    threshold = 100
    vgg_fake = vgg_count > threshold
    vgg_correct = vgg_fake != is_real

    resnet_fake = resnet_count > threshold
    resnet_correct = resnet_fake != is_real

    return vgg_correct, resnet_correct


def test_data() -> None:
    # [false positive, true negative, false negative, true positive]
    vgg_correct = [0, 0, 0, 0]
    resnet_correct = [0, 0, 0, 0]

    # pre-load models for performance
    vgg_model = load_model(vgg_model_path)
    resnet_model = load_model(resnet_model_path)

    # loop over the test videos
    video_paths = [
        (f"{path_to_test_data}/{type}/{video}", type == "real")
        for type in ["real", "fake"]
        for video in listdir(f"{path_to_test_data}/{type}")
    ]

    for video_path, is_real in video_paths:
        # process the video and update the counts
        vgg, resnet = process_video(video_path, is_real, vgg_model, resnet_model)
        vgg_correct[2 * is_real + vgg] += 1
        resnet_correct[2 * is_real + resnet] += 1

    # print the results
    print("VGG19")
    print(f"True Positives: {vgg_correct[3]}")
    print(f"True Negatives: {vgg_correct[1]}")
    print(f"False Positives: {vgg_correct[0]}")
    print(f"False Negatives: {vgg_correct[2]}")
    accuracy = (vgg_correct[1] + vgg_correct[3]) / sum(vgg_correct)
    print(f"Accuracy: {accuracy}")
    print()
    print("ResNet50")
    print(f"True Positives: {resnet_correct[3]}")
    print(f"True Negatives: {resnet_correct[1]}")
    print(f"False Positives: {resnet_correct[0]}")
    print(f"False Negatives: {resnet_correct[2]}")
    accuracy = (resnet_correct[1] + resnet_correct[3]) / sum(resnet_correct)
    print(f"Accuracy: {accuracy}")


if __name__ == "__main__":
    dataset_created = False
    if not dataset_created:
        print("Creating dataset")
        create_image_dataset()
    print("Dataset created")

    vgg_trained = False
    if not vgg_trained:
        # print(list_physical_devices("GPU"))
        train_vgg_model()
    print("VGG model trained")

    resnet_trained = False
    if not resnet_trained:
        train_resnet_model()
    print("ResNet model trained")

    test_data()
    print("done :)")
