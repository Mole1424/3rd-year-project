# vgg19 model based on https://github.com/rahul9903/Deepfake/blob/main/Deepfake_detection.ipynb
# thanks to Pradyumna Yadav, Priyansh Sharma, and Sakshi Verma

from os import listdir
from pathlib import Path

import cv2
import dlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.config import list_physical_devices
from tensorflow.keras import optimizers
from tensorflow.keras.applications import VGG19
from tensorflow.keras.callbacks import History
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical

path_to_data = "/dcs/large/u2204489/faceforensics/train"
model_path = "vgg19.h5"
input_shape = (128, 128, 3)


def create_image_dataset() -> None:
    detector = dlib.get_frontal_face_detector()
    for type in ["fake", "real"]:
        videos = [f for f in listdir(f"{path_to_data}/{type}") if f.endswith(".mp4")]
        for video in videos:
            count = 0
            cap = cv2.VideoCapture(str(Path(path_to_data) / type / video))
            frame_rate = cap.get(5)
            while cap.isOpened():
                frame_id = cap.get(1)
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_id % (int(frame_rate) + 1) == 0:
                    face_rects, _, _ = detector.run(frame, 0)
                    for _, d in enumerate(face_rects):
                        x1 = d.left()
                        y1 = d.top()
                        x2 = d.right()
                        y2 = d.bottom()
                        crop = frame[y1:y2, x1:x2]
                        if crop.shape[0] > 0 and crop.shape[1] > 0:
                            cv2.imwrite(
                                str(Path(path_to_data) / type / f"{video}{count}.png"),
                                cv2.resize(crop, (128, 128)),
                            )
                            count += 1
            cap.release()


def save_graphs(history: History, epochs: int) -> None:
    plt.figure()
    epoch_list = list(range(1, epochs + 1))
    plt.plot(epoch_list, history.history["accuracy"], label="Train Accuracy")
    plt.plot(epoch_list, history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy for VGG19")
    plt.savefig("accuracy.png")

    plt.figure()
    plt.plot(epoch_list, history.history["loss"], label="Train Loss")
    plt.plot(epoch_list, history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss for VGG19")
    plt.savefig("loss.png")


def train_model() -> None:
    X, Y = [], []  # noqa: N806

    real_data = [f for f in listdir(f"{path_to_data}/real") if f.endswith(".png")]
    fake_data = [f for f in listdir(f"{path_to_data}/fake") if f.endswith(".png")]

    for img in real_data:
        X.append(img_to_array(load_img(f"{path_to_data}/real/{img}")).flatten() / 255.0)
        Y.append(0)
    for img in fake_data:
        X.append(img_to_array(load_img(f"{path_to_data}/fake/{img}")).flatten() / 255.0)
        Y.append(1)

    X = np.array(X)  # noqa: N806
    Y = to_categorical(Y, 2)  # noqa: N806

    X = X.reshape(-1, 128, 128, 3)  # noqa: N806

    X_train, X_val, Y_train, Y_val = train_test_split(  # noqa: N806
        X, Y, test_size=0.2, random_state=5
    )

    vgg19 = VGG19(include_top=False, weights="imagenet", input_shape=input_shape)
    vgg19.trainable = True

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
    model.save(model_path)
    save_graphs(history, epochs)


if __name__ == "__main__":
    dataset_created = True
    if not dataset_created:
        print("Creating dataset")
        create_image_dataset()
    print("Dataset created")

    model_trained = True
    if not model_trained:
        print(list_physical_devices("GPU"))
        train_model()
    print("Model trained")
