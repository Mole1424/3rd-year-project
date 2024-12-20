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
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical

path_to_train_data = "/dcs/large/u2204489/faceforensics/train"
path_to_test_data = "/dcs/large/u2204489/faceforensics"
model_path = "vgg19.h5"
input_shape = (128, 128, 3)


def create_image_dataset() -> None:
    detector = dlib.get_frontal_face_detector()
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
                    # detect faces in the frame and iterate over them
                    face_rects, _, _ = detector.run(frame, 0)
                    for _, d in enumerate(face_rects):
                        # crop the face and resize it to 128x128
                        x1 = d.left()
                        y1 = d.top()
                        x2 = d.right()
                        y2 = d.bottom()
                        crop = frame[y1:y2, x1:x2]
                        if crop.shape[0] > 0 and crop.shape[1] > 0:
                            # save the cropped face as a png file
                            cv2.imwrite(
                                str(
                                    Path(path_to_train_data)
                                    / type
                                    / f"{video}{count}.png"
                                ),
                                cv2.resize(crop, (128, 128)),
                            )
                            count += 1
            cap.release()


def save_graphs(history: History, epochs: int) -> None:
    # plot the accuracy and loss graphs
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

    X = X.reshape(-1, 128, 128, 3)  # noqa: N806

    X_train, X_val, Y_train, Y_val = train_test_split(  # noqa: N806
        X, Y, test_size=0.2, random_state=5
    )

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
    model.save(model_path)
    save_graphs(history, epochs)


def process_video(
    video_path: str,
    is_real: bool,
    model: Sequential,
    detector: dlib.fhog_object_detector,
) -> bool:
    # open the video and get the frame rate
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(5)
    count = 0
    while cap.isOpened():
        # attempt to read the video frame by frame
        frame_id = cap.get(1)
        ret, frame = cap.read()
        if not ret:
            break

        # only process every nth frame
        if frame_id % (int(frame_rate) + 1) == 0:
            # detect faces in the frame and iterate over them
            face_rects, _, _ = detector.run(frame, 0)
            for _, d in enumerate(face_rects):
                # crop the face and resize it to 128x128
                x1 = d.left()
                y1 = d.top()
                x2 = d.right()
                y2 = d.bottom()
                crop = frame[y1:y2, x1:x2]
                if crop.shape[0] > 0 and crop.shape[1] > 0:
                    img = cv2.resize(crop, (128, 128))
                    # classify the frame as real or fake
                    img = img_to_array(img).flatten() / 255.0
                    img = img.reshape(-1, 128, 128, 3)
                    prediction = model.predict(img, verbose=0)
                    if np.argmax(prediction) == 1:
                        count += 1
    cap.release()
    # a video is classified as fake if >10 frames are fake
    threshold = 10
    fake = count > threshold
    correct = fake != is_real
    print(
        f"video: {video_path}, is_real: {is_real}, count: {count}, correct: {correct}"
    )
    return correct


def test_data() -> None:
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    # pre-load models for performance
    model = load_model(model_path)
    detector = dlib.get_frontal_face_detector()

    # loop over the test videos
    video_paths = [
        (f"{path_to_test_data}/{type}/{video}", type == "real")
        for type in ["real", "fake"]
        for video in listdir(f"{path_to_test_data}/{type}")
    ]

    for video_path, is_real in video_paths:
        # process the video and update the counts
        is_correct = process_video(video_path, is_real, model, detector)
        true_positives += is_real and is_correct
        false_negatives += is_real and not is_correct
        true_negatives += not is_real and is_correct
        false_positives += not is_real and not is_correct

    print(f"True Positives: {true_positives}")
    print(f"True Negatives: {true_negatives}")
    print(f"False Positives: {false_positives}")
    print(f"False Negatives: {false_negatives}")
    accuracy = (true_positives + true_negatives) / (
        true_positives + true_negatives + false_positives + false_negatives
    )
    print(f"Accuracy: {accuracy}")


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

    test_data()
    print("done :)")
