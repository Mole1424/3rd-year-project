from math import floor
from multiprocessing import Pool
from os import listdir
from pathlib import Path
from typing import Tuple, Union

import cv2 as cv
import mediapipe as mp
import numpy as np
import tensorflow as tf
from foolbox import TensorFlowModel  # type: ignore
from foolbox.attacks import LinfFastGradientAttack  # type: ignore
from foolbox.criteria import Misclassification
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array  # type: ignore

path_to_dataset = "/dcs/large/u2204489/faceforensics"

# load the VGG19 model and create a foolbox model
vgg19_model = load_model("/dcs/large/u2204489/vgg19.keras")
foolbox_model = TensorFlowModel(vgg19_model, bounds=(0, 255))

# load the ResNet50 model
resnet_model = load_model("/dcs/large/u2204489/resnet50.keras")

# load the mediapipe model with options
mediapipe_model_path = "/dcs/large/u2204489/face_landmarker.task"
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=mediapipe_model_path),
    running_mode=VisionRunningMode.VIDEO,
)


# given a list of eye landmarks, calculate the eye aspect ratio (EAR)
def calculate_ear(eye_landmarks: list) -> float:
    p2_p6 = ((eye_landmarks[1].x - eye_landmarks[5].x) ** 2) + (
        (eye_landmarks[1].y - eye_landmarks[5].y) ** 2
    )
    p3_p5 = ((eye_landmarks[2].x - eye_landmarks[4].x) ** 2) + (
        (eye_landmarks[2].y - eye_landmarks[4].y) ** 2
    )
    p1_p4 = ((eye_landmarks[0].x - eye_landmarks[3].x) ** 2) + (
        (eye_landmarks[0].y - eye_landmarks[3].y) ** 2
    )
    return (p2_p6 + p3_p5) / (2.0 * p1_p4)


# calculate the eye aspect ratio (EAR) for a frame (if possible)
def process_frame_blink(
    frame: np.ndarray,
    timestamp: int,
    landmarker: mp.tasks.vision.FaceLandmarker,  # type: ignore
) -> float:
    # pre-processing of frame
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    face_landmarker_result = landmarker.detect_for_video(mp_image, timestamp)

    if not face_landmarker_result.face_landmarks:
        return -1

    # get the landmarks of the left and right eye
    face_landmarks = face_landmarker_result.face_landmarks[0]
    left_eye_indices = [33, 160, 158, 133, 153, 144]
    right_eye_indices = [362, 385, 387, 263, 373, 380]
    left_eye_landmarks = [face_landmarks[i] for i in left_eye_indices]
    right_eye_landmarks = [face_landmarks[i] for i in right_eye_indices]

    # calculate eye aspect ratio (EAR)
    left_eye_ear = calculate_ear(left_eye_landmarks)
    right_eye_ear = calculate_ear(right_eye_landmarks)
    return (left_eye_ear + right_eye_ear) / 2


# create a plot of the EARs
def create_ear_plot(  # noqa: PLR0913
    ears: list,
    threshold: float,
    average_ear: float,
    min_blinks: int,
    blink_count: int,
    is_correct: int,
    video_name: str,
    perturbated: bool,
) -> None:
    # save graph of EARs
    plt.figure()
    plt.plot(ears, label="EAR")
    plt.axhline(
        y=threshold, color="r", linestyle="--", label=f"Threshold ({threshold:.2f})"
    )
    plt.axhline(
        y=average_ear,
        color="g",
        linestyle="--",
        label=f"Average ({average_ear:.2f})",
    )
    plt.legend()
    plt.xlabel("Frame")
    plt.ylabel("EAR")
    plt.title(
        f"Fake: <={min_blinks}, Actual: {blink_count}, Correct: {is_correct}, Perturbated: {perturbated}"  # noqa: E501
    )
    plt.savefig(f"EARs/{video_name}_{perturbated}.png")
    plt.close()


# given a list of EARs, calculate if the video is real or fake
def process_ears(
    ears: list, video_length: int, is_real: bool, name: str, perturbated: bool
) -> Union[bool, int]:
    # need 30 frames for accurate detection
    minimum_frames = 30
    if len(ears) < minimum_frames:
        return 2

    # calculate the threshold for the EAR
    average_ear = sum(ears) / len(ears)
    standard_deviation = (
        sum([(ear - average_ear) ** 2 for ear in ears]) / len(ears)
    ) ** 0.5
    threshold = min(ears) + 0.5 * standard_deviation

    # calculate number of blinks
    blink_count = 0
    blink_occuring = False
    for ear in ears:
        if ear < threshold:
            if not blink_occuring:
                blink_count += 1
                blink_occuring = True
        else:
            blink_occuring = False

    # average human blinks 14 times per minute
    min_blinks = floor(14 * video_length / 60)
    correct = (min_blinks <= blink_count) == is_real

    # don't create plot for real and perturbated videos (as they are not perturbated)
    if not (is_real and perturbated):
        create_ear_plot(
            ears,
            threshold,
            average_ear,
            min_blinks,
            blink_count,
            correct,
            name,
            perturbated,
        )
    return correct


# classify a frame using the VGG19 and ResNet50 model
def process_frame_models(frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # resize the face to 256x256 and normalise the pixel values
    frame = cv.resize(frame, (256, 256))
    frame = img_to_array(frame).flatten() / 255.0
    frame = frame.reshape(-1, 256, 256, 3)
    # classify using vgg19
    return vgg19_model.predict(frame, verbose=0), resnet_model.predict(frame, verbose=0)


# add guided noise to a frame
def perturbate_frame(frame: np.ndarray) -> np.ndarray:
    # save the original frame
    original_dimensions = frame.shape
    original_frame = frame

    # pre-process frame for the VGG19 model
    frame = cv.resize(frame, (256, 256))
    frame = img_to_array(frame) / 255.0
    frame = np.expand_dims(frame, axis=0)
    frame = tf.convert_to_tensor(frame, dtype=tf.float32)  # type: ignore

    # generate noise using the fast gradient sign attack
    attack = LinfFastGradientAttack()
    epsilon = 0.1
    noise = attack.run(
        foolbox_model,
        frame,
        # attempt to misclassify the frame as real
        Misclassification(tf.constant([1], dtype=tf.int32)),
        epsilon=epsilon,
    )

    # post-process the noise and add it to the original frame
    noise = np.squeeze(noise, axis=0)
    noise = np.clip(noise, 0, 1)
    noise = (noise * 255).astype(np.uint8)
    noise = cv.resize(noise, (original_dimensions[1], original_dimensions[0]))
    return cv.addWeighted(original_frame, 1, noise, 1, 0)


# process a video and return the classification results
def process_video(
    video_path: str, is_real: bool
) -> Tuple[bool, bool, int, bool, bool, int, bool]:
    print(f"Processing video: {video_path}")
    # get the video and metadata
    video = cv.VideoCapture(video_path)
    frame_rate = video.get(cv.CAP_PROP_FPS)
    video_length = int(video.get(cv.CAP_PROP_FRAME_COUNT) / frame_rate)

    # initalise trackers for vgg, resnet, and blinking
    ears, perturbated_ears = [], []
    vgg_fake_frames, perturbated_vgg_fake_frames = 0, 0
    resnet_fake_frames, perturbated_resnet_fake_frames = 0, 0

    with (
        FaceLandmarker.create_from_options(options) as landmarker,
        FaceLandmarker.create_from_options(options) as perturbated_landmarker,
    ):
        while video.isOpened():
            # read the video frame by frame
            success, frame = video.read()
            if not success:
                break

            # perturbate the frame only if fake
            perturbated_frame = frame if is_real else perturbate_frame(frame)

            # classify the frame using the VGG19 and ResNet50 model
            vgg_prediction, resnet_prediction = process_frame_models(frame)
            if len(vgg_prediction) == 0 or np.argmax(vgg_prediction) == 1:
                vgg_fake_frames += 1
            if len(resnet_prediction) == 0 or np.argmax(resnet_prediction) == 1:
                resnet_fake_frames += 1
            perturbated_vgg_prediction, perturbated_resnet_prediction = (
                process_frame_models(perturbated_frame)
            )
            if (
                len(perturbated_vgg_prediction) == 0
                or np.argmax(perturbated_vgg_prediction) == 1
            ):
                perturbated_vgg_fake_frames += 1
            if (
                len(perturbated_resnet_prediction) == 0
                or np.argmax(perturbated_resnet_prediction) == 1
            ):
                perturbated_resnet_fake_frames += 1

            # get the timestamp of the frame and detect face landmarks
            timestamp = int(video.get(cv.CAP_PROP_POS_MSEC))
            ear = process_frame_blink(frame, timestamp, landmarker)
            if ear != -1:
                ears.append(ear)
            perturbared_ear = process_frame_blink(
                perturbated_frame, timestamp, perturbated_landmarker
            )
            if perturbared_ear != -1:
                perturbated_ears.append(perturbared_ear)
    video.release()

    # classify video using vgg
    # if >100 frames are faked, classify as fake
    vgg_threshold = 100
    vgg_correct = (vgg_fake_frames <= vgg_threshold) == is_real
    perturbated_vgg_correct = (perturbated_vgg_fake_frames <= vgg_threshold) == is_real

    # classify video using resnet
    resnet_threshold = 100
    resnet_correct = (resnet_fake_frames <= resnet_threshold) == is_real
    perturbated_resnet_correct = (
        perturbated_resnet_fake_frames <= resnet_threshold
    ) == is_real

    # classify video using blink detection
    blink_correct = process_ears(
        ears, video_length, is_real, video_path.split("/")[-1], False
    )
    perturbated_blink_correct = process_ears(
        perturbated_ears, video_length, is_real, video_path.split("/")[-1], True
    )

    return (
        vgg_correct,
        resnet_correct,
        blink_correct,
        perturbated_vgg_correct,
        perturbated_resnet_correct,
        perturbated_blink_correct,
        is_real,
    )


# print the results of the classification
def print_results(model_name: str, correct: list) -> None:
    print(f"Model: {model_name}")
    print(f"True Positives: {correct[3]}")
    print(f"True Negatives: {correct[1]}")
    print(f"False Positives: {correct[0]}")
    print(f"False Negatives: {correct[2]}")
    accuracy = (correct[1] + correct[3]) / (
        correct[0] + correct[1] + correct[2] + correct[3]
    )
    print(f"Accuracy: {accuracy}")
    if len(correct) == 6:  # noqa: PLR2004
        print(f"Unknown Fake: {correct[4]}")
        print(f"Unknown Real: {correct[5]}")
        accuracy = (correct[1] + correct[3] + correct[4]) / sum(correct)
        print(f"Accuracy (assuming unkown as fake): {accuracy}")
    print("========================================")


def main() -> None:
    # clear the EARs directory
    for file in listdir("EARs"):
        Path(f"EARs/{file}").unlink()

    video_paths = [
        (f"{path_to_dataset}/{type}/{video}", type == "real")
        for type in ["fake", "real"]
        for video in listdir(f"{path_to_dataset}/{type}")
    ]

    # process videos in parallel
    # results = [process_video(*video_path) for video_path in video_paths]
    with Pool() as pool:
        results = pool.starmap(process_video, video_paths)

    # [false positive, true negative, false negative, true positive]
    # for original and perturbated
    vgg_correct = [0, 0, 0, 0, 0, 0, 0, 0]
    resnet_correct = [0, 0, 0, 0, 0, 0, 0, 0]
    # [..., unkown fake, unkown real]
    blink_correct = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # aggregate the results
    for (
        vgg,
        resnet,
        blink,
        perturbated_vgg,
        perturbated_resnet,
        perturbated_blink,
        is_real,
    ) in results:
        vgg_correct[2 * is_real + vgg] += 1
        vgg_correct[4 + 2 * is_real + perturbated_vgg] += 1
        resnet_correct[2 * is_real + resnet] += 1
        resnet_correct[4 + 2 * is_real + perturbated_resnet] += 1

        if blink == 2:  # noqa: PLR2004
            blink_correct[4 + is_real] += 1
        else:
            blink_correct[2 * is_real + blink] += 1
        if perturbated_blink == 2:  # noqa: PLR2004
            blink_correct[10 + is_real] += 1
        else:
            blink_correct[6 + 2 * is_real + perturbated_blink] += 1

    print_results("VGG19", vgg_correct[:4])
    print_results("Perturbated VGG19", vgg_correct[4:])
    print_results("ResNet50", resnet_correct[:4])
    print_results("Perturbated ResNet50", resnet_correct[4:])
    print_results("Blink Detection", blink_correct[6:])
    print_results("Perturbated Blink Detection", blink_correct[:6])


if __name__ == "__main__":
    main()
