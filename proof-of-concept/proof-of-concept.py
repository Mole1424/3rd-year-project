from math import floor
from os import listdir
from typing import Tuple

import cv2 as cv
import dlib
import mediapipe as mp
import numpy as np
from cleverhans.tf2.attacks.carlini_wagner_l2 import CarliniWagnerL2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

path_to_dataset = "/dcs/large/u2204489/faceforensics"
vgg19_model = load_model("/dcs/large/u2204489/vgg19.h5")
detector = dlib.get_frontal_face_detector()


def process_frame_vgg(frame: np.ndarray) -> np.ndarray:
    # detect faces in the frame and select the first one
    faces, _, _ = detector.run(frame, 0)
    if len(faces) == 0:
        return np.array([])
    face = faces[0]

    # crop the face out
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    crop = frame[y1:y2, x1:x2]
    if crop.shape[0] == 0 and crop.shape[1] == 0:
        return np.array([])

    # resize the face to 128x128 and classify it, returning the prediction
    img = cv.resize(crop, (128, 128))
    img = img_to_array(img).flatten() / 255.0
    img = img.reshape(-1, 128, 128, 3)
    return vgg19_model.predict(img, verbose=0)


mediapipe_model_path = "/dcs/large/u2204489/face_landmarker.task"
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=mediapipe_model_path),
    running_mode=VisionRunningMode.VIDEO,
)


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


def process_video(video_path: str, is_real: bool) -> Tuple[bool, bool, bool]:
    # get the video and metadata
    video = cv.VideoCapture(video_path)
    frame_rate = video.get(cv.CAP_PROP_FPS)
    video_length = int(video.get(cv.CAP_PROP_FRAME_COUNT) / frame_rate)

    # initalise trackers for vgg and blinking
    ears = []
    vgg_fake_frames = 0

    with FaceLandmarker.create_from_options(options) as landmarker:
        while video.isOpened():
            # read the video frame by frame
            success, frame = video.read()
            if not success:
                break

            # classify the frame using the VGG19 model
            vgg_prediction = process_frame_vgg(frame)
            if len(vgg_prediction) == 0 or np.argmax(vgg_prediction) == 1:
                vgg_fake_frames += 1

            # get the timestamp of the frame and detect face landmarks
            timestamp = int(video.get(cv.CAP_PROP_POS_MSEC))
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            face_landmarker_result = landmarker.detect_for_video(mp_image, timestamp)

            if not face_landmarker_result.face_landmarks:
                continue

            # get the landmarks of the left and right eye
            face_landmarks = face_landmarker_result.face_landmarks[0]
            left_eye_indices = [33, 160, 158, 133, 153, 144]
            right_eye_indices = [362, 385, 387, 263, 373, 380]
            left_eye_landmarks = [face_landmarks[i] for i in left_eye_indices]
            right_eye_landmarks = [face_landmarks[i] for i in right_eye_indices]

            # calculate eye aspect ratio (EAR)
            left_eye_ear = calculate_ear(left_eye_landmarks)
            right_eye_ear = calculate_ear(right_eye_landmarks)
            ear = (left_eye_ear + right_eye_ear) / 2
            ears.append(ear)
    video.release()

    # classify video using vgg
    # if >10 frames are faked, classify as fake
    vgg_threshold = 10
    vgg_correct = (vgg_fake_frames <= vgg_threshold) == is_real

    minimum_frames = 30
    if len(ears) < minimum_frames:
        return vgg_correct, not is_real, is_real

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

    min_blinks = floor(14 * video_length / 60)
    blink_correct = (min_blinks <= blink_count) == is_real

    return vgg_correct, blink_correct, is_real


def print_results(model_name: str, correct: list) -> None:
    print(f"Model: {model_name}")
    print(f"True Positives: {correct[3]}")
    print(f"True Negatives: {correct[1]}")
    print(f"False Positives: {correct[0]}")
    print(f"False Negatives: {correct[2]}")
    accuracy = (correct[1] + correct[3]) / sum(correct)
    print(f"Accuracy: {accuracy}")


def main() -> None:
    video_paths = [
        (f"{path_to_dataset}/{type}/{video}", type == "real")
        for type in ["real", "fake"]
        for video in listdir(f"{path_to_dataset}/{type}")
    ]

    results = []
    for video_path, is_real in video_paths:
        results.append(process_video(video_path, is_real))

    # [false positive, true negative, false negative, true positive]
    vgg_correct = [0, 0, 0, 0]
    blink_correct = [0, 0, 0, 0]

    for vgg, blink, is_real in results:
        vgg_correct[2 * is_real + vgg] += 1
        blink_correct[2 * is_real + blink] += 1

    print_results("VGG19", vgg_correct)
    print_results("Blink Detection", blink_correct)


if __name__ == "__main__":
    main()
