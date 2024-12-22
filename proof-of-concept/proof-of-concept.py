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

vgg19_model = load_model("/dcs/large/u2204489/vgg19.h5")
detector = dlib.get_frontal_face_detector()


def process_video_vgg(video_path: str, is_real: bool) -> bool:
    # open the video and get the frame rate
    video = cv.VideoCapture(video_path)
    frame_rate = video.get(5)

    # count the number of frames that are classified as fake
    fake_frames = 0

    while video.isOpened():
        # attempt to read the video frame by frame
        frame_id = video.get(1)
        success, frame = video.read()
        if not success:
            continue

        # only process every nth frame
        if frame_id % (int(frame_rate) + 1) == 0:
            # detect faces in the frame and iterate over them
            faces, _, _ = detector.run(frame, 0)
            for _, d in enumerate(faces):
                # crop the face and resize it to 128x128
                x1 = d.left()
                y1 = d.top()
                x2 = d.right()
                y2 = d.bottom()
                crop = frame[y1:y2, x1:x2]
                if crop.shape[0] > 0 and crop.shape[1] > 0:
                    img = cv.resize(crop, (128, 128))

                    # classify the frame as real or fake
                    img = img_to_array(img).flatten() / 255.0
                    img = img.reshape(-1, 128, 128, 3)
                    prediction = vgg19_model.predict(img, verbose=0)
                    if np.argmax(prediction) == 1:
                        fake_frames += 1
    video.release()

    # a video is classified as fake if >10 frames are fake
    threshold = 10
    fake = fake_frames > threshold
    correct = fake != is_real
    print(
        f"video: {video_path}, is_real: {is_real}, fake_frames: {fake_frames}, correct: {correct}"
    )
    return correct


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


def process_video_blink(video_path: str, is_real: bool) -> bool:
    ears = []
    # open the video and get the length
    video = cv.VideoCapture(video_path)
    video_length = int(video.get(cv.CAP_PROP_FRAME_COUNT)) / video.get(cv.CAP_PROP_FPS)

    with FaceLandmarker.create_from_options(options) as landmarker:
        while video.isOpened():
            # read the video frame by frame
            success, frame = video.read()
            if not success:
                break

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

    minimum_frames = 30
    # if cannot reliably detect blinks, assume the video is fake
    if len(ears) <= minimum_frames:
        return not is_real

    # calculate blink threshold
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

    # the average human blinks around 14 times per minute
    min_blinks = floor(14 * video_length / 60)
    is_correct = (min_blinks <= blink_count) == is_real
    print(
        f"video: {video_path}, is_real: {is_real}, blink_count: {blink_count}, is_correct: {is_correct}"
    )
    return is_correct


path_to_dataset = "/dcs/large/u2204489/faceforensics"


def print_results(model_name: str, correct: list) -> None:
    print(f"Model: {model_name}")
    print(f"True Positives: {correct[0]}")
    print(f"True Negatives: {correct[2]}")
    print(f"False Positives: {correct[3]}")
    print(f"False Negatives: {correct[1]}")
    accuracy = (correct[0] + correct[2]) / sum(correct)
    print(f"Accuracy: {accuracy}")


def process_video(video_path: str, is_real: bool) -> Tuple[bool, bool, bool]:
    vgg = process_video_vgg(video_path, is_real)
    blink = process_video_blink(video_path, is_real)
    return (vgg, blink, is_real)


def main() -> None:
    video_paths = [
        (f"{path_to_dataset}/{type}/{video}", type == "real")
        for type in ["real", "fake"]
        for video in listdir(f"{path_to_dataset}/{type}")
    ]

    results = []
    for video_path, is_real in video_paths:
        results.append(process_video(video_path, is_real))

    # [true_positives, false_negatives, true_negatives, false_positives]
    vgg_correct = [0, 0, 0, 0]
    blink_correct = [0, 0, 0, 0]

    for vgg, blink, is_real in results:
        vgg_correct[2 * is_real + vgg] += 1
        blink_correct[2 * is_real + blink] += 1

    print_results("VGG19", vgg_correct)
    print_results("Blink Detection", blink_correct)


if __name__ == "__main__":
    print("running main")
    main()
