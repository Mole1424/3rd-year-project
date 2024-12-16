import mediapipe as mp
import cv2 as cv
from os import listdir, remove
from concurrent.futures import ThreadPoolExecutor
from matplotlib import pyplot as plt
from math import floor

model_path = "face_landmarker.task"
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO
)

def process_video(video_path, is_real):
    """Attempt to detect if a video is real or fake based on the number of blinks."""
    print(f"Processing {video_path}")
    EARs = []
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
            def calculate_EAR(eye_landmarks):
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

            left_eye_ear = calculate_EAR(left_eye_landmarks)
            right_eye_ear = calculate_EAR(right_eye_landmarks)
            EAR = (left_eye_ear + right_eye_ear) / 2
            EARs.append(EAR)

    video.release()

    if len(EARs) <= 30:
        print(f"Failed to process {video_path}")
        return (is_real, 0, 0, 1)

    # calculate blink threshold
    average_EAR = sum(EARs) / len(EARs)
    standard_deviation = (
        sum([(ear - average_EAR) ** 2 for ear in EARs]) / len(EARs)
    ) ** 0.5
    threshold = min(EARs) + 0.5 * standard_deviation

    # calculate number of blinks
    blink_count = 0
    blink_occuring = False
    for ear in EARs:
        if ear < threshold:
            if not blink_occuring:
                blink_count += 1
                blink_occuring = True
        else:
            blink_occuring = False

    # the average human blinks around 14 times per minute
    min_blinks = floor(14 * video_length / 60)
    is_correct = (min_blinks <= blink_count) == is_real

    # save graph of EARs
    plt.figure()
    plt.plot(EARs, label="EAR")
    plt.axhline(
        y=threshold, color="r", linestyle="--", label=f"Threshold ({threshold:.2f})"
    )
    plt.axhline(
        y=average_EAR,
        color="g",
        linestyle="--",
        label=f"Average ({average_EAR:.2f})",
    )
    plt.legend()
    plt.xlabel("Frame")
    plt.ylabel("EAR")
    plt.title(f"Expected: <={min_blinks}, Actual: {blink_count}, Correct: {is_correct}")
    plt.savefig(f"EARs/{video_path.split('/')[-1]}.png")
    return (is_real, 1 if is_correct else 0, 0 if is_correct else 1, 0)

def process_dataset():
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    unkown_real = 0
    unkown_fake = 0

    path_to_dataset = "/dcs/large/u2204489/faceforensics"

    # get the paths of all the videos in the dataset
    video_paths = [
        (f"{path_to_dataset}/{type}/{video}", type == "real")
        for type in ["real", "fake"]
        for video in listdir(f"{path_to_dataset}/{type}")
    ]

    results = []
    # process the videos in parallel
    with ThreadPoolExecutor() as executor:
        results = executor.map(lambda args: process_video(*args), video_paths)

    # process the videos in serial (slower but avoids parallelism issues in matplotlib)
    # for video_path, is_real in video_paths:
    #     results.append(process_video(video_path, is_real))

    # count the number of correct and incorrect results
    for real, correct, incorrect, unkown in results:
        if real:
            true_positives += correct
            false_negatives += incorrect
            unkown_real += unkown
        else:
            true_negatives += correct
            false_positives += incorrect
            unkown_fake += unkown

    print(f"True Positives: {true_positives}")
    print(f"True Negatives: {true_negatives}")
    print(f"False Positives: {false_positives}")
    print(f"False Negatives: {false_negatives}")
    print(f"Unknown Real: {unkown_real}")
    print(f"Unknown Fake: {unkown_fake}")
    print(f"Accuracy: {(true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)}")

if __name__ == "__main__":
    # remove content of EARs directory
    for file in listdir("EARs"):
        remove(f"EARs/{file}")
    process_dataset()
