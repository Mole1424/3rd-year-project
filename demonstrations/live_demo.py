import sys
import threading

import cv2 as cv
import matplotlib.pyplot as plt
import mediapipe as mp
from matplotlib.animation import FuncAnimation

mediapipe_model_path = "face_landmarker.task"
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=mediapipe_model_path),
    running_mode=VisionRunningMode.VIDEO,
)

ears = []


def plot_ears() -> None:
    figure, axis = plt.subplots()
    x_data, y_data = [], []

    def update(frame) -> None:  # noqa: ANN001, ARG001
        if len(ears) > len(x_data):
            x_data.append(len(x_data))
            y_data.append(ears[len(x_data) - 1])
        axis.clear()
        axis.set_xlabel("Time")
        axis.set_ylabel("EAR")
        axis.plot(x_data, y_data, label="EAR")

    _ = FuncAnimation(figure, update, interval=2)  # type: ignore
    plt.show()


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


def process_video(video: cv.VideoCapture, live: bool) -> None:
    with FaceLandmarker.create_from_options(options) as face_landmarker:
        while video.isOpened():
            success, frame = video.read()
            if not success:
                break

            frame_height, frame_width, _ = frame.shape
            if live:
                frame = cv.resize(frame, (frame_width * 2, frame_height * 2))
                frame_height, frame_width, _ = frame.shape
            original_frame = frame.copy()

            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            face_landmarker_result = face_landmarker.detect_for_video(
                mp_image, int(video.get(cv.CAP_PROP_POS_MSEC))
            )

            if not face_landmarker_result.face_landmarks:
                continue

            face_landmarks = face_landmarker_result.face_landmarks[0]
            left_eye_indices = [33, 160, 158, 133, 153, 144]
            right_eye_indices = [362, 385, 387, 263, 373, 380]
            left_eye_landmarks = [face_landmarks[i] for i in left_eye_indices]
            right_eye_landmarks = [face_landmarks[i] for i in right_eye_indices]

            left_ear = calculate_ear(left_eye_landmarks)
            right_ear = calculate_ear(right_eye_landmarks)
            ears.append((left_ear + right_ear) / 2)

            for landmark in right_eye_landmarks + left_eye_landmarks:
                cv.circle(
                    original_frame,
                    (int(landmark.x * frame_width), int(landmark.y * frame_height)),
                    radius=3,
                    color=(0, 0, 255),
                    thickness=-1,
                )
            # add ear to top right of frame
            cv.putText(
                original_frame,
                f"EAR: {ears[-1]:.2f}",
                (frame_width - 200, 50),
                cv.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv.LINE_AA,
            )
            cv.imshow("Frame", original_frame)
            cv.waitKey(1)
            if live and cv.waitKey(1) & 0xFF == ord("q"):
                break
    video.release()


if __name__ == "__main__":
    arg = None
    try:
        arg = sys.argv[1]
    except IndexError:
        print('Please provide an argument: "video" or "live"')

    source = 0 if arg == "live" else "09_13__kitchen_pan__21H6XSPE.mp4"
    video = cv.VideoCapture(source)
    threading.Thread(target=process_video, args=(video, arg == "live")).start()
    plot_ears()
