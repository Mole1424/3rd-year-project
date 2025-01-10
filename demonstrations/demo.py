import sys

import cv2 as cv
import mediapipe as mp

path_to_dataset = "/dcs/large/u2204489/faceforensics"

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


def run_video():
    path_to_video = f"{path_to_dataset}/fake/09_13__kitchen_pan__21H6XSPE.mp4"
    video = cv.VideoCapture(path_to_video)
    ears = []

    with FaceLandmarker.create_from_options(options) as face_landmarker:
        while video.isOpened():
            success, frame = video.read()
            if not success:
                break

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
            cv.imshow("Frame", original_frame)
            cv.waitKey(1)

    video.release()


if __name__ == "__main__":
    arg = None
    try:
        arg = sys.argv[1]
    except IndexError:
        print('Please provide an argument: "video" or "live"')
    if arg == "video":
        run_video()
    elif arg == "live":
        run_live()
    else:
        print('Invalid argument. Please provide "video" or "live"')
