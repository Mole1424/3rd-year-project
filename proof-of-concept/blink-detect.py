import mediapipe as mp
import cv2 as cv

path_to_video = "sample.mp4"
video = cv.VideoCapture(path_to_video)

model_path = "face_landmarker.task"
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# default options for face landmark detection
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
)

EARs = []

with FaceLandmarker.create_from_options(options) as landmarker:
    while video.isOpened():
        # read the video frame by frame
        success, frame = video.read()
        if not success:
            break

        # get the timestamp of the frame and detect face landmarks
        timestamp = int(video.get(cv.CAP_PROP_POS_MSEC))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        face_landmarker_result = landmarker.detect_for_video(mp_image, timestamp)

        if not face_landmarker_result.face_landmarks:
            print(f"No face detected at {timestamp}ms")
            continue

        # Assuming the first face's landmarks are required
        face_landmarks = face_landmarker_result.face_landmarks[0]

        # face landmark indices for left and right eyes
        left_eye_indices = [33, 160, 158, 133, 153, 144]
        right_eye_indices = [362, 385, 387, 263, 373, 380]

        # get eye landmarks
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

print(EARs)
