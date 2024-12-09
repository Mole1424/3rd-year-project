import mediapipe as mp
import cv2 as cv

# get video as numpy array
path_to_video = "/dcs/large/u2204489/datasets/"
video = cv.VideoCapture(path_to_video)
frame_rate = video.get(cv.CAP_PROP_FPS)

model_path = "face_landmarker.task"
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
)

with FaceLandmarker.create_from_options(options) as landmarker:
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        timestamp = video.get(cv.CAP_PROP_POS_MSEC)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        face_landmarker_result = landmarker.detect_for_video(mp_image, timestamp)

        # show image with landmarks
        cv.imshow("Face Landmarks", face_landmarker_result.image)
