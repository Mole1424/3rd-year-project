import json
from pathlib import Path

import cv2 as cv
import numpy as np
import tensorflow as tf
from ear_analysis import EarAnalysis
from eye_detection import EyeLandmarker
from mtcnn import MTCNN  # type: ignore
from tensorflow.keras.models import Model, load_model  # type: ignore


def create_dataset(path_to_dataset: str) -> list[tuple[str, int]]:
    """creates a dataset of video paths and labels"""
    videos = list(map(str, Path(path_to_dataset).rglob("*.mp4")))
    labels = [int("real" in str(video)) for video in videos]
    return list(zip(videos, labels))

def save_results(results: dict, path: str) -> None:
    """saves the results to a json file"""
    with Path(path).open("w") as f:
        json.dump(results, f, indent=4)

def load_results(path: str) -> dict:
    """loads the results from a json file"""
    with Path(path).open("r") as f:
        return json.load(f)

def get_custom_model(
    hrnet: bool, base_model_path: str, ear_path: str
) -> tuple[EyeLandmarker, EarAnalysis]:
    """loads custom model"""
    if hrnet:
        # load hrnet config
        hrnet_config = {
            "NUM_JOINTS": 68,
            "FINAL_CONV_KERNEL": 1,
            "STAGE2": {
                "NUM_MODULES": 1,
                "NUM_BRANCHES": 2,
                "NUM_BLOCKS": [4, 4],
                "NUM_CHANNELS": [18, 36],
            },
            "STAGE3": {
                "NUM_MODULES": 4,
                "NUM_BRANCHES": 3,
                "NUM_BLOCKS": [4, 4, 4],
                "NUM_CHANNELS": [18, 36, 72],
            },
            "STAGE4": {
                "NUM_MODULES": 3,
                "NUM_BRANCHES": 4,
                "NUM_BLOCKS": [4, 4, 4, 4],
                "NUM_CHANNELS": [18, 36, 72, 144],
            },
        }
        landmarker = EyeLandmarker(hrnet_config, base_model_path + "hrnet.weights.h5")
    else:
        # otherwise use PFLD
        landmarker = EyeLandmarker(None, base_model_path + "pfld.keras")

    ear_analyser = EarAnalysis(base_model_path + ear_path, None, "", "", "")

    return landmarker, ear_analyser

def calculate_ears(points: np.ndarray) -> np.ndarray:
    """calcualte eye aspect ratio"""
    # calculate for each eye
    p2_p6 = np.linalg.norm(points[:,1] - points[:,5], axis=1)
    p3_p5 = np.linalg.norm(points[:,2] - points[:,4], axis=1)
    p1_p4 = np.linalg.norm(points[:,0] - points[:,3], axis=1)
    ear_l = np.clip((p2_p6 + p3_p5) / (2.0 * p1_p4), 0, 1)

    p8_p12 = np.linalg.norm(points[:,7] - points[:,11], axis=1)
    p9_p11 = np.linalg.norm(points[:,8] - points[:,10], axis=1)
    p7_p10 = np.linalg.norm(points[:,6] - points[:,9], axis=1)
    ear_r = np.clip((p8_p12 + p9_p11) / (2.0 * p7_p10), 0, 1)

    # return mean of the 2
    return np.nanmean(np.array([ear_l, ear_r]), axis=0)

def classify_video_custom(
    faces: np.ndarray, landmarker: EyeLandmarker, ear_analyser: EarAnalysis
) -> bool:
    """classifies a video using custom models (true=real, false=fake)"""
    ears = []

    # get landmarks from video
    landmarks = landmarker.get_landmarks(faces)

    # calculate ear for frames
    ears = calculate_ears(landmarks)

    if len(ears) == 0:
        return False

    # return the prediction of the ear analyser
    return bool(ear_analyser.predict(np.array(ears)))

def pre_process_frames(faces: np.ndarray) -> np.ndarray:
    """pre-process a frames for traditional models"""
    # normalise to 0-1
    return faces.astype(np.float32) / 255.0

def classify_video_classical(video: np.ndarray, model: Model) -> bool:
    """classifies a video using a pre-existing model (1 real, 0 fake)"""

    # pre-process frames
    frames = pre_process_frames(video)

    # get predictions for each frame
    predictions = model.predict(frames, verbose=0)

    # if more than 50% of frames are classified as real, then real
    real_frames = np.sum(np.argmax(predictions, axis=1))
    real_frame_threshold = 0.5
    return bool(real_frames / len(video) > real_frame_threshold)  # thanks bool_

def get_faces_from_video(frames: np.ndarray) -> np.ndarray:
    """get faces from video"""

    yunet = cv.FaceDetectorYN().create(
        "face_detection_yunet_2023mar.onnx", "", (frames.shape[2], frames.shape[1]), 0.7
    )
    device = "GPU:0" if tf.config.list_physical_devices("GPU") else "CPU:0"
    mtcnn = MTCNN("face_detection_only", device)

    faces = [] # main face in each frame
    faces_per_frame = [np.ndarray([])] * len(frames) # faces within each frame
    mtcnn_indices = [] # indices of frames where mtcnn is needed
    mtcnn_frames = [] # frames where mtcnn is needed

    for i, frame in enumerate(frames):
        # attempt to detect faces with yunet
        _, detected_faces = yunet.detect(frame)

        if detected_faces is not None:
            # if face is detected, add to lists
            face_list = []
            for face in detected_faces:
                x, y, w, h = map(int, face[:4])
                # put in dict to align with mtcnn output
                face_list.append({"box": [x, y, w, h]})
            faces_per_frame[i] = face_list # type: ignore
        else:
            # otherwise, tag to use mtcnn
            mtcnn_indices.append(i)
            mtcnn_frames.append(frame)

    if mtcnn_frames:
        batch_size = 8 # batch mtcnn for speed
        for i in range(0, len(mtcnn_frames), batch_size):
            batch_frames = mtcnn_frames[i : i + batch_size]
            mtcnn_results = mtcnn.detect_faces(batch_frames)

            # add mtcnn results to list
            for j, result in enumerate(mtcnn_results): # type: ignore
                frame_idx = mtcnn_indices[i + j]
                faces_per_frame[frame_idx] = result

    previous_face = None
    for i, frame_faces in enumerate(faces_per_frame):
        if len(frame_faces) > 0:
            best_face = None
            if previous_face is None:
                # if first frame, set previous face to first face
                best_face = frame_faces[0]["box"]
            else:
                # otherwise, get face closest to previous
                best_face = min(
                    frame_faces,
                    key=lambda x: np.linalg.norm(
                        np.array(previous_face) - np.array(x["box"])
                    )
                )["box"]
            previous_face = best_face
            x, y, w, h = best_face
            x = max(0, x)
            y = max(0, y)
            w = min(frames[i].shape[1] - x, w)
            h = min(frames[i].shape[0] - y, h)
            face_crop = cv.resize(
                frames[i][y : y + h, x : x + w], (256, 256)
            )
            faces.append(face_crop)
        else:
            faces.append(cv.resize(frames[i], (256, 256)))
    return np.array(faces)

def process_video(
    video_path: str,
    customs: list[tuple[EyeLandmarker, EarAnalysis]],
    models: list[Model],
) -> dict[str, bool]:
    """classifies a video and returns the results"""

    print(f"Processing {video_path}")

    # get central 512 frames from video
    video = cv.VideoCapture(video_path)
    total_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    frames = []
    max_frames = 512
    start_frame = (total_frames - max_frames) // 2 if total_frames > max_frames else 0
    video.set(cv.CAP_PROP_POS_FRAMES, start_frame)
    for _ in range(max_frames):
        success, frame = video.read()
        if not success:
            break
        frames.append(frame)
    video.release()
    frames = np.array(frames)

    faces = get_faces_from_video(frames)

    predictions = {}

    # classify with custom models
    for i, (landmarker, ear_analyser) in enumerate(customs):
        predictions[f"custom_model_{i}"] = classify_video_custom(
            faces, landmarker, ear_analyser
        )

    # classify with classical models
    for i, model in enumerate(models):
        predictions[f"classical_model_{i}"] = classify_video_classical(
            faces, model
        )

    return predictions

def main() -> None:
    # allow for memory growth on GPU
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    path_to_dataset = "/dcs/large/u2204489/fakeavceleb/"
    path_to_models = "/dcs/large/u2204489/"
    path_to_save = "transferability_results.txt"

    # load models
    # load custom models
    faceforensics_hrnet = get_custom_model(
        True, path_to_models, "faceforensics_hrnet_time_series_forest.joblib"
    )
    faceforensics_pfld = get_custom_model(
        False, path_to_models, "faceforensics_pfld_learning_shapelets.joblib"
    )
    celeb_df_hrnet = get_custom_model(
        True, path_to_models, "celeb-df_hrnet_learning_shapelets.joblib"
    )
    celeb_df_pfld = get_custom_model(
        False, path_to_models, "celeb-df_pfld_learning_shapelets.joblib"
    )
    custom_models = [
        faceforensics_hrnet,
        faceforensics_pfld,
        celeb_df_hrnet,
        celeb_df_pfld,
    ]

    # load classical models
    model_paths = [
        "celeb-df_efficientnet.keras",
        "celeb-df_resnet50.keras",
        "celeb-df_vgg19.keras",
        "celeb-df_xception.keras",
        "faceforensics_efficientnet.keras",
        "faceforensics_resnet50.keras",
        "faceforensics_vgg19.keras",
        "faceforensics_xception.keras",
    ]
    models = [
        load_model(path_to_models + model_path) for model_path in model_paths
    ]

    all_models = [
        "faceforensics_hrnet",
        "faceforensics_pfld",
        "celeb_df_hrnet",
        "celeb_df_pfld",
        *model_paths
    ]

    # create dataset
    dataset = create_dataset(path_to_dataset)
    print(len(dataset))

    results = load_results(path_to_save) if Path(path_to_save).exists() else {}
    num_processed = sum(results["faceforensics_hrnet"].values()) if results else 0

    if results == {}:
        results = {
            model: {"tp": 0, "fp": 0, "tn": 0, "fn": 0} for model in all_models
        }

    for i, (video_path, label) in enumerate(
        dataset[num_processed:], start=num_processed + 1
    ):
        print(f"{i}/{len(dataset)}: {video_path} - {label}")

        predictions = process_video(video_path, custom_models, models)
        for (_, pred), res_name in zip(predictions.items(), all_models):
            if pred and label == 1:
                results[res_name]["tp"] += 1
            elif not pred and label == 1:
                results[res_name]["fn"] += 1
            elif pred and label == 0:
                results[res_name]["fp"] += 1
            elif not pred and label == 0:
                results[res_name]["tn"] += 1

        # save results every 25 videos
        if i % 25 == 0:
            save_results(results, path_to_save)

if __name__ == "__main__":
    main()
