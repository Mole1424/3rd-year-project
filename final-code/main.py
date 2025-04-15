import json
import pickle
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
import tensorflow as tf
from ear_analysis import EarAnalysis
from eye_detection import EyeLandmarker
from foolbox import TargetedMisclassification, TensorFlowModel  # type: ignore
from foolbox.attacks import LinfPGD
from mtcnn import MTCNN  # type: ignore
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model  # type: ignore
from traditional_detectors import train_detectors


def generate_datasets(path_to_dataset: str) -> list[list[tuple[str, int]]]:
    """creates dataset of video paths and labels, splitting into train and test sets"""
    videos = list(Path(path_to_dataset).rglob("*.mp4"))

    # split videos into real and fake
    real_videos = [str(video) for video in videos if "real" in str(video)]
    fake_videos = [str(video) for video in videos if "fake" in str(video)]

    # real dataset is the limiting factor
    train_size = int(0.8 * len(real_videos))

    # split into train and test sets
    train_real, test_real = train_test_split(
        real_videos, train_size=train_size, random_state=42
    )
    train_fake, test_fake = train_test_split(
        fake_videos, train_size=train_size, random_state=42
    )

    # aggregate into 2 datasets
    train_data = (
        [(video, 0) for video in train_fake] + [(video, 1) for video in train_real]
    )
    test_data = (
        [(video, 0) for video in test_fake] + [(video, 1) for video in test_real]
    )

    return [train_data, test_data]


def save_progress(results: dict, path_to_ear_anylyser: str, path_output: str) -> None:
    """saves current progress to file"""

    with Path(path_output).open("w") as f:
        # the path to best ear analyser model
        f.write(path_to_ear_anylyser + "\n")
        # current stats for each model
        json.dump(results, f, indent=4)

def load_progress(path_output: str) -> tuple[str, dict]:
    """loads progress from file"""

    path = Path(path_output)
    if not path.exists():
        # if no path, no data
        return "", {}

    # otherwise read the data
    with path.open("r") as f:
        path_to_ear_anylyser = f.readline().strip()
        results = json.load(f)
    return path_to_ear_anylyser, results


def get_custom_model(
    hrnet: bool,
    train_set: list[tuple[str, int]],
    path_to_models: str,
    dataset_name: str,
    path_to_ear_anylyser: str
) -> tuple[EyeLandmarker, EarAnalysis, str]:
    """loads (and trains) models for custom classification"""
    eye_landmarker = "hrnet" if hrnet else "pfld"

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
        landmarker = EyeLandmarker(hrnet_config, path_to_models + "hrnet.weights.h5")
    else:
        # otherwise use PFLD
        landmarker = EyeLandmarker(None, path_to_models + "pfld.keras")

    if path_to_ear_anylyser != "":
        # if path to ear analyser is provided, load it
        ear_analyser = EarAnalysis(path_to_ear_anylyser, None, "", "", "")
        return landmarker, ear_analyser, path_to_ear_anylyser

    # otherwise check if pickle file for ears dataset exists
    path = Path(f"{path_to_models}/{dataset_name}_{eye_landmarker}_ears_dataset.pkl")
    if path.exists():
        # if it does, load it
        with path.open("rb") as file:
            ears_dataset = pickle.load(file)
    else:
        # otherwise need to create from scratch
        ears_dataset = []
        for video in train_set:
            video_path, label = video
            print(f"Processing {video_path}")

            # load middle 512 frames from video
            video = cv.VideoCapture(video_path)  # noqa: PLW2901
            total_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))
            frames = []
            max_frames = 512
            start_frame = (
                (total_frames - max_frames) // 2 if total_frames > max_frames else 0
            )
            video.set(cv.CAP_PROP_POS_FRAMES, start_frame)
            for _ in range(max_frames):
                success, frame = video.read()
                if not success:
                    break
                frames.append(frame)
            video.release()
            frames = np.array(frames)

            # get faces from video
            faces = get_faces_from_video(frames)

            # get landmarks from video
            landmarks = landmarker.get_landmarks(faces)

            # calculate ear aspect accross frames
            ears = calculate_ears(landmarks)
            # pad if necessary
            if len(ears) < max_frames:
                ears = np.pad(
                    ears,
                    (0, max_frames - len(ears)),
                    "constant",
                    constant_values=-1
                )
            ears_dataset.append((ears, label))

        # split ears dataset into train and test
        ears_dataset = train_test_split(ears_dataset, train_size=0.8, random_state=42)
        # save dataset
        with path.open("wb") as file:
            pickle.dump(ears_dataset, file)

    # training models doesnt like being accross multiple GPUs
    ear_analyser = EarAnalysis(
        None, ears_dataset, path_to_models, dataset_name, eye_landmarker
    )
    path = ear_analyser.get_best_path()

    return landmarker, ear_analyser, path

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


def post_process_frames(frames: np.ndarray) -> np.ndarray:
    """post-process frames from noise attacks"""
    # convert back to uint8 and 0-255
    return np.clip(frames * 255, 0, 255).astype(np.uint8)

def perturbate_frames(
    frames: np.ndarray, epsilon: float, **models: Model
) -> tuple[np.ndarray, ...]:
    """Add adversarial noise to video frames for multiple models."""

    # pre-process frames
    frames = pre_process_frames(frames)

    # intialise attack (FGSM)
    attack = LinfPGD(steps=1)
    batch_size = 16

    # set frames in tensorflow dataset
    dataset = tf.data.Dataset.from_tensor_slices(frames).batch(batch_size)

    # adverserial frames for each model
    adv_frames = {name: [] for name in models}

    for batch in dataset:
        # batch size may not be equal to batch_size (last batch)
        batch_size_actual = tf.shape(batch)[0]  # type: ignore

        # targeted misclassification for real (1)
        target = TargetedMisclassification(tf.ones(batch_size_actual, dtype=tf.int32))

        # for each model, get adverserial frames
        for name, model in models.items():
            adv_batch = attack.run(
                TensorFlowModel(model, bounds=(0, 1)), batch, target, epsilon=epsilon
            )
            adv_frames[name].extend(post_process_frames(adv_batch)) # type: ignore

    return tuple(np.array(adv_frames[name]) for name in models)


def get_faces_from_video(frames: np.ndarray) -> np.ndarray:
    """get faces from video"""

    yunet = cv.FaceDetectorYN().create(
        "face_detection_yunet_2023mar.onnx", "", (frames.shape[2], frames.shape[1]), 0.7
    )
    device = "GPU:0" if tf.config.list_physical_devices("GPU") else "CPU:0"
    mtcnn = MTCNN("face_detection_only", device)

    faces = []
    faces_per_frame = [np.ndarray([])] * len(frames)
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
    video_info: tuple[str, int],
    epsilon: float,
    landmarker: EyeLandmarker,
    ear_analyser: EarAnalysis,
    **models: Model,
) -> tuple[int, dict[str, bool]]:
    """Processes a single video and returns classification results."""

    video_path, label = video_info

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

    # get initial predictions
    predictions = {"custom": classify_video_custom(faces, landmarker, ear_analyser)}
    for name, model in models.items():
        predictions[name] = classify_video_classical(faces, model)

    # if video is fake, perturbate frames and classify them
    if label == 0:
        perturbed_faces = perturbate_frames(faces, epsilon, **models)

        for i, model_name in enumerate(models.keys()):
            adv_faces = perturbed_faces[i]

            # reclassify perturbed frames using custom models
            predictions[f"custom_{model_name}"] = classify_video_custom(
                adv_faces, landmarker, ear_analyser
            )

            # reclassify perturbed frames using classical models
            for target_model_name, target_model in models.items():
                predictions[
                    f"{model_name}_{target_model_name}"
                ] = classify_video_classical(adv_faces, target_model)
    else:
        # if video is real, copy predictions
        for model_name in models:
            predictions[f"custom_{model_name}"] = predictions["custom"]
            for target_model in models:
                predictions[f"{model_name}_{target_model}"] = predictions[target_model]

    # print results
    print("Finished processing:", video_path)
    print("Label:", bool(label))
    print("Correct?:")
    for name, prediction in predictions.items():
        print(f"{name}: {prediction == bool(label)}")
    print("=" * 20)

    return label, predictions


def main(
    path_to_dataset: str, path_to_models: str, epsilon: float, eye_model: str
) -> None:
    # allow for memory growth on GPU
    for gpu in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    # get dataset name
    dataset_name = path_to_dataset.split("/")[-2]
    path_to_save = f"{dataset_name}_{eye_model}_{epsilon}_results.txt"

    # attempt to load progress
    path_to_ear_anylyser, loaded_results = load_progress(path_to_save)

    # generate dataset
    print("Generating datasets")
    train_set, test_set = generate_datasets(path_to_dataset)
    print("Datasets generated")

    # get custom model
    print("Getting custom model")
    landmarker, ear_analyser, best_path = get_custom_model(
        eye_model=="hrnet",
        train_set,
        path_to_models,
        dataset_name,
        path_to_ear_anylyser,
    )
    print("Custom model loaded")

    # save ear analyser path
    save_progress(loaded_results, best_path, path_to_save)
    tf.keras.backend.clear_session() # type: ignore

    # train traditional models
    print("Training traditional models")
    models = train_detectors(path_to_models, dataset_name, path_to_dataset)
    print("Traditional models trained")

    tf.keras.backend.clear_session() # type: ignore

    # initialise results
    traditional_models = ["vgg", "resnet", "xception", "efficientnet"]
    all_models = ["custom", *traditional_models]
    if loaded_results == {}:
        results = {model: {"tp": 0, "tn": 0, "fp": 0, "fn": 0} for model in all_models}
        results.update(
            {f"{model}_{target}": {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
            for model in all_models for target in traditional_models}
        )
    else:
        results = loaded_results

    num_vidoes_processed = sum(results["custom"].values())

    # process each video in the dataset
    print("Processing videos")
    for i, video in enumerate(
        test_set[num_vidoes_processed:], start=num_vidoes_processed + 1
    ):
        print(f"{i}/{len(test_set)}")
        label, predictions = process_video(
            video,
            epsilon,
            landmarker,
            ear_analyser,
            **dict(zip(traditional_models, models))
        )
        for model_name, prediction in predictions.items():
            if label == 1 and prediction:
                results[model_name]["tp"] += 1
            elif label == 1 and not prediction:
                results[model_name]["fn"] += 1
            elif label == 0 and prediction:
                results[model_name]["fp"] += 1
            elif label == 0 and not prediction:
                results[model_name]["tn"] += 1

        # save progress every now and then
        if i % 25 == 0:
            save_progress(results, best_path, path_to_save)

    save_progress(results, best_path, path_to_save)

    # print final results
    for model_name, metrics in results.items():
        accuracy = (metrics["tp"] + metrics["tn"]) / len(test_set)
        print("=" * 20)
        print(f"Model: {model_name}")
        print(f"Accuracy: {accuracy}")
        print(f"True Positives: {metrics["tp"]}")
        print(f"True Negatives: {metrics["tn"]}")
        print(f"False Positives: {metrics["fp"]}")
        print(f"False Negatives: {metrics["fn"]}")

    print("done :)")


if __name__ == "__main__":
    try:
        path_to_dataset = sys.argv[1]
        path_to_models = sys.argv[2]
        epsilon = float(sys.argv[3])
        eye_model = sys.argv[4]
    except (IndexError, ValueError):
        print("Please provide the path to the dataset, path to models, epsilon value, and eye model (hrnet or pfld)")  # noqa: E501
        sys.exit(1)
    main(path_to_dataset, path_to_models, epsilon, eye_model)
