import sys
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np
import tensorflow as tf
from ear_analysis import EarAnalysis
from foolbox import TargetedMisclassification, TensorFlowModel  # type: ignore
from foolbox.attacks import LinfPGD
from hrnet import HRNet
from pfld import PFLD
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model  # type: ignore
from traditional_detectors import train_detectors


def generate_datasets(path_to_dataset: str) -> tuple[list[tuple[str, int]]]:
    """creates dataset of video paths and labels, splitting into train and test sets"""
    videos = Path(path_to_dataset).rglob("*.mp4")

    dataset = [(str(video), int("real" in str(video))) for video in videos]

    return train_test_split(dataset, train_size=0.8, random_state=42)

def calculate_ear(points: np.ndarray) -> float:
    """calcualte eye aspect ratio"""
    p2_p6 = np.linalg.norm(points[1] - points[5])
    p3_p5 = np.linalg.norm(points[2] - points[4])
    p1_p4 = np.linalg.norm(points[0] - points[3])

    return float((p2_p6 + p3_p5) / (2.0 * p1_p4))


def classify_video_custom(
    video: np.ndarray, landmarker: HRNet | PFLD, ear_analyser: EarAnalysis
) -> bool:
    """classifies a video using custom models (true real, false fake)"""
    ears = []

    landmarks = landmarker.get_landmarks(video)
    for frame_landmarks in landmarks:
        if frame_landmarks is None:
            continue
        for landmark in frame_landmarks:
            if len(landmark) != 12:  # noqa: PLR2004
                continue
            ear_l = calculate_ear(landmark[0:6])
            ear_r = calculate_ear(landmark[6:12])
            ears.append(min((ear_l + ear_r) / 2, 1))

    if len(ears) == 0:
        return False

    # return the prediction of the ear analyser
    return bool(ear_analyser.predict(np.array(ears)))


def pre_process_frames(frames: np.ndarray) -> np.ndarray:
    """pre-process a frames for traditional models"""
    processed_frames = []
    num, height, width, channels = frames.shape

    for i in range(0, num, 170):
        batch = frames[i : i + 170]
        num_frames = len(batch)

        # hack to resize the images in one go using opencv
        # https://stackoverflow.com/questions/65154879/using-opencv-resize-multiple-the-same-size-of-images-at-once-in-python
        # channels * num <= 512 due to opencv limitations so need to take batches of 170
        batch = batch.transpose((1, 2, 3, 0))
        batch = batch.reshape((height, width, channels * num_frames))
        batch = cv.resize(batch, (256, 256))
        batch = batch.reshape((256, 256, channels, num_frames))
        batch = batch.transpose((3, 0, 1, 2))

        batch = batch / 255.0
        processed_frames.extend(batch)

    return np.array(processed_frames)

def classify_video_classical(video: np.ndarray, model: Model) -> bool:
    """classifies a video using a pre-existing model (1 real, 0 fake)"""
    frames = pre_process_frames(video)
    predictions = model.predict(frames, verbose=0)
    real_frames = np.sum(np.argmax(predictions, axis=1))

    # assume if >50% of frames are classified as real, the video is real
    real_frame_threshold = 0.5
    return bool(real_frames / len(video) > real_frame_threshold)  # thanks bool_


def post_process_frames(frames: Any, height: int, width: int) -> np.ndarray:  # noqa: ANN401
    """post-process frames from noise attacks"""
    frames = np.clip(frames.numpy(), 0, 1) * 255
    frames = frames.astype(np.uint8)

    num, fheight, fwidth, channels = frames.shape
    frames = frames.transpose((1, 2, 3, 0))
    frames = frames.reshape((fheight, fwidth, channels * num))
    frames = cv.resize(frames, (width, height))
    frames = frames.reshape((height, width, channels, num))

    return frames.transpose((3, 0, 1, 2))


def perturbate_frames(
    frames: np.ndarray, **models: Model
) -> tuple[np.ndarray, ...]:
    """Add adversarial noise to video frames for multiple models."""

    frames = pre_process_frames(frames)
    attack = LinfPGD(steps=1)
    epsilon = 0.1
    batch_size = 16

    dataset = tf.data.Dataset.from_tensor_slices(frames).batch(batch_size)
    adv_frames = {name: [] for name in models}

    for batch in dataset:
        batch_size_actual = tf.shape(batch)[0]  # type: ignore
        target = TargetedMisclassification(tf.ones(batch_size_actual, dtype=tf.int32))

        for name, model in models.items():
            adv_batch = attack.run(
                TensorFlowModel(model, bounds=(0, 1)), batch, target, epsilon=epsilon
            )
            adv_frames[name].extend(post_process_frames(adv_batch, *frames.shape[1:3]))

    return tuple(np.array(adv_frames[name]) for name in models)


def process_video(
    video_info: tuple[str, int],
    landmarker: HRNet | PFLD,
    ear_analyser: EarAnalysis,
    **models: Model,
) -> tuple[int, ...]:
    """Processes a single video and returns classification results."""

    video_path, label = video_info

    print(f"Processing {video_path}")

    # get all frames and convert into numpy array
    video = cv.VideoCapture(video_path)
    frames = []
    max_frames = 256
    while video.isOpened():
        success, frame = video.read()
        if not success or len(frames) >= max_frames:
            break
        frames.append(frame)
    frames = np.array(frames)

    # get initial predictions
    predictions = {"custom": classify_video_custom(frames, landmarker, ear_analyser)}
    for name, model in models.items():
        predictions[name] = classify_video_classical(frames, model)

    # if video is fake, perturbate frames and classify them
    if label == 0:
        perturbed_frames = perturbate_frames(frames, **models)

        for i, model_name in enumerate(models.keys()):
            adv_frames = perturbed_frames[i]

            # reclassify perturbed frames using custom models
            predictions[f"custom_{model_name}"] = classify_video_custom(
                adv_frames, landmarker, ear_analyser
            )

            # reclassify perturbed frames using classical models
            for target_model_name, target_model in models.items():
                predictions[
                    f"{model_name}_{target_model_name}"
                ] = classify_video_classical(adv_frames, target_model)
    else:
        # if video is real, copy predictions
        for model_name in models:
            predictions[f"custom_{model_name}"] = predictions["custom"]
            for target_model in models:
                predictions[f"{model_name}_{target_model}"] = predictions[target_model]

    print(f"Finished processing {video_path}")
    print(f"Label: {bool(label)}")
    for key, value in predictions.items():
        print(f"{key.replace("_", " ").title()}: {value}")

    return (label, *predictions.values())


def main(path_to_dataset: str, path_to_models: str) -> None:
    # allow for memory growth on GPU
    for gpu in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    # generate dataset
    train_set, test_set = generate_datasets(path_to_dataset)

    # load models
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
    landmarker = HRNet(hrnet_config, path_to_models + "hrnet.weights.h5")
    ear_analyser = EarAnalysis(path_to_models + "fullyconvolutionalneuralnetwork.keras")

    models = train_detectors(
        path_to_models, path_to_dataset.replace("/", "_"), train_set, path_to_dataset
    )

    # initialise results
    traditional_models = ["vgg", "resnet", "xception", "efficientnet"]
    all_models = ["custom", *traditional_models]

    results = {model: {"tp": 0, "tn": 0, "fp": 0, "fn": 0} for model in all_models}
    results.update(
        {f"{model}_{target}": {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
        for model in all_models for target in traditional_models}
    )

    # process each video in the dataset
    for video in test_set:
        label, *predictions = process_video(
            video, landmarker, ear_analyser, **dict(zip(traditional_models, models))
        )
        for model_name, prediction in zip(results.keys(), predictions):
            if label == 1 and prediction:
                results[model_name]["tp"] += 1
            elif label == 1 and not prediction:
                results[model_name]["fn"] += 1
            elif label == 0 and prediction:
                results[model_name]["fp"] += 1
            elif label == 0 and not prediction:
                results[model_name]["tn"] += 1

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
    except IndexError:
        print("Please provide the path to the dataset and models")
        sys.exit(1)
    main(path_to_dataset, path_to_models)
