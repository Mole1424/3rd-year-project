import sys
from pathlib import Path
from time import time

import cv2 as cv
import numpy as np
import tensorflow as tf
from ear_analysis import EarAnalysis
from foolbox import TargetedMisclassification, TensorFlowModel  # type: ignore
from foolbox.attacks import LinfPGD
from hrnet import HRNet
from pfld import PFLD
from tensorflow.keras.models import Model, load_model  # type: ignore


def generate_datasets(path_to_dataset: str) -> list[tuple[str, int]]:
    """creates dataset of video paths and labels"""
    videos = Path(path_to_dataset).rglob("*.mp4")

    dataset = []

    for video in videos:
        label = int("real" in str(video))
        # if label == 1:
        #     continue
        dataset.append((str(video), label))

    return dataset

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


def perturbate_frames(
    frames: np.ndarray, xception: Model, efficientnet: Model
) -> tuple[np.ndarray, np.ndarray]:
    """add noise to a video"""

    _, height, width, _ = frames.shape

    frames = pre_process_frames(frames)

    attack = LinfPGD(steps=1)
    epsilon = 0.35

    batch_size = 16
    dataset = tf.data.Dataset.from_tensor_slices(frames).batch(batch_size)

    xception_adv = []
    efficientnet_adv = []

    for batch in dataset:
        batch_size_actual = tf.shape(batch)[0] # type: ignore

        xception_batch = attack.run(
            TensorFlowModel(xception, bounds=(0, 1)),
            batch,
            TargetedMisclassification(tf.ones(batch_size_actual, dtype=tf.int32)),
            epsilon=epsilon,
        )

        efficientnet_batch = attack.run(
            TensorFlowModel(efficientnet, bounds=(0, 1)),
            batch,
            TargetedMisclassification(tf.ones(batch_size_actual, dtype=tf.int32)),
            epsilon=epsilon,
        )

        xception_batch = np.clip(xception_batch.numpy(), 0, 1) * 255 # type: ignore
        efficientnet_batch = np.clip(efficientnet_batch.numpy(), 0, 1) * 255 # type: ignore
        xception_batch = xception_batch.astype(np.uint8)
        efficientnet_batch = efficientnet_batch.astype(np.uint8)

        xnum, xheight, xwidth, xchannels = xception_batch.shape
        xception_batch = xception_batch.transpose((1,2,3,0))
        xception_batch = xception_batch.reshape((xheight, xwidth, xnum*xchannels))
        xception_batch = cv.resize(xception_batch, (width, height))
        xception_batch = xception_batch.reshape((height, width, xchannels, xnum))
        xception_batch = xception_batch.transpose((3,0,1,2))

        enum, eheight, ewidth, echannels = efficientnet_batch.shape
        efficientnet_batch = efficientnet_batch.transpose((1,2,3,0))
        efficientnet_batch = efficientnet_batch.reshape(
            (eheight, ewidth, enum * echannels)
        )
        efficientnet_batch = cv.resize(efficientnet_batch, (width, height))
        efficientnet_batch = efficientnet_batch.reshape(
            (height, width, echannels, enum)
        )
        efficientnet_batch = efficientnet_batch.transpose((3,0,1,2))

        xception_adv.extend(xception_batch)
        efficientnet_adv.extend(efficientnet_batch)

    return np.array(xception_adv), np.array(efficientnet_adv)



def process_video(
    video_info: tuple[str, int],
    landmarker: HRNet | PFLD,
    ear_analyser: EarAnalysis,
    xception: Model,
    efficientnet: Model,
) -> tuple[int, bool, bool, bool, bool, bool, bool, bool, bool, bool]:
    """Processes a single video and returns classification results."""
    video_path, label = video_info

    print(f"Processing {video_path}")
    start = time()

    # get all frames and convert into numpy array
    video = cv.VideoCapture(video_path)
    frames = []
    max_frames = 256
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        frames.append(frame)
        if len(frames) == max_frames:
            break
    frames = np.array(frames)

    # predict the video using all models
    custom_prediction = classify_video_custom(frames, landmarker, ear_analyser)
    xception_prediction = classify_video_classical(frames, xception)
    efficientnet_prediction = classify_video_classical(frames, efficientnet)

    if label == 0:
        # perturbate and classify fake frames
        perturbarted_xception, perturbarted_efficientnet = perturbate_frames(
            frames, xception, efficientnet
        )

        custom_xception_prediction = classify_video_custom(
            perturbarted_xception, landmarker, ear_analyser
        )
        custom_efficientnet_prediction = classify_video_custom(
            perturbarted_efficientnet, landmarker, ear_analyser
        )

        xception_xception_prediction = classify_video_classical(
            perturbarted_xception, xception
        )
        xception_efficientnet_prediction = classify_video_classical(
            perturbarted_efficientnet, xception
        )

        efficientnet_xception_prediction = classify_video_classical(
            perturbarted_xception, efficientnet
        )
        efficientnet_efficientnet_prediction = classify_video_classical(
            perturbarted_efficientnet, efficientnet
        )
    else:
        # otherwise the predictions are the same
        custom_xception_prediction = custom_efficientnet_prediction = custom_prediction
        xception_xception_prediction = xception_efficientnet_prediction = (
            xception_prediction
        )
        efficientnet_xception_prediction = efficientnet_efficientnet_prediction = (
            efficientnet_prediction
        )

    end = time()

    # print results for video in case of error
    print(f"Finished processing {video_path}")
    print(f"Label: {bool(label)}")
    print(f"Custom: {custom_prediction}")
    print(f"Xception: {xception_prediction}")
    print(f"EfficientNet: {efficientnet_prediction}")
    print(f"Custom Xception: {custom_xception_prediction}")
    print(f"Xception Xception: {xception_xception_prediction}")
    print(f"EfficientNet Xception: {efficientnet_xception_prediction}")
    print(f"Custom EfficientNet: {custom_efficientnet_prediction}")
    print(f"Xception EfficientNet: {xception_efficientnet_prediction}")
    print(f"EfficientNet EfficientNet: {efficientnet_efficientnet_prediction}")

    print(f"Time taken: {end - start:.2f}s")

    return (
        label,
        custom_prediction,
        xception_prediction,
        efficientnet_prediction,
        custom_xception_prediction,
        xception_xception_prediction,
        efficientnet_xception_prediction,
        custom_efficientnet_prediction,
        xception_efficientnet_prediction,
        efficientnet_efficientnet_prediction,
    )


def main(path_to_dataset: str) -> None:
    # allow for memory growth on GPU
    for gpu in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    # generate dataset
    path_to_large = "/dcs/large/u2204489/"
    path_to_dataset = path_to_large + path_to_dataset
    dataset = generate_datasets(path_to_dataset)

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
    landmarker = HRNet(hrnet_config, path_to_large + "hrnet.weights.h5")
    ear_analyser = EarAnalysis(path_to_large + "fullyconvolutionalneuralnetwork.keras")

    xception = load_model(path_to_large + "xception.keras")
    efficientnet = load_model(path_to_large + "efficientnet_b4.keras")

    # create results dictionary
    results = {
        "custom": {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
        "custom_xception": {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
        "custom_efficientnet": {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
        "xception": {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
        "xception_xception": {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
        "xception_efficientnet": {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
        "efficientnet": {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
        "efficientnet_xception": {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
        "efficientnet_efficientnet": {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
    }

    for video in dataset:
        # process video
        (
            label,
            custom_pred,
            xception_pred,
            efficientnet_pred,
            custom_xception_pred,
            xception_xception_pred,
            efficientnet_xception_pred,
            custom_efficientnet_pred,
            xception_efficientnet_pred,
            efficientnet_efficientnet_pred,
        ) = process_video(video, landmarker, ear_analyser, xception, efficientnet)

        # update results dictionary
        for model_name, pred in zip(
            results.keys(),
            [
                custom_pred,
                xception_pred,
                efficientnet_pred,
                custom_xception_pred,
                xception_xception_pred,
                efficientnet_xception_pred,
                custom_efficientnet_pred,
                xception_efficientnet_pred,
                efficientnet_efficientnet_pred,
            ],
        ):
            if pred:
                if label:
                    results[model_name]["tp"] += 1
                else:
                    results[model_name]["fp"] += 1
            else:  # noqa: PLR5501
                if label:
                    results[model_name]["fn"] += 1
                else:
                    results[model_name]["tn"] += 1

    # print results
    for model_name in results:
        tp, fp, tn, fn = (
            results[model_name]["tp"],
            results[model_name]["fp"],
            results[model_name]["tn"],
            results[model_name]["fn"],
        )
        accuracy = (tp + tn) / len(dataset)
        print("\n" + "=" * 20)
        print(f"{model_name} Model:")
        print(f"False Positives: {fp}")
        print(f"False Negatives: {fn}")
        print(f"True Positives: {tp}")
        print(f"True Negatives: {tn}")
        print(f"Accuracy: {accuracy}")

    print("done :)")


if __name__ == "__main__":
    path_to_dataset = sys.argv[1]
    main(path_to_dataset)
