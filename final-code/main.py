import sys
from pathlib import Path

import cv2 as cv
import numpy as np
import tensorflow as tf
from ear_analysis import EarAnalysis
from foolbox import TargetedMisclassification, TensorFlowModel  # type: ignore
from foolbox.attacks import LinfPGD
from hrnet import HRNet
from pfld import PFLD
from tensorflow.keras.models import Model, load_model  # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array  # type: ignore


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

    for frame in video:
        # get landmarks of the first face detected
        landmarks = landmarker.get_landmarks(frame)

        if len(landmarks) == 0:
            print("No face detected")
            continue

        landmarks = landmarks[0]
        if len(landmarks) != 12:  # noqa: PLR2004
            print("Not enough landmarks detected")
            continue

        # calculate the eye aspect ratio for each eye and take the average
        ear_l = calculate_ear(landmarks[0:6])
        ear_r = calculate_ear(landmarks[6:12])
        ears.append((ear_l + ear_r) / 2)

    if len(ears) == 0:
        return False

    # return the prediction of the ear analyser
    return bool(ear_analyser.predict(np.array(ears)))


def classify_video_classical(video: np.ndarray, model: Model) -> bool:
    """classifies a video using a pre-existing model (1 real, 0 fake)"""
    real_frames = 0

    for frame in video:
        # pre-process frame for model
        input_frame = cv.resize(frame, (256, 256))
        input_frame = img_to_array(input_frame).flatten() / 255.0
        input_frame = np.reshape(input_frame, (-1, 256, 256, 3))

        # predict the frame
        prediction = model.predict(input_frame, verbose=0)
        real_frames += np.argmax(prediction)

    # assume if >50% of frames are classified as real, the video is real
    real_frame_threshold = 0.5
    return bool(real_frames / len(video) > real_frame_threshold)  # thanks bool_


def perturbate_frames(
    frames: np.ndarray, xception: Model, efficientnet: Model
) -> tuple[np.ndarray, np.ndarray]:
    """add noise to a video"""

    return_frames = ([], [])
    for frame in frames:
        # save the original frame
        original_dimensions = frame.shape

        # pre-process frame for models
        frame = cv.resize(frame, (256, 256))  # noqa: PLW2901
        frame = frame / 255.0  # noqa: PLW2901
        frame = np.expand_dims(frame, axis=0)  # noqa: PLW2901
        frame = tf.convert_to_tensor(frame, dtype=tf.float32)  # noqa: PLW2901

        # generate noise using the fast gradient sign attack
        attack = LinfPGD(steps=1)
        epsilon = 0.001
        xception_frame = attack.run(
            TensorFlowModel(xception, bounds=(0, 1)),
            frame,
            # attempt to misclassify the frame as real
            TargetedMisclassification(tf.constant([1], dtype=tf.int32)),
            epsilon=epsilon,
        )
        efficientnet_frame = attack.run(
            TensorFlowModel(efficientnet, bounds=(0, 1)),
            frame,
            TargetedMisclassification(tf.constant([1], dtype=tf.int32)),
            epsilon=epsilon,
        )

        # post-process the frames to return to original quality
        xception_frame = np.squeeze(xception_frame, axis=0)
        efficientnet_frame = np.squeeze(efficientnet_frame, axis=0)
        xception_frame = np.clip(xception_frame, 0, 1) * 255
        efficientnet_frame = np.clip(efficientnet_frame, 0, 1) * 255
        xception_frame = xception_frame.astype(np.uint8)
        efficientnet_frame = efficientnet_frame.astype(np.uint8)
        xception_frame = cv.resize(
            xception_frame, (original_dimensions[1], original_dimensions[0])
        )
        efficientnet_frame = cv.resize(
            efficientnet_frame, (original_dimensions[1], original_dimensions[0])
        )

        return_frames[0].append(xception_frame)
        return_frames[1].append(efficientnet_frame)

    return np.array(return_frames[0]), np.array(return_frames[1])


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
