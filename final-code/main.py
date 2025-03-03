import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2 as cv
import numpy as np
import tensorflow as tf
from ear_analysis import EarAnalysis
from foolbox import Misclassification, TensorFlowModel  # type: ignore
from foolbox.attacks import LinfFastGradientAttack  # type: ignore
from hrnet import HRNet
from pfld import PFLD
from tensorflow.keras.models import Model, load_model  # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array  # type: ignore


def generate_datasets(path_to_dataset: str) -> list[tuple[str, int]]:
    """creates dataset of video paths and labels"""
    videos = Path(path_to_dataset).rglob("*.mp4")

    dataset = []

    for video in videos:
        label = int("real" in video.name)
        dataset.append((str(video), label))

    return dataset


def classify_video_custom(
    video: np.ndarray, landmarker: HRNet | PFLD, ear_analyser: EarAnalysis
) -> bool:
    """classifies a video using custom models (1 real, 0 fake)"""
    ears = []

    for frame in video:
        # get landmarks of the first face detected
        landmarks = landmarker.get_landmarks(frame)[0]

        # calculate the eye aspect ratio for each eye and take the average
        ear_l = (
            np.linalg.norm(landmarks[1] - landmarks[5])
            + np.linalg.norm(landmarks[2] - landmarks[4])
        ) / (2 * np.linalg.norm(landmarks[0] - landmarks[3]))
        ear_r = (
            np.linalg.norm(landmarks[6] - landmarks[10])
            + np.linalg.norm(landmarks[7] - landmarks[9])
        ) / (2 * np.linalg.norm(landmarks[8] - landmarks[11]))
        ears.append((ear_l + ear_r) / 2)

    # return the prediction of the ear analyser
    return bool(ear_analyser.predict(np.array(ears)))


def classify_video_classical(video: np.ndarray, model: Model) -> bool:
    """classifies a video using a pre-existing model (1 real, 0 fake)"""
    real_frames = 0

    for frame in video:
        # pre-process frame for model
        input_frame = cv.resize(frame, (256, 256))
        input_frame = img_to_array(input_frame) / 255.0
        input_frame = np.expand_dims(input_frame, axis=0)

        # predict the frame
        prediction = model.predict(input_frame, verbose=0)
        real_frames += np.argmax(prediction)

    # assume if >50% of frames are classified as real, the video is real
    real_frame_threshold = 0.5
    return bool(real_frames / len(video) > real_frame_threshold)  # thanks bool_


# config adapted from https://github.com/HRNet/HRNet-Facial-Landmark-Detection/blob/master/experiments/wflw/face_alignment_wflw_hrnet_w18.yaml
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
        frame = img_to_array(frame) / 255.0  # noqa: PLW2901
        frame = np.expand_dims(frame, axis=0)  # noqa: PLW2901
        frame = tf.convert_to_tensor(frame, dtype=tf.float32)  # type: ignore # noqa: PLW2901

        # generate noise using the fast gradient sign attack
        attack = LinfFastGradientAttack()
        epsilon = 0.02
        xception_frame = attack.run(
            TensorFlowModel(xception, bounds=(0, 1)),
            frame,
            # attempt to misclassify the frame as real
            Misclassification(tf.constant([1], dtype=tf.int32)),
            epsilon=epsilon,
        )
        efficientnet_frame = attack.run(
            TensorFlowModel(efficientnet, bounds=(0, 1)),
            frame,
            # attempt to misclassify the frame as real
            Misclassification(tf.constant([1], dtype=tf.int32)),
            epsilon=epsilon,
        )

        frame_set = (xception_frame, efficientnet_frame)

        # post-process frame
        for frame in frame_set:  # noqa: PLW2901
            frame = np.squeeze(frame, axis=0)  # noqa: PLW2901
            frame = np.clip(frame, 0, 1)  # noqa: PLW2901
            frame = (frame * 255).astype(np.uint8)  # noqa: PLW2901
            frame = cv.resize(  # noqa: PLW2901
                frame, (original_dimensions[1], original_dimensions[0])
            )
        return_frames[0].append(frame_set[0])
        return_frames[1].append(frame_set[1])

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
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        frames.append(frame)
    frames = np.array(frames)

    # predict the video using all models
    custom_prediction = classify_video_custom(frames, landmarker, ear_analyser)
    xception_prediction = classify_video_classical(frames, xception)
    efficientnet_prediction = classify_video_classical(frames, efficientnet)

    # perturbate and classify fake frames
    if label == 0:
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


def worker(
    video_info: tuple[str, int]
) -> tuple[int, bool, bool, bool, bool, bool, bool, bool, bool, bool]:
    """Processes a single video and returns classification results"""

    path_to_large = "/dcs/large/u2204489/"

    # load models
    landmarker = HRNet(hrnet_config, path_to_large + "hrnet.weights.h5")
    ear_analyser = EarAnalysis(path_to_large + "fullyconvolutionalneuralnetwork.keras")

    xception = load_model(path_to_large + "xception.keras")
    efficientnet = load_model(path_to_large + "efficientnet_b4.keras")

    # process video
    return process_video(
        video_info,
        landmarker,
        ear_analyser,
        xception,
        efficientnet,
    )


def main(path_to_dataset: str) -> None:

    # allow for memory growth on GPU
    for gpu in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    # generate dataset
    path_to_large = "/dcs/large/u2204489/"
    path_to_dataset = path_to_large + path_to_dataset
    dataset = generate_datasets(path_to_dataset)

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

    # process videos in parallel for performance
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(worker, video_info) for video_info in dataset]

        for future in as_completed(futures):
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
            ) = future.result()

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

    # print final results
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
