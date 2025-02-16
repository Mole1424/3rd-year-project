import sys
from pathlib import Path
from typing import Generator

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from hrnet import HRNET
from tensorflow.keras.callbacks import History  # type: ignore
from tensorflow.keras.losses import MeanSquaredError  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore

IMAGE_SIZE = (512, 512)


def points_to_heatmap(
    points: np.ndarray, heatmap_size: tuple[int, int], sigma: float
) -> np.ndarray:
    """converts points to heatmap"""
    h, w = heatmap_size
    num_points = points.shape[0]
    heatmap = np.zeros((h, w, num_points), dtype=np.float32)

    scale_x = h / IMAGE_SIZE[0]
    scale_y = w / IMAGE_SIZE[1]

    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))

    for i, (x, y) in enumerate(points):
        cx = x * scale_x
        cy = y * scale_y

        d2 = (grid_x - cx) ** 2 + (grid_y - cy) ** 2
        exponent = d2 / 2.0 / sigma / sigma
        heatmap[:, :, i] = np.exp(-exponent)

    return heatmap


def dataset_generator(path: str, file_type: str, heatmap: bool) -> Generator:
    """generator for dataset"""
    for file in Path(path).glob("*.pts"):
        image = cv.imread(str(file).replace("pts", file_type))
        original_size = image.shape[:2]
        image = cv.resize(image, IMAGE_SIZE)
        points = np.loadtxt(file, comments=("version:", "n_points:", "{", "}"))
        points[:, 0] = points[:, 0] * IMAGE_SIZE[0] / original_size[0]
        points[:, 1] = points[:, 1] * IMAGE_SIZE[1] / original_size[1]

        # convert to heatmap
        points = points_to_heatmap(points, (128, 128), 1.5) if heatmap else points

        yield image, points


def load_dataset(path: str, file_type: str, heatmap: bool) -> tf.data.Dataset:
    """load dataset"""
    return tf.data.Dataset.from_generator(
        lambda: dataset_generator(path, file_type, heatmap),
        output_signature=(
            tf.TensorSpec(shape=(*IMAGE_SIZE, 3), dtype=tf.float32),  # type: ignore
            (
                tf.TensorSpec(shape=(128, 128, 68), dtype=tf.float32)  # type: ignore
                if heatmap
                else tf.TensorSpec(shape=(68, 2), dtype=tf.float32)  # type: ignore
            ),
        ),
    )


def combine_datasets(
    datasets: list[tf.data.Dataset], batch_size: int, debug: bool
) -> tf.data.Dataset:
    """combine datasets, shuffle and batch"""
    full_dataset = datasets[0]
    for dataset in datasets[1:]:
        full_dataset = full_dataset.concatenate(dataset)
    # shuffle only if not in debug mode to save times
    full_dataset = full_dataset.shuffle(13000) if not debug else full_dataset
    return full_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def load_datasets(
    path_to_large: str, batch_size: int, heatmap: bool, debug: bool
) -> tf.data.Dataset:
    """load all datasets and combine them"""
    path_to_datasets = path_to_large + "eyes/"
    datasets = [
        load_dataset(path_to_datasets + "300w/", "png", heatmap),
        load_dataset(path_to_datasets + "afw/", "jpg", heatmap),
        load_dataset(path_to_datasets + "cofw/", "jpg", heatmap),
        load_dataset(path_to_datasets + "helen/", "jpg", heatmap),
        load_dataset(path_to_datasets + "ibug/", "jpg", heatmap),
        load_dataset(path_to_datasets + "lfpw/trainset/", "png", heatmap),
        load_dataset(path_to_datasets + "wflw/", "jpg", heatmap),
    ]
    return combine_datasets(datasets, batch_size, debug)


def save_graphs(history: History, epochs: int, model_type: str) -> None:
    """save accuracy and loss graphs"""

    plt.figure()
    epoch_list = list(range(1, epochs + 1))
    plt.plot(epoch_list, history.history["accuracy"], label="Train Accuracy")
    plt.plot(epoch_list, history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.xticks(epoch_list)
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title(f"Accuracy for {model_type}")
    plt.savefig(f"{model_type}-accuracy.png")

    plt.figure()
    plt.plot(epoch_list, history.history["loss"], label="Train Loss")
    plt.plot(epoch_list, history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.xticks(epoch_list)
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"Loss for {model_type}")
    plt.savefig(f"{model_type}-loss.png")


def main(debug: bool) -> None:
    path_to_large = "/dcs/large/u2204489/"

    batch_size = 16

    hrnet_dataset = load_datasets(path_to_large, batch_size, True, debug)

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

    # create and train model
    model = HRNET(hrnet_config)
    model.compile(optimizer=Adam(0.001), loss=MeanSquaredError())
    history = model.fit(hrnet_dataset, epochs=60)

    # save model and graphs
    model.save_weights(f"{path_to_large}hrnet.weights.h5")
    save_graphs(history, 60, "hrnet")

    print("done :)")


if __name__ == "__main__":
    main(sys.argv[1] == "debug")
