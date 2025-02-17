import sys
from pathlib import Path
from typing import Generator

import cv2 as cv
import numpy as np
import tensorflow as tf
from hrnet import HRNET
from tensorflow.keras.losses import MeanSquaredError  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore

IMAGE_SIZE = (256, 256)


def points_to_heatmap(
    points: np.ndarray, heatmap_size: tuple[int, int], sigma: float
) -> np.ndarray:
    """converts points to heatmap"""
    h, w = heatmap_size
    num_points = points.shape[0]
    heatmap = np.zeros((h, w, num_points), dtype=np.float32)

    scale_x = w / IMAGE_SIZE[0]
    scale_y = h / IMAGE_SIZE[1]

    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))

    for i, (x, y) in enumerate(points):
        cx = x * scale_x
        cy = y * scale_y

        d2 = (grid_x - cx) ** 2 + (grid_y - cy) ** 2
        exponent = d2 / (2.0 * sigma**2)
        heatmap[:, :, i] = np.exp(-exponent)

    return heatmap


def dataset_generator(path: str, file_type: str, heatmap: bool) -> Generator:
    """generator for dataset"""
    for file in Path(path).glob("*.pts"):
        image = cv.imread(str(file).replace("pts", file_type))
        points = np.loadtxt(
            file, np.float32, comments=("version:", "n_points:", "{", "}")
        )
        print(points)

        # crop image to points
        x, y, w, h = cv.boundingRect(points)
        # add 2% padding
        x -= int(w * 0.02)
        y -= int(h * 0.02)
        w += int(w * 0.04)
        h += int(h * 0.04)

        # Clamp the coordinates to be within image bounds
        x = max(0, x)
        y = max(0, y)
        w = min(w, image.shape[1] - x)
        h = min(h, image.shape[0] - y)

        image = cv.resize(image[int(y) : int(y + h), int(x) : int(x + w)], IMAGE_SIZE)

        # scale points
        points[:, 0] = (points[:, 0] - x) / w * IMAGE_SIZE[0]
        points[:, 1] = (points[:, 1] - y) / h * IMAGE_SIZE[1]

        # convert to heatmap
        points = points_to_heatmap(points, (64, 64), 1.5) if heatmap else points

        yield image, points


def load_dataset(path: str, file_type: str, heatmap: bool) -> tf.data.Dataset:
    """load dataset"""
    return tf.data.Dataset.from_generator(
        lambda: dataset_generator(path, file_type, heatmap),
        output_signature=(
            tf.TensorSpec(shape=(*IMAGE_SIZE, 3), dtype=tf.float32),  # type: ignore
            (
                tf.TensorSpec(shape=(64, 64, 68), dtype=tf.float32)  # type: ignore
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


def main(debug: bool) -> None:
    path_to_large = "/dcs/large/u2204489/"

    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    batch_size = 16
    epochs = 60
    # steps per epoch = dataset/batch ~= 13000/16 ~= 812 ~= 1000
    steps_per_epoch = 1000

    hrnet_dataset = load_datasets(path_to_large, batch_size, True, debug)

    # create and train model
    model = HRNET(hrnet_config)
    model.compile(optimizer=Adam(0.001), loss=MeanSquaredError())
    model.fit(hrnet_dataset, epochs=epochs, steps_per_epoch=steps_per_epoch)
    model.save_weights(f"{path_to_large}hrnet.weights.h5")

    print("done :)")


if __name__ == "__main__":
    main(sys.argv[1] == "debug")
