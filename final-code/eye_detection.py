import sys
from pathlib import Path
from typing import Generator

import cv2 as cv
import numpy as np
import tensorflow as tf
from eyes import HRNET, EyeLandmarker, auxiliary_net, pfld, pfld_loss
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
        # apply downscaling
        cx = x * scale_x
        cy = y * scale_y

        # apply guassian blur to form heatmap
        d2 = (grid_x - cx) ** 2 + (grid_y - cy) ** 2
        exponent = d2 / (2.0 * sigma**2)
        heatmap[:, :, i] = np.exp(-exponent)

    return heatmap

def get_angles(points: np.ndarray, image: np.ndarray) -> np.ndarray:
    """converts facial landmarks to pitch, yaw, roll"""
    # adapted from https://github.com/jerryhouuu/Face-Yaw-Roll-Pitch-from-Pose-Estimation-using-OpenCV/blob/master/pose_estimation.py

    # 6 points for pose estimation
    # nose, chin, left eye, right eye, left mouth, right mouth
    image_points = np.array([
        points[30], points[8], points[36], points[45], points[48], points[54]
    ], dtype=np.float32)

    # those points in a sample 3D model
    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0)
    ])

    # compute camera matrix
    center = (image.shape[1]//2, image.shape[0]//2)
    focal_length = center[0] / np.tan(60/2 * np.pi / 180)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float32)

    # solve pnp
    dist_coeffs = np.zeros((4, 1))
    (_, rotation_vector, translation_vector) = cv.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv.SOLVEPNP_ITERATIVE
    )

    # calculate euler angles
    rvec_matrix = cv.Rodrigues(rotation_vector)[0]
    proj_matrix = np.hstack((rvec_matrix, translation_vector))
    euler_angles = cv.decomposeProjectionMatrix(proj_matrix)[6]

    pitch, yaw, roll = [angle[0] for angle in euler_angles]
    return np.array([pitch, yaw, roll], dtype=np.float32)



def dataset_generator(path: str, file_type: str, heatmap: bool) -> Generator:
    """generator for dataset"""
    for file in Path(path).glob("*.pts"):
        image = cv.imread(str(file).replace("pts", file_type))
        points = np.loadtxt(
            file, np.float32, comments=("version:", "n_points:", "{", "}")
        )

        # crop image to points with some padding
        x, y, w, h = cv.boundingRect(points)
        x -= int(w * 0.02)
        y -= int(h * 0.02)
        w += int(w * 0.04)
        h += int(h * 0.04)

        # clamp to image size
        x = max(0, x)
        y = max(0, y)
        w = min(w, image.shape[1] - x)
        h = min(h, image.shape[0] - y)

        image = cv.resize(image[int(y) : int(y + h), int(x) : int(x + w)], IMAGE_SIZE)

        # scale points
        points[:, 0] = (points[:, 0] - x) / w * IMAGE_SIZE[0]
        points[:, 1] = (points[:, 1] - y) / h * IMAGE_SIZE[1]

        if heatmap:
            yield image, points_to_heatmap(points, (64, 64), 1.5)
        else:
            yield image, points, get_angles(points, image)


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
            tf.TensorSpec(shape=(3,), dtype=tf.float32) if not heatmap else None, # type: ignore
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
        # load_dataset(path_to_datasets + "lfpw/testset/", "png", heatmap),
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


def main(debug: bool, hrnet: bool) -> None:
    path_to_large = "/dcs/large/u2204489/"

    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    if hrnet:
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
    else:
        batch_size = 128
        epochs = 60

        pfld_dataset = load_datasets(path_to_large, batch_size, False, debug)

        pfld_model = pfld()
        aux_model = auxiliary_net()

        optimiser = Adam(1e-4)

        @tf.function
        def train_step(
            images: tf.Tensor, points: tf.Tensor, angles: tf.Tensor
        ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
            with tf.GradientTape() as tape:
                out1, landmarks = pfld_model(images)
                pred_angles = aux_model(out1)

                landmarks = tf.reshape(landmarks, (-1, 68, 2))

                loss, landmark_loss, angle_loss = pfld_loss(
                    points, landmarks, pred_angles, angles
                )

            gradients = tape.gradient(
                loss, pfld_model.trainable_variables + aux_model.trainable_variables
            )
            optimiser.apply_gradients(
                zip(
                    gradients, # type: ignore
                    pfld_model.trainable_variables + aux_model.trainable_variables,
                )
            )

            return loss, landmark_loss, angle_loss

        for epoch in range(epochs):
            for i, (images, points, angles) in enumerate(pfld_dataset):
                loss, landmark_loss, angle_loss = train_step(images, points, angles) # type: ignore
                print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss}, Landmark Loss: {landmark_loss}, Angle Loss: {angle_loss}")
            print("=" * 20)
            print(f"Epoch: {epoch}, Loss: {loss}") # type: ignore
            print("=" * 20)

        pfld_model.save(f"{path_to_large}pfld.keras")

    print("done :)")


def test_model() -> None:
    # delete all files in test-images from previous runs
    for file in Path("test-images/").glob("*"):
        file.unlink()

    path_to_large = "/dcs/large/u2204489/"
    path_to_hrnet = path_to_large + "hrnet.weights.h5"
    path_to_testset = path_to_large + "eyes/lfpw/testset/"

    # get 5 images from testset
    images = Path(path_to_testset).glob("*.png")
    images = [image for image, _ in zip(images, range(5))]
    model = EyeLandmarker(hrnet_config, path_to_hrnet)

    for image in images:
        print("Processing image:", image)
        img = cv.imread(str(image))
        points = model.get_landmarks([img])[0] # type: ignore

        for i, face_points in enumerate(points):
            for j, (x, y) in enumerate(face_points):
                # for each face and point draw labelled point
                cv.circle(img, (int(x), int(y)), 2, (0, 255, 0), -1)
                cv.putText(
                    img,
                    str(i) + "_" + str(j + 1),
                    (int(x), int(y)),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

        # save image for testing
        new_path = "test-images/" + str(image).split("/")[-1]
        cv.imshow(new_path, img)
    cv.waitKey(0)
    cv.destroyAllWindows()

    print("done :)")


def calculate_ear(points: np.ndarray) -> float:
    """calcualte eye aspect ratio"""
    p2_p6 = np.linalg.norm(points[1] - points[5])
    p3_p5 = np.linalg.norm(points[2] - points[4])
    p1_p4 = np.linalg.norm(points[0] - points[3])

    return float((p2_p6 + p3_p5) / (2.0 * p1_p4))


def calculate_ears() -> None:
    path_to_large = "/dcs/large/u2204489/"
    path_to_hrnet = path_to_large + "hrnet.weights.h5"
    path_to_videos = path_to_large + "/faceforensics/"

    model = EyeLandmarker(hrnet_config, path_to_hrnet)

    for video_path in Path(path_to_videos).rglob("*.mp4"):
        print("Processing video:", video_path)

        video = cv.VideoCapture(str(video_path))

        ears = np.zeros(int(video.get(cv.CAP_PROP_FRAME_COUNT)))
        previous_points = []
        while video.isOpened():
            success, frame = video.read()
            if not success:
                break

            points = model.get_landmarks(frame)
            if len(points) == 0:
                continue

            # for first frame choose first face
            if len(previous_points) == 0:
                points = points[0]
                previous_points = points
            else:
                # otherwise choose face with the most overlap to previous frame
                max_overlap = 0
                max_index = 0
                for i, face_points in enumerate(points):
                    overlap = np.sum(
                        np.linalg.norm(previous_points - face_points, axis=1)
                    )
                    if overlap > max_overlap:
                        max_overlap = overlap
                        max_index = i
                previous_points = points[max_index]
                points = previous_points

            # ear is average of left and right eye
            ear_l = calculate_ear(points[0:6])
            ear_r = calculate_ear(points[6:12])
            ear = (ear_l + ear_r) / 2
            ears[int(video.get(cv.CAP_PROP_POS_FRAMES)) - 1] = ear

        video.release()
        np.save(str(video_path).replace("mp4", "npy"), ears)

    print("done :)")


if __name__ == "__main__":
    arg = None
    try:
        arg = sys.argv[1]
    except IndexError:
        print("Please provide an argument")
        sys.exit(1)

    if arg == "test":
        test_model()
    elif arg == "ear":
        calculate_ears()
    else:
        main(arg == "debug", False)
