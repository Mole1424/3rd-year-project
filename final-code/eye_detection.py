import sys
from pathlib import Path
from typing import Generator

import cv2 as cv
import numpy as np
import tensorflow as tf
from eyes import HRNET, EyeLandmarker, auxiliary_net, pfld, pfld_loss
from foolbox import TargetedMisclassification, TensorFlowModel  # type: ignore
from foolbox.attacks import LinfPGD
from mtcnn import MTCNN  # type: ignore
from tensorflow.keras.losses import MeanSquaredError  # type: ignore
from tensorflow.keras.models import Model, load_model  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore

IMAGE_SIZE = (256, 256)


def points_to_heatmap(
    points: np.ndarray, heatmap_size: tuple[int, int], sigma: float
) -> np.ndarray:
    """converts points to heatmap"""

    # create empty heatmap
    h, w = heatmap_size
    num_points = points.shape[0]
    heatmap = np.zeros((h, w, num_points), dtype=np.float32)

    # scale points to heatmap size
    scale_x = w / IMAGE_SIZE[0]
    scale_y = h / IMAGE_SIZE[1]

    # create grid for heatmap
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

    # extract pitch, yaw, roll
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

        # resize image
        image = cv.resize(image[int(y) : int(y + h), int(x) : int(x + w)], IMAGE_SIZE)

        # scale points
        points[:, 0] = (points[:, 0] - x) / w * IMAGE_SIZE[0]
        points[:, 1] = (points[:, 1] - y) / h * IMAGE_SIZE[1]

        # return heatmap, or angles
        if heatmap:
            yield image, points_to_heatmap(points, (64, 64), 1.5)
        else:
            yield image, points, get_angles(points, image)


def load_dataset(path: str, file_type: str, heatmap: bool) -> tf.data.Dataset:
    """load dataset from path"""

    return tf.data.Dataset.from_generator(
        lambda: dataset_generator(path, file_type, heatmap),
        output_signature=(
            # the image
            tf.TensorSpec(shape=(*IMAGE_SIZE, 3), dtype=tf.float32),  # type: ignore
            (
                # the heatmap or points
                tf.TensorSpec(shape=(64, 64, 68), dtype=tf.float32)  # type: ignore
                if heatmap
                else tf.TensorSpec(shape=(68, 2), dtype=tf.float32)  # type: ignore
            ),
            # pitch, yaw, roll if applicable
            tf.TensorSpec(shape=(3,), dtype=tf.float32) if not heatmap else None, # type: ignore # angles
        ),
    )


def combine_datasets(
    datasets: list[tf.data.Dataset], batch_size: int, debug: bool
) -> tf.data.Dataset:
    """combine datasets, shuffle and batch"""
    full_dataset = datasets[0]
    for dataset in datasets[1:]:
        full_dataset = full_dataset.concatenate(dataset)
    # shuffle only if not in debug mode to save time
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

    # allow memory growth on gpu
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    if hrnet:
        # train hrnet
        batch_size = 16
        epochs = 60
        # steps per epoch = dataset/batch ~= 13000/16 ~= 812 ~= 1000
        steps_per_epoch = 1000

        hrnet_dataset = load_datasets(path_to_large, batch_size, True, debug)

        # create and train model
        model = HRNET(hrnet_config)
        model.compile(optimizer=Adam(0.001), loss=MeanSquaredError())
        model.fit(
            hrnet_dataset, epochs=epochs, steps_per_epoch=steps_per_epoch, verbose=2
        )
        model.save_weights(f"{path_to_large}hrnet.weights.h5")
    else:
        # train pfld
        batch_size = 128
        epochs = 500

        pfld_dataset = load_datasets(path_to_large, batch_size, False, debug)

        pfld_model = pfld()
        aux_model = auxiliary_net()

        optimiser = Adam(1e-4)

        @tf.function
        def train_step(
            images: tf.Tensor, points: tf.Tensor, angles: tf.Tensor
        ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
            """custom training for pfld to allow for custom loss function"""
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
                print(
                    f"Epoch: {epoch}, Batch: {i}, Loss: {loss} "
                    f"Landmark Loss: {landmark_loss}, Angle Loss: {angle_loss}"
                )
            print("=" * 20)
            print(f"Epoch: {epoch}, Loss: {loss}") # type: ignore
            print("=" * 20)
            if epoch % 50 == 0:
                pfld_model.save(f"{path_to_large}pfld.keras")
                aux_model.save(f"{path_to_large}auxiliary.keras")

        pfld_model.save(f"{path_to_large}pfld.keras")
        aux_model.save(f"{path_to_large}auxiliary.keras")

    print("done :)")


def test_model(hrnet: bool) -> None:
    # delete all files in test-images from previous runs
    for file in Path("test-images/").glob("*"):
        file.unlink()

    # load model and testset
    path_to_large = "/dcs/large/u2204489/"
    path_to_model = (
        path_to_large + "hrnet.weights.h5" if hrnet else path_to_large + "pfld.keras"
    )
    path_to_testset = path_to_large + "eyes/lfpw/testset/"

    # get 5 images from testset
    images = Path(path_to_testset).glob("*.png")
    images = [image for image, _ in zip(images, range(5))]
    model = EyeLandmarker(hrnet_config if hrnet else None, path_to_model)

    for image in images:
        print("Processing image:", image)
        img = cv.imread(str(image))
        points = model.get_landmarks(np.array([img]))[0]

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
        print("Saving image to:", new_path)
        cv.imwrite(new_path, img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    print("done :)")


def calculate_ear(points: np.ndarray) -> float:
    """calcualte eye aspect ratio"""
    p2_p6 = np.linalg.norm(points[1] - points[5])
    p3_p5 = np.linalg.norm(points[2] - points[4])
    p1_p4 = np.linalg.norm(points[0] - points[3])

    return float((p2_p6 + p3_p5) / (2.0 * p1_p4))

def noise_test() -> None:
    path_to_large = "/dcs/large/u2204489/"
    path_to_hrnet = path_to_large + "hrnet.weights.h5"
    path_to_pfld = path_to_large + "pfld.keras"
    path_to_vgg = path_to_large + "faceforensics_vgg19.keras"

    hrnet = EyeLandmarker(hrnet_config, path_to_hrnet)
    pfld = EyeLandmarker(None, path_to_pfld)
    vgg = load_model(path_to_vgg)

    mtcnn = MTCNN("face_detection_only")

    path_to_image = path_to_large + "eyes/lfpw/testset/image_0063.png"

    image = cv.imread(path_to_image)
    mtcnn_boxes = mtcnn.detect_faces(image)
    if mtcnn_boxes is None:
        print("No faces detected")
        return

    # get bounding box
    mtcnn_boxes = mtcnn_boxes[0]
    x, y, w, h = mtcnn_boxes["box"]
    face_crop = image[y:y+h, x:x+w].copy()
    orig_h, orig_w = face_crop.shape[:2]
    resized_face = cv.resize(face_crop, IMAGE_SIZE)
    resized_face = np.expand_dims(resized_face, axis=0)

    # get clean landmarks
    hrnet_points = hrnet.get_landmarks(resized_face)[0]
    pfld_points = pfld.get_landmarks(resized_face)[0]

    # add noise
    noisy_frame = perturb_frame(resized_face.copy(), vgg)

    # get noisy landmarks
    noisy_hrnet_points = hrnet.get_landmarks(noisy_frame)[0]
    noisy_pfld_points = pfld.get_landmarks(noisy_frame)[0]

    def resize_point(point: np.ndarray) -> np.ndarray:
        """resize point to original image size"""
        x, y = point
        x = int(x * orig_w / IMAGE_SIZE[0])
        y = int(y * orig_h / IMAGE_SIZE[1])
        return np.array([x, y], dtype=np.float32)

    hrnet_points = np.array([resize_point(point) for point in hrnet_points])
    pfld_points = np.array([resize_point(point) for point in pfld_points])
    noisy_hrnet_points = np.array([resize_point(point) for point in noisy_hrnet_points])
    noisy_pfld_points = np.array([resize_point(point) for point in noisy_pfld_points])

    # add hrnet points, in red (dots for clean, cross for noisy)
    # add pfld points, in blue (dots for clean, cross for noisy)
    # Prepare two copies of the face crop
    face_crop_hrnet = face_crop.copy()
    face_crop_pfld = face_crop.copy()

    for idx, (
        clean_hrnet_point, clean_pfld_point,
        noisy_hrnet_point, noisy_pfld_point
    ) in enumerate(
        zip(hrnet_points, pfld_points, noisy_hrnet_points, noisy_pfld_points)
    ):
        # Draw HRNet points on face_crop_hrnet (red)
        cv.circle(face_crop_hrnet, tuple(clean_hrnet_point.astype(int)), 2, (0, 0, 255), -1)
        cv.putText(
            face_crop_hrnet,
            f"H{idx}",
            tuple(clean_hrnet_point.astype(int) + np.array([5, -5])),
            cv.FONT_HERSHEY_SIMPLEX,
            0.3,
            (0, 0, 255),
            1,
        )
        cv.drawMarker(
            face_crop_hrnet,
            tuple(noisy_hrnet_point.astype(int)),
            (0, 0, 255),
            markerType=cv.MARKER_TILTED_CROSS,
            markerSize=6,
            thickness=1
        )

        # Draw PFLD points on face_crop_pfld (blue)
        cv.circle(face_crop_pfld, tuple(clean_pfld_point.astype(int)), 2, (255, 0, 0), -1)
        cv.putText(
            face_crop_pfld,
            f"P{idx}",
            tuple(clean_pfld_point.astype(int) + np.array([5, 5])),
            cv.FONT_HERSHEY_SIMPLEX,
            0.3,
            (255, 0, 0),
            1
        )
        cv.drawMarker(
            face_crop_pfld,
            tuple(noisy_pfld_point.astype(int)),
            (255, 0, 0),
            markerType=cv.MARKER_TILTED_CROSS,
            markerSize=6,
            thickness=1
        )

    # Save two separate images
    cv.imwrite("test-images/noisy_frame_hrnet.png", face_crop_hrnet)
    cv.imwrite("test-images/noisy_frame_pfld.png", face_crop_pfld)
    print("done :)")


def preprocess_image(image:np.ndarray) -> np.ndarray:
    """preprocess image for vgg"""
    return image.astype(np.float32) / 255.0

def postprocess_image(image:np.ndarray) -> np.ndarray:
    """postprocess image for vgg"""
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)

def perturb_frame(image: np.ndarray, vgg: Model) -> np.ndarray:
    """perturb image with foolbox"""
    image = preprocess_image(image)
    image = tf.convert_to_tensor(image, dtype=tf.float32) # type: ignore
    attack = LinfPGD(steps=1)
    noisy_frame = attack.run(
        TensorFlowModel(vgg, bounds=(0, 1)),
        image,
        TargetedMisclassification(tf.constant([1], dtype=tf.int32)),
        epsilon=0.05,
    )

    return postprocess_image(noisy_frame)

if __name__ == "__main__":
    arg = None
    try:
        arg = sys.argv[1]
    except IndexError:
        print("Please provide an argument")
        sys.exit(1)

    if arg == "test":
        test_model(False)
    elif arg == "noise":
        noise_test()
    else:
        main(arg == "debug", False)
