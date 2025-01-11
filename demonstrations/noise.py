import cv2 as cv
import numpy as np
import tensorflow as tf
from foolbox import TensorFlowModel  # type: ignore
from foolbox.attacks import LinfFastGradientAttack  # type: ignore
from foolbox.criteria import Misclassification  # type: ignore
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array  # type: ignore

model = load_model("/dcs/large/u2204489/vgg19.keras")
foolbox_model = TensorFlowModel(model, bounds=(0, 255))


def perturbate_frame(frame: np.ndarray, frame_num: int) -> np.ndarray:
    original_dimensions = frame.shape
    original_frame = frame.copy()

    frame = cv.resize(frame, (256, 256))
    frame = img_to_array(frame) / 255.0
    frame = np.expand_dims(frame, axis=0)
    frame = tf.convert_to_tensor(frame, dtype=tf.float32)  # type: ignore

    attack = LinfFastGradientAttack()
    epsilon = 0.02
    noisy_frame = attack.run(
        foolbox_model,
        frame,
        Misclassification(tf.constant([1], dtype=tf.int32)),
        epsilon=epsilon,
    )

    noisy_frame = np.squeeze(noisy_frame, axis=0)
    noisy_frame = np.clip(noisy_frame, 0, 1)
    noisy_frame = (noisy_frame * 255).astype(np.uint8)
    noisy_frame = cv.resize(
        noisy_frame, (original_dimensions[1], original_dimensions[0])
    )

    noise = np.abs(noisy_frame - original_frame)

    if frame_num % 100 == 0:
        cv.imwrite(f"images/{frame_num}-original.png", original_frame)
        cv.imwrite(f"images/{frame_num}-noisy.png", noisy_frame)
        cv.imwrite(f"images/{frame_num}-noise.png", noise)

    return noisy_frame


def process_frame(frame: np.ndarray) -> np.ndarray:
    frame = cv.resize(frame, (256, 256))
    frame = img_to_array(frame).flatten() / 255.0
    frame = frame.reshape(-1, 256, 256, 3)
    return model.predict(frame, verbose=0)


def main() -> None:
    video = "/dcs/large/u2204489/faceforensics/fake/09_13__kitchen_pan__21H6XSPE.mp4"
    video = cv.VideoCapture(video)

    frame_num = 0
    fake_frames = 0
    perturbated_fake_frames = 0

    while video.isOpened():
        success, frame = video.read()
        if not success:
            break

        print("Processing frame:", frame_num)

        perturbated_frame = perturbate_frame(frame, frame_num)

        prediction = process_frame(frame)
        if len(prediction) == 0 or np.argmax(prediction) == 1:
            fake_frames += 1
        prediction = process_frame(perturbated_frame)
        if len(prediction) == 0 or np.argmax(prediction) == 1:
            perturbated_fake_frames += 1

        frame_num += 1
    video.release()

    print(fake_frames)
    print(perturbated_fake_frames)


if __name__ == "__main__":
    main()
