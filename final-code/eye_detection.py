# an adapted version of "Eye landmarks detection via weakly supervised learning"
# available at https://www.sciencedirect.com/science/article/pii/S0031320319303772
# thanks to Bin Huang, Renwen Chen, Qinbang Zhou, and Wang Xu

# rpn work was aided by "Facial landmark detection by semi-supervised deep learning"
# available at https://www.sciencedirect.com/science/article/pii/S0031320319303772
# thanks to Xin Tang, Fang Guo, Jianbing Shen, and Tianyuan Du
# and "Region Proposal Network(RPN) (in Faster RCNN) from scratch in Keras"
# available at https://martian1231-py.medium.com/region-proposal-network-rpn-in-faster-rcnn-from-scratch-in-keras-1311c67c13cf
# thanks to Akash Kewar

# faster rcnn from tf2-faster-rcnn
# available at https://github.com/hxuaj/tf2-faster-rcnn/tree/main/model
# thanks to hxuaj

from pathlib import Path
from typing import Generator

import cv2 as cv
import numpy as np
import tensorflow as tf
from roi_pooling import ROIPoolingLayer
from tensorflow.image import non_max_suppression  # type: ignore
from tensorflow.keras import Input, Model  # type: ignore
from tensorflow.keras.applications import VGG16  # type: ignore
from tensorflow.keras.layers import (  # type: ignore
    LSTM,
    Conv2D,
    Dense,
    Lambda,
    RepeatVector,
    TimeDistributed,
)
from tensorflow.keras.losses import (  # type: ignore
    Huber,
    MeanSquaredError,
)
from tensorflow.keras.optimizers import SGD, Adam  # type: ignore


class RPN(Model):
    def __init__(self, ratio: float) -> None:
        super(RPN, self).__init__()
        self.ratio = ratio
        self.num_points_per_anchor = 9

        self.conv1 = Conv2D(
            512, (3, 3), activation="relu", padding="same", name="rpn_conv1"
        )
        self.regressor = Conv2D(
            8 * self.num_points_per_anchor,
            (1, 1),
            activation="linear",
            padding="same",
            name="rpn_regressor",
        )
        self.classifier = Conv2D(
            1 * self.num_points_per_anchor,
            (1, 1),
            activation="sigmoid",
            padding="same",
            name="rpn_classifier",
        )
        self.eye_landmark_classifier = Conv2D(
            24,  # 6 landmarks * 2 eyes * 2 pointss
            (1, 1),
            activation="linear",
            padding="same",
            name="rpn_eye_landmark_classifier",
        )

        self.num_landmarks = 6

    def call(
        self, features: tf.Tensor, image: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        generates region proposals and initial eye landmarks

        features: features from the backbone
        image: input image

        output is 3 tensors:
        - region proposals (x, y, w, h)
        - initial eye landmarks (x1, y1, ..., x6, y6) x2
        - anchors (x1, y1, x2, y2)
        """

        # generate anchors
        features_shape = tf.shape(features)
        image_shape = tf.shape(image)
        anchors = self.generate_anchors(
            (features_shape[1], features_shape[2]), (image_shape[1], image_shape[2])  # type: ignore
        )

        # pass through the network
        x = self.conv1(features)
        anchor_deltas = self.regressor(x)
        objectivness_scores = self.classifier(x)
        eye_landmarks = self.eye_landmark_classifier(x)

        # reduce eye landmarks from (None, None, None, 24) -> (24)
        eye_landmarks = tf.reshape(eye_landmarks, [-1, 24])

        # convert to x, y, w, h
        w_anchors = anchors[:, 2] - anchors[:, 0]  # type: ignore
        h_anchors = anchors[:, 3] - anchors[:, 1]  # type: ignore
        x_anchors = anchors[:, 0] + w_anchors / 2  # type: ignore
        y_anchors = anchors[:, 1] + h_anchors / 2  # type: ignore

        anchor_deltas = tf.reshape(anchor_deltas, [-1, tf.shape(anchors)[0], 4])  # type: ignore
        objectivness_scores = tf.reshape(
            objectivness_scores, [-1, tf.shape(anchors)[0]]  # type: ignore
        )

        # post processing
        rois = self.post_processing(
            x_anchors,
            y_anchors,
            w_anchors,
            h_anchors,
            anchor_deltas,
            objectivness_scores,
            image,
        )

        return rois, eye_landmarks, anchors

    def generate_anchors(
        self, features_shape: tuple[int, int], image_shape: tuple[int, int]
    ) -> tf.Tensor:
        """generate anchors for the image"""
        features_height, features_width = features_shape
        image_height, image_width = image_shape

        x_stride = tf.cast(image_width / features_width, tf.float32)
        y_stride = tf.cast(image_height / features_height, tf.float32)

        # find centers of each anchor
        x_centers = tf.range(x_stride / 2, image_width, x_stride, dtype=tf.float32)  # type: ignore
        y_centers = tf.range(y_stride / 2, image_height, y_stride, dtype=tf.float32)  # type: ignore
        x_centers, y_centers = tf.meshgrid(x_centers, y_centers, indexing="xy")
        centers = tf.stack(
            [tf.reshape(x_centers, [-1]), tf.reshape(y_centers, [-1])], axis=-1
        )

        # initial anchor params
        anchor_ratios = tf.constant([0.5, 1, 2], dtype=tf.float32)
        anchor_scales = tf.constant([8, 16, 32], dtype=tf.float32)

        # get height and width of each anchor
        scales, ratios = tf.meshgrid(anchor_scales, anchor_ratios, indexing="xy")
        scales = tf.reshape(scales, [-1])
        ratios = tf.reshape(ratios, [-1])
        heights = tf.reshape(tf.sqrt(scales**2 / ratios) * y_stride, [-1])
        widths = tf.reshape(heights * ratios * x_stride / y_stride, [-1])

        # compute all anchors
        num_centers = tf.shape(centers)[0]  # type: ignore
        num_anchors = tf.size(heights)

        centers = tf.tile(tf.expand_dims(centers, axis=1), [1, num_anchors, 1])
        heights = tf.tile(tf.expand_dims(heights, axis=0), [num_centers, 1])
        widths = tf.tile(tf.expand_dims(widths, axis=0), [num_centers, 1])

        x_min = centers[..., 0] - widths / 2
        y_min = centers[..., 1] - heights / 2
        x_max = centers[..., 0] + widths / 2
        y_max = centers[..., 1] + heights / 2

        anchors = tf.reshape(tf.stack([x_min, y_min, x_max, y_max], axis=-1), [-1, 4])

        # get anchors that are inside the image
        inside_mask = (
            (anchors[:, 0] >= 0.0)
            & (anchors[:, 1] >= 0.0)
            & (anchors[:, 2] <= tf.cast(image_width, tf.float32))
            & (anchors[:, 3] <= tf.cast(image_height, tf.float32))
        )

        return tf.boolean_mask(anchors, inside_mask)

    def iou(self, box1: tf.Tensor, box2: tf.Tensor) -> float:
        cx1, cy1, w1, h1 = box1
        cx2, cy2, w2, h2 = box2

        x1 = max(cx1 - w1 / 2, cx2 - w2 / 2)
        y1 = max(cy1 - h1 / 2, cy2 - h2 / 2)
        x2 = min(cx1 + w1 / 2, cx2 + w2 / 2)
        y2 = min(cy1 + h1 / 2, cy2 + h2 / 2)

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        union = w1 * h1 + w2 * h2 - intersection
        return intersection / union

    def post_processing(  # noqa: PLR0913
        self,
        x_anchors: tf.Tensor,
        y_anchors: tf.Tensor,
        w_anchors: tf.Tensor,
        h_anchors: tf.Tensor,
        anchor_deltas: tf.Tensor,
        objectivness_scores: tf.Tensor,
        image: tf.Tensor,
    ) -> tf.Tensor:
        # apply deltas
        x1 = x_anchors + w_anchors * anchor_deltas[:, :, 0]  # type: ignore
        y1 = y_anchors + h_anchors * anchor_deltas[:, :, 1]  # type: ignore
        w = w_anchors * tf.exp(anchor_deltas[:, :, 2])  # type: ignore
        h = h_anchors * tf.exp(anchor_deltas[:, :, 3])  # type: ignore

        # convert to x1, y1, x2, y2
        x1 = x1 - w / 2
        y1 = y1 - h / 2
        x2 = x1 + w
        y2 = y1 + h

        # clip to image
        predicted_anchors = tf.stack([x1, y1, x2, y2], axis=-1)
        image_shape = tf.cast(tf.shape(image)[:2], tf.float32)  # type: ignore
        max_values = tf.stack(
            [image_shape[1], image_shape[0], image_shape[1], image_shape[0]]  # type: ignore
        )
        max_values = tf.reshape(max_values, [1, 1, 4])
        predicted_anchors = tf.clip_by_value(predicted_anchors, 0.0, max_values)

        predicted_anchors = tf.reshape(predicted_anchors, [-1, 4])
        objectivness_scores = tf.reshape(objectivness_scores, [-1])

        # non max suppression
        nms_indices = non_max_suppression(
            predicted_anchors,
            objectivness_scores,
            max_output_size=self.num_landmarks,
        )

        # get back to x, y, w, h
        x1 = tf.gather(predicted_anchors[:, 0], nms_indices)
        y1 = tf.gather(predicted_anchors[:, 1], nms_indices)
        x2 = tf.gather(predicted_anchors[:, 2], nms_indices)
        y2 = tf.gather(predicted_anchors[:, 3], nms_indices)

        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        return tf.stack([x, y, w, h], axis=-1)


class FasterRCNN(Model):
    def __init__(self, ratio: float) -> None:
        super(FasterRCNN, self).__init__()
        self.backbone = self.shared_convolutional_model()
        self.rpn = RPN(ratio)
        self.roi_pooling = ROIPoolingLayer(2, 2, name="roi_pooling")  # type: ignore
        self.fc1 = Dense(512, activation="relu", name="frcnn_fc1")
        self.fc2 = Dense(256, activation="relu", name="frcnn_fc2")

    def call(self, image: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        takes in an image tensor

        return 3 tensors of the following form:
        - region proposals (x, y, w, h)
        - initial eye landmarks (x1, y1, ..., x6, y6) x2
        - anchors (x1, y1, x2, y2)
        """
        features = self.backbone(image)
        rois = self.rpn(features)
        pooled = self.roi_pooling([features, rois])  # type: ignore
        x = self.fc1(pooled)
        return self.fc2(x)

    def shared_convolutional_model(self) -> Model:
        """shared convolutional area to act as the backbone"""
        # VGG16 without the top and final max pooling layer
        vgg16 = VGG16(include_top=False, input_shape=(None, None, 3))
        custom_vgg = vgg16.get_layer("block5_conv3").output
        return Model(
            inputs=vgg16.input, outputs=custom_vgg, name="shared_convolutional_model"
        )


def recurrent_learning_module(time_steps: int) -> Model:
    """
    fine tunes the original eye landmarks using LSTM
    """

    num_ltsm_units = 256
    num_landmarks = 6

    initial_landmarks = Input(shape=(num_landmarks * 2,))

    x = Dense(256, activation="relu", name="rlm_fc1")(initial_landmarks)
    x = RepeatVector(time_steps, name="rlm_repeat_vector")(x)
    x = LSTM(num_ltsm_units, return_sequences=True, name="rlm_lstm")(x)

    x = TimeDistributed(Dense(num_landmarks * 2, activation="relu", name="rlm_time"))(x)

    def update_landmarks(inputs: tf.Tensor) -> tf.Tensor:
        initial, deltas = inputs
        updated = tf.cumsum(deltas, axis=1) + tf.expand_dims(initial, axis=1)
        return updated[:, -1, :]

    x = Lambda(update_landmarks, name="rlm_lambda")([initial_landmarks, x])

    return Model(
        inputs=initial_landmarks,
        outputs=x,
        name="recurrent_learning_module",
    )


class EyeLandmarks(Model):
    def __init__(self, ratio: float, time_steps: int) -> None:
        super(EyeLandmarks, self).__init__()
        self.faster_rcnn = FasterRCNN(ratio)
        self.recurrent_learning_module = recurrent_learning_module(time_steps)

    def call(self, image: tf.Tensor) -> tf.Tensor:
        """
        return 3 tensors of the following form:
        - initial eye landmarks (x1, y1, ..., x6, y6) x2
        """
        _, eyes, _ = self.faster_rcnn(image)

        # extract the eye landmarks and bounding boxes
        left_eye_landmarks = eyes[:12]
        right_eye_landmarks = eyes[12:]

        # fine tune the eye landmarks
        left_eye_landmarks = self.recurrent_learning_module(left_eye_landmarks)
        right_eye_landmarks = self.recurrent_learning_module(right_eye_landmarks)

        # append to the end
        return tf.concat([left_eye_landmarks, right_eye_landmarks], axis=0)  # type: ignore


def dataset_generator(path: str, file_type: str, fully_labeled: bool) -> Generator:
    for file in Path(path).glob("*.txt"):
        image = cv.imread(str(file).replace("txt", file_type))
        original_shape = image.shape
        label = (
            np.loadtxt(file, max_rows=6),
            (
                np.loadtxt(file, skiprows=6).flatten()
                if fully_labeled
                else np.zeros((24,))
            ),
            1 if fully_labeled else 0,
        )
        yield image, label, original_shape


def load_dataset(path: str, file_type: str, fully_labeled: bool) -> tf.data.Dataset:
    return tf.data.Dataset.from_generator(
        lambda: dataset_generator(path, file_type, fully_labeled),
        output_signature=(
            tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),  # type: ignore
            (
                tf.TensorSpec(shape=(6, 4), dtype=tf.float32),  # type: ignore
                tf.TensorSpec(shape=(24,), dtype=tf.float32),  # type: ignore
                tf.TensorSpec(shape=(), dtype=tf.uint8),  # type: ignore
            ),
            tf.TensorSpec(shape=(3,), dtype=tf.int32),  # type: ignore
        ),
    )


def load_datsets(path_to_large: str, batch_size: int) -> tf.data.Dataset:
    path_to_datasets = path_to_large + "eyes/"

    datasets = [
        load_dataset(path_to_datasets + "300w/", "png", True),
        load_dataset(path_to_datasets + "helen/", "jpg", True),
        load_dataset(path_to_datasets + "lfpw/trainset/", "png", True),
        load_dataset(path_to_datasets + "afw/", "jpg", True),
        load_dataset(path_to_datasets + "aflw/", "jpg", False),
    ]

    return combine_datasets(datasets, batch_size)


def combine_datasets(datasets: list, batch_size: int) -> tf.data.Dataset:
    full_dataset = datasets[0]
    for dataset in datasets[1:]:
        full_dataset = full_dataset.concatenate(dataset)
    full_dataset = full_dataset.shuffle(32946)
    return full_dataset.padded_batch(
        batch_size, padded_shapes=([None, None, 3], ([6, 4], [24], []), [3])
    ).prefetch(tf.data.AUTOTUNE)


def remove_padding(image: tf.Tensor, shape: tf.Tensor) -> tf.Tensor:
    return image[: shape[0], : shape[1], :]  # type: ignore


def get_backbone_rpn_model(eye_landmarks: EyeLandmarks) -> Model:
    image = Input(shape=(None, None, 3))
    original_shape = Input(shape=(3,), dtype=tf.int32)
    image = Lambda(
        remove_padding, output_shape=(None, None, 3), name="backbone_lambda"
    )([image, original_shape])
    features = eye_landmarks.faster_rcnn.backbone(image)
    rois, eye_landmarkers, anchors = eye_landmarks.faster_rcnn.rpn(features, image)
    return Model(
        inputs=[image, original_shape], outputs=[rois, eye_landmarkers, anchors]
    )


@tf.function
def frcnn_loss(
    y_true: tuple[tf.Tensor, tf.Tensor, tf.Tensor],
    y_pred: tuple[tf.Tensor, tf.Tensor, tf.Tensor],
) -> tf.Tensor:
    ratio = 3

    rois, eye_landmarks, anchors = y_pred
    true_bboxs, true_landmarks, _ = y_true

    x_anchors = anchors[:, 0] + anchors[:, 2] / 2  # type: ignore
    y_anchors = anchors[:, 1] + anchors[:, 3] / 2  # type: ignore
    w_anchors = anchors[:, 2] - anchors[:, 0]  # type: ignore
    h_anchors = anchors[:, 3] - anchors[:, 1]  # type: ignore

    # find ious between anchors and ground truths
    ious = compute_ious(anchors, true_bboxs)

    best_anchors, labels = compute_labels(ious, anchors)

    # subsample anchors
    labels = filter_labels(labels, 256, ratio)

    # find deltas
    w_truths = true_bboxs[:, 2] - true_bboxs[:, 0]  # type: ignore
    h_truths = true_bboxs[:, 3] - true_bboxs[:, 1]  # type: ignore
    x_truths = true_bboxs[:, 0] + w_truths / 2  # type: ignore
    y_truths = true_bboxs[:, 1] + h_truths / 2  # type: ignore

    eps = tf.constant(1e-7, dtype=tf.float32)
    w_anchors = tf.maximum(w_anchors, eps)
    h_anchors = tf.maximum(h_anchors, eps)

    tx = (x_truths - x_anchors) / w_anchors
    ty = (y_truths - y_anchors) / h_anchors
    tw = tf.math.log(w_truths / w_anchors)
    th = tf.math.log(h_truths / h_anchors)

    deltas = tf.stack([tx, ty, tw, th], axis=1)

    positive_indices = tf.where(tf.equal(labels, 1))
    best_anchor_positives = tf.boolean_mask(best_anchors, tf.equal(labels, 1))
    best_anchor_positives = tf.squeeze(best_anchor_positives, axis=-1)
    update_values = tf.gather(deltas, best_anchor_positives)

    offsets = tf.zeros((tf.shape(anchors)[0], 4), dtype=tf.float32)  # type: ignore
    offsets = tf.tensor_scatter_nd_update(offsets, positive_indices, update_values)
    offsets = tf.expand_dims(offsets, axis=0)

    converted_left_eye = convert_eye_landmarks(eye_landmarks[:6], rois[0])  # type: ignore
    converted_right_eye = convert_eye_landmarks(eye_landmarks[6:], rois[1])  # type: ignore
    converted_landmarks = tf.concat([converted_left_eye, converted_right_eye], axis=0)

    # find the loss
    return frcnn_loss_function(y_true, (offsets, converted_landmarks), ratio)  # type: ignore


def compute_ious(anchors: tf.Tensor, truths: tf.Tensor) -> tf.Tensor:
    """
    computes the ious between anchors and ground truths
    boxes in form cx, cy, w, h
    """

    x1 = tf.maximum(anchors[:, 0] - anchors[:, 2] / 2, truths[:, 0] - truths[:, 2] / 2)  # type: ignore
    y1 = tf.maximum(anchors[:, 1] - anchors[:, 3] / 2, truths[:, 1] - truths[:, 3] / 2)  # type: ignore
    x2 = tf.minimum(anchors[:, 0] + anchors[:, 2] / 2, truths[:, 0] + truths[:, 2] / 2)  # type: ignore
    y2 = tf.minimum(anchors[:, 1] + anchors[:, 3] / 2, truths[:, 1] + truths[:, 3] / 2)  # type: ignore

    intersection = tf.maximum(0.0, x2 - x1) * tf.maximum(0.0, y2 - y1)
    union = anchors[:, 2] * anchors[:, 3] + truths[:, 2] * truths[:, 3] - intersection  # type: ignore

    return intersection / union


def compute_labels(ious: tf.Tensor, anchors: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    # find best anchors
    best_iou_indexes = tf.argmax(ious, axis=1)
    best_ious = tf.gather(ious, best_iou_indexes, axis=1)
    best_anchors = tf.argmax(ious, axis=0)
    best_gt_ious = tf.gather(ious, best_anchors, axis=0)
    best_anchors = tf.where(ious == best_gt_ious)[:, 0]

    # assign labels to anchors
    # 0. background, 1. foreground, -1. ignore
    iou_threshold = 0.5
    labels = tf.fill([tf.shape(anchors)[0]], -1)  # type: ignore

    # background indeces added to 0
    background_indices = tf.where(best_ious < iou_threshold)
    labels = tf.tensor_scatter_nd_update(
        labels,
        background_indices,
        tf.zeros_like([tf.shape(background_indices)[0]], dtype=tf.int32),  # type: ignore
    )
    # foreground indices added to 1
    foreground_indices = tf.expand_dims(best_anchors, axis=1)
    labels = tf.tensor_scatter_nd_update(
        labels,
        foreground_indices,
        tf.ones_like([tf.shape(foreground_indices)[0]], dtype=tf.int32),  # type: ignore
    )
    high_iou_indices = tf.where(best_ious >= iou_threshold)
    labels = tf.tensor_scatter_nd_update(
        labels,
        high_iou_indices,
        tf.ones_like([tf.shape(high_iou_indices)[0]], dtype=tf.int32),  # type: ignore
    )

    return best_anchors, labels


def filter_labels(labels: tf.Tensor, batch_size: int, ratio: int) -> tf.Tensor:
    """proposes a balanced (according to ratio) set of labels of size batch_size"""
    num_positives = int(batch_size * (1 - ratio))
    positive_indices = tf.where(tf.equal(labels, 1))[:, 0]
    num_current_positives = tf.shape(positive_indices)[0]  # type: ignore

    # add more positives if necessary
    positive_diff = num_current_positives - num_positives
    if positive_diff > 0:
        added_positives = tf.random.shuffle(positive_indices)[:positive_diff]
        update_values = tf.fill([positive_diff], -1)
        labels = tf.tensor_scatter_nd_update(
            labels, tf.expand_dims(added_positives, axis=1), update_values
        )

    num_negatives = batch_size - num_positives
    negative_indices = tf.where(tf.equal(labels, 0))[:, 0]
    num_current_negatives = tf.shape(negative_indices)[0]  # type: ignore

    # add more negatives if necessary
    negative_diff = num_current_negatives - num_negatives
    if negative_diff > 0:
        added_negatives = tf.random.shuffle(negative_indices)[:negative_diff]
        update_neg_values = tf.fill([negative_diff], -1)
        labels = tf.tensor_scatter_nd_update(
            labels, tf.expand_dims(added_negatives, axis=1), update_neg_values
        )

    return labels


def convert_eye_landmarks(proposal: tf.Tensor, bounding_box: tf.Tensor) -> tf.Tensor:
    indices_x = tf.range(1, 12, 2)
    indices_y = tf.range(2, 11, 2)

    proposal_x = (tf.gather(proposal, indices_x) - bounding_box[0]) / bounding_box[2]  # type: ignore
    proposal_y = (tf.gather(proposal, indices_y) - bounding_box[1]) / bounding_box[3]  # type: ignore

    proposal = tf.tensor_scatter_nd_update(
        proposal, tf.expand_dims(indices_x, axis=1), proposal_x
    )
    return tf.tensor_scatter_nd_update(
        proposal, tf.expand_dims(indices_y, axis=1), proposal_y
    )


def frcnn_loss_function(
    y_true: tuple[tf.Tensor, tf.Tensor, tf.Tensor],
    y_pred: tuple[tf.Tensor, tf.Tensor],
    ratio: int,
) -> tf.Tensor:
    # lambda_c = 1 / ratio
    lambda_l = 1
    lambda_s = ratio

    offsets, predicted_eye_landmarks = y_pred
    true_bboxs, true_eye_landmarks, has_landmarks = y_true

    total_loss = lambda_l * Huber()(true_bboxs, offsets)

    mse_loss = lambda_s * MeanSquaredError()(
        true_eye_landmarks, predicted_eye_landmarks
    )
    total_loss += mse_loss * tf.cast(has_landmarks, tf.float32)

    return total_loss  # type: ignore


def rlm_loss_function(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    cumulative_prediction = tf.cumsum(y_pred, axis=1)
    final_prediction = cumulative_prediction[:, -1, :]
    return tf.reduce_mean(tf.reduce_sum(tf.square(y_true - final_prediction), axis=-1))


def step_decay(epoch: int, lr: float) -> float:
    if epoch % 8 == 0 and epoch:
        return lr * 0.8
    return lr


@tf.function
def rpn_train_loop(
    rpn_model: Model, dataset: tf.data.Dataset, optimizer: SGD, epochs: int
) -> None:
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for _, (x, y, original_shape) in enumerate(dataset):
            with tf.GradientTape() as tape:
                predictions = rpn_model([x, original_shape])
                loss = frcnn_loss(y, predictions)
            grads = tape.gradient(loss, rpn_model.trainable_weights)
            optimizer.apply_gradients(zip(grads, rpn_model.trainable_weights))  # type: ignore
        optimizer.learning_rate = step_decay(epoch, optimizer.learning_rate)


def train_model() -> None:
    path_to_large = "/dcs/large/u2204489/"

    batch_size = 2

    # check if the dataset already exists
    if Path(path_to_large + "eyedataset").exists():
        dataset = tf.data.Dataset.load(path_to_large + "eyedataset")
        dataset.save(path_to_large + "eyedataset")
    else:
        dataset = load_datsets(path_to_large, batch_size)

    # create model
    eye_landmarks = EyeLandmarks(3, 10)
    eye_landmarks.faster_rcnn.backbone.trainable = False

    # train FRCNN in 4step process as described by
    # "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks"
    # https://ieeexplore.ieee.org/abstract/document/7485869 (section 3.2)

    # step 1 train soley RPN
    eye_landmarks.faster_rcnn.rpn.trainable = True
    eye_landmarks.faster_rcnn.fc1.trainable = False
    eye_landmarks.faster_rcnn.fc2.trainable = False

    backbone_rpn_model = get_backbone_rpn_model(eye_landmarks)

    rpn_train_loop(
        backbone_rpn_model,
        dataset,
        SGD(learning_rate=0.02, momentum=0.9, weight_decay=0.0001),
        8 * 10,
    )
    backbone_rpn_model.save_weights(f"{path_to_large}step1.keras")

    # step 2, train fast-RCNN with fixed RPN
    eye_landmarks.faster_rcnn.rpn.trainable = False
    eye_landmarks.faster_rcnn.roi_pooling.trainable = True
    eye_landmarks.faster_rcnn.fc1.trainable = True
    eye_landmarks.faster_rcnn.fc2.trainable = True

    rpn_train_loop(
        backbone_rpn_model,
        dataset,
        SGD(learning_rate=0.02, momentum=0.9, weight_decay=0.0001),
        8 * 10,
    )
    eye_landmarks.faster_rcnn.save_weights(f"{path_to_large}step2.keras")

    # step 3, train RPN with fixed fast-RCNN
    eye_landmarks.faster_rcnn.rpn.trainable = True
    eye_landmarks.faster_rcnn.roi_pooling.trainable = False
    eye_landmarks.faster_rcnn.fc1.trainable = False
    eye_landmarks.faster_rcnn.fc2.trainable = False

    rpn_train_loop(
        backbone_rpn_model,
        dataset,
        SGD(learning_rate=0.02, momentum=0.9, weight_decay=0.0001),
        8 * 10,
    )
    eye_landmarks.faster_rcnn.save_weights(f"{path_to_large}step3.keras")

    # step 4, train everything together
    eye_landmarks.faster_rcnn.roi_pooling.trainable = True
    eye_landmarks.faster_rcnn.fc1.trainable = True
    eye_landmarks.faster_rcnn.fc2.trainable = True

    rpn_train_loop(
        backbone_rpn_model,
        dataset,
        SGD(learning_rate=0.02, momentum=0.9, weight_decay=0.0001),
        8 * 10,
    )
    eye_landmarks.faster_rcnn.save_weights(f"{path_to_large}step4.keras")

    eye_landmarks.faster_rcnn.trainable = False

    # use faster_rcnn to generate starting landmarks for the RLM
    x_train = []
    y_train = []
    for x, y in dataset:  # type: ignore
        x_train.append(x)
        y_train.append(y)

    predictions = eye_landmarks.faster_rcnn(x_train)
    l_eye, r_eye, _ = predictions
    eyes = tf.concat([l_eye, r_eye], axis=0)

    eye_landmarks.recurrent_learning_module.compile(
        optimizer=Adam(learning_rate=0.0001), loss=rlm_loss_function
    )
    eye_landmarks.recurrent_learning_module.fit(
        eyes,
        y_train,
        batch_size=batch_size,
        epochs=10,
    )

    eye_landmarks.save_weights(f"{path_to_large}final.keras")
    print("Training complete :)")


if __name__ == "__main__":
    train_model()
