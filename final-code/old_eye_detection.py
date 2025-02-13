###################################################################################################
# MARK:THIS CODE DOES NOT WORK AND IS NO LONGER BEING DEVELOPED, IT IS HERE FOR ARCHIVAL PURPOSES #
###################################################################################################

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

import sys
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
    Flatten,
    Lambda,
    RepeatVector,
    TimeDistributed,
)
from tensorflow.keras.losses import Huber, MeanSquaredError  # type: ignore
from tensorflow.keras.optimizers import SGD, Adam  # type: ignore


class RPN(Model):
    def __init__(self, ratio: float) -> None:
        super(RPN, self).__init__()
        self.ratio = ratio
        self.num_points_per_anchor = 9

        # pre-convolutional layer for pre-processing
        self.conv1 = Conv2D(
            512, (3, 3), activation="relu", padding="same", name="rpn_conv1"
        )
        # reigon location identifier
        self.regressor = Conv2D(
            4 * self.num_points_per_anchor,
            (1, 1),
            activation="linear",
            padding="same",
            name="rpn_regressor",
        )
        # objectivness score
        self.classifier = Conv2D(
            1 * self.num_points_per_anchor,
            (1, 1),
            activation="sigmoid",
            padding="same",
            name="rpn_classifier",
        )
        # eye landmark location identifier
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
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        generates region proposals and initial eye landmarks

        features: features from the backbone
        image: input image

        output is 2 tensors:
        - region proposals (cx, cy, w, h)
        - initial eye landmarks (x1, y1, ..., x6, y6) x2
        - anchors (for loss calculation)
        - offsets (for loss calculation)
        """
        batch_size = tf.shape(image)[0]  # type: ignore

        # generate anchors
        features_shape = tf.shape(features)
        image_shape = tf.shape(image)
        anchors, mask = self.generate_anchors(
            (features_shape[1], features_shape[2]), (image_shape[1], image_shape[2])  # type: ignore
        )

        anchors = tf.tile(tf.expand_dims(anchors, axis=0), [batch_size, 1, 1])

        # pass through the network
        x = self.conv1(features)
        predicted_offsets = self.regressor(x)
        objectivness_scores = self.classifier(x)
        eye_landmarks = self.eye_landmark_classifier(x)

        # reduce eye landmarks from (None, None, None, 24) -> (None, 24)
        eye_landmarks = tf.reshape(eye_landmarks, [-1, 24])

        # reshape for post processing
        predicted_offsets = tf.reshape(predicted_offsets, [batch_size, -1, 4])
        objectivness_scores = tf.reshape(objectivness_scores, [batch_size, -1, 1])

        # post processing
        def post_processing_fn(
            inputs: tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]
        ) -> tuple[tf.Tensor, tf.Tensor]:
            anchors, anchor_deltas, objectivness_scores, image = inputs
            anchor_deltas = tf.boolean_mask(anchor_deltas, mask)
            objectivness_scores = tf.boolean_mask(objectivness_scores, mask)
            return self.post_processing(
                anchors,
                anchor_deltas,
                objectivness_scores,
                image,
            )

        # apply post processing to each image in the batch
        rois, original_predictions = tf.map_fn(
            post_processing_fn,
            (anchors, predicted_offsets, objectivness_scores, image),
            fn_output_signature=(tf.float32, tf.float32),
        )  # type: ignore
        rois = tf.stack(rois, axis=0)
        original_predictions = tf.stack(original_predictions, axis=0)

        return rois, eye_landmarks, original_predictions, anchors

    def generate_anchors(
        self, features_shape: tuple[int, int], image_shape: tuple[int, int]
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """
        generate anchors for the image

        output is 2 tensors:
        - anchors (x1, y1, x2, y2)
        - mask for anchors that are inside the image
        """
        features_height, features_width = features_shape
        image_height, image_width = image_shape

        # find strides to work out anchor centers
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

        # tile centers and dimensions
        centers = tf.tile(tf.expand_dims(centers, axis=1), [1, num_anchors, 1])
        heights = tf.tile(tf.expand_dims(heights, axis=0), [num_centers, 1])
        widths = tf.tile(tf.expand_dims(widths, axis=0), [num_centers, 1])

        # reformat to x1, y1, x2, y2
        x_min = centers[:, :, 0] - widths / 2
        y_min = centers[:, :, 1] - heights / 2
        x_max = centers[:, :, 0] + widths / 2
        y_max = centers[:, :, 1] + heights / 2

        anchors = tf.reshape(tf.stack([x_min, y_min, x_max, y_max], axis=-1), [-1, 4])

        # get anchors that are inside the image
        inside_mask = (
            (anchors[:, 0] >= 0.0)
            & (anchors[:, 1] >= 0.0)
            & (anchors[:, 2] <= tf.cast(image_width, tf.float32))
            & (anchors[:, 3] <= tf.cast(image_height, tf.float32))
        )

        return tf.boolean_mask(anchors, inside_mask), inside_mask

    def post_processing(
        self,
        anchors: tf.Tensor,
        anchor_deltas: tf.Tensor,
        objectivness_scores: tf.Tensor,
        image: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """
        post processing for region proposals
        gets the best 7 via non max suppression and converts to cx, cy, w, h
        """
        # convert anchor from (x1, y1, x2, y2) to (cx, cy, w, h)
        cx = (anchors[:, 0] + anchors[:, 2]) / 2  # type: ignore
        cy = (anchors[:, 1] + anchors[:, 3]) / 2  # type: ignore
        w = anchors[:, 2] - anchors[:, 0]  # type: ignore
        h = anchors[:, 3] - anchors[:, 1]  # type: ignore

        # apply deltas
        cx = anchor_deltas[:, 0] * w + cx  # type: ignore
        cy = anchor_deltas[:, 1] * h + cy  # type: ignore
        w = tf.exp(anchor_deltas[:, 2]) * w  # type: ignore
        h = tf.exp(anchor_deltas[:, 3]) * h  # type: ignore

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        # clip to image
        predicted_anchors = tf.stack([x1, y1, x2, y2], axis=-1)
        image_shape = tf.cast(tf.shape(image)[:2], tf.float32)  # type: ignore
        max_values = tf.stack(
            [image_shape[1], image_shape[0], image_shape[1], image_shape[0]]  # type: ignore
        )
        max_values = tf.reshape(max_values, [1, 4])
        predicted_anchors = tf.clip_by_value(predicted_anchors, 0.0, max_values)

        # reshape for nms
        predicted_anchors = tf.reshape(predicted_anchors, [-1, 4])
        objectivness_scores = tf.reshape(objectivness_scores, [-1])

        # non max suppression
        nms_indices = non_max_suppression(
            predicted_anchors,
            objectivness_scores,
            max_output_size=256,
        )

        # get best anchors
        x1 = tf.gather(predicted_anchors[:, 0], nms_indices)
        y1 = tf.gather(predicted_anchors[:, 1], nms_indices)
        x2 = tf.gather(predicted_anchors[:, 2], nms_indices)
        y2 = tf.gather(predicted_anchors[:, 3], nms_indices)

        # reshape for to cx, cy, w, h
        rois = tf.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=-1)

        # get anchor offsets
        deltas_x = tf.gather(anchor_deltas[:, 0], nms_indices)  # type: ignore
        deltas_y = tf.gather(anchor_deltas[:, 1], nms_indices)  # type: ignore
        deltas_w = tf.gather(anchor_deltas[:, 2], nms_indices)  # type: ignore
        deltas_h = tf.gather(anchor_deltas[:, 3], nms_indices)  # type: ignore

        offsets = tf.stack([deltas_x, deltas_y, deltas_w, deltas_h], axis=-1)

        return rois, offsets


class FasterRCNN(Model):
    def __init__(self, ratio: float) -> None:
        super(FasterRCNN, self).__init__()
        # layers as per Bin Huang et al.
        self.backbone = self.shared_convolutional_model()
        self.rpn = RPN(ratio)
        self.roi_pooling = ROIPoolingLayer(2, 2)
        self.flatten = Flatten()
        self.fc1 = Dense(512, activation="relu", name="frcnn_fc1")
        self.fc2 = Dense(256, activation="relu", name="frcnn_fc2")
        self.fc3 = Dense(4, activation="linear", name="frcnn_fc3")

    def call(
        self, image: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        takes in an image tensor

        return 2 tensors of the following form:
        - region proposals (x, y, w, h)
        - initial eye landmarks (x1, y1, ..., x6, y6) x2
        """
        features = self.backbone(image)
        rois, eye_landmarks, anchors, original_prediction = self.rpn(features, image)
        converted_rois = self.convert_rois(rois)
        pooled = self.roi_pooling([features, converted_rois])
        x = self.flatten(pooled)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return tf.expand_dims(x, axis=0), eye_landmarks, anchors, original_prediction

    def shared_convolutional_model(self) -> Model:
        """shared convolutional area to act as the backbone"""
        vgg16 = VGG16(include_top=False, input_shape=(None, None, 3))
        custom_vgg = vgg16.get_layer("block5_conv3").output
        return Model(
            inputs=vgg16.input, outputs=custom_vgg, name="shared_convolutional_model"
        )

    def convert_rois(self, rois: tf.Tensor) -> tf.Tensor:
        """
        convert rois from (cx, cy, w, h) to (x1, y1, x2, y2)
        """
        x1 = rois[:, 0] - rois[:, 2] / 2  # type: ignore
        y1 = rois[:, 1] - rois[:, 3] / 2  # type: ignore
        x2 = rois[:, 0] + rois[:, 2] / 2  # type: ignore
        y2 = rois[:, 1] + rois[:, 3] / 2  # type: ignore
        return tf.cast(tf.stack([x1, y1, x2, y2], axis=-1), tf.int32)  # type: ignore


def recurrent_learning_module(time_steps: int) -> Model:
    """
    fine tunes the original eye landmarks using LSTM
    """

    num_ltsm_units = 256
    num_landmarks = 6

    initial_landmarks = Input(shape=(num_landmarks * 2,))

    # again layers as per Bin Huang et al.
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
        _, eyes, _, _ = self.faster_rcnn(image)

        # extract the eye landmarks and bounding boxes
        left_eye_landmarks = eyes[:12]
        right_eye_landmarks = eyes[12:]

        # fine tune the eye landmarks
        left_eye_landmarks = self.recurrent_learning_module(left_eye_landmarks)
        right_eye_landmarks = self.recurrent_learning_module(right_eye_landmarks)

        # append to the end
        return tf.concat([left_eye_landmarks, right_eye_landmarks], axis=0)  # type: ignore


def dataset_generator(path: str, file_type: str, fully_labeled: bool) -> Generator:
    """generator for the dataset"""
    for file in Path(path).glob("*.txt"):
        image = cv.imread(str(file).replace("txt", file_type))
        image = cv.resize(image, (512, 512))
        label = (
            np.loadtxt(file, max_rows=6),
            (
                np.loadtxt(file, skiprows=6).flatten()
                if fully_labeled
                else np.zeros((24,))
            ),
            1 if fully_labeled else 0,
        )
        yield image, label


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
        ),
    )


def load_datsets(path_to_large: str, batch_size: int, debug: bool) -> tf.data.Dataset:
    path_to_datasets = path_to_large + "eyes/"
    datasets = [
        load_dataset(path_to_datasets + "300w/", "png", True),
        load_dataset(path_to_datasets + "helen/", "jpg", True),
        load_dataset(path_to_datasets + "lfpw/trainset/", "png", True),
        load_dataset(path_to_datasets + "afw/", "jpg", True),
        load_dataset(path_to_datasets + "aflw/", "jpg", False),
    ]
    return combine_datasets(datasets, batch_size, debug)


def combine_datasets(datasets: list, batch_size: int, debug: bool) -> tf.data.Dataset:
    full_dataset = datasets[0]
    for dataset in datasets[1:]:
        full_dataset = full_dataset.concatenate(dataset)
    # 32946 is the total number of images in the aflw dataset
    full_dataset = full_dataset.shuffle(32946) if not debug else full_dataset
    return full_dataset.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)


def get_backbone_rpn_model(eye_landmarks: EyeLandmarks) -> Model:
    # combine RPN with the backbone
    image = Input(shape=(None, None, 3))
    features = eye_landmarks.faster_rcnn.backbone(image)
    rois, eye_landmarkers, anchors, offsets = eye_landmarks.faster_rcnn.rpn(
        features, image
    )
    return Model(inputs=image, outputs=[rois, eye_landmarkers, anchors, offsets])


def compute_ious(boxes1: tf.Tensor, boxes2: tf.Tensor) -> tf.Tensor:
    """
    compute the intersection over union of two sets of boxes
    """
    boxes1 = tf.expand_dims(boxes1, axis=1)
    boxes2 = tf.expand_dims(boxes2, axis=0)

    x1 = tf.maximum(boxes1[..., 0], boxes2[..., 0])  # type: ignore
    y1 = tf.maximum(boxes1[..., 1], boxes2[..., 1])  # type: ignore
    x2 = tf.minimum(boxes1[..., 2], boxes2[..., 2])  # type: ignore
    y2 = tf.minimum(boxes1[..., 3], boxes2[..., 3])  # type: ignore

    intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])  # type: ignore
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])  # type: ignore

    union = area1 + area2 - intersection
    return intersection / union


# MARK: GTS IS THE OFFSSETS FROM ANCHOR TO GT


@tf.function
def frcnn_loss(  # noqa: PLR0915
    y_true: tuple[tf.Tensor, tf.Tensor, tf.Tensor],
    y_pred: tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
) -> tf.Tensor:
    """
    loss function for the faster rcnn
    """

    def frcnn_loss_per_sample(  # noqa: PLR0915
        input: tuple[
            tuple[tf.Tensor, tf.Tensor, tf.Tensor],
            tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
        ],
    ) -> tf.Tensor:
        # number of +ve to -ve examples
        ratio = 3.0

        y_true, y_pred = input
        cx_rois, eye_landmarks, anchors, original_prediction = y_pred
        cx_true_bboxs, true_eye_landmarks, has_landmarks = y_true

        # convert to (x1, y1, x2, y2)
        x1_rois = tf.stack(
            [
                cx_rois[:, 0] - cx_rois[:, 2] / 2,  # type: ignore
                cx_rois[:, 1] - cx_rois[:, 3] / 2,  # type: ignore
                cx_rois[:, 0] + cx_rois[:, 2] / 2,  # type: ignore
                cx_rois[:, 1] + cx_rois[:, 3] / 2,  # type: ignore
            ],
            axis=-1,
        )
        x1_true_bboxs = tf.stack(
            [
                cx_true_bboxs[:, 0] - cx_true_bboxs[:, 2] / 2,  # type: ignore
                cx_true_bboxs[:, 1] - cx_true_bboxs[:, 3] / 2,  # type: ignore
                cx_true_bboxs[:, 0] + cx_true_bboxs[:, 2] / 2,  # type: ignore
                cx_true_bboxs[:, 1] + cx_true_bboxs[:, 3] / 2,  # type: ignore
            ],
            axis=-1,
        )
        l_eye_bbox, r_eye_bbox = x1_true_bboxs[0], x1_true_bboxs[1]  # type: ignore

        # compute ious
        ious = compute_ious(x1_rois, x1_true_bboxs)
        argmax_ious = tf.argmax(ious, axis=1)
        max_ious = tf.reduce_max(ious, axis=1)
        truth_argmax_indices = tf.argmax(ious, axis=0)

        # assign labels
        labels = tf.fill([tf.shape(anchors)[0]], -1)  # type: ignore
        iou_threshold = 0.5
        background_indices = tf.where(max_ious < iou_threshold)
        labels = tf.tensor_scatter_nd_update(
            labels,
            background_indices,
            tf.zeros([tf.shape(background_indices)[0]], dtype=tf.int32),  # type: ignore
        )
        foreground_indices = tf.expand_dims(truth_argmax_indices, axis=1)
        labels = tf.tensor_scatter_nd_update(
            labels,
            foreground_indices,
            tf.ones([tf.shape(truth_argmax_indices)[0]], dtype=tf.int32),  # type: ignore
        )
        high_iou_indices = tf.where(max_ious > iou_threshold)
        labels = tf.tensor_scatter_nd_update(
            labels,
            high_iou_indices,
            tf.ones([tf.shape(high_iou_indices)[0]], dtype=tf.int32),  # type: ignore
        )

        # subsample anchors
        num_positives = int(256 * (1 - 1 / ratio))
        num_negatives = 256 - num_positives
        positive_indices = tf.where(labels == 1)
        num_positive = tf.shape(positive_indices)[0]  # type: ignore
        if num_positive > num_positives:
            disable_indices = tf.random.shuffle(positive_indices)[
                : (num_positive - num_positives)
            ]
            labels = tf.tensor_scatter_nd_update(
                labels, disable_indices, tf.fill([tf.shape(disable_indices)[0]], -1)  # type: ignore
            )
        negative_indices = tf.where(labels == 0)
        num_negative = tf.shape(negative_indices)[0]  # type: ignore
        if num_negative > num_negatives:
            disable_indices = tf.random.shuffle(negative_indices)[
                : (num_negative - num_negatives)
            ]
            labels = tf.tensor_scatter_nd_update(
                labels, disable_indices, tf.fill([tf.shape(disable_indices)[0]], -1)  # type: ignore
            )

        # only keep positive and negative anchors
        positive_anchor_indices = tf.where(labels != 1)[:, 0]
        cx_rois = tf.gather(cx_rois, positive_anchor_indices)
        x1_rois = tf.gather(x1_rois, positive_anchor_indices)
        cx_true_bboxs = tf.gather(
            cx_true_bboxs, tf.gather(argmax_ious, positive_anchor_indices)
        )
        x1_true_bboxs = tf.gather(
            x1_true_bboxs, tf.gather(argmax_ious, positive_anchor_indices)
        )

        # compute offsets for bbox loss
        eps = tf.constant(np.finfo(cx_rois.dtype.as_numpy_dtype).eps)  # type: ignore
        h = tf.maximum(cx_rois[:, 2], eps)  # type: ignore
        w = tf.maximum(cx_rois[:, 3], eps)  # type: ignore

        dx = (cx_true_bboxs[:, 0] - cx_rois[:, 0]) / w  # type: ignore
        dy = (cx_true_bboxs[:, 1] - cx_rois[:, 1]) / h  # type: ignore
        dw = tf.math.log(cx_true_bboxs[:, 2] / w)  # type: ignore
        dh = tf.math.log(cx_true_bboxs[:, 3] / h)  # type: ignore
        offsets = tf.stack([dx, dy, dw, dh], axis=-1)

        # smooth L1 loss for bounding boxes
        bbox_loss = Huber()(
            tf.gather(original_prediction, positive_anchor_indices), offsets
        )

        # get best roi for each eye
        ious_left = tf.squeeze(compute_ious(x1_rois, l_eye_bbox), axis=1)
        ious_right = tf.squeeze(compute_ious(x1_rois, r_eye_bbox), axis=1)
        left_roi = cx_rois[tf.argmax(ious_left, axis=0)]  # type: ignore
        right_roi = cx_rois[tf.argmax(ious_right, axis=0)]  # type: ignore

        pred_left = tf.reshape(eye_landmarks[:12], [6, 2])  # type: ignore
        pred_right = tf.reshape(eye_landmarks[12:], [6, 2])  # type: ignore

        # convert eye landmarks to be relative to the roi
        converted_left = (
            pred_left - tf.stack([left_roi[0], left_roi[1]])  # type: ignore
        ) / tf.stack(
            [left_roi[2], left_roi[3]]  # type: ignore
        )
        converted_right = (
            pred_right - tf.stack([right_roi[0], right_roi[1]])  # type: ignore
        ) / tf.stack(
            [right_roi[2], right_roi[3]]  # type: ignore
        )

        converted_left = tf.reshape(converted_left, [-1])
        converted_right = tf.reshape(converted_right, [-1])

        # MSE loss (euclidean distance) for eye landmarks
        left_loss = MeanSquaredError()(converted_left, true_eye_landmarks[:12])  # type: ignore
        right_loss = MeanSquaredError()(converted_right, true_eye_landmarks[12:])  # type: ignore
        eye_loss = ratio * left_loss + right_loss * tf.cast(has_landmarks, tf.float32)

        return bbox_loss + eye_loss

    # map loff function over each sample
    return tf.reduce_mean(
        tf.map_fn(
            frcnn_loss_per_sample, (y_true, y_pred), fn_output_signature=tf.float32
        )
    )


def rlm_loss_function(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean(
        tf.reduce_sum(tf.square(y_true - tf.cumsum(y_pred, axis=1)[:, -1, :]), axis=-1)
    )


def step_decay(epoch: int, lr: float) -> float:
    if epoch % 8 == 0 and epoch:
        return lr * 0.8
    return lr


@tf.function
def rpn_train_loop(
    rpn_model: Model, dataset: tf.data.Dataset, optimizer: SGD, epochs: int
) -> None:
    """
    custom training loop for the RPN
    (because tensorflow doesnt allow passing tuples of tensors)
    """
    # loop talen from tensorflow guide
    # https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for _, (x, y) in enumerate(dataset):
            with tf.GradientTape() as tape:
                predictions = rpn_model(x)
                loss = frcnn_loss(y, predictions)
            grads = tape.gradient(loss, rpn_model.trainable_weights)
            optimizer.apply_gradients(zip(grads, rpn_model.trainable_weights))  # type: ignore
        optimizer.learning_rate = step_decay(epoch, optimizer.learning_rate)


def train_model(debug: bool) -> None:  # noqa: PLR0915
    path_to_large = "/dcs/large/u2204489/"

    # attempt to limit gpu memory usage
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    batch_size = 2  # same batch size as in the paper

    dataset = load_datsets(path_to_large, batch_size, debug)

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

    step1_weights = f"{path_to_large}eyesstep1.weights.h5"
    if Path(step1_weights).exists():
        eye_landmarks.faster_rcnn.rpn.load_weights(step1_weights)
    else:
        rpn_train_loop(
            backbone_rpn_model,
            dataset,
            SGD(learning_rate=0.02, momentum=0.9, weight_decay=0.0001),
            8 * 10,
        )
        eye_landmarks.faster_rcnn.rpn.save_weights(step1_weights)
    print("Step 1 complete")

    # step 2, train fast-RCNN with fixed RPN
    eye_landmarks.faster_rcnn.rpn.trainable = False
    eye_landmarks.faster_rcnn.roi_pooling.trainable = True
    eye_landmarks.faster_rcnn.fc1.trainable = True
    eye_landmarks.faster_rcnn.fc2.trainable = True

    step2_weights = f"{path_to_large}eyesstep2.weights.h5"
    if Path(step2_weights).exists():
        eye_landmarks.faster_rcnn.load_weights(step2_weights)
    else:
        rpn_train_loop(
            eye_landmarks.faster_rcnn,
            dataset,
            SGD(learning_rate=0.02, momentum=0.9, weight_decay=0.0001),
            8 * 10,
        )
        eye_landmarks.faster_rcnn.save_weights(step2_weights)
    print("Step 2 complete")

    # step 3, train RPN with fixed fast-RCNN
    eye_landmarks.faster_rcnn.rpn.trainable = True
    eye_landmarks.faster_rcnn.roi_pooling.trainable = False
    eye_landmarks.faster_rcnn.fc1.trainable = False
    eye_landmarks.faster_rcnn.fc2.trainable = False

    step3_weights = f"{path_to_large}eyesstep3.weights.h5"
    if Path(step3_weights).exists():
        eye_landmarks.faster_rcnn.load_weights(step3_weights)
    else:
        rpn_train_loop(
            eye_landmarks.faster_rcnn,
            dataset,
            SGD(learning_rate=0.02, momentum=0.9, weight_decay=0.0001),
            8 * 10,
        )
        eye_landmarks.faster_rcnn.save_weights(step3_weights)
    print("Step 3 complete")

    # step 4, train everything together
    eye_landmarks.faster_rcnn.roi_pooling.trainable = True
    eye_landmarks.faster_rcnn.fc1.trainable = True
    eye_landmarks.faster_rcnn.fc2.trainable = True

    step4_weights = f"{path_to_large}eyesstep4.weights.h5"
    if Path(step4_weights).exists():
        eye_landmarks.faster_rcnn.load_weights(step4_weights)
    else:
        rpn_train_loop(
            eye_landmarks.faster_rcnn,
            dataset,
            SGD(learning_rate=0.02, momentum=0.9, weight_decay=0.0001),
            8 * 10,
        )
        eye_landmarks.faster_rcnn.save_weights(step4_weights)
    print("Step 4 complete")

    eye_landmarks.faster_rcnn.trainable = False

    rlm_weights = f"{path_to_large}eyesrlm.weights.h5"
    if Path(rlm_weights).exists():
        eye_landmarks.recurrent_learning_module.load_weights(rlm_weights)
    else:
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

        eye_landmarks.save_weights(rlm_weights)
    print("Training complete :)")


if __name__ == "__main__":
    train_model(sys.argv[1] == "debug")
