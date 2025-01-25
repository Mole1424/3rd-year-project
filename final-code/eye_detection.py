# an adapted version of "Eye landmarks detection via weakly supervised learning"
# available at https://www.sciencedirect.com/science/article/pii/S0031320319303772
# thanks to Bin Huang, Renwen Chen, Qinbang Zhou, and Wang Xu

# rpn work was aided by "Facial landmark detection by semi-supervised deep learning"
# available at https://www.sciencedirect.com/science/article/pii/S0031320319303772
# thanks to Xin Tang, Fang Guo, Jianbing Shen, and Tianyuan Du
# and "Region Proposal Network(RPN) (in Faster RCNN) from scratch in Keras"
# available at https://martian1231-py.medium.com/region-proposal-network-rpn-in-faster-rcnn-from-scratch-in-keras-1311c67c13cf
# thanks to Akash Kewar

import numpy as np
import pandas as pd
import tensorflow as tf
from roi_pooling import ROIPoolingLayer
from tensorflow.keras import Input, Model  # type: ignore
from tensorflow.keras.applications import VGG16  # type: ignore
from tensorflow.keras.layers import (  # type: ignore
    LSTM,
    Concatenate,
    Dense,
    GlobalMaxPool2D,
    Lambda,
    RepeatVector,
    TimeDistributed,
)
from tensorflow.keras.losses import CategoricalCrossentropy, Huber  # type: ignore


def shared_convolutional_model() -> Model:
    """shared convolutional area to act as the backbone"""
    # VGG16 without the top and final max pooling layer
    vgg16 = VGG16(include_top=False, input_shape=(None, None, 3))
    custom_vgg = vgg16.get_layer("block5_conv3").output
    return Model(
        inputs=vgg16.input, outputs=custom_vgg, name="shared_convolutional_model"
    )


def facial_classification_network() -> Model:
    """
    facial classification network to classify facial features
    (1. Eyebrow, 2. Eye, 3. Nose, 4. Mouth)
    """
    input = Input(shape=(None, None, 3))

    x = shared_convolutional_model()(input)

    x = GlobalMaxPool2D()(x)
    x = Dense(512, activation="relu")(x)
    x = Dense(512, activation="relu")(x)
    x = Dense(4, activation="softmax")(x)

    return Model(inputs=input, outputs=x, name="facial_classification_network")


def iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    intersection over union
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    x1 = max(x1, x2)
    y1 = max(y1, y2)
    x2 = min(x1 + w1, x2 + w2)
    y2 = min(y1 + h1, y2 + h2)

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = w1 * h1 + w2 * h2 - intersection
    if union == 0:
        return 0

    return intersection / union


def train_regional_proposal_network(
    image: np.ndarray, bounding_boxes: np.ndarray
) -> Model:
    """
    Regional proposal network to detect facial regions
    Output:
        - Label in the form (1, 0, 0, 0)
        - 6x region proposals in the form (x, y, w, h)
        - 12x landmarks in the form (x1, y1, ..., x6, y6)
    """
    image_height, image_width, _ = image.shape

    # labels are 0..4 for 4 facial regions and 5 for background
    labels = np.zeros(5)

    # run through vgg16
    backbones = shared_convolutional_model()
    features = backbones.predict(np.expand_dims(image, axis=0))
    _, feature_width, feature_height, _ = features.shape

    # calculate stride for anchors
    x_stride, y_stride = image_width / feature_width, image_height / feature_height

    # find centers of each anchor
    x_centers = np.arange(x_stride / 2, image_width, x_stride)
    y_centers = np.arange(y_stride / 2, image_height, y_stride)
    centers = np.array(np.meshgrid(x_centers, y_centers, indexing="xy")).T.reshape(
        -1, 2
    )

    # initial anchor params
    anchor_ratios = [0.5, 1, 2]
    anchor_scales = [8, 16, 32]
    anchors = np.zeros(
        (feature_width * feature_height * len(anchor_ratios) * len(anchor_scales), 4)
    )

    # generate anchors for all centers
    for i, (x, y) in enumerate(centers):
        for ratio in anchor_ratios:
            for scale in anchor_scales:
                h = np.sqrt(scale**2 / ratio) * y_stride
                w = h * ratio * x_stride / y_stride
                anchors[i] = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]

    # we only care about anchors that are inside the image
    inside_anchors = anchors[
        (anchors[:, 0] >= 0)
        & (anchors[:, 1] >= 0)
        & (anchors[:, 2] <= image_width)
        & (anchors[:, 3] <= image_height)
    ]

    # find iou for each anchor with each bounding box
    ious = np.array(
        [
            [iou(anchor, ground_truth) for ground_truth in bounding_boxes]
            for anchor in inside_anchors
        ]
    )

    return Model()  # keep pylance happy for the time being


def faster_rcnn() -> Model:
    """
    the faster rcnn model to detect facial reigons, classifications, and landmarks
    """
    input = Input(shape=(None, None, 3))

    x = shared_convolutional_model()(input)
    rpn = reigonal_proposal_network()(x)

    x = ROIPoolingLayer(2, 2)([x, rpn])

    x = Dense(512, activation="relu")(x)
    x = Dense(256, activation="relu")(x)

    return Model(inputs=input, outputs=x, name="faster_rcnn")


num_reigon_proposals = 3


@tf.function
def faster_rcnn_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    lambda_c = 1 / num_reigon_proposals
    lambda_l = 1
    lambda_s = num_reigon_proposals

    total_loss = 0
    for i in range(num_reigon_proposals):
        predicted_label = y_pred[i, :4]
        true_label = y_true[i, :4]
        total_loss += lambda_c * CategoricalCrossentropy()(true_label, predicted_label)

        if predicted_label >= 1:
            predicted_bbox = y_pred[i, 4:8]
            true_bbox = y_true[i, 4:8]
            total_loss += lambda_l * Huber(reduction="sum")(true_bbox, predicted_bbox)
        else:
            predicted_landmarks = y_pred[i, 8:]
            true_landmarks = y_true[i, 8:]
            squared_diff = tf.square(predicted_landmarks - true_landmarks)
            distance_per_landmark = tf.reduce_sum(squared_diff, axis=-1)
            total_loss += lambda_s * tf.reduce_sum(distance_per_landmark, axis=-1)

    return total_loss


time_steps = 4


def recurrent_learning_module() -> Model:
    """
    fine tunes the original eye landmarks using LSTM
    """

    num_ltsm_units = 256
    num_landmarks = 6

    initial_landmarks = Input(shape=(num_landmarks * 2,))
    eye_reigonal_proposal = Input(shape=(None, None, 3))

    x = Concatenate()([initial_landmarks, eye_reigonal_proposal])

    x = Dense(256, activation="relu")(x)
    x = RepeatVector(time_steps)(x)
    x = LSTM(num_ltsm_units, return_sequences=True)(x)

    x = TimeDistributed(Dense(num_landmarks * 2, activation="relu"))(x)

    def update_landmarks(inputs: tf.Tensor) -> tf.Tensor:
        initial, deltas = inputs
        updated = tf.cumsum(deltas, axis=1) + tf.expand_dims(initial, axis=1)
        return updated[:, -1, :]

    x = Lambda(update_landmarks)([initial_landmarks, x])

    return Model(
        inputs=[initial_landmarks, eye_reigonal_proposal],
        outputs=x,
        name="recurrent_learning_module",
    )


@tf.function
def recurrent_learning_module_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    cumulative_prediction = tf.cumsum(y_pred, axis=1)
    final_prediction = cumulative_prediction[:, -1, :]
    return tf.reduce_mean(tf.reduce_sum(tf.square(y_true - final_prediction), axis=-1))
