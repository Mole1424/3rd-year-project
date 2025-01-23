# an adapted version of Eye landmarks detection via weakly supervised learning
# available at https://www.sciencedirect.com/science/article/pii/S0031320319303772
# thanks to Bin Huang, Renwen Chen, Qinbang Zhou, and Wang Xu

import tensorflow as tf
from roi_pooling import ROIPoolingLayer
from tensorflow.keras import Input, Model  # type: ignore
from tensorflow.keras.layers import (  # type: ignore
    LSTM,
    Concatenate,
    Conv2D,
    Dense,
    GlobalMaxPool2D,
    Lambda,
    MaxPool2D,
    RepeatVector,
    TimeDistributed,
)
from tensorflow.keras.losses import CategoricalCrossentropy, Huber  # type: ignore


def shared_convolutional_model() -> Model:
    """shared convolutional area to act as the backbone"""
    input = Input(shape=(None, None, 3))

    x = Conv2D(64, (3, 3), activation="relu", padding="same")(input)
    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = MaxPool2D()(x)  # default for MaxPool2D is (2, 2)

    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = MaxPool2D()(x)

    x = Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = MaxPool2D()(x)

    x = Conv2D(512, (3, 3), activation="relu", padding="same")(x)
    x = Conv2D(512, (3, 3), activation="relu", padding="same")(x)
    x = Conv2D(512, (3, 3), activation="relu", padding="same")(x)
    x = MaxPool2D()(x)

    x = Conv2D(512, (3, 3), activation="relu", padding="same")(x)
    x = Conv2D(512, (3, 3), activation="relu", padding="same")(x)
    x = Conv2D(512, (3, 3), activation="relu", padding="same")(x)

    return Model(inputs=input, outputs=x, name="shared_convolutional_model")


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


def reigonal_proposal_network() -> Model:
    """
    reigonal proposal network to detect facial reigons
    output is:
        label in the form (1, 0, 0, 0)
        reigon proposals in the form (x, y, w, h)
        landmarks in the form (x1, y1, ..., x6, y6)
    """
    pass


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
