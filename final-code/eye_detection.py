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


import numpy as np
import tensorflow as tf
from roi_pooling import ROIPoolingLayer
from tensorflow.image import non_max_suppression  # type: ignore
from tensorflow.keras import Input, Model  # type: ignore
from tensorflow.keras.applications import VGG16  # type: ignore
from tensorflow.keras.callbacks import LearningRateScheduler  # type: ignore
from tensorflow.keras.initializers import RandomNormal  # type: ignore
from tensorflow.keras.layers import (  # type: ignore
    LSTM,
    Concatenate,
    Conv2D,
    Dense,
    Lambda,
    RepeatVector,
    TimeDistributed,
)
from tensorflow.keras.losses import CategoricalCrossentropy, Huber  # type: ignore
from tensorflow.keras.optimizers import SGD, Adam  # type: ignore


class RPN(Model):
    def __init__(self, ratio: float) -> None:
        super(RPN, self).__init__()
        self.ratio = ratio
        self.initaliser = RandomNormal()
        self.num_points_per_anchor = 9

        self.conv1 = Conv2D(
            512,
            (3, 3),
            activation="relu",
            padding="same",
            kernel_initializer=self.initaliser,
        )
        self.regressor = Conv2D(
            8 * self.num_points_per_anchor,
            (1, 1),
            activation="linear",
            padding="same",
            kernel_initializer=self.initaliser,
        )
        self.classifier = Conv2D(
            1 * self.num_points_per_anchor,
            (1, 1),
            activation="sigmoid",
            padding="same",
            kernel_initializer=self.initaliser,
        )
        # self.eye_landmark_classifier = Conv2D(
        #     6 * self.num_points_per_anchor,
        #     (1, 1),
        #     activation="linear",
        #     padding="same",
        #     kernel_initializer=self.initalisers,
        # )

        self.num_landmarks = 6

    @tf.function
    def call(
        self,
        features: tf.Tensor,
        image: tf.Tensor,
    ) -> tf.Tensor:
        """
        generates region proposals and initial eye landmarks

        features: features from the backbone
        image: input image
        ground_truths: ground truth eye landmarks in the form
        - left eye landmarks (x1, y1, ..., x6, y6)
        - right eye landmarks
        - left eye bounding box (center_x, center_y, w, h)
        - right eye bounding box
        - left eye brow
        - right eye brow
        - nose
        - mouth

        output is a 2d tensor:
        - 1st row is left eye landmarks (x1, y1, ..., x6, y6)
        - 2nd row is right eye landmarks
        - 3rd row is left eye bounding box (center_x, center_y, w, h)
        - 4th row is right eye bounding box
        - 5th row is left eye brow
        - 6th row is right eye brow
        - 7th row is nose
        - 8th row is mouth
        - 9th row is the anchor boxes
        """

        # generate anchors
        anchors = self.generate_anchors(
            (features.shape[1], features.shape[2]), (image.shape[1], image.shape[2])  # type: ignore
        )

        # pass through the network
        x = self.conv1(features)
        anchor_deltas = self.regressor(x)
        objectivness_scores = self.classifier(x)
        # eye_landmarks = self.eye_landmark_classifier(x)

        # convert to x, y, w, h
        w_anchors = anchors[:, 2] - anchors[:, 0]  # type: ignore
        h_anchors = anchors[:, 3] - anchors[:, 1]  # type: ignore
        x_anchors = anchors[:, 0] + w_anchors / 2  # type: ignore
        y_anchors = anchors[:, 1] + h_anchors / 2  # type: ignore

        anchor_deltas = anchor_deltas.reshape(-1, len(anchors), 4)
        objectivness_scores = objectivness_scores.reshape(-1, len(anchors))

        # post processing
        rois = self.post_processing(
            anchors,
            x_anchors,
            y_anchors,
            w_anchors,
            h_anchors,
            anchor_deltas,
            objectivness_scores,
            image,
        )

        return tf.concat([rois, anchors], axis=0)  # type: ignore

    def generate_anchors(
        self, features_shape: tuple[int, int], image_shape: tuple[int, int]
    ) -> tf.Tensor:
        """generate anchors for the image"""
        features_height, features_width = features_shape
        image_height, image_width = image_shape

        x_stride, y_stride = (
            image_width / features_width,
            image_height / features_height,
        )

        # find centers of each anchor
        x_centers = np.arange(x_stride / 2, image_width, x_stride)
        y_centers = np.arange(y_stride / 2, image_height, y_stride)
        centers = np.array(np.meshgrid(x_centers, y_centers, indexing="xy")).T.reshape(
            -1, 2
        )

        # initial anchor params
        anchor_ratios = [0.5, 1, 2]
        anchor_scales = [8, 16, 32]
        anchors = tf.zeros(
            (
                features_width
                * features_height
                * len(anchor_ratios)
                * len(anchor_scales),
                4,
            )
        )

        # generate anchors for all centers
        for i, (x, y) in enumerate(centers):
            for ratio in anchor_ratios:
                for scale in anchor_scales:
                    h = tf.sqrt(scale**2 / ratio) * y_stride
                    w = h * ratio * x_stride / y_stride
                    anchors[i] = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]

        # get anchors that are inside the image
        inside_anchors_ids = anchors[
            (anchors[:, 0] >= 0)
            & (anchors[:, 1] >= 0)
            & (anchors[:, 2] <= image_width)
            & (anchors[:, 3] <= image_height)
        ][0]

        return anchors[inside_anchors_ids]

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

    def convert_landmakrs(
        self, proposal: tf.Tensor, bounding_box: tf.Tensor
    ) -> tf.Tensor:
        g_x, g_y, g_w, g_h = bounding_box
        for i in range(1, 12, 2):
            proposal[i] = (proposal[i] - g_x) / g_w  # type: ignore
            proposal[i + 1] = (proposal[i + 1] - g_y) / g_h  # type: ignore
        return proposal

    def post_processing(  # noqa: PLR0913
        self,
        anchors: tf.Tensor,
        x_anchors: tf.Tensor,
        y_anchors: tf.Tensor,
        w_anchors: tf.Tensor,
        h_anchors: tf.Tensor,
        anchor_deltas: tf.Tensor,
        objectivness_scores: tf.Tensor,
        image: tf.Tensor,
    ) -> tf.Tensor:
        # apply deltas
        predicted_anchors = np.zeros((len(anchors), 4))
        predicted_anchors[:, 0] = x_anchors + w_anchors * anchor_deltas[:, :, 0]  # type: ignore
        predicted_anchors[:, 1] = y_anchors + h_anchors * anchor_deltas[:, :, 1]  # type: ignore
        predicted_anchors[:, 2] = w_anchors * np.exp(anchor_deltas[:, :, 2])  # type: ignore
        predicted_anchors[:, 3] = h_anchors * np.exp(anchor_deltas[:, :, 3])  # type: ignore

        # convert proposals from (center_x, center_y, w, h) to (x1, y1, x2, y2)
        predicted_anchors[:, 0] = predicted_anchors[:, 0] - predicted_anchors[:, 2] / 2
        predicted_anchors[:, 1] = predicted_anchors[:, 1] - predicted_anchors[:, 3] / 2
        predicted_anchors[:, 2] = predicted_anchors[:, 0] + predicted_anchors[:, 2]
        predicted_anchors[:, 3] = predicted_anchors[:, 1] + predicted_anchors[:, 3]

        # clip proposals to the image
        predicted_anchors[:, 0] = np.clip(predicted_anchors[:, 0], 0, image.shape[1])  # type: ignore
        predicted_anchors[:, 1] = np.clip(predicted_anchors[:, 1], 0, image.shape[0])  # type: ignore
        predicted_anchors[:, 2] = np.clip(predicted_anchors[:, 2], 0, image.shape[1])  # type: ignore
        predicted_anchors[:, 3] = np.clip(predicted_anchors[:, 3], 0, image.shape[0])  # type: ignore

        # non max suppression
        nms_indices = non_max_suppression(
            predicted_anchors,
            objectivness_scores,
            max_output_size=self.num_landmarks,
        )
        return tf.gather(predicted_anchors, nms_indices)


class FasterRCNN(Model):
    def __init__(self, ratio: float) -> None:
        super(FasterRCNN, self).__init__()
        self.backbone = self.shared_convolutional_model()
        self.rpn = RPN(ratio)
        self.roi_pooling = ROIPoolingLayer(2, 2)
        self.fc1 = Dense(512, activation="relu")
        self.fc2 = Dense(256, activation="relu")

    def call(self, image: tf.Tensor) -> tf.Tensor:
        """
        takes in an image tensor

        return a tensor of the following form:
        - 1st row is left eye landmarks (x1, y1, ..., x6, y6)
        - 2nd row is right eye landmarks
        - 3rd row is left eye bounding box (center_x, center_y, w, h)
        - 4th row is right eye bounding box
        - 5th row is left eye brow
        - 6th row is right eye brow
        - 7th row is nose
        - 8th row is mouth
        - 9th row is the anchor boxes
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


class EyeLandmarks(Model):
    def __init__(self, ratio: float, time_steps: int) -> None:
        super(EyeLandmarks, self).__init__()
        self.faster_rcnn = FasterRCNN(ratio)
        self.recurrent_learning_module = recurrent_learning_module(time_steps)

    def call(self, image: tf.Tensor) -> tf.Tensor:
        """
        return tensor of the following form:
        - 1st row is left eye landmarks (x1, y1, ..., x6, y6)
        - 2nd row is right eye landmarks
        - 3rd row is left eye bounding box (center_x, center_y, w, h)
        - 4th row is right eye bounding box
        - 5th row is left eye brow
        - 6th row is right eye brow
        - 7th row is nose
        - 8th row is mouth
        - 9th row is the anchor boxes
        - 10th row is the updated left eye landmarks
        - 11th row is the updated right eye landmarks
        """
        rois = self.faster_rcnn(image)

        # extract the eye landmarks and bounding boxes
        left_eye_landmarks = rois[0]
        right_eye_landmarks = rois[1]
        left_eye_box = rois[2]
        right_eye_box = rois[3]

        # fine tune the eye landmarks
        left_eye_landmarks = self.recurrent_learning_module(
            left_eye_landmarks, left_eye_box
        )
        right_eye_landmarks = self.recurrent_learning_module(
            right_eye_landmarks, right_eye_box
        )

        # append to the end
        return tf.concat([rois, left_eye_landmarks, right_eye_landmarks], axis=0)  # type: ignore


@tf.function
def frcnn_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    ratio = 3

    anchors = y_pred[8]  # type: ignore
    x_anchors = anchors[:, 0] + anchors[:, 2] / 2
    y_anchors = anchors[:, 1] + anchors[:, 3] / 2
    w_anchors = anchors[:, 2] - anchors[:, 0]
    h_anchors = anchors[:, 3] - anchors[:, 1]

    ground_truths = y_true[:8]  # type: ignore

    # find ious between anchors and ground truths
    ious = np.zeros((len(anchors), len(ground_truths)), dtype=np.float32)
    for i, anchor in enumerate(anchors):
        for j, truth in enumerate(ground_truths[2:]):  # type: ignore
            ious[i, j] = iou(anchor, truth)

    best_anchors, labels = compute_labels(ious, anchors)

    # subsample anchors
    labels = filter_labels(labels, 256, ratio)

    # find deltas
    relevant_truths = ground_truths[2:]  # type: ignore
    w_truths = relevant_truths[:, 2] - relevant_truths[:, 0]
    h_truths = relevant_truths[:, 3] - relevant_truths[:, 1]
    x_truths = relevant_truths[:, 0] + w_truths / 2
    y_truths = relevant_truths[:, 1] + h_truths / 2

    eps = np.finfo(anchors.dtype).eps  # type: ignore
    w_anchors = np.maximum(w_anchors, eps)
    h_anchors = np.maximum(h_anchors, eps)

    tx = (x_truths - x_anchors) / w_anchors
    ty = (y_truths - y_anchors) / h_anchors
    tw = np.log(w_truths / w_anchors)
    th = np.log(h_truths / h_anchors)

    deltas = np.stack([tx, ty, tw, th], axis=1)

    offsets = np.zeros((len(anchors), 4))
    offsets[labels == 1] = deltas[best_anchors]
    offsets = np.expand_dims(offsets, axis=0)

    # find the loss
    return frcnn_loss_function(
        tf.convert_to_tensor(offsets, dtype=tf.float32), y_pred, ratio
    )


def iou(box1: tf.Tensor, box2: tf.Tensor) -> float:
    """
    intersection over union
    """
    cx1, cy1, w1, h1 = box1
    cx2, cy2, w2, h2 = box2

    x1 = max(cx1 - w1 / 2, cx2 - w2 / 2)
    y1 = max(cy1 - h1 / 2, cy2 - h2 / 2)
    x2 = min(cx1 + w1 / 2, cx2 + w2 / 2)
    y2 = min(cy1 + h1 / 2, cy2 + h2 / 2)

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = w1 * h1 + w2 * h2 - intersection
    return intersection / union


def compute_labels(
    ious: np.ndarray, anchors: tf.Tensor
) -> tuple[np.ndarray, np.ndarray]:
    # find best anchors
    best_iou_indexes = np.argmax(ious, axis=1)
    best_ious = ious[np.arange(len(anchors)), best_iou_indexes]
    best_anchors = np.argmax(ious, axis=0)
    best_gt_ious = ious[best_anchors, np.arange(ious.shape[1])]
    best_anchors = np.where(ious == best_gt_ious)[0]

    # assign labels to anchors
    # 0. background, 1. foreground, -1. ignore
    iou_threshold = 0.5
    labels = np.zeros(len(anchors))
    labels.fill(-1)
    labels[best_ious < iou_threshold] = 0
    labels[best_anchors] = 1
    labels[best_ious >= iou_threshold] = 1

    return best_anchors, labels


def filter_labels(labels: np.ndarray, batch_size: int, ratio: int) -> np.ndarray:
    num_positives = int(batch_size * (1 - ratio))
    positive_indices = np.where(labels == 1)[0]

    if len(positive_indices) > num_positives:
        labels[
            np.random.choice(
                positive_indices,
                len(positive_indices) - num_positives,
                replace=False,
            )
        ] = -1
    num_negatives = batch_size - num_positives
    negative_indices = np.where(labels == 0)[0]
    if len(negative_indices) > num_negatives:
        labels[
            np.random.choice(
                negative_indices,
                len(negative_indices) - num_negatives,
                replace=False,
            )
        ] = -1

    return labels


def frcnn_loss_function(
    y_true: tf.Tensor, y_pred: tf.Tensor, ratio: float
) -> tf.Tensor:
    lambda_c = 1 / ratio
    lambda_l = 1
    lambda_s = ratio

    total_loss = 0
    predicted_label = y_pred[0]  # type: ignore
    true_label = y_true[0]  # type: ignore
    total_loss += lambda_c * CategoricalCrossentropy()(true_label, predicted_label)

    for i in [1, 2]:
        predicted_landmarks = y_pred[i]  # type: ignore
        true_landmarks = y_true[i]  # type: ignore
        squared_diff = tf.square(predicted_landmarks - true_landmarks)
        distance_per_landmark = tf.reduce_sum(squared_diff, axis=-1)
        total_loss += lambda_s * tf.reduce_sum(distance_per_landmark, axis=-1)
    for i in [2, 3, 4, 5, 6, 7, 8]:
        predicted_bbox = y_pred[i]  # type: ignore
        true_bbox = y_true[i]  # type: ignore
        total_loss += lambda_l * Huber(reduction="sum")(true_bbox, predicted_bbox)

    return total_loss  # type: ignore


def rlm_loss_function(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    cumulative_prediction = tf.cumsum(y_pred, axis=1)
    final_prediction = cumulative_prediction[:, -1, :]
    return tf.reduce_mean(tf.reduce_sum(tf.square(y_true - final_prediction), axis=-1))


def step_decay(epoch: int, lr: float) -> float:
    if epoch % 8 == 0 and epoch:
        return lr * 0.8
    return lr


def train_model() -> None:
    # get data TODO
    x_train = ...
    y_train = ...

    path_to_weights = "/dcs/large/u2204489/"

    # create model
    eye_landmarks = EyeLandmarks(3, 10)
    eye_landmarks.faster_rcnn.backbone.trainable = False

    eye_landmarks.faster_rcnn.rpn.compile(
        optimizer=SGD(learning_rate=0.02, momentum=0.9, weight_decay=0.0001),
        loss=frcnn_loss,
    )
    eye_landmarks.faster_rcnn.rpn.fit(
        x_train,
        y_train,
        batch_size=2,
        epochs=8 * 10,
        callbacks=[LearningRateScheduler(step_decay)],
    )
    eye_landmarks.faster_rcnn.rpn.save_weights(f"{path_to_weights}step1.keras")

    eye_landmarks.faster_rcnn.rpn.trainable = False
    eye_landmarks.faster_rcnn.roi_pooling.trainable = True
    eye_landmarks.faster_rcnn.fc1.trainable = True
    eye_landmarks.faster_rcnn.fc2.trainable = True

    eye_landmarks.faster_rcnn.compile(
        optimizer=SGD(learning_rate=0.02, momentum=0.9, weight_decay=0.0001),
        loss=frcnn_loss,
    )
    eye_landmarks.faster_rcnn.fit(
        x_train,
        y_train,
        batch_size=2,
        epochs=8 * 10,
        callbacks=[LearningRateScheduler(step_decay)],
    )
    eye_landmarks.faster_rcnn.save_weights(f"{path_to_weights}step2.keras")

    eye_landmarks.faster_rcnn.rpn.trainable = True
    eye_landmarks.faster_rcnn.roi_pooling.trainable = False
    eye_landmarks.faster_rcnn.fc1.trainable = False
    eye_landmarks.faster_rcnn.fc2.trainable = False

    eye_landmarks.faster_rcnn.compile(
        optimizer=SGD(learning_rate=0.02, momentum=0.9, weight_decay=0.0001),
        loss=frcnn_loss,
    )
    eye_landmarks.faster_rcnn.fit(
        x_train,
        y_train,
        batch_size=2,
        epochs=8 * 10,
        callbacks=[LearningRateScheduler(step_decay)],
    )
    eye_landmarks.faster_rcnn.save_weights(f"{path_to_weights}step3.keras")

    eye_landmarks.faster_rcnn.roi_pooling.trainable = True
    eye_landmarks.faster_rcnn.fc1.trainable = True
    eye_landmarks.faster_rcnn.fc2.trainable = True

    eye_landmarks.faster_rcnn.compile(
        optimizer=SGD(learning_rate=0.02, momentum=0.9, weight_decay=0.0001),
        loss=frcnn_loss,
    )
    eye_landmarks.faster_rcnn.fit(
        x_train,
        y_train,
        batch_size=2,
        epochs=8 * 10,
        callbacks=[LearningRateScheduler(step_decay)],
    )
    eye_landmarks.faster_rcnn.save_weights(f"{path_to_weights}step4.keras")

    eye_landmarks.faster_rcnn.trainable = False

    # use faster_rcnn to generate starting landmarks for the RLM
    predictions = eye_landmarks.faster_rcnn(x_train)
    l_eye, r_eye, l_box, r_box, _ = predictions

    eye_landmarks.recurrent_learning_module.compile(
        optimizer=Adam(learning_rate=0.0001), loss=rlm_loss_function
    )
    eye_landmarks.recurrent_learning_module.fit(
        zip([l_eye, l_box], [r_eye, r_box]),
        y_train,
        batch_size=2,
        epochs=10,
    )

    eye_landmarks.save_weights(f"{path_to_weights}final.keras")
    print("Training complete :)")


if __name__ == "__main__":
    train_model()
