# an adapted version of Eye landmarks detection via weakly supervised learning
# available at https://www.sciencedirect.com/science/article/pii/S0031320319303772
# thanks to Bin Huang, Renwen Chen, Qinbang Zhou, and Wang Xu

import tensorflow as tf
from roi_pooling import ROIPoolingLayer
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    LSTM,
    Conv2D,
    Dense,
    Input,
    MaxPooling2D,
)
from tesorflow.keras.optimizers import Adam


def shared_convolutional_module() -> Model:
    input_layer = Input(shape=(256, 256, 3))
    x = Conv2D(64, (3, 3), activation="relu", padding="same")(input_layer)
    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(512, (3, 3), activation="relu", padding="same")(x)
    x = Conv2D(512, (3, 3), activation="relu", padding="same")(x)
    x = Conv2D(512, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(512, (3, 3), activation="relu", padding="same")(x)
    x = Conv2D(512, (3, 3), activation="relu", padding="same")(x)
    x = Conv2D(512, (3, 3), activation="relu", padding="same")(x)

    return Model(input_layer, x, name="shared_convolutional_module")


def reigon_proposal_network() -> Model:
    pass


def recurrent_module() -> Model:
    lstm_units = 256
    time_steps = 4

    # Input layers
    eye_region_input = Input(shape=(64, 64, 3), name="eye_region")
    # 6 landmarks (x, y) = 12
    initial_landmarks_input = Input(shape=(12,), name="initial_landmarks")

    # Repeat initial landmarks for time steps
    repeated_landmarks = tf.repeat(
        tf.expand_dims(initial_landmarks_input, axis=1), time_steps, axis=1
    )

    # LSTM layers
    lstm = LSTM(lstm_units, return_sequences=True, return_state=False, name="lstm")
    lstm_out = lstm(repeated_landmarks)

    # Fully connected layers for refining landmark positions
    refined_landmarks = []
    for t in range(time_steps):
        x = Dense(128, activation="relu", name=f"fc1_time_{t}")(lstm_out[:, t, :])
        x = Dense(12, name=f"fc2_time_{t}")(x)  # Output for 6 landmarks (x, y) = 12
        refined_landmarks.append(x)

    # Stack refined landmarks across time steps
    refined_landmarks = tf.stack(refined_landmarks, axis=1, name="refined_landmarks")

    # Define the model
    return Model(
        inputs=[eye_region_input, initial_landmarks_input],
        outputs=refined_landmarks,
        name="RecurrentLearningModule",
    )


def eye_landmarks_detection() -> Model:
    input_layer = Input(shape=(256, 256, 3), name="input_image")

    # Shared convolutional module
    shared_conv_module = shared_convolutional_module()
    shared_conv_out = shared_conv_module(input_layer)

    # Region proposal network
    rpn = reigon_proposal_network()
    rpn_out = rpn(shared_conv_out)

    # Pooling
    roi_pooling = ROIPoolingLayer(2, 2)
    roi_pooling_out = roi_pooling([shared_conv_out, rpn_out])
    dense_1 = Dense(512, activation="relu")(roi_pooling_out)
    dense_2 = Dense(256, activation="relu")(dense_1)

    # Recurrent module
    refinement = recurrent_module()
    final_landmarks = refinement([roi_pooling_out, dense_2])

    return Model(
        inputs=input_layer, outputs=final_landmarks, name="EyeLandmarksDetection"
    )


def train_model() -> None:
    # load data
    supervised_data = ...
    weakly_supervised_data = ...

    model = eye_landmarks_detection()

    def multi_task_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return tf.reduce_mean(tf.square(y_true - y_pred))

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss=multi_task_loss,
        metrics=["mse"],
    )

    # Training loop
    for epoch in range(50):
        print(f"Epoch {epoch + 1}/50")

        # Train on supervised data
        model.fit(
            supervised_data["images"],
            supervised_data["landmarks"],
            batch_size=16,
            epochs=1,
            verbose=1,
        )

        # Train on weakly supervised data
        weakly_supervised_images = weakly_supervised_data["images"]
        model.fit(
            weakly_supervised_images,
            weakly_supervised_images,  # Dummy target for unsupervised training
            batch_size=16,
            epochs=1,
            verbose=1,
        )

    # Save the trained model
    model.save("/dcs/large/u2204489/eye_landmarks_model.keras")
