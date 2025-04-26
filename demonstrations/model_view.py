from pathlib import Path

import visualkeras
from tensorflow.keras import Model  # type: ignore
from tensorflow.keras.applications import (  # type: ignore
    VGG19,
    EfficientNetB4,
    ResNet50,
    Xception,
)
from tensorflow.keras.layers import (  # type: ignore
    Activation,
    AveragePooling1D,
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling1D,
    GlobalAveragePooling2D,
    Input,
    add,
)

INPUT_LENGTH = 512

# define all models
def mlp() -> Model:
    input = Input(shape=(INPUT_LENGTH, 1))

    x = Dropout(0.1)(input)
    x = Dense(500, activation="relu")(x)

    x = Dropout(0.2)(x)
    x = Dense(500, activation="relu")(x)

    x = Dropout(0.3)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(2, activation="softmax")(x)

    return Model(inputs=input, outputs=x)

def fcnn() -> Model:
    input = Input(shape=(INPUT_LENGTH, 1))

    x = Conv1D(128, 8, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv1D(256, 5, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv1D(128, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = GlobalAveragePooling1D()(x)
    x = Dense(2, activation="softmax")(x)

    return Model(inputs=input, outputs=x)

def time_cnn() -> Model:
    input = Input(shape=(INPUT_LENGTH, 1))

    x = Conv1D(6, 7, padding="same", activation="sigmoid")(input)
    x = AveragePooling1D(3, padding="same")(x)

    x = Conv1D(12, 7, padding="same", activation="sigmoid")(x)
    x = AveragePooling1D(3, padding="same")(x)

    x = Flatten()(x)
    x = Dense(2, activation="softmax")(x)

    return Model(inputs=input, outputs=x)


def resnet() -> Model:
    n_feature_maps = 64

    input = Input(shape=(INPUT_LENGTH, 1))

    x = Conv1D(n_feature_maps, 8, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv1D(n_feature_maps, 5, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv1D(n_feature_maps, 3, padding="same")(x)
    x = BatchNormalization()(x)

    y = Conv1D(n_feature_maps, 1, padding="same")(input)
    y = BatchNormalization()(y)

    block1 = add([x, y])
    block1 = Activation("relu")(block1)

    x = Conv1D(n_feature_maps * 2, 8, padding="same")(block1)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv1D(n_feature_maps * 2, 5, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv1D(n_feature_maps * 2, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    y = Conv1D(n_feature_maps * 2, 1, padding="same")(block1)
    y = BatchNormalization()(y)

    block2 = add([x, y])
    block2 = Activation("relu")(block2)

    x = Conv1D(n_feature_maps * 2, 8, padding="same")(block2)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv1D(n_feature_maps * 2, 5, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv1D(n_feature_maps * 2, 3, padding="same")(x)
    x = BatchNormalization()(x)

    y = BatchNormalization()(block2)

    block3 = add([x, y])
    block3 = Activation("relu")(block3)

    x = GlobalAveragePooling1D()(block3)
    x = Dense(2, activation="softmax")(x)

    return Model(inputs=input, outputs=x)

def efficientnet_b4() -> Model:
    backbone = EfficientNetB4(include_top=False, input_shape=(256, 256, 3))

    inputs = Input(shape=(256, 256, 3))
    x = backbone(inputs)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(2, activation="softmax")(x)

    return Model(inputs=inputs, outputs=x)

def xception() -> Model:
    backbone = Xception(include_top=False, input_shape=(256, 256, 3))

    inputs = Input(shape=(256, 256, 3))
    x = backbone(inputs)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(2, activation="softmax")(x)

    return Model(inputs=inputs, outputs=x)

def vgg19() -> Model:
    backbone = VGG19(include_top=False, input_shape=(256, 256, 3))

    inputs = Input(shape=(256, 256, 3))
    x = backbone(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dense(2, activation="softmax")(x)

    return Model(inputs=inputs, outputs=x)

def resnet50() -> Model:
    backbone = ResNet50(include_top=False, input_shape=(256, 256, 3))

    inputs = Input(shape=(256, 256, 3))
    x = backbone(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(2, activation="softmax")(x)

    return Model(inputs=inputs, outputs=x)

def main() -> None:
    # delete all "models_*" images in the images folder
    for image in Path("images").glob("models_*"):
        image.unlink()

    # create models
    models = {
        "MLP": mlp(),
        "FCNN": fcnn(),
        "Time CNN": time_cnn(),
        "ResNet": resnet(),
        "EfficientNetB4": efficientnet_b4(),
        "Xception": xception(),
        "VGG19": vgg19(),
        "ResNet50": resnet50(),
    }

    # models which have a backbone
    backbone_names = [
        "EfficientNetB4",
        "Xception",
        "VGG19",
        "ResNet50",
    ]

    # iterate over models and create images
    for name, model in models.items():
        print(f"Model: {name}")

        visualkeras.layered_view(
            model,
            to_file=f"images/models_{name}.png",
            legend=True,
            max_xy=200, # reduce the size of some models
            # this is hackily added to add the name of the backbone to the legend
            backbone_name=name if name in backbone_names else None,
        )

    print("done :)")


if __name__ == "__main__":
    main()
