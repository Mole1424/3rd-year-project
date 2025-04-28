from pathlib import Path

import tensorflow as tf
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
    AveragePooling2D,
    BatchNormalization,
    Conv1D,
    Conv2D,
    Dense,
    DepthwiseConv2D,
    Dropout,
    Flatten,
    GlobalAveragePooling1D,
    GlobalAveragePooling2D,
    Input,
    Layer,
    MaxPooling2D,
    ReLU,
    add,
)
from tensorflow.keras.saving import register_keras_serializable  # type: ignore

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

def conv2d(x: tf.Tensor, filters: int, kernel_size: int, stride: int) -> tf.Tensor:
    """common convolutional layer for PFLD"""
    x = Conv2D(filters, kernel_size, stride, padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    return ReLU()(x)

@register_keras_serializable(package="Custom", name="InvertedResidual")
class InvertedResidual(Layer):
    def __init__(
            self,
            input_channels: int,
            output_channels: int,
            stride: int,
            skip: bool,
            expand_ratio: int,
            **kwargs: dict,
    ) -> None:
        super().__init__(**kwargs)
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = stride
        self.skip = skip
        self.expand_ratio = expand_ratio
        self.hidden_dim = input_channels * expand_ratio

    def build(self) -> None:
        self.conv1 = Conv2D(self.hidden_dim, 1, 1, padding="same", use_bias=False)
        self.bn1 = BatchNormalization()
        self.relu1 = ReLU()

        self.depthwise_conv = DepthwiseConv2D(
            3, self.stride, padding="same", use_bias=False
        )
        self.bn2 = BatchNormalization()
        self.relu2 = ReLU()

        self.conv2 = Conv2D(self.output_channels, 1, 1, padding="same", use_bias=False)
        self.bn3 = BatchNormalization()

    def call(self, x: tf.Tensor) -> tf.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.depthwise_conv(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv2(out)
        out = self.bn3(out)

        if self.skip:
            return x + out
        return out

# cutom layer to concatenate multi-scale features
@register_keras_serializable(package="Custom", name="MultiScaleLayer")
class MultiScaleLayer(Layer):
    def call(self, x: tf.Tensor) -> tf.Tensor:
        x1, x2, x3 = x
        return tf.concat([x1, x2, x3], axis=1) # type: ignore

def pfld() -> Model:
    """PFLD model"""
    inputs = Input(shape=(256, 256, 3))

    x = conv2d(inputs, 64, 3, 2)
    x = conv2d(x, 64, 3, 1)

    x = InvertedResidual(64, 64, 2, False, 2)(x)
    x = InvertedResidual(64, 64, 1, True, 2)(x)
    x = InvertedResidual(64, 64, 1, True, 2)(x)
    x = InvertedResidual(64, 64, 1, True, 2)(x)
    # intermediate layer for auxiliary net to pick up from
    out1 = InvertedResidual(64, 64, 1, True, 2)(x)

    x = InvertedResidual(64, 128, 2, False, 2)(out1)
    x = InvertedResidual(128, 128, 1, False, 4)(x)
    x = InvertedResidual(128, 128, 1, True, 4)(x)
    x = InvertedResidual(128, 128, 1, True, 4)(x)
    x = InvertedResidual(128, 128, 1, True, 4)(x)
    x = InvertedResidual(128, 128, 1, True, 4)(x)
    x = InvertedResidual(128, 128, 1, True, 4)(x)

    x = InvertedResidual(128, 16, 1, False, 2)(x)

    x1 = AveragePooling2D(14)(x)
    x1 = Flatten()(x1)

    x = Conv2D(32, 3, 2, padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x2 = AveragePooling2D(7)(x)
    x2 = Flatten()(x2)

    x = Conv2D(32, 3, 2, padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x3 = Flatten()(x)

    multi_scale = MultiScaleLayer()([x1, x2, x3])
    landmarks = Dense(136)(multi_scale)

    return Model(inputs, [out1, landmarks], name="PFLD")

def auxiliary_net() -> Model:
    """AuxiliaryNet for PFLD"""
    inputs = Input((64, 64, 64))

    x = conv2d(inputs, 128, 3, 2)
    x = conv2d(x, 128, 3, 1)
    x = conv2d(x, 32, 3, 2)
    x = conv2d(x, 128, 7, 1)
    x = MaxPooling2D(3)(x)
    x = Flatten()(x)
    x = Dense(32)(x)
    output = Dense(3)(x)

    return Model(inputs, output, name="AuxiliaryNet")

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
        "PFLD": pfld(),
        "AuxiliaryNet": auxiliary_net(),
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
