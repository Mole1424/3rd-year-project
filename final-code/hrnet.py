# HRNET from paper "High-Resolution Representations for Labeling Pixels and Regions"
# https://arxiv.org/abs/1904.04514v1
# thanks to Ke Sun et al.
# adapted to tensorflow from the official pytorch implementation (https://github.com/HRNet/HRNet-Facial-Landmark-Detection)

import tensorflow as tf
from tensorflow.keras import Model, Sequential  # type: ignore
from tensorflow.keras.layers import (  # type: ignore
    BatchNormalization,
    Conv2D,
    Layer,
    ReLU,
    Softmax,
)

MOMENTUM = 0.99


class BasicBlock(Layer):
    expansion = 1

    def __init__(self, filters: int, stride: int = 1, downsample: Layer = None) -> None:
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2D(filters, 3, stride, padding="same", use_bias=False)
        self.bn1 = BatchNormalization(momentum=MOMENTUM, epsilon=1e-5)
        self.relu = ReLU()
        self.conv2 = Conv2D(filters, 3, stride, padding="same", use_bias=False)
        self.bn2 = BatchNormalization(momentum=MOMENTUM, epsilon=1e-5)
        self.downsample = downsample
        self.stride = stride

    def call(self, x: tf.Tensor) -> tf.Tensor:
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        return self.relu(x + residual)  # type: ignore


class Bottleneck(Layer):
    expansion = 4

    def __init__(self, filters: int, stride: int = 1, downsample: Layer = None) -> None:
        super(Bottleneck, self).__init__()
        self.conv1 = Conv2D(filters, 1, stride, padding="same", use_bias=False)
        self.bn1 = BatchNormalization(momentum=MOMENTUM, epsilon=1e-5)
        self.conv2 = Conv2D(filters, 3, stride, padding="same", use_bias=False)
        self.bn2 = BatchNormalization(momentum=MOMENTUM, epsilon=1e-5)
        self.conv3 = Conv2D(
            filters * self.expansion, 1, stride, padding="same", use_bias=False
        )
        self.bn3 = BatchNormalization(momentum=MOMENTUM, epsilon=1e-5)
        self.relu = ReLU()

        self.downsample = downsample
        self.stride = stride

    def call(self, x: tf.Tensor) -> tf.Tensor:
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        return self.relu(x + residual)  # type: ignore


class HRNetModule(Model):
    def __init__(
        self,
        num_branches: int,
        block: Layer,
        num_blocks: list[int],
        num_inchannels: list[int],
        num_filters: list[int],
    ) -> None:
        super(HRNetModule, self).__init__()

        self.num_branches = num_branches
        self.block = block
        self.num_blocks = num_blocks
        self.num_inchannels = num_inchannels
        self.num_filters = num_filters
        self.check_branches()
        self.branches = [self.make_one_branch(i) for i in range(self.num_branches)]
        self.fuse_layers = self.make_fuse_layers()
        self.relu = ReLU()

    def check_branches(self) -> None:
        assert self.num_branches == len(
            self.num_blocks
        ), "NUM_BRANCHES != len(NUM_BLOCKS)"
        assert self.num_branches == len(
            self.num_filters
        ), "NUM_BRANCHES != len(NUM_FILTERS)"
        assert self.num_branches == len(
            self.num_inchannels
        ), "NUM_BRANCHES != len(NUM_INCHANNELS)"

    def make_one_branch(self, branch_index: int, stride: int = 1) -> Layer:
        downsample = None
        if (
            stride != 1
            or self.num_inchannels[branch_index]
            != self.num_filters[branch_index] * self.block.expansion
        ):
            downsample = Sequential(
                Conv2D(
                    self.num_filters[branch_index] * self.block.expansion,
                    1,
                    stride,
                    use_bias=False,
                ),
                BatchNormalization(momentum=MOMENTUM, epsilon=1e-5),
            )

        layers = []
        layers.append(self.block(self.num_filters[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = (
            self.num_filters[branch_index] * self.block.expansion
        )
        for _ in range(1, self.num_blocks[branch_index]):
            layers.append(self.block(self.num_filters[branch_index]))  # noqa: PERF401

        return Sequential(layers)

    def make_fuse_layers(self) -> list[Layer] | None:
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []

        for i in range(num_branches):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        Sequential(
                            Conv2D(num_inchannels[j], 1, use_bias=False),
                            BatchNormalization(momentum=MOMENTUM, epsilon=1e-5),
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    convs = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outs = num_inchannels[i]
                            convs.append(
                                Sequential(
                                    Conv2D(num_outs, 3, 2, use_bias=False),
                                    BatchNormalization(momentum=MOMENTUM, epsilon=1e-5),
                                )
                            )
                        else:
                            num_outs = num_inchannels[j]
                            convs.append(
                                Sequential(
                                    Conv2D(num_outs, 3, 2, use_bias=False),
                                    BatchNormalization(momentum=MOMENTUM, epsilon=1e-5),
                                    ReLU(),
                                )
                            )
                fuse_layer.append(Sequential(convs))  # type: ignore
            fuse_layers.append(fuse_layer)

        return fuse_layers

    def call(self, x: tf.Tensor) -> tf.Tensor:
        if self.num_branches == 1:
            return [self.branches[0](x[0])]  # type: ignore

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])  # type: ignore

        x_fuse = []
        for i in range(len(self.fuse_layers)):  # type: ignore
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])  # type: ignore
            for j in range(1, self.num_branches):
                if i == j:
                    y += x[j]  # type: ignore
                elif j > i:
                    width_output = x[i].shape[2]  # type: ignore
                    height_output = x[i].shape[1]  # type: ignore
                    y += tf.image.resize(
                        self.fuse_layers[i][j](x[j]),  # type: ignore
                        (height_output, width_output),
                        method="bilinear",
                    )
                else:
                    y += self.fuse_layers[i][j](x[j])  # type: ignore
            x_fuse.append(self.relu(y))
        return x_fuse  # type: ignore


config = {
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


class HRNET(Model):
    def __init__(self) -> None:
        self.inplanes = 64
        super(HRNET, self).__init__()

        self.conv1 = Conv2D(64, 3, 2, padding="same", use_bias=False)
        self.bn1 = BatchNormalization(momentum=MOMENTUM, epsilon=1e-5)
        self.conv2 = Conv2D(64, 3, 2, padding="same", use_bias=False)
        self.bn2 = BatchNormalization(momentum=MOMENTUM, epsilon=1e-5)
        self.relu = ReLU()
        self.sf = Softmax(1)  # ??????????????

        self.layer1 = self.make_layer(Bottleneck, 64, 64, 4)

        self.stage2_cfg = config["STAGE2"]
        block = BasicBlock
        num_channels = self.stage2_cfg["NUM_CHANNELS"]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition1 = self.make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self.make_stage(self.stage2_cfg, num_channels)

        self.stage3_cfg = config["STAGE3"]
        block = BasicBlock
        num_channels = self.stage3_cfg["NUM_CHANNELS"]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition2 = self.make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self.make_stage(self.stage3_cfg, num_channels)

        self.stage4_cfg = self.config["STAGE4"]
        block = BasicBlock
        num_channels = self.stage4_cfg["NUM_CHANNELS"]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition3 = self.make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self.make_stage(self.stage4_cfg, num_channels)

        final_channels = sum(pre_stage_channels)

        self.head = Sequential(
            Conv2D(final_channels, 1, "same"),
            BatchNormalization(momentum=MOMENTUM, epsilon=1e-5),
            ReLU(),
            Conv2D(68, 1, "same"),
        )

    def make_transition_layer(
        self, num_channels_pre_layer: list[int], num_channels_cur_layer: list[int]
    ) -> Layer:
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        Sequential(
                            Conv2D(
                                num_channels_cur_layer[i],
                                3,
                                padding="same",
                                use_bias=False,
                            ),
                            BatchNormalization(momentum=MOMENTUM, epsilon=1e-5),
                            ReLU(),
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                convs = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] if j == i else inchannels
                    convs.append(
                        Sequential(
                            Conv2D(outchannels, 3, 2, padding="same"),
                            BatchNormalization(momentum=MOMENTUM, epsilon=1e-5),
                            ReLU(),
                        )
                    )
                transition_layers.append(Sequential(convs))

        return transition_layers

    def make_layer(
        self, block: Layer, inplanes: int, planes: int, blocks: int, stride: int = 1
    ) -> Layer:
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = Sequential(
                Conv2D(planes * block.expansion, 1, stride, use_bias=False),
                BatchNormalization(momentum=MOMENTUM, epsilon=1e-5),
            )

        layers = []
        layers.append(block(planes, stride, downsample))
        for _ in range(1, blocks):
            layers.append(block(planes))  # noqa: PERF401

        return Sequential(layers)

    def make_stage(
        self, config: dict, num_inchannels: list[int]
    ) -> tuple[list[Layer], list[int]]:
        num_modules = config["NUM_MODULES"]
        num_branches = config["NUM_BRANCHES"]
        num_blocks = config["NUM_BLOCKS"]
        num_channels = config["NUM_CHANNELS"]

        modules = []
        for _ in range(num_modules):
            modules.append(
                HRNetModule(
                    num_branches,
                    BasicBlock,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                )
            )
            num_inchannels = modules[-1].num_inchannels

        return modules, num_inchannels

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg["NUM_BRANCHES"]):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)  # type: ignore

        x_list = []
        for i in range(self.stage3_cfg["NUM_BRANCHES"]):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)  # type: ignore

        x_list = []
        for i in range(self.stage4_cfg["NUM_BRANCHES"]):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)  # type: ignore

        height, width = y_list[0].shape[1], y_list[0].shape[2]
        x1 = tf.image.resize(x[0], (height, width), method="bilinear")  # type: ignore
        x2 = tf.image.resize(x[1], (height, width), method="bilinear")  # type: ignore
        x3 = tf.image.resize(x[2], (height, width), method="bilinear")  # type: ignore
        x4 = tf.image.resize(x[3], (height, width), method="bilinear")  # type: ignore

        x = tf.concat([x1, x2, x3, x4], axis=1)  # type: ignore
        return self.head(x)
