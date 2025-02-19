from typing import List
from time import time

import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential  # type: ignore
from tensorflow.keras.layers import (  # type: ignore
    BatchNormalization,
    Conv2D,
    Layer,
    ReLU,
    Softmax,
)

# momentum changed to reflect tensorflow's BatchNormalization implementation
MOMENTUM = 0.99


class BasicBlock(Layer):
    """Basic Block for HRNet"""

    # how much the number of filters will be increased
    expansion = 1

    def __init__(self, filters: int, stride: int = 1, downsample: Layer = None) -> None:
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2D(filters, 3, stride, padding="same", use_bias=False)
        self.bn1 = BatchNormalization(momentum=MOMENTUM, epsilon=1e-5)
        self.relu = ReLU()
        self.conv2 = Conv2D(filters, 3, 1, padding="same", use_bias=False)
        self.bn2 = BatchNormalization(momentum=MOMENTUM, epsilon=1e-5)
        self.downsample = downsample
        self.stride = stride

    def call(self, x: tf.Tensor) -> tf.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # downsample if needed
        if self.downsample is not None:
            residual = self.downsample(x)

        return self.relu(out + residual)


class Bottleneck(Layer):
    """Bottleneck Block for HRNet"""

    expansion = 4

    def __init__(self, filters: int, stride: int = 1, downsample: Layer = None) -> None:
        super(Bottleneck, self).__init__()
        self.conv1 = Conv2D(filters, 1, 1, padding="same", use_bias=False)
        self.bn1 = BatchNormalization(momentum=MOMENTUM, epsilon=1e-5)
        self.conv2 = Conv2D(filters, 3, stride, padding="same", use_bias=False)
        self.bn2 = BatchNormalization(momentum=MOMENTUM, epsilon=1e-5)
        self.conv3 = Conv2D(
            filters * self.expansion, 1, 1, padding="same", use_bias=False
        )
        self.bn3 = BatchNormalization(momentum=MOMENTUM, epsilon=1e-5)
        self.relu = ReLU()
        self.downsample = downsample
        self.stride = stride

    def call(self, x: tf.Tensor) -> tf.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        return self.relu(out + residual)


class HRNetModule(Model):
    """HRNet Module"""

    def __init__(
        self,
        num_branches: int,
        block: Layer,
        num_blocks: List[int],
        num_inchannels: List[int],
        num_filters: List[int],
    ) -> None:
        super(HRNetModule, self).__init__()

        self.num_branches = num_branches
        self.block = block
        self.num_blocks = num_blocks
        self.num_inchannels = num_inchannels
        self.num_filters = num_filters

        self.check_branches()
        # generate branches and fuse layers
        self.branches = [self.make_one_branch(i) for i in range(self.num_branches)]
        self.fuse_layers = self.make_fuse_layers()

        self.relu = ReLU()

    def check_branches(self) -> None:
        """cheks if the branching is valid"""
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
        """creates a single branch for the module"""

        # apply downsampling if needed
        downsample = None
        if (
            stride != 1
            or self.num_inchannels[branch_index]
            != self.num_filters[branch_index] * self.block.expansion
        ):
            downsample = Sequential(
                [
                    Conv2D(
                        self.num_filters[branch_index] * self.block.expansion,
                        1,
                        stride,
                        use_bias=False,
                    ),
                    BatchNormalization(momentum=MOMENTUM, epsilon=1e-5),
                ]
            )

        # create the layers
        layers = []
        # first layer is the block with specified params
        layers.append(self.block(self.num_filters[branch_index], stride, downsample))
        # adjust the number of input channels
        self.num_inchannels[branch_index] = (
            self.num_filters[branch_index] * self.block.expansion
        )
        # all future layers are default
        for _ in range(1, self.num_blocks[branch_index]):
            layers.append(self.block(self.num_filters[branch_index]))  # noqa: PERF401

        return Sequential(layers)

    def make_fuse_layers(self) -> List[List[Layer]] | None:
        """creates the fuse layers for the module"""

        # if there is only one branch, there is no need for fuse layers
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels

        # create layers to fuse the branches
        fuse_layers = []
        for i in range(num_branches):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    # if the branch is higher than the current one, apply downsampling
                    fuse_layer.append(
                        Sequential(
                            [
                                Conv2D(num_inchannels[i], 1, use_bias=False),
                                BatchNormalization(momentum=MOMENTUM, epsilon=1e-5),
                            ]
                        )
                    )
                elif j == i:
                    # if the branch is the same as the current one, apply nothing
                    fuse_layer.append(None)
                else:
                    # if the branch is lower than the current one, apply upsampling
                    convs = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            # if it is the last layer, apply downsampling
                            num_outs = num_inchannels[i]
                            convs.append(
                                Sequential(
                                    [
                                        Conv2D(
                                            num_outs,
                                            kernel_size=3,
                                            strides=2,
                                            padding="same",
                                            use_bias=False,
                                        ),
                                        BatchNormalization(
                                            momentum=MOMENTUM, epsilon=1e-5
                                        ),
                                    ]
                                )
                            )
                        else:
                            # if it is not the last layer, apply normal convolutions
                            num_outs = num_inchannels[j]
                            convs.append(
                                Sequential(
                                    [
                                        Conv2D(
                                            num_outs,
                                            3,
                                            2,
                                            padding="same",
                                            use_bias=False,
                                        ),
                                        BatchNormalization(
                                            momentum=MOMENTUM, epsilon=1e-5
                                        ),
                                        ReLU(),
                                    ]
                                )
                            )
                    fuse_layer.append(Sequential(convs))
            fuse_layers.append(fuse_layer)

        return fuse_layers

    def call(self, x: List[tf.Tensor]) -> List[tf.Tensor]:
        # if there is only one branch, apply it and return
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        # otherwise, apply all branches and fuse
        x_branches = [self.branches[i](x[i]) for i in range(self.num_branches)]

        x_fuse = []
        for i in range(len(self.fuse_layers)):  # type: ignore
            if i == 0:
                # if it is the first layer, apply the first branch
                y = x_branches[0]
            else:
                # otherwise, apply the first branch and resize
                y = self.fuse_layers[i][0](x_branches[0])  # type: ignore
                target_shape = tf.shape(x_branches[i])[1:3]  # type: ignore
                y = tf.image.resize(y, target_shape)

            # apply the rest of the branches
            for j in range(1, self.num_branches):
                if i == j:
                    # if it is the same branch, apply it
                    y += x_branches[j]
                elif j > i:
                    # if it is a higher branch, apply the fuse layer and resize
                    target_shape = tf.shape(x_branches[i])[1:3]  # type: ignore
                    y += tf.image.resize(
                        self.fuse_layers[i][j](x_branches[j]), target_shape  # type: ignore
                    )
                else:
                    # if it is a lower branch, apply the fuse layers
                    y += self.fuse_layers[i][j](x_branches[j])  # type: ignore
            x_fuse.append(self.relu(y))
        return x_fuse


class HRNET(Model):
    def __init__(self, config: dict) -> None:
        super(HRNET, self).__init__()

        # stage 1 is the same for all HRNet models
        self.conv1 = Conv2D(64, 3, 2, padding="same", use_bias=False)
        self.bn1 = BatchNormalization(momentum=MOMENTUM, epsilon=1e-5)
        self.conv2 = Conv2D(64, 3, 2, padding="same", use_bias=False)
        self.bn2 = BatchNormalization(momentum=MOMENTUM, epsilon=1e-5)
        self.relu = ReLU()
        self.sf = Softmax(axis=1)  # ?????????

        self.layer1 = self.make_layer(Bottleneck, inplanes=64, planes=64, blocks=4)

        # stage 2
        self.stage2_cfg = config["STAGE2"]
        block = BasicBlock
        num_channels = self.stage2_cfg["NUM_CHANNELS"]
        num_channels = [c * block.expansion for c in num_channels]
        self.transition1 = self.make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self.make_stage(self.stage2_cfg, num_channels)

        # stage 3
        self.stage3_cfg = config["STAGE3"]
        block = BasicBlock
        num_channels = self.stage3_cfg["NUM_CHANNELS"]
        num_channels = [c * block.expansion for c in num_channels]
        self.transition2 = self.make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self.make_stage(self.stage3_cfg, num_channels)

        # stage 4
        self.stage4_cfg = config["STAGE4"]
        block = BasicBlock
        num_channels = self.stage4_cfg["NUM_CHANNELS"]
        num_channels = [c * block.expansion for c in num_channels]
        self.transition3 = self.make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self.make_stage(self.stage4_cfg, num_channels)

        # apply final convolutions
        final_channels = sum(pre_stage_channels)

        self.head = Sequential(
            [
                Conv2D(final_channels, 1, padding="same"),
                BatchNormalization(momentum=MOMENTUM, epsilon=1e-5),
                ReLU(),
                Conv2D(
                    config["NUM_JOINTS"], config["FINAL_CONV_KERNEL"], padding="same"
                ),
            ]
        )

    def make_transition_layer(
        self, num_channels_pre_layer: List[int], num_channels_cur_layer: List[int]
    ) -> List[Layer]:
        """creates the transition layers between stages"""

        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    # if the number of channels is different, apply a convolution
                    transition_layers.append(
                        Sequential(
                            [
                                Conv2D(
                                    num_channels_cur_layer[i],
                                    3,
                                    padding="same",
                                    use_bias=False,
                                ),
                                BatchNormalization(momentum=MOMENTUM, epsilon=1e-5),
                                ReLU(),
                            ]
                        )
                    )
                else:
                    # otherwise, apply nothing
                    transition_layers.append(None)
            else:
                # if the branch is new, apply convolutions to downsample
                convs = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = (
                        num_channels_cur_layer[i]
                        if j == i - num_branches_pre
                        else inchannels
                    )
                    convs.append(
                        Sequential(
                            [
                                Conv2D(
                                    outchannels, 3, 2, padding="same", use_bias=False
                                ),
                                BatchNormalization(momentum=MOMENTUM, epsilon=1e-5),
                                ReLU(),
                            ]
                        )
                    )
                transition_layers.append(Sequential(convs))
        return transition_layers

    def make_layer(
        self, block: Layer, inplanes: int, planes: int, blocks: int, stride: int = 1
    ) -> Layer:
        """creates a layer for HRNet"""

        # initialise downsampling
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = Sequential(
                [
                    Conv2D(planes * block.expansion, 1, stride, use_bias=False),
                    BatchNormalization(momentum=MOMENTUM, epsilon=1e-5),
                ]
            )

        # create the layers
        layers_list = []
        layers_list.append(block(planes, stride, downsample))
        for _ in range(1, blocks):
            layers_list.append(block(planes))  # noqa: PERF401
        return Sequential(layers_list)

    def make_stage(
        self, config: dict, num_inchannels: List[int]
    ) -> tuple[List[Layer], List[int]]:
        """creates a stage for HRNet"""

        num_modules = config["NUM_MODULES"]
        num_branches = config["NUM_BRANCHES"]
        num_blocks = config["NUM_BLOCKS"]
        num_channels = config["NUM_CHANNELS"]

        # create the modules
        modules = [
            HRNetModule(
                num_branches, BasicBlock, num_blocks, num_inchannels, num_channels
            )
            for _ in range(num_modules)
        ]
        return modules, modules[-1].num_inchannels

    def call(self, x: tf.Tensor) -> tf.Tensor:  # noqa: PLR0912
        # stage 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        # stage 2
        x_list = []
        for i in range(self.stage2_cfg["NUM_BRANCHES"]):
            # apply the transition layers
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        # apply the modules
        for module in self.stage2:
            x_list = module(x_list)

        # stage 3
        x_list2 = []
        for i in range(self.stage3_cfg["NUM_BRANCHES"]):
            if self.transition2[i] is not None:
                x_list2.append(self.transition2[i](x_list[-1]))
            else:
                x_list2.append(x_list[i])
        x_list = x_list2
        for module in self.stage3:
            x_list = module(x_list)

        # stage 4
        x_list2 = []
        for i in range(self.stage4_cfg["NUM_BRANCHES"]):
            if self.transition3[i] is not None:
                x_list2.append(self.transition3[i](x_list[-1]))
            else:
                x_list2.append(x_list[i])
        x_list = x_list2
        for module in self.stage4:
            x_list = module(x_list)

        # resize the branches and concatenate
        target_shape = tf.shape(x_list[0])[1:3]  # type: ignore
        resized_branches = [x_list[0]]
        for i in range(1, len(x_list)):
            resized_branches.append(  # noqa: PERF401
                tf.image.resize(x_list[i], target_shape)
            )
        x_cat = tf.concat(resized_branches, axis=-1)
        # apply final convolutions
        return self.head(x_cat)


class HRNet:
    """public interface for HRNet"""

    def __init__(self, config: dict, path_to_weights: str) -> None:
        """initialises the HRNet model"""
        self.cropped_image_size = (256, 256)

        self.model = HRNET(config)

        # build the model and load the weights
        self.model(tf.zeros((1, *self.cropped_image_size, 3)))
        self.model.load_weights(path_to_weights)

        # get opencv face detector
        self.face_cascade = cv.CascadeClassifier(
            cv.data.haarcascades + "haarcascade_frontalface_default.xml"  # type: ignore
        )

    def get_landmarks(self, image: np.ndarray) -> np.ndarray:
        """gets landmarks from the image"""

        # crop image to faces
        faces = self.face_cascade.detectMultiScale(
            cv.cvtColor(image, cv.COLOR_BGR2GRAY), 1.3, 5
        )
        if len(faces) == 0:
            return np.array([])

        face_crops = []
        for x, y, w, h in faces:
            face_crop = cv.resize(image[y : y + h, x : x + w], self.cropped_image_size)
            face_crops.append(face_crop)
        face_crops = tf.cast(tf.convert_to_tensor(face_crops), tf.float32)

        heatmaps = self.model(face_crops).numpy()
        heatmap_size = heatmaps.shape[1:3]

        multi_landmarks = []
        for idx, (x, y, w, h) in enumerate(faces):
            heatmap = heatmaps[idx]
            landmarks = [
                self.heatmap_to_landmark(heatmap[:, :, i]) for i in range(36, 48)
            ]

            landmarks = (
                np.array(landmarks)
                / np.array(heatmap_size)
                * np.array(self.cropped_image_size)
            )
            landmarks = np.array([x, y]) + landmarks * np.array([w, h]) / np.array(
                self.cropped_image_size
            )

            multi_landmarks.append(landmarks)

        return np.array(multi_landmarks)


    def heatmap_to_landmark(self, heatmap: np.ndarray) -> np.ndarray:
        """converts a heatmap to a landmark with refinement from CoM7"""
        max_y, max_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)

        fx_3 = heatmap[max_y, max_x - 3] if max_x - 3 >= 0 else 0
        fx_2 = heatmap[max_y, max_x - 2] if max_x - 2 >= 0 else 0
        fx_1 = heatmap[max_y, max_x - 1] if max_x - 1 >= 0 else 0
        fx1 = heatmap[max_y, max_x + 1] if max_x + 1 < heatmap.shape[1] else 0
        fx2 = heatmap[max_y, max_x + 2] if max_x + 2 < heatmap.shape[1] else 0
        fx3 = heatmap[max_y, max_x + 3] if max_x + 3 < heatmap.shape[1] else 0
        fy_3 = heatmap[max_y - 3, max_x] if max_y - 3 >= 0 else 0
        fy_2 = heatmap[max_y - 2, max_x] if max_y - 2 >= 0 else 0
        fy_1 = heatmap[max_y - 1, max_x] if max_y - 1 >= 0 else 0
        fy1 = heatmap[max_y + 1, max_x] if max_y + 1 < heatmap.shape[0] else 0
        fy2 = heatmap[max_y + 2, max_x] if max_y + 2 < heatmap.shape[0] else 0
        fy3 = heatmap[max_y + 3, max_x] if max_y + 3 < heatmap.shape[0] else 0
        fxy = heatmap[max_y, max_x]

        dx = (3 * fx3 + 2 * fx2 + fx1 - fx_1 - 2 * fx_2 - 3 * fx_3) / (
            fx3 + fx2 + fx1 + fxy + fx_1 + fx_2 + fx_3
        )
        dy = (3 * fy3 + 2 * fy2 + fy1 - fy_1 - 2 * fy_2 - 3 * fy_3) / (
            fy3 + fy2 + fy1 + fxy + fy_1 + fy_2 + fy_3
        )

        return np.array([max_x + dx, max_y + dy])
