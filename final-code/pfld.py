# A tensorflow implementation of "PFLD: A Practical Facial Landmark Detector"
# Paper: https://arxiv.org/abs/1902.10859

# implmentation is adapted from https://github.com/polarisZhao/PFLD-pytorch
# thanks to Zhichao Zhao, Harry Guo, and Andres

# slightly adapted to be in tensorflow and interchangable backbones

import tensorflow as tf
from tensorflow.keras import Model  # type: ignore
from tensorflow.keras.applications import MobileNetV2, ResNet50  # type: ignore
from tensorflow.keras.layers import Conv2D, Dense  # type: ignore


class PFLD(Model):
    pass
