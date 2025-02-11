# ROI Pooling layer adapted from original implementation by Jaime Sevilla
# original available at https://gist.github.com/Jsevillamol/0daac5a6001843942f91f2a3daea27a7

import tensorflow as tf
from tensorflow.keras.layers import Layer  # type: ignore


class ROIPoolingLayer(Layer):
    """Implements Region Of Interest Max Pooling
    for channel-first images and relative bounding box coordinates

    # Constructor parameters
        pooled_height, pooled_width (int) --
          specify height and width of layer outputs

    Shape of inputs
        [(batch_size, pooled_height, pooled_width, n_channels),
         (batch_size, num_rois, 4)]

    Shape of output
        (batch_size, num_rois, pooled_height, pooled_width, n_channels)

    """

    def __init__(self, pooled_height: int, pooled_width: int, **kwargs: dict) -> None:
        self.pooled_height = pooled_height
        self.pooled_width = pooled_width

        super(ROIPoolingLayer, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        """Returns the shape of the ROI Layer output"""
        feature_map_shape, rois_shape = input_shape
        assert feature_map_shape[0] == rois_shape[0]
        batch_size = feature_map_shape[0]
        n_rois = rois_shape[1]
        n_channels = feature_map_shape[3]
        return (batch_size, n_rois, self.pooled_height, self.pooled_width, n_channels)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Maps the input tensor of the ROI layer to its output

        # Parameters
            x[0] -- Convolutional feature map tensor,
                    shape (batch_size, pooled_height, pooled_width, n_channels)
            x[1] -- Tensor of region of interests from candidate bounding boxes,
                    shape (batch_size, num_rois, 4)
                    Each region of interest is defined by four relative
                    coordinates (x_min, y_min, x_max, y_max) between 0 and 1
        # Output
            pooled_areas -- Tensor with the pooled region of interest, shape
                (batch_size, num_rois, pooled_height, pooled_width, n_channels)
        """

        def curried_pool_rois(x: tf.Tensor) -> tf.Tensor:
            return ROIPoolingLayer._pool_rois(
                x[0], x[1], self.pooled_height, self.pooled_width  # type: ignore
            )

        return tf.map_fn(curried_pool_rois, x, fn_output_signature=tf.float32)  # type: ignore

    @staticmethod
    def _pool_rois(
        feature_map: tf.Tensor, rois: tf.Tensor, pooled_height: int, pooled_width: int
    ) -> tf.Tensor:
        """Applies ROI pooling for a single image and varios ROIs"""

        def curried_pool_roi(roi: tf.Tensor) -> tf.Tensor:
            return ROIPoolingLayer._pool_roi(
                feature_map, roi, pooled_height, pooled_width
            )

        return tf.map_fn(curried_pool_roi, rois, fn_output_signature=tf.float32)  # type: ignore

    @staticmethod
    def _pool_roi(
        feature_map: tf.Tensor, roi: tf.Tensor, pooled_height: int, pooled_width: int
    ) -> tf.Tensor:
        """Applies ROI pooling to a single image and a single region of interest"""

        # Compute the region of interest
        feature_map_height = int(tf.shape(feature_map)[0])  # type: ignore
        feature_map_width = int(tf.shape(feature_map)[1])  # type: ignore

        h_start = tf.cast(feature_map_height * roi[0], "int32")  # type: ignore
        w_start = tf.cast(feature_map_width * roi[1], "int32")  # type: ignore
        h_end = tf.cast(feature_map_height * roi[2], "int32")  # type: ignore
        w_end = tf.cast(feature_map_width * roi[3], "int32")  # type: ignore

        region = feature_map[h_start:h_end, w_start:w_end, :]  # type: ignore

        # Divide the region into non overlapping areas
        region_height = h_end - h_start  # type: ignore
        region_width = w_end - w_start  # type: ignore
        h_step = tf.cast(region_height / pooled_height, "int32")
        w_step = tf.cast(region_width / pooled_width, "int32")

        areas = [
            [
                (
                    i * h_step,  # type: ignore
                    j * w_step,  # type: ignore
                    (i + 1) * h_step if i + 1 < pooled_height else region_height,  # type: ignore
                    (j + 1) * w_step if j + 1 < pooled_width else region_width,  # type: ignore
                )
                for j in range(pooled_width)
            ]
            for i in range(pooled_height)
        ]

        # take the maximum of each area and stack the result
        def pool_area(x: tuple) -> tf.Tensor:
            return tf.math.reduce_max(region[x[0] : x[2], x[1] : x[3], :], axis=[0, 1])

        return tf.stack([[pool_area(x) for x in row] for row in areas])
