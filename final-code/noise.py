from typing import Any

import numpy as np
import tensorflow as tf

# MARK: CWL2
# From paper: "Towards Evaluating the Robustness of Neural Networks"
# https://ieeexplore.ieee.org/abstract/document/7958570
# thanks to Nicholas Carlini and David Wagner
# implementation adapted from https://github.com/carlini/nn_robust_attacks/ and https://github.com/kkew3/
# thanks to Kaiwen Wu


class CWL2:
    def __init__(self, model: Any) -> None:  # noqa: ANN401
        self.model = model

        self.confidence = 0
        self.c_range = (1e-3, 1e10)
        self.search_steps = 9
        self.max_steps = 1000
        self.abort_early = True
        self.abort_early_tolerance = 1e-4
        self.box = (-1.0, 1.0)
        self.learning_rate = 1e-2

        self.optimiser = tf.keras.optimizers.Adam(self.learning_rate)  # type: ignore

    def attack(  # noqa: PLR0912
        self, images: np.ndarray, targets: np.ndarray
    ) -> np.ndarray:
        num_classes = 2
        batch_size = images.shape[0]

        lower_bounds_np = np.zeros(batch_size)
        upper_bounds_np = np.ones(batch_size) * self.c_range[1]
        scale_consts_np = np.ones(batch_size) * self.c_range[0]

        o_best_l2 = np.ones(batch_size) * np.inf
        o_best_l2_ppred = -np.ones(batch_size)
        o_best_advx = np.copy(images)

        inputs_tanh = self._to_tanh(images)
        inputs_tanh_var = tf.Variable(inputs_tanh)

        targets_oh_var = tf.one_hot(targets, num_classes)

        pert_tanh_var = tf.Variable(tf.random.normal(images.shape))

        for sstep in range(self.search_steps):
            if sstep == self.search_steps - 1:
                scale_consts_np = upper_bounds_np

            scale_consts_var = tf.Variable(scale_consts_np)

            best_l2 = np.ones(batch_size) * np.inf
            best_l2_ppred = -np.ones(batch_size)
            prev_batch_loss = np.inf

            for optim_step in range(self.max_steps):
                batch_loss, pert_norms_np, pert_outputs_np, advxs_np = self._optimise(
                    inputs_tanh_var, pert_tanh_var, targets_oh_var, scale_consts_var
                )

                if self.abort_early and not optim_step % (self.max_steps // 10):
                    if batch_loss > prev_batch_loss * (1 - self.abort_early_tolerance):
                        break
                    prev_batch_loss = batch_loss

                pert_predictions_np = np.argmax(pert_outputs_np, axis=1)
                compensate = (
                    np.copy(pert_predictions_np)[np.arange(targets.shape[0]), targets]
                    - self.confidence
                )
                comp_pert_predictions_np = np.argmax(compensate, axis=1)

                for i in range(batch_size):
                    l2 = pert_norms_np[i]
                    cppred = comp_pert_predictions_np[i]
                    ppred = pert_predictions_np[i]
                    tlabel = targets[i]
                    ax = advxs_np[i]
                    if cppred == tlabel:
                        if l2 < best_l2[i]:
                            best_l2[i] = l2
                            best_l2_ppred[i] = ppred
                        if l2 < o_best_l2[i]:
                            o_best_l2[i] = l2
                            o_best_l2_ppred[i] = ppred
                            o_best_advx[i] = ax

            for i in range(batch_size):
                tlabel = targets[i]
                if best_l2_ppred[i] != -1:
                    upper_bounds_np[i] = min(upper_bounds_np[i], scale_consts_np[i])
                    if upper_bounds_np[i] < self.c_range[1] * 0.1:
                        scale_consts_np[i] = (
                            lower_bounds_np[i] + upper_bounds_np[i]
                        ) / 2
                else:
                    lower_bounds_np[i] = max(lower_bounds_np[i], scale_consts_np[i])
                    if upper_bounds_np[i] < self.c_range[1] * 0.1:
                        scale_consts_np[i] = (
                            lower_bounds_np[i] + upper_bounds_np[i]
                        ) / 2
                    else:
                        scale_consts_np[i] *= 10

        return o_best_advx

    def _to_tanh(self, x: np.ndarray) -> np.ndarray:
        mul = (self.box[1] - self.box[0]) / 2.0
        add = (self.box[1] + self.box[0]) / 2.0
        return np.arctanh((x - add) / mul)

    def _from_tanh(self, x: np.ndarray) -> np.ndarray:
        mul = (self.box[1] - self.box[0]) / 2.0
        add = (self.box[1] + self.box[0]) / 2.0
        return np.tanh(x) * mul + add

    def _optimise(
        self,
        inputs_tanh_var: tf.Variable,
        pert_tanh_var: tf.Variable,
        targets_oh_var: tf.Variable,
        c_var: tf.Variable,
    ) -> tuple:
        advxs_var = self._from_tanh(inputs_tanh_var + pert_tanh_var)  # type: ignore
        pert_outputs_var = self.model(advxs_var)

        inputs_var = self._from_tanh(inputs_tanh_var)  # type: ignore

        perts_norm_var = tf.pow(advxs_var - inputs_var, 2)
        perts_norm_var = tf.reduce_sum(
            tf.reshape(perts_norm_var, perts_norm_var.shape[0], -1), axis=1
        )

        target_activ_var = tf.reduce_sum(targets_oh_var * pert_outputs_var, axis=1)
        inf = tf.constant(1e10, dtype=tf.float32)

        maxother_activ_car = tf.reduce_max(
            (1 - targets_oh_var) * pert_outputs_var - targets_oh_var * inf, axis=1  # type: ignore
        )[0]

        f_var = tf.clip_by_value(
            maxother_activ_car - target_activ_var + self.confidence, 0, inf
        )

        batch_loss_var = tf.reduce_sum(perts_norm_var + c_var * f_var)  # type: ignore

        self.optimiser.minimize(batch_loss_var, [pert_tanh_var])

        batch_loss = batch_loss_var[0].numpy()
        pert_norms_np = perts_norm_var.numpy()
        pert_outputs_np = pert_outputs_var.numpy()

        return batch_loss, pert_norms_np, pert_outputs_np, advxs_var
