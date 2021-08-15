# TODO: do we need this control dependencies stuff or does that make things slower unnecessarily
# TODO: why does training seem to be slower when we use a weight_decay scheduler as compared to a float
import numpy as np
import tensorflow as tf



SKIP_WEIGHT_DECAY_FOR_BATCH_NORM = True



class SGDW(tf.keras.optimizers.SGD):
    def __init__(self, weight_decay, **kwargs):
        super(SGDW, self).__init__(**kwargs)

        self._set_hyper("weight_decay", weight_decay)

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(SGDW, self)._prepare_local(var_device, var_dtype, apply_state)

        apply_state[(var_device, var_dtype)]["weight_decay"] = tf.identity(self._decayed_weight_decay(var_dtype))

    def _decayed_weight_decay(self, var_dtype):
        weight_decay = self._get_hyper("weight_decay", var_dtype)
        if isinstance(weight_decay, tf.keras.optimizers.schedules.LearningRateSchedule):
            local_step = tf.cast(self.iterations, var_dtype)
            weight_decay = tf.cast(weight_decay(local_step), var_dtype)
        return weight_decay

    def _decay_weights_op(self, var, coefficients):
        # def true_fn():
        #     return var
        # def false_fn():
        #     return var.assign_sub(coefficients["weight_decay"] * var, self._use_locking)
        #
        # return tf.cond(SKIP_WEIGHT_DECAY_FOR_BATCH_NORM and tf.strings.regex_full_match(var.name, '.*bn.*'),
        #                true_fn, false_fn)

        return var.assign_sub(coefficients["weight_decay"] * var, self._use_locking)

    # def _decay_weights_sparse_op(self, var, indices, coefficients):
    #     update = (-coefficients["weight_decay"] * tf.gather(var, indices))
    #     return self._resource_scatter_add(var, indices, update)

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))

        with tf.control_dependencies([self._decay_weights_op(var, coefficients)]):
            return super(SGDW, self)._resource_apply_dense(grad, var, apply_state)
        # self._decay_weights_op(var, coefficients)
        # return super(SGDW, self)._resource_apply_dense(grad, var, apply_state)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))

        with tf.control_dependencies([self._decay_weights_sparse_op(var, indices, coefficients)]):
            return super(SGDW, self)._resource_apply_sparse(grad, var, indices, apply_state)
        # self._decay_weights_sparse_op(var, indices, coefficients)
        # return super(SGDW, self)._resource_apply_sparse(grad, var, indices, apply_state)

    def get_config(self):
        config = super(SGDW, self).get_config()
        config.update({
            'weight_decay': self._serialize_hyperparameter('weight_decay'),
        })
        return config



def piecewise_scheduler(boundaries, decay_rates, base_rate=0.1, boundaries_as='epochs', num_images=None,
                        batch_size=None):
    # use rate while <= boundary
    # iterations: the number of training steps this optimizer has run
    assert boundaries_as in ['epochs', 'iterations'], 'boundaries_as must be in [epochs, iterations]'

    if boundaries_as == 'epochs':
        iterations_per_epoch = np.floor(num_images/batch_size)
        boundary_iterations = [(iterations_per_epoch * epoch) - 1 for epoch in boundaries]
    else:
        boundary_iterations = boundaries

    boundary_values = [base_rate * decay for decay in decay_rates]

    return tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundary_iterations, boundary_values)


