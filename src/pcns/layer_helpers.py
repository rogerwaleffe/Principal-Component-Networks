import tensorflow as tf



# PRODUCTION VERSIONS
class DensePCALayer(tf.keras.layers.Layer):
    def __init__(self, units, mu, V, name=None, **kwargs):
        super(DensePCALayer, self).__init__(name=name)

        if tf.is_tensor(mu) and tf.is_tensor(V):
            self.mu = self.add_weight('mu', shape=[1, mu.shape[0]],
                                      initializer=constant_initializer_from_tensor(tf.expand_dims(mu, 0)),
                                      regularizer=None, constraint=None, dtype=self.dtype, trainable=False)
            self.V = self.add_weight('V', shape=V.shape, initializer=constant_initializer_from_tensor(V),
                                     regularizer=None, constraint=None, dtype=self.dtype, trainable=False)
        else:
            self.mu = self.add_weight('mu', shape=mu, initializer=tf.keras.initializers.get('zeros'),
                                      regularizer=None, constraint=None, dtype=self.dtype, trainable=False)
            self.V = self.add_weight('V', shape=V, initializer=tf.keras.initializers.get('zeros'),
                                     regularizer=None, constraint=None, dtype=self.dtype, trainable=False)

        self.dense_layer = tf.keras.layers.Dense(units, **kwargs)

        self.units = units

    def build(self, input_shape):
        super(DensePCALayer, self).build(input_shape)

    def call(self, inputs):
        inputs = tf.linalg.matmul(tf.math.subtract(inputs, self.mu), self.V)

        return self.dense_layer(inputs)

    def get_config(self):
        config = super(DensePCALayer, self).get_config()
        config.update({'units': self.units})
        config.update({'mu': self.mu.shape})
        config.update({'V': self.V.shape})
        config.update({
            'activation': tf.keras.activations.serialize(self.dense_layer.activation),
            'use_bias': self.dense_layer.use_bias,
            'kernel_initializer': tf.keras.initializers.serialize(tf.keras.initializers.get('zeros')),
            'bias_initializer': tf.keras.initializers.serialize(tf.keras.initializers.get('zeros')),
            'kernel_regularizer': tf.keras.regularizers.serialize(self.dense_layer.kernel_regularizer),
            'bias_regularizer': tf.keras.regularizers.serialize(self.dense_layer.bias_regularizer),
            'activity_regularizer': tf.keras.regularizers.serialize(self.dense_layer.activity_regularizer),
            'kernel_constraint': tf.keras.constraints.serialize(self.dense_layer.kernel_constraint),
            'bias_constraint': tf.keras.constraints.serialize(self.dense_layer.bias_constraint)
        })

        return config



# TODO: for these conv layers, support tuple kernel sizes and strides
# TODO: for Conv2DPCALayer support ignoring padding transformation
class Conv2DPCALayer(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, mu, V, data_format='channels_last', name=None, **kwargs):
        super(Conv2DPCALayer, self).__init__(name=name)

        if tf.is_tensor(mu) and tf.is_tensor(V):
            self.mu = self.add_weight('mu', shape=[1, mu.shape[0]],
                                      initializer=constant_initializer_from_tensor(tf.expand_dims(mu, 0)),
                                      regularizer=None, constraint=None, dtype=self.dtype, trainable=False)
            self.V = self.add_weight('V', shape=V.shape, initializer=constant_initializer_from_tensor(V),
                                     regularizer=None, constraint=None, dtype=self.dtype, trainable=False)
        else:
            self.mu = self.add_weight('mu', shape=mu, initializer=tf.keras.initializers.get('zeros'),
                                      regularizer=None, constraint=None, dtype=self.dtype, trainable=False)
            self.V = self.add_weight('V', shape=V, initializer=tf.keras.initializers.get('zeros'),
                                     regularizer=None, constraint=None, dtype=self.dtype, trainable=False)

        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg

        self.zero_pad = ((pad_beg, pad_end), (pad_beg, pad_end))
        self.zero_pad_layer = tf.keras.layers.ZeroPadding2D(padding=self.zero_pad, data_format=data_format)
        self.conv_layer = tf.keras.layers.Conv2D(filters, kernel_size, data_format=data_format, **kwargs)

        # NOTE: TO IGNORE PADDING, and comment out inputs=self.zero_pad_layer in call
        # kwargs.pop('padding')
        # self.zero_pad = ((0, 0), (0, 0))
        # self.conv_layer = tf.keras.layers.Conv2D(filters, kernel_size, data_format=data_format, padding='SAME',
        #                                          **kwargs)

        self.height, self.width, self.pre_pca_depth, self.post_pca_depth = None, None, None, None

        self.filters = filters
        self.kernel_size = kernel_size
        self.data_format = data_format

    def build(self, input_shape):
        super(Conv2DPCALayer, self).build(input_shape)

        self.height = input_shape[-3] + self.zero_pad[0][0] + self.zero_pad[0][1]
        self.width = input_shape[-2] + self.zero_pad[1][0] + self.zero_pad[1][1]
        self.pre_pca_depth = input_shape[-1]
        self.post_pca_depth = self.V.shape[1]

    def call(self, inputs):
        inputs = self.zero_pad_layer(inputs)

        inputs = tf.reshape(inputs, (-1, self.pre_pca_depth))
        inputs = tf.linalg.matmul(tf.math.subtract(inputs, self.mu), self.V)
        inputs = tf.reshape(inputs, (-1, self.height, self.width, self.post_pca_depth))

        return self.conv_layer(inputs)

    def get_config(self):
        config = super(Conv2DPCALayer, self).get_config()
        config.update({'filters': self.filters})
        config.update({'kernel_size': self.kernel_size})
        config.update({'mu': self.mu.shape})
        config.update({'V': self.V.shape})
        config.update({'data_format': self.data_format})
        config.update({
            'strides': self.conv_layer.strides,
            'padding': self.conv_layer.padding,
            'dilation_rate': self.conv_layer.dilation_rate,
            'activation': tf.keras.activations.serialize(self.conv_layer.activation),
            'use_bias': self.conv_layer.use_bias,
            'kernel_initializer': tf.keras.initializers.serialize(tf.keras.initializers.get('zeros')),
            'bias_initializer': tf.keras.initializers.serialize(tf.keras.initializers.get('zeros')),
            'kernel_regularizer': tf.keras.regularizers.serialize(self.conv_layer.kernel_regularizer),
            'bias_regularizer': tf.keras.regularizers.serialize(self.conv_layer.bias_regularizer),
            'activity_regularizer': tf.keras.regularizers.serialize(self.conv_layer.activity_regularizer),
            'kernel_constraint': tf.keras.constraints.serialize(self.conv_layer.kernel_constraint),
            'bias_constraint': tf.keras.constraints.serialize(self.conv_layer.bias_constraint)
        })

        return config



class Conv2DExplicitPadding(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, data_format='channels_last', name=None, **kwargs):
        super(Conv2DExplicitPadding, self).__init__(name=name)

        # NOTE: hack rn b/c we do not support tuple strides (we don't use them rn)
        # if isinstance(strides, (list, tuple)):
        #     strides = strides[0]

        if strides > 1:
            pad_total = kernel_size - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg

            zero_pad = ((pad_beg, pad_end), (pad_beg, pad_end))
            self.zero_pad_layer = tf.keras.layers.ZeroPadding2D(padding=zero_pad, data_format=data_format)

            self.explicit_pad = True
        else:
            self.explicit_pad = False

        self.conv_layer = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides,
                                                 padding=('SAME' if strides == 1 else 'VALID'), data_format=data_format,
                                                 **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.data_format = data_format

    def build(self, input_shape):
        super(Conv2DExplicitPadding, self).build(input_shape)

    def call(self, inputs):
        if self.explicit_pad:
            inputs = self.zero_pad_layer(inputs)
        return self.conv_layer(inputs)

    def get_config(self):
        config = super(Conv2DExplicitPadding, self).get_config()
        config.update({'filters': self.filters})
        config.update({'kernel_size': self.kernel_size})
        config.update({'strides': self.strides})
        config.update({'data_format': self.data_format})
        config.update({
            'dilation_rate': self.conv_layer.dilation_rate,
            'activation': tf.keras.activations.serialize(self.conv_layer.activation),
            'use_bias': self.conv_layer.use_bias,
            'kernel_initializer': tf.keras.initializers.serialize(tf.keras.initializers.get('zeros')),
            'bias_initializer': tf.keras.initializers.serialize(tf.keras.initializers.get('zeros')),
            'kernel_regularizer': tf.keras.regularizers.serialize(self.conv_layer.kernel_regularizer),
            'bias_regularizer': tf.keras.regularizers.serialize(self.conv_layer.bias_regularizer),
            'activity_regularizer': tf.keras.regularizers.serialize(self.conv_layer.activity_regularizer),
            'kernel_constraint': tf.keras.constraints.serialize(self.conv_layer.kernel_constraint),
            'bias_constraint': tf.keras.constraints.serialize(self.conv_layer.bias_constraint)
        })

        return config



class IdentityLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(IdentityLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(IdentityLayer, self).build(input_shape)

    def call(self, inputs):
        return inputs



class constant_initializer_from_tensor(tf.keras.initializers.Initializer):
    def __init__(self, value=None):
        # TODO: check value is a Tensor if not none
        self.value = value

    def __call__(self, shape, dtype=None):
        """
        Returns a tensor object initialized as specified by the initializer.

        Args:
          shape: Shape of the tensor.
          dtype: Optional dtype of the tensor. If not provided the dtype of the tensor created will be the type
                 of the initial value.
        """
        if self.value is None:
            return tf.zeros(shape, dtype=dtype)

        # TODO: check shape matches shape of value, possibly convert dtype (if dtype not None and diff than value dtype)
        return self.value










# # TEST VERSIONS:
# class DensePCALayer(tf.keras.layers.Layer):
#     def __init__(self, units, mu, V, name=None, **kwargs):
#         super(DensePCALayer, self).__init__(name=name)
#
#         self.mu = self.add_weight('mu', shape=[1, mu.shape[0]], initializer=constant_initializer_from_tensor(tf.expand_dims(mu, 0)),
#                                   regularizer=None, constraint=None, dtype=self.dtype, trainable=False)
#         self.V = self.add_weight('V', shape=V.shape, initializer=constant_initializer_from_tensor(V),
#                                  regularizer=None, constraint=None, dtype=self.dtype, trainable=False)
#
#         self.dense_layer = tf.keras.layers.Dense(units, **kwargs)
#
#     def build(self, input_shape):
#         super(DensePCALayer, self).build(input_shape)
#
#     def call(self, inputs):
#         inputs = tf.linalg.matmul(tf.math.subtract(inputs, self.mu), self.V)
#         # inputs = inputs[:, :self.V.shape[1]]
#
#         return self.dense_layer(inputs)
#
#
#
# class Conv2DPCALayer(tf.keras.layers.Layer):
#     def __init__(self, filters, kernel_size, mu, V, zero_pad=((0,0),(0,0)), name=None, **kwargs):
#         super(Conv2DPCALayer, self).__init__(name=name)
#
#         self.mu = self.add_weight('mu', shape=[1, mu.shape[0]], initializer=constant_initializer_from_tensor(tf.expand_dims(mu, 0)),
#                                   regularizer=None, constraint=None, dtype=self.dtype, trainable=False)
#         self.V = self.add_weight('V', shape=V.shape, initializer=constant_initializer_from_tensor(V),
#                                  regularizer=None, constraint=None, dtype=self.dtype, trainable=False)
#
#         self.zero_pad = zero_pad
#         self.zero_pad_layer = tf.keras.layers.ZeroPadding2D(padding=zero_pad)
#         self.conv_layer = tf.keras.layers.Conv2D(filters, kernel_size, **kwargs) # TODO
#
#         self.flatten = None
#         self.un_flatten = None
#
#     def build(self, input_shape):
#         super(Conv2DPCALayer, self).build(input_shape)
#
#         # input_shape (None, height, width, depth) is before PCA, it needs to be modified as necessary
#         # height and width are padded with 0s before PCA transformation, depth is smaller after PCA transformation
#
#         self.height = input_shape[-3] + self.zero_pad[0][0] + self.zero_pad[0][1] # TODO
#         self.width = input_shape[-2] + self.zero_pad[1][0] + self.zero_pad[1][1]
#         self.pre_pca_depth = input_shape[-1]
#         self.post_pca_depth = self.V.shape[1]
#
#         # self.flatten = tf.keras.layers.Reshape((self.height*self.width, self.pre_pca_depth))
#         # self.un_flatten = tf.keras.layers.Reshape((self.height, self.width, self.post_pca_depth))
#
#         # print(input_shape)
#         # tf.print(input_shape)
#         # tf.print(tf.TensorShape([input_shape[0], height, width, post_pca_depth]))
#         # self.conv_layer.build(tf.TensorShape([input_shape[0], height, width, post_pca_depth]))
#
#     def call(self, inputs):
#         # print("\nCALL", self)
#         # tf.print(tf.shape(inputs))
#         # print(inputs.shape)
#         # if inputs.shape[0] is None:
#         #     # print("INPUT IS NONE")
#         #     inputs = self.zero_pad_layer(inputs)
#         #     inputs = tf.reshape(inputs, (-1, self.pre_pca_depth))
#         #     inputs = tf.linalg.matmul(tf.math.subtract(inputs, self.mu), self.V)
#         #     inputs = tf.reshape(inputs, (-1, self.height, self.width, self.post_pca_depth))
#         #     # X_p = inputs[:, :, :, :self.V.shape[1]]
#         #     return self.conv_layer(inputs)
#
#
#         inputs = self.zero_pad_layer(inputs)
#         # inputs = tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]])
#
#
#         # V1
#         # X_p_flat = self.flatten(tf.math.subtract(inputs, self.mu))
#         # batch_V = tf.tile(tf.expand_dims(self.V, 0), [tf.shape(inputs)[0], 1, 1])
#         # X_p_flat = tf.linalg.matmul(X_p_flat, batch_V)
#         # X_p = self.un_flatten(X_p_flat)
#
#         # inputs = self.flatten(tf.math.subtract(inputs, self.mu))
#         # batch_V = tf.tile(tf.expand_dims(self.V, 0), [tf.shape(inputs)[0], 1, 1])
#         # inputs = tf.linalg.matmul(inputs, batch_V)
#         # inputs = self.un_flatten(inputs)
#
#
#         # V2
#         # batch_size = tf.shape(inputs)[0]
#         inputs = tf.reshape(inputs, (-1, self.pre_pca_depth))
#         inputs = tf.linalg.matmul(tf.math.subtract(inputs, self.mu), self.V)
#         inputs = tf.reshape(inputs, (-1, self.height, self.width, self.post_pca_depth))
#         # tf.print(tf.shape(X_f))
#         # tf.print(tf.shape(X_p))
#
#
#         # X_p = inputs[:, :, :, :self.V.shape[1]]
#         # X_p = X_p[:, :, :, :self.V.shape[1]]
#         # inputs = inputs[:, :, :, :self.V.shape[1]]
#
#         # return self.conv_layer(tf.pad(X_p, [[0, 0], [1, 1], [1, 1], [0, 0]]))
#         return self.conv_layer(inputs)
#
#         # return self.conv_layer(tf.reshape(tf.linalg.matmul(tf.math.subtract(
#         #        tf.reshape(self.zero_pad_layer(inputs), (batch_size*self.height*self.width, self.pre_pca_depth)),
#         #        self.mu), self.V), (batch_size, self.height, self.width, self.post_pca_depth)))
#
#     # def compute_output_shape(self, input_shape):
#     #     print("COMPUTE_OUTPUT_SHAPE", self)
#     #     tf.print(input_shape)
#     #     return tf.TensorShape([input_shape[0], input_shape[1], input_shape[2], self.V.shape[1]])










# OLD VERSIONS:
# from tensorflow import dtypes
# from tensorflow.keras import activations
# from tensorflow.keras import backend as K
# from tensorflow.keras import constraints
# from tensorflow.keras import initializers
# from tensorflow.keras import regularizers
# from tensorflow.keras.layers import InputSpec

# class DensePCALayer(tf.keras.layers.Dense):
#     def __init__(self, units, mu, V, initial_weights, initial_bias, **kwargs):
#         super(DensePCALayer, self).__init__(units, **kwargs)
#
#         self.mu = self.add_weight('mu', shape=[1, mu.shape[0]], initializer=constant_initializer_from_tensor(tf.expand_dims(mu, 0)),
#                                   regularizer=None, constraint=None, dtype=self.dtype, trainable=False)
#         self.V = self.add_weight('V', shape=V.shape, initializer=constant_initializer_from_tensor(V),
#                                  regularizer=None, constraint=None, dtype=self.dtype, trainable=False)
#         # V here is the truncated matrix (m' x m)
#
#         self.initial_weights = initial_weights
#         self.initial_bias = initial_bias
#
#         self.input_spec = InputSpec(min_ndim=2)
#         self.kernel = None
#         self.bias = None
#         self.built = False
#
#     def build(self, input_shape):
#         dtype = dtypes.as_dtype(self.dtype or K.floatx())
#         if not (dtype.is_floating or dtype.is_complex):
#             raise TypeError('Unable to build `Dense` layer with non-floating point '
#                             'dtype %s' % (dtype,))
#         input_shape = tf.TensorShape(input_shape)  # generally [None, number of inputs]
#         if input_shape[-1] is None:
#             raise ValueError('The last dimension of the inputs to `Dense` '
#                              'should be defined. Found `None`.')
#         last_dim = input_shape[-1]
#         self.input_spec = InputSpec(min_ndim=2,
#                                     axes={-1: last_dim})
#
#         self.kernel = self.add_weight(
#             'kernel',
#             shape=[self.initial_weights.shape[0], self.units],
#             initializer=constant_initializer_from_tensor(self.initial_weights),
#             regularizer=self.kernel_regularizer,
#             constraint=self.kernel_constraint,
#             dtype=self.dtype,
#             trainable=True)
#         if self.use_bias:
#             self.bias = self.add_weight(
#                 'bias',
#                 shape=[self.units, ],
#                 initializer=constant_initializer_from_tensor(tf.squeeze(self.initial_bias)),
#                 regularizer=self.bias_regularizer,
#                 constraint=self.bias_constraint,
#                 dtype=self.dtype,
#                 trainable=True)
#         else:
#             self.bias = None
#
#         self.built = True
#
#     def call(self, inputs, training=None):
#         inputs = dtypes.cast(inputs, self._compute_dtype)
#
#         inputs_prime = tf.linalg.matmul(tf.math.subtract(inputs, self.mu), self.V)
#         outputs = tf.linalg.matmul(inputs_prime, self.kernel)
#         # outputs = tf.math.add(outputs, tf.linalg.matmul(self.mu, self.kernel))
#
#         if self.use_bias:
#             outputs = tf.nn.bias_add(outputs, self.bias)
#
#         if self.activation is not None:
#             outputs = self.activation(outputs)  # pylint: disable=not-callable
#
#         return outputs



# class Conv2DPCATransformLayer(tf.keras.layers.Layer):
#     def __init__(self, mu, V, **kwargs):
#         super(Conv2DPCATransformLayer, self).__init__(**kwargs)
#
#         self.mu = self.add_weight('mu', shape=[1, mu.shape[0]], initializer=constant_initializer_from_tensor(tf.expand_dims(mu, 0)),
#                                   regularizer=None, constraint=None, dtype=self.dtype, trainable=False)
#         self.V = self.add_weight('V', shape=V.shape, initializer=constant_initializer_from_tensor(V),
#                                  regularizer=None, constraint=None, dtype=self.dtype, trainable=False)
#
#         self.flatten = None
#         self.un_flatten = None
#
#     def build(self, input_shape):
#         super(Conv2DPCATransformLayer, self).build(input_shape)
#
#         self.flatten = tf.keras.layers.Reshape((input_shape[-3]*input_shape[-2], input_shape[-1]))
#         self.un_flatten = tf.keras.layers.Reshape((input_shape[-3], input_shape[-2], self.V.shape[1]))
#
#     def call(self, inputs):
#         X_p_flat = self.flatten(tf.math.subtract(inputs, self.mu))
#         batch_V = tf.tile(tf.expand_dims(self.V, 0), [tf.shape(inputs)[0], 1, 1])
#         X_p_flat = tf.linalg.matmul(X_p_flat, batch_V)
#         X_p = self.un_flatten(X_p_flat)
#
#         return X_p