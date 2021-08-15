import tensorflow as tf



class DenseTransformLayer(tf.keras.layers.Layer):
    def __init__(self, units, mu, Vt, Vb, Wt, Wb, bp, include_offset=False, train_top_basis='NO', add_bias_if_nec=True,
                 name=None, **kwargs):
        super(DenseTransformLayer, self).__init__(name=name)

        assert train_top_basis in ['NO', 'YES', 'AS_KERNEL'], 'train_top_basis must be in [NO, YES, AS_KERNEL]'
        should_train = False if train_top_basis == 'NO' else True
        regularizer = kwargs.get('kernel_regularizer', None) if train_top_basis == 'AS_KERNEL' else None
        constraint = kwargs.get('kernel_constraint', None) if train_top_basis == 'AS_KERNEL' else None
        regularizer = tf.keras.regularizers.get(regularizer)
        constraint = tf.keras.constraints.get(constraint)

        if mu is not None:
            shape = [1, mu.shape[0]] if tf.is_tensor(mu) else mu
            initializer = cift(tf.expand_dims(mu, 0)) if tf.is_tensor(mu) else tf.keras.initializers.get('zeros')
            self.mu = self.add_weight('mu', shape=shape, initializer=initializer, regularizer=regularizer,
                                      constraint=constraint, dtype=self.dtype, trainable=should_train)
        else:
            self.mu = None

        shape = Vt.shape if tf.is_tensor(Vt) else Vt
        initializer = cift(Vt) if tf.is_tensor(Vt) else tf.keras.initializers.get('zeros')
        self.Vt = self.add_weight('Vt', shape=shape, initializer=initializer, regularizer=regularizer,
                                  constraint=constraint, dtype=self.dtype, trainable=should_train)

        if Vb is not None:
            shape = Vb.shape if tf.is_tensor(Vb) else Vb
            initializer = cift(Vb) if tf.is_tensor(Vb) else tf.keras.initializers.get('zeros')
            self.Vb = self.add_weight('Vb', shape=shape, initializer=initializer, regularizer=None, constraint=None,
                                      dtype=self.dtype, trainable=False)
            shape = Wb.shape if tf.is_tensor(Wb) else Wb
            initializer = cift(Wb) if tf.is_tensor(Wb) else tf.keras.initializers.get('zeros')
            self.Wb = self.add_weight('Wb', shape=shape, initializer=initializer, regularizer=None, constraint=None,
                                      dtype=self.dtype, trainable=False)
        else:
            self.Vb, self.Wb = None, None

        self.activation = tf.keras.activations.get(kwargs.pop('activation', None))

        use_bias = kwargs.get('use_bias', True)
        self.previous_layer_used_bias = kwargs.pop('previous_layer_used_bias', use_bias)
        if self.previous_layer_used_bias is False and mu is not None:
            if add_bias_if_nec is True:
                use_bias = True
                kwargs.update({'use_bias': use_bias})
            else:
                self.mu = None

        kwargs.pop('kernel_initializer', '')
        kwargs.pop('bias_initializer', '')
        k_init = cift(Wt) if Wt is not None else tf.keras.initializers.get('zeros')
        b_init = cift(tf.squeeze(bp)) if bp is not None else tf.keras.initializers.get('zeros')
        self.dense_layer = tf.keras.layers.Dense(units, activation=None, kernel_initializer=k_init,
                                                 bias_initializer=b_init, **kwargs)

        self.units = units
        self.include_offset = include_offset
        self.train_top_basis = train_top_basis
        self.add_bias_if_nec = add_bias_if_nec

    def build(self, input_shape):
        super(DenseTransformLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        h_top = tf.matmul(inputs, self.Vt) if self.mu is None else tf.matmul(tf.subtract(inputs, self.mu), self.Vt)
        output = self.dense_layer(h_top)

        if self.include_offset:
            h_bot = tf.matmul(inputs, self.Vb) if self.mu is None else tf.matmul(tf.subtract(inputs, self.mu), self.Vb)
            output += tf.matmul(h_bot, self.Wb)

        return self.activation(output)

    def get_config(self):
        config = super(DenseTransformLayer, self).get_config()
        config.update({'units': self.units})
        config.update({'mu': self.mu.shape if self.mu is not None else None})
        config.update({'Vt': self.Vt.shape})
        config.update({'Vb': self.Vb.shape if self.Vb is not None else None})
        config.update({'Wt': None})
        config.update({'Wb': self.Wb.shape if self.Wb is not None else None})
        config.update({'bp': None})
        config.update({'include_offset': self.include_offset})
        config.update({'train_top_basis': self.train_top_basis})
        config.update({'add_bias_if_nec': self.add_bias_if_nec})
        config.update({'previous_layer_used_bias': self.previous_layer_used_bias})
        config.update({
            'activation': tf.keras.activations.serialize(self.activation),
            'use_bias': self.dense_layer.use_bias,
            'kernel_initializer': None,
            'bias_initializer': None,
            'kernel_regularizer': tf.keras.regularizers.serialize(self.dense_layer.kernel_regularizer),
            'bias_regularizer': tf.keras.regularizers.serialize(self.dense_layer.bias_regularizer),
            'activity_regularizer': tf.keras.regularizers.serialize(self.dense_layer.activity_regularizer),
            'kernel_constraint': tf.keras.constraints.serialize(self.dense_layer.kernel_constraint),
            'bias_constraint': tf.keras.constraints.serialize(self.dense_layer.bias_constraint)
        })



class Conv2DTransformLayer(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, mu, Vt, Vb, Wt, Wb, bp, include_offset=False, train_top_basis='NO',
                 add_bias_if_nec=True, previous_layer_type=None, name=None, **kwargs):
        super(Conv2DTransformLayer, self).__init__(name=name)

        assert train_top_basis in ['NO', 'YES', 'AS_KERNEL'], 'train_top_basis must be in [NO, YES, AS_KERNEL]'
        should_train = False if train_top_basis == 'NO' else True
        regularizer = kwargs.get('kernel_regularizer', None) if train_top_basis == 'AS_KERNEL' else None
        constraint = kwargs.get('kernel_constraint', None) if train_top_basis == 'AS_KERNEL' else None
        regularizer = tf.keras.regularizers.get(regularizer)
        constraint = tf.keras.constraints.get(constraint)

        if mu is not None:
            shape = [1, mu.shape[0]] if tf.is_tensor(mu) else mu
            initializer = cift(tf.expand_dims(mu, 0)) if tf.is_tensor(mu) else tf.keras.initializers.get('zeros')
            self.mu = self.add_weight('mu', shape=shape, initializer=initializer, regularizer=regularizer,
                                      constraint=constraint, dtype=self.dtype, trainable=should_train)
        else:
            self.mu = None

        shape = Vt.shape if tf.is_tensor(Vt) else Vt
        initializer = cift(Vt) if tf.is_tensor(Vt) else tf.keras.initializers.get('zeros')
        self.Vt = self.add_weight('Vt', shape=shape, initializer=initializer, regularizer=regularizer,
                                  constraint=constraint, dtype=self.dtype, trainable=should_train)

        if Vb is not None:
            shape = Vb.shape if tf.is_tensor(Vb) else Vb
            initializer = cift(Vb) if tf.is_tensor(Vb) else tf.keras.initializers.get('zeros')
            self.Vb = self.add_weight('Vb', shape=shape, initializer=initializer, regularizer=None, constraint=None,
                                      dtype=self.dtype, trainable=False)
            shape = Wb.shape if tf.is_tensor(Wb) else Wb
            initializer = cift(Wb) if tf.is_tensor(Wb) else tf.keras.initializers.get('zeros')
            self.Wb = self.add_weight('Wb', shape=shape, initializer=initializer, regularizer=None, constraint=None,
                                      dtype=self.dtype, trainable=False)
        else:
            self.Vb, self.Wb = None, None

        # TODO: support tuple kernel sizes
        if isinstance(kernel_size, (list, tuple)):
            kernel_size = kernel_size[0]

        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg

        self.zero_pad = ((pad_beg, pad_end), (pad_beg, pad_end))
        self.zero_pad_layer = tf.keras.layers.ZeroPadding2D(padding=self.zero_pad,
                                                            data_format=kwargs.get('data_format', None))

        self.activation = tf.keras.activations.get(kwargs.pop('activation', None))

        use_bias = kwargs.get('use_bias', True)
        self.previous_layer_used_bias = kwargs.pop('previous_layer_used_bias', use_bias)
        if self.previous_layer_used_bias is False and mu is not None:
            if add_bias_if_nec is True:
                use_bias = True
                kwargs.update({'use_bias': use_bias})
            else:
                self.mu = None

        self.previous_layer_padding = kwargs.pop('previous_layer_padding', kwargs.get('padding', 'valid'))
        kwargs.pop('padding', '')
        kwargs.pop('kernel_initializer', '')
        kwargs.pop('bias_initializer', '')
        k_init = cift(Wt) if Wt is not None else tf.keras.initializers.get('zeros')
        b_init = cift(bp) if bp is not None else tf.keras.initializers.get('zeros')
        self.conv_layer = tf.keras.layers.Conv2D(filters, kernel_size, padding='valid', activation=None,
                                                 kernel_initializer=k_init, bias_initializer=b_init, **kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.include_offset = include_offset
        self.train_top_basis = train_top_basis
        self.add_bias_if_nec = add_bias_if_nec
        self.previous_layer_type = previous_layer_type

        self.height, self.width, self.pre_transform_depth = None, None, None
        self.post_transform_top_depth, self.post_transform_bot_depth = None, None

    def build(self, input_shape):
        super(Conv2DTransformLayer, self).build(input_shape)

        self.height = input_shape[-3] + self.zero_pad[0][0] + self.zero_pad[0][1]
        self.width = input_shape[-2] + self.zero_pad[1][0] + self.zero_pad[1][1]
        self.pre_transform_depth = input_shape[-1]
        self.post_transform_top_depth = self.Vt.shape[1]
        self.post_transform_bot_depth = self.Vb.shape[1] if self.Vb is not None else None

    def call(self, inputs, **kwargs):
        inputs = self.zero_pad_layer(inputs)

        inputs = tf.reshape(inputs, (-1, self.pre_transform_depth))

        h_top = tf.matmul(inputs, self.Vt) if self.mu is None else tf.matmul(tf.subtract(inputs, self.mu), self.Vt)
        h_top = tf.reshape(h_top, (-1, self.height, self.width, self.post_transform_top_depth))
        output = self.conv_layer(h_top)

        if self.include_offset:
            h_bot = tf.matmul(inputs, self.Vb) if self.mu is None else tf.matmul(tf.subtract(inputs, self.mu), self.Vb)
            h_bot = tf.reshape(h_bot, (-1, self.height, self.width, self.post_transform_bot_depth))
            output += tf.nn.conv2d(h_bot, self.Wb, strides=self.conv_layer.strides, padding='VALID',
                                   data_format='NHWC', dilations=None)

        return self.activation(output)

    def get_config(self):
        config = super(Conv2DTransformLayer, self).get_config()
        config.update({'filters': self.filters})
        config.update({'kernel_size': self.kernel_size})
        config.update({'mu': self.mu.shape if self.mu is not None else None})
        config.update({'Vt': self.Vt.shape})
        config.update({'Vb': self.Vb.shape if self.Vb is not None else None})
        config.update({'Wt': None})
        config.update({'Wb': self.Wb.shape if self.Wb is not None else None})
        config.update({'bp': None})
        config.update({'include_offset': self.include_offset})
        config.update({'train_top_basis': self.train_top_basis})
        config.update({'add_bias_if_nec': self.add_bias_if_nec})
        config.update({'previous_layer_type': self.previous_layer_type})
        config.update({'previous_layer_used_bias': self.previous_layer_used_bias})
        config.update({'previous_layer_padding': self.previous_layer_padding})
        config.update({
            'strides': self.conv_layer.strides,
            'padding': self.conv_layer.padding,
            'data_format': self.conv_layer.data_format,
            'dilation_rate': self.conv_layer.dilation_rate,
            'groups': self.conv_layer.groups,
            'activation': tf.keras.activations.serialize(self.activation),
            'use_bias': self.conv_layer.use_bias,
            'kernel_initializer': None,
            'bias_initializer': None,
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

        # TODO: support tuple strides and kernel sizes
        if isinstance(strides, (list, tuple)):
            strides = strides[0]
        if isinstance(kernel_size, (list, tuple)):
            kernel_size = kernel_size[0]

        if strides > 1:
            pad_total = kernel_size - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg

            zero_pad = ((pad_beg, pad_end), (pad_beg, pad_end))
            self.zero_pad_layer = tf.keras.layers.ZeroPadding2D(padding=zero_pad, data_format=data_format)

            self.explicit_pad = True
        else:
            self.explicit_pad = False

        kwargs.pop('padding', '')
        self.conv_layer = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides,
                                                 padding=('SAME' if strides == 1 else 'VALID'),
                                                 data_format=data_format, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.data_format = data_format

    def build(self, input_shape):
        super(Conv2DExplicitPadding, self).build(input_shape)

    def call(self, inputs, **kwargs):
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
            'padding': self.conv_layer.padding,
            'dilation_rate': self.conv_layer.dilation_rate,
            'groups': self.conv_layer.groups,
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

    def call(self, inputs, **kwargs):
        return inputs





class cift(tf.keras.initializers.Initializer):
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