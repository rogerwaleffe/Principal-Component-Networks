# TODO: I don't think full input/output based compression will work if the conv layers have biases
import tensorflow as tf

from src.pcns.layer_helpers import IdentityLayer, Conv2DExplicitPadding, Conv2DPCALayer, DensePCALayer
from src.pcns.layer_helpers import constant_initializer_from_tensor as cift
from src.pcns.compression_helpers import tf_pca, tf_kill_outputs, tf_transform_conv_weights, \
                                                          tf_transform_dense_weights, tf_get_indices_from_muVs, \
                                                          decode_cc



BN_GAMMA_REGULARIZER = None
BN_BETA_REGULARIZER = None
# TODO: not sure these are implemented for the compression layers
CNN_REGULARIZER = tf.keras.regularizers.l2(l=0.00005)
DENSE_REGULARIZER = tf.keras.regularizers.l2(l=0.00005)



def batch_norm(data_format='channels_last', gamma_regularizer=BN_GAMMA_REGULARIZER,
               beta_regularizer=BN_BETA_REGULARIZER, **kwargs):
    return tf.keras.layers.BatchNormalization(axis=1 if data_format == 'channels_first' else 3,
                                              momentum=0.997, epsilon=1e-5, gamma_regularizer=gamma_regularizer,
                                              beta_regularizer=beta_regularizer, **kwargs)

def conv2d(filters, kernel_size, strides=1, data_format='channels_last',
           kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_out',
                                                                    distribution='untruncated_normal'),
           kernel_regularizer=CNN_REGULARIZER, **kwargs):
    return Conv2DExplicitPadding(filters, kernel_size, strides=strides, data_format=data_format, use_bias=False,
                                 kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, **kwargs)
    # return tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, data_format=data_format, use_bias=False,
    #                               kernel_initializer=kernel_initializer, **kwargs)

def copy_bn_layer(bn_layer, indices=None):
    gamma, beta = bn_layer.weights[0], bn_layer.weights[1]
    moving_mean, moving_var = bn_layer.weights[2], bn_layer.weights[3]
    if indices is not None:
        gamma, beta = tf.gather(gamma, indices), tf.gather(beta, indices)
        moving_mean, moving_var = tf.gather(moving_mean, indices), tf.gather(moving_var, indices)
    return batch_norm(data_format='channels_first' if bn_layer.axis == 1 else 'channels_last', name=bn_layer.name,
                      gamma_initializer=cift(gamma), beta_initializer=cift(beta),
                      moving_mean_initializer=cift(moving_mean), moving_variance_initializer=cift(moving_var))



def create_ResNetBlockV1(filters, kernel_size, strides=1, conv0=None, data_format='channels_last',
                         compression_config=None, **kwargs):
    bn0 = batch_norm(data_format=data_format, name='bn0') if conv0 is not None else None
    bn1 = batch_norm(data_format=data_format, name='bn1')
    bn2 = batch_norm(data_format=data_format, name='bn2')
    conv1 = conv2d(filters, kernel_size, strides=strides, data_format=data_format, name='conv1')
    conv2 = conv2d(filters, kernel_size, strides=1, data_format=data_format, name='conv2')

    return ResNetBlockV1(conv0, bn0, conv1, conv2, bn1, bn2, kernel_size, strides, data_format, compression_config,
                         **kwargs)

def create_ResNetBlockV2(filters, kernel_size, strides=1, conv0=None, data_format='channels_last',
                         compression_config=None, **kwargs):
    bn1 = batch_norm(data_format=data_format, name='bn1')
    bn2 = batch_norm(data_format=data_format, name='bn2')
    conv1 = conv2d(filters, kernel_size=kernel_size, strides=strides, data_format=data_format, name='conv1')
    conv2 = conv2d(filters, kernel_size=kernel_size, strides=1, data_format=data_format, name='conv2')

    return ResNetBlockV2(conv0, conv1, conv2, bn1, bn2, kernel_size, strides, data_format, compression_config, **kwargs)



class ResNetBlock(tf.keras.layers.Layer):
    def __init__(self, kernel_size, strides, data_format, compression_config, **kwargs):
        super(ResNetBlock, self).__init__(**kwargs)

        # __init__ params used elsewhere
        self.kernel_size = kernel_size
        self.strides = strides
        self.data_format = data_format
        self.cc = compression_config
        self.cc_params = None

        # helper variables for printing etc.
        self.layers = {'before': [], 'path0': [], 'path1': [], 'after': []}
        self.all_layers = []
        self.all_layer_names = []

    def build(self, input_shape):
        super(ResNetBlock, self).build(input_shape)

        if self.cc is not None:
            W_c0_initial = self.conv0 if self.conv0 is not None else None
            self.cc_params = {'mu_i': None, 'V_i': None, 'e_i':None,
                              'W_c0': (None, W_c0_initial), 'W_c1': (None, self.conv1), 'W_c2': (None, self.conv2),
                              'mu_c1': None, 'V_c1': None, 'e_c1': None,
                              'in_indices': None, 'out_indices': None}

        all_layers = []
        for key in list(self.layers.keys()):
            all_layers.extend(self.layers[key])
        self.all_layers = all_layers
        self.all_layer_names = [l.name for l in all_layers]

    def get_layer(self, name):
        index = self.all_layer_names.index(name)
        return self.all_layers[index]

    def compute_initial_pca(self, inputs, version, verbose=True):
        if version == 'V2':
            inputs = self.bn1(inputs)
            inputs = self.relu1(inputs)

        if self.cc[0] is not None:
            num, ut = decode_cc(self.cc[0])
            mu, V, e = tf_pca(inputs, num, num_as_threshold=ut, conv=True, verbose=verbose,
                              prefix=' ({} conv1)'.format(self.name))
        else:
            mu, V, e = None, None, None
        return inputs, mu, V, e

    def set_cc_param(self, names, values):
        for n, v in zip(names, values):
            if 'W_' in n:
                self.cc_params[n] = (v, None)
            else:
                self.cc_params[n] = v

    def get_cc_param(self, names):
        if len(names) == 1:
            n = names[0]
            if 'W_' in n:
                return self.cc_params[n][0] if self.cc_params[n][0] is not None else self.cc_params[n][1].weights[0]
            else:
                return self.cc_params[n]

        values = []
        for n in names:
            if 'W_' in n:
                values.append(self.cc_params[n][0] if self.cc_params[n][0] is not None else
                              self.cc_params[n][1].weights[0])
            else:
                values.append(self.cc_params[n])
        return values

    def get_compressed_block(self, version, verbose):
        mu_i, V_i = self.get_cc_param(['mu_i', 'V_i'])

        if self.conv0 is not None:
            W_c0 = self.get_cc_param(['W_c0'])
            if mu_i is not None:
                W_p, b_p = tf_transform_conv_weights(mu_i, V_i, W_c0, tf.zeros([W_c0.shape[-1], ]))
                conv0 = Conv2DPCALayer(W_p.shape[-1], 1, mu_i, V_i, strides=self.strides, kernel_initializer=cift(W_p),
                                       bias_initializer=cift(b_p), data_format=self.data_format, name='conv0',
                                       kernel_regularizer=CNN_REGULARIZER, bias_regularizer=CNN_REGULARIZER)
            else:
                conv0 = conv2d(W_c0.shape[-1], kernel_size=1, strides=self.strides, data_format=self.data_format,
                               kernel_initializer=cift(W_c0), name='conv0')
        else:
            conv0 = None


        # conv1, need to remove filters
        W_c1, mu_c1, V_c1, W_c2 = self.get_cc_param(['W_c1', 'mu_c1', 'V_c1', 'W_c2'])

        if self.cc[1] is not None and mu_c1 is not None:
            num, ut = decode_cc(self.cc[1])
            _, W_c1, b_c1, mu_c1, V_c1, W_c2, indices_c1 = \
                tf_kill_outputs(W_c1, tf.zeros([W_c1.shape[-1], ]), mu_c1, V_c1, W_c2, num, num_as_threshold=ut,
                                conv=True, verbose=verbose, return_indices=True)
        else:
            b_c1 = tf.zeros([W_c1.shape[-1], ])
            indices_c1 = None

        if mu_i is not None:
            W_p, b_p = tf_transform_conv_weights(mu_i, V_i, W_c1, b_c1)
            conv1 = Conv2DPCALayer(W_p.shape[-1], self.kernel_size, mu_i, V_i, strides=self.strides,
                                   kernel_initializer=cift(W_p), bias_initializer=cift(b_p),
                                   data_format=self.data_format, name='conv1', kernel_regularizer=CNN_REGULARIZER,
                                   bias_regularizer=CNN_REGULARIZER)
        else:
            conv1 = conv2d(W_c1.shape[-1], kernel_size=self.kernel_size, strides=self.strides,
                           data_format=self.data_format, kernel_initializer=cift(W_c1), name='conv1')


        # conv2
        if mu_c1 is not None:
            W_p, b_p = tf_transform_conv_weights(mu_c1, V_c1, W_c2, tf.zeros([W_c2.shape[-1], ]))
            conv2 = Conv2DPCALayer(W_p.shape[-1], self.kernel_size, mu_c1, V_c1, strides=1,
                                   kernel_initializer=cift(W_p), bias_initializer=cift(b_p),
                                   data_format=self.data_format, name='conv2', kernel_regularizer=CNN_REGULARIZER,
                                   bias_regularizer=CNN_REGULARIZER)
        else:
            conv2 = conv2d(W_c2.shape[-1], kernel_size=self.kernel_size, strides=1,
                           data_format=self.data_format, kernel_initializer=cift(W_c2), name='conv2')


        if version == 'V2':
            in_indices = self.get_cc_param(['in_indices'])
            bn1 = copy_bn_layer(self.bn1, in_indices)
            bn2 = copy_bn_layer(self.bn2, indices_c1)
            new_block = ResNetBlockV2(conv0, conv1, conv2, bn1, bn2, name=self.name)
        else:
            out_indices = self.get_cc_param(['out_indices'])
            bn0 = copy_bn_layer(self.bn0, out_indices) if self.conv0 is not None else None
            bn1 = copy_bn_layer(self.bn1, indices_c1)
            bn2 = copy_bn_layer(self.bn2, out_indices)
            new_block = ResNetBlockV1(conv0, bn0, conv1, conv2, bn1, bn2, name=self.name)

        return new_block



class ResNetBlockV1(ResNetBlock):
    def __init__(self, conv0, bn0, conv1, conv2, bn1, bn2, kernel_size=None, strides=None, data_format=None,
                 compression_config=None, **kwargs):
        super(ResNetBlockV1, self).__init__(kernel_size, strides, data_format, compression_config, **kwargs)

        self.conv0 = conv0
        self.bn0 = bn0
        self.conv1 = conv1
        self.conv2 = conv2
        self.bn1 = bn1
        self.bn2 = bn2
        self.relu1 = tf.keras.layers.ReLU(name='relu1')
        self.relu2 = tf.keras.layers.ReLU(name='relu2')

    def build(self, input_shape):
        if self.conv0 is not None:
            self.layers['path0'].extend([self.conv0, self.bn0])

        self.layers['path1'].extend([self.conv1, self.bn1, self.relu1, self.conv2, self.bn2])
        self.layers['after'].extend([self.relu2])

        super(ResNetBlockV1, self).build(input_shape)

    def call(self, inputs, training=False, progress_input_arr=False, verbose=True):
        shortcut = inputs

        if self.conv0 is not None:
            shortcut = self.conv0(inputs)
            shortcut = self.bn0(shortcut, training=training)

        inputs = self.conv1(inputs)
        inputs = self.bn1(inputs, training=training)
        inputs = self.relu1(inputs)

        if progress_input_arr is True:
            # compute intermediate pca
            if self.cc[2] is not None:
                num, ut = decode_cc(self.cc[2])
                mu_c1, V_c1, e_c1 = tf_pca(inputs, num, num_as_threshold=ut, conv=True, verbose=verbose,
                                           prefix=' ({} conv2)'.format(self.name))
            else:
                mu_c1, V_c1, e_c1 = None, None, None
            self.set_cc_param(['mu_c1', 'V_c1', 'e_c1'], [mu_c1, V_c1, e_c1])

        inputs = self.conv2(inputs)
        inputs = self.bn2(inputs, training=training)

        inputs = inputs + shortcut
        inputs = self.relu2(inputs)

        return inputs



class ResNetBlockV2(ResNetBlock):
    def __init__(self, conv0, conv1, conv2, bn1, bn2, kernel_size=None, strides=None, data_format=None,
                 compression_config=None, **kwargs):
        super(ResNetBlockV2, self).__init__(kernel_size, strides, data_format, compression_config, **kwargs)

        self.conv0 = conv0
        self.conv1 = conv1
        self.conv2 = conv2
        self.bn1 = bn1
        self.bn2 = bn2
        self.relu1 = tf.keras.layers.ReLU(name='relu1')
        self.relu2 = tf.keras.layers.ReLU(name='relu2')

    def build(self, input_shape):
        if self.conv0 is not None:
            self.layers['before'].extend([self.bn1, self.relu1])
            self.layers['path0'].extend([self.conv0])
            self.layers['path1'].extend([self.conv1, self.bn2, self.relu2, self.conv2])
        else:
            self.layers['path1'].extend([self.bn1, self.relu1, self.conv1, self.bn2, self.relu2, self.conv2])

        super(ResNetBlockV2, self).build(input_shape)

    def call(self, inputs, training=False, progress_input_arr=False, verbose=True):
        shortcut = inputs

        inputs = self.bn1(inputs, training=training)
        inputs = self.relu1(inputs)

        if self.conv0 is not None:
            # From [2]: "when pre-activation is used, these projection shortcuts are also with pre-activation"
            shortcut = self.conv0(inputs)

        inputs = self.conv1(inputs)
        inputs = self.bn2(inputs, training=training)
        inputs = self.relu2(inputs)

        if progress_input_arr is True:
            # compute intermediate pca
            if self.cc[2] is not None:
                num, ut = decode_cc(self.cc[2])
                mu_c1, V_c1, e_c1 = tf_pca(inputs, num, num_as_threshold=ut, conv=True, verbose=verbose,
                                           prefix=' ({} conv2)'.format(self.name))
            else:
                mu_c1, V_c1, e_c1 = None, None, None
            self.set_cc_param(['mu_c1', 'V_c1', 'e_c1'], [mu_c1, V_c1, e_c1])

        inputs = self.conv2(inputs)

        return shortcut + inputs










def create_ResNetBottleneckBlockV1(filters, kernel_size, strides=1, conv0=None, data_format='channels_last',
                                   compression_config=None, **kwargs):
    bn0 = batch_norm(data_format=data_format, name='bn0') if conv0 is not None else None
    bn1 = batch_norm(data_format=data_format, name='bn1')
    bn2 = batch_norm(data_format=data_format, name='bn2')
    bn3 = batch_norm(data_format=data_format, name='bn3', gamma_initializer='zeros') # This initialization is a trick
    conv1 = conv2d(filters, kernel_size=1, strides=strides, data_format=data_format, name='conv1')
    conv2 = conv2d(filters, kernel_size=kernel_size, strides=1, data_format=data_format, name='conv2')
    conv3 = conv2d(4*filters, kernel_size=1, strides=1, data_format=data_format, name='conv3')

    return ResNetBottleneckBlockV1(conv0, bn0, conv1, conv2, conv3, bn1, bn2, bn3, kernel_size, strides, data_format,
                                   compression_config, **kwargs)

# V1.5 moves the stride to the 3x3 conv rather than the first 1x1 conv in the bottleneck. This change results in
# higher and more stable accuracy with less epochs than the original v1. This version requires ~12% more compute to
# train and has 6% reduced throughput for inference.
def create_ResNetBottleneckBlockV1_5(filters, kernel_size, strides=1, conv0=None, data_format='channels_last',
                                     compression_config=None, **kwargs):
    bn0 = batch_norm(data_format=data_format, name='bn0') if conv0 is not None else None
    bn1 = batch_norm(data_format=data_format, name='bn1')
    bn2 = batch_norm(data_format=data_format, name='bn2')
    bn3 = batch_norm(data_format=data_format, name='bn3', gamma_initializer='zeros') # This initialization is a trick
    conv1 = conv2d(filters, kernel_size=1, strides=1, data_format=data_format, name='conv1')
    conv2 = conv2d(filters, kernel_size=kernel_size, strides=strides, data_format=data_format, name='conv2')
    conv3 = conv2d(4*filters, kernel_size=1, strides=1, data_format=data_format, name='conv3')

    return ResNetBottleneckBlockV1(conv0, bn0, conv1, conv2, conv3, bn1, bn2, bn3, kernel_size, strides, data_format,
                                   compression_config, **kwargs)



class ResNetBottleneckBlockV1(tf.keras.layers.Layer): # TODO: ResNetBottleneckBlock
    def __init__(self, conv0, bn0, conv1, conv2, conv3, bn1, bn2, bn3, kernel_size=None, strides=None, data_format=None,
                 compression_config=None, **kwargs):
        super(ResNetBottleneckBlockV1, self).__init__(**kwargs)

        if not isinstance(conv1, (Conv2DExplicitPadding, Conv2DPCALayer, tf.keras.layers.Conv2D)):
            custom_objects = {'Conv2DExplicitPadding': Conv2DExplicitPadding, 'Conv2DPCALayer': Conv2DPCALayer,
                              'constant_initializer_from_tensor': cift}
            conv0 = None if conv0 is None else tf.keras.layers.deserialize(conv0, custom_objects=custom_objects)
            bn0 = None if bn0 is None else tf.keras.layers.deserialize(bn0, custom_objects=custom_objects)
            conv1 = tf.keras.layers.deserialize(conv1, custom_objects=custom_objects)
            conv2 = tf.keras.layers.deserialize(conv2, custom_objects=custom_objects)
            conv3 = tf.keras.layers.deserialize(conv3, custom_objects=custom_objects)
            bn1 = tf.keras.layers.deserialize(bn1, custom_objects=custom_objects)
            bn2 = tf.keras.layers.deserialize(bn2, custom_objects=custom_objects)
            bn3 = tf.keras.layers.deserialize(bn3, custom_objects=custom_objects)

        self.kernel_size = kernel_size
        self.strides = strides
        self.data_format = data_format
        self.cc = compression_config

        self.cc_params = None

        self.conv0 = conv0
        self.bn0 = bn0
        self.conv1 = conv1
        self.conv2 = conv2
        self.conv3 = conv3
        self.bn1 = bn1
        self.bn2 = bn2
        self.bn3 = bn3
        self.relu1 = tf.keras.layers.ReLU(name='relu1')
        self.relu2 = tf.keras.layers.ReLU(name='relu2')
        self.relu3 = tf.keras.layers.ReLU(name='relu3')

    def build(self, input_shape):
        super(ResNetBottleneckBlockV1, self).build(input_shape)

        W_c0_initial = self.conv0 if self.conv0 is not None else None
        self.cc_params = {'mu_i': None, 'V_i': None, 'e_i': None,
                          'W_c0': (None, W_c0_initial), 'W_c1': (None, self.conv1), 'W_c2': (None, self.conv2),
                          'W_c3': (None, self.conv3),
                          'mu_c1': None, 'V_c1': None, 'e_c1': None,
                          'mu_c2': None, 'V_c2': None, 'e_c2': None,
                          'in_indices': None, 'out_indices': None}

    def call(self, inputs, training=False):
        shortcut = inputs

        if self.conv0 is not None:
            shortcut = self.conv0(inputs)
            shortcut = self.bn0(shortcut, training=training)

        inputs = self.conv1(inputs)
        inputs = self.bn1(inputs, training=training)
        inputs = self.relu1(inputs)

        inputs = self.conv2(inputs)
        inputs = self.bn2(inputs, training=training)
        inputs = self.relu2(inputs)

        inputs = self.conv3(inputs)
        inputs = self.bn3(inputs, training=training)

        inputs = inputs + shortcut
        inputs = self.relu3(inputs)

        return inputs

    def get_config(self):
        config = super(ResNetBottleneckBlockV1, self).get_config()
        config.update({
            'conv0': None if self.conv0 is None else tf.keras.layers.serialize(self.conv0),
            'bn0': None if self.bn0 is None else tf.keras.layers.serialize(self.bn0),
            'conv1': tf.keras.layers.serialize(self.conv1),
            'conv2': tf.keras.layers.serialize(self.conv2),
            'conv3': tf.keras.layers.serialize(self.conv3),
            'bn1': tf.keras.layers.serialize(self.bn1),
            'bn2': tf.keras.layers.serialize(self.bn2),
            'bn3': tf.keras.layers.serialize(self.bn3),
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'data_format': self.data_format,
            'compression_config': self.cc
        })

        return config

    def set_cc_param(self, names, values):
        for n, v in zip(names, values):
            if 'W_' in n:
                self.cc_params[n] = (v, None)
            else:
                self.cc_params[n] = v

    def get_cc_param(self, names):
        if len(names) == 1:
            n = names[0]
            if 'W_' in n:
                return self.cc_params[n][0] if self.cc_params[n][0] is not None else self.cc_params[n][1].weights[0]
            else:
                return self.cc_params[n]

        values = []
        for n in names:
            if 'W_' in n:
                values.append(self.cc_params[n][0] if self.cc_params[n][0] is not None else
                              self.cc_params[n][1].weights[0])
            else:
                values.append(self.cc_params[n])
        return values

    def get_compressed_block(self, version, verbose):
        # TODO: easier to switch between v1 and v1.5
        # conv0
        mu_i, V_i = self.get_cc_param(['mu_i', 'V_i'])

        if self.conv0 is not None:
            W_c0 = self.get_cc_param(['W_c0'])
            if mu_i is not None:
                W_p, b_p = tf_transform_conv_weights(mu_i, V_i, W_c0, tf.zeros([W_c0.shape[-1], ]))
                conv0 = Conv2DPCALayer(W_p.shape[-1], 1, mu_i, V_i, strides=self.strides, kernel_initializer=cift(W_p),
                                       bias_initializer=cift(b_p), data_format=self.data_format, name='conv0',
                                       kernel_regularizer=CNN_REGULARIZER, bias_regularizer=CNN_REGULARIZER)
            else:
                conv0 = conv2d(W_c0.shape[-1], kernel_size=1, strides=self.strides, data_format=self.data_format,
                               kernel_initializer=cift(W_c0), name='conv0')
        else:
            conv0 = None


        # conv1, need to remove filters
        W_c1, mu_c1, V_c1, W_c2 = self.get_cc_param(['W_c1', 'mu_c1', 'V_c1', 'W_c2'])

        if self.cc[1] is not None and mu_c1 is not None:
            num, ut = decode_cc(self.cc[1])
            _, W_c1, b_c1, mu_c1, V_c1, W_c2, indices_c1 = \
                tf_kill_outputs(W_c1, tf.zeros([W_c1.shape[-1], ]), mu_c1, V_c1, W_c2, num, num_as_threshold=ut,
                                conv=True, verbose=verbose, return_indices=True)
        else:
            b_c1 = tf.zeros([W_c1.shape[-1], ])
            indices_c1 = None

        if mu_i is not None:
            W_p, b_p = tf_transform_conv_weights(mu_i, V_i, W_c1, b_c1)
            conv1 = Conv2DPCALayer(W_p.shape[-1], 1, mu_i, V_i, strides=1, kernel_initializer=cift(W_p),
                                   bias_initializer=cift(b_p), data_format=self.data_format, name='conv1',
                                   kernel_regularizer=CNN_REGULARIZER, bias_regularizer=CNN_REGULARIZER)
        else:
            conv1 = conv2d(W_c1.shape[-1], kernel_size=1, strides=1, data_format=self.data_format,
                           kernel_initializer=cift(W_c1), name='conv1')


        # conv2, need to remove filters
        mu_c2, V_c2, W_c3 = self.get_cc_param(['mu_c2', 'V_c2', 'W_c3'])

        if self.cc[3] is not None and mu_c2 is not None:
            num, ut = decode_cc(self.cc[3])
            _, W_c2, b_c2, mu_c2, V_c2, W_c3, indices_c2 = \
                tf_kill_outputs(W_c2, tf.zeros([W_c2.shape[-1], ]), mu_c2, V_c2, W_c3, num, num_as_threshold=ut,
                                conv=True, verbose=verbose, return_indices=True)
        else:
            b_c2 = tf.zeros([W_c2.shape[-1], ])
            indices_c2 = None

        if mu_c1 is not None:
            W_p, b_p = tf_transform_conv_weights(mu_c1, V_c1, W_c2, b_c2)
            conv2 = Conv2DPCALayer(W_p.shape[-1], self.kernel_size, mu_c1, V_c1, strides=self.strides,
                                   kernel_initializer=cift(W_p), bias_initializer=cift(b_p),
                                   data_format=self.data_format, name='conv2', kernel_regularizer=CNN_REGULARIZER,
                                   bias_regularizer=CNN_REGULARIZER)
        else:
            conv2 = conv2d(W_c2.shape[-1], kernel_size=self.kernel_size, strides=self.strides,
                           data_format=self.data_format, kernel_initializer=cift(W_c2), name='conv2')


        # conv3
        if mu_c2 is not None:
            W_p, b_p = tf_transform_conv_weights(mu_c2, V_c2, W_c3, tf.zeros([W_c3.shape[-1], ]))
            conv3 = Conv2DPCALayer(W_p.shape[-1], 1, mu_c2, V_c2, strides=1, kernel_initializer=cift(W_p),
                                   bias_initializer=cift(b_p), data_format=self.data_format, name='conv3',
                                   kernel_regularizer=CNN_REGULARIZER, bias_regularizer=CNN_REGULARIZER)
        else:
            conv3 = conv2d(W_c3.shape[-1], kernel_size=1, strides=1, data_format=self.data_format,
                           kernel_initializer=cift(W_c3), name='conv3')

        # if version == 'V2':
        #     in_indices = self.get_cc_param(['in_indices'])
        #     bn1 = copy_bn_layer(self.bn1, in_indices)
        #     bn2 = copy_bn_layer(self.bn2, indices)
        #     new_block = ResNetBlockV2(conv0, conv1, conv2, bn1, bn2, name=self.name)
        # else:

        out_indices = self.get_cc_param(['out_indices'])
        bn0 = copy_bn_layer(self.bn0, out_indices) if self.conv0 is not None else None
        bn1 = copy_bn_layer(self.bn1, indices_c1)
        bn2 = copy_bn_layer(self.bn2, indices_c2)
        bn3 = copy_bn_layer(self.bn3, out_indices)
        new_block = ResNetBottleneckBlockV1(conv0, bn0, conv1, conv2, conv3, bn1, bn2, bn3, kernel_size=None,
                                            strides=None, data_format=None, compression_config=None, name=self.name)

        return new_block










class ResNet:
    def __init__(self, input_shape, num_classes, bottleneck, num_filters_at_start, initial_kernel_size,
                 initial_conv_strides, initial_pool_size, initial_pool_strides, num_residual_blocks_per_stage,
                 first_block_strides_per_stage, kernel_size, project_first_residual=True, version='V2',
                 data_format='channels_last', compression_config=None):

        assert bottleneck in [True, False], 'bottleneck should be boolean in [True, False]'
        assert version in ['V1', 'V2'], 'version should be in [V1, V2]'
        assert data_format in ['channels_first', 'channels_last'], 'data_format in [channels_first, channels_last]'

        if version == 'V2' and not bottleneck:
            self.block = create_ResNetBlockV2
        elif version == 'V1' and not bottleneck:
            self.block = create_ResNetBlockV1
        elif version == 'V1':
            self.block = create_ResNetBottleneckBlockV1_5
        else:
            raise NotImplementedError("Not Implemented Yet")

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.bottleneck = bottleneck
        self.num_filters_at_start = num_filters_at_start
        self.initial_kernel_size = initial_kernel_size
        self.initial_conv_strides = initial_conv_strides
        self.initial_pool_size = initial_pool_size
        self.initial_pool_strides = initial_pool_strides
        self.num_residual_blocks_per_stage = num_residual_blocks_per_stage
        self.first_block_strides_per_stage = first_block_strides_per_stage
        self.kernel_size = kernel_size
        self.project_first_residual = project_first_residual
        self.version = version
        self.data_format = data_format
        self.cc = compression_config

    def get_model(self):
        # create the model
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=self.input_shape, name='input'))

        if self.data_format == 'channels_first':
            # Convert the inputs from channels_last (NHWC) to channels_first (NCHW) (provides performance boost on GPU).
            model.add(tf.keras.layers.Permute((3, 1, 2), name='permute'))

        # initial conv layer
        model.add(conv2d(self.num_filters_at_start, self.initial_kernel_size, strides=self.initial_conv_strides,
                         data_format=self.data_format, name='initial_conv'))

        # We do not include batch normalization or activation functions in V2 for the initial conv because the first
        # ResNet unit will perform these for both the shortcut and non-shortcut paths as part of the block's projection.
        if self.version == 'V1':
            model.add(batch_norm(data_format=self.data_format, name='initial_bn'))
            model.add(tf.keras.layers.ReLU(name='initial_relu'))

        # optional pooling after initial conv
        if self.initial_pool_size:
            model.add(tf.keras.layers.MaxPooling2D(pool_size=self.initial_pool_size, strides=self.initial_pool_strides,
                                                   padding='same', data_format=self.data_format, name='initial_mp'))

        # stages of ResNet blocks
        for i, num_blocks in enumerate(self.num_residual_blocks_per_stage):
            filters = self.num_filters_at_start * (2 ** i)
            filters_out = filters * 4 if self.bottleneck else filters

            if i == 0 and self.project_first_residual is False:
                conv0 = IdentityLayer(name='identity')
            else:
                conv0 = conv2d(filters_out, 1, strides=self.first_block_strides_per_stage[i],
                               data_format=self.data_format, name='conv0')

            cc = None if self.cc is None else self.cc['{}_{}'.format(i+1, 1)]
            model.add(self.block(filters, self.kernel_size, strides=self.first_block_strides_per_stage[i],
                                 conv0=conv0, data_format=self.data_format, compression_config=cc,
                                 name='stage{}_block{}'.format(i+1, 1)))

            for j in range(1, num_blocks):
                cc = None if self.cc is None else self.cc['{}_{}'.format(i+1, j+1)]
                model.add(self.block(filters, self.kernel_size, strides=1, conv0=None, data_format=self.data_format,
                                     compression_config=cc, name='stage{}_block{}'.format(i+1, j+1)))

        # From [2]: for the last Residual Unit (...), we adopt an extra activation right after its element-wise addition
        if self.version == 'V2':
            model.add(batch_norm(data_format=self.data_format, name='final_bn'))
            model.add(tf.keras.layers.ReLU(name='final_relu'))

        # final layers
        model.add(tf.keras.layers.GlobalAveragePooling2D(data_format=self.data_format, name='ap'))
        model.add(tf.keras.layers.Dense(self.num_classes, activation=None, use_bias=True, name='fc1',
                                        kernel_regularizer=DENSE_REGULARIZER, bias_regularizer=DENSE_REGULARIZER))
        # model.add(tf.keras.layers.Softmax(name='softmax'))

        return model

    def get_compressed_model(self, input_arr, overall_model, verbose=True):
        # TODO: beyond bottleneck=False, handle: channels_first implementation, no projection at beginning

        # create the model
        compressed_model = tf.keras.Sequential()
        compressed_model.add(tf.keras.layers.InputLayer(input_shape=self.input_shape, name='input'))

        # progress through initial conv layer
        input_arr = overall_model.get_layer('initial_conv')(input_arr)
        if self.version == 'V1':
            input_arr = overall_model.get_layer('initial_bn')(input_arr)
            input_arr = overall_model.get_layer('initial_relu')(input_arr)

        if self.initial_pool_size:
            input_arr = overall_model.get_layer('initial_mp')(input_arr)

        # stages of ResNet blocks
        previous_muVs = []
        for i, num_blocks in enumerate(self.num_residual_blocks_per_stage):

            first_block_in_stage = overall_model.get_layer('stage{}_block{}'.format(i+1, 1))
            _, mu, V, e = first_block_in_stage.compute_initial_pca(input_arr, self.version, verbose=verbose)


            # finalize previous layer/stage
            if i == 0:
                # kill outputs in initial_conv (using mu, V), update W_c0 and W_c1 for first_block_in_stage
                if mu is not None and self.cc['initial_conv'][1] is not None:
                    W_i = overall_model.get_layer('initial_conv').weights[0]
                    b_i = tf.zeros([W_i.shape[-1], ])
                    W_c0, W_c1 = first_block_in_stage.get_cc_param(['W_c0', 'W_c1'])

                    num, ut = decode_cc(self.cc['initial_conv'][1])
                    n_i, W_i, b_i, mu, V, W_c0, indices = \
                        tf_kill_outputs(W_i, b_i, mu, V, W_c0, num, num_as_threshold=ut, conv=True, verbose=verbose,
                                        return_indices=True)
                    W_c1 = tf.gather(W_c1, indices, axis=2)

                    first_block_in_stage.set_cc_param(['mu_i', 'V_i', 'W_c0', 'W_c1', 'in_indices'],
                                                      [mu, V, W_c0, W_c1, indices])

                    new_layer = conv2d(n_i, self.initial_kernel_size, strides=self.initial_conv_strides,
                                       data_format=self.data_format, name='initial_conv', kernel_initializer=cift(W_i))
                else:
                    W_i = overall_model.get_layer('initial_conv').weights[0]
                    indices = None
                    first_block_in_stage.set_cc_param(['mu_i', 'V_i', 'in_indices'], [mu, V, indices])

                    new_layer = conv2d(self.num_filters_at_start, self.initial_kernel_size,
                                       strides=self.initial_conv_strides, data_format=self.data_format,
                                       name='initial_conv', kernel_initializer=cift(W_i))

                compressed_model.add(new_layer)
                self.add_other_initial_layers(compressed_model, overall_model, indices)


            else:
                # using previous_muVs, mu, V decide which filters of previous stage (conv0, all conv2) to kill
                if self.cc[str(i)] is not None:
                    num, ut = decode_cc(self.cc[str(i)])
                    indices = tf_get_indices_from_muVs(previous_muVs, V, num, num_as_threshold=ut, verbose=verbose,
                                                       use_all=self.cc['use_all_muVs'],
                                                       evals=e if self.cc['weighted_row_sum'] is True else None)
                else:
                    indices = None

                # previous stage: update previous_muVs, conv1 for blocks >= 2, kill filters in conv0, all conv2
                self.finalize_stage(i, compressed_model, overall_model, previous_muVs, indices, verbose=verbose)

                # this stage: update mu, V, conv0, conv1 for block 1 to handle filter removal
                if indices is not None:
                    W_c0, W_c1 = first_block_in_stage.get_cc_param(['W_c0', 'W_c1'])
                    mu, V = tf.gather(mu, indices), tf.gather(V, indices, axis=0)
                    W_c0, W_c1 = tf.gather(W_c0, indices, axis=2), tf.gather(W_c1, indices, axis=2)

                    first_block_in_stage.set_cc_param(['mu_i', 'V_i', 'W_c0', 'W_c1', 'in_indices'],
                                                      [mu, V, W_c0, W_c1, indices])
                else:
                    first_block_in_stage.set_cc_param(['mu_i', 'V_i', 'in_indices'], [mu, V, indices])

                previous_muVs = []

            # continue going through this stage
            input_arr = first_block_in_stage.call(input_arr, training=False, progress_input_arr=True, verbose=verbose)

            # other blocks in stage
            for j in range(1, num_blocks):
                block_in_stage = overall_model.get_layer('stage{}_block{}'.format(i+1, j+1))
                _, mu, V, e = block_in_stage.compute_initial_pca(input_arr, self.version, verbose=verbose)
                previous_muVs.append((mu, V, e))

                input_arr = block_in_stage.call(input_arr, training=False, progress_input_arr=True, verbose=verbose)

        # final layers
        if self.version == 'V2':
            input_arr = overall_model.get_layer('final_bn')(input_arr)
            input_arr = overall_model.get_layer('final_relu')(input_arr)

        if self.cc['fc1'][0] is not None:
            input_arr = overall_model.get_layer('ap')(input_arr)
            num, ut = decode_cc(self.cc['fc1'][0])
            mu, V, e = tf_pca(input_arr, num, num_as_threshold=ut, verbose=verbose, prefix='{} '.format(' (fc1)'))
        else:
            mu, V, e = None, None, None

        # using previous_muVs, mu, V decide which filters of previous stage (conv0, all conv2) to kill
        index = len(self.num_residual_blocks_per_stage)
        if self.cc[str(index)] is not None:
            num, ut = decode_cc(self.cc[str(index)])
            indices = tf_get_indices_from_muVs(previous_muVs, V, num, num_as_threshold=ut, verbose=verbose,
                                               use_all=self.cc['use_all_muVs'],
                                               evals=e if self.cc['weighted_row_sum'] is True else None)
        else:
            indices = None

        # previous stage: update previous_muVs, conv1 for blocks >= 2, kill filters in conv0, all conv2
        self.finalize_stage(index, compressed_model, overall_model, previous_muVs, indices, verbose=verbose)

        # update mu, V, fc1 weights to handle filter removal
        W_fc1, b_fc1 = overall_model.get_layer('fc1').weights[0], overall_model.get_layer('fc1').weights[1]
        if indices is not None:
            mu, V = tf.gather(mu, indices), tf.gather(V, indices, axis=0)
            W_fc1 = tf.gather(W_fc1, indices, axis=0)

        if self.version == 'V2':
            compressed_model.add(copy_bn_layer(overall_model.get_layer('final_bn'), indices))
            compressed_model.add(tf.keras.layers.ReLU(name='final_relu'))

        compressed_model.add(tf.keras.layers.GlobalAveragePooling2D(data_format=self.data_format, name='ap'))
        if mu is not None:
            W_fc1_p, b_fc1_p = tf_transform_dense_weights(mu, V, W_fc1, b_fc1)
            compressed_model.add(DensePCALayer(self.num_classes, mu, V, kernel_initializer=cift(W_fc1_p),
                                               bias_initializer=cift(tf.squeeze(b_fc1_p)), activation=None, name='fc1',
                                               kernel_regularizer=DENSE_REGULARIZER,
                                               bias_regularizer=DENSE_REGULARIZER))
        else:
            compressed_model.add(tf.keras.layers.Dense(self.num_classes, kernel_initializer=cift(W_fc1),
                                                       bias_initializer=cift(tf.squeeze(b_fc1)), activation=None,
                                                       name='fc1', kernel_regularizer=DENSE_REGULARIZER,
                                                       bias_regularizer=DENSE_REGULARIZER))
        # compressed_model.add(tf.keras.layers.Softmax(name='softmax'))

        return compressed_model

    def add_other_initial_layers(self, cm, m, indices):
        if self.version == 'V1':
            cm.add(copy_bn_layer(m.get_layer('initial_bn'), indices))
            cm.add(tf.keras.layers.ReLU(name='initial_relu'))
        if self.initial_pool_strides:
            cm.add(tf.keras.layers.MaxPooling2D(pool_size=self.initial_pool_size, strides=self.initial_pool_strides,
                                                padding='same', data_format=self.data_format, name='initial_mp'))

    def finalize_stage(self, i, cm, m, muVs, indices, verbose=True):
        for j in range(0, self.num_residual_blocks_per_stage[i-1]):
            p_block = m.get_layer('stage{}_block{}'.format(i, j+1))
            if j == 0:
                W_c0, W_c2 = p_block.get_cc_param(['W_c0', 'W_c2'])
                # remove filters from conv0, conv2, give indices back for bn0, bn2
                if indices is not None:
                    W_c0, W_c2 = tf.gather(W_c0, indices, axis=3), tf.gather(W_c2, indices, axis=3)
                p_block.set_cc_param(['W_c0', 'W_c2', 'out_indices'], [W_c0, W_c2, indices])
            else:
                mu_i, V_i, _ = muVs[j-1]
                W_c1, W_c2 = p_block.get_cc_param(['W_c1', 'W_c2'])
                #process muVs, W_c1 to handle filter removal, remove filters from conv2, give indices for bn2
                if indices is not None:
                    mu_i, V_i = tf.gather(mu_i, indices), tf.gather(V_i, indices, axis=0)
                    W_c1 = tf.gather(W_c1, indices, axis=2)
                    W_c2 = tf.gather(W_c2, indices, axis=3)
                p_block.set_cc_param(['mu_i', 'V_i', 'W_c1', 'W_c2', 'in_indices', 'out_indices'],
                                     [mu_i, V_i, W_c1, W_c2, indices, indices])

            # get the new compressed block and add it to the compressed model
            new_block = p_block.get_compressed_block(self.version, verbose=verbose)
            cm.add(new_block)

    @staticmethod
    def print_ResNet(model):
        for layer in model.layers:
            print(layer.name)
            if 'block' in layer.name:
                for internal_layer in layer.layers['before']:
                    print('\t' + internal_layer.name)
                print('\tPath0:', end=' ')
                for internal_layer in layer.layers['path0']:
                    print(internal_layer.name, end=' ')
                print()
                print('\tPath1:', end=' ')
                for internal_layer in layer.layers['path1']:
                    print(internal_layer.name, end=' ')
                print()
                for internal_layer in layer.layers['after']:
                    print('\t' + internal_layer.name)










    def finalize_stage_v2(self, i, cm, m, muVs, indices, verbose=True):
        for j in range(0, self.num_residual_blocks_per_stage[i-1]):
            p_block = m.get_layer('stage{}_block{}'.format(i, j+1))
            if j == 0:
                W_c0, W_c3 = p_block.get_cc_param(['W_c0', 'W_c3'])
                # remove filters from conv0, conv3, give indices back for bn0, bn3
                if indices is not None:
                    W_c0, W_c3 = tf.gather(W_c0, indices, axis=3), tf.gather(W_c3, indices, axis=3)
                p_block.set_cc_param(['W_c0', 'W_c3', 'out_indices'], [W_c0, W_c3, indices])
            else:
                mu_i, V_i, _ = muVs[j-1]
                W_c1, W_c3 = p_block.get_cc_param(['W_c1', 'W_c3'])
                #process muVs, W_c1 to handle filter removal, remove filters from conv3, give indices for bn3
                if indices is not None:
                    mu_i, V_i = tf.gather(mu_i, indices), tf.gather(V_i, indices, axis=0)
                    W_c1 = tf.gather(W_c1, indices, axis=2)
                    W_c3 = tf.gather(W_c3, indices, axis=3)
                p_block.set_cc_param(['mu_i', 'V_i', 'W_c1', 'W_c3', 'in_indices', 'out_indices'],
                                     [mu_i, V_i, W_c1, W_c3, indices, indices])

            # get the new compressed block and add it to the compressed model
            new_block = p_block.get_compressed_block(self.version, verbose=verbose)
            cm.add(new_block)

    def get_compressed_model_given_muVs(self, vc, overall_model, mu_V_e_list, verbose=True):
        # TODO: bottleneck V2, handle: channels_first implementation, no projection at beginning

        # create the model
        compressed_model = tf.keras.Sequential()
        compressed_model.add(tf.keras.layers.InputLayer(input_shape=self.input_shape, name='input'))

        # stages of ResNet blocks
        previous_muVs = []
        stage_start_index = 0
        for i, num_blocks in enumerate(self.num_residual_blocks_per_stage):

            first_block_in_stage = overall_model.get_layer('stage{}_block{}'.format(i+1, 1))
            mu, V, e = mu_V_e_list[stage_start_index][:3] # initial PCA


            # finalize previous layer/stage
            if i == 0:
                # kill outputs in initial_conv (using mu, V), update W_c0 and W_c1 for first_block_in_stage
                if mu is not None and self.cc['initial_conv'][1] is not None:
                    W_i = overall_model.get_layer('initial_conv').weights[0]
                    b_i = tf.zeros([W_i.shape[-1], ])
                    W_c0, W_c1 = first_block_in_stage.get_cc_param(['W_c0', 'W_c1'])

                    num, ut = decode_cc(self.cc['initial_conv'][1])
                    n_i, W_i, b_i, mu, V, W_c0, indices = \
                        tf_kill_outputs(W_i, b_i, mu, V, W_c0, num, num_as_threshold=ut, conv=True, verbose=verbose,
                                        return_indices=True)
                    W_c1 = tf.gather(W_c1, indices, axis=2)

                    first_block_in_stage.set_cc_param(['mu_i', 'V_i', 'W_c0', 'W_c1', 'in_indices'],
                                                      [mu, V, W_c0, W_c1, indices])

                    new_layer = conv2d(n_i, self.initial_kernel_size, strides=self.initial_conv_strides,
                                       data_format=self.data_format, name='initial_conv', kernel_initializer=cift(W_i))
                else:
                    W_i = overall_model.get_layer('initial_conv').weights[0]
                    indices = None
                    first_block_in_stage.set_cc_param(['mu_i', 'V_i', 'in_indices'], [mu, V, indices])

                    new_layer = conv2d(self.num_filters_at_start, self.initial_kernel_size,
                                       strides=self.initial_conv_strides, data_format=self.data_format,
                                       name='initial_conv', kernel_initializer=cift(W_i))

                compressed_model.add(new_layer)
                self.add_other_initial_layers(compressed_model, overall_model, indices)


            else:
                # using previous_muVs, mu, V decide which filters of previous stage (conv0, all conv3) to kill
                if self.cc[str(i)] is not None:
                    num, ut = decode_cc(self.cc[str(i)])
                    indices = tf_get_indices_from_muVs(previous_muVs, V, num, num_as_threshold=ut, verbose=verbose,
                                                       use_all=vc['use_all_muVs'],
                                                       evals=e if vc['weighted_row_sum'] is True else None)
                else:
                    indices = None

                # previous stage: update previous_muVs, conv1 for blocks >= 2, kill filters in conv0, all conv3
                self.finalize_stage_v2(i, compressed_model, overall_model, previous_muVs, indices, verbose=verbose)

                # this stage: update mu, V, conv0, conv1 for block 1 to handle filter removal
                if indices is not None:
                    W_c0, W_c1 = first_block_in_stage.get_cc_param(['W_c0', 'W_c1'])
                    mu, V = tf.gather(mu, indices), tf.gather(V, indices, axis=0)
                    W_c0, W_c1 = tf.gather(W_c0, indices, axis=2), tf.gather(W_c1, indices, axis=2)

                    first_block_in_stage.set_cc_param(['mu_i', 'V_i', 'W_c0', 'W_c1', 'in_indices'],
                                                      [mu, V, W_c0, W_c1, indices])
                else:
                    first_block_in_stage.set_cc_param(['mu_i', 'V_i', 'in_indices'], [mu, V, indices])

                previous_muVs = []

            # continue going through this stage
            first_block_in_stage.set_cc_param(['mu_c1', 'V_c1', 'e_c1', 'mu_c2', 'V_c2', 'e_c2'],
                                              list(mu_V_e_list[stage_start_index][3:]))

            # other blocks in stage
            for j in range(1, num_blocks):
                block_in_stage = overall_model.get_layer('stage{}_block{}'.format(i+1, j+1))
                mu, V, e = mu_V_e_list[stage_start_index+j][:3]
                previous_muVs.append((mu, V, e))

                block_in_stage.set_cc_param(['mu_c1', 'V_c1', 'e_c1', 'mu_c2', 'V_c2', 'e_c2'],
                                            list(mu_V_e_list[stage_start_index+j][3:]))

            stage_start_index += num_blocks

        # final layers
        # if self.version == 'V2':
        #     input_arr = overall_model.get_layer('final_bn')(input_arr)
        #     input_arr = overall_model.get_layer('final_relu')(input_arr)
        #
        # input_arr = overall_model.get_layer('ap')(input_arr)
        # num, ut = decode_cc(self.cc['fc1'][0])
        # mu, V, e = tf_pca(input_arr, num, num_as_threshold=ut, verbose=verbose, prefix='{} '.format(' (fc1)'))
        mu, V, e = mu_V_e_list[-1]

        # using previous_muVs, mu, V decide which filters of previous stage (conv0, all conv3) to kill
        index = len(self.num_residual_blocks_per_stage)
        if self.cc[str(index)] is not None:
            num, ut = decode_cc(self.cc[str(index)])
            indices = tf_get_indices_from_muVs(previous_muVs, V, num, num_as_threshold=ut, verbose=verbose,
                                               use_all=vc['use_all_muVs'],
                                               evals=e if vc['weighted_row_sum'] is True else None)
        else:
            indices = None

        # previous stage: update previous_muVs, conv1 for blocks >= 2, kill filters in conv0, all conv3
        self.finalize_stage_v2(index, compressed_model, overall_model, previous_muVs, indices, verbose=verbose)

        # update mu, V, fc1 weights to handle filter removal
        W_fc1, b_fc1 = overall_model.get_layer('fc1').weights[0], overall_model.get_layer('fc1').weights[1]
        if indices is not None:
            mu, V = tf.gather(mu, indices), tf.gather(V, indices, axis=0)
            W_fc1 = tf.gather(W_fc1, indices, axis=0)

        # if self.version == 'V2':
        #     compressed_model.add(copy_bn_layer(overall_model.get_layer('final_bn'), indices))
        #     compressed_model.add(tf.keras.layers.ReLU(name='final_relu'))
        #
        compressed_model.add(tf.keras.layers.GlobalAveragePooling2D(data_format=self.data_format, name='ap'))
        if mu is not None:
            W_fc1_p, b_fc1_p = tf_transform_dense_weights(mu, V, W_fc1, b_fc1)
            compressed_model.add(DensePCALayer(self.num_classes, mu, V, kernel_initializer=cift(W_fc1_p),
                                               bias_initializer=cift(tf.squeeze(b_fc1_p)), activation=None, name='fc1',
                                               kernel_regularizer=DENSE_REGULARIZER,
                                               bias_regularizer=DENSE_REGULARIZER))
        else:
            compressed_model.add(tf.keras.layers.Dense(self.num_classes, kernel_initializer=cift(W_fc1),
                                                       bias_initializer=cift(tf.squeeze(b_fc1)), activation=None,
                                                       name='fc1', kernel_regularizer=DENSE_REGULARIZER,
                                                       bias_regularizer=DENSE_REGULARIZER))
        # compressed_model.add(tf.keras.layers.Softmax(name='softmax'))

        return compressed_model




















# def ResNet(input_shape, num_classes, bottleneck, num_filters_at_start, kernel_size, initial_conv_strides,
#            initial_pool_size, initial_pool_strides, num_residual_blocks_per_stage, first_block_strides_per_stage,
#            project_first_residual=False, version='V2', data_format='channels_last'):

    # assert bottleneck in [True, False], 'bottleneck should be boolean in [True, False]'
    # assert version in ['V1', 'V2'], 'version should be in [V1, V2]'
    # assert data_format in ['channels_first', 'channels_last'], 'data_format should b in [channels_first, channels_last]'
    #
    # if version == 'V2' and not bottleneck:
    #     Block = ResNetBlockV2
    # elif version == 'V1' and not bottleneck:
    #     Block = ResNetBlockV1
    # else:
    #     raise NotImplementedError("Not Implemented Yet")


    # # Create the model
    # model = tf.keras.Sequential()
    # model.add(tf.keras.layers.InputLayer(input_shape=input_shape, name='input'))

    # if data_format == 'channels_first':
    #     # Convert the inputs from channels_last (NHWC) to channels_first (NCHW). Provides a performance boost on GPU.
    #     model.add(tf.keras.layers.Permute((3, 1, 2), name='permute'))
    #
    # # Initial Conv Layer
    # model.add(conv2d(num_filters_at_start, kernel_size, initial_conv_strides, data_format=data_format,
    #                  name='initial_conv'))

    # # We do not include batch normalization or activation functions in V2 for the initial conv because the first ResNet
    # # unit will perform these for both the shortcut and non-shortcut paths as part of the first block's projection.
    # if version == 'V1':
    #     model.add(batch_norm(data_format=data_format, name='initial_bn'))
    #     model.add(tf.keras.layers.ReLU(name='initial_relu'))
    #
    # # Optional Pooling After Initial Conv
    # if initial_pool_size:
    #     model.add(tf.keras.layers.MaxPooling2D(pool_size=initial_pool_size, strides=initial_pool_strides,
    #                                            padding='same', data_format=data_format, name='initial_mp'))

    # # Stages of ResNet Blocks
    # for i, num_blocks in enumerate(num_residual_blocks_per_stage):
    #     filters = num_filters_at_start * (2**i)
    #     filters_out = filters * 4 if bottleneck else filters
    #
    #     if i == 0 and project_first_residual is False:
    #         projection_shortcut = IdentityLayer(name='identity')
    #     else:
    #         projection_shortcut = conv2d(filters_out, kernel_size=1, strides=first_block_strides_per_stage[i],
    #                                      data_format=data_format, name='conv0')
    #
    #     model.add(Block(filters, first_block_strides_per_stage[i], projection_shortcut, data_format=data_format,
    #                     name='stage{}_block{}'.format(i+1, 1)))
    #
    #     for j in range(1, num_blocks):
    #         model.add(Block(filters, strides=1, projection_shortcut=None, data_format=data_format,
    #                         name='stage{}_block{}'.format(i+1, j+1)))

    # # From [2]: for the last Residual Unit (...), we adopt an extra activation right after its element-wise addition
    # if version == 'V2':
    #     model.add(batch_norm(data_format=data_format, name='final_bn'))
    #     model.add(tf.keras.layers.ReLU(name='final_relu'))
    #
    # # Final Layers
    # model.add(tf.keras.layers.GlobalAveragePooling2D(data_format=data_format, name='ap'))
    # model.add(tf.keras.layers.Dense(num_classes, activation=None, use_bias=True, name='fc1'))
    # model.add(tf.keras.layers.Softmax(name='softmax'))
    #
    # return model



# def print_ResNet(model):
#     for layer in model.layers:
#         print(layer.name)
#         if 'block' in layer.name:
#             for internal_layer in layer.layers['before']:
#                 print('\t'+internal_layer.name)
#
#             print('\tPath0:', end=' ')
#             for internal_layer in layer.layers['path0']:
#                 print(internal_layer.name, end=' ')
#             print()
#
#             print('\tPath1:', end=' ')
#             for internal_layer in layer.layers['path1']:
#                 print(internal_layer.name, end=' ')
#             print()
#
#             for internal_layer in layer.layers['after']:
#                 print('\t'+internal_layer.name)



# if __name__ == "__main__":
#     # ResNet20 = ResNet(input_shape=(32, 32, 3), num_classes=10, bottleneck=False, num_filters_at_start=16,
#     #                   kernel_size=3, initial_conv_strides=1, first_pool_size=None, first_pool_strides=None,
#     #                   num_residual_blocks=[3, 3, 3], first_block_strides=[1, 2, 2], project_first_residual=True,
#     #                   version='V1', data_format='channels_last')
#     #
#     # print(ResNet20.summary())
#
#     ResNet50 = ResNet(input_shape=(224, 224, 3), num_classes=1000, bottleneck=True, num_filters_at_start=64,
#                       initial_kernel_size=7, initial_conv_strides=2, initial_pool_size=3, initial_pool_strides=2,
#                       num_residual_blocks_per_stage=[3, 4, 6, 3], first_block_strides_per_stage=[1, 2, 2, 2],
#                       kernel_size=3, project_first_residual=True, version='V1', data_format='channels_last',
#                       compression_config=None)
#
#     print(ResNet50.get_model().summary())