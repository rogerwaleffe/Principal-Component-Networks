import tensorflow as tf

from src.pcns_v2.helpers.layer_helpers import Conv2DTransformLayer, Conv2DExplicitPadding, IdentityLayer
from src.pcns_v2.helpers.compression_helpers import tf_pca, copy_bn_layer, copy_conv_layer, copy_dense_layer, \
    from_conv_layer, from_dense_layer, from_conv_transform_layer, from_dense_transform_layer



HE_INIT = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_out', distribution='untruncated_normal')
WD = tf.keras.regularizers.l2(l=0.00005)



# Layers used to make up ResNet
def batch_norm(data_format='channels_last', **kwargs):
    defaults = {'momentum': 0.997, 'epsilon': 1e-5, 'gamma_regularizer': None, 'beta_regularizer': None}
    defaults.update(kwargs)
    return tf.keras.layers.BatchNormalization(axis=1 if data_format == 'channels_first' else 3, **defaults)



def conv2d(filters, kernel_size, strides=1, data_format='channels_last', **kwargs):
    defaults = {'use_bias': False, 'kernel_initializer': HE_INIT, 'kernel_regularizer': WD} # correct implementation
    # defaults = {'use_bias': True, 'kernel_initializer': HE_INIT, 'kernel_regularizer': WD, 'bias_regularizer': WD}
    defaults.update(kwargs)
    return Conv2DExplicitPadding(filters, kernel_size, strides=strides, data_format=data_format, **defaults)



def dense(units, **kwargs):
    defaults = {'activation': None, 'use_bias': True, 'kernel_regularizer': WD, 'bias_regularizer': WD}
    defaults.update(kwargs)
    return tf.keras.layers.Dense(units, **defaults)



"""
Create normal ResNet blocks
"""
def create_ResNetBlockV1(filters, kernel_size, strides=1, data_format='channels_last', conv0=None, **kwargs):
    bn0 = batch_norm(data_format=data_format, name='bn0') if conv0 is not None else None
    bn1 = batch_norm(data_format=data_format, name='bn1')
    bn2 = batch_norm(data_format=data_format, name='bn2')
    conv1 = conv2d(filters, kernel_size, strides=strides, data_format=data_format, name='conv1')
    conv2 = conv2d(filters, kernel_size, strides=1, data_format=data_format, name='conv2')

    return ResNetBlockV1(conv0, bn0, conv1, bn1, conv2, bn2, **kwargs)



def create_ResNetBlockV2(filters, kernel_size, strides=1, data_format='channels_last', conv0=None, **kwargs):
    bn1 = batch_norm(data_format=data_format, name='bn1')
    bn2 = batch_norm(data_format=data_format, name='bn2')
    conv1 = conv2d(filters, kernel_size=kernel_size, strides=strides, data_format=data_format, name='conv1')
    conv2 = conv2d(filters, kernel_size=kernel_size, strides=1, data_format=data_format, name='conv2')

    return ResNetBlockV2(conv0, conv1, bn1, conv2, bn2, **kwargs)



"""
Normal ResNet blocks
"""
class ResNetBlock(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ResNetBlock, self).__init__(**kwargs)

        self.cc = None
        self.cc_params = None

        # helper variables for printing etc.
        self.layers = {'before': [], 'path0': [], 'path1': [], 'after': []}
        self.all_layers = []
        self.all_layer_names = []

    def build(self, input_shape):
        super(ResNetBlock, self).build(input_shape)

        all_layers = []
        for key in list(self.layers.keys()):
            all_layers.extend(self.layers[key])
        self.all_layers = all_layers
        self.all_layer_names = [l.name for l in all_layers]

    def get_config(self):
        config = super(ResNetBlock, self).get_config()

        return config

    def get_layer(self, name):
        index = self.all_layer_names.index(name)
        return self.all_layers[index]

    def compute_initial_pca(self, inputs, version, forget_bottom=True, pca_centering=True, verbose=True):
        if version == 'V2':
            inputs = self.bn1(inputs)
            inputs = self.relu1(inputs)

        mu, Vt, Et, Vb, Eb = None, None, None, None, None
        if self.cc[0] is not None:
            mu, Vt, Et, Vb, Eb = tf_pca(inputs, self.cc[0], centering=pca_centering, conv=True, verbose=verbose,
                                        prefix='({} conv1)'.format(self.name))
        if forget_bottom:
            Vb, Eb = None, None

        return inputs, mu, Vt, Et, Vb, Eb

    def set_cc(self, cc):
        self.cc = cc

        W_c0_initial = self.conv0 if self.conv0 is not None else None
        self.cc_params = {'mu_1': None, 'Vt_1': None, 'Et_1': None, 'Vb_1': None, 'Eb_1': None,
                          'mu_2': None, 'Vt_2': None, 'Et_2': None, 'Vb_2': None, 'Eb_2': None,
                          'W_0': (None, W_c0_initial), 'W_1': (None, self.conv1), 'W_2': (None, self.conv2),
                          'in_indices': None, 'out_indices': None}

    def set_cc_param(self, names, values):
        for n, v in zip(names, values):
            if 'W_' in n:
                self.cc_params[n] = (v, None)
            else:
                self.cc_params[n] = v

    def get_cc_param(self, names):
        values = []
        for n in names:
            if 'W_' in n:
                val = self.cc_params[n][0] if self.cc_params[n][0] is not None else self.cc_params[n][1].weights[0]
            else:
                val = self.cc_params[n]

            if len(names) == 1:
                return val
            values.append(val)
        return values

    def forward_transform(self, version, include_offset=False, train_top_basis='NO', add_bias_if_nec=True,
                          verbose=True):
        def get_new_layer(old_layer, mu, Vt, Vb):
            if Vt is not None:
                new_layer = from_conv_layer(old_layer, mu, Vt, Vb, include_offset=include_offset,
                                            train_top_basis=train_top_basis, add_bias_if_nec=add_bias_if_nec)
            else:
                new_layer = copy_conv_layer(old_layer)
            return new_layer

        mu_1, Vt_1, Vb_1 = self.get_cc_param(['mu_1', 'Vt_1', 'Vb_1'])

        # conv0
        if self.conv0 is not None:
            conv0 = get_new_layer(self.conv0, mu_1, Vt_1, Vb_1)
        else:
            conv0 = None

        # conv1
        conv1 = get_new_layer(self.conv1, mu_1, Vt_1, Vb_1)

        mu_2, Vt_2, Vb_2 = self.get_cc_param(['mu_2', 'Vt_2', 'Vb_2'])

        # conv2
        conv2 = get_new_layer(self.conv2, mu_2, Vt_2, Vb_2)

        if version == 'V2':
            bn1 = copy_bn_layer(self.bn1, None)
            bn2 = copy_bn_layer(self.bn2, None)
            new_block = ResNetBlockV2(conv0, conv1, bn1, conv2, bn2, name=self.name)
        else:
            bn0 = copy_bn_layer(self.bn0, None) if self.conv0 is not None else None
            bn1 = copy_bn_layer(self.bn1, None)
            bn2 = copy_bn_layer(self.bn2, None)
            new_block = ResNetBlockV1(conv0, bn0, conv1, bn1, conv2, bn2, name=self.name)

        return new_block

    def undo_forward_transform(self, version):
        def get_new_layer(l):
            return from_conv_transform_layer(l) if isinstance(l, Conv2DTransformLayer) else copy_conv_layer(l)

        # conv0
        if self.conv0 is not None:
            conv0 = get_new_layer(self.conv0)
        else:
            conv0 = None

        # conv1/conv2
        conv1 = get_new_layer(self.conv1)
        conv2 = get_new_layer(self.conv2)

        if version == 'V2':
            bn1 = copy_bn_layer(self.bn1, None)
            bn2 = copy_bn_layer(self.bn2, None)
            new_block = ResNetBlockV2(conv0, conv1, bn1, conv2, bn2, name=self.name)
        else:
            bn0 = copy_bn_layer(self.bn0, None) if self.conv0 is not None else None
            bn1 = copy_bn_layer(self.bn1, None)
            bn2 = copy_bn_layer(self.bn2, None)
            new_block = ResNetBlockV1(conv0, bn0, conv1, bn1, conv2, bn2, name=self.name)

        return new_block



class ResNetBlockV1(ResNetBlock):
    def __init__(self, conv0, bn0, conv1, bn1, conv2, bn2, **kwargs):
        super(ResNetBlockV1, self).__init__(**kwargs)

        if not isinstance(conv1, (Conv2DTransformLayer, Conv2DExplicitPadding, tf.keras.layers.Conv2D)):
            custom_objects = {'Conv2DTransformLayer': Conv2DTransformLayer,
                              'Conv2DExplicitPadding': Conv2DExplicitPadding, 'IdentityLayer': IdentityLayer}
            conv0 = None if conv0 is None else tf.keras.layers.deserialize(conv0, custom_objects=custom_objects)
            bn0 = None if bn0 is None else tf.keras.layers.deserialize(bn0, custom_objects=custom_objects)
            conv1 = tf.keras.layers.deserialize(conv1, custom_objects=custom_objects)
            bn1 = tf.keras.layers.deserialize(bn1, custom_objects=custom_objects)
            conv2 = tf.keras.layers.deserialize(conv2, custom_objects=custom_objects)
            bn2 = tf.keras.layers.deserialize(bn2, custom_objects=custom_objects)

        self.conv0 = conv0
        self.bn0 = bn0
        self.conv1 = conv1
        self.bn1 = bn1
        self.conv2 = conv2
        self.bn2 = bn2

        self.relu1 = tf.keras.layers.ReLU(name='relu1')
        self.relu2 = tf.keras.layers.ReLU(name='relu2')

    def build(self, input_shape):
        if self.conv0 is not None:
            self.layers['path0'].extend([self.conv0, self.bn0])

        self.layers['path1'].extend([self.conv1, self.bn1, self.relu1, self.conv2, self.bn2])
        self.layers['after'].extend([self.relu2])

        super(ResNetBlockV1, self).build(input_shape)

    def call(self, inputs, training=False, progress_input_arr=False, forget_bottom=True, pca_centering=True,
             verbose=True):
        shortcut = inputs

        if self.conv0 is not None:
            shortcut = self.conv0(inputs)
            shortcut = self.bn0(shortcut, training=training)

        inputs = self.conv1(inputs)
        inputs = self.bn1(inputs, training=training)
        inputs = self.relu1(inputs)

        # Compute intermediate PCA if desired
        mu, Vt, Et, Vb, Eb = None, None, None, None, None
        if progress_input_arr is True:
            if self.cc[2] is not None:
                mu, Vt, Et, Vb, Eb = tf_pca(inputs, self.cc[2], centering=pca_centering, conv=True, verbose=verbose,
                                            prefix='({} conv2)'.format(self.name))
            if forget_bottom:
                Vb, Eb = None, None

        inputs = self.conv2(inputs)
        inputs = self.bn2(inputs, training=training)

        inputs = inputs + shortcut
        inputs = self.relu2(inputs)

        if progress_input_arr is True:
            return inputs, mu, Vt, Et, Vb, Eb
        else:
            return inputs

    def get_config(self):
        config = super(ResNetBlockV1, self).get_config()
        config.update({
            'conv0': None if self.conv0 is None else tf.keras.layers.serialize(self.conv0),
            'bn0': None if self.bn0 is None else tf.keras.layers.serialize(self.bn0),
            'conv1': tf.keras.layers.serialize(self.conv1),
            'bn1': tf.keras.layers.serialize(self.bn1),
            'conv2': tf.keras.layers.serialize(self.conv2),
            'bn2': tf.keras.layers.serialize(self.bn2)
        })

        return config



class ResNetBlockV2(ResNetBlock):
    def __init__(self, conv0, conv1, bn1, conv2, bn2, **kwargs):
        super(ResNetBlockV2, self).__init__(**kwargs)

        if not isinstance(conv1, (Conv2DTransformLayer, Conv2DExplicitPadding, tf.keras.layers.Conv2D)):
            custom_objects = {'Conv2DTransformLayer': Conv2DTransformLayer,
                              'Conv2DExplicitPadding': Conv2DExplicitPadding, 'IdentityLayer': IdentityLayer}
            conv0 = None if conv0 is None else tf.keras.layers.deserialize(conv0, custom_objects=custom_objects)
            conv1 = tf.keras.layers.deserialize(conv1, custom_objects=custom_objects)
            bn1 = tf.keras.layers.deserialize(bn1, custom_objects=custom_objects)
            conv2 = tf.keras.layers.deserialize(conv2, custom_objects=custom_objects)
            bn2 = tf.keras.layers.deserialize(bn2, custom_objects=custom_objects)

        self.conv0 = conv0
        self.conv1 = conv1
        self.bn1 = bn1
        self.conv2 = conv2
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

    def call(self, inputs, training=False, progress_input_arr=False, forget_bottom=True, pca_centering=True,
             verbose=True):
        shortcut = inputs

        inputs = self.bn1(inputs, training=training)
        inputs = self.relu1(inputs)

        if self.conv0 is not None:
            # From [2]: "when pre-activation is used, these projection shortcuts are also with pre-activation"
            shortcut = self.conv0(inputs)

        inputs = self.conv1(inputs)
        inputs = self.bn2(inputs, training=training)
        inputs = self.relu2(inputs)

        # Compute intermediate PCA if desired
        mu, Vt, Et, Vb, Eb = None, None, None, None, None
        if progress_input_arr is True:
            if self.cc[2] is not None:
                mu, Vt, Et, Vb, Eb = tf_pca(inputs, self.cc[2], centering=pca_centering, conv=True, verbose=verbose,
                                            prefix='({} conv2)'.format(self.name))
            if forget_bottom:
                Vb, Eb = None, None

        inputs = self.conv2(inputs)

        inputs = shortcut + inputs

        if progress_input_arr is True:
            return inputs, mu, Vt, Et, Vb, Eb
        else:
            return inputs

    def get_config(self):
        config = super(ResNetBlockV2, self).get_config()
        config.update({
            'conv0': None if self.conv0 is None else tf.keras.layers.serialize(self.conv0),
            'conv1': tf.keras.layers.serialize(self.conv1),
            'bn1': tf.keras.layers.serialize(self.bn1),
            'conv2': tf.keras.layers.serialize(self.conv2),
            'bn2': tf.keras.layers.serialize(self.bn2)
        })

        return config



"""
ResNet class
"""
class ResNet:
    def __init__(self, input_shape, num_classes, bottleneck, num_filters_at_start, initial_kernel_size,
                 initial_conv_strides, initial_pool_size, initial_pool_strides, num_residual_blocks_per_stage,
                 first_block_strides_per_stage, kernel_size, project_first_residual=True, version='V1',
                 data_format='channels_last'):

        assert bottleneck in [True, False], 'bottleneck should be boolean in [True, False]'
        assert version in ['V1', 'V15', 'V2'], 'version should be in [V1, V15, V2]'
        assert data_format in ['channels_first', 'channels_last'], 'data_format in [channels_first, channels_last]'
        if bottleneck is False:
            assert version != 'V15', 'version 1.5 only applicable to bottleneck layers'

        if version == 'V1' and not bottleneck:
            self.block = create_ResNetBlockV1
        elif version == 'V2' and not bottleneck:
            self.block = create_ResNetBlockV2
        # elif version == 'V1':
        #     self.block = create_ResNetBottleneckBlockV1
        # elif version == 'V15':
        #     self.block = create_ResNetBottleneckBlockV1_5
        # elif version == 'V2':
        #     self.block = create_ResNetBottleneckBlockV2
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

    def get_model(self):
        N = None
        cc = {}
        # Create the model
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=self.input_shape, name='input'))

        if self.data_format == 'channels_first':
            # Convert the inputs from channels_last (NHWC) to channels_first (NCHW) (provides performance boost on GPU?)
            model.add(tf.keras.layers.Permute((3, 1, 2), name='permute'))

        # Initial conv layer
        model.add(conv2d(self.num_filters_at_start, self.initial_kernel_size, strides=self.initial_conv_strides,
                         data_format=self.data_format, name='initial_conv'))
        cc['initial_conv'] = (N, N)

        # We do not include batch normalization or activation functions in V2 for the initial conv because the first
        # ResNet unit will perform these for both the shortcut and non-shortcut paths as part of the block's projection.
        if self.version == 'V1':
            model.add(batch_norm(data_format=self.data_format, name='initial_bn'))
            model.add(tf.keras.layers.ReLU(name='initial_relu'))

        # Optional pooling after initial conv
        if self.initial_pool_size:
            model.add(tf.keras.layers.MaxPooling2D(pool_size=self.initial_pool_size, strides=self.initial_pool_strides,
                                                   padding='same', data_format=self.data_format, name='initial_mp'))

        # Stages of ResNet blocks
        for i, num_blocks in enumerate(self.num_residual_blocks_per_stage):
            filters = self.num_filters_at_start * (2 ** i)
            filters_out = filters * 4 if self.bottleneck else filters

            if i == 0 and self.project_first_residual is False:
                conv0 = IdentityLayer(name='identity')
            else:
                conv0 = conv2d(filters_out, 1, strides=self.first_block_strides_per_stage[i],
                               data_format=self.data_format, name='conv0')

            model.add(self.block(filters, self.kernel_size, strides=self.first_block_strides_per_stage[i],
                                 data_format=self.data_format, conv0=conv0, name='stage{}_block{}'.format(i+1, 1)))
            cc['stage{}_block{}'.format(i+1, 1)] = (N, N, N) if self.bottleneck is False else (N, N, N, N, N)

            for j in range(1, num_blocks):
                model.add(self.block(filters, self.kernel_size, strides=1, data_format=self.data_format, conv0=None,
                                     name='stage{}_block{}'.format(i+1, j+1)))
                cc['stage{}_block{}'.format(i+1, j+1)] = (N, N, N) if self.bottleneck is False else (N, N, N, N, N)
            cc['stage{}'.format(i+1)] = N

        # From [2]: for the last Residual Unit (...), we adopt an extra activation right after its element-wise addition
        if self.version == 'V2':
            model.add(batch_norm(data_format=self.data_format, name='final_bn'))
            model.add(tf.keras.layers.ReLU(name='final_relu'))

        # Final layers
        model.add(tf.keras.layers.GlobalAveragePooling2D(data_format=self.data_format, name='ap'))
        model.add(dense(self.num_classes, name='fc1'))
        cc['fc1'] = (N, N)

        # model.add(tf.keras.layers.Softmax(name='softmax'))

        return model, cc

    def compute_activation_bases_cifar(self, cc, input_arr, full_model, forget_bottom=True, pca_centering=True,
                                       verbose=True):
        # TODO: handle: bottleneck=True, channels_first implementation, no projection at beginning
        bases = {}

        # Progress through initial layers
        input_arr = full_model.get_layer('initial_conv')(input_arr)
        if self.version == 'V1':
            input_arr = full_model.get_layer('initial_bn')(input_arr)
            input_arr = full_model.get_layer('initial_relu')(input_arr)

        if self.initial_pool_size:
            input_arr = full_model.get_layer('initial_mp')(input_arr)

        # ResNet blocks
        for i, num_blocks in enumerate(self.num_residual_blocks_per_stage):
            for j in range(0, num_blocks):
                l_name = 'stage{}_block{}'.format(i+1, j+1)
                block_in_stage = full_model.get_layer(l_name)
                block_in_stage.set_cc(cc[l_name])

                bases[l_name] = {}
                _, mu, Vt, Et, Vb, Eb = block_in_stage.compute_initial_pca(input_arr, self.version,
                                                                           forget_bottom=forget_bottom,
                                                                           pca_centering=pca_centering, verbose=verbose)
                bases[l_name]['conv1'] = (mu, Vt, Et, Vb, Eb)
                input_arr, mu, Vt, Et, Vb, Eb = block_in_stage.call(input_arr, training=False, progress_input_arr=True,
                                                                    forget_bottom=forget_bottom,
                                                                    pca_centering=pca_centering, verbose=verbose)
                bases[l_name]['conv2'] = (mu, Vt, Et, Vb, Eb)

        # Progress through final layers
        if self.version == 'V2':
            input_arr = full_model.get_layer('final_bn')(input_arr)
            input_arr = full_model.get_layer('final_relu')(input_arr)

        mu, Vt, Et, Vb, Eb = None, None, None, None, None
        if cc['fc1'][0] is not None:
            input_arr = full_model.get_layer('ap')(input_arr)
            mu, Vt, Et, Vb, Eb = tf_pca(input_arr, cc['fc1'][0], centering=pca_centering, conv=False, verbose=verbose,
                                        prefix='(fc1)')
            if forget_bottom:
                Vb, Eb = None, None
        bases['fc1'] = (mu, Vt, Et, Vb, Eb)

        return bases

    def forward_transform(self, bases, full_model, include_offset=False, train_top_basis='NO', add_bias_if_nec=True,
                          verbose=True):
        # TODO: handle: bottleneck=True, etc.

        # Create the model
        new_model = tf.keras.Sequential()
        new_model.add(tf.keras.layers.InputLayer(input_shape=self.input_shape, name='input'))

        # Initial conv layer
        new_model.add(copy_conv_layer(full_model.get_layer('initial_conv')))
        self.add_other_initial_layers(new_model, full_model, None)

        # Stages of ResNet blocks
        for i, num_blocks in enumerate(self.num_residual_blocks_per_stage):
            for j in range(0, num_blocks):
                l_name = 'stage{}_block{}'.format(i + 1, j + 1)
                block_in_stage = full_model.get_layer(l_name)

                mu, Vt, Et, Vb, Eb = bases[l_name]['conv1']
                block_in_stage.set_cc_param(['mu_1', 'Vt_1', 'Vb_1'], [mu, Vt, Vb])
                mu, Vt, Et, Vb, Eb = bases[l_name]['conv2']
                block_in_stage.set_cc_param(['mu_2', 'Vt_2', 'Vb_2'], [mu, Vt, Vb])

                new_block = block_in_stage.forward_transform(self.version, include_offset=include_offset,
                                                             train_top_basis=train_top_basis,
                                                             add_bias_if_nec=add_bias_if_nec, verbose=verbose)
                new_model.add(new_block)

        # Final layers
        if self.version == 'V2':
            new_model.add(copy_bn_layer(full_model.get_layer('final_bn'), None))
            new_model.add(tf.keras.layers.ReLU(name='final_relu'))

        new_model.add(tf.keras.layers.GlobalAveragePooling2D(data_format=self.data_format, name='ap'))

        dense_layer = full_model.get_layer('fc1')
        mu, Vt, Et, Vb, Eb = bases['fc1']
        if Vt is not None:
            new_model.add(from_dense_layer(dense_layer, mu, Vt, Vb, include_offset=include_offset,
                                           train_top_basis=train_top_basis, add_bias_if_nec=add_bias_if_nec))
        else:
            new_model.add(copy_dense_layer(dense_layer))

        return new_model

    def undo_forward_transform(self, c_model):
        # Create the model
        full_model = tf.keras.Sequential()
        full_model.add(tf.keras.layers.InputLayer(input_shape=self.input_shape, name='input'))

        # Initial conv layer
        full_model.add(copy_conv_layer(c_model.get_layer('initial_conv')))
        self.add_other_initial_layers(full_model, c_model, None)

        # Stages of ResNet blocks
        for i, num_blocks in enumerate(self.num_residual_blocks_per_stage):
            for j in range(0, num_blocks):
                l_name = 'stage{}_block{}'.format(i + 1, j + 1)
                block_in_stage = c_model.get_layer(l_name)

                full_block = block_in_stage.undo_forward_transform(self.version)
                full_model.add(full_block)

        # Final layers
        if self.version == 'V2':
            full_model.add(copy_bn_layer(c_model.get_layer('final_bn'), None))
            full_model.add(tf.keras.layers.ReLU(name='final_relu'))

        full_model.add(tf.keras.layers.GlobalAveragePooling2D(data_format=self.data_format, name='ap'))

        dense_layer = c_model.get_layer('fc1')
        if isinstance(dense_layer, tf.keras.layers.Dense):
            full_model.add(copy_dense_layer(dense_layer))
        else:
            full_model.add(from_dense_transform_layer(dense_layer))

        return full_model

    def add_other_initial_layers(self, nm, m, indices):
        if self.version == 'V1':
            nm.add(copy_bn_layer(m.get_layer('initial_bn'), indices))
            nm.add(tf.keras.layers.ReLU(name='initial_relu'))
        if self.initial_pool_strides:
            nm.add(tf.keras.layers.MaxPooling2D(pool_size=self.initial_pool_size, strides=self.initial_pool_strides,
                                                padding='same', data_format=self.data_format, name='initial_mp'))

    @staticmethod
    def print_ResNet(model):
        print("\nResNet Architecture:")
        for layer in model.layers:
            print(layer.name)
            if 'block' in layer.name:
                print('\tFirst:', end=' ')
                for internal_layer in layer.layers['before']:
                    print(internal_layer.name, end=' ')
                print()
                print('\tPath0:', end=' ')
                for internal_layer in layer.layers['path0']:
                    print(internal_layer.name, end=' ')
                print()
                print('\tPath1:', end=' ')
                for internal_layer in layer.layers['path1']:
                    print(internal_layer.name, end=' ')
                print()
                print('\tAfter:', end=' ')
                for internal_layer in layer.layers['after']:
                    print(internal_layer.name, end=' ')
                print()
        print()
