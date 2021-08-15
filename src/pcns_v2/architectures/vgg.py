import tensorflow as tf

from src.pcns_v2.helpers.compression_helpers import tf_pca, copy_bn_layer, copy_conv_layer, copy_dense_layer, \
    from_conv_layer, from_dense_layer



HE_INIT = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_out', distribution='untruncated_normal')
WD = tf.keras.regularizers.l2(l=0.00005)



def vgg16A(input_shape=(32, 32, 3), output_shape=10, conv_kwargs=None, dense_kwargs=None):
    if conv_kwargs is None:
        conv_kwargs = {'padding': 'same', 'activation': None, 'use_bias': False,
                       'kernel_initializer': HE_INIT, 'kernel_regularizer': WD}
    if dense_kwargs is None:
        dense_kwargs = {'activation': None, 'kernel_regularizer': WD, 'bias_regularizer': WD}

    full_model = tf.keras.Sequential()
    full_model.add(tf.keras.layers.InputLayer(input_shape=input_shape, name='input'))

    # Convolutions
    num_filters = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]

    conv_count = 1
    mp_count = 1
    for n in num_filters:
        if n == 'M':
            full_model.add(tf.keras.layers.MaxPooling2D((2, 2), name='mp{}'.format(mp_count)))
            mp_count += 1
        else:
            full_model.add(tf.keras.layers.Conv2D(n, (3, 3), name='conv{}'.format(conv_count), **conv_kwargs))
            full_model.add(tf.keras.layers.BatchNormalization(name='bn{}'.format(conv_count)))
            full_model.add(tf.keras.layers.ReLU(name='relu{}'.format(conv_count)))
            conv_count += 1
    full_model.add(tf.keras.layers.GlobalAveragePooling2D(name='ap'))

    # Output
    full_model.add(tf.keras.layers.Dense(output_shape, name='output', **dense_kwargs))
    # full_model.add(tf.keras.layers.Softmax(name='softmax'))

    # Compression config
    cc = {}
    for layer in full_model.layers:
        if 'conv' in layer.name or 'output' in layer.name:
            cc[layer.name] = (None, None)

    return full_model, cc










def verify_cc(cc, full_model):
    # Get information about the layers in the original network
    compute_layers = []
    for l in full_model.layers: # note: the input layer is not included in overall_model.layers
        if 'conv' in l.name or 'fc' in l.name or 'output' in l.name:
            compute_layers.append(l.name)

    # Verify necessary config constraints
    cc_keys = list(cc.keys())
    cc_values = list(cc.values())

    assert cc_keys == compute_layers, 'compression config must have keys for all compute layers (conv, fc, output)'
    assert cc_values[0][0] is None, 'currently we are not considering compression of the network input'
    for index in range(len(cc_keys)):
        if cc_keys[index] in ['output']:
            continue
        if cc_values[index][1] is not None:
            assert cc_values[index+1][0] is not None, 'to kill outputs the next layer must perform PCA compression'
    assert cc_values[-1][1] is None, 'can not kill columns in the output layer'



def compute_activation_bases_cifar(cc, input_arr, full_model, forget_bottom=True, verbose=True):
    verify_cc(cc, full_model)

    cc_keys = list(cc.keys())
    bases = {}
    for l in full_model.layers:
        if l.name in cc_keys:
            if cc[l.name][0] is None:
                mu, Vt, Et, Vb, Eb = None, None, None, None, None
            else:
                mu, Vt, Et, Vb, Eb = tf_pca(input_arr, cc[l.name][0], centering=True, conv='conv' in l.name,
                                            verbose=verbose, prefix=l.name)
            if forget_bottom:
                Vb, Eb = None, None

            bases[l.name] = (mu, Vt, Et, Vb, Eb)

        input_arr = l(input_arr)

    return bases



def forward_transform(bases, full_model, include_offset=False, train_top_basis='NO', verbose=True):
    new_model = tf.keras.Sequential()
    new_model.add(tf.keras.layers.InputLayer(input_shape=full_model.layers[0].input_shape[1:], name='input'))

    for l in full_model.layers:
        if 'mp' in l.name:
            new_model.add(tf.keras.layers.MaxPooling2D((2, 2), name=l.name))
        elif 'ap' in l.name:
            new_model.add(tf.keras.layers.GlobalAveragePooling2D(name=l.name))
        elif 'bn' in l.name:
            new_model.add(copy_bn_layer(l))
        elif 'relu' in l.name:
            new_model.add(tf.keras.layers.ReLU(name=l.name))
        elif 'flatten' in l.name:
            new_model.add(tf.keras.layers.Flatten(name=l.name))
        elif 'softmax' in l.name:
            new_model.add(tf.keras.layers.Softmax(name=l.name))
        # TODO: add dropout layer here

        else:
            # Compute layer
            if bases[l.name] == (None, None, None, None, None): # can just do Vt == None, but the whole tuple is ok too
                if 'conv' in l.name:
                    new_model.add(copy_conv_layer(l))
                else:
                    new_model.add(copy_dense_layer(l))
                continue
            # Layer to transform
            mu, Vt, Et, Vb, Eb = bases[l.name]

            if 'conv' in l.name:
                new_model.add(from_conv_layer(l, mu, Vt, Vb, include_offset=include_offset,
                                              train_top_basis=train_top_basis))

            else:
                new_model.add(from_dense_layer(l, mu, Vt, Vb, include_offset=include_offset,
                                               train_top_basis=train_top_basis))

    return new_model