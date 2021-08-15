import numpy as np
import tensorflow as tf

from src.pcns.layer_helpers import Conv2DPCALayer, DensePCALayer
from src.pcns.compression_helpers import tf_pca, tf_kill_outputs, tf_transform_conv_weights, \
                                                          tf_transform_dense_weights, decode_cc

from src.pcns.layer_helpers import constant_initializer_from_tensor as cift



def get_compression_layer(layer_input_shape, new_shape, compression_layer_type):
    ph = layer_input_shape[0] // new_shape[0]
    pw = layer_input_shape[1] // new_shape[1]

    if compression_layer_type == 'MaxPooling':
        extra_layer = tf.keras.layers.MaxPooling2D((ph, pw))

    elif compression_layer_type == 'AveragePooling':
        extra_layer = tf.keras.layers.AveragePooling2D((ph, pw))

    elif compression_layer_type == 'RandomCrop':
        class RandomCropLayer(tf.keras.layers.Layer):
            def __init__(self, shape, **kwargs):
                self.shape = shape

                super(RandomCropLayer, self).__init__(**kwargs)

            def build(self, input_shape):
                super(RandomCropLayer, self).build(input_shape)

            def call(self, inputs):
                inputs = tf.map_fn(lambda image: tf.image.random_crop(image, self.shape), inputs, back_prop=False,
                                   parallel_iterations=None, swap_memory=False)
                return inputs

        extra_layer = RandomCropLayer(new_shape)

    elif compression_layer_type == 'CenterCrop':
        class CenterCropLayer(tf.keras.layers.Layer):
            def __init__(self, full_shape, center_shape, **kwargs):
                amount_to_be_cropped_h = (full_shape[0] - center_shape[0])
                self.crop_top = amount_to_be_cropped_h // 2
                amount_to_be_cropped_w = (full_shape[1] - center_shape[1])
                self.crop_left = amount_to_be_cropped_w // 2
                self.shape = center_shape

                super(CenterCropLayer, self).__init__(**kwargs)

            def build(self, input_shape):
                super(CenterCropLayer, self).build(input_shape)

            def call(self, inputs):
                return tf.slice(inputs, [0, self.crop_top, self.crop_left, 0], [-1, self.shape[0], self.shape[1], -1])

        extra_layer = CenterCropLayer(layer_input_shape, new_shape)

    elif compression_layer_type == 'EvenlySpacedPixels':
        class ExtractPatchesLayer(tf.keras.layers.Layer):
            def __init__(self, sizes, strides, rates, padding, shape, **kwargs):
                self.sizes = sizes
                self.strides = strides
                self.rates = rates
                self.padding = padding
                self.shape = shape

                super(ExtractPatchesLayer, self).__init__(**kwargs)

            def build(self, input_shape):
                super(ExtractPatchesLayer, self).build(input_shape)

            def call(self, inputs):
                inputs = tf.image.extract_patches(inputs, self.sizes, self.strides, self.rates, self.padding)
                return tf.reshape(inputs, (-1, self.shape[0], self.shape[1], self.shape[2]))

        extra_layer = ExtractPatchesLayer(sizes=[1, new_shape[0], new_shape[1], 1],
                                          strides=[1, layer_input_shape[0], layer_input_shape[1], 1],
                                          rates=[1, ph, pw, 1], padding='VALID', shape=new_shape)

    else:
        raise NotImplementedError()

    return extra_layer



def predict(overall_model, dataset, strategy, layer_index, extra_layer):
    # TODO: fine to reuse strategy?
    # create the model inside the scope to ensure that any variables are mirrored
    with strategy.scope():
        intermediate_model = tf.keras.Sequential(layers=overall_model.layers[:layer_index])
        if extra_layer is not None:
            intermediate_model.add(extra_layer)

    # distribute the dataset based on the strategy
    dist_dataset = strategy.experimental_distribute_dataset(dataset)

    @tf.function
    def forward_pass(dist_inputs):
        def step_fn(inputs):
            output = intermediate_model(inputs)
            return output

        outputs = strategy.experimental_run_v2(step_fn, args=(dist_inputs,))
        local_tensors = strategy.experimental_local_results(outputs)
        return tf.concat(local_tensors, axis=0)

    batch_tensors = []
    with strategy.scope():
        print("Beginning forward pass")
        for i, (batch_x, batch_y) in enumerate(dist_dataset):
            temp = forward_pass(batch_x)
            batch_tensors.append(temp)
        print("Done with forward pass")

    total = tf.concat(batch_tensors, axis=0)
    return total



def get_layer_input_possibly_compressed(layer_index, dataset, overall_model, var_config, strategy):
    layer_input_shape = overall_model.layers[layer_index-1].output_shape[1:]

    image_size = 1
    for val in layer_input_shape:
        image_size *= val

    new_shape = layer_input_shape
    while image_size > var_config['max_image_size']:
        if len(new_shape) == 3:
            # image is still an 'image'
            new_shape = (new_shape[0]//2, new_shape[1]//2, new_shape[2])
        else:
            # TODO: do we need to compress inputs to FC layers?
            break
        image_size = 1
        for val in new_shape:
            image_size *= val

    print("\tlayer input shape {}, compressed input shape {}".format(layer_input_shape, new_shape))

    if new_shape != layer_input_shape:
        print("\tadding {} compression layer to reduce input shape".format(var_config['compression_layer_type']))
        extra_layer = get_compression_layer(layer_input_shape, new_shape, var_config['compression_layer_type'])
    else:
        extra_layer = None

    output = predict(overall_model, dataset, strategy, layer_index, extra_layer)

    return output, new_shape != layer_input_shape



#@tf.function
def compute_all_muVs(compression_config, var_config, dataset, overall_model, strategy):
    all_layers = []
    pca_layers_input = {}
    for i, l in enumerate(overall_model.layers):  # note: the input layer is not included in overall_model.layers
        if ('conv' in l.name and 'conv1' != l.name) or 'fc' in l.name or 'output' in l.name:
            pca_layers_input[l.name] = i
        all_layers.append(l.name)

    # we want to do PCA on the input to all layers in pca_layers_input
    muVes = []

    input_arr = None
    prev_layer_index = None
    for l_name, layer_index in pca_layers_input.items():
        print()
        print("LAYER: {}, overall layer 0 based index: {}".format(l_name, layer_index))

        if compression_config[l_name][0] is None: # NOTE this isn't general, can't skip once input_arr is not None
            print("\tskipping PCA for this layer")
            mu, V, e = None, None, None
            muVes.append((mu, V, e))
            continue

        if input_arr is None or var_config['compute_all_from_scratch'] is True:
            # we have to start from the beginning
            layer_input, did_compress = get_layer_input_possibly_compressed(layer_index, dataset, overall_model,
                                                                            var_config, strategy)

            if not did_compress and var_config['compute_all_from_scratch'] is False:
                input_arr = layer_input
                prev_layer_index = layer_index
        else:
            # we can start with input_arr (input of prev pca layer) and just call until we get to this layer
            while prev_layer_index < layer_index:
                print("\tpassing through layer {}".format(all_layers[prev_layer_index]))
                input_arr = overall_model.get_layer(all_layers[prev_layer_index])(input_arr)
                prev_layer_index += 1

            layer_input = input_arr

        num, ut = decode_cc(compression_config[l_name][0])
        conv = 'conv' in l_name
        mu, V, e = tf_pca(layer_input, num, num_as_threshold=ut, conv=conv, verbose=True, prefix=' {}'.format(l_name))
        muVes.append((mu, V, e))

    return muVes



def imagenet_vgg_compression(compression_config, var_config, overall_model, muVes, strategy, optimizer, verbose=True):
    """

    :param compression_config:
    :param var_config:
    :param overall_model:
    :param muVes:
    :param strategy:
    :param optimizer:
    :param verbose:
    :return:
    """
    cc = compression_config


    # Get information about the layers in the original network
    all_layers = []
    n_full = {}
    last_conv = ""
    for l in overall_model.layers:  # note: the input layer is not included in overall_model.layers
        if 'conv' in l.name or 'fc' in l.name or 'output' in l.name:
            n_full[l.name] = l.output_shape[-1]
        if 'conv' in l.name:
            last_conv = l.name
        all_layers.append(l.name)


    # Verify necessary config constraints
    cc_keys = list(cc.keys())
    cc_values = list(cc.values())

    assert cc_keys == list(n_full.keys()), 'compression config must have keys for all compute layers (conv, fc, output)'
    assert cc_values[0][0] is None, 'currently we are not considering compression of the network input'
    for index in range(len(cc_keys)):
        if cc_keys[index] in ['output']:
            continue
        if cc_values[index][1] is not None:
            assert cc_values[index+1][0] is not None, 'to kill outputs the next layer must perform PCA compression'
    assert cc_values[-1][1] is None, 'can not kill columns in the output layer'

    assert len(cc_keys)-1 == len(muVes), 'PCA for each compute layer (except conv1) should already be computed'


    # Go through the layers and create the (possibly) compressed layers
    # TODO: model creation inside distribute strategy
    with strategy.scope():
        compressed_model = tf.keras.Sequential()
        compressed_model.add(tf.keras.layers.InputLayer(input_shape=overall_model.layers[0].input_shape[1:],
                                                        name='input'))

        mu_c, V_c, W_n = None, None, None

        for index in range(len(cc_keys)):
            prev_layer = cc_keys[index-1] if index-1 >= 0 else None
            curr_layer = cc_keys[index]
            next_layer = cc_keys[index+1] if index+1 <= len(cc_keys)-1 else None

            n_c = n_full[curr_layer]
            W_c = overall_model.get_layer(curr_layer).weights[0] if W_n is None else W_n
            b_c = overall_model.get_layer(curr_layer).weights[1]
            W_n = overall_model.get_layer(next_layer).weights[0] if next_layer is not None else None

            mu_p, V_p = mu_c, V_c # previous layer could have modified mu, V if it killed columns

            conv_c = True if 'conv' in curr_layer else False

            # generic iteration
            if cc[curr_layer][0] is not None and cc[prev_layer][1] is None:
                mu_p, V_p, e_p = muVes[index-1] # want output of previous layer (input to this layer), recall offset
            if cc[curr_layer][1] is not None:
                mu_c, V_c, e_c = muVes[index] # want mu, V output of this layer (input to next layer), recall offset
                num, ut = decode_cc(cc[curr_layer][1])
                n_c, W_c, b_c, mu_c, V_c, W_n = tf_kill_outputs(W_c, b_c, mu_c, V_c, W_n, num, num_as_threshold=ut,
                                                                conv=conv_c, conv_to_dense=curr_layer == last_conv,
                                                                verbose=verbose, prefix=' {}'.format(curr_layer))
            else:
                mu_c, V_c = None, None

            # add layer
            activation = None if curr_layer == 'output' else 'relu'
            if conv_c and cc[curr_layer][0] is not None:
                W_c_p, b_c_p = tf_transform_conv_weights(mu_p, V_p, W_c, b_c)
                compressed_model.add(Conv2DPCALayer(int(n_c), 3, mu_p, V_p, kernel_initializer=cift(W_c_p),
                                                    bias_initializer=cift(b_c_p), activation='relu', name=curr_layer))
            elif conv_c:
                compressed_model.add(tf.keras.layers.Conv2D(int(n_c), 3, padding='same', kernel_initializer=cift(W_c),
                                                            bias_initializer=cift(b_c), activation='relu',
                                                            name=curr_layer))
            elif cc[curr_layer][0] is not None:
                W_c_p, b_c_p = tf_transform_dense_weights(mu_p, V_p, W_c, b_c)
                compressed_model.add(DensePCALayer(int(n_c), mu_p, V_p, kernel_initializer=cift(W_c_p),
                                                   bias_initializer=cift(tf.squeeze(b_c_p)), activation=activation,
                                                   name=curr_layer))
            else:
                compressed_model.add(tf.keras.layers.Dense(int(n_c), kernel_initializer=cift(W_c),
                                                           bias_initializer=cift(b_c), activation=activation,
                                                           name=curr_layer))

            # add extra layers if necessary
            c_i = all_layers.index(curr_layer)
            n_i = all_layers.index(next_layer) if next_layer is not None else len(all_layers)
            while c_i < n_i - 1:
                c_i += 1
                if 'mp' in all_layers[c_i]:
                    compressed_model.add(tf.keras.layers.MaxPooling2D((2, 2), name=all_layers[c_i]))
                elif 'flatten' in all_layers[c_i]:
                    compressed_model.add(tf.keras.layers.Flatten(name=all_layers[c_i]))
                # TODO: do you want to add dropout back in, does dropout scaling affect compression??
                elif 'dropout' in all_layers[c_i] and var_config['add_dropout_to_compressed_model'] is True:
                    compressed_model.add(tf.keras.layers.Dropout(0.5))
                elif 'softmax' in all_layers[c_i]:
                    compressed_model.add(tf.keras.layers.Softmax(name=all_layers[c_i]))

        # Optimizer
        # TODO: fix this
        compressed_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                                 optimizer=optimizer, metrics=['accuracy'])
        # compressed_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
        #                          optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

    return compressed_model