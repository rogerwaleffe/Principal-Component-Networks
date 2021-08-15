# TODO: this code is a mess
import tensorflow as tf

from src.pcns.imagenet.imagenet_compression import get_compression_layer
from src.pcns.compression_helpers import decode_cc, tf_pca



def predict(intermediate_model, dataset, strategy):
    # # TODO: fine to reuse strategy?
    # # create the model inside the scope to ensure that any variables are mirrored
    # with strategy.scope():
    #     intermediate_model = tf.keras.Sequential(layers=overall_model.layers[:layer_index])
    #     if extra_layer is not None:
    #         intermediate_model.add(extra_layer)

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
        # print("Beginning forward pass")
        for i, (batch_x, batch_y) in enumerate(dist_dataset):
            temp = forward_pass(batch_x)
            batch_tensors.append(temp)
        # print("Done with forward pass")

    total = tf.concat(batch_tensors, axis=0)
    return total



def get_compressed_layer_shape(layer_input_shape, var_config):
    image_size = 1
    for val in layer_input_shape:
        image_size *= val

    new_shape = layer_input_shape
    while image_size > var_config['max_image_size']:
        if len(new_shape) == 3:
            # image is still an 'image'
            new_shape = (new_shape[0] // 2, new_shape[1] // 2, new_shape[2])
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

    return new_shape, extra_layer



def compute_all_muVs(compression_config, var_config, dataset, overall_model, strategy):
    all_layers = []
    pca_layers_input = {} # layer name: overall index
    for i, l in enumerate(overall_model.layers):  # note: the input layer is not included in overall_model.layers
        if 'block' in l.name or 'fc' in l.name:
            pca_layers_input[l.name] = i
        all_layers.append(l.name)

    # we want to do PCA on the input to all layers in pca_layers_input
    muVes = []

    input_arr = None
    # prev_layer_index = None
    for l_name, layer_index in pca_layers_input.items():
        print()
        print("LAYER: {}, overall layer 0 based index: {}".format(l_name, layer_index))

        if 'block' in l_name:
            cc_key = '{}_{}'.format(l_name[5], l_name[-1])
        else:
            cc_key = l_name

        # NOTE this isn't general, can't skip once input_arr is not None
        if compression_config[cc_key] == (None, None, None, None, None):
            print("\tskipping PCA for this layer")
            mu1, V1, e1, mu2, V2, e2, mu3, V3, e3 = None, None, None, None, None, None, None, None, None
            muVes.append((mu1, V1, e1, mu2, V2, e2, mu3, V3, e3))
            continue
        if compression_config[cc_key] == (None, None):
            print("\tskipping PCA for this layer")
            mu, V, e = None, None, None
            muVes.append((mu, V, e))
            continue

        if input_arr is None or var_config['compute_all_from_scratch'] is True:
            # we have to start from the beginning, TODO: bottleneck v2
            if 'block' in l_name:
                # for conv1 (and conv0), input is just output of previous layer (block)
                if compression_config[cc_key][0] is not None:
                    layer_input_shape = overall_model.layers[layer_index-1].output_shape[1:]
                    _, extra_layer = get_compressed_layer_shape(layer_input_shape, var_config)
                    with strategy.scope():
                        intermediate_model = tf.keras.Sequential(layers=overall_model.layers[:layer_index])
                        if extra_layer is not None:
                            intermediate_model.add(extra_layer)

                    layer_input = predict(intermediate_model, dataset, strategy)
                    num, ut = decode_cc(compression_config[cc_key][0])
                    mu1, V1, e1 = tf_pca(layer_input, num, num_as_threshold=ut, conv=True, verbose=True,
                                         prefix=' {}'.format(l_name))
                else:
                    mu1, V1, e1 = None, None, None


                # for conv2, input needs to pass through conv1/bn1/relu1 of this block
                if compression_config[cc_key][2] is not None:
                    layer_input_shape = overall_model.layers[layer_index].relu1.output_shape[1:]
                    _, extra_layer = get_compressed_layer_shape(layer_input_shape, var_config)
                    with strategy.scope():
                        intermediate_model = tf.keras.Sequential(layers=overall_model.layers[:layer_index])
                        intermediate_model.add(overall_model.layers[layer_index].conv1)
                        intermediate_model.add(overall_model.layers[layer_index].bn1)
                        intermediate_model.add(overall_model.layers[layer_index].relu1)
                        if extra_layer is not None:
                            intermediate_model.add(extra_layer)

                    layer_input = predict(intermediate_model, dataset, strategy)
                    num, ut = decode_cc(compression_config[cc_key][2])
                    mu2, V2, e2 = tf_pca(layer_input, num, num_as_threshold=ut, conv=True, verbose=True,
                                         prefix=' {}'.format(l_name))
                else:
                    mu2, V2, e2 = None, None, None


                # for conv3, input needs to pass through above, and conv2/bn2/relu2 of this block
                if compression_config[cc_key][4] is not None:
                    layer_input_shape = overall_model.layers[layer_index].relu2.output_shape[1:]
                    _, extra_layer = get_compressed_layer_shape(layer_input_shape, var_config)
                    with strategy.scope():
                        intermediate_model = tf.keras.Sequential(layers=overall_model.layers[:layer_index])
                        intermediate_model.add(overall_model.layers[layer_index].conv1)
                        intermediate_model.add(overall_model.layers[layer_index].bn1)
                        intermediate_model.add(overall_model.layers[layer_index].relu1)
                        intermediate_model.add(overall_model.layers[layer_index].conv2)
                        intermediate_model.add(overall_model.layers[layer_index].bn2)
                        intermediate_model.add(overall_model.layers[layer_index].relu2)
                        if extra_layer is not None:
                            intermediate_model.add(extra_layer)

                    layer_input = predict(intermediate_model, dataset, strategy)
                    num, ut = decode_cc(compression_config[cc_key][4])
                    mu3, V3, e3 = tf_pca(layer_input, num, num_as_threshold=ut, conv=True, verbose=True,
                                         prefix=' {}'.format(l_name))
                else:
                    mu3, V3, e3 = None, None, None


                muVes.append((mu1, V1, e1, mu2, V2, e2, mu3, V3, e3))

            else:
                layer_input_shape = overall_model.layers[layer_index-1].output_shape[1:]
                _, extra_layer = get_compressed_layer_shape(layer_input_shape, var_config)
                with strategy.scope():
                    intermediate_model = tf.keras.Sequential(layers=overall_model.layers[:layer_index])
                    if extra_layer is not None:
                        intermediate_model.add(extra_layer)

                layer_input = predict(intermediate_model, dataset, strategy)
                num, ut = decode_cc(compression_config[cc_key][0])
                mu, V, e = tf_pca(layer_input, num, num_as_threshold=ut, conv=False, verbose=True,
                                  prefix=' {}'.format(l_name))

                muVes.append((mu, V, e))

    return muVes

    #     if input_arr is None or var_config['compute_all_from_scratch'] is True:
    #         # we have to start from the beginning
    #         layer_input, did_compress = get_layer_input_possibly_compressed(layer_index, dataset, overall_model,
    #                                                                         var_config, strategy)
    #
    #         if not did_compress and var_config['compute_all_from_scratch'] is False:
    #             input_arr = layer_input
    #             prev_layer_index = layer_index
    #     else:
    #         # we can start with input_arr (input of prev pca layer) and just call until we get to this layer
    #         while prev_layer_index < layer_index:
    #             print("\tpassing through layer {}".format(all_layers[prev_layer_index]))
    #             input_arr = overall_model.get_layer(all_layers[prev_layer_index])(input_arr)
    #             prev_layer_index += 1
    #
    #         layer_input = input_arr
    #
    #     num, ut = decode_cc(compression_config[l_name][0])
    #     conv = 'conv' in l_name
    #     mu, V, e = tf_pca(layer_input, num, num_as_threshold=ut, conv=conv, verbose=True, prefix=' {}'.format(l_name))
    #     muVes.append((mu, V, e))
    #
    # return muVes