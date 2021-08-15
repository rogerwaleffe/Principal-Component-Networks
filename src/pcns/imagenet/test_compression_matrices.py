import time
import os
import numpy as np
import tensorflow as tf

from src.pcns.imagenet.imagenet_helpers import input_fn, create_and_cache_tf_records, input_fn_given_dataset
from src.pcns.imagenet.vgg import get_full_vgg19_model
from src.pcns.optimizer import SGDW
from src.pcns.layer_helpers import Conv2DPCALayer, DensePCALayer
from src.pcns.compression_helpers import tf_pca, tf_kill_outputs, tf_transform_conv_weights, \
                                                          tf_transform_dense_weights, decode_cc

from src.pcns.layer_helpers import constant_initializer_from_tensor as cift



# DATA_DIR = 'mnt/imagenet_tf_record_data'
# CHECKPOINT_DIR = 'mnt/model_data'
DATA_DIR = '/mnt/imagenet_tf_record_data'
CHECKPOINT_DIR = '/mnt'

MAX_IMAGE_SIZE = 25088 # 100352 # 25088
CENTER_CROPS_FOR_TRAIN = True
COMPRESSION_LAYER_TYPE = 'MaxPooling'
NUM_SAMPLES_FOR_COMPRESSION = 10000
C_BATCH_SIZE = 256
COMPUTE_ALL_FROM_SCRATCH = False



def get_compression_layer(layer_input_shape, new_shape):
    ph = layer_input_shape[0] // new_shape[0]
    pw = layer_input_shape[1] // new_shape[1]

    if COMPRESSION_LAYER_TYPE == 'MaxPooling':
        extra_layer = tf.keras.layers.MaxPooling2D((ph, pw))

    elif COMPRESSION_LAYER_TYPE == 'AveragePooling':
        extra_layer = tf.keras.layers.AveragePooling2D((ph, pw))

    elif COMPRESSION_LAYER_TYPE == 'RandomCrop':
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

    elif COMPRESSION_LAYER_TYPE == 'CenterCrop':
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

    elif COMPRESSION_LAYER_TYPE == 'EvenlySpacedPixels':
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

    # elif COMPRESSION_LAYER_TYPE == 'Resize':
    #     pass

    else:
        raise NotImplementedError()

    return extra_layer



# TODO
# TODO: worth trying to get the whole thing in a tf.function using TensorArray or some sort of scatter updates stuff
# @tf.function
def predict(overall_model, dataset, layer_index, extra_layer, compression_config, l_name, new_shape):
    # strategy = tf.distribute.MirroredStrategy()
    # TODO: create the model inside the scope to ensure that any variables are mirrored
    with strategy.scope():
        intermediate_model = tf.keras.Sequential(layers=overall_model.layers[:layer_index])
        if extra_layer is not None:
            intermediate_model.add(extra_layer)

    # TODO: distribute the dataset based on the strategy
    dist_dataset = strategy.experimental_distribute_dataset(dataset)

    @tf.function
    def forward_pass(dist_inputs):
        def step_fn(inputs):
            output = intermediate_model(inputs)
            return output

        outputs = strategy.experimental_run_v2(step_fn, args=(dist_inputs, ))
        local_tensors = strategy.experimental_local_results(outputs)
        # print(outputs)
        # print(local_tensors)
        # return tf.concat(outputs, axis=0)
        return tf.concat(local_tensors, axis=0)
        # return tf.parallel_stack(local_tensors)

    # t1 = time.time()
    # print("\nBeginning time {}".format(t1))
    # for i, _ in enumerate(dist_dataset):
    #     print(i)
    # print("\nEnd time {}, total {}".format(time.time(), time.time() - t1))

    batch_tensors = []
    with strategy.scope():
        for i, (batch_x, batch_y) in enumerate(dist_dataset):
            temp = forward_pass(batch_x)
            batch_tensors.append(temp)
            print(i)





    # print((4*256,)+new_shape)
    # @tf.function
    # def combine_forward_passes():
    #     ta = tf.TensorArray(tf.float32, size=0, dynamic_size=True, infer_shape=True)
    #                         # element_shape=tf.TensorShape((4*256,)+new_shape))
    #     with strategy.scope():
    #         for ii, (bx, by) in enumerate(dist_dataset):
    #             temp = forward_pass(bx)
    #             ta = ta.write(ii, temp)
    #             print(ii)
    #     return ta.concat()
    # return combine_forward_passes()






    # with strategy.scope():
    # intermediate_model = tf.keras.Sequential(layers=overall_model.layers[:layer_index])
    # if extra_layer is not None:
    #     intermediate_model.add(extra_layer)
    #
    # return intermediate_model.predict(dataset, steps=np.ceil(NUM_SAMPLES_FOR_COMPRESSION/C_BATCH_SIZE), verbose=1)



    # @tf.function
    # def combine_predict_pca(m):
    #     temp = m.predict(dataset, steps=np.ceil(NUM_SAMPLES_FOR_COMPRESSION/C_BATCH_SIZE), verbose=1)
    #
    #     num, ut = decode_cc(compression_config[l_name][0])
    #     conv = 'conv' in l_name
    #     mu, V, e = tf_pca(temp, num, num_as_threshold=ut, conv=conv, verbose=True, prefix=' {}'.format(l_name))
    #     return mu, V, e
    #
    # with strategy.scope():
    #     intermediate_model = tf.keras.Sequential(layers=overall_model.layers[:layer_index])
    #     if extra_layer is not None:
    #         intermediate_model.add(extra_layer)
    #
    # return combine_predict_pca(intermediate_model)




    # @tf.function
    # def call(m, b):
    #     return m(b)
    #
    # intermediate_model = tf.keras.Sequential(layers=overall_model.layers[:layer_index])
    # if extra_layer is not None:
    #     intermediate_model.add(extra_layer)
    #
    # # print(len(dataset))
    # # we could:
    # # concatenate tensors
    # # stack tensors
    # # parallel stack tensors (might have to reshape for stack or parallel stack)
    # # scatter_nd_update into pre-allocated tensor of zeros
    #
    # # # total = tf.zeros([20*256, 7, 7, 512])
    # # # temp_tensor = tf.zeros([256, 7, 7, 512])
    # # # ta = tf.TensorArray(tf.float32, size=196, dynamic_size=False, element_shape=[256, 7, 7, 512])
    # # #
    # batch_tensors = []
    # # # index = 0
    # for i, (batch_x, batch_y) in enumerate(dataset):
    # #     # temp_tensor = intermediate_model(batch_x)
    # #     # indices = tf.reshape(tf.range(256 * index, 256 * (index + 1)), [256, 1])
    # #     # tf.tensor_scatter_nd_update(total, indices, temp_tensor)
    # #     # index += 1
    # #     # temp = intermediate_model.predict(batch_x)
    # #     # with strategy.scope():
    #     temp = call(intermediate_model, batch_x)
    # #     # print(temp)
    #     batch_tensors.append(temp)
    # #     # batch_tensors.append(call(intermediate_model, batch_x))
    #     print(i)


        # ta.write(index, temp_tensor)
        # batch_tensors.append(temp_tensor)
    # total = tf.parallel_stack(batch_tensors)
    # return temp_tensor
    # return ta.concat()
    # print(len(batch_tensors))
    # with tf.device('/device:gpu:0'):

    total = tf.concat(batch_tensors, axis=0)
    return total




def get_layer_input_possibly_compressed(layer_index, dataset, overall_model, compression_config, l_name):
    layer_input_shape = overall_model.layers[layer_index-1].output_shape[1:]

    image_size = 1
    for val in layer_input_shape:
        image_size *= val

    new_shape = layer_input_shape
    while image_size > MAX_IMAGE_SIZE:
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

    # # TODO fine to reuse strategy so that batches are computed in parallel?
    # # strategy = tf.distribute.MirroredStrategy()
    # with strategy.scope():
    #     intermediate_model = tf.keras.Sequential(layers=overall_model.layers[:layer_index])
    #     if new_shape != layer_input_shape:
    #         print("\tadding {} compression layer to reduce input shape".format(COMPRESSION_LAYER_TYPE))
    #         extra_layer = get_compression_layer(layer_input_shape, new_shape)
    #         intermediate_model.add(extra_layer)

    # output = intermediate_model.predict(dataset, steps=np.ceil(NUM_SAMPLES_FOR_COMPRESSION/C_BATCH_SIZE), verbose=1)

    if new_shape != layer_input_shape:
        print("\tadding {} compression layer to reduce input shape".format(COMPRESSION_LAYER_TYPE))
        extra_layer = get_compression_layer(layer_input_shape, new_shape)
    else:
        extra_layer = None

    output = predict(overall_model, dataset, layer_index, extra_layer, compression_config, l_name, new_shape)
    # print(output)
    # print("OUTPUT SHAPE {}".format(output.shape))

    return output, new_shape != layer_input_shape



#@tf.function
def compute_all_muVs(compression_config, dataset, overall_model):
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

        if input_arr is None or COMPUTE_ALL_FROM_SCRATCH is True:
            # we have to start from the beginning
            layer_input, did_compress = get_layer_input_possibly_compressed(layer_index, dataset, overall_model,
                                                                            compression_config, l_name)

            if not did_compress and COMPUTE_ALL_FROM_SCRATCH is False:
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










def imagenet_vgg_compression(compression_config, overall_model, muVes, verbose=True):
    """

    :param compression_config:
    :param overall_model:
    :param muVes:
    :param verbose:
    :return:
    """

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
    cc_keys = list(compression_config.keys())
    cc_values = list(compression_config.values())

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
    compressed_model = tf.keras.Sequential()
    compressed_model.add(tf.keras.layers.InputLayer(input_shape=overall_model.layers[0].input_shape[1:], name='input'))

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
            num, ut = decode_cc(compression_config[curr_layer][1])
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
                                                        bias_initializer=cift(b_c), activation='relu', name=curr_layer))
        elif cc[curr_layer][0] is not None:
            W_c_p, b_c_p = tf_transform_dense_weights(mu_p, V_p, W_c, b_c)
            compressed_model.add(DensePCALayer(int(n_c), mu_p, V_p, kernel_initializer=cift(W_c_p),
                                               bias_initializer=cift(tf.squeeze(b_c_p)), activation=activation,
                                               name=curr_layer))
        else:
            compressed_model.add(tf.keras.layers.Dense(int(n_c), kernel_initializer=cift(W_c), bias_initializer=cift(b_c),
                                                       activation=activation, name=curr_layer))

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
            # elif 'dropout' in all_layers[c_i]:
            #     compressed_model.add(tf.keras.layers.Dropout(0.5))
            elif 'softmax' in all_layers[c_i]:
                compressed_model.add(tf.keras.layers.Softmax(name=all_layers[c_i]))

    # Optimizer
    # TODO: fix this
    compressed_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                             optimizer=SGDW(0.01 * 5*10e-4, learning_rate=0.01, momentum=0.9), metrics=['accuracy'])

    return compressed_model










if __name__ == "__main__":
    # Create the model
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = get_full_vgg19_model(dropout=True)
    #
    #     # lr = 0.01
    #     # optimizer = SGDW(lr * 5 * 10e-4, learning_rate=lr, momentum=0.9)
    #     # loss = tf.keras.losses.SparseCategoricalCrossentropy()
    #     #
    #     # model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    #     # print(model.summary())
    #
    # TODO: should this be in the scope
    model.load_weights(os.path.join(CHECKPOINT_DIR, 'weights_11.hdf5'))
    # print(model.to_json())


    # Test Compression
    val_dataset = input_fn(DATA_DIR, is_training=False, num_epochs=1, batch_size=256)
    train_dataset = input_fn(DATA_DIR, is_training=True, num_epochs=1, batch_size=256).take(10).cache()
    # TODO: does this have enough randomness, should we use center_crops_for_train?
    compression_dataset = input_fn(DATA_DIR, is_training=True, num_epochs=1,
                                   batch_size=C_BATCH_SIZE, center_crops_for_train=CENTER_CROPS_FOR_TRAIN)
    compression_dataset = compression_dataset.take(np.ceil(NUM_SAMPLES_FOR_COMPRESSION/C_BATCH_SIZE))
    # compression_dataset = compression_dataset.cache() #TODO .cache()

    # t1 = time.time()
    # print("\nBeginning time {}".format(t1))
    # for i, (batch_x, batch_y) in enumerate(compression_dataset):
    #     print(i, batch_x.shape)
    #     print(batch_y)
    # print("\nEnd time {}, total {}".format(time.time(), time.time() - t1))

    # print(model.evaluate(val_dataset, steps=np.ceil(50000/256)))
    # model.predict(compression_dataset, steps=np.ceil(num_samples_for_compression/C_BATCH_SIZE), verbose=1)

    # print("Start")
    # strategy = tf.distribute.MirroredStrategy()
    # with strategy.scope():
    #     intermediate_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('flatten').output)
    # intermediate_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('flatten').output)
    # print("Starting Prediction")
    # X = intermediate_model.predict(compression_dataset)
    # print(X.shape)
    # print("Ending Prediction, Starting PCA")
    # mu, V, e = tf_pca(X, 25088, num_as_threshold=False, conv=False, verbose=True)
    # print("End")
    # tf.print(e, summarize=-1)


    # for batch in compression_dataset:
    #     print(len(batch))
    #     batch = batch[0]
    #     print(batch.shape)
    #     for layer in model.layers:
    #         batch = layer(batch)
    #     print("Done with batch")
    #     print(batch.shape)

    # (input image size reduction factor, input pca num, output columns/filters num
    # cc = {
    #     'conv1': (None, 'd64'), 'conv2': ('d64', 'd64'), 'conv3': ('d64', 'd128'), 'conv4': ('d128', 'd128'),
    #     'conv5': ('d128', 'd256'), 'conv6': ('d256', 'd256'), 'conv7': ('d256', 'd256'), 'conv8': ('d256', 'd256'),
    #     'conv9': ('d256', 'd512'), 'conv10': ('d512', 'd512'), 'conv11': ('d512', 'd512'), 'conv12': ('d512', 'd512'),
    #     'conv13': ('d512', 'd512'), 'conv14': ('d512', 'd512'), 'conv15': ('d512', 'd512'), 'conv16': ('d512', 'd512'),
    #     'fc1': ('d25088', 'd4096'), 'fc2': ('d4096', 'd4096'), 'output': ('d4096', None),
    #     'weighted_row_sum': True
    # }

    cc = {
        'conv1': (None, None), 'conv2': (None, None), 'conv3': ('d64', None), 'conv4': (None, None),
        'conv5': (None, None), 'conv6': (None, None), 'conv7': (None, None), 'conv8': (None, None),
        'conv9': (None, 't1.0'), 'conv10': ('d256', 't1.0'), 'conv11': ('d192', 't1.0'), 'conv12': ('d192', 't1.0'),
        'conv13': ('d192', 't1.0'), 'conv14': ('d256', 't1.0'), 'conv15': ('d256', 't1.0'), 'conv16': ('d256', 't10.0'),
        'fc1': ('d250', None), 'fc2': ('d500', None), 'output': (None, None),
    }
    # cc = {
    #     'conv1': (None, None), 'conv2': (None, None), 'conv3': (None, None), 'conv4': (None, None),
    #     'conv5': (None, None), 'conv6': (None, None), 'conv7': (None, None), 'conv8': (None, None),
    #     'conv9': (None, None), 'conv10': (None, None), 'conv11': (None, None), 'conv12': (None, None),
    #     'conv13': (None, None), 'conv14': (None, None), 'conv15': (None, None), 'conv16': ('d256', 't1.0'),
    #     'fc1': ('d250', None), 'fc2': (None, None), 'output': (None, None),
    # }

    # #'weighted_row_sum': True
    #
    mu_V_e_list = compute_all_muVs(cc, compression_dataset, model)
    new_model = imagenet_vgg_compression(cc, model, mu_V_e_list, verbose=True)

    print(new_model.summary())

    # print(new_model.get_layer('conv16').get_config())
    # new_layer = Conv2DPCALayer.from_config(new_model.get_layer('conv16').get_config())

    # print(new_model.evaluate(val_dataset, steps=np.ceil(50000/256)))
    new_model.fit(train_dataset)
    new_model.save_weights('test_save_compression_model/')
    new_model.fit(train_dataset)



    print("TEST MODEL")
    json = new_model.to_json()
    # print(json)

    with open("model.json", "w") as fp:
        fp.write(json)

    with open("model.json", "r") as fp:
        json = fp.read()

    # new_model.save_weights('test_save_compression_model/')

    test_model = tf.keras.models.model_from_json(json, custom_objects={'DensePCALayer': DensePCALayer,
                                                                       'Conv2DPCALayer': Conv2DPCALayer,
                                                                       'constant_initializer_from_tensor': cift})

    # test_model.load_weights('test_save_compression_model/')

    test_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                       optimizer=SGDW(0.0, learning_rate=1000.0, momentum=0.0), metrics=['accuracy'])

    # NOTE: is the optimizer loaded regardless of whether this is done or not?? NO! You do have to do this.
    load_opt = True
    if load_opt:
        init_dataset = train_dataset.take(1)
        print("LOADING WEIGHTS: training on one batch to initialize optimizer vars so they can be restored")
        test_model.train_on_batch(init_dataset)

    test_model.load_weights('test_save_compression_model/')
    # test_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    #                    optimizer=SGDW(0, learning_rate=1000.0, momentum=0.0), metrics=['accuracy'])

    print(test_model.summary())
    # print(test_model.evaluate(val_dataset, steps=np.ceil(50000 / 256)))

    test_model.fit(train_dataset)





    # new_model.save('test', save_format='tf')
    # del new_model
    # new_model = tf.keras.models.load_model('test')
    # # print(new_model.summary())
    #
    # new_model.evaluate(val_dataset, steps=np.ceil(50000/256))
    #
    # print(new_model.summary())
    #
    # lr = 0.01
    # optimizer = SGDW(lr * 5 * 10e-4, learning_rate=lr, momentum=0.9)
    # loss = tf.keras.losses.SparseCategoricalCrossentropy()
    #
    # new_model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])





    # train_tf_records_cached = create_and_cache_tf_records(DATA_DIR, is_training=True)
    #
    # train_dataset = input_fn_given_dataset(train_tf_records_cached, is_training=True, num_epochs=1,
    #                                        # shuffle_buffer=1281167,
    #                                        batch_size=256,
    #                                        drop_remainder=False, center_crops_for_train=False)
    # train_dataset = input_fn(DATA_DIR, is_training=True, num_epochs=1, batch_size=256)
    # compression_dataset = input_fn(DATA_DIR, is_training=True, num_epochs=1, batch_size=C_BATCH_SIZE,
    #                                center_crops_for_train=CENTER_CROPS_FOR_TRAIN)
    # compression_dataset = compression_dataset.take(np.ceil(NUM_SAMPLES_FOR_COMPRESSION/C_BATCH_SIZE))  # .cache() #TODO .cache()
    # compression_dataset = input_fn_given_dataset(train_tf_records_cached, is_training=True, num_epochs=1,
    #                                              shuffle_buffer=1,
    #                                              batch_size=5,
    #                                              drop_remainder=False, center_crops_for_train=CENTER_CROPS_FOR_TRAIN)
    #
    # counter = tf.data.experimental.Counter()
    # compression_dataset = tf.data.Dataset.zip((counter, compression_dataset))
    #
    # compression_dataset = compression_dataset.take(np.ceil(NUM_SAMPLES_FOR_COMPRESSION/C_BATCH_SIZE))

    # import time
    # # # dataset = input_fn('/mnt/imagenet_tf_record_data', is_training=True, num_epochs=1, batch_size=256)
    # #
    # t1 = time.time()
    # print("\nBeginning time {}".format(t1))
    # for i, (batch_x, batch_y) in enumerate(train_dataset):
    #     print(i, batch_x.shape)
    #     # print(batch_y)
    # print("\nEnd time {}, total {}".format(time.time(), time.time()-t1))

    # # t1 = time.time()
    # # print("\nBeginning time {}".format(t1))
    # # for i, (batch_x, batch_y) in enumerate(train_dataset):
    # #     print(i, batch_x.shape)
    # #     # print(batch_y)
    # # print("\nEnd time {}, total {}".format(time.time(), time.time() - t1))
    #
    #
    # t1 = time.time()
    # print("\nBeginning time {}".format(t1))
    # for i, (batch_x, batch_y) in enumerate(compression_dataset):
    #     print(i, batch_x.shape)
    #     print(batch_y)
    # print("\nEnd time {}, total {}".format(time.time(), time.time() - t1))
    #
    # t1 = time.time()
    # print("\nBeginning time {}".format(t1))
    # for i, (batch_x, batch_y) in enumerate(compression_dataset):
    #     print(i, batch_x.shape)
    #     print(batch_y)
    # print("\nEnd time {}, total {}".format(time.time(), time.time() - t1))
    #
    # mu_V_e_list = compute_all_muVs(cc, compression_dataset, model)
    #
    #
    #
    #
    #
    # from src.pcns.imagenet.imagenet_helpers import parse_record
    # counter = tf.data.experimental.Counter()
    # train_tf_records_cached = tf.data.Dataset.zip((tf.data.Dataset.range(1282167), train_tf_records_cached))
    # dataset = train_tf_records_cached
    #
    # dataset = dataset.prefetch(buffer_size=C_BATCH_SIZE)
    # dataset = dataset.shuffle(buffer_size=1, reshuffle_each_iteration=True)
    # dataset = dataset.repeat(1)
    # dataset = dataset.map(lambda v1, v2: (v1, parse_record(v2, True, center_crops_for_train=True)),
    #                       num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # dataset = dataset.batch(C_BATCH_SIZE, drop_remainder=False)
    # dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    # dataset = dataset.take(np.ceil(NUM_SAMPLES_FOR_COMPRESSION/C_BATCH_SIZE))
    #
    #
    #
    # # t1 = time.time()
    # # print("\nBeginning time {}".format(t1))
    # # for i, (counter, (batch_x, batch_y)) in enumerate(dataset.take(np.ceil(NUM_SAMPLES_FOR_COMPRESSION/C_BATCH_SIZE))):
    # #     print(i, batch_x.shape)
    # #     print(counter, batch_y)
    # # print("\nEnd time {}, total {}".format(time.time(), time.time() - t1))
    # #
    # # t1 = time.time()
    # # print("\nBeginning time {}".format(t1))
    # # for i, (counter, (batch_x, batch_y)) in enumerate(dataset.skip(5*np.ceil(NUM_SAMPLES_FOR_COMPRESSION/C_BATCH_SIZE)).take(np.ceil(NUM_SAMPLES_FOR_COMPRESSION/C_BATCH_SIZE))):
    # #     print(i, batch_x.shape)
    # #     print(counter, batch_y)
    # # print("\nEnd time {}, total {}".format(time.time(), time.time() - t1))
    #
    # t1 = time.time()
    # print("\nBeginning time {}".format(t1))
    # i = 0
    # for j, (counter, (batch_x, batch_y)) in dataset.enumerate(start=0):
    #     print(j, batch_x.shape)
    #     print(counter, batch_y)
    #     i += 1
    #     if i >= 9:
    #         break
    # print("\nEnd time {}, total {}".format(time.time(), time.time() - t1))
    #
    # t1 = time.time()
    # print("\nBeginning time {}".format(t1))
    # i = 0
    # for j, (counter, (batch_x, batch_y)) in dataset.enumerate(start=5*np.ceil(NUM_SAMPLES_FOR_COMPRESSION/C_BATCH_SIZE)):
    #     print(j, batch_x.shape)
    #     print(counter, batch_y)
    #     i += 1
    #     if i >= 9:
    #         break
    # print("\nEnd time {}, total {}".format(time.time(), time.time() - t1))
