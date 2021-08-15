import time
import numpy as np
import tensorflow as tf

from datetime import datetime

from src.pcns.resnet.resnet_model import ResNet
from src.pcns.optimizer import SGDW, piecewise_scheduler

HEIGHT, WIDTH, NUM_CHANNELS = 32, 32, 3


CHANNEL_MEANS = [125.3, 123.0, 113.9] # RGB
CHANNEL_STANDARDS = [63.0,  62.1,  66.7]
MEANS = tf.broadcast_to(CHANNEL_MEANS, [HEIGHT, WIDTH, NUM_CHANNELS])
STANDARDS = tf.broadcast_to(CHANNEL_STANDARDS, [HEIGHT, WIDTH, NUM_CHANNELS])



def mean_channel_subtraction(image):
    return image - MEANS



def channel_standardization(image):
    return image/STANDARDS



def get_dataset(samples_for_val):
    cifar10 = tf.keras.datasets.cifar10

    (xTrain, yTrain), (xTest, yTest) = cifar10.load_data()

    # Create Validation Set
    random_indices = np.random.permutation(np.arange(xTrain.shape[0]))

    xVal = xTrain[random_indices[:samples_for_val]]
    yVal = yTrain[random_indices[:samples_for_val]]

    xTrain = xTrain[random_indices[samples_for_val:]]
    yTrain = yTrain[random_indices[samples_for_val:]]

    print("Dataset created. Sizes:\n(train, val, test, yTrain, yVal, yTest): ({}, {}, {}, {}, {}, {})"
          "".format(xTrain.shape, xVal.shape, xTest.shape, yTrain.shape, yVal.shape, yTest.shape))

    return (xTrain, yTrain), (xVal, yVal), (xTest, yTest)

    # print("Dataset created. Sizes:\n(train, test, yTrain, yTest): ({}, {}, {}, {})"
    #       "".format(xTrain.shape, xTest.shape, yTrain.shape, yTest.shape))
    #
    # return (xTrain, yTrain), (xTest, yTest)



def preprocess_image(image, label, is_training):
    """
    Preprocess a single image of layout [height, width, depth].
    """
    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.float32) # yTrain was float64 in dataset_helpers creation, but this should be fine

    if is_training:
        # Resize the image to add four extra pixels on each side.
        image = tf.image.resize_with_crop_or_pad(image, HEIGHT + 8, WIDTH + 8)

        # Randomly crop a [HEIGHT, WIDTH] section of the image.
        image = tf.image.random_crop(image, [HEIGHT, WIDTH, NUM_CHANNELS])

        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)

    # Subtract off the mean and divide by the variance of the pixels.
    # image = tf.image.per_image_standardization(image) # TODO: per-pixel mean subtraction, not whatever this is
    image = mean_channel_subtraction(image)
    image = channel_standardization(image)

    return image, label



def input_fn(images, labels, map_fn=preprocess_image, num_epochs=1, num_images=1, batch_size=128, is_training=False,
             drop_remainder=False):
    """
    :return: The tf.data.Dataset
    """
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))

    # Prefetch a batch at a time to smooth out the time taken for processing.
    dataset = dataset.prefetch(buffer_size=batch_size)

    if is_training:
        # Shuffles records before repeating to respect epoch boundaries. Show every element of one epoch before next.
        dataset = dataset.shuffle(buffer_size=num_images, reshuffle_each_iteration=True)

    dataset = dataset.repeat(num_epochs)

    # Preprocess images and labels. TODO: with map before repeat, we may not be preprocessing differently per epoch
    dataset = dataset.map(lambda image, label: map_fn(image, label, is_training),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

    # Shuffle/batch records before repeating to respect epoch boundaries. Last batch in each epoch may not be full.
    # Repeats the dataset for the number of epochs to train.
    # dataset = dataset.repeat(num_epochs)

    # Operations between the final prefetch and the get_next call to the iterator
    # will happen synchronously during run time. We prefetch here again to
    # background all of the above processing work and keep it out of the
    # critical training path. Setting buffer_size to tf.data.experimental.AUTOTUNE
    # allows DistributionStrategies to adjust how many batches to fetch based
    # on how many devices are present.
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset



if __name__ == "__main__":
    # for _ in range(5):
    print("Running cifar10.py: {}".format(datetime.now()))

    # {}_{}: (conv1 input, conv1 output, conv2 input), note that conv1 input also applies to conv0 input for {}_1 blocks
    # {}: conv0 and all conv2 outputs for stage
    # cc = {
    #     'initial_conv': (None, 'd64'),
    #     '1_1': ('d64', 'd64', 'd64'), '1_2': ('d64', 'd64', 'd64'), '1_3': ('d64', 'd64', 'd64'),
    #     '1': 'd64',
    #
    #     '2_1': ('d64', 'd128', 'd128'), '2_2': ('d128', 'd128', 'd128'), '2_3': ('d128', 'd128', 'd128'),
    #     '2': 'd128',
    #
    #     '3_1': ('d128', 'd256', 'd256'), '3_2': ('d256', 'd256', 'd256'), '3_3': ('d256', 'd256', 'd256'),
    #     '3': 'd256',
    #
    #     'fc1': ('d256', None),
    #
    #     'use_all_muVs': True, 'weighted_row_sum': True
    # }
    cc = {
        'initial_conv': (None, None),
        '1_1': (None, None, None), '1_2': (None, None, None), '1_3': (None, None, None),
        '1': None,

        '2_1': (None, None, None), '2_2': (None, None, None), '2_3': (None, None, None),
        '2': None,

        '3_1': (None, None, 'd256'), '3_2': ('d256', None, 'd256'), '3_3': ('d256', None, 'd256'),
        '3': None,

        'fc1': ('d256', None),

        'use_all_muVs': True, 'weighted_row_sum': False
    }
    # config = {'compression_config': cc, 'num_samples_for_compression': 5000,
    #           'compression_after_epoch': 10, 'total_epochs': 20, 'batch_size': 128}
    # TODO: constraint checks for the compression_config? all conv1s >= block 2 plus next stage conv1 same input size? (technically no?)



    NUM_VAL = 5000
    NUM_SAMPLES_FOR_COMPRESSION = 5000
    EPOCHS_BEFORE_COMPRESSION = 5
    TOTAL_EPOCHS = 182
    VERBOSE = 2 # 1 = progress bar, 2 = one line per epoch

    NUM_TEST = 10000
    BATCH_SIZE = 128

    measure_base_perf = False
    measure_c_perf = False





    # Get train/test data sets
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = get_dataset(NUM_VAL)
    num_train = train_x.shape[0]

    train_dataset = input_fn(train_x, train_y, num_epochs=EPOCHS_BEFORE_COMPRESSION, num_images=num_train,
                             batch_size=BATCH_SIZE, is_training=True)
    val_dataset = input_fn(val_x, val_y, batch_size=BATCH_SIZE)
    test_dataset = input_fn(test_x, test_y, batch_size=BATCH_SIZE)


    # Create ResNet model
    resnet = ResNet(input_shape=(32, 32, 3), num_classes=10, bottleneck=False, num_filters_at_start=64,
                    initial_kernel_size=3, initial_conv_strides=1, initial_pool_size=None, initial_pool_strides=None,
                    num_residual_blocks_per_stage=[3, 3, 3], first_block_strides_per_stage=[1, 2, 2], kernel_size=3,
                    project_first_residual=True, version='V1', data_format='channels_last', compression_config=cc)
    model = resnet.get_model()

    lr_schedule = piecewise_scheduler([91, 136], [1.0, 0.1, 0.01], base_rate=0.1, boundaries_as='epochs',
                                      num_images=num_train, batch_size=BATCH_SIZE)
    # optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)
    # wd_schedule = piecewise_scheduler([91, 136], [1.0, 0.1, 0.01], base_rate=0.1*1e-4, boundaries_as='epochs',
    #                                   num_images=num_train, batch_size=BATCH_SIZE)
    # optimizer = SGDW(wd_schedule, learning_rate=lr_schedule, momentum=0.9)
    # optimizer = SGDW(1e-5, learning_rate=0.1, momentum=0.9)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    print(model.summary())



    # if measure_base_perf is True:
    #     print("\nBeginning performance measurements for full model, time {}".format(time.time()))
    #     print("\nTraining 5 epochs, time {}".format(time.time()))
    #     model.fit(train_dataset, epochs=5, steps_per_epoch=np.floor(num_train/BATCH_SIZE), verbose=VERBOSE)
    #     print("\nTraining 5 epochs again, time {}".format(time.time()))
    #     model.fit(train_dataset, epochs=5, steps_per_epoch=np.floor(num_train/BATCH_SIZE), verbose=VERBOSE)
    #
    #     print("\nValidating 5 times on test set, using given batch size, time {}".format(time.time()))
    #     for _ in range(5):
    #         print("\nStarting validation, time {}".format(time.time()))
    #         model.evaluate(test_dataset, steps=np.ceil(NUM_TEST/BATCH_SIZE), verbose=0)
    #         print("\nValidation done, time {}".format(time.time()))
    #     print("\nValidating 5 times on test set again, using given batch size, time {}".format(time.time()))
    #     for _ in range(5):
    #         print("\nStarting validation, time {}".format(time.time()))
    #         model.evaluate(test_dataset, steps=np.ceil(NUM_TEST/BATCH_SIZE), verbose=0)
    #         print("\nValidation done, time {}".format(time.time()))
    #     print("\nDone measuring performance, time {}".format(time.time()))
    #     import sys
    #     sys.exit(0)





    # Figure 2b like plot:
    # for epoch in range(EPOCHS_BEFORE_COMPRESSION):
    #     model.fit(train_dataset, epochs=1, steps_per_epoch=np.floor(num_train/BATCH_SIZE), verbose=VERBOSE)
    #     print("Epoch: {}".format(epoch+1))
    #
    #     score = model.evaluate(val_dataset, steps=np.ceil(NUM_VAL/BATCH_SIZE), verbose=0)
    #     print("Val Loss {:.5f}, Val Acc: {:.5f}".format(score[0], score[1]))
    #     score = model.evaluate(test_dataset, steps=np.ceil(NUM_TEST/BATCH_SIZE), verbose=0)
    #     print("Test Loss {:.5f}, Test Acc: {:.5f}".format(score[0], score[1]))
    #
    #     compression_dataset = input_fn(train_x, train_y, num_images=num_train, batch_size=NUM_SAMPLES_FOR_COMPRESSION,
    #                                    is_training=False)
    #     compression_dataset = compression_dataset.take(1)
    #
    #     input_arr = []
    #     for i, (im, lab) in enumerate(compression_dataset):
    #         input_arr.append(im)
    #     input_arr = tf.concat(input_arr, axis=0)
    #
    #     compressed_model = resnet.get_compressed_model(input_arr, model, verbose=False)
    #
    # import sys
    # sys.exit()





    # Train model
    model.fit(train_dataset, epochs=EPOCHS_BEFORE_COMPRESSION, steps_per_epoch=np.floor(num_train/BATCH_SIZE),
              validation_data=val_dataset, validation_steps=np.ceil(NUM_VAL/BATCH_SIZE), validation_freq=1,
              verbose=VERBOSE)
    if VERBOSE == 1:
        model.evaluate(val_dataset, steps=np.ceil(NUM_VAL/BATCH_SIZE))
        model.evaluate(test_dataset, steps=np.ceil(NUM_TEST/BATCH_SIZE))
    else:
        score = model.evaluate(val_dataset, steps=np.ceil(NUM_VAL/BATCH_SIZE), verbose=0)
        print("TRAINING: before compression, Val Loss {:.5f}, Val Acc: {:.5f}".format(score[0], score[1]))
        score = model.evaluate(test_dataset, steps=np.ceil(NUM_TEST/BATCH_SIZE), verbose=0)
        print("TRAINING: before compression, Test Loss {:.5f}, Test Acc: {:.5f}".format(score[0], score[1]))





    # Compress the model
    # NOTE: with the training data, but is_training=False we won't do data augmentation for the compression dataset
    compression_dataset = input_fn(train_x, train_y, num_images=num_train, batch_size=NUM_SAMPLES_FOR_COMPRESSION,
                                   is_training=False)
    compression_dataset = compression_dataset.take(1)

    input_arr = []
    for i, (im, lab) in enumerate(compression_dataset):
        input_arr.append(im)
    input_arr = tf.concat(input_arr, axis=0)


    print("COMPRESSION: getting compressed model, time {}".format(time.time()))
    compressed_model = resnet.get_compressed_model(input_arr, model, verbose=True)
    print("COMPRESSION: done getting compressed model, time {}".format(time.time()))

    boundaries = [91-EPOCHS_BEFORE_COMPRESSION, 136-EPOCHS_BEFORE_COMPRESSION]
    lr_schedule = piecewise_scheduler(boundaries, [1.0, 0.1, 0.01],  base_rate=0.1, boundaries_as='epochs',
                                      num_images=num_train, batch_size=BATCH_SIZE)
    # wd_schedule = piecewise_scheduler(boundaries, [1.0, 0.1, 0.01], base_rate=0.1*1e-4, boundaries_as='epochs',
    #                                   num_images=num_train, batch_size=BATCH_SIZE)
    # optimizer = SGDW(wd_schedule, learning_rate=lr_schedule, momentum=0.9)
    # optimizer = SGDW(1e-5, learning_rate=0.1, momentum=0.9)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    compressed_model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    print(compressed_model.summary())
    # print_ResNet(compressed_model)

    if VERBOSE == 1:
        compressed_model.evaluate(val_dataset, steps=np.ceil(NUM_VAL/BATCH_SIZE))
        compressed_model.evaluate(test_dataset, steps=np.ceil(NUM_TEST/BATCH_SIZE))
    else:
        score = compressed_model.evaluate(val_dataset, steps=np.ceil(NUM_VAL/BATCH_SIZE), verbose=0)
        print("TRAINING: before compression, Val Loss {:.5f}, Val Acc: {:.5f}".format(score[0], score[1]))
        score = compressed_model.evaluate(test_dataset, steps=np.ceil(NUM_TEST/BATCH_SIZE), verbose=0)
        print("TRAINING: before compression, Test Loss {:.5f}, Test Acc: {:.5f}".format(score[0], score[1]))


    train_dataset = input_fn(train_x, train_y, num_epochs=TOTAL_EPOCHS-EPOCHS_BEFORE_COMPRESSION, num_images=num_train,
                             batch_size=BATCH_SIZE, is_training=True)



    # if measure_c_perf is True:
    #     print("\nBeginning performance measurements for compressed model, time {}".format(time.time()))
    #     print("\nTraining 5 epochs, time {}".format(time.time()))
    #     compressed_model.fit(train_dataset, epochs=5, steps_per_epoch=np.floor(num_train/BATCH_SIZE), verbose=VERBOSE)
    #     print("\nTraining 5 epochs again, time {}".format(time.time()))
    #     compressed_model.fit(train_dataset, epochs=5, steps_per_epoch=np.floor(num_train/BATCH_SIZE), verbose=VERBOSE)
    #
    #     print("\nValidating 5 times on test set, using given batch size, time {}".format(time.time()))
    #     for _ in range(5):
    #         print("\nStarting validation, time {}".format(time.time()))
    #         compressed_model.evaluate(test_dataset, steps=np.ceil(NUM_TEST/BATCH_SIZE), verbose=0)
    #         print("\nValidation done, time {}".format(time.time()))
    #     print("\nValidating 5 times on test set again, using given batch size, time {}".format(time.time()))
    #     for _ in range(5):
    #         print("\nStarting validation, time {}".format(time.time()))
    #         compressed_model.evaluate(test_dataset, steps=np.ceil(NUM_TEST/BATCH_SIZE), verbose=0)
    #         print("\nValidation done, time {}".format(time.time()))
    #     print("\nDone measuring performance, time {}".format(time.time()))
    #     import sys
    #     sys.exit(0)



    compressed_model.fit(train_dataset, initial_epoch=EPOCHS_BEFORE_COMPRESSION,
                         epochs=TOTAL_EPOCHS, steps_per_epoch=np.floor(num_train/BATCH_SIZE),
                         validation_data=val_dataset, validation_steps=np.ceil(NUM_VAL/BATCH_SIZE), validation_freq=1,
                         verbose=VERBOSE)

    if VERBOSE == 1:
        compressed_model.evaluate(val_dataset, steps=np.ceil(NUM_VAL/BATCH_SIZE))
        compressed_model.evaluate(test_dataset, steps=np.ceil(NUM_TEST/BATCH_SIZE))
    else:
        score = compressed_model.evaluate(val_dataset, steps=np.ceil(NUM_VAL/BATCH_SIZE), verbose=0)
        print("TRAINING: before compression, Val Loss {:.5f}, Val Acc: {:.5f}".format(score[0], score[1]))
        score = compressed_model.evaluate(test_dataset, steps=np.ceil(NUM_TEST/BATCH_SIZE), verbose=0)
        print("TRAINING: before compression, Test Loss {:.5f}, Test Acc: {:.5f}".format(score[0], score[1]))




















    # VIEWING OF PCA EIGENVALUE MAGNITUDES
    # from src.pcns.compression_helpers import tf_pca
    # initial_epoch = 0
    # for epoch in range(0, EPOCHS):
    #     model.fit(train_dataset, initial_epoch=initial_epoch, epochs=initial_epoch+1, steps_per_epoch=np.ceil(num_train/BATCH_SIZE))
    #     score_val = model.evaluate(val_dataset, verbose=0)
    #     score_test = model.evaluate(test_dataset, verbose=0)
    #     print("Epoch: {}, Val Acc: {:.5f}, Val Loss {:.5f}".format(epoch + 1, score_val[1], score_val[0]))
    #     print("Epoch: {}, Test Acc: {:.5f}, Test Loss {:.5f}".format(epoch + 1, score_test[1], score_test[0]))
    #
    #     initial_epoch += 1
    #
    #     if initial_epoch % 10 == 0:
    #         for layer in model.layers:
    #             print(layer.name)
    #             if 'initial_relu' in layer.name:
    #                 temp_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('initial_relu').output)
    #                 mu, V, e = tf_pca(temp_model.predict(val_dataset), 0.0001, num_as_threshold=True, conv=True, verbose=True)
    #                 print(e)
    #
    #             if 'block' in layer.name:
    #                 for internal_layer in layer.layers:
    #                     print("\t"+internal_layer.name)
    #
    #                     if 'relu1' in internal_layer.name:
    #                         temp_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer(layer.name).relu1.output)
    #                         mu, V, e = tf_pca(temp_model.predict(val_dataset), 0.0001, num_as_threshold=True, conv=True, verbose=True)
    #                         print(e)
    #
    #                 temp_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer(layer.name).output)
    #                 mu, V, e = tf_pca(temp_model.predict(val_dataset), 0.0001, num_as_threshold=True, conv=True, verbose=True)
    #                 print(e)

    # from src.pcns.resnet.resnet_model import print_ResNet
    # resnet.print_ResNet(model)





    # COMPRESSION TEST
    # from src.pcns.compression_helpers import tf_pca, tf_transform_conv_weights
    # from src.pcns.resnet.resnet_model import conv2d, batch_norm, create_ResNetBlockV1
    # from src.pcns.layer_helpers import Conv2DPCALayer
    # from src.pcns.layer_helpers import constant_initializer_from_tensor as cift
    # print("\nCompression:")
    #
    # def make_layer(l, f, k, s):
    #     if 'conv' in l.name:
    #         W = l.weights[0]
    #         return conv2d(f, k, s, data_format='channels_last', name=l.name, kernel_initializer=cift(W))
    #     if 'bn' in l.name:
    #         gamma = l.weights[0]
    #         beta = l.weights[1]
    #         moving_mean = l.weights[2]
    #         moving_var = l.weights[3]
    #         return batch_norm(data_format='channels_last', name=l.name, gamma_initializer=cift(gamma),
    #                           beta_initializer=cift(beta), moving_mean_initializer=cift(moving_mean),
    #                           moving_variance_initializer=cift(moving_var))
    #     if 'relu' in l.name:
    #         return tf.keras.layers.ReLU(name=l.name)
    #
    # def make_compressed_layer(l, p, f, k, s):
    #     # if s != 1:
    #     #     return make_layer(l, f, k, s)
    #     # if k != 3:
    #     #     return make_layer(l, f, k, s)
    #
    #     # print('K='+str(k))
    #     # pad_total = k - 1
    #     # pad_beg = pad_total // 2
    #     # pad_end = pad_total - pad_beg
    #
    #     # print(l.name, prev_layer.name)
    #     temp_model = tf.keras.Model(inputs=model.input, outputs=p.output)
    #     input_to_layer = temp_model.predict(val_dataset)
    #     mu_p, V_p, e = tf_pca(input_to_layer, int(input_to_layer.shape[-1]*6/8), num_as_threshold=False, conv=True, verbose=True)
    #     print(e)
    #
    #     W_c = l.weights[0]
    #     b_c = tf.zeros([W_c.shape[-1], ])
    #     print(W_c.shape)
    #     W_c_p, b_c_p = tf_transform_conv_weights(mu_p, V_p, W_c, b_c)
    #
    #     new_l = Conv2DPCALayer(f, k, mu_p, V_p, strides=s, kernel_initializer=cift(W_c_p), bias_initializer=cift(b_c_p),
    #                            data_format='channels_last', name=l.name)
    #
    #     return new_l
    #
    # # Create the model
    # compressed_model = tf.keras.Sequential()
    # compressed_model.add(tf.keras.layers.InputLayer(input_shape=(32, 32, 3), name='input'))
    #
    # prev_layer = None
    # filters = 16
    # first_block_strides = 1
    # # stage = 1
    # for layer in model.layers:
    #     if 'initial' in layer.name:
    #         compressed_model.add(make_layer(layer, filters, 3, first_block_strides))
    #         prev_layer = layer
    #
    #     if 'block' in layer.name:
    #         if 'stage2_block1' in layer.name:
    #             filters = filters*2
    #             first_block_strides = 2
    #         elif 'stage3_block1' in layer.name:
    #             filters = filters*2
    #             first_block_strides = 2
    #         else:
    #             first_block_strides = 1
    #
    #         projection_shortcut = None
    #         bn0 = None
    #         # for internal_layer in layer.layers['before']:
    #         #
    #         for internal_layer in layer.layers['path0']:
    #             if 'conv' in internal_layer.name:
    #                 # projection_shortcut = make_layer(internal_layer, filters, 1, first_block_strides)
    #                 projection_shortcut = make_compressed_layer(internal_layer, prev_layer, filters, 1, first_block_strides)
    #             else:
    #                 bn0 = make_layer(internal_layer, None, None, None)
    #
    #         for internal_layer in layer.layers['path1']:
    #             if 'conv1' in internal_layer.name:
    #                 # conv1 = make_layer(internal_layer, filters, 3, first_block_strides)
    #                 conv1 = make_compressed_layer(internal_layer, prev_layer, filters, 3, first_block_strides)
    #             if 'bn1' in internal_layer.name:
    #                 bn1 = make_layer(internal_layer, None, None, None)
    #             if 'relu1' in internal_layer.name:
    #                 relu1 = make_layer(internal_layer, None, None, None)
    #             if 'conv2' in internal_layer.name:
    #                 prev_layer = layer.get_layer('relu1')
    #                 # conv2 = make_layer(internal_layer, filters, 3, 1)
    #                 conv2 = make_compressed_layer(internal_layer, prev_layer, filters, 3, 1)
    #             if 'bn2' in internal_layer.name:
    #                 bn2 = make_layer(internal_layer, None, None, None)
    #
    #         for internal_layer in layer.layers['after']:
    #             relu2 = make_layer(internal_layer, None, None, None)
    #
    #         block = create_ResNetBlockV1(filters, 3, first_block_strides, projection_shortcut,
    #                                      data_format='channels_last', name=layer.name)
    #         block.batch_norm0 = bn0
    #         block.conv1 = conv1
    #         block.batch_norm1 = bn1
    #         # block.relu1 = relu1
    #         block.conv2 = conv2
    #         block.batch_norm2 = bn2
    #         # block.relu2 = relu2
    #         compressed_model.add(block)
    #
    #         prev_layer = layer
    #
    #     if 'ap' in layer.name:
    #         compressed_model.add(tf.keras.layers.GlobalAveragePooling2D(data_format='channels_last', name='ap'))
    #     if 'fc' in layer.name:
    #         compressed_model.add(tf.keras.layers.Dense(10, activation=None, use_bias=True, name='fc1',
    #                                                    kernel_initializer=cift(layer.weights[0]),
    #                                                    bias_initializer=cift(layer.weights[1])))
    #     if 'softmax' in layer.name:
    #         compressed_model.add(tf.keras.layers.Softmax(name='softmax'))
    #
    #
    # lr_schedule = piecewise_scheduler([91-EPOCHS_BEFORE_COMPRESSION, 136-EPOCHS_BEFORE_COMPRESSION], [1.0, 0.1, 0.01],
    #                                   base_rate=0.1, boundaries_as='epochs', num_images=num_train, batch_size=BATCH_SIZE)
    # # optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)
    # wd_schedule = piecewise_scheduler([91-EPOCHS_BEFORE_COMPRESSION, 136-EPOCHS_BEFORE_COMPRESSION], [1.0, 0.1, 0.01],
    #                                   base_rate=0.1 * 1e-4, boundaries_as='epochs', num_images=num_train, batch_size=BATCH_SIZE)
    # optimizer = SGDW(wd_schedule, learning_rate=lr_schedule, momentum=0.9)
    # loss = tf.keras.losses.SparseCategoricalCrossentropy()
    #
    # compressed_model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    # print(compressed_model.summary())
    # # print_ResNet(compressed_model)
    #
    # print(compressed_model.evaluate(test_dataset))
    #
    # # temp_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('stage2_block2').output)
    # # mu, V, e = tf_pca(temp_model.predict(val_dataset), 0.0001, num_as_threshold=True, conv=True, verbose=True)
    #
    # compressed_model.fit(train_dataset, initial_epoch=EPOCHS_BEFORE_COMPRESSION, epochs=TOTAL_EPOCHS,
    #                      steps_per_epoch=np.ceil(num_train/BATCH_SIZE),
    #                      validation_data=val_dataset, validation_freq=2)
    # print(compressed_model.evaluate(test_dataset))