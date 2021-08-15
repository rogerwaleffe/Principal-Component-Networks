import time
import numpy as np
import tensorflow as tf

from src.pcns.dataset_helpers import get_a_cifar10_data_set
from src.pcns.train_helpers import vgg_train
from src.pcns.compression_helpers import temp_compression
from src.pcns.resnet.cifar10 import get_dataset, input_fn
from src.pcns.optimizer import piecewise_scheduler



def get_full_vgg16A_model(input_shape=(32, 32, 3), output_shape=10, dropout=True):
    CNN_REGULARIZER = tf.keras.regularizers.l2(l=0.00005)
    DENSE_REGULARIZER = tf.keras.regularizers.l2(l=0.00005)

    full_model = tf.keras.Sequential()
    full_model.add(tf.keras.layers.InputLayer(input_shape=input_shape, name='input'))

    # Convolution Layers
    num_filters = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
    # num_filters = [48, 64, 'M', 124, 128, 'M', 242, 229, 173, 'M', 31, 2, 'M', 226]
    # num_filters = [62, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 364, 147, 14, 'M', 2, 1, 432]

    conv_count = 1
    mp_count = 1
    for n in num_filters:
        if n == 'M':
            full_model.add(tf.keras.layers.MaxPooling2D((2, 2), name='mp{}'.format(mp_count)))
            mp_count += 1
        else:
            full_model.add(tf.keras.layers.Conv2D(n, (3, 3), padding='same', activation=None, use_bias=False,
                                                  kernel_initializer= tf.keras.initializers.
                                                      VarianceScaling(scale=2.0, mode='fan_out',
                                                                      distribution='untruncated_normal'),
                                                  kernel_regularizer=CNN_REGULARIZER,
                                                  name='conv{}'.format(conv_count)))
            full_model.add(tf.keras.layers.BatchNormalization(#gamma_initializer=tf.keras.initializers.constant(0.5),
                                                              name='bn{}'.format(conv_count)))
            full_model.add(tf.keras.layers.ReLU(name='relu{}'.format(conv_count)))
            conv_count += 1
    full_model.add(tf.keras.layers.GlobalAveragePooling2D(name='ap'))
    # Output Layer
    full_model.add(tf.keras.layers.Dense(output_shape, activation=None,
                                         #kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.01),
                                         kernel_regularizer=DENSE_REGULARIZER,
                                         bias_regularizer=DENSE_REGULARIZER, name='output'))
    # full_model.add(tf.keras.layers.Softmax(name='softmax'))

    return full_model



if __name__ == "__main__":
    cc = {'conv1': (None, None), 'conv2': (None, None), 'conv3': (None, None), 'conv4': (None, None),
          'conv5': (None, None), 'conv6': (None, None), 'conv7': (None, None),
          'conv8': (64, None), 'conv9': (64, None), 'conv10': (64, None),
          'conv11': (32, None), 'conv12': (32, None), 'conv13': (32, None),
          'output': (None, None)}



    NUM_VAL = 5000
    NUM_SAMPLES_FOR_COMPRESSION = 5000
    EPOCHS_BEFORE_COMPRESSION = 1
    TOTAL_EPOCHS = 160
    VERBOSE = 1  # 1 = progress bar, 2 = one line per epoch

    NUM_TEST = 10000
    BATCH_SIZE = 256



    # Get train/test data sets
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = get_dataset(NUM_VAL)
    num_train = train_x.shape[0]

    train_dataset = input_fn(train_x, train_y, num_epochs=EPOCHS_BEFORE_COMPRESSION, num_images=num_train,
                             batch_size=BATCH_SIZE, is_training=True)
    val_dataset = input_fn(val_x, val_y, batch_size=BATCH_SIZE)
    test_dataset = input_fn(test_x, test_y, batch_size=BATCH_SIZE)


    model = get_full_vgg16A_model()
    lr_schedule = piecewise_scheduler([80, 120], [1.0, 0.1, 0.01], base_rate=0.1, boundaries_as='epochs',
                                      num_images=num_train, batch_size=BATCH_SIZE)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    print(model.summary())



    # Model FLOPs measurements
    # @tf.function
    # def predict_function(data):
    #     return model.predict_step(data)
    # print(predict_function)
    #
    # input_data = tf.random.uniform([1, 32, 32, 3], dtype=tf.float32)
    # run_meta = tf.compat.v1.RunMetadata()
    # opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    # flops = tf.compat.v1.profiler.profile(graph=predict_function.get_concrete_function(input_data).graph,
    #                                       run_meta=run_meta, op_log=None, cmd='scope', options=opts)
    # print("Predict function: {}".format(flops.total_float_ops))
    #
    # @tf.function
    # def train_function(data):
    #     return model.train_step(data)
    # print(train_function)
    #
    # input_data = (tf.random.uniform([BATCH_SIZE, 32, 32, 3], dtype=tf.float32), tf.random.uniform([BATCH_SIZE, 1], dtype=tf.float32))
    # run_meta = tf.compat.v1.RunMetadata()
    # opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    # flops = tf.compat.v1.profiler.profile(graph=train_function.get_concrete_function(input_data).graph,
    #                                       run_meta=run_meta, op_log=None, cmd='scope', options=opts)
    # print("Train function: {}".format(flops.total_float_ops))
    #
    # import sys
    # sys.exit(0)



    # Train model
    # model.fit(train_dataset, epochs=EPOCHS_BEFORE_COMPRESSION, steps_per_epoch=np.floor(num_train/BATCH_SIZE),
    #           validation_data=val_dataset, validation_steps=np.ceil(NUM_VAL/BATCH_SIZE), validation_freq=1,
    #           verbose=VERBOSE)
    model.fit(train_dataset, epochs=EPOCHS_BEFORE_COMPRESSION, steps_per_epoch=np.floor(num_train/BATCH_SIZE),
              validation_data=test_dataset, validation_steps=np.ceil(NUM_TEST/BATCH_SIZE), validation_freq=1,
              verbose=VERBOSE)
    if VERBOSE == 1:
        model.evaluate(val_dataset, steps=np.ceil(NUM_VAL / BATCH_SIZE))
        model.evaluate(test_dataset, steps=np.ceil(NUM_TEST / BATCH_SIZE))
    else:
        score = model.evaluate(val_dataset, steps=np.ceil(NUM_VAL / BATCH_SIZE), verbose=0)
        print("TRAINING: before compression, Val Loss {:.5f}, Val Acc: {:.5f}".format(score[0], score[1]))
        score = model.evaluate(test_dataset, steps=np.ceil(NUM_TEST / BATCH_SIZE), verbose=0)
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
    compressed_model = temp_compression(cc, input_arr, model, nums_as='dimension', verbose=True)
    print("COMPRESSION: done getting compressed model, time {}".format(time.time()))

    boundaries = [80 - EPOCHS_BEFORE_COMPRESSION, 120 - EPOCHS_BEFORE_COMPRESSION]
    # boundaries = [80, 120]
    lr_schedule = piecewise_scheduler(boundaries, [1.0, 0.1, 0.01], base_rate=0.1, boundaries_as='epochs',
                                      num_images=num_train, batch_size=BATCH_SIZE)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    compressed_model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    print(compressed_model.summary())

    if VERBOSE == 1:
        compressed_model.evaluate(val_dataset, steps=np.ceil(NUM_VAL/BATCH_SIZE))
        compressed_model.evaluate(test_dataset, steps=np.ceil(NUM_TEST/BATCH_SIZE))
    else:
        score = compressed_model.evaluate(val_dataset, steps=np.ceil(NUM_VAL/BATCH_SIZE), verbose=0)
        print("TRAINING: before compression, Val Loss {:.5f}, Val Acc: {:.5f}".format(score[0], score[1]))
        score = compressed_model.evaluate(test_dataset, steps=np.ceil(NUM_TEST/BATCH_SIZE), verbose=0)
        print("TRAINING: before compression, Test Loss {:.5f}, Test Acc: {:.5f}".format(score[0], score[1]))



    # @tf.function
    # def predict_function(data):
    #     return compressed_model.predict_step(data)
    # print(predict_function)
    #
    # input_data = tf.random.uniform([1, 32, 32, 3], dtype=tf.float32)
    # run_meta = tf.compat.v1.RunMetadata()
    # opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    # flops = tf.compat.v1.profiler.profile(graph=predict_function.get_concrete_function(input_data).graph,
    #                                       run_meta=run_meta, op_log=None, cmd='scope', options=opts)
    # print("Predict function: {}".format(flops.total_float_ops))
    #
    # @tf.function
    # def train_function(data):
    #     return compressed_model.train_step(data)
    # print(train_function)
    #
    # input_data = (
    # tf.random.uniform([BATCH_SIZE, 32, 32, 3], dtype=tf.float32), tf.random.uniform([BATCH_SIZE, 1], dtype=tf.float32))
    # run_meta = tf.compat.v1.RunMetadata()
    # opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    # flops = tf.compat.v1.profiler.profile(graph=train_function.get_concrete_function(input_data).graph,
    #                                       run_meta=run_meta, op_log=None, cmd='scope', options=opts)
    # print("Train function: {}".format(flops.total_float_ops))
    #
    # import sys
    # sys.exit(0)



    # Continue training
    train_dataset = input_fn(train_x, train_y, num_epochs=TOTAL_EPOCHS - EPOCHS_BEFORE_COMPRESSION,
                             num_images=num_train, batch_size=BATCH_SIZE, is_training=True)
    # train_dataset = input_fn(train_x, train_y, num_epochs=TOTAL_EPOCHS,
    #                          num_images=num_train, batch_size=BATCH_SIZE, is_training=True)

    # compressed_model.fit(train_dataset, initial_epoch=EPOCHS_BEFORE_COMPRESSION,
    #                      epochs=TOTAL_EPOCHS, steps_per_epoch=np.floor(num_train/BATCH_SIZE),
    #                      validation_data=val_dataset, validation_steps=np.ceil(NUM_VAL/BATCH_SIZE), validation_freq=1,
    #                      verbose=VERBOSE)
    compressed_model.fit(train_dataset, initial_epoch=EPOCHS_BEFORE_COMPRESSION,
                         epochs=TOTAL_EPOCHS, steps_per_epoch=np.floor(num_train/BATCH_SIZE),
                         validation_data=test_dataset, validation_steps=np.ceil(NUM_TEST/BATCH_SIZE), validation_freq=1,
                         verbose=VERBOSE)
    # compressed_model.fit(train_dataset, epochs=TOTAL_EPOCHS, steps_per_epoch=np.floor(num_train/BATCH_SIZE),
    #                      validation_data=test_dataset, validation_steps=np.ceil(NUM_TEST/BATCH_SIZE),
    #                      validation_freq=1, verbose=VERBOSE)

    if VERBOSE == 1:
        compressed_model.evaluate(val_dataset, steps=np.ceil(NUM_VAL / BATCH_SIZE))
        compressed_model.evaluate(test_dataset, steps=np.ceil(NUM_TEST / BATCH_SIZE))
    else:
        score = compressed_model.evaluate(val_dataset, steps=np.ceil(NUM_VAL / BATCH_SIZE), verbose=0)
        print("TRAINING: before compression, Val Loss {:.5f}, Val Acc: {:.5f}".format(score[0], score[1]))
        score = compressed_model.evaluate(test_dataset, steps=np.ceil(NUM_TEST / BATCH_SIZE), verbose=0)
        print("TRAINING: before compression, Test Loss {:.5f}, Test Acc: {:.5f}".format(score[0], score[1]))