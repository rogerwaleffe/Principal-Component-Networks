import os
import time
import numpy as np
import tensorflow as tf

from datetime import datetime

from src.pcns.imagenet.imagenet_helpers import input_fn, create_and_cache_tf_records, \
                                                                input_fn_given_dataset
from src.pcns.imagenet.resnet_imagenet_compression import compute_all_muVs

from src.pcns.resnet.resnet_model import ResNet, ResNetBottleneckBlockV1

from src.pcns.optimizer import SGDW, piecewise_scheduler
from src.pcns.layer_helpers import DensePCALayer, Conv2DPCALayer, Conv2DExplicitPadding
from src.pcns.layer_helpers import constant_initializer_from_tensor as cift



NUM_TRAIN = 1281167
NUM_VAL = 50000

# COSMOS
BASE_DATA_DIR = '/mnt/imagenet_tf_record_data'
CHECKPOINT_DIR = '/mnt/resnet'
#
# BASE_DATA_DIR = 'mnt/imagenet_tf_record_data'
# CHECKPOINT_DIR = 'mnt/model_data'
# AWS



def get_resnet50_model(compression_config):
    ResNetClass = ResNet(input_shape=(224, 224, 3), num_classes=1000, bottleneck=True, num_filters_at_start=64,
                         initial_kernel_size=7, initial_conv_strides=2, initial_pool_size=3, initial_pool_strides=2,
                         num_residual_blocks_per_stage=[3, 4, 6, 3], first_block_strides_per_stage=[1, 2, 2, 2],
                         kernel_size=3, project_first_residual=True, version='V1', data_format='channels_last',
                         compression_config=compression_config)

    return ResNetClass, ResNetClass.get_model()



def load_weights_and_possibly_optimizer(m, base_file_name):
    print("LOADING WEIGHTS")
    old_file_name = os.path.join(CHECKPOINT_DIR, '{}_{}.hdf5'.format(base_file_name, START_AFTER_EPOCH))
    if os.path.exists(old_file_name):
        print("LOADING WEIGHTS: found .hdf5 file format, using it to load weights, no optimizer state, model should "
              "already be compiled")
        m.load_weights(old_file_name)
    else:
        # to restore optimizer state 1) compile model with the same args and 2) call train_on_batch to
        # initialize optimizer variables before load_weights
        print("LOADING WEIGHTS: using TensorFlow format, ensure model was already compiled with the same args")
        init_dataset = input_fn_given_dataset(train_records_cached, is_training=True, shuffle_buffer=1,
                                              batch_size=BATCH_SIZE).take(1)
        print("LOADING WEIGHTS: training on one batch to initialize optimizer vars so they can be restored")
        # m.train_on_batch(init_dataset) # was giving errors regardless of whether fxn was called in or out of scope
        m.fit(init_dataset, verbose=0)

        new_file_name = os.path.join(CHECKPOINT_DIR, '{}_{}/'.format(base_file_name, START_AFTER_EPOCH))
        m.load_weights(new_file_name)
        # NOTE: this probably overrides any optimizer variables from already compiling?

        if update_optimizer:
            print("LOADING WEIGHTS: loaded optimizer, but recompiling with the given new optimizer")
            with strategy.scope():
                m.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])



if __name__ == "__main__":
    print("Running resnet.py: {}".format(datetime.now()))



    ID_IN = '_trainable_2'
    ID_OUT = '_trainable_2'
    START_AFTER_EPOCH = 18
    COMPRESSION_AFTER_EPOCH = 0
    TOTAL_EPOCHS = 18

    BATCH_SIZE = 256
    VERBOSE = 2 # 1 = progress bar, 2 = one line per epoch

    initial_lr = 0.1
    # initial_wd = initial_lr * 0.0001
    boundaries = [30, 60]
    decay_rates = [1.0, 0.1, 0.01]
    update_optimizer = False



    C_BATCH_SIZE = 256
    CENTER_CROPS_FOR_COMPRESSION = True
    SINGLE_SCALE_FOR_COMPRESSION = True
    NUM_SAMPLES_FOR_COMPRESSION = 75000
    # NOTE: need to be careful with constraints: if you want to kill columns make sure all necessary muVs are calculated
    # {}_{}: (conv1 input, conv1 output, conv2 input, conv2 output, conv3 input), conv0 input = conv1 input for {}_1
    # {}: conv0 and all conv3 outputs for stage
    N = None
    # cc = {
    #     'initial_conv': (N, N),
    #     '1_1': ('d64', N, 'd64', N, 'd64'), '1_2': ('d256', N, 'd64', N, 'd64'),
    #     '1_3': ('d256', N, 'd64', N, 'd64'),
    #     '1': N,
    #
    #     '2_1': ('d256', N, 'd128', N, 'd128'), '2_2': ('d512', N, 'd128', N, 'd128'),
    #     '2_3': ('d512', N, 'd128', N, 'd128'), '2_4': ('d512', N, 'd128', N, 'd128'),
    #     '2': N,
    #
    #     '3_1': ('d512', N, 'd256', N, 'd256'), '3_2': ('d1024', N, 'd256', N, 'd256'),
    #     '3_3': ('d1024', N, 'd256', N, 'd256'), '3_4': ('d1024', N, 'd256', N, 'd256'),
    #     '3_5': ('d1024', N, 'd256', N, 'd256'), '3_6': ('d1024', N, 'd256', N, 'd256'),
    #     '3': N,
    #
    #     '4_1': ('d1024', N, 'd512', N, 'd512'), '4_2': ('d2048', N, 'd512', N, 'd512'),
    #     '4_3': ('d2048', N, 'd512', N, 'd512'),
    #     '4': N,
    #
    #     'fc1': ('d2048', N),
    # }
    cc = {
        'initial_conv': (N, N),
        '1_1': (N, N, N, N, N), '1_2': (N, N, N, N, N),
        '1_3': (N, N, N, N, N),
        '1': N,

        '2_1': (N, N, N, N, N), '2_2': (N, N, N, N, N),
        '2_3': (N, N, N, N, N), '2_4': (N, N, N, N, N),
        '2': N,

        '3_1': (N, N, N, N, N), '3_2': (N, N, N, N, N),
        '3_3': (N, N, N, N, N), '3_4': (N, N, N, N, N),
        '3_5': (N, N, N, N, N), '3_6': (N, N, N, N, N),
        '3': N,

        '4_1': ('d128', N, 'd128', N, 'd128'), '4_2': ('d128', N, 'd128', N, 'd128'),
        '4_3': ('d128', N, 'd128', N, 'd128'),
        '4': N,

        'fc1': (N, N),
    }
    vc = {'use_all_muVs': True, 'weighted_row_sum': False,
          'max_image_size': 36864, 'compression_layer_type': 'AveragePooling', 'compute_all_from_scratch': True,
          'num_samples_for_compression': NUM_SAMPLES_FOR_COMPRESSION, 'c_batch_size': C_BATCH_SIZE,
          'center_crops': CENTER_CROPS_FOR_COMPRESSION, 'single_scale': SINGLE_SCALE_FOR_COMPRESSION}
          # 'add_dropout_to_compressed_model': True}



    base_weight_in_name = 'weights{}'.format(ID_IN)
    base_compressed_in_name = 'c_model{}'.format(ID_IN)

    base_weight_out_name = 'weights{}'.format(ID_OUT)
    base_compressed_out_name = 'c_model{}'.format(ID_OUT)


    strategy = tf.distribute.MirroredStrategy()
    train_records_cached = create_and_cache_tf_records(BASE_DATA_DIR, is_training=True)
    val_records_cached = create_and_cache_tf_records(BASE_DATA_DIR, is_training=False)
    val_dataset = input_fn_given_dataset(val_records_cached, is_training=False, shuffle_buffer=1, batch_size=BATCH_SIZE,
                                         resnet_preprocessing=True)


    lr_schedule = piecewise_scheduler([x-START_AFTER_EPOCH for x in boundaries], decay_rates, base_rate=initial_lr,
                                      boundaries_as='epochs', num_images=NUM_TRAIN, batch_size=BATCH_SIZE)
    # wd_schedule = piecewise_scheduler([x-START_AFTER_EPOCH for x in boundaries], decay_rates, base_rate=initial_wd,
    #                                   boundaries_as='epochs', num_images=NUM_TRAIN, batch_size=BATCH_SIZE)
    # optimizer = SGDW(wd_schedule, learning_rate=lr_schedule, momentum=0.9)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=False)
    # loss = tf.keras.losses.SparseCategoricalCrossentropy()
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True) # NOTE: label smoothing done by one-hot encoding

    # NOTE: this is just temporary
    # optimizer.weight_decay = 1.0 * initial_wd
    # optimizer.learning_rate = 1.0 * initial_lr



    if START_AFTER_EPOCH <= COMPRESSION_AFTER_EPOCH:
        # Need to at least compress the original model

        # Create the original model
        print("CREATING: original model, time {}".format(time.time()))

        with strategy.scope():
            ResNet50, model = get_resnet50_model(cc)
            model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy',
                                                                   tf.keras.metrics.TopKCategoricalAccuracy(k=5)])
        print(model.summary())
        print("CREATING: original model, done, time {}".format(time.time()))


        # Load weights if necessary # TODO: do we need to load weights in the scope? doesn't seem like it
        if START_AFTER_EPOCH > 0:
            load_weights_and_possibly_optimizer(model, base_weight_in_name)

            if VERBOSE == 1:
                model.evaluate(val_dataset, steps=np.ceil(NUM_VAL/BATCH_SIZE))
            else:
                score = model.evaluate(val_dataset, steps=np.ceil(NUM_VAL/BATCH_SIZE), verbose=0)
                print("TRAINING: before compression, Val Loss {:.5f}, Val Acc: {:.5f}".format(score[0], score[1]))



        # NOTE: Performance measurements for the original model
        # model.evaluate(val_dataset, steps=np.ceil(NUM_VAL/BATCH_SIZE))
        # model.evaluate(val_dataset, steps=np.ceil(NUM_VAL/BATCH_SIZE))
        # model.evaluate(val_dataset, steps=np.ceil(NUM_VAL/BATCH_SIZE))
        #
        # train_dataset = input_fn_given_dataset(train_records_cached, is_training=True, num_epochs=3,
        #                                        shuffle_buffer=NUM_TRAIN, batch_size=BATCH_SIZE,
        #                                        resnet_preprocessing=True)
        # print("TRAINING: time {}".format(time.time()))
        # model.fit(train_dataset, initial_epoch=START_AFTER_EPOCH, epochs=START_AFTER_EPOCH+3,
        #           steps_per_epoch=np.floor(NUM_TRAIN/BATCH_SIZE), verbose=VERBOSE)

        # @tf.function
        # def predict_function(data):
        #     return model.predict_step(data)
        # print(predict_function)
        #
        # input_data = tf.random.uniform([1, 224, 224, 3], dtype=tf.float32)
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
        # input_data = (tf.random.uniform([BATCH_SIZE, 224, 224, 3], dtype=tf.float32), tf.random.uniform([BATCH_SIZE, 1000], dtype=tf.float32))
        # run_meta = tf.compat.v1.RunMetadata()
        # opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        # flops = tf.compat.v1.profiler.profile(graph=train_function.get_concrete_function(input_data).graph,
        #                                       run_meta=run_meta, op_log=None, cmd='scope', options=opts)
        # print("Train function: {}".format(flops.total_float_ops))
        #
        # import sys
        # sys.exit(0)



        # Train before compression if necessary
        if START_AFTER_EPOCH < COMPRESSION_AFTER_EPOCH:
            print("TRAINING: before compression, time {}".format(time.time()))
            num_epochs = COMPRESSION_AFTER_EPOCH - START_AFTER_EPOCH
            train_dataset = input_fn_given_dataset(train_records_cached, is_training=True, num_epochs=num_epochs,
                                                   shuffle_buffer=NUM_TRAIN, batch_size=BATCH_SIZE,
                                                   resnet_preprocessing=True)

            # NOTE: this was just a test
            # for batch_x, batch_y in train_dataset.take(1):
            #     print(batch_x.shape)

            checkpoint = tf.keras.callbacks.ModelCheckpoint(
                os.path.join(CHECKPOINT_DIR, base_weight_out_name + '_{epoch:02d}/'), save_weights_only=True)

            # if VERBOSE == 1:
            #     model.evaluate(val_dataset, steps=np.ceil(NUM_VAL/BATCH_SIZE))
            # else:
            #     score = model.evaluate(val_dataset, steps=np.ceil(NUM_VAL/BATCH_SIZE), verbose=0)
            #     print("TRAINING: before compression, Val Loss {:.5f}, Val Acc: {:.5f}".format(score[0], score[1]))
            print("TRAINING: before compression, calling model.fit(), time {}".format(time.time()))
            model.fit(train_dataset, initial_epoch=START_AFTER_EPOCH, epochs=COMPRESSION_AFTER_EPOCH,
                      steps_per_epoch=np.floor(NUM_TRAIN/BATCH_SIZE), validation_data=val_dataset,
                      validation_steps=np.ceil(NUM_VAL/BATCH_SIZE), validation_freq=1,
                      callbacks=[checkpoint], verbose=VERBOSE)


        # Compress the model
        print("COMPRESSION: beginning, time {}".format(time.time()))
        print("COMPRESSION: compression config {}".format(cc))
        print("COMPRESSION: variable config {}".format(vc))
        # # TODO: add option here to try and use cached train records
        compression_dataset = input_fn(BASE_DATA_DIR, is_training=True, num_epochs=1, batch_size=C_BATCH_SIZE,
                                       center_crops_for_train=CENTER_CROPS_FOR_COMPRESSION,
                                       single_scale=SINGLE_SCALE_FOR_COMPRESSION, standardize_train=True)
        compression_dataset = compression_dataset.take(np.ceil(NUM_SAMPLES_FOR_COMPRESSION/C_BATCH_SIZE))

        c_boundaries = [x-COMPRESSION_AFTER_EPOCH for x in boundaries]
        c_lr_schedule = piecewise_scheduler(c_boundaries, decay_rates, base_rate=initial_lr, boundaries_as='epochs',
                                            num_images=NUM_TRAIN, batch_size=BATCH_SIZE)
        # c_wd_schedule = piecewise_scheduler(c_boundaries, decay_rates, base_rate=initial_wd, boundaries_as='epochs',
        #                                     num_images=NUM_TRAIN, batch_size=BATCH_SIZE)
        # c_optimizer = SGDW(c_wd_schedule, learning_rate=c_lr_schedule, momentum=0.9)
        c_optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=False)

        mu_V_e_list = compute_all_muVs(cc, vc, compression_dataset, model, strategy)
        with strategy.scope():
            new_model = ResNet50.get_compressed_model_given_muVs(vc, model, mu_V_e_list, verbose=True)
            new_model.compile(loss=loss, optimizer=c_optimizer, metrics=['accuracy'])


        print("COMPRESSION: saving compressed model, time {}".format(time.time()))
        json = new_model.to_json()
        json_file_name = os.path.join(CHECKPOINT_DIR, "{}.json".format(base_compressed_out_name))
        with open(json_file_name, "w") as fp:
            fp.write(json)
        new_model.save_weights(os.path.join(CHECKPOINT_DIR, "{}_{:02d}/".format(base_compressed_out_name,
                                                                                COMPRESSION_AFTER_EPOCH)))

        print(new_model.summary())
        if VERBOSE == 1:
            new_model.evaluate(val_dataset, steps=np.ceil(NUM_VAL/BATCH_SIZE))
        else:
            score = new_model.evaluate(val_dataset, steps=np.ceil(NUM_VAL/BATCH_SIZE), verbose=0)
            print("COMPRESSION, Val Loss {:.5f}, Val Acc: {:.5f}".format(score[0], score[1]))



        # @tf.function
        # def predict_function(data):
        #     return new_model.predict_step(data)
        # print(predict_function)
        #
        # input_data = tf.random.uniform([1, 224, 224, 3], dtype=tf.float32)
        # run_meta = tf.compat.v1.RunMetadata()
        # opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        # flops = tf.compat.v1.profiler.profile(graph=predict_function.get_concrete_function(input_data).graph,
        #                                       run_meta=run_meta, op_log=None, cmd='scope', options=opts)
        # print("Predict function: {}".format(flops.total_float_ops))
        #
        # @tf.function
        # def train_function(data):
        #     return new_model.train_step(data)
        # print(train_function)
        #
        # input_data = (tf.random.uniform([BATCH_SIZE, 224, 224, 3], dtype=tf.float32), tf.random.uniform([BATCH_SIZE, 1000], dtype=tf.float32))
        # run_meta = tf.compat.v1.RunMetadata()
        # opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        # flops = tf.compat.v1.profiler.profile(graph=train_function.get_concrete_function(input_data).graph,
        #                                       run_meta=run_meta, op_log=None, cmd='scope', options=opts)
        # print("Train function: {}".format(flops.total_float_ops))
        #
        # import sys
        # sys.exit(0)

    else:
        # Load compressed model and continue training it, TODO: does this loaded model train in parallel?
        print("LOADING: compressed model, time {}".format(time.time()))

        json_file_name = os.path.join(CHECKPOINT_DIR, "{}.json".format(base_compressed_in_name))
        with open(json_file_name, "r") as fp:
            json = fp.read()

        with strategy.scope():
            new_model = tf.keras.Sequential()
            new_model.add(tf.keras.layers.InputLayer(input_shape=(224, 224, 3), name='input'))

            json_layers = tf.keras.models.model_from_json(json, custom_objects={
                'Conv2DExplicitPadding': Conv2DExplicitPadding, 'constant_initializer_from_tensor': cift,
                'ResNetBottleneckBlockV1': ResNetBottleneckBlockV1, 'DensePCALayer': DensePCALayer})
            for layer in json_layers.layers:
                new_model.add(layer)

            # new_model = tf.keras.models.model_from_json(json, custom_objects={
            #     'Conv2DExplicitPadding': Conv2DExplicitPadding, 'constant_initializer_from_tensor': cift,
            #     'ResNetBottleneckBlockV1': ResNetBottleneckBlockV1, 'DensePCALayer': DensePCALayer,
            #     'Conv2DPCALayer': Conv2DPCALayer})

            new_model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
                                                                       # tf.keras.metrics.TopKCategoricalAccuracy(k=5)])

            # load_weights_and_possibly_optimizer(new_model, base_compressed_in_name) # NOTE: this was just a test
        print(new_model.summary())

        load_weights_and_possibly_optimizer(new_model, base_compressed_in_name)



        # NOTE: Performance measurements for the compressed model
        # new_model.evaluate(val_dataset, steps=np.ceil(NUM_VAL/BATCH_SIZE))
        # new_model.evaluate(val_dataset, steps=np.ceil(NUM_VAL/BATCH_SIZE))
        # new_model.evaluate(val_dataset, steps=np.ceil(NUM_VAL/BATCH_SIZE))
        #
        # train_dataset = input_fn_given_dataset(train_records_cached, is_training=True, num_epochs=3,
        #                                        shuffle_buffer=NUM_TRAIN, batch_size=BATCH_SIZE,
        #                                        resnet_preprocessing=True)
        # print("TRAINING: time {}".format(time.time()))
        # new_model.fit(train_dataset, initial_epoch=START_AFTER_EPOCH, epochs=START_AFTER_EPOCH+3,
        #           steps_per_epoch=np.floor(NUM_TRAIN/BATCH_SIZE), verbose=VERBOSE)

        # @tf.function
        # def predict_function(data):
        #     return new_model.predict_step(data)
        # print(predict_function)
        #
        # input_data = tf.random.uniform([1, 224, 224, 3], dtype=tf.float32)
        # run_meta = tf.compat.v1.RunMetadata()
        # opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        # flops = tf.compat.v1.profiler.profile(graph=predict_function.get_concrete_function(input_data).graph,
        #                                       run_meta=run_meta, op_log=None, cmd='scope', options=opts)
        # print("Predict function: {}".format(flops.total_float_ops))
        #
        # @tf.function
        # def train_function(data):
        #     return new_model.train_step(data)
        # print(train_function)
        #
        # input_data = (tf.random.uniform([256, 224, 224, 3], dtype=tf.float32), tf.random.uniform([256, 1000], dtype=tf.float32))
        # run_meta = tf.compat.v1.RunMetadata()
        # opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        # flops = tf.compat.v1.profiler.profile(graph=train_function.get_concrete_function(input_data).graph,
        #                                       run_meta=run_meta, op_log=None, cmd='scope', options=opts)
        # print("Train function: {}".format(flops.total_float_ops))
        #
        # import sys
        # sys.exit(0)



        if VERBOSE == 1:
            new_model.evaluate(val_dataset, steps=np.ceil(NUM_VAL/BATCH_SIZE))
        else:
            score = new_model.evaluate(val_dataset, steps=np.ceil(NUM_VAL/BATCH_SIZE), verbose=0)
            print("LOADING: compressed model, Val Loss {:.5f}, Val Acc: {:.5f}".format(score[0], score[1]))

        # NOTE: this is just temporary
        # new_model.get_layer('dropout_2').rate = 0.50
        # new_model.get_layer('dropout_3').rate = 0.50
        # optimizer.weight_decay = 1.0 * initial_wd
        # optimizer.learning_rate = 1.0 * initial_lr


    # Continue training
    print("TRAINING: after compression, time {}".format(time.time()))
    initial_epoch = COMPRESSION_AFTER_EPOCH if START_AFTER_EPOCH <= COMPRESSION_AFTER_EPOCH else START_AFTER_EPOCH
    num_epochs = TOTAL_EPOCHS - initial_epoch
    train_dataset = input_fn_given_dataset(train_records_cached, is_training=True, num_epochs=num_epochs,
                                           shuffle_buffer=NUM_TRAIN, batch_size=BATCH_SIZE, resnet_preprocessing=True)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(CHECKPOINT_DIR, base_compressed_out_name + '_{epoch:02d}/'), save_weights_only=True)

    new_model.fit(train_dataset, initial_epoch=initial_epoch, epochs=TOTAL_EPOCHS,
                  steps_per_epoch=np.floor(NUM_TRAIN/BATCH_SIZE), validation_data=val_dataset,
                  validation_steps=np.ceil(NUM_VAL/BATCH_SIZE), validation_freq=1,
                  callbacks=[checkpoint], verbose=VERBOSE)



