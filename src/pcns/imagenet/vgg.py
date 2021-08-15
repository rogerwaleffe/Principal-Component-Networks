import os
import time
import numpy as np
import tensorflow as tf

from datetime import datetime

from src.pcns.imagenet.imagenet_helpers import input_fn, create_and_cache_tf_records, \
                                                                input_fn_given_dataset
from src.pcns.imagenet.imagenet_compression import compute_all_muVs, imagenet_vgg_compression

from src.pcns.optimizer import SGDW, piecewise_scheduler
from src.pcns.layer_helpers import DensePCALayer, Conv2DPCALayer
from src.pcns.layer_helpers import constant_initializer_from_tensor as cift



NUM_TRAIN = 1281167
NUM_VAL = 50000

# BASE_DATA_DIR = '/mnt/imagenet_tf_record_data'
# CHECKPOINT_DIR = '/mnt'
BASE_DATA_DIR = 'mnt/imagenet_tf_record_data'
CHECKPOINT_DIR = 'mnt/model_data'



def get_full_vgg19_model(input_shape=(224, 224, 3), output_shape=1000, dropout=True):
    full_model = tf.keras.Sequential()
    full_model.add(tf.keras.layers.InputLayer(input_shape=input_shape, name='input'))

    # Convolution Layers
    full_model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='conv1', data_format='channels_last'))
    full_model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='conv2', data_format='channels_last'))
    full_model.add(tf.keras.layers.MaxPooling2D((2, 2), name='mp1'))
    full_model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='conv3', data_format='channels_last'))
    full_model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='conv4', data_format='channels_last'))
    full_model.add(tf.keras.layers.MaxPooling2D((2, 2), name='mp2'))
    full_model.add(tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu', name='conv5', data_format='channels_last'))
    full_model.add(tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu', name='conv6', data_format='channels_last'))
    full_model.add(tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu', name='conv7', data_format='channels_last'))
    full_model.add(tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu', name='conv8', data_format='channels_last'))
    full_model.add(tf.keras.layers.MaxPooling2D((2, 2), name='mp3'))
    full_model.add(tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu', name='conv9', data_format='channels_last'))
    full_model.add(tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu', name='conv10', data_format='channels_last'))
    full_model.add(tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu', name='conv11', data_format='channels_last'))
    full_model.add(tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu', name='conv12', data_format='channels_last'))
    full_model.add(tf.keras.layers.MaxPooling2D((2, 2), name='mp4'))
    full_model.add(tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu', name='conv13', data_format='channels_last'))
    full_model.add(tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu', name='conv14', data_format='channels_last'))
    full_model.add(tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu', name='conv15', data_format='channels_last'))
    full_model.add(tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu', name='conv16', data_format='channels_last'))
    full_model.add(tf.keras.layers.MaxPooling2D((2, 2), name='mp5'))

    # Fully Connected Layers
    full_model.add(tf.keras.layers.Flatten(name='flatten'))
    if dropout:
        full_model.add(tf.keras.layers.Dropout(0.5))
    full_model.add(tf.keras.layers.Dense(4096, activation='relu', name='fc1'))
    if dropout:
        full_model.add(tf.keras.layers.Dropout(0.5))
    full_model.add(tf.keras.layers.Dense(4096, activation='relu', name='fc2'))

    # Output Layer
    full_model.add(tf.keras.layers.Dense(output_shape, activation=None, name='output'))
    full_model.add(tf.keras.layers.Softmax(name='softmax'))

    return full_model



def get_full_vgg16_model(input_shape=(224, 224, 3), output_shape=1000, dropout=True):
    full_model = tf.keras.Sequential()
    full_model.add(tf.keras.layers.InputLayer(input_shape=input_shape, name='input'))

    # Convolution Layers
    full_model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='conv1', data_format='channels_last'))
    full_model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='conv2', data_format='channels_last'))
    full_model.add(tf.keras.layers.MaxPooling2D((2, 2), name='mp1'))
    full_model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='conv3', data_format='channels_last'))
    full_model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='conv4', data_format='channels_last'))
    full_model.add(tf.keras.layers.MaxPooling2D((2, 2), name='mp2'))
    full_model.add(tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu', name='conv5', data_format='channels_last'))
    full_model.add(tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu', name='conv6', data_format='channels_last'))
    full_model.add(tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu', name='conv7', data_format='channels_last'))
    full_model.add(tf.keras.layers.MaxPooling2D((2, 2), name='mp3'))
    full_model.add(tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu', name='conv8', data_format='channels_last'))
    full_model.add(tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu', name='conv9', data_format='channels_last'))
    full_model.add(tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu', name='conv10', data_format='channels_last'))
    full_model.add(tf.keras.layers.MaxPooling2D((2, 2), name='mp4'))
    full_model.add(tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu', name='conv11', data_format='channels_last'))
    full_model.add(tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu', name='conv12', data_format='channels_last'))
    full_model.add(tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu', name='conv13', data_format='channels_last'))
    full_model.add(tf.keras.layers.MaxPooling2D((2, 2), name='mp5'))

    # Fully Connected Layers
    full_model.add(tf.keras.layers.Flatten(name='flatten'))
    if dropout:
        full_model.add(tf.keras.layers.Dropout(0.5))
    full_model.add(tf.keras.layers.Dense(4096, activation='relu', name='fc1'))
    if dropout:
        full_model.add(tf.keras.layers.Dropout(0.5))
    full_model.add(tf.keras.layers.Dense(4096, activation='relu', name='fc2'))

    # Output Layer
    full_model.add(tf.keras.layers.Dense(output_shape, activation=None, name='output'))
    full_model.add(tf.keras.layers.Softmax(name='softmax'))

    return full_model



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
    print("Running vgg.py: {}".format(datetime.now()))



    ID_IN = '_full_model_1'
    ID_OUT = '_test'
    START_AFTER_EPOCH = 70
    COMPRESSION_AFTER_EPOCH = 70
    TOTAL_EPOCHS = 70

    BATCH_SIZE = 256
    MODEL_WITH_DROPOUT = True
    VERBOSE = 1 # 1 = progress bar, 2 = one line per epoch

    initial_lr = 0.01
    initial_wd = initial_lr * 5*10e-4
    boundaries = [50, 60]
    decay_rates = [1.0, 0.1, 0.01]
    update_optimizer = False

    C_BATCH_SIZE = 256
    CENTER_CROPS_FOR_COMPRESSION = True
    NUM_SAMPLES_FOR_COMPRESSION = 50000
    cc = {
        'conv1': (None, None), 'conv2': (None, None), 'conv3': (None, None), 'conv4': (None, None),
        'conv5': (None, None), 'conv6': (None, None), 'conv7': (None, None), 'conv8': (None, None),
        'conv9': (None, None), 'conv10': (None, None), 'conv11': (None, None), 'conv12': (None, None),
        'conv13': (None, None), 'conv14': (None, None), 'conv15': (None, None), 'conv16': (None, None),
        'fc1': ('d350', None), 'fc2': ('d400', None), 'output': (None, None),
    }
    vc = {'max_image_size': 25088, 'compression_layer_type': 'AveragePooling', 'compute_all_from_scratch': True,
          'num_samples_for_compression': NUM_SAMPLES_FOR_COMPRESSION, 'c_batch_size': C_BATCH_SIZE,
          'add_dropout_to_compressed_model': True}



    base_weight_in_name = 'weights{}'.format(ID_IN) if MODEL_WITH_DROPOUT is True else 'weights_nd{}'.format(ID_IN)
    base_compressed_in_name = 'c_model{}'.format(ID_IN)

    base_weight_out_name = 'weights{}'.format(ID_OUT) if MODEL_WITH_DROPOUT is True else 'weights_nd{}'.format(ID_OUT)
    base_compressed_out_name = 'c_model{}'.format(ID_OUT)


    strategy = tf.distribute.MirroredStrategy()
    train_records_cached = create_and_cache_tf_records(BASE_DATA_DIR, is_training=True)
    val_records_cached = create_and_cache_tf_records(BASE_DATA_DIR, is_training=False)
    val_dataset = input_fn_given_dataset(val_records_cached, is_training=False, shuffle_buffer=1, batch_size=BATCH_SIZE)


    lr_schedule = piecewise_scheduler([x-START_AFTER_EPOCH for x in boundaries], decay_rates, base_rate=initial_lr,
                                      boundaries_as='epochs', num_images=NUM_TRAIN, batch_size=BATCH_SIZE)
    wd_schedule = piecewise_scheduler([x-START_AFTER_EPOCH for x in boundaries], decay_rates, base_rate=initial_wd,
                                      boundaries_as='epochs', num_images=NUM_TRAIN, batch_size=BATCH_SIZE)
    optimizer = SGDW(wd_schedule, learning_rate=lr_schedule, momentum=0.9)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()

    # NOTE: this is just temporary
    # optimizer.weight_decay = 1.0 * initial_wd
    # optimizer.learning_rate = 1.0 * initial_lr



    if START_AFTER_EPOCH <= COMPRESSION_AFTER_EPOCH:
        # Need to at least compress the original model

        # Create the original model
        print("CREATING: original model, time {}".format(time.time()))

        with strategy.scope():
            model = get_full_vgg19_model(dropout=MODEL_WITH_DROPOUT)
            model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy',
                                                                   tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)])
        print(model.summary())
        print("CREATING: original model, done, time {}".format(time.time()))


        # Load weights if necessary # TODO: do we need to load weights in the scope? doesn't seem like it
        if START_AFTER_EPOCH > 0:
            load_weights_and_possibly_optimizer(model, base_weight_in_name)



        # NOTE: Performance and Evaluation of Full Model (added TopK above to model compilation)
        # model.evaluate(val_dataset, steps=np.ceil(NUM_VAL/BATCH_SIZE))
        # train_dataset = input_fn_given_dataset(train_records_cached, is_training=True, num_epochs=3,
        #                                        shuffle_buffer=NUM_TRAIN, batch_size=BATCH_SIZE)
        # print("\nTraining 5 epochs, time {}".format(time.time()))
        # model.fit(train_dataset, epochs=3, steps_per_epoch=np.floor(NUM_TRAIN/BATCH_SIZE), verbose=2)
        # import sys
        # sys.exit(0)



        # Train before compression if necessary
        if START_AFTER_EPOCH < COMPRESSION_AFTER_EPOCH:
            print("TRAINING: before compression, time {}".format(time.time()))
            num_epochs = COMPRESSION_AFTER_EPOCH - START_AFTER_EPOCH
            train_dataset = input_fn_given_dataset(train_records_cached, is_training=True, num_epochs=num_epochs,
                                                   shuffle_buffer=NUM_TRAIN, batch_size=BATCH_SIZE)

            checkpoint = tf.keras.callbacks.ModelCheckpoint(
                os.path.join(CHECKPOINT_DIR, base_weight_out_name + '_{epoch:02d}/'), save_weights_only=True)

            if VERBOSE == 1:
                model.evaluate(val_dataset, steps=np.ceil(NUM_VAL/BATCH_SIZE))
            else:
                score = model.evaluate(val_dataset, steps=np.ceil(NUM_VAL/BATCH_SIZE), verbose=0)
                print("TRAINING: before compression, Val Loss {:.5f}, Val Acc: {:.5f}".format(score[0], score[1]))
            print("TRAINING: before compression, calling model.fit(), time {}".format(time.time()))
            model.fit(train_dataset, initial_epoch=START_AFTER_EPOCH, epochs=COMPRESSION_AFTER_EPOCH,
                      steps_per_epoch=np.floor(NUM_TRAIN/BATCH_SIZE), validation_data=val_dataset,
                      validation_steps=np.ceil(NUM_VAL/BATCH_SIZE), validation_freq=1,
                      callbacks=[checkpoint], verbose=VERBOSE)


        # Compress the model
        print("COMPRESSION: beginning, time {}".format(time.time()))
        print("COMPRESSION: compression config {}".format(cc))
        print("COMPRESSION: variable config {}".format(vc))
        # TODO: add option here to try and use cached train records
        compression_dataset = input_fn(BASE_DATA_DIR, is_training=True, num_epochs=1, batch_size=C_BATCH_SIZE,
                                       center_crops_for_train=CENTER_CROPS_FOR_COMPRESSION)
        compression_dataset = compression_dataset.take(np.ceil(NUM_SAMPLES_FOR_COMPRESSION/C_BATCH_SIZE))

        c_boundaries = [x-COMPRESSION_AFTER_EPOCH for x in boundaries]
        c_lr_schedule = piecewise_scheduler(c_boundaries, decay_rates, base_rate=initial_lr, boundaries_as='epochs',
                                            num_images=NUM_TRAIN, batch_size=BATCH_SIZE)
        c_wd_schedule = piecewise_scheduler(c_boundaries, decay_rates, base_rate=initial_wd, boundaries_as='epochs',
                                            num_images=NUM_TRAIN, batch_size=BATCH_SIZE)
        c_optimizer = SGDW(c_wd_schedule, learning_rate=c_lr_schedule, momentum=0.9)

        mu_V_e_list = compute_all_muVs(cc, vc, compression_dataset, model, strategy)
        new_model = imagenet_vgg_compression(cc, vc, model, mu_V_e_list, strategy, c_optimizer, verbose=True)


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

    else:
        # Load compressed model and continue training it, TODO: does this loaded model train in parallel?
        print("LOADING: compressed model, time {}".format(time.time()))

        json_file_name = os.path.join(CHECKPOINT_DIR, "{}.json".format(base_compressed_in_name))
        # json_file_name = os.path.join(CHECKPOINT_DIR, "c_model_official_1_trial2.json")
        with open(json_file_name, "r") as fp:
            json = fp.read()

        with strategy.scope():
            new_model = tf.keras.models.model_from_json(json, custom_objects={'DensePCALayer': DensePCALayer,
                                                                              'Conv2DPCALayer': Conv2DPCALayer,
                                                                              'constant_initializer_from_tensor': cift})
            new_model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy',
                                                                       tf.keras.metrics.SparseTopKCategoricalAccuracy(
                                                                           k=5)])
            # load_weights_and_possibly_optimizer(new_model, base_compressed_in_name) # NOTE: this was a test
        print(new_model.summary())

        load_weights_and_possibly_optimizer(new_model, base_compressed_in_name)

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
                                           shuffle_buffer=NUM_TRAIN, batch_size=BATCH_SIZE)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(CHECKPOINT_DIR, base_compressed_out_name + '_{epoch:02d}/'), save_weights_only=True)

    new_model.fit(train_dataset, initial_epoch=initial_epoch, epochs=TOTAL_EPOCHS,
                  steps_per_epoch=np.floor(NUM_TRAIN/BATCH_SIZE), validation_data=val_dataset,
                  validation_steps=np.ceil(NUM_VAL/BATCH_SIZE), validation_freq=1,
                  callbacks=[checkpoint], verbose=VERBOSE)
