import os
import copy
import numpy as np
import tensorflow as tf

from datetime import datetime

from src.pcns.dataset_helpers import get_a_cifar10_data_set # create/load_cifar10_data_set
from src.pcns.train_helpers import vgg_train

DATA_DIRECTORY = 'data/'



def get_full_model(input_shape=(32, 32, 3), output_shape=10):
    full_model = tf.keras.Sequential()
    full_model.add(tf.keras.layers.InputLayer(input_shape=input_shape, name='input'))

    # Convolution Layers
    full_model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='conv1'))
    full_model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='conv2'))
    full_model.add(tf.keras.layers.MaxPooling2D((2, 2), name='mp1'))
    full_model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='conv3'))
    full_model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='conv4'))
    full_model.add(tf.keras.layers.MaxPooling2D((2, 2), name='mp2'))
    full_model.add(tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu', name='conv5'))
    full_model.add(tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu', name='conv6'))
    full_model.add(tf.keras.layers.MaxPooling2D((2, 2), name='mp3'))

    # Fully Connected Layers
    full_model.add(tf.keras.layers.Flatten(name='flatten'))
    full_model.add(tf.keras.layers.Dense(256, activation='relu', name='fc1'))
    full_model.add(tf.keras.layers.Dense(256, activation='relu', name='fc2'))

    # Output Layer
    full_model.add(tf.keras.layers.Dense(output_shape, activation=None, name='output'))
    full_model.add(tf.keras.layers.Softmax(name='softmax'))

    # Optimizer
    full_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                       optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

    return full_model



def baseline():
    baseline_config = {'compression_config': {'conv1': (None, None), 'conv2': (None, None), 'conv3': (None, None),
                                              'conv4': (None, None), 'conv5': (None, None), 'conv6': (None, None),
                                              'fc1': (None, None), 'fc2': (None, None), 'output': (None, None)},
                      'nums_as': 'dimension',
                      'num_samples_for_compression': 5000, 'compression_after_epoch': 20, 'total_epochs': 20,
                      'batch_size': 60}

    num_to_average = 5
    for _ in range(num_to_average):
        vgg_train(baseline_config, get_full_model, get_a_cifar10_data_set, verbose=1)



def individual_layer_pca_experiments():
    generic_config = {'compression_config': {'conv1': (None, None), 'conv2': (None, None), 'conv3': (None, None),
                                             'conv4': (None, None), 'conv5': (None, None), 'conv6': (None, None),
                                             'fc1': (None, None), 'fc2': (None, None), 'output': (None, None)},
                      'nums_as': 'threshold',
                      'num_samples_for_compression': 5000, 'compression_after_epoch': None, 'total_epochs': 20,
                      'batch_size': 60}

    compression_after_epoch = [1, 2, 3, 4, 5]

    conv2 = {'conv2': [1e-5, 5e-5, 1e-4, 4e-4, 1e-3, 2.5e-3, 5e-3]}
    conv3 = {'conv3': [1e-5, 5e-5, 1e-4, 4e-4, 1e-3, 2.5e-3, 5e-3]}
    conv4 = {'conv4': [1e-5, 5e-5, 1e-4, 4e-4, 1e-3, 2.5e-3, 5e-3]}
    conv5 = {'conv5': [1e-5, 5e-5, 1e-4, 4e-4, 1e-3, 2.5e-3, 5e-3]}
    conv6 = {'conv6': [1e-5, 5e-5, 1e-4, 4e-4, 1e-3, 2.5e-3, 5e-3]}
    fc1 = {'fc1': [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 2.5e-1, 5e-1]}
    fc2 = {'fc2': [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 2.5e-1, 5e-1]}
    output = {'output': [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 2.5e-1, 5e-1]}

    all_dicts = [conv2, conv3, conv4, conv5, conv6, fc1, fc2, output]
    num_to_average = 5

    for epoch in compression_after_epoch:
        for single_dict in all_dicts:
            layer = list(single_dict.keys())[0]
            values = single_dict[layer]
            for val in values:
                single_run_config = copy.deepcopy(generic_config)
                single_run_config['compression_after_epoch'] = epoch
                single_run_config['compression_config'][layer] = (val, None)
                for _ in range(num_to_average):
                    vgg_train(single_run_config, get_full_model, get_a_cifar10_data_set, verbose=1)



def all_layer_pca_experiments():
    generic_config = {'compression_config': {'conv1': (None, None), 'conv2': (None, None), 'conv3': (None, None),
                                             'conv4': (None, None), 'conv5': (None, None), 'conv6': (None, None),
                                             'fc1': (None, None), 'fc2': (None, None), 'output': (None, None)},
                      'nums_as': 'threshold',
                      'num_samples_for_compression': 5000, 'compression_after_epoch': None, 'total_epochs': 20,
                      'batch_size': 60}

    compression_after_epoch = [1, 2, 3, 4, 5]

    conv_fc_var = [(1e-5, 1e-3), (5e-5, 5e-3), (1e-4, 1e-2), (5e-4, 5e-2), (1e-3, 1e-1), (2.5e-3, 2.5e-1), (5e-3, 5e-1),
                   (1e-5, 5e-1), (5e-5, 2.5e-1), (1e-4, 1e-1),
                   (5e-3, 1e-3), (2.5e-3, 5e-3), (1e-3, 1e-2)]
    num_to_average = 5

    for epoch in compression_after_epoch:
        for val in conv_fc_var:
            single_run_config = copy.deepcopy(generic_config)
            single_run_config['compression_after_epoch'] = epoch
            for layer in list(single_run_config['compression_config'].keys()):
                if layer == 'conv1':
                    continue
                elif 'conv' in layer:
                    single_run_config['compression_config'][layer] = (val[0], None)
                else:
                    single_run_config['compression_config'][layer] = (val[1], None)
            for _ in range(num_to_average):
                vgg_train(single_run_config, get_full_model, get_a_cifar10_data_set, verbose=1)



if __name__ == "__main__":
    print("Running conv6.py: {}".format(datetime.now()))

    # if os.path.exists(DATA_DIRECTORY+'cifar10_xTrain.npy'):
    #     (train_x, train_y), (val_x, val_y), (test_x, test_y) = load_cifar10_data_set(DATA_DIRECTORY)
    # else:
    #     (train_x, train_y), (val_x, val_y), (test_x, test_y) = create_cifar10_data_set(DATA_DIRECTORY)
    #
    # train_x = np.array(train_x, dtype=np.float32)
    # val_x = np.array(val_x, dtype=np.float32)
    # test_x = np.array(test_x, dtype=np.float32)
    #
    # data = ((train_x, train_y), (val_x, val_y), (test_x, test_y))



    config = {'compression_config': {'conv1': (None, 32), 'conv2': (20, 32), 'conv3': (30, 64),
                                     'conv4': (60, 64), 'conv5': (60, 128), 'conv6': (100, 128),
                                     'fc1': (35, 70), 'fc2': (20, 170), 'output': (15, None)},
              'nums_as': 'dimension',
              'num_samples_for_compression': 5000, 'compression_after_epoch': 2, 'total_epochs': 20, 'batch_size': 60}
    # config = {'compression_config': {'conv1': (None, 64), 'conv2': (64, 64), 'conv3': (64, 128),
    #                                  'conv4': (128, 128), 'conv5': (128, 256), 'conv6': (256, 256),
    #                                  'fc1': (4096, 256), 'fc2': (256, 256), 'output': (256, None)},
    #           'nums_as': 'dimension',
    #           'num_samples_for_compression': 5000, 'compression_after_epoch': 2, 'total_epochs': 20, 'batch_size': 60}

    vgg_train(config, get_full_model, get_a_cifar10_data_set, verbose=2)



    # individual_layer_pca_experiments()
    # all_layer_pca_experiments()
    # baseline()