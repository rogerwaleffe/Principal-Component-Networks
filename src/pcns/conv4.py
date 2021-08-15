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
                                              'conv4': (None, None), 'fc1': (None, None), 'fc2': (None, None),
                                              'output': (None, None)},
                      'nums_as': 'dimension',
                      'num_samples_for_compression': 5000, 'compression_after_epoch': 20, 'total_epochs': 20,
                      'batch_size': 60}

    num_to_average = 5
    for _ in range(num_to_average):
        vgg_train(baseline_config, get_full_model, get_a_cifar10_data_set, verbose=1)



def individual_layer_pca_experiments():
    generic_config = {'compression_config': {'conv1': (None, None), 'conv2': (None, None), 'conv3': (None, None),
                                             'conv4': (None, None), 'fc1': (None, None), 'fc2': (None, None),
                                             'output': (None, None)},
                      'nums_as': 'threshold',
                      'num_samples_for_compression': 5000, 'compression_after_epoch': None, 'total_epochs': 20,
                      'batch_size': 60}

    compression_after_epoch = [1, 2, 3, 4, 5]

    conv2 = {'conv2': [1e-5, 5e-5, 1e-4, 4e-4, 1e-3, 2.5e-3, 5e-3]}
    conv3 = {'conv3': [1e-5, 5e-5, 1e-4, 4e-4, 1e-3, 2.5e-3, 5e-3]}
    conv4 = {'conv4': [1e-5, 5e-5, 1e-4, 4e-4, 1e-3, 2.5e-3, 5e-3]}
    fc1 = {'fc1': [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 2.5e-1, 5e-1]}
    fc2 = {'fc2': [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 2.5e-1, 5e-1]}
    output = {'output': [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 2.5e-1, 5e-1]}

    all_dicts = [conv2, conv3, conv4, fc1, fc2, output]
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



def just_fc1_experiments():
    generic_config = {'compression_config': {'conv1': (None, None), 'conv2': (None, None), 'conv3': (None, None),
                                             'conv4': (None, None), 'fc1': (0.1, None), 'fc2': (None, None),
                                             'output': (None, None)},
                      'nums_as': 'threshold',
                      'num_samples_for_compression': 5000, 'compression_after_epoch': None, 'total_epochs': 20,
                      'batch_size': 60}

    compression_after_epoch = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    num_to_average = 5
    for epoch in compression_after_epoch:
        single_run_config = copy.deepcopy(generic_config)
        single_run_config['compression_after_epoch'] = epoch
        for _ in range(num_to_average):
            vgg_train(single_run_config, get_full_model, get_a_cifar10_data_set, verbose=1)



def all_layer_pca_experiments():
    generic_config = {'compression_config': {'conv1': (None, None), 'conv2': (None, None), 'conv3': (None, None),
                                             'conv4': (None, None), 'fc1': (None, None), 'fc2': (None, None),
                                             'output': (None, None)},
                      'nums_as': 'threshold',
                      'num_samples_for_compression': 5000, 'compression_after_epoch': None, 'total_epochs': 20,
                      'batch_size': 60}

    compression_after_epoch = [1, 2, 3, 4, 5]

    # conv_fc_var = [(1e-5, 1e-3), (5e-5, 5e-3), (1e-4, 1e-2), (5e-4, 5e-2), (1e-3, 1e-1), (2.5e-3, 2.5e-1), (5e-3, 5e-1),
    #                (1e-5, 5e-1), (5e-5, 2.5e-1), (1e-4, 1e-1),
    #                (5e-3, 1e-3), (2.5e-3, 5e-3), (1e-3, 1e-2)]
    conv_fc_var = [(1e-4, 0.05), (1e-4, 0.1), (1e-3, 0.1), (1e-3, 0.2)]
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



def all_but_one_layer_pca_experiments():
    generic_config = {'compression_config': {'conv1': (None, None), 'conv2': (None, None), 'conv3': (None, None),
                                             'conv4': (None, None), 'fc1': (None, None), 'fc2': (None, None),
                                             'output': (None, None)},
                      'nums_as': 'threshold',
                      'num_samples_for_compression': 5000, 'compression_after_epoch': None, 'total_epochs': 20,
                      'batch_size': 60}

    # full_net_configs = [(0.0001, 0.1, 2), (0.001, 0.1, 3), (0.0001, 0.25, 4), (0.001, 0.25, 4)]
    full_net_configs = [(1e-4, 0.05, 3), (1e-4, 0.1, 3), (1e-3, 0.1, 3), (1e-3, 0.2, 3)]

    num_to_average = 5

    for conf in full_net_configs:
        single_run_config = copy.deepcopy(generic_config)
        single_run_config['compression_after_epoch'] = conf[-1]
        for layer in list(single_run_config['compression_config'].keys()):
            if layer == 'conv1':
                continue
            elif 'conv' in layer:
                single_run_config['compression_config'][layer] = (conf[0], None)
            else:
                single_run_config['compression_config'][layer] = (conf[1], None)

        # baseline for this config
        for _ in range(num_to_average):
            vgg_train(single_run_config, get_full_model, get_a_cifar10_data_set, verbose=1)

        # remove a layer
        for layer in list(single_run_config['compression_config'].keys()):
            if layer == 'conv1':
                continue
            else:
                temp_config = copy.deepcopy(single_run_config)
                temp_config['compression_config'][layer] = (None, None)
                for _ in range(num_to_average):
                    vgg_train(temp_config, get_full_model, get_a_cifar10_data_set, verbose=1)



if __name__ == "__main__":
    print("Running conv4.py: {}".format(datetime.now()))

    # if os.path.exists(DATA_DIRECTORY+'cifar10_xTrain.npy'):
    #     (train_x, train_y), (val_x, val_y), (test_x, test_y) = load_cifar10_data_set(DATA_DIRECTORY)
    # else:
    #     (train_x, train_y), (val_x, val_y), (test_x, test_y) = create_cifar10_data_set(DATA_DIRECTORY)

    # train_x = np.array(train_x, dtype=np.float32)
    # val_x = np.array(val_x, dtype=np.float32)
    # test_x = np.array(test_x, dtype=np.float32)

    # data = ((train_x, train_y), (val_x, val_y), (test_x, test_y))



    # config = {'compression_config': {'conv1': (None, None), 'conv2': (None, None), 'conv3': (None, None),
    #                                           'conv4': (None, None), 'fc1': (0.1, None), 'fc2': (None, None),
    #                                           'output': (None, None)},
    #           'nums_as': 'threshold',
    #           'num_samples_for_compression': 5000, 'compression_after_epoch': 20, 'total_epochs': 20,
    #           'batch_size': 60}
    # config = {'compression_config': {'conv1': (None, 0.2), 'conv2': (0.0002, 0.2), 'conv3': (0.0002, 0.2),
    #                                  'conv4': (0.0002, 0.2), 'fc1': (0.2, 0.2), 'fc2': (0.2, 0.2),
    #                                  'output': (0.2, None)},
    #           'nums_as': 'threshold',
    #           'num_samples_for_compression': 5000, 'compression_after_epoch': 20, 'total_epochs': 20, 'batch_size': 60}
    config = {'compression_config': {'conv1': (None, 40), 'conv2': (20, 50), 'conv3': (40, 100),
                                     'conv4': (80, 60), 'fc1': (50, 90), 'fc2': (40, 180),
                                     'output': (30, None)},
              'nums_as': 'dimension',
              'num_samples_for_compression': 5000, 'compression_after_epoch': 2, 'total_epochs': 20, 'batch_size': 60}
    # config = {'compression_config': {'conv1': (None, None), 'conv2': (20, None), 'conv3': (40, None),
    #                                  'conv4': (80, None), 'fc1': (50, None), 'fc2': (40, None),
    #                                  'output': (30, None)},
    #           'nums_as': 'dimension',
    #           'num_samples_for_compression': 5000, 'compression_after_epoch': 2, 'total_epochs': 20, 'batch_size': 60}

    # vgg_train(config, get_full_model, get_a_cifar10_data_set, verbose=2) measure_base_perf=False, measure_c_perf=False
    # for _ in range(5):
    #     vgg_train(config, get_full_model, get_a_cifar10_data_set, verbose=2)
    vgg_train(config, get_full_model, get_a_cifar10_data_set, verbose=2, measure_base_perf=False, measure_c_perf=False)



    # individual_layer_pca_experiments()
    # just_fc1_experiments()
    # all_layer_pca_experiments()
    # all_but_one_layer_pca_experiments()
    # baseline()




















# l_c1, l_c2, l_c3, l_c4, l_fc1, l_fc2, l_o = compression_v1(compression_config=train_config['compression_config'],
#                                                            input_arr=input_arr, overall_model=model,
#                                                            verbose=False if verbose == 0 else True)
# new_model = get_compressed_model(l_c1, l_c2, l_c3, l_c4, l_fc1, l_fc2, l_o)

# def get_compressed_model(l_c1, l_c2, l_c3, l_c4, l_fc1, l_fc2, l_o, input_shape=(32, 32, 3), output_shape=10):
#     compressed_model = tf.keras.Sequential()
#     compressed_model.add(tf.keras.layers.InputLayer(input_shape=input_shape, name='input'))
#
#     # Convolution Layers
#     compressed_model.add(Conv2DPCALayer(l_c1[0], (3, 3), l_c1[1], l_c1[2], zero_pad=((1, 1), (1, 1)),
#                                         kernel_initializer=cift(l_c1[3]), bias_initializer=cift(l_c1[4]),
#                                         activation='relu', name='conv1'))
#     compressed_model.add(Conv2DPCALayer(l_c2[0], (3, 3), l_c2[1], l_c2[2], zero_pad=((1, 1), (1, 1)),
#                                         kernel_initializer=cift(l_c2[3]), bias_initializer=cift(l_c2[4]),
#                                         activation='relu', name='conv2'))
#     compressed_model.add(tf.keras.layers.MaxPooling2D((2, 2), name='mp1'))
#     compressed_model.add(Conv2DPCALayer(l_c3[0], (3, 3), l_c3[1], l_c3[2], zero_pad=((1, 1), (1, 1)),
#                                         kernel_initializer=cift(l_c3[3]), bias_initializer=cift(l_c3[4]),
#                                         activation='relu', name='conv3'))
#     compressed_model.add(Conv2DPCALayer(l_c4[0], (3, 3), l_c4[1], l_c4[2], zero_pad=((1, 1), (1, 1)),
#                                         kernel_initializer=cift(l_c4[3]), bias_initializer=cift(l_c4[4]),
#                                         activation='relu', name='conv4'))
#     compressed_model.add(tf.keras.layers.MaxPooling2D((2, 2), name='mp2'))
#
#     # Fully Connected Layers
#     compressed_model.add(tf.keras.layers.Flatten(name='flatten'))
#     compressed_model.add(DensePCALayer(l_fc1[0], l_fc1[1], l_fc1[2], kernel_initializer=cift(l_fc1[3]),
#                                        bias_initializer=cift(tf.squeeze(l_fc1[4])), activation='relu', name='fc1'))
#     compressed_model.add(DensePCALayer(l_fc2[0], l_fc2[1], l_fc2[2], kernel_initializer=cift(l_fc2[3]),
#                                        bias_initializer=cift(tf.squeeze(l_fc2[4])), activation='relu', name='fc2'))
#
#     # Output Layer
#     compressed_model.add(DensePCALayer(l_o[0], l_o[1], l_o[2], kernel_initializer=cift(l_o[3]),
#                                        bias_initializer=cift(tf.squeeze(l_o[4])), activation=None, name='output'))
#     compressed_model.add(tf.keras.layers.Softmax())
#
#     # Optimizer
#     compressed_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
#                              optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
#
#     return compressed_model
#
#
#
# # @tf.function
# def compression_v1(compression_config, input_arr, overall_model, verbose=True):
#     assert compression_config['nums_as'] in ['dimension', 'threshold'], 'nums_as should be in {dimension, threshold}'
#     if compression_config['nums_as'] == 'dimension':
#         ut = False
#     else:
#         ut = True
#     cc = compression_config
#
#
#     # naming convention: _layer means output distribution of layer
#     mu_i, V_i, _ = tf_pca(input_arr, cc['conv1'][0], num_as_threshold=ut, conv=True, verbose=verbose)
#     input_arr = overall_model.get_layer('conv1')(input_arr)
#     mu_c1, V_c1, _ = tf_pca(input_arr, cc['conv2'][0], num_as_threshold=ut, conv=True, verbose=verbose)
#     input_arr = overall_model.get_layer('conv2')(input_arr)
#     input_arr = overall_model.get_layer('mp1')(input_arr)
#     mu_mp1, V_mp1, _ = tf_pca(input_arr, cc['conv3'][0], num_as_threshold=ut, conv=True, verbose=verbose)
#     input_arr = overall_model.get_layer('conv3')(input_arr)
#     mu_c3, V_c3, _ = tf_pca(input_arr, cc['conv4'][0], num_as_threshold=ut, conv=True, verbose=verbose)
#     input_arr = overall_model.get_layer('conv4')(input_arr)
#     input_arr = overall_model.get_layer('mp2')(input_arr)
#     input_arr = overall_model.get_layer('flatten')(input_arr)
#     mu_f, V_f, _ = tf_pca(input_arr, cc['fc1'][0], num_as_threshold=ut, verbose=verbose)
#     input_arr = overall_model.get_layer('fc1')(input_arr)
#     mu_fc1, V_fc1, _ = tf_pca(input_arr, cc['fc2'][0], num_as_threshold=ut, verbose=verbose)
#     input_arr = overall_model.get_layer('fc2')(input_arr)
#     mu_fc2, V_fc2, _ = tf_pca(input_arr, cc['output'][0], num_as_threshold=ut, verbose=verbose)
#
#
#     # naming convention: _layer means weights of layer
#     W_c1 = overall_model.get_layer('conv1').weights[0]
#     b_c1 = overall_model.get_layer('conv1').weights[1]
#     W_c2 = overall_model.get_layer('conv2').weights[0]
#     b_c2 = overall_model.get_layer('conv2').weights[1]
#     W_c3 = overall_model.get_layer('conv3').weights[0]
#     b_c3 = overall_model.get_layer('conv3').weights[1]
#     W_c4 = overall_model.get_layer('conv4').weights[0]
#     b_c4 = overall_model.get_layer('conv4').weights[1]
#     W_fc1 = overall_model.get_layer('fc1').weights[0]
#     b_fc1 = overall_model.get_layer('fc1').weights[1]
#     W_fc2 = overall_model.get_layer('fc2').weights[0]
#     b_fc2 = overall_model.get_layer('fc2').weights[1]
#     W_o = overall_model.get_layer('output').weights[0]
#     b_o = overall_model.get_layer('output').weights[1]
#
#
#     # pick filters to kill
#     n_c1, W_c1, b_c1, mu_c1, V_c1, W_c2 = tf_kill_columns_or_filters(W_c1, b_c1, mu_c1, V_c1, W_c2, cc['conv1'][1],
#                                                                      num_as_threshold=ut, conv=True, verbose=verbose)
#     n_c2, W_c2, b_c2, mu_mp1, V_mp1, W_c3 = tf_kill_columns_or_filters(W_c2, b_c2, mu_mp1, V_mp1, W_c3, cc['conv2'][1],
#                                                                        num_as_threshold=ut, conv=True, verbose=verbose)
#     n_c3, W_c3, b_c3, mu_c3, V_c3, W_c4 = tf_kill_columns_or_filters(W_c3, b_c3, mu_c3, V_c3, W_c4, cc['conv3'][1],
#                                                                      num_as_threshold=ut, conv=True, verbose=verbose)
#     n_c4, W_c4, b_c4, mu_f, V_f, W_fc1 = tf_kill_filters_to_dense(W_c4, b_c4, mu_f, V_f, W_fc1, cc['conv4'][1],
#                                                                   num_as_threshold=ut, verbose=True)
#     n_fc1, W_fc1, b_fc1, mu_fc1, V_fc1, W_fc2 = tf_kill_columns_or_filters(W_fc1, b_fc1, mu_fc1, V_fc1, W_fc2,
#                                                                            cc['fc1'][1],
#                                                                            num_as_threshold=ut, verbose=verbose)
#     n_fc2, W_fc2, b_fc2, mu_fc2, V_fc2, W_o = tf_kill_columns_or_filters(W_fc2, b_fc2, mu_fc2, V_fc2, W_o, cc['fc2'][1],
#                                                                          num_as_threshold=ut, verbose=verbose)
#
#
#     # compute primed variables
#     W_c1_p, b_c1_p = tf_transform_conv_weights(mu_i, V_i, W_c1, b_c1)
#     W_c2_p, b_c2_p = tf_transform_conv_weights(mu_c1, V_c1, W_c2, b_c2)
#     W_c3_p, b_c3_p = tf_transform_conv_weights(mu_mp1, V_mp1, W_c3, b_c3)
#     W_c4_p, b_c4_p = tf_transform_conv_weights(mu_c3, V_c3, W_c4, b_c4)
#     W_fc1_p, b_fc1_p = tf_transform_dense_weights(mu_f, V_f, W_fc1, b_fc1)
#     W_fc2_p, b_fc2_p = tf_transform_dense_weights(mu_fc1, V_fc1, W_fc2, b_fc2)
#     W_o_p, b_o_p = tf_transform_dense_weights(mu_fc2, V_fc2, W_o, b_o)
#
#     return (n_c1, mu_i, V_i, W_c1_p, b_c1_p), (n_c2, mu_c1, V_c1, W_c2_p, b_c2_p), \
#            (n_c3, mu_mp1, V_mp1, W_c3_p, b_c3_p), (n_c4, mu_c3, V_c3, W_c4_p, b_c4_p), \
#            (n_fc1, mu_f, V_f, W_fc1_p, b_fc1_p), (n_fc2, mu_fc1, V_fc1, W_fc2_p, b_fc2_p), \
#            (10, mu_fc2, V_fc2, W_o_p, b_o_p)