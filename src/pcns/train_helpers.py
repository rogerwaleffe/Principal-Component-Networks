import time
import numpy as np
from src.pcns.compression_helpers import vgg_compression



def vgg_train(train_config, get_full_model_fxn, get_data_fxn, verbose=0, measure_base_perf=False, measure_c_perf=False):
    """

    :param train_config:
    :param get_full_model_fxn:
    :param get_data_fxn:
    :param verbose: 0 print nothing, 1 print minimal statistics, 2 print everything
    :param measure_base_perf
    :param measure_c_perf
    :return:
    """
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = get_data_fxn()

    if verbose != 0:
        print("\nENTERING VGG TRAIN METHOD")
        print("TRAIN CONFIG: {}".format(train_config))

    model = get_full_model_fxn()
    if verbose == 2:
        print(model.summary())



    # Rewind stuff
    # import tensorflow as tf
    # model_copy = tf.keras.models.clone_model(model)



    # if measure_base_perf is True:
    #     print("\nBeginning performance measurements for full model, time {}".format(time.time()))
    #     print("\nTraining 5 epochs, time {}".format(time.time()))
    #     model.fit(train_x, train_y, epochs=5, batch_size=train_config['batch_size'], verbose=1 if verbose == 2 else 0,
    #               shuffle=True)
    #     print("\nTraining 5 epochs again, time {}".format(time.time()))
    #     model.fit(train_x, train_y, epochs=5, batch_size=train_config['batch_size'], verbose=1 if verbose == 2 else 0,
    #               shuffle=True)
    #
    #     bb = train_config['batch_size']
    #     print("\nValidating 5 times on test set, using given batch size, time {}".format(time.time()))
    #     for _ in range(5):
    #         print("\nStarting validation, time {}".format(time.time()))
    #         model.evaluate(test_x, test_y, batch_size=bb, verbose=0)
    #         print("\nValidation done, time {}".format(time.time()))
    #     print("\nValidating 5 times on test set again, using given batch size, time {}".format(time.time()))
    #     for _ in range(5):
    #         print("\nStarting validation, time {}".format(time.time()))
    #         model.evaluate(test_x, test_y, batch_size=bb, verbose=0)
    #         print("\nValidation done, time {}".format(time.time()))
    #     print("\nDone measuring performance, time {}".format(time.time()))
    #     return



    # Train for a little
    epochs = train_config['compression_after_epoch']
    batch_size = train_config['batch_size']

    if verbose != 0:
        print("\nBeginning training, time {}".format(time.time()))

    for epoch in range(0, epochs):
        model.fit(train_x, train_y, epochs=1, batch_size=batch_size, verbose=1 if verbose == 2 else 0, shuffle=True)
        score_val = model.evaluate(val_x, val_y, batch_size=val_x.shape[0], verbose=0)
        score_test = model.evaluate(test_x, test_y, batch_size=test_x.shape[0], verbose=0)
        if verbose != 0:
            print("Epoch: {}, Val Acc: {:.5f}, Val Loss {:.5f}".format(epoch + 1, score_val[1], score_val[0]))
            print("Epoch: {}, Test Acc: {:.5f}, Test Loss {:.5f}".format(epoch + 1, score_test[1], score_test[0]))


    # Compression
    if verbose != 0:
        print("\nBeginning compression, time {}".format(time.time()))

    input_arr = np.random.permutation(np.arange(train_x.shape[0]))[:train_config['num_samples_for_compression']]
    input_arr = train_x[input_arr]

    new_model = vgg_compression(compression_config=train_config['compression_config'], input_arr=input_arr,
                                overall_model=model, nums_as=train_config['nums_as'],
                                verbose=False if verbose == 0 else True)

    if verbose != 0:
        print("\nCompression Complete, time {}".format(time.time()))
        print(new_model.summary())
        score_val = new_model.evaluate(val_x, val_y, batch_size=val_x.shape[0], verbose=0)
        score_test = new_model.evaluate(test_x, test_y, batch_size=test_x.shape[0], verbose=0)
        print("Compression Complete, Val Acc: {:.5f}, Val Loss {:.5f}".format(score_val[1], score_val[0]))
        print("Compression Complete, Test Acc: {:.5f}, Test Loss {:.5f}".format(score_test[1], score_test[0]))



    # Rewind stuff
    # mu_Vs = []
    # for layer in new_model.layers:
    #     if len(layer.trainable_weights) == 0:
    #         continue
    #     if 'PCA' in type(layer).__name__:
    #         mu_Vs.append((tf.squeeze(layer.mu), layer.V, None))
    #     else:
    #         pass
    #         # mu_Vs.append((None, None))
    #
    # from src.pcns.imagenet.imagenet_compression import imagenet_vgg_compression
    # strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    #
    # cc = {'conv1': (None, None), 'conv2': ('d20', None), 'conv3': ('d40', None), 'conv4': ('d80', None),
    #       'fc1': ('d50', None), 'fc2': ('d40', None), 'output': ('d30', None)}
    # rewind_model = imagenet_vgg_compression(cc, None, model_copy, mu_Vs, strategy, None, verbose=True)
    # print("\nRewind Model Complete, time {}".format(time.time()))
    # print(rewind_model.summary())
    # score_val = rewind_model.evaluate(val_x, val_y, batch_size=val_x.shape[0], verbose=0)
    # score_test = rewind_model.evaluate(test_x, test_y, batch_size=test_x.shape[0], verbose=0)
    # print("Rewind Complete, Val Acc: {:.5f}, Val Loss {:.5f}".format(score_val[1], score_val[0]))
    # print("Rewind Complete, Test Acc: {:.5f}, Test Loss {:.5f}".format(score_test[1], score_test[0]))
    #
    # for epoch in range(0, train_config['total_epochs']):
    #     rewind_model.fit(train_x, train_y, epochs=1, batch_size=batch_size, verbose=1 if verbose == 2 else 0, shuffle=True)
    #     score_val = rewind_model.evaluate(val_x, val_y, batch_size=val_x.shape[0], verbose=0)
    #     score_test = rewind_model.evaluate(test_x, test_y, batch_size=test_x.shape[0], verbose=0)
    #     if verbose != 0:
    #         print("Epoch: {}, Val Acc: {:.5f}, Val Loss {:.5f}".format(epoch + 1, score_val[1], score_val[0]))
    #         print("Epoch: {}, Test Acc: {:.5f}, Test Loss {:.5f}".format(epoch + 1, score_test[1], score_test[0]))




    # if measure_c_perf is True:
    #     print("\nBeginning performance measurements for compressed model, time {}".format(time.time()))
    #     print("\nTraining 5 epochs, time {}".format(time.time()))
    #     new_model.fit(train_x, train_y, epochs=5, batch_size=train_config['batch_size'],
    #                   verbose=1 if verbose == 2 else 0, shuffle=True)
    #     print("\nTraining 5 epochs again, time {}".format(time.time()))
    #     new_model.fit(train_x, train_y, epochs=5, batch_size=train_config['batch_size'],
    #                   verbose=1 if verbose == 2 else 0, shuffle=True)
    #
    #     bb = train_config['batch_size']
    #     print("\nValidating 5 times on test set, using given batch size, time {}".format(time.time()))
    #     for _ in range(5):
    #         print("\nStarting validation, time {}".format(time.time()))
    #         new_model.evaluate(test_x, test_y, batch_size=bb, verbose=0)
    #         print("\nValidation done, time {}".format(time.time()))
    #     print("\nValidating 5 times on test set again, using given batch size, time {}".format(time.time()))
    #     for _ in range(5):
    #         print("\nStarting validation, time {}".format(time.time()))
    #         new_model.evaluate(test_x, test_y, batch_size=bb, verbose=0)
    #         print("\nValidation done, time {}".format(time.time()))
    #     print("\nDone measuring performance, time {}".format(time.time()))
    #     return



    # Continue Training
    if verbose != 0:
        print("\nContinuing training, time {}".format(time.time()))

    for epoch in range(train_config['compression_after_epoch'], train_config['total_epochs']):
        new_model.fit(train_x, train_y, epochs=1, batch_size=batch_size, verbose=1 if verbose == 2 else 0, shuffle=True)
        score_val = new_model.evaluate(val_x, val_y, batch_size=val_x.shape[0], verbose=0)
        score_test = new_model.evaluate(test_x, test_y, batch_size=test_x.shape[0], verbose=0)
        if verbose != 0:
            print("Epoch: {}, Val Acc: {:.5f}, Val Loss {:.5f}".format(epoch + 1, score_val[1], score_val[0]))
            print("Epoch: {}, Test Acc: {:.5f}, Test Loss {:.5f}".format(epoch + 1, score_test[1], score_test[0]))

    if verbose != 0:
        print("Training done, time {}".format(time.time()))