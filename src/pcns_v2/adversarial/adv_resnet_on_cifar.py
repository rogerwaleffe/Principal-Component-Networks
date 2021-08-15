import argparse
import numpy as np
import tensorflow as tf

from datetime import datetime

from src.pcns_v2.architectures.resnet import ResNet
from src.pcns_v2.datasets.cifar10 import get_dataset, input_fn
from src.pcns_v2.helpers.optimization_helpers import piecewise_scheduler
from src.pcns_v2.adversarial.helpers import advTrainModel, PGDAttack



if __name__ == "__main__":
    print("Running cifar10.py: {}".format(datetime.now()))

    parser = argparse.ArgumentParser(description='PCN in adversarial training')
    parser.add_argument("--prune", help='Prune by PCN', action="store_true")
    parser.add_argument('--adv_method', type=str, default='None', help="Method to generate adversarial examples")
    parser.add_argument('--scale', type=float, default=1.0, help="Width of the network")

    args = parser.parse_args()
    isPrune = args.prune
    scale = args.scale
    adv_method = args.adv_method
    print("Run PCN pruning: %s" % isPrune)
    print("Method to generate adversarial samples: %s" % adv_method)
    print("Width factor: %f" % scale)



    NUM_SAMPLES_FOR_COMPRESSION = 5000
    EPOCHS_BEFORE_COMPRESSION = 15
    TOTAL_EPOCHS = 182
    BATCH_SIZE = 128
    VERBOSE = 2  # 1 = progress bar, 2 = one line per epoch



    # Get train/test data sets
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = get_dataset()
    num_train = train_x.shape[0]
    num_test = test_x.shape[0]

    if isPrune:
        train_dataset = input_fn(train_x, train_y, num_epochs=EPOCHS_BEFORE_COMPRESSION, batch_size=BATCH_SIZE,
                                 is_training=True, normalize=False)
    else:
        train_dataset = input_fn(train_x, train_y, num_epochs=TOTAL_EPOCHS, batch_size=BATCH_SIZE,
                                 is_training=True, normalize=False)
    test_dataset = input_fn(test_x, test_y, batch_size=BATCH_SIZE, normalize=False)



    # Create ResNet model
    resnet = ResNet(input_shape=(32, 32, 3), num_classes=10, bottleneck=False, num_filters_at_start=int(16*scale),
                    initial_kernel_size=3, initial_conv_strides=1, initial_pool_size=None, initial_pool_strides=None,
                    num_residual_blocks_per_stage=[3, 3, 3], first_block_strides_per_stage=[1, 2, 2], kernel_size=3,
                    project_first_residual=True, version='V2', data_format='channels_last')
    model, cc = resnet.get_model()
    model = advTrainModel(model, attack=adv_method)

    lr_schedule = piecewise_scheduler([91, 136], [1.0, 0.1, 0.01], base_rate=0.1, boundaries_as='epochs',
                                      num_images=num_train, batch_size=BATCH_SIZE)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    print(model.base_model.summary())
    resnet.print_ResNet(model.base_model)



    # Train model
    if isPrune:
        model.fit(train_dataset, epochs=EPOCHS_BEFORE_COMPRESSION, steps_per_epoch=np.floor(num_train/BATCH_SIZE),
                  validation_data=test_dataset, validation_steps=np.ceil(num_test/BATCH_SIZE), validation_freq=1,
                  verbose=VERBOSE)
    else:
        model.fit(train_dataset, epochs=TOTAL_EPOCHS, steps_per_epoch=np.floor(num_train/BATCH_SIZE),
                  validation_data=test_dataset, validation_steps=np.ceil(num_test/BATCH_SIZE), validation_freq=1,
                  verbose=VERBOSE)

    if VERBOSE == 1:
        model.evaluate(test_dataset, steps=np.ceil(num_test/BATCH_SIZE))
    else:
        print('Natural Loss and Acc')
        model.adv_attack = 'None'
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        score = model.evaluate(test_dataset, steps=np.ceil(num_test/BATCH_SIZE), verbose=0)
        print("TRAINING: before compression, Test Loss {:.5f}, Test Acc: {:.5f}".format(score[0], score[1]))

        print('FGSM Loss and Acc')
        model.adv_attack = 'FGSM'
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        score = model.evaluate(test_dataset, steps=np.ceil(num_test/BATCH_SIZE), verbose=0)
        print("TRAINING: before compression, Test Loss {:.5f}, Test Acc: {:.5f}".format(score[0], score[1]))

        print('PGD Loss and Acc')
        model.adv_attack = 'PGD'
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        score = model.evaluate(test_dataset, steps=np.ceil(num_test/BATCH_SIZE), verbose=0)
        print("TRAINING: before compression, Test Loss {:.5f}, Test Acc: {:.5f}".format(score[0], score[1]))



    if isPrune:
        # Compress the model
        # NOTE: with the training data, but is_training=False we won't do data augmentation for the compression dataset
        compression_dataset = input_fn(train_x, train_y, batch_size=BATCH_SIZE, is_training=False, normalize=False)
        compression_dataset = compression_dataset.take(NUM_SAMPLES_FOR_COMPRESSION // BATCH_SIZE)

        input_arr = []
        for i, (im, lab) in enumerate(compression_dataset):
            # TODO: these samples should probably depend on adv_method
            print('Batch for attack: %d / %d' % (i, NUM_SAMPLES_FOR_COMPRESSION // BATCH_SIZE))
            im_adv = PGDAttack(im, lab, model.base_model, loss, epsilon=8, num_steps=10, step_size=2, random_start=True)
            input_arr.append(im_adv)
            # input_arr.append(im)
        input_arr = tf.concat(input_arr, axis=0)


        cc['stage2_block1'] = (None, None, 'd16')
        cc['stage2_block2'] = cc['stage2_block3'] = ('d16', None, 'd16')
        cc['stage3_block1'] = ('d16', None, 'd32')
        cc['stage3_block2'] = cc['stage3_block3'] = ('d32', None, 'd32')

        bases = resnet.compute_activation_bases_cifar(cc, input_arr, model.base_model, forget_bottom=True)
        compressed_model = resnet.forward_transform(bases, model.base_model, include_offset=False, train_top_basis='NO')
        compressed_model = advTrainModel(compressed_model, attack=adv_method)


        boundaries = [91-EPOCHS_BEFORE_COMPRESSION, 136-EPOCHS_BEFORE_COMPRESSION]
        lr_schedule = piecewise_scheduler(boundaries, [1.0, 0.1, 0.01], base_rate=0.1, boundaries_as='epochs',
                                          num_images=num_train, batch_size=BATCH_SIZE)
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

        compressed_model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        print(compressed_model.base_model.summary())

        if VERBOSE == 1:
            compressed_model.evaluate(test_dataset, steps=np.ceil(num_test/BATCH_SIZE))
        else:
            score = compressed_model.evaluate(test_dataset, steps=np.ceil(num_test/BATCH_SIZE), verbose=0)
            print("COMPRESSION: Test Loss {:.5f}, Test Acc: {:.5f}".format(score[0], score[1]))



        # Continue training PCN:
        train_dataset = input_fn(train_x, train_y, num_epochs=TOTAL_EPOCHS-EPOCHS_BEFORE_COMPRESSION,
                                 batch_size=BATCH_SIZE, is_training=True, normalize=False)

        compressed_model.fit(train_dataset, initial_epoch=EPOCHS_BEFORE_COMPRESSION,
                             epochs=TOTAL_EPOCHS, steps_per_epoch=np.floor(num_train/BATCH_SIZE),
                             validation_data=test_dataset, validation_steps=np.ceil(num_test/BATCH_SIZE),
                             validation_freq=1, verbose=VERBOSE)

        if VERBOSE == 1:
            compressed_model.evaluate(test_dataset, steps=np.ceil(num_test/BATCH_SIZE))
        else:
            print('Natural Loss and Acc')
            compressed_model.adv_attack = 'None'
            compressed_model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
            score = compressed_model.evaluate(test_dataset, steps=np.ceil(num_test/BATCH_SIZE), verbose=0)
            print("TRAINING: after compression, Test Loss {:.5f}, Test Acc: {:.5f}".format(score[0], score[1]))

            print('FGSM Loss and Acc')
            compressed_model.adv_attack = 'FGSM'
            compressed_model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
            score = compressed_model.evaluate(test_dataset, steps=np.ceil(num_test/BATCH_SIZE), verbose=0)
            print("TRAINING: after compression, Test Loss {:.5f}, Test Acc: {:.5f}".format(score[0], score[1]))

            print('PGD Loss and Acc')
            compressed_model.adv_attack = 'PGD'
            compressed_model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
            score = compressed_model.evaluate(test_dataset, steps=np.ceil(num_test/BATCH_SIZE), verbose=0)
            print("TRAINING: after compression, Test Loss {:.5f}, Test Acc: {:.5f}".format(score[0], score[1]))