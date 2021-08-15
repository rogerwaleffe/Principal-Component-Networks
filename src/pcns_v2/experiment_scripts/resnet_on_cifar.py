import numpy as np
import tensorflow as tf

from src.pcns_v2.architectures.resnet import ResNet, ResNetBlockV1, ResNetBlockV2
from src.pcns_v2.helpers.layer_helpers import Conv2DExplicitPadding, DenseTransformLayer
from src.pcns_v2.datasets.cifar10 import get_dataset, input_fn
from src.pcns_v2.helpers.optimization_helpers import piecewise_scheduler



if __name__ == "__main__":
    # NUM_VAL = 5000
    NUM_SAMPLES_FOR_COMPRESSION = 5000
    EPOCHS_BEFORE_COMPRESSION = 15
    TOTAL_EPOCHS = 164
    BATCH_SIZE = 128
    VERBOSE = 1  # 1 = progress bar, 2 = one line per epoch



    # Get train/test data sets
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = get_dataset()
    num_train = train_x.shape[0]
    num_test = test_x.shape[0]

    train_dataset = input_fn(train_x, train_y, num_epochs=EPOCHS_BEFORE_COMPRESSION, batch_size=BATCH_SIZE,
                             is_training=True, normalize=True)
    # val_dataset = input_fn(val_x, val_y, batch_size=BATCH_SIZE, normalize=True)
    test_dataset = input_fn(test_x, test_y, batch_size=BATCH_SIZE, normalize=True)



    # Create ResNet model
    resnet = ResNet(input_shape=(32, 32, 3), num_classes=10, bottleneck=False, num_filters_at_start=16,
                    initial_kernel_size=3, initial_conv_strides=1, initial_pool_size=None, initial_pool_strides=None,
                    num_residual_blocks_per_stage=[3, 3, 3], first_block_strides_per_stage=[1, 2, 2], kernel_size=3,
                    project_first_residual=True, version='V1', data_format='channels_last')
    model, cc = resnet.get_model()

    lr_schedule = piecewise_scheduler([82, 123], [1.0, 0.1, 0.01], base_rate=0.1, boundaries_as='epochs',
                                      num_images=num_train, batch_size=BATCH_SIZE)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    print(model.summary())
    resnet.print_ResNet(model)



    # Train model
    model.fit(train_dataset, epochs=EPOCHS_BEFORE_COMPRESSION, steps_per_epoch=np.floor(num_train/BATCH_SIZE),
              validation_data=test_dataset, validation_steps=np.ceil(num_test/BATCH_SIZE), validation_freq=1,
              verbose=VERBOSE)
    if VERBOSE == 1:
        model.evaluate(test_dataset, steps=np.ceil(num_test/BATCH_SIZE))
    else:
        score = model.evaluate(test_dataset, steps=np.ceil(num_test/BATCH_SIZE), verbose=0)
        print("TRAINING: before compression, Test Loss {:.5f}, Test Acc: {:.5f}".format(score[0], score[1]))



    # Compression
    # NOTE: use the training data, but is_training=False so we don't do data augmentation for the compression dataset
    compression_dataset = input_fn(train_x, train_y, batch_size=NUM_SAMPLES_FOR_COMPRESSION,
                                   is_training=False, normalize=True)
    compression_dataset = compression_dataset.take(1)

    # Can put whole data set in one tensor for CIFAR-10
    input_arr = []
    for i, (im, lab) in enumerate(compression_dataset):
        input_arr.append(im)
    input_arr = tf.concat(input_arr, axis=0)


    # TODO: need to explain this
    # {}_{}: (conv1 input, conv1 output, conv2 input), note that conv1 input also applies to conv0 input for {}_1 blocks
    # {}: conv0 and all conv2 outputs for stage
    # output transformations not yet supported in v2
    # 't0.25': use threshold 0.25
    # 'd64': use a fixed dimension of 64
    # 'f0.25': use fraction 0.25
    cc['stage1_block1'] = ('f1.0', None, 'f1.0')
    cc['stage1_block2'] = cc['stage1_block3'] = ('f1.0', None, 'f1.0')
    cc['stage2_block1'] = ('f1.0', None, 'f1.0')
    cc['stage2_block2'] = cc['stage2_block3'] = ('f1.0', None, 'f1.0')
    cc['stage3_block1'] = ('f1.0', None, 'f1.0')
    cc['stage3_block2'] = cc['stage3_block3'] = ('f1.0', None, 'f1.0')
    cc['fc1'] = ('f1.0', None)

    bases = resnet.compute_activation_bases_cifar(cc, input_arr, model, forget_bottom=True, pca_centering=True)
    new_model = resnet.forward_transform(bases, model, include_offset=False, train_top_basis='NO', add_bias_if_nec=True)


    boundaries = [82-EPOCHS_BEFORE_COMPRESSION, 123-EPOCHS_BEFORE_COMPRESSION]
    lr_schedule = piecewise_scheduler(boundaries, [1.0, 0.1, 0.01], base_rate=0.1, boundaries_as='epochs',
                                      num_images=num_train, batch_size=BATCH_SIZE)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    new_model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    print(new_model.summary())

    if VERBOSE == 1:
        new_model.evaluate(test_dataset, steps=np.ceil(num_test/BATCH_SIZE))
    else:
        score = new_model.evaluate(test_dataset, steps=np.ceil(num_test/BATCH_SIZE), verbose=0)
        print("TRAINING: after compression, Test Loss {:.5f}, Test Acc: {:.5f}".format(score[0], score[1]))



    # Continue training
    # checkpoint = tf.keras.callbacks.ModelCheckpoint('run_name_{epoch:03d}/', monitor='val_accuracy', verbose=0,
    #                                                 save_best_only=False, save_weights_only=False, mode='auto',
    #                                                 save_freq='epoch', options=None)

    train_dataset = input_fn(train_x, train_y, num_epochs=TOTAL_EPOCHS-EPOCHS_BEFORE_COMPRESSION,
                             batch_size=BATCH_SIZE, is_training=True, normalize=True)
    new_model.fit(train_dataset, initial_epoch=EPOCHS_BEFORE_COMPRESSION,
                  epochs=TOTAL_EPOCHS, steps_per_epoch=np.floor(num_train/BATCH_SIZE),
                  validation_data=test_dataset, validation_steps=np.ceil(num_test/BATCH_SIZE),
                  validation_freq=1, verbose=VERBOSE)#, callbacks=[checkpoint])

    if VERBOSE == 1:
        new_model.evaluate(test_dataset, steps=np.ceil(num_test/BATCH_SIZE))
    else:
        score = new_model.evaluate(test_dataset, steps=np.ceil(num_test/BATCH_SIZE), verbose=0)
        print("TRAINING: after compression, Test Loss {:.5f}, Test Acc: {:.5f}".format(score[0], score[1]))

    # # Load saved model example
    # new_model = tf.keras.models.load_model('run_name_002/', compile=True,
    #                                        custom_objects={'Conv2DExplicitPadding': Conv2DExplicitPadding,
    #                                                        'ResNetBlockV1': ResNetBlockV1,
    #                                                        'ResNetBlockV2': ResNetBlockV2,
    #                                                        'DenseTransformLayer': DenseTransformLayer})
    # new_model.evaluate(test_dataset, steps=np.ceil(num_test/BATCH_SIZE))



    # # Undo model
    # model = resnet.undo_forward_transform(new_model)
    # optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)
    # loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    #
    # model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    # print(model.summary())
    #
    # model.evaluate(test_dataset, steps=np.ceil(num_test / BATCH_SIZE))