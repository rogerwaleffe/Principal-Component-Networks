import numpy as np
import tensorflow as tf

from src.pcns_v2.architectures.vgg import vgg16A, compute_activation_bases_cifar, forward_transform
from src.pcns_v2.datasets.cifar10 import get_dataset, input_fn
from src.pcns_v2.helpers.optimization_helpers import piecewise_scheduler



if __name__ == "__main__":
    # NUM_VAL = 5000
    NUM_SAMPLES_FOR_COMPRESSION = 5000
    EPOCHS_BEFORE_COMPRESSION = 1
    TOTAL_EPOCHS = 160
    BATCH_SIZE = 256
    VERBOSE = 1  # 1 = progress bar, 2 = one line per epoch



    # Get train/test data sets
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = get_dataset()
    num_train = train_x.shape[0]
    num_test = test_x.shape[0]

    train_dataset = input_fn(train_x, train_y, num_epochs=EPOCHS_BEFORE_COMPRESSION, batch_size=BATCH_SIZE,
                             is_training=True, normalize=True)
    # val_dataset = input_fn(val_x, val_y, batch_size=BATCH_SIZE, normalize=True)
    test_dataset = input_fn(test_x, test_y, batch_size=BATCH_SIZE, normalize=True)



    # Get model
    model, cc = vgg16A()
    lr_schedule = piecewise_scheduler([80, 120], [1.0, 0.1, 0.01], base_rate=0.1, boundaries_as='epochs',
                                      num_images=num_train, batch_size=BATCH_SIZE)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    print(model.summary())



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


    cc['conv8'] = cc['conv9'] = cc['conv10'] = ('d64', None)
    cc['conv11'] = cc['conv12'] = cc['conv13'] = ('d32', None)

    bases = compute_activation_bases_cifar(cc, input_arr, model, forget_bottom=True)
    new_model = forward_transform(bases, model, include_offset=False, train_top_basis='NO')


    boundaries = [80-EPOCHS_BEFORE_COMPRESSION, 120-EPOCHS_BEFORE_COMPRESSION]
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
    train_dataset = input_fn(train_x, train_y, num_epochs=TOTAL_EPOCHS-EPOCHS_BEFORE_COMPRESSION,
                             batch_size=BATCH_SIZE, is_training=True, normalize=True)
    new_model.fit(train_dataset, initial_epoch=EPOCHS_BEFORE_COMPRESSION,
                         epochs=TOTAL_EPOCHS, steps_per_epoch=np.floor(num_train/BATCH_SIZE),
                         validation_data=test_dataset, validation_steps=np.ceil(num_test/BATCH_SIZE),
                         validation_freq=1, verbose=VERBOSE)

    if VERBOSE == 1:
        new_model.evaluate(test_dataset, steps=np.ceil(num_test/BATCH_SIZE))
    else:
        score = new_model.evaluate(test_dataset, steps=np.ceil(num_test/BATCH_SIZE), verbose=0)
        print("TRAINING: after compression, Test Loss {:.5f}, Test Acc: {:.5f}".format(score[0], score[1]))

