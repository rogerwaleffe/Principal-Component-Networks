import numpy as np
import tensorflow as tf



HEIGHT, WIDTH, NUM_CHANNELS = 32, 32, 3

CHANNEL_MEANS = [125.3, 123.0, 113.9] # RGB
CHANNEL_STANDARDS = [63.0,  62.1,  66.7]
MEANS = tf.broadcast_to(CHANNEL_MEANS, [HEIGHT, WIDTH, NUM_CHANNELS])
STANDARDS = tf.broadcast_to(CHANNEL_STANDARDS, [HEIGHT, WIDTH, NUM_CHANNELS])



def mean_channel_subtraction(image):
    return image - MEANS



def channel_standardization(image):
    return image/STANDARDS



def get_dataset(num_val=0, verbose=True):
    cifar10 = tf.keras.datasets.cifar10

    (xTrain, yTrain), (xTest, yTest) = cifar10.load_data()

    # One hot encoding
    num_classes = 10
    yTrain = yTrain.reshape((yTrain.shape[0],))
    yTest = yTest.reshape((yTest.shape[0],))
    yTrain = np.eye(num_classes)[yTrain]
    yTest = np.eye(num_classes)[yTest]

    # Create validation set
    random_indices = np.random.permutation(np.arange(xTrain.shape[0]))

    xVal = xTrain[random_indices[:num_val]]
    yVal = yTrain[random_indices[:num_val]]

    xTrain = xTrain[random_indices[num_val:]]
    yTrain = yTrain[random_indices[num_val:]]

    if verbose:
        print("CIFAR-10 data set created. Sizes:\n(train, val, test, yTrain, yVal, yTest): ({}, {}, {}, {}, {}, {})"
              "".format(xTrain.shape, xVal.shape, xTest.shape, yTrain.shape, yVal.shape, yTest.shape))

    return (xTrain, yTrain), (xVal, yVal), (xTest, yTest)



def preprocess_image(image, label, is_training, normalize):
    """
    Preprocess a single image of layout [height, width, depth].
    """
    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.float32)

    if is_training:
        # Resize the image to add four extra pixels on each side.
        image = tf.image.resize_with_crop_or_pad(image, HEIGHT + 8, WIDTH + 8)

        # Randomly crop a [HEIGHT, WIDTH] section of the image.
        image = tf.image.random_crop(image, [HEIGHT, WIDTH, NUM_CHANNELS])

        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)

    # Subtract off the mean and divide by the variance of the pixels.
    if normalize:
        image = mean_channel_subtraction(image)
        image = channel_standardization(image)

    return image, label



def input_fn(images, labels, num_epochs=1, batch_size=128, is_training=False, normalize=True, shuffle=True):
             # map_fn=preprocess_image, num_images=1, drop_remainder=False):
    """
    :return: The tf.data.Dataset
    """
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))

    # Prefetch a batch at a time to smooth out the time taken for processing.
    dataset = dataset.prefetch(buffer_size=batch_size)

    if is_training and shuffle is True:
        # Shuffles records before repeating to respect epoch boundaries. Show every element of one epoch before next.
        dataset = dataset.shuffle(buffer_size=images.shape[0], reshuffle_each_iteration=True)

    dataset = dataset.repeat(num_epochs)

    # Preprocess images and labels. With map before repeat, may not be preprocessing differently per epoch.
    dataset = dataset.map(lambda image, label: preprocess_image(image, label, is_training, normalize),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=False)

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