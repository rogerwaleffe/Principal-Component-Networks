"""
ImageNet Processing Helpers for TensorFlow
"""

import os
import math
import random
import tensorflow as tf



# Image Preprocessing Defaults (can not modify elsewhere at this point)
DEFAULT_IMAGE_SIZE = 224
NUM_CHANNELS = 3
TRAIN_SCALE_S = 256
TEST_SCALE_S = TRAIN_SCALE_S

NUM_TRAIN_FILES = 1024
NUM_TEST_FILES = 128

# Can be modified in input_fn definition
DEFAULT_SHUFFLE_BUFFER = 10000



R_MEAN = 123.68
G_MEAN = 116.78
B_MEAN = 103.94
CHANNEL_MEANS = [R_MEAN, G_MEAN, B_MEAN]
CHANNEL_STANDARDS = [58.393, 57.12, 57.375]
MEANS = tf.broadcast_to(CHANNEL_MEANS, [DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, NUM_CHANNELS])
STANDARDS = tf.broadcast_to(CHANNEL_STANDARDS, [DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, NUM_CHANNELS])

eig_val = tf.reshape(tf.constant([0.2175, 0.0188, 0.0045], dtype=tf.float32), [3, 1])
eig_vec = tf.constant([[-0.5675,  0.7192,  0.4009],
                       [-0.5808, -0.0045, -0.8140],
                       [-0.5836, -0.6948,  0.4203]])



def get_filenames(is_training, data_dir):
    if is_training:
        return [os.path.join(data_dir, 'train-%05d-of-01024' % i) for i in range(NUM_TRAIN_FILES)]
    else:
        return [os.path.join(data_dir, 'val-%05d-of-00128' % i) for i in range(NUM_TEST_FILES)]



def central_crop(image, crop_height, crop_width):
    shape = tf.shape(image)
    height = shape[0]
    width = shape[1]

    amount_to_be_cropped_h = (height - crop_height)
    crop_top = amount_to_be_cropped_h // 2
    amount_to_be_cropped_w = (width - crop_width)
    crop_left = amount_to_be_cropped_w // 2

    return tf.slice(image, [crop_top, crop_left, 0], [crop_height, crop_width, -1])



def smallest_size_at_least(height, width, resize_min):
    resize_min = tf.cast(resize_min, tf.float32)
    height, width = tf.cast(height, tf.float32), tf.cast(width, tf.float32)

    smaller_dim = tf.math.minimum(height, width)
    scale_ratio = resize_min/smaller_dim

    new_height = tf.cast(tf.math.rint(height*scale_ratio), tf.int32)
    new_width = tf.cast(tf.math.rint(width*scale_ratio), tf.int32)

    return new_height, new_width



def aspect_preserving_resize(image, resize_min):
    shape = tf.shape(image)
    height = shape[0]
    width = shape[1]

    new_height, new_width = smallest_size_at_least(height, width, resize_min)
    return tf.image.resize(image, [new_height, new_width])



def mean_channel_subtraction(image):
    return image - MEANS



def channel_standardization(image):
    return image/STANDARDS



def pca_color_augmentation(image):
    alpha = tf.random.normal([3, 1], mean=0.0, stddev=0.1, dtype=tf.float32)

    noise = tf.reshape(tf.linalg.matmul(eig_vec, tf.math.multiply(alpha, eig_val)), [3, ])
    noise = noise * 255
    # tf.print(noise)
    noise = tf.broadcast_to(noise, tf.shape(image))

    return image + noise



def resnet_train_preprocessing(image):

    def get_params(height, width, random_area=(0.08, 1.0), random_aspect_ratio=(3./4., 4./3.)):
        area = height * width

        i, j, h, w = -1, -1, -1, -1
        iteration = 0
        while tf.math.less(iteration, 10):
            target_area = tf.random.uniform([], minval=random_area[0], maxval=random_area[1])*tf.cast(area, tf.float32)
            aspect_ratio = tf.random.uniform([], minval=random_aspect_ratio[0], maxval=random_aspect_ratio[1])

            w = tf.cast(tf.math.round(tf.math.sqrt(target_area*aspect_ratio)), tf.int32)
            h = tf.cast(tf.math.round(tf.math.sqrt(target_area/aspect_ratio)), tf.int32)

            if tf.math.less(0, w) and tf.math.less_equal(w, width) \
                    and tf.math.less(0, h) and tf.math.less_equal(h, height):
                i = tf.random.uniform([], minval=0, maxval=height-h+1, dtype=tf.int32)
                j = tf.random.uniform([], minval=0, maxval=width-w+1, dtype=tf.int32)
                iteration = iteration + 10 # to stop the loop
                # return i, j, h, w
            else:
                iteration = iteration + 1

        if tf.math.not_equal(i, -1):
            return i, j, h, w
        else:
            # Fallback to central crop
            in_ratio = tf.cast(width, tf.float32) / tf.cast(height, tf.float32)
            if tf.math.less(in_ratio, tf.math.minimum(random_aspect_ratio[0], random_aspect_ratio[1])):
                w = width
                h = tf.cast(tf.math.round(tf.cast(w, tf.float32)/tf.math.minimum(random_aspect_ratio[0],
                                                                                 random_aspect_ratio[1])), tf.int32)
            elif tf.math.greater(in_ratio, tf.math.maximum(random_aspect_ratio[0], random_aspect_ratio[1])):
                h = height
                w = tf.cast(tf.math.round(tf.cast(h, tf.float32)*tf.math.maximum(random_aspect_ratio[0],
                                                                                 random_aspect_ratio[1])), tf.int32)
            else: # whole image
                w = width
                h = height
            i = (height - h) // 2
            j = (width - w) // 2
            return i, j, h, w

    # crop
    top, left, hh, ww = get_params(tf.shape(image)[0], tf.shape(image)[1])
    image = tf.slice(image, [top, left, 0], [hh, ww, -1])
    # resize to 224*224*3
    image = tf.image.resize(image, [DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE])
    # flip with probability 0.5
    image = tf.image.random_flip_left_right(image)
    # hue, saturation, brightness
    image = tf.image.random_brightness(image, 0.4, seed=None)
    image = tf.image.random_saturation(image, 0.6, 1.4, seed=None)
    image = tf.image.random_hue(image, 0.4, seed=None)
    # PCA Noise
    image = pca_color_augmentation(image)
    # normalize channels
    image = mean_channel_subtraction(image)
    image = channel_standardization(image)

    return image



def preprocess_image(image_buffer, is_training, center_crops_for_train=False, single_scale=True, color_aug=False,
                     standardize_train=False, resnet_preprocessing=False): # Default is what we used for VGG
    image = tf.io.decode_jpeg(image_buffer, channels=NUM_CHANNELS)

    if is_training:
        if resnet_preprocessing:
            image = resnet_train_preprocessing(image)
        else:
            # rescale (isotropically-rescale) to training scale S, note this also returns a float32 image
            # single train scale
            if single_scale:
                image = aspect_preserving_resize(image, TRAIN_SCALE_S)
            # multiple train scales
            else:
                train_scale_s = tf.random.uniform(shape=[], minval=256, maxval=481, dtype=tf.int32) # max is exclusive
                image = aspect_preserving_resize(image, train_scale_s)

            if center_crops_for_train:
                image = central_crop(image, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)
            else:
                # random 224x224 crop with random horizontal flip
                image = tf.image.random_crop(image, [DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, NUM_CHANNELS])
                image = tf.image.random_flip_left_right(image)

            # random RGB colour shift
            if color_aug:
                image = pca_color_augmentation(image)

            # subtract mean RGB value, computed on training set, from each pixel
            image = mean_channel_subtraction(image)

            if standardize_train:
                image = channel_standardization(image)

    # validation
    else:
        # rescale to smallest image side (test scale Q, Q=S for fixed S)
        image = aspect_preserving_resize(image, TEST_SCALE_S)

        # central 224*224 crop
        image = central_crop(image, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)

        # subtract mean RGB value, computed on training set, from each pixel
        image = mean_channel_subtraction(image)

        if resnet_preprocessing:
            image = channel_standardization(image)

        # TODO: some other things that involve network predictions?

    return image



def parse_example_proto(example_serialized):
    """
    Parses an Example proto containing an image.

    :param example_serialized: scalar Tensor tf.string
    :return:
        image_buffer: Tensor tf.string with contents of a JPEG file
        label: Tensor tf.float32 containing the label
    """
    feature_map = {'image/encoded': tf.io.FixedLenFeature([], dtype=tf.string, default_value=''),
                   'image/class/label': tf.io.FixedLenFeature([], dtype=tf.int64, default_value=-1),}

    features = tf.io.parse_single_example(serialized=example_serialized, features=feature_map)

    # want labels [0, 999] instead of [1, 1000]
    one_hot = True
    if one_hot:
        # one hot encoding with label smoothing
        smoothing = 0.1
        label = tf.one_hot(features['image/class/label'] - 1, 1000, on_value=1-smoothing, off_value=smoothing/(1000-1),
                           dtype=tf.float32)
    else:
        label = tf.cast(features['image/class/label'] - 1, dtype=tf.float32)

    return features['image/encoded'], label



def parse_record(raw_record, is_training, dtype=tf.float32, center_crops_for_train=False, single_scale=True,
                 color_aug=False, standardize_train=False, resnet_preprocessing=False):
    """
    Parse a record containing an image. Record is parsed into a label and image, and the image is passed through
    preprocessing steps.

    :param raw_record:
    :param is_training:
    :param dtype:
    :param center_crops_for_train
    :param single_scale
    :param color_aug
    :param standardize_train
    :param resnet_preprocessing
    :return:
    """
    image_buffer, label = parse_example_proto(raw_record)

    image = preprocess_image(image_buffer, is_training=is_training, center_crops_for_train=center_crops_for_train,
                             single_scale=single_scale, color_aug=color_aug, standardize_train=standardize_train,
                             resnet_preprocessing=resnet_preprocessing)
    image = tf.cast(image, dtype)

    return image, label



def input_fn(base_data_dir, is_training, num_epochs=1, shuffle_buffer=DEFAULT_SHUFFLE_BUFFER, batch_size=256,
             drop_remainder=False, center_crops_for_train=False, single_scale=True, color_aug=False,
             standardize_train=False, resnet_preprocessing=False):
    """
    :return: The tf.data.Dataset
    """
    data_dir = 'train' if is_training is True else 'val'

    filenames = get_filenames(is_training, os.path.join(base_data_dir, data_dir))
    dataset = tf.data.Dataset.from_tensor_slices(filenames)

    if is_training:
        # Shuffle the input files.
        dataset = dataset.shuffle(buffer_size=NUM_TRAIN_FILES)

    # Convert to individual records. cycle_length = X means that up to X files will be read and deserialized in
    # parallel. Increase this number if you have a large number of CPU cores. cycle through cycle_length input elements
    # producing block_length consecutive elements from each iterator.
    dataset = dataset.interleave(lambda x: tf.data.TFRecordDataset(x, buffer_size=None, num_parallel_reads=None),
                                 #cycle_length=NUM_TRAIN_FILES if is_training else NUM_TEST_FILES, block_length=1,
                                 num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Prefetch a batch at a time to smooth out the time taken for processing.
    dataset = dataset.prefetch(buffer_size=batch_size)

    if is_training:
        # Shuffles records before repeating to respect epoch boundaries. Show every element of one epoch before next.
        dataset = dataset.shuffle(buffer_size=shuffle_buffer, reshuffle_each_iteration=True)

    # Repeats the dataset for the number of epochs to train.
    dataset = dataset.repeat(num_epochs)

    # Parses the raw records into images and labels.
    dataset = dataset.map(lambda value: parse_record(value, is_training, center_crops_for_train=center_crops_for_train,
                                                     single_scale=single_scale, color_aug=color_aug,
                                                     standardize_train=standardize_train,
                                                     resnet_preprocessing=resnet_preprocessing),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Batch after repeat yields batches that straddle epoch boundaries. Only the very last batch may not be full.
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

    # Operations between the final prefetch and the get_next call to the iterator
    # will happen synchronously during run time. We prefetch here again to
    # background all of the above processing work and keep it out of the
    # critical training path. Setting buffer_size to tf.data.experimental.AUTOTUNE
    # allows DistributionStrategies to adjust how many batches to fetch based
    # on how many devices are present.
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset



def create_and_cache_tf_records(base_data_dir, is_training):
    data_dir = 'train' if is_training is True else 'val'

    filenames = get_filenames(is_training, os.path.join(base_data_dir, data_dir))
    dataset = tf.data.Dataset.from_tensor_slices(filenames)

    if is_training:
        # Shuffle the input files. NOTE: this probably isn't necessary since the idea is to shuffle after the interleave
        dataset = dataset.shuffle(buffer_size=NUM_TRAIN_FILES)

    # Convert to individual records. cycle_length = X means that up to X files will be read and deserialized in
    # parallel. Increase this number if you have a large number of CPU cores. cycle through cycle_length input elements
    # producing block_length consecutive elements from each iterator.
    dataset = dataset.interleave(lambda x: tf.data.TFRecordDataset(x, buffer_size=None, num_parallel_reads=None),
                                 # cycle_length=NUM_TRAIN_FILES if is_training else NUM_TEST_FILES, block_length=1,
                                 num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.cache()

    return dataset



def input_fn_given_dataset(dataset, is_training, num_epochs=1, shuffle_buffer=DEFAULT_SHUFFLE_BUFFER, batch_size=256,
                           drop_remainder=False, center_crops_for_train=False, single_scale=True, color_aug=False,
                           standardize_train=False, resnet_preprocessing=False):
    # Prefetch a batch at a time to smooth out the time taken for processing. NOTE: is this needed?
    dataset = dataset.prefetch(buffer_size=batch_size)

    if is_training:
        # Shuffles records before repeating to respect epoch boundaries. Show every element of one epoch before next.
        dataset = dataset.shuffle(buffer_size=shuffle_buffer, reshuffle_each_iteration=True)

    # Repeats the dataset for the number of epochs to train.
    dataset = dataset.repeat(num_epochs)

    # Parses the raw records into images and labels.
    dataset = dataset.map(lambda value: parse_record(value, is_training, center_crops_for_train=center_crops_for_train,
                                                     single_scale=single_scale, color_aug=color_aug,
                                                     standardize_train=standardize_train,
                                                     resnet_preprocessing=resnet_preprocessing),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Batch after repeat yields batches that straddle epoch boundaries. Only the very last batch may not be full.
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

    # Operations between the final prefetch and the get_next call to the iterator
    # will happen synchronously during run time. We prefetch here again to
    # background all of the above processing work and keep it out of the
    # critical training path. Setting buffer_size to tf.data.experimental.AUTOTUNE
    # allows DistributionStrategies to adjust how many batches to fetch based
    # on how many devices are present.
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset