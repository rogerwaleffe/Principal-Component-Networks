import os
import math
import random
import tensorflow as tf
import tensorflow.compat.v1.logging as logging



DATA_DIRECTORY = 'attached_vol/hongyi_imagenet_data'
TRAINING_DIRECTORY = 'train'
VALIDATION_DIRECTORY = 'val'

OUTPUT_DIRECTORY = 'attached_vol/imagenet_tf_record_data'
TRAINING_SHARDS = 1024
VALIDATION_SHARDS = 128



def _check_or_create_dir(directory):
    """
    Check if directory exists otherwise create it.
    """
    if not tf.io.gfile.exists(directory):
        tf.io.gfile.makedirs(directory)



def _is_png(filename):
    """
    Determine if a file contains a PNG format image.
    """
    # File list from: https://github.com/cytsai/ilsvrc-cmyk-image-list
    return 'n02105855_2933.JPEG' in filename



def _is_cmyk(filename):
    """
    Determine if file contains a CMYK JPEG format image.
    """
    # File list from: https://github.com/cytsai/ilsvrc-cmyk-image-list
    blacklist = {'n01739381_1309.JPEG', 'n02077923_14822.JPEG', 'n02447366_23489.JPEG', 'n02492035_15739.JPEG',
                 'n02747177_10752.JPEG', 'n03018349_4028.JPEG', 'n03062245_4620.JPEG', 'n03347037_9675.JPEG',
                 'n03467068_12171.JPEG', 'n03529860_11437.JPEG', 'n03544143_17228.JPEG', 'n03633091_5218.JPEG',
                 'n03710637_5125.JPEG', 'n03961711_5286.JPEG', 'n04033995_2932.JPEG', 'n04258138_17003.JPEG',
                 'n04264628_27969.JPEG', 'n04336792_7448.JPEG', 'n04371774_5854.JPEG', 'n04596742_4225.JPEG',
                 'n07583066_647.JPEG', 'n13037406_4650.JPEG'}
    return os.path.basename(filename) in blacklist



def _int64_feature(value):
    """
    Wrapper for inserting int64 features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))



def _bytes_feature(value):
    """
    Wrapper for inserting bytes features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))



class ImageCoder(object):
    """
    Helper class that provides TensorFlow image coding utilities.
    """
    @staticmethod
    def png_to_jpeg(image_data):
        image = tf.io.decode_png(image_data, channels=3)
        return tf.io.encode_jpeg(image, format='rgb', quality=100).numpy()

    @staticmethod
    def cmyk_to_rgb(image_data):
        image = tf.io.decode_jpeg(image_data, channels=0)
        return tf.io.encode_jpeg(image, format='rgb', quality=100).numpy()

    @ staticmethod
    def decode_jpeg(image_data):
        image = tf.io.decode_jpeg(image_data, channels=3)

        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image



def _convert_to_example(filename, image_buffer, label, synset, height, width):
    """
    Build an Example proto for an example.
    """
    colorspace = 'RGB'
    channels = 3
    image_format = 'JPEG'

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/colorspace': _bytes_feature(tf.compat.as_bytes(colorspace)),
        'image/channels': _int64_feature(channels),
        'image/class/label': _int64_feature(label),
        'image/class/synset': _bytes_feature(tf.compat.as_bytes(synset)),
        'image/format': _bytes_feature(tf.compat.as_bytes(image_format)),
        'image/filename': _bytes_feature(tf.compat.as_bytes(os.path.basename(filename))),
        'image/encoded': _bytes_feature(tf.compat.as_bytes(image_buffer))}))
    return example



def _process_image(filename, coder):
    """
    Process a single image file.

    :param filename: string, path to an image file e.g., '/path/to/example.JPG'
    :param coder: instance of ImageCoder to provide TensorFlow image coding utils

    :return:
    """
    # Read the image file.
    with tf.io.gfile.GFile(filename, 'rb') as f:
        image_data = f.read()

    # Clean the dirty data.
    if _is_png(filename):
        # 1 image is a PNG.
        logging.info('Converting PNG to JPEG for %s' % filename)
        image_data = coder.png_to_jpeg(image_data)
    elif _is_cmyk(filename):
        # 22 JPEG images are in CMYK colorspace.
        logging.info('Converting CMYK to RGB for %s' % filename)
        image_data = coder.cmyk_to_rgb(image_data)

    # Decode the RGB JPEG.
    image = coder.decode_jpeg(image_data)

    # Check that image converted to RGB
    height = image.shape[0]
    width = image.shape[1]

    return image_data, height, width



def _process_image_files_batch(coder, output_file, filenames, synsets, labels):
    """
    Processes and saves list of images as TFRecords.

    :param coder: instance of ImageCoder to provide TensorFlow image coding utils
    :param output_file: string, unique identifier specifying the data set
    :param filenames: list of strings; each string is a path to an image file
    :param synsets: list of strings; each string is a unique WordNet ID
    :param labels: map of string to integer; id for all synset labels
    """
    writer = tf.io.TFRecordWriter(output_file)

    for filename, synset in zip(filenames, synsets):
        image_buffer, height, width = _process_image(filename, coder)
        label = labels[synset]
        example = _convert_to_example(filename, image_buffer, label, synset, height, width)
        writer.write(example.SerializeToString())

    writer.close()



def _process_dataset(filenames, synsets, labels, output_directory, prefix, num_shards):
    """
    Processes and saves list of images as TFRecords.

    :param filenames: list of strings; each string is a path to an image file
    :param synsets: list of strings; each string is a unique WordNet ID
    :param labels: map of string to integer; id for all synset labels
    :param output_directory: path where output files should be created
    :param prefix: string; prefix for each file
    :param num_shards: number of chucks to split the filenames into

    :return: list of TFRecord file paths created from processing the dataset
    """

    _check_or_create_dir(output_directory)
    chunksize = int(math.ceil(len(filenames)/num_shards))
    coder = ImageCoder()

    files = []

    for shard in range(num_shards):
        chunk_files = filenames[shard * chunksize: (shard + 1) * chunksize]
        chunk_synsets = synsets[shard * chunksize: (shard + 1) * chunksize]
        output_file = os.path.join(output_directory, '%s-%.5d-of-%.5d' % (prefix, shard, num_shards))

        _process_image_files_batch(coder, output_file, chunk_files, chunk_synsets, labels)
        logging.info('Finished writing file: %s' % output_file)
        files.append(output_file)
    return files



def convert_to_tf_records(raw_data_dir):
    """
    Convert the ImageNet dataset into TFRecord dumps.
    """

    # Shuffle training records to ensure we are distributing classes across the batches.
    random.seed(0)
    def make_shuffle_idx(n):
        order = list(range(n))
        random.shuffle(order)
        return order

    # Glob all the training files
    training_files = tf.io.gfile.glob(os.path.join(raw_data_dir, TRAINING_DIRECTORY, '*', '*.JPEG'))

    # Get training file synset labels from the directory name
    training_synsets = [os.path.basename(os.path.dirname(f)) for f in training_files]

    # Shuffle the training set
    training_shuffle_idx = make_shuffle_idx(len(training_files))
    training_files = [training_files[i] for i in training_shuffle_idx]
    training_synsets = [training_synsets[i] for i in training_shuffle_idx]

    # Glob all the validation files
    validation_files = tf.io.gfile.glob(os.path.join(raw_data_dir, VALIDATION_DIRECTORY, '*', '*.JPEG'))

    # Get validation file synset labels from the directory name
    validation_synsets = [os.path.basename(os.path.dirname(f)) for f in validation_files]

    # Create unique ids for all synsets (this is a dictionary, synset -> integer)
    labels = {v: k + 1 for k, v in enumerate(sorted(set(validation_synsets + training_synsets)))}

    # Create training data
    logging.info('Processing the training data.')
    training_records = _process_dataset(training_files, training_synsets, labels,
                                        os.path.join(OUTPUT_DIRECTORY, TRAINING_DIRECTORY), TRAINING_DIRECTORY,
                                        TRAINING_SHARDS)

    # Create validation data
    logging.info('Processing the validation data.')
    validation_records = _process_dataset(validation_files, validation_synsets, labels,
                                          os.path.join(OUTPUT_DIRECTORY, VALIDATION_DIRECTORY), VALIDATION_DIRECTORY,
                                          VALIDATION_SHARDS)

    return training_records, validation_records



if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    convert_to_tf_records(DATA_DIRECTORY)