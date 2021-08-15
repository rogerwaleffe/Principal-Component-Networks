import numpy as np
import tensorflow as tf



def get_a_cifar10_data_set(num_val=5000):
    cifar10 = tf.keras.datasets.cifar10

    (xTrain, yTrain), (xTest, yTest) = cifar10.load_data()
    xTrain, xTest = xTrain / 255.0, xTest / 255.0

    yTrain = yTrain.reshape((yTrain.shape[0],))
    yTest = yTest.reshape((yTest.shape[0],))

    # One hot encoding
    NUM_CLASSES = 10
    yTrain = np.eye(NUM_CLASSES)[yTrain]
    yTest = np.eye(NUM_CLASSES)[yTest]

    # Create Validation Set
    random_indices = np.random.permutation(np.arange(xTrain.shape[0]))

    xVal = xTrain[random_indices[:num_val]]
    yVal = yTrain[random_indices[:num_val]]

    xTrain = xTrain[random_indices[num_val:]]
    yTrain = yTrain[random_indices[num_val:]]

    print("Dataset created. Sizes:\n(train, val, test, yTrain, yVal, yTest): ({}, {}, {}, {}, {}, {})"
          "".format(xTrain.shape, xVal.shape, xTest.shape, yTrain.shape, yVal.shape, yTest.shape))

    xTrain = np.array(xTrain, dtype=np.float32)
    xVal = np.array(xVal, dtype=np.float32)
    xTest = np.array(xTest, dtype=np.float32)

    return (xTrain, yTrain), (xVal, yVal), (xTest, yTest)



def create_cifar10_data_set(data_directory):
    # SET UP DATA SET
    # 32x32x3 images
    cifar10 = tf.keras.datasets.cifar10

    (xTrain, yTrain), (xTest, yTest) = cifar10.load_data()
    xTrain, xTest = xTrain / 255.0, xTest / 255.0

    yTrain = yTrain.reshape((yTrain.shape[0],))
    yTest = yTest.reshape((yTest.shape[0],))

    # One hot encoding
    NUM_CLASSES = 10
    yTrain = np.eye(NUM_CLASSES)[yTrain]
    yTest = np.eye(NUM_CLASSES)[yTest]

    # Create Validation Set
    VAL_SIZE = 5000
    random_indices = np.random.permutation(np.arange(xTrain.shape[0]))

    xVal = xTrain[random_indices[:VAL_SIZE]]
    yVal = yTrain[random_indices[:VAL_SIZE]]

    xTrain = xTrain[random_indices[VAL_SIZE:]]
    yTrain = yTrain[random_indices[VAL_SIZE:]]

    print("(train, val, test, yTrain, yVal, yTest): ({}, {}, {}, {}, {}, {})"
          "".format(xTrain.shape, xVal.shape, xTest.shape, yTrain.shape, yVal.shape, yTest.shape))

    np.save(data_directory+'cifar10_xTrain.npy', xTrain)
    np.save(data_directory+'cifar10_xVal.npy', xVal)
    np.save(data_directory+'cifar10_xTest.npy', xTest)
    np.save(data_directory+'cifar10_yTrain.npy', yTrain)
    np.save(data_directory+'cifar10_yVal.npy', yVal)
    np.save(data_directory+'cifar10_yTest.npy', yTest)

    return (xTrain, yTrain), (xVal, yVal), (xTest, yTest)



def load_cifar10_data_set(data_directory):
    xTrain = np.load(data_directory + 'cifar10_xTrain.npy')
    xVal = np.load(data_directory + 'cifar10_xVal.npy')
    xTest = np.load(data_directory + 'cifar10_xTest.npy')
    yTrain = np.load(data_directory + 'cifar10_yTrain.npy')
    yVal = np.load(data_directory + 'cifar10_yVal.npy')
    yTest = np.load(data_directory + 'cifar10_yTest.npy')

    print("(train, val, test, yTrain, yVal, yTest): ({}, {}, {}, {}, {}, {})"
          "".format(xTrain.shape, xVal.shape, xTest.shape, yTrain.shape, yVal.shape, yTest.shape))

    return (xTrain, yTrain), (xVal, yVal), (xTest, yTest)