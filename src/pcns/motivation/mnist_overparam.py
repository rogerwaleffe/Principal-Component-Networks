import numpy as np
import tensorflow as tf

from src.pcns.compression_helpers import tf_pca



def get_a_mnist_data_set(num_val=5000):
    mnist = tf.keras.datasets.mnist

    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    xTrain, xTest = xTrain / 255.0, xTest / 255.0

    # Reshape from images to vectors
    xTrain = xTrain.reshape((xTrain.shape[0], xTrain.shape[1] * xTrain.shape[2]))
    xTest = xTest.reshape((xTest.shape[0], xTest.shape[1] * xTest.shape[2]))

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

    print("{}, {}, {}, {}, {}, {}".format(xTrain.shape, xVal.shape, xTest.shape, yTrain.shape, yVal.shape, yTest.shape))

    return (xTrain, yTrain), (xVal, yVal), (xTest, yTest)



def get_compiled_model(hidden_units, input_shape=784, output_shape=10):
    full_model = tf.keras.Sequential()
    full_model.add(tf.keras.layers.InputLayer(input_shape=input_shape, name='input'))

    # Hidden Layers
    full_model.add(tf.keras.layers.Dense(hidden_units, activation='sigmoid', name='fc1'))

    # Output Layer
    full_model.add(tf.keras.layers.Dense(output_shape, activation='sigmoid', name='output'))

    full_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                       optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

    intermediate_models = {'fc1': tf.keras.Model(inputs=full_model.input, outputs=full_model.get_layer('fc1').output)}

    return full_model, intermediate_models



if __name__ == "__main__":
    EPOCHS = 50
    BATCH_SIZE = 60

    num_to_avg = 5
    dims = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750] # , 800, 850, 900, 950, 1000,
            # 1050, 1100, 1150, 1200, 1250]

    test_acc = []
    variances = []
    for h in dims:
        local_dim_test_acc = []
        local_dim_variances = []
        for _ in range(num_to_avg):
            model, partial_models = get_compiled_model(h)
            (train_x, train_y), (val_x, val_y), (test_x, test_y) = get_a_mnist_data_set()

            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.0005, patience=2,
                                                              restore_best_weights=True)
            model.fit(train_x, train_y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(val_x, val_y),
                      callbacks=[early_stopping])

            intermediate_out = partial_models['fc1'].predict(train_x)
            mu, V, e = tf_pca(intermediate_out, h, num_as_threshold=False, conv=False, verbose=True)

            score = model.evaluate(test_x, test_y, batch_size=test_x.shape[0], verbose=0)
            local_dim_test_acc.append(score[1])

            local_dim_variances.append([x for x in e.numpy()])

        test_acc.append(local_dim_test_acc)
        variances.append(local_dim_variances)

    test_acc = np.array(test_acc)
    np.save('test_acc.npy', test_acc)

    variances = np.array(variances)
    np.save('variances.npy', variances)