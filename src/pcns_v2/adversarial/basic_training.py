import argparse
import numpy as np
import tensorflow as tf

from src.pcns_v2.datasets.cifar10 import get_dataset, input_fn
from src.pcns_v2.architectures.resnet import ResNet
from src.pcns_v2.adversarial.helpers import advTrainModel
from src.pcns_v2.helpers.optimization_helpers import piecewise_scheduler



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ResNet adversarial training')
    parser.add_argument('--adv_method', type=str, default='None', help="Method to generate adversarial examples")
    parser.add_argument('--scale', type=float, default=1.0, help="Width of the network")

    args = parser.parse_args()
    adv_method = args.adv_method
    scale = args.scale
    print("Method to generate adversarial samples: %s" % adv_method)
    print("Width factor: %f" % scale)



    TOTAL_EPOCHS = 164
    BATCH_SIZE = 128
    VERBOSE = 1  # 1 = progress bar, 2 = one line per epoch



    # Get train/test data sets
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = get_dataset()
    num_train = train_x.shape[0]
    num_test = test_x.shape[0]

    train_dataset = input_fn(train_x, train_y, num_epochs=TOTAL_EPOCHS, batch_size=BATCH_SIZE,
                                 is_training=True, normalize=False)
    test_dataset = input_fn(test_x, test_y, batch_size=BATCH_SIZE, normalize=False)



    # Create ResNet model
    resnet = ResNet(input_shape=(32, 32, 3), num_classes=10, bottleneck=False, num_filters_at_start=int(16*scale),
                    initial_kernel_size=3, initial_conv_strides=1, initial_pool_size=None, initial_pool_strides=None,
                    num_residual_blocks_per_stage=[3, 3, 3], first_block_strides_per_stage=[1, 2, 2], kernel_size=3,
                    project_first_residual=True, version='V1', data_format='channels_last')
    model, _ = resnet.get_model()
    model = advTrainModel(model, attack=adv_method)

    lr_schedule = piecewise_scheduler([82, 123], [1.0, 0.1, 0.01], base_rate=0.1, boundaries_as='epochs',
                                      num_images=num_train, batch_size=BATCH_SIZE)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    print(model.base_model.summary())
    resnet.print_ResNet(model.base_model)



    # Train model
    history = model.fit(train_dataset, epochs=TOTAL_EPOCHS, steps_per_epoch=np.floor(num_train/BATCH_SIZE),
                        validation_data=test_dataset, validation_steps=np.ceil(num_test/BATCH_SIZE), validation_freq=1,
                        verbose=VERBOSE)



    # Best model in terms of validation accuracy, would have to checkpoint this model during training to evaluate
    # all three natural/FGSM/PGD accuracies (see experiment_scripts/resnet_on_cifar for model checkpointing)
    best_epoch = np.argmax(history.history['val_accuracy'])
    best_acc = history.history['val_accuracy'][best_epoch]
    print("Best accuracy: {:.5f} after epoch {}".format(best_acc, best_epoch+1))



    # Evaluate fully trained model
    print("Evaluation of fully trained model:")

    print('Natural Loss and Acc')
    model.adv_attack = 'None'
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    score = model.evaluate(test_dataset, steps=np.ceil(num_test/BATCH_SIZE), verbose=0)
    print("Test Loss {:.5f}, Test Acc: {:.5f}".format(score[0], score[1]))

    print('FGSM Loss and Acc')
    model.adv_attack = 'FGSM'
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    score = model.evaluate(test_dataset, steps=np.ceil(num_test/BATCH_SIZE), verbose=0)
    print("Test Loss {:.5f}, Test Acc: {:.5f}".format(score[0], score[1]))

    print('PGD Loss and Acc')
    model.adv_attack = 'PGD'
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    score = model.evaluate(test_dataset, steps=np.ceil(num_test/BATCH_SIZE), verbose=0)
    print("Test Loss {:.5f}, Test Acc: {:.5f}".format(score[0], score[1]))





    # # Something like this might be helpful:
    # loss = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    # test_set_losses = [] # hold the loss of every example from the test set
    # for idx, batch in enumerate(test_dataset):
    #     x, y = batch
    #     # attack x
    #     y_pred = model.base_model(x, training=False)
    #     batch_losses = loss(y, y_pred)
    #     test_set_losses.append(batch_losses)
    #
    # test_set_losses = tf.concat(test_set_losses, axis=0).numpy() # array of length 10,000k
