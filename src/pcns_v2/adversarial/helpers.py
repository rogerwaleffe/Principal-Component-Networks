import tensorflow as tf



def FGSMAttack(images, labels, model, loss_fxn, epsilon=8.0, random_start=False): # False for backward compatibility
    if random_start:
        # method 1
        # adv_images = images + epsilon/2 * tf.sign(tf.random.uniform(tf.shape(images), -1.0, 1.0))
        # adv_images = tf.clip_by_value(adv_images, 0, 255)
        # new_epsilon = epsilon/2

        # method 2
        adv_images = images + tf.random.uniform(tf.shape(images), -epsilon, epsilon)
        adv_images = tf.clip_by_value(adv_images, 0, 255)
        new_epsilon = epsilon
    else:
        adv_images = images
        new_epsilon = epsilon

    with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape:
        tape.watch(adv_images)
        predictions = model(adv_images, training=False)
        loss = loss_fxn(labels, predictions)

    # Get the gradients of the loss w.r.t to the input.
    gradient = tape.gradient(loss, adv_images)

    # Get the sign of the gradients to create the perturbation
    signed_grad = tf.sign(gradient)
    adv_images = tf.stop_gradient(adv_images + new_epsilon*signed_grad)
    adv_images = tf.clip_by_value(adv_images, images - epsilon, images + epsilon)
    adv_images = tf.clip_by_value(adv_images, 0, 255)

    return adv_images



def PGDAttack(images, labels, model, loss_fxn, epsilon=8.0, num_steps=10, step_size=2.0, random_start=True):
    if random_start:
        adv_images = images + tf.random.uniform(tf.shape(images), -epsilon, epsilon)
        adv_images = tf.clip_by_value(adv_images, 0, 255)
    else:
        adv_images = images

    for _ in range(num_steps):
        with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape:
            tape.watch(adv_images)
            predictions = model(adv_images, training=False)
            loss = loss_fxn(labels, predictions)

        gradient = tape.gradient(loss, adv_images)

        # Get the sign of the gradients to create the perturbation
        signed_grad = tf.sign(gradient)
        adv_images = tf.stop_gradient(adv_images + step_size*signed_grad)
        adv_images = tf.clip_by_value(adv_images, images - epsilon, images + epsilon)
        adv_images = tf.clip_by_value(adv_images, 0, 255)

    return adv_images



class advTrainModel(tf.keras.Model):
    def __init__(self, base_model, num_steps=10, step_size=2.0, epsilon=8.0, random_start=True, attack='None',
                 test_attack=None):
        super(advTrainModel, self).__init__()

        assert attack in ['PGD', 'FGSM', 'None'], 'attack must be in [PGD, FGSM, None]'
        if test_attack is not None:
            assert test_attack in ['PGD', 'FGSM', 'None'], 'test_attack must be in [PGD, FGSM, None]'

        self.base_model = base_model
        self.adv_num_steps = num_steps
        self.adv_step_size = step_size
        self.adv_epsilon = epsilon
        self.adv_random_start = random_start
        self.adv_attack = attack

        if test_attack is None:
            self.test_attack = self.adv_attack
        else:
            self.test_attack = test_attack

        self.loss_fxn = None

    def compile(self, loss=None, **kwargs):
        super(advTrainModel, self).compile(loss=loss, **kwargs)

        self.loss_fxn = loss

    # def load_weights(self, filepath, **kwargs):
    #     self.base_model.load_weights(filepath, **kwargs)

    # def save(self, filepath, **kwargs):
    #     self.base_model.save(filepath, **kwargs)

    # def save_weights(self, filepath, **kwargs):
    #     self.base_model.save_weights(filepath, **kwargs)

    def train_step(self, data):
        x, y = data

        if self.adv_attack == 'PGD':
            x_train = PGDAttack(x, y, self.base_model, self.loss_fxn, self.adv_epsilon, self.adv_num_steps,
                                self.adv_step_size, self.adv_random_start)
        elif self.adv_attack == 'FGSM':
            x_train = FGSMAttack(x, y, self.base_model, self.loss_fxn, self.adv_epsilon, self.adv_random_start)
        else:
            x_train = x

        with tf.GradientTape() as tape:
            y_pred = self.base_model(tf.stop_gradient(x_train), training=True) # might not need stop_gradient here
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.compiled_metrics.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data

        if self.test_attack == 'PGD':
            x_test = PGDAttack(x, y, self.base_model, self.loss_fxn, self.adv_epsilon, self.adv_num_steps,
                               self.adv_step_size, self.adv_random_start)
        elif self.test_attack == 'FGSM':
            x_test = FGSMAttack(x, y, self.base_model, self.loss_fxn, self.adv_epsilon, self.adv_random_start)
        else:
            x_test = x

        y_pred = self.base_model(x_test, training=False)

        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}