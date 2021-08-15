import numpy as np
import tensorflow as tf



def piecewise_scheduler(boundaries, decay_rates, base_rate=0.1, boundaries_as='epochs', num_images=None,
                        batch_size=None):
    # use rate while <= boundary
    # iterations: the number of training steps this optimizer has run
    assert boundaries_as in ['epochs', 'iterations'], 'boundaries_as must be in [epochs, iterations]'

    if boundaries_as == 'epochs':
        iterations_per_epoch = np.floor(num_images/batch_size)
        boundary_iterations = [(iterations_per_epoch * epoch) - 1 for epoch in boundaries]
    else:
        boundary_iterations = boundaries

    boundary_values = [base_rate * decay for decay in decay_rates]

    return tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundary_iterations, boundary_values)