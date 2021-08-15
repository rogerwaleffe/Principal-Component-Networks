import tensorflow as tf

from src.pcns_v2.helpers.layer_helpers import DenseTransformLayer, Conv2DTransformLayer, Conv2DExplicitPadding, cift



"""
TensorFlow PCA Based Compression Implementation Helpers
"""
def decode_cc(string_value):
    # 't0.25': use threshold 0.25
    # 'd64': use a fixed dimension of 64
    # 'f0.25': use fraction 0.25
    num_type = string_value[0]
    num = float(string_value[1:])
    return num_type, num



# @tf.function
def tf_pca(X, config, centering=True, conv=False, verbose=True, prefix=''):
    prefix = ' ' + prefix
    if verbose:
        print("Computing PCA{}: input shape {}, config {}, centering {}, conv {}"
              .format(prefix, tf.shape(X), config, centering, conv))

    if conv:
        X = tf.reshape(X, [tf.shape(X)[0] * tf.shape(X)[1] * tf.shape(X)[2], tf.shape(X)[3]])

    N = tf.shape(X)[0]
    d = tf.shape(X)[1]

    if centering:
        avg = tf.math.reduce_mean(X, axis=0)
        X = tf.math.subtract(X, avg)
    else:
        avg = None

    # form the sample covariance matrix
    S = tf.linalg.matmul(X, X, transpose_a=True)
    S = tf.math.divide(S, tf.cast(N - 1, dtype=S.dtype))

    # symmetric eigenvalue solver, routine returns them in non-decreasing order, column V[:, i] is the eigenvector
    # routine returns real numbers, but occasionally one is slightly negative (should be all non negative)
    e, V = tf.linalg.eigh(S)
    # print("PCA{}: ".format(prefix), end="")
    # print(*e.numpy())

    num_type, num = decode_cc(config)
    if num_type == 't':
        count = tf.math.count_nonzero(tf.math.greater_equal(e, num), dtype=tf.int32)
        dim = count
        if verbose:
            print("Finished PCA{}: using fixed variance threshold, {} eigenvalues greater than or equal to "
                  "{} out of {}, max eigenvalue {}".format(prefix, count, num, tf.shape(e), e[d-1]))
    elif num_type == 'd':
        dim = int(num)
        if verbose:
            print("Finished PCA{}: using fixed dimension {} out of {}, resulting cutoff variance {}, max variance {}"
                  "".format(prefix, dim, tf.shape(e), e[d-dim], e[d-1]))
    elif num_type == 'f':
        dim = tf.cast(tf.cast(d, tf.float32)*num, tf.int32)
        if verbose:
            print("Finished PCA{}: using fixed dimension {} (fraction {} of {}), resulting cutoff variance {}, "
                  "max variance {}".format(prefix, dim, num, tf.shape(e), e[d-dim], e[d-1]))
    else:
        raise Exception("Invalid config for tf_pca.")

    indices_top = tf.range(d-1, d-(dim+1), -1)
    indices_bot = tf.range(d-(dim+1), -1, -1)
    return avg, tf.gather(V, indices_top, axis=1), tf.gather(e, indices_top), \
           tf.gather(V, indices_bot, axis=1), tf.gather(e, indices_bot)



# @tf.function
def tf_transform_conv_weights(mu, V, W, b, transform_bias=True, undo_bias=False):
    # compute primed variables
    W_flat = tf.reshape(W, [tf.shape(W)[0] * tf.shape(W)[1], tf.shape(W)[2], tf.shape(W)[3]])
    W_flat = tf.transpose(W_flat, perm=[2, 1, 0])
    batch_V = tf.tile(tf.expand_dims(V, 0), [tf.shape(W_flat)[0], 1, 1])
    W_p_flat = tf.linalg.matmul(batch_V, W_flat, transpose_a=True)
    W_p_flat = tf.transpose(W_p_flat, perm=[2, 1, 0])
    W_p = tf.reshape(W_p_flat, [tf.shape(W)[0], tf.shape(W)[1], tf.shape(W_p_flat)[1], tf.shape(W)[3]])

    if transform_bias is False:
        return W_p, b

    if undo_bias is False:
        batch_mu = tf.tile(tf.reshape(mu, [1, 1, tf.shape(mu)[0]]), [tf.shape(W_flat)[0], 1, 1])
        T = tf.squeeze(tf.matmul(batch_mu, W_flat), axis=1)
        b_p = tf.math.add(b, tf.math.reduce_sum(T, axis=1))
    else:
        W_p_flat = tf.transpose(W_p_flat, perm=[2, 1, 0])
        batch_mu = tf.tile(tf.reshape(mu, [1, 1, tf.shape(mu)[0]]), [tf.shape(W_p_flat)[0], 1, 1])
        T = tf.squeeze(tf.matmul(batch_mu, W_p_flat), axis=1)
        b_p = tf.math.subtract(b, tf.math.reduce_sum(T, axis=1))

    return W_p, b_p



# @tf.function
def tf_transform_dense_weights(mu, V, W, b, transform_bias=True, undo_bias=False):
    W_p = tf.linalg.matmul(V, W, transpose_a=True)

    if transform_bias is False:
        return W_p, b

    if undo_bias is False:
        b_p = tf.math.add(b, tf.linalg.matmul(tf.expand_dims(mu, 0), W))
    else:
        b_p = tf.math.subtract(b, tf.linalg.matmul(tf.expand_dims(mu, 0), W_p))

    return W_p, b_p










def copy_bn_layer(bn_layer, indices=None):
    gamma, beta = bn_layer.weights[0], bn_layer.weights[1]
    moving_mean, moving_var = bn_layer.weights[2], bn_layer.weights[3]
    if indices is not None:
        gamma, beta = tf.gather(gamma, indices), tf.gather(beta, indices)
        moving_mean, moving_var = tf.gather(moving_mean, indices), tf.gather(moving_var, indices)

    # TODO: fix this renorm thing
    return tf.keras.layers.BatchNormalization(axis=bn_layer.axis, momentum=bn_layer.momentum, epsilon=bn_layer.epsilon,
           center=bn_layer.center, scale=bn_layer.scale, beta_initializer=cift(beta), gamma_initializer=cift(gamma),
           moving_mean_initializer=cift(moving_mean), moving_variance_initializer=cift(moving_var),
           beta_regularizer=bn_layer.beta_regularizer, gamma_regularizer=bn_layer.gamma_regularizer,
           beta_constraint=bn_layer.beta_constraint, gamma_constraint=bn_layer.gamma_constraint, renorm=bn_layer.renorm,
           #renorm_clipping=bn_layer.renorm_clipping, renorm_momentum=bn_layer.renorm_momentum,
           fused=bn_layer.fused, trainable=bn_layer.trainable, virtual_batch_size=bn_layer.virtual_batch_size,
           adjustment=bn_layer.adjustment, name=bn_layer.name)



def copy_conv_layer(l):
    if isinstance(l, tf.keras.layers.Conv2D):
        name = l.name
        o = tf.keras.layers.Conv2D
    elif isinstance(l, Conv2DExplicitPadding):
        name = l.name
        l = l.conv_layer
        o = Conv2DExplicitPadding
    else:
        raise Exception()

    W = l.weights[0]
    b = l.weights[1] if len(l.weights) > 1 else tf.zeros([W.shape[-1], ])

    return o(l.filters, l.kernel_size, strides=l.strides, padding=l.padding, data_format=l.data_format,
             dilation_rate=l.dilation_rate, groups=l.groups, activation=l.activation, use_bias=l.use_bias,
             kernel_initializer=cift(W), bias_initializer=cift(b), kernel_regularizer=l.kernel_regularizer,
             bias_regularizer=l.bias_regularizer, activity_regularizer=l.activity_regularizer,
             kernel_constraint=l.kernel_constraint, bias_constraint=l.bias_constraint, name=name)

def from_conv_layer(l, mu, Vt, Vb, include_offset=False, train_top_basis='NO', add_bias_if_nec=True):
    if isinstance(l, tf.keras.layers.Conv2D):
        name = l.name
        previous_layer_type = 'Conv2D'
    elif isinstance(l, Conv2DExplicitPadding):
        name = l.name
        l = l.conv_layer
        previous_layer_type = 'Conv2DExplicitPadding'
    else:
        raise Exception()

    W = l.weights[0]
    b = l.weights[1] if len(l.weights) > 1 else tf.zeros([W.shape[-1], ])

    Wt, bp = tf_transform_conv_weights(mu, Vt, W, b, transform_bias=mu is not None)
    Wb, _ = tf_transform_conv_weights(None, Vb, W, None, transform_bias=False) if Vb is not None else (None, None)

    return Conv2DTransformLayer(l.filters, l.kernel_size, mu, Vt, Vb, Wt, Wb, bp,
                                include_offset=include_offset, train_top_basis=train_top_basis,
                                add_bias_if_nec=add_bias_if_nec, previous_layer_type=previous_layer_type,
                                strides=l.strides, padding=l.padding, data_format=l.data_format,
                                dilation_rate=l.dilation_rate, groups=l.groups, activation=l.activation,
                                use_bias=l.use_bias, kernel_initializer=l.kernel_initializer,
                                bias_initializer=l.bias_initializer, kernel_regularizer=l.kernel_regularizer,
                                bias_regularizer=l.bias_regularizer, activity_regularizer=l.activity_regularizer,
                                kernel_constraint=l.kernel_constraint, bias_constraint=l.bias_constraint, name=name)

def from_conv_transform_layer(l):
    if l.train_top_basis != 'NO' or l.include_offset is False:
        if l.Vb.shape[-1] == 0:
            pass
        else:
            raise Exception("TBD on if it makes sense to undo transformation for this layer.")

    V = tf.concat([l.Vt, l.Vb], axis=1)
    W_tilde = tf.concat([l.conv_layer.weights[0], l.Wb], axis=2)
    b_tilde = l.conv_layer.weights[1] if len(l.conv_layer.weights) > 1 else tf.zeros([W_tilde.shape[-1], ])

    if l.previous_layer_used_bias is False:
        if l.mu is None:
            # b = b_tilde = None
            W, b = tf_transform_conv_weights(l.mu, tf.transpose(V), W_tilde, b_tilde, transform_bias=False)
        else:
            # b = ???
            raise Exception("A bias was added to the transform layer, can not go backwards to a layer without a bias.")
    else:
        if l.mu is None:
            # b = b_tilde
            W, b = tf_transform_conv_weights(l.mu, tf.transpose(V), W_tilde, b_tilde, transform_bias=False)
        else:
            # b = b_tilde-mu*W
            W, b = tf_transform_conv_weights(tf.squeeze(l.mu), tf.transpose(V), W_tilde, b_tilde, undo_bias=True)

    if l.previous_layer_type == 'Conv2D':
        o = tf.keras.layers.Conv2D
    elif l.previous_layer_type == 'Conv2DExplicitPadding':
        o = Conv2DExplicitPadding
    else:
        raise Exception()

    return o(l.filters, l.kernel_size, strides=l.conv_layer.strides, padding=l.previous_layer_padding,
             data_format=l.conv_layer.data_format, dilation_rate=l.conv_layer.dilation_rate, groups=l.conv_layer.groups,
             activation=l.activation, use_bias=l.previous_layer_used_bias, kernel_initializer=cift(W),
             bias_initializer=cift(b), kernel_regularizer=l.conv_layer.kernel_regularizer,
             bias_regularizer=l.conv_layer.bias_regularizer, activity_regularizer=l.conv_layer.activity_regularizer,
             kernel_constraint=l.conv_layer.kernel_constraint, bias_constraint=l.conv_layer.bias_constraint,
             name=l.name)



def copy_dense_layer(l):
    W = l.weights[0]
    b = l.weights[1] if len(l.weights) > 1 else tf.zeros([W.shape[-1], ])

    o = tf.keras.layers.Dense
    return o(l.units, activation=l.activation, use_bias=l.use_bias, kernel_initializer=cift(W),
             bias_initializer=cift(b), kernel_regularizer=l.kernel_regularizer, bias_regularizer=l.bias_regularizer,
             activity_regularizer=l.activity_regularizer, kernel_constraint=l.kernel_constraint,
             bias_constraint=l.bias_constraint, name=l.name)

def from_dense_layer(l, mu, Vt, Vb, include_offset=False, train_top_basis='NO', add_bias_if_nec=True):
    W = l.weights[0]
    b = l.weights[1] if len(l.weights) > 1 else tf.zeros([W.shape[-1], ])

    Wt, bp = tf_transform_dense_weights(mu, Vt, W, b, transform_bias=mu is not None)
    Wb, _ = tf_transform_dense_weights(None, Vb, W, None, transform_bias=False) if Vb is not None else (None, None)

    return DenseTransformLayer(l.units, mu, Vt, Vb, Wt, Wb, bp,
                               include_offset=include_offset, train_top_basis=train_top_basis,
                               add_bias_if_nec=add_bias_if_nec,
                               activation=l.activation, use_bias=l.use_bias, kernel_initializer=l.kernel_initializer,
                               bias_initializer=l.bias_initializer, kernel_regularizer=l.kernel_regularizer,
                               bias_regularizer=l.bias_regularizer, activity_regularizer=l.activity_regularizer,
                               kernel_constraint=l.kernel_constraint, bias_constraint=l.bias_constraint, name=l.name)

def from_dense_transform_layer(l):
    if l.train_top_basis != 'NO' or l.include_offset is False:
        if l.Vb.shape[-1] == 0:
            pass
        else:
            raise Exception("TBD on if it makes sense to undo transformation for this layer.")

    V = tf.concat([l.Vt, l.Vb], axis=1)
    W_tilde = tf.concat([l.dense_layer.weights[0], l.Wb], axis=0)
    b_tilde = l.dense_layer.weights[1] if len(l.dense_layer.weights) > 1 else tf.zeros([W_tilde.shape[-1], ])

    if l.previous_layer_used_bias is False:
        if l.mu is None:
            W, b = tf_transform_dense_weights(l.mu, tf.transpose(V), W_tilde, b_tilde, transform_bias=False)
        else:
            raise Exception("A bias was added to the transform layer, can not go backwards to a layer without a bias.")
    else:
        if l.mu is None:
            W, b = tf_transform_dense_weights(l.mu, tf.transpose(V), W_tilde, b_tilde, transform_bias=False)
        else:
            W, b = tf_transform_dense_weights(tf.squeeze(l.mu), tf.transpose(V), W_tilde, b_tilde, undo_bias=True)

    o = tf.keras.layers.Dense
    return o(l.units, activation=l.activation, use_bias=l.previous_layer_used_bias, kernel_initializer=cift(W),
             bias_initializer=cift(tf.squeeze(b)), kernel_regularizer=l.dense_layer.kernel_regularizer,
             bias_regularizer=l.dense_layer.bias_regularizer, activity_regularizer=l.dense_layer.activity_regularizer,
             kernel_constraint=l.dense_layer.kernel_constraint, bias_constraint=l.dense_layer.bias_constraint,
             name=l.name)