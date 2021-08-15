# TODO: add a percentage of the max threshold option
import tensorflow as tf
from src.pcns.layer_helpers import DensePCALayer, Conv2DPCALayer
from src.pcns.layer_helpers import constant_initializer_from_tensor as cift



# TensorFlow PCA Based Compression Implementation
def decode_cc(string_value):
    ut = True if string_value[0] == 't' else False
    num = float(string_value[1:]) if ut else int(string_value[1:])
    return num, ut



def tf_pca_no_centering(X, num, num_as_threshold=False, conv=False, verbose=True, prefix=''):
    if conv:
        X = tf.reshape(X, [tf.shape(X)[0] * tf.shape(X)[1] * tf.shape(X)[2], tf.shape(X)[3]])

    if verbose:
        print("PCA No Centering{}: computing, input shape {}".format(prefix, tf.shape(X)))

    N = tf.shape(X)[0]
    d = tf.shape(X)[1]

    # form the sample covariance matrix
    S = tf.linalg.matmul(X, X, transpose_a=True)
    S = tf.math.divide(S, tf.cast(N - 1, dtype=S.dtype))

    # symmetric eigenvalue solver, routine returns them in non-decreasing order, column V[:, i] is the eigenvector
    # routine returns real numbers, but occasionally one is slightly negative (should be all non negative)
    e, V = tf.linalg.eigh(S)

    if num_as_threshold:
        count = tf.math.count_nonzero(tf.math.greater_equal(e, num), dtype=tf.int32)
        dim = count

        if verbose:
            print("PCA No Centering{}: done, using fixed variance threshold, {} eigenvalues greater than or equal to "
                  "{} out of {}, max eigenvalue {}".format(prefix, count, num, tf.shape(e), e[d-1]))
    else:
        dim = num
        if verbose:
            print("PCA No Centering{}: done, using fixed dimension {} out of {}, resulting cutoff variance {}, "
                  "max variance {}".format(prefix, dim, tf.shape(e), e[d-dim], e[d-1]))

    indices = tf.range(d-1, d-(dim+1), -1)
    return None, tf.gather(V, indices, axis=1), tf.gather(e, indices)



# @tf.function
def tf_pca(X, num, num_as_threshold=False, conv=False, verbose=True, prefix=''):
    if conv:
        X = tf.reshape(X, [tf.shape(X)[0] * tf.shape(X)[1] * tf.shape(X)[2], tf.shape(X)[3]])

    if verbose:
        print("PCA{}: computing, input shape {}".format(prefix, tf.shape(X)))

    N = tf.shape(X)[0]
    d = tf.shape(X)[1]

    avg = tf.math.reduce_mean(X, axis=0)
    X_centered = tf.math.subtract(X, avg)

    # form the sample covariance matrix
    S = tf.linalg.matmul(X_centered, X_centered, transpose_a=True)
    S = tf.math.divide(S, tf.cast(N - 1, dtype=S.dtype))

    # symmetric eigenvalue solver, routine returns them in non-decreasing order, column V[:, i] is the eigenvector
    # routine returns real numbers, but occasionally one is slightly negative (should be all non negative)
    e, V = tf.linalg.eigh(S)
    # print("PCA{}: ".format(prefix), end="")
    # print(*e.numpy())

    if num_as_threshold:
        count = tf.math.count_nonzero(tf.math.greater_equal(e, num), dtype=tf.int32)
        dim = count

        if verbose:
            print("PCA{}: done, using fixed variance threshold, {} eigenvalues greater than or equal to "
                  "{} out of {}, max eigenvalue {}".format(prefix, count, num, tf.shape(e), e[d-1]))
    else:
        dim = num
        if verbose:
            print("PCA{}: done, using fixed dimension {} out of {}, resulting cutoff variance {}, max variance {}"
                  "".format(prefix, dim, tf.shape(e), e[d-dim], e[d-1]))

    indices = tf.range(d-1, d-(dim+1), -1)
    return avg, tf.gather(V, indices, axis=1), tf.gather(e, indices)



# @tf.function
def tf_kill_columns_or_filters(W, b, mu, V, W2, num, num_as_threshold=False, conv=False, verbose=True,
                               return_indices=False, prefix=''):
    row_sums = tf.math.reduce_sum(tf.math.abs(V), axis=1)
    row_sums_argsort = tf.argsort(row_sums, direction='DESCENDING')

    if verbose:
        print("Analyzing Columns/Filters{}: row sum max {}, mean {}, std {}"
              "".format(prefix, row_sums[row_sums_argsort[0]], tf.math.reduce_mean(row_sums),
                        tf.math.reduce_std(row_sums)))

    if num_as_threshold:
        count = tf.math.count_nonzero(tf.math.greater_equal(row_sums, num), dtype=tf.int32)
        dim = count

        if verbose:
            print("Analyzing Columns/Filters{}: using fixed threshold, retaining {} columns/filters with row sum "
                  "greater than or equal to {} out of {}".format(prefix, dim, num, tf.shape(row_sums)[0]))
    else:
        dim = num
        if verbose:
            print("Analyzing Columns/Filters{}: using fixed dimension, retaining {} columns/filters out of {}, "
                  "resulting cutoff row sum {}"
                  "".format(prefix, dim, tf.shape(row_sums)[0], row_sums[row_sums_argsort[dim-1]]))

    keep_columns = tf.gather(row_sums_argsort, tf.range(0, dim, 1))

    b = tf.gather(b, keep_columns)
    V = tf.gather(V, keep_columns, axis=0)
    mu = tf.gather(mu, keep_columns)
    if conv is False:
        W = tf.gather(W, keep_columns, axis=1)
        W2 = tf.gather(W2, keep_columns, axis=0)
    else:
        W = tf.gather(W, keep_columns, axis=3)
        W2 = tf.gather(W2, keep_columns, axis=2)

    if return_indices is False:
        return dim, W, b, mu, V, W2
    else:
        return dim, W, b, mu, V, W2, keep_columns



# @tf.function
def tf_kill_filters_to_dense(W, b, mu, V, W2, num, num_as_threshold=False, verbose=True, prefix=''):
    tile_size = tf.math.floordiv(tf.shape(V)[0], tf.shape(W)[3]) # this has to be an integer

    row_sums = tf.math.reduce_sum(tf.math.abs(V), axis=1)
    filter_ids = tf.range(0, tf.shape(W)[3], 1)
    filter_ids_all = tf.tile(filter_ids, [tile_size])
    filter_sums = tf.math.unsorted_segment_sum(row_sums, filter_ids_all, tf.shape(W)[3]) # not segment_sum!
    filter_sums_argsort = tf.argsort(filter_sums, direction='DESCENDING')

    if verbose:
        print("Analyzing Filters Before Dense Layer{}: filter sum max {}, mean {}, std {}"
              "".format(prefix, filter_sums[filter_sums_argsort[0]], tf.math.reduce_mean(filter_sums),
                        tf.math.reduce_std(filter_sums)))

    if num_as_threshold:
        count = tf.math.count_nonzero(tf.math.greater_equal(filter_sums, num), dtype=tf.int32)
        dim = count

        if verbose:
            print("Analyzing Filters Before Dense Layer{}: using fixed threshold, retaining {} filters with filter "
                  "sum greater than or equal to {} out of {}".format(prefix, dim, num, tf.shape(filter_sums)[0]))
    else:
        dim = num
        if verbose:
            print("Analyzing Filters Before Dense Layer{}: using fixed dimension, retaining {} filters out of {}, "
                  "resulting cutoff filter sum {}"
                  "".format(prefix, dim, tf.shape(filter_sums)[0], filter_sums[filter_sums_argsort[dim-1]]))

    keep_filters = tf.gather(filter_sums_argsort, tf.range(0, dim, 1))

    keep_rows = tf.reshape(tf.tile(keep_filters, [tile_size]), [tile_size, tf.shape(keep_filters)[0]])
    offset = tf.reshape(tf.range(0, tf.shape(row_sums)[0], tf.shape(W)[3]), [tile_size, 1])
    keep_rows = tf.reshape(tf.math.add(keep_rows, offset), [tf.shape(keep_filters)[0]*tile_size])

    W = tf.gather(W, keep_filters, axis=3)
    b = tf.gather(b, keep_filters)

    V = tf.gather(V, keep_rows, axis=0)
    mu = tf.gather(mu, keep_rows)
    W2 = tf.gather(W2, keep_rows, axis=0)

    return dim, W, b, mu, V, W2



# @tf.function
def tf_kill_outputs(W, b, mu, V, W2, num, num_as_threshold=False, conv=False, conv_to_dense=False, verbose=True,
                    return_indices=False, prefix=''):
    if conv_to_dense is False:
        return tf_kill_columns_or_filters(W, b, mu, V, W2, num, num_as_threshold=num_as_threshold, conv=conv,
                                          verbose=verbose, return_indices=return_indices, prefix=prefix)
    else:
        return tf_kill_filters_to_dense(W, b, mu, V, W2, num, num_as_threshold=num_as_threshold, verbose=verbose,
                                        prefix=prefix)



def tf_get_indices_from_muVs(muVs, V, num, num_as_threshold=False, verbose=True, use_all=True, evals=None):
    if evals is None:
        def op(X, _): return tf.math.reduce_sum(tf.math.abs(X), axis=1)
    else:
        def op(X, e): return tf.reshape(tf.linalg.matmul(tf.math.abs(X), tf.reshape(e, [X.shape[1], 1])), [X.shape[0],])

    if use_all is True:
        row_sums = op(V, evals)
        for _, V_p, e_p in muVs:
            row_sums = tf.add(op(V_p, e_p), row_sums)
        row_sums = row_sums / (len(muVs) + 1)
        # print(tf.sort(row_sums))
    else:
        row_sums = op(V, evals)

    row_sums_argsort = tf.argsort(row_sums, direction='DESCENDING')

    if verbose:
        print("Analyzing Output Filters From ResNet Stage: row sum max {}, mean {}, std {}"
              "".format(row_sums[row_sums_argsort[0]], tf.math.reduce_mean(row_sums), tf.math.reduce_std(row_sums)))

    if num_as_threshold:
        count = tf.math.count_nonzero(tf.math.greater_equal(row_sums, num), dtype=tf.int32)
        dim = count

        if verbose:
            print("Analyzing Output Filters From ResNet Stage: using fixed threshold, retaining {} filters with row "
                  "sum greater than or equal to {} out of {}".format(dim, num, tf.shape(row_sums)[0]))
    else:
        dim = num
        if verbose:
            print("Analyzing Output Filters From ResNet Stage: using fixed dimension, retaining {} columns/filters out "
                  "of {}, resulting cutoff row sum {}".format(dim, tf.shape(row_sums)[0],
                                                              row_sums[row_sums_argsort[dim-1]]))

    keep_filters = tf.gather(row_sums_argsort, tf.range(0, dim, 1))

    return keep_filters



# @tf.function
def tf_transform_conv_weights(mu, V, W, b, transform_bias=True):
    # compute primed variables
    W_flat = tf.reshape(W, [tf.shape(W)[0] * tf.shape(W)[1], tf.shape(W)[2], tf.shape(W)[3]])
    W_flat = tf.transpose(W_flat, perm=[2, 1, 0])
    batch_V = tf.tile(tf.expand_dims(V, 0), [tf.shape(W_flat)[0], 1, 1])
    W_p_flat = tf.linalg.matmul(batch_V, W_flat, transpose_a=True)
    W_p_flat = tf.transpose(W_p_flat, perm=[2, 1, 0])
    W_p = tf.reshape(W_p_flat, [tf.shape(W)[0], tf.shape(W)[1], tf.shape(W_p_flat)[1], tf.shape(W)[3]])

    if transform_bias is False:
        return W_p

    batch_mu = tf.tile(tf.reshape(mu, [1, 1, tf.shape(mu)[0]]), [tf.shape(W_flat)[0], 1, 1])
    T = tf.squeeze(tf.matmul(batch_mu, W_flat), axis=1)
    b_p = tf.math.add(b, tf.math.reduce_sum(T, axis=1))

    return W_p, b_p



# @tf.function
def tf_transform_dense_weights(mu, V, W, b):
    W_p = tf.linalg.matmul(V, W, transpose_a=True)
    b_p = tf.math.add(b, tf.linalg.matmul(tf.expand_dims(mu, 0), W))

    return W_p, b_p










def vgg_compression(compression_config, input_arr, overall_model, nums_as='dimension', verbose=True):
    assert nums_as in ['dimension', 'threshold'], 'nums_as should be in {dimension, threshold}'
    ut = False if nums_as == 'dimension' else True
    cc = compression_config


    # Get information about the layers in the original network
    all_layers = []
    n_full = {}
    last_conv = ""
    for l in overall_model.layers: # note: the input layer is not included in overall_model.layers
        if 'conv' in l.name or 'fc' in l.name or 'output' in l.name:
            n_full[l.name] = l.output_shape[-1]
        if 'conv' in l.name:
            last_conv = l.name
        all_layers.append(l.name)


    # Verify necessary config constraints
    cc_keys = list(cc.keys())
    cc_values = list(cc.values())

    assert cc_keys == list(n_full.keys()), 'compression config must have keys for all compute layers (conv, fc, output)'
    assert cc_values[0][0] is None, 'currently we are not considering compression of the network input'
    for index in range(len(cc_keys)):
        if cc_keys[index] in ['output']:
            continue
        if cc_values[index][1] is not None:
            assert cc_values[index+1][0] is not None, 'to kill outputs the next layer must perform PCA compression'
    assert cc_values[-1][1] is None, 'can not kill columns in the output layer'


    def advance_input_arr(cl, nl, arr):
        if cl == 'output':
            return None
        arr = overall_model.get_layer(cl)(arr)
        if nl is not None:
            curr_index = all_layers.index(cl)
            next_index = all_layers.index(nl)
            while curr_index < next_index - 1:
                curr_index += 1
                arr = overall_model.get_layer(all_layers[curr_index])(arr)
        return arr


    # Go through the layers and create the (possibly) compressed layers
    compressed_model = tf.keras.Sequential()
    compressed_model.add(tf.keras.layers.InputLayer(input_shape=overall_model.layers[0].input_shape[1:], name='input'))

    mu_c, V_c, W_n = None, None, None

    for index in range(len(cc_keys)):
        prev_layer = cc_keys[index-1] if index-1 >= 0 else None
        curr_layer = cc_keys[index]
        next_layer = cc_keys[index+1] if index+1 <= len(cc_keys)-1 else None

        n_c = n_full[curr_layer]
        W_c = overall_model.get_layer(curr_layer).weights[0] if W_n is None else W_n
        b_c = overall_model.get_layer(curr_layer).weights[1]
        W_n = overall_model.get_layer(next_layer).weights[0] if next_layer is not None else None

        mu_p, V_p = mu_c, V_c

        conv_c = True if 'conv' in curr_layer else False
        conv_n = (True if 'conv' in next_layer else False) if next_layer is not None else None

        # generic iteration
        if cc[curr_layer][0] is not None and cc[prev_layer][1] is None:
            mu_p, V_p, _ = tf_pca(input_arr, cc[curr_layer][0], num_as_threshold=ut, conv=conv_c, verbose=verbose)
        input_arr = advance_input_arr(curr_layer, next_layer, input_arr)
        if cc[curr_layer][1] is not None:
            mu_c, V_c, _ = tf_pca(input_arr, cc[next_layer][0], num_as_threshold=ut, conv=conv_n, verbose=verbose)
            n_c, W_c, b_c, mu_c, V_c, W_n = tf_kill_outputs(W_c, b_c, mu_c, V_c, W_n, cc[curr_layer][1],
                                                            num_as_threshold=ut, conv=conv_c,
                                                            conv_to_dense=curr_layer == last_conv, verbose=verbose)
        else:
            mu_c, V_c = None, None

        # add layer
        activation = None if curr_layer == 'output' else 'relu'
        if conv_c and cc[curr_layer][0] is not None:
            W_c_p, b_c_p = tf_transform_conv_weights(mu_p, V_p, W_c, b_c)
            compressed_model.add(Conv2DPCALayer(n_c, 3, mu_p, V_p, kernel_initializer=cift(W_c_p),
                                                bias_initializer=cift(b_c_p), activation='relu', name=curr_layer))
        elif conv_c:
            compressed_model.add(tf.keras.layers.Conv2D(n_c, 3, padding='same', kernel_initializer=cift(W_c),
                                                        bias_initializer=cift(b_c), activation='relu', name=curr_layer))
        elif cc[curr_layer][0] is not None:
            W_c_p, b_c_p = tf_transform_dense_weights(mu_p, V_p, W_c, b_c)
            compressed_model.add(DensePCALayer(n_c, mu_p, V_p, kernel_initializer=cift(W_c_p),
                                               bias_initializer=cift(tf.squeeze(b_c_p)), activation=activation,
                                               name=curr_layer))
        else:
            compressed_model.add(tf.keras.layers.Dense(n_c, kernel_initializer=cift(W_c), bias_initializer=cift(b_c),
                                                       activation=activation, name=curr_layer))

        # add extra layers if necessary
        c_i = all_layers.index(curr_layer)
        n_i = all_layers.index(next_layer) if next_layer is not None else len(all_layers)
        while c_i < n_i - 1:
            c_i += 1
            if 'mp' in all_layers[c_i]:
                compressed_model.add(tf.keras.layers.MaxPooling2D((2, 2), name=all_layers[c_i]))
            elif 'flatten' in all_layers[c_i]:
                compressed_model.add(tf.keras.layers.Flatten(name=all_layers[c_i]))
            elif 'softmax' in all_layers[c_i]:
                compressed_model.add(tf.keras.layers.Softmax(name=all_layers[c_i]))


    # Optimizer
    compressed_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                             optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

    return compressed_model



# For the VGG-16 Average Pooling architecture
CNN_REGULARIZER = tf.keras.regularizers.l2(l=0.00005)
DENSE_REGULARIZER = tf.keras.regularizers.l2(l=0.00005)

def copy_bn_layer(bn_layer, indices=None):
    gamma, beta = bn_layer.weights[0], bn_layer.weights[1]
    moving_mean, moving_var = bn_layer.weights[2], bn_layer.weights[3]
    if indices is not None:
        gamma, beta = tf.gather(gamma, indices), tf.gather(beta, indices)
        moving_mean, moving_var = tf.gather(moving_mean, indices), tf.gather(moving_var, indices)
    return tf.keras.layers.BatchNormalization(name=bn_layer.name, gamma_initializer=cift(gamma),
                                              beta_initializer=cift(beta), moving_mean_initializer=cift(moving_mean),
                                              moving_variance_initializer=cift(moving_var))

def temp_compression(compression_config, input_arr, overall_model, nums_as='dimension', verbose=True):
    assert nums_as in ['dimension', 'threshold'], 'nums_as should be in {dimension, threshold}'
    ut = False if nums_as == 'dimension' else True
    cc = compression_config


    # Get information about the layers in the original network
    all_layers = []
    n_full = {}
    last_conv = ""
    for l in overall_model.layers: # note: the input layer is not included in overall_model.layers
        if 'conv' in l.name or 'fc' in l.name or 'output' in l.name:
            n_full[l.name] = l.output_shape[-1]
        if 'conv' in l.name:
            last_conv = l.name
        all_layers.append(l.name)


    # Verify necessary config constraints
    cc_keys = list(cc.keys())
    cc_values = list(cc.values())

    assert cc_keys == list(n_full.keys()), 'compression config must have keys for all compute layers (conv, fc, output)'
    assert cc_values[0][0] is None, 'currently we are not considering compression of the network input'
    for index in range(len(cc_keys)):
        if cc_keys[index] in ['output']:
            continue
        if cc_values[index][1] is not None:
            assert cc_values[index+1][0] is not None, 'to kill outputs the next layer must perform PCA compression'
    assert cc_values[-1][1] is None, 'can not kill columns in the output layer'


    def advance_input_arr(cl, nl, arr):
        if cl == 'output':
            return None
        arr = overall_model.get_layer(cl)(arr)
        if nl is not None:
            curr_index = all_layers.index(cl)
            next_index = all_layers.index(nl)
            while curr_index < next_index - 1:
                curr_index += 1
                arr = overall_model.get_layer(all_layers[curr_index])(arr)
        return arr


    # Go through the layers and create the (possibly) compressed layers
    compressed_model = tf.keras.Sequential()
    compressed_model.add(tf.keras.layers.InputLayer(input_shape=overall_model.layers[0].input_shape[1:], name='input'))

    mu_c, V_c, W_n = None, None, None

    for index in range(len(cc_keys)):
        prev_layer = cc_keys[index-1] if index-1 >= 0 else None
        curr_layer = cc_keys[index]
        next_layer = cc_keys[index+1] if index+1 <= len(cc_keys)-1 else None

        n_c = n_full[curr_layer]
        W_c = overall_model.get_layer(curr_layer).weights[0] if W_n is None else W_n
        if len(overall_model.get_layer(curr_layer).weights) > 1:
            b_c = overall_model.get_layer(curr_layer).weights[1]
        else:
            b_c = tf.zeros([W_c.shape[-1], ])
        W_n = overall_model.get_layer(next_layer).weights[0] if next_layer is not None else None

        mu_p, V_p = mu_c, V_c

        conv_c = True if 'conv' in curr_layer else False
        conv_n = (True if 'conv' in next_layer else False) if next_layer is not None else None

        # generic iteration
        if cc[curr_layer][0] is not None and cc[prev_layer][1] is None:
            mu_p, V_p, _ = tf_pca(input_arr, cc[curr_layer][0], num_as_threshold=ut, conv=conv_c, verbose=verbose)
        input_arr = advance_input_arr(curr_layer, next_layer, input_arr)
        if cc[curr_layer][1] is not None:
            mu_c, V_c, _ = tf_pca(input_arr, cc[next_layer][0], num_as_threshold=ut, conv=conv_n, verbose=verbose)
            n_c, W_c, b_c, mu_c, V_c, W_n = tf_kill_outputs(W_c, b_c, mu_c, V_c, W_n, cc[curr_layer][1],
                                                            num_as_threshold=ut, conv=conv_c,
                                                            conv_to_dense=curr_layer == last_conv, verbose=verbose)
        else:
            mu_c, V_c = None, None

        # add layer
        # activation = None if curr_layer == 'output' else 'relu'
        if conv_c and cc[curr_layer][0] is not None:
            W_c_p, b_c_p = tf_transform_conv_weights(mu_p, V_p, W_c, b_c)
            compressed_model.add(Conv2DPCALayer(n_c, 3, mu_p, V_p, kernel_initializer=cift(W_c_p),
                                                bias_initializer=cift(b_c_p), activation=None,
                                                kernel_regularizer=CNN_REGULARIZER, bias_regularizer=CNN_REGULARIZER,
                                                name=curr_layer))
        elif conv_c:
            compressed_model.add(tf.keras.layers.Conv2D(n_c, 3, padding='same', kernel_initializer=cift(W_c),
                                                        activation=None, use_bias=False,
                                                        kernel_regularizer=CNN_REGULARIZER, name=curr_layer))
        elif cc[curr_layer][0] is not None:
            W_c_p, b_c_p = tf_transform_dense_weights(mu_p, V_p, W_c, b_c)
            compressed_model.add(DensePCALayer(n_c, mu_p, V_p, kernel_initializer=cift(W_c_p),
                                               bias_initializer=cift(tf.squeeze(b_c_p)), activation=None,
                                               kernel_regularizer=DENSE_REGULARIZER, bias_regularizer=DENSE_REGULARIZER,
                                               name=curr_layer))
        else:
            compressed_model.add(tf.keras.layers.Dense(n_c, kernel_initializer=cift(W_c), bias_initializer=cift(b_c),
                                                       activation=None, kernel_regularizer=DENSE_REGULARIZER,
                                                       bias_regularizer=DENSE_REGULARIZER, name=curr_layer))

        # add extra layers if necessary
        c_i = all_layers.index(curr_layer)
        n_i = all_layers.index(next_layer) if next_layer is not None else len(all_layers)
        while c_i < n_i - 1:
            c_i += 1
            if 'mp' in all_layers[c_i]:
                compressed_model.add(tf.keras.layers.MaxPooling2D((2, 2), name=all_layers[c_i]))
            elif 'ap' in all_layers[c_i]:
                compressed_model.add(tf.keras.layers.GlobalAveragePooling2D(name=all_layers[c_i]))
            elif 'bn' in all_layers[c_i]:
                compressed_model.add(copy_bn_layer(overall_model.get_layer(all_layers[c_i])))
            elif 'relu' in all_layers[c_i]:
                compressed_model.add(tf.keras.layers.ReLU(name=all_layers[c_i]))
            elif 'flatten' in all_layers[c_i]:
                compressed_model.add(tf.keras.layers.Flatten(name=all_layers[c_i]))
            elif 'softmax' in all_layers[c_i]:
                compressed_model.add(tf.keras.layers.Softmax(name=all_layers[c_i]))


    # Optimizer
    # compressed_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
    #                          optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

    return compressed_model