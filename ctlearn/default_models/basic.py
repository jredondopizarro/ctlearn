import tensorflow as tf
import tensorflow_probability as tfp

def conv_block(input, params):

    # Get standard hyperparameters
    bn_momentum = params.get('batchnorm_decay', 0.99)
    # Get custom hyperparameters
    filters_list = [layer['filters'] for layer in
            params['basic']['conv_block']['layers']]
    kernel_sizes = [layer['kernel_size'] for layer in
            params['basic']['conv_block']['layers']]
    max_pool = params['basic']['conv_block']['max_pool']
    bottleneck_filters = params['basic']['conv_block']['bottleneck']
    batchnorm = params['basic']['conv_block'].get('batchnorm', False)

    x = input
    if batchnorm:
        x = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(x)

    for i, (filters, kernel_size) in enumerate(
            zip(filters_list, kernel_sizes)):
        x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                activation=tf.nn.relu, padding="same", name="conv_{}".format(i+1))(x)
        if max_pool:
            x = tf.keras.layers.MaxPool2D(pool_size=max_pool['size'],
                    strides=max_pool['strides'], name="pool_{}".format(i+1))(x)
        if batchnorm:
            x = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(x)

    # bottleneck layer
    if bottleneck_filters:
        x = tf.keras.layers.Conv2D(filters=bottleneck_filters,
                kernel_size=1, activation=tf.nn.relu, padding="same",
                name="bottleneck")(x)
        if batchnorm:
            x = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(x)

    return x

def fc_head(inputs, tasks_dict, expected_logits_dimension):

    layers = tasks_dict['fc_head']['layers']

    if layers[-1] != expected_logits_dimension:
        print("Warning:fc_head: Last logit unit '{}' of the fc_head array differs from the expected_logits_dimension '{}'. The expected logits dimension '{}' will be appended.".format(layers[-1], expected_logits_dimension))
        layers.append(expected_logits_dimension)

    x = inputs
    activation=tf.nn.relu
    for i, units in enumerate(layers):
        if i == len(layers)-1:
            activation=None
        x = tf.keras.layers.Dense(units=units, activation=activation,
                name="fc_{}_{}".format(tasks_dict['name'], i+1))(x)
    return x

def conv_head(input, params):

    # Get standard hyperparameters
    bn_momentum = params.get('batchnorm_decay', 0.99)
    
    # Get custom hyperparameters
    filters_list = [layer['filters'] for layer in
            params['basic']['conv_head']['layers']]
    kernel_sizes = [layer['kernel_size'] for layer in
            params['basic']['conv_head']['layers']]
    final_avg_pool = params['basic']['conv_head'].get('final_avg_pool', True)
    batchnorm = params['basic']['conv_head'].get('batchnorm', False)

    x = input
    for i, (filters, kernel_size) in enumerate(zip(filters_list, kernel_sizes)):
        x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                activation=tf.nn.relu, padding="same",
                name="conv_{}".format(i+1))(x)
        if batchnorm:
            x = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(x)

    # Average over remaining width and length
    if final_avg_pool:
        x = tf.keras.layers.AveragePooling2D(
                pool_size=x.get_shape().as_list()[1],
                strides=1, name="global_avg_pool")(x)

    flat = tf.keras.layers.Flatten()(x)

    return flat


def bayesian_conv_block(input, params, num_training_examples):
        # Get standard hyperparameters
        bn_momentum = params.get('batchnorm_decay', 0.99)
        # Get custom hyperparameters
        filters_list = [layer['filters'] for layer in
                        params['bayesian']['bayesian_conv_block']['layers']]
        kernel_sizes = [layer['kernel_size'] for layer in
                        params['bayesian']['bayesian_conv_block']['layers']]
        max_pool = params['bayesian']['bayesian_conv_block']['max_pool']
        bottleneck_filters = params['bayesian']['bayesian_conv_block']['bottleneck']
        batchnorm = params['bayesian']['bayesian_conv_block'].get('batchnorm', False)

        kl_divergence_function = (lambda q, p, _: tfp.distributions.kl_divergence(q, p) /
                                  tf.cast(num_training_examples, dtype=tf.float32))

        x = input
        if batchnorm:
            x = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(x)

        for i, (filters, kernel_size) in enumerate(
                zip(filters_list, kernel_sizes)):
            x = tfp.layers.Convolution2DFlipout(filters=filters,
                                                kernel_size=kernel_size,
                                                activation=tf.nn.relu,
                                                padding="same",
                                                kernel_divergence_fn = kl_divergence_function,
                                                name="bayes_conv_{}".format(i + 1))(x)
            if max_pool:
                x = tf.keras.layers.MaxPool2D(pool_size=max_pool['size'],
                                              strides=max_pool['strides'], name="pool_{}".format(i + 1))(x)
            if batchnorm:
                x = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(x)

        # bottleneck layer
        if bottleneck_filters:
            x = tfp.layers.Convolution2DFlipout(filters=bottleneck_filters,
                                                kernel_size=1,
                                                activation=tf.nn.relu,
                                                padding="same",
                                                kernel_divergence_fn=kl_divergence_function,
                                                name="bayes_bottleneck")(x)
            if batchnorm:
                x = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(x)

        return x

