import tensorflow as tf

def conv_block(x, params):

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

    if batchnorm:
        x = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(x)

    for i, (filters, kernel_size) in enumerate(
                zip(filters_list, kernel_sizes)):
        x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                        activation=tf.nn.relu, padding="same",
                        name="conv_{}".format(i+1))(x)

        if max_pool:
             x = tf.keras.layers.MaxPool2D(pool_size=max_pool['size'],
                    strides=max_pool['strides'], name="pool_{}".format(i+1))(x)
        if batchnorm:
             x = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(x)

    # bottleneck layer
    if bottleneck_filters:
        x = tf.keras.layers.Conv2D(filters=bottleneck_filters,
                kernel_size=1, activation=tf.nn.relu, padding="same", name="bottleneck")(x)
        if batchnorm:
             x = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(x)

    return x

def fc_head(inputs, training, params):

    # Get standard hyperparameters
    bn_momentum = params['basic'].get('batchnorm_decay', 0.99)
    
    # Get custom hyperparameters
    layers = params['basic']['fc_head']['layers']
    batchnorm = params['basic']['fc_head'].get('batchnorm', False)

    x = tf.keras.layers.flatten(inputs)

    for i, units in enumerate(layers):
        x = tf.keras.layers.dense(x, units=units, activation=tf.nn.relu,
                name="fc_{}".format(i+1))
        if batchnorm:
            x = tf.keras.layers.batch_normalization(x, momentum=bn_momentum,
                    training=training)

    return x

def conv_head(inputs, training, params):

    # Get standard hyperparameters
    bn_momentum = params.get('batchnorm_decay', 0.99)
    
    # Get custom hyperparameters
    filters_list = [layer['filters'] for layer in
            params['basic']['conv_head']['layers']]
    kernel_sizes = [layer['kernel_size'] for layer in
            params['basic']['conv_head']['layers']]
    final_avg_pool = params['basic']['conv_head'].get('final_avg_pool', True)
    batchnorm = params['basic']['conv_head'].get('batchnorm', False)

    x = inputs

    for i, (filters, kernel_size) in enumerate(zip(filters_list, kernel_sizes)):
        x = tf.keras.layers.conv2d(x, filters=filters, kernel_size=kernel_size,
                activation=tf.nn.relu, padding="same",
                name="conv_{}".format(i+1))
        if batchnorm:
            x = tf.keras.layers.batch_normalization(x, momentum=bn_momentum,
                    training=training)

    # Average over remaining width and length
    if final_avg_pool:
        x = tf.keras.layers.average_pooling2d(x,
                pool_size=x.get_shape().as_list()[1],
                strides=1, name="global_avg_pool")
    
    flat = tf.keras.layers.flatten(x)
    
    return flat
