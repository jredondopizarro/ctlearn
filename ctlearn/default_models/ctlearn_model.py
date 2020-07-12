import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow_probability as tfp

# common
BAYESIAN_MODEL = True
FILTER_LIST = [32, 32, 64, 128]
KERNEL_SIZES = [3, 3, 3, 3]
POOL_SIZE = 2
POOL_STRIDES = 2

# keras hyperparameters
LEARNING_RATE = 0.0005
EPSILON = 1.0e-8


def build_bayesian_model_keras(feature_shapes, kl_weight):

    def custom_loss(labels, logits):
        neg_log_likelihood = K.sum(K.binary_crossentropy(labels, logits), axis=-1)
        loss = neg_log_likelihood
        return loss

    # bayes conv layers hyperparameters
    filters_list = FILTER_LIST
    kernel_sizes = KERNEL_SIZES

    # max pool layer hyperparameters
    pool_size = POOL_SIZE
    pool_strides = POOL_STRIDES

    # weight KL divergence
    kl_divergence_function = (lambda q, p, _: tfp.distributions.kl_divergence(q, p) * kl_weight)

    # build model
    inputs = tf.keras.layers.Input(shape=tf.TensorShape(feature_shapes['image']), name='image')

    x = inputs

    if BAYESIAN_MODEL:

        for i, (filters, kernel_size) in enumerate(
                zip(filters_list, kernel_sizes)):
            x = tfp.layers.Convolution2DFlipout(filters=filters,
                                                kernel_size=kernel_size,
                                                activation=tf.nn.relu,
                                                padding="same",
                                                kernel_divergence_fn=kl_divergence_function,
                                                name="bayes_conv_{}".format(i + 1))(x)

            x = tf.keras.layers.MaxPooling2D(pool_size=pool_size,
                                             strides=pool_strides,
                                             name="pool_{}".format(i+1))(x)

        x = tf.keras.layers.Flatten()(x)

        outputs = tfp.layers.DenseFlipout(1,
                                          kernel_divergence_fn=kl_divergence_function,
                                          activation='sigmoid',
                                          name='particletype')(x)

    else:

        for i, (filters, kernel_size) in enumerate(
                zip(filters_list, kernel_sizes)):
            x = tf.keras.layers.Conv2D(filters=filters,
                                       kernel_size=kernel_size,
                                       activation=tf.nn.relu,
                                       padding="same",
                                       name="conv_{}".format(i + 1))(x)

            x = tf.keras.layers.MaxPooling2D(pool_size=pool_size,
                                             strides=pool_strides,
                                             name="pool_{}".format(i + 1))(x)

        x = tf.keras.layers.Flatten()(x)

        outputs = tf.keras.layers.Dense(1,
                                        activation='sigmoid',
                                        name='particletype')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE,
                                         epsilon=EPSILON)

    # compile model
    model.compile(loss=custom_loss,
                  optimizer=optimizer,
                  metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()],
                  experimental_run_tf_function=False)

    return model


def build_bayesian_model_tf(feature_shapes):


    # bayes conv layers hyperparameters
    filters_list = FILTER_LIST
    kernel_sizes = KERNEL_SIZES

    # max pool layer hyperparameters
    pool_size = POOL_SIZE
    pool_strides = POOL_STRIDES

    # build model
    inputs = tf.keras.layers.Input(shape=tf.TensorShape(feature_shapes['image']), name='image')

    x = inputs

    if BAYESIAN_MODEL:

        for i, (filters, kernel_size) in enumerate(
                zip(filters_list, kernel_sizes)):
            x = tfp.layers.Convolution2DFlipout(filters=filters,
                                                kernel_size=kernel_size,
                                                activation=tf.nn.relu,
                                                padding="same",
                                                name="bayes_conv_{}".format(i + 1))(x)

            x = tf.keras.layers.MaxPooling2D(pool_size=pool_size,
                                             strides=pool_strides,
                                             name="pool_{}".format(i+1))(x)

        x = tf.keras.layers.Flatten()(x)

        outputs = tfp.layers.DenseFlipout(1,
                                          name='particletype')(x)

    else:

        for i, (filters, kernel_size) in enumerate(
                zip(filters_list, kernel_sizes)):
            x = tf.keras.layers.Conv2D(filters=filters,
                                       kernel_size=kernel_size,
                                       activation=tf.nn.relu,
                                       padding="same",
                                       name="conv_{}".format(i + 1))(x)

            x = tf.keras.layers.MaxPooling2D(pool_size=pool_size,
                                             strides=pool_strides,
                                             name="pool_{}".format(i + 1))(x)

        x = tf.keras.layers.Flatten()(x)

        outputs = tf.keras.layers.Dense(1,
                                        name='particletype')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model