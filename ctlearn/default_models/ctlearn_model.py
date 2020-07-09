import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow_probability as tfp


def build_bayesian_model(feature_shapes, kl_weight):

    def custom_loss(labels, logits):
        neg_log_likelihood = K.sum(K.binary_crossentropy(labels, logits), axis=-1)
        loss = neg_log_likelihood
        return loss

    bayesian_model = True

    # optimizer hyperparameters
    learning_rate = 0.0005
    epsilon = 1.0e-8

    # bayes conv layers hyperparameters
    filters_list = [32, 32, 64, 128]
    kernel_sizes = [3, 3, 3, 3]

    # max pool layer hyperparameters
    pool_size = 2
    pool_strides = 2

    # weight KL divergence
    kl_divergence_function = (lambda q, p, _: tfp.distributions.kl_divergence(q, p) * kl_weight)

    # build model
    inputs = tf.keras.layers.Input(shape=tf.TensorShape(feature_shapes['image']), name='image')

    x = inputs

    if bayesian_model:

        for i, (filters, kernel_size) in enumerate(
                zip(filters_list, kernel_sizes)):
            x = tfp.layers.Convolution2DFlipout(filters=filters,
                                                kernel_size=kernel_size,
                                                activation=tf.nn.relu,
                                                padding="same",
                                                kernel_divergence_fn=kl_divergence_function)(x)

            x = tf.keras.layers.MaxPooling2D(pool_size=pool_size,
                                             strides=pool_strides)(x)

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
                                       padding="same")(x)

            x = tf.keras.layers.MaxPooling2D(pool_size=pool_size,
                                             strides=pool_strides)(x)

        x = tf.keras.layers.Flatten()(x)

        outputs = tf.keras.layers.Dense(1,
                                        activation='sigmoid',
                                        name='particletype')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                         epsilon=epsilon)

    # compile model
    model.compile(loss=custom_loss,
                  optimizer=optimizer,
                  metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()],
                  experimental_run_tf_function=False)

    return model

