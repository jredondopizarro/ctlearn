import importlib
import sys

import tensorflow as tf

import ctlearn
import ctlearn.default_models.cnn_blocks as cnn_blocks
import ctlearn.default_models.ct_heads as ct_heads

def build_model(feature_shapes, params):

    cnn_block_fn = getattr(cnn_blocks,
                           params['model_settings']['cnn_block'])
    input, cnn_block = cnn_block_fn(feature_shapes, params)
    
    ct_head_fn = getattr(ct_heads,
                         params['model_settings']['ct_head'])
    model = ct_head_fn(input, cnn_block, params)
    
    learning_rate = params['base_learning_rate']
    # Select optimizer with appropriate arguments
    # Dict of optimizer_name: (optimizer_fn, optimizer_args)
    optimizers = {
        'Adadelta': (tf.keras.optimizers.Adadelta,
                     dict(learning_rate=learning_rate)),
        'Adam': (tf.keras.optimizers.Adam,
                 dict(learning_rate=learning_rate,
                 epsilon=params['adam_epsilon'])),
        'RMSProp': (tf.keras.optimizers.RMSprop,
                    dict(learning_rate=learning_rate)),
        'SGD': (tf.keras.optimizers.SGD,
                dict(learning_rate=learning_rate))
        }

    optimizer_fn, optimizer_args = optimizers[params['optimizer']]
    optimizer = optimizer_fn(**optimizer_args)

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()])
    # return compiled model
    return model


def build_bayesian_model(feature_shapes, params, num_training_examples):
    cnn_block_fn = getattr(cnn_blocks,
                           params['model_settings']['cnn_block'])
    input, cnn_block = cnn_block_fn(feature_shapes, params, num_training_examples)

    ct_head_fn = getattr(ct_heads,
                         params['model_settings']['ct_head'])
    model = ct_head_fn(input, cnn_block, params, num_training_examples)




    learning_rate = params['base_learning_rate']
    # Select optimizer with appropriate arguments
    # Dict of optimizer_name: (optimizer_fn, optimizer_args)
    optimizers = {
        'Adadelta': (tf.keras.optimizers.Adadelta,
                     dict(learning_rate=learning_rate)),
        'Adam': (tf.keras.optimizers.Adam,
                 dict(learning_rate=learning_rate,
                      epsilon=params['adam_epsilon'])),
        'RMSProp': (tf.keras.optimizers.RMSprop,
                    dict(learning_rate=learning_rate)),
        'SGD': (tf.keras.optimizers.SGD,
                dict(learning_rate=learning_rate))
    }

    optimizer_fn, optimizer_args = optimizers[params['optimizer']]
    optimizer = optimizer_fn(**optimizer_args)



    # def custom_loss_function(labels, logits):
    #     #neg_log_likelihood = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    #     neg_log_likelihood = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(labels, logits)
    #     kl = sum(model.losses)/num_training_examples
    #     loss = neg_log_likelihood + kl
    #     return loss




    # logits = model(features)
    # neg_log_likelihood = tf.nn.softmax_cross_entropy_with_logits(
    #     labels=labels, logits=logits)
    # kl = sum(model.losses)
    # loss = neg_log_likelihood + kl
    # Compile the model

    model.compile(loss='binary_crossentropy', optimizer=optimizer,
                  metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()])
    # return compiled model
    return model

