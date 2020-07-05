import importlib
import sys

import tensorflow as tf
import tensorflow_probability as tfp

from ctlearn.default_models.basic import *

# Only for testing
def single_tel_head(input, cnn_output, model_params):
    output_flattened = tf.keras.layers.Flatten()(cnn_output)
    logits = tf.keras.layers.Dense(1, activation='sigmoid', name="particletype")(output_flattened)
    
    tf.keras.backend.set_learning_phase(True)
    model = tf.keras.Model(inputs=input, outputs=logits)
    
    return model

def custom_head(input, cnn_output, model_params):

    output_flattened = tf.keras.layers.Flatten()(cnn_output)
    output_globalpooled = tf.reduce_mean(cnn_output, axis=[1,2])
    
    model_outputs = []
    tasks_dict = model_params['custom_head']
    for task in tasks_dict:
        tasks_dict[task].update({'name': task})
        if task == 'particletype':
            expected_logits_dimension = len(tasks_dict[task]['class_names'])
            logit = fc_head(output_flattened, tasks_dict[task], expected_logits_dimension)
            gammaness = tf.nn.softmax(logit)
            model_outputs.append(tf.cast(tf.argmax(gammaness, axis=1),
                                         tf.int32, name="predicted_classes"))
            model_outputs.append(gammaness)
        else:
            output = output_globalpooled if task == 'energy' else output_flattened
            expected_logits_dimension = 2 if task in ['direction', 'impact'] else 1
            model_outputs.append(fc_head(output, tasks_dict[task], expected_logits_dimension))

    tf.keras.backend.set_learning_phase(True)
    model = tf.keras.Model(inputs=input, outputs=model_outputs)

    return model


def rnn_head(inputs, telescope_outputs, model_params):

    num_tels_triggered = tf.to_int32(tf.reduce_sum(inputs[1],1))
    #combine image embeddings (batch_size, num_tel, num_units_embedding)
    embeddings = tf.stack(telescope_outputs,axis=1)

    #implement attention mechanism with range num_tel (covering all timesteps)
    #define LSTM cell size
    LSTM_SIZE = 2048
    
    #TODO: Update to TF2
    rnn_cell = tf.keras.layers.LSTMCell(LSTM_SIZE)(embeddings)
    outputs, _ = tf.nn.dynamic_rnn(
                        rnn_cell,
                        embeddings,
                        dtype=tf.float32,
                        swap_memory=True,
                        sequence_length=num_tels_triggered)

    # (batch_size, max_num_tel * LSTM_SIZE)
    outputs = tf.keras.layers.Flatten()(outputs)
    output_dropout = tf.keras.layers.Dropout(dropout_rate, name="rnn_output_dropout")(outputs)
           
    fc1 = tf.keras.layers.Dense(1024, kernel_regularizer=tf.keras.regularizers.l2(0.004), name="fc1")(output_dropout)
    dropout_1 = tf.keras.layers.Dropout(dropout_rate)(fc1)
           
    fc2 = tf.keras.layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.004), name="fc2")(dropout_1)
    dropout_2 = tf.keras.layers.dropout(dropout_rate)(fc2)

    model = tf.keras.Model(inputs=input, outputs=dropout_2)
    return model
    
def gammaPhysNet_head(inputs, model_params):
    raise NotImplementedError

def attention_head(inputs, model_params):
    raise NotImplementedError

def bayesian_head(inputs, cnn_output, model_params, num_training_examples):
    kl_divergence_function = (lambda q, p, _: tfp.kl_divergence(q, p) /
                              tf.cast(num_training_examples, dtype=tf.float32))

    output_flattened = tf.keras.layers.Flatten()(cnn_output)
    logits = tfp.layers.DenseFlipout(1,
                                     kernel_divergence_fn=kl_divergence_function,
                                     activation='sigmoid',
                                     name="particletype")(output_flattened)

    tf.keras.backend.set_learning_phase(True)
    model = tf.keras.Model(inputs=inputs, outputs=logits)

    return model

    
