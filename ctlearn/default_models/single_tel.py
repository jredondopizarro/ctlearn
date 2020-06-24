import importlib
import sys

import tensorflow as tf

def single_tel_model(inputs, model_params, training):

    # Load neural network model
    sys.path.append(model_params['model_directory'])
    network_module = importlib.import_module(model_params['single_tel']['network']['module'])
    network = getattr(network_module,
                      model_params['single_tel']['network']['function'])

    #inputs = tf.keras.layers.Input(shape=model_params['model_input'])
    output = network(inputs, params=model_params, training=training)

    #if model_params['single_tel']['pretrained_weights']:    tf.contrib.framework.init_from_checkpoint(model_params['single_tel']['pretrained_weights'],{'Network/':'Network/'})

    return output
