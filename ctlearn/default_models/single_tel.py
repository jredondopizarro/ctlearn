import importlib
import sys

import tensorflow as tf
from ctlearn.ct_heads import *

def single_tel_model(features, model_params, example_description, training):
    
    # Reshape inputs into proper dimensions
    #for (name, f), d in zip(features.items(), example_description):
    #    if name == 'image':
    telescope_data = features['image']
    #telescope_data = tf.reshape(features, [-1, *d['shape']])
    num_classes = len(model_params['label_names']['particletype'])
    
    # Load neural network model
    sys.path.append(model_params['model_directory'])
    network_module = importlib.import_module(model_params['single_tel']['network']['module'])
    network = getattr(network_module,
                      model_params['single_tel']['network']['function'])

    if model_params['single_tel']['mode'] == 'hard':
        inputShape = telescope_data.get_shape()
        inputs = tf.keras.layers.Input(shape=inputShape[1:])

        x = network(inputs, params=model_params, training=training)

        if model_params['single_tel']['pretrained_weights']:    tf.contrib.framework.init_from_checkpoint(model_params['single_tel']['pretrained_weights'],{'Network/':'Network/'})
        
        x = tf.keras.layers.Flatten()(x)
        #model.add(tf.keras.layers.Flatten())
    
    outputs = []
    multihead_array = []
    for task in model_params['label_names']:

        '''
        if model_params['single_tel']['mode'] == 'soft':
            with tf.variable_scope("Network"):
                output = network(telescope_data, params=model_params, training=training)

            if model_params['single_tel']['pretrained_weights']:    tf.contrib.framework.init_from_checkpoint(model_params['single_tel']['pretrained_weights'],{'Network/':'Network/'})
            
            output_flattened = tf.keras.layers.flatten(output)
        ''' 
        if task == 'particletype':
            if num_classes == 2:
                logit_units = 1
                multihead_array.append(tf.estimator.BinaryClassHead(name=task))
            else: 
                logit_units = num_classes
                multihead_array.append(tf.estimator.MultiClassHead(name=task, n_classes=logit_units))
        elif task in ['energy', 'showermaximum']:
            logit_units = 1
            multihead_array.append(tf.estimator.RegressionHead(name=task,label_dimension=logit_units))
        elif task in ['direction', 'impact']:
            logit_units = 2
            multihead_array.append(tf.estimator.RegressionHead(name=task,label_dimension=logit_units))

        y = tf.keras.layers.Dense(units=logit_units)(x)
        outputs.append(y)

    tf.keras.backend.set_learning_phase(True)
    if len(model_params['label_names']) == 1:
        model = tf.keras.Model(inputs=inputs, outputs=outputs[0])
    else:
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

    predictions = model(features)

    logits = {}
    for ind, task in enumerate(model_params['label_names']):
        logits[task] = predictions[ind]
    print(model.summary())
    
    print("model.updates: ", model.updates)
    print("model.weigths: ", model.weights)
    #print("model.sample_weights: ", model.sample_weights)
    #print("model.state_updates: ", model.state_updates)
    return model, multihead_array, logits
