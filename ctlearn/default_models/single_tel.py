import importlib
import sys

import tensorflow as tf

def single_tel_model(model_params, example_description):
    
    # Reshape inputs into proper dimensions
    #for (name, f), d in zip(features.items(), example_description):
    #    if name == 'image':
    #telescope_data = features['image']
    #telescope_data = tf.reshape(features, [-1, *d['shape']])
    #num_classes = model_params['num_classes']devices=["/gpu:0","/gpu:1"]
    #strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    '''
    inputShape = (114,114,1)
    inputs = tf.keras.layers.Input(shape=inputShape, name='image')
    
    # Load neural network model
    sys.path.append(model_params['model_directory'])
    cnn_module = importlib.import_module(model_params['single_tel']['cnn_block']['module'])
    cnn_block = getattr(cnn_module, model_params['single_tel']['cnn_block']['function'])

    if model_params['single_tel']['mode'] == 'hard': 
        output = cnn_block(inputs, params=model_params)
        output_flattened = tf.keras.layers.Flatten()(output)
           
    outputs = []
    losses = {}
    lossWeights = {}
    metrics = {}
    task_dict = model_params['task']
    for task in task_dict:

        if model_params['single_tel']['mode'] == 'soft':
            output = cnn_block(inputs, params=model_params, training=training)
            output_flattened = tf.keras.layers.Flatten()(output)

        for i, logit_unit in enumerate(task_dict[task]['fc_head']):
            if logit_unit == task_dict[task]['fc_head'][-1]:
                layer_name = task
            else:
                layer_name = "{}_fc_{}".format(task,i+1)
            output_fc = tf.keras.layers.Dense(units=logit_unit, name=layer_name)(output_flattened)
        outputs.append(output_fc)

        losses[task] = task_dict[task]['loss']
        lossWeights[task] = task_dict[task]['weight']
        metrics[task] = task_dict[task]['metric']

    if len(outputs) == 1:
        model = tf.keras.Model(inputs=inputs, outputs=outputs[0])
    else:
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

    op = tf.keras.optimizers.Adam(learning_rate=5.0e-05, beta_1=1.0e-08)
    model.compile(optimizer=op, loss=losses, loss_weights=lossWeights, metrics=metrics)

    '''
    with tf.device('/device:gpu:0'):
        inputShape = (114,114,1)
        inputs = tf.keras.layers.Input(shape=inputShape, name='image')

        # Load neural network model
        sys.path.append(model_params['model_directory'])
        cnn_module = importlib.import_module(model_params['single_tel']['cnn_block']['module'])
        cnn_block = getattr(cnn_module, model_params['single_tel']['cnn_block']['function'])

        if model_params['single_tel']['mode'] == 'hard':
            output = cnn_block(inputs, params=model_params)
            output_flattened = tf.keras.layers.Flatten()(output)

        outputs, losses, lossWeights, metrics = [],{},{},{}
        activation = None
        task_dict = model_params['task']
        for task in task_dict:

            if model_params['single_tel']['mode'] == 'soft':
                output = cnn_block(inputs, params=model_params, training=training)
                output_flattened = tf.keras.layers.Flatten()(output)

            for i, logit_unit in enumerate(task_dict[task]['fc_head']):
                if i == len(task_dict[task]['fc_head'])-1:
                    layer_name = task
                    if task == 'particletype':
                        activation = tf.nn.softmax
                else:
                    layer_name = "{}_fc_{}".format(task,i+1)
                output_fc = tf.keras.layers.Dense(units=logit_unit, name=layer_name, activation=activation)(output_flattened)
            outputs.append(output_fc)

            losses[task] = task_dict[task]['loss']
            lossWeights[task] = task_dict[task]['weight']
            metrics[task] = task_dict[task]['metric']

        #if len(outputs) == 1:
        #    model = tf.keras.Model(inputs=inputs, outputs=outputs[0])
        #else:
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        op = tf.keras.optimizers.Adam(learning_rate=5.0e-05, beta_1=1.0e-08)
        model.compile(optimizer=op, loss=losses, loss_weights=lossWeights, metrics=metrics)

    return model
