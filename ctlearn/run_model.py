import argparse
import importlib
import logging
import math
from random import randint
import os
from pprint import pformat
import sys

import numpy as np
import yaml

import tensorflow as tf
from tensorflow.python import debug as tf_debug

from dl1_data_handler.reader import DL1DataReader
from ctlearn.data_loading import *
from ctlearn.utils import *

# Disable Tensorflow info and warning messages (not error messages)
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#tf.logging.set_verbosity(tf.logging.WARN)

def run_model(config, mode="train", debug=False, log_to_file=False, multiple_runs=1):

    #print(tf.config.experimental.list_logical_devices('GPU'))

    #gpus = tf.config.experimental.list_physical_devices('GPU')
    #tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    #tf.config.experimental.set_memory_growth(gpus[0], True)

    #print(gpus)
    # Load options relating to logging and checkpointing
    root_model_dir = model_dir = config['Logging']['model_directory']
    # Create model directory if it doesn't exist already
    if not os.path.exists(root_model_dir):
        if mode == 'predict':
            raise ValueError("Invalid model directory '{}'. "
            "Must be a path to an existing directory in the predict mode.".format(config['Logging']['model_directory']))
        os.makedirs(root_model_dir)

    random_seed = None
    if multiple_runs != 1:
        random_seed = config['Data']['seed']
        if mode=='train':
            model_dir += "/experiment_{}".format(random_seed)
            os.makedirs(model_dir)

    # Set up logging, saving the config and optionally logging to a file
    logger = setup_logging(config, model_dir, debug, log_to_file)

    # Log the loaded configuration
    logger.debug(pformat(config))

    logger.info("Logging has been correctly set up")

    # Create params dictionary that will be passed to the model
    params = {}
    '''
    # Load options to specify the model
    try:
        model_directory = config['Model']['model_directory']
        if model_directory is None:
            raise KeyError
    except KeyError:
        model_directory = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "default_models/"))
    sys.path.append(model_directory)
    model_module = importlib.import_module(config['Model']['model']['module'])
    ctlearn_model = getattr(model_module, config['Model']['model']['function'])

    # Write the model parameters in the params dictionary
    params['model'] = {**config['Model'], **config.get('Model Parameters', {})}
    '''

    labels = config['Model']['task']

    # Set up the DL1DataReader
    config['Data'] = setup_DL1DataReader(config, mode)

    # Create data reader
    logger.info("Loading data:")
    logger.info("For a large dataset, this may take a while...")
    reader = DL1DataReader(**config['Data'])
    params['example_description'] = reader.example_description

    # Set up the TensorFlow dataset
    if 'Input' not in config:
        config['Input'] = {}
    config['Input'] = setup_TFdataset_format(config, reader.example_description, labels)
    batch_size = config['Input'].get('batch_size', 1)
    logger.info("Batch size: {}".format(batch_size))

    # Load either training or prediction options
    # and log information about the data set
    indices = list(range(len(reader)))
    
    if mode in ['train', 'load_only']:

        # Write the training configuration in the params dict
        params['training'] = config['Training']

        validation_split = config['Training']['validation_split']
        if not 0.0 < validation_split < 1.0:
            raise ValueError("Invalid validation split: {}. "
                             "Must be between 0.0 and 1.0".format(
                                 validation_split))
        num_training_examples = math.floor((1 - validation_split) * len(reader))
        training_indices = indices[:num_training_examples]
        validation_indices = indices[num_training_examples:]
        logger.info("Number of training steps per epoch: {}".format(
            int(num_training_examples / batch_size)))
        logger.info("Number of training steps between validations: {}".format(
            config['Training']['num_training_steps_per_validation']))
            
        # Write the evaluation configuration in the params dict
        params['evaluation'] = config['Evaluation']

    if mode == 'predict':
        # Write the prediction configuration in the params dict
        params['prediction'] = config['Prediction']

    if mode == 'load_only':
        log_examples(reader, indices, labels, 'total dataset')
        log_examples(reader, training_indices, labels, 'training')
        log_examples(reader, validation_indices, labels, 'validation')
        # If only loading data, can end now that dataset logging is complete
        return

    if mode == 'train' and config['Training']['apply_class_weights']:
        num_class_examples = log_examples(reader, training_indices,
                                          labels, 'training')
        class_weights = compute_class_weights(labels, num_class_examples)
        params['training']['class_weights'] = class_weights

    # Load options for TensorFlow
    run_tfdbg = config.get('TensorFlow', {}).get('run_TFDBG', False)


    training_data = input_fn(reader, training_indices, mode='train', **config['Input'])
    validation_data = input_fn(reader, validation_indices, mode='eval', **config['Input'])

    model_file = config['Model'].get('model_file', None)
    if model_file:
        model = tf.saved_model.load(model_file)
    else:
        # Load options to specify the model
        try:
            model_directory = config['Model']['model_directory']
            if model_directory is None:
                raise KeyError
        except KeyError:
            model_directory = os.path.abspath(os.path.join(
                os.path.dirname(__file__), "default_models/"))
        sys.path.append(model_directory)
        if 'ctlearn_model.h5' in np.array([os.listdir(model_dir)]):
            model = tf.saved_model.load(model_dir+'/ctlearn_model.h5')
        else:
            model_module = importlib.import_module(config['Model']['model']['module'])
            ctlearn_model = getattr(model_module, config['Model']['model']['function'])
            # Write the model parameters in the params dictionary
            params['model'] = {**config['Model'], **config.get('Model Parameters', {})}
            model = ctlearn_model(params['model'], params['example_description'])

    '''
    # Scale the learning rate so batches with fewer triggered
    # telescopes don't have smaller gradients
    # Only apply learning rate scaling for array-level models
    optimizer = {}
    if training:
        training_params = params['training']
        if (training_params['scale_learning_rate'] and params['model']['function'] in ['cnn_rnn_model', 'variable_input_model']):
            trigger_rate = tf.reduce_mean(tf.cast(
                            features['telescope_triggers'], tf.float32),
                            name="trigger_rate")
            trigger_rate = tf.maximum(trigger_rate, 0.1) # Avoid division by 0
            scaling_factor = tf.reciprocal(trigger_rate, name="scaling_factor")
            learning_rate = tf.multiply(scaling_factor,
                                        training_params['base_learning_rate'],
                                        name="learning_rate")
        else:
            learning_rate = training_params['base_learning_rate']
        
        # Select optimizer with appropriate arguments
        # Dict of optimizer_name: (optimizer_fn, optimizer_args)
        optimizers = {
            'Adadelta': (tf.keras.optimizers.Adadelta,
                         dict(learning_rate=learning_rate)),
            'Adam': (tf.keras.optimizers.Adam,
                     dict(learning_rate=learning_rate,
                     epsilon=training_params['adam_epsilon'])),
            'RMSProp': (tf.keras.optimizers.RMSprop,
                        dict(learning_rate=learning_rate)),
            'SGD': (tf.keras.optimizers.SGD,
                    dict(learning_rate=learning_rate))
            }

        optimizer_fn, optimizer_args = optimizers[training_params['optimizer']]
        optimizer = optimizer_fn(**optimizer_args)
    '''
    
    #model.fit(training_data, epochs=1)
    model.fit(training_data, epochs=100, validation_data=validation_data)
    tf.saved_model.save(model, model_dir+'/ctlearn_model.h5')

    evaluations = model.predict(validation_data)

    print(evaluations)
    print(type(evaluations))
    print(evaluations.shape)
    '''
    # Define model function with model, mode (train/predict),
    # metrics, optimizer, learning rate, etc.
    # to pass into TF Estimator
    def model_fn(features, labels, mode, params):
    
        training = True if mode == tf.estimator.ModeKeys.TRAIN else False

        mod, multihead_array, logits = model(features, params['model'], params['example_description'], training)

        # Combine the several heads in the multi_head class
        multi_head = tf.estimator.MultiHead(multihead_array)
        
        # Scale the learning rate so batches with fewer triggered
        # telescopes don't have smaller gradients
        # Only apply learning rate scaling for array-level models
        optimizer = {}
        if training:
            training_params = params['training']
            if (training_params['scale_learning_rate'] and params['model']['function'] in ['cnn_rnn_model', 'variable_input_model']):
                trigger_rate = tf.reduce_mean(tf.cast(
                                features['telescope_triggers'], tf.float32),
                                name="trigger_rate")
                trigger_rate = tf.maximum(trigger_rate, 0.1) # Avoid division by 0
                scaling_factor = tf.reciprocal(trigger_rate, name="scaling_factor")
                learning_rate = tf.multiply(scaling_factor,
                                            training_params['base_learning_rate'],
                                            name="learning_rate")
            else:
                learning_rate = training_params['base_learning_rate']
        
            # Select optimizer with appropriate arguments
            # Dict of optimizer_name: (optimizer_fn, optimizer_args)
            optimizers = {
                'Adadelta': (tf.keras.optimizers.Adadelta,
                             dict(learning_rate=learning_rate)),
                'Adam': (tf.keras.optimizers.Adam,
                         dict(learning_rate=learning_rate,
                         epsilon=training_params['adam_epsilon'])),
                'RMSProp': (tf.keras.optimizers.RMSprop,
                            dict(learning_rate=learning_rate)),
                'SGD': (tf.keras.optimizers.SGD,
                        dict(learning_rate=learning_rate))
                }

            optimizer_fn, optimizer_args = optimizers[training_params['optimizer']]
            optimizer = optimizer_fn(**optimizer_args)

        #return multi_head.create_estimator_spec(features=features, mode=mode, logits=logits, labels=labels, optimizer=optimizer)

        return multi_head.create_estimator_spec(features=features, mode=mode, logits=logits, labels=labels, trainable_variables=mod.weights, update_ops=mod.updates, optimizer=optimizer)
    
    estimator = tf.estimator.Estimator(
        model_fn,
        model_dir=model_dir,
        params=params)

    hooks = None
    # Activate Tensorflow debugger if appropriate option set
    if run_tfdbg:
        if not isinstance(hooks, list):
            hooks = []
        hooks.append(tf_debug.LocalCLIDebugHook())

    if mode == 'train':

        # Train and evaluate the model
        logger.info("Training and evaluating...")
        num_validations = config['Training']['num_validations']
        steps = config['Training']['num_training_steps_per_validation']
        train_forever = False if num_validations != 0 else True
        num_validations_remaining = num_validations
        while train_forever or num_validations_remaining:
            print(steps)
            epoch = num_validations-num_validations_remaining+1
            estimator.train(
                lambda: input_fn(reader, training_indices, mode='train', **config['Input']),
                steps=steps, hooks=hooks)
            if params['evaluation']['default_tensorflow']:
                logger.info("Evaluate with the default TensorFlow evaluation...")
                estimator.evaluate(
                    lambda: input_fn(reader, validation_indices, mode='eval', **config['Input']),
                    hooks=hooks, name='validation')

            if params['evaluation']['custom_ctlearn']['final_evaluation'] and num_validations_remaining == 1:
                logger.info("Evaluate with the custom CTLearn evaluation...")
                evaluations = estimator.predict(
                    lambda: input_fn(reader, validation_indices, mode='predict', **config['Input']),
                    hooks=hooks)

                evaluation = list(evaluations)
                
                # Open the the h5 to dump the final evaluation information in the selected format
                if params['evaluation']['custom_ctlearn']['final_evaluation_file_name']:
                    final_eval_file = os.path.abspath(os.path.join(os.path.dirname(__file__), model_dir+"/{}.h5".format(params['evaluation']['custom_ctlearn']['final_evaluation_file_name'])))
                else:
                    final_eval_file = os.path.abspath(os.path.join(os.path.dirname(__file__), model_dir+"/experiment.h5"))
                write_output(h5file=final_eval_file, reader=reader, indices=validation_indices, example_description=params['example_description'], predictions=evaluation)
            
            if not train_forever:
                num_validations_remaining -= 1

    elif mode == 'predict':

        # Generate predictions and add to output
        logger.info("Predicting...")
        
        # Open the the h5 to dump the final evaluation information in the selected format
        if params['prediction']['prediction_file_name']:
            predict_file = os.path.abspath(os.path.join(os.path.dirname(__file__), root_model_dir+"/{}.h5".format(params['prediction']['prediction_file_name'])))
        else:
            predict_file = os.path.abspath(os.path.join(os.path.dirname(__file__), root_model_dir+"/ctlearn_prediction.h5"))
        predictions = estimator.predict(
            lambda: input_fn(reader, indices, mode='predict', **config['Input']),
            hooks=hooks)
        prediction = list(predictions)
        write_output(h5file=predict_file, reader=reader, indices=indices, example_description=params['example_description'], predictions=prediction, mode='predict', seed=random_seed)
    '''
    # clear the handlers, shutdown the logging and delete the logger
    logger.handlers.clear()
    logging.shutdown()
    del logger
    return
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=("Train/Predict with a CTLearn model."))
    parser.add_argument(
        '--mode',
        default="train",
        help="Mode to run in (train/predict/trainandpredict/load_only)")
    parser.add_argument(
        'config_file',
        help="path to YAML configuration file with training options")
    parser.add_argument(
        '--debug',
        action='store_true',
        help="print debug/logger messages")
    parser.add_argument(
        '--log_to_file',
        action='store_true',
        help="log to a file in model directory instead of terminal")
    parser.add_argument(
        '--multiple_runs',
        default=1,
        type=int,
        help="run the same model multiple times with the same config file")

    args = parser.parse_args()

    random_seeds = []
    for run in np.arange(args.multiple_runs):
        with open(args.config_file, 'r') as config_file:
            config = yaml.safe_load(config_file)
        if args.multiple_runs != 1:
            # Create and overwrite the random seed in the config file
            while True:
                random_seed = randint(1000,9999)
                if random_seed not in random_seeds:
                    random_seeds.append(random_seed)
                    break
            
            config['Data']['seed'] = random_seed
            print("CTLearn run {} with random seed '{}':".format(run+1,config['Data']['seed']))
        config['Data']['shuffle'] = False if args.mode == 'predict' else True

        if args.mode != 'trainandpredict':
            run_model(config, mode=args.mode, debug=args.debug, log_to_file=args.log_to_file, multiple_runs=args.multiple_runs)
        else:
            run_model(config, mode='train', debug=args.debug, log_to_file=args.log_to_file, multiple_runs=args.multiple_runs)
            with open(args.config_file, 'r') as config_file:
                config = yaml.safe_load(config_file)
            if args.multiple_runs != 1:
                config['Data']['seed'] = random_seed
            config['Data']['shuffle'] = False
            run_model(config, mode='predict', debug=args.debug, log_to_file=args.log_to_file, multiple_runs=args.multiple_runs)
