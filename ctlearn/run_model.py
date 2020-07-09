import argparse
import importlib
import logging
import math
from random import randint
import os
from pprint import pformat
import sys
from functools import partial
import pickle
import numpy as np
import yaml

import tensorflow as tf
from tensorflow.keras import backend as K

from dl1_data_handler.reader import DL1DataReader
from ctlearn.data_loader import *
from ctlearn.utils import *

# Disable Tensorflow info and warning messages (not error messages)
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#tf.logging.set_verbosity(tf.logging.WARN)

ANNEAL_KL = False
# epochs to anneal the KL term (anneals from 0 to 1)
KL_ANNEALING = 5
# whether to use the alternative annealing strategy
ALTERNATIVE_ANNEALING = False

def run_model(config, mode="train", debug=False, log_to_file=False, multiple_runs=1):

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
        model_dir += "/experiment_{}".format(random_seed)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    # Set up logging, saving the config and optionally logging to a file
    logger = setup_logging(config, model_dir, debug, log_to_file)

    # Log the loaded configuration
    logger.debug(pformat(config))

    logger.info("Logging has been correctly set up")

    # Create params dictionary that will be passed to the model_fn
    params = {}
    
    tasks = config['Model Parameters']['custom_head']

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
    config['Input'], feature_shapes = setup_TFdataset_format(config, params['example_description'], tasks)
    batch_size = config['Input'].get('batch_size', 1)
    config['Training']['batch_size'] = batch_size
    logger.info("Batch size: {}".format(batch_size))

    # Load either training or prediction options
    # and log information about the data set
    indices = reader_indices = list(range(len(reader)))

    # Write the training configuration in the params dict
    params['training'] = config['Training']
    # Write the evaluation configuration in the params dict
    params['evaluation'] = config['Evaluation']
    
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
        validation_indices = reader_indices = indices[num_training_examples:]
        training_steps_per_epoch = int(num_training_examples / batch_size)
        logger.info("Number of training steps per epoch: {}".format(
            training_steps_per_epoch))
        training_steps = config['Training'].get('num_training_steps_per_validation', training_steps_per_epoch)
        logger.info("Number of training steps between validations: {}".format(
            training_steps))

    if mode == 'load_only':

        log_examples(reader, indices, tasks, 'total dataset')
        log_examples(reader, training_indices, tasks, 'training')
        log_examples(reader, validation_indices, tasks, 'validation')
        # If only loading data, can end now that dataset logging is complete
        return

    if mode == 'train' and config['Training']['apply_class_weights']:
        num_class_examples = log_examples(reader, training_indices,
                                          tasks, 'training')
        class_weights = compute_class_weights(tasks, num_class_examples)
        params['training']['class_weights'] = class_weights

    # Load options for TensorFlow
    run_tfdbg = config.get('TensorFlow', {}).get('run_TFDBG', False)

    model_file = config['Model'].get('model_file', None)
    if model_file:
        model = tf.keras.models.load_model(model_file)
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
            model = tf.keras.models.load_model(model_dir+'/ctlearn_model.h5')
        else:
            
            params['model'] = {**config['Model'], **config.get('Model Parameters', {}), **config.get('Training', {}), **config.get('Losses', {})}
            
            model_module = importlib.import_module(config['Model']['model']['module'])
            model_fn = getattr(model_module, config['Model']['model']['function'])

            callbacks_list = []

            # KL ANNEALING
            class AnnealingCallback(tf.keras.callbacks.Callback):
                def __init__(self, t):
                    super(AnnealingCallback, self).__init__()
                    self.t = t

                def on_train_batch_end(self, batch, logs=None):
                    new_t = K.get_value(self.t) + 1
                    K.set_value(self.t, new_t)
                    print('Current model losses: ' + str(sum(model.losses)))
                    if not ALTERNATIVE_ANNEALING:
                        print('Current KL Regularizer: ' + str(K.minimum(1, K.get_value(kl_regularizer))))
                    else:
                        print('Current KL Weight: ' + str(K.get_value(kl_weight)))

            if ANNEAL_KL:

                if not ALTERNATIVE_ANNEALING:
                    t = K.variable(0.0)
                    kl_regularizer = t / (KL_ANNEALING * num_training_examples / batch_size)
                    kl_weight = 1 / num_training_examples * K.minimum(1.0, kl_regularizer)

                else:
                    t = K.variable(1.0)
                    kl_weight = 2 ** (training_steps_per_epoch - t) / (2 ** training_steps_per_epoch - 1)

                callbacks_list.append(AnnealingCallback(t))

            else:
                kl_weight = 1 / num_training_examples

            # get the model
            model = model_fn(feature_shapes, kl_weight)
            print(model.summary())

    if mode == 'train':

        training_data = input_fn(reader, training_indices, mode='train', **config['Input'])
        validation_data = input_fn(reader, validation_indices, mode='eval', **config['Input'])
        
        history = model.fit(training_data,
                            epochs=params['training']['num_epochs'],
                            steps_per_epoch=training_steps_per_epoch,
                            validation_data=validation_data,
                            verbose=params['training']['verbose'],
                            callbacks=callbacks_list)

        model.save(model_dir+'/ctlearn_model.h5')

        with open(model_dir + '/history.pickle', 'wb') as file:
            pickle.dump(history, file)


    elif mode == 'predict':

        prediction_data = input_fn(reader, indices, mode='predict', **config['Input'])

        if config['Prediction'].get('monte_carlo_sampling', False):
            num_samples = config['Prediction'].get('num_samples', 100)
            predictions = [model.predict(prediction_data) for _ in num_samples]

        else:
            predictions = model.predict(prediction_data)

        with open(model_dir + '/predictions.pickle', 'wb') as file:
            pickle.dump(predictions, file)

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
        random_seed = config['Data']['seed']
        
        if args.multiple_runs != 1:
            # Create and overwrite the random seed in the config file
            while True:
                random_seed = randint(1000,9999)
                if random_seed not in random_seeds:
                    random_seeds.append(random_seed)
                    break
            config['Data']['seed'] = random_seed
            print("CTLearn run {} with random seed '{}':".format(run+1,config['Data']['seed']))
        
        if args.mode == 'train':
            run_model(config, mode=args.mode, debug=args.debug, log_to_file=args.log_to_file, multiple_runs=args.multiple_runs)
        elif args.mode == 'predict':
            for key in config['Prediction']['prediction_file_lists']:
                with open(args.config_file, 'r') as config_file:
                    config = yaml.safe_load(config_file)
                config['Data']['seed'] = random_seed
                config['Data']['shuffle'] = False
                config['Prediction']['prediction_label'] = key
                run_model(config, mode=args.mode, debug=args.debug, log_to_file=args.log_to_file, multiple_runs=args.multiple_runs)
        else:
            run_model(config, mode='train', debug=args.debug, log_to_file=args.log_to_file, multiple_runs=args.multiple_runs)
            for key in config['Prediction']['prediction_file_lists']:
                with open(args.config_file, 'r') as config_file:
                    config = yaml.safe_load(config_file)
                config['Data']['seed'] = random_seed
                config['Data']['shuffle'] = False
                config['Prediction']['prediction_label'] = key
                run_model(config, mode='predict', debug=args.debug, log_to_file=args.log_to_file, multiple_runs=args.multiple_runs)


