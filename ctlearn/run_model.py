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
import time
import tensorflow as tf
from tensorflow.keras import backend as K

from dl1_data_handler.reader import DL1DataReader
from ctlearn.data_loader import *
from ctlearn.utils import *

# Disable Tensorflow info and warning messages (not error messages)
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#tf.logging.set_verbosity(tf.logging.WARN)

#tf.config.experimental_run_functions_eagerly(True)


# tf hyperparameters:
LEARNING_RATE = 0.0005
EPSILON = 1.0e-8
LOG_FREQ = 10

# whether to anneal the KL divergence term
ANNEAL_KL = False
# epochs to anneal the KL term (anneals from 0 to 1)
KL_ANNEALING = 5
# whether to use the alternative annealing strategy, only if ANNEAL_KL is True (anneals from 0.5 to 0)
ALTERNATIVE_ANNEALING = False


def run_model_keras(config, mode="train", debug=False, log_to_file=False, multiple_runs=1):

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
            model_fn = getattr(model_module, 'build_bayesian_model_keras')

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
                            verbose=params['training']['verbose'])

        model.save(model_dir+'/ctlearn_model.h5')

        with open(model_dir + '/history.pickle', 'wb') as file:
            pickle.dump(history, file)


    elif mode == 'predict':

        prediction_data = input_fn(reader, indices, mode='predict', **config['Input'])

        predictions = model.predict(prediction_data)

    # clear the handlers, shutdown the logging and delete the logger
    logger.handlers.clear()
    logging.shutdown()
    del logger
    return


def run_model_tf(config, mode="train", debug=False, log_to_file=False, multiple_runs=1):
    # Load options relating to logging and checkpointing
    root_model_dir = model_dir = config['Logging']['model_directory']
    # Create model directory if it doesn't exist already
    if not os.path.exists(root_model_dir):
        if mode == 'predict':
            raise ValueError("Invalid model directory '{}'. "
                             "Must be a path to an existing directory in the predict mode.".format(
                config['Logging']['model_directory']))
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
            model = tf.keras.models.load_model(model_dir + '/ctlearn_model.h5')
        else:

            params['model'] = {**config['Model'], **config.get('Model Parameters', {}), **config.get('Training', {}),
                               **config.get('Losses', {})}

            model_module = importlib.import_module(config['Model']['model']['module'])
            model_fn = getattr(model_module, 'build_bayesian_model_tf')

            # get the model
            model = model_fn(feature_shapes)

    if mode == 'train':

        training_data = input_fn(reader, training_indices, mode='train', **config['Input'])
        validation_data = input_fn(reader, validation_indices, mode='eval', **config['Input'])

        if ANNEAL_KL:

            if not ALTERNATIVE_ANNEALING:
                t = K.variable(0.0)
                kl_regularizer = t / (KL_ANNEALING * num_training_examples / batch_size)
                kl_weight = 1 / num_training_examples * K.minimum(1.0, kl_regularizer)

            else:
                t = K.variable(1.0)
                kl_weight = 2 ** (training_steps_per_epoch - t) / (2 ** training_steps_per_epoch - 1)

        else:
            t = K.variable(0.0) # no sirve para nada en este caso
            kl_weight = 1 / num_training_examples

        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE,
                                             epsilon=EPSILON)

        train_total_loss_metric = tf.keras.metrics.Mean(name='train_total_loss')
        train_kl_divergence_metric = tf.keras.metrics.Mean(name='train_kl_divergence')
        train_accuracy_metric = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
        train_auc_metric = tf.keras.metrics.AUC(name='train_auc')

        val_total_loss_metric = tf.keras.metrics.Mean(name='val_total_loss')
        val_kl_divergence_metric = tf.keras.metrics.Mean(name='val_kl_divergence')
        val_accuracy_metric = tf.keras.metrics.BinaryAccuracy(name='val_accuracy')
        val_auc_metric = tf.keras.metrics.AUC(name='val_auc')

        @tf.function
        def train_step(inputs, labels):
            labels = tf.reshape(tf.cast(labels['particletype'], dtype=tf.float32), (-1, 1))
            with tf.GradientTape() as tape:
                predictions = model(inputs, training=True)
                neg_log_likelihood = K.sum(K.binary_crossentropy(labels, predictions), axis=-1)
                kl_divergence = sum(model.losses) * kl_weight
                loss = neg_log_likelihood + kl_divergence
            # update the weights
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            # update the metrics
            train_total_loss_metric.update_state(loss)
            train_kl_divergence_metric.update_state(kl_divergence)
            train_accuracy_metric.update_state(labels, predictions)
            train_auc_metric.update_state(labels, predictions)

        def test_step(inputs, labels):
            labels = tf.reshape(tf.cast(labels['particletype'], dtype=tf.float32), (-1, 1))
            predictions = model(inputs)
            neg_log_likelihood = K.sum(K.binary_crossentropy(labels, predictions), axis=-1)
            kl_divergence = sum(model.losses) * kl_weight
            loss = neg_log_likelihood + kl_divergence

            val_total_loss_metric.update_state(loss)
            val_kl_divergence_metric.update_state(kl_divergence)
            val_accuracy_metric.update_state(labels, predictions)
            val_auc_metric.update_state(labels, predictions)

        for epoch in range(params['training']['num_epochs']):
            print(f'Beginning epoch: {epoch+1}')

            print(f'Beginning training - Epoch: {epoch+1}')
            for batch_idx, (inputs, labels) in enumerate(training_data):

                start = time.perf_counter()
                train_step(inputs, labels)
                step_time = time.perf_counter() - start

                K.set_value(t, K.get_value(t) + 1)

                if batch_idx % LOG_FREQ == 0:
                    mean_total_loss = train_total_loss_metric.result().numpy()
                    mean_kl_divergence = train_kl_divergence_metric.result().numpy()
                    mean_accuracy = train_accuracy_metric.result().numpy()
                    mean_auc = train_auc_metric.result().numpy()

                    print(f'Epoch: {epoch+1}')
                    print(f'Step: {batch_idx}')
                    print(f'Train total loss: {mean_total_loss:.3f}. Train KL div: {mean_kl_divergence:.5f}')
                    print(f'Train accuracy: {mean_accuracy:.3f}. Train auc: {mean_auc:.3f}')
                    print(f'Current KL weight: {kl_weight:.10f}')
                    print(f'Steps per second: {1/step_time}')
                    print(f'Aprox. remaining time: {(training_steps_per_epoch-batch_idx)/step_time} s')

                    train_total_loss_metric.reset_states()
                    train_kl_divergence_metric.reset_states()
                    train_accuracy_metric.reset_states()
                    train_auc_metric.reset_states()

            print(f'Ending training - Epoch: {epoch+1}')

            print(f'Beginning validation - Epoch: {epoch+1}')
            for inputs, labels in validation_data:
                test_step(inputs, labels)

            mean_total_loss = val_total_loss_metric.result().numpy()
            mean_kl_divergence = val_kl_divergence_metric.result().numpy()
            mean_accuracy = val_accuracy_metric.result().numpy()
            mean_auc = val_auc_metric.result().numpy()

            print(f'Epoch: {epoch + 1}')
            print(f'Val total loss: {mean_total_loss:.3f}. Val KL div: {mean_kl_divergence:.5f}')
            print(f'Val accuracy: {mean_accuracy:.3f}. Val auc: {mean_auc:.3f}')

            val_total_loss_metric.reset_states()
            val_kl_divergence_metric.reset_states()
            val_accuracy_metric.reset_states()
            val_auc_metric.reset_states()

            print(f'Ending validation - Epoch: {epoch+1}')
            print("Finished epoch")

        model.save(model_dir + '/ctlearn_model.h5')


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

