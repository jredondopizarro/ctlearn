import importlib
import logging
import numpy as np
import os
import pkg_resources
import sys
import pandas as pd
import time
import yaml

import tensorflow as tf

def setup_DL1DataReader(config, mode):

    # Parse file list or prediction file list
    if mode in ['train', 'load_only']:
        if isinstance(config['Data']['file_list'], str):
            data_files = []
            with open(config['Data']['file_list']) as f:
                for line in f:
                    line = line.strip()
                    if line and line[0] != "#":
                        data_files.append(line)
            config['Data']['file_list'] = data_files
        if not isinstance(config['Data']['file_list'], list):
            raise ValueError("Invalid file list '{}'. "
                             "Must be list or path to file".format(config['Data']['file_list']))
    else:
        if isinstance(config['Prediction']['prediction_file_list'], str):
            data_files = []
            with open(config['Prediction']['prediction_file_list']) as f:
                for line in f:
                    line = line.strip()
                    if line and line[0] != "#":
                        data_files.append(line)
            config['Data']['file_list'] = data_files
        if not isinstance(config['Data']['file_list'], list):
            raise ValueError("Invalid prediction file list '{}'. "
                             "Must be list or path to file".format(config['Prediction']['prediction_file_list']))
                             
    # Parse list of event selection filters
    event_selection = {}
    for s in config['Data'].get('event_selection', {}):
        s = {'module': 'dl1_data_handler.filters', **s}
        filter_fn, filter_params = load_from_module(**s)
        event_selection[filter_fn] = filter_params
    config['Data']['event_selection'] = event_selection

    # Parse list of image selection filters
    image_selection = {}
    for s in config['Data'].get('image_selection', {}):
        s = {'module': 'dl1_data_handler.filters', **s}
        filter_fn, filter_params = load_from_module(**s)
        image_selection[filter_fn] = filter_params
    config['Data']['image_selection'] = image_selection

    # Parse list of Transforms
    transforms = []
    for t in config['Data'].get('transforms', {}):
        t = {'module': 'dl1_data_handler.transforms', **t}
        transform, args = load_from_module(**t)
        transforms.append(transform(**args))
    config['Data']['transforms'] = transforms

    # Convert interpolation image shapes from lists to tuples, if present
    if 'interpolation_image_shape' in config['Data'].get('mapping_settings',{}):
        config['Data']['mapping_settings']['interpolation_image_shape'] = {
            k: tuple(l) for k, l in config['Data']['mapping_settings']['interpolation_image_shape'].items()}

    
    # Possibly add additional info to load if predicting to write later
    if mode == 'predict':

        if 'Prediction' not in config:
            config['Prediction'] = {}

        if config['Prediction'].get('save_identifiers', False):
            if 'event_info' not in config['Data']:
                config['Data']['event_info'] = []
            config['Data']['event_info'].extend(['event_id', 'obs_id'])
            if config['Data']['mode'] == 'mono':
                if 'array_info' not in config['Data']:
                    config['Data']['array_info'] = []
                config['Data']['array_info'].append('id')
    
    return config['Data']

def load_from_module(name, module, path=None, args=None):
    if path is not None and path not in sys.path:
        sys.path.append(path)
    mod = importlib.import_module(module)
    fn = getattr(mod, name)
    params = args if args is not None else {}
    return fn, params


# Define format for TensorFlow dataset
def setup_TFdataset_format(config, example_description, labels):

    config['Input']['output_names'] = names = [d['name'] for d in example_description]
    shapes = [tf.TensorShape(d['shape']) for d in example_description]
    # TensorFlow does not support conversion for NumPy unsigned dtypes
    # other than int8. Work around this by doing a manual conversion.
    dtypes = [d['dtype'] for d in example_description]
    for i, dtype in enumerate(dtypes):
        for utype, stype in [(np.uint16, np.int32), (np.uint32, np.int64)]:
            if dtype == utype:
                dtypes[i] = stype
    dtypes = [tf.as_dtype(d) for d in dtypes]
    feature_names, label_names = [], []
    feature_dtypes, label_dtypes, feature_shapes, label_shapes = {}, {}, {}, {}
    for i, (name, dtype, shape) in enumerate(zip(names, dtypes, shapes)):
        if name in labels.keys():
            label_names.append(name)
            label_dtypes[name] = dtype
            label_shapes[name] = shape
        else:
            feature_names.append(name)
            feature_dtypes[name] = dtype
            feature_shapes[name] = shape

    config['Input']['feature_names'] = feature_names
    config['Input']['label_names'] = label_names
    config['Input']['output_dtypes'] = (feature_dtypes, label_dtypes)
    config['Input']['output_shapes'] = (feature_shapes, label_shapes)

    return config['Input']

# Define input function
def input_fn(reader, indices, output_names, output_dtypes, output_shapes,
             feature_names, label_names, mode='train', seed=None,
             batch_size=1, shuffle_buffer_size=None, prefetch_buffer_size=1):

    def generator(indices):
        for idx in indices:
            [features, labels] = map(lambda keys: {name: reader[idx][i] for i,name in enumerate(output_names) if name in keys}, [feature_names, label_names])
            yield (features, labels)

    dataset = tf.data.Dataset.from_generator(generator, output_dtypes,
                                             output_shapes=output_shapes,
                                             args=(indices,))
       
    # Only shuffle the data, when train mode is selected.
    if mode == 'train':
        if shuffle_buffer_size is None:
            shuffle_buffer_size = len(indices)
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, seed=seed)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(prefetch_buffer_size)

    return dataset


