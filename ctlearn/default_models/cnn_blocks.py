import importlib
import sys

import tensorflow as tf

from ctlearn.default_models.basic import *

def single_cnn(feature_shapes, model_params):
    image_input = tf.keras.layers.Input(shape=tf.TensorShape(feature_shapes['image']), name='image')
    output = conv_block(image_input, params=model_params)
    return image_input, output

def bayesian_single_cnn(feature_shapes, model_params, num_training_examples):
    image_input = tf.keras.layers.Input(shape=tf.TensorShape(feature_shapes['image']), name='image')
    output = bayesian_conv_block(image_input, model_params, num_training_examples)
    return image_input, output

def multi_cnn(feature_shapes, model_params):
    
    image_input = tf.keras.layers.Input(shape=tf.TensorShape(feature_shapes['image']), name='images')
    trigger_input = tf.keras.layers.Input(shape=tf.TensorShape(feature_shapes['trigger']), name='triggers')
    num_telescopes = feature_shapes['feature_shapes']
    
    # Transpose image_input from [batch_size,num_tel,length,width,channels]
    # to [num_tel,batch_size,length,width,channels].
    permuted_image_input = tf.keras.layers.Permute((1, 0, 2, 3, 4))(image_input)
    
    telescope_outputs = []
    for telescope_index in range(num_telescopes):
        # Set all telescopes after the first to share weights
        #reuse = None if telescope_index == 0 else True
        output = conv_block(tf.gather(permuted_image_input, telescope_index), params=model_params)
        
        #flatten output of embedding CNN to (batch_size, _)
        image_embedding = tf.keras.layers.Flatten(name='image_embedding')(output)
        image_embedding_dropout = tf.keras.layers.Dropout()(image_embedding)
        telescope_outputs.append(image_embedding_dropout)
        
    return [image_input, trigger_input], telescope_outputs

def keras_cnn(feature_shapes, model_params):

    image_input = tf.keras.layers.Input(shape=tf.TensorShape(feature_shapes['image']), name='images')
    
    model = getattr(tf.keras.applications, keras_models[model_params[keras]])(
        input_tensor=image_input,
        include_top=False,
        pooling='avg')
        
    return image_input, model.layers[-1].output
