import os
import tensorflow as tf

os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'

import IPython.display as display

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False

import numpy as np
import PIL.Image
import time
import functools
from Preprocess import load_img, tensor_to_image

content_path = input('content_path: ')
style_path = input('style_path: ')
content_img = load_img(content_path)
style_img = load_img(style_path)

# vgg =  tf.keras.applications.VGG19(include_top= True,weights='imagenet',)

content_layers = [
    'block5_conv2']  # In expermient Researcher found out these layers to be good for style and content layers

style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)


def vgg_layers(layers):
    vgg = tf.keras.applications.VGG19(include_top=True, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(layer).output for layer in layers]
    model = tf.keras.Model([vgg.input], outputs)
    return model


# style_extractor = vgg_layers(style_layers)  # returns a model(x)
# style_output = style_extractor(style_img * 255)


def compute_gram_matrix(feature_map):
    flattened = tf.reshape(feature_map, [-1, feature_map.shape[-1]])
    gram_matrix = tf.matmul(flattened, flattened, transpose_a=True)
    num_elements = tf.reduce_prod(feature_map.shape[:-1])
    gram_matrix /= tf.cast(num_elements, tf.float32)
    return gram_matrix
