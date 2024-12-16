# # Lint as: python
# #
# # Authors: Vittorio | Francesco
# # Location: Turin, Biella, Ivrea
# #
# # This file is based on the work of Francisco Dorr - PROBA-V-3DWDSR (https://github.com/frandorr/PROBA-V-3DWDSR)


# """RAMS functions and building"""
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras import datasets, layers, models, Input, Model, regularizers
# import tensorflow_addons as tfa

# #-------------
# # Settings
# #-------------
# MEAN = 7433.6436 # mean of the proba-v dataset
# STD = 2353.0723 # std of proba-v the dataset

# def normalize(x):
#     """Normalize tensor"""
#     return (x-MEAN)/STD

# def denormalize(x):
#     """Denormalize tensor"""
#     return x * STD + MEAN 

# def conv3d_weightnorm(filters, kernel_size, padding='same', activation=None, **kwargs):
#     """3D convolution with weight normalization"""
#     return tfa.layers.WeightNormalization(layers.Conv3D(filters, kernel_size, padding=padding, activation=activation, **kwargs), data_init=False)

# def conv2d_weightnorm(filters, kernel_size, padding='same', activation=None, **kwargs):
#     """2D convolution with weight normalization"""
#     return tfa.layers.WeightNormalization(layers.Conv2D(filters, kernel_size, padding=padding, activation=activation, **kwargs), data_init=False)

# def reflective_padding(name):
#     """Reflecting padding on H and W dimension"""
#     return layers.Lambda(lambda x: tf.pad(x, [[0,0],[1,1],[1,1],[0,0],[0,0]], mode='REFLECT', name=name))


# def RFAB(x, filters, kernel_size, r):
#     """Residual Feature attention Block"""
#     x_res = x
    
#     # Attention
#     x = conv3d_weightnorm(filters, kernel_size)(x)
#     x = layers.ReLU()(x)
#     x = conv3d_weightnorm(filters, kernel_size)(x)
    
#     x_to_scale = x
    
#     x = layers.GlobalAveragePooling3D()(x)
#     # expand to 1x1x1xc
#     for i in range(3):
#         x = layers.Lambda(lambda x: tf.expand_dims(x, axis=(-2)))(x)   
#     x = conv3d_weightnorm(int(filters/r), 1)(x)
#     x = layers.ReLU()(x)
#     x = conv3d_weightnorm(filters, 1, activation='sigmoid')(x)
    
#     x_scaled = x_to_scale * x
    
#     return x_scaled + x_res

# def RTAB(x, filters, kernel_size, r):
#     """Residual Temporal Attention Blcok"""
#     # residual
#     x_res = x
    
#     # Attention
#     x = conv2d_weightnorm(filters, kernel_size)(x)
#     x = layers.ReLU()(x)
#     x = conv2d_weightnorm(filters, kernel_size)(x)
    
#     x_to_scale = x
    
#     x = layers.GlobalAveragePooling2D()(x)
#     # expand to 1x1xc
#     for i in range(2):
#         x = layers.Lambda(lambda x: tf.expand_dims(x, axis=(-2)))(x)   
#     x = conv2d_weightnorm(int(filters/r), 1)(x)
#     x = layers.ReLU()(x)
#     x = conv2d_weightnorm(filters, 1, activation='sigmoid')(x)
    
#     x_scaled = x_to_scale * x
    
#     return x_scaled + x_res



# def RAMS(scale, filters, kernel_size, channels, r, N):
#     """
#     Build RAMS Deep Neural Network
    
#     Parameters
#     ----------
#     scale: int
#         uscale factor
#     filters: int
#         number of filters
#     kernel_size: int
#         convolutional kernel dimension
#     channels: int
#         number of channels
#     r: int
#         compression factor
#     N: int
#         number of RFAB
#     """
#     img_inputs = Input(shape=(None,None,channels))

#     # normalize, expand and pad
#     x = layers.Lambda(normalize)(img_inputs)
#     x_global_res = x
#     x = layers.Lambda(lambda x: tf.expand_dims(x, -1))(x)
#     x = reflective_padding(name="initial_padding")(x)
    
#     # low level features extraction
#     x = conv3d_weightnorm(filters, kernel_size)(x)
    
#     # LSC
#     x_res = x
      
#     for i in range(N):
#         x = RFAB(x, filters, kernel_size, r)
        
#     x = conv3d_weightnorm(filters, kernel_size)(x)
    
#     x = x + x_res
    
#     # Temporal Reduction out: HxWxC
#     for i in range(0, np.floor((channels-1)/(kernel_size-1)-1).astype(int)):
#         x = reflective_padding(name="ref_padding_{}".format(i))(x)
#         x = RFAB(x, filters, kernel_size, r)
#         x = conv3d_weightnorm(filters, (3,3,3), padding='valid', activation='relu',
#                               name="conv_reduction_{}".format(i))(x)

#     # Upscaling    
#     x = conv3d_weightnorm(scale ** 2, (3,3,3), padding='valid')(x)
#     x = layers.Lambda(lambda x: x[...,0,:])(x)
#     x = layers.Lambda(lambda x: tf.nn.depth_to_space(x, scale))(x)
    
    
#     # global path
#     x_global_res = layers.Lambda(lambda x: tf.pad(x,[[0,0],[1,1],[1,1],[0,0]], mode='REFLECT', name='padding_2d'))(x_global_res)
#     x_global_res = RTAB(x_global_res, 9, kernel_size, r)
#     x_global_res = conv2d_weightnorm(scale ** 2, (3,3), padding='valid')(x_global_res)
#     x_global_res = layers.Lambda(lambda x: tf.nn.depth_to_space(x, scale))(x_global_res)
    
#     # output
#     x = x + x_global_res
#     outputs = layers.Lambda(denormalize)(x)
    

#     return Model(img_inputs, outputs, name="RAMS")

# Lint as: python
#
# Authors: Vittorio | Francesco
# Location: Turin, Biella, Ivrea
#
# This file is based on the work of Francisco Dorr - PROBA-V-3DWDSR (https://github.com/frandorr/PROBA-V-3DWDSR)


import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model, layers
import tensorflow_addons as tfa

#-------------
# Settings
#-------------
MEAN = 7433.6436  # mean of the proba-v dataset
STD = 2353.0723   # std of the proba-v dataset

def normalize(x):
    """Normalize tensor"""
    return (x - MEAN) / STD

def denormalize(x):
    """Denormalize tensor"""
    return x * STD + MEAN

def conv3d_weightnorm(filters, kernel_size, padding='same', activation=None, **kwargs):
    """3D convolution with weight normalization"""
    return tfa.layers.WeightNormalization(layers.Conv3D(filters, kernel_size, padding=padding, activation=activation, **kwargs), data_init=False)

def conv2d_weightnorm(filters, kernel_size, padding='same', activation=None, **kwargs):
    """2D convolution with weight normalization"""
    return tfa.layers.WeightNormalization(layers.Conv2D(filters, kernel_size, padding=padding, activation=activation, **kwargs), data_init=False)

def reflective_padding(name):
    """Reflecting padding on H and W dimension for 3D input (N,H,W,C,T)"""
    return layers.Lambda(lambda x: tf.pad(x, [[0,0],[1,1],[1,1],[0,0],[0,0]], mode='REFLECT', name=name))

def reflective_padding_2d(name):
    """Reflecting padding on H and W dimension for 2D input (N,H,W,C)"""
    return layers.Lambda(lambda x: tf.pad(x, [[0,0],[1,1],[1,1],[0,0]], mode='REFLECT', name=name))

def simple_3d_res_block(x, filters, kernel_size):
    """A simpler 3D residual block without attention."""
    x_res = x
    x = conv3d_weightnorm(filters, kernel_size)(x)
    x = layers.ReLU()(x)
    x = conv3d_weightnorm(filters, kernel_size)(x)
    return x + x_res

def simple_2d_res_block(x, filters, kernel_size):
    """A simpler 2D residual block without attention."""
    x_res = x
    x = conv2d_weightnorm(filters, kernel_size)(x)
    x = layers.ReLU()(x)
    x = conv2d_weightnorm(filters, kernel_size)(x)
    return x + x_res

def RAMS(scale, filters, kernel_size, channels, r, N):
    """
    Build a simplified RAMS Deep Neural Network

    Parameters
    ----------
    scale: int
        Upscale factor
    filters: int
        Number of filters
    kernel_size: int
        Convolutional kernel dimension
    channels: int
        Number of input channels
    r: int
        Compression factor (not used in simplified blocks, but retained for signature)
    N: int
        Number of residual blocks (originally RFAB, now simple 3D res blocks)
    """
    img_inputs = Input(shape=(None, None, channels))

    # normalize and expand
    x = layers.Lambda(normalize)(img_inputs)
    x_global_res = x
    x = layers.Lambda(lambda x: tf.expand_dims(x, -1))(x)
    x = reflective_padding(name="initial_padding")(x)

    # Low level features
    x = conv3d_weightnorm(filters, kernel_size)(x)

    # A series of simple 3D residual blocks
    x_res = x
    for i in range(N):
        x = simple_3d_res_block(x, filters, kernel_size)
    x = conv3d_weightnorm(filters, kernel_size)(x)
    x = x + x_res

    # Temporal reduction: In original code, complex logic. Here just add a couple of conv3D layers.
    # We maintain reflective padding and a few residual passes without attention.
    # The logic for temporal reduction (depending on channels and kernel_size) is simplified.
    # We'll just do one or two residual blocks with valid padding to mimic dimension changes.
    
    # Example: Perform one more residual block with padding and a conv3D reduction
    x = reflective_padding(name="ref_padding_temporal")(x)
    x = simple_3d_res_block(x, filters, kernel_size)
    x = conv3d_weightnorm(filters, (3,3,3), padding='valid', activation='relu')(x)

    # Upscaling path
    x = conv3d_weightnorm(scale ** 2, (3,3,3), padding='valid')(x)
    # Extract the central feature map (T dimension)
    x = layers.Lambda(lambda x: x[...,0,:])(x)
    # Depth to space
    x = layers.Lambda(lambda x: tf.nn.depth_to_space(x, scale))(x)

    # Global path: simplified as well.
    x_global_res = reflective_padding_2d(name="padding_2d")(x_global_res)
    x_global_res = simple_2d_res_block(x_global_res, 9, kernel_size)
    x_global_res = conv2d_weightnorm(scale ** 2, (3,3), padding='valid')(x_global_res)
    x_global_res = layers.Lambda(lambda x: tf.nn.depth_to_space(x, scale))(x_global_res)

    # Combine paths and denormalize
    x = x + x_global_res
    outputs = layers.Lambda(denormalize)(x)

    return Model(img_inputs, outputs, name="RAMS")