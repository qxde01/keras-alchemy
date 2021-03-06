# Copyright 2019 The TensorFlow Authors, Pavel Yakubovskiy, Björn Barz. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains definitions for EfficientNet model.
[1] Mingxing Tan, Quoc V. Le
  EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.
  ICML'19, https://arxiv.org/abs/1905.11946
"""

# Code of this model implementation is mostly written by
# Björn Barz ([@Callidior](https://github.com/Callidior))
# https://github.com/qubvel/efficientnet
from __future__ import absolute_import,division,print_function
import tensorflow as tf
if tf.__version__<'2.0':
    import keras
else:
    from tensorflow import keras

import math
import string
import collections
#import numpy as np
from six.moves import xrange


BASE_WEIGHTS_PATH = (
    'https://github.com/Callidior/keras-applications/'
    'releases/download/efficientnet/')

BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'strides', 'se_ratio'
])
# defaults will be a public argument for namedtuple in Python 3.7
# https://docs.python.org/3/library/collections.html#collections.namedtuple
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)

DEFAULT_BLOCKS_ARGS = [
    BlockArgs(kernel_size=3, num_repeat=1, input_filters=32, output_filters=16,
              expand_ratio=1, id_skip=True, strides=[1, 1], se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=2, input_filters=16, output_filters=24,
              expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25), #strides=[2, 2]->[1,1]
    BlockArgs(kernel_size=5, num_repeat=2, input_filters=24, output_filters=40,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=3, input_filters=40, output_filters=80,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=3, input_filters=80, output_filters=112,
              expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=4, input_filters=112, output_filters=192,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=1, input_filters=192, output_filters=320,
              expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25)
]

CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        # EfficientNet actually uses an untruncated normal distribution for
        # initializing conv layers, but keras.initializers.VarianceScaling use
        # a truncated distribution.
        # We decided against a custom initializer for better serializability.
        'distribution': 'normal'
    }
}

DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}





def get_swish(**kwargs):
    def swish(x):
        """Swish activation function: x * sigmoid(x).
        Reference: [Searching for Activation Functions](https://arxiv.org/abs/1710.05941)
        """

        if keras.backend.backend() == 'tensorflow':
            try:
                # The native TF implementation has a more
                # memory-efficient gradient implementation
                return keras.backend.tf.nn.swish(x)
            except AttributeError:
                pass

        return x * keras.backend.sigmoid(x)
    return  swish

def swish(x):
    return x * keras.backend.sigmoid(x)

# ** update custom Activate functions


def get_dropout(**kwargs):
    """Wrapper over custom dropout. Fix problem of ``None`` shape for tf.keras.
    It is not possible to define FixedDropout class as global object,
    because we do not have modules for inheritance at first time.
    Issue:
        https://github.com/tensorflow/tensorflow/issues/30946
    """
    #backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    class FixedDropout(keras.layers.Dropout):
        def _get_noise_shape(self, inputs):
            if self.noise_shape is None:
                return self.noise_shape

            symbolic_shape = keras.backend.shape(inputs)
            noise_shape = [symbolic_shape[axis] if shape is None else shape
                           for axis, shape in enumerate(self.noise_shape)]
            return tuple(noise_shape)

    return FixedDropout

FixedDropout = get_dropout()
keras.utils.get_custom_objects().update({'custom_activation': keras.layers.Activation(swish)})
keras.utils.get_custom_objects().update({'FixedDropout': FixedDropout})

def round_filters(filters, width_coefficient, depth_divisor):
    """Round number of filters based on width multiplier."""

    filters *= width_coefficient
    new_filters = int(filters + depth_divisor / 2) // depth_divisor * depth_divisor
    new_filters = max(depth_divisor, new_filters)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return int(new_filters)


def round_repeats(repeats, depth_coefficient):
    """Round number of repeats based on depth multiplier."""

    return int(math.ceil(depth_coefficient * repeats))


def mb_conv_block(inputs, block_args, activation, drop_rate=None, prefix='', ):
    """Mobile Inverted Residual Bottleneck."""

    has_se = (block_args.se_ratio is not None) and (0 < block_args.se_ratio <= 1)
    bn_axis = 3 if keras.backend.image_data_format() == 'channels_last' else 1

    # workaround over non working dropout with None in noise_shape in tf.keras
    #Dropout = get_dropout()

    # Expansion phase
    filters = block_args.input_filters * block_args.expand_ratio
    if block_args.expand_ratio != 1:
        x = keras.layers.Conv2D(filters, 1,
                          padding='same',
                          use_bias=False,
                          kernel_initializer=CONV_KERNEL_INITIALIZER,
                          name=prefix + 'expand_conv')(inputs)
        x = keras.layers.BatchNormalization(axis=bn_axis, name=prefix + 'expand_bn')(x)
        x = keras.layers.Activation(activation, name=prefix + 'expand_activation')(x)
    else:
        x = inputs

    # Depthwise Convolution
    x = keras.layers.DepthwiseConv2D(block_args.kernel_size,
                               strides=block_args.strides,
                               padding='same',
                               use_bias=False,
                               depthwise_initializer=CONV_KERNEL_INITIALIZER,
                               name=prefix + 'dwconv')(x)
    x = keras.layers.BatchNormalization(axis=bn_axis, name=prefix + 'bn')(x)
    x = keras.layers.Activation(activation, name=prefix + 'activation')(x)

    # Squeeze and Excitation phase
    if has_se:
        num_reduced_filters = max(1, int(
            block_args.input_filters * block_args.se_ratio
        ))
        se_tensor = keras.layers.GlobalAveragePooling2D(name=prefix + 'se_squeeze')(x)

        target_shape = (1, 1, filters) if keras.backend.image_data_format() == 'channels_last' else (filters, 1, 1)
        se_tensor = keras.layers.Reshape(target_shape, name=prefix + 'se_reshape')(se_tensor)
        se_tensor = keras.layers.Conv2D(num_reduced_filters, 1,
                                  activation=activation,
                                  padding='same',
                                  use_bias=True,
                                  kernel_initializer=CONV_KERNEL_INITIALIZER,
                                  name=prefix + 'se_reduce')(se_tensor)
        se_tensor = keras.layers.Conv2D(filters, 1,
                                  activation='sigmoid',
                                  padding='same',
                                  use_bias=True,
                                  kernel_initializer=CONV_KERNEL_INITIALIZER,
                                  name=prefix + 'se_expand')(se_tensor)
        # if backend.backend() == 'theano':
        #     # For the Theano backend, we have to explicitly make
        #     # the excitation weights broadcastable.
        #     pattern = ([True, True, True, False] if backend.image_data_format() == 'channels_last'
        #                else [True, False, True, True])
        #     se_tensor = layers.Lambda(
        #         lambda x: backend.pattern_broadcast(x, pattern),
        #         name=prefix + 'se_broadcast')(se_tensor)
        x = keras.layers.multiply([x, se_tensor], name=prefix + 'se_excite')

    # Output phase
    x = keras.layers.Conv2D(block_args.output_filters, 1,
                      padding='same',
                      use_bias=False,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      name=prefix + 'project_conv')(x)
    x = keras.layers.BatchNormalization(axis=bn_axis, name=prefix + 'project_bn')(x)
    if block_args.id_skip and all(
            s == 1 for s in block_args.strides
    ) and block_args.input_filters == block_args.output_filters:
        if drop_rate and (drop_rate > 0):
            x = FixedDropout(drop_rate,noise_shape=(None, 1, 1, 1),name=prefix + 'drop')(x)
        x = keras.layers.add([x, inputs], name=prefix + 'add')

    return x


def EfficientNet(width_coefficient,
                 depth_coefficient,
                 dropout_rate=0.2,
                 drop_connect_rate=0.2,
                 depth_divisor=8,
                 blocks_args=DEFAULT_BLOCKS_ARGS,
                 model_name='efficientnet',
                 include_top=True,
                 input_shape=None,
                 pooling=None,
                 classes=1000,
                 **kwargs):
    """Instantiates the EfficientNet architecture using given scaling coefficients.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    # Arguments
        width_coefficient: float, scaling coefficient for network width.
        depth_coefficient: float, scaling coefficient for network depth.
        default_resolution: int, default input image size.
        dropout_rate: float, dropout rate before final classifier layer.
        drop_connect_rate: float, dropout rate at skip connections.
        depth_divisor: int.
        blocks_args: A list of BlockArgs to construct block modules.
        model_name: string, model name.
        include_top: whether to include the fully-connected
            layer at the top of the network.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False.
            It should have exactly 3 inputs channels.
        pooling: optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """

    img_input = keras.layers.Input(shape=input_shape)

    bn_axis = 3 if keras.backend.image_data_format() == 'channels_last' else 1
    #activation = get_swish(**kwargs)
    # Build stem
    x = img_input
    # strides=(2, 2)-> (1,1)
    x = keras.layers.Conv2D(round_filters(32, width_coefficient, depth_divisor), 3,
                      strides=(1, 1),
                      padding='same',
                      use_bias=False,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      name='stem_conv')(x)
    x = keras.layers.BatchNormalization(axis=bn_axis, name='stem_bn')(x)
    x = keras.layers.Activation(swish, name='stem_activation')(x)

    # Build blocks
    num_blocks_total = sum(block_args.num_repeat for block_args in blocks_args)
    block_num = 0
    for idx, block_args in enumerate(blocks_args):
        print(idx,block_args)
        assert block_args.num_repeat > 0
        # Update block input and output filters based on depth multiplier.
        block_args = block_args._replace(
            input_filters=round_filters(block_args.input_filters,
                                        width_coefficient, depth_divisor),
            output_filters=round_filters(block_args.output_filters,
                                         width_coefficient, depth_divisor),
            num_repeat=round_repeats(block_args.num_repeat, depth_coefficient))

        # The first block needs to take care of stride and filter size increase.
        drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
        x = mb_conv_block(x, block_args,
                          activation=swish,
                          drop_rate=drop_rate,
                          prefix='block{}a_'.format(idx + 1))
        block_num += 1
        if block_args.num_repeat > 1:
            # pylint: disable=protected-access
            block_args = block_args._replace(
                input_filters=block_args.output_filters, strides=[1, 1])
            # pylint: enable=protected-access
            for bidx in xrange(block_args.num_repeat - 1):
                drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
                block_prefix = 'block{}{}_'.format(
                    idx + 1,
                    string.ascii_lowercase[bidx + 1]
                )
                x = mb_conv_block(x, block_args,
                                  activation=swish,
                                  drop_rate=drop_rate,
                                  prefix=block_prefix)
                block_num += 1

    # Build top
    x = keras.layers.Conv2D(round_filters(1280, width_coefficient, depth_divisor), 1,
                      padding='same',
                      use_bias=False,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      name='top_conv')(x)
    x = keras.layers.BatchNormalization(axis=bn_axis, name='top_bn')(x)
    x = keras.layers.Activation(swish, name='top_activation')(x)
    if include_top:
        x = keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
        if dropout_rate and dropout_rate > 0:
            x = keras.layers.Dropout(dropout_rate, name='top_dropout')(x)
        x = keras.layers.Dense(classes,
                         activation='softmax',
                         kernel_initializer=DENSE_KERNEL_INITIALIZER,
                         name='probs')(x)
    else:
        if pooling == 'avg':
            x = keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = keras.layers.GlobalMaxPooling2D(name='max_pool')(x)

     # Create model.
    model = keras.models.Model(img_input, x, name=model_name)
    return model


def EfficientNetB0(include_top=True,
                   input_shape=(224,224,3),
                   pooling=None,
                   classes=1000,
                   **kwargs):
    return EfficientNet(1.0, 1.0, 0.2,
                        model_name='efficientnet-b0',
                        include_top=include_top,
                        input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        **kwargs)


def EfficientNetB1(include_top=True,
                   input_shape=(240,240,3),
                   pooling=None,
                   classes=1000,
                   **kwargs):
    return EfficientNet(1.0, 1.1, 0.2,
                        model_name='efficientnet-b1',
                        include_top=include_top, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        **kwargs)


def EfficientNetB2(include_top=True,
                   input_shape=(260,260,3),
                   pooling=None,
                   classes=1000,
                   **kwargs):
    return EfficientNet(1.1, 1.2, 0.3,
                        model_name='efficientnet-b2',
                        include_top=include_top, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        **kwargs)


def EfficientNetB3(include_top=True,
                   input_shape=(300,300,3),
                   pooling=None,
                   classes=1000,
                   **kwargs):
    return EfficientNet(1.2, 1.4, 0.3,
                        model_name='efficientnet-b3',
                        include_top=include_top, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        **kwargs)


def EfficientNetB4(include_top=True,
                   input_shape=(380,380,3),
                   pooling=None,
                   classes=1000,
                   **kwargs):
    return EfficientNet(1.4, 1.8, 0.4,
                        model_name='efficientnet-b4',
                        include_top=include_top, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        **kwargs)


def EfficientNetB5(include_top=True,
                   input_shape=(456,456,3),
                   pooling=None,
                   classes=1000,
                   **kwargs):
    return EfficientNet(1.6, 2.2, 0.4,
                        model_name='efficientnet-b5',
                        include_top=include_top, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        **kwargs)


def EfficientNetB6(include_top=True,
                   input_shape=(528,528,3),
                   pooling=None,
                   classes=1000,
                   **kwargs):
    return EfficientNet(1.8, 2.6, 0.5,
                        model_name='efficientnet-b6',
                        include_top=include_top, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        **kwargs)


def EfficientNetB7(include_top=True,
                   input_shape=(600.600,3),
                   pooling=None,
                   classes=1000,
                   **kwargs):
    return EfficientNet(2.0, 3.1, 0.5,
                        model_name='efficientnet-b7',
                        include_top=include_top, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        **kwargs)


setattr(EfficientNetB0, '__doc__', EfficientNet.__doc__)
setattr(EfficientNetB1, '__doc__', EfficientNet.__doc__)
setattr(EfficientNetB2, '__doc__', EfficientNet.__doc__)
setattr(EfficientNetB3, '__doc__', EfficientNet.__doc__)
setattr(EfficientNetB4, '__doc__', EfficientNet.__doc__)
setattr(EfficientNetB5, '__doc__', EfficientNet.__doc__)
setattr(EfficientNetB6, '__doc__', EfficientNet.__doc__)
setattr(EfficientNetB7, '__doc__', EfficientNet.__doc__)

if __name__ == '__main__':
    # B0,4,177,664
    # B1,6,703,332
    model = EfficientNetB1(include_top=True,input_shape=(32,32,3), classes=100)
    print(model.summary())
    keras.utils.plot_model(model, 'EfficientNetB1.png', show_shapes=True)
