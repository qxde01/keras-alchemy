"""Xception V1 model for Keras.

On ImageNet, this model gets to a top-1 validation accuracy of 0.790
and a top-5 validation accuracy of 0.945.

Do note that the input image format for this model is different than for
the VGG16 and ResNet models (299x299 instead of 224x224),
and that the input preprocessing function
is also different (same as Inception V3).

# Reference

- [Xception: Deep Learning with Depthwise Separable Convolutions](
    https://arxiv.org/abs/1610.02357) (CVPR 2017)

"""
#from __future__ import absolute_import,division,print_function
import tensorflow as tf
if tf.__version__<'2.0':
    import keras
else:
    from tensorflow import keras

#import os
#import warnings

#from . import get_submodules_from_kwargs
#from . import imagenet_utils
#from .imagenet_utils import decode_predictions
#from .imagenet_utils import _obtain_input_shape


TF_WEIGHTS_PATH = (
    'https://github.com/fchollet/deep-learning-models/'
    'releases/download/v0.4/'
    'xception_weights_tf_dim_ordering_tf_kernels.h5')
TF_WEIGHTS_PATH_NO_TOP = (
    'https://github.com/fchollet/deep-learning-models/'
    'releases/download/v0.4/'
    'xception_weights_tf_dim_ordering_tf_kernels_notop.h5')


def conv2d_bn(x, filters, kernel_size=3, padding='same', strides=(1, 1), activation='relu', name=None):
    x = keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=False,
                            kernel_initializer='he_normal',name=name+'_conv2d')(x)
    x = keras.layers.BatchNormalization(axis=3, scale=True, momentum=0.95,name=name+'_conv2d_bn')(x)
    if activation is not None:
        x = keras.layers.Activation(activation, name=name+'_conv2d_bn_relu')(x)
    return x

def sepconv2d_bn(x, filters, kernel_size=3, padding='same', activation='relu', name=None):
    x = keras.layers.SeparableConv2D(filters, kernel_size=kernel_size, padding=padding, use_bias=False,
                            kernel_initializer='he_normal',name=name+'_sepconv2d')(x)
    x = keras.layers.BatchNormalization(axis=3, scale=True, momentum=0.95,name=name+'_sepconv2d_bn')(x)
    if activation is not None:
        x = keras.layers.Activation(activation, name=name+'_sepconv2d_bn_relu')(x)
    return x

def Xception(include_top=True,
             input_shape=None,
             pooling='avg',
             classes=1000,
             **kwargs):
    """Instantiates the Xception architecture.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    Note that the default input image size for this model is 299x299.

    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(299, 299, 3)`.
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 71.
            E.g. `(150, 150, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True,
            and if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
        RuntimeError: If attempting to run this model with a
            backend that does not support separable convolutions.
    """
    if input_shape[0]>112:
        stride0=2
    else:
        stride0=1
    img_input = keras.layers.Input(shape=input_shape)
    channel_axis = -1 #1 if backend.image_data_format() == 'channels_first' else -1

    #strides=(2,2)-->(1,1)
    x=conv2d_bn(img_input,filters=32,kernel_size=3,strides=(stride0,stride0),name='block1_conv1')
    # strides=(2,2)-->(1,1)
    x = conv2d_bn(x, filters=64, kernel_size=3, strides=(2, 2), name='block1_conv2')

    residual = conv2d_bn(x, filters=128, kernel_size=1, strides=(2, 2),activation=None, name='block1_residual1')

    x=sepconv2d_bn(x, filters=128, kernel_size=3, padding='same', activation='relu', name='block2_sepconv1')
    x = sepconv2d_bn(x, filters=128, kernel_size=3, padding='same', activation=None, name='block2_sepconv2')
    x = keras.layers.MaxPooling2D((3, 3),strides=(2, 2), padding='same', name='block2_pool')(x)
    x = keras.layers.add([x, residual])

    residual = conv2d_bn(x, filters=256, kernel_size=1, strides=(2, 2), activation=None, name='block2_residual2')

    x = sepconv2d_bn(x, filters=256, kernel_size=3, padding='same', activation='relu', name='block3_sepconv1')
    x = sepconv2d_bn(x, filters=256, kernel_size=3, padding='same', activation=None, name='block3_sepconv2')

    x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block3_pool')(x)
    x = keras.layers.add([x, residual])

    residual = conv2d_bn(x, filters=728, kernel_size=1, strides=(2, 2), activation=None, name='block3_residual3')

    x = sepconv2d_bn(x, filters=728, kernel_size=3, padding='same', activation='relu', name='block4_sepconv1')
    x = sepconv2d_bn(x, filters=728, kernel_size=3, padding='same', activation=None, name='block4_sepconv2')

    x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same',name='block4_pool')(x)
    x = keras.layers.add([x, residual])

    for i in range(8):
        residual = x
        prefix = 'block' + str(i + 5)
        x = keras.layers.Activation('relu', name=prefix + '_sepconv1_act')(x)
        x = sepconv2d_bn(x, filters=728, kernel_size=3, padding='same', activation='relu', name=prefix+'_sepconv1')
        x = sepconv2d_bn(x, filters=728, kernel_size=3, padding='same', activation='relu', name=prefix + '_sepconv2')
        x = sepconv2d_bn(x, filters=728, kernel_size=3, padding='same', activation=None, name=prefix + '_sepconv3')
        x = keras.layers.add([x, residual])


    residual = conv2d_bn(x, filters=1024, kernel_size=1, strides=(2, 2), activation=None, name='block13_residual')

    x = keras.layers.Activation('relu', name='block13_sepconv1_act')(x)
    x = sepconv2d_bn(x, filters=728, kernel_size=3, padding='same', activation='relu', name='block13_sepconv1')
    x = sepconv2d_bn(x, filters=1024, kernel_size=3, padding='same', activation='relu', name='block13_sepconv2')


    x = keras.layers.MaxPooling2D((3, 3),strides=(2, 2),  padding='same', name='block13_pool')(x)
    x = keras.layers.add([x, residual])

    x = sepconv2d_bn(x, filters=1536, kernel_size=3, padding='same', activation='relu', name='block14_sepconv1')
    x = sepconv2d_bn(x, filters=2048, kernel_size=3, padding='same', activation='relu', name='block14_sepconv2')

    if pooling == 'avg':
        x = keras.layers.GlobalAveragePooling2D()(x)
    elif pooling == 'max':
        x = keras.layers.GlobalMaxPooling2D()(x)

    if include_top:
        x = keras.layers.Dense(classes, activation='softmax', name='predictions')(x)

    model = keras.models.Model(img_input, x, name='xception')
    return model


if __name__ == '__main__':
    #21,066,380
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    input_shape = (32, 32, 3)
    model = Xception(input_shape=input_shape, classes=100)
    model.summary()
    keras.utils.plot_model(model, 'Xception.png', show_shapes=True)
