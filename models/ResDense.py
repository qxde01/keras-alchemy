# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
if tf.__version__<'2.0':
    import keras
else:
    from tensorflow import keras

def basic_block(x,
    filters,
    stage=0,
    block=0,
    kernel_size=3,
    numerical_name=False,
    stride=None
):
    """
    A two-dimensional basic block.
    :param filters: the output’s feature space
    :param stage: int representing the stage of this block (starting from 0)
    :param block: int representing this block (starting from 0)
    :param kernel_size: size of the kernel
    :param numerical_name: if true, uses numbers to represent blocks instead of chars (ResNet{101, 152, 200})
    :param stride: int representing the stride used in the shortcut and the first conv layer, default derives stride from block id
    """
    if stride is None:
        if block != 0 or stage == 0:
            stride = 1
        else:
            stride = 2
    axis = 3

    if block > 0 and numerical_name:
        block_char = "b{}".format(block)
    else:
        block_char = chr(ord('a') + block)

    stage_char = str(stage + 2)
    print('stride:',stride)
    y = keras.layers.ZeroPadding2D(padding=1, name="padding{}{}_branch2a".format(stage_char, block_char))(x)
    y = keras.layers.Conv2D(filters, kernel_size, strides=stride,kernel_initializer='he_normal' ,use_bias=False, name="res{}{}_branch2a".format(stage_char, block_char))(y)
    y = keras.layers.BatchNormalization(axis=axis, epsilon=1e-5, name="bn{}{}_branch2a".format(stage_char, block_char))(y)
    y = keras.layers.Activation("relu", name="res{}{}_branch2a_relu".format(stage_char, block_char))(y)
    y = keras.layers.ZeroPadding2D(padding=1, name="padding{}{}_branch2b".format(stage_char, block_char))(y)
    y = keras.layers.Conv2D(filters, kernel_size, use_bias=False,kernel_initializer='he_normal' , name="res{}{}_branch2b".format(stage_char, block_char))(y)
    y = keras.layers.BatchNormalization(axis=axis, epsilon=1e-5, name="bn{}{}_branch2b".format(stage_char, block_char))(y)
    #print(block,y)

    if block == 0 :
        #print(block,stride,y)
        shortcut = keras.layers.Conv2D(filters, (1, 1), strides=stride, use_bias=False, name="res{}{}_branch1".format(stage_char, block_char))(x)
        shortcut = keras.layers.BatchNormalization(axis=axis, epsilon=1e-5,  name="bn{}{}_branch1".format(stage_char, block_char))(shortcut)
    else:
        shortcut = x
    #print('shotcut:',shortcut)
    y = keras.layers.Add(name="res{}{}".format(stage_char, block_char))([y, shortcut])
    y = keras.layers.Activation("relu", name="res{}{}_relu".format(stage_char, block_char))(y)
    return y

def stack_basic(x, filters, blocks, stride=2,stage=0):
    """A set of stacked residual blocks.

    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer in a block.
        blocks: integer, blocks in the stacked blocks.
        stride1: default 2, stride of the first layer in the first block.
        name: string, stack label.

    # Returns
        Output tensor for the stacked blocks.
    """
    x = basic_block(x, filters, block=0,stage=stage)
    print(x)
    for i in range(1, blocks):
         x = basic_block(x, filters,block=i,stride=stride,stage=stage)
         print(i,x)
    return x



def block2(x, filters, kernel_size=3, stride=1, conv_shortcut=False, name=None):
    """A residual block.

    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer.
        kernel_size: default 3, kernel size of the bottleneck layer.
        stride: default 1, stride of the first layer.
        conv_shortcut: default False, use convolution shortcut if True,
            otherwise identity shortcut.
        name: string, block label.

    # Returns
        Output tensor for the residual block.
    """
    bn_axis = 3 if keras.backend.image_data_format() == 'channels_last' else 1
    preact = keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,name=name + '_preact_bn')(x)
    preact = keras.layers.Activation('relu', name=name + '_preact_relu')(preact)
    if conv_shortcut is True:
        shortcut = keras.layers.Conv2D(4 * filters, 1, strides=stride,name=name + '_0_conv')(preact)
    else:
        shortcut = keras.layers.MaxPooling2D(1, strides=stride)(x) if stride > 1 else x
    x = keras.layers.Conv2D(filters, 1, strides=1, use_bias=False, name=name + '_1_conv')(preact)
    x = keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
    x = keras.layers.Activation('relu', name=name + '_1_relu')(x)
    x = keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + '_2_pad')(x)
    x = keras.layers.Conv2D(filters, kernel_size, strides=stride,use_bias=False, name=name + '_2_conv')(x)
    x = keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
    x = keras.layers.Activation('relu', name=name + '_2_relu')(x)
    x = keras.layers.Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
    x = keras.layers.Add(name=name + '_out')([shortcut, x])
    return x


def stack2(x, filters, blocks, stride1=2, name=None):
    """A set of stacked residual blocks.

    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer in a block.
        blocks: integer, blocks in the stacked blocks.
        stride1: default 2, stride of the first layer in the first block.
        name: string, stack label.

    # Returns
        Output tensor for the stacked blocks.
    """
    x = block2(x, filters, conv_shortcut=True, name=name + '_block0')
    if blocks>1:
        for i in range(2, blocks):
            x = block2(x, filters, name=name + '_block' + str(i))
    x = block2(x, filters, stride=stride1, name=name + '_block' + str(blocks))
    return x


def dense_block(x, blocks, name,do_norm=True):
    """A dense block.
    # Arguments
        x: input tensor.
        blocks: integer, the number of building blocks.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    x = conv_block(x, 32, name=name + '_block' + str( 1),do_norm=do_norm)
    if blocks>1:
        for i in range(1,blocks):
            x = conv_block(x, 32, name=name + '_block' + str(i + 1))
    return x

def transition_block(x, reduction, name):
    """A transition block.
    # Arguments
        x: input tensor.
        reduction: float, compression rate at transition layers.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    bn_axis = 3 if keras.backend.image_data_format() == 'channels_last' else 1
    x = keras.layers.BatchNormalization(axis=bn_axis, momentum=0.9, epsilon=1.001e-5,name=name + '_bn')(x)
    x = keras.layers.Activation('relu', name=name + '_relu')(x)
    x = keras.layers.Conv2D(int(keras.backend.int_shape(x)[bn_axis] * reduction), 1,  use_bias=False, kernel_initializer='Orthogonal',  name=name + '_conv')(x)
    x = keras.layers.AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    return x

def conv_block(x, growth_rate, name,do_norm=True):
    """A building block for a dense block.
    # Arguments
        x: input tensor.
        growth_rate: float, growth rate at dense layers.
        name: string, block label.
    # Returns
        Output tensor for the block.
    """
    bn_axis = 3 #if keras.backend.image_data_format() == 'channels_last' else 1
    if do_norm == True:
        x1 = keras.layers.BatchNormalization(axis=bn_axis,momentum=0.9, epsilon=1.001e-5, name=name + '_0_bn')(x)
        x1 = keras.layers.Activation('relu', name=name + '_0_relu')(x1)
    else:
        x1=x
    x1 = keras.layers.Conv2D(4 * growth_rate, 1,  use_bias=False,kernel_initializer='Orthogonal',   name=name + '_1_conv')(x1)
    x1 = keras.layers.BatchNormalization(axis=bn_axis,momentum=0.9, epsilon=1.001e-5,  name=name + '_1_bn')(x1)
    x1 = keras.layers.Activation('relu', name=name + '_1_relu')(x1)
    x1 = keras.layers.Conv2D(growth_rate, 3,  padding='same', use_bias=False, kernel_initializer='Orthogonal',   name=name + '_2_conv')(x1)
    x = keras.layers.Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x

def ResDenseNet(blocks=[2,2,2,2],
           model_name='ResDenseNet',
           include_top=True,
           input_shape=(32,32,3),
           pooling='avg',
           classes=1000):
    img_input = keras.layers.Input(shape=input_shape)
    bn_axis = 3 #if keras.backend.image_data_format() == 'channels_last' else 1

    x = keras.layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')(img_input)
    x = keras.layers.Conv2D(64, 7, strides=1, use_bias=False, name='conv1_conv')(x)
    x = keras.layers.BatchNormalization(axis=bn_axis, momentum=0.9, epsilon=1.001e-5, name= 'conv1/bn')(x)
    x = keras.layers.Activation('relu', name='conv1/bn1/relu')(x)

    #x1 = keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='DBlock1/zeroPad2')(x)
    x1 = keras.layers.MaxPooling2D(2, strides=1, name='DBlock1/pool',padding='same')(x)
    x1 = dense_block(x1, blocks[0], name='DBlock1/conv') ## 128
    #r1 = stack2(x, filters=32, blocks=blocks[0], stride1=1, name='ResBlock1') ## 128
    r1 = stack_basic(x, filters=32, blocks=blocks[0], stride=1)

    rx1=keras.layers.concatenate([x1,r1],name= 'RDBlock1') #256
    rx1 = transition_block(rx1, 0.5, name= 'RDBlock1/pool') #128

    x2 = dense_block(rx1, blocks[1], name= 'DBlock2/conv')#192
    r2 = stack2(rx1, filters=48, blocks=blocks[1], stride1=1, name='RBlock2') #192

    rx2 = keras.layers.concatenate([x2, r2], name='RDBlock2') #384
    rx2 = transition_block(rx2, 0.5, name='RDBlock2/pool') #192

    x3 = dense_block(rx2, blocks[2], name='DBlock3/conv') #256
    r3 = stack2(rx2, filters=72, blocks=blocks[2], stride1=1, name='RBlock3')
    #x3 = transition_block(x3, 0.5, name='DenseBlock3/pool') #128
    #r3 = stack2(rx2, filters=72, blocks=blocks[2], stride1=2, name='ResBlock3') #448
    rx3 = keras.layers.concatenate([x3, r3], name='RDBlock3/concat')
    rx3 = transition_block(rx3, 0.5, name='RDBlock3/pool')

    x4 = dense_block(rx3, blocks[3], name='DBlock4/conv') #480
    #x4 = transition_block(x4, 0.5, name='DenseBlock4/pool') #320
    r4 = stack2(rx3, filters=96, blocks=blocks[3], stride1=1, name='RBlock4')
    rx4 = keras.layers.concatenate([x4, r4], name='RDBlock4/concat')

    '''
    x3 = dense_block(rx2, blocks[2], name='DenseBlock3/conv3')
    r3 = stack2(rx2, filters=64, blocks=blocks[2], stride1=1, name='ResBlock3')
    rx3 = keras.layers.concatenate([x3, r3], name='ResDenseBlock3/pool3')
    rx3 = transition_block(rx3, 0.5, name='ResDenseBlock3/pool3')

    x4 = dense_block(rx3, blocks[3], name= 'DenseBlock4/conv4')
    r4 = stack2(rx3, filters=80, blocks=blocks[3], stride1=1, name='ResBlock4')
    rx4 = keras.layers.concatenate([x4, r4], name='ResDenseBlock4')
   '''
    output = keras.layers.BatchNormalization(axis=bn_axis, momentum=0.9, epsilon=1.001e-5, name='concat/bn')(rx4)
    output = keras.layers.Activation('relu', name='concat/bn/relu')(output)

    if pooling == 'avg':
        output = keras.layers.GlobalAveragePooling2D(name='avg_pool')(output)
    elif pooling == 'max':
        output = keras.layers.GlobalMaxPooling2D(name='max_pool')(output)
    if include_top:
        output = keras.layers.Dense(classes, activation='softmax', name= 'FC')(output)
    model = keras.models.Model(inputs=img_input, outputs=output,name=model_name)
    return model

def ResDenseNetSmall(
           include_top=True,
           input_shape=(32,32,3),
           pooling='avg',
           classes=100):
    return ResDenseNet(blocks=[2,2,2,2],
           model_name='ResDenseNetSmall',
           include_top=include_top,
           input_shape=input_shape,
           pooling=pooling,
           classes=classes)

def ResDenseNetMedium(
           include_top=True,
           input_shape=(32,32,3),
           pooling='avg',
           classes=100):
    return ResDenseNet(blocks=[3,4,6,3],
           model_name='ResDenseNetMedium',
           include_top=include_top,
           input_shape=input_shape,
           pooling=pooling,
           classes=classes)

if __name__ == "__main__":
    #blocks=[2,2,2,2],     1,524,724 30--Conv2d
    # blocks=[3,4,6,3],    2,976,004 53--Conv2d
    # blocks=[6,12,24,16],14,349,012 176-Conv2d
    # blocks=[3,4,23,3]    7,407,972 104-Conv2d
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    model=ResDenseNet(blocks=[2,2,2,2],include_top=True,input_shape=(32,32,3),pooling='avg', classes=100)
    model.summary()
    keras.utils.plot_model(model, 'ResDenseL.png', show_shapes=True)
