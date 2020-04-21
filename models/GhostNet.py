"""
    GhostNet for ImageNet-1K, implemented in TensorFlow.
    Original paper: 'GhostNet: More Features from Cheap Operations,' https://arxiv.org/abs/1911.11907.
"""

__all__ = ['GhostNet']

import os
import math
import tensorflow as tf
if tf.__version__<'2.0':
    import keras
else:
    from tensorflow import keras
# https://github.com/YeFeng1993/GhostNet-Keras

def conv2d_bn(x, filters, kernel_size=3, padding='same', strides=1, activation='relu', name=None):
    x = keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=False,
                            kernel_initializer='he_normal',name=name+'_conv2d')(x)
    x = keras.layers.BatchNormalization(axis=-1, scale=True, momentum=0.95,name=name+'_conv2d_bn')(x)
    if activation is not None:
        x = keras.layers.Activation(activation, name=name+'_conv2d_bn_relu')(x)
    return x

def sepconv2d_bn(x, filters, kernel_size=3, padding='same', activation='relu', name=None):
    x = keras.layers.SeparableConv2D(filters, kernel_size=kernel_size, padding=padding, use_bias=False,
                            kernel_initializer='he_normal',name=name+'_sepconv2d')(x)
    x = keras.layers.BatchNormalization(axis=-1, scale=True, momentum=0.95,name=name+'_sepconv2d_bn')(x)
    if activation is not None:
        x = keras.layers.Activation(activation, name=name+'_sepconv2d_bn_relu')(x)
    return x

def dwconv2d_bn(x,  kernel_size=3,strides=1, padding='same', activation='relu',depth_multiplier=1, name=None):
    x = keras.layers.DepthwiseConv2D(kernel_size=kernel_size,strides=strides, padding=padding,depth_multiplier=depth_multiplier, use_bias=False,name=name+'_sepconv2d')(x)
    x = keras.layers.BatchNormalization(axis=-1, scale=True, momentum=0.95,name=name+'_dwconv2d_bn')(x)
    if activation is not None:
        x = keras.layers.Activation(activation, name=name+'_dwconv2d_bn_relu')(x)
    return x

def slices(x,channel):
    y = x[:,:,:,:channel]
    return y


def GhostModule(x, outchannels, ratio, convkernel, dwkernel, padding='same', strides=1, activation=None,name=None):
    conv_out_channel = math.ceil(outchannels * 1.0 / ratio)
    #x = keras.layers.Conv2D(int(conv_out_channel), (convkernel, convkernel), strides=(strides, strides),
    #                        padding=padding, data_format=data_format, activation=activation, use_bias=use_bias)(x)
    x=conv2d_bn(x,filters=int(conv_out_channel),kernel_size=convkernel,strides=strides,padding=padding,activation=activation,name=name+'_conv')
    if (ratio == 1):
        return x

    #dw = keras.layers.DepthwiseConv2D(dwkernel, strides, padding=padding, depth_multiplier=ratio - 1,
    #                                  data_format=data_format, activation=activation, use_bias=use_bias)(x)
    dw=dwconv2d_bn(x, kernel_size=dwkernel, strides=strides, padding=padding, activation=activation, depth_multiplier=ratio - 1, name=name+'_dw')
    dw = keras.layers.Lambda(slices, arguments={'channel': int(outchannels - conv_out_channel)})(dw)
    x = keras.layers.Concatenate(axis=-1)([x, dw])
    return x



def SEModule(x,outchannels,ratio,name=None):
    x1=keras.layers.GlobalAveragePooling2D()(x)
    #squeeze = keras.layers.Reshape([-1, 1, 1, x1.get_shape[1]])(x1)
    squeeze = keras.layers.Reshape((1, 1, int(x1.shape[1])))(x1)
    fc1=conv2d_bn(squeeze,filters=int(outchannels/ratio),strides=1,padding='same',activation='relu',name=name+'_fc1')
    fc2 = conv2d_bn(fc1, filters=int(outchannels), strides=1, padding='same', activation='hard_sigmoid',name=name + '_fc2')
    scale = keras.layers.Multiply()([x, fc2])
    return scale

def GhostBottleneck(x,dwkernel,strides,exp,out,ratio,use_se,name=None):
    #x1 = keras.layers.DepthwiseConv2D(dwkernel,strides,padding='same',depth_multiplier=ratio-1,data_format='channels_last',activation=None,use_bias=False)(x)
    #x1 = keras.layers.BatchNormalization(axis=-1)(x1)
    x1=dwconv2d_bn(x, kernel_size=dwkernel, strides=strides, padding='same', activation=None, depth_multiplier=ratio - 1, name=name+'_dw')
    #x1 = keras.layers.Conv2D(out,(1,1),strides=(1,1),padding='same',data_format='channels_last', activation=None,use_bias=False)(x1)
    #x1 = keras.layers.BatchNormalization(axis=-1)(x1)
    x1=conv2d_bn(x1,out,kernel_size=1,strides=1,padding='same',activation=None,name=name+'_conv')
    y = GhostModule(x,exp,ratio,1,3,activation='relu',name=name+'_ghost1')
    #y = keras.layers.BatchNormalization(axis=-1)(y)
    #y = keras.layers.Activation('relu')(y)
    if(strides>1):
        #y = keras.layers.DepthwiseConv2D(dwkernel,strides,padding='same',depth_multiplier=ratio-1,data_format='channels_last', activation=None,use_bias=False)(y)
        #y = keras.layers.BatchNormalization(axis=-1)(y)
        #y = keras.layers.Activation('relu')(y)
        y=dwconv2d_bn(y, kernel_size=dwkernel, strides=strides, padding='same', activation='relu',depth_multiplier=ratio - 1, name=name + '_idw')
    if(use_se==True):
        y = SEModule(y,exp,ratio,name=name+'_SE')
    y = GhostModule(y,out,ratio,1,3,activation=None, name=name+'_ghost2')
    #y = keras.layers.BatchNormalization(axis=-1)(y)
    y = keras.layers.add([x1,y])
    return y

def GhostNet(include_top=True,
             input_shape=(224,224,3),
             pooling='avg',
             classes=1000):
    img_input = keras.layers.Input(shape=input_shape)

    if input_shape[0]>112:
        stride0=2
    else:
        stride0=1
    x = conv2d_bn(img_input, filters=16, kernel_size=3, strides=stride0,padding='same', name='init')

    x = GhostBottleneck(x, 3, 1, 16, 16, 2, False,name='Bottle1')
    x = GhostBottleneck(x, 3, 2, 48, 24, 2, False,name='Bottle2')
    x = GhostBottleneck(x, 3, 1, 72, 24, 2, False,name='Bottle3')
    x = GhostBottleneck(x, 5, 2, 72, 40, 2, True,name='Bottle4')
    x = GhostBottleneck(x, 5, 1, 120, 40, 2, True,name='Bottle5')
    x = GhostBottleneck(x, 3, 2, 240, 80, 2, False,name='Bottle6')
    x = GhostBottleneck(x, 3, 1, 200, 80, 2, False,name='Bottle7')
    x = GhostBottleneck(x, 3, 1, 184, 80, 2, False,name='Bottle8')
    x = GhostBottleneck(x, 3, 1, 184, 80, 2, False,name='Bottle9')
    x = GhostBottleneck(x, 3, 1, 480, 112, 2, True,name='Bottle10')
    x = GhostBottleneck(x, 3, 1, 672, 112, 2, True,name='Bottle11')
    x = GhostBottleneck(x, 5, 2, 672, 160, 2, True,name='Bottle12')
    x = GhostBottleneck(x, 5, 1, 960, 160, 2, False,name='Bottle13')
    x = GhostBottleneck(x, 5, 1, 960, 160, 2, True,name='Bottle14')
    x = GhostBottleneck(x, 5, 1, 960, 160, 2, False,name='Bottle15')
    x = GhostBottleneck(x, 5, 1, 960, 160, 2, True,name='Bottle16')
    print(x)
    x = conv2d_bn(x, filters=960, kernel_size=1, strides=1, padding='same', activation='relu',name='last2')
    print(x)
    if pooling=='avg':
        x=keras.layers.GlobalAveragePooling2D()(x)
    else:
        x=keras.layers.GlobalMaxPooling2D()(x)
    print(x)
    if include_top:
        x=keras.layers.Reshape((1,1,int(x.shape[1]) ))(x)
        #print(x)
        x = conv2d_bn(x, filters=1280, kernel_size=1, strides=1, padding='same', activation='relu', name='last1')
        #print(x)
        x=keras.layers.Conv2D(classes,kernel_size=1,strides=1,activation=None,name='last0')(x)
        #print(x)
        x=keras.layers.Activation('softmax')(x)
        #print(x)
        x = keras.layers.Flatten()(x)
    model=keras.models.Model(img_input,x)
    return model


if __name__ == "__main__":
    input_shape = (32, 32, 3)
    model = GhostNet(input_shape=input_shape, classes=100)
    model.summary()
    keras.utils.plot_model(model, 'GhostNet.png', show_shapes=True)

