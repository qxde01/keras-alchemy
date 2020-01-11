"""ResNet, ResNetV2, and ResNeXt models for Keras.

# Reference papers

- [Deep Residual Learning for Image Recognition]
  (https://arxiv.org/abs/1512.03385) (CVPR 2016 Best Paper Award)
- [Identity Mappings in Deep Residual Networks]
  (https://arxiv.org/abs/1603.05027) (ECCV 2016)
- [Aggregated Residual Transformations for Deep Neural Networks]
  (https://arxiv.org/abs/1611.05431) (CVPR 2017)

# Reference implementations

- [TensorNets]
  (https://github.com/taehoonlee/tensornets/blob/master/tensornets/resnets.py)
- [Caffe ResNet]
  (https://github.com/KaimingHe/deep-residual-networks/tree/master/prototxt)
- [Torch ResNetV2]
  (https://github.com/facebook/fb.resnet.torch/blob/master/models/preresnet.lua)
- [Torch ResNeXt]
  (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)

"""
from __future__ import absolute_import,division,print_function
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
        print(block,stride,y)
        shortcut = keras.layers.Conv2D(filters, (1, 1), strides=stride, use_bias=False, name="res{}{}_branch1".format(stage_char, block_char))(x)
        shortcut = keras.layers.BatchNormalization(axis=axis, epsilon=1e-5,  name="bn{}{}_branch1".format(stage_char, block_char))(shortcut)
    else:
        shortcut = x
    print('shotcut:',shortcut)
    y = keras.layers.Add(name="res{}{}".format(stage_char, block_char))([y, shortcut])
    y = keras.layers.Activation("relu", name="res{}{}_relu".format(stage_char, block_char))(y)
    return y




def block1(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
    """A residual block.

    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer.
        kernel_size: default 3, kernel size of the bottleneck layer.
        stride: default 1, stride of the first layer.
        conv_shortcut: default True, use convolution shortcut if True,
            otherwise identity shortcut.
        name: string, block label.

    # Returns
        Output tensor for the residual block.
    """
    bn_axis = 3 if keras.backend.image_data_format() == 'channels_last' else 1

    if conv_shortcut is True:
        shortcut = keras.layers.Conv2D(4 * filters, 1, strides=stride,name=name + '_0_conv')(x)
        shortcut = keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    x = keras.layers.Conv2D(filters, 1, strides=stride, name=name + '_1_conv')(x)
    x = keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,name=name + '_1_bn')(x)
    x = keras.layers.Activation('relu', name=name + '_1_relu')(x)

    x = keras.layers.Conv2D(filters, kernel_size, padding='SAME',name=name + '_2_conv')(x)
    x = keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,name=name + '_2_bn')(x)
    x = keras.layers.Activation('relu', name=name + '_2_relu')(x)

    x = keras.layers.Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
    x = keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,name=name + '_3_bn')(x)

    x = keras.layers.Add(name=name + '_add')([shortcut, x])
    x = keras.layers.Activation('relu', name=name + '_out')(x)
    return x


def stack1(x, filters, blocks, stride1=2, name=None):
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
    x = block1(x, filters, stride=stride1, name=name + '_block1')
    for i in range(2, blocks + 1):
        x = block1(x, filters, conv_shortcut=False, name=name + '_block' + str(i))
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
        shortcut = keras.layers.Conv2D(4 * filters, 1, strides=stride,
                                 name=name + '_0_conv')(preact)
    else:
        shortcut = keras.layers.MaxPooling2D(1, strides=stride)(x) if stride > 1 else x

    x = keras.layers.Conv2D(filters, 1, strides=1, use_bias=False,
                      name=name + '_1_conv')(preact)
    x = keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_1_bn')(x)
    x = keras.layers.Activation('relu', name=name + '_1_relu')(x)

    x = keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + '_2_pad')(x)
    x = keras.layers.Conv2D(filters, kernel_size, strides=stride,
                      use_bias=False, name=name + '_2_conv')(x)
    x = keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_2_bn')(x)
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
    x = block2(x, filters, conv_shortcut=True, name=name + '_block1')
    for i in range(2, blocks):
        x = block2(x, filters, name=name + '_block' + str(i))
    x = block2(x, filters, stride=stride1, name=name + '_block' + str(blocks))
    return x


def block3(x, filters, kernel_size=3, stride=1, groups=32,  conv_shortcut=True, name=None):
    """A residual block.

    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer.
        kernel_size: default 3, kernel size of the bottleneck layer.
        stride: default 1, stride of the first layer.
        groups: default 32, group size for grouped convolution.
        conv_shortcut: default True, use convolution shortcut if True,
            otherwise identity shortcut.
        name: string, block label.

    # Returns
        Output tensor for the residual block.
    """
    bn_axis = 3 if keras.backend.image_data_format() == 'channels_last' else 1

    if conv_shortcut is True:
        shortcut = keras.layers.Conv2D((64 // groups) * filters, 1, strides=stride,
                                 use_bias=False, name=name + '_0_conv')(x)
        shortcut = keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                             name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    x = keras.layers.Conv2D(filters, 1, use_bias=False, name=name + '_1_conv')(x)
    x = keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_1_bn')(x)
    x = keras.layers.Activation('relu', name=name + '_1_relu')(x)

    c = filters // groups
    x = keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + '_2_pad')(x)
    x = keras.layers.DepthwiseConv2D(kernel_size, strides=stride, depth_multiplier=c,
                               use_bias=False, name=name + '_2_conv')(x)
    x_shape = keras.backend.int_shape(x)[1:-1]
    x = keras.layers.Reshape(x_shape + (groups, c, c))(x)
    output_shape = x_shape + (groups, c) if keras.backend.backend() == 'theano' else None
    x = keras.layers.Lambda(lambda x: sum([x[:, :, :, :, i] for i in range(c)]),
                      output_shape=output_shape, name=name + '_2_reduce')(x)
    x = keras.layers.Reshape(x_shape + (filters,))(x)
    x = keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_2_bn')(x)
    x = keras.layers.Activation('relu', name=name + '_2_relu')(x)

    x = keras.layers.Conv2D((64 // groups) * filters, 1,use_bias=False, name=name + '_3_conv')(x)
    x = keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_3_bn')(x)

    x = keras.layers.Add(name=name + '_add')([shortcut, x])
    x = keras.layers.Activation('relu', name=name + '_out')(x)
    return x


def stack3(x, filters, blocks, stride1=2, groups=32, name=None):
    """A set of stacked residual blocks.

    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer in a block.
        blocks: integer, blocks in the stacked blocks.
        stride1: default 2, stride of the first layer in the first block.
        groups: default 32, group size for grouped convolution.
        name: string, stack label.

    # Returns
        Output tensor for the stacked blocks.
    """
    x = block3(x, filters, stride=stride1, groups=groups, name=name + '_block1')
    for i in range(2, blocks + 1):
        x = block3(x, filters, groups=groups, conv_shortcut=False, name=name + '_block' + str(i))
    return x


def ResNet(stack_fn,
           preact,
           use_bias,
           model_name='resnet',
           include_top=True,
           input_shape=None,
           pooling=None,
           classes=1000):
    """Instantiates the ResNet, ResNetV2, and ResNeXt architecture.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    # Arguments
        stack_fn: a function that returns output tensor for the
            stacked residual blocks.
        preact: whether to use pre-activation or not
            (True for ResNetV2, False for ResNet and ResNeXt).
        use_bias: whether to use biases for convolutional layers or not
            (True for ResNet and ResNetV2, False for ResNeXt).
        model_name: string, model name.
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
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
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

    x = keras.layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')(img_input)
    x = keras.layers.Conv2D(64, 7, strides=2, use_bias=use_bias, name='conv1_conv')(x)

    if preact is False:
        x = keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,name='conv1_bn')(x)
        x = keras.layers.Activation('relu', name='conv1_relu')(x)

    x = keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
    x = keras.layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)

    x = stack_fn(x)

    if preact is True:
        x = keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,name='post_bn')(x)
        x = keras.layers.Activation('relu', name='post_relu')(x)

    if include_top:
        x = keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = keras.layers.Dense(classes, activation='softmax', name='probs')(x)
    else:
        if pooling == 'avg':
            x = keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = keras.layers.GlobalMaxPooling2D(name='max_pool')(x)
    # Create model.
    model = keras.models.Model(img_input, x, name=model_name)
    return model


def ResNet50(include_top=True,input_shape=None,pooling=None,classes=1000):
    def stack_fn(x):
        x = stack1(x, 64, 3, stride1=1, name='conv2')
        x = stack1(x, 128, 4, name='conv3')
        x = stack1(x, 256, 6, name='conv4')
        x = stack1(x, 512, 3, name='conv5')
        return x
    return ResNet(stack_fn, False, True, 'resnet50',
                  include_top, input_shape,pooling, classes)


def ResNet101(include_top=True,input_shape=None,pooling=None, classes=1000):
    def stack_fn(x):
        x = stack1(x, 64, 3, stride1=1, name='conv2')
        x = stack1(x, 128, 4, name='conv3')
        x = stack1(x, 256, 23, name='conv4')
        x = stack1(x, 512, 3, name='conv5')
        return x
    return ResNet(stack_fn, False, True, 'resnet101',include_top,input_shape,pooling, classes)


def ResNet152(include_top=True,
              input_shape=None,
              pooling=None,
              classes=1000):
    def stack_fn(x):
        x = stack1(x, 64, 3, stride1=1, name='conv2')
        x = stack1(x, 128, 8, name='conv3')
        x = stack1(x, 256, 36, name='conv4')
        x = stack1(x, 512, 3, name='conv5')
        return x
    return ResNet(stack_fn, False, True, 'resnet152',
                  include_top,input_shape,pooling, classes)


def ResNet50V2(include_top=True,
               input_shape=None,
               pooling=None,
               classes=1000):
    def stack_fn(x):
        x = stack2(x, 64, 3,stride1=1, name='conv2')
        x = stack2(x, 128, 4, name='conv3')
        x = stack2(x, 256, 6, name='conv4')
        x = stack2(x, 512, 3, stride1=1, name='conv5')
        return x
    return ResNet(stack_fn, True, True, 'resnet50v2',
                  include_top, input_shape,pooling, classes)


def ResNet101V2(include_top=True,
                input_shape=None,
                pooling=None,
                classes=1000):
    def stack_fn(x):
        x = stack2(x, 64, 3,stride1=1, name='conv2')
        x = stack2(x, 128, 4, name='conv3')
        x = stack2(x, 256, 23, name='conv4')
        x = stack2(x, 512, 3, stride1=1, name='conv5')
        return x
    return ResNet(stack_fn, True, True, 'resnet101v2',
                  include_top, input_shape, pooling, classes)


def ResNet152V2(include_top=True,
                input_shape=None,
                pooling=None,
                classes=1000):
    def stack_fn(x):
        x = stack2(x, 64, 3,stride1=1, name='conv2')
        x = stack2(x, 128, 8, name='conv3')
        x = stack2(x, 256, 36, name='conv4')
        x = stack2(x, 512, 3, stride1=1, name='conv5')
        return x
    return ResNet(stack_fn, True, True, 'resnet152v2',
                  include_top, input_shape,pooling, classes)


def ResNeXt50(include_top=True,
              input_shape=None,
              pooling=None,
              classes=1000):
    def stack_fn(x):
        x = stack3(x, 128, 3, stride1=1, name='conv2')
        x = stack3(x, 256, 4, name='conv3')
        x = stack3(x, 512, 6, name='conv4')
        x = stack3(x, 1024, 3, name='conv5')
        return x
    return ResNet(stack_fn, False, False, 'resnext50',
                  include_top,input_shape,pooling, classes)


def ResNeXt101(include_top=True,input_shape=None,pooling=None, classes=1000):
    def stack_fn(x):
        x = stack3(x, 128, 3, stride1=1, name='conv2')
        x = stack3(x, 256, 4, name='conv3')
        x = stack3(x, 512, 23, name='conv4')
        x = stack3(x, 1024, 3, name='conv5')
        return x
    return ResNet(stack_fn, False, False, 'resnext101',include_top, input_shape, pooling, classes)


def ResNet18(include_top=True,input_shape=None,pooling='avg', classes=1000):
    img_input = keras.layers.Input(shape=input_shape)
    x = keras.layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')(img_input)
    x = keras.layers.Conv2D(64, 7, strides=1, use_bias=False, name='conv1_conv')(x)
    print('block1')
    x=basic_block(x,64,block=1,kernel_size=3,stage=1)
    x = basic_block(x, 64, block=2, kernel_size=3,stage=1)
    print('block2')
    x = basic_block(x, 128, block=0, kernel_size=3,stage=2,stride=1)
    x = basic_block(x, 128, block=1, kernel_size=3,stage=2)
    print('block3')
    x = basic_block(x, 256, block=0, kernel_size=3,stage=3)
    x = basic_block(x, 256, block=1, kernel_size=3,stage=3)
    print('block4')
    x = basic_block(x, 512, block=0, kernel_size=3,stage=4)
    x = basic_block(x, 512, block=1, kernel_size=3,stage=4)
    if pooling == 'avg':
        x = keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
    elif pooling == 'max':
        x = keras.layers.GlobalMaxPooling2D(name='max_pool')(x)
    if include_top:
        x=keras.layers.Dense(classes,activation='softmax',name='fc')(x)
    model=keras.models.Model(img_input,x,name='ResNet18')
    return model

def ResNet34(include_top=True,input_shape=None,pooling='avg', classes=1000):
    img_input = keras.layers.Input(shape=input_shape)
    x = keras.layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')(img_input)
    x = keras.layers.Conv2D(64, 7, strides=2, use_bias=False, name='conv1_conv')(x)
    # blocks=3,4,6,3
    print('block1')
    x = basic_block(x, 64, block=1, kernel_size=3, stage=1)
    x = basic_block(x, 64, block=2, kernel_size=3, stage=1)
    x = basic_block(x, 64, block=3, kernel_size=3, stage=1)
    print('block2')
    x = basic_block(x, 128, block=0, kernel_size=3, stage=2,stride=1)
    x = basic_block(x, 128, block=1, kernel_size=3, stage=2)
    x = basic_block(x, 128, block=2, kernel_size=3, stage=2)
    x = basic_block(x, 128, block=3, kernel_size=3, stage=2)
    print('block3')
    x = basic_block(x, 256, block=0, kernel_size=3, stage=3)
    x = basic_block(x, 256, block=1, kernel_size=3, stage=3)
    x = basic_block(x, 256, block=2, kernel_size=3, stage=3)
    x = basic_block(x, 256, block=3, kernel_size=3, stage=3)
    x = basic_block(x, 256, block=4, kernel_size=3, stage=3)
    x = basic_block(x, 256, block=5, kernel_size=3, stage=3)
    print('block4')
    x = basic_block(x, 512, block=0, kernel_size=3, stage=4)
    x = basic_block(x, 512, block=1, kernel_size=3, stage=4)
    x = basic_block(x, 512, block=2, kernel_size=3, stage=4)

    if pooling == 'avg':
        x = keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
    elif pooling == 'max':
        x = keras.layers.GlobalMaxPooling2D(name='max_pool')(x)
    if include_top:
        x = keras.layers.Dense(classes, activation='softmax', name='fc')(x)
    model = keras.models.Model(img_input, x, name='ResNet18')

    return model



setattr(ResNet18, '__doc__', ResNet.__doc__)
setattr(ResNet34, '__doc__', ResNet.__doc__)
setattr(ResNet50, '__doc__', ResNet.__doc__)
setattr(ResNet101, '__doc__', ResNet.__doc__)
setattr(ResNet152, '__doc__', ResNet.__doc__)
#setattr(ResNet18V2, '__doc__', ResNet.__doc__)
#setattr(ResNet34V2, '__doc__', ResNet.__doc__)
setattr(ResNet50V2, '__doc__', ResNet.__doc__)
setattr(ResNet101V2, '__doc__', ResNet.__doc__)
setattr(ResNet152V2, '__doc__', ResNet.__doc__)
setattr(ResNeXt50, '__doc__', ResNet.__doc__)
setattr(ResNeXt101, '__doc__', ResNet.__doc__)

if __name__ == "__main__":
    #ResNet18  11,237,156
    #ResNet34 ,21,352,740
    #ResNet50.,23,792,612
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    model=ResNet34(include_top=True,input_shape=(32,32,3),pooling='avg', classes=100)
    model.summary()
    #keras.utils.plot_model(model, 'ResNet34.png', show_shapes=True)
