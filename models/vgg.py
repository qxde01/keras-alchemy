"""VGG19 model for Keras.

# Reference

- [Very Deep Convolutional Networks for Large-Scale Image Recognition](
    https://arxiv.org/abs/1409.1556) (ICLR 2015)

"""

import os
import tensorflow as tf
if tf.__version__<'2.0':
    import keras
else:
    from tensorflow import keras

def conv_bn( x, filters, kernel_size=3, stride=1, padding='same',name='conv'):
        '''
        Combination of Conv and BN layers since these always appear together.
        '''
        x = keras.layers.Conv2D(filters=filters, kernel_size=(kernel_size, kernel_size),
                   strides=(stride, stride), padding=padding,kernel_initializer='he_normal',name=name)(x)
        x = keras.layers.BatchNormalization(momentum=0.9,epsilon=1e-5,name=name+'_bn')(x)
        x=keras.layers.Activation('relu',name=name+'_bn_relu')(x)
        return x

def VGG19(include_top=True, input_shape=None, pooling='avg', classes=100,
          **kwargs):
    """Instantiates the VGG19 architecture.
    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)`
            (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(200, 200, 3)` would be one valid value.
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
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """

    img_input = keras.layers.Input(shape=input_shape)

    # Block 1
    x=conv_bn(img_input, 64, kernel_size=3, stride=1, padding='same', name='block1_conv1')
    x = conv_bn(x, 64, kernel_size=3, stride=1, padding='same', name='block1_conv2')
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = conv_bn(x, 128, kernel_size=3, stride=1, padding='same', name='block2_conv1')
    x = conv_bn(x, 128, kernel_size=3, stride=1, padding='same', name='block2_conv2')
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = conv_bn(x, 256, kernel_size=3, stride=1, padding='same', name='block3_conv1')
    x = conv_bn(x, 256, kernel_size=3, stride=1, padding='same', name='block3_conv2')
    x = conv_bn(x, 256, kernel_size=3, stride=1, padding='same', name='block3_conv3')
    x = conv_bn(x, 256, kernel_size=3, stride=1, padding='same', name='block3_conv4')
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = conv_bn(x, 512, kernel_size=3, stride=1, padding='same', name='block4_conv1')
    x = conv_bn(x, 512, kernel_size=3, stride=1, padding='same', name='block4_conv2')
    x = conv_bn(x, 512, kernel_size=3, stride=1, padding='same', name='block4_conv3')
    x = conv_bn(x, 512, kernel_size=3, stride=1, padding='same', name='block4_conv4')
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = conv_bn(x, 512, kernel_size=3, stride=1, padding='same', name='block5_conv1')
    x = conv_bn(x, 512, kernel_size=3, stride=1, padding='same', name='block5_conv2')
    x = conv_bn(x, 512, kernel_size=3, stride=1, padding='same', name='block5_conv3')
    x = conv_bn(x, 512, kernel_size=3, stride=1, padding='same', name='block5_conv4')
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if include_top:
        # Classification block
        x = keras.layers.Flatten(name='flatten')(x)
        x = keras.layers.Dense(4096, activation='relu', name='fc1')(x)
        x = keras.layers.Dense(4096, activation='relu', name='fc2')(x)
        x = keras.layers.Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = keras.layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = keras.layers.GlobalMaxPooling2D()(x)
    # Create model.
    model = keras.models.Model(img_input, x, name='vgg19')
    return model

def VGG16(include_top=True,
           input_shape=None,
          pooling='avg',
          classes=100,
          **kwargs):
    """Instantiates the VGG16 architecture.
    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)`
            (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 input channels,
            and width and height should be no smaller than 32.
            E.g. `(200, 200, 3)` would be one valid value.
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
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    img_input = keras.layers.Input(shape=input_shape)

    # Block 1
    x=conv_bn(img_input, 64, kernel_size=3, stride=1, padding='same', name='block1_conv1')
    x = conv_bn(x, 64, kernel_size=3, stride=1, padding='same', name='block1_conv2')
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = conv_bn(x, 128, kernel_size=3, stride=1, padding='same', name='block2_conv1')
    x = conv_bn(x, 128, kernel_size=3, stride=1, padding='same', name='block2_conv2')
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = conv_bn(x, 256, kernel_size=3, stride=1, padding='same', name='block3_conv1')
    x = conv_bn(x, 256, kernel_size=3, stride=1, padding='same', name='block3_conv2')
    x = conv_bn(x, 256, kernel_size=3, stride=1, padding='same', name='block3_conv3')
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = conv_bn(x, 512, kernel_size=3, stride=1, padding='same', name='block4_conv1')
    x = conv_bn(x, 512, kernel_size=3, stride=1, padding='same', name='block4_conv2')
    x = conv_bn(x, 512, kernel_size=3, stride=1, padding='same', name='block4_conv3')
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = conv_bn(x, 512, kernel_size=3, stride=1, padding='same', name='block5_conv1')
    x = conv_bn(x, 512, kernel_size=3, stride=1, padding='same', name='block5_conv2')
    x = conv_bn(x, 512, kernel_size=3, stride=1, padding='same', name='block5_conv3')
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    if include_top:
        # Classification block
        x = keras.layers.Flatten(name='flatten')(x)
        x = keras.layers.Dense(4096, activation='relu', name='fc1')(x)
        x = keras.layers.Dense(4096, activation='relu', name='fc2')(x)
        x = keras.layers.Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = keras.layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = keras.layers.GlobalMaxPooling2D()(x)
    # Create model.
    model = keras.models.Model(img_input, x, name='vgg16')
    return model

def VGG13(include_top=True,
           input_shape=None,
          pooling='avg',
          classes=100,
          **kwargs):

    img_input = keras.layers.Input(shape=input_shape)

    # Block 1
    x = conv_bn(img_input, 64, kernel_size=3, stride=1, padding='same', name='block1_conv1')
    x = conv_bn(x, 64, kernel_size=3, stride=1, padding='same', name='block1_conv2')
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = conv_bn(x, 128, kernel_size=3, stride=1, padding='same', name='block2_conv1')
    x = conv_bn(x, 128, kernel_size=3, stride=1, padding='same', name='block2_conv2')
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = conv_bn(x, 256, kernel_size=3, stride=1, padding='same', name='block3_conv1')
    x = conv_bn(x, 256, kernel_size=3, stride=1, padding='same', name='block3_conv2')
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = conv_bn(x, 512, kernel_size=3, stride=1, padding='same', name='block4_conv1')
    x = conv_bn(x, 512, kernel_size=3, stride=1, padding='same', name='block4_conv2')
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = conv_bn(x, 512, kernel_size=3, stride=1, padding='same', name='block5_conv1')
    x = conv_bn(x, 512, kernel_size=3, stride=1, padding='same', name='block5_conv2')
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    if include_top:
        # Classification block
        x = keras.layers.Flatten(name='flatten')(x)
        x = keras.layers.Dense(4096, activation='relu', name='fc1')(x)
        x = keras.layers.Dense(4096, activation='relu', name='fc2')(x)
        x = keras.layers.Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = keras.layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = keras.layers.GlobalMaxPooling2D()(x)
    # Create model.
    model = keras.models.Model(img_input, x, name='vgg13')
    return model

def VGG11(include_top=True,
           input_shape=None,
          pooling='avg',
          classes=100,
          **kwargs):

    img_input = keras.layers.Input(shape=input_shape)

    # Block 1
    x = conv_bn(img_input, 64, kernel_size=3, stride=1, padding='same', name='block1_conv1')
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = conv_bn(x, 128, kernel_size=3, stride=1, padding='same', name='block2_conv1')
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = conv_bn(x, 256, kernel_size=3, stride=1, padding='same', name='block3_conv1')
    x = conv_bn(x, 256, kernel_size=3, stride=1, padding='same', name='block3_conv2')
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = conv_bn(x, 512, kernel_size=3, stride=1, padding='same', name='block4_conv1')
    x = conv_bn(x, 512, kernel_size=3, stride=1, padding='same', name='block4_conv2')
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = conv_bn(x, 512, kernel_size=3, stride=1, padding='same', name='block5_conv1')
    x = conv_bn(x, 512, kernel_size=3, stride=1, padding='same', name='block5_conv2')
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    if include_top:
        # Classification block
        x = keras.layers.Flatten(name='flatten')(x)
        x = keras.layers.Dense(4096, activation='relu', name='fc1')(x)
        x = keras.layers.Dense(4096, activation='relu', name='fc2')(x)
        x = keras.layers.Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = keras.layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = keras.layers.GlobalMaxPooling2D()(x)
    # Create model.
    model = keras.models.Model(img_input, x, name='vgg11')
    return model


def VGG19Small(include_top=True, input_shape=None, pooling='avg', classes=100,
          **kwargs):
    img_input = keras.layers.Input(shape=input_shape)

    # Block 1
    x=conv_bn(img_input, 64, kernel_size=3, stride=1, padding='same', name='block1_conv1')
    x = conv_bn(x, 64, kernel_size=3, stride=1, padding='same', name='block1_conv2')
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = conv_bn(x, 96, kernel_size=3, stride=1, padding='same', name='block2_conv1')
    x = conv_bn(x, 96, kernel_size=3, stride=1, padding='same', name='block2_conv2')
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = conv_bn(x, 144, kernel_size=3, stride=1, padding='same', name='block3_conv1')
    x = conv_bn(x, 144, kernel_size=3, stride=1, padding='same', name='block3_conv2')
    x = conv_bn(x, 144, kernel_size=3, stride=1, padding='same', name='block3_conv3')
    x = conv_bn(x, 144, kernel_size=3, stride=1, padding='same', name='block3_conv4')
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = conv_bn(x, 216, kernel_size=3, stride=1, padding='same', name='block4_conv1')
    x = conv_bn(x, 216, kernel_size=3, stride=1, padding='same', name='block4_conv2')
    x = conv_bn(x, 216, kernel_size=3, stride=1, padding='same', name='block4_conv3')
    x = conv_bn(x, 216, kernel_size=3, stride=1, padding='same', name='block4_conv4')
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = conv_bn(x, 216, kernel_size=3, stride=1, padding='same', name='block5_conv1')
    x = conv_bn(x, 216, kernel_size=3, stride=1, padding='same', name='block5_conv2')
    x = conv_bn(x, 216, kernel_size=3, stride=1, padding='same', name='block5_conv3')
    x = conv_bn(x, 216, kernel_size=3, stride=1, padding='same', name='block5_conv4')
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if include_top:
        # Classification block
        x = keras.layers.Flatten(name='flatten')(x)
        x = keras.layers.Dense(864, activation='relu', name='fc1')(x)
        x = keras.layers.Dense(864, activation='relu', name='fc2')(x)
        x = keras.layers.Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = keras.layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = keras.layers.GlobalMaxPooling2D()(x)
    # Create model.
    model = keras.models.Model(img_input, x, name='vgg19')
    return model

def VGG11Small(include_top=True,
           input_shape=None,
          pooling='avg',
          classes=100,
          **kwargs):

    img_input = keras.layers.Input(shape=input_shape)

    # Block 1
    x = conv_bn(img_input, 64, kernel_size=3, stride=1, padding='same', name='block1_conv1')
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = conv_bn(x, 96, kernel_size=3, stride=1, padding='same', name='block2_conv1')
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = conv_bn(x, 144, kernel_size=3, stride=1, padding='same', name='block3_conv1')
    x = conv_bn(x, 144, kernel_size=3, stride=1, padding='same', name='block3_conv2')
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = conv_bn(x, 216, kernel_size=3, stride=1, padding='same', name='block4_conv1')
    x = conv_bn(x, 216, kernel_size=3, stride=1, padding='same', name='block4_conv2')
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = conv_bn(x, 216, kernel_size=3, stride=1, padding='same', name='block5_conv1')
    x = conv_bn(x, 216, kernel_size=3, stride=1, padding='same', name='block5_conv2')
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    if include_top:
        # Classification block
        x = keras.layers.Flatten(name='flatten')(x)
        x = keras.layers.Dense(864, activation='relu', name='fc1')(x)
        x = keras.layers.Dense(864, activation='relu', name='fc2')(x)
        x = keras.layers.Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = keras.layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = keras.layers.GlobalMaxPooling2D()(x)
    # Create model.
    model = keras.models.Model(img_input, x, name='vgg11small')
    return model

if __name__ == '__main__':
    #VGG19,39,338,660
    # VGG16,34,023,844
    # VGG13,28,709,028
    # VGG11,     28,523,748
    #VGG11Small ,2,932,996
    model = VGG11Small(input_shape=(32,32,3))
    print(model.summary())
    keras.utils.plot_model(model, 'VGG11Small.png', show_shapes=True)
