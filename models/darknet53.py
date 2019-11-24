"""Darknet-53 for yolo v3.
"""
#https://github.com/xiaochus/YOLOv3
import tensorflow as tf
if tf.__version__<'2.0':
    import keras
else:
    from tensorflow import keras

def conv2d_unit(x, filters, kernels, strides=1):
    """Convolution Unit
    This function defines a 2D convolution operation with BN and LeakyReLU.

    # Arguments
        x: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernels: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and
            height. Can be a single integer to specify the same value for
            all spatial dimensions.

    # Returns
            Output tensor.
    """
    x = keras.layers.Conv2D(filters, kernels,
               padding='same',
               strides=strides,
               activation='linear',
               kernel_regularizer=keras.regularizers.l2(5e-4),kernel_initializer='he_normal')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha=0.1)(x)

    return x


def residual_block(inputs, filters):
    """Residual Block
    This function defines a 2D convolution operation with BN and LeakyReLU.

    # Arguments
        x: Tensor, input tensor of residual block.
        kernels: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.

    # Returns
        Output tensor.
    """
    x = conv2d_unit(inputs, filters, (1, 1))
    x = conv2d_unit(x, 2 * filters, (3, 3))
    x = keras.layers.add([inputs, x])
    x = keras.layers.Activation('linear')(x)

    return x


def stack_residual_block(inputs, filters, n):
    """Stacked residual Block
    """
    x = residual_block(inputs, filters)
    for i in range(n - 1):
        x = residual_block(x, filters)
    return x


def darknet_base(inputs):
    """Darknet-53 base model.
    """

    x = conv2d_unit(inputs, 32, (7, 7))

    x = conv2d_unit(x, 64, (3, 3), strides=2)
    x = stack_residual_block(x, 32, n=1)

    x = conv2d_unit(x, 128, (3, 3), strides=2)
    x = stack_residual_block(x, 64, n=2)

    x = conv2d_unit(x, 256, (3, 3), strides=2)
    x = stack_residual_block(x, 128, n=8)

    x = conv2d_unit(x, 512, (3, 3), strides=2)
    x = stack_residual_block(x, 256, n=8)

    x = conv2d_unit(x, 1024, (3, 3), strides=2)
    x = stack_residual_block(x, 512, n=4)

    return x


def darknet(include_top=True,input_shape=(224,224,3),classes=100,pooling='avg'):
    """Darknet-53 classifier.
    """
    inputs = keras.layers.Input(shape=input_shape)
    x = darknet_base(inputs)
    if pooling=='avg':
        x = keras.layers.GlobalAveragePooling2D()(x)
    else:
        x=keras.layers.GlobalMaxPooling2D()(x)
    if include_top==True:
        x = keras.layers.Dense(classes, activation='softmax')(x)
    model = keras.models.Model(inputs, x)
    return model


if __name__ == '__main__':
    model = darknet(input_shape=(32,32,3))
    print(model.summary())
    keras.utils.plot_model(model, 'darknet53.png', show_shapes=True)
