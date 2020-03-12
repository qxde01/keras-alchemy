import tensorflow as tf
if tf.__version__<'2.0':
    import keras
else:
    from tensorflow import keras

# https://github.com/nibtehaz/MultiResUNet

def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), activation='relu', name=None):
    '''
    2D Convolutional layers
    
    Arguments:
        x {keras layer} -- input layer 
        filters {int} -- number of filters
        num_row {int} -- number of rows in filters
        num_col {int} -- number of columns in filters
    
    Keyword Arguments:
        padding {str} -- mode of padding (default: {'same'})
        strides {tuple} -- stride of convolution operation (default: {(1, 1)})
        activation {str} -- activation function (default: {'relu'})
        name {str} -- name of the layer (default: {None})
    
    Returns:
        [keras layer] -- [output layer]
    '''

    x = keras.layers.Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=False, kernel_initializer='he_normal')(x)
    x = keras.layers.BatchNormalization(axis=3, scale=True,momentum=0.95)(x)
    if activation == None:
        return x
    else:
        x = keras.layers.Activation(activation, name=name)(x)
        return x


def trans_conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(2, 2), name=None):
    '''
    2D Transposed Convolutional layers
    Arguments:
        x {keras layer} -- input layer 
        filters {int} -- number of filters
        num_row {int} -- number of rows in filters
        num_col {int} -- number of columns in filters
    Keyword Arguments:
        padding {str} -- mode of padding (default: {'same'})
        strides {tuple} -- stride of convolution operation (default: {(2, 2)})
        name {str} -- name of the layer (default: {None})
    Returns:
        [keras layer] -- [output layer]
    '''
    x = keras.layers.Conv2DTranspose(filters, (num_row, num_col), strides=strides, padding=padding, kernel_initializer='he_normal')(x)
    x = keras.layers.BatchNormalization(axis=3, scale=False)(x)
    return x


def MultiResBlock(U, inp, alpha=1.67):
    '''
    MultiRes Block
    
    Arguments:
        U {int} -- Number of filters in a corrsponding UNet stage
        inp {keras layer} -- input layer 
    
    Returns:
        [keras layer] -- [output layer]
    '''
    W = alpha * U
    shortcut = inp
    shortcut = conv2d_bn(shortcut, int(W * 0.167) + int(W * 0.333) + int(W * 0.5), 1, 1, activation=None, padding='same')
    conv3x3 = conv2d_bn(inp, int(W * 0.167), 3, 3, activation='relu', padding='same')
    conv5x5 = conv2d_bn(conv3x3, int(W * 0.333), 3, 3, activation='relu', padding='same')
    conv7x7 = conv2d_bn(conv5x5, int(W * 0.5), 3, 3, activation='relu', padding='same')
    out = keras.layers.concatenate([conv3x3, conv5x5, conv7x7], axis=3)
    #out = keras.layers.BatchNormalization(axis=3,momentum=0.95)(out)
    out = keras.layers.add([shortcut, out])
    out = keras.layers.BatchNormalization(axis=3, momentum=0.95)(out)
    out = keras.layers.Activation('relu')(out)

    # out=Dropout(0.5)(out)
    return out


def ResPath(filters, length, inp):
    '''
    ResPath
    
    Arguments:
        filters {int} -- [description]
        length {int} -- length of ResPath
        inp {keras layer} -- input layer 
    
    Returns:
        [keras layer] -- [output layer]
    '''
    shortcut = inp
    shortcut = conv2d_bn(shortcut, filters, 1, 1, activation=None, padding='same')
    out = conv2d_bn(inp, filters, 3, 3, activation='relu', padding='same')
    out = keras.layers.add([shortcut, out])
    out = keras.layers.BatchNormalization(axis=3)(out)
    out = keras.layers.Activation('relu')(out)

    for i in range(length - 1):
        shortcut = out
        shortcut = conv2d_bn(shortcut, filters, 1, 1, activation=None, padding='same')
        out = conv2d_bn(out, filters, 3, 3, activation='relu', padding='same')
        out = keras.layers.add([shortcut, out])
        out = keras.layers.BatchNormalization(axis=3)(out)
        out = keras.layers.Activation('relu')(out)

    return out




def MultiResNet(input_shape=(224, 224, 3), classes=1000,pooling='avg',filters=64):

    inputs = keras.layers.Input(shape=input_shape)
    mresblock1 = MultiResBlock(filters, inputs)  # 224X224X51
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(mresblock1)  # 112*112*51

    mresblock1 = ResPath(filters, 4, mresblock1)  # 224X224X32
    mresblock1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(mresblock1)  # 112, 112, 32

    mresblock2 = MultiResBlock(filters * 2, pool1)  # 112X112X64
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(mresblock2)  # 56X56X105
    mresblock2 = ResPath(filters * 2, 3, mresblock2)  # 112X112X64

    mresblock2 = keras.layers.concatenate([mresblock1, mresblock2], name='concat_m12')  # 112, 112, 96
    mresblock2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(mresblock2)  # 56, 56, 96

    mresblock3 = MultiResBlock(filters * 4, pool2)  # 56, 56, 128
    pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(mresblock3)  # 28, 28, 212
    mresblock3 = ResPath(filters * 4, 2, mresblock3)  # 56, 56, 128
    mresblock3 = keras.layers.concatenate([mresblock2, mresblock3], name='concat_m23')  # 56, 56, 224
    mresblock3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(mresblock3)  # 28, 28, 224

    mresblock4 = MultiResBlock(filters * 8, pool3)  # 28, 28, 256
    pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(mresblock4)  # 14, 14, 426
    mresblock4 = ResPath(filters * 8, 1, mresblock4)  # 28, 28, 256
    mresblock4 = keras.layers.concatenate([mresblock3, mresblock4], name='concat_m34')  # 28, 28, 480
    mresblock4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(mresblock4)  # 14, 14, 480

    #mresblock5 = MultiResBlock(filters * 16, pool4)  # 14, 14, 853
    output=MultiResBlock(filters * 16, pool4)
    #output = keras.layers.concatenate([mresblock4, mresblock5], name='concat_m45')
    if pooling=='avg':
        output = keras.layers.GlobalAveragePooling2D()(output)
    else:
        output=keras.layers.GlobalMaxPooling2D()(output)
    output = keras.layers.Dense(classes, activation='softmax')(output)
    model = keras.models.Model(inputs=inputs, outputs=output,name='MultiResUnet')
    return model


def main():
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    input_shape = (32, 32, 3)
    model = MultiResNet(input_shape=input_shape, classes=100)
    keras.utils.plot_model(model, 'MultiResnet.png', show_shapes=True)
    print(model.summary())


if __name__ == '__main__':
    #filters-64: 12,813,482
    #filters-32:3, 242, 279
    main()
