import tensorflow as tf
if tf.__version__<'2.0':
    import keras
else:
    tf.compat.v1.disable_eager_execution()
    from tensorflow import keras

## https://github.com/qubvel/residual_attention_network


def residual_block(input, input_channels=None, output_channels=None, kernel_size=(3, 3), stride=1):
    """
    full pre-activation residual block
    https://arxiv.org/pdf/1603.05027.pdf
    """
    if output_channels is None:
        output_channels = input._shape_tuple()[-1]
    if input_channels is None:
        input_channels = output_channels // 4
    strides = (stride, stride)
    x = keras.layers.BatchNormalization()(input)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Conv2D(input_channels, (1, 1))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Conv2D(input_channels, kernel_size, padding='same', strides=stride)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Conv2D(output_channels, (1, 1), padding='same')(x)
    if input_channels != output_channels or stride != 1:
        input = keras.layers.Conv2D(output_channels, (1, 1), padding='same', strides=strides)(input)
    x = keras.layers.Add()([x, input])
    return x


def attention_block(input, input_channels=None, output_channels=None, encoder_depth=1):
    """
    attention block
    https://arxiv.org/abs/1704.06904
    """
    p = 1
    t = 2
    r = 1
    print(input,input.shape)
    if input_channels is None:
        input_channels =input._shape_tuple()[-1]
    if output_channels is None:
        output_channels = input_channels

    # First Residual Block
    for i in range(p):
        input = residual_block(input)

    # Trunc Branch
    output_trunk = input
    for i in range(t):
        output_trunk = residual_block(output_trunk)

    # Soft Mask Branch

    ## encoder
    ### first down sampling
    output_soft_mask = keras.layers.MaxPool2D(padding='same')(input)  # 32x32
    for i in range(r):
        output_soft_mask = residual_block(output_soft_mask)

    skip_connections = []
    for i in range(encoder_depth - 1):

        ## skip connections
        output_skip_connection = residual_block(output_soft_mask)
        skip_connections.append(output_skip_connection)
        # print ('skip shape:', output_skip_connection.get_shape())

        ## down sampling
        output_soft_mask = keras.layers.MaxPool2D(padding='same')(output_soft_mask)
        for _ in range(r):
            output_soft_mask = residual_block(output_soft_mask)

            ## decoder
    skip_connections = list(reversed(skip_connections))
    for i in range(encoder_depth - 1):
        ## upsampling
        for _ in range(r):
            output_soft_mask = residual_block(output_soft_mask)
        output_soft_mask = keras.layers.UpSampling2D()(output_soft_mask)
        ## skip connections
        output_soft_mask = keras.layers.Add()([output_soft_mask, skip_connections[i]])

    ### last upsampling
    for i in range(r):
        output_soft_mask = residual_block(output_soft_mask)
    output_soft_mask = keras.layers.UpSampling2D()(output_soft_mask)

    ## Output
    output_soft_mask = keras.layers.Conv2D(input_channels, (1, 1))(output_soft_mask)
    output_soft_mask = keras.layers.Conv2D(input_channels, (1, 1))(output_soft_mask)
    output_soft_mask = keras.layers.Activation('sigmoid')(output_soft_mask)

    # Attention: (1 + output_soft_mask) * output_trunk
    output = keras.layers.Lambda(lambda x: x + 1)(output_soft_mask)
    output = keras.layers.Multiply()([output, output_trunk])  #

    # Last Residual Block
    for i in range(p):
        output = residual_block(output)

    return output

def AttentionResNet92(include_top=True,input_shape=(224, 224, 3), n_channels=64, n_classes=100,
                      dropout=0, regularization=0.01):
    """
    Attention-92 ResNet
    https://arxiv.org/abs/1704.06904
    """
    regularizer = keras.regularizers.l2(regularization)

    input_ = keras.layers.Input(shape=input_shape)
    x = keras.layers.Conv2D(n_channels, (7, 7), strides=(2, 2), padding='same')(input_) # 112x112
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)  # 56x56

    x = residual_block(x, output_channels=n_channels * 4)  # 56x56
    x = attention_block(x, encoder_depth=3)  # bottleneck 7x7

    x = residual_block(x, output_channels=n_channels * 8, stride=2)  # 28x28
    x = attention_block(x, encoder_depth=2)  # bottleneck 7x7
    x = attention_block(x, encoder_depth=2)  # bottleneck 7x7

    x = residual_block(x, output_channels=n_channels * 16, stride=2)  # 14x14
    x = attention_block(x, encoder_depth=1)  # bottleneck 7x7
    x = attention_block(x, encoder_depth=1)  # bottleneck 7x7
    x = attention_block(x, encoder_depth=1)  # bottleneck 7x7

    x = residual_block(x, output_channels=n_channels * 32, stride=2)  # 7x7
    x = residual_block(x, output_channels=n_channels * 32)
    x = residual_block(x, output_channels=n_channels * 32)

    pool_size = (x.get_shape()[1].value, x.get_shape()[2].value)
    x = keras.layers.AveragePooling2D(pool_size=pool_size, strides=(1, 1))(x)
    x = keras.layers.Flatten()(x)
    if dropout:
        x = keras.layers.Dropout(dropout)(x)
    if include_top:
        output = keras.layers.Dense(n_classes, kernel_regularizer=regularizer, activation='softmax')(x)
    else:
        output=x
    model = keras.models.Model(input_, output)
    return model


def AttentionResNet56(include_top=True,input_shape=(224, 224, 3), n_channels=64, n_classes=100,
                      dropout=0, regularization=0.01):
    """
    Attention-56 ResNet
    https://arxiv.org/abs/1704.06904
    """

    regularizer = keras.regularizers.l2(regularization)

    input_ = keras.layers.Input(shape=input_shape)
    x = keras.layers.Conv2D(n_channels, (7, 7), strides=(2, 2), padding='same')(input_) # 112x112
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)  # 56x56

    x = residual_block(x, output_channels=n_channels * 4)  # 56x56
    x = attention_block(x, encoder_depth=3)  # bottleneck 7x7

    x = residual_block(x, output_channels=n_channels * 8, stride=2)  # 28x28
    x = attention_block(x, encoder_depth=2)  # bottleneck 7x7

    x = residual_block(x, output_channels=n_channels * 16, stride=2)  # 14x14
    x = attention_block(x, encoder_depth=1)  # bottleneck 7x7

    x = residual_block(x, output_channels=n_channels * 32, stride=2)  # 7x7
    x = residual_block(x, output_channels=n_channels * 32)
    x = residual_block(x, output_channels=n_channels * 32)

    pool_size = (x._shape_tuple()[1], x._shape_tuple()[2])
    x = keras.layers.AveragePooling2D(pool_size=pool_size, strides=(1, 1))(x)
    x = keras.layers.Flatten()(x)
    if dropout:
        x = keras.layers.Dropout(dropout)(x)
    if include_top:
        output = keras.layers.Dense(n_classes, kernel_regularizer=regularizer, activation='softmax')(x)
    else:
        output=x
    model = keras.models.Model(input_, output)
    return model

if __name__ == "__main__":

    model=AttentionResNet56(include_top=True,input_shape=(224,224,3),n_channels=32,n_classes=100)
    model.summary()
    keras.utils.plot_model(model, 'AttentionResNet56.png', show_shapes=True)
