import tensorflow as tf
if tf.__version__<'2.0':
    import keras
else:
    from tensorflow import keras

# https://github.com/vietanhdev/blazeface_keras/blob/master/keras_layers/keras_layer_BlazeFace.py
def channel_padding(x):
    """
    zero padding in an axis of channel
    """
    #keras.backend.concatenate([x, tf.zeros_like(x)], axis=-1)
    x0=keras.layers.Activation('sigmoid')(x)
    return keras.backend.concatenate([x, x0], axis=-1)


def single_blaze_block(x, filters=24, kernel_size=5, strides=1, padding='same'):
    # depth-wise separable convolution
    x_0 = keras.layers.SeparableConv2D(filters=filters,kernel_size=kernel_size, strides=strides,padding=padding, use_bias=False)(x)
    x_1 = keras.layers.BatchNormalization()(x_0) #keras.layers.BatchNormalization
    # Residual connection
    if strides == 2:
        input_channels = x.shape[-1]
        output_channels = x_1.shape[-1]
        x_ = keras.layers.MaxPooling2D()(x)
        #print('x_:', x_.shape)
        if output_channels - input_channels != 0:
            # channel padding
            x_ = keras.layers.Lambda(channel_padding)(x_)
        out = keras.layers.Add()([x_1, x_])
        #return keras.layers.Activation("relu")(out)
    else:
        out = keras.layers.Add()([x_1, x])
    out=keras.layers.Activation("relu")(out)
    print('single:',out.shape)
    return out


def double_blaze_block(x, filters_1=24, filters_2=96,kernel_size=5, strides=1, padding='same'):
    # depth-wise separable convolution, project
    print('x',strides,x.shape)
    x_0 = keras.layers.SeparableConv2D(
        filters=filters_1,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=False)(x)
    x_1 = keras.layers.BatchNormalization()(x_0)
    x_2 = keras.layers.Activation("relu")(x_1)
    # depth-wise separable convolution, expand
    x_3 = keras.layers.SeparableConv2D(
        filters=filters_2,
        kernel_size=kernel_size,
        strides=1,
        padding=padding,
        use_bias=False)(x_2)
    x_4 = keras.layers.BatchNormalization()(x_3)
    #print('x_4', x_4.shape)
    # Residual connection
    if strides == 2:
        input_channels = x.shape[-1]
        output_channels = x_4.shape[-1]
        x_ = keras.layers.MaxPooling2D()(x)
        print('x_:',x_.shape)
        if output_channels - input_channels != 0:
            # channel padding
            x_ = keras.layers.Lambda(channel_padding)(x_)
        out = keras.layers.Add()([x_4, x_])
        return keras.layers.Activation("relu")(out)
    out = keras.layers.Add()([x_4, x])
    return keras.layers.Activation("relu")(out)


def BlazeFace(include_top=True,input_shape=(32,32,3),filters=64,classes=100):
    inputs = keras.layers.Input(shape=input_shape)
    if input_shape[1]<=64:
        stride0=1
    else:
        stride0=2
    x_0 = keras.layers.Conv2D(filters=filters, kernel_size=1, strides=stride0, padding='same')(inputs)
    print(x_0,stride0)
    x_0 = keras.layers.BatchNormalization()(x_0)
    x_0 = keras.layers.Activation("relu")(x_0)
    # Single BlazeBlock phase
    x_1 = single_blaze_block(x_0,filters=filters)
    x_2 = single_blaze_block(x_1,filters=filters)
    #2 -> stride0
    x_3 = single_blaze_block(x_2, strides=2, filters=2*filters)
    x_4 = single_blaze_block(x_3, filters=2*filters)
    x_5 = single_blaze_block(x_4, filters=2*filters)
    print('x_5:', x_5.shape)
    # Double BlazeBlock phase
    # 2 -> stride0
    x_6 = double_blaze_block(x_5, strides=2,filters_1=filters,filters_2=4*filters)
    x_7 = double_blaze_block(x_6,filters_1=filters,filters_2=4*filters)
    x_8 = double_blaze_block(x_7,filters_1=filters,filters_2=4*filters)
    print('x_8:',x_8.shape)
    x_9 = double_blaze_block(x_8, strides=2,filters_1=filters,filters_2=4*filters)
    print('x_9:', x_9.shape)
    x_10 = double_blaze_block(x_9,filters_1=filters,filters_2=4*filters)
    x_11 = double_blaze_block(x_10,filters_1=filters,filters_2=4*filters)
    print('x_11:',x_11.shape)
    output = keras.layers.Flatten()(x_11)
    if include_top:
        output=keras.layers.Dense(classes,activation='softmax',name='softmax')(output)
    model = keras.models.Model(inputs=inputs, outputs=output)
    return model

if __name__ == "__main__":
    #filters-24, 212,524
    #filters-32, 298,180
    #filters-64, 715,044

    model = BlazeFace()
    model.summary()
    keras.utils.plot_model(model, 'BlazeFace1.png', show_shapes=True)
    model1 = BlazeFace(include_top=False)
    keras.utils.plot_model(model1, 'BlazeFace2.png', show_shapes=True)
