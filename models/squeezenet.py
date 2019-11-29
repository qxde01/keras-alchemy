import tensorflow as tf
if tf.__version__<'2.0':
    import keras
else:
    from tensorflow import keras

sq1x1 = "squeeze1x1"
exp1x1 = "expand1x1"
exp3x3 = "expand3x3"
relu = "relu_"
# Modular function for Fire Node
# https://github.com/rcmalli/keras-squeezenet/blob/master/keras_squeezenet/squeezenet.py
def fire_module(x, fire_id, squeeze=16, expand=64):
    s_id = 'fire' + str(fire_id) + '/'
    channel_axis = 3
    x = keras.layers.Conv2D(squeeze, (1, 1), padding='valid', use_bias=True,kernel_initializer='he_normal',name=s_id + sq1x1)(x)
    x = keras.layers.BatchNormalization(axis=3, name=s_id + sq1x1+'/bn')(x)
    x = keras.layers.Activation('relu', name=s_id + relu + sq1x1)(x)
    left = keras.layers.Conv2D(expand//2, (1, 1), padding='valid',  use_bias=True,kernel_initializer='he_normal',name=s_id + exp1x1)(x)
    left = keras.layers.BatchNormalization(axis=3, name=s_id + exp1x1 + '/bn')(left)
    left = keras.layers.Activation('relu', name=s_id + relu + exp1x1)(left)
    right = keras.layers.Conv2D(expand//2, (3, 3), padding='same', use_bias=True,kernel_initializer='he_normal', name=s_id + exp3x3)(x)
    right = keras.layers.BatchNormalization(axis=3, name=s_id + exp3x3 + '/bn')(right)
    right = keras.layers.Activation('relu', name=s_id + relu + exp3x3)(right)
    x = keras.layers.concatenate([left, right], axis=channel_axis, name=s_id + 'concat')
    #print(x)
    return x


def SqueezeNet(include_top=True,input_shape=(224,224,3), pooling='avg',classes=100):
    """Instantiates the SqueezeNet architecture.
    """
    img_input = keras.layers.Input(shape=input_shape)
    x = keras.layers.Conv2D(96, (3, 3), strides=(3, 3), padding='same',  use_bias=True,kernel_initializer='he_normal',name='conv1')(img_input) #111, 111, 64
    x=keras.layers.BatchNormalization(axis=3,name='conv1_bn')(x)
    x = keras.layers.Activation('relu', name='relu_conv1')(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(x) #55, 55, 64

    f2 = fire_module(x, fire_id=2, squeeze=16, expand=128) #55, 55, 128
    f30 = fire_module(f2, fire_id=3, squeeze=16, expand=128) #55, 55, 128
    f3= keras.layers.add([f2,f30],name='fire2_3')
    f4= fire_module(f3, fire_id=4, squeeze=32, expand=256)  # 27, 27, 256
    f4 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), name='pool3')(f4) #27, 27, 128

    f50 = fire_module(f4, fire_id=5, squeeze=32, expand=256)  # 27, 27, 256
    f5 = keras.layers.add([f50, f4], name='fire4_5')
    f6=fire_module(f5, fire_id=6, squeeze=48, expand=384)

    f70 = fire_module(f6, fire_id=7, squeeze=48, expand=384)
    f7 = keras.layers.add([f70,f6] ,name='fire5_6')   # 13, 13, 384
    f8 = fire_module(f7, fire_id=8, squeeze=64, expand=512)  # 13, 13, 512

    f8 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool8')(f8)
    #x = fire_module(x, fire_id=8, squeeze=64, expand=256) #13, 13, 512
    f9 = fire_module(f8, fire_id=9, squeeze=64, expand=512) #13, 13, 512)

    # It's not obvious where to cut the network...
    # Could do the 8th or 9th layer... some work recommends cutting earlier layers.
    #x = keras.layers.Dropout(0.5, name='drop9')(x)
    x = keras.layers.Conv2D(classes, (1, 1), padding='valid', use_bias=True,kernel_regularizer=keras.regularizers.l2(10e-5), name='conv10')(f9) # 13, 13, 1000
    #x = keras.layers.Activation('relu', name='relu_conv10')(x)
    if pooling == 'avg':
        x = keras.layers.GlobalAveragePooling2D()(x)
    if pooling == 'max':
        x = keras.layers.GlobalMaxPooling2D()(x)
    #x = layers.Flatten(name='flatten')(x)
    if include_top:
        x = keras.layers.Activation('softmax', name='softmax')(x)
    model = keras.models.Model(inputs=img_input, outputs=x, name='squeezenet')
    return model

if __name__ == "__main__":
    model=SqueezeNet()
    model.summary()
    keras.utils.plot_model(model, 'SqueezeNet.png', show_shapes=True)
