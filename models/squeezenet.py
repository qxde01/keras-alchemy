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
    x = keras.layers.Conv2D(squeeze, (1, 1), padding='valid', use_bias=False,kernel_initializer='he_normal',name=s_id + sq1x1)(x)
    x = keras.layers.Activation('relu', name=s_id + relu + sq1x1)(x)
    left = keras.layers.Conv2D(expand, (1, 1), padding='valid',  use_bias=False,kernel_initializer='he_normal',name=s_id + exp1x1)(x)
    left = keras.layers.Activation('relu', name=s_id + relu + exp1x1)(left)
    right = keras.layers.Conv2D(expand, (3, 3), padding='same', use_bias=False,kernel_initializer='he_normal', name=s_id + exp3x3)(x)
    right = keras.layers.Activation('relu', name=s_id + relu + exp3x3)(right)
    x = keras.layers.concatenate([left, right], axis=channel_axis, name=s_id + 'concat')
    return x


# Original SqueezeNet from paper.

def SqueezeNet(include_top=True,input_shape=(224,224,3), pooling='avg',classes=1000):
    """Instantiates the SqueezeNet architecture.
    """
    img_input = keras.layers.Input(shape=input_shape)
    x = keras.layers.Conv2D(96, (7, 7), strides=(2, 2), padding='same',  use_bias=False,kernel_initializer='he_normal',name='conv1')(img_input) #112, 112, 64
    x = keras.layers.Activation('relu', name='relu_conv1')(x)
    x = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x) #55, 55, 64

    x = fire_module(x, fire_id=2, squeeze=16, expand=64) #55, 55, 128
    x = fire_module(x, fire_id=3, squeeze=16, expand=64) #55, 55, 128
    x = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x) #27, 27, 128

    x = fire_module(x, fire_id=4, squeeze=32, expand=128) #27, 27, 256
    x = fire_module(x, fire_id=5, squeeze=32, expand=128) #27, 27, 256
    x = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x) #13, 13, 256

    x = fire_module(x, fire_id=6, squeeze=48, expand=192) # 13, 13, 384
    x = fire_module(x, fire_id=7, squeeze=48, expand=192) #13, 13, 384
    x = fire_module(x, fire_id=8, squeeze=64, expand=256) #13, 13, 512
    x = fire_module(x, fire_id=9, squeeze=64, expand=256) #13, 13, 512)

    # It's not obvious where to cut the network...
    # Could do the 8th or 9th layer... some work recommends cutting earlier layers.
    
    x = keras.layers.Dropout(0.5, name='drop9')(x)
    x = keras.layers.Conv2D(classes, (1, 1), padding='valid', use_bias=False,kernel_regularizer=keras.regularizers.l2(10e-5), name='conv10')(x) # 13, 13, 1000
    x = keras.layers.Activation('relu', name='relu_conv10')(x)
    if pooling == 'avg':
        x = keras.layers.GlobalAveragePooling2D()(x)
    if pooling == 'max':
        x = keras.layers.GlobalMaxPooling2D()(x)
    #x = layers.Flatten(name='flatten')(x)
    if include_top:
        x = keras.layers.Activation('softmax', name='loss')(x)
    model = keras.models.Model(inputs=img_input, outputs=x, name='squeezenet')
    return model

if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    model=SqueezeNet()
    model.summary()
    keras.utils.plot_model(model, 'png/SqueezeNet.png', show_shapes=True)
