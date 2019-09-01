
#https://github.com/godofpdog/MobileNetV3_keras/blob/master/src/MobileNet_V3.py
import tensorflow as tf
if tf.__version__<'2.0':
    import keras
    from keras import backend as K
else:
    from tensorflow import keras
    import tensorflow.keras.backend as K

""" Define layers block functions """
def Hswish(x):
    return x * tf.nn.relu6(x + 3) / 6

# ** update custom Activate functions
keras.utils.get_custom_objects().update({'custom_activation': keras.layers.Activation(Hswish)})

def __conv2d_block(_inputs, filters, kernel, strides, is_use_bias=False, padding='same', activation='RE', name=None):
    x = keras.layers.Conv2D(filters, kernel, strides= strides, padding=padding, use_bias=is_use_bias)(_inputs)
    x = keras.layers.BatchNormalization()(x)
    if activation == 'RE':
        x = keras.layers.ReLU(name=name)(x)
    elif activation == 'HS':
        x = keras.layers.Activation(Hswish, name=name)(x)
    else:
        raise NotImplementedError
    return x

def __depthwise_block(_inputs, kernel=(3, 3), strides=(1, 1), activation='RE', is_use_se=True, num_layers=0):
    x = keras.layers.DepthwiseConv2D(kernel_size=kernel, strides=strides, depth_multiplier=1, padding='same')(_inputs)
    x = keras.layers.BatchNormalization()(x)
    if is_use_se:
        x = __se_block(x)
    if activation == 'RE':
        x = keras.layers.ReLU()(x)
    elif activation == 'HS':
        x = keras.layers.Activation(Hswish)(x)
    else:
        raise NotImplementedError
    return x

def __global_depthwise_block(_inputs):
    print(_inputs)
    #assert _inputs._keras_shape[1] == _inputs._keras_shape[2]
    assert _inputs.shape[1] == _inputs.shape[2]
    #kernel_size = _inputs._keras_shape[1]
    kernel_size = _inputs.shape[1]
    x = keras.layers.DepthwiseConv2D((kernel_size, kernel_size), strides=(1, 1), depth_multiplier=1, padding='valid')(_inputs)
    return x

def __se_block(_inputs, ratio=4, pooling_type='avg'):
    #print('============'*10)
    #print(_inputs,_inputs.shape,_inputs.shape[-1])
    #filters = _inputs._keras_shape[-1]
    filters = _inputs.shape[-1]
    se_shape = (1, 1, filters)
    if pooling_type == 'avg':
        se = keras.layers.GlobalAveragePooling2D()(_inputs)
    elif pooling_type == 'depthwise':
        se = __global_depthwise_block(_inputs)
    else:
        raise NotImplementedError
    se = keras.layers.Reshape(se_shape)(se)
    se = keras.layers.Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = keras.layers.Dense(filters, activation='hard_sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    return keras.layers.multiply([_inputs, se])

def __bottleneck_block(_inputs, out_dim, kernel, strides, expansion_dim, is_use_bias=False, shortcut=True, is_use_se=True, activation='RE', num_layers=0, *args):
    with tf.name_scope('bottleneck_block'):
        # ** to high dim 
        bottleneck_dim = expansion_dim

        # ** pointwise conv 
        x = __conv2d_block(_inputs, bottleneck_dim, kernel=(1, 1), strides=(1, 1), is_use_bias=is_use_bias, activation=activation)

        # ** depthwise conv
        x = __depthwise_block(x, kernel=kernel, strides=strides, is_use_se=is_use_se, activation=activation, num_layers=num_layers)

        # ** pointwise conv
        x = keras.layers.Conv2D(out_dim, (1, 1), strides=(1, 1), padding='same')(x)
        x = keras.layers.BatchNormalization()(x)

        if shortcut and strides == (1, 1):
            in_dim = K.int_shape(_inputs)[-1]
            if in_dim != out_dim:
                ins = keras.layers.Conv2D(out_dim, (1, 1), strides=(1, 1), padding='same')(_inputs)
                x = keras.layers.Add()([x, ins])
            else:
                x = keras.layers.Add()([x, _inputs])
    return x

def build_mobilenet_v3(input_shape=(224,224,3), num_classes=1000, model_type='large', pooling_type='avg', include_top=True):
    # ** input layer
    inputs = keras.layers.Input(shape=input_shape)

    # ** feature extraction layers
    net = __conv2d_block(inputs, 16, kernel=(3, 3), strides=(2, 2), is_use_bias=False, padding='same', activation='HS') 

    if model_type == 'large':
        config_list = large_config_list
    elif model_type == 'small':
        config_list = small_config_list
    else:
        raise NotImplementedError
        
    for config in config_list:
        net = __bottleneck_block(net, *config)
    
    # ** final layers
    net = __conv2d_block(net, 960, kernel=(3, 3), strides=(1, 1), is_use_bias=True, padding='same', activation='HS', name='output_map')

    if pooling_type == 'avg':
        net = keras.layers.GlobalAveragePooling2D()(net)
    elif pooling_type == 'depthwise':
        net = __global_depthwise_block(net)
    else:
        raise NotImplementedError

    # ** shape=(None, channel) --> shape(1, 1, channel) 
    pooled_shape = (1, 1, net.shape[-1])

    net = keras.layers.Reshape(pooled_shape)(net)
    net = keras.layers.Conv2D(1280, (1, 1), strides=(1, 1), padding='valid', use_bias=True)(net)
    
    if include_top:
        net = keras.layers.Conv2D(num_classes, (1, 1), strides=(1, 1), padding='valid', use_bias=True)(net)
        net = keras.layers.Flatten()(net)
        net = keras.layers.Softmax()(net)

    model = keras.models.Model(inputs=inputs, outputs=net)

    return model

""" define bottleneck structure """
# ** 
# **             
global large_config_list    
global small_config_list

large_config_list = [[16,  (3, 3), (1, 1), 16,  False, False, False, 'RE',  0],
                     [24,  (3, 3), (2, 2), 64,  False, False, False, 'RE',  1],
                     [24,  (3, 3), (1, 1), 72,  False, True,  False, 'RE',  2],
                     [40,  (5, 5), (2, 2), 72,  False, False, True,  'RE',  3],
                     [40,  (5, 5), (1, 1), 120, False, True,  True,  'RE',  4],
                     [40,  (5, 5), (1, 1), 120, False, True,  True,  'RE',  5],
                     [80,  (3, 3), (2, 2), 240, False, False, False, 'HS',  6],
                     [80,  (3, 3), (1, 1), 200, False, True,  False, 'HS',  7],
                     [80,  (3, 3), (1, 1), 184, False, True,  False, 'HS',  8],
                     [80,  (3, 3), (1, 1), 184, False, True,  False, 'HS',  9],
                     [112, (3, 3), (1, 1), 480, False, False, True,  'HS', 10],
                     [112, (3, 3), (1, 1), 672, False, True,  True,  'HS', 11],
                     [160, (5, 5), (1, 1), 672, False, False, True,  'HS', 12],
                     [160, (5, 5), (2, 2), 672, False, True,  True,  'HS', 13],
                     [160, (5, 5), (1, 1), 960, False, True,  True,  'HS', 14]]

small_config_list = [[16,  (3, 3), (2, 2), 16,  False, False, True,  'RE', 0],
                     [24,  (3, 3), (2, 2), 72,  False, False, False, 'RE', 1],
                     [24,  (3, 3), (1, 1), 88,  False, True,  False, 'RE', 2],
                     [40,  (5, 5), (1, 1), 96,  False, False, True,  'HS', 3],
                     [40,  (5, 5), (1, 1), 240, False, True,  True,  'HS', 4], 
                     [40,  (5, 5), (1, 1), 240, False, True,  True,  'HS', 5],
                     [48,  (5, 5), (1, 1), 120, False, False, True,  'HS', 6],
                     [48,  (5, 5), (1, 1), 144, False, True,  True,  'HS', 7],
                     [96,  (5, 5), (2, 2), 288, False, False, True,  'HS', 8],
                     [96,  (5, 5), (1, 1), 576, False, True,  True,  'HS', 9],
                     [96,  (5, 5), (1, 1), 576, False, True,  True,  'HS', 10]]

def MobileNetV3Large(include_top=True,input_shape=(416,416,3), num_classes=10, pooling='avg'):
    return build_mobilenet_v3(input_shape=input_shape, num_classes=num_classes, model_type='large', pooling_type=pooling, include_top=include_top)

def MobileNetV3Small(include_top=True,input_shape=(224,224,3), num_classes=10, pooling='avg'):
    return build_mobilenet_v3(input_shape=input_shape, num_classes=num_classes, model_type='small', pooling_type=pooling, include_top=include_top)

""" build MobileNet V3 model """
if __name__ == '__main__':
    model = MobileNetV3Large(include_top=True,input_shape=(416,416,3), num_classes=10, pooling='avg')
    print(model.summary())
    keras.utils.plot_model(model, 'MobileNetV3Large.png', show_shapes=True)
    #print(model.layers)
