# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
if tf.__version__<'2.0':
    import keras
else:
    from tensorflow import keras

#https://github.com/opconty/keras-shufflenetV2/blob/master/shufflenetv2.py
def dense_block(x, blocks, name,do_norm=True):
    """A dense block.
    # Arguments
        x: input tensor.
        blocks: integer, the number of building blocks.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    x = conv_block(x, 32, name=name + '_block' + str( 1),do_norm=do_norm)
    if blocks>1:
        for i in range(1,blocks):
            x = conv_block(x, 32, name=name + '_block' + str(i + 1))
    return x

def transition_block(x, reduction, name):
    """A transition block.
    # Arguments
        x: input tensor.
        reduction: float, compression rate at transition layers.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    bn_axis = 3 if keras.backend.image_data_format() == 'channels_last' else 1
    x = keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,name=name + '_bn')(x)
    x = keras.layers.Activation('relu', name=name + '_relu')(x)
    x = keras.layers.Conv2D(int(keras.backend.int_shape(x)[bn_axis] * reduction), 1,  use_bias=False, kernel_initializer='he_normal',  name=name + '_conv')(x)
    x = keras.layers.AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    return x

def conv_block(x, growth_rate, name,do_norm=True):
    """A building block for a dense block.
    # Arguments
        x: input tensor.
        growth_rate: float, growth rate at dense layers.
        name: string, block label.
    # Returns
        Output tensor for the block.
    """
    bn_axis = 3 if keras.backend.image_data_format() == 'channels_last' else 1
    if do_norm == True:
        x1 = keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(x)
        x1 = keras.layers.Activation('relu', name=name + '_0_relu')(x1)
    else:
        x1=x
    x1 = keras.layers.Conv2D(4 * growth_rate, 1,  use_bias=False,kernel_initializer='he_normal',   name=name + '_1_conv')(x1)
    x1 = keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,  name=name + '_1_bn')(x1)
    x1 = keras.layers.Activation('relu', name=name + '_1_relu')(x1)
    x1 = keras.layers.Conv2D(growth_rate, 3,  padding='same', use_bias=False, kernel_initializer='he_normal',   name=name + '_2_conv')(x1)
    x = keras.layers.Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x


def channel_split(x, name=''):
    # equipartition
    in_channles = x.shape.as_list()[-1]
    ip = in_channles // 2
    c_hat = keras.layers.Lambda(lambda z: z[:, :, :, 0:ip], name='%s/sp%d_slice' % (name, 0))(x)
    c = keras.layers.Lambda(lambda z: z[:, :, :, ip:], name='%s/sp%d_slice' % (name, 1))(x)
    return c_hat, c

def channel_shuffle(x):
    height, width, channels = x.shape.as_list()[1:]
    channels_per_split = channels // 2
    x = keras.backend.reshape(x, [-1, height, width, 2, channels_per_split])
    x = keras.backend.permute_dimensions(x, (0,1,2,4,3))
    x = keras.backend.reshape(x, [-1, height, width, channels])
    return x


def shuffle_unit(inputs, out_channels, bottleneck_ratio,strides=2,stage=1,block=1):
    if keras.backend.image_data_format() == 'channels_last':
        bn_axis = -1
    else:
        raise ValueError('Only channels last supported')
    prefix = 'stage{}/block{}'.format(stage, block)
    bottleneck_channels = int(out_channels * bottleneck_ratio)
    if strides < 2:
        c_hat, c = channel_split(inputs, '{}/spl'.format(prefix))
        inputs = c
    x = keras.layers.Conv2D(bottleneck_channels, kernel_size=(1,1), strides=1, padding='same',kernel_initializer='he_normal',  name='{}/1x1conv_1'.format(prefix))(inputs)
    x =  keras.layers.BatchNormalization(axis=bn_axis, name='{}/bn_1x1conv_1'.format(prefix))(x)
    x =  keras.layers.Activation('relu', name='{}/relu_1x1conv_1'.format(prefix))(x)
    x =  keras.layers.DepthwiseConv2D(kernel_size=3, strides=strides, padding='same', name='{}/3x3dwconv'.format(prefix))(x)
    x =  keras.layers.BatchNormalization(axis=bn_axis, name='{}/bn_3x3dwconv'.format(prefix))(x)
    x =  keras.layers.Conv2D(bottleneck_channels, kernel_size=1,strides=1,padding='same',kernel_initializer='he_normal',  name='{}/1x1conv_2'.format(prefix))(x)
    x =  keras.layers.BatchNormalization(axis=bn_axis, name='{}/bn_1x1conv_2'.format(prefix))(x)
    x =  keras.layers.Activation('relu', name='{}/relu_1x1conv_2'.format(prefix))(x)
    if strides < 2:
        ret =  keras.layers.Concatenate(axis=bn_axis, name='{}/concat_1'.format(prefix))([x, c_hat])
    else:
        s2 =  keras.layers.DepthwiseConv2D(kernel_size=3, strides=2, padding='same', name='{}/3x3dwconv_2'.format(prefix))(inputs)
        s2 =  keras.layers.BatchNormalization(axis=bn_axis, name='{}/bn_3x3dwconv_2'.format(prefix))(s2)
        s2 = keras.layers.Conv2D(bottleneck_channels, kernel_size=1,strides=1,kernel_initializer='he_normal', padding='same', name='{}/1x1_conv_3'.format(prefix))(s2)
        s2 =  keras.layers.BatchNormalization(axis=bn_axis, name='{}/bn_1x1conv_3'.format(prefix))(s2)
        s2 =  keras.layers.Activation('relu', name='{}/relu_1x1conv_3'.format(prefix))(s2)
        ret = keras.layers.Concatenate(axis=bn_axis, name='{}/concat_2'.format(prefix))([x, s2])
    ret =  keras.layers.Lambda(channel_shuffle, name='{}/channel_shuffle'.format(prefix))(ret)
    return ret


def block(x, channel_map, bottleneck_ratio, repeat=1, stage=1):
    x = shuffle_unit(x, out_channels=channel_map[stage-1], strides=2,bottleneck_ratio=bottleneck_ratio,stage=stage,block=1)
    for i in range(1, repeat+1):
        print(channel_map[stage-1])
        x = shuffle_unit(x, out_channels=channel_map[stage-1],strides=1, bottleneck_ratio=bottleneck_ratio,stage=stage, block=(1+i))
    return x


def DenseShuffleV1(include_top=True,blocks=[2,2,2],input_shape=(160,160,3),num_shuffle_units=[3, 7, 3]
                ,scale_factor = 1.0,bottleneck_ratio = 1,pooling='avg',name='rgb',classes=1108,dropout_rate=0.5):
    #out_dim_stage_two = {0.5: 48, 1: 116, 1.5: 176, 2: 244}
    #out_dim_stage_two = {0.5: 48, 1: 64, 1.5: 128, 2: 192}
    out_dim_stage_two = {0.5: 48, 1: 32, 1.5: 64, 2: 128}
    print(out_dim_stage_two)
    bn_axis = 3  if keras.backend.image_data_format() == 'channels_last' else 1
    img_input = keras.layers.Input(shape=input_shape, name=name + '/input')
    x = keras.layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name=name + '/zeroPad1')(img_input)
    x0 = keras.layers.Conv2D(64, 7, strides=2, use_bias=False,kernel_initializer='he_normal',  name=name + '/conv1/conv')(x)  # 80, 80, 64
    x = keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '/conv1/bn')(x0)
    x = keras.layers.Activation('relu', name=name + '/conv1/relu')(x)
    x = keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + '/zeroPad2')(x)  # 82, 82, 64
    x = keras.layers.MaxPooling2D(3, strides=2, name=name + '/pool1')(x)  # 40, 40, 64
    x = dense_block(x, blocks[0], name=name + '/conv2')  # 40, 40, 128
    x = transition_block(x, 0.5, name=name + '/pool2')  # 20, 20, 64
    exp = np.insert(np.arange(len(num_shuffle_units), dtype=np.float32), 0, 0)  # [0., 0., 1., 2.]
    out_channels_in_stage = 2 ** exp  # 1 1 2 4
    out_channels_in_stage *= out_dim_stage_two[bottleneck_ratio]  # calculate output channels for each stage 116., 116., 232., 464.
    out_channels_in_stage[0] = 24  # first stage has always 24 output channels
    out_channels_in_stage *= scale_factor  # 24., 116., 232., 464
    out_channels_in_stage = out_channels_in_stage.astype(int)
    print(out_channels_in_stage)
    z1 = keras.layers.Conv2D(filters=out_channels_in_stage[0], kernel_size=(3, 3), padding='same', use_bias=False, strides=(2, 2),kernel_initializer='he_normal',  activation='relu', name='conv1')(img_input)  # 80, 80, 24
    z1 = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='maxpool1')(z1)  # 40, 40, 24
    z1 = block(z1, out_channels_in_stage, repeat=num_shuffle_units[0], bottleneck_ratio=bottleneck_ratio, stage=0 + 2)  # 20, 20, 232)
    z2 = keras.layers.concatenate([z1, x])
    z2 = keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '/concat1/bn')(z2)
    z2 = keras.layers.Dropout(0.5)(z2)
    z21 = dense_block(z2, blocks[1], name=name + '/conv3',do_norm=False)  # 20, 20, 360
    z21 = transition_block(z21, 0.5, name=name + '/pool3')  # 10, 10, 180
    z3 = block(z2, out_channels_in_stage, repeat=num_shuffle_units[1], bottleneck_ratio=bottleneck_ratio, stage=1 + 2)  # 10, 10, 464
    z4 = keras.layers.concatenate([z21, z3])  # 10, 10, 644
    z4 = keras.layers.Dropout(dropout_rate)(z4)
    z4 = keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '/concat2/bn')(z4)

    z41 = dense_block(z4, blocks[2], name=name + '/conv4',do_norm=False)
    z41 = transition_block(z41, 0.5, name=name + '/pool4')  # 5, 5, 354
    z5 = block(z4, out_channels_in_stage, repeat=num_shuffle_units[2], bottleneck_ratio=bottleneck_ratio, stage=2 + 2)  # 5, 5, 928)
    z6 = keras.layers.concatenate([z5, z41])  # 5, 5, 1282
    z6 = keras.layers.Dropout(dropout_rate)(z6)
    z6 = keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '/concat3/bn')(z6)
    if bottleneck_ratio < 2:
        #k = 1024
        k=512
    else:
        k = 2048
    output = keras.layers.Conv2D(k, kernel_size=1, padding='same', strides=1, name='1x1conv5_out', kernel_initializer='he_normal', activation='relu')(z6)  # 5, 5, 1024
    if pooling == 'avg':
        output = keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(output)
    elif pooling == 'max':
        output = keras.layers.GlobalMaxPooling2D(name='global_max_pool')(output)
    #z8=keras.layers.Dropout(0.5)(z8)
    if include_top:
        output = keras.layers.Dense(classes, activation='softmax',name=name+'/FC')(output)
    model = keras.models.Model(inputs=img_input, outputs=output)
    return model

def DenseShuffleV2(include_top=True,blocks=[2,2,2],input_shape=(160,160,3),num_shuffle_units=[3, 7, 3]
                ,scale_factor = 1.0,bottleneck_ratio = 1,pooling='avg',name='rgb',classes=1108,dropout_rate=0.5):
    #out_dim_stage_two = {0.5: 48, 1: 116, 1.5: 176, 2: 244}
    #out_dim_stage_two = {0.5: 48, 1: 64, 1.5: 128, 2: 192}
    out_dim_stage_two = {0.5: 48, 1: 32, 1.5: 64, 2: 128}
    print(out_dim_stage_two)
    bn_axis = 3   if keras.backend.image_data_format() == 'channels_last' else 1
    img_input = keras.layers.Input(shape=input_shape, name=name + '/input')
    x = keras.layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name=name + '/zeroPad1')(img_input)
    x0 = keras.layers.Conv2D(64, 7, strides=2, use_bias=False,kernel_initializer='he_normal',  name=name + '/conv1/conv')(x)  # 80, 80, 64
    x = keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '/conv1/bn')(x0)
    x = keras.layers.Activation('relu', name=name + '/conv1/relu')(x)
    x = keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + '/zeroPad2')(x)  # 82, 82, 64
    x = keras.layers.MaxPooling2D(3, strides=2, name=name + '/pool1')(x)  # 40, 40, 64
    x = dense_block(x, blocks[0], name=name + '/conv2')  # 40, 40, 128
    x = transition_block(x, 0.5, name=name + '/pool2')  # 20, 20, 64
    exp = np.insert(np.arange(len(num_shuffle_units), dtype=np.float32), 0, 0)  # [0., 0., 1., 2.]
    out_channels_in_stage = 2 ** exp  # 1 1 2 4
    out_channels_in_stage *= out_dim_stage_two[bottleneck_ratio]  # calculate output channels for each stage 116., 116., 232., 464.
    out_channels_in_stage[0] = 24  # first stage has always 24 output channels
    out_channels_in_stage *= scale_factor  # 24., 116., 232., 464
    out_channels_in_stage = out_channels_in_stage.astype(int)
    print(out_channels_in_stage)
    z1 = keras.layers.Conv2D(filters=out_channels_in_stage[0], kernel_size=(3, 3), padding='same', use_bias=False, strides=(2, 2),kernel_initializer='he_normal',  activation='relu', name='conv1')(img_input)  # 80, 80, 24
    z1 = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='maxpool1')(z1)  # 40, 40, 24
    z1 = block(z1, out_channels_in_stage, repeat=num_shuffle_units[0], bottleneck_ratio=bottleneck_ratio, stage=0 + 2)  # 20, 20, 232)
    z2 = keras.layers.concatenate([z1, x])
    z2 = keras.layers.Dropout(dropout_rate)(z2)
    z2 = keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '/concat1/bn')(z2)

    z21 = dense_block(z2, blocks[1], name=name + '/conv3',do_norm=False)  # 20, 20, 360
    z21 = transition_block(z21, 0.5, name=name + '/pool3')  # 10, 10, 180
    z3 = block(z1, out_channels_in_stage, repeat=num_shuffle_units[1], bottleneck_ratio=bottleneck_ratio, stage=1 + 2)  # 10, 10, 464
    z4 = keras.layers.concatenate([z21, z3])  # 10, 10, 644
    z4 = keras.layers.Dropout(dropout_rate)(z4)
    z4 = keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '/concat2/bn')(z4)

    z41 = dense_block(z4, blocks[2], name=name + '/conv4',do_norm=False)
    z41 = transition_block(z41, 0.5, name=name + '/pool4')  # 5, 5, 354
    z5 = block(z3, out_channels_in_stage, repeat=num_shuffle_units[2], bottleneck_ratio=bottleneck_ratio, stage=2 + 2)  # 5, 5, 928)
    z6 = keras.layers.concatenate([z5, z41])  # 5, 5, 1282
    z6 = keras.layers.Dropout(dropout_rate)(z6)
    z6 = keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '/concat3/bn')(z6)
    if bottleneck_ratio < 2:
        #k = 1024
        k=512
    else:
        k = 2048
    output = keras.layers.Conv2D(k, kernel_size=1, padding='same', strides=1, name='1x1conv5_out', kernel_initializer='he_normal', activation='relu')(z6)  # 5, 5, 1024
    if pooling == 'avg':
        output = keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(output)
    elif pooling == 'max':
        output = keras.layers.GlobalMaxPooling2D(name='global_max_pool')(output)
    #z8=keras.layers.Dropout(0.5)(z8)
    if include_top:
        output = keras.layers.Dense(classes, activation='softmax',name=name+'/FC')(output)
    model = keras.models.Model(inputs=img_input, outputs=output)
    return model

def DenseNet(include_top=True,blocks=[2,2,2,2],input_shape=(192,192,3),pooling='max',classes=10,name='rgb'):
    bn_axis = 3 if keras.backend.image_data_format() == 'channels_last' else 1
    img_input = keras.layers.Input(shape=input_shape, name=name+'/input')
    x = keras.layers.ZeroPadding2D(padding=((3, 3), (3, 3)),name=name+'/zeroPad1')(img_input)
    x = keras.layers.Conv2D(64, 7, strides=2, use_bias=False, name=name+'/conv1/conv')(x)
    x = keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name+'/conv1/bn')(x)
    x = keras.layers.Activation('relu', name=name+'/conv1/relu')(x)
    x = keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)),name=name+'/zeroPad2')(x)
    x = keras.layers.MaxPooling2D(3, strides=2, name=name+'/pool1')(x)
    x = dense_block(x, blocks[0], name=name+'/conv2')
    x = transition_block(x, 0.5, name=name+'/pool2')
    x = dense_block(x, blocks[1], name=name+'/conv3')
    x = transition_block(x, 0.5, name=name+'/pool3')
    x = dense_block(x, blocks[2], name=name+'/conv4')
    x = transition_block(x, 0.5, name=name+'/pool4')
    x = dense_block(x, blocks[3], name=name+'/conv5')
    x = keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name+'/bn')(x)
    output = keras.layers.Activation('relu', name=name+'/relu')(x)
    if pooling == 'avg':
        output = keras.layers.GlobalAveragePooling2D(name=name+'/avg_pool')(output)
    elif pooling == 'max':
        output = keras.layers.GlobalMaxPooling2D(name=name+'/max_pool')(output)
    if include_top:
        output = keras.layers.Dense(classes, activation='softmax', name=name + '/FC')(output)
    model = keras.models.Model(inputs=img_input, outputs=output,name=name)
    return model

def ShuffleNetV2(include_top=True,
                 scale_factor=1.0,
                 pooling='max',
                 input_shape=(224,224,3),
                 num_shuffle_units=[3,7,3],
                 bottleneck_ratio=1,
                 classes=1000):

    name = 'ShuffleNetV2_{}_{}_{}'.format(scale_factor, bottleneck_ratio, "".join([str(x) for x in num_shuffle_units]))
    out_dim_stage_two = {0.5:48, 1:116, 1.5:176, 2:244}

    if pooling not in ['max', 'avg']:
        raise ValueError('Invalid value for pooling')
    if not (float(scale_factor)*4).is_integer():
        raise ValueError('Invalid value for scale_factor, should be x over 4')
    exp = np.insert(np.arange(len(num_shuffle_units), dtype=np.float32), 0, 0)  # [0., 0., 1., 2.]
    out_channels_in_stage = 2**exp
    out_channels_in_stage *= out_dim_stage_two[bottleneck_ratio]  #  calculate output channels for each stage
    out_channels_in_stage[0] = 24  # first stage has always 24 output channels
    out_channels_in_stage *= scale_factor
    out_channels_in_stage = out_channels_in_stage.astype(int)


    img_input = keras.layers.Input(shape=input_shape)

    # create shufflenet architecture
    x = keras.layers.Conv2D(filters=out_channels_in_stage[0], kernel_size=(3, 3), padding='same', use_bias=False, strides=(2, 2),
               activation='relu', name='conv1')(img_input)
    x = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='maxpool1')(x)

    # create stages containing shufflenet units beginning at stage 2
    for stage in range(len(num_shuffle_units)):
        repeat = num_shuffle_units[stage]
        x = block(x, out_channels_in_stage, repeat=repeat,bottleneck_ratio=bottleneck_ratio, stage=stage + 2)

    if bottleneck_ratio < 2:
        k = 1024
    else:
        k = 2048
    x = keras.layers.Conv2D(k, kernel_size=1, padding='same', strides=1, name='1x1conv5_out', activation='relu')(x)

    if pooling == 'avg':
        x = keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    elif pooling == 'max':
        x = keras.layers.GlobalMaxPooling2D(name='global_max_pool')(x)

    if include_top:
        x = keras.layers.Dense(classes, name='fc')(x)
        x = keras.layers.Activation('softmax', name='softmax')(x)
    model = keras.models.Model(img_input,x, name=name)
    return model

if __name__ == "__main__":
    blocks=[6,12,24]
    blocks = [2, 4, 8]
    num_shuffle_units = [2, 3, 2]
    shape=(224,224,3)
    name = '%s_%s%s%s' % (sum(blocks) * 2 + 4, num_shuffle_units[0], num_shuffle_units[1], num_shuffle_units[2])
    model=DenseShuffleV1(blocks=blocks,input_shape=shape,num_shuffle_units=num_shuffle_units,scale_factor = 1.0,bottleneck_ratio = 1,pooling='max',name='rgb',classes=1108)
    model.summary()
    keras.utils.plot_model(model, 'DenseShuffleV1_%s.png' %name, show_shapes=True)