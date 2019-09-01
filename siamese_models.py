import sys
import tensorflow as tf
if tf.__version__<'2.0':
    import keras
    import keras.backend as K
else:
    from tensorflow import keras
    import tensorflow.keras.backend as K

from models.mobilenet import MobileNet
from models.mobilenet_v2 import MobileNetV2
from models.nasnet import NASNetMobile
from models.squeezenet import SqueezeNet
from models.DenseShuffle import DenseNet,DenseShuffleV1,DenseShuffleV2,ShuffleNetV2
from models.resnet import ResNet18,ResNet18V2,ResNet34,ResNet34V2
from models.mobilenet_v3 import MobileNetV3Small,MobileNetV3Large
from models.shufflenet import ShuffleNet


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def build_model(net='MobileNet',input_shape=(224,224,3),siamese_weights=None,share=True):
    if net=='MobileNet' :
        base_model=MobileNet(include_top=False,input_shape=input_shape)
    elif net =='MobileNetV2':
        base_model = MobileNetV2(include_top=False,input_shape=input_shape)
    elif net=='NASNetMobile':
        base_model=NASNetMobile(include_top=False,input_shape=input_shape)
    elif net=='ResNet18':
        base_model=ResNet18(include_top=False,input_shape=input_shape)
    elif net=='ResNet18V2':
        base_model=ResNet18V2(include_top=False,input_shape=input_shape)
    elif net=='ResNet34':
        base_model=ResNet34(include_top=False,input_shape=input_shape)
    elif net=='ResNet34V2':
        base_model=ResNet34V2(include_top=False,input_shape=input_shape)
    elif net == 'DenseNet21':
        base_model = DenseNet(include_top=False, blocks=[2, 2, 2, 2], input_shape=input_shape,name='a')
        if share ==False:
            base_model_b = DenseNet(include_top=False, blocks=[2, 2, 2, 2], input_shape=input_shape, name='b')
    elif net=='DenseNet69':
        base_model = DenseNet(include_top=False,blocks=[6,8,10,8],input_shape=input_shape)
        if share ==False:
            base_model_b = DenseNet(include_top=False, blocks=[6, 8, 10, 8], input_shape=input_shape, name='b')
    elif net=='DenseNet109':
        base_model = DenseNet(include_top=False,blocks=[6,12,18,16],input_shape=input_shape)
        if share ==False:
            base_model_b = DenseNet(include_top=False, blocks=[6, 12, 18, 16], input_shape=input_shape, name='b')
    elif net == 'DenseShuffleV1_57_373':
        base_model=DenseShuffleV1(include_top=False,blocks=[6, 8, 12], input_shape=input_shape, num_shuffle_units=[3, 7, 3],
                             scale_factor=1.0, bottleneck_ratio=1, dropout_rate=0.5)
    elif net == 'DenseShuffleV2_57_373':
        base_model=DenseShuffleV2(include_top=False,blocks=[6, 8, 12], input_shape=input_shape, num_shuffle_units=[3, 7, 3],
                             scale_factor=1.0, bottleneck_ratio=1, dropout_rate=0.5)
    elif net == 'DenseShuffleV2_17_232':
        base_model=DenseShuffleV2(include_top=False,blocks=[2, 2, 2], input_shape=input_shape, num_shuffle_units=[2, 3, 2],
                             scale_factor=1.0, bottleneck_ratio=1, dropout_rate=0.5)
    elif net=='ShuffleNetV2':
        base_model=ShuffleNetV2(include_top=False, scale_factor=1.0,pooling='avg',input_shape=input_shape,
                 num_shuffle_units=[3,7,3],bottleneck_ratio=1)
    elif net == 'ShuffleNet':
        base_model = ShuffleNet(include_top=False, scale_factor=1.0, pooling='avg', input_shape=input_shape,
                                  num_shuffle_units=[3, 7, 3], bottleneck_ratio=1)
    elif net=='MobileNetV3Small':
        base_model = MobileNetV3Small(include_top=False,input_shape=input_shape)
    elif net=='SqueezeNet':
        base_model = SqueezeNet(include_top=False,input_shape=input_shape)
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    input_a = keras.layers.Input(shape=input_shape, name='input_a')
    input_b = keras.layers.Input(shape=input_shape, name='input_b')
    processed_a = base_model(input_a)

    if share:
        processed_b = base_model(input_b)
    else:
        processed_b = base_model_b(input_b)

    #processed_a = keras.layers.Activation('sigmoid', name='sigmoid_a')(processed_a)
    #processed_b = keras.layers.Activation('sigmoid', name='sigmoid_b')(processed_b)
    normalize = keras.layers.Lambda(lambda x: K.l2_normalize(x, axis=-1), name='normalize')
    processed_a = normalize(processed_a)
    processed_b = normalize(processed_b)
    distance = keras.layers.Lambda(euclidean_distance,output_shape=eucl_dist_output_shape,
                                   name='dist')([processed_a, processed_b])
    model = keras.models.Model([input_a, input_b], distance)
    if siamese_weights is not None:
        print('load siamses weights ....')
        model.load_weights(siamese_weights)
    print('hahahaha')
    return model

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1.0
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0.))
    return K.mean(y_true * square_pred + (1. - y_true) * margin_square)

def covert2cassify(model,num_classes,trainable=False):
    base_model=model.layers[2]
    print('trainable:',trainable)
    if trainable==False:
        n=len(base_model.layers)
        for i in range(0,n):
            base_model.layers[i].trainable=False
    output=keras.layers.Dense(num_classes,activation='softmax',name='softmax')(base_model.layers[-1].output)
    model=keras.models.Model(base_model.input,output)
    return model

