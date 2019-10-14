import sys,math
import numpy as np
import tensorflow as tf
if tf.__version__<'2.0':
    import keras
    import keras.backend as K
else:
    from tensorflow import keras
    import tensorflow.keras.backend as K

from models.mobilenet import MobileNet
from models.mobilenet_v2 import MobileNetV2
from models.nasnet import NASNetMobile,NASNetLarge
from models.squeezenet import SqueezeNet
from models.DenseShuffle import DenseNet,DenseShuffleV1,DenseShuffleV2,ShuffleNetV2
from models.resnet import ResNet18,ResNet18V2,ResNet34,ResNet34V2
from models.mobilenet_v3 import MobileNetV3Small,MobileNetV3Large
from models.shufflenet import ShuffleNet
from models.mixnets import MixNetSmall,MixNetLarge,MixNetMedium

def build_model(net='MobileNet',input_shape=(224,224,3),classes=100):
    if net=='MobileNet' :
        base_model=MobileNet(include_top=True,input_shape=input_shape,classes=classes)
    elif net =='MobileNetV2':
        base_model = MobileNetV2(include_top=True,input_shape=input_shape,classes=classes)
    elif net=='NASNetMobile':
        base_model=NASNetMobile(include_top=True,input_shape=input_shape,classes=classes)
    elif net=='NASNetLarge':
        base_model=NASNetLarge(include_top=True,input_shape=input_shape,classes=classes)
    elif net=='ResNet18':
        base_model=ResNet18(include_top=True,input_shape=input_shape,classes=classes)
    elif net=='ResNet18V2':
        base_model=ResNet18V2(include_top=True,input_shape=input_shape,classes=classes)
    elif net=='ResNet34':
        base_model=ResNet34(include_top=True,input_shape=input_shape,classes=classes)
    elif net=='ResNet34V2':
        base_model=ResNet34V2(include_top=True,input_shape=input_shape,classes=classes)
    elif net == 'DenseNet21':
        base_model = DenseNet(include_top=True, blocks=[2, 2, 2, 2], input_shape=input_shape,classes=classes,name='a')
    elif net=='DenseNet69':
        base_model = DenseNet(include_top=True,blocks=[6,8,10,8],input_shape=input_shape,classes=classes)
    elif net=='DenseNet109':
        base_model = DenseNet(include_top=True,blocks=[6,12,18,16],input_shape=input_shape,classes=classes)
    elif net == 'DenseShuffleV1_57_373':
        base_model=DenseShuffleV1(include_top=True,blocks=[6, 8, 12], input_shape=input_shape, num_shuffle_units=[3, 7, 3],
                             scale_factor=1.0, bottleneck_ratio=1, dropout_rate=0.5,classes=classes)
    elif net == 'DenseShuffleV2_49_353':
        base_model=DenseShuffleV2(include_top=True,blocks=[6, 8, 8], input_shape=input_shape, num_shuffle_units=[3, 5, 3],
                             scale_factor=1.0, bottleneck_ratio=1, dropout_rate=0.5,classes=classes)
    elif net == 'DenseShuffleV2_57_373':
        base_model=DenseShuffleV2(include_top=True,blocks=[6, 8, 12], input_shape=input_shape, num_shuffle_units=[3, 7, 3],
                             scale_factor=1.0, bottleneck_ratio=1, dropout_rate=0.5,classes=classes)

    elif net == 'DenseShuffleV2_17_232':
        base_model=DenseShuffleV2(include_top=True,blocks=[2, 2, 2], input_shape=input_shape, num_shuffle_units=[2, 3, 2],
                             scale_factor=1.0, bottleneck_ratio=1, dropout_rate=0.5)
    elif net == 'DenseShuffleV1_17_232':
        base_model=DenseShuffleV1(include_top=True,blocks=[2, 2, 2], input_shape=input_shape, num_shuffle_units=[2, 3, 2],
                             scale_factor=1.0, bottleneck_ratio=1, dropout_rate=0.5,classes=classes)

    elif net=='ShuffleNetV2':
        base_model=ShuffleNetV2(include_top=True, scale_factor=1.0,pooling='avg',input_shape=input_shape,
                 num_shuffle_units=[3,7,3],bottleneck_ratio=1,classes=classes)
    elif net == 'ShuffleNet':
        base_model = ShuffleNet(include_top=True, scale_factor=1.0, pooling='avg', input_shape=input_shape,
                                  num_shuffle_units=[3, 7, 3], bottleneck_ratio=1,classes=classes)
    elif net=='MobileNetV3Small':
        base_model = MobileNetV3Small(include_top=True,input_shape=input_shape,classes=classes)
    elif net=='MobileNetV3Large':
        base_model = MobileNetV3Large(include_top=True,input_shape=input_shape,classes=classes)
    elif net=='SqueezeNet':
        base_model = SqueezeNet(include_top=True,input_shape=input_shape,classes=classes)
    elif net=='MixNetMedium':
        base_model=MixNetMedium(include_top=True,input_shape=input_shape,classes=classes)
    elif net=='MixNetSmall':
        base_model=MixNetSmall(include_top=True,input_shape=input_shape,classes=classes)
    elif net=='MixNetLarge':
        base_model=MixNetLarge(include_top=True,input_shape=input_shape,classes=classes)
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()
    return base_model
