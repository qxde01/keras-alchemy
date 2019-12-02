import sys,math
import numpy as np
import tensorflow as tf
if tf.__version__<'2.0':
    import keras
    import keras.backend as K
else:
    tf.compat.v1.disable_eager_execution()
    from tensorflow import keras
    import tensorflow.keras.backend as K

from models.mobilenet import MobileNet
from models.mobilenet_v2 import MobileNetV2
from models.nasnet import NASNetMobile,NASNetLarge
from models.squeezenet import SqueezeNet
from models.DenseShuffle import DenseNet,DenseShuffleV1,DenseShuffleV2,ShuffleNetV2
from models.resnet import ResNet18,ResNet18V2,ResNet34,ResNet34V2,ResNet50,ResNet50V2,ResNeXt50
from models.mobilenet_v3 import MobileNetV3Small,MobileNetV3Large
from models.shufflenet import ShuffleNet
from models.mixnets import MixNetSmall,MixNetLarge,MixNetMedium
from models.residual_attention_network import AttentionResNet56,AttentionResNet92
from models.DualPathNetwork import  DPN92,DPN98,DPN107,DPN137
from models.darknet53 import darknet
from models.Mnasnet import MnasNet

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
    elif net=='ResNet50':
        base_model=ResNet50(include_top=True,input_shape=input_shape,classes=classes)
    elif net=='ResNet50V2':
        base_model=ResNet50V2(include_top=True,input_shape=input_shape,classes=classes)
    elif net == 'DenseNet21':
        base_model = DenseNet(include_top=True, blocks=[2, 2, 2, 2], input_shape=input_shape,classes=classes,name='a')
    elif net=='DenseNet69':
        base_model = DenseNet(include_top=True,blocks=[6,8,10,8],input_shape=input_shape,classes=classes)
    elif net=='DenseNet109':
        base_model = DenseNet(include_top=True,blocks=[6,12,18,16],input_shape=input_shape,classes=classes)
    elif net=='DenseNet121':
        base_model = DenseNet(include_top=True,blocks=[6,12,24,16],input_shape=input_shape,classes=classes)
   elif net == 'DenseShuffleV1_49_233':
        base_model=DenseShuffleV1(include_top=True,blocks=[4, 8, 10], input_shape=input_shape, num_shuffle_units=[2, 3, 3],
                             scale_factor=1.0, bottleneck_ratio=1, dropout_rate=0.0,classes=classes)
    elif net == 'DenseShuffleV1_37_551':
        base_model=DenseShuffleV1(include_top=True,blocks=[4, 8, 4], input_shape=input_shape, num_shuffle_units=[5, 5, 1],
                             scale_factor=1.0, bottleneck_ratio=1, dropout_rate=0.0,classes=classes)
    elif net == 'DenseShuffleV1_37_532':
        base_model = DenseShuffleV1(include_top=True, blocks=[6, 6, 4], input_shape=input_shape, num_shuffle_units=[5, 3, 2],
                                    scale_factor=1.0, bottleneck_ratio=1, dropout_rate=0.0, classes=classes)
    elif net == 'DenseShuffleV2_49_353':
        base_model=DenseShuffleV2(include_top=True,blocks=[6, 8, 8], input_shape=input_shape, num_shuffle_units=[3, 5, 3],
                             scale_factor=1.0, bottleneck_ratio=1, dropout_rate=0.5,classes=classes)
    elif net == 'DenseShuffleV2_57_373':
        base_model=DenseShuffleV2(include_top=True,blocks=[6, 8, 12], input_shape=input_shape, num_shuffle_units=[3, 7, 3],
                             scale_factor=1.0, bottleneck_ratio=1, dropout_rate=0.5,classes=classes)

    elif net == 'DenseShuffleV2_17_232':
        base_model=DenseShuffleV2(include_top=True,blocks=[2, 2, 2], input_shape=input_shape, num_shuffle_units=[2, 3, 2],
                             scale_factor=1.0, bottleneck_ratio=1, dropout_rate=0.5,classes=classes)
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
        base_model = MobileNetV3Small(include_top=True,input_shape=input_shape,num_classes=classes)
    elif net=='MobileNetV3Large':
        base_model = MobileNetV3Large(include_top=True,input_shape=input_shape,num_classes=classes)
    elif net=='SqueezeNet':
        base_model = SqueezeNet(include_top=True,input_shape=input_shape,classes=classes)
    elif net=='MixNetMedium':
        base_model=MixNetMedium(include_top=True,input_shape=input_shape,classes=classes)
    elif net=='MixNetSmall':
        base_model=MixNetSmall(include_top=True,input_shape=input_shape,classes=classes)
    elif net=='MixNetLarge':
        base_model=MixNetLarge(include_top=True,input_shape=input_shape,classes=classes)
    elif net == 'AttentionResNet56':
        base_model = AttentionResNet56(include_top=True, input_shape=input_shape,n_channels=32,n_classes=classes)
    elif net == 'AttentionResNet92':
        base_model = AttentionResNet92(include_top=True, input_shape=input_shape,n_channels=32,n_classes=classes)
    elif net=='DPN92':
        base_model=DPN92(include_top=True,input_shape=input_shape,classes=classes)    
    elif net=='DPN98':
        base_model=DPN98(include_top=True,input_shape=input_shape,classes=classes)    
    elif net=='DPN107':
        base_model=DPN107(include_top=True,input_shape=input_shape,classes=classes)
    elif net=='DPN137':
        base_model=DPN137(include_top=True,input_shape=input_shape,classes=classes)
    elif net=='DarkNet53':
        base_model=darknet(include_top=True,input_shape=input_shape,classes=classes)
    elif net=='MnasNet':
        base_model=MnasNet(include_top=True,input_shape=input_shape,classes=classes)
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()
    return base_model
