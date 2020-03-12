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
from models.squeezenet import SqueezeNet,SqueezeCapsule
from models.DenseShuffle import DenseNet,DenseShuffleV1,DenseShuffleV2,ShuffleNetV2,ShuffleNetV2Capsule
from models.resnet import ResNet18,ResNet34,ResNet50,ResNet101,ResNet152,ResNet50V2,ResNet101V2,ResNet152V2,ResNeXt50,ResNeXt101
from models.mobilenet_v3 import MobileNetV3Small,MobileNetV3Large
from models.shufflenet import ShuffleNet
from models.mixnets import MixNetSmall,MixNetLarge,MixNetMedium
from models.residual_attention_network import AttentionResNet56,AttentionResNet92
from models.DualPathNetwork import  DPN92,DPN98,DPN107,DPN137
from models.darknet53 import darknet
from models.Mnasnet import MnasNet
from models.SimpleNet import SimpleNetV1,SimpleNetV2
from models.vgg import VGG11Small,VGG11,VGG13,VGG16,VGG19,VGG19Small
from models.efficientnet import EfficientNetB0,EfficientNetB1,EfficientNetB2,EfficientNetB3,EfficientNetB4,EfficientNetB5,EfficientNetB6,EfficientNetB7
from models.ResDense import ResDenseNet,ResDenseNetMedium,ResDenseNetSmall
from models.MultiResNet import MultiResNet
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
    elif net=='ResNet34':
        base_model=ResNet34(include_top=True,input_shape=input_shape,classes=classes)
    elif net=='ResNet50':
        base_model=ResNet50(include_top=True,input_shape=input_shape,classes=classes)
    elif net=='ResNet101':
        base_model=ResNet101(include_top=True,input_shape=input_shape,classes=classes)
    elif net=='ResNet152':
        base_model=ResNet152(include_top=True,input_shape=input_shape,classes=classes)
    elif net=='ResNet50V2':
        base_model=ResNet50V2(include_top=True,input_shape=input_shape,classes=classes)
    elif net=='ResNeXt50':
        base_model=ResNeXt50(include_top=True,input_shape=input_shape,classes=classes)
    elif net == 'DenseNet21':
        base_model = DenseNet(include_top=True, blocks=[2, 2, 2, 2], input_shape=input_shape,classes=classes,name='a')
    elif net == 'DenseNet21Capsule':
        base_model = DenseNet(include_top=True, blocks=[2, 2, 2, 2], input_shape=input_shape,classes=classes,pooling='avg',name='a',capsule=True)
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
    elif net=='SimpleNetV1':
        base_model=SimpleNetV1(include_top=True,input_shape=input_shape,classes=classes)
    elif net=='SimpleNetV2':
        base_model=SimpleNetV2(include_top=True,input_shape=input_shape,classes=classes,pooling='max')
    elif net=='SimpleNetV2Capsule':
        base_model=SimpleNetV2(include_top=True,input_shape=input_shape,classes=classes,capsule=True)
    elif net=='VGG11Small':
        base_model=VGG11Small(include_top=True,input_shape=input_shape,classes=classes)
    elif net=='VGG19Small':
        base_model=VGG19Small(include_top=True,input_shape=input_shape,classes=classes)
    elif net=='VGG11':
        base_model=VGG11(include_top=True,input_shape=input_shape,classes=classes)
    elif net=='VGG13':
        base_model=VGG13(include_top=True,input_shape=input_shape,classes=classes)
    elif net=='VGG16':
        base_model=VGG16(include_top=True,input_shape=input_shape,classes=classes)
    elif net=='VGG19':
        base_model=VGG19(include_top=True,input_shape=input_shape,classes=classes)
    elif net=='EfficientNetB0':
        base_model=EfficientNetB0(include_top=True,input_shape=input_shape,classes=classes)
    elif net=='EfficientNetB1':
        base_model=EfficientNetB1(include_top=True,input_shape=input_shape,classes=classes)
    elif net=='EfficientNetB2':
        base_model=EfficientNetB2(include_top=True,input_shape=input_shape,classes=classes)
    elif net=='EfficientNetB3':
        base_model=EfficientNetB3(include_top=True,input_shape=input_shape,classes=classes)
    elif net=='EfficientNetB4':
        base_model=EfficientNetB4(include_top=True,input_shape=input_shape,classes=classes)
    elif net=='ResDenseNet30':
        base_model=ResDenseNetSmall(include_top=True,input_shape=input_shape,classes=classes)
    elif net=='ResDenseNet30Capsule':
        #3,719,424
        base_model=ResDenseNet(blocks=[2,2,2,2],include_top=True,input_shape=input_shape,classes=classes,capsule=True)
    elif net=='ResDenseNet53':
        base_model=ResDenseNetMedium(include_top=True,input_shape=input_shape,classes=classes)
    elif net=='ResDenseNet104':
        base_model=ResDenseNet(blocks=[3,4,23,3],include_top=True,input_shape=input_shape,classes=classes)
    elif net=='ResDenseNet176':
        base_model=ResDenseNet(blocks=[6,12,24,16],include_top=True,input_shape=input_shape,classes=classes)
    elif net=='SqueezeCapsule':
        base_model=SqueezeCapsule(include_top=True,input_shape=input_shape,classes=classes)
    elif net == 'ShuffleNetV2Capsule':
        base_model = ShuffleNetV2Capsule(include_top=True, input_shape=input_shape, classes=classes)
    elif net=='MultiResNet':
        base_model=MultiResNet(input_shape=input_shape, classes=classes)
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()
    return base_model

def RandSqeenzeLayer(model,alpha=0.5):
    layers_list=model.layers
    layers_list=[str(a).split(' ')[0] for a in layers_list]
    layers_list = [a.split('.')[-1] for a in layers_list]
    sq=['Conv2D','DepthwiseConv2D','Dense']
    layers_index=[]
    n=len(layers_list)
    for i in range(0,n-1):
        if sq.__contains__(layers_list[i]):
            layers_index.append(i)
    #layers_index=[layers_list.index(a) for a in layers_list if ['Conv2D','DepthwiseConv2D','Dense'].__contains__(a)]
    num_layers=len(layers_index)
    #sn=np.random.randint(1,num_layers,int(num_layers*alpha))
    sn=np.random.choice(a=layers_index, size=int(num_layers*alpha), replace=False, p=None)
    sn.sort()
    for i in sn:
        print(i,model.get_layer(index=i))
        model.layers[i].trainable=False
    return model
