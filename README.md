# keras-alchemy
keras alchemy of image classification

## Model list(net args)
```
MobileNet
MobileNetV2
MobileNetV3Small
MobileNetV3Large
NASNetMobile
NASNetLarge
SqueezeNet
ShuffleNet
ShuffleNetV2
MixNetSmall
MixNetMedium
MixNetLarge
ResNet18
ResNet34
ResNet50
ResNet101
ResNet152
ResNet50V2
ResNeXt50
DenseNet21
DenseNet121
AttentionResNet56
AttentionResNet92
DPN92
DPN98
DPN107
DPN137
DarkNet53
MnasNet
SimpleNetV1
SimpleNetV2
VGG11
VGG13
VGG16
VGG19
VGG19Small
EfficientNetB0
ResDenseNet-30
MultiResNet
BlazeFace
```
## cifar100 Results

|network|params|val_acc|
|:---:|---:|:---:|
|SimpleNetV1|18,113,508|0.7295|
|SimpleNetV2|5,522,739|0.7305|
|SimpleNetV2Capsule|6,861,839|0.7583|
|SqueezeNet|787,108|0.7068|
|SqueezeCapsule|2,374,208|0.7295|
|ShuffleNet|1,027,324|0.6708|
|ShuffleNetV2|4,121,336|0.7205|
|ShuffleNetV2Capsule|7,295,636|0.7125|
|MobileNet|3,331,364|0.7305|
|MobileNetV2|2,386,084|0.6496|
|MobileNetV3Small|3,073,916|0.6503|
|MobileNetV3Large|5,252,092|0.7387|
|VGG11|28,523,748|0.6621|
|VGG11Small|2,932,996|0.6330|
|VGG19|39,338,660|0.6791|
|VGG19Small|5,114,852|0.6578|
|ResNet18|11,237,156|0.7563|
|ResNet34|21,352,740|0.7785|
|ResNet50V2|23,769,700|0.7593|
|ResDenseNet-30|1,524,724|0.7305|
|ResDenseNet-30-Capsule|3,719,424|0.7402|
|ResDenseNet-53|2,976,004|0.7559|
|ResDenseNet-104|7,407,972|0.7689|
|MnasNet|3,217,900|0.7521|
|MultiResNet|6,014,038|0.6921|
|BlazeFace|715,044|0.6236|


# Reference
* [keras-applications](https://github.com/keras-team/keras-applications)
* [Keras_ResNet18_of_Tricks](https://github.com/Tony607/Keras_Bag_of_Tricks)
* [MobileNet_V3](https://github.com/godofpdog/MobileNetV3_keras/)
* [shufflenet](https://github.com/scheckmedia/keras-shufflenet/)
* [shufflenetv2](https://github.com/opconty/keras-shufflenetV2/)
* [keras-squeezenet](https://github.com/rcmalli/keras-squeezenet/)
* [keras-resnet](https://github.com/broadinstitute/keras-resnet)
* [keras-radam](https://github.com/CyberZHG/keras-radam)
* [mnist_siamese](https://github.com/keras-team/keras/blob/master/examples/mnist_siamese.py)
* [pytorch-cifar100](https://github.com/weiaicunzai/pytorch-cifar100)
* [keras-CLR](https://github.com/bckenstler/CLR)
* [keras_mixnets](https://github.com/titu1994/keras_mixnets)
* [residual_attention_network](https://github.com/qubvel/residual_attention_network)
* [Keras-DualPathNetworks](https://github.com/titu1994/Keras-DualPathNetworks)
* [DarkNet53](https://github.com/xiaochus/YOLOv3)
* [MnasNet](https://github.com/Shathe/MNasNet-Keras-Tensorflow)
* [SimpleNet](https://github.com/Coderx7/SimpleNet)
* [efficientnet](https://github.com/qubvel/efficientnet)
* [Capsule](https://github.com/bojone/Capsule)
* [MultiResUNet](https://github.com/nibtehaz/MultiResUNet)
* [BlazeFace](https://github.com/vietanhdev/blazeface_keras)
