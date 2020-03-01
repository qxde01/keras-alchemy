import tensorflow as tf
if tf.__version__<'2.0':
    import keras
else:
    from tensorflow import keras
from .Capsule import Capsule

# https://github.com/Coderx7/SimpleNet
def SimpleNetV1(include_top=True,input_shape=(224,224,3),pooling='max',classes=100,dropout_rate=0.1):
    img_input = keras.layers.Input(shape=input_shape, name= 'input')
    conv1=keras.layers.Conv2D(64,kernel_size=3,strides=(1,1),padding='same',name='conv1',kernel_initializer='he_normal')(img_input)
    conv1=keras.layers.BatchNormalization(axis=-1,momentum=0.05,epsilon=1e-5,name='conv1_bn')(conv1)
    conv1=keras.layers.ReLU(name='conv1_bn_relu')(conv1)

    conv2 = keras.layers.Conv2D(128, kernel_size=3, strides=(1, 1), padding='same', name='conv2',
                                kernel_initializer='he_normal')(conv1)
    conv2 = keras.layers.BatchNormalization(axis=-1, momentum=0.05, epsilon=1e-5, name='conv2_bn')(conv2)
    conv2 = keras.layers.ReLU(name='conv2_bn_relu')(conv2)

    conv3 = keras.layers.Conv2D(128, kernel_size=3, strides=(1, 1), padding='same', name='conv3',
                                kernel_initializer='he_normal')(conv2)
    conv3 = keras.layers.BatchNormalization(axis=-1, momentum=0.05, epsilon=1e-5, name='conv3_bn')(conv3)
    conv3 = keras.layers.ReLU(name='conv3_bn_relu')(conv3)

    conv4 = keras.layers.Conv2D(128, kernel_size=3, strides=(1, 1), padding='same', name='conv4',
                                kernel_initializer='he_normal')(conv3)
    conv4 = keras.layers.BatchNormalization(axis=-1, momentum=0.05, epsilon=1e-5, name='conv4_bn')(conv4)
    conv4 = keras.layers.ReLU(name='conv4_bn_relu')(conv4)

    maxpool1=keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2),padding='same',data_format='channels_last',name='maxpool1')(conv4)
    if dropout_rate > 0.0:
        maxpool1=keras.layers.Dropout(rate=dropout_rate,name='dropout1')(maxpool1)

    conv5 = keras.layers.Conv2D(128, kernel_size=3, strides=(1, 1), padding='same', name='conv5',
                                kernel_initializer='he_normal')(maxpool1)
    conv5 = keras.layers.BatchNormalization(axis=-1, momentum=0.05, epsilon=1e-5, name='conv5_bn')(conv5)
    conv5 = keras.layers.ReLU(name='conv5_bn_relu')(conv5)

    conv6 = keras.layers.Conv2D(128, kernel_size=3, strides=(1, 1), padding='same', name='conv6',
                                kernel_initializer='he_normal')(conv5)
    conv6 = keras.layers.BatchNormalization(axis=-1, momentum=0.05, epsilon=1e-5, name='conv6_bn')(conv6)
    conv6 = keras.layers.ReLU(name='conv6_bn_relu')(conv6)

    conv7 = keras.layers.Conv2D(256, kernel_size=3, strides=(1, 1), padding='same', name='conv7',
                                kernel_initializer='he_normal')(conv6)
    conv7 = keras.layers.BatchNormalization(axis=-1, momentum=0.05, epsilon=1e-5, name='conv7_bn')(conv7)
    conv7 = keras.layers.ReLU(name='conv7_bn_relu')(conv7)

    maxpool2 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same', data_format='channels_last',
                                      name='maxpool2')(conv7)
    if dropout_rate > 0.0:
        maxpool2 = keras.layers.Dropout(rate=dropout_rate, name='dropout2')(maxpool2)

    conv8 = keras.layers.Conv2D(256, kernel_size=3, strides=(1, 1), padding='same', name='conv8',
                                kernel_initializer='he_normal')(maxpool2)
    conv8 = keras.layers.BatchNormalization(axis=-1, momentum=0.05, epsilon=1e-5, name='conv8_bn')(conv8)
    conv8 = keras.layers.ReLU(name='conv8_bn_relu')(conv8)

    conv9 = keras.layers.Conv2D(256, kernel_size=3, strides=(1, 1), padding='same', name='conv9',
                                kernel_initializer='he_normal')(conv8)
    conv9 = keras.layers.BatchNormalization(axis=-1, momentum=0.05, epsilon=1e-5, name='conv9_bn')(conv9)
    conv9 = keras.layers.ReLU(name='conv9_bn_relu')(conv9)

    maxpool3 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same', data_format='channels_last',
                                      name='maxpool3')(conv9)
    if dropout_rate > 0.0:
        maxpool3 = keras.layers.Dropout(rate=dropout_rate, name='dropout3')(maxpool3)

    conv10 = keras.layers.Conv2D(512, kernel_size=3, strides=(1, 1), padding='same', name='conv10',
                                kernel_initializer='he_normal')(maxpool3)
    conv10 = keras.layers.BatchNormalization(axis=-1, momentum=0.05, epsilon=1e-5, name='conv10_bn')(conv10)
    conv10 = keras.layers.ReLU(name='conv10_bn_relu')(conv10)

    maxpool4 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same', data_format='channels_last',
                                      name='maxpool4')(conv10)
    if dropout_rate > 0.0:
        maxpool4 = keras.layers.Dropout(rate=dropout_rate, name='dropout4')(maxpool4)

    conv11 = keras.layers.Conv2D(2048, kernel_size=3, strides=(1, 1), padding='same', name='conv11',
                                 kernel_initializer='he_normal')(maxpool4)
    conv11 = keras.layers.BatchNormalization(axis=-1, momentum=0.05, epsilon=1e-5, name='conv11_bn')(conv11)
    conv11 = keras.layers.ReLU(name='conv11_bn_relu')(conv11)

    conv12 = keras.layers.Conv2D(256, kernel_size=3, strides=(1, 1), padding='same', name='conv12',
                                 kernel_initializer='he_normal')(conv11)
    conv12 = keras.layers.BatchNormalization(axis=-1, momentum=0.05, epsilon=1e-5, name='conv12_bn')(conv12)
    conv12 = keras.layers.ReLU(name='conv12_bn_relu')(conv12)

    maxpool5 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same', data_format='channels_last',
                                      name='maxpool5')(conv12)
    if dropout_rate > 0.0:
        maxpool5 = keras.layers.Dropout(rate=dropout_rate, name='dropout5')(maxpool5)

    conv13 = keras.layers.Conv2D(256, kernel_size=3, strides=(1, 1), padding='same', name='conv13',
                                 kernel_initializer='he_normal')(maxpool5)
    conv13 = keras.layers.BatchNormalization(axis=-1, momentum=0.05, epsilon=1e-5, name='conv13_bn')(conv13)
    conv13 = keras.layers.ReLU(name='conv13_bn_relu')(conv13)

    if pooling=='avg':
        output=keras.layers.GlobalAveragePooling2D(name='globalAvgpool')(conv13)
    else:
        output = keras.layers.GlobalMaxPooling2D(name='globalMaxpool')(conv13)
    if dropout_rate>0.0:
        output = keras.layers.Dropout(rate=dropout_rate, name='dropout6')(output)
    if include_top:
        output=keras.layers.Dense(classes,activation='softmax')(output)
    model=keras.models.Model(inputs=img_input,outputs=output,name='SimpleNetV1')
    return model
#4,857,267
def SimpleNetV2(include_top=True,input_shape=(224,224,3),pooling='avg',classes=100,dropout_rate=0.0,capsule=False):
    img_input = keras.layers.Input(shape=input_shape, name= 'input')
    conv1=keras.layers.Conv2D(66,kernel_size=3,strides=(1,1),padding='same',name='conv1',kernel_initializer='he_normal')(img_input)
    conv1=keras.layers.BatchNormalization(axis=-1,momentum=0.95,epsilon=1e-5,name='conv1_bn')(conv1)
    conv1=keras.layers.ReLU(name='conv1_bn_relu')(conv1)
    #conv1 = keras.layers.Dropout(rate=dropout_rate, name='dropout1')(conv1)

    conv2 = keras.layers.Conv2D(128, kernel_size=3, strides=(1, 1), padding='same', name='conv2',
                                kernel_initializer='he_normal')(conv1)
    conv2 = keras.layers.BatchNormalization(axis=-1, momentum=0.95, epsilon=1e-5, name='conv2_bn')(conv2)
    conv2 = keras.layers.ReLU(name='conv2_bn_relu')(conv2)
    #conv2 = keras.layers.Dropout(rate=dropout_rate, name='dropout2')(conv2)

    conv3 = keras.layers.Conv2D(128, kernel_size=3, strides=(1, 1), padding='same', name='conv3',
                                kernel_initializer='he_normal')(conv2)
    conv3 = keras.layers.BatchNormalization(axis=-1, momentum=0.95, epsilon=1e-5, name='conv3_bn')(conv3)
    conv3 = keras.layers.ReLU(name='conv3_bn_relu')(conv3)
    #conv3 = keras.layers.Dropout(rate=dropout_rate, name='dropout3')(conv3)

    conv4 = keras.layers.Conv2D(128, kernel_size=3, strides=(1, 1), padding='same', name='conv4',
                                kernel_initializer='he_normal')(conv3)
    conv4 = keras.layers.BatchNormalization(axis=-1, momentum=0.95, epsilon=1e-5, name='conv4_bn')(conv4)
    conv4 = keras.layers.ReLU(name='conv4_bn_relu')(conv4)
    #conv4 = keras.layers.Dropout(rate=dropout_rate, name='dropout4')(conv4)

    conv5 = keras.layers.Conv2D(192, kernel_size=3, strides=(1, 1), padding='same', name='conv5',
                                kernel_initializer='he_normal')(conv4)
    conv5 = keras.layers.BatchNormalization(axis=-1, momentum=0.95, epsilon=1e-5, name='conv5_bn')(conv5)
    conv5 = keras.layers.ReLU(name='conv5_bn_relu')(conv5)
    #conv5 = keras.layers.Dropout(rate=dropout_rate, name='dropout5')(conv5)

    maxpool1 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same', data_format='channels_last',
                                      name='maxpool1')(conv5)
    if dropout_rate > 0.0:
        maxpool1 = keras.layers.Dropout(rate=dropout_rate, name='dropout6')(maxpool1)

    conv6 = keras.layers.Conv2D(192, kernel_size=3, strides=(1, 1), padding='same', name='conv6',
                                kernel_initializer='he_normal')(maxpool1)
    conv6 = keras.layers.BatchNormalization(axis=-1, momentum=0.95, epsilon=1e-5, name='conv6_bn')(conv6)
    conv6 = keras.layers.ReLU(name='conv6_bn_relu')(conv6)
    #conv6 = keras.layers.Dropout(rate=dropout_rate, name='dropout7')(conv6)

    conv7 = keras.layers.Conv2D(192, kernel_size=3, strides=(1, 1), padding='same', name='conv7',
                                kernel_initializer='he_normal')(conv6)
    conv7 = keras.layers.BatchNormalization(axis=-1, momentum=0.95, epsilon=1e-5, name='conv7_bn')(conv7)
    conv7 = keras.layers.ReLU(name='conv7_bn_relu')(conv7)
    #conv7 = keras.layers.Dropout(rate=dropout_rate, name='dropout8')(conv7)

    conv8 = keras.layers.Conv2D(192, kernel_size=3, strides=(1, 1), padding='same', name='conv8',
                                kernel_initializer='he_normal')(conv7)
    conv8 = keras.layers.BatchNormalization(axis=-1, momentum=0.95, epsilon=1e-5, name='conv8_bn')(conv8)
    conv8 = keras.layers.ReLU(name='conv8_bn_relu')(conv8)
    #conv8 = keras.layers.Dropout(rate=dropout_rate, name='dropout9')(conv8)

    conv9 = keras.layers.Conv2D(192, kernel_size=3, strides=(1, 1), padding='same', name='conv9',
                                kernel_initializer='he_normal')(conv8)
    conv9 = keras.layers.BatchNormalization(axis=-1, momentum=0.95, epsilon=1e-5, name='conv9_bn')(conv9)
    conv9 = keras.layers.ReLU(name='conv9_bn_relu')(conv9)
    #conv9 = keras.layers.Dropout(rate=dropout_rate, name='dropout10')(conv9)

    conv10 = keras.layers.Conv2D(288, kernel_size=3, strides=(1, 1), padding='same', name='conv10',
                                kernel_initializer='he_normal')(conv9)
    conv10 = keras.layers.BatchNormalization(axis=-1, momentum=0.95, epsilon=1e-5, name='conv10_bn')(conv10)
    conv10 = keras.layers.ReLU(name='conv10_bn_relu')(conv10)
    #conv10 = keras.layers.Dropout(rate=dropout_rate, name='dropout10')(conv10)

    maxpool2 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same', data_format='channels_last',
                                      name='maxpool2')(conv10)
    if dropout_rate > 0.0:
        maxpool2 = keras.layers.Dropout(rate=dropout_rate, name='dropout11')(maxpool2)

    conv11 = keras.layers.Conv2D(288, kernel_size=3, strides=(1, 1), padding='same', name='conv11',
                                 kernel_initializer='he_normal')(maxpool2)
    conv11 = keras.layers.BatchNormalization(axis=-1, momentum=0.95, epsilon=1e-5, name='conv11_bn')(conv11)
    conv11 = keras.layers.ReLU(name='conv11_bn_relu')(conv11)
    #conv11 = keras.layers.Dropout(rate=dropout_rate, name='dropout12')(conv11)

    conv12 = keras.layers.Conv2D(355, kernel_size=3, strides=(1, 1), padding='same', name='conv12',
                                 kernel_initializer='he_normal')(conv11)
    conv12 = keras.layers.BatchNormalization(axis=-1, momentum=0.95, epsilon=1e-5, name='conv12_bn')(conv12)
    conv12 = keras.layers.ReLU(name='conv12_bn_relu')(conv12)
    #conv12 = keras.layers.Dropout(rate=dropout_rate, name='dropout12')(conv12)

    conv13 = keras.layers.Conv2D(432, kernel_size=3, strides=(1, 1), padding='same', name='conv13',
                                 kernel_initializer='he_normal')(conv12)
    conv13 = keras.layers.BatchNormalization(axis=-1, momentum=0.95, epsilon=1e-5, name='conv13_bn')(conv13)
    conv13 = keras.layers.ReLU(name='conv13_bn_relu')(conv13)
    #conv13 = keras.layers.Dropout(rate=dropout_rate, name='dropout13')(conv12)

    if pooling=='avg':
        output=keras.layers.GlobalAveragePooling2D(name='globalAvgpool')(conv13)
    else:
        output = keras.layers.GlobalMaxPooling2D(name='globalMaxpool')(conv13)
    if dropout_rate>0.0:
        output = keras.layers.Dropout(rate=dropout_rate, name='global_dropout')(output)
    if include_top:
        if capsule :
            output = keras.layers.Reshape((-1, int(output.get_shape()[-1])))(output)
            output = Capsule(classes, 32, 3, True)(output)
            output = keras.layers.Lambda(lambda x: keras.backend.sqrt(keras.backend.sum(keras.backend.square(x), 2)), output_shape=(classes,))(output)
        else:
            output=keras.layers.Dense(classes,activation='softmax')(output)
    model=keras.models.Model(inputs=img_input,outputs=output,name='SimpleNetV2')
    return model

if __name__ == "__main__":
    model=SimpleNetV2(include_top=True,input_shape=(32,32,3),classes=100)
    model.summary()
    keras.utils.plot_model(model, 'SimpleNetV2.png' , show_shapes=True)

