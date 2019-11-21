#from __future__ import absolute_import,print_function
import numpy as np
import tensorflow as tf
import argparse,os
if tf.__version__<'2.0':
    import keras
    from keras import backend as K
else:
    from tensorflow import keras
    import tensorflow.keras.backend as K
    os.environ['TF_KERAS'] = '1'
from keras_radam import RAdam
from keras_lookahead import Lookahead
from classify import build_model
from data_generator import DataGenerator_classify,smooth_labels,preprocess_input
from warmup import LRTensorBoard,WarmUpLearningRateScheduler,CosineAnnealingScheduler,CyclicLR

if os.path.exists('saved')==False:
    os.makedirs('saved')
global init_lr

#def schler(epoch,init_lr=0.02,apha=0.01):
#    return init_lr/(1+apha*epoch)

def scheduler(epoch):
    if epoch<=50:
        lr=0.01
    elif epoch>50 and epoch <=100:
        lr=0.005
    elif epoch>100 and epoch <=150:
        lr=0.001
    else:
        lr=0.0005
    print('epoch:',epoch,'lr:',lr)
    return lr


def get_args():
    parser = argparse.ArgumentParser(description="cifar100 train.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--net", "-net",type=str,  help=" net type",default='MobileNetV2')
    parser.add_argument("--batch_size","-batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--epochs","-epochs", type=int, default=500, help="number of epochs")
    parser.add_argument('--pretrained',"-p",type=str,default=None,help=' softmax model weights  h5 file')
    parser.add_argument("--learning_rate",'-lr', type=float,default=0.01, help="learning_rate")
    parser.add_argument("--max_lr", '-max_lr', type=float, default=0.01, help="max learning_rate")
    parser.add_argument("--warmup", '-wp', type=int, default=1, help="warmup ")
    parser.add_argument("--lr_scheduler",'-lrs', type=int,default=None,help="1-stage decay,2-CosineAnnealingScheduler")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    print(args)
    net = args.net
    lr=args.learning_rate
    init_lr=lr
    max_lr=args.max_lr
    epochs = args.epochs
    batch_size = args.batch_size
    #size=32
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_test = preprocess_input(x_test)
    input_shape = x_train.shape[1:]
    classes = len(np.unique(y_train))
    #opt =Lookahead( RAdam(lr=lr, weight_decay=0.00005, decay=0.0, min_lr=0.00001),sync_period=5,slow_step=0.5)
    opt=RAdam(lr=lr, weight_decay=0.00005, decay=0.0, min_lr=0.00001)
    #opt=keras.optimizers.SGD(lr=lr,momentum=0.9,nesterov=True)
    print('build model:',net)
    model = build_model(net=net, input_shape=input_shape, classes=classes)
    if args.pretrained is not None:
        model.load_weights(args.pretrained)
    model.compile(loss=["categorical_crossentropy"], optimizer=opt, metrics=['accuracy'])
    sample_num = x_train.shape[0]
    y_train = smooth_labels(keras.utils.to_categorical(y_train, num_classes=classes), 0.1)
    y_test = smooth_labels(keras.utils.to_categorical(y_test, num_classes=classes), 0.1)
    train_gen = DataGenerator_classify(x_train=x_train, y_train=y_train, batch_size=batch_size, size=32, shuffle=True, num_classes=classes)
    val_data = (x_test, y_test)
    model.summary()
    filepath = 'saved/classify_%s_{val_loss:.4f}-{val_acc:.4f}-{epoch:03d}.h5' % net
    checkpoint=keras.callbacks.ModelCheckpoint(filepath,save_best_only=False,verbose=1,monitor='val_loss',save_weights_only=True)
    callbacks_list=[checkpoint,LRTensorBoard(log_dir='logs')]
    if args.warmup>0:
        warmup_epoch = args.warmup
        warmup_batches = warmup_epoch * sample_num // batch_size+1
        print('warmup_batches:',warmup_batches)
        warm_up_lr = WarmUpLearningRateScheduler(init_lr=max_lr, warmup_batches=warmup_batches)
        callbacks_list=callbacks_list+[warm_up_lr]
    if args.lr_scheduler is not None:
        if args.lr_scheduler == 1:
            #print('1/sqrt(1+apha*epoch)')
            lrs= keras.callbacks.LearningRateScheduler(scheduler)
            callbacks_list = callbacks_list + [lrs]
        elif args.lr_scheduler == 2:
            print('cosin decay.')
            lrs=CosineAnnealingScheduler(T_max=epochs, eta_max=max_lr, eta_min=0.0001, verbose=1)
            callbacks_list = callbacks_list + [lrs]
        elif args.lr_scheduler == 3:
            print('exp_range decay.')
            lrs = CyclicLR(mode='exp_range',base_lr=lr,max_lr=max_lr, gamma=0.99994)
            callbacks_list = callbacks_list + [lrs]
        elif args.lr_scheduler == 4:
            print('exp_range cycle decay.')
            lrs = CyclicLR(mode='exp_range', base_lr=lr, max_lr=max_lr, gamma=0.99994,scale_mode='cycle')
            callbacks_list = callbacks_list + [lrs]
        elif args.lr_scheduler == 5:
            print('triangular2 cycle decay.')
            lrs = CyclicLR(mode='triangular2', base_lr=lr, max_lr=max_lr, gamma=0.99994,scale_mode='cycle')
            callbacks_list = callbacks_list + [lrs]
        elif args.lr_scheduler ==6:
            lrs = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=6, verbose=1, mode='min',
                                         min_delta=0.0001, cooldown=0, min_lr=0.00005)
            callbacks_list = callbacks_list + [lrs]
        else:
            pass

    model.fit_generator(train_gen, steps_per_epoch=sample_num//batch_size, epochs=epochs, validation_data=val_data,
                   callbacks=callbacks_list, max_queue_size=20, shuffle=True, use_multiprocessing=False, workers=1)
