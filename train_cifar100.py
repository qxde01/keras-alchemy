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
from siamese_models import build_model,contrastive_loss,covert2cassify
from siamese_data_generator import DataGenerator,create_pairs,preprocess_input
from siamese_data_generator import DataGenerator_classify,smooth_labels
from warmup import LRTensorBoard,WarmUpLearningRateScheduler,CosineAnnealingScheduler,CyclicLR

global init_lr

#def schler(epoch,init_lr=0.02,apha=0.01):
#    return init_lr/(1+apha*epoch)

def scheduler(epoch,init_lr=0.01,apha=0.01):
    if epoch<=40:
        lr=init_lr / np.sqrt((1 + apha * epoch))
    elif epoch>40 and epoch <=80:
        lr=0.5*init_lr / np.sqrt(1 + apha * (epoch - 39))
    elif epoch>80 and epoch <=120:
        lr=0.25*init_lr / np.sqrt(1 + apha * (epoch-79))
    else:
        lr=0.125 * init_lr /np.sqrt(1 + apha * (epoch - 119) )
    print('epoch:',epoch,'lr:',lr)
    return lr


def dacc(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_test = preprocess_input(x_test)
input_shape = x_train.shape[1:]
num_classes = len(np.unique(y_train))

print(num_classes)



def get_args():
    parser = argparse.ArgumentParser(description="cifar100 train.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--net", "-net",type=str,  help=" net type",default='MobileNetV2')
    parser.add_argument("--siamese_weights", "-sw", type=str, default=None, help="siamese_weights path ")
    parser.add_argument("--batch_size","-batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--epochs","-epochs", type=int, default=500, help="number of epochs")
    parser.add_argument('--classify_weights',"-cw",type=str,default=None,help=' softmax model weights  h5 file')
    parser.add_argument("--convert", '-cv', type=int, default=None, help="convert to softmax")
    parser.add_argument("--share", '-share', type=int, default=0, help="share=0,share siamese network,share=1,pseudo-siamese network")
    parser.add_argument("--learning_rate",'-lr', type=float,default=0.01, help="learning_rate")
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
    epochs = args.epochs
    batch_size = args.batch_size
    size=32
    opt = RAdam(lr=lr, weight_decay=0.00005, decay=0.0, min_lr=0.00001)
    print('build model:',net)
    if args.share == 0:
        model = build_model(net=net, input_shape=input_shape, siamese_weights=args.siamese_weights, share=True)
    else:
        model = build_model(net=net, input_shape=input_shape, siamese_weights=args.siamese_weights, share=False)

    if args.convert is  None:
        filepath = 'saved/siam_%s_{val_loss:.4f}-{val_dacc:.4f}-{epoch:03d}.h5' % net
        model.compile(loss=[contrastive_loss], optimizer=opt, metrics=[dacc])
        print('train siamese net.')
        digit_indices = [np.where(y_train == i)[0] for i in range(num_classes)]
        tr_pairs, tr_y = create_pairs(x_train, digit_indices, num_classes)
        print(tr_pairs.shape)
        digit_indices = [np.where(y_test == i)[0] for i in range(num_classes)]
        te_pairs, te_y = create_pairs(x_test, digit_indices, num_classes)
        print(te_pairs.shape)
        sample_num = tr_pairs.shape[0]
        del (x_train, x_test)
        train_gen = DataGenerator('train', tr_pairs, tr_y, batch_size=batch_size, size=32)
        val_data = ([te_pairs[:, 0], te_pairs[:, 1]], te_y)
    else:
        print('convert siamese net to softmax net')
        filepath = 'saved/classify_%s_{val_loss:.4f}-{val_accuracy:.4f}-{epoch:03d}.h5' % net
        if args.classify_weights is not None:
            model = covert2cassify(model, num_classes,trainable=True)
            print('load classify weights:',args.classify_weights)
            model.load_weights(args.classify_weights)
        else:
            if args.convert==0:
                print('Freeze siamese pretrained layers.')
                model = covert2cassify(model, num_classes)
            else:
                print('Train all layers.' )
                model = covert2cassify(model, num_classes, trainable=True)

        model.compile(loss=["categorical_crossentropy"], optimizer=opt, metrics=['accuracy'])
        sample_num = x_train.shape[0]
        y_train = smooth_labels(keras.utils.to_categorical(y_train, num_classes=num_classes), 0.1)
        y_test = smooth_labels(keras.utils.to_categorical(y_test, num_classes=num_classes), 0.1)
        train_gen = DataGenerator_classify(x_train=x_train, y_train=y_train, batch_size=batch_size, size=size, shuffle=True, num_classes=num_classes)
        val_data = (x_test, y_test)
    model.summary()
    checkpoint=keras.callbacks.ModelCheckpoint(filepath,save_best_only=False,verbose=1,monitor='val_loss',save_weights_only=True)
    callbacks_list=[checkpoint,LRTensorBoard(log_dir='logs')]
    if args.warmup>0:
        warmup_epoch = args.warmup
        warmup_batches = warmup_epoch * sample_num // batch_size+1
        print('warmup_batches:',warmup_batches)
        warm_up_lr = WarmUpLearningRateScheduler(init_lr=lr, warmup_batches=warmup_batches)
        callbacks_list=callbacks_list+[warm_up_lr]
    if args.lr_scheduler is not None:
        if args.lr_scheduler == 1:
            #print('1/sqrt(1+apha*epoch)')
            lrs= keras.callbacks.LearningRateScheduler(scheduler)
        elif args.lr_scheduler == 2:
            print('cosin decay.')
            lrs=CosineAnnealingScheduler(T_max=epochs, eta_max=lr, eta_min=0.0001, verbose=1)
        elif args.lr_scheduler == 3:
            lrs = CyclicLR(mode='exp_range',base_lr=lr,max_lr=3*lr, gamma=0.99994)
        elif args.lr_scheduler == 4:
            lrs = CyclicLR(mode='exp_range', base_lr=lr, max_lr=3 * lr, gamma=0.99994,scale_mode='cycle')
        else:
            pass
        callbacks_list = callbacks_list+[lrs]
    #del (x_train, x_test)

    model.fit_generator(train_gen, steps_per_epoch=sample_num//batch_size, epochs=epochs, validation_data=val_data,
                    callbacks=callbacks_list, max_queue_size=20, shuffle=True, use_multiprocessing=False, workers=1)
