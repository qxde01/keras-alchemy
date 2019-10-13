import numpy as np
import  math,gc,random
import tensorflow as tf
import skimage,cv2
if tf.__version__<'2.0':
    import keras
else:
    from tensorflow import keras


def Shparpen(image):
    k=np.random.randint(8,12,1)[0]
    #print('    sharpen:',k)
    kernel=np.array([[-1,-1,-1],[-1,k,-1],[-1,-1,-1]])
    return cv2.filter2D(image,-1,kernel=kernel)

def Excessive(image):
    k = -np.random.randint(6, 9, 1)[0]
    #print('    Excessive:',k)
    kernel = np.array([[1,1,1], [1,k,1], [1,1,1]])
    return cv2.filter2D(image, -1, kernel=kernel)

def EdgeEnhance(image):
    #generating the kernels
    #print('     EdgeEnhance')
    kernel = np.array([[-1,-1,-1,-1,-1],
                               [-1,2,2,2,-1],
                               [-1,2,8,2,-1],
                               [-2,2,2,2,-1],
                               [-1,-1,-1,-1,-1]])/8.0
    return cv2.filter2D(image, -1, kernel=kernel)


def random_crop(image, crop_shape, padding=None):
    oshape = np.shape(image)
    if padding:
        oshape = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)
        npad = ((padding, padding), (padding, padding), (0, 0))
        #print('   ', oshape,npad)
        image_pad = np.lib.pad(image, pad_width=npad, mode='constant', constant_values=0)
        nh = np.random.randint(0, oshape[0] - crop_shape[0])
        nw = np.random.randint(0, oshape[1] - crop_shape[1])
        image_crop = image_pad[nh:nh + crop_shape[0], nw:nw + crop_shape[1]]
        return image_crop
    else:
        # print("WARNING!!! nothing to do!!!")
        return image


def AugImage(image, size=224):
    kidx = np.random.randint(0, 9, 1)[0]
    h, w = image.shape[:2]
    #print('   >>>>',kidx)
    if kidx == 0 or kidx==1:
        padding =  int(h* np.random.randint(40, 125,1)/1000.)
        crop_shape = (h - padding, w - padding)
        image = random_crop(image, crop_shape=crop_shape, padding=padding)
        image = cv2.resize(image, (size, size))
    elif kidx == 2:
        image = skimage.util.random_noise(image, mode='gaussian',mean=0,var=0.001)
        image= np.array(image*255,dtype=np.uint8)
    elif kidx == 3:
        sigma = np.random.randint(1, 9, 1)[0]
        if sigma % 2 == 0:
            sigma = sigma + 1
        image = cv2.GaussianBlur(image, ksize=(sigma, sigma), sigmaX=0, sigmaY=0)
    elif kidx == 4:
        image = np.fliplr(image)
    elif kidx == 5:
        center = cv2.getRotationMatrix2D((w / 2, h / 2), np.random.randint(5, 90, 1), 1)
        image = cv2.warpAffine(image, center, (w, h))
    elif kidx==6:
        image=Shparpen(image)
    elif kidx==7:
        image=Excessive(image)
    elif kidx==8:
        image=EdgeEnhance(image)
    else:
        image = image
    return image


def preprocess_input(x):
    if x.dtype not in ['float32', 'float64', 'float']:
        x = x.astype(np.float32)
    x /= 127.5
    x -= 1.
    return x

def create_pairs(x, digit_indices,num_classes):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
    for d in range(num_classes):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, num_classes)
            dn = (d + inc) % num_classes
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1., 0.]
    return np.array(pairs), np.array(labels)


class DataGenerator(keras.utils.Sequence):
#class DataGenerator:
    def __init__(self,usage,pairs,y, batch_size=1,size=256, shuffle=True):
        self.usage=usage
        self.batch_size = batch_size
        self.size=size
        self.pairs=pairs
        self.y = y
        #print('----------------')
        #print(self.pairs.shape)
        self.indexes = np.arange(len(self.pairs))
        self.shuffle = shuffle
        print(len(self.indexes))
        #print(self.std)

    def __len__(self):
        # 计算每一个epoch的迭代次数
        return math.ceil(len(self.indexes) / float(self.batch_size))

    def __getitem__(self, index):
        # 生成每个batch数据，这里就根据自己对数据的读取方式进行发挥了
        # 生成batch_size个索引
        batch_indexs = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # 根据索引获取datas集合中的数据
        batch_images = self.pairs[batch_indexs]
        Y=None
        X1=X2=None
        Y=self.y[batch_indexs]
        # 生成数据
        #print('Y',len(Y))
        if self=='train':
            X1 = self.process_batch(batch_images[:, 0])
            X2 = self.process_batch(batch_images[:, 1])
        else:
            X1=preprocess_input(batch_images[:, 0])
            X2 = preprocess_input(batch_images[:, 1])

        return [X1,X2],Y

    def on_epoch_end(self):
        # 在每一次epoch结束是否需要进行一次随机，重新随机一下index
        #print('epoch.....')
        gc.collect()
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    def process_batch(self,batch_images):
        #print('batch')
        m = batch_images.shape[0]
        arr = np.zeros((m, self.size, self.size, 3), dtype=np.float32)
        for line in range(0, m):
            img=batch_images[line]
            img=AugImage(img,size=self.size)
            arr[line, :, :, :] = preprocess_input(img)
        return arr



class DataGenerator_classify(keras.utils.Sequence):
#class DataGenerator:
    def __init__(self, x_train,y_train, batch_size=1,size=256, shuffle=True,num_classes=1108):
        self.batch_size = batch_size
        self.size=size
        self.x_train = x_train
        self.y_train=y_train
        self.indexes = np.arange(self.x_train.shape[0])
        self.shuffle = shuffle
        self.num_classes=num_classes
        #self.mean=[float(self.x_train[:,:,:,0].mean()),float(self.x_train[:,:,:,1].mean()),float(self.x_train[:,:,:,2].mean())]
        #self.std = [float(self.x_train[:, :, :, 0].std()), float(self.x_train[:, :, :, 1].std()), float(self.x_train[:, :, :, 2].std())]
        #print(self.mean)
        #print(self.std)

    def __len__(self):
        # 计算每一个epoch的迭代次数
        return math.ceil(len(self.indexes) / float(self.batch_size))

    def __getitem__(self, index):
        # 生成每个batch数据，这里就根据自己对数据的读取方式进行发挥了
        # 生成batch_size个索引
        batch_indexs = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # 根据索引获取datas集合中的数据
        batch_images = self.x_train[batch_indexs]
        y=None
        X=None
        y=self.y_train[batch_indexs]
        # 生成数据
        X = self.process_batch(batch_images)
        return X, y

    def on_epoch_end(self):
        # 在每一次epoch结束是否需要进行一次随机，重新随机一下index
        #print('epoch.....')
        gc.collect()
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    def process_batch(self,batch_images):
        m = batch_images.shape[0]
        arr = np.zeros((m, self.size, self.size, 3), dtype=np.float32)
        for line in range(0, m):
            img=batch_images[line]
            img=AugImage(img,size=self.size)
            arr[line, :, :, :] = preprocess_input(img)
        return arr

def smooth_labels(y, smooth_factor):
    '''Convert a matrix of one-hot row-vector labels into smoothed versions.
    # Arguments
        y: matrix of one-hot row-vector labels to be smoothed
        smooth_factor: label smoothing factor (between 0 and 1)
    # Returns
        A matrix of smoothed labels.
    '''
    assert len(y.shape) == 2
    if 0 <= smooth_factor <= 1:
        # label smoothing ref: https://www.robots.ox.ac.uk/~vgg/rg/papers/reinception.pdf
        y *= 1 - smooth_factor
        y += smooth_factor / y.shape[1]
    else:
        raise Exception(
            'Invalid label smoothing factor: ' + str(smooth_factor))
    return y
