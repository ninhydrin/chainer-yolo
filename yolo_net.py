import chainer
import chainer.functions as F
import chainer.links as L
from chainer import serializers
from multi_part_loss import *
import numpy as np
class Yolo(chainer.Chain):

    insize = 448

    def __init__(self, divide_num, B_num, batch_size, class_num=20):
        super(Yolo, self).__init__()
        self.trunk = [
            ("conv1", L.Convolution2D(3,  64, 7, stride=2, pad=1)),
            ('relu1', F.LeakyReLU(slope=0.2)),
            ('pool1', F.MaxPooling2D(2, 2)),

            ("conv2", L.Convolution2D(64, 192,  3, stride=2, pad=1)),
            ('relu2', F.LeakyReLU(slope=0.2)),
            ('pool2', F.MaxPooling2D(2, 2)),

            ("conv3-1", L.Convolution2D(192, 128,  1, pad=1)),
            ('relu3-1', F.LeakyReLU(slope=0.2)),
            ("conv3-2", L.Convolution2D(128, 256,  3, stride=2, pad=1)),
            ('relu3-2', F.LeakyReLU(slope=0.2)),
            ("conv3-3", L.Convolution2D(256, 128,  1, pad=1)),
            ('relu3-3', F.LeakyReLU(slope=0.2)),
            ("conv3-4", L.Convolution2D(128, 256,  3, pad=1)),
            ('relu3-4', F.LeakyReLU(slope=0.2)),
            ("conv3-5", L.Convolution2D(256, 128,  1, pad=1)),
            ('relu3-5', F.LeakyReLU(slope=0.2)),
            ("conv3-6", L.Convolution2D(128, 512,  3, stride=2, pad=1)),
            ('relu3-6', F.LeakyReLU(slope=0.2)),
            ('pool3', F.MaxPooling2D(2, 2)),

            ("conv4-1-1", L.Convolution2D(512, 256,  1, pad=1)),
            ('relu4-1-1', F.LeakyReLU(slope=0.2)),
            ("conv4-1-2", L.Convolution2D(256, 512,  3, pad=1)),
            ('relu4-1-2', F.LeakyReLU(slope=0.2)),
            ("conv4-2-1", L.Convolution2D(512, 256,  1, pad=1)),
            ('relu4-2-1', F.LeakyReLU(slope=0.2)),
            ("conv4-2-2", L.Convolution2D(256, 512,  1, pad=1)),
            ('relu4-2-2', F.LeakyReLU(slope=0.2)),
            ("conv4-3-1", L.Convolution2D(512, 256,  1, pad=1)),
            ('relu4-3-1', F.LeakyReLU(slope=0.2)),
            ("conv4-3-2", L.Convolution2D(256, 512,  3, pad=1)),
            ('relu4-3-2', F.LeakyReLU(slope=0.2)),
            ("conv4-4-1", L.Convolution2D(512, 256,  1, pad=1)),
            ('relu4-4-1', F.LeakyReLU(slope=0.2)),
            ("conv4-4-2", L.Convolution2D(256, 512,  3, pad=1)),
            ('relu4-4-2', F.LeakyReLU(slope=0.2)),
            ("conv4-5", L.Convolution2D(512, 256,  1, pad=1)),
            ('relu4-5', F.LeakyReLU(slope=0.2)),
            ("conv4-6", L.Convolution2D(256, 1024,  3, stride=2, pad=1)),
            ('relu4-6', F.LeakyReLU(slope=0.2)),
            ('pool4', F.MaxPooling2D(2, 2)),

            ("conv5-1-1", L.Convolution2D(1024, 512,  1, pad=1)),
            ('relu5-1-1', F.LeakyReLU(slope=0.2)),
            ("conv5-1-2", L.Convolution2D(512, 1024,  3, pad=1)),
            ('relu5-1-2', F.LeakyReLU(slope=0.2)),
            ("conv5-2-1", L.Convolution2D(1024, 1024,  3, pad=1)),
            ('relu5-2-1', F.LeakyReLU(slope=0.2)),
            ("conv5-2-2", L.Convolution2D(1024, 1024,  3, stride=2, pad=1)),
            ('relu5-2-2', F.LeakyReLU(slope=0.2)),

            ("conv6-1", L.Convolution2D(1024, 1024,  3, pad=1)),
            ('relu6-1', F.LeakyReLU(slope=0.2)),
            ("conv6-2", L.Convolution2D(1024, 512,  3, pad=1)),
            ('relu6-2', F.LeakyReLU(slope=0.2)),
            ("fc1", L.Linear(8192, 2096)),
            ('relu-fc1', F.LeakyReLU(slope=0.2)),
            ("fc2", L.Linear(2096, divide_num*divide_num*(B_num*5 + class_num))),
            ('relu-fc2', F.LeakyReLU(slope=0.2)),

        ]
        for name, link in self.trunk:
            if 'conv' in name:
                self.add_link(name, link)
        self.batch_size = batch_size
        self.B_num = B_num
        self.class_num = class_num
        self.divide_num = divide_num
        self.train = False
        self.fc1_dropout_rate=0.5

    def clear(self):
        self.loss = None
        self.accuracy = None

    def __call__(self, x, t):
        self.clear()
        for name, f in self.trunk:
            x = (getattr(self, name) if 'conv' in name else f)(x)
        x = F.reshape(x, (self.batch_size, self.B_num*5 + self.class_num, self.divide_num, self.divide_num))
        self.loss = F.mean_squared_error(x, t)
        self.accuracy = self.loss
        return self.loss


class YoloPreTrain(chainer.Chain):
    insize = 448

    def __init__(self):
        super(YoloPreTrain, self).__init__()
        self.trunk = [
            ("conv1", L.Convolution2D(3,  64, 7, stride=2, pad=1)),
            ('relu1', F.LeakyReLU(slope=0.2)),
            ('pool1', F.MaxPooling2D(2, 2)),

            ("conv2", L.Convolution2D(64, 192,  3, stride=2, pad=1)),
            ('relu2', F.LeakyReLU(slope=0.2)),
            ('pool2', F.MaxPooling2D(2, 2)),

            ("conv3-1", L.Convolution2D(192, 128,  1, pad=1)),
            ('relu3-1', F.LeakyReLU(slope=0.2)),
            ("conv3-2", L.Convolution2D(128, 256,  3, stride=2, pad=1)),
            ('relu3-2', F.LeakyReLU(slope=0.2)),
            ("conv3-3", L.Convolution2D(256, 128,  1, pad=1)),
            ('relu3-3', F.LeakyReLU(slope=0.2)),
            ("conv3-4", L.Convolution2D(128, 256,  3, pad=1)),
            ('relu3-4', F.LeakyReLU(slope=0.2)),
            ("conv3-5", L.Convolution2D(256, 128,  1, pad=1)),
            ('relu3-5', F.LeakyReLU(slope=0.2)),
            ("conv3-6", L.Convolution2D(128, 512,  3, stride=2, pad=1)),
            ('relu3-6', F.LeakyReLU(slope=0.2)),
            ('pool3', F.MaxPooling2D(2, 2)),

            ("conv4-1-1", L.Convolution2D(512, 256,  1, pad=1)),
            ('relu4-1-1', F.LeakyReLU(slope=0.2)),
            ("conv4-1-2", L.Convolution2D(256, 512,  3, pad=1)),
            ('relu4-1-2', F.LeakyReLU(slope=0.2)),
            ("conv4-2-1", L.Convolution2D(512, 256,  1, pad=1)),
            ('relu4-2-1', F.LeakyReLU(slope=0.2)),
            ("conv4-2-2", L.Convolution2D(256, 512,  1, pad=1)),
            ('relu4-2-2', F.LeakyReLU(slope=0.2)),
            ("conv4-3-1", L.Convolution2D(512, 256,  1, pad=1)),
            ('relu4-3-1', F.LeakyReLU(slope=0.2)),
            ("conv4-3-2", L.Convolution2D(256, 512,  3, pad=1)),
            ('relu4-3-2', F.LeakyReLU(slope=0.2)),
            ("conv4-4-1", L.Convolution2D(512, 256,  1, pad=1)),
            ('relu4-4-1', F.LeakyReLU(slope=0.2)),
            ("conv4-4-2", L.Convolution2D(256, 512,  3, pad=1)),
            ('relu4-4-2', F.LeakyReLU(slope=0.2)),
            ("conv4-5", L.Convolution2D(512, 256,  1, pad=1)),
            ('relu4-5', F.LeakyReLU(slope=0.2)),
            ("conv4-6", L.Convolution2D(256, 1024,  3, stride=2, pad=1)),
            ('relu4-6', F.LeakyReLU(slope=0.2)),
            ('pool4', F.MaxPooling2D(2, 2)),

            ("conv5-1-1", L.Convolution2D(1024, 512,  1, pad=1)),
            ('relu5-1-1', F.LeakyReLU(slope=0.2)),
            ("conv5-1-2", L.Convolution2D(512, 1024,  3, pad=1)),
            ('relu5-1-2', F.LeakyReLU(slope=0.2)),

            ("average_pool", F.AveragePooling2D(2)),
            ("fc", L.Linear(9216,1000))
        ]
        for name, link in self.trunk:
            if 'conv' in name or "fc" in name:
                self.add_link(name, link)
        self.train = False

    def clear(self):
        self.loss = None
        self.accuracy = None

    def __call__(self, x, t):
        self.clear()
        for name, f in self.trunk:
            x = (getattr(self, name) if 'conv' in name else f)(x)
        self.loss = F.softmax_cross_entropy(x, t)
        self.accuracy = F.accuracy(x, t)
        return self.loss

class Alex2(chainer.Chain):
    """Single-GPU AlexNet without partition toward the channel axis."""

    insize = 448

    def __init__(self):
        super(Alex2, self).__init__()
        self.trunk=[
            ("conv1", L.Convolution2D(3,  96, 11, stride=4)),
            ('relu1', F.ReLU()),
            ("lrn1", F.LocalResponseNormalization(alpha=0.00002, k=1)),
            ("pool1", F.MaxPooling2D(3, stride=2)),
            ("conv2", L.Convolution2D(96, 256,  5, pad=2)),
            ('relu2', F.ReLU()),
            ("lrn2",F.LocalResponseNormalization(alpha=0.00002, k=1)),
            ("pool2", F.MaxPooling2D(3, stride=2)),
            ("conv3", L.Convolution2D(256, 384,  3, pad=1)),
            ('relu3', F.ReLU()),
            ("conv4", L.Convolution2D(384, 384,  3, pad=1)),
            ('relu4', F.ReLU()),
            ("pool4", F.MaxPooling2D(3, stride=2)),
            ("conv5", L.Convolution2D(384, 256,  3, pad=1)),
            ('relu5', F.ReLU()),
            ("pool5", F.MaxPooling2D(3, stride=2)),
            ("fc6", L.Linear(9216, 4096)),
            ('relu6', F.ReLU()),
            ("drop6", F.Dropout(0.5)),
            ("fc7", L.Linear(4096, 4096)),
            ('relu7', F.ReLU()),
            ("drop7", F.Dropout(0.5)),
            ("fc8", L.Linear(4096, 1000))
        ]
        for name, link in self.trunk:
            if 'conv' in name or "fc" in name:
                self.add_link(name, link)

        self.train = False

    def clear(self):
        self.loss = None
        self.accuracy = None

    def __call__(self, x, t):
        self.clear()
        for name, f in self.trunk:
            x = (getattr(self, name) if 'conv' in name or "fc" in name else f)(x)
        self.loss = F.softmax_cross_entropy(x, t)
        self.accuracy = F.accuracy(x, t)
        return self.loss


class ValAlex(chainer.Chain):

    """Single-GPU AlexNet without partition toward the channel axis."""

    insize = 227

    def __init__(self):
        super(ValAlex, self).__init__(
            conv1=L.Convolution2D(3,  96, 11, stride=4),
            conv2=L.Convolution2D(96, 256,  5, pad=2),
            conv3=L.Convolution2D(256, 384,  3, pad=1),
            conv4=L.Convolution2D(384, 384,  3, pad=1),
            conv5=L.Convolution2D(384, 256,  3, pad=1),
            fc6=L.Linear(9216, 4096),
            fc7=L.Linear(4096, 4096),
            fc8=L.Linear(4096, 1000),
        )
        self.train = False
        self.fc6_dropout_rate=0.5
        self.fc7_dropout_rate=0.5

    def clear(self):
        self.loss = None
        self.accuracy = None

    def __call__(self, x, t):
        self.clear()
        h = F.max_pooling_2d(
            F.local_response_normalization(F.relu(self.conv1(x)),alpha=0.00002,k=1), 3, stride=2)
        h = F.max_pooling_2d(
            F.local_response_normalization(F.relu(self.conv2(h)),alpha=0.00002,k=1), 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        #h = F.max_pooling_2d(F.relu(self.conv4(h)), 3, stride=2)
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride=2)
        self.h1 = F.dropout(F.relu(self.fc6(h)), ratio=self.fc6_dropout_rate, train=self.train)
        self.h2 = F.dropout(F.relu(self.fc7(self.h1)), ratio=self.fc7_dropout_rate, train=self.train)
        h = self.fc8(self.h2)

        self.loss = F.softmax_cross_entropy(h, t)
        self.accuracy = F.accuracy(h, t)
        return self.loss

class Alex(chainer.Chain):

    """Single-GPU AlexNet without partition toward the channel axis."""

    insize = 227


    def __init__(self, divide_num, B_num, batch_size, class_num=20):
        super(Alex, self).__init__(
            conv1=L.Convolution2D(3,  96, 11, stride=4),
            conv2=L.Convolution2D(96, 256,  5, pad=2),
            conv3=L.Convolution2D(256, 384,  3, pad=1),
            conv4=L.Convolution2D(384, 384,  3, pad=1),
            conv5=L.Convolution2D(384, 256,  3, pad=1),
            fc1=L.Linear(9216, 4096),
            fc2=L.Linear(4096, (B_num*5+class_num)*divide_num*divide_num)
        )
        self.train = True

        self.batch_size = batch_size
        self.B_num = B_num
        self.class_num = class_num
        self.divide_num = divide_num

    def copy_yolo(self, model_path):
        model = ValAlex()
        serializers.load_npz(model_path, model)
        self.conv1.copyparams(model.conv1)
        self.conv2.copyparams(model.conv2)
        self.conv3.copyparams(model.conv3)
        self.conv4.copyparams(model.conv4)
        self.conv5.copyparams(model.conv5)

    def __call__(self, x, t):
        h = F.max_pooling_2d(
            F.local_response_normalization(F.relu(self.conv1(x)),alpha=0.00002,k=1), 3, stride=2)
        h = F.max_pooling_2d(
            F.local_response_normalization(F.relu(self.conv2(h)),alpha=0.00002,k=1), 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride=2)
        h = F.dropout(F.relu(self.fc1(h)), train=self.train)
        h = F.relu(self.fc2(h))
        h = F.reshape(h, (self.batch_size, self.B_num*5 + self.class_num, self.divide_num, self.divide_num))

        #self.loss = F.mean_squared_error(h, t)
        self.loss = multi_part_loss(h, t)
        self.accuracy = self.loss
        self.img = (x, h)
        return self.loss
"""
model = Alex(7,2,10)
from chainer import cuda
cuda.check_cuda_available()
cuda.get_device(0).use()
model.to_gpu()
xp = cuda.cupy
a = chainer.Variable(xp.array(np.random.random([10,3,227,227]).astype(np.float32)))
b = chainer.Variable(xp.array(np.random.random([10,30,7,7]).astype(np.float32)))
loss = model(a,b)
"""
