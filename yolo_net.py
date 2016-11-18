# -*- coding: utf-8 -*-
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import serializers
import numpy as np
from chainer  import cuda
import math

def overlap (x1, w1, x2, w2):
    l1 = x1 - w1/2
    l2 = x2 - w2/2
    left = l1 if l1 > l2 else l2

    r1 = x1 + w1/2
    r2 = x2 + w2/2
    right = r1 if r1 > r2 else r2
    return right - left

def box_intersection(box_a, box_b):
    #print(box_a, box_b)
    w = overlap(box_a[0], box_a[2], box_b[0], box_b[2])
    h = overlap(box_a[1], box_a[3], box_b[1], box_b[3])
    if w == 0 or h == 0:
        return 0
    return h*w

def box_union(box_a, box_b):
    box_a, box_b = box_a.data, box_b.data 
    i = box_intersection(box_a, box_b)
    u = box_a[2]*box_a[3] + box_b[2]*box_b[3] - i
    return u

def box_rmse(box_a, box_b):
    box_a, box_b = box_a.data, box_b.data
    a = math.sqrt(sum((box_a - box_b)**2))
    return a

class YOLO(chainer.Chain):
    insize = 448

    def __init__(self, divide_num, B_num, class_num=20, coord=5, no_obj=0.5):
        self.train = False
        self.pre_train = False

        self.loss_func = YoloLoss(divide_num, B_num, class_num, no_obj, coord)
        self.coord = coord
        self.no_obj = no_obj
        self.B_num = B_num
        self.class_num = class_num
        self.divide_num = divide_num
        self.vec_len = 5 * B_num + class_num

        super(YOLO, self).__init__(
            conv1 =  L.Convolution2D(3,  64, 7, stride=2, pad=3),
            conv2 = L.Convolution2D(64, 192,  3, pad=1),
            conv3 = L.Convolution2D(192, 128,  1),
            conv4 = L.Convolution2D(128, 256,  3, pad=1),
            conv5 = L.Convolution2D(256, 256,  1),
            conv6 = L.Convolution2D(256, 512,  3, pad=1),

            conv7 = L.Convolution2D(512, 256,  1),
            conv8 = L.Convolution2D(256, 512,  3, pad=1),
            conv9 = L.Convolution2D(512, 256,  1),
            conv10 = L.Convolution2D(256, 512,  3, pad=1),
            conv11 = L.Convolution2D(512, 256,  1),
            conv12 = L.Convolution2D(256, 512,  3, pad=1),
            conv13 = L.Convolution2D(512, 256,  1),
            conv14 = L.Convolution2D(256, 512,  3, pad=1),
            conv15 = L.Convolution2D(512, 512,  1),
            conv16 = L.Convolution2D(512, 1024, 3, pad=1),

            conv17 = L.Convolution2D(1024, 512,  1),
            conv18 = L.Convolution2D(512, 1024,  3, pad=1),
            conv19 = L.Convolution2D(1024, 512,  1),
            conv20 = L.Convolution2D(512, 1024,  3, pad=1),
            conv21 = L.Convolution2D(1024, 1024,  3, pad=1),#ここまででImageNet
            conv22 = L.Convolution2D(1024, 1024,  3, stride=2, pad=1),
            conv23 = L.Convolution2D(1024, 1024,  3, pad=1),
            conv24 = L.Convolution2D(1024, 1024,  3, pad=1),

            fc25 = L.Linear(50176, 4096),
            fc26 = L.Linear(4096, self.vec_len * 49),

            #fc_pre = L.Linear(16384,1000),
            #fc_pre = L.Linear(50176,1000),
        )

    def __call__(self, x, t):
        h = F.max_pooling_2d(F.leaky_relu(self.conv1(x)),2,2)
        h = F.max_pooling_2d(F.leaky_relu(self.conv2(h)),2,2)
        h = F.leaky_relu(self.conv3(h))
        h = F.leaky_relu(self.conv4(h))
        h = F.leaky_relu(self.conv5(h))
        h = F.max_pooling_2d(F.leaky_relu(self.conv6(h)),2,2)


        h = F.leaky_relu(self.conv7(h))
        h = F.leaky_relu(self.conv8(h))
        h = F.leaky_relu(self.conv9(h))
        h = F.leaky_relu(self.conv10(h))
        h = F.leaky_relu(self.conv11(h))
        h = F.leaky_relu(self.conv12(h))
        h = F.leaky_relu(self.conv13(h))
        h = F.leaky_relu(self.conv14(h))
        h = F.leaky_relu(self.conv15(h))
        h = F.max_pooling_2d(F.leaky_relu(self.conv16(h)),2,2)


        h = F.leaky_relu(self.conv17(h))
        h = F.leaky_relu(self.conv18(h))
        h = F.leaky_relu(self.conv19(h))

        if self.pre_train:
            h = F.average_pooling_2d(h, 2, 2)
            h = self.fc_pre(h)
            self.loss = F.softmax_cross_entropy(h, t)
            self.accuracy = F.accuracy(h, t)
            return self.loss
        else:
            h = F.leaky_relu(self.conv20(h))
            h = F.leaky_relu(self.conv21(h))
            h = F.leaky_relu(self.conv22(h))
            h = F.leaky_relu(self.conv23(h))
            h = F.leaky_relu(self.conv24(h))
            self.h = h
            h = F.leaky_relu(self.fc25(h))
            h = F.relu(self.fc26(h))
            #self.loss = self.loss_func(h, t)
            #self.accuracy = self.loss
            self.img = (x, h)
            #return self.loss

class YoloLoss:
    def __init__(self, divide_num, B_num, class_num=20, no_obj=0.5, coord=5):
        self.coord = coord
        self.no_obj = no_obj
        self.B_num = B_num
        self.class_num = class_num
        self.divide_num = divide_num
        self.vec_len = 5 * B_num + class_num

    def __call__(self, x, t):
        #(C1,x1,y1,w1,h1,C2,x2,y2,w2,h2,classes)
        xp = cuda.get_array_module(x)
        best_iou = 0
        best_index = 0
        best_rmse = 10000
        loss = chainer.Variable(xp.array(0, dtype=x.data.dtype.type))
        for batch in range(x.shape[0]):
            for i in range(self.divide_num**2):
                target_x = x[batch, i]
                target_t = t[batch, i]
                conf_t = target_t[0]
                if conf_t.data == 0: #no response
                    for b in range(self.B_num):
                        loss+= self.no_obj * (target_x[b*5])**2
                    continue
                class_t = target_t[-self.class_num]
                class_x = target_x[-self.class_num]
                coord_t = target_t[1:5]
                for b in range(self.B_num):
                    coord_x = target_x[b * 5 + 1:b * 5 + 5]
                    iou = box_union(coord_t, coord_x)
                    if best_iou > 0 or iou > 0:
                        if iou > best_iou:
                            best_index = b
                            best_iou = iou
                    else:
                        rmse = box_rmse(coord_t, coord_x)
                        if rmse < best_rmse:
                            best_index = b
                            best_rmse = rmse
                coord_best = target_x[b * 5 + 1:b * 5 + 5]
                for b in range(self.B_num):

                    if b != best_index:
                        loss += F.sum((coord_t - target_x[b * 5 + 1:b * 5 + 5])**2)
                        loss += F.sum((conf_t - target_x[b * 5])**2)
                loss += F.sum((coord_t - coord_best)**2 * self.coord)
                loss += F.sum((class_t - class_x)**2)
            loss /= len(x.data)
            return loss

a = YoloLoss(7, 2)
b = chainer.Variable(np.ones([10,49,25]))
c = chainer.Variable(np.ones([10,49,30]))
a(b, c)
