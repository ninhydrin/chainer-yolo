#!/usr/bin/env python
# coding: utf-8
import argparse
import datetime
import json
import multiprocessing
import os
import random
import sys
import threading
import time
import cv2
import numpy as np
import six
import six.moves.cPickle as pickle
from six.moves import queue
import pickle
import chainer
from chainer import cuda
from chainer import optimizers
from chainer import serializers


parser = argparse.ArgumentParser(
    description='Learning blackboard mask from ILSVRC2012 dataset')

parser.add_argument('--batchsize', '-B', type=int, default=50,
                    help='Learning minibatch size')
parser.add_argument('--val_batchsize', '-b', type=int, default=50,
                    help='Validation minibatch size')
parser.add_argument('--epoch', '-E', default=2000, type=int,
                    help='Number of epochs to learn')
parser.add_argument('--gpu', '-g', default=0, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--loaderjob', '-j', default=20, type=int,
                    help='Number of parallel data loading processes')
parser.add_argument('--out', '-o', default='model_2007',
                    help='Path to save model on each validation')
parser.add_argument('--outstate', '-s', default='state',
                    help='Path to save optimizer state on each validation')
parser.add_argument('--initmodel', '-m', default='',
                    help='Initialize the model from given file')
parser.add_argument('--resume', default='',
                    help='Resume the optimization from snapshot')
parser.add_argument('--year', default='2007',
                    help='Resume the optimization from snapshot')

args = parser.parse_args()

if args.gpu >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if args.gpu >= 0 else np

assert 50000 % args.val_batchsize == 0

year = args.year
image_path = "VOCdevkit/VOC{}/ImageSets/Main/".format(year)

#train_list = list(open("trainval_voc{}.pkl".format(year),"rb"))
#test_list = list(open("test_voc{}.pkl".format(year),"rb"))

import make_label
train_list, val_list= make_label.make_data(year)

# Prepare dataset
img_size = 227
img_pre_size = 256
divide_num = 7
class_num = 20
B_num = 2
vec_len = 30
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

import yolo_net
#model = yolo_net.Yolo(divide_num, B_num, args.batchsize)
model = yolo_net.Alex(divide_num, B_num, args.batchsize)
model.copy_yolo("alex_npz.model")
optimizer = optimizers.MomentumSGD(lr=0.001, momentum=0.9)
optimizer.setup(model)

if args.initmodel:
    print('Load model from', args.initmodel)
    serializers.load_hdf5(args.initmodel, model)
if args.resume:
    print('Load optimizer state from', args.resume)
    serializers.load_hdf5(args.resume, optimizer)

if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()

data_q = queue.Queue(maxsize=1)
res_q = queue.Queue()

def draw_bbox(img, data,name):
    flag = False
    """
    try:
        class_name = classes[vec[10:].argmax()]
    except ValueError as e:
        pass
    """
    for tate in data:
        for vec in tate:
            if vec[0]==0:
                continue
            h, w, _ = img.shape
            center_x = vec[1]*w
            center_y = vec[2]*h
            ww = vec[3]*w/2
            hh = vec[4]*h/2
            x1 = max(0,int(center_x - ww))
            y1 = max(0,int(center_y - hh))
            x2 = min(w-1, int(center_x + ww))
            y2 = min(h-1, int(center_y + hh))
            img = cv2.rectangle(img.copy(), (x1,y1),(x2,y2),(255,0,0), thickness=1)
            flag =True
            if vec[5]==0:
                continue
            center_x = vec[6]*w
            center_y = vec[7]*h
            ww = vec[8]*w/2
            hh = vec[9]*h/2
            x1 = max(0,int(center_x - ww))
            y1 = max(0,int(center_y - hh))
            x2 = min(w-1, int(center_x + ww))
            y2 = min(h-1, int(center_y + hh))
            img = cv2.rectangle(img.copy(), (x1,y1),(x2,y2),(0,255,0), thickness=1)
            #cv2.putText(image,text,(start_x,start_y),font, font size,color)
    if flag:
        cv2.imwrite(name, cv2.resize(img,(400,400)))

def read_image(path, is_val=False):
    img = cv2.resize(cv2.imread(path), (img_size, img_size))
    return img.transpose(2, 0, 1)[::-1]

    if random.randint(0, 1):
        return back_img[:, ::-1,:].transpose(2,0,1), black_img[:, ::-1]
    else:
        return back_img.transpose(2,0,1), black_img

def locate(x, y):
    px = int(x*divide_num)
    py = int(y*divide_num)
    return px, py

def make_ans(label):
    #vec = [(class or back,x,y,w,h)*B,classes]
    vec = []
    ans = np.zeros([divide_num, divide_num, vec_len],dtype=np.float32 )
    obj = {}
    random.shuffle(label)
    for num, prediction in enumerate(label):
        x,y = locate(prediction[1], prediction[2])
        obj_vec = ans[y, x] #画像は(y, x, 3)
        obj_vec[0] = 1 # confidence
        if sum(obj_vec[10:]) == 0: #まだオブジェクト無し
            obj_vec[0] = 1
            obj_vec[1:5]=prediction[1:]
        elif sum(obj_vec[10:]) == 1: #一つある
            obj_vec[5] = 1
            obj_vec[6:10]=prediction[1:]
        else: #すでに２つある場合、ランダムに入れるか消すかする
            pass
        obj_vec[10:][prediction[0]]=1 #class label
    return ans.transpose(2,0,1)

#sys.exit()
def feed_data():
    # Data feeder
    i = 0
    count = 0

    x_batch = np.ndarray((args.batchsize, 3, model.insize, model.insize), dtype=np.float32)
    y_batch = np.ndarray((args.batchsize, 30, divide_num, divide_num), dtype=np.float32)
    val_x_batch = np.ndarray((args.val_batchsize, 3, model.insize, model.insize), dtype=np.float32)
    val_y_batch = np.ndarray((args.val_batchsize, 30, divide_num, divide_num), dtype=np.float32)

    batch_pool = [None] * args.batchsize
    val_batch_pool = [None] * args.val_batchsize
    pool = multiprocessing.Pool(args.loaderjob)
    data_q.put('train')

    for epoch in six.moves.xrange(1, 1 + args.epoch):
        print('epoch', epoch, file=sys.stderr)
        print('learning rate', optimizer.lr, file=sys.stderr)
        perm = np.random.permutation(len(train_list))
        for idx in perm:
            path, label = train_list[idx]
            batch_pool[i] = pool.apply_async(read_image, (path, False))
            y_batch[i]= make_ans(label)
            i += 1
            if i == args.batchsize:
                for j, x in enumerate(batch_pool):
                    x_batch[j] = x.get()
                data_q.put((x_batch.copy(), y_batch.copy()))
                i = 0

            count += 1
            if count % 100000 == 0:
                data_q.put('val')
                j = 0
                for path, label in val_list[:100]:
                    val_batch_pool[j] = pool.apply_async(read_image, (path, True))
                    val_y_batch[j] = make_ans(label)
                    j += 1
                    if j == args.val_batchsize:
                        for k, x in enumerate(val_batch_pool):
                            val_x_batch[k] = x.get()
                        data_q.put((val_x_batch.copy(), val_y_batch.copy()))
                        j = 0
                data_q.put('train')

        serializers.save_hdf5(args.out+"epoch_"+str(epoch), model)
        serializers.save_hdf5(args.outstate+"epoch_"+str(epoch), optimizer)
        optimizer.lr *= 0.99
    pool.close()
    pool.join()
    data_q.put('end')

def log_result():
    train_count = 0
    train_cur_loss = 0
    train_cur_accuracy = 0
    begin_at = time.time()
    val_begin_at = None

    while True:
        result = res_q.get()
        if result == 'end':
            print(file=sys.stderr)
            break
        elif result == 'train':
            print(file=sys.stderr)
            train = True
            if val_begin_at is not None:
                begin_at += time.time() - val_begin_at
                val_begin_at = None
            continue
        elif result == 'val':
            print(file=sys.stderr)
            train = False
            val_count = val_loss = val_accuracy = 0
            val_begin_at = time.time()
            continue

        loss, img_list, img = result

        if train:
            del img
            train_count += 1
            duration = time.time() - begin_at
            throughput = train_count * args.batchsize / duration
            sys.stderr.write(
                '\rtrain {0} updates ({1} samples) time: {2} ({3:3.2} images/sec loss={4:.4})'
                .format(train_count, train_count * args.batchsize,
                        datetime.timedelta(seconds=duration), throughput, loss))

            train_cur_loss += loss
            if train_count % 1000 == 0:
                mean_loss = train_cur_loss / 1000
                print(file=sys.stderr)
                print(json.dumps({'type': 'train', 'iteration': train_count,
                                  'loss': mean_loss}))
                sys.stdout.flush()
                train_cur_loss = 0
                train_cur_accuracy = 0
        else:
            img, datas = img
            img.to_cpu()
            datas.to_cpu()
            mm = 0
            for im, da in zip(img,datas):
                im = im.data.transpose(1,2,0)
                da = da.data.transpose(1,2,0)
                draw_bbox(im,da,"ans/im{}.jpg".format(mm))
                mm+=1
            val_count += args.val_batchsize
            duration = time.time() - val_begin_at
            throughput = val_count / duration
            sys.stderr.write(
                '\rval   {0} batches ({1} samples) time: {2} ({3:3.3} images/sec)'
                .format(val_count / args.val_batchsize, val_count,
                        datetime.timedelta(seconds=duration), throughput))

            val_loss += loss
            if val_count == 50000:
                mean_loss = val_loss * args.val_batchsize / 50000
                print(file=sys.stderr)
                print(json.dumps({'type': 'val', 'iteration': train_count,
                                  'loss': mean_loss}))
                sys.stdout.flush()

def train_loop():
    while True:
        while data_q.empty():
            time.sleep(0.1)
        inp = data_q.get()
        if inp == 'end':  # quit
            res_q.put('end')
            break
        elif inp == 'train':  # restart training
            res_q.put('train')
            model.train = True
            continue
        elif inp == 'val':  # start validation
            res_q.put('val')
            serializers.save_hdf5(args.out, model)
            serializers.save_hdf5(args.outstate, optimizer)
            model.train = False
            continue

        volatile = 'off' if model.train else 'on'
        x = chainer.Variable(xp.asarray(inp[0]), volatile=volatile)
        t = chainer.Variable(xp.asarray(inp[1]), volatile=volatile)

        if model.train:
            optimizer.update(model, x, t)
        else:
            model(x, t)

        res_q.put((float(model.loss.data), model.accuracy, model.img))
        del x, t

# Invoke threads
feeder = threading.Thread(target=feed_data)
feeder.daemon = True
feeder.start()
logger = threading.Thread(target=log_result)
logger.daemon = True
logger.start()

train_loop()
feeder.join()
logger.join()

# Save final model
serializers.save_hdf5(args.out, model)
serializers.save_hdf5(args.outstate, optimizer)
