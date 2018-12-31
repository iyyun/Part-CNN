'''

tensorflow version r1.4

created by Inyong yun, 2018.05.26
Copyright (c) 2018 DSPL Sungkyunkwan Univ

'''
import cv2 as cv
import os
import glob
from sklearn.utils import shuffle
import numpy as np
import random


def load_cls_information(f_path, classes):
    images = []  # image names
    labels = []  # labels (number)

    print('read image information!')
    for fields in classes:
        index = classes.index(fields)
        #if index > 2:
        #    path = os.path.join(f_path, 'part/%s' % fields, '*g')
        #else:
        path = os.path.join(f_path, fields, '*g')

        files = glob.glob(path)
        for fi in files:
            if '.png' not in fi:
                continue
            images.append(fi)
            label = np.zeros(len(classes))
            label[index] = 1.0
            labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    print 'size = ', len(images)
    print 'first image name  = ', images[0]
    print 'first image label = ', labels[0]

    return images, labels


class DataSet(object):

    def __init__(self, mode, images, labels):
        self._mode = mode
        self._num_examples = images.shape[0]

        self._images = images
        self._labels = labels
        self._epochs = 0
        self._index_in_epoch = 0
        self._hflip = ['True', 'False']

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def index_in_epoch(self):
        return self._index_in_epoch

    @property
    def epoch_done(self):
        return self._epochs

    def open_data(self, start_index, end_index, isflip, width = 96, height = 160, isResize = False):
        imgs = []  # image names
        for i in range(start_index, end_index):
            img = cv.imread(self._images[i]) - np.array([[[102.9801, 115.9465, 122.7717]]])
            if isResize:
                img = cv.resize(img, (width, height), interpolation=cv.INTER_LINEAR)
            if isflip: # horizontal flip
                img = cv.flip(img, 1)
            img = img.astype(np.float32)
            img = np.multiply(img, 1.0 / 255.0)
            # full
            #x1 = random.randint(0, 32)
            #y1 = random.randint(0, 32)
            #x2 = x1 + 64
            #y2 = y1 + 128

            # part
            x1 = random.randint(0, 8) # 64 + 8
            y1 = random.randint(0, 8) # 64 + 8
            x2 = x1 + 64
            y2 = y1 + 64

            part_img = img[y1:y2, x1:x2]
            imgs.append(part_img)
        imgs = np.array(imgs)
        return imgs

    def next_batch(self, batch_size, width=96, height=160, resize=False):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            self._epochs += 1
            start = 0
            self._index_in_epoch = batch_size

        end = self._index_in_epoch

        if self._mode == 'train':
            idx = random.randint(0, 1)
            h_flip = self._hflip[idx]
        else:
            h_flip = False

        # read image
        imgs = self.open_data(start, end, h_flip, width=width, height=height, isResize=resize)

        return imgs, self._labels[start:end]


def get_db(path, classes, mode='train', validation_size=0.2):
    class DataSets(object):
        pass

    data_set = DataSets()
    random.seed(1)
    images, labels = load_cls_information(path, classes)
    images, labels = shuffle(images, labels)

    if mode == 'train':
        if isinstance(validation_size, float):
            validation_size = int(validation_size * images.shape[0])

        validation_images = images[:validation_size]
        validation_labels = labels[:validation_size]

        train_images = images[validation_size:]
        train_labels = labels[validation_size:]

        data_set.train = DataSet(mode, train_images, train_labels)
        data_set.valid = DataSet(mode, validation_images, validation_labels)

        return data_set

    data_set.test = DataSet(mode, images, labels)

    return data_set