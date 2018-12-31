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

caltech_image_data_folder = ['set00', 'set01', 'set02', 'set03', 'set04', 'set05', 'set06', 'set07', 'set08', 'set09', 'set10']
caltech_image_data_count = [14, 5, 11, 12, 11, 12, 18, 11, 10, 11, 11]


def load_saliency_information(f_path, mode):
    images = []  # image names
    labels = []  # labels (number)

    start = 0
    end = 11

    print('read image information!')
    for field in caltech_image_data_folder[start:end]:
        index = caltech_image_data_folder.index(field)
        include_video_set_count = caltech_image_data_count[index]
        for i in range(include_video_set_count + 1):
            image_root = 'images/%s/V%03d' % (field, i)
            label_root = 'saliency/%s/V%03d' % (field, i)

            if mode == 'train':
                ipath = os.path.join(f_path, image_root)
                spath = os.path.join(f_path, label_root, '*g')
                files = glob.glob(spath)
            else:
                ipath = os.path.join(f_path, image_root, '*g')
                spath = os.path.join(f_path, label_root)
                files = glob.glob(ipath)
                files = np.sort(files)
                skip = 30

            ii = 0
            for fi in files:
                if '.png' not in fi:
                    continue
                if mode == 'train':
                    name = fi.split('/')
                    fl = os.path.join(ipath, name[-1])
                    images.append(fl)
                    labels.append(fi)
                else:

                    ii += 1
                    if ii == (skip-1):
                        images.append(fi)
                        skip += 30

    images = np.array(images)
    if mode == 'train':
        labels = np.array(labels)

    print 'size = ', len(images)
    print 'first image name  = ', images[0]
    if mode == 'train':
        print 'first saliency name = ', labels[0]
    print ('read image information - done!')

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

    def open_data(self, start_index, end_index, isflip, isResize=False, min_size=600, max_size=1500):
        imgs = []  # image
        sals = []  # saliency
        for i in range(start_index, end_index):
            img = cv.imread(self._images[i])
            if self._mode == 'train':
                sal = cv.imread(self._labels[i], 0)# gray scale
            if isResize:
                im_shape = img.shape
                im_size_min = np.min(im_shape[0:2])
                im_size_max = np.max(im_shape[0:2])
                scale = float(min_size) / float(im_size_min)
                if np.round(scale * im_size_max) > max_size:
                    scale = float(max_size) / float(im_size_max)
                img = cv.resize(img, None, None, fx=scale, fy=scale, interpolation=cv.INTER_LINEAR)

                if self._mode == 'train':
                    sal = cv.resize(sal, (769, 577), interpolation=cv.INTER_LINEAR)
                #sal = cv.resize(sal, None, None, fx=scale, fy=scale, interpolation=cv.INTER_LINEAR)

            if isflip: # horizontal flip
                img = cv.flip(img, 1)
                if self._mode == 'train':
                    sal = cv.flip(sal, 1)

            img = img.astype(np.float32)
            img = np.multiply(img, 1.0 / 255.0)
            imgs.append(img)
            if self._mode == 'train':
                sal = sal.astype(np.float32)
                sal = np.multiply(sal, 1.0 / 255.0)
                sals.append(sal)

        imgs = np.array(imgs)
        if self._mode == 'train':
            sals = np.array(sals)
            sals = np.expand_dims(sals, axis=4)

        return imgs, sals

    def next_batch(self, batch_size, resize=False):
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
        imgs, sals = self.open_data(start, end, h_flip, isResize=resize)
        #imgs = self.open_data(start, end, h_flip, width=width, height=height, isResize=resize)

        return imgs, sals, self._images[start:end]


def get_db(path, mode='train', validation_size=0.2):
    class DataSets(object):
        pass

    data_set = DataSets()
    random.seed(1)
    #images = load_saliency_information(path)

    images, labels = load_saliency_information(path, mode)
    if mode == 'train':
        images, labels = shuffle(images, labels)

        #print 'sorry! can not operate! => using test mode'

        if isinstance(validation_size, float):
            validation_size = int(validation_size * images.shape[0])

        validation_images = images[:validation_size]
        validation_labels = labels[:validation_size]


        train_images = images[validation_size:]
        train_labels = labels[validation_size:]


        data_set.train = DataSet(mode, train_images, train_labels)
        data_set.valid = DataSet(mode, validation_images, validation_labels)

        return data_set

    #images = np.sort(images)
    data_set.test = DataSet(mode, images, labels)

    return data_set