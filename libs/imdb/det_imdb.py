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
import json


class tData:
    def __init__(self, frame=-1, obj_type="unset", occlusion=-1, \
                 obs_angle=-10, x1=-1, y1=-1, x2=-1, y2=-1, track_id=-1):
        """
            Constructor, initializes the object given the parameters.
        """

        # init object data
        self.img_root = 'none'
        self.sal_root = 'none'
        self.frame = frame
        self.track_id = track_id
        self.obj_type = obj_type
        self.occlusion = occlusion
        self.obs_angle = obs_angle
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.vx1 = 0
        self.vy1 = 0
        self.vx2 = 0
        self.vy2 = 0

folder = ['USA', 'ETH', 'INRIA', 'TudBrussels']

def get_dbinfo(db_index, mode):
    s = []
    v = []
    i = 0
    j = 0

    if folder[db_index] == 'USA':
        s = ['set00', 'set01', 'set02', 'set03', 'set04', 'set05', 'set06', 'set07', 'set08', 'set09', 'set10']
        v = [15, 6, 12, 13, 12, 13, 19, 12, 11, 12, 12]
        if mode == 'train':
            i = 0
            j = 6
        else:
            i = 6
            j = 11
    elif folder[db_index] == 'ETH':
        s = ['set00', 'set01', 'set02']
        v = [1, 1, 1]
        if mode == 'train':
            i = 0
            j = 3
        else:
            i = 0
            j = 3
    elif folder[db_index] == 'INRIA':
        s = ['set00', 'set01']
        v = [2, 1]
        if mode == 'train':
            i = 0
            j = 1
        else:
            i = 1
            j = 2
    elif folder[db_index] == 'TudBrussels':
        s = ['set00']
        v = [1]
        i = 0
        j = 1

    return s, v, i, j

def load_label_file(f_path, mode, dset):
    labels = []

    for DB in dset:
        s_f, v_f, start, end = get_dbinfo(folder.index(DB), mode)

        if mode == 'train':
            for s in s_f[start:end]:
                idx = s_f.index(s)
                for v in range(0, v_f[idx]):
                    path = os.path.join(f_path, 'data-%s/images/%s/V%03d' % (DB, s, v), '*g')
                    img_files = glob.glob(path)
                    if mode == 'test':
                        img_files = np.sort(img_files)

                    for fi in img_files:
                        # print fi
                        fn = fi.split('/')[-1].split('.')[0]
                        # label data reas
                        label_path = os.path.join(f_path, 'data-%s/annotations/%s/V%03d/%s.txt' % (DB, s, v, fn))

                        label_file = open(label_path, 'r')

                        add_labels = True
                        tmP_cnt = 0
                        for line in label_file:
                            line_s = line.strip().split(' ')
                            if line_s[0] == '%':
                                # tmP_cnt += 1
                                continue

                            tmP_cnt += 1
                            if tmP_cnt > 0:

                                t_data = tData()

                                t_data.frame = fn
                                t_data.img_root = fi
                                t_data.sal_root = '%s/data-%s/saliency/%s/V%03d/%s.jpg' % (f_path, DB, s, v, fn)
                                t_data.obj_type = line_s[0]

                                if t_data.obj_type == 'people' and mode == 'train':
                                    continue

                                t_data.obj_type = 'pedestrian'

                                t_data.x1 = int(line_s[1])
                                t_data.y1 = int(line_s[2])
                                t_data.x2 = int(line_s[1]) + int(line_s[3])
                                t_data.y2 = int(line_s[2]) + int(line_s[4])

                                if (t_data.y2 - t_data.y1) < 35 and mode == 'train':  #30
                                    continue

                                t_data.occlusion = int(line_s[5])

                                if t_data.occlusion == 1:

                                    t_data.x1 = int(line_s[6])
                                    t_data.y1 = int(line_s[7])
                                    t_data.x2 = int(line_s[6]) + int(line_s[8])
                                    t_data.y2 = int(line_s[7]) + int(line_s[9])

                                    if (t_data.y2 - t_data.y1) < 15 and mode == 'train': # 25
                                        continue
                                    if (t_data.x2 - t_data.x1) < 15 and mode == 'train':
                                        continue

                                if add_labels:
                                    labels.append([])
                                    add_labels = False

                                labels[-1].append(t_data)

                        if tmP_cnt == 0 and mode == 'test':
                            t_data = tData()
                            t_data.frame = fn
                            t_data.img_root = fi
                            t_data.sal_root = '%s/data-%s/saliency/%s/V%03d/%s.jpg' % (f_path, DB, s, v, fn)
                            t_data.obj_type = 'pedestrian'

                            if add_labels:
                                labels.append([])
                                # add_labels = False

                            labels[-1].append(t_data)


                        label_file.close()

        else:
            for s in s_f[start:end]:
                idx = s_f.index(s)
                for v in range(0, v_f[idx]):
                    path = os.path.join(f_path, 'data-%s/images/%s/V%03d' % (DB, s, v), '*g')
                    img_files = glob.glob(path)
                    img_files = np.sort(img_files)

                    for fi in img_files:
                        # print fi
                        fn = fi.split('/')[-1].split('.')[0]

                        t_data = tData()
                        t_data.frame = fn
                        t_data.img_root = fi

                        labels.append(t_data)



    #print(len(labels), np.shape(labels[0]))
    return labels


class DataSet(object):
    def __init__(self, db, classes, mode):
        self._num_examples = len(db)
        self._data = db
        self._classes = classes
        self._epochs = 0
        self._index_in_epoch = 0
        self._mode = mode

    @property
    def images(self):
        return self._data

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def index_in_epoch(self):
        return self._index_in_epoch

    @property
    def epoch_done(self):
        return self._epochs

    def pre_image(self, root, gray=1, min_size = 600, max_size = 1200):
        #print(root)
        if gray:
            img = cv.imread(root, gray) - np.array([[[102.9801, 115.9465, 122.7717]]])
        else:
            img = cv.imread(root, gray)

        #print(np.shape(img))
        im_shape = np.shape(img) #.shape
        #print(im_shape)
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        scale = float(min_size) / float(im_size_min)
        if np.round(scale * im_size_max) > max_size:
            scale = float(max_size) / float(im_size_max)

        img = cv.resize(img, None, None, fx=scale, fy=scale, interpolation=cv.INTER_LINEAR)

        #print(np.shape(img))

        img = img.astype(np.float32)
        img = np.multiply(img, 1.0 / 255.0)

        return img, scale

    def open_data(self, start_index, end_index):
        c_imgs = []
        ## train
        #if self._mode == 'train':
        s_imgs = []

        imgs_info = []
        ## train
        #if self._mode == 'train':
        labels = []

        for i in range(start_index, end_index):
            #if self._mode == 'train':
            t_data = self._data[i][0]
            #else:
            #    t_data = self._data[i]

            ## image
            img, scale = self.pre_image(t_data.img_root)
            ## only train
            #if self._mode == 'train':
            sal, _ = self.pre_image(t_data.sal_root, 0)

            c_imgs.append(img)
            ## train
            #if self._mode == 'train':
            s_imgs.append(sal)

            _info = np.array([img.shape[0], img.shape[1], scale]).astype(np.float32)
            imgs_info.append(_info)

            ## labels
            # for train
            #if self._mode == 'train':
            labels.append([])
            for j in range(0, len(self._data[i])):
                i_data = self._data[i][j]
                x1 = i_data.x1 * scale
                y1 = i_data.y1 * scale
                x2 = i_data.x2 * scale
                y2 = i_data.y2 * scale
                ca = self._classes.index(i_data.obj_type)

                rect = np.array((x1, y1, x2, y2, ca)).astype(np.float32)
                labels[-1].append(rect)

        c_imgs = np.array(c_imgs)
        # train
        #if self._mode == 'train':
        s_imgs = np.array(s_imgs)
        s_imgs = np.expand_dims(s_imgs, axis=4)

        imgs_info = np.array(imgs_info)
        # train
        #if self._mode == 'train':
        labels = np.array(labels)
        gt_inds = np.shape(labels)
        gt_boxes = np.empty((gt_inds[1], 5), dtype=np.float32)
        gt_boxes = labels[0][:]


        ## train
        #if self._mode == 'train':
        return c_imgs, s_imgs, gt_boxes, imgs_info, self._data[start_index:end_index]
        #else: # test
        #    return c_imgs, imgs_info, self._data[start_index:end_index]



    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch >= self._num_examples:
            # random suhffle
            self._data = shuffle(self._data)
            self._epochs += 1
            start = 0
            self._index_in_epoch = batch_size

        end = self._index_in_epoch

        return self.open_data(start, end)


def get_db(path, classes, mode='train', validation_size=0.05, which_data=['USA']):
    class DataSets(object):
        pass

    data_set = DataSets()
    random.seed(1)

    gt_label = load_label_file(path, mode, which_data)
    if mode == 'train':
        gt_label = shuffle(gt_label)
        if isinstance(validation_size, float):
            validation_size = int(validation_size * len(gt_label))

        validation_labels = gt_label[:validation_size]
        train_labels = gt_label #[validation_size:]

        data_set.train = DataSet(train_labels, classes, mode)
        data_set.valid = DataSet(validation_labels, classes, mode)

        return data_set

    data_set.test = DataSet(gt_label, classes, mode)

    return data_set




"""

back up!
def load_label_file(f_path, mode, dset):
    labels = []

    for DB in dset:
        s_f, v_f, start, end = get_dbinfo(folder.index(DB), mode)

        for s in s_f[start:end]:
            idx = s_f.index(s)
            for v in range(0, v_f[idx]):
                if DB == 'USA':
                    if mode == 'train':
                        path = os.path.join(f_path, 'data-%s/annotations/%s/V%03d_all_v5_train.txt' % (DB, s, v))
                    else:
                        path = os.path.join(f_path, 'data-%s/annotations/%s/V%03d_all_v5_test.txt' % (DB, s, v))
                else:
                    path = os.path.join(f_path, 'data-%s/annotations/%s/V%03d.txt' % (DB, s, v))

                print path
                files = open(path, 'r')
                last_fid = -1

                for line in files:
                    line = line.strip()
                    fields = line.split(' ')
                    fid = int(float(fields[0]))

                    t_data = tData()
                    t_data.frame = int(float(fields[0]))  # frame
                    if DB == 'USA':
                        t_data.img_root = '%s/data-%s/images/%s/V%03d/Frame_%05d.png' % (f_path, DB, s, v, fid)
                        t_data.sal_root = '%s/data-%s/saliency/%s/V%03d/Frame_%05d.png' % (f_path, DB, s, v, fid)

                        if not os.path.exists(t_data.sal_root):
                            continue

                        t_data.track_id = int(float(fields[1]))
                        t_data.obj_type = fields[2].lower()
                        t_data.occlusion = int(float(fields[4]))  # occlusion  [-1,0,1,2]
                        t_data.obs_angle = 0.0  # float(fields[5])  # observation angle [rad]
                        #
                        t_data.x1 = float(fields[6])  # left   [px]
                        t_data.y1 = float(fields[7])  # top    [px]
                        t_data.x2 = float(fields[8])  # right  [px]
                        t_data.y2 = float(fields[9])  # bottom [px]
                        #
                        t_data.vx1 = float(fields[10])  # v left , if occlusion = 1
                        t_data.vy1 = float(fields[11])  # v top  [m]
                        t_data.vx2 = float(fields[12])  # v right [m]
                        t_data.vy2 = float(fields[13])  # v bottom [m]

                        if t_data.occlusion == 1:
                            t_data.x1 = t_data.vx1
                            t_data.y1 = t_data.vy1
                            t_data.x2 = t_data.vx2
                            t_data.y2 = t_data.vy2

                        # limit
                        if t_data.obj_type == '\'people\'':
                            continue
                        elif t_data.obj_type == 'bg':
                            t_data.obj_type = 'bg'
                        else:
                            t_data.obj_type = 'pedestrian'

                    else:
                        t_data.img_root = '%s/data-%s/images/%s/V%03d/I%05d.png' % (f_path, DB, s, v, fid)
                        t_data.sal_root = '%s/data-%s/saliency/%s/V%03d/I%05d.png' % (f_path, DB, s, v, fid)
                        if not os.path.exists(t_data.sal_root):
                            continue

                        t_data.track_id = int(float(fields[1]))
                        t_data.obj_type = 'pedestrian'  # fields[2].lower()
                        t_data.occlusion = int(float(fields[3]))  # occlusion  [-1,0,1,2]
                        t_data.obs_angle = 0.0  # float(fields[5])  # observation angle [rad]
                        #
                        t_data.x1 = float(fields[6])  # left   [px]
                        t_data.y1 = float(fields[7])  # top    [px]
                        t_data.x2 = float(fields[8])  # right  [px]
                        t_data.y2 = float(fields[9])  # bottom [px]
                        #
                        t_data.vx1 = float(fields[10])  # v left , if occlusion = 1
                        t_data.vy1 = float(fields[11])  # v top  [m]
                        t_data.vx2 = float(fields[12])  # v right [m]
                        t_data.vy2 = float(fields[13])  # v bottom [m]

                    if mode == 'train':
                        if np.abs(t_data.y2 - t_data.y1) < 40.:
                            continue

                        if np.abs(t_data.x2 - t_data.x1) < 16.:
                            continue

                    if last_fid != fid:
                        labels.append([])
                    last_fid = fid

                    labels[-1].append(t_data)

                    #

                files.close()


    print len(labels)
    return labels



f_path = '/media/xidian/Data/dataset/Pedstrian'
classes = ['bg', 'pedestrian']

sets = get_db(f_path, classes, which_data='INRIA')

for i in range(0, 10):
    t_imgs, t_sal, t_bbox, t_info = sets.train.next_batch(1)

    for j in range(0, len(t_bbox)):
        gts = t_bbox[j].astype(np.int)
        print gts
        cv.rectangle(t_imgs[0], (gts[0], gts[1]), (gts[2], gts[3]), (0,0,1), 2)

    cv.imshow('frame', t_imgs[0])
    cv.imshow('saliency', t_sal[0])
    key = cv.waitKey(0)
    if key == 27:
        break

"""
















