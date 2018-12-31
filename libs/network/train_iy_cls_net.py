'''

tensorflow version r1.4

created by Inyong yun, 2018.05.26
Copyright (c) 2018 DSPL Sungkyunkwan Univ

'''
import tensorflow as tf
from network.iy_layer import Network

num_class = 4
img_ch = 3


class train_iy_cls_net(Network):
    def __init__(self):
        self.inputs = []
        self.img_data = tf.placeholder(tf.float32, shape=[None, 64, 64, img_ch])
        self.img_label = tf.placeholder(tf.float32, shape=[None, num_class])
        self.keep_prob = tf.placeholder(tf.float32)
        self.layers = dict({'img_data': self.img_data, 'img_label': self.img_label})
        self.setup()

    def setup(self):
        # root classification
        (self.feed('img_data')
         .conv2d(3, 3, 1, 1, 64, name='conv1_1', trainable=False)
         .conv2d(3, 3, 1, 1, 64, name='conv1_2', trainable=False)
         .max_pool2d(2, 2, 2, 2, name='pool1', padding='SAME')  # h=128 -> 64 / w=128 -> 64  / s = 2
         .conv2d(3, 3, 1, 1, 128, name='conv2_1', trainable=False)
         .conv2d(3, 3, 1, 1, 128, name='conv2_2', trainable=False)
         .max_pool2d(2, 2, 2, 2, name='pool2', padding='SAME')  # h=64 -> 32 / w=64 -> 32  / s = 4
         .conv2d(3, 3, 1, 1, 256, name='conv3_1')
         .conv2d(3, 3, 1, 1, 256, name='conv3_2')
         .conv2d(3, 3, 1, 1, 256, name='conv3_3')
         .max_pool2d(2, 2, 2, 2, name='pool3', padding='SAME')  # h=32 -> 16 / w=32 -> 16  / s = 8
         .conv2d(3, 3, 1, 1, 512, name='conv4_1')
         .conv2d(3, 3, 1, 1, 512, name='conv4_2')
         .conv2d(3, 3, 1, 1, 512, name='conv4_3')
         .max_pool2d(2, 2, 2, 2, name='pool4', padding='SAME')  # h=16 -> 8 / w=16 -> 8  / s = 16
         .conv2d(3, 3, 1, 1, 512, name='conv5_1')
         .conv2d(3, 3, 1, 1, 512, name='conv5_2')
         .conv2d(3, 3, 1, 1, 512, name='conv5_3')
         .max_pool2d(2, 2, 2, 2, name='pool5', padding='SAME')
         .fc(1024, name='fc6', relu=True, withCAM=True)
         .dropout(self.keep_prob, name='drop_fc6')
         .fc(num_class, name='cls_score_fc', relu=False, withCAM=True))

        (self.feed('conv4_3')
         .global_avg_pool2d(name='gap')
         .dropout(self.keep_prob, name='drop_gap')
         .fc(num_class, name='cls_score_gap', relu=False, withCAM=True))

