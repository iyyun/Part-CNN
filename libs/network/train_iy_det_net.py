'''

tensorflow version r1.4

created by Inyong yun, 2018.05.26
Copyright (c) 2018 DSPL Sungkyunkwan Univ

'''
import tensorflow as tf
from network.iy_layer import Network

num_class = 2
img_ch = 3


_feat_stride = [16,]
anchor_scales = [8, 16, 32] # inria = [8, 16, 32]

class train_iy_det_net(Network):
    def __init__(self):
        self.inputs = []
        self.img_data = tf.placeholder(tf.float32, shape=[None, None, None, img_ch])
        self.img_sal_data = tf.placeholder(tf.float32, shape=[None, None, None, 1])
        self.img_info = tf.placeholder(tf.float32, shape=[None, 3])
        self.img_label = tf.placeholder(tf.float32, shape=[None, 5])
        self.keep_prob = tf.placeholder(tf.float32)
        self.layers = dict({'img_data': self.img_data, 'img_info': self.img_info, 'img_label': self.img_label, \
                            'img_sal_data': self.img_sal_data })
        self.setup()

    def setup(self):
        # root classification
        (self.feed('img_data')
         .conv2d(3, 3, 1, 1, 64, name='conv1_1', trainable=False)
         .conv2d(3, 3, 1, 1, 64, name='conv1_2', trainable=False)
         .max_pool2d(2, 2, 2, 2, name='pool1', padding='SAME')  # h=256 -> 128 / w=128 -> 64  / s = 2
         .conv2d(3, 3, 1, 1, 128, name='conv2_1', trainable=False)
         .conv2d(3, 3, 1, 1, 128, name='conv2_2', trainable=False)
         .max_pool2d(2, 2, 2, 2, name='pool2', padding='SAME')  # h=128 -> 64 / w=64 -> 32  / s = 4
         .conv2d(3, 3, 1, 1, 256, name='conv3_1')
         .conv2d(3, 3, 1, 1, 256, name='conv3_2')
         .conv2d(3, 3, 1, 1, 256, name='conv3_3')
         .max_pool2d(2, 2, 2, 2, name='pool3', padding='SAME')  # h=64 -> 32 / w=32 -> 16  / s = 8
         .conv2d(3, 3, 1, 1, 512, name='conv4_1')
         .conv2d(3, 3, 1, 1, 512, name='conv4_2')
         .conv2d(3, 3, 1, 1, 512, name='conv4_3')
         .max_pool2d(2, 2, 2, 2, name='pool4', padding='SAME')  # h=32 -> 16 / w=16 -> 8  / s = 16
         .conv2d(3, 3, 1, 1, 512, name='conv5_1')
         .conv2d(3, 3, 1, 1, 512, name='conv5_2')
         .conv2d(3, 3, 1, 1, 512, name='conv5_3')

         # for Saliency Map
         .max_pool2d(2, 2, 2, 2, name='pool5', padding='SAME')  # s = 32
         .conv2d(3, 3, 1, 1, 1024, name='conv6_1')
         .conv2d(3, 3, 1, 1, 1024, name='conv6_2')
         .conv2d(3, 3, 1, 1, 1024, name='conv6_3')
         .upsample2d(1024, 2, name='dconv7_1')  # s = 32 -> s = 16
         .conv2d(3, 3, 1, 1, 512, name='dconv7_2')
         .conv2d(3, 3, 1, 1, 512, name='dconv7_3')
         .upsample2d(512, 2, name='dconv8_1')  # s = 16 -> s = 8
         .conv2d(3, 3, 1, 1, 256, name='dconv8_2')
         .conv2d(3, 3, 1, 1, 256, name='dconv8_3')
         .upsample2d(256, 2, name='dconv9_1')  # s = 8 -> s = 4
         .conv2d(3, 3, 1, 1, 128, name='dconv9_2')
         .upsample2d(128, 2, name='dconv10_1')  # s = 4 -> s = 2
         .conv2d(3, 3, 1, 1, 64, name='dconv10_2')
         .upsample2d(64, 2, name='dconv11_1')  # s = 2 -> s = 1
         .conv2d(3, 3, 1, 1, 32, name='dconv11_2')
         .dropout(self.keep_prob, name='drop_saliency')
         .conv2d(3, 3, 1, 1, 1, name='dconv11_3', relu=False)
         .sigmoid(name='saliency_score'))

        #RPN HIGH
        (self.feed('conv5_3')
            .conv2d(3, 3, 1, 1, 512, name='rpn_conv_high')
            .conv2d(1, 1, 1, 1, len(anchor_scales)*3*2, name='rpn_cls_score_high', relu=False))

        (self.feed('rpn_cls_score_high', 'img_label', 'img_info', 'img_data')
            .anchor_target_layer(_feat_stride, anchor_scales, name='rpn_data_high'))

        (self.feed('rpn_conv_high')
            .conv2d(1, 1, 1, 1, len(anchor_scales) * 3 * 4, name='rpn_bbox_pred_high', relu=False))

        (self.feed('rpn_cls_score_high')
            .reshape_layer(2, name='rpn_cls_score_reshape_high')
            .softmax(name='rpn_cls_prob_high'))

        (self.feed('rpn_cls_prob_high')
            .reshape_layer(len(anchor_scales) * 3 * 2, name='rpn_cls_prob_reshape_high'))

        (self.feed('rpn_cls_prob_reshape_high', 'rpn_bbox_pred_high', 'img_info')
            .proposal_layer(_feat_stride, anchor_scales, 'TRAIN', name='rpn_rois_high'))

        (self.feed('rpn_rois_high', 'img_label')
            .proposal_target_layer(num_class, name='roi_data_high'))
        #END

        (self.feed('conv5_3', 'roi_data_high')
            .roi_pool(1, 1, 1.0/16.0, name='GMP')   # h=1  / w=1  / s = NP
            # fc layer
            .dropout(self.keep_prob, name='drop_gmp')
            .fc(num_class, name='cls_score', relu=False)
            .softmax(name='cls_prob'))

        (self.feed('drop_gmp')
            .fc(num_class * 4, name='bbox_pred', relu=False))