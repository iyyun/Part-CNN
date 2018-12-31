'''

tensorflow version r1.4

created by Inyong yun, 2018.05.26
Copyright (c) 2018 DSPL Sungkyunkwan Univ

'''
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import numpy as np

import roi.roi_pooling_op as roi_pool_op
import roi.roi_pooling_op_grad
from rpn.anchor_target_layer import anchor_target_layer as anchor_target_layer_py
from rpn.proposal_layer import proposal_layer as proposal_layer_py
from rpn.proposal_target_layer import proposal_target_layer as proposal_target_layer_py

def layer(op):
    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))

        # Figure out the layer inputs.
        if len(self.inputs) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.inputs) == 1:
            layer_input = self.inputs[0]
        else:
            layer_input = list(self.inputs)

        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)

        # print layer name
        print (layer_output)

        # Add to layer LUT.
        self.layers[name] = layer_output

        # This output is now the input for the next layer.
        self.feed(layer_output)

        # Return self for chained calls.
        return self

    return layer_decorated


class Network(object):
    def __init__(self, inputs):
        self.inputs = []
        self.layers = dict(inputs)
        self.setup()

    def setup(self):
        raise NotImplementedError('Must be subclassed.')

    def get_unique_name(self, prefix):
        id = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        return '%s_%d' % (prefix, id)

    def feed(self, *args):
        assert len(args) != 0
        self.inputs = []
        for layer in args:
            if isinstance(layer, basestring):
                try:
                    layer = self.layers[layer]
                    print (layer)
                except KeyError:
                    print (self.layers.keys())
                    raise KeyError('Unknown layer name fed: %s' % layer)
            self.inputs.append(layer)
        return self

    def load(self, saver, sess, ckpt, ignore_missing=True, use_restore=False):
        if use_restore:
            saver.restore(sess, ckpt)
            print (str('weight and variable restore from %s' % ckpt))
        else:
            data_dict = np.load(ckpt).item()
            for key in data_dict:
                with tf.variable_scope(key, reuse=True):
                    for subkey in data_dict[key]:
                        try:
                            var = tf.get_variable(subkey)
                            sess.run(var.assign(data_dict[key][subkey]))
                            print "assign pretrain model " + subkey + " to " + key
                        except ValueError:
                            print "ignore " + key
                            if not ignore_missing:
                                raise

            #reader = pywrap_tensorflow.NewCheckpointReader(ckpt)
            #var_to_shape_map = reader.get_variable_to_shape_map()
            #for key in var_to_shape_map:
            #    if key.find('/') > -1:
            #        index = key.find('/') + 1
            #        names = key.split('/')
            #        sub_key = key[index:]
            #        with tf.variable_scope(names[0], reuse=True):
            #            try:
            #                var = tf.get_variable(sub_key)
            #                tensor = reader.get_tensor(key)
            #                sess.run(tf.assign(var, tensor))
            #                print ("assign pretrain model " + str('%s' % sub_key) + " to " + str('%s' % names))
            #            except ValueError:
            #                print ("ignore " + str('%s' % names))
            #                if not ignore_missing:
            #                    raise

    def get_output(self, layer):
        try:
            layer = self.layers[layer]
        except KeyError:
            print (self.layers.keys())
            raise KeyError('Unknown layer name fed: %s' % layer)
        return layer

    def mk_variable(self, name, shape, initializer=None, trainable=True, validate_shape=True):
        return tf.get_variable(name, shape, initializer=initializer, trainable=trainable, validate_shape=validate_shape)

    @layer
    def relu(self, input, name):
        return tf.nn.relu(input, name=name)

    @layer
    def sigmoid(self, input, name):
        return tf.nn.sigmoid(input, name=name)

    @layer
    def conv2d(self, input, h_k, w_k, h_s, w_s, c_output, name, padding='SAME', relu=True, trainable=True):
        # input ch
        c_input = input.get_shape()[-1]

        conv = lambda i, k: tf.nn.conv2d(i, k, [1, h_s, w_s, 1], padding=padding)
        with tf.variable_scope(name) as scope:
            init_weight = tf.contrib.layers.xavier_initializer()
            init_baises = tf.constant_initializer(0.1)

            w = self.mk_variable('weights', [h_k, w_k, c_input, c_output], init_weight, trainable=trainable)
            b = self.mk_variable('biases', [c_output], init_baises, trainable=trainable)

            c = conv(input, w)

            if relu:
                bias = tf.nn.bias_add(c, b)
                return tf.nn.relu(bias, name=scope.name)

            return tf.nn.bias_add(c, b, name=scope.name)

    @layer
    def fc(self, input, num_out, name, relu=True, trainable=True, withCAM = False):
        with tf.variable_scope(name) as scope:

            if isinstance(input, tuple):
                input = input[0]

            input_shape = input.get_shape()
            if input_shape.ndims == 4:
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                #feed_in = tf.reshape(tf.transpose(input, [0, 3, 1, 2]), [-1, dim])
                feed_in = tf.reshape(input, [-1, dim])
            else:
                #print('x?')
                #print(input_shape[-1].value)
                feed_in, dim = (input, input_shape[-1].value)

            init_weight = tf.contrib.layers.xavier_initializer()
            init_baises = tf.constant_initializer(0.1)

            w = self.mk_variable('weights', [dim, num_out], init_weight, trainable=trainable, validate_shape=True)
            b = self.mk_variable('biases', [num_out], init_baises, trainable=trainable)

            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, w, b, name=scope.name)

            if withCAM:
                return fc, w, b #tf.nn.bias_add(w, b)

            return fc

    @layer
    def max_pool2d(self, input, h_k, w_k, h_s, w_s, name, padding='SAME'):
        return tf.nn.max_pool(input, ksize=[1, h_k, w_k, 1], strides=[1, h_s, w_s, 1], padding=padding, name=name)

    @layer
    def global_avg_pool2d(self, input, name):
        return tf.reduce_mean(input, [1, 2], name=name)

    @layer
    def global_max_pool2d(self, input, name):
        return tf.reduce_max(input, [1, 2], name=name)


    @layer
    def avg_pool2d(self, input, h_k, w_k, h_s, w_s, name, padding='SAME'):
        return tf.nn.avg_pool(input, ksize=[1, h_k, w_k, 1], strides=[1, h_s, w_s, 1], padding=padding, name=name)

    @layer
    def softmax(self, input, name):
        return tf.nn.softmax(input, name=name)

    @layer
    def dropout(self, input, keep_prob, name):
        if isinstance(input, tuple):
            input = input[0]

        return tf.nn.dropout(input, keep_prob=keep_prob, name=name)

    # addition for detection layers
    @layer
    def anchor_target_layer(self, input, _feat_stride, anchor_scales, name):
        if isinstance(input[0], tuple):
            input[0] = input[0][0]

        with tf.variable_scope(name) as scope:
            rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = tf.py_func(
                anchor_target_layer_py, [input[0], input[1], input[2], input[3], _feat_stride, anchor_scales],
                [tf.float32, tf.float32, tf.float32, tf.float32])

            rpn_labels = tf.convert_to_tensor(tf.cast(rpn_labels, tf.int32), name='rpn_labels')
            rpn_bbox_targets = tf.convert_to_tensor(rpn_bbox_targets, name='rpn_bbox_targets')
            rpn_bbox_inside_weights = tf.convert_to_tensor(rpn_bbox_inside_weights, name='rpn_bbox_inside_weights')
            rpn_bbox_outside_weights = tf.convert_to_tensor(rpn_bbox_outside_weights,
                                                                name='rpn_bbox_outside_weights')

            return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights

    @layer
    def reshape_layer(self, input, d, name):
        input_shape = tf.shape(input)
        if name == 'rpn_cls_prob_reshape_low' or name == 'rpn_cls_prob_reshape_high':
            return tf.transpose(tf.reshape(tf.transpose(input, [0, 3, 1, 2]), [input_shape[0], int(d),\
                                                                               tf.cast(tf.cast(input_shape[1], tf.float32) / tf.cast(d, tf.float32) * tf.cast(input_shape[3], tf.float32), \
                                                                                       tf.int32), input_shape[2]]), [0, 2, 3, 1], name=name)
        else:
            return tf.transpose(tf.reshape(tf.transpose(input, [0, 3, 1, 2]), [input_shape[0],\
                                                                               int(d), tf.cast(tf.cast(input_shape[1], tf.float32) * (tf.cast(input_shape[3], tf.float32) / tf.cast(d, tf.float32)), tf.int32), \
                                                                               input_shape[2]]), [0, 2, 3, 1], name=name)

    @layer
    def proposal_layer(self, input, _feat_stride, anchor_scales, cfg_key, name):
        if isinstance(input[0], tuple):
            input[0] = input[0][0]
        return tf.reshape(
            tf.py_func(proposal_layer_py, [input[0], input[1], input[2], cfg_key, _feat_stride, anchor_scales], [tf.float32]), [-1, 5], name=name)

    @layer
    def proposal_target_layer(self, input, classes, name):
        if isinstance(input[0], tuple):
            input[0] = input[0][0]
        with tf.variable_scope(name) as scope:
            rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = tf.py_func( proposal_target_layer_py, [input[0], input[1], classes],\
                                                                                                [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32])

            rois = tf.reshape(rois, [-1, 5], name='rois')
            labels = tf.convert_to_tensor(tf.cast(labels, tf.int32), name='labels')
            bbox_targets = tf.convert_to_tensor(bbox_targets, name='bbox_targets')
            bbox_inside_weights = tf.convert_to_tensor(bbox_inside_weights, name='bbox_inside_weights')
            bbox_outside_weights = tf.convert_to_tensor(bbox_outside_weights, name='bbox_outside_weights')

            return rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights

    @layer
    def roi_pool(self, input, pooled_height, pooled_width, spatial_scale, name):
        # only use the first input
        if isinstance(input[0], tuple):
            input[0] = input[0][0]

        if isinstance(input[1], tuple):
            input[1] = input[1][0]

        # print input
        return roi_pool_op.roi_pool(input[0], input[1],
                                        pooled_height,
                                        pooled_width,
                                        spatial_scale,
                                        name=name)[0]

    # saliency map
    @layer
    def sigmoid(self, input, name):
        return tf.nn.sigmoid(input, name=name)

    @layer
    def upsample2d(self, input, n_channels, upscale_factor, name):
        kernel_size = 2 * upscale_factor - upscale_factor % 2
        stride = upscale_factor
        strides = [1, stride, stride, 1]

        dconv = lambda i, k, o, s: tf.nn.conv2d_transpose(i, k, o, s, padding='SAME')

        with tf.variable_scope(name) as scope:
            # Shape of the bottom tensor
            in_shape = tf.shape(input)

            h = ((in_shape[1] - 1) * stride) + 1 #in_shape[1] * stride #((in_shape[1] - 1) * stride) + 1
            w = ((in_shape[2] - 1) * stride) + 1 #in_shape[2] * stride #((in_shape[2] - 1) * stride) + 1
            new_shape = [in_shape[0], h, w, n_channels]
            output_shape = tf.stack(new_shape)


            filter_shape = [kernel_size, kernel_size, n_channels, n_channels]

            ####
            k_s = filter_shape[1]

            if k_s % 2 == 1:
                centre_location = upscale_factor - 1
            else:
                centre_location = upscale_factor - 0.5

            bilinear = np.zeros([filter_shape[0], filter_shape[1]])
            for x in range(filter_shape[0]):
                for y in range(filter_shape[1]):
                    ##Interpolation Calculation
                    value = (1 - abs((x - centre_location) / upscale_factor)) * (
                            1 - abs((y - centre_location) / upscale_factor))
                    bilinear[x, y] = value
            weights = np.zeros(filter_shape)
            for i in range(filter_shape[2]):
                weights[:, :, i, i] = bilinear
            init = tf.constant_initializer(value=weights,
                                           dtype=tf.float32)

            bilinear_weights = self.mk_variable('w', weights.shape, init, trainable=True)
            ####

            # weights = self.get_bilinear_filter(filter_shape, upscale_factor)

            dc = dconv(input, bilinear_weights, output_shape, strides)

        return tf.nn.relu(dc, name=scope.name)
















