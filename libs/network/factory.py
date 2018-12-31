'''

tensorflow version r1.4

created by Inyong yun, 2018.05.26
Copyright (c) 2018 DSPL Sungkyunkwan Univ

'''
import network.train_iy_det_net
import network.test_iy_det_net

import network.train_iy_cls_net

def get_network(mode, method):

    if mode == 'train':
        if method == 'det':
            print ('det network')
            return network.train_iy_det_net()
        elif method == 'cls':
            print ('cls network')
            return network.train_iy_cls_net()
        else:
            raise KeyError('Unknown Network: {}'.format(method))
    elif mode == 'test':
        if method == 'det':
            return network.test_iy_det_net()
        else:
            raise KeyError('Unknown Network: {}'.format(method))
    else:
        raise KeyError('Unknown mode: {}'.format(mode))