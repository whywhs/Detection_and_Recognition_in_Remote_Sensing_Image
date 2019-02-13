# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Guodong Zhang
# Modified by Bin Xiao
# --------------------------------------------------------

import cPickle
import mxnet as mx
from utils.symbol import Symbol
from operator_py.proposal import *
from operator_py.proposal_quadrangle import *
from operator_py.proposal_target import *
from operator_py.proposal_target_quadrangle import *
from operator_py.box_annotator_ohem import *
from operator_py.focal_loss import *
from symbol_dpn import *

class resnet_v1_101_rcnn_quadrangle(Symbol):
    def __init__(self):
        """
        Use __init__ to define parameter network needs
        """
        self.eps = 1e-5
        self.use_global_stats = True
        self.workspace = 512
        self.units = (3, 4, 23, 3)  # use for 101
        self.filter_list = [256, 512, 1024, 2048]

        self.k_R = 200

        self.G   = 50

        self.k_sec  = {  2: 4, \
                    3: 8, \
                    4: 20, \
                    5: 3   }

        self.inc_sec= {  2: 20, \
                    3: 64, \
                    4: 64, \
                    5: 128 }

    def get_resnet_v1_conv4(self, data):
        # conv1 = (data+1)/2  stride2
        conv1 = mx.symbol.Convolution(name='conv1', data=data, num_filter=64, pad=(3, 3), kernel=(7, 7), stride=(2, 2),
                                      no_bias=True)

        bn_conv1 = mx.symbol.BatchNorm(name='bn_conv1', data=conv1, use_global_stats=True, fix_gamma=False,
                                       eps=self.eps)
        scale_conv1 = bn_conv1
        conv1_relu = mx.symbol.Activation(name='conv1_relu', data=scale_conv1, act_type='relu')
        # pool1 = (conv1+1)/2  stride4
        pool1 = mx.symbol.Pooling(name='pool1', data=conv1_relu, pooling_convention='full', pad=(0, 0), kernel=(3, 3),
                                  stride=(2, 2), pool_type='max')



        res2a_branch1 = mx.symbol.Convolution(name='res2a_branch1', data=pool1, num_filter=256, pad=(0, 0),
                                              kernel=(1, 1),
                                              stride=(1, 1), no_bias=True)
        bn2a_branch1 = mx.symbol.BatchNorm(name='bn2a_branch1', data=res2a_branch1, use_global_stats=True,
                                           fix_gamma=False, eps=self.eps)
        scale2a_branch1 = bn2a_branch1
        res2a_branch2a = mx.symbol.Convolution(name='res2a_branch2a', data=pool1, num_filter=64, pad=(0, 0),
                                               kernel=(1, 1),
                                               stride=(1, 1), no_bias=True)
        bn2a_branch2a = mx.symbol.BatchNorm(name='bn2a_branch2a', data=res2a_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale2a_branch2a = bn2a_branch2a
        res2a_branch2a_relu = mx.symbol.Activation(name='res2a_branch2a_relu', data=scale2a_branch2a, act_type='relu')
        res2a_branch2b = mx.symbol.Convolution(name='res2a_branch2b', data=res2a_branch2a_relu, num_filter=64,
                                               pad=(1, 1),
                                               kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn2a_branch2b = mx.symbol.BatchNorm(name='bn2a_branch2b', data=res2a_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale2a_branch2b = bn2a_branch2b
        res2a_branch2b_relu = mx.symbol.Activation(name='res2a_branch2b_relu', data=scale2a_branch2b, act_type='relu')
        res2a_branch2c = mx.symbol.Convolution(name='res2a_branch2c', data=res2a_branch2b_relu, num_filter=256,
                                               pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn2a_branch2c = mx.symbol.BatchNorm(name='bn2a_branch2c', data=res2a_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale2a_branch2c = bn2a_branch2c
        res2a = mx.symbol.broadcast_add(name='res2a', *[scale2a_branch1, scale2a_branch2c])
        res2a_relu = mx.symbol.Activation(name='res2a_relu', data=res2a, act_type='relu')
        res2b_branch2a = mx.symbol.Convolution(name='res2b_branch2a', data=res2a_relu, num_filter=64, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn2b_branch2a = mx.symbol.BatchNorm(name='bn2b_branch2a', data=res2b_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale2b_branch2a = bn2b_branch2a
        res2b_branch2a_relu = mx.symbol.Activation(name='res2b_branch2a_relu', data=scale2b_branch2a, act_type='relu')
        res2b_branch2b = mx.symbol.Convolution(name='res2b_branch2b', data=res2b_branch2a_relu, num_filter=64,
                                               pad=(1, 1),
                                               kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn2b_branch2b = mx.symbol.BatchNorm(name='bn2b_branch2b', data=res2b_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale2b_branch2b = bn2b_branch2b
        res2b_branch2b_relu = mx.symbol.Activation(name='res2b_branch2b_relu', data=scale2b_branch2b, act_type='relu')
        res2b_branch2c = mx.symbol.Convolution(name='res2b_branch2c', data=res2b_branch2b_relu, num_filter=256,
                                               pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn2b_branch2c = mx.symbol.BatchNorm(name='bn2b_branch2c', data=res2b_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale2b_branch2c = bn2b_branch2c
        res2b = mx.symbol.broadcast_add(name='res2b', *[res2a_relu, scale2b_branch2c])
        res2b_relu = mx.symbol.Activation(name='res2b_relu', data=res2b, act_type='relu')
        res2c_branch2a = mx.symbol.Convolution(name='res2c_branch2a', data=res2b_relu, num_filter=64, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn2c_branch2a = mx.symbol.BatchNorm(name='bn2c_branch2a', data=res2c_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale2c_branch2a = bn2c_branch2a
        res2c_branch2a_relu = mx.symbol.Activation(name='res2c_branch2a_relu', data=scale2c_branch2a, act_type='relu')
        res2c_branch2b = mx.symbol.Convolution(name='res2c_branch2b', data=res2c_branch2a_relu, num_filter=64,
                                               pad=(1, 1),
                                               kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn2c_branch2b = mx.symbol.BatchNorm(name='bn2c_branch2b', data=res2c_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale2c_branch2b = bn2c_branch2b
        res2c_branch2b_relu = mx.symbol.Activation(name='res2c_branch2b_relu', data=scale2c_branch2b, act_type='relu')
        res2c_branch2c = mx.symbol.Convolution(name='res2c_branch2c', data=res2c_branch2b_relu, num_filter=256,
                                               pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn2c_branch2c = mx.symbol.BatchNorm(name='bn2c_branch2c', data=res2c_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale2c_branch2c = bn2c_branch2c
        res2c = mx.symbol.broadcast_add(name='res2c', *[res2b_relu, scale2c_branch2c])
        res2c_relu = mx.symbol.Activation(name='res2c_relu', data=res2c, act_type='relu')







        # res3a_branch1 = (res2c_reslu+1)/2    stride8
        res3a_branch1 = mx.symbol.Convolution(name='res3a_branch1', data=res2c_relu, num_filter=512, pad=(0, 0),
                                              kernel=(1, 1), stride=(2, 2), no_bias=True)

        bn3a_branch1 = mx.symbol.BatchNorm(name='bn3a_branch1', data=res3a_branch1, use_global_stats=True,
                                           fix_gamma=False, eps=self.eps)
        scale3a_branch1 = bn3a_branch1
        res3a_branch2a = mx.symbol.Convolution(name='res3a_branch2a', data=res2c_relu, num_filter=128, pad=(0, 0),
                                               kernel=(1, 1), stride=(2, 2), no_bias=True)
        bn3a_branch2a = mx.symbol.BatchNorm(name='bn3a_branch2a', data=res3a_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale3a_branch2a = bn3a_branch2a
        res3a_branch2a_relu = mx.symbol.Activation(name='res3a_branch2a_relu', data=scale3a_branch2a, act_type='relu')
        res3a_branch2b = mx.symbol.Convolution(name='res3a_branch2b', data=res3a_branch2a_relu, num_filter=128,
                                               pad=(1, 1),
                                               kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn3a_branch2b = mx.symbol.BatchNorm(name='bn3a_branch2b', data=res3a_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale3a_branch2b = bn3a_branch2b
        res3a_branch2b_relu = mx.symbol.Activation(name='res3a_branch2b_relu', data=scale3a_branch2b, act_type='relu')
        res3a_branch2c = mx.symbol.Convolution(name='res3a_branch2c', data=res3a_branch2b_relu, num_filter=512,
                                               pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3a_branch2c = mx.symbol.BatchNorm(name='bn3a_branch2c', data=res3a_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale3a_branch2c = bn3a_branch2c
        res3a = mx.symbol.broadcast_add(name='res3a', *[scale3a_branch1, scale3a_branch2c])
        res3a_relu = mx.symbol.Activation(name='res3a_relu', data=res3a, act_type='relu')
        res3b1_branch2a = mx.symbol.Convolution(name='res3b1_branch2a', data=res3a_relu, num_filter=128, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3b1_branch2a = mx.symbol.BatchNorm(name='bn3b1_branch2a', data=res3b1_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale3b1_branch2a = bn3b1_branch2a
        res3b1_branch2a_relu = mx.symbol.Activation(name='res3b1_branch2a_relu', data=scale3b1_branch2a,
                                                    act_type='relu')
        res3b1_branch2b = mx.symbol.Convolution(name='res3b1_branch2b', data=res3b1_branch2a_relu, num_filter=128,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn3b1_branch2b = mx.symbol.BatchNorm(name='bn3b1_branch2b', data=res3b1_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale3b1_branch2b = bn3b1_branch2b
        res3b1_branch2b_relu = mx.symbol.Activation(name='res3b1_branch2b_relu', data=scale3b1_branch2b,
                                                    act_type='relu')
        res3b1_branch2c = mx.symbol.Convolution(name='res3b1_branch2c', data=res3b1_branch2b_relu, num_filter=512,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3b1_branch2c = mx.symbol.BatchNorm(name='bn3b1_branch2c', data=res3b1_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale3b1_branch2c = bn3b1_branch2c
        res3b1 = mx.symbol.broadcast_add(name='res3b1', *[res3a_relu, scale3b1_branch2c])
        res3b1_relu = mx.symbol.Activation(name='res3b1_relu', data=res3b1, act_type='relu')
        res3b2_branch2a = mx.symbol.Convolution(name='res3b2_branch2a', data=res3b1_relu, num_filter=128, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3b2_branch2a = mx.symbol.BatchNorm(name='bn3b2_branch2a', data=res3b2_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale3b2_branch2a = bn3b2_branch2a
        res3b2_branch2a_relu = mx.symbol.Activation(name='res3b2_branch2a_relu', data=scale3b2_branch2a,
                                                    act_type='relu')
        res3b2_branch2b = mx.symbol.Convolution(name='res3b2_branch2b', data=res3b2_branch2a_relu, num_filter=128,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn3b2_branch2b = mx.symbol.BatchNorm(name='bn3b2_branch2b', data=res3b2_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale3b2_branch2b = bn3b2_branch2b
        res3b2_branch2b_relu = mx.symbol.Activation(name='res3b2_branch2b_relu', data=scale3b2_branch2b,
                                                    act_type='relu')
        res3b2_branch2c = mx.symbol.Convolution(name='res3b2_branch2c', data=res3b2_branch2b_relu, num_filter=512,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3b2_branch2c = mx.symbol.BatchNorm(name='bn3b2_branch2c', data=res3b2_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale3b2_branch2c = bn3b2_branch2c
        res3b2 = mx.symbol.broadcast_add(name='res3b2', *[res3b1_relu, scale3b2_branch2c])
        res3b2_relu = mx.symbol.Activation(name='res3b2_relu', data=res3b2, act_type='relu')
        res3b3_branch2a = mx.symbol.Convolution(name='res3b3_branch2a', data=res3b2_relu, num_filter=128, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3b3_branch2a = mx.symbol.BatchNorm(name='bn3b3_branch2a', data=res3b3_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale3b3_branch2a = bn3b3_branch2a
        res3b3_branch2a_relu = mx.symbol.Activation(name='res3b3_branch2a_relu', data=scale3b3_branch2a,
                                                    act_type='relu')
        res3b3_branch2b = mx.symbol.Convolution(name='res3b3_branch2b', data=res3b3_branch2a_relu, num_filter=128,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn3b3_branch2b = mx.symbol.BatchNorm(name='bn3b3_branch2b', data=res3b3_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale3b3_branch2b = bn3b3_branch2b
        res3b3_branch2b_relu = mx.symbol.Activation(name='res3b3_branch2b_relu', data=scale3b3_branch2b,
                                                    act_type='relu')
        res3b3_branch2c = mx.symbol.Convolution(name='res3b3_branch2c', data=res3b3_branch2b_relu, num_filter=512,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3b3_branch2c = mx.symbol.BatchNorm(name='bn3b3_branch2c', data=res3b3_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale3b3_branch2c = bn3b3_branch2c
        res3b3 = mx.symbol.broadcast_add(name='res3b3', *[res3b2_relu, scale3b3_branch2c])
        res3b3_relu = mx.symbol.Activation(name='res3b3_relu', data=res3b3, act_type='relu')






        # res4a_branch1 : stride16
        res4a_branch1 = mx.symbol.Convolution(name='res4a_branch1', data=res3b3_relu, num_filter=1024, pad=(0, 0),
                                              kernel=(1, 1), stride=(2, 2), no_bias=True)
        bn4a_branch1 = mx.symbol.BatchNorm(name='bn4a_branch1', data=res4a_branch1, use_global_stats=True,
                                           fix_gamma=False, eps=self.eps)
        scale4a_branch1 = bn4a_branch1
        res4a_branch2a = mx.symbol.Convolution(name='res4a_branch2a', data=res3b3_relu, num_filter=256, pad=(0, 0),
                                               kernel=(1, 1), stride=(2, 2), no_bias=True)
        bn4a_branch2a = mx.symbol.BatchNorm(name='bn4a_branch2a', data=res4a_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale4a_branch2a = bn4a_branch2a
        res4a_branch2a_relu = mx.symbol.Activation(name='res4a_branch2a_relu', data=scale4a_branch2a, act_type='relu')
        res4a_branch2b = mx.symbol.Convolution(name='res4a_branch2b', data=res4a_branch2a_relu, num_filter=256,
                                               pad=(1, 1),
                                               kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4a_branch2b = mx.symbol.BatchNorm(name='bn4a_branch2b', data=res4a_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale4a_branch2b = bn4a_branch2b
        res4a_branch2b_relu = mx.symbol.Activation(name='res4a_branch2b_relu', data=scale4a_branch2b, act_type='relu')
        res4a_branch2c = mx.symbol.Convolution(name='res4a_branch2c', data=res4a_branch2b_relu, num_filter=1024,
                                               pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4a_branch2c = mx.symbol.BatchNorm(name='bn4a_branch2c', data=res4a_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale4a_branch2c = bn4a_branch2c
        res4a = mx.symbol.broadcast_add(name='res4a', *[scale4a_branch1, scale4a_branch2c])
        res4a_relu = mx.symbol.Activation(name='res4a_relu', data=res4a, act_type='relu')
        res4b1_branch2a = mx.symbol.Convolution(name='res4b1_branch2a', data=res4a_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b1_branch2a = mx.symbol.BatchNorm(name='bn4b1_branch2a', data=res4b1_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b1_branch2a = bn4b1_branch2a
        res4b1_branch2a_relu = mx.symbol.Activation(name='res4b1_branch2a_relu', data=scale4b1_branch2a,
                                                    act_type='relu')
        res4b1_branch2b = mx.symbol.Convolution(name='res4b1_branch2b', data=res4b1_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b1_branch2b = mx.symbol.BatchNorm(name='bn4b1_branch2b', data=res4b1_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b1_branch2b = bn4b1_branch2b
        res4b1_branch2b_relu = mx.symbol.Activation(name='res4b1_branch2b_relu', data=scale4b1_branch2b,
                                                    act_type='relu')
        res4b1_branch2c = mx.symbol.Convolution(name='res4b1_branch2c', data=res4b1_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b1_branch2c = mx.symbol.BatchNorm(name='bn4b1_branch2c', data=res4b1_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b1_branch2c = bn4b1_branch2c
        res4b1 = mx.symbol.broadcast_add(name='res4b1', *[res4a_relu, scale4b1_branch2c])
        res4b1_relu = mx.symbol.Activation(name='res4b1_relu', data=res4b1, act_type='relu')
        res4b2_branch2a = mx.symbol.Convolution(name='res4b2_branch2a', data=res4b1_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b2_branch2a = mx.symbol.BatchNorm(name='bn4b2_branch2a', data=res4b2_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b2_branch2a = bn4b2_branch2a
        res4b2_branch2a_relu = mx.symbol.Activation(name='res4b2_branch2a_relu', data=scale4b2_branch2a,
                                                    act_type='relu')
        res4b2_branch2b = mx.symbol.Convolution(name='res4b2_branch2b', data=res4b2_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b2_branch2b = mx.symbol.BatchNorm(name='bn4b2_branch2b', data=res4b2_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b2_branch2b = bn4b2_branch2b
        res4b2_branch2b_relu = mx.symbol.Activation(name='res4b2_branch2b_relu', data=scale4b2_branch2b,
                                                    act_type='relu')
        res4b2_branch2c = mx.symbol.Convolution(name='res4b2_branch2c', data=res4b2_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b2_branch2c = mx.symbol.BatchNorm(name='bn4b2_branch2c', data=res4b2_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b2_branch2c = bn4b2_branch2c
        res4b2 = mx.symbol.broadcast_add(name='res4b2', *[res4b1_relu, scale4b2_branch2c])
        res4b2_relu = mx.symbol.Activation(name='res4b2_relu', data=res4b2, act_type='relu')
        res4b3_branch2a = mx.symbol.Convolution(name='res4b3_branch2a', data=res4b2_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b3_branch2a = mx.symbol.BatchNorm(name='bn4b3_branch2a', data=res4b3_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b3_branch2a = bn4b3_branch2a
        res4b3_branch2a_relu = mx.symbol.Activation(name='res4b3_branch2a_relu', data=scale4b3_branch2a,
                                                    act_type='relu')
        res4b3_branch2b = mx.symbol.Convolution(name='res4b3_branch2b', data=res4b3_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b3_branch2b = mx.symbol.BatchNorm(name='bn4b3_branch2b', data=res4b3_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b3_branch2b = bn4b3_branch2b
        res4b3_branch2b_relu = mx.symbol.Activation(name='res4b3_branch2b_relu', data=scale4b3_branch2b,
                                                    act_type='relu')
        res4b3_branch2c = mx.symbol.Convolution(name='res4b3_branch2c', data=res4b3_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b3_branch2c = mx.symbol.BatchNorm(name='bn4b3_branch2c', data=res4b3_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b3_branch2c = bn4b3_branch2c
        res4b3 = mx.symbol.broadcast_add(name='res4b3', *[res4b2_relu, scale4b3_branch2c])
        res4b3_relu = mx.symbol.Activation(name='res4b3_relu', data=res4b3, act_type='relu')
        res4b4_branch2a = mx.symbol.Convolution(name='res4b4_branch2a', data=res4b3_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b4_branch2a = mx.symbol.BatchNorm(name='bn4b4_branch2a', data=res4b4_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b4_branch2a = bn4b4_branch2a
        res4b4_branch2a_relu = mx.symbol.Activation(name='res4b4_branch2a_relu', data=scale4b4_branch2a,
                                                    act_type='relu')
        res4b4_branch2b = mx.symbol.Convolution(name='res4b4_branch2b', data=res4b4_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b4_branch2b = mx.symbol.BatchNorm(name='bn4b4_branch2b', data=res4b4_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b4_branch2b = bn4b4_branch2b
        res4b4_branch2b_relu = mx.symbol.Activation(name='res4b4_branch2b_relu', data=scale4b4_branch2b,
                                                    act_type='relu')
        res4b4_branch2c = mx.symbol.Convolution(name='res4b4_branch2c', data=res4b4_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b4_branch2c = mx.symbol.BatchNorm(name='bn4b4_branch2c', data=res4b4_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b4_branch2c = bn4b4_branch2c
        res4b4 = mx.symbol.broadcast_add(name='res4b4', *[res4b3_relu, scale4b4_branch2c])
        res4b4_relu = mx.symbol.Activation(name='res4b4_relu', data=res4b4, act_type='relu')
        res4b5_branch2a = mx.symbol.Convolution(name='res4b5_branch2a', data=res4b4_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b5_branch2a = mx.symbol.BatchNorm(name='bn4b5_branch2a', data=res4b5_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b5_branch2a = bn4b5_branch2a
        res4b5_branch2a_relu = mx.symbol.Activation(name='res4b5_branch2a_relu', data=scale4b5_branch2a,
                                                    act_type='relu')
        res4b5_branch2b = mx.symbol.Convolution(name='res4b5_branch2b', data=res4b5_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b5_branch2b = mx.symbol.BatchNorm(name='bn4b5_branch2b', data=res4b5_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b5_branch2b = bn4b5_branch2b
        res4b5_branch2b_relu = mx.symbol.Activation(name='res4b5_branch2b_relu', data=scale4b5_branch2b,
                                                    act_type='relu')
        res4b5_branch2c = mx.symbol.Convolution(name='res4b5_branch2c', data=res4b5_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b5_branch2c = mx.symbol.BatchNorm(name='bn4b5_branch2c', data=res4b5_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b5_branch2c = bn4b5_branch2c
        res4b5 = mx.symbol.broadcast_add(name='res4b5', *[res4b4_relu, scale4b5_branch2c])
        res4b5_relu = mx.symbol.Activation(name='res4b5_relu', data=res4b5, act_type='relu')
        res4b6_branch2a = mx.symbol.Convolution(name='res4b6_branch2a', data=res4b5_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b6_branch2a = mx.symbol.BatchNorm(name='bn4b6_branch2a', data=res4b6_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b6_branch2a = bn4b6_branch2a
        res4b6_branch2a_relu = mx.symbol.Activation(name='res4b6_branch2a_relu', data=scale4b6_branch2a,
                                                    act_type='relu')
        res4b6_branch2b = mx.symbol.Convolution(name='res4b6_branch2b', data=res4b6_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b6_branch2b = mx.symbol.BatchNorm(name='bn4b6_branch2b', data=res4b6_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b6_branch2b = bn4b6_branch2b
        res4b6_branch2b_relu = mx.symbol.Activation(name='res4b6_branch2b_relu', data=scale4b6_branch2b,
                                                    act_type='relu')
        res4b6_branch2c = mx.symbol.Convolution(name='res4b6_branch2c', data=res4b6_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b6_branch2c = mx.symbol.BatchNorm(name='bn4b6_branch2c', data=res4b6_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b6_branch2c = bn4b6_branch2c
        res4b6 = mx.symbol.broadcast_add(name='res4b6', *[res4b5_relu, scale4b6_branch2c])
        res4b6_relu = mx.symbol.Activation(name='res4b6_relu', data=res4b6, act_type='relu')
        res4b7_branch2a = mx.symbol.Convolution(name='res4b7_branch2a', data=res4b6_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b7_branch2a = mx.symbol.BatchNorm(name='bn4b7_branch2a', data=res4b7_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b7_branch2a = bn4b7_branch2a
        res4b7_branch2a_relu = mx.symbol.Activation(name='res4b7_branch2a_relu', data=scale4b7_branch2a,
                                                    act_type='relu')
        res4b7_branch2b = mx.symbol.Convolution(name='res4b7_branch2b', data=res4b7_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b7_branch2b = mx.symbol.BatchNorm(name='bn4b7_branch2b', data=res4b7_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b7_branch2b = bn4b7_branch2b
        res4b7_branch2b_relu = mx.symbol.Activation(name='res4b7_branch2b_relu', data=scale4b7_branch2b,
                                                    act_type='relu')
        res4b7_branch2c = mx.symbol.Convolution(name='res4b7_branch2c', data=res4b7_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b7_branch2c = mx.symbol.BatchNorm(name='bn4b7_branch2c', data=res4b7_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b7_branch2c = bn4b7_branch2c
        res4b7 = mx.symbol.broadcast_add(name='res4b7', *[res4b6_relu, scale4b7_branch2c])
        res4b7_relu = mx.symbol.Activation(name='res4b7_relu', data=res4b7, act_type='relu')
        res4b8_branch2a = mx.symbol.Convolution(name='res4b8_branch2a', data=res4b7_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b8_branch2a = mx.symbol.BatchNorm(name='bn4b8_branch2a', data=res4b8_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b8_branch2a = bn4b8_branch2a
        res4b8_branch2a_relu = mx.symbol.Activation(name='res4b8_branch2a_relu', data=scale4b8_branch2a,
                                                    act_type='relu')
        res4b8_branch2b = mx.symbol.Convolution(name='res4b8_branch2b', data=res4b8_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b8_branch2b = mx.symbol.BatchNorm(name='bn4b8_branch2b', data=res4b8_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b8_branch2b = bn4b8_branch2b
        res4b8_branch2b_relu = mx.symbol.Activation(name='res4b8_branch2b_relu', data=scale4b8_branch2b,
                                                    act_type='relu')
        res4b8_branch2c = mx.symbol.Convolution(name='res4b8_branch2c', data=res4b8_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b8_branch2c = mx.symbol.BatchNorm(name='bn4b8_branch2c', data=res4b8_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b8_branch2c = bn4b8_branch2c
        res4b8 = mx.symbol.broadcast_add(name='res4b8', *[res4b7_relu, scale4b8_branch2c])
        res4b8_relu = mx.symbol.Activation(name='res4b8_relu', data=res4b8, act_type='relu')
        res4b9_branch2a = mx.symbol.Convolution(name='res4b9_branch2a', data=res4b8_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b9_branch2a = mx.symbol.BatchNorm(name='bn4b9_branch2a', data=res4b9_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b9_branch2a = bn4b9_branch2a
        res4b9_branch2a_relu = mx.symbol.Activation(name='res4b9_branch2a_relu', data=scale4b9_branch2a,
                                                    act_type='relu')
        res4b9_branch2b = mx.symbol.Convolution(name='res4b9_branch2b', data=res4b9_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b9_branch2b = mx.symbol.BatchNorm(name='bn4b9_branch2b', data=res4b9_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b9_branch2b = bn4b9_branch2b
        res4b9_branch2b_relu = mx.symbol.Activation(name='res4b9_branch2b_relu', data=scale4b9_branch2b,
                                                    act_type='relu')
        res4b9_branch2c = mx.symbol.Convolution(name='res4b9_branch2c', data=res4b9_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b9_branch2c = mx.symbol.BatchNorm(name='bn4b9_branch2c', data=res4b9_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b9_branch2c = bn4b9_branch2c
        res4b9 = mx.symbol.broadcast_add(name='res4b9', *[res4b8_relu, scale4b9_branch2c])
        res4b9_relu = mx.symbol.Activation(name='res4b9_relu', data=res4b9, act_type='relu')
        res4b10_branch2a = mx.symbol.Convolution(name='res4b10_branch2a', data=res4b9_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b10_branch2a = mx.symbol.BatchNorm(name='bn4b10_branch2a', data=res4b10_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b10_branch2a = bn4b10_branch2a
        res4b10_branch2a_relu = mx.symbol.Activation(name='res4b10_branch2a_relu', data=scale4b10_branch2a,
                                                     act_type='relu')
        res4b10_branch2b = mx.symbol.Convolution(name='res4b10_branch2b', data=res4b10_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b10_branch2b = mx.symbol.BatchNorm(name='bn4b10_branch2b', data=res4b10_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b10_branch2b = bn4b10_branch2b
        res4b10_branch2b_relu = mx.symbol.Activation(name='res4b10_branch2b_relu', data=scale4b10_branch2b,
                                                     act_type='relu')
        res4b10_branch2c = mx.symbol.Convolution(name='res4b10_branch2c', data=res4b10_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b10_branch2c = mx.symbol.BatchNorm(name='bn4b10_branch2c', data=res4b10_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b10_branch2c = bn4b10_branch2c
        res4b10 = mx.symbol.broadcast_add(name='res4b10', *[res4b9_relu, scale4b10_branch2c])
        res4b10_relu = mx.symbol.Activation(name='res4b10_relu', data=res4b10, act_type='relu')
        res4b11_branch2a = mx.symbol.Convolution(name='res4b11_branch2a', data=res4b10_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b11_branch2a = mx.symbol.BatchNorm(name='bn4b11_branch2a', data=res4b11_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b11_branch2a = bn4b11_branch2a
        res4b11_branch2a_relu = mx.symbol.Activation(name='res4b11_branch2a_relu', data=scale4b11_branch2a,
                                                     act_type='relu')
        res4b11_branch2b = mx.symbol.Convolution(name='res4b11_branch2b', data=res4b11_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b11_branch2b = mx.symbol.BatchNorm(name='bn4b11_branch2b', data=res4b11_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b11_branch2b = bn4b11_branch2b
        res4b11_branch2b_relu = mx.symbol.Activation(name='res4b11_branch2b_relu', data=scale4b11_branch2b,
                                                     act_type='relu')
        res4b11_branch2c = mx.symbol.Convolution(name='res4b11_branch2c', data=res4b11_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b11_branch2c = mx.symbol.BatchNorm(name='bn4b11_branch2c', data=res4b11_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b11_branch2c = bn4b11_branch2c
        res4b11 = mx.symbol.broadcast_add(name='res4b11', *[res4b10_relu, scale4b11_branch2c])
        res4b11_relu = mx.symbol.Activation(name='res4b11_relu', data=res4b11, act_type='relu')
        res4b12_branch2a = mx.symbol.Convolution(name='res4b12_branch2a', data=res4b11_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b12_branch2a = mx.symbol.BatchNorm(name='bn4b12_branch2a', data=res4b12_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b12_branch2a = bn4b12_branch2a
        res4b12_branch2a_relu = mx.symbol.Activation(name='res4b12_branch2a_relu', data=scale4b12_branch2a,
                                                     act_type='relu')
        res4b12_branch2b = mx.symbol.Convolution(name='res4b12_branch2b', data=res4b12_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b12_branch2b = mx.symbol.BatchNorm(name='bn4b12_branch2b', data=res4b12_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b12_branch2b = bn4b12_branch2b
        res4b12_branch2b_relu = mx.symbol.Activation(name='res4b12_branch2b_relu', data=scale4b12_branch2b,
                                                     act_type='relu')
        res4b12_branch2c = mx.symbol.Convolution(name='res4b12_branch2c', data=res4b12_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b12_branch2c = mx.symbol.BatchNorm(name='bn4b12_branch2c', data=res4b12_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b12_branch2c = bn4b12_branch2c
        res4b12 = mx.symbol.broadcast_add(name='res4b12', *[res4b11_relu, scale4b12_branch2c])
        res4b12_relu = mx.symbol.Activation(name='res4b12_relu', data=res4b12, act_type='relu')
        res4b13_branch2a = mx.symbol.Convolution(name='res4b13_branch2a', data=res4b12_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b13_branch2a = mx.symbol.BatchNorm(name='bn4b13_branch2a', data=res4b13_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b13_branch2a = bn4b13_branch2a
        res4b13_branch2a_relu = mx.symbol.Activation(name='res4b13_branch2a_relu', data=scale4b13_branch2a,
                                                     act_type='relu')
        res4b13_branch2b = mx.symbol.Convolution(name='res4b13_branch2b', data=res4b13_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b13_branch2b = mx.symbol.BatchNorm(name='bn4b13_branch2b', data=res4b13_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b13_branch2b = bn4b13_branch2b
        res4b13_branch2b_relu = mx.symbol.Activation(name='res4b13_branch2b_relu', data=scale4b13_branch2b,
                                                     act_type='relu')
        res4b13_branch2c = mx.symbol.Convolution(name='res4b13_branch2c', data=res4b13_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b13_branch2c = mx.symbol.BatchNorm(name='bn4b13_branch2c', data=res4b13_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b13_branch2c = bn4b13_branch2c
        res4b13 = mx.symbol.broadcast_add(name='res4b13', *[res4b12_relu, scale4b13_branch2c])
        res4b13_relu = mx.symbol.Activation(name='res4b13_relu', data=res4b13, act_type='relu')
        res4b14_branch2a = mx.symbol.Convolution(name='res4b14_branch2a', data=res4b13_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b14_branch2a = mx.symbol.BatchNorm(name='bn4b14_branch2a', data=res4b14_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b14_branch2a = bn4b14_branch2a
        res4b14_branch2a_relu = mx.symbol.Activation(name='res4b14_branch2a_relu', data=scale4b14_branch2a,
                                                     act_type='relu')
        res4b14_branch2b = mx.symbol.Convolution(name='res4b14_branch2b', data=res4b14_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b14_branch2b = mx.symbol.BatchNorm(name='bn4b14_branch2b', data=res4b14_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b14_branch2b = bn4b14_branch2b
        res4b14_branch2b_relu = mx.symbol.Activation(name='res4b14_branch2b_relu', data=scale4b14_branch2b,
                                                     act_type='relu')
        res4b14_branch2c = mx.symbol.Convolution(name='res4b14_branch2c', data=res4b14_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b14_branch2c = mx.symbol.BatchNorm(name='bn4b14_branch2c', data=res4b14_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b14_branch2c = bn4b14_branch2c
        res4b14 = mx.symbol.broadcast_add(name='res4b14', *[res4b13_relu, scale4b14_branch2c])
        res4b14_relu = mx.symbol.Activation(name='res4b14_relu', data=res4b14, act_type='relu')
        res4b15_branch2a = mx.symbol.Convolution(name='res4b15_branch2a', data=res4b14_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b15_branch2a = mx.symbol.BatchNorm(name='bn4b15_branch2a', data=res4b15_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b15_branch2a = bn4b15_branch2a
        res4b15_branch2a_relu = mx.symbol.Activation(name='res4b15_branch2a_relu', data=scale4b15_branch2a,
                                                     act_type='relu')
        res4b15_branch2b = mx.symbol.Convolution(name='res4b15_branch2b', data=res4b15_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b15_branch2b = mx.symbol.BatchNorm(name='bn4b15_branch2b', data=res4b15_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b15_branch2b = bn4b15_branch2b
        res4b15_branch2b_relu = mx.symbol.Activation(name='res4b15_branch2b_relu', data=scale4b15_branch2b,
                                                     act_type='relu')
        res4b15_branch2c = mx.symbol.Convolution(name='res4b15_branch2c', data=res4b15_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b15_branch2c = mx.symbol.BatchNorm(name='bn4b15_branch2c', data=res4b15_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b15_branch2c = bn4b15_branch2c
        res4b15 = mx.symbol.broadcast_add(name='res4b15', *[res4b14_relu, scale4b15_branch2c])
        res4b15_relu = mx.symbol.Activation(name='res4b15_relu', data=res4b15, act_type='relu')
        res4b16_branch2a = mx.symbol.Convolution(name='res4b16_branch2a', data=res4b15_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b16_branch2a = mx.symbol.BatchNorm(name='bn4b16_branch2a', data=res4b16_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b16_branch2a = bn4b16_branch2a
        res4b16_branch2a_relu = mx.symbol.Activation(name='res4b16_branch2a_relu', data=scale4b16_branch2a,
                                                     act_type='relu')
        res4b16_branch2b = mx.symbol.Convolution(name='res4b16_branch2b', data=res4b16_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b16_branch2b = mx.symbol.BatchNorm(name='bn4b16_branch2b', data=res4b16_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b16_branch2b = bn4b16_branch2b
        res4b16_branch2b_relu = mx.symbol.Activation(name='res4b16_branch2b_relu', data=scale4b16_branch2b,
                                                     act_type='relu')
        res4b16_branch2c = mx.symbol.Convolution(name='res4b16_branch2c', data=res4b16_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b16_branch2c = mx.symbol.BatchNorm(name='bn4b16_branch2c', data=res4b16_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b16_branch2c = bn4b16_branch2c
        res4b16 = mx.symbol.broadcast_add(name='res4b16', *[res4b15_relu, scale4b16_branch2c])
        res4b16_relu = mx.symbol.Activation(name='res4b16_relu', data=res4b16, act_type='relu')
        res4b17_branch2a = mx.symbol.Convolution(name='res4b17_branch2a', data=res4b16_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b17_branch2a = mx.symbol.BatchNorm(name='bn4b17_branch2a', data=res4b17_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b17_branch2a = bn4b17_branch2a
        res4b17_branch2a_relu = mx.symbol.Activation(name='res4b17_branch2a_relu', data=scale4b17_branch2a,
                                                     act_type='relu')
        res4b17_branch2b = mx.symbol.Convolution(name='res4b17_branch2b', data=res4b17_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b17_branch2b = mx.symbol.BatchNorm(name='bn4b17_branch2b', data=res4b17_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b17_branch2b = bn4b17_branch2b
        res4b17_branch2b_relu = mx.symbol.Activation(name='res4b17_branch2b_relu', data=scale4b17_branch2b,
                                                     act_type='relu')
        res4b17_branch2c = mx.symbol.Convolution(name='res4b17_branch2c', data=res4b17_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b17_branch2c = mx.symbol.BatchNorm(name='bn4b17_branch2c', data=res4b17_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b17_branch2c = bn4b17_branch2c
        res4b17 = mx.symbol.broadcast_add(name='res4b17', *[res4b16_relu, scale4b17_branch2c])
        res4b17_relu = mx.symbol.Activation(name='res4b17_relu', data=res4b17, act_type='relu')
        res4b18_branch2a = mx.symbol.Convolution(name='res4b18_branch2a', data=res4b17_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b18_branch2a = mx.symbol.BatchNorm(name='bn4b18_branch2a', data=res4b18_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b18_branch2a = bn4b18_branch2a
        res4b18_branch2a_relu = mx.symbol.Activation(name='res4b18_branch2a_relu', data=scale4b18_branch2a,
                                                     act_type='relu')
        res4b18_branch2b = mx.symbol.Convolution(name='res4b18_branch2b', data=res4b18_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b18_branch2b = mx.symbol.BatchNorm(name='bn4b18_branch2b', data=res4b18_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b18_branch2b = bn4b18_branch2b
        res4b18_branch2b_relu = mx.symbol.Activation(name='res4b18_branch2b_relu', data=scale4b18_branch2b,
                                                     act_type='relu')
        res4b18_branch2c = mx.symbol.Convolution(name='res4b18_branch2c', data=res4b18_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b18_branch2c = mx.symbol.BatchNorm(name='bn4b18_branch2c', data=res4b18_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b18_branch2c = bn4b18_branch2c
        res4b18 = mx.symbol.broadcast_add(name='res4b18', *[res4b17_relu, scale4b18_branch2c])
        res4b18_relu = mx.symbol.Activation(name='res4b18_relu', data=res4b18, act_type='relu')
        res4b19_branch2a = mx.symbol.Convolution(name='res4b19_branch2a', data=res4b18_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b19_branch2a = mx.symbol.BatchNorm(name='bn4b19_branch2a', data=res4b19_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b19_branch2a = bn4b19_branch2a
        res4b19_branch2a_relu = mx.symbol.Activation(name='res4b19_branch2a_relu', data=scale4b19_branch2a,
                                                     act_type='relu')
        res4b19_branch2b = mx.symbol.Convolution(name='res4b19_branch2b', data=res4b19_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b19_branch2b = mx.symbol.BatchNorm(name='bn4b19_branch2b', data=res4b19_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b19_branch2b = bn4b19_branch2b
        res4b19_branch2b_relu = mx.symbol.Activation(name='res4b19_branch2b_relu', data=scale4b19_branch2b,
                                                     act_type='relu')
        res4b19_branch2c = mx.symbol.Convolution(name='res4b19_branch2c', data=res4b19_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b19_branch2c = mx.symbol.BatchNorm(name='bn4b19_branch2c', data=res4b19_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b19_branch2c = bn4b19_branch2c
        res4b19 = mx.symbol.broadcast_add(name='res4b19', *[res4b18_relu, scale4b19_branch2c])
        res4b19_relu = mx.symbol.Activation(name='res4b19_relu', data=res4b19, act_type='relu')
        res4b20_branch2a = mx.symbol.Convolution(name='res4b20_branch2a', data=res4b19_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b20_branch2a = mx.symbol.BatchNorm(name='bn4b20_branch2a', data=res4b20_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b20_branch2a = bn4b20_branch2a
        res4b20_branch2a_relu = mx.symbol.Activation(name='res4b20_branch2a_relu', data=scale4b20_branch2a,
                                                     act_type='relu')
        res4b20_branch2b = mx.symbol.Convolution(name='res4b20_branch2b', data=res4b20_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b20_branch2b = mx.symbol.BatchNorm(name='bn4b20_branch2b', data=res4b20_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b20_branch2b = bn4b20_branch2b
        res4b20_branch2b_relu = mx.symbol.Activation(name='res4b20_branch2b_relu', data=scale4b20_branch2b,
                                                     act_type='relu')
        res4b20_branch2c = mx.symbol.Convolution(name='res4b20_branch2c', data=res4b20_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b20_branch2c = mx.symbol.BatchNorm(name='bn4b20_branch2c', data=res4b20_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b20_branch2c = bn4b20_branch2c
        res4b20 = mx.symbol.broadcast_add(name='res4b20', *[res4b19_relu, scale4b20_branch2c])
        res4b20_relu = mx.symbol.Activation(name='res4b20_relu', data=res4b20, act_type='relu')
        res4b21_branch2a = mx.symbol.Convolution(name='res4b21_branch2a', data=res4b20_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b21_branch2a = mx.symbol.BatchNorm(name='bn4b21_branch2a', data=res4b21_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b21_branch2a = bn4b21_branch2a
        res4b21_branch2a_relu = mx.symbol.Activation(name='res4b21_branch2a_relu', data=scale4b21_branch2a,
                                                     act_type='relu')
        res4b21_branch2b = mx.symbol.Convolution(name='res4b21_branch2b', data=res4b21_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b21_branch2b = mx.symbol.BatchNorm(name='bn4b21_branch2b', data=res4b21_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b21_branch2b = bn4b21_branch2b
        res4b21_branch2b_relu = mx.symbol.Activation(name='res4b21_branch2b_relu', data=scale4b21_branch2b,
                                                     act_type='relu')
        res4b21_branch2c = mx.symbol.Convolution(name='res4b21_branch2c', data=res4b21_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b21_branch2c = mx.symbol.BatchNorm(name='bn4b21_branch2c', data=res4b21_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b21_branch2c = bn4b21_branch2c
        res4b21 = mx.symbol.broadcast_add(name='res4b21', *[res4b20_relu, scale4b21_branch2c])
        res4b21_relu = mx.symbol.Activation(name='res4b21_relu', data=res4b21, act_type='relu')
        res4b22_branch2a = mx.symbol.Convolution(name='res4b22_branch2a', data=res4b21_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b22_branch2a = mx.symbol.BatchNorm(name='bn4b22_branch2a', data=res4b22_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b22_branch2a = bn4b22_branch2a
        res4b22_branch2a_relu = mx.symbol.Activation(name='res4b22_branch2a_relu', data=scale4b22_branch2a,
                                                     act_type='relu')
        res4b22_branch2b = mx.symbol.Convolution(name='res4b22_branch2b', data=res4b22_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b22_branch2b = mx.symbol.BatchNorm(name='bn4b22_branch2b', data=res4b22_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b22_branch2b = bn4b22_branch2b
        res4b22_branch2b_relu = mx.symbol.Activation(name='res4b22_branch2b_relu', data=scale4b22_branch2b,
                                                     act_type='relu')
        res4b22_branch2c = mx.symbol.Convolution(name='res4b22_branch2c', data=res4b22_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b22_branch2c = mx.symbol.BatchNorm(name='bn4b22_branch2c', data=res4b22_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b22_branch2c = bn4b22_branch2c
        res4b22 = mx.symbol.broadcast_add(name='res4b22', *[res4b21_relu, scale4b22_branch2c])
        res4b22_relu = mx.symbol.Activation(name='res4b22_relu', data=res4b22, act_type='relu')
        return res4b22_relu , res2c_relu , res3b3_relu

    def get_resnet_v1_conv5(self, conv_feat):


        #res5a_branch1 = mx.symbol.Convolution(name='res5a_branch1', data=conv_feat, num_filter=2048, pad=(0, 0),
        #                                      kernel=(1, 1), stride=(1, 1), no_bias=True)

        res5a_branch1 = mx.symbol.Convolution(name='res5a_branch1', data=conv_feat, num_filter=2048, pad=(0, 0),
                                              kernel=(1, 1), stride=(2, 2), no_bias=True)

        bn5a_branch1 = mx.symbol.BatchNorm(name='bn5a_branch1', data=res5a_branch1, use_global_stats=True,
                                           fix_gamma=False, eps=self.eps)
        scale5a_branch1 = bn5a_branch1

        #res5a_branch2a = mx.symbol.Convolution(name='res5a_branch2a', data=conv_feat, num_filter=512, pad=(0, 0),
        #                                       kernel=(1, 1), stride=(1, 1), no_bias=True)

        res5a_branch2a = mx.symbol.Convolution(name='res5a_branch2a', data=conv_feat, num_filter=512, pad=(0, 0),
                                               kernel=(1, 1), stride=(2, 2), no_bias=True)



        bn5a_branch2a = mx.symbol.BatchNorm(name='bn5a_branch2a', data=res5a_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5a_branch2a = bn5a_branch2a
        res5a_branch2a_relu = mx.symbol.Activation(name='res5a_branch2a_relu', data=scale5a_branch2a, act_type='relu')
        res5a_branch2b = mx.symbol.Convolution(name='res5a_branch2b', data=res5a_branch2a_relu, num_filter=512,
                                               pad=(2, 2),
                                               kernel=(3, 3), stride=(1, 1), dilate=(2, 2), no_bias=True)
        bn5a_branch2b = mx.symbol.BatchNorm(name='bn5a_branch2b', data=res5a_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5a_branch2b = bn5a_branch2b
        res5a_branch2b_relu = mx.symbol.Activation(name='res5a_branch2b_relu', data=scale5a_branch2b, act_type='relu')
        res5a_branch2c = mx.symbol.Convolution(name='res5a_branch2c', data=res5a_branch2b_relu, num_filter=2048,
                                               pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5a_branch2c = mx.symbol.BatchNorm(name='bn5a_branch2c', data=res5a_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5a_branch2c = bn5a_branch2c
        res5a = mx.symbol.broadcast_add(name='res5a', *[scale5a_branch1, scale5a_branch2c])
        res5a_relu = mx.symbol.Activation(name='res5a_relu', data=res5a, act_type='relu')
        res5b_branch2a = mx.symbol.Convolution(name='res5b_branch2a', data=res5a_relu, num_filter=512, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5b_branch2a = mx.symbol.BatchNorm(name='bn5b_branch2a', data=res5b_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5b_branch2a = bn5b_branch2a
        res5b_branch2a_relu = mx.symbol.Activation(name='res5b_branch2a_relu', data=scale5b_branch2a, act_type='relu')
        res5b_branch2b = mx.symbol.Convolution(name='res5b_branch2b', data=res5b_branch2a_relu, num_filter=512,
                                               pad=(2, 2),
                                               kernel=(3, 3), stride=(1, 1), dilate=(2, 2), no_bias=True)
        bn5b_branch2b = mx.symbol.BatchNorm(name='bn5b_branch2b', data=res5b_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5b_branch2b = bn5b_branch2b
        res5b_branch2b_relu = mx.symbol.Activation(name='res5b_branch2b_relu', data=scale5b_branch2b, act_type='relu')
        res5b_branch2c = mx.symbol.Convolution(name='res5b_branch2c', data=res5b_branch2b_relu, num_filter=2048,
                                               pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5b_branch2c = mx.symbol.BatchNorm(name='bn5b_branch2c', data=res5b_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5b_branch2c = bn5b_branch2c
        res5b = mx.symbol.broadcast_add(name='res5b', *[res5a_relu, scale5b_branch2c])
        res5b_relu = mx.symbol.Activation(name='res5b_relu', data=res5b, act_type='relu')
        res5c_branch2a = mx.symbol.Convolution(name='res5c_branch2a', data=res5b_relu, num_filter=512, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5c_branch2a = mx.symbol.BatchNorm(name='bn5c_branch2a', data=res5c_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5c_branch2a = bn5c_branch2a
        res5c_branch2a_relu = mx.symbol.Activation(name='res5c_branch2a_relu', data=scale5c_branch2a, act_type='relu')
        res5c_branch2b = mx.symbol.Convolution(name='res5c_branch2b', data=res5c_branch2a_relu, num_filter=512,
                                               pad=(2, 2),
                                               kernel=(3, 3), stride=(1, 1), dilate=(2, 2), no_bias=True)
        bn5c_branch2b = mx.symbol.BatchNorm(name='bn5c_branch2b', data=res5c_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5c_branch2b = bn5c_branch2b
        res5c_branch2b_relu = mx.symbol.Activation(name='res5c_branch2b_relu', data=scale5c_branch2b, act_type='relu')
        res5c_branch2c = mx.symbol.Convolution(name='res5c_branch2c', data=res5c_branch2b_relu, num_filter=2048,
                                               pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5c_branch2c = mx.symbol.BatchNorm(name='bn5c_branch2c', data=res5c_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5c_branch2c = bn5c_branch2c
        res5c = mx.symbol.broadcast_add(name='res5c', *[res5b_relu, scale5c_branch2c])
        res5c_relu = mx.symbol.Activation(name='res5c_relu', data=res5c, act_type='relu')
        return res5c_relu

    
    '''
    def DPN_net(self,data):
        ## define Dual Path Network
        data = mx.symbol.Variable(name="data")

        # conv1
        conv1_x_1  = Conv(data=data,  num_filter=128,  kernel=(7, 7), name='conv1_x_1', pad=(3,3), stride=(2,2))
        conv1_x_1  = BN_AC(conv1_x_1, name='conv1_x_1__relu-sp')
        conv1_x_x  = mx.symbol.Pooling(data=conv1_x_1, pool_type="max", kernel=(3, 3),  pad=(1,1), stride=(2,2), name="pool1")

        # conv2
        bw = 256
        inc= self.inc_sec[2]
        R  = (self.k_R*bw)/256 
        conv2_x_x  = DualPathFactory(     conv1_x_x,   R,   R,   bw,  'conv2_x__1',           inc,   self.G,  'proj'  )
        for i_ly in range(2, self.k_sec[2]+1):
            conv2_x_x  = DualPathFactory( conv2_x_x,   R,   R,   bw, ('conv2_x__%d'% i_ly),   inc,   self.G,  'normal')

        conv2_x_x_con  = mx.symbol.Concat(*[conv2_x_x[0], conv2_x_x[1]],  name='conv2_x_x_cat-final')
        #conv2_x_x_con = BN_AC(conv2_x_x_con, name='conv2_x_x__relu-sp')

        # conv3
        bw = 512
        inc= self.inc_sec[3]
        R  = (self.k_R*bw)/256
        conv3_x_x  = DualPathFactory(     conv2_x_x,   R,   R,   bw,  'conv3_x__1',           inc,   self.G,  'down'  )
        for i_ly in range(2, self.k_sec[3]+1):
            conv3_x_x  = DualPathFactory( conv3_x_x,   R,   R,   bw, ('conv3_x__%d'% i_ly),   inc,   self.G,  'normal')

        conv3_x_x_con  = mx.symbol.Concat(*[conv3_x_x[0], conv3_x_x[1]],  name='conv3_x_x_cat-final')
        #conv3_x_x_con = BN_AC(conv3_x_x_con, name='conv3_x_x__relu-sp')

        # conv4
        bw = 1024
        inc= self.inc_sec[4]
        R  = (self.k_R*bw)/256
        conv4_x_x  = DualPathFactory(     conv3_x_x,   R,   R,   bw,  'conv4_x__1',           inc,   self.G,  'down'  )
        for i_ly in range(2, self.k_sec[4]+1):
            conv4_x_x  = DualPathFactory( conv4_x_x,   R,   R,   bw, ('conv4_x__%d'% i_ly),   inc,   self.G,  'normal')

        conv4_x_x_con  = mx.symbol.Concat(*[conv4_x_x[0], conv4_x_x[1]],  name='conv4_x_x_cat-final')
        #conv4_x_x_con = BN_AC(conv4_x_x_con, name='conv4_x_x__relu-sp')

        # conv5
        bw = 2048
        inc= self.inc_sec[5]
        R  = (self.k_R*bw)/256
        conv5_x_x  = DualPathFactory(     conv4_x_x,   R,   R,   bw,  'conv5_x__1',           inc,   self.G,  'down'  )
        for i_ly in range(2, self.k_sec[5]+1):
            conv5_x_x  = DualPathFactory( conv5_x_x,   R,   R,   bw, ('conv5_x__%d'% i_ly),   inc,   self.G,  'normal')

        # output: concat
        conv5_x_x  = mx.symbol.Concat(*[conv5_x_x[0], conv5_x_x[1]],  name='conv5_x_x_cat-final')
        conv5_x_x = BN_AC(conv5_x_x, name='conv5_x_x__relu-sp')
        
        return conv5_x_x, conv4_x_x_con, conv3_x_x_con, conv2_x_x_con
    '''

    def get_resnet_panet(self,conv_feat):

        N2 = conv_feat[4]
        N2_sample = mx.symbol.Convolution(data=N2, kernel=(3,3), num_filter=256, pad=(1,1),stride=(2,2),name='N2_sample')

        N2_sample_clip = mx.symbol.Crop(*[N2_sample,conv_feat[3]],name='N2_sample_clip')
        N3_conv = mx.sym.ElementWiseSum(*[N2_sample_clip,conv_feat[3]],name='N3_conv')
        N3 = mx.symbol.Convolution(data=N3_conv, kernel=(3,3), num_filter=256, pad=(1,1),stride=(1,1),name='N3')
        N3_sample = mx.symbol.Convolution(data=N3, kernel=(3,3), num_filter=256, pad=(1,1),stride=(2,2),name='N3_sample')
        
        N3_sample_clip = mx.symbol.Crop(*[N3_sample,conv_feat[2]],name='N3_sample_clip')
        N4_conv = mx.sym.ElementWiseSum(*[N3_sample_clip,conv_feat[2]],name='N4_conv')
        N4 = mx.symbol.Convolution(data=N4_conv, kernel=(3,3), num_filter=256, pad=(1,1),stride=(1,1),name='N4')
        N4_sample = mx.symbol.Convolution(data=N4, kernel=(3,3), num_filter=256, pad=(1,1),stride=(2,2),name='N4_sample')

        N4_sample_clip = mx.symbol.Crop(*[N4_sample,conv_feat[1]],name='N4_sample_clip')
        N5_conv = mx.sym.ElementWiseSum(*[N4_sample_clip,conv_feat[1]],name='N5_conv')
        N5 = mx.symbol.Convolution(data=N5_conv, kernel=(3,3), num_filter=256, pad=(1,1),stride=(1,1),name='N5')
        # N3_sample = mx.symbol.Convolution(data=N3, kernel=(3,3), num_filter=256, pad=(1,1),stride=(2,2),name='N3_sample')

        conv_panet_feat = dict()
        conv_panet_feat.update({'panet32':N5,'panet16':N4,'panet8':N3,'panet4':N2})

        return conv_panet_feat, [N5, N4, N3, N2]

    def get_resnet_conv_down(self,conv_feat):
        # C5 to P5, 1x1 dimension reduction to 256
        P5 = mx.symbol.Convolution(data=conv_feat[0], kernel=(1, 1), num_filter=256, name="P5_lateral")
        # P5 2x upsampling + C4 = P4
        P5_up   = mx.symbol.UpSampling(P5, scale=2, sample_type='bilinear', workspace=512, name='P5_upsampling', num_filter=256)

        P4_la   = mx.symbol.Convolution(data=conv_feat[1], kernel=(1, 1), num_filter=256, name="P4_lateral")
        P5_clip = mx.symbol.Crop(*[P5_up, P4_la], name="P4_clip")
        P4      = mx.sym.ElementWiseSum(*[P5_clip, P4_la], name="P4_sum")
        P4      = mx.symbol.Convolution(data=P4, kernel=(3, 3), pad=(1, 1), num_filter=256, name="P4_aggregate")

        # P4 2x upsampling + C3 = P3
        P4_up   = mx.symbol.UpSampling(P4, scale=2, sample_type='bilinear', workspace=512, name='P4_upsampling', num_filter=256)
        P3_la   = mx.symbol.Convolution(data=conv_feat[2], kernel=(1, 1), num_filter=256, name="P3_lateral")
        P4_clip = mx.symbol.Crop(*[P4_up, P3_la], name="P3_clip")
        P3      = mx.sym.ElementWiseSum(*[P4_clip, P3_la], name="P3_sum")
        P3      = mx.symbol.Convolution(data=P3, kernel=(3, 3), pad=(1, 1), num_filter=256, name="P3_aggregate")

        # P3 2x upsampling + C2 = P2
        P3_up   = mx.symbol.UpSampling(P3, scale=2, sample_type='bilinear', workspace=512, name='P3_upsampling', num_filter=256)
        P2_la   = mx.symbol.Convolution(data=conv_feat[3], kernel=(1, 1), num_filter=256, name="P2_lateral")
        P3_clip = mx.symbol.Crop(*[P3_up, P2_la], name="P2_clip")
        P2      = mx.sym.ElementWiseSum(*[P3_clip, P2_la], name="P2_sum")
        P2      = mx.symbol.Convolution(data=P2, kernel=(3, 3), pad=(1, 1), num_filter=256, name="P2_aggregate")

        # P6 2x subsampling P5
        P6 = mx.symbol.Pooling(data=P5, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max', name='P6_subsampling')

        conv_fpn_feat = dict()
        conv_fpn_feat.update({"stride64":P6, "stride32":P5, "stride16":P4, "stride8":P3, "stride4":P2})

        return conv_fpn_feat, [P6, P5, P4, P3, P2]



   
    def get_resnet_conv_down_cat(self,conv_feat):

        # P5
        P5 = mx.symbol.Convolution(data=conv_feat[0], kernel=(1, 1), num_filter=256, name="P5_lateral")

        # P4
        P5_up   = mx.symbol.UpSampling(P5, scale=2, sample_type='bilinear', workspace=512, name='P5_upsampling', num_filter=256)
        P4_la   = mx.symbol.Convolution(data=conv_feat[1], kernel=(1, 1), num_filter=256, name="P4_lateral")
        P5_clip = mx.symbol.Crop(*[P5_up, P4_la], name="P4_clip")
        P4_cat      = mx.symbol.concat(P5_clip, P4_la, dim=1,name='P4_cat')
        P4      = mx.symbol.Convolution(data=P4_cat, kernel=(3, 3), pad=(1, 1), num_filter=256, name="P4_aggregate")

        # P3
        P4_up   = mx.symbol.UpSampling(P4_la, scale=2, sample_type='bilinear', workspace=512, name='P4_upsampling', num_filter=256)
        P5_up_3 = mx.symbol.UpSampling(P5, scale=4, sample_type='bilinear', workspace=512, name='P5_upsampling_3', num_filter=256)
        P3_la   = mx.symbol.Convolution(data=conv_feat[2], kernel=(1, 1), num_filter=256, name="P3_lateral")
        P4_clip = mx.symbol.Crop(*[P4_up, P3_la], name="P3_clip")
        P5_clip_3 = mx.symbol.Crop(*[P5_up_3,P3_la],name = "P5_clip_3")
        P3_cat     = mx.symbol.concat(P5_clip_3, P4_clip, P3_la,dim=1, name="P3_cat")
        P3      = mx.symbol.Convolution(data=P3_cat, kernel=(3, 3), pad=(1, 1), num_filter=256, name="P3_aggregate")

        # P3 2x upsampling + C2 = P2
        P4_up_2   = mx.symbol.UpSampling(P4_la, scale=4, sample_type='bilinear', workspace=512, name='P4_upsampling_2', num_filter=256)
        P5_up_2 = mx.symbol.UpSampling(P5, scale=8, sample_type='bilinear', workspace=512, name='P5_upsampling_2', num_filter=256)
        P3_up   = mx.symbol.UpSampling(P3_la, scale=2, sample_type='bilinear', workspace=512, name='P3_upsampling', num_filter=256)
        P2_la   = mx.symbol.Convolution(data=conv_feat[3], kernel=(1, 1), num_filter=256, name="P2_lateral")
        P3_clip = mx.symbol.Crop(*[P3_up, P2_la], name="P2_clip")
        P4_clip_2 = mx.symbol.Crop(*[P4_up_2, P2_la], name="P3_clip_2")
        P5_clip_2 = mx.symbol.Crop(*[P5_up_2,P2_la],name = "P5_clip_2")
        P2_cat     = mx.symbol.concat(P5_clip_2, P4_clip_2, P3_clip,P2_la,dim=1, name="P2_cat")
        P2      = mx.symbol.Convolution(data=P2_cat, kernel=(3, 3), pad=(1, 1), num_filter=256, name="P2_aggregate")

        # P6 2x subsampling P5
        P6 = mx.symbol.Pooling(data=P5, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max', name='P6_subsampling')

        conv_fpn_feat = dict()
        conv_fpn_feat.update({"stride64":P6, "stride32":P5, "stride16":P4, "stride8":P3, "stride4":P2})

        return conv_fpn_feat, [P6, P5, P4, P3, P2]
  

    def get_symbol(self, cfg, is_train=True):

        # config alias for convenient
        num_classes = cfg.dataset.NUM_CLASSES
        num_reg_classes = (2 if cfg.CLASS_AGNOSTIC else num_classes)
        num_anchors = cfg.network.NUM_ANCHORS

        # input init
        if is_train:
            data = mx.sym.Variable(name="data")
            im_info = mx.sym.Variable(name="im_info")
            gt_boxes = mx.sym.Variable(name="gt_boxes")
            rpn_label = mx.sym.Variable(name='label')
            rpn_bbox_target = mx.sym.Variable(name='bbox_target')
            rpn_bbox_weight = mx.sym.Variable(name='bbox_weight')
        else:
            data = mx.sym.Variable(name="data")
            im_info = mx.sym.Variable(name="im_info")

        # shared convolutional layers
    
        conv_feat,res2,res3 = self.get_resnet_v1_conv4(data)
        relu1 = self.get_resnet_v1_conv5(conv_feat)


        conv_feat1 = [relu1,conv_feat,res3,res2]
   

        conv_fpn_feat ,px = self.get_resnet_conv_down(conv_feat1)
        conv_panet_feat, _ = self.get_resnet_panet(px)

        # shared parameters for predictions
        rpn_conv_weight      = mx.symbol.Variable('rpn_conv_weight')
        rpn_conv_bias        = mx.symbol.Variable('rpn_conv_bias')
        rpn_conv_cls_weight  = mx.symbol.Variable('rpn_conv_cls_weight')
        rpn_conv_cls_bias    = mx.symbol.Variable('rpn_conv_cls_bias')
        rpn_conv_bbox_weight = mx.symbol.Variable('rpn_conv_bbox_weight')
        rpn_conv_bbox_bias   = mx.symbol.Variable('rpn_conv_bbox_bias')

        rpn_cls_score_list = []
        rpn_bbox_pred_list = []
        rpn_cls_prob_dict = {}
        rpn_bbox_pred_dict = {}

        for stride in cfg.network.RPN_FEAT_STRIDE:
            rpn_conv = mx.symbol.Convolution(data=conv_panet_feat['panet%s'%stride],
                                             kernel=(3, 3), pad=(1, 1),
                                             num_filter=512,
                                             weight=rpn_conv_weight,
                                             bias=rpn_conv_bias)
            rpn_relu = mx.symbol.Activation(data=rpn_conv,
                                            act_type="relu",
                                            name="rpn_relu")
            rpn_cls_score = mx.symbol.Convolution(data=rpn_relu,
                                                  kernel=(1, 1), pad=(0, 0),
                                                  num_filter=2 * num_anchors,
                                                  weight=rpn_conv_cls_weight,
                                                  bias=rpn_conv_cls_bias,
                                                  name="rpn_cls_score_stride%s"%stride)
            rpn_bbox_pred = mx.symbol.Convolution(data=rpn_relu,
                                                  kernel=(1, 1), pad=(0, 0),
                                                  num_filter=4 * num_anchors,
                                                  weight=rpn_conv_bbox_weight,
                                                  bias=rpn_conv_bbox_bias,
                                                  name="rpn_bbox_pred_stride%s" % stride)

            # ROI Proposal
            rpn_cls_score_reshape = mx.symbol.Reshape(data=rpn_cls_score,
                                                      shape=(0, 2, -1),
                                                      name="rpn_cls_score_reshape_stride%s"%stride)

            rpn_bbox_pred_reshape = mx.symbol.Reshape(data=rpn_bbox_pred,
                                                      shape = (0,0,-1),
                                                      name = 'rpn_bbox_pred_reshape_stride%s'%stride)
            
            rpn_bbox_pred_list.append(rpn_bbox_pred_reshape)
            rpn_cls_score_list.append(rpn_cls_score_reshape)

            rpn_cls_score_reshape_t = mx.symbol.Reshape(data=rpn_cls_score,
                                                      shape=(0, 2, -1 ,0),
                                                      name="rpn_cls_score_reshape_t")
            rpn_cls_prob = mx.symbol.SoftmaxActivation(data=rpn_cls_score_reshape_t,
                                                      mode="channel",
                                                      name="rpn_cls_prob_stride")
            rpn_cls_prob_reshape = mx.symbol.Reshape(data=rpn_cls_prob,
                                                   shape=(0,2*num_anchors,-1,0),
                                                   name='rpn_cls_prob_reshape')


            rpn_cls_prob_dict.update({'cls_prob_stride%s'%stride:rpn_cls_prob_reshape})
            rpn_bbox_pred_dict.update({'bbox_pred_stride%s'%stride:rpn_bbox_pred})

        if is_train:
            rpn_bbox_pred_concat = mx.symbol.concat(*rpn_bbox_pred_list,dim=2)

            rpn_cls_score_concat = mx.symbol.concat(*rpn_cls_score_list,dim=2)


            # classification
            rpn_cls_prob = mx.sym.SoftmaxOutput(data=rpn_cls_score_concat, label=rpn_label, multi_output=True,
                                                normalization='valid', use_ignore=True, ignore_label=-1,
                                                name="rpn_cls_prob")

            # bounding box regression
            rpn_bbox_loss_ = rpn_bbox_weight * mx.sym.smooth_l1(name='rpn_bbox_loss_', scalar=3.0,
                                                                data=(rpn_bbox_pred_concat - rpn_bbox_target))

            rpn_bbox_loss = mx.sym.MakeLoss(name='rpn_bbox_loss', data=rpn_bbox_loss_,
                                            grad_scale=1.0 / cfg.TRAIN.RPN_BATCH_SIZE)

            args_dict = dict(rpn_cls_prob_dict.items()+rpn_bbox_pred_dict.items())


            rois_list = []

            for s in cfg.network.RPN_FEAT_STRIDE:
                rois = mx.contrib.sym.Proposal(
                    cls_prob=args_dict['cls_prob_stride%s'%s], bbox_pred=args_dict['bbox_pred_stride%s'%s], im_info=im_info, name='rois',
                    feature_stride=s, scales=tuple(cfg.network.ANCHOR_SCALES),
                    ratios=tuple(cfg.network.ANCHOR_RATIOS),
                    rpn_pre_nms_top_n=cfg.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TRAIN.RPN_POST_NMS_TOP_N,
                    threshold=cfg.TRAIN.RPN_NMS_THRESH, rpn_min_size=cfg.TRAIN.RPN_MIN_SIZE)
                rois_list.append(rois)
            rois_concat = mx.symbol.concat(*rois_list,dim=0)

            # ROI proposal target
            gt_boxes_reshape = mx.sym.Reshape(data=gt_boxes, shape=(-1, 9), name='gt_boxes_reshape')

            rois, label, bbox_target, bbox_weight = mx.sym.Custom(rois=rois_concat, gt_boxes=gt_boxes_reshape,
                                                                  op_type='proposal_target_quadrangle',
                                                                  num_classes=num_reg_classes,
                                                                  batch_images=cfg.TRAIN.BATCH_IMAGES,
                                                                  batch_rois=cfg.TRAIN.BATCH_ROIS,
                                                                  cfg=cPickle.dumps(cfg),
                                                                  fg_fraction=cfg.TRAIN.FG_FRACTION)
        else:

            args_dict = dict(rpn_cls_prob_dict.items()+rpn_bbox_pred_dict.items())

            # ROI Proposal
            rois_all = []

            for s in cfg.network.RPN_FEAT_STRIDE:
                rois_list = mx.contrib.sym.Proposal(
                    cls_prob=args_dict['cls_prob_stride%s'%s], bbox_pred=args_dict['bbox_pred_stride%s'%s], im_info=im_info, name='rois',
                    feature_stride=s, scales=tuple(cfg.network.ANCHOR_SCALES),
                    ratios=tuple(cfg.network.ANCHOR_RATIOS),
                    rpn_pre_nms_top_n=cfg.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TEST.RPN_POST_NMS_TOP_N,
                    threshold=cfg.TEST.RPN_NMS_THRESH, rpn_min_size=cfg.TEST.RPN_MIN_SIZE)
                rois_all.append(rois_list)
            rois = mx.symbol.concat(*rois_all,dim=0,name='rois')

        rcnn_fc6_weight = mx.symbol.Variable('rcnn_fc6_weight')
        rcnn_fc6_bias = mx.symbol.Variable('rcnn_fc6_bias')

        rcnn_fc7_weight = mx.symbol.Variable('rcnn_fc7_weight')
        rcnn_fc7_bias = mx.symbol.Variable('rcnn_fc7_bias')

        rcnn_fc8_weight = mx.symbol.Variable('rcnn_fc8_weight')
        rcnn_fc8_bias = mx.symbol.Variable('rcnn_fc8_bias')

        roi_append_list_m = dict()
        roi_append_list_s = dict()
        roi_append_list_l = dict()

        roi_append_list = dict()
       
        for ii in cfg.network.RCNN_FEAT_STRIDE:

        
            roi_pool = mx.symbol.contrib.ROIAlign(
                name='roi_pool_%s'%ii, data=conv_fpn_feat['stride%s'%ii], rois=rois, pooled_size=(7, 7), spatial_scale=1.0/ii)

            #conv_panet_feat['panet%s'%ii]
            # 2 fc
            print 'roialign is done'
            flatten = mx.symbol.Flatten(data=roi_pool,name='flatten_%s'%ii)
            fc_new_1 = mx.symbol.FullyConnected(name='fc_new_1_%s'%ii, data=flatten, num_hidden=1024,weight=rcnn_fc6_weight,bias=rcnn_fc6_bias)
            fc_new_1_relu = mx.sym.Activation(data=fc_new_1, act_type='relu', name='fc_new_1_relu_%s'%ii)
            roi_append_list.update({'fc_new%s'%ii:fc_new_1_relu})
          


            '''
            roi_pool_m = mx.symbol.contrib.ROIAlign(
                name='roi_pool_m_%s'%ii, data=conv_panet_feat['panet%s'%ii], rois=rois, pooled_size=(7,7), spatial_scale=1.0/ii)

            # 2 fc
            #print 'roialign is done'
            flatten_m = mx.symbol.Flatten(data=roi_pool_m,name='flatten_m_%s'%ii)
            fc_new_1_m = mx.symbol.FullyConnected(name='fc_new_1_m_%s'%ii, data=flatten_m, num_hidden=512,weight=rcnn_fc6_weight,bias=rcnn_fc6_bias)
            fc_new_1_relu_m = mx.sym.Activation(data=fc_new_1_m, act_type='relu', name='fc_new_1_relu_m_%s'%ii)
            roi_append_list_m.update({'fc_new_m%s'%ii:fc_new_1_relu_m})


            #if len(rois_d['rois_14']) == 0:
            #    fc_new_1_relu_l = mx.symbol((1, 512), dtype=rois_d.dtype)
            #    roi_append_list_l.update({'fc_new_l%s'%ii:fc_new_1_relu_l})

            #else:    
            roi_pool_l = mx.symbol.contrib.ROIAlign(
                name='roi_pool_l_%s'%ii, data=conv_panet_feat['panet%s'%ii], rois=rois, pooled_size=(4,14), spatial_scale=1.0/ii)

            # 2 fc
            #print 'roialign is done'
            flatten_l = mx.symbol.Flatten(data=roi_pool_l,name='flatten_l_%s'%ii)
            fc_new_1_l = mx.symbol.FullyConnected(name='fc_new_1_l_%s'%ii, data=flatten_l, num_hidden=512,weight=rcnn_fc7_weight,bias=rcnn_fc7_bias)
            fc_new_1_relu_l = mx.sym.Activation(data=fc_new_1_l, act_type='relu', name='fc_new_1_relu_l_%s'%ii)
            roi_append_list_l.update({'fc_new_l%s'%ii:fc_new_1_relu_l})


            
            #if len(rois_d['rois_41']) == 0:
            #    fc_new_1_relu_s = mx.symbol((1, 512), dtype=rois_d.dtype)
            #    roi_append_list_s.update({'fc_new_s%s'%ii:fc_new_1_relu_s})

            #else:    

            roi_pool_s = mx.symbol.contrib.ROIAlign(
                name='roi_pool_s_%s'%ii, data=conv_panet_feat['panet%s'%ii], rois=rois, pooled_size=(14,4), spatial_scale=1.0/ii)

            # 2 fc
            #print 'roialign is done'
            flatten_s = mx.symbol.Flatten(data=roi_pool_s,name='flatten_s_%s'%ii)
            fc_new_1_s = mx.symbol.FullyConnected(name='fc_new_1_s_%s'%ii, data=flatten_s, num_hidden=512,weight=rcnn_fc8_weight,bias=rcnn_fc8_bias)
            fc_new_1_relu_s = mx.sym.Activation(data=fc_new_1_s, act_type='relu', name='fc_new_1_relu_s_%s'%ii)
            roi_append_list_s.update({'fc_new_s%s'%ii:fc_new_1_relu_s})
          



        roi_sum_m = roi_append_list_m['fc_new_m32'] + roi_append_list_m['fc_new_m16'] + roi_append_list_m['fc_new_m8'] + roi_append_list_m['fc_new_m4']
        roi_sum_l = roi_append_list_l['fc_new_l32'] + roi_append_list_l['fc_new_l16'] + roi_append_list_l['fc_new_l8'] + roi_append_list_l['fc_new_l4']
        roi_sum_s = roi_append_list_s['fc_new_s32'] + roi_append_list_s['fc_new_s16'] + roi_append_list_s['fc_new_s8'] + roi_append_list_s['fc_new_s4']

        roi_sum = mx.symbol.concat(roi_sum_m,roi_sum_s,roi_sum_l,dim=1)
        '''

      
        roi_sum = roi_append_list['fc_new32'] + roi_append_list['fc_new16'] + roi_append_list['fc_new8'] + roi_append_list['fc_new4']

        fc_new_2 = mx.symbol.FullyConnected(name='fc_new_2', data=roi_sum, num_hidden=1024)
        fc_new_2_relu = mx.sym.Activation(data=fc_new_2, act_type='relu', name='fc_new_2_relu')

        # cls_score/bbox_pred
        cls_score = mx.symbol.FullyConnected(name='cls_score', data=fc_new_2_relu, num_hidden=num_classes)
        bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=fc_new_2_relu, num_hidden=num_reg_classes * 8)

        if is_train:

            if cfg.TRAIN.ENABLE_OHEM:
                labels_ohem, bbox_weights_ohem = mx.sym.Custom(op_type='BoxAnnotatorOHEM', num_classes=num_classes,
                                                               num_reg_classes=num_reg_classes,
                                                               roi_per_img=cfg.TRAIN.BATCH_ROIS_OHEM,
                                                               cls_score=cls_score, bbox_pred=bbox_pred, labels=label,
                                                               bbox_targets=bbox_target, bbox_weights=bbox_weight)
                cls_prob = mx.sym.SoftmaxOutput(name='cls_prob', data=cls_score, label=labels_ohem,
                                                normalization='valid', use_ignore=True, ignore_label=-1)
                bbox_loss_ = bbox_weights_ohem * mx.sym.smooth_l1(name='bbox_loss_', scalar=1.0,
                                                                  data=(bbox_pred - bbox_target))
                bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_,
                                            grad_scale=1.0 / cfg.TRAIN.BATCH_ROIS_OHEM)
                rcnn_label = labels_ohem
            else:
                # cls_prob = mx.sym.SoftmaxOutput(name='cls_prob', data=cls_score, label=label, normalization='valid')
                cls_prob = mx.sym.Custom(op_type='FocalLoss', name='cls_prob', data=cls_score, labels=label, gamma= 2,alpha = 0.25)
                bbox_loss_ = bbox_weight * mx.sym.smooth_l1(name='bbox_loss_', scalar=1.0,
                                                            data=(bbox_pred - bbox_target))
                bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=1.0 / cfg.TRAIN.BATCH_ROIS)
                rcnn_label = label

            # reshape output
            rcnn_label = mx.sym.Reshape(data=rcnn_label, shape=(cfg.TRAIN.BATCH_IMAGES, -1), name='label_reshape')
            cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TRAIN.BATCH_IMAGES, -1, num_classes),
                                      name='cls_prob_reshape')
            bbox_loss = mx.sym.Reshape(data=bbox_loss, shape=(cfg.TRAIN.BATCH_IMAGES, -1, 8 * num_reg_classes),
                                       name='bbox_loss_reshape')
            group = mx.sym.Group([rpn_cls_prob, rpn_bbox_loss, cls_prob, bbox_loss, mx.sym.BlockGrad(rcnn_label)])
        else:
            cls_prob = mx.sym.SoftmaxActivation(name='cls_prob', data=cls_score)
            cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TEST.BATCH_IMAGES, -1, num_classes),
                                      name='cls_prob_reshape')
            bbox_pred = mx.sym.Reshape(data=bbox_pred, shape=(cfg.TEST.BATCH_IMAGES, -1, 8 * num_reg_classes),
                                       name='bbox_pred_reshape')
            group = mx.sym.Group([rois, cls_prob, bbox_pred])

        self.sym = group
        return group

    def init_weight_rcnn(self, cfg, arg_params, aux_params):


        arg_params['rcnn_fc6_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rcnn_fc6_weight'])
        arg_params['rcnn_fc6_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rcnn_fc6_bias'])
        #arg_params['rcnn_fc7_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rcnn_fc7_weight'])
        #arg_params['rcnn_fc7_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rcnn_fc7_bias'])
        #arg_params['rcnn_fc8_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rcnn_fc8_weight'])
        #arg_params['rcnn_fc8_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rcnn_fc8_bias'])


        arg_params['fc_new_2_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fc_new_2_weight'])
        arg_params['fc_new_2_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fc_new_2_bias'])

        arg_params['cls_score_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['cls_score_weight'])
        arg_params['cls_score_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['cls_score_bias'])
        arg_params['bbox_pred_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['bbox_pred_weight'])
        arg_params['bbox_pred_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['bbox_pred_bias'])

    def init_weight_rpn(self, cfg, arg_params, aux_params):
        arg_params['rpn_conv_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rpn_conv_weight'])
        arg_params['rpn_conv_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_conv_bias'])
        arg_params['rpn_conv_cls_weight'] = mx.random.normal(0, 0.01,
                                                              shape=self.arg_shape_dict['rpn_conv_cls_weight'])
        arg_params['rpn_conv_cls_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_conv_cls_bias'])

        arg_params['rpn_conv_bbox_weight'] = mx.random.normal(0, 0.01,
                                                              shape=self.arg_shape_dict['rpn_conv_bbox_weight'])
        arg_params['rpn_conv_bbox_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_conv_bbox_bias'])


        #arg_params['conv_feat_sample_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['conv_feat_sample_weight'])


        arg_params['P5_lateral_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['P5_lateral_weight'])
        arg_params['P5_lateral_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['P5_lateral_bias'])
        arg_params['P4_lateral_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['P4_lateral_weight'])
        arg_params['P4_lateral_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['P4_lateral_bias'])
        arg_params['P3_lateral_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['P3_lateral_weight'])
        arg_params['P3_lateral_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['P3_lateral_bias'])
        arg_params['P2_lateral_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['P2_lateral_weight'])
        arg_params['P2_lateral_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['P2_lateral_bias'])

        arg_params['P4_aggregate_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['P4_aggregate_weight'])
        arg_params['P4_aggregate_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['P4_aggregate_bias'])
        arg_params['P3_aggregate_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['P3_aggregate_weight'])
        arg_params['P3_aggregate_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['P3_aggregate_bias'])
        arg_params['P2_aggregate_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['P2_aggregate_weight'])
        arg_params['P2_aggregate_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['P2_aggregate_bias'])

       
        arg_params['N2_sample_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['N2_sample_weight'])
        arg_params['N2_sample_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['N2_sample_bias'])
        arg_params['N3_sample_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['N3_sample_weight'])
        arg_params['N3_sample_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['N3_sample_bias'])
        arg_params['N4_sample_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['N4_sample_weight'])
        arg_params['N4_sample_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['N4_sample_bias'])

        arg_params['N3_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['N3_weight'])
        arg_params['N3_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['N3_bias'])
        arg_params['N4_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['N4_weight'])
        arg_params['N4_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['N4_bias'])
        arg_params['N5_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['N5_weight'])
        arg_params['N5_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['N5_bias'])
       

        arg_params['P5_upsampling_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['P5_upsampling_weight'])
        arg_params['P4_upsampling_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['P4_upsampling_weight'])
        arg_params['P3_upsampling_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['P3_upsampling_weight'])

        #arg_params['P5_upsampling_2_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['P5_upsampling_2_weight'])
        #arg_params['P5_upsampling_3_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['P5_upsampling_3_weight'])
        #arg_params['P4_upsampling_2_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['P4_upsampling_2_weight'])



    def init_weight(self, cfg, arg_params, aux_params):
        self.init_weight_rpn(cfg, arg_params, aux_params)
        self.init_weight_rcnn(cfg, arg_params, aux_params)

