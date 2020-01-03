#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
     # @Time    : 2019-04-27 16:18
     # @Author  : Awiny
     # @Site    :
     # @Project : Action_Video_Visualization
     # @File    : mp_i3d.py
     # @Software: PyCharm
     # @Github  : https://github.com/FingerRec
     # @Blog    : http://fingerrec.github.io
"""
import scipy.io
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #close the warning
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
     # @Time    : 2019-04-08 18:32
     # @Author  : Awiny
     # @Site    :
     # @Project : pytorch_i3d
     # @File    : multi_path_i3d.py
     # @Software: PyCharm
     # @Github  : https://github.com/FingerRec
     # @Blog    : http://fingerrec.github.io
"""
import scipy.io
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #close the warning
import scipy.io
import os
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #close the warning
#========================================================================================
#This network is designed to capture different range dependencies and cobine them.
#With dilated conv and downsample, i want to down the number of parameters.
#The network are divided into 3 parllel network. and across information between them.
#1:64frame input, 56 x 56 input, long range temporal dependencies, call s
#2:16frame input, 112x112, middle range temporal dependencies, call m
#3:4frame input, 224x224, shortest temporal dependencies, call l
#after these network, use tpp to combine them and put it into fc layer
#========================================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import math
from math import exp
import os
import sys
from collections import OrderedDict

class MaxPool3dSamePadding(nn.MaxPool3d):

    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        # print t,h,w
        out_t = np.ceil(float(t) / float(self.stride[0]))
        out_h = np.ceil(float(h) / float(self.stride[1]))
        out_w = np.ceil(float(w) / float(self.stride[2]))
        # print out_t, out_h, out_w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        # print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        # print x.size()
        # print pad
        x = F.pad(x, pad)
        return super(MaxPool3dSamePadding, self).forward(x)

class Unit3D(nn.Module):

    def __init__(self, in_channels,
                 output_channels,
                 kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1),
                 padding=0,
                 dilation=1,
                 activation_fn=F.relu,
                 use_batch_norm=True,
                 use_bias=False,
                 name='unit_3d'):

        """Initializes Unit3D module."""
        super(Unit3D, self).__init__()

        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding

        self.conv3d = nn.Conv3d(in_channels=in_channels,
                                out_channels=self._output_channels,
                                kernel_size=self._kernel_shape,
                                stride=self._stride,
                                dilation=dilation,
                                padding=0,
                                # we always want padding to be 0 here. We will dynamically pad based on input size in forward function
                                bias=self._use_bias)

        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels, eps=0.001, momentum=0.01)

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        # print t,h,w
        out_t = np.ceil(float(t) / float(self._stride[0]))
        out_h = np.ceil(float(h) / float(self._stride[1]))
        out_w = np.ceil(float(w) / float(self._stride[2]))
        # print out_t, out_h, out_w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        # print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        # print x.size()
        # print pad
        x = F.pad(x, pad)
        # print x.size()

        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x

class TemporalPyramidPool3D_2(nn.Module):
    """
    Args:
        out_side (tuple): Length of side in the pooling results of each pyramid layer.

    Inputs:
        - `input`: the input Tensor to invert ([batch, channel, width, height])
    """

    def __init__(self, out_side):
        super(TemporalPyramidPool3D_2, self).__init__()
        self.out_side = out_side
        self.out_t = out_side[0] + out_side[1] + out_side[2]

    def forward(self, x):
        out = None
        for n in self.out_side:
            t_r, w_r, h_r = map(lambda s: math.ceil(s / n), x.size()[2:])  # Receptive Field Size
            s_t, s_w, s_h = map(lambda s: math.floor(s / n), x.size()[2:])  # Stride
            max_pool = nn.MaxPool3d(kernel_size=(t_r, w_r, h_r), stride=(s_t, s_w, s_h))
            y = max_pool(x)
            avg_pool = nn.AdaptiveAvgPool3d((y.size(2), 1, 1))
            y = avg_pool(y)
            # print(y.size())
            if out is None:
                out = y.view(y.size()[0], y.size()[1], -1, 1, 1)
            else:
                out = torch.cat((out, y.view(y.size()[0], y.size()[1], -1, 1, 1)), 2)
        return out

class TemporalPyramidPool3D(nn.Module):
    """
    Args:
        out_side (tuple): Length of side in the pooling results of each pyramid layer.

    Inputs:
        - `input`: the input Tensor to invert ([batch, channel, width, height])
    """

    def __init__(self, out_side):
        super(TemporalPyramidPool3D, self).__init__()
        self.out_side = out_side

    def forward(self, x):
        out = None
        for n in self.out_side:
            avg_pool = nn.AdaptiveMaxPool3d((n, 1, 1))
            y = avg_pool(x)
            if out is None:
                out = y.view(y.size()[0], y.size()[1], -1, 1, 1)
            else:
                out = torch.cat((out, y.view(y.size()[0], y.size()[1], -1, 1, 1)), 2)
        return out

class SpatialPyramidPool3D(nn.Module):
    """
    Args:
        out_side (tuple): Length of side in the pooling results of each pyramid layer.

    Inputs:
        - `input`: the input Tensor to invert ([batch, channel, width, height])
    """

    def __init__(self, out_side):
        super(SpatialPyramidPool3D, self).__init__()
        self.out_side = out_side

    def forward(self, x):
        out = None
        for n in self.out_side:
            max_pool = nn.AdaptiveMaxPool3d((1, n, n))
            y = max_pool(x)
            if out is None:
                out = y.view(y.size()[0], y.size()[1], 1, n*n, 1)
            else:
                out = torch.cat((out, y.view(y.size()[0], y.size()[1], 1, n*n, 1)), 3)
        return out

class InplaceShift(torch.autograd.Function):
    # Special thanks to @raoyongming for the help to this function
    @staticmethod
    def forward(ctx, input, fold):
        # not support higher order gradient
        # input = input.detach_()
        ctx.fold_ = fold
        n, t, c, h, w = input.size()
        buffer = input.data.new(n, t, fold, h, w).zero_()
        buffer[:, :-1] = input.data[:, 1:, :fold]
        input.data[:, :, :fold] = buffer
        buffer.zero_()
        buffer[:, 1:] = input.data[:, :-1, fold: 2 * fold]
        input.data[:, :, fold: 2 * fold] = buffer
        return input

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output = grad_output.detach_()
        fold = ctx.fold_
        n, t, c, h, w = grad_output.size()
        buffer = grad_output.data.new(n, t, fold, h, w).zero_()
        buffer[:, 1:] = grad_output.data[:, :-1, :fold]
        grad_output.data[:, :, :fold] = buffer
        buffer.zero_()
        buffer[:, :-1] = grad_output.data[:, 1:, fold: 2 * fold]
        grad_output.data[:, :, fold: 2 * fold] = buffer
        return grad_output, None

class TemporalShuffle(nn.Module):
    def __init__(self, fold_div=8):
        super(TemporalShuffle, self).__init__()
        self.fold_div = fold_div

    def forward(self, x):
        b, t, c, h, w = x.size()
        fold = c // self.fold_div
        out = InplaceShift.apply(x, fold)
        return out.view(b, t, c, h, w)

class MultiDependBlock(nn.Module):
    def __init__(self, in_channel, out_channel, concat=False, fc=False):
        super(MultiDependBlock, self).__init__()
        self.out_channel = out_channel
        self.channel_compress = Unit3D(in_channels=in_channel, output_channels=out_channel,
               kernel_shape=[1, 1, 1],
               stride=(1, 1, 1),
               padding=0,
               activation_fn=None,
               use_batch_norm=False,
               use_bias=True,
               name='channel_compress')
        self.long_range_depen = Unit3D(in_channels=out_channel, output_channels=out_channel,
               kernel_shape=[2, 1, 1],
               stride=(1, 1, 1),
               padding=0,
               activation_fn=None,
               use_batch_norm=False,
               use_bias=True,
               name='long_range_depen')
        self.middle_range_depen = Unit3D(in_channels=out_channel, output_channels=out_channel,
               kernel_shape=[2, 1, 1],
               stride=(1, 1, 1),
               padding=0,
               activation_fn=None,
               use_batch_norm=False,
               use_bias=True,
               name='middle_range_depen')
        self.small_range_depen = Unit3D(in_channels=out_channel, output_channels=out_channel,
               kernel_shape=[2, 1, 1],
               stride=(1, 1, 1),
               padding=0,
               activation_fn=None,
               use_batch_norm=False,
               use_bias=True,
               name='small_range_depen')
        self.local_range_depen = Unit3D(in_channels=out_channel, output_channels=out_channel,
               kernel_shape=[1, 1, 1],
               stride=(1, 1, 1),
               padding=0,
               activation_fn=None,
               use_batch_norm=False,
               use_bias=True,
               name='local_range_depen')
        '''
        self.single_range_depen = Unit3D(in_channels=out_channel, output_channels=out_channel,
               kernel_shape=[1, 1, 1],
               stride=(1, 1, 1),
               padding=0,
               activation_fn=None,
               use_batch_norm=False,
               use_bias=True,
               name='single_range_depen')
        '''
        self.concat = concat
        self.fc = fc
        if self.fc:
            self.fc_fusion = nn.Sequential(
                nn.ReLU(),
                nn.Linear(3 * out_channel, 128),
                nn.ReLU(),
                nn.Linear(128, out_channel),
            )
        #self.dropout_probality = 0.05
    def forward(self, x):
        b, c, t, h, w = x.size()
        spatial_pool_x = nn.AdaptiveAvgPool3d((t,1,1))(x)/2 + nn.AdaptiveMaxPool3d((t,1,1))(x)/2
        #spatial_pool_x = nn.Dropout(self.dropout_probality)(spatial_pool_x)
        spatial_pool_x = self.channel_compress(spatial_pool_x)
        long_range_depen = self.long_range_depen(spatial_pool_x[:,:,::(t-1),:,:])
        middle_range_depen = self.middle_range_depen(spatial_pool_x[:,:,::(t-1)//2,:,:])
        small_range_depen = self.small_range_depen(spatial_pool_x[:,:,::(t-1)//4,:,:])
        local_range_depen = self.local_range_depen(spatial_pool_x[:,:,::(t-1)//7,:,:])
        #single_range_depen = self.single_range_depen(spatial_pool_x[:, :, ::1, :, :])
        '''
        long_range_depen = self.long_range_depen(spatial_pool_x[:,:,::7,:,:])
        middle_range_depen = self.middle_range_depen(spatial_pool_x[:,:,::4,:,:])
        small_range_depen = self.small_range_depen(spatial_pool_x[:,:,::2,:,:])
        local_range_depen = self.local_range_depen(spatial_pool_x[:,:,::1,:,:])
        '''
        if self.fc:
            out = torch.cat((nn.AdaptiveMaxPool3d((1, 1, 1))(long_range_depen).squeeze(2).squeeze(2).squeeze(2), nn.AdaptiveMaxPool3d((1, 1, 1))(middle_range_depen).squeeze(2).squeeze(2).squeeze(2), nn.AdaptiveMaxPool3d((1, 1, 1))(small_range_depen).squeeze(2).squeeze(2).squeeze(2)), dim = 1)
            return self.fc_fusion(out)
        elif self.concat:
            return torch.cat((nn.AdaptiveMaxPool3d((1, 1, 1))(long_range_depen).squeeze(2).squeeze(2).squeeze(2), nn.AdaptiveMaxPool3d((1, 1, 1))(middle_range_depen).squeeze(2).squeeze(2).squeeze(2), nn.AdaptiveMaxPool3d((1, 1, 1))(small_range_depen).squeeze(2).squeeze(2).squeeze(2)), dim = 1)
        else:
            return nn.AdaptiveMaxPool3d((1, 1, 1))(long_range_depen).squeeze(2).squeeze(2).squeeze(2) + \
                   nn.AdaptiveMaxPool3d((1, 1, 1))(middle_range_depen).squeeze(2).squeeze(2).squeeze(2) + \
                   nn.AdaptiveMaxPool3d((1, 1, 1))(small_range_depen).squeeze(2).squeeze(2).squeeze(2) + \
                   nn.AdaptiveMaxPool3d((1, 1, 1))(local_range_depen).squeeze(2).squeeze(2).squeeze(2) #+ \
                   #nn.AdaptiveMaxPool3d((1, 1, 1))(single_range_depen).squeeze(2).squeeze(2).squeeze(2)

class TemporalDependBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(TemporalDependBlock, self).__init__()
        self.out_channel = out_channel
        self.channel_compress = Unit3D(in_channels=in_channel, output_channels=out_channel,
               kernel_shape=[1, 1, 1],
               stride=(1, 1, 1),
               padding=0,
               activation_fn=None,
               use_batch_norm=False,
               use_bias=True,
               name='channel_compress')
        self.tpp = TemporalPyramidPool3D((1,2,4,8))
        self.temporal_conv = Unit3D(in_channels=out_channel, output_channels=out_channel,
                                                kernel_shape=[15, 1, 1],
                                                stride=(15, 1, 1),
                                                padding=0,
                                                activation_fn=None,
                                                use_batch_norm=False,
                                                use_bias=True,
                                                name='latter_temporal_conv')
    def forward(self, x):
        b, c, t, h, w = x.size()
        compress = self.channel_compress(x)
        tpp = self.tpp(compress)
        out = self.temporal_conv(tpp)
        return out.view(b, out.size(1))

class HeavyMultiDependBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(HeavyMultiDependBlock, self).__init__()
        self.out_channel = out_channel
        self.channel_compress = Unit3D(in_channels=in_channel, output_channels=out_channel,
               kernel_shape=[1, 1, 1],
               stride=(1, 1, 1),
               padding=0,
               activation_fn=None,
               use_batch_norm=False,
               use_bias=True,
               name='channel_compress')
        self.long_range_depen = Unit3D(in_channels=out_channel, output_channels=out_channel,
               kernel_shape=[2, 1, 1],
               stride=(1, 1, 1),
               padding=0,
               activation_fn=None,
               use_batch_norm=False,
               use_bias=True,
               name='long_range_depen')
        self.middle_range_depen = Unit3D(in_channels=out_channel, output_channels=out_channel,
               kernel_shape=[2, 1, 1],
               stride=(1, 1, 1),
               padding=0,
               activation_fn=None,
               use_batch_norm=False,
               use_bias=True,
               name='middle_range_depen')
        self.small_range_depen = Unit3D(in_channels=out_channel, output_channels=out_channel,
               kernel_shape=[2, 1, 1],
               stride=(1, 1, 1),
               padding=0,
               activation_fn=None,
               use_batch_norm=False,
               use_bias=True,
               name='small_range_depen')
        self.tpp_1 = TemporalPyramidPool3D((1,2,4))
        self.fusion_1 = Unit3D(in_channels=out_channel, output_channels=out_channel,
               kernel_shape=[7, 1, 1],
               stride=(7, 1, 1),
               padding=0,
               activation_fn=None,
               use_batch_norm=False,
               use_bias=True,
               name='long_range_depen')
        self.tpp_2 = TemporalPyramidPool3D((1,2,4))
        self.fusion_2 = Unit3D(in_channels=out_channel, output_channels=out_channel,
               kernel_shape=[7, 1, 1],
               stride=(7, 1, 1),
               padding=0,
               activation_fn=None,
               use_batch_norm=False,
               use_bias=True,
               name='middle_range_depen')
        self.tpp_3 = TemporalPyramidPool3D((1,2,4))
        self.fusion_3 = Unit3D(in_channels=out_channel, output_channels=out_channel,
               kernel_shape=[7, 1, 1],
               stride=(7, 1, 1),
               padding=0,
               activation_fn=None,
               use_batch_norm=False,
               use_bias=True,
               name='small_range_depen')
    def forward(self, x):
        b, c, t, h, w = x.size()
        spatial_pool_x = nn.AdaptiveAvgPool3d((t,1,1))(x)/2 + nn.AdaptiveMaxPool3d((t,1,1))(x)/2
        spatial_pool_x = self.channel_compress(spatial_pool_x)
        long_range_depen = self.long_range_depen(spatial_pool_x[:,:,::4,:,:])
        middle_range_depen = self.middle_range_depen(spatial_pool_x[:,:,::2,:,:])
        small_range_depen = self.small_range_depen(spatial_pool_x[:,:,::1,:,:])
        long_range_depen = self.tpp_1(long_range_depen)
        middle_range_depen = self.tpp_2(middle_range_depen)
        small_range_depen = self.tpp_3(small_range_depen)
        return self.fusion_1(long_range_depen).squeeze(2).squeeze(2).squeeze(2) + self.fusion_2(middle_range_depen).squeeze(2).squeeze(2).squeeze(2) + self.fusion_3(small_range_depen).squeeze(2).squeeze(2).squeeze(2)

class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, name):
        super(InceptionModule, self).__init__()

        self.b0 = Unit3D(in_channels=in_channels, output_channels=out_channels[0], kernel_shape=[1, 1, 1], padding=0,
                         name=name + '/Branch_0/Conv3d_0a_1x1')
        self.b1a = Unit3D(in_channels=in_channels, output_channels=out_channels[1], kernel_shape=[1, 1, 1], padding=0,
                          name=name + '/Branch_1/Conv3d_0a_1x1')
        self.b1b = Unit3D(in_channels=out_channels[1], output_channels=out_channels[2], kernel_shape=[3, 3, 3],
                          name=name + '/Branch_1/Conv3d_0b_3x3')
        self.b2a = Unit3D(in_channels=in_channels, output_channels=out_channels[3], kernel_shape=[1, 1, 1], padding=0,
                          name=name + '/Branch_2/Conv3d_0a_1x1')
        self.b2b = Unit3D(in_channels=out_channels[3], output_channels=out_channels[4], kernel_shape=[3, 3, 3],
                          name=name + '/Branch_2/Conv3d_0b_3x3')
        self.b3a = MaxPool3dSamePadding(kernel_size=[3, 3, 3],
                                        stride=(1, 1, 1), padding=0)
        self.b3b = Unit3D(in_channels=in_channels, output_channels=out_channels[5], kernel_shape=[1, 1, 1], padding=0,
                          name=name + '/Branch_3/Conv3d_0b_1x1')
        #self.temporal_shift = TemporalShuffle(fold_div=16)
        self.name = name

    def forward(self, x):
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return  torch.cat([b0, b1, b2, b3], dim=1)
        """
        out = torch.cat([b0, b1, b2, b3], dim=1)
        b, c, t, h, w = x.size()
        if t > 16:
            ts_1 = self.temporal_shift(out)
            return out + ts_1
        else:
            '''
            tb0 = self.tba(x)
            tb1 = self.tbb(tb0)
            tb2 = self.tbc(tb1)
            '''
            return out
        """
class TemporalInceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, name):
        super(TemporalInceptionModule, self).__init__()

        self.b0 = Unit3D(in_channels=in_channels, output_channels=out_channels[0], kernel_shape=[1, 1, 1], padding=0,
                         name=name + '/Branch_0/Conv3d_0a_1x1')
        self.b1a = Unit3D(in_channels=in_channels, output_channels=out_channels[1], kernel_shape=[1, 1, 1], padding=0,
                          name=name + '/Branch_1/Conv3d_0a_1x1')
        self.b1b = Unit3D(in_channels=out_channels[1], output_channels=out_channels[2], kernel_shape=[1, 3, 3],
                          name=name + '/Branch_1/Conv3d_0b_3x3')
        self.b2a = Unit3D(in_channels=in_channels, output_channels=out_channels[3], kernel_shape=[1, 1, 1], padding=0,
                          name=name + '/Branch_2/Conv3d_0a_1x1')
        self.b2b = Unit3D(in_channels=out_channels[3], output_channels=out_channels[4], kernel_shape=[1, 3, 3],
                          name=name + '/Branch_2/Conv3d_0b_3x3')
        self.b3a = MaxPool3dSamePadding(kernel_size=[1, 3, 3],
                                        stride=(1, 1, 1), padding=0)
        self.b3b = Unit3D(in_channels=in_channels, output_channels=out_channels[5], kernel_shape=[1, 1, 1], padding=0,
                          name=name + '/Branch_3/Conv3d_0b_1x1')
        self.name = name

    def forward(self, x):
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0, b1, b2, b3], dim=1)

class MultiPathI3d(nn.Module):
    def __init__(self, num_classes=400, spatial_squeeze=True, in_channels=3, dropout_prob=0.5):

        super(MultiPathI3d, self).__init__()
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self.logits = None

        self.Conv3d_1a_7x7 = Unit3D(in_channels=in_channels, output_channels=64, kernel_shape=[7, 7, 7],
                                            stride=(2, 2, 2), padding=(3, 3, 3), name='conv3d_1a_7_7')

        self.MaxPool3d_2a_3x3 = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                          padding=0)
        self.Conv3d_2b_1x1 = Unit3D(in_channels=64, output_channels=64, kernel_shape=[1, 1, 1], padding=0,name='Conv3d_2b_1x1')
        self.Conv3d_2c_3x3 = Unit3D(in_channels=64, output_channels=192, kernel_shape=[3, 3, 3], padding=1, name='Conv3d_2c_3x3')
        self.maxpool_1 = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2), padding=0)
        self.Mixed_3b = InceptionModule(192, [64, 96, 128, 16, 32, 32], 'Mixed_3b')
        self.Mixed_3c = InceptionModule(256, [128, 128, 192, 32, 96, 64], 'Mixed_3c')
        self.maxpool_2 = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2), padding=0)
        self.Mixed_4b = InceptionModule(128 + 192 + 96 + 64, [192, 96, 208, 16, 48, 64], 'Mixed_4b')
        self.Mixed_4c = InceptionModule(192 + 208 + 48 + 64, [160, 112, 224, 24, 64, 64], 'Mixed_4c')
        self.Mixed_4d = InceptionModule(160 + 224 + 64 + 64, [128, 128, 256, 24, 64, 64], 'Mixed_4d')
        self.Mixed_4e = InceptionModule(128 + 256 + 64 + 64, [112, 144, 288, 32, 64, 64], 'Mixed_4e')
        self.Mixed_4f = InceptionModule(112 + 288 + 64 + 64, [256, 160, 320, 32, 128, 128], 'Mixed_4f')
        self.maxpool_3 = MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2), padding=0)
        self.Mixed_5b = InceptionModule(256 + 320 + 128 + 128, [256, 160, 320, 32, 128, 128], 'Mixed_5b')
        self.Mixed_5c = InceptionModule(256 + 320 + 128 + 128, [384, 192, 384, 48, 128, 128], 'Mixed_5c')
        self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7],
                                     stride=(1, 1, 1))
        self.dropout = nn.Dropout(dropout_prob)
        self.dropout_probality = dropout_prob


        #=================================Multi Stride Multi Path Compress Network======================
        self.s_depend = MultiDependBlock(480, self._num_classes, concat=False, fc=False)
        self.m_depend = MultiDependBlock(832, self._num_classes, concat=False, fc=False)
        self.l_depend = MultiDependBlock(1024, self._num_classes, concat=False, fc=False)
        self.concat = False
        self.fc_fusion = False
        if self.concat:
            self.fc = nn.Linear(self._num_classes*9, self._num_classes)
    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, T, H, W = y.size()
        return F.interpolate(x, size=(T, H, W), mode='trilinear', align_corners=True)/2 + y/2
        #return F.upsample(x, size=(T, H, W), mode='trilinear') + y

    def constrain(self, x):
        alpha = 0.2
        beta = 5
        return 1/(beta+exp(-x)) + alpha

    def forward(self, x):
        x = self.Conv3d_1a_7x7(x)
        x = self.MaxPool3d_2a_3x3(x)
        x = self.Conv3d_2b_1x1(x)
        x = self.Conv3d_2c_3x3(x)
        x = self.maxpool_1(x)
        x = self.Mixed_3b(x)
        x = self.Mixed_3c(x)
        path_s = x
        x = self.maxpool_2(x)
        x = self.Mixed_4b(x)
        x = self.Mixed_4c(x)
        x = self.Mixed_4d(x)
        x = self.Mixed_4e(x)
        x = self.Mixed_4f(x)
        path_m = x
        x = self.maxpool_3(x)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        path_l = x
        plot_s = path_s
        plot_m = path_m
        plot_l = path_l
        path_s = self.s_depend(path_s)
        path_m = self.m_depend(path_m)
        path_l = self.l_depend(path_l)
        #main_path = self.main_depend(x)
        main_path = path_m + path_l + path_s
        if self.concat:
            out =  torch.cat((self.s_depend(path_s), self.m_depend(path_m), self.l_depend(path_l)), dim=1)
            return self.fc(out) #+ temporal_path
        else:
            return main_path, plot_s, plot_m, plot_l, path_s, path_m, path_l
