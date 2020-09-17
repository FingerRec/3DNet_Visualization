#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
     # @Time    : 2020-09-17 16:02
     # @Author  : Awiny
     # @Site    :
     # @Project : amax_Action_Video_Visualization
     # @File    : model.py
     # @Software: PyCharm
     # @Github  : https://github.com/FingerRec
     # @Blog    : http://fingerrec.github.io
"""
import scipy.io
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #close the warning

from torch import nn
import torch.nn.functional as F
import torch
import numpy as np

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.view(input.size(0), -1)


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1./self.power)
        out = x.div(norm)
        return out


class Sharpen(nn.Module):
    def __init__(self, tempeature=0.5):
        super(Sharpen, self).__init__()
        self.T = tempeature

    def forward(self, probabilities):
        tempered = torch.pow(probabilities, 1 / self.T)
        tempered = tempered / tempered.sum(dim=-1, keepdim=True)
        return tempered

class MotionEnhance(nn.Module):
    def __init__(self, beta=1, maxium_radio=0.3):
        super(MotionEnhance, self).__init__()
        self.beta = beta
        self.maxium_radio = maxium_radio

    def forward(self, x):
        b, c, t, h, w = x.size()
        mean = nn.AdaptiveAvgPool3d((1, h, w))(x)
        lam = np.random.beta(self.beta, self.beta) * self.maxium_radio
        out = (x - mean * lam) * (1 / (1 - lam))
        return out