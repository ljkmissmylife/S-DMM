#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 12:38:20 2018

@author: dengbin
"""

import torch
import torch.nn as nn


# Encoder network
class CNNEncoder(nn.Module):
    """Deep Embedding Module"""

    def __init__(self, input_channels, feature_dim=64):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, feature_dim, kernel_size=1, padding=0),
            nn.BatchNorm2d(feature_dim, momentum=1, affine=True),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=1, padding=0),
            nn.BatchNorm2d(feature_dim, momentum=1, affine=True),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=1, padding=0),
            nn.BatchNorm2d(feature_dim, momentum=1, affine=True),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=1, padding=0),
            nn.BatchNorm2d(feature_dim, momentum=1, affine=True),
            nn.ReLU())

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out  # 64
