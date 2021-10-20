#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 12:38:20 2018

@author: dengbin
"""

import torch
import torch.nn as nn


# Relation network
class RelationNetwork(nn.Module):
    """Deep Metric Module"""

    def __init__(self, sample_size=5, feature_dim=64):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=1, padding=0),
            nn.BatchNorm2d(feature_dim, momentum=1, affine=True),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=1, padding=0),
            nn.BatchNorm2d(feature_dim, momentum=1, affine=True),
            nn.ReLU())
        self.layer3 = nn.Conv2d(feature_dim, 1, kernel_size=sample_size, padding=0)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = torch.sigmoid(out)
        return out
