#!/usr/bin/python3
# coding=utf-8
import torch.nn as nn
import torch.nn.functional as F

from model.modules.weight_init import weight_init


class DCM(nn.Module):
    def __init__(self):
        super(DCM, self).__init__()

        self.convLR0 = nn.Conv2d(64, 32, kernel_size=1)
        self.bnLR0 = nn.BatchNorm2d(32)

        self.convLR1 = nn.Conv2d(64, 32, kernel_size=1)
        self.bnLR1 = nn.BatchNorm2d(32)

        self.convLR2 = nn.Conv2d(64, 32, kernel_size=1)
        self.bnLR2 = nn.BatchNorm2d(32)

        self.convLR3 = nn.Conv2d(32, 32, kernel_size=1)
        self.bnLR3 = nn.BatchNorm2d(32)

    def forward(self, featLR, featHR):

        temp = F.relu(self.bnLR0(self.convLR0(featLR[0])), inplace=True)
        featHR0 = featHR[0] - temp
        temp = F.relu(self.bnLR1(self.convLR1(featLR[1])), inplace=True)
        featHR1 = featHR[1] - temp
        temp = F.relu(self.bnLR2(self.convLR2(featLR[2])), inplace=True)
        featHR2 = featHR[2] - temp

        return featHR0, featHR1, featHR2

    def initialize(self):
        weight_init(self)
