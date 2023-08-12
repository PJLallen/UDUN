#!/usr/bin/python3
# coding=utf-8
import torch.nn as nn
import torch.nn.functional as F

from model.modules.weight_init import weight_init


class TSA(nn.Module):
    def __init__(self):
        super(TSA, self).__init__()

        self.act = nn.Sigmoid()
        self.conv1 = nn.Conv2d(64, 1, kernel_size=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(32)

    def forward(self, feat_trunk, feat_struct):

        y = F.relu(self.bn4(self.conv4(feat_trunk)), inplace=True)
        x = self.act(self.conv1(feat_trunk))
        x = x * feat_struct
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = F.relu(self.bn3(self.conv3(x + feat_struct)), inplace=True)
        y = F.relu(self.bn5(self.conv5(x + y)), inplace=True)

        return y

    def initialize(self):
        weight_init(self)


class MSA_256(nn.Module):
    def __init__(self):
        super(MSA_256, self).__init__()

        self.act = nn.Sigmoid()
        self.conv1 = nn.Conv2d(32, 1, kernel_size=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(32)

    def forward(self, feat_trunk, feat_struct):

        y = F.relu(self.bn4(self.conv4(feat_trunk)), inplace=True)
        x = self.act(self.conv1(feat_trunk))
        x = x * feat_struct
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = F.relu(self.bn3(self.conv3(x + feat_struct)), inplace=True)
        y = F.relu(self.bn5(self.conv5(x + y)), inplace=True)

        return y

    def initialize(self):
        weight_init(self)

class MSA_512(nn.Module):
    def __init__(self):
        super(MSA_512, self).__init__()

        self.act = nn.Sigmoid()
        self.conv1 = nn.Conv2d(32, 1, kernel_size=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(32)

    def forward(self, feat_mask, feat_struct):

        y = F.relu(self.bn4(self.conv4(feat_mask )), inplace=True)
        x = self.act(self.conv1(feat_mask))
        x = x * feat_struct
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = F.relu(self.bn3(self.conv3(x + feat_struct)), inplace=True)
        y = F.relu(self.bn5(self.conv5(x+y)), inplace=True)

        return y

    def initialize(self):
        weight_init(self)



class MSA_1024(nn.Module):
    def __init__(self):
        super(MSA_1024, self).__init__()

        self.act = nn.Sigmoid()
        self.conv1 = nn.Conv2d(32, 1, kernel_size=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(32, 16, kernel_size=1)
        self.bn4 = nn.BatchNorm2d(16)
        self.conv5 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(16)

    def forward(self, feat_mask, feat_struct):

        y = F.relu(self.bn4(self.conv4(feat_mask)), inplace=True)
        x = self.act(self.conv1(feat_mask))
        x = x * feat_struct
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = F.relu(self.bn3(self.conv3(x + feat_struct)), inplace=True)
        y = F.relu(self.bn5(self.conv5(x+y)), inplace=True)

        return y

    def initialize(self):
        weight_init(self)





class UnionDe(nn.Module):
    def __init__(self):
        super(UnionDe, self).__init__()
        self.TSA_0 = TSA()
        self.TSA_1 = TSA()
        self.TSA_2 = TSA()
        self.MSA_3 = MSA_256()
        self.MSA_4 = MSA_512()
        self.MSA_5 = MSA_1024()

        self.conv_1 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.conv_2 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.conv_3 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.conv_4 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.conv_4_reduce = nn.Sequential(nn.Conv2d(32, 16, kernel_size=3, padding=1),
                                           nn.BatchNorm2d(16), nn.ReLU(inplace=True))
        self.conv_5 = nn.Sequential(nn.Conv2d(16, 16, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(16), nn.ReLU(inplace=True))

    def forward(self, feat_trunk, feat_struct):
        mask = self.TSA_0(feat_trunk[0], feat_struct[0])

        temp = self.TSA_1(feat_trunk[1], feat_struct[1])
        maskup = F.interpolate(mask, size=temp.size()[2:], mode='bilinear')
        temp = maskup + temp
        mask = self.conv_1(temp)

        temp = self.TSA_2(feat_trunk[2], feat_struct[2])
        maskup = F.interpolate(mask, size=temp.size()[2:], mode='bilinear')
        temp = maskup + temp
        mask = self.conv_2(temp)

        maskup = F.interpolate(mask, size=feat_struct[3].size()[2:], mode='bilinear')
        temp = self.MSA_3(maskup, feat_struct[3])
        temp = maskup + temp
        mask = self.conv_3(temp)

        maskup = F.interpolate(mask, size=feat_struct[4].size()[2:], mode='bilinear')
        temp = self.MSA_4(maskup, feat_struct[4])
        temp = maskup + temp
        mask = self.conv_4(temp)

        maskup = F.interpolate(mask, size=feat_struct[5].size()[2:], mode='bilinear')
        temp = self.MSA_5(maskup, feat_struct[5])
        maskup = self.conv_4_reduce(maskup)
        temp = maskup + temp
        mask = self.conv_5(temp)

        return mask

    def initialize(self):
        weight_init(self)
