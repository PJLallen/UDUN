#!/usr/bin/python3
# coding=utf-8
import torch.nn as nn
import torch.nn.functional as F
from model.modules.weight_init import weight_init

class TrunkDe(nn.Module):
    def __init__(self):
        super(TrunkDe, self).__init__()

        # T1
        self.conv_1_2 = nn.Conv2d(64, 64, kernel_size=1)
        self.bn_1_2 = nn.BatchNorm2d(64)
        # T2
        self.conv_2_2 = nn.Conv2d(64, 64, kernel_size=1)
        self.bn_2_2 = nn.BatchNorm2d(64)
        self.conv_2_3 = nn.Conv2d(64, 64, kernel_size=1)
        self.bn_2_3 = nn.BatchNorm2d(64)
        # T3
        self.conv_3_3 = nn.Conv2d(64, 64, kernel_size=1)
        self.bn_3_3 = nn.BatchNorm2d(64)
        self.conv_3_4 = nn.Conv2d(64, 64, kernel_size=1)
        self.bn_3_4 = nn.BatchNorm2d(64)
        # T4
        self.conv_4_4 = nn.Conv2d(64, 64, kernel_size=1)
        self.bn_4_4 = nn.BatchNorm2d(64)
        self.conv_4_5 = nn.Conv2d(64, 64, kernel_size=1)
        self.bn_4_5 = nn.BatchNorm2d(64)
        # T5
        self.conv_5_5 = nn.Conv2d(64, 64, kernel_size=1)
        self.bn_5_5 = nn.BatchNorm2d(64)
        # sum
        self.conv_21 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn_21 = nn.BatchNorm2d(64)
        self.conv_31 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn_31 = nn.BatchNorm2d(64)
        self.conv_41 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn_41 = nn.BatchNorm2d(64)
        self.conv_51 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn_51 = nn.BatchNorm2d(64)

        # T21
        self.conv_21_3 = nn.Conv2d(64, 64, kernel_size=1)
        self.bn_21_3 = nn.BatchNorm2d(64)
        # T31
        self.conv_31_3 = nn.Conv2d(64, 64, kernel_size=1)
        self.bn_31_3 = nn.BatchNorm2d(64)
        self.conv_31_4 = nn.Conv2d(64, 64, kernel_size=1)
        self.bn_31_4 = nn.BatchNorm2d(64)
        # T41
        self.conv_41_4 = nn.Conv2d(64, 64, kernel_size=1)
        self.bn_41_4 = nn.BatchNorm2d(64)
        self.conv_41_5 = nn.Conv2d(64, 64, kernel_size=1)
        self.bn_41_5 = nn.BatchNorm2d(64)
        # T51
        self.conv_51_5 = nn.Conv2d(64, 64, kernel_size=1)
        self.bn_51_5 = nn.BatchNorm2d(64)
        # sum
        self.conv_32 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn_32 = nn.BatchNorm2d(64)
        self.conv_42 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn_42 = nn.BatchNorm2d(64)
        self.conv_52 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn_52 = nn.BatchNorm2d(64)

        # T32
        self.conv_32_4 = nn.Conv2d(64, 64, kernel_size=1)
        self.bn_32_4 = nn.BatchNorm2d(64)
        # T42
        self.conv_42_4 = nn.Conv2d(64, 64, kernel_size=1)
        self.bn_42_4 = nn.BatchNorm2d(64)
        self.conv_42_5 = nn.Conv2d(64, 64, kernel_size=1)
        self.bn_42_5 = nn.BatchNorm2d(64)
        # T52
        self.conv_52_5 = nn.Conv2d(64, 64, kernel_size=1)
        self.bn_52_5 = nn.BatchNorm2d(64)
        # sum
        self.conv_43 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn_43 = nn.BatchNorm2d(64)
        self.conv_53 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn_53 = nn.BatchNorm2d(64)

        # T43
        self.conv_43_5 = nn.Conv2d(64, 64, kernel_size=1)
        self.bn_43_5 = nn.BatchNorm2d(64)
        # T53
        self.conv_53_5 = nn.Conv2d(64, 64, kernel_size=1)
        self.bn_53_5 = nn.BatchNorm2d(64)
        # sum
        self.conv_54 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn_54 = nn.BatchNorm2d(64)


    def forward(self, in_feat):
        # T1
        out_1_2 = F.interpolate(in_feat[0], size=in_feat[1].size()[2:], mode='bilinear')
        out_1_2 = F.relu(self.bn_1_2(self.conv_1_2(out_1_2)), inplace=True)
        # T2
        out_2_2 = in_feat[1]
        out_2_2 = F.relu(self.bn_2_2(self.conv_2_2(out_2_2)), inplace=True)
        out_2_3 = F.interpolate(in_feat[1], size=in_feat[2].size()[2:], mode='bilinear')
        out_2_3 = F.relu(self.bn_2_3(self.conv_2_3(out_2_3)), inplace=True)
        # T3
        out_3_3 = in_feat[2]
        out_3_3 = F.relu(self.bn_3_3(self.conv_3_3(out_3_3)), inplace=True)
        out_3_4 = F.interpolate(in_feat[2], size=in_feat[3].size()[2:], mode='bilinear')
        out_3_4 = F.relu(self.bn_3_4(self.conv_3_4(out_3_4)), inplace=True)
        # T4
        out_4_4 = in_feat[3]
        out_4_4 = F.relu(self.bn_4_4(self.conv_4_4(out_4_4)), inplace=True)
        out_4_5 = F.interpolate(in_feat[3], size=in_feat[4].size()[2:], mode='bilinear')
        out_4_5 = F.relu(self.bn_4_5(self.conv_4_5(out_4_5)), inplace=True)
        # T5
        out_5_5 = in_feat[4]
        out_5_5 = F.relu(self.bn_5_5(self.conv_5_5(out_5_5)), inplace=True)
        # sum
        out_21 = out_1_2 + out_2_2
        out_21 = F.relu(self.bn_21(self.conv_21(out_21)), inplace=True)
        out_31 = out_2_3 + out_3_3
        out_31 = F.relu(self.bn_31(self.conv_31(out_31)), inplace=True)
        out_41 = out_3_4 + out_4_4
        out_41 = F.relu(self.bn_41(self.conv_41(out_41)), inplace=True)
        out_51 = out_4_5 + out_5_5
        out_51 = F.relu(self.bn_51(self.conv_51(out_51)), inplace=True)

        # T21
        out_21_3 = F.interpolate(out_21, size=in_feat[2].size()[2:], mode='bilinear')
        out_21_3 = F.relu(self.bn_21_3(self.conv_21_3(out_21_3)), inplace=True)
        # T31
        out_31_3 = out_31
        out_31_3 = F.relu(self.bn_31_3(self.conv_31_3(out_31_3)), inplace=True)
        out_31_4 = F.interpolate(out_31, size=in_feat[3].size()[2:], mode='bilinear')
        out_31_4 = F.relu(self.bn_31_4(self.conv_31_4(out_31_4)), inplace=True)
        # T41
        out_41_4 = out_41
        out_41_4 = F.relu(self.bn_41_4(self.conv_41_4(out_41_4)), inplace=True)
        out_41_5 = F.interpolate(out_41, size=in_feat[4].size()[2:], mode='bilinear')
        out_41_5 = F.relu(self.bn_41_5(self.conv_41_5(out_41_5)), inplace=True)
        # T51
        out_51_5 = out_51
        out_51_5 = F.relu(self.bn_51_5(self.conv_51_5(out_51_5)), inplace=True)
        # sum
        out_32 = out_21_3 + out_31_3
        out_32 = F.relu(self.bn_32(self.conv_32(out_32)), inplace=True)
        out_42 = out_31_4 + out_41_4
        out_42 = F.relu(self.bn_42(self.conv_42(out_42)), inplace=True)
        out_52 = out_41_5 + out_51_5
        out_52 = F.relu(self.bn_52(self.conv_52(out_52)), inplace=True)

        # T32
        out_32_4 = F.interpolate(out_32, size=in_feat[3].size()[2:], mode='bilinear')
        out_32_4 = F.relu(self.bn_32_4(self.conv_32_4(out_32_4)), inplace=True)
        # T42
        out_42_4 = out_42
        out_42_4 = F.relu(self.bn_42_4(self.conv_42_4(out_42_4)), inplace=True)
        out_42_5 = F.interpolate(out_42, size=in_feat[4].size()[2:], mode='bilinear')
        out_42_5 = F.relu(self.bn_42_5(self.conv_42_5(out_42_5)), inplace=True)
        # T52
        out_52_5 = out_52
        out_52_5 = F.relu(self.bn_52_5(self.conv_52_5(out_52_5)), inplace=True)
        # sum
        out_43 = out_32_4 + out_42_4
        out_43 = F.relu(self.bn_43(self.conv_43(out_43)), inplace=True)  #
        out_53 = out_52_5 + out_42_5
        out_53 = F.relu(self.bn_53(self.conv_53(out_53)), inplace=True)  #

        # T43
        out_43_5 = F.interpolate(out_43, size=in_feat[4].size()[2:], mode='bilinear')
        out_43_5 = F.relu(self.bn_43_5(self.conv_43_5(out_43_5)), inplace=True)
        # T53
        out_53_5 = out_53
        out_53_5 = F.relu(self.bn_53_5(self.conv_53_5(out_53_5)), inplace=True)
        # sum
        out_54 = out_43_5 + out_53_5
        out_54 = F.relu(self.bn_54(self.conv_54(out_54)), inplace=True)  #

        return out_32, out_43, out_54

    def initialize(self):
        weight_init(self)
