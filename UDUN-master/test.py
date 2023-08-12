#!/usr/bin/python3
# coding=utf-8
import os
import sys

sys.path.insert(0, '/')
sys.dont_write_bytecode = True
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import torch.nn.functional as F

plt.ion()
import torch
import dataset
from torch.utils.data import DataLoader

from model.UDUNet import UDUN


class Test(object):
    def __init__(self, Dataset, Network, Path, weight):

        ## dataset
        self.cfg = Dataset.Config(datapath=Path, snapshot=weight, mode='test')
        # self.cfg = Dataset.Config(datapath=Path, snapshot='', mode='test')

        self.data = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=0)
        ## network
        self.net = Network(self.cfg)
        self.net.train(False)
        self.net.cuda()

    def save_body_detail(self, save_name):
        with torch.no_grad():
            res = []
            for image, (H, W), name in self.loader:
                image, shape = image.cuda().float(), (H, W)

                # start = time.time()
                out_trunk, out_struct, out_mask = self.net(image)
                out_mask = F.interpolate(out_mask, size=shape, mode='bilinear')
                # end = time.time()
                # res.append(end - start)

                predmask = torch.sigmoid(out_mask[0, 0]).cpu().numpy() * 255
                head = self.cfg.datapath + save_name
                if not os.path.exists(head):
                    os.makedirs(head)
                cv2.imwrite(head + '/' + name[0] + '.png', np.round(predmask))

            # time_sum = 0
            # for i in res:
            #     time_sum += i
            # print("FPS: %f" % (1.0 / (time_sum / len(res))))


if __name__ == '__main__':
    # 测试集合
    t = Test(dataset, UDUN, r'/media/pjl307/data/experiment/ZZJ/DIS/DIS5K/DIS-VD', weight=r'/home/pjl307/ZZJ/UDUN-master/model/new_params.pth')
    # 保存body和detail map
    t.save_body_detail(save_name=r'/media/DIS-VD')
