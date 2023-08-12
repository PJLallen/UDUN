#!/usr/bin/python3
# coding=utf-8

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


########################### Data Augmentation ###########################
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask=None, trunk=None, struct=None):
        image = (image - self.mean) / self.std
        if mask is None:
            return image
        return image, mask / 255, trunk / 255, struct / 255


class RandomCrop(object):
    def __call__(self, image, mask=None, trunk=None, struct=None):
        H, W, _ = image.shape
        randw = np.random.randint(W / 8)
        randh = np.random.randint(H / 8)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth, H + offseth - randh, offsetw, W + offsetw - randw
        if mask is None:
            return image[p0:p1, p2:p3, :]
        return image[p0:p1, p2:p3, :], mask[p0:p1, p2:p3], trunk[p0:p1, p2:p3], struct[p0:p1, p2:p3]


class RandomFlip(object):
    def __call__(self, image, mask=None, trunk=None, struct=None):
        if np.random.randint(2) == 0:
            if mask is None:
                return image[:, ::-1, :].copy()
            return image[:, ::-1, :].copy(), mask[:, ::-1].copy(), trunk[:, ::-1].copy(), struct[:, ::-1].copy()
        else:
            if mask is None:
                return image
            return image, mask, trunk, struct


class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask=None, trunk=None, struct=None):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        if mask is None:
            return image
        mask = cv2.resize(mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        trunk = cv2.resize(trunk, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        struct = cv2.resize(struct, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return image, mask, trunk, struct


class ToTensor(object):
    def __call__(self, image, mask=None, trunk=None, struct=None):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        if mask is None:
            return image
        mask = torch.from_numpy(mask)
        trunk = torch.from_numpy(trunk)
        struct = torch.from_numpy(struct)
        return image, mask, trunk, struct


########################### Config File ###########################
class Config(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.mean = np.array([[[124.55, 118.90, 102.94]]])
        self.std = np.array([[[56.77, 55.97, 57.50]]])
        print('\nParameters...')
        for k, v in self.kwargs.items():
            print('%-10s: %s' % (k, v))

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            return None


########################### Dataset Class ###########################
class Data(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.normalize = Normalize(mean=cfg.mean, std=cfg.std)
        self.randomcrop = RandomCrop()
        self.randomflip = RandomFlip()
        self.resize = Resize(1024, 1024)
        self.totensor = ToTensor()

        self.samples = os.listdir(os.path.join(self.cfg.datapath, 'im'))

    def __getitem__(self, idx):
        name = self.samples[idx]
        name = name[0:-4]

        image = cv2.imread(self.cfg.datapath + '/im/' + name + '.jpg')[:, :, ::-1].astype(np.float32)

        if self.cfg.mode == 'train':
            mask = cv2.imread(self.cfg.datapath + '/gt/' + name + '.png', 0).astype(np.float32)
            trunk = cv2.imread(self.cfg.datapath + '/trunk-origin/' + name + '.png', 0).astype(np.float32)
            struct = cv2.imread(self.cfg.datapath + '/struct-origin/' + name + '.png', 0).astype(np.float32)

            image, mask, trunk, struct = self.normalize(image, mask, trunk, struct)
            image, mask, trunk, struct = self.randomcrop(image, mask, trunk, struct)  # 裁剪之后所以输入还是1024吗
            image, mask, trunk, struct = self.randomflip(image, mask, trunk, struct)
            return image, mask, trunk, struct
        else:
            shape = image.shape[:2]
            image = self.normalize(image)
            image = self.resize(image)
            image = self.totensor(image)
            return image, shape, name

    def __len__(self):
        return len(self.samples)

    def collate(self, batch):
        size = [1024, 1024, 1024, 1024, 1024][np.random.randint(0, 5)]
        # size = [960, 992, 1024, 1056, 1088][np.random.randint(0, 5)]
        image, mask, trunk, struct = [list(item) for item in zip(*batch)]
        for i in range(len(batch)):
            image[i] = cv2.resize(image[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            mask[i] = cv2.resize(mask[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            trunk[i] = cv2.resize(trunk[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            struct[i] = cv2.resize(struct[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
        image = torch.from_numpy(np.stack(image, axis=0)).permute(0, 3, 1, 2)
        mask = torch.from_numpy(np.stack(mask, axis=0)).unsqueeze(1)
        trunk = torch.from_numpy(np.stack(trunk, axis=0)).unsqueeze(1)
        struct = torch.from_numpy(np.stack(struct, axis=0)).unsqueeze(1)

        return image, mask, trunk, struct
