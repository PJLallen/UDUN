# coding=utf-8
import os
import cv2
import numpy as np

# body detail
def split_map(datapath):
    """
    From https://https://github.com/weijun88/LDF/blob/master/utils.py
    """
    print(datapath)
    for name in os.listdir(datapath + '/gt'):
        mask = cv2.imread(datapath + '/gt/' + name, 0)
        body = cv2.blur(mask, ksize=(5, 5))
        body = cv2.distanceTransform(body, distanceType=cv2.DIST_L2, maskSize=5)
        body = body ** 0.5
        tmp = body[np.where(body > 0)]

        if len(tmp) != 0:
            body[np.where(body > 0)] = np.floor(
                tmp / np.max(tmp) * 255)

        if not os.path.exists(datapath+'/trunk-origin/'):
            os.makedirs(datapath+'/trunk-origin/')
        cv2.imwrite(datapath+'/trunk-origin/'+name, body)

        if not os.path.exists(datapath+'/struct-origin/'):
            os.makedirs(datapath+'/struct-origin/')
        cv2.imwrite(datapath+'/struct-origin/'+name, mask-body)


if __name__ == '__main__':
    split_map('../gt')


