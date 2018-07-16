import os
import numpy as np

import torch, cv2
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def fixed_resize(sample, resolution, flagval=None):
    if flagval is None:
        if sample.ndim == 2:
            flagval = cv2.INTER_NEAREST
        else:
            flagval = cv2.INTER_CUBIC

    if sample.ndim == 2 or (sample.ndim == 3 and sample.shape[2] == 3):
        sample = cv2.resize(sample, resolution, interpolation=flagval)
    else:
        tmp = sample
        sample = np.zeros(np.append(resolution, tmp.shape[2]), dtype=np.float32)
        for ii in range(sample.shape[2]):
            sample[:, :, ii] = cv2.resize(tmp[:, :, ii], resolution, interpolation=flagval)

    return sample

def generate_param_report(logfile, param):
    log_file = open(logfile, 'w')
    for key, val in param.items():
        log_file.write(key + ':' + str(val) + '\n')
    log_file.close()

def cross_entropy2d(logit, target, ignore_index=255, weight=None, size_average=True, batch_average=True):
    n, c, h, w = logit.size()
    # logit = logit.permute(0, 2, 3, 1)
    target = target.squeeze(1)
    criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, size_average=False)
    loss = criterion(logit, target.long())

    if size_average:
        loss /= (h * w)

    if batch_average:
        loss /= n

    return loss

def lr_poly(base_lr, iter_, max_iter=100, power=0.9):
    return base_lr * ((1 - float(iter_) / max_iter) ** power)

def add_size(bbox, imgsize, add_pix=50):
    '''
    bbox: (x1, y1, x2, y2)
    imgsize: [h,w,c]
    '''
    h,w = imgsize[:2]
    # extend bbox with ad_pix
    bbox_new = [bbox[0]-add_pix, bbox[1]-add_pix, bbox[2]+add_pix, bbox[3]+add_pix]
    # check size whether cross the border
    bbox_new = [max(0, bbox_new[0]), max(0, bbox_new[1]), min(w-1, bbox_new[2]), min(h-1, bbox_new[3])]
    return bbox_new

#--------------------------------------------------------------------------------
#-- function: jitter bbox
def jitterBox(bbox, bboxSize, jit_scale = 0.05):
    """
    bbox: (x1, y1, x2, y2)
    jitter random 5%
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    wj = int(w * jit_scale)
    hj = int(h * jit_scale)
    return [max(0, x1+int(random.uniform(-wj, wj))),
            max(0, y1+int(random.uniform(-hj, hj))),
            min(bboxSize[1]-1, x2+int(random.uniform(-wj, wj))),
            min(bboxSize[0]-1, y2+int(random.uniform(-hj, hj)))]

def get_distance_map(img, bbox):
    """
    img: croped image, PIL image  (h,w,3)
    bbox: (x1,y1,x2,y2) bbox from annotation

    return:
        distance_map: numpy array of (h,w) with distance map in 
            ECCV paper "Deep GrabCut for Object Selection"
    """
    img_arr = np.asarray(img)
    distance_map = np.zeros(img_arr.shape[:2])
    for i in range(img_arr.shape[0]):
        for j in range(img_arr.shape[1]):
            if i<=bbox[3] and i>=bbox[1] and j<=bbox[2] and i>=bbox[0]:
                distance_map[i][j] = 128 + min( abs(i-bbox[3]), abs(i-bbox[1]), abs(j-bbox[0]), abs(j-bbox[2]) )
            else:
                distance_map[i][j] = 128 - min( abs(i-bbox[3]), abs(i-bbox[1]), abs(j-bbox[0]), abs(j-bbox[2]) )
    return distance_map