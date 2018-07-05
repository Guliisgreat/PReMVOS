import numpy as np
import sys,os
import cv2

import torch

from utils.davis import *
import vgg_osvos as vo

class OSVOS():
    """
    osvos v2 class for mask tracker
    """

    def __init__(self, cfgs, seqname, obj_id):
        self.seqname = seqname
        self.obj_id = obj_id
        self.cfgs = cfgs
        self.results = None
        self.net = None
        if cfgs.OSVOS_RESULTS_1 is not None and cfgs.OSVOS_RESULTS_2 is not None:
            img_path = os.path.join(cfgs.IMAGE_ROOT, self.seqname)
            names_img = np.sort([file_[:-4] for file_ in os.listdir(img_path) if not os.path.isdir(os.path.join(img_path,file_))])
            # 1
            result_path_1 = os.path.join(cfgs.OSVOS_RESULTS_1, self.seqname, str(self.obj_id))
            img_list_1 = list(map(lambda x: os.path.join(result_path_1, x+'_1osvos.png'), names_img))
            # 2
            result_path_2 = os.path.join(cfgs.OSVOS_RESULTS_2, self.seqname, str(self.obj_id))
            img_list_2 = list(map(lambda x: os.path.join(result_path_2, x+'.png'), names_img))
            # add
            self.results = [img_list_1, img_list_2]
        elif cfgs.NET_PATH is not None:
            # Network definition
            net = vo.OSVOS(pretrained=0)
            net = torch.nn.DataParallel(net)
            net.load_state_dict(torch.load(os.path.join(cfgs.NET_PATH, self.seqname, str(self.obj_id), 
                                    self.seqname+'_epoch-9999.pth'), map_location=lambda storage, loc: storage))
            self.net = net
        else:
            raise Exception,"Invalid OSVOS!"

    def get_segmentation(self, img_id):
        res_num = len(self.results)
        masks = []
        for i in range(res_num):
            if self.results[i][img_id] is not None:
                mask = cv2.imread(self.results[i][img_id])
                mask = np.squeeze(mask)
                if len(mask.shape) == 3:
                    mask = mask[...,0]
                masks.append(mask/255.0)
            else:
                raise Exception,"No OSVOS Results OR NET!"
        gt_mask = np.zeros(masks[0].shape).astype(np.float)
        for i in len(masks):
            gt_mask = gt_mask + masks[i]
        gt_mask = gt_mask / res_num
        gt_mask[gt_mask<0.4] = 0
        return valid_mask(self.cfgs, gt_mask), gt_mask

            
        
