import math
import numpy as np
import os

class Config(object):
    """Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    """
    # experiment name
    NAME = 'Davis_2017'

    USE_CUDA = True
    GPU_IDS = [0,1,2,3,4,5]
    # mask tracker path (trained caffe model of deeplabv2 for lucid davis)
    TARGET_DATASET = 'test-dev'
    
    TRACKER_MODEL_PROTOTXT = 'prototxt/test_online.prototxt'

    DATA_ROOT = '/fid/shuangjiexu/data_buffer/DAVIS_2017'
    # flow data from opticalflow_flownet2
    FLOW_ROOT = '/app/home/DAVIS_2017/Results/Segmentations/480p/opticalflow_flownet2/'
    
    # results of osvos or net
    OSVOS_RESULTS_1 = '/app/home/baofusev3_2kiter' # '/app/home/osvos_results/DAVIS_2017/Results/'
    OSVOS_RESULTS_2 = '/app/home/osvos_results/DAVIS_2017/Results/'
    # save path
    SAVE_PATH = 'out'
    # final result path
    PALETTE = np.loadtxt('config/palette.txt',dtype=np.uint8).reshape(-1,3)
    NET_PATH = None
    # SETTINGS
    PREDICT_BBOX_MARGIN = 0.15
    PIXEL_DECAY = 0.9
    BBOX_DELAY = 0.8
    # re-id setting
    MIN_PIXEL_NUMBER = 20*20
    MIN_PIXEL_NUMBER_LARGE = 50*50
    # to detect occlusion, min ratio of mask pix over bbox area, excluding border
    MIN_SEG_OVER_BBOX = 0.2 * (1/(1+PREDICT_BBOX_MARGIN)**2)
    # to detect occlusion, min ratio of mask pix over init mask pix
    MIN_SEG_OVER_INIT = 0.2
    # to detect occlusion, min ratio of mask pix over smoothed history numpix
    MIN_SEG_OVER_HISTRY = 0.5
    # to detect occlusion, min iou of warped mask and cnn response
    MIN_PRIOR_CNN_IOU = 0.2
    # 0.6 to detect occlusion, the forward/backward mask overlap
    MIN_FORD_BACK_OVERLAP = 0.4
    # threshold value for mask
    VALID_TH = 0.3
    # used when compute overlap between two mask
    OVERLAP_TH = 0.5
    DILATION_COEFFICIENT = 1.05
    # when objsz changed too radically stop updating 
    OBJSIZE_UPDATE_LOWTH = 0.6
    # when objsz changed too radically stop updating 
    OBJSIZE_UPDATE_HIGHTH = 1.4
    

    def __init__(self):
        """Set values of computed attributes."""
        self.RESULT_PATH = os.path.join('out/', self.NAME)
        self.IMAGE_ROOT = os.path.join(self.DATA_ROOT, 'JPEGImages/480p/')
        self.MASK_ROOT = os.path.join(self.DATA_ROOT, 'Annotations/480p/')

        if self.TARGET_DATASET == 'test-dev':
            self.TRACKER_MODEL_ROOT = '/app/home/maskrefine_models/davis2017testdev'
        elif self.TARGET_DATASET == 'challenge':
            self.TRACKER_MODEL_ROOT = '/app/home/maskrefine_models/davis2017challenge'
        else:
            self.TRACKER_MODEL_ROOT = '/app/home/maskrefine_models/davis2017val'
        self.SEQUENCE_FILE_NAME = self.TARGET_DATASET + '.txt'
        # test seqence file path
        self.TEST_SEQUENCE_FILE = os.path.join(self.DATA_ROOT, 'ImageSets/2017', self.SEQUENCE_FILE_NAME)
        self.SEQUENCES = self.read_seq_name(self.TEST_SEQUENCE_FILE)


    def read_seq_name(self, path):
        with open(path, 'r') as f:
            contents = f.readlines()
        return [content.strip() for content in contents]


    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
