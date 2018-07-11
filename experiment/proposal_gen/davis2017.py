"""
Mask R-CNN
Configurations and data loading code for Davis 2017.

Licensed under the MIT License (see LICENSE for details)
Written by Shuangjie Xu

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained DAVIS weights
    python davis2017.py train --dataset=/path/to/davis/ --model=davis

    # Train a new model starting from ImageNet weights
    python davis2017.py train --dataset=/path/to/davis/ --model=imagenet

    # Continue training a model that you had trained earlier
    python davis2017.py train --dataset=/path/to/davis/ --model=/path/to/weights.h5

    # Continue training the last model you trained
    python davis2017.py train --dataset=/path/to/davis/ --model=last

    # Run DAVIS evaluatoin on the last model you trained
    python davis2017.py evaluate --dataset=/path/to/davis/ --model=last
"""

import os
import time
import numpy as np

import zipfile
# import urllib.request
from urllib2 import urlopen
import shutil
import random

from .config import Config
from . import utils
from . import model as modellib

import torch
import skimage

############################################################
#  Argument
############################################################

# Root directory of the project
ROOT_DIR = os.getcwd()

# Path to trained weights file
DAVIS_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_davis.pth")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.pth")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "out", "logs")
DEFAULT_DATASET_YEAR = "2017"


import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(
    description='Train Mask R-CNN on MS COCO.')
parser.add_argument("command",
                    metavar="<command>",
                    help="'train' or 'evaluate' on MS COCO")
parser.add_argument('--dataset', required=False,
                    default='/data1/shuangjiexu/data/DAVIS_2017',
                    metavar="/path/to/davis/",
                    help='Directory of the DAVIS dataset')
parser.add_argument('--year', required=False,
                    default=DEFAULT_DATASET_YEAR,
                    metavar="<year>",
                    help='Year of the MS-COCO dataset (2014 or 2017) (default=2014)')
parser.add_argument('--model', required=False,
                    default="coco",
                    metavar="/path/to/weights/",
                    help="Path to weights .pth file or 'coco'")
parser.add_argument('--logs', required=False,
                    default=DEFAULT_LOGS_DIR,
                    metavar="/path/to/logs/",
                    help='Logs and checkpoints directory (default=logs/)')
parser.add_argument('--limit', required=False,
                    default=100,
                    metavar="<image count>",
                    help='Images to use for evaluation (default=500)')
parser.add_argument('--seq', required=False,
                    default='bike-packing',
                    metavar="<sequence class name>",
                    help='Sequence class name in DAVIS (default=bike-packing)')
parser.add_argument('--augment_method', required=False,
                    default='xu_val_augment_2500',
                    metavar="<augment methods name>",
                    help='Augment methods name in Matlab code (default=xu_val_augment_2500)')
args = parser.parse_args()
print("Command: ", args.command)
print("Model: ", args.model)
print("Dataset: ", args.dataset)
print("Year: ", args.year)
print("Logs: ", args.logs)
print("Class: ", args.seq)
print("Limit: ", args.limit)
print("Augment method: ", args.augment_method)


############################################################
#  Dataset
############################################################

# NEED DAVIS TOOL in https://github.com/fperazzi/davis-2017
from davis import cfg, phase, io, DAVISLoader, Annotation

an = Annotation(args.seq, single_object=0)
OBJ_NUMBER = an.n_objects

# Load dataset
db = DAVISLoader(year=args.year, phase=phase.TESTDEV)

AugmentImgPath = os.path.join(args.dataset, 'Augmentations', args.augment_method, 'JPEGImages', '480p', args.seq)
AugmentAnnoPath = os.path.join(args.dataset, 'Augmentations', args.augment_method, 'Annotations', '480p', args.seq)
# read all the file list
file_names = next(os.walk(AugmentImgPath))[2]
random.shuffle (file_names)
test_files = [x[:-4] for x in file_names[:args.limit]]
train_files =  [x[:-4] for x in file_names[args.limit:]]

class DavisDataset(utils.Dataset):
    def load_davis(self, subset):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, test)
        year: What dataset year to load (2014, 2017) as a string, not an integer
        """
        # Add classes
        for i in range(1, OBJ_NUMBER):
            self.add_class("davis", i, 'obj_'+str(i))

        if subset == 'train':
            image_list = train_files
        else:
            image_list = test_files
        # Add images
        # annotations is a mask of w*h with value [0,1,2,...]
        for i in range(len(image_list)):
            self.add_image(
                "davis", image_id=i,
                path=os.path.join(AugmentImgPath, image_list[i]+'.jpg'),
                annotations=skimage.io.imread(os.path.join(AugmentAnnoPath, image_list[i]+'.png')))

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "davis":
            return super(DavisDataset, self).load_mask(image_id)
        img = skimage.io.imread(image_info["path"])
        h,w = img.shape[:2]

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for class_id in np.unique(annotations):
            if class_id == 0:
                continue
            mask = annotations.copy()
            mask[mask!=class_id] = 0
            mask[mask==class_id] = 1

            # and end up rounded out. Skip those objects.
            if mask.max() < 1:
                continue

            instance_masks.append(mask)
            class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(DavisDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        """Return a path to the image in the DAVIS 2017 Augmentations."""
        info = self.image_info[image_id]
        if info["source"] == "coco":
            return info["path"]
        else:
            super(DavisDataset, self).image_reference(image_id)



############################################################
#  Configurations
############################################################

class DavisConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "davis"

    # We use one GPU with 8GB memory, which can fit one image.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Uncomment to train on 8 GPUs (default is 1)
    GPU_COUNT = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + OBJ_NUMBER  # COCO has 80 classes

    STEPS_PER_EPOCH = 100


############################################################
#  DAVIS Evaluation
############################################################


############################################################
#  Training
############################################################

if __name__ == '__main__':
    # Configurations
    if args.command == "train":
        config = DavisConfig()
    else:
        class InferenceConfig(DavisConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(config=config,
                                  model_dir=args.logs)
    if config.GPU_COUNT:
        model = model.cuda()

    # TODO: add model load methods for the different class
    # Select weights file to load
    if args.model:
        if args.model.lower() == "coco":
            model_path = COCO_MODEL_PATH
        elif args.model.lower() == "last":
            # Find last trained weights
            model_path = model.find_last()[1]
        elif args.model.lower() == "imagenet":
            # Start from ImageNet trained weights
            model_path = config.IMAGENET_MODEL_PATH
        else:
            model_path = args.model
    else:
        model_path = ""

    # Load weights
    print("Loading weights ", model_path)
    model.load_weights(model_path, utils.state_modifier)

    # Train or evaluate
    if args.command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = DavisDataset()
        dataset_train.load_davis("train")
        dataset_train.prepare()

        # Validation dataset
        dataset_val = DavisDataset()
        dataset_val.load_davis("val")
        dataset_val.prepare()

        # TODO: change epoches because we fine tune it
        # lr and epoch number decrese 10
        config.LEARNING_RATE = config.LEARNING_RATE / 10
        # Training - Stage 1
        print("Training network heads")
        model.train_model(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=20,
                    layers='heads')

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train_model(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=60,
                    layers='4+')

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train_model(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=80,
                    layers='all')

    elif args.command == "evaluate":
        # Validation dataset
        dataset_val = DavisDataset()
        # TODO: how to load the test dev image
        davis = dataset_val.load_davis("minival")
        dataset_val.prepare()
        print("Running COCO evaluation on {} images.".format(args.limit))
        print('Undo')
        # evaluate_davis(model, dataset_val, davis, "bbox", limit=int(args.limit))
        # evaluate_davis(model, dataset_val, davis, "segm", limit=int(args.limit))
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))