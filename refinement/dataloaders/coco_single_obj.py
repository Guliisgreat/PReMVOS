from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import numpy as np
import torch
import codecs
import json
import random

from .utils import add_size, jitterBox, get_distance_map
import torchvision.transforms.functional as F

from pycocotools import mask as maskUtils


class COCO(data.Dataset):
    """`COCO Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        year: int number for coco dataset in which year
    """

    def __init__(self, root, train=True, transform=None, target_transform=None, year=2014):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.year = year

        if self.train:
            self.img_root = os.path.join(self.root, 'train%04d' % year)
            self.json_root = os.path.join(self.root, 'annotations', 'instances_train%04d.json' % year)
        else:
            self.img_root = os.path.join(self.root, 'val%04d' % year)
            self.json_root = os.path.join(self.root, 'annotations', 'instances_val%04d.json' % year)

        if not self._check_exists():
            raise RuntimeError('Dataset not found.')

        with open(self.json_root, 'r') as f:
            self.anno = json.load(f)
        self.info = self.anno['annotations']
        self.train_labels = torch.FloatTensor([self.info[index]['category_id'] for index in range(len(self.info))])
        self.test_labels = torch.FloatTensor([self.info[index]['category_id'] for index in range(len(self.info))])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # COCO_train2014_000000233296.jpg
        if self.train:
            img_path = os.path.join(self.img_root, 'COCO_train%04d_%012d.jpg' % (self.year, self.info[index]['image_id']))
        else:
            img_path = os.path.join(self.img_root, 'COCO_val%04d_%012d.jpg' % (self.year, self.info[index]['image_id']))
        target = self.test_labels[index]
        bbox = self.info[index]['bbox']  # [x1,y1,width,height]

        # solve 0 height sample such as 390267 with skis
        if bbox[2]<2:
            bbox[2] += 2
        if bbox[3]<2:
            bbox[3] += 2

        # solve some gray image
        img = Image.open(img_path)
        img = img.convert('RGB')
        img = np.array(img)
        img = Image.im = Image.fromarray(img[...,:3])
        img_size = img.size
        # extend bbox with 50 pixel and jitter
        bbox = [bbox[0], bbox[1], bbox[0]+bbox[2]-1, bbox[1]+bbox[3]-1] # (x1, y1, x2, y2)
        bbox_jitter = jitterBox(bbox, img_size)
        bbox_crop = add_size(bbox_jitter, img_size)
        # img
        img = img.crop( bbox_crop )

        # random gamma adjust
        rand = random.randint(1,2) 
        if rand == 2:
            gamma = random.uniform(0.2, 3)
            img = F.adjust_gamma(img, gamma)
        # img = img.resize((28,28))

        # solve mask
        m = self.annToMask(self.info[index], img_size[0], img_size[1])
        mask = m[bbox_crop[1]:bbox_crop[3]+1, bbox_crop[0]:bbox_crop[2]+1]

        # solve density map
        distance_map = get_distance_map(img, bbox)

        # TODO: add density map to the fourth channel in img

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m

    def __len__(self):
        return len(self.info)

    def _check_exists(self):
        return os.path.exists(self.img_root) and os.path.exists(self.json_root)

    def _categories_num(self):
        return len(self.anno['categories'])

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
