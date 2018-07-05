from __future__ import division

import math
import os
from copy import deepcopy
import scipy.io
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.modules as modules
from torch.autograd import Variable
from torch.nn import functional as F

def logit(x):
    return np.log(x/(1-x+1e-08)+1e-08)


def sigmoid_np(x):
    return 1/(1+np.exp(-x))


def class_balanced_cross_entropy_loss(output, label, size_average=True, batch_average=True):
    """Define the class balanced cross entropy loss to train the network
    Args:
    output: Output of the network
    label: Ground truth label
    Returns:
    Tensor that evaluates the loss
    """

    labels = torch.ge(label, 0.5).float()

    num_labels_pos = torch.sum(labels)
    num_labels_neg = torch.sum(1.0 - labels)
    num_total = num_labels_pos + num_labels_neg

    output_gt_zero = torch.ge(output, 0).float()
    loss_val = torch.mul(output, (labels - output_gt_zero)) - torch.log(
        1 + torch.exp(output - 2 * torch.mul(output, output_gt_zero)))

    loss_pos = torch.sum(-torch.mul(labels, loss_val))
    loss_neg = torch.sum(-torch.mul(1.0 - labels, loss_val))

    final_loss = num_labels_neg / num_total * loss_pos + num_labels_pos / num_total * loss_neg

    if size_average:
        final_loss /= np.prod(label.size())
    elif batch_average:
        final_loss /= label.size()[0]

    return final_loss


def center_crop(x, height, width):
    crop_h = torch.FloatTensor([x.size()[2]]).sub(height).div(-2)
    crop_w = torch.FloatTensor([x.size()[3]]).sub(width).div(-2)

    return F.pad(x, [
        crop_w.ceil().int()[0], crop_w.floor().int()[0],
        crop_h.ceil().int()[0], crop_h.floor().int()[0],
    ])


def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)


# set parameters s.t. deconvolutional layers compute bilinear interpolation
# this is for deconvolution without groups
def interp_surgery(lay):
        m, k, h, w = lay.weight.data.size()
        if m != k:
            print('input + output channels need to be the same')
            raise ValueError
        if h != w:
            print('filters need to be square')
            raise ValueError
        filt = upsample_filt(h)

        for i in range(m):
            lay.weight[i, i, :, :].data.copy_(torch.from_numpy(filt))

        return lay.weight.data

class OSVOS(nn.Module):
    def __init__(self, pretrained=1):
        super(OSVOS, self).__init__()
        lay_list = [[64, 64],
                    ['M', 128, 128],
                    ['M', 256, 256, 256],
                    ['M', 512, 512, 512],
                    ['M', 512, 512, 512]]
        in_channels = [3, 64, 128, 256, 512]

        print("Constructing OSVOS architecture..")
        stages = modules.ModuleList()
        side_prep = modules.ModuleList()
        score_dsn = modules.ModuleList()
        upscale = modules.ModuleList()
        upscale_ = modules.ModuleList()

        # Construct the network
        for i in range(0, len(lay_list)):
            # Make the layers of the stages
            stages.append(make_layers_osvos(lay_list[i], in_channels[i]))

            # Attention, side_prep and score_dsn start from layer 2
            if i > 0:
                # Make the layers of the preparation step
                side_prep.append(nn.Conv2d(lay_list[i][-1], 16, kernel_size=3, padding=1))

                # Make the layers of the score_dsn step
                score_dsn.append(nn.Conv2d(16, 1, kernel_size=1, padding=0))
                upscale_.append(nn.ConvTranspose2d(1, 1, kernel_size=2 ** (1 + i), stride=2 ** i, bias=False))
                upscale.append(nn.ConvTranspose2d(16, 16, kernel_size=2 ** (1 + i), stride=2 ** i, bias=False))

        self.upscale = upscale
        self.upscale_ = upscale_
        self.stages = stages
        self.side_prep = side_prep
        self.score_dsn = score_dsn

        self.fuse = nn.Conv2d(64, 1, kernel_size=1, padding=0)

        print("Initializing weights..")
        self._initialize_weights(pretrained)

    def forward(self, x):
        crop_h, crop_w = int(x.size()[-2]), int(x.size()[-1])
        x = self.stages[0](x)

        side = []
        side_out = []
        for i in range(1, len(self.stages)):
            x = self.stages[i](x)
            side_temp = self.side_prep[i - 1](x)
            side.append(center_crop(self.upscale[i - 1](side_temp), crop_h, crop_w))
            side_out.append(center_crop(self.upscale_[i - 1](self.score_dsn[i - 1](side_temp)), crop_h, crop_w))

        out = torch.cat(side[:], dim=1)
        out = self.fuse(out)
        side_out.append(out)
        return side_out

    def _initialize_weights(self, pretrained):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.001)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.zero_()
                m.weight.data = interp_surgery(m)

        if pretrained == 1:
            print("Loading weights from PyTorch VGG")
            vgg_structure = [64, 64, 'M', 128, 128, 'M', 256, 256, 256,
                             'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
            _vgg = VGG(make_layers(vgg_structure))

            # Load the weights from saved model
            _vgg.load_state_dict(torch.load(os.path.join('./pre_trained_models', 'vgg_pytorch.pth'),
                                            map_location=lambda storage, loc: storage))

            inds = find_conv_layers(_vgg)
            k = 0
            for i in range(len(self.stages)):
                for j in range(len(self.stages[i])):
                    if isinstance(self.stages[i][j], nn.Conv2d):
                        self.stages[i][j].weight = deepcopy(_vgg.features[inds[k]].weight)
                        self.stages[i][j].bias = deepcopy(_vgg.features[inds[k]].bias)
                        k += 1
        elif pretrained == 2:
            print("Loading weights from Caffe VGG")
            # Load weights from Caffe
            caffe_weights = scipy.io.loadmat(os.path.join('./pre_trained_models', 'vgg_caffe.mat'))
            # Core network
            caffe_ind = 0
            for ind, layer in enumerate(self.stages.parameters()):
                if ind % 2 == 0:
                    c_w = torch.from_numpy(caffe_weights['weights'][0][caffe_ind].transpose())
                    assert (layer.data.shape == c_w.shape)
                    layer.data = c_w
                else:
                    c_b = torch.from_numpy(caffe_weights['biases'][0][caffe_ind][:, 0])
                    assert (layer.data.shape == c_b.shape)
                    layer.data = c_b
                    caffe_ind += 1


def find_conv_layers(_vgg):
    inds = []
    for i in range(len(_vgg.features)):
        if isinstance(_vgg.features[i], nn.Conv2d):
            inds.append(i)
    return inds


def make_layers_osvos(cfg, in_channels):
    layers = []
    for v in cfg:
        if v == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers.extend([conv2d, nn.ReLU(inplace=True)])
            in_channels = v
    return nn.Sequential(*layers)


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
