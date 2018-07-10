import socket
import timeit
from datetime import datetime
import os
import glob
from collections import OrderedDict
from PIL import Image
import numpy as np

# PyTorch includes
import torch
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms
from torch.nn.functional import upsample

# Custom includes
from networks import deeplab_xception
from dataloaders import custom_transforms as tr

image_path='imgs/test.jpg'
gpu_id = -1
print('Using GPU: {} '.format(gpu_id))
# Setting parameters
resume_epoch=300
save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]

# get run id and dir
runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
run_id = int(runs[-1].split('_')[-1]) if runs else 0

save_dir = os.path.join(save_dir_root, 'run', 'run_' + str(run_id))

# Network definition
net = deeplab_xception.DeepLabv3_plus(nInputChannels=3, n_classes=21, pretrained=True)
modelName = 'deeplabv3+'

print("Initializing weights from: {}...".format(
    os.path.join(save_dir, 'models', modelName + '_epoch-' + str(resume_epoch - 1) + '.pth')))
net.load_state_dict(
    torch.load(os.path.join(save_dir, 'models', modelName + '_epoch-' + str(resume_epoch - 1) + '.pth'),
                map_location=lambda storage, loc: storage)) # Load all tensors onto the CPU

if gpu_id >= 0:
    torch.cuda.set_device(device=gpu_id)
    net.cuda()

composed_transforms_ts = transforms.Compose([
    tr.FixedResize(size=513),
    tr.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(1.0, 1.0, 1.0)),
    tr.ToTensor()])

net.eval()
# chaneg to a single image
inputs = composed_transforms_ts({'image': np.array(Image.open(image_path).convert('RGB')).astype(np.float32)})
# Forward pass of the mini-batch
inputs = Variable(inputs['image'].unsqueeze_(0), requires_grad=True)
if gpu_id >= 0:
    inputs = inputs.cuda()
with torch.no_grad():
    output = net.forward(inputs)
output = upsample(output, size=(513, 513), mode='bilinear', align_corners=True)
# save image
print(output.shape)
result = transforms.ToPILImage()(output[0][0].unsqueeze(0))
result.save("imgs/result.jpg", "JPEG")
