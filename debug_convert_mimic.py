import os
import json
import argparse
from random import shuffle, seed
import string
# non-standard dependencies:
import h5py
from six.moves import cPickle
import numpy as np
import torch
import torchvision.models as models
import skimage.io
import matplotlib.pyplot as plt
from torchvision import transforms as trn
preprocess = trn.Compose([
                #trn.ToTensor(),
                trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
from captioning.utils.resnet_utils import myResnet
import captioning.utils.resnet as resnet
from PIL import Image

params = {}
params['input_json'] = 'data/dataset_retina_resize.json'
params['output_json'] = 'data/retina_resize'

params['images_root'] = ''
params['att_size'] = 14
params['model'] = 'resnet101'
params['model_root'] = 'data/imagenet_weights'
newsize = (256, 256)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = getattr(resnet, params['model'])()
net.load_state_dict(torch.load(os.path.join(params['model_root'],params['model']+'.pth')))
my_resnet = myResnet(net)
#my_resnet.cuda()
my_resnet.to(device)
my_resnet.eval()

file_path = '/media/hdd/data/imcaption/retina_dataset_resize/resize/1_2_826_0_1_3680043_9_5115_636252259520332334/1_3_6_1_4_1_33437_10_4_11578754_13134123662_18471_4_1_0_0.png'
# I = skimage.io.imread(file_path)

# Resample image 
original_image_path = '/media/hdd/data/imcaption/retina_dataset_resize/out/1_2_826_0_1_3680043_9_5115_636252259520332334/1_3_6_1_4_1_33437_10_4_11578754_13134123662_18471_4_1_0_0.png'
image = Image.open(original_image_path)
# print('image size before resizing', image.size)
reim = image.resize(newsize)
# im.save(imgPath)
# print('image size after resizing', reim.size)
imsavepath = '/media/hdd/data/imcaption/retina_dataset_resize/resize/1_2_826_0_1_3680043_9_5115_636252259520332334/1_3_6_1_4_1_33437_10_4_11578754_13134123662_18471_4_1_0_0.png'
# I = skimage.io.imread(file_path)
# print('the current image being saved is', imsavepath)
reim.save(imsavepath)
