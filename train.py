'''
Run this code for training
'''
# imports
from __future__ import print_function
# torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
# np, images
import numpy as np
# import cv2
# data processing 
import pandas as pd
import glob
import time
from model_fused import AVNet, VideoNet, AudioNet

# parameters and hyper params
batch_size = 1
nepochs = 5
LR = 0.001

#gpu settings
use_cuda = torch.cuda.is_available()
print('gpu status ===',use_cuda)
torch.manual_seed(1)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

def main():
    # initialise the models
    vmodel = VideoNet().to(device)
    amodel = AudioNet().to(device)
    avmodel = AVNet().to(device)
    params = list(vmodel.parameters())+list(amodel.parameters())+list(avmodel.parameters())
    optimiser = optim.Adam(params,lr=LR)
    train(vmodel, amodel, avmodel, optimiser)

if __name__ == '__main__':
    main()    