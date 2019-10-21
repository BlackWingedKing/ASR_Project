'''
This code contains the neural network model used for our project
The network contains the image extractor and audio extractor which extracts 
the features from the images and audio respectively
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
import cv2
# data processing 
import pandas as pd
import glob
import time

# parameters and hyper params
batch_size = 10

#gpu settings
use_cuda = torch.cuda.is_available()
print('gpu status ===',use_cuda)
torch.manual_seed(1)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

class AVNet(nn.Module):
    def __init__(self):
        super(AVNet, self).__init__()
        # AV network
        # x and y inputs corresponding to image frames and audio frames
        # The paper assumes random cropped images of 256*256 resized to 224*224
        # define the layers as described in the paper
        # layers for image
        self.im_conv1 = nn.Conv3d(in_channels=3,out_channels=64,kernel_size=[5,7,7],stride=[2,2,2])
        self.im_conv2 = nn.Conv3d(in_channels=64,out_channels=64,kernel_size=[3,3,3],stride=[2,2,2])
        self.im_pool1 = nn.Maxpool3d(kernel_size=[1,2,2])

        # audio layers
        self.a_conv1 = nn.Conv3d(in_channels=cainput,out_channels=64,kernel_size=[65,1,1],stride=4)
        self.a_conv2 = nn.Conv3d(in_channels=64,out_channels=128,kernel_size=[15,1,1],stride=4)
        self.a_conv3 = nn.Conv3d(in_channels=128,out_channels=128,kernel_size=[15,1,1],stride=4)
        self.a_conv4 = nn.Conv3d(in_channels=128,out_channels=256,kernel_size=[15,1,1],stride=4)
        self.a_conv5 = nn.Conv3d(in_channels=256,out_channels=256,kernel_size=[15,1,1],stride=4)
        self.a_conv6 = nn.Conv3d(in_channels=256,out_channels=128,kernel_size=[3,1,1])

        self.a_pool1 = nn.Maxpool1d(kernel_size=4)
        self.a_pool2 = nn.Maxpool1d(kernel_size=3)

        # fusion layers
        self.f_conv1 = nn.Conv3d(in_channels=,out_channels=512,kernel_size=[1,1,1])
        self.f_conv2 = nn.Conv3d(in_channels=512,out_channels=128,kernel_size=[1,1,1])
        self.f_conv3 = nn.Conv3d(in_channels=128,out_channels=128,kernel_size=[3,3,3],stride=[2,2,2])
        self.f_conv4 = nn.Conv3d(in_channels=128,out_channels=256,kernel_size=[3,3,3],stride=[1,2,2])
        self.f_conv5 = nn.Conv3d(in_channels=256,out_channels=256,kernel_size=[3,3,3],stride=[1,2,2])
        self.f_conv6 = nn.Conv3d(in_channels=256,out_channels=512,kernel_size=[3,3,3],stride=[1,2,2])
        self.f_conv7 = nn.Conv3d(in_channels=512,out_channels=512,kernel_size=[3,3,3],stride=[1,2,2])
        self.f_fcn = nn.Linear(sasa,1)
        
        # activations and fc layers
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLu()
        self.logsigmoid = nn.LogSigmoid()

    def forward(self, x, y):
        # x is the rgb image and y is the audio 
        # x shape is assumed to be Bx3x125x224x224 and y's shape is assumed to be ..
        # first the image part
        x = self.im_conv1(x)
        x = self.im_pool1(x)
        for i in range(0,4):
            x = self.im_conv2(x)

        # now extract from the audio
        y = self.a_conv1(y)
        y = self.a_pool1(y)
        y = self.a_conv2(y)
        for i in range(0,4):
            y = self.a_conv3(y)
        y = self.a_conv4(y)
        y = self.a_conv5(y)
        y = self.a_pool2(y)
        y = self.a_conv6(y)

        # now combine both of these variables
        # fu = # tile command and conacatenate these
        fu = self.f_conv1(fu)
        fu = self.f_conv2(fu)
        for i in range(0,4):
            fu = self.f_conv3(fu)
        fu = self.f_conv4(fu)
        for i in range(0,3):
            fu = self.f_conv5(fu)
        fu = self.f_conv6(fu)
        for i in range(0,3):
            fu = self.f_conv7(fu)
        fu = fu.view(batch_size,-1)
        fu = self.f_fcn(fu)
        fu = self.sigmoid(fu)
        return fu