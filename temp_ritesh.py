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
# import cv2
# data processing 
import pandas as pd
import glob
import time

class residual_block(nn.Module):
    def __init__(self,in_feats,out_feats,kernel,padding,stride=1):
        super(residual_block,self).__init__()
        self.conv1 = nn.Conv3d(in_channels=in_feats,out_channels=out_feats,kernel_size=kernel,stride=stride,padding=padding)
        self.conv2 = nn.Conv3d(in_channels=out_feats,out_channels=out_feats,kernel_size=kernel,padding=padding)
        self.bn = nn.BatchNorm3d(out_feats)
        self.relu = nn.ReLU(inplace=True)
        self.downsample_conv = 0
        if(in_feats==out_feats and stride!=[1,1,1]):
            self.downsample_conv = 1
            self.downsample_layer =nn.MaxPool3d(kernel_size=[1,1,1],stride=stride)
        if(in_feats!=out_feats and stride!=[1,1,1]):
            self.downsample_conv = 2
            self.downsample_layer = nn.Conv3d(in_channels=in_feats,out_channels=out_feats,kernel_size=[1,1,1],stride=stride)
    def forward(self,x):
        identity = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        # if(self.downsample_conv==1):
        identity = self.downsample_layer(identity)
        out = out + identity
        out = self.bn(out)
        out = self.relu(out)
        return out 

class VideoNet(nn.Module):
    def __init__(self):
        super(VideoNet, self).__init__()
        # Video feature extraction network
        # x and y inputs corresponding to image frames and audio frames
        # The paper assumes random cropped images of 256*256 resized to 224*224
        # define the layers as described in the paper
        # layers for image
        self.im_conv1 = nn.Conv3d(in_channels=3,out_channels=64,kernel_size=[5,7,7],stride=[2,2,2],padding=(2,3,3))
        self.im_pool1 = nn.MaxPool3d(kernel_size=[1,3,3],stride=[1,2,2],padding=(0,1,1))
        self.im_res1 = residual_block(64,64,[3,3,3],[1,1,1],1)
        self.im_res2 = residual_block(64,64,[3,3,3],[1,1,1],[2,2,2])

    def forward(self, x):
        # x is the rgb image and y is the audio 
        # x shape is assumed to be Bx3x125x224x224 
        # and y's shape is assumed to be B*2*n_samples*1*1
        # first the image part
        x = F.relu(self.im_conv1(x))
        print(x.shape)
        x = self.im_pool1(x)
        print(x.shape)
        x = self.im_res1(x)
        print(x.shape)
        x= self.im_res2(x)
        print(x.shape)
        return x

class AudioNet(nn.Module):
    def __init__(self):
        super(AudioNet, self).__init__()
        # audio layers
        # input of form [Batch,channels,time,height width]
        self.a_conv1 = nn.Conv3d(in_channels=2,out_channels=64,kernel_size=[65,1,1],stride=4,padding=(32,1,1))
        self.a_pool1 = nn.MaxPool3d(kernel_size=[4,1,1],stride=[4,1,1],padding=(1,0,0))
        self.a_res1 = residual_block(64,128,[15,1,1],(7,0,0),[4,1,1])
        self.a_res2 = residual_block(128,128,[15,1,1],(7,0,0),[4,1,1])
        self.a_res3 = residual_block(128,256,[15,1,1],(7,0,0),[4,1,1])
        self.a_pool2 = nn.FractionalMaxPool3d(kernel_size=[3,1,1],output_size=(32,1,1))
        self.a_conv2 = nn.Conv3d(in_channels=256,out_channels=128,kernel_size=[3,1,1],padding=(1,0,0))
    
    def forward(self, y):
        # now extract from the audio
        y = F.relu(self.a_conv1(y))
        print(y.shape)
        y = self.a_pool1(y)
        print(y.shape)
        y = self.a_res1(y)
        print(y.shape)
        y = self.a_res2(y)
        print(y.shape)
        y = self.a_res3(y)
        print(y.shape)
        y = self.a_pool2(y.repeat([1,1,1,2,2]))
        print(y.shape)
        y = F.relu(self.a_conv2(y))
        print(y.shape)
        return y

# code for testing
Vmodel = VideoNet()
Amodel = AudioNet()
print(Vmodel.state_dict())