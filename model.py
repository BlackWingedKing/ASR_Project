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

class residual_block(nn.Module):
    def __init__(self,in_feats,out_feats,kernel,stride=1):
        super(residual_block,self).__init__()
        self.conv1 = nn.Conv3d(in_channels=in_feats,out_channels=out_feats,kernel_size=kernel,stride=stride)
        self.conv2 = nn.Conv3d(in_channels=out_feats,out_channels=out_feats,kernel=kernel)
        self.bn = nn.BatchNorm3d(out_feats)
        self.relu = nn.ReLu(inplace=True)
        self.downsample_conv = 0
        if(in_feats==out_feats and stride!=1):
            self.downsample_conv = 1
            self.downsample_layer =nn.Maxpool3d(kernel_size=[1,1,1],stride=stride)
        if(in_feats!=out_feats and stride!=1):
            self.downsample_conv = 2
            self.downsample_layer = nn.Conv3d(in_channels=in_feats,out_channels=out_feats,kernel_size=[1,1,1],stride=stride)
    def forward(self,x):
        identity = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        if(downsample_conv==1):
            identity = self.downsample_layer(identity)
        out = out + identity
        out = self.bn(out)
        out = self.relu(out)
        return out 


class AVNet(nn.Module):
    

    def __init__(self):
        super(AVNet, self).__init__()
        # AV network
        # x and y inputs corresponding to image frames and audio frames
        # The paper assumes random cropped images of 256*256 resized to 224*224
        # define the layers as described in the paper
        # layers for image
        self.im_conv1 = nn.Conv3d(in_channels=3,out_channels=64,kernel_size=[7,7,5],stride=[2,2,2])
        self.im_pool1 = nn.Maxpool3d(kernel_size=[1,3,3],stride=[1,2,2])
        self.im_res1 = residual_block(64,64,[3,3,3],1)
        self.im_res2 = residual_block(64,64,[3,3,3],[2,2,2])
        # audio layers
        # input of form [Batch,channels,time,height width]
        self.a_conv1 = nn.Conv3d(in_channels=2,out_channels=64,kernel_size=[65,1,1],stride=4)
        self.a_pool1 = nn.Maxpool3d(kernel_size=[4,1,1],stride=[4,1,1])
        self.a_res1 = residual_block(64,128,[15,1,1],[4,1,1])
        self.a_res2 = residual_block(128,128,[15,1,1],[4,1,1])
        self.a_res3 = residual_block(128,256,[15,1,1],[4,1,1])
        self.a_pool2 = nn.Maxpool3d(kernel_size=[3,1,1],stride=[3,1,1])
        self.a_conv2 = nn.Conv3d(in_channels=256,out_channels=128,kernel_size=[3,1,1])
        # fusion layers
        self.f_conv1 = nn.Conv3d(in_channels=192,out_channels=512,kernel_size=[1,1,1])
        self.f_conv2 = nn.Conv3d(in_channels=512,out_channels=128,kernel_size=[1,1,1])
        self.bn_f = nn.BatchNorm3d(128)
        self.relu_f = nn.ReLu(inplace=True)
        
        self.c_res1 = residual_block(128,128,[3,3,3],1)
        self.c_res2 = residual_block(128,128,[3,3,3],1)
        
        self.c_res3 = residual_block(128,256,[3,3,3],[2,2,2])
        self.c_res4 = residual_block(256,256,[3,3,3],1)
        
        self.c_res5 = residual_block(256,512,[3,3,3],[1,2,2])
        self.c_res6 = residual_block(512,512,[3,3,3],1)
        
        self.avgpool = nn.AvgPool3d([16,7,7])
        self.f_fcn = nn.Linear(512,1)
        self.cmm_weights = self.f_fcn.weight
        # activations and fc layers
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLu()
        self.logsigmoid = nn.LogSigmoid()

    def forward(self, x, y):
        # x is the rgb image and y is the audio 
        # x shape is assumed to be Bx3x125x224x224 
        # and y's shape is assumed to be B*2*1*1*n_samples
        # first the image part
        x = F.relu(self.im_conv1(x))
        x = self.im_pool1(x)
        x = self.im_res1(x)
        x= self.im_res2(x)
        # now extract from the audio
        y = F.relu(self.a_conv1(y))
        y = self.a_pool1(y)
        y = self.a_res1(y)
        y = self.a_res2(y)
        y = self.a_res3(y)
        y = self.a_pool2(y)
        y = F.relu(self.a_conv2(y))
        # now combine both of these variables

        shape = list(x.size())
        y_tiled = y.repeat([1,1,shape[2],shape[3],1])
        combined = torch.cat([x,y_tiled],4)
        short = torch.cat([combined[:,:,:,:,:64],combined[:,:,:,:,-64:]],4)

        combined = F.relu(self.f_conv1(combined))
        combined = self.f_conv2(combined)
        combined = self.bn_f(combined+short)
        combined = self.relu_f(combined)

        combined = self.c_res1(combined)
        combined = self.c_res2(combined)
        combined = self.c_res3(combined)
        combined = self.c_res4(combined)
        combined = self.c_res5(combined)
        combined = self.c_res6(combined)

        gap = self.avgpool(combined)
        logits = self.f_fcn(gap[:,0,0,0,:])
        logits = self.sigmoid(logits)
        probs, idxs = logits.sort(1, True)
        class_idx = idxs[:, 0]
        cam = torch.bmm(self.cmm_weights[class_idx].unsqueeze(1), combined)
        return logits,cam
