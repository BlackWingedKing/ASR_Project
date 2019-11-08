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
from data_loader import DataLoaderTrain
# parameters and hyper params
batch_size = 1
nepochs = 1
LR = 0.001

#gpu settings
use_cuda = torch.cuda.is_available()
print('gpu status ===',use_cuda)
torch.manual_seed(1)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

ce_loss = nn.CrossEntropyLoss()
def train(vmodel, amodel, avmodel, optimiser, epochs, train_loader):
    # train function 
    # load data from the dataloader
    loss_list = []
    it = 0
    for vid, aus, au in enumerate(train_loader):
        vid = vid.to(device)
        aus = aus.to(device)        
        au = au.to(device)
        vfeat = vmodel(vid)
        afeat = amodel(au)
        asfeat = amodel(aus)
        p = avmodel(vfeat, afeat)
        ps = avmodel(vfeat, asfeat)
        gt = torch.ones_like(p).to(device)
        # loss = torch.mean(torch.log(p) + torch.log(1-ps))
        loss = torch.mean(ce_loss(p, gt) + ce_loss((1-ps), gt))
        loss.backward()
        loss_list.append(loss.item())
        print('training ','iteration: ',it,'avg loss: ',loss.item())

def main():
    # initialise the models
    vmodel = VideoNet().to(device)
    amodel = AudioNet().to(device)
    avmodel = AVNet().to(device)
    params = list(vmodel.parameters())+list(amodel.parameters())+list(avmodel.parameters())
    optimiser = optim.Adam(params,lr=LR)
    train_loader = DataLoaderTrain()
    train(vmodel, amodel, avmodel, optimiser, nepochs, train_loader)

if __name__ == '__main__':
    main()    