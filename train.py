'''
Run this code for training
'''
# imports
from __future__ import print_function
# torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchvision.models as models
from torchvision import transforms, models
import torch.optim as optim
# np, images
import numpy as np
# import cv2
# data processing
import os
import pandas as pd
import glob
import time
from model_fused import AVNet, VideoNet, AudioNet
from data_loader import DataLoader, Rescale, RandomCrop
import utils
# parameters and hyper params
batch_size = 1
nepochs = 1
LR = 0.001

#gpu settings
use_cuda = torch.cuda.is_available()
print('gpu status ===', use_cuda)
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
        p, _ = avmodel(vfeat, afeat)
        ps, _ = avmodel(vfeat, asfeat)
        gt = torch.ones_like(p).to(device)
        # loss = torch.mean(torch.log(p) + torch.log(1-ps))
        loss = torch.mean(ce_loss(p, gt) + ce_loss((1-ps), gt))
        loss.backward()
        loss_list.append(loss.item())
        print('training ', 'iteration: ', it, 'avg loss: ', loss.item())


def main():
    # initialise the models
    vmodel = VideoNet().to(device)
    amodel = AudioNet().to(device)
    avmodel = AVNet().to(device)
    params = list(vmodel.parameters())+list(amodel.parameters())+list(avmodel.parameters())
    optimiser = optim.Adam(params, lr=LR)
    list_vid = os.listdir('data/train/full_vid')  # ensure no extra files like .DS_Store are present
    train_list, val_list = utils.split_data(list_vid, 0.8, 0.2)
    # log the list for reference
    utils.log_list(train_list, 'data/train_list.txt')
    utils.log_list(val_list, 'data/val_list.txt')
    ## uncomment following to read previous list
    # train_list = utils.read_list('data/train_list.txt')
    # val_list = utils.read_list('data/val_list.txt')
    composed = transforms.Compose([RandomCrop(224), Rescale(256)])
    train_loader = DataLoader(train_list, transform=composed)
    val_loader = DataLoader(val_list, transform=composed)

    train(vmodel, amodel, avmodel, optimiser, nepochs, train_loader)


if __name__ == '__main__':
    main()
