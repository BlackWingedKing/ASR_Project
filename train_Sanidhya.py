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
from data_loader import AVDataset, Resize, RandomCrop
import utils
from tqdm import tqdm
from findcam import *
# parameters and hyper params
batch_size = 1
test_batch_size = 1
nepochs = 100
LR = 1e-4

#gpu settings
use_cuda = torch.cuda.is_available()
print('gpu status ===', use_cuda)
torch.manual_seed(1)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

bce_loss = nn.BCELoss()

def train(vmodel, amodel, avmodel, optimiser, epochs, train_loader, val_loader):
    # train function
    # load data from the dataloader
    loss_list = []
    val_list = []
    p_list = []
    ps_list = []
    it = 0
    prev_loss = 10000
    for e in range(0,epochs):
        vmodel.train()
        amodel.train()
        avmodel.train()
        trainloss = 0.0
        i = 0
        for batch_id, (vid, aus, au) in enumerate(train_loader):
            i+=1
            optimiser.zero_grad()
            vmodel.zero_grad()
            amodel.zero_grad()
            avmodel.zero_grad()
            aus1 = aus.unsqueeze(3).unsqueeze(4)#.to(device)
            au1 = au.unsqueeze(3).unsqueeze(4)#.to(device)
            vid1 = torch.cat([vid,vid]).to(device)
            au2 = torch.cat([au1,aus1]).to(device)

            vfeat = vmodel(vid1)
            afeat = amodel(au2)
            q, _ = avmodel(vfeat, afeat)
            ql = torch.chunk(q, 2, dim=0)
            p = ql[0]
            ps = ql[1]
            gt = torch.ones_like(p).to(device)
            gt2 = torch.zeros_like(ps).to(device)
            loss = bce_loss(p,gt) + bce_loss(ps,gt2)
            loss.backward()
            optimiser.step()
            print('train stats:', p.data,ps.data)
            p_list.append(p.data[0][0])
            ps_list.append(ps.data[0][0])
            trainloss+=loss.item()
        trainloss = trainloss*batch_size
        trainloss/=len(train_loader)
        # valoss,_,_ = val(vmodel, amodel, avmodel, val_loader)
        loss_list.append(trainloss)
        # val_list.append(valoss)

        print('epoch: ', e, 'iteration: ', it, 'train loss: ', trainloss)#, 'val loss: ', valoss)
    #     if(prev_loss >= valoss):
    #          print('saving the model ')
    #          torch.save(amodel.state_dict(), 'amodel.pt')
    #          torch.save(vmodel.state_dict(), 'vmodel.pt')
    #          torch.save(avmodel.state_dict(), 'avmodel.pt')
    #          prev_loss = valoss
    #          print('model saved')
    dicty = {'train_loss': loss_list,'p':p_list, 'ps':ps_list}#, 'val loss': val_list}
    dft = pd.DataFrame(dicty)
    dft.to_hdf('logtest.h5', key='data')

def val(vmodel, amodel, avmodel, val_loader):
    vmodel.eval()
    amodel.eval()
    avmodel.eval()
    avgloss = 0.0
    with torch.no_grad():
        print("Validation:")
        for batch_id, (vid, aus, au) in enumerate(val_loader):
            vid = vid.to(device)
            aus = aus.unsqueeze(3).unsqueeze(4).to(device)
            au = au.unsqueeze(3).unsqueeze(4).to(device)
            vfeat = vmodel(vid)
            afeat = amodel(au)
            asfeat = amodel(aus)
            p, cam = avmodel(vfeat, afeat)
            ps, _ = avmodel(vfeat, asfeat)
            print('val stats:', p.data,ps.data)
            gt = torch.ones_like(p).to(device)
            # loss = torch.mean(torch.log(p) + torch.log(1-ps))
            loss = torch.mean(bce_loss(p,gt) + bce_loss((1-ps),gt))
            avgloss+= loss.item()
        return avgloss*test_batch_size/len(val_loader),p,cam

def main():
    # initialise the models
    vmodel = VideoNet().to(device)
    amodel = AudioNet().to(device)
    avmodel = AVNet().to(device)
    vmodel.load_state_dict(torch.load('vmodel_final.pt'))
    amodel.load_state_dict(torch.load('amodel_final.pt'))
    avmodel.load_state_dict(torch.load('avmodel_final.pt'))
    print('loaded model')
    params = list(vmodel.parameters())+list(amodel.parameters())+list(avmodel.parameters())
    # optimiser = optim.Adam(params, lr=LR)
    optimiser = optim.SGD(params, lr=LR, momentum=0.9)

    list_vid = os.listdir('data/train/full_vid')  # ensure no extra files like .DS_Store are present
    train_list, val_list = utils.split_data(list_vid, 0.8, 0.2)
    # log the list for reference
    utils.log_list(train_list, 'data/train_list.txt')
    utils.log_list(val_list, 'data/val_list.txt')
    # uncomment following to read previous list
    # train_list = utils.read_list('data/train_list.txt')
    # val_list = utils.read_list('data/val_list.txt')
    train_list = ['video_001.mp4']
    composed = transforms.Compose([Resize(256), RandomCrop(224)])
    # composed = transforms.Compose([Resize(256)])
    train_loader = torch.utils.data.DataLoader(AVDataset(train_list[:1], transform=composed), batch_size=batch_size, shuffle=False, num_workers=4)
    val_loader = torch.utils.data.DataLoader(AVDataset(train_list[:1], transform=composed), batch_size=batch_size,shuffle=False, num_workers=4)
    l,p,cam=val(vmodel,amodel,avmodel,val_loader)
    print(p,cam.shape)
    import skvideo.io
    vids=skvideo.io.vread('data/train/'+'snippet/video_001.mp4')
    # print('vids',vids)
    findcam(np.expand_dims(vids,0),np.abs(cam.cpu().numpy()))
    # train(vmodel, amodel, avmodel, optimiser, nepochs, train_loader, val_loader)


if __name__ == '__main__':
    main()
