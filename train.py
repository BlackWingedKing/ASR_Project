'''
Run this code for training
'''
# imports
from __future__ import print_function
# torch imports
import torch
import torch.nn as nn
import torch.nn.init as init
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
# from findcam import *
# parameters and hyper params
batch_size = 2
test_batch_size = 2
nepochs = 100
LR = 5e-4

#gpu settings
use_cuda = torch.cuda.is_available()
print('gpu status ===', use_cuda)
torch.manual_seed(1)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

bce_loss = nn.BCELoss()


def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)

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
        mp,mps = 0.0, 0.0
        for batch_id, (vid, aus, au) in enumerate(train_loader):
            i+=1
            print('running ', batch_id, 'out of',len(train_loader))
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
            # loss = bce_loss(p,gt) + bce_loss(ps,gt2)
            loss = -1*torch.mean(p*torch.log(p) + (1-ps)*torch.log(1-ps))
            loss.backward()
            
            optimiser.step()
            trainloss+=loss.item()
            mp+=torch.mean(p).item()
            mps+=torch.mean(ps).item()
        trainloss = trainloss*batch_size
        trainloss/=len(train_loader)
        valoss,_,_ = val(vmodel, amodel, avmodel, val_loader)
        loss_list.append(trainloss)
        val_list.append(valoss)
        mp = mp/len(train_loader)
        mps = mps/len(train_loader)
        p_list.append(mp)
        ps_list.append(mps)

        print('epoch: ', e, 'iteration: ', it, 'train loss: ', trainloss, 'val loss: ',valoss, 'mean p: ', mp,'mean ps:', mps)
        if(prev_loss >= valoss):
             print('saving the model ')
             torch.save(amodel.state_dict(), 'amodel_final.pt')
             torch.save(vmodel.state_dict(), 'vmodel_final.pt')
             torch.save(avmodel.state_dict(), 'avmodel_final.pt')
             prev_loss = valoss
             print('model saved')
    dicty = {'train_loss': loss_list,'p':p_list, 'ps':ps_list, 'val loss': val_list}
    dft = pd.DataFrame(dicty)
    dft.to_hdf('logtest.h5', key='data')

def val(vmodel, amodel, avmodel, val_loader):
    vmodel.eval()
    amodel.eval()
    avmodel.eval()
    avgloss = 0.0
    with torch.no_grad():
        print("Validating....")
        for batch_id, (vid, aus, au) in enumerate(val_loader):
            vid = vid.to(device)
            aus = aus.unsqueeze(3).unsqueeze(4).to(device)
            au = au.unsqueeze(3).unsqueeze(4).to(device)
            vfeat = vmodel(vid)
            afeat = amodel(au)
            asfeat = amodel(aus)
            p, cam = avmodel(vfeat, afeat)
            ps, _ = avmodel(vfeat, asfeat)
            # print('val stats:', p.data,ps.data)
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
    
#     amodel.apply(weight_init)
#     avmodel.apply(weight_init)
#     vmodel.apply(weight_init)
    vmodel.load_state_dict(torch.load('./pretrained/tfvmodel.pt'))
    amodel.load_state_dict(torch.load('./pretrained/tfamodel.pt'))
    avmodel.load_state_dict(torch.load('./pretrained/tfavmodel.pt'))
    print('loaded model')
    
    params = list(vmodel.parameters())+list(amodel.parameters())+list(avmodel.parameters())
    optimiser = optim.Adam(params, lr=LR)
#     optimiser = optim.SGD(params, lr=LR, momentum=0.9)

    list_vid = os.listdir('data/train/full_vid')  # ensure no extra files like .DS_Store are present
    train_list, val_list = utils.split_data(list_vid, 0.9, 0.1)
    # log the list for reference
    utils.log_list(train_list, 'data/train_list.txt')
    utils.log_list(val_list, 'data/val_list.txt')
    # uncomment following to read previous list
    # train_list = utils.read_list('data/train_list.txt')
    # val_list = utils.read_list('data/val_list.txt')
    # train_list = ['video_1.mp4']
    composed = transforms.Compose([Resize(256), RandomCrop(224)])
    # composed = transforms.Compose([Resize(256)])
    train_loader = torch.utils.data.DataLoader(AVDataset(train_list, transform=composed), batch_size=batch_size, shuffle=False, num_workers=4)
    val_loader = torch.utils.data.DataLoader(AVDataset(val_list, transform=composed), batch_size=batch_size,shuffle=False, num_workers=4)
    # l,p,cam=val(vmodel,amodel,avmodel,val_loader)
    # print(p,cam.shape)
    # import skvideo.io
    # vids=skvideo.io.vread('data/train/'+'snippet/video_1.mp4')
    # findcam(np.expand_dims(vids,0),cam.cpu().numpy())
    train(vmodel, amodel, avmodel, optimiser, nepochs, train_loader, val_loader)


if __name__ == '__main__':
    main()
