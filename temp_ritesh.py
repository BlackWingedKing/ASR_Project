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
import os
import tensorflow as tf
from pprint import pprint

def calprod(a):
    s=1
    for i in a:
        s*=i
    return s
# code for testing
Vmodel = VideoNet()
Amodel = AudioNet()
AVmodel = AVNet()
adict = Amodel.state_dict()
vdict = Vmodel.state_dict()
avdict = AVmodel.state_dict()
akeys = adict.keys()
vkeys = vdict.keys()
avkeys = avdict.keys()
ashapes = []
vshapes = []
avshapes = []

ashape = []
vshape = []
avshape = []

for i in akeys:
    x = adict[i].shape
    ashapes.append(calprod(x))
    ashape.append((i, x))

for i in vkeys:
    x = vdict[i].shape
    vshapes.append(calprod(x))
    vshape.append((i, x))

for i in avkeys:
    x = avdict[i].shape
    avshapes.append(calprod(x))
    avshape.append((i, x))

# akeys, ashapes, vkeys, vshapes are in sync

tf_path = os.path.abspath('/home/ritesh/Desktop/multisensory/results/nets/shift/net.tf-650000')  # Path to our TensorFlow checkpoint
tf_vars = tf.train.list_variables(tf_path)

pprint(tf_vars)
# print(tf_vars)
# for i in tf_vars:
#     print'\'':str(i[0]),'\'':':': i[1])
tfkeys = []
tfshapes = []

# for i in tf_vars:
#     tfkeys.append(i[0])
#     tfshapes.append(calprod(i[1]))
# print(len(tfshapes), len(ashapes), len(vshapes), len(avshapes))

# pprint(vshape)
# pprint(ashape)
# pprint(avshape)

# stfshapes, stfkeys = zip(*sorted(zip(tfshapes, tfkeys)))
# sashapes, sakeys = zip(*sorted(zip(ashapes, akeys)))
# svshapes, svkeys = zip(*sorted(zip(vshapes, vkeys)))
# savshapes, savkeys = zip(*sorted(zip(avshapes, avkeys)))

# ptkeys = akeys + vkeys + avkeys
# ptshapes = ashapes + vshapes + avshapes
# sptshapes, sptkeys = zip(*sorted(zip(ptshapes, ptkeys)))

# print(len(list(set(list(sptshapes)))), len(list(set(list(stfshapes)))))

# here the dict creation starts
vmapper = {
 'im_conv1.weight': 'im/conv1/weights',
#  'im_conv1.bias': 'n',
 'bn.weight': 'n',
 'bn.bias': 'im/conv1/BatchNorm/beta',
 'bn.running_mean': 'im/conv1/BatchNorm/moving_mean',
 'bn.running_var': 'im/conv1/BatchNorm/moving_variance',
 'bn.num_batches_tracked': 'n',
 'im_res1.conv1.weight': 'im/conv2_1_1/weights',
#  'im_res1.conv1.bias': 'n',
 'im_res1.conv2.weight': 'im/conv2_1_2/weights',
 'im_res1.conv2.bias': 'im/conv2_1_2/biases',
 'im_res1.bn1.weight': 'n',
 'im_res1.bn1.bias': 'im/conv2_1_1/BatchNorm/beta',
 'im_res1.bn1.running_mean': 'im/conv2_1_1/BatchNorm/moving_mean',
 'im_res1.bn1.running_var': 'im/conv2_1_1/BatchNorm/moving_variance',
 'im_res1.bn1.num_batches_tracked': 'n',
 'im_res1.bn2.weight': 'n',
 'im_res1.bn2.bias': 'im/conv2_1_bn/beta',
 'im_res1.bn2.running_mean': 'im/conv2_1_bn/moving_mean',
 'im_res1.bn2.running_var': 'im/conv2_1_bn/moving_variance',
 'im_res1.bn2.num_batches_tracked': 'n',
 'im_res2.conv1.weight': 'im/conv2_2_1/weights',
#  'im_res2.conv1.bias': 'n',
 'im_res2.conv2.weight': 'im/conv2_2_2/weights',
 'im_res2.conv2.bias': 'im/conv2_2_2/biases',
 'im_res2.bn1.weight': 'n',
 'im_res2.bn1.bias': 'im/conv2_2_1/BatchNorm/beta',
 'im_res2.bn1.running_mean': 'im/conv2_2_1/BatchNorm/moving_mean',
 'im_res2.bn1.running_var': 'im/conv2_2_1/BatchNorm/moving_variance',
 'im_res2.bn1.num_batches_tracked': 'n',
 'im_res2.bn2.weight': 'n',
 'im_res2.bn2.bias': 'im/conv2_2_bn/beta',
 'im_res2.bn2.running_mean': 'im/conv2_2_bn/moving_mean',
 'im_res2.bn2.running_var': 'im/conv2_2_bn/moving_variance',
 'im_res2.bn2.num_batches_tracked': 'n'
}

amapper = {
 'a_conv1.weight': 'sf/conv1_1/weights',
#  'a_conv1.bias': 'n',
 'bn1.weight': 'n',
 'bn1.bias': 'sf/conv1_1/BatchNorm/beta',
 'bn1.running_mean': 'sf/conv1_1/BatchNorm/moving_mean',
 'bn1.running_var': 'sf/conv1_1/BatchNorm/moving_variance',
 'bn1.num_batches_tracked': 'n',
 'a_res1.conv1.weight': 'sf/conv2_1_1/weights',
 'a_res1.conv1.bias': 'n',
 'a_res1.conv2.weight': 'sf/conv2_1_2/weights',
 'a_res1.conv2.bias': 'sf/conv2_1_2/biases',
 'a_res1.bn1.weight': 'n',
 'a_res1.bn1.bias': 'sf/conv2_1_1/BatchNorm/beta',
 'a_res1.bn1.running_mean': 'sf/conv2_1_1/BatchNorm/moving_mean',
 'a_res1.bn1.running_var': 'sf/conv2_1_1/BatchNorm/moving_variance',
 'a_res1.bn1.num_batches_tracked': 'n',
 'a_res1.bn2.weight': 'n',
 'a_res1.bn2.bias': 'sf/conv2_1_bn/beta',
 'a_res1.bn2.running_mean': 'sf/conv2_1_bn/moving_mean',
 'a_res1.bn2.running_var': 'sf/conv2_1_bn/moving_variance',
 'a_res1.bn2.num_batches_tracked': 'n',
 'a_res1.downsample_layer.weight': 'sf/conv2_1_short/weights',
 'a_res1.downsample_layer.bias': 'n',
 ######  here missed a batch norm for downsample layer model change needed
 'a_res2.conv1.weight': 'sf/conv3_1_1/weights',
 'a_res2.conv1.bias': 'n',
 'a_res2.conv2.weight': 'sf/conv3_1_2/weights',
 'a_res2.conv2.bias': 'sf/conv3_1_2/biases',
 'a_res2.bn1.weight': 'n',
 'a_res2.bn1.bias': 'sf/conv3_1_1/BatchNorm/beta',
 'a_res2.bn1.running_mean': 'sf/conv3_1_1/BatchNorm/moving_mean',
 'a_res2.bn1.running_var': 'sf/conv3_1_1/BatchNorm/moving_variance',
 'a_res2.bn1.num_batches_tracked': 'n',
 'a_res2.bn2.weight': 'n',
 'a_res2.bn2.bias': 'sf/conv3_1_bn/beta',
 'a_res2.bn2.running_mean': 'sf/conv3_1_bn/moving_mean',
 'a_res2.bn2.running_var': 'sf/conv3_1_bn/moving_variance',
 'a_res2.bn2.num_batches_tracked': 'n',
 'a_res3.conv1.weight': 'sf/conv4_1_1/weights',
 'a_res3.conv1.bias': 'n',
 'a_res3.conv2.weight': 'sf/conv4_1_2/weights',
 'a_res3.conv2.bias': 'sf/conv4_1_2/biases',
 'a_res3.bn1.weight': 'n',
 'a_res3.bn1.bias': 'sf/conv4_1_1/BatchNorm/beta',
 'a_res3.bn1.running_mean': 'sf/conv4_1_1/BatchNorm/moving_mean',
 'a_res3.bn1.running_var': 'sf/conv4_1_1/BatchNorm/moving_variance',
 'a_res3.bn1.num_batches_tracked': 'n',
 'a_res3.bn2.weight': 'n',
 'a_res3.bn2.bias': 'sf/conv4_1_bn/beta',
 'a_res3.bn2.running_mean': 'sf/conv4_1_bn/moving_mean',
 'a_res3.bn2.running_var': 'sf/conv4_1_bn/moving_variance',
 'a_res3.bn2.num_batches_tracked': 'n',
 'a_res3.downsample_layer.weight': 'sf/conv4_1_short/weights',
 'a_res3.downsample_layer.bias': 'n',
 ############### here missing anpther batch norm
 'a_conv2.weight': 'sf/conv5_1/weights',
 'a_conv2.bias': 'n',
 'bn2.weight': 'n',
 'bn2.bias': 'sf/conv5_1/BatchNorm/beta',
 'bn2.running_mean': 'sf/conv5_1/BatchNorm/moving_mean',
 'bn2.running_var': 'sf/conv5_1/BatchNorm/moving_variance',
 'bn2.num_batches_tracked': 'n'
}

avmapper = {
 'cmm_weights': 'joint/logits/weights',
 'f_conv1.weight': 'im/merge1/weights',
 'f_conv1.bias': 'n',
############# missing batch norm for the merge1 or f_conv1 
 'f_conv2.weight': 'im/merge2/weights',
 'f_conv2.bias': 'im/merge2/biases',
 'bn_f.weight': 'n',
 'bn_f.bias': 'im/merge_block_bn/beta',
 'bn_f.running_mean': 'im/merge_block_bn/moving_mean',
 'bn_f.running_var': 'im/merge_block_bn/moving_variance',
 'bn_f.num_batches_tracked': 'n',
 'c_res1.conv1.weight': 'im/conv3_1_1/weights',
 'c_res1.conv1.bias': 'n',
 'c_res1.conv2.weight': 'im/conv3_1_2/weights',
 'c_res1.conv2.bias': 'im/conv3_1_2/biases',
 'c_res1.bn1.weight': 'n',
 'c_res1.bn1.bias': 'im/conv3_1_1/BatchNorm/beta',
 'c_res1.bn1.running_mean': 'im/conv3_1_1/BatchNorm/moving_mean',
 'c_res1.bn1.running_var': 'im/conv3_1_1/BatchNorm/moving_variance',
 'c_res1.bn1.num_batches_tracked': 'n',
 'c_res1.bn2.weight': 'n',
 'c_res1.bn2.bias': 'im/conv3_1_bn/beta',
 'c_res1.bn2.running_mean': 'im/conv3_1_bn/moving_mean',
 'c_res1.bn2.running_var': 'im/conv3_1_bn/moving_variance',
 'c_res1.bn2.num_batches_tracked': 'n',
 'c_res2.conv1.weight': 'im/conv3_2_1/weights',
 'c_res2.conv1.bias': 'n',
 'c_res2.conv2.weight': 'im/conv3_2_2/weights',
 'c_res2.conv2.bias': 'im/conv3_2_2/biases',
 'c_res2.bn1.weight': 'n',
 'c_res2.bn1.bias': 'im/conv3_2_1/BatchNorm/beta',
 'c_res2.bn1.running_mean': 'im/conv3_2_1/BatchNorm/moving_mean',
 'c_res2.bn1.running_var': 'im/conv3_2_1/BatchNorm/moving_variance',
 'c_res2.bn1.num_batches_tracked': 'n',
 'c_res2.bn2.weight': 'n',
 'c_res2.bn2.bias': 'im/conv3_2_bn/beta',
 'c_res2.bn2.running_mean': 'im/conv3_2_bn/moving_mean',
 'c_res2.bn2.running_var': 'im/conv3_2_bn/moving_variance',
 'c_res2.bn2.num_batches_tracked': 'n',
 'c_res3.conv1.weight': 'im/conv4_1_1/weights',
 'c_res3.conv1.bias': 'n',
 'c_res3.conv2.weight': 'im/conv4_1_2/weights',
 'c_res3.conv2.bias': 'im/conv4_1_2/biases',
 'c_res3.bn1.weight': 'n',
 'c_res3.bn1.bias': 'im/conv4_1_1/BatchNorm/beta',
 'c_res3.bn1.running_mean': 'im/conv4_1_1/BatchNorm/moving_mean',
 'c_res3.bn1.running_var': 'im/conv4_1_1/BatchNorm/moving_variance',
 'c_res3.bn1.num_batches_tracked': 'n',
 'c_res3.bn2.weight': 'n',
 'c_res3.bn2.bias': 'im/conv4_1_bn/beta',
 'c_res3.bn2.running_mean': 'im/conv4_1_bn/moving_mean',
 'c_res3.bn2.running_var': 'im/conv4_1_bn/moving_variance',
 'c_res3.bn2.num_batches_tracked': 'n',
 'c_res3.downsample_layer.weight': 'im/conv4_1_short/weights',
 'c_res3.downsample_layer.bias': 'n',
 ###################### misssing a bn for short or downsample layer here
 'c_res4.conv1.weight': 'im/conv4_2_1/weights',
 'c_res4.conv1.bias': 'n',
 'c_res4.conv2.weight': 'im/conv4_2_2/weights',
 'c_res4.conv2.bias': 'im/conv4_2_2/biases',
 'c_res4.bn1.weight': 'n',
 'c_res4.bn1.bias': 'im/conv4_2_1/BatchNorm/beta',
 'c_res4.bn1.running_mean': 'im/conv4_2_1/BatchNorm/moving_mean',
 'c_res4.bn1.running_var': 'im/conv4_2_1/BatchNorm/moving_variance',
 'c_res4.bn1.num_batches_tracked': 'n',
 'c_res4.bn2.weight': 'n',
 'c_res4.bn2.bias': 'im/conv4_2_bn/beta',
 'c_res4.bn2.running_mean': 'im/conv4_2_bn/moving_mean',
 'c_res4.bn2.running_var': 'im/conv4_2_bn/moving_variance',
 'c_res4.bn2.num_batches_tracked': 'n',
 'c_res5.conv1.weight': 'im/conv5_1_1/weights',
 'c_res5.conv1.bias': 'n',
 'c_res5.conv2.weight': 'im/conv5_1_2/weights',
 'c_res5.conv2.bias': 'im/conv5_1_2/biases',
 'c_res5.bn1.weight': 'n',
 'c_res5.bn1.bias': 'im/conv5_1_1/BatchNorm/beta',
 'c_res5.bn1.running_mean': 'im/conv5_1_1/BatchNorm/moving_mean',
 'c_res5.bn1.running_var': 'im/conv5_1_1/BatchNorm/moving_variance',
 'c_res5.bn1.num_batches_tracked': 'n',
 'c_res5.bn2.weight': 'n',
 'c_res5.bn2.bias': 'im/conv5_1_bn/beta',
 'c_res5.bn2.running_mean': 'im/conv5_1_bn/moving_mean',
 'c_res5.bn2.running_var': 'im/conv5_1_bn/moving_variance',
 'c_res5.bn2.num_batches_tracked': 'n',
 'c_res5.downsample_layer.weight': 'im/conv5_1_short/weights',
 'c_res5.downsample_layer.bias': 'n',
 ############## misssing batch norm for short layers here
 'c_res6.conv1.weight': 'im/conv5_2_1/weights',
 'c_res6.conv1.bias': 'n',
 'c_res6.conv2.weight': 'im/conv5_2_2/weights',
 'c_res6.conv2.bias': 'im/conv5_2_2/biases',
 'c_res6.bn1.weight': 'n',
 'c_res6.bn1.bias': 'im/conv5_2_1/BatchNorm/beta',
 'c_res6.bn1.running_mean': 'im/conv5_2_1/BatchNorm/moving_mean',
 'c_res6.bn1.running_var': 'im/conv5_2_1/BatchNorm/moving_variance',
 'c_res6.bn1.num_batches_tracked': 'n',
 'c_res6.bn2.weight': 'n',
 'c_res6.bn2.bias': 'im/conv5_2_bn/beta',
 'c_res6.bn2.running_mean': 'im/conv5_2_bn/moving_mean',
 'c_res6.bn2.running_var': 'im/conv5_2_bn/moving_variance',
 'c_res6.bn2.num_batches_tracked': 'n',
 'f_fcn.weight': 'joint/logits/weights',
 'f_fcn.bias': 'joint/logits/biases'
}