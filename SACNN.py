#!/usr/bin/env python
# coding: utf-8
"""
All network architectures: FBCNet, EEGNet, DeepConvNet
@author: Ravikiran Mane
"""
import torch
import torch.nn as nn
import sys
current_module = sys.modules[__name__]
import numpy as np
from torch.nn import init
from torch.nn.functional import elu

from braindecode.models.base import BaseModel
from braindecode.torch_ext.modules import Expression, AvgPool2dWithConv
from braindecode.torch_ext.functions import identity
from braindecode.torch_ext.util import np_to_var
import math
import torch.nn.functional as F


class SACNN(BaseModel):
    def __init__(self,in_chans,n_classes,input_time_length):
        self.in_chans=in_chans
        self.n_classes=n_classes
        self.input_time_length=input_time_length

    def create_network(self):
        model = SACNN_net()
        return model


class SACNN_net(nn.Module):

    def SpatialModule(self):    #feature generalization module
        net=nn.Sequential(
                nn.Conv2d(25, 25, (1, 4), padding=0),
                nn.Conv2d(25, 25, (32, 1), padding = 0, bias= False),
                nn.BatchNorm2d(25),
                nn.ELU(),
                nn.MaxPool2d((1,3), stride = (1,3)),
                )
        return net
    def SAModule(self, Q, K, V,x,mode=1):
        d_k = K.size(-1)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(d_k)  ##### 1，2--》0,1
        alpha_n = F.softmax(scores, dim=-1)
        weiht=torch.unsqueeze(torch.sum(alpha_n.mean(dim=0),-2),-1)
        weiht=1+weiht/torch.max(weiht)
        if mode==0:
            context = weiht * V
        else:
            context = weiht*x

        return context


    def FeatureGeneralizationModule (self):
        net=nn.Sequential(
                nn.Dropout(p=0.25),
                nn.Conv2d(25, 50, (1,10), bias=False),
                nn.BatchNorm2d(50),
                nn.ELU(),
                nn.MaxPool2d((1, 3), stride=(1, 3)),

                nn.Dropout(p=0.25),
                nn.Conv2d(50, 100, (1, 10), bias=False),
                nn.BatchNorm2d(100),
                nn.ELU(),
                nn.MaxPool2d((1, 3), stride=(1, 3)),

                nn.Dropout(p=0.25),
                nn.Conv2d(100, 200, (1, 10), bias=False),
                nn.BatchNorm2d(200),
                nn.ELU(),
                nn.MaxPool2d((1, 3), stride=(1, 3)),

                nn.Conv2d(200, 4, (1, 7), bias=False),         # Classes and final kernel_size need to be changed.
                nn.LogSoftmax(dim=1)
                )
        return net
    def TemproalModule(self):
        return(nn.Conv2d(1,25, (1,4), padding = 0))

    def __init__(self):
        super(SACNN_net, self).__init__()

        self.SpatialLayer=self.SpatialModule()
        self.TemproalLayer=self.TemproalModule()
        self.FeatureGeneralizationLayer=self.FeatureGeneralizationModule()

        self.LinearQ = nn.Linear(997, 997)
        self.LinearK = nn.Linear(997, 997)
        self.LinearV = nn.Linear(997, 997)


    def forward(self, x):
        x=torch.permute(x,(0,3,1,2))
        x=self.TemproalLayer(x)
        Q=self.LinearQ(x)
        K=self.LinearK(x)
        V=self.LinearV(x)
        x=self.SAModule(Q,K,V,x)
        x=self.SpatialLayer(x)
        x=self.FeatureGeneralizationLayer(x)
        x = torch.squeeze(x, 3)
        x = torch.squeeze(x, 2)
        return x

if __name__=='__main__':
    from torchsummary.torchsummary import summary
    net=SACNN_net().to('cuda:0')
    print(summary(net,(32,1000,1),device='cuda'))
    print(net)