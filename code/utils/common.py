import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os
import time
import numpy as np
import torchfile
from config import cfg

class residualBlocks(nn.Module):
    def __init__(self,channels):
        self.resBlock = nn.Sequential (
            nn.Conv2d(channels,channels,3,stride = 1, padding =1 , bias = False),
            nn.BatchNorm2D(channels),
            nn.ReLU(True),
            nn.Conv2d(channels,channels,3,stride = 1, padding =1 , bias = False),
            nn.BatchNorm2D(channels))
        self.relu  = nn.ReLU(True)

    def forward(self,input):
        output = self.resBlock(input)
        output = output + input
        output = self.relu(output)
        return output
 
def upSamplingAtomic(inputChannels, outputChannels):
    upSamplingBlock = nn.Sequential (
        nn.Upsample(2,'nearest'),
        nn.Conv2d(inputChannels,outputChannels,3,stride = 1, padding =1 , bias = False),
        nn.BatchNorm2D(outputChannels),
        nn.ReLU(True)
    )
    return upSamplingBlock

def downSamplingAtomic(inputChannels,outputChannels):
    downSamplingBlock = nn.Sequential(
         nn.Conv2d(inputChannels, outputChannels, 4, 2, 1, bias=False),
         nn.BatchNorm2d(outputChannels),
         nn.ReLU(True),
    ) 
    return downSamplingBlock

def KL_loss(mu, logvar):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD

class ConditioningAugment(nn.Module):
    def __init__(self):
        super(ConditioningAugment,self).__init__()
        self.tDim = cfg.TextDim
        self.cDim = cfg.ConditionDim
        self.fc = nn.Linear(self.tDim,self.cDim * 2)
        self.relu = nn.ReLU()
    
    def encode(self,embeddings):
        out = self.fc(embeddings)
        out = self.relu(out)
        mu = out[:,:self.cDim]
        var = out[:,self.cDim:]
        return mu,var

    def forward(self,embeddings):
        mu, var = self.encode(embeddings)
        std = var.mul(0.5).exp_()
        if cfg.CUDA:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        temp = eps.mul(std).add_(mu)
        return temp, mu, var



    



	
