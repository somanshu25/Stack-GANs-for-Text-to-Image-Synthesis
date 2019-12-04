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
from common import ConditioningAugment
from common import downSamplingAtomic, upSamplingAtomic, LogitsForDiscriminator
from stage1 import STAGE1_Generator,STAGE1_Discriminator
from computeLoss import computeGeneratorLoss,computeDiscriminatorLoss

class STAGE1_GAN:
    def __init__(self):
        pass

    def loadStage1(self):
        netG = STAGE1_Generator()           # Initialize Generator
        netD = STAGE1_Discriminator()       # Initialize Discriminator

        netG.apply(weights_init)
        netD.apply(weights_init)

        if cfg.CUDA:
            netG = netG.cuda()
            netD = netD.cuda()

        return netG,netD
    
    def train1(self):
        netG,netD = self.loadStage1()
        nz = cfg.LatentDim
        latentInitial = Variable(torch.FloatTensor(batch,nz))
        realLabels = Variable(torch.FloatTensor(batch).fill_(1))
        fakeLabels = Variable(torch.FloatTensor(batch).fill_(0))

        if cfg.CUDA:
            realLabels = realLabels.cuda()
            fakeLabels = fakeLabels.cuda()
        
        ## Parameters for Generator and Discriminator
        genLR = cfg.GenLR
        discLR = cfg.DiscLR
        lrDecayEpoch = cfg.LrDecayEpoch

        optimizerD = \
            optim.Adam(netD.parameters(),
                       lr=discLR, betas=(0.5, 0.999))
        netG_para = []
        for p in netG.parameters():
            if p.requires_grad:
                netG_para.append(p)
        optimizerG = optim.Adam(netG_para,
                                lr=cfg.TRAIN.GENERATOR_LR,
                                betas=(0.5, 0.999))
        
        count = 0
        for epoch in range(self.max_epoch):
            start_t = time.time()
            if epoch % lr_decay_step == 0 and epoch > 0:
                generator_lr *= 0.5
                for param_group in optimizerG.param_groups:
                    param_group['lr'] = generator_lr
                discriminator_lr *= 0.5
                for param_group in optimizerD.param_groups:
                    param_group['lr'] = discriminator_lr

        





