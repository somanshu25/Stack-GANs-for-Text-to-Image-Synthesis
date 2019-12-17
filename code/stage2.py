import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os
import time
import numpy as np
import torchfile
from utils.config import cfg
from utils.common import ConditioningAugment
from utils.common import upSamplingAtomic, LogitsForDiscriminator, residualBlocks, downSamplingAtomicII
from stage1 import STAGE1_Generator


def DiscdownSampling(inputChannels, outputChannels):
    DiscdownSamplingBlock = nn.Sequential(
        nn.Conv2d(inputChannels, outputChannels, 4, 2, 1, bias=False),
        nn.BatchNorm2d(outputChannels),
        nn.LeakyReLU(0.2, inplace=True),
    )
    return DiscdownSamplingBlock


class GenStageII(nn.Module):
    def __init__(self, STAGE1_Generator):
        super(GenStageII, self).__init__()

        for param in STAGE1_Generator.parameters():
            param.requires_grad = False
        self.network()

        self.gDim = cfg.GenInputDim
        self.cDim = cfg.ConditionDim
        self.zDim = cfg.LatentDim
        self.STAGE1_Generator = STAGE1_Generator

    def createLayer(self, resblock, numChannel):

        layers = []
        for i in range(cfg.RNum):
            layers.append(resblock(numChannel))
        return nn.Sequential(*layers)

    def network(self):

        GID = self.gDim
        self.downSamplingAtomicII = downSamplingAtomicII()
        self.ConditioningAugment = ConditioningAugment()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, GID, 3, stride=1, padding=1, bias=False),
            nn.ReLU(True))
        self.downSample1 = self.downSamplingAtomicII(GID, GID * 2)
        self.downSample2 = self.downSamplingAtomicII(GID * 2, GID * 4)

        self.HighRes = nn.Sequential(
            nn.Conv2d(self.cDim + GID * 4, GID * 4, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(GID * 4),
            nn.ReLU(True)
        )
        self.residual = self.createLayer(residualBlocks, GID * 4)

        self.upSample1 = upSamplingAtomic(GID * 4, GID * 2)
        self.upSample2 = upSamplingAtomic(GID * 2, GID)
        self.upSample3 = upSamplingAtomic(GID, GID // 2)
        self.upSample4 = upSamplingAtomic(GID // 2, GID // 4)

        self.img = nn.Sequential(
            nn.Conv2d(GID // 4, 3, 3, stride=1, padding=1, bias=False),
            nn.Tanh())

    def forward(self, textEmbedding, noise):
        a, imgStageI, c, d = self.STAGE1_Generator(textEmbedding, noise)
        imgStageI = imgStageI.detach()
        encodedImg = self.encoder(imgStageI)
        tempencodedImg = self.downSample1(encodedImg)
        FinEncodedImg = self.downSample2(tempencodedImg)

        temp, mu, var = self.ConditioningAugment(textEmbedding)
        temp = temp.view(-1, self.cDim, 1, 1)
        temp = temp.repeat(1, 1, 16, 16)
        temp1 = torch.cat([FinEncodedImg, temp], 1)
        temp2 = self.HighRes(temp1)
        temp2 = self.residual(temp2)

        temp2 = self.upSample1(temp2)
        temp2 = self.upSample2(temp2)
        temp2 = self.upSample3(temp2)
        temp2 = self.upSample4(temp2)

        fakeImg = self.img(temp2)

        return imgStageI, fakeImg, mu, var


class DiscStageII(nn.Module):

    def __init__(self):
        super(DiscStageII, self).__init__()
        self.dDim = cfg.DiscInputDim
        self.cDim = cfg.ConditionDim
        self.network()

    def network(self):
        DID = self.dDim
        CD = self.cDim
        self.ImgEncode = nn.Sequential(
            nn.Conv2d(3, DID, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True))

        self.down1 = DiscdownSampling(DID, DID * 2)
        self.down2 = DiscdownSampling(DID * 2, DID * 4)
        self.down3 = DiscdownSampling(DID * 4, DID * 8)
        self.down4 = DiscdownSampling(DID * 8, DID * 16)
        self.down5 = DiscdownSampling(DID * 16, DID * 32)
        self.tempEncode = nn.Sequential(
            nn.Conv2d(DID * 32, DID * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(DID * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(DID * 16, DID * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(DID * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.getConditionalLogits  = LogitsForDiscriminator(DID, CD, conditionCrit=True)
        self.getUnconditionalLogits  = LogitsForDiscriminator(DID, CD, conditionCrit=False)

    def forward(self, img):
        imgEmbedding = self.ImgEncode(img)
        downSample1 = self.down1(imgEmbedding)
        downSample2 = self.down2(downSample1)
        downSample3 = self.down3(downSample2)
        downSample4 = self.down4(downSample3)
        downSample5 = self.down5(downSample4)
        finalOutput = self.tempEncode(downSample5)
        return finalOutput
