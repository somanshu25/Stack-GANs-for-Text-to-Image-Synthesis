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
from StageI import GenStageI



def DiscdownSampling(inputChannels,outputChannels):
    DiscdownSamplingBlock = nn.Sequential(
         nn.Conv2d(inputChannels, outputChannels, 4, 2, 1, bias=False),
         nn.BatchNorm2d(outputChannels),
         nn.LeakyReLU(0.2, inplace = True),
    ) 
    return DiscdownSamplingBlock
  
  
  class GenStageII(nn.Module):
  def __init__(self, StageI):
    super(GenStageII, self).__init__()
    self.gDim = cfg.GenInputDim
    self.cDim = cfg.ConditionDim
    self.zDim = cfg.LatentDim
    self.GenStageI = GenStageI

    for param in GenStageI.parameters():
      param.requires_grad = False
    self.network()


    def network(self):
      GID = self.gDim
      self.downSamplingAtomic = downSamplingAtomic()
      self.ConditioningAugment = ConditioningAugment()
      self.encoder = nn.Sequential(
          nn.Conv2d(3,GID,3,stride = 1, padding =1 , bias = False),
          nn.ReLU(True)
          self.downSamplingAtomic(GID, GID*2)
          self.downSamplingAtomic(GID*2, GID*4)
      )
      self.HighRes = nn.Sequential(
          nn.Conv2d(self.cDim+GID*4, GID*4, 3, stride = 1, padding =1, bias = False),
          nn.BatchNorm2d(GID*4)
          nn.ReLU(True)
      )

      
    # def _make_layer(self, block, channel_num):
    #     layers = []
    #     for i in range(cfg.GAN.R_NUM):
    #         layers.append(block(channel_num))
    #     return nn.Sequential(*layers)
    
      self.residual = self._make_layer(ResBlock, GID * 4)
      
      self.upSample1 = upSamplingAtomic(GID * 4, GID * 2)
      self.upSample2 = upSamplingAtomic(GID * 2, GID)
      self.upSample3 = upSamplingAtomic(GID, GID // 2)
      self.upSample4 = upSamplingAtomic(GID // 2, GID // 4)
     
      self.img = nn.Sequential(
          nn.Conv2d(GID // 4, 3, 3, stride = 1, padding =1, bias = False),
          nn.Tanh())

      def forward(self, textEmbedding, noise):
        a, imgStageI, c, d = self.GenStageI(textEmbedding, noise)
        imgStageI = imgStageI.detach()
        encodedImg = self.encoder(imgStageI)

        temp, mu, var = self.ConditioningAugment(textEmbedding)
        temp = temp.view(-1, self.cDim, 1, 1)
        temp = temp.repeat(1, 1, 16, 16)
        temp1 = torch.cat([encodedImg, temp], 1)
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
        nn.LeakyReLU(0.2, inplace=True),
        DiscdownSampling(DID, DID*2),
        DiscdownSampling(DID*2, DID*4),
        DiscdownSampling(DID*4, DID*8),
        DiscdownSampling(DID*8, DID*16),
        DiscdownSampling(DID*16, DID*32),
        nn.Conv2d(DID*32, DID*16, 4, 2, 1, bias=False),
        nn.BatchNorm2d(DID*16),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(DID*16, DID*8, 4, 2, 1, bias=False),
        nn.BatchNorm2d(DID*8),
        nn.LeakyReLU(0.2, inplace=True)    
    )

    self.condLogits = LogitsForDiscriminator(DID, CD, bcondition = True)
    self.uncondLogits = LogitsForDiscriminator(DID, CD, bcondition = False)

    def forward(self, img):
      imgEmbedding = self.ImgEncode(img)
      return imgEmbedding 
