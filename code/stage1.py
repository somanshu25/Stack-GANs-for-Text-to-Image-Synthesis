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
from utils.common import downSamplingAtomic, upSamplingAtomic, LogitsForDiscriminator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class STAGE1_Generator(nn.Module):
    def __init__(self):
        super(STAGE1_Generator, self).__init__()
        self.LatentDim = cfg.LatentDim
        self.ConditionDim = cfg.ConditionDim
        self.input = cfg.GenInputDim * 8          #1024
        self.gen_module()

    def gen_module(self):
        num_input = self.LatentDim + self.ConditionDim  # 100 + 128 = 228
        num_text_input = self.input  #1024
        self.ConditioningAugment = ConditioningAugment()

        #Fully connected layer
        self.fc1 = nn.Sequential(
            nn.Linear(num_input, num_text_input * 4 * 4),
            nn.BatchNorm1d(num_text_input * 4 * 4),
            nn.ReLU(True))

        #Upsampling
        self.up1 = upSamplingAtomic(num_text_input, num_text_input // 2)            # 1024 x 4 x 4 - 512 x 8 x 8
        self.up2 = upSamplingAtomic(num_text_input // 2, num_text_input // 4)       # 512 x 8 x 8 - 256 x 16 x 16
        self.up3 = upSamplingAtomic(num_text_input // 4, num_text_input // 8)       # 256 x 16 x 16 - 128 x 32 x 32
        self.up4 = upSamplingAtomic(num_text_input // 8, num_text_input // 16)      # 128 x 32 x 32 - 64 x 64 x 64
        self.image1 = nn.Sequential(                          # 64 x 64 x 64 - 3 x 64 x 64 (Reducing number of channels to 3)
            nn.Conv2d(num_text_input // 16, 3,3,stride = 1, padding =1 , bias = False),
            nn.Tanh())

    def forward(self, txt_embed, noise):
        print(txt_embed.device)

        txt_embed = txt_embed.cuda(device)
        print(txt_embed.device)
        noise = noise.cuda(device)
      
        #Conditional Augmentation
        cond, mu, logvar = self.ConditioningAugment(txt_embed)
        cond = cond.cuda(device)
        #Add noise
        cond_with_noise = torch.cat((noise, cond), 1)
        #Fully connected layer
        h1 = self.fc1(cond_with_noise)

        h1 = h1.view(-1, self.input, 4, 4)
        
        #Upsampling
        h2 = self.up1(h1)
        h3 = self.up2(h2)
        h4 = self.up3(h3)
        h5 = self.up4(h4)
        
        generated_image = self.image1(h5)
        
        return None, generated_image, mu, logvar

class STAGE1_Discriminator(nn.Module):
    def __init__(self):
        super(STAGE1_Discriminator, self).__init__()
        self.DiscDim = cfg.DiscInputDim
        self.ConditionDim = cfg.ConditionDim
        self.disc_module()

    def disc_module(self):
        discDim = self.DiscDim      # 64
        conditionDIm = self.ConditionDim    #128
        # 3 X 64 X 64
        self.convLayer = nn.Sequential(             
            nn.Conv2d(3,discDim,4,2,1,bias=False),
            nn.LeakyReLU(0.2,inplace = True)
        )
        # 64 X 32 X 32
        self.down1 = downSamplingAtomic(discDim, discDim * 2)
        # 128 X 16 X 16
        self.down2 = downSamplingAtomic(discDim *2 , discDim *4)
        # 256 X 8 X 8
        self.down3 = downSamplingAtomic(discDim * 4, discDim * 8)
        # 512 X 4 X 4

        self.getConditionalLogits = LogitsForDiscriminator(discDim,conditionDIm)
    
    def forward(self,image):
        image.cuda(device)
        c1 = self.convLayer(image)
        imgEncode1 = self.down1(c1)
        imgEncode2 = self.down2(imgEncode1)
        imgEncode = self.down3(imgEncode2)

        return imgEncode

      
