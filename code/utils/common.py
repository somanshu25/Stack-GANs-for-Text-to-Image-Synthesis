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
import os
import errno

class residualBlocks(nn.Module):
    def __init__(self,channels):
        super(residualBlocks,self).__init__()
        self.resBlock = nn.Sequential (
            nn.Conv2d(channels,channels,3,stride = 1, padding =1 , bias = False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels,channels,3,stride = 1, padding =1 , bias = False),
            nn.BatchNorm2d(channels))
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
        nn.BatchNorm2d(outputChannels),
        nn.ReLU(True)
    )
    return upSamplingBlock

def downSamplingAtomic(inputChannels,outputChannels):
    downSamplingBlock = nn.Sequential(
         nn.Conv2d(inputChannels, outputChannels, 4, 2, 1, bias=False),
         nn.BatchNorm2d(outputChannels),
         nn.LeakyReLU(True),
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

class LogitsForDiscriminator(nn.Module):
    def __init__(self,tDim,cDim,conditionCrit = True):
        super(LogitsForDiscriminator,self).__init__()
        self.tDim = tDim * 8
        self.cDim = cDim
        self.conditionCrit = conditionCrit
        if self.conditionCrit:
            self.output = nn.Sequential(
                nn.Conv2d(self.tDim + self.cDim,self.tDim,3,stride = 1, padding =1 , bias = False),
                nn.BatchNorm2d(self.tDim),
                nn.LeakyReLU(0.2,True),
                nn.Conv2d(self.tDim,1,kernel_size = 4,stride = 4),
                nn.Sigmoid()
            )
        else:
            self.output = nn.Sequential(
                nn.Conv2d(self.tDim,1,kernel_size = 4,stride = 4),
                nn.Sigmoid()  
            ) 
    
    def forward(self,h,c = None):
        if self.conditionCrit and c is not None:
            c = c.view(-1,self.cDim,1,1)
            c = c.repeat(1,1,4,4)
            hc = torch.cat((h,c),1)
        else:
            hc = h
        
        output = self.output(hc)
        return output.view(-1)


def computeGeneratorLoss(netDisc,fakeImages,realLabels,conditions,gpus):
    criterion = nn.BCELoss()
    cond = conditions.detach()

    ## fake logits and error
    _,fakeLogits = nn.parallel.data.parallel(netDisc,(fakeImages,cond),gpus)
    errorDiscFake = criterion(fakeLogits,realLabels)

    return errorDiscFake

def computeDiscriminatorLoss(netDisc,fakeImages,realImages,fakeLabels,realLabels,conditions,gpus):
    criterion = nn.BCELoss()
    cond = conditions.detach()
    fakeImages = fakeImages.detach()

    ## fake pairs
    fakeEmbedding = nn.parallel.data.parallel(netDisc,(fakeImages),gpus)
    fakeLogits = nn.parallel.data.parallel(netDisc.getConditionalLogits,(fakeEmbedding,cond),gpus)
    errDiscFake = criterion(fakeLogits,realLabels)

    ## real paris
    realEmbedding = nn.parallel.data.parallel(netDisc,(realImages),gpus)
    realLogits = nn.parallel.data.parallel(netDisc.getConditionalLogits,(realEmbedding,cond),gpus)
    errDiscReal = criterion(realLogits,realLabels)

    ## wrong paris
    wrongLogits = nn.parallel.data.parallel(netDisc.getConditionalLogits,(realEmbedding[:realImages.shape[0]-1],cond[1:0]),gpus)
    errDiscWrong = criterion(wrongLogits,fakeLabels[1:0])

    return errDiscReal + 0.5*errDiscFake + 0.5*errDiscWrong, errDiscReal, errDiscWrong, errDiscFake

def save_images(data_image, fake_im, epoch, image_dir):
    img1_size = cfg.ImageSizeStageI
    fake_im = fake[0:img1_size]
    if data_image is not None:
        data_image = data_image[0:img1_size]
        torchvision.utils.save_image(data_image, '%s/true_samples.png' % image_dir, normalize=True)
        torchvision.utils.save_image(fake_im.data, '%s/fake_samples_epoch_%03d.png' % (image_dir, epoch), normalize=True)
               
def save_model(netG, netD, epoch, model_dir):
  
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (model_dir, epoch))
    torch.save(netD.state_dict(),' %s/netD_epoch_last.pth' % (model_dir))
    print('Models saved')
    
def makedir(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
    

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


	
