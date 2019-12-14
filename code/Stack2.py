#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install torch
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os
import time
import numpy as np
import torchfile
print('Torch', torch.__version__, 'CUDA', torch.version.cuda)
print('Device:', torch.device('cuda:0'))
torch.cuda.get_device_name(0)


# In[2]:


get_ipython().run_line_magic('cd', 'C:\\Users\\somanshu\\Desktop\\Stack-GANs-for-Text-to-Image-Synthesis\\code')


# In[3]:


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
        txt_embed = txt_embed.cuda(device)
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
        h1 = self.up1(h1)
        h1 = self.up2(h1)
        h1 = self.up3(h1)
        h1 = self.up4(h1)
        
        generated_image = self.image1(h1)
        
        del h1
        return None, generated_image, mu, logvar


# In[4]:


import numpy as np
import os.path as osp
from easydict import EasyDict as edict

__C = edict()
cfg = __C

# Config variables for General 

__C.TextDim = 1024
__C.LatentDim = 100
# __C.ImageSizeStageI = 64
# __C.ImageSizeStageII = 256
__C.CUDA = True
__C.GPU_ID = '0'
__C.DatasetName = ''
__C.Workers = 1
__C.ConfigName = ''
__C.Stage = 1
__C.DataDir = ''
__C.ImSize = 64
# Config variables for Training GANS

__C.BatchSize = 4
__C.NumEpochs = 50
__C.LrDecayEpoch = 600
__C.saveModelEpoch = 50
__C.DiscLR = 0.0002
__C.GenLR = 0.0002
__C.KLCoeff = 2
__C.RNum = 4
# Config parameters for mode options for GANS

__C.ConditionDim = 128
__C.DiscInputDim = 64
__C.GenInputDim = 128
__C.StageIGen= ''
# __C.DataDir= '..\\Data\\birds'
def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if not k in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)


# In[5]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import torch.utils.data as data
from PIL import Image
import PIL
import os
import os.path
import pickle
import random
import numpy as np
import pandas as pd

# from utils.config import cfg


class TextDataset(data.Dataset):
    def __init__(self, data_dir, split='train', embedding_type='cnn-rnn',
                 imsize=64, transform=None, target_transform=None):

        self.transform = transform
        self.target_transform = target_transform
        self.imsize = imsize
        self.data = []
        self.data_dir = data_dir
        if data_dir.find('birds') != -1:
            self.bbox = self.load_bbox()
        else:
            self.bbox = None
        split_dir = os.path.join(data_dir, split)
        # C:\Users\vedire\Desktop\Stack-GANs-for-Text-to-Image-Synthesis\data\coco\train

        self.filenames = self.load_filenames(split_dir)
        self.embeddings = self.load_embedding(split_dir, embedding_type)
        self.class_id = self.load_class_id(split_dir, len(self.filenames))
        # self.captions = self.load_all_captions()

    def get_img(self, img_path, bbox):
        img = Image.open(img_path).convert('RGB')
        width, height = img.size
        if bbox is not None:
            R = int(np.maximum(bbox[2], bbox[3]) * 0.75)
            center_x = int((2 * bbox[0] + bbox[2]) / 2)
            center_y = int((2 * bbox[1] + bbox[3]) / 2)
            y1 = np.maximum(0, center_y - R)
            y2 = np.minimum(height, center_y + R)
            x1 = np.maximum(0, center_x - R)
            x2 = np.minimum(width, center_x + R)
            img = img.crop([x1, y1, x2, y2])
        load_size = int(self.imsize * 76 / 64)
        img = img.resize((load_size, load_size), PIL.Image.BILINEAR)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def load_bbox(self):
        data_dir = self.data_dir
        bbox_path = os.path.join(data_dir, 'CUB_200_2011/bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)
        #
        filepath = os.path.join(data_dir, 'CUB_200_2011/images.txt')
        df_filenames =             pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        print('Total filenames: ', len(filenames), filenames[0])
        #
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in range(0, numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()

            key = filenames[i][:-4]
            filename_bbox[key] = bbox
        #
        return filename_bbox

    def load_all_captions(self):
        caption_dict = {}
        for key in self.filenames:
            caption_name = '%s/text/%s.txt' % (self.data_dir, key)
            captions = self.load_captions(caption_name)
            caption_dict[key] = captions
        return caption_dict

    def load_captions(self, caption_name):
        cap_path = caption_name
        with open(cap_path, "r") as f:
            captions = f.read().decode('utf8').split('\n')
        captions = [cap.replace("\ufffd\ufffd", " ")
                    for cap in captions if len(cap) > 0]
        return captions

    def load_embedding(self, data_dir, embedding_type):
        if embedding_type == 'cnn-rnn':
            # embedding_filename = '/char-CNN-RNN-embeddings.pickle'
            embedding_filename = '/char-CNN-RNN-embeddings.pickle'
            #embedding_filename = '\char-CNN-RNN-embeddings-subset.pickle'
        elif embedding_type == 'cnn-gru':
            embedding_filename = '/char-CNN-GRU-embeddings.pickle'
        elif embedding_type == 'skip-thought':
            embedding_filename = '/skip-thought-embeddings.pickle'

        with open(data_dir + embedding_filename, 'rb') as f:
            embeddings = pickle.load(f,encoding='latin1')
            embeddings = np.array(embeddings)
            #subset_embeddings = np.array(embeddings)
            # embedding_shape = [embeddinqqgs.shape[-1]]
        #return embeddings
        return embeddings

    def load_class_id(self, data_dir, total_num):
        print('loading: ', data_dir)
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f,encoding='latin1')

        else:
            class_id = np.arange(total_num)
        return class_id

    def load_filenames(self, data_dir):
        filepath = os.path.join(data_dir, 'filenames.pickle')
        with open(filepath, 'rb') as f:
            filenames = pickle.load(f,encoding='latin1')
        print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        # return filenames
        return filenames

    def __getitem__(self, index):
        key = self.filenames[index]
        # cls_id = self.class_id[index]
        #
        if self.bbox is not None:
            bbox = self.bbox[key]
            data_dir = '%s/CUB_200_2011' % self.data_dir
        else:
            bbox = None
            data_dir = self.data_dir

        # captions = self.captions[key]
        embeddings = self.embeddings[index, :, :]
        img_name = '%s/images/%s.jpg' % (data_dir, key)
        img = self.get_img(img_name, bbox)

        embedding_ix = random.randint(0, embeddings.shape[0]-1)
        embedding = embeddings[embedding_ix, :]
        if self.target_transform is not None:
            embedding = self.target_transform(embedding)
        return img, embedding

    def __len__(self):
        return len(self.filenames)
        #return len(self.embeddings)


# In[6]:


import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torch.optim as optim
import os
import time
import numpy as np
import torchfile
# from utils.config import cfg
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
        nn.Upsample(scale_factor=2,mode='nearest'),
        nn.Conv2d(inputChannels,outputChannels,3,stride = 1, padding =1 , bias = False),
        nn.BatchNorm2d(outputChannels),
        nn.ReLU(True)
    )
    return upSamplingBlock

def downSamplingAtomic(inputChannels,outputChannels):
    downSamplingBlock = nn.Sequential(
         nn.Conv2d(inputChannels, outputChannels, 4, 2, 1, bias=False),
         nn.BatchNorm2d(outputChannels),
         nn.LeakyReLU(0.2,True)        ##   Change 2: Leaky ReLu with 0.2 argument
    ) 
    return downSamplingBlock

def downSamplingAtomicII(inputChannels,outputChannels):
    downSamplingBlockII = nn.Sequential(
         nn.Conv2d(inputChannels, outputChannels, 4, 2, 1, bias=False),
         nn.BatchNorm2d(outputChannels),
         nn.ReLU(True)    
    ) 
    return downSamplingBlockII

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
        
        del c,hc
        return output.view(-1)

import pdb 

def computeGeneratorLoss(netDisc,fakeImages,realLabels,conditions,gpus):
    criterion = nn.BCELoss()
    cond = conditions.detach()

    fakeFeatures = netDisc(fakeImages)
    ## fake logits and error
    #_,fakeLogits = nn.parallel.data.parallel(netDisc,(fakeImages,cond),gpus)
    fakeLogits = netDisc.getConditionalLogits(fakeFeatures,cond)
    errorDiscFake = criterion(fakeLogits,realLabels)
    
    if netDisc.getUnconditionalLogits is not None:
        fakeLogits = netDisc.getUnconditionalLogits(fakeFeatures)
        errDiscFakeUncond = criterion(fakeLogits,realLabels)
        errorDiscFake += errDiscFakeUncond
    
    del fakeFeatures,fakeLogits
    return errorDiscFake

def computeDiscriminatorLoss(netDisc,fakeImages,realImages,fakeLabels,realLabels,conditions,gpus):
    criterion = nn.BCELoss()
    cond = conditions.detach()
    fakeImages = fakeImages.detach()
    #pdb.set_trace()
    ## fake pairs
    # fakeEmbedding = nn.parallel.data_parallel(netDisc,(fakeImages),gpus)
    fakeEmbedding = netDisc(fakeImages)
    #fakeLogits = nn.parallel.data.parallel(netDisc.getConditionalLogits,(fakeEmbedding,cond),gpus)
    
    fakeLogits = netDisc.getConditionalLogits(fakeEmbedding,cond)
    errDiscFake = criterion(fakeLogits,fakeLabels)

    ## real paris
    #realEmbedding = nn.parallel.data.parallel(netDisc,(realImages),gpus)
    realEmbedding = netDisc(realImages)

    #realLogits = nn.parallel.data.parallel(netDisc.getConditionalLogits,(realEmbedding,cond),gpus)
    realLogits = netDisc.getConditionalLogits(realEmbedding,cond)
    errDiscReal = criterion(realLogits,realLabels)

    ## wrong paris
    #wrongLogits = nn.parallel.data.parallel(netDisc.getConditionalLogits,(realEmbedding[:realImages.shape[0]-1],cond[1:0]),gpus)
    wrongLogits = netDisc.getConditionalLogits(realEmbedding[:(realImages.shape[0]-1)],cond[1:])
    errDiscWrong = criterion(wrongLogits,fakeLabels[1:])

    if netDisc.getUnconditionalLogits is not None:
        realLogits = netDisc.getUnconditionalLogits(realEmbedding)
        fakeLogits = netDisc.getUnconditionalLogits(fakeEmbedding)

        errDiscRealUncond = criterion(realLogits, realLabels)
        errDiscFakeUncond = criterion(fakeLogits, fakeLabels)

        errDisc = ((errDiscReal + errDiscRealUncond) / 2. +
                (errDiscFake + errDiscWrong + errDiscFakeUncond) / 3.)
        errDiscReal = (errDiscReal + errDiscRealUncond) / 2.
        errDiscFake = (errDiscFake + errDiscFakeUncond) / 2.

    else:
        errDisc = errDiscReal + 0.5*errDiscFake + 0.5*errDiscWrong
    
    del fakeEmbedding,fakeLogits,realLogits,realEmbedding
    return errDisc, errDiscReal, errDiscWrong, errDiscFake

def save_images(data_image, fake_im, stage1Img, epoch, image_dir):
    img1_size = cfg.ImSize
    fake_im = fake_im[0:img1_size]
    if data_image is not None:
        data_image = data_image[0:img1_size]
        torchvision.utils.save_image(data_image, '%s/true_samples.png' % image_dir, normalize=True)
        print('Saving True image')
        torchvision.utils.save_image(fake_im.data, '%s/fake_samples_epoch_%03d.png' % (image_dir, epoch), normalize=True)
        print('Saving Fake image')
        torchvision.utils.save_image(stage1Img.data, '%s/stage1Img_samples_epoch_%03d.png' % (image_dir, epoch), normalize=True)
        print('Saving stage1 output image')
        
               
def save_model(netG, netD, epoch, model_dir):
  
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (model_dir, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (model_dir, epoch))
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



# In[7]:


import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os
import time
import numpy as np
import torchfile
from torchvision import models
# from utils.config import cfg
# from utils.common import ConditioningAugment
# from utils.common import upSamplingAtomic, LogitsForDiscriminator, residualBlocks, downSamplingAtomicII
# from stage1 import STAGE1_Generator


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
        
        self.gDim = cfg.GenInputDim
        self.cDim = cfg.ConditionDim
        self.zDim = cfg.LatentDim
        self.STAGE1_Generator = STAGE1_Generator
        self.network()

    def createLayer(self, resblock, numChannel):

        layers = []
        for i in range(cfg.RNum):
            layers.append(resblock(numChannel))
        return nn.Sequential(*layers)

    def network(self):

        GID = self.gDim
        
        self.ConditioningAugment = ConditioningAugment()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, GID, 3, stride=1, padding=1, bias=False),
            nn.ReLU(True))
        self.downSample1 = downSamplingAtomicII(GID, GID * 2)
        self.downSample2 = downSamplingAtomicII(GID * 2, GID * 4)

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
            nn.Conv2d(DID * 32, DID * 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(DID * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(DID * 16, DID * 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(DID * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.getConditionalLogits  = LogitsForDiscriminator(DID, CD, conditionCrit=True)
        self.getUnconditionalLogits  = LogitsForDiscriminator(DID, CD, conditionCrit=False)

    def forward(self, img):
        #pdb.set_trace()
        imgEmbedding = self.ImgEncode(img)
        down1 = self.down1(imgEmbedding)
        down1 = self.down2(down1)
        down1 = self.down3(down1)
        down1 = self.down4(down1)
        down1 = self.down5(down1)
        finalOutput = self.tempEncode(down1)
        del down1
        return finalOutput


# In[ ]:





# In[8]:


import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os
import time
import numpy as np
import torchfile
# from utils.config import cfg
# from stage1 import STAGE1_Generator
# from stage2 import GenStageII,DiscStageII
# from utils.common import computeGeneratorLoss,computeDiscriminatorLoss, makedir, weights_init, save_images, save_model, KL_loss
import errno

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class STAGE2_GAN:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.model_dir = os.path.join(output_dir,'Model2')
        self.image_dir = os.path.join(output_dir,'Image2')
        self.log_dir = os.path.join(output_dir,'Log2')
        makedir(self.model_dir)
        makedir(self.image_dir)
        makedir(self.log_dir)
        #self.summary_writer = FileWriter(self.log_dir)

        self.max_epoch = cfg.NumEpochs
        s_gpus = cfg.GPU_ID.split(',')
        self.gpus = [int(ix) for ix in s_gpus]
        self.num_gpus = len(self.gpus)
        self.batch_size = cfg.BatchSize * self.num_gpus


    def loadStage2(self):
        stage1 = STAGE1_Generator()
        netG = GenStageII(stage1)
        netD = DiscStageII()
        netG.apply(weights_init)
        netD.apply(weights_init)
        if cfg.StageIGen != '':
            state_dict = torch.load(cfg.StageIGen,map_location=lambda storage, loc: storage)
            netG.STAGE1_Generator.load_state_dict(state_dict, strict=False)
            print('Load from: ', cfg.StageIGen)

        else:
            print("For Stage2, needs weights of Stage1 Generator")
            return

        if cfg.CUDA:
            netG = netG.cuda()
            netD = netD.cuda()
            
        del stage1,state_dict
        
        return netG,netD

    def train2(self,data_loader):
        netG, netD = self.loadStage2()       # Initialize Generator and Discriminator
        netG = netG.to(device)
        netD = netD.to(device)
        nz = cfg.LatentDim                  # 100
        latentInitial = Variable(torch.FloatTensor(self.batch_size,nz))   ## Latent variables
        realLabels = Variable(torch.FloatTensor(self.batch_size).fill_(1))  ## Real values (1)
        fakeLabels = Variable(torch.FloatTensor(self.batch_size).fill_(0))  ## Fake values (0)
        if cfg.CUDA:
            realLabels = realLabels.cuda()
            fakeLabels = fakeLabels.cuda()
            ## Changes: Add latentInitial into the cuda device
            latentInitial = latentInitial.cuda()

        ## Parameters for Generator and Discriminator
        genLR = cfg.GenLR
        discLR = cfg.DiscLR
        lrDecayEpoch = cfg.LrDecayEpoch
        netG_para = []
        for p in netG.parameters():
            if p.requires_grad:
                netG_para.append(p)
        optimizerG = optim.Adam(netG_para,lr=genLR,betas=(0.5, 0.999))

        optimizerD = optim.Adam(netD.parameters(), lr=discLR, betas=(0.5, 0.999))

        kl_loss_values= []
        errDiscValues= []
        errDiscRealValues = []
        errDiscWrongValues = []
        errDiscFakeValues = []
        errGenValues = []
        for epoch in range(self.max_epoch):
            print(epoch)
            #### Decay the learning rates after some epochs
            start_t = time.time()
            if epoch % lrDecayEpoch == 0 and epoch > 0:
                genLR *= 0.5
                for param_group in optimizerG.param_groups:
                    param_group['lr'] = genLR
                discLR *= 0.5
                for param_group in optimizerD.param_groups:
                    param_group['lr'] = discLR
            kl_loss_total = 0
            errDiscTotal= 0
            errDiscRealTotal = 0
            errDiscWrongTotal = 0
            errDiscFakeTotal = 0
            errGenTotal = 0

            for i, data in enumerate(data_loader):
                ##### Prepare training data
                realImgs, txtEmbedding = data
                realImgs = Variable(realImgs)
                txtEmbedding = Variable(txtEmbedding)
                if cfg.CUDA:
                    realImgs = realImgs.cuda()
                    txtEmbedding = txtEmbedding.cuda()

                ##### Generate fake images
                latentInitial.data.normal_(0, 1)
                inputs = (txtEmbedding, latentInitial)
#                 _, fakeImgs, mu, logvar = nn.parallel.data_parallel(netG, inputs, self.gpus)
                stage1Img, fakeImgs,mu,logvar = netG(txtEmbedding, latentInitial)

                #####  Update D network
                netD.zero_grad()
                errDisc, errDiscReal, errDiscWrong, errDiscFake = computeDiscriminatorLoss(netD, realImgs, fakeImgs, realLabels, fakeLabels, mu, self.gpus)
                errDisc.backward(retain_graph=True)
                optimizerD.step()

                #####  Update G network
                netG.zero_grad()
                errGen = computeGeneratorLoss(netD, fakeImgs, realLabels, mu, self.gpus)
                kl_loss = KL_loss(mu, logvar)
                print(kl_loss,errGen)
                errG_total = errGen + kl_loss * cfg.KLCoeff
                errG_total.backward()
                optimizerG.step()

                if i % 100 ==0 and epoch % 5 == 0:
                    save_images(realImgs, fakeImgs ,stage1Img, epoch, self.image_dir)

                end_t = time.time()
                print('''[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f Loss_KL: %.4f
                     Loss_real: %.4f Loss_wrong:%.4f Loss_fake %.4f
                     Total Time: %.2f sec
                     '''
                      
                     % (epoch, self.max_epoch, i, len(data_loader), errDisc.data.item(), errGen.data.item(), kl_loss.data.item(), errDiscReal.item(), errDiscWrong.item(), errDiscFake.item(), (end_t - start_t)))

                kl_loss_total += kl_loss.data.item()
                errDiscTotal += errDisc.data.item()
                errGenTotal += errGen.data.item()
                errDiscRealTotal += errDiscReal.item()
                errDiscWrongTotal += errDiscWrong.item()
                errDiscFakeTotal += errDiscWrong.item()

            if(epoch % cfg.saveModelEpoch == 0 or epoch == self.max_epoch-1):
                save_model(netG, netD, epoch, self.model_dir)

            kl_loss_values.append(kl_loss_total)
            errDiscValues.append(errDiscTotal)
            errGenValues.append(errGenTotal)
            errDiscRealValues.append(errDiscRealTotal)
            errDiscWrongValues.append(errDiscWrongTotal)
            errDiscFakeValues.append(errDiscFakeTotal)

        return kl_loss_values,errDiscValues,errGenValues,errDiscRealValues,errDiscWrongValues,errDiscFakeValues


# In[ ]:





# In[9]:


def get_tensorboard_logger():
  logs_base_dir = "./logs"
  logs_dir = logs_base_dir + "/run_" + str(time.mktime(datetime.datetime.now().timetuple()))
  os.makedirs(logs_dir, exist_ok=True)
  from torch.utils.tensorboard import SummaryWriter
  get_ipython().run_line_magic('tensorboard', '--logdir {logs_base_dir}')
  logger = SummaryWriter(logs_dir)
  return logger


# In[10]:


import numpy as np
import torch.backends.cudnn as cudnn
import torch
import torchvision.transforms as transforms

import argparse
import os
import random
import sys
import pprint
import datetime
import dateutil
import dateutil.tz

# from utils.datasets import TextDataset
# from utils.config import cfg,cfg_from_file
# from utils.common import makedir
# from train import STAGE1_GAN
import pdb
get_ipython().run_line_magic('load_ext', 'tensorboard')
import os
import time, datetime


if __name__ == "__main__":
    cfg_from_file('cfg\stage2_birds.yml')
    torch.cuda.manual_seed_all(0)
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '..\output\%s_%s_%s' % (cfg.DatasetName,cfg.ConfigName,timestamp)
    num_gpu = len(cfg.GPU_ID.split(','))
    image_transform = transforms.Compose([
            transforms.RandomCrop(cfg.ImSize),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    print("Cfg datadir ", cfg.DataDir)
    dataset = TextDataset(cfg.DataDir, 'train',imsize=cfg.ImSize, transform=image_transform)
    # assert dataset
    dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=cfg.BatchSize * num_gpu,
            drop_last=True, shuffle=True)

    algo = STAGE2_GAN(output_dir)
    #print(len(dataloader))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    algo.train2(dataloader)  #cfg.Stage


# In[ ]:





# In[ ]:


get_ipython().system('pwd')


# In[ ]:




