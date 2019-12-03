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

    return errDiscReal + 0.5*errDiscFake + 0.5*errDiscWrong





