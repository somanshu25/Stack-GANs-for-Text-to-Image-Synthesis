import numpy as np
import os.path as osp
from easydict import EasyDict as edict

__C = edict()
cfg = __C

# Config variables for General 

__C.TextDim = 1024
__C.LatentDim = 100
__C.ImageSizeStageI = 64
__C.ImageSizeStageII = 256 
__C.CUDA = True
__C.GPU_ID = '0'

# Config variables for Training GANS

__C.BatchSize = 64
__C.NumEpochs = 600
__C.LrDecayEpoch = 600
__C.saveModelEpoch = 50
__C.DiscLR = 0.0002
__C.GenLR = 0.0002
__C.KLCoeff = 2

# Config parameters for mode options for GANS

__C.ConditionDim = 128
__C.DiscInputDim = 64
__C.GenInputDim = 128



