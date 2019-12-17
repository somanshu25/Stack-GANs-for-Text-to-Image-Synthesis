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

from utils.datasets import TextDataset
from utils.config import cfg,cfg_from_file
from utils.common import makedir
from train import STAGE2_GAN
import pdb
%load_ext tensorboard
import os
import time, datetime

if __name__ == "__main__":
    cfg_from_file('cfg\\birds_eval.yml')
    np.random.seed(0)
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
    dataset = TextDataset(cfg.DataDir, 'test',imsize=cfg.ImSize, transform=image_transform)
    assert dataset
    #print(dataset.filenames)
    dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=cfg.BatchSize * num_gpu,
            drop_last=True, shuffle=False)

    listIndex = np.random.choice(len(dataloader),30)
    listIndex = np.sort(listIndex)
    print(listIndex)
    fileNamesSample = []
    for i in listIndex:
        fileNamesSample.append(dataset.filenames[i])

    print(fileNamesSample)
    algo = STAGE2_GAN(output_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    algo.eval(dataloader,listIndex)  #cfg.Stage
