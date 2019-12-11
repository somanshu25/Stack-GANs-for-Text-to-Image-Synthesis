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
from train import STAGE1_GAN
import pdb


if __name__ == "__main__":
    #cfg_from_file('cfg\cocoStage1.yml')
    cfg_from_file('cfg\stage1_birds.yml')
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '..\output\%s_%s_%s' % (cfg.DatasetName,cfg.ConfigName,timestamp)
    num_gpu = len(cfg.GPU_ID.split(','))
    image_transform = transforms.Compose([
            transforms.RandomCrop(cfg.ImSize),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    #print("Cfg datadir ", cfg.DataDir)
    dataset = TextDataset(cfg.DataDir, 'train',imsize=cfg.ImSize, transform=image_transform)
    # assert dataset
    dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=cfg.BatchSize * num_gpu,
            drop_last=True, shuffle=True, num_workers=int(cfg.Workers))

    algo = STAGE1_GAN(output_dir)
    #print(len(dataloader))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #print(device)
    algo.train1(dataloader)  #cfg.Stage
