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
from utils.config import cfg
from utils.common import make_dir,STAGE1_GAN

if __name__ == "__main__":
    now = 