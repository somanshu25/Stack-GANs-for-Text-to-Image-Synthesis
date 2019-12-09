# -*- coding: utf-8 -*-
"""GAN_stage1_init.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/195S3pktp1LTzA_ckRMSQdFSjSH8AsUqp
"""

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
from stage1 import STAGE1_Generator,STAGE1_Discriminator
from utils.common import computeGeneratorLoss,computeDiscriminatorLoss, makedir, weights_init, save_images, save_model, KL_loss
#from tensorboard import summary
#from tensorboard import FileWriter
import errno

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class STAGE1_GAN:
    def __init__(self, output_dir):
        self.model_dir = os.path.join(output_dir,'Model')
        self.image_dir = os.path.join(output_dir,'Image')
        self.log_dir = os.path.join(output_dir,'Log')
        makedir(self.model_dir)
        makedir(self.image_dir)
        makedir(self.log_dir)
        #self.summary_writer = FileWriter(self.log_dir)

        self.max_epoch = cfg.NumEpochs
        s_gpus = cfg.GPU_ID.split(',')
        self.gpus = [int(ix) for ix in s_gpus]
        self.num_gpus = len(self.gpus)
        self.batch_size = cfg.BatchSize * self.num_gpus


    def loadStage1(self):
        netG = STAGE1_Generator()           
        netD = STAGE1_Discriminator()       

        netG.apply(weights_init)            # Initialize the weights for Generator
        netD.apply(weights_init)            # Initialize the weights for Discriminator

        if cfg.CUDA:
            netG = netG.cuda()
            netD = netD.cuda()

        return netG,netD
    
    def train1(self,data_loader):
        netG, netD = self.loadStage1()       # Initialize Generator and Discriminator
        netG = netG.to(device)
        netD = netD.to(device)
        nz = cfg.LatentDim                  # 100
        latentInitial = Variable(torch.FloatTensor(self.batch_size,nz))   ## Latent variables
        realLabels = Variable(torch.FloatTensor(self.batch_size).fill_(1))  ## Real values (1)
        fakeLabels = Variable(torch.FloatTensor(self.batch_size).fill_(0))  ## Fake values (0)

        if cfg.CUDA:
            realLabels = realLabels.cuda()
            fakeLabels = fakeLabels.cuda()
        
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
        
        for epoch in range(self.max_epoch):
            print(epoch)
            #### Decay the learning rates after some epochs
            start_t = time.time()
            # if epoch % lrDecayEpoch == 0 and epoch > 0:
            #     genLR *= 0.5
            #     for param_group in optimizerG.param_groups:
            #         param_group['lr'] = genLR
            #     discLR *= 0.5
            #     for param_group in optimizerD.param_groups:
            #         param_group['lr'] = discLR
            print(len(data_loader))
            for i, data in enumerate(data_loader):
                #print('XYZ')

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
                #_, fakeImgs, mu, logvar = nn.parallel.data_parallel(netG, inputs, self.gpus)
                _, fakeImgs,mu,logvar = netG(txtEmbedding, latentInitial)

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
                    # summary_D = summary.scalar('D_loss', errD.data[0])
                    # summary_D_r = summary.scalar('D_loss_real', errD_real)
                    # summary_D_w = summary.scalar('D_loss_wrong', errD_wrong)
                    # summary_D_f = summary.scalar('D_loss_fake', errD_fake)
                    # summary_G = summary.scalar('G_loss', errG.data[0])
                    # summary_KL = summary.scalar('KL_loss', kl_loss.data[0])
                    #
                    # self.summary_writer.add_summary(summary_D, count)
                    # self.summary_writer.add_summary(summary_D_r, count)
                    # self.summary_writer.add_summary(summary_D_w, count)
                    # self.summary_writer.add_summary(summary_D_f, count)
                    # self.summary_writer.add_summary(summary_G, count)
                    # self.summary_writer.add_summary(summary_KL, count)
                    save_images(realImgs, fakeImgs , epoch, self.image_dir)

                end_t = time.time()
                # print('''[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f Loss_KL: %.4f
                #      Loss_real: %.4f Loss_wrong:%.4f Loss_fake %.4f
                #      Total Time: %.2fsec
                #      '''
                #      % (epoch, self.max_epoch, i, len(data_loader), errDisc.data[0], errGen.data[0], kl_loss.data[0], errDiscReal, errDiscWrong, errDiscFake, (end_t - start_t)))

                # if(epoch % cfg.saveModelEpoch == 0):
                #   save_model(netG, netD, epoch, self.model_dir)
