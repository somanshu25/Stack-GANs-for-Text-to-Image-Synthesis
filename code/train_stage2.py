
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
from stage1 import STAGE1_Generator
from stage2 import GenStageII,DiscStageII
from utils.common import computeGeneratorLoss,computeDiscriminatorLoss, makedir, weights_init, save_images, save_model, KL_loss
import errno

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class STAGE2_GAN:
    def __init__(self, output_dir):
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

        if cfg.Stage1Gen != '':
            state_dict = torch.load(cfg.Stage1Gen,map_location=lambda storage, loc: storage)
            netG.STAGE1_Generator.load_state_dict(state_dict)
            print('Load from: ', cfg.Stage1Gen)

        else:
            print("For Stage2, needs weights of Stage1 Generator")
            return

        if cfg.CUDA:
            netG = netG.cuda()
            netD = netD.cuda()

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
                # print('''[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f Loss_KL: %.4f
                #      Loss_real: %.4f Loss_wrong:%.4f Loss_fake %.4f
                #      Total Time: %.2f sec
                #      '''
                #
                #      % (epoch, self.max_epoch, i, len(data_loader), errDisc.data.item(), errGen.data.item(), kl_loss.data.item(), errDiscReal.item(), errDiscWrong.item(), errDiscFake.item(), (end_t - start_t)))
                #

            if(epoch % cfg.saveModelEpoch == 0 or epoch == self.max_epoch-1):
                save_model(netG, netD, epoch, self.model_dir)

   def eval(self,dataloader,listIndex):
        netG, _ = self.loadStage2()       # Initialize Generator
        nz = cfg.LatentDim                  # 100
        latentInitial = Variable(torch.FloatTensor(self.batch_size,nz))   ## Latent variables

        if cfg.CUDA:
            latentInitial = latentInitial.cuda()

        for i, data in enumerate(dataloader):

            if i in listIndex:
                realImgs, txtEmbedding = data
                realImgs = Variable(realImgs)
                txtEmbedding = Variable(txtEmbedding)
                if cfg.CUDA:
                    realImgs = realImgs.cuda()
                    txtEmbedding = txtEmbedding.cuda()

                ##### Generate fake images
                latentInitial.data.normal_(0, 1)
                netG.eval()
                stage1Img, fakeImgs,mu,logvar = netG(txtEmbedding, latentInitial)
                save_images(realImgs, fakeImgs ,stage1Img, i, self.image_dir)
