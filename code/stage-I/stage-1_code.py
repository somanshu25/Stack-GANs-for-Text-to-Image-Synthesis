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
            nn.ReLU)

        #Upsampling
        self.up1 = upSamplingAtomic(num_text_input, num_text_input // 2)            # 1024 x 4 x 4 - 512 x 8 x 8
        self.up2 = upSamplingAtomic(num_text_input // 2, num_text_input // 4)       # 512 x 8 x 8 - 256 x 16 x 16
        self.up3 = upSamplingAtomic(num_text_input // 4, num_text_input // 8)       # 256 x 16 x 16 - 128 x 32 x 32
        self.up4 = upSamplingAtomic(num_text_input // 8, num_text_input // 16)      # 128 x 32 x 32 - 64 x 64 x 64
        self.image1 = nn.Sequential(                          # 64 x 64 x 64 - 3 x 64 x 64 (Reducing number of channels to 3)
            conv3x3(num_text_input // 16, 3),
            nn.Tanh())

    def forward(self, txt_embed, noise):
      
        #Conditional Augmentation
        cond, mu, logvar = self.ConditioningAugment(txt_embed)
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
        
        return generated_image
      
      
      
