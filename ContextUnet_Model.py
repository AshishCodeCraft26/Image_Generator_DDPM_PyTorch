import torch
import torch.nn as nn
import torch.nn.functional as F

from Unet_Models import *

class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat=256, n_cfeat=10, height=28):
        super(ContextUnet, self).__init__()

        # number of input channels, number of intermediate feature maps and number of classes
        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_cfeat = n_cfeat
        self.h = height #assume h == w. must be divisible by 4, so 28,24,20,16...

        # Initialize the initial convolutional layer
        self.init_conv = ResidualConvBlock(in_channels=in_channels, out_channels=n_feat, is_res=True)

        # Initialize the down-sampling path of the U-Net with two levels
        self.down1 = UnetDown(in_channels=n_feat, out_channels=n_feat)
        self.down2 = UnetDown(in_channels=n_feat, out_channels=2*n_feat)

        # Initialise the fully-coonected layers for flatten
        self.to_vec = ToVec()

        # Embed the timestep and context labels with a one-layer fully connected neural network
        self.timeembed1 = EmbedFC(input_dim=1, emb_dim=2*n_feat)
        self.timeembed2 = EmbedFC(input_dim=1, emb_dim=1*n_feat)
        self.contextembed1 = EmbedFC(input_dim=n_cfeat, emb_dim=2*n_feat)
        self.contextembed2 = EmbedFC(input_dim=n_cfeat, emb_dim=1*n_feat)

        # Initialize the up-sampling path of the U-Net with three levels
        self.up0 = InitializeUpSampling(in_channels=2*n_feat, out_channels=2*n_feat, height=self.h)
        self.up1 = UnetUp(in_channels=4*n_feat, out_channels=n_feat)
        self.up2 = UnetUp(in_channels=2*n_feat, out_channels=n_feat)


        # Initialize the final convolutional layers to map to the same number of channels as the input image
        self.out =  InitializeOut(in_channels=2*n_feat, out_channels=n_feat, IN_Channel=self.in_channels)


    def forward(self, x, t, c=None):

        """
        x : (batch, n_feat, h, w) : input image
        t : (batch, n_cfeat)      : time step
        c : (batch, n_classes)    : context label
        """
        # x is the input image, c is the context label, t is the timestep, context_mask says which samples to block the context on

        # pass the input image through the initial convolutional layer
        x = self.init_conv(x)

        # pass the result through the down-sampling path
        down1 = self.down1(x)
        down2 = self.down2(down1)

        # convert the feature maps to a vector and apply an activation
        hiddenvec = self.to_vec(down2)

        # mask out context if context_mask == 1
        if c is None:
            c = torch.zeros(x.shape[0], self.n_cfeat).to(x)
        
        # embed context and timestep
        cemb1 = self.contextembed1(c).view(-1, 2 * self.n_feat, 1, 1)
        temb1 = self.timeembed1(t).view(-1, 2 * self.n_feat, 1, 1)
        cemb2 =  self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        up0 = self.up0(hiddenvec)
        up1 = self.up1(cemb1*up0 + temb1, down2) # add and multiply embeddings
        up2 = self.up2(cemb2*up1 + temb2, down1)

        out = self.out(torch.cat((up2,x), 1))

        return out


        
