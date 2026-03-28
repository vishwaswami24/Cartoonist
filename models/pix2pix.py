"""
Pix2Pix Model Implementation
Image-to-image translation using conditional GANs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Convolutional block with BatchNorm and LeakyReLU"""
    
    def __init__(self, in_channels, out_channels, stride=1, use_batchnorm=True):
        super().__init__()
        padding = 1
        
        if use_batchnorm:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=4, 
                         stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=4, 
                         stride=stride, padding=padding, bias=True),
                nn.LeakyReLU(0.2, inplace=True)
            )
    
    def forward(self, x):
        return self.conv(x)


class Generator(nn.Module):
    """U-Net based Generator for Pix2Pix"""
    
    def __init__(self, in_channels=3, out_channels=3, ngf=64, dropout=0.5):
        super().__init__()
        
        # Encoder (downsampling)
        self.enc1 = ConvBlock(in_channels, ngf, stride=2, use_batchnorm=False)
        self.enc2 = ConvBlock(ngf, ngf*2, stride=2)
        self.enc3 = ConvBlock(ngf*2, ngf*4, stride=2)
        self.enc4 = ConvBlock(ngf*4, ngf*8, stride=2)
        self.enc5 = ConvBlock(ngf*8, ngf*8, stride=2)
        self.enc6 = ConvBlock(ngf*8, ngf*8, stride=2)
        self.enc7 = ConvBlock(ngf*8, ngf*8, stride=2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(ngf*8, ngf*8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.ConvTranspose2d(ngf*8, ngf*8, kernel_size=4, stride=2, 
                              padding=1, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Decoder (upsampling) with skip connections
        self.dec7 = nn.Sequential(
            nn.ConvTranspose2d(ngf*16, ngf*8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        self.dec6 = nn.Sequential(
            nn.ConvTranspose2d(ngf*16, ngf*8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(ngf*16, ngf*8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(ngf*16, ngf*8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(inplace=True)
        )
        
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(ngf*16, ngf*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(inplace=True)
        )
        
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(ngf*16, ngf*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(inplace=True)
        )
        
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(ngf*4, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True)
        )
        
        # Output layer
        self.output = nn.Sequential(
            nn.ConvTranspose2d(ngf*2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        # Encoder with skip connections
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        
        # Bottleneck
        bottleneck = self.bottleneck(e7)
        
        # Decoder with skip connections
        d7 = self.dec7(torch.cat([bottleneck, e7], dim=1))
        d6 = self.dec6(torch.cat([d7, e6], dim=1))
        d5 = self.dec5(torch.cat([d6, e5], dim=1))
        d4 = self.dec4(torch.cat([d5, e4], dim=1))
        d3 = self.dec3(torch.cat([d4, e3], dim=1))
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))
        
        # Output
        return self.output(torch.cat([d1, x], dim=1))


class Discriminator(nn.Module):
    """PatchGAN Discriminator"""
    
    def __init__(self, in_channels=3, ndf=64):
        super().__init__()
        
        self.model = nn.Sequential(
            ConvBlock(in_channels * 2, ndf, stride=2, use_batchnorm=False),
            ConvBlock(ndf, ndf * 2, stride=2),
            ConvBlock(ndf * 2, ndf * 4, stride=2),
            ConvBlock(ndf * 4, ndf * 8, stride=1),
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1)
        )
    
    def forward(self, x, y):
        # Concatenate input and target images
        x_y = torch.cat([x, y], dim=1)
        return self.model(x_y)


class Pix2PixLoss(nn.Module):
    """Combined GAN + L1 loss for Pix2Pix"""
    
    def __init__(self, lambda_l1=100.0):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()
    
    def forward(self, fake_pred, real_pred, generated, target):
        # GAN loss
        fake_loss = self.bce_loss(fake_pred, torch.zeros_like(fake_pred))
        real_loss = self.bce_loss(real_pred, torch.ones_like(real_pred))
        gan_loss = (fake_loss + real_loss) / 2
        
        # L1 loss
        l1_loss = self.l1_loss(generated, target) * self.lambda_l1
        
        # Generator loss
        gen_gan_loss = self.bce_loss(fake_pred, torch.ones_like(fake_pred))
        gen_loss = gen_gan_loss + l1_loss
        
        return gen_loss, gan_loss, l1_loss
