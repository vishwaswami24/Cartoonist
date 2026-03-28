"""
StyleGAN Model Implementation
High-quality image generation with style-based architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PixelNorm(nn.Module):
    """Pixel-wise normalization"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)


class EqualizedLR(nn.Module):
    """Equalized Learning Rate"""
    
    def __init__(self, module, gain=math.sqrt(2)):
        super().__init__()
        self.module = module
        self.gain = gain
        
        # Initialize and save original weights
        fan_in = module.weight.data[0].numel()
        self.scale = gain / math.sqrt(fan_in)
    
    def forward(self, x):
        weight = self.module.weight * self.scale
        bias = self.module.bias if self.module.bias is not None else 0
        
        if isinstance(self.module, nn.Conv2d):
            return F.conv2d(x, weight, bias, self.module.stride, 
                           self.module.padding, self.module.dilation, 
                           self.module.groups)
        elif isinstance(self.module, nn.Linear):
            return F.linear(x, weight, bias)
        elif isinstance(self.module, nn.ConvTranspose2d):
            return F.conv_transpose2d(x, weight, bias, self.module.stride, 
                                     self.module.padding, self.module.output_padding, 
                                     self.module.groups, self.module.dilation)


class ModulatedConv2d(nn.Module):
    """Modulated convolution for StyleGAN"""
    
    def __init__(self, in_channels, out_channels, kernel_size, style_dim, 
                 demodulate=True, upsample=False, downsample=False):
        super().__init__()
        padding = kernel_size // 2
        
        self.weight = nn.Parameter(torch.randn(1, out_channels, in_channels, kernel_size, kernel_size))
        self.style_modulation = nn.Linear(style_dim, in_channels)
        
        self.demodulate = demodulate
        self.upsample = upsample
        self.downsample = downsample
        
        # Bias
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
    
    def forward(self, x, style):
        batch_size = x.shape[0]
        
        # Get modulation coefficients
        style = self.style_modulation(style).view(batch_size, -1, 1, 1, 1)
        weight = self.weight * style
        
        # Demodulation
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch_size, -1, 1, 1, 1)
        
        weight = weight.view(batch_size * self.weight.shape[1], 
                            self.weight.shape[2], 
                            self.weight.shape[3], 
                            self.weight.shape[4])
        
        x = x.view(1, batch_size * x.shape[1], x.shape[2], x.shape[3])
        
        # Upsample
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        # Convolution
        x = F.conv2d(x, weight, padding=self.weight.shape[-1]//2, groups=batch_size)
        
        # Downsample
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        
        x = x.view(batch_size, -1, x.shape[2], x.shape[3]) + self.bias
        
        return x


class NoiseInjection(nn.Module):
    """Noise injection layer"""
    
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))
    
    def forward(self, x):
        batch_size = x.shape[0]
        noise = torch.randn(batch_size, 1, x.shape[2], x.shape[3], device=x.device)
        return x + self.weight * noise


class StyleBlock(nn.Module):
    """Style block for StyleGAN generator"""
    
    def __init__(self, in_channels, out_channels, style_dim, upsample=True):
        super().__init__()
        
        self.conv1 = ModulatedConv2d(in_channels, out_channels, kernel_size=3, 
                                     style_dim=style_dim, upsample=upsample)
        self.noise1 = NoiseInjection(out_channels)
        self.activate1 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv2 = ModulatedConv2d(out_channels, out_channels, kernel_size=3, 
                                     style_dim=style_dim)
        self.noise2 = NoiseInjection(out_channels)
        self.activate2 = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x, style):
        x = self.conv1(x, style)
        x = self.noise1(x)
        x = self.activate1(x)
        
        x = self.conv2(x, style)
        x = self.noise2(x)
        x = self.activate2(x)
        
        return x


class MappingNetwork(nn.Module):
    """Mapping network from latent space to style space"""
    
    def __init__(self, z_dim, w_dim, num_layers=8):
        super().__init__()
        
        layers = []
        for i in range(num_layers):
            in_dim = z_dim if i == 0 else w_dim
            out_dim = w_dim
            
            linear = EqualizedLR(nn.Linear(in_dim, out_dim))
            layers.extend([linear, nn.LeakyReLU(0.2, inplace=True)])
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, z):
        return self.network(z)


class Generator(nn.Module):
    """StyleGAN Generator"""
    
    def __init__(self, z_dim=512, w_dim=512, img_resolution=256, 
                 channel_multiplier=1, mapping_layers=8):
        super().__init__()
        
        self.img_resolution = img_resolution
        self.channel_multiplier = channel_multiplier
        
        # Channel mapping
        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256,
            128: 128,
            256: 64,
            512: 32,
        }
        
        # Mapping network
        self.mapping = MappingNetwork(z_dim, w_dim, num_layers=mapping_layers)
        
        # Input layer (4x4)
        self.input_layer = nn.Parameter(torch.randn(1, channels[4], 4, 4))
        self.input_style = StyleBlock(channels[4], channels[4], w_dim, upsample=False)
        
        # Progressive layers
        resolutions = [4, 8, 16, 32, 64, 128, 256, 512]
        self.layers = nn.ModuleList()
        
        for res in resolutions[1:]:
            in_ch = channels[res // 2]
            out_ch = channels[res]
            
            layer = StyleBlock(in_ch, out_ch, w_dim, upsample=True)
            self.layers.append(layer)
        
        # Output layer
        self.to_rgb = nn.Sequential(
            nn.Conv2d(channels[img_resolution], 3, kernel_size=1),
            nn.Tanh()
        )
    
    def forward(self, z):
        # Map latent vector to style space
        style = self.mapping(z)
        
        # Start with input layer
        x = self.input_layer.expand(z.shape[0], -1, -1, -1)
        x = self.input_style(x, style)
        
        # Progress through layers
        for layer in self.layers:
            x = layer(x, style)
        
        # Generate RGB image
        rgb = self.to_rgb(x)
        
        return rgb


class Discriminator(nn.Module):
    """StyleGAN Discriminator"""
    
    def __init__(self, img_resolution=256, channel_multiplier=1):
        super().__init__()
        
        self.img_resolution = img_resolution
        
        # Channel mapping
        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256,
            128: 128,
            256: 64,
            512: 32,
        }
        
        resolutions = [512, 256, 128, 64, 32, 16, 8, 4]
        
        # Input layer
        self.from_rgb = nn.Conv2d(3, channels[img_resolution], kernel_size=1)
        
        # Discriminator blocks
        self.blocks = nn.ModuleList()
        for res in resolutions:
            in_ch = channels[res]
            out_ch = channels[res // 2] if res > 4 else channels[4]
            
            block = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.AvgPool2d(2) if res > 4 else nn.Identity()
            )
            self.blocks.append(block)
        
        # Output
        self.output = nn.Sequential(
            nn.Conv2d(channels[4], channels[4], kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(channels[4] * 16, 1)
        )
    
    def forward(self, x):
        x = self.from_rgb(x)
        
        for block in self.blocks:
            x = block(x)
        
        return self.output(x)
