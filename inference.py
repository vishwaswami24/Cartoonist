"""
Inference script for trained GAN models
Generate images using pre-trained models
"""

import os
import torch
import argparse
from PIL import Image
from torchvision import transforms
import numpy as np

from models.pix2pix import Generator as Pix2PixGenerator
from models.stylegan import Generator as StyleGANGenerator
from data.dataset import load_image, save_image
from config import Pix2PixConfig, StyleGANConfig


class Inference:
    """Inference class for GAN models"""
    
    def __init__(self, model_type='pix2pix', checkpoint_path=None, device=None):
        """
        Args:
            model_type: 'pix2pix' or 'stylegan'
            checkpoint_path: Path to model checkpoint
            device: Device to use (cuda/cpu)
        """
        self.model_type = model_type
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load configuration
        if model_type == 'pix2pix':
            self.config = Pix2PixConfig()
            self.model = Pix2PixGenerator(in_channels=3, out_channels=3, ngf=self.config.ngf)
        elif model_type == 'stylegan':
            self.config = StyleGANConfig()
            self.model = StyleGANGenerator(
                z_dim=self.config.z_dim,
                w_dim=self.config.w_dim,
                img_resolution=self.config.img_size,
                channel_multiplier=self.config.channel_multiplier,
                mapping_layers=self.config.mapping_layers
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model.to(self.device)
        
        # Load checkpoint
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        
        self.model.eval()
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if self.model_type == 'pix2pix':
            self.model.load_state_dict(checkpoint['generator_state_dict'])
        elif self.model_type == 'stylegan':
            # Try to load EMA weights first
            if 'g_ema_shadow' in checkpoint:
                self.model.load_state_dict(checkpoint['g_ema_shadow'])
            else:
                self.model.load_state_dict(checkpoint['generator_state_dict'])
        
        print(f"Loaded checkpoint from: {checkpoint_path}")
    
    @torch.no_grad()
    def translate_image(self, input_path, output_path):
        """
        Translate image using Pix2Pix model
        
        Args:
            input_path: Path to input image
            output_path: Path to save output image
        """
        if self.model_type != 'pix2pix':
            raise ValueError("This method is only available for pix2pix model")
        
        # Load and preprocess image
        input_tensor = load_image(input_path, img_size=self.config.img_size)
        input_tensor = input_tensor.to(self.device)
        
        # Generate translation
        output_tensor = self.model(input_tensor)
        
        # Save result
        save_image(output_tensor, output_path, denormalize=True)
        print(f"Translation saved to: {output_path}")
    
    @torch.no_grad()
    def generate_cartoon(self, num_images=8, output_dir='outputs/generated'):
        """
        Generate cartoon faces using StyleGAN
        
        Args:
            num_images: Number of images to generate
            output_dir: Directory to save generated images
        """
        if self.model_type != 'stylegan':
            raise ValueError("This method is only available for stylegan model")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Sample latent vectors
        z = torch.randn(num_images, self.config.z_dim).to(self.device)
        
        # Generate images
        generated = self.model(z)
        
        # Save individual images
        for i in range(generated.shape[0]):
            output_path = os.path.join(output_dir, f'cartoon_{i+1:03d}.png')
            save_image(generated[i:i+1], output_path, denormalize=True)
            print(f"Generated: {output_path}")
        
        # Save grid
        if num_images > 1:
            nrow = int(np.sqrt(num_images))
            from torchvision.utils import make_grid
            grid = make_grid(generated, nrow=nrow, padding=2)
            grid_path = os.path.join(output_dir, 'grid.png')
            save_image(grid.unsqueeze(0), grid_path, denormalize=True)
            print(f"Grid saved to: {grid_path}")
    
    @torch.no_grad()
    def interpolate_latent(self, z1, z2, num_steps=10, output_dir='outputs/interpolation'):
        """
        Interpolate between two latent vectors
        
        Args:
            z1: First latent vector
            z2: Second latent vector
            num_steps: Number of interpolation steps
            output_dir: Directory to save results
        """
        if self.model_type != 'stylegan':
            raise ValueError("This method is only available for stylegan model")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create interpolation
        alphas = torch.linspace(0, 1, num_steps).view(-1, 1).to(self.device)
        interpolated = z1 * (1 - alphas) + z2 * alphas
        
        # Generate images
        generated = self.model(interpolated)
        
        # Save images
        for i in range(generated.shape[0]):
            output_path = os.path.join(output_dir, f'interp_{i+1:03d}.png')
            save_image(generated[i:i+1], output_path, denormalize=True)
        
        # Save grid
        from torchvision.utils import make_grid
        grid = make_grid(generated, nrow=num_steps, padding=2)
        grid_path = os.path.join(output_dir, 'interpolation_grid.png')
        save_image(grid.unsqueeze(0), grid_path, denormalize=True)
        
        print(f"Interpolation saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='GAN Model Inference')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--type', type=str, default='pix2pix',
                       choices=['pix2pix', 'stylegan'],
                       help='Model type')
    parser.add_argument('--input', type=str,
                       help='Input image path (for pix2pix)')
    parser.add_argument('--output', type=str, default='outputs/result.png',
                       help='Output image path')
    parser.add_argument('--generate', type=int, default=0,
                       help='Number of images to generate (for stylegan)')
    args = parser.parse_args()
    
    # Initialize inference
    inference = Inference(
        model_type=args.type,
        checkpoint_path=args.model
    )
    
    # Run inference
    if args.type == 'pix2pix' and args.input:
        inference.translate_image(args.input, args.output)
    elif args.type == 'stylegan' and args.generate > 0:
        inference.generate_cartoon(num_images=args.generate)
    else:
        print("Please specify --input for pix2pix or --generate for stylegan")


if __name__ == '__main__':
    main()
