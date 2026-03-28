"""
Demo script for interactive image generation
Simple GUI and command-line interface for testing models
"""

import os
import torch
import argparse
from PIL import Image
import numpy as np
from pathlib import Path

from models.pix2pix import Generator as Pix2PixGenerator
from models.stylegan import Generator as StyleGANGenerator
from data.dataset import load_image, save_image
from config import Pix2PixConfig, StyleGANConfig


class DemoApp:
    """Interactive demo application for GAN models"""
    
    def __init__(self, pix2pix_checkpoint=None, stylegan_checkpoint=None):
        """Initialize demo with optional pre-loaded checkpoints"""
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        self.pix2pix_model = None
        self.stylegan_model = None
        
        # Load models if checkpoints provided
        if pix2pix_checkpoint:
            self.load_pix2pix(pix2pix_checkpoint)
        
        if stylegan_checkpoint:
            self.load_stylegan(stylegan_checkpoint)
    
    def load_pix2pix(self, checkpoint_path):
        """Load Pix2Pix model"""
        
        config = Pix2PixConfig()
        self.pix2pix_model = Pix2PixGenerator(
            in_channels=3, 
            out_channels=3, 
            ngf=config.ngf
        ).to(self.device)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.pix2pix_model.load_state_dict(checkpoint['generator_state_dict'])
        self.pix2pix_model.eval()
        
        print(f"✓ Pix2Pix model loaded: {checkpoint_path}")
    
    def load_stylegan(self, checkpoint_path):
        """Load StyleGAN model"""
        
        config = StyleGANConfig()
        self.stylegan_model = StyleGANGenerator(
            z_dim=config.z_dim,
            w_dim=config.w_dim,
            img_resolution=config.img_size,
            channel_multiplier=config.channel_multiplier,
            mapping_layers=config.mapping_layers
        ).to(self.device)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Use EMA weights if available
        if 'g_ema_shadow' in checkpoint:
            self.stylegan_model.load_state_dict(checkpoint['g_ema_shadow'])
        else:
            self.stylegan_model.load_state_dict(checkpoint['generator_state_dict'])
        
        self.stylegan_model.eval()
        
        print(f"✓ StyleGAN model loaded: {checkpoint_path}")
    
    @torch.no_grad()
    def cartoonize(self, input_path, output_path='outputs/cartoonized.png'):
        """Convert photo to cartoon using Pix2Pix"""
        
        if not self.pix2pix_model:
            raise RuntimeError("Pix2Pix model not loaded")
        
        # Load image
        input_tensor = load_image(input_path, img_size=Pix2PixConfig.img_size)
        input_tensor = input_tensor.to(self.device)
        
        # Generate cartoon
        output_tensor = self.pix2pix_model(input_tensor)
        
        # Save result
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        save_image(output_tensor, output_path, denormalize=True)
        
        print(f"✓ Cartoon saved to: {output_path}")
        return output_path
    
    @torch.no_grad()
    def generate_cartoons(self, num_images=8, output_dir='outputs/generated'):
        """Generate random cartoon faces using StyleGAN"""
        
        if not self.stylegan_model:
            raise RuntimeError("StyleGAN model not loaded")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Sample latent vectors
        z = torch.randn(num_images, StyleGANConfig.z_dim).to(self.device)
        
        # Generate images
        generated = self.stylegan_model(z)
        
        # Save individual images
        paths = []
        for i in range(generated.shape[0]):
            output_path = os.path.join(output_dir, f'cartoon_{i+1:03d}.png')
            save_image(generated[i:i+1], output_path, denormalize=True)
            paths.append(output_path)
            print(f"✓ Generated: {output_path}")
        
        # Save grid
        if num_images > 1:
            nrow = int(np.sqrt(num_images))
            from torchvision.utils import make_grid
            grid = make_grid(generated, nrow=nrow, padding=2)
            grid_path = os.path.join(output_dir, 'grid.png')
            save_image(grid.unsqueeze(0), grid_path, denormalize=True)
            print(f"✓ Grid saved to: {grid_path}")
        
        return paths
    
    @torch.no_grad()
    def interpolate(self, seed1=42, seed2=100, num_steps=10, 
                   output_dir='outputs/interpolation'):
        """Interpolate between two cartoon faces"""
        
        if not self.stylegan_model:
            raise RuntimeError("StyleGAN model not loaded")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Set seeds
        torch.manual_seed(seed1)
        z1 = torch.randn(1, StyleGANConfig.z_dim).to(self.device)
        
        torch.manual_seed(seed2)
        z2 = torch.randn(1, StyleGANConfig.z_dim).to(self.device)
        
        # Create interpolation
        alphas = torch.linspace(0, 1, num_steps).view(-1, 1).to(self.device)
        interpolated = z1 * (1 - alphas) + z2 * alphas
        
        # Generate images
        generated = self.stylegan_model(interpolated)
        
        # Save images
        for i in range(generated.shape[0]):
            output_path = os.path.join(output_dir, f'interp_{i+1:03d}.png')
            save_image(generated[i:i+1], output_path, denormalize=True)
        
        # Save grid
        from torchvision.utils import make_grid
        grid = make_grid(generated, nrow=num_steps, padding=2)
        grid_path = os.path.join(output_dir, 'interpolation_grid.png')
        save_image(grid.unsqueeze(0), grid_path, denormalize=True)
        
        print(f"✓ Interpolation saved to: {output_dir}")
        return output_dir


def main():
    parser = argparse.ArgumentParser(description='GAN Demo Application')
    parser.add_argument('--pix2pix', type=str, 
                       help='Path to Pix2Pix checkpoint')
    parser.add_argument('--stylegan', type=str,
                       help='Path to StyleGAN checkpoint')
    parser.add_argument('--mode', type=str, default='interactive',
                       choices=['interactive', 'cartoonize', 'generate', 'interpolate'],
                       help='Operation mode')
    parser.add_argument('--input', type=str,
                       help='Input image path (for cartoonize mode)')
    parser.add_argument('--output', type=str, default='outputs/result.png',
                       help='Output path')
    parser.add_argument('--num', type=int, default=8,
                       help='Number of images to generate')
    parser.add_argument('--seed1', type=int, default=42,
                       help='First seed for interpolation')
    parser.add_argument('--seed2', type=int, default=100,
                       help='Second seed for interpolation')
    
    args = parser.parse_args()
    
    # Initialize demo
    demo = DemoApp(
        pix2pix_checkpoint=args.pix2pix,
        stylegan_checkpoint=args.stylegan
    )
    
    # Run in specified mode
    if args.mode == 'cartoonize':
        if not args.input:
            print("Error: --input required for cartoonize mode")
            return
        demo.cartoonize(args.input, args.output)
    
    elif args.mode == 'generate':
        demo.generate_cartoons(num_images=args.num)
    
    elif args.mode == 'interpolate':
        demo.interpolate(seed1=args.seed1, seed2=args.seed2)
    
    elif args.mode == 'interactive':
        print("\n" + "="*50)
        print("Interactive GAN Demo")
        print("="*50)
        print("\nCommands:")
        print("  c <input> <output> - Cartoonize image")
        print("  g [num]           - Generate cartoon faces")
        print("  i [seed1] [seed2] - Interpolate between faces")
        print("  q                 - Quit")
        print("="*50 + "\n")
        
        while True:
            try:
                cmd = input("> ").strip().split()
                
                if not cmd:
                    continue
                
                if cmd[0] == 'q':
                    print("Goodbye!")
                    break
                
                elif cmd[0] == 'c':
                    if len(cmd) < 3:
                        print("Usage: c <input> <output>")
                        continue
                    demo.cartoonize(cmd[1], cmd[2])
                
                elif cmd[0] == 'g':
                    num = int(cmd[1]) if len(cmd) > 1 else 8
                    demo.generate_cartoons(num_images=num)
                
                elif cmd[0] == 'i':
                    seed1 = int(cmd[1]) if len(cmd) > 1 else 42
                    seed2 = int(cmd[2]) if len(cmd) > 2 else 100
                    demo.interpolate(seed1=seed1, seed2=seed2)
                
                else:
                    print("Unknown command. Use: c, g, i, or q")
            
            except Exception as e:
                print(f"Error: {e}")
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break


if __name__ == '__main__':
    main()
