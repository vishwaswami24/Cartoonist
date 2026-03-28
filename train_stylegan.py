"""
Training script for StyleGAN model
High-quality cartoon face generation
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import numpy as np
from PIL import Image

from models.stylegan import Generator, Discriminator
from data.dataset import CartoonDataset, create_dataloader, save_image
from config import StyleGANConfig


class EMA:
    """Exponential Moving Average for model weights"""
    
    def __init__(self, model, decay=0.995):
        self.model = model
        self.decay = decay
        self.shadow = {key: value.clone() for key, value in model.state_dict().items()}
    
    def update(self):
        """Update shadow weights"""
        for key, value in self.model.state_dict().items():
            self.shadow[key] = self.shadow[key] * self.decay + value * (1 - self.decay)
    
    def apply_shadow(self):
        """Return a copy of the model with shadow weights"""
        backup = {key: value.clone() for key, value in self.model.state_dict().items()}
        self.model.load_state_dict(self.shadow)
        return backup
    
    def restore(self, backup):
        """Restore original weights"""
        self.model.load_state_dict(backup)


class StyleGANTrainer:
    """Trainer class for StyleGAN model"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.generator = Generator(
            z_dim=config.z_dim,
            w_dim=config.w_dim,
            img_resolution=config.img_size,
            channel_multiplier=config.channel_multiplier,
            mapping_layers=config.mapping_layers
        ).to(self.device)
        
        self.discriminator = Discriminator(
            img_resolution=config.img_size,
            channel_multiplier=config.channel_multiplier
        ).to(self.device)
        
        # EMA for generator
        self.g_ema = EMA(self.generator, decay=config.ema_decay)
        
        # Initialize optimizers
        self.optimizer_G = optim.Adam(
            self.generator.parameters(),
            lr=config.learning_rate,
            betas=(0.0, 0.99)
        )
        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(),
            lr=config.learning_rate,
            betas=(0.0, 0.99)
        )
        
        # TensorBoard writer
        self.writer = None
        
        # Create save directory
        os.makedirs(config.save_dir, exist_ok=True)
        
        # Fixed noise for consistent sampling
        self.fixed_noise = torch.randn(8, config.z_dim).to(self.device)
    
    def r1_regularity_loss(self, real_imgs, real_preds):
        """R1 gradient penalty regularization"""
        
        real_imgs.requires_grad_(True)
        real_preds = self.discriminator(real_imgs)
        
        gradients = torch.autograd.grad(
            outputs=[real_preds.sum()],
            inputs=[real_imgs],
            create_graph=True,
            only_inputs=True
        )[0]
        
        gradients = gradients.view(gradients.size(0), -1)
        r1_penalty = gradients.pow(2).sum(1)
        
        return r1_penalty.mean() * self.config.r1_reg_weight / 2
    
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch"""
        
        self.generator.train()
        self.discriminator.train()
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')
        
        total_g_loss = 0
        total_d_loss = 0
        
        for batch_idx, batch in enumerate(progress_bar):
            # Get real images
            real_imgs = batch['image'].to(self.device)
            batch_size = real_imgs.shape[0]
            
            # Sample latent vectors
            z = torch.randn(batch_size, self.config.z_dim).to(self.device)
            
            # --- Train Discriminator ---
            self.optimizer_D.zero_grad()
            
            # Generate fake images
            with torch.no_grad():
                fake_imgs = self.generator(z)
            
            # Discriminator predictions
            real_preds = self.discriminator(real_imgs)
            fake_preds = self.discriminator(fake_imgs.detach())
            
            # Discriminator loss
            d_loss_real = -torch.mean(real_preds)
            d_loss_fake = -torch.mean(fake_preds)
            d_loss = d_loss_real + d_loss_fake
            
            # R1 regularization
            r1_loss = self.r1_regularity_loss(real_imgs, real_preds)
            d_loss = d_loss + r1_loss
            
            d_loss.backward()
            self.optimizer_D.step()
            
            # --- Train Generator ---
            self.optimizer_G.zero_grad()
            
            # Generate fake images
            z = torch.randn(batch_size, self.config.z_dim).to(self.device)
            fake_imgs = self.generator(z)
            
            # Discriminator prediction
            fake_preds = self.discriminator(fake_imgs)
            
            # Generator loss
            g_loss = -torch.mean(fake_preds)
            
            g_loss.backward()
            self.optimizer_G.step()
            
            # Update EMA
            self.g_ema.update()
            
            # Update statistics
            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'G_loss': f'{g_loss.item():.4f}',
                'D_loss': f'{d_loss.item():.4f}'
            })
            
            # Log to TensorBoard
            if self.writer and batch_idx % 10 == 0:
                step = epoch * len(dataloader) + batch_idx
                self.writer.add_scalar('Loss/G_total', g_loss.item(), step)
                self.writer.add_scalar('Loss/D_total', d_loss.item(), step)
                self.writer.add_scalar('Loss/D_real', -d_loss_real.item(), step)
                self.writer.add_scalar('Loss/D_fake', -d_loss_fake.item(), step)
        
        # Return average losses
        return {
            'g_loss': total_g_loss / len(dataloader),
            'd_loss': total_d_loss / len(dataloader)
        }
    
    @torch.no_grad()
    def generate_samples(self, epoch):
        """Generate sample images"""
        
        # Use EMA weights
        backup = self.g_ema.apply_shadow()
        
        self.generator.eval()
        
        # Generate images
        fake_imgs = self.generator(self.fixed_noise)
        
        # Denormalize
        samples = (fake_imgs + 1) / 2
        samples = torch.clamp(samples, 0, 1)
        
        # Save individual images
        for i in range(samples.shape[0]):
            save_path = os.path.join(self.config.save_dir, f'sample_{epoch:04d}_{i}.png')
            save_image(samples[i:i+1], save_path, denormalize=False)
        
        # Save grid
        nrow = int(np.sqrt(samples.shape[0]))
        from torchvision.utils import make_grid
        grid = make_grid(samples, nrow=nrow, padding=2)
        grid_path = os.path.join(self.config.save_dir, f'grid_{epoch:04d}.png')
        save_image(grid.unsqueeze(0), grid_path, denormalize=False)
        
        # Restore original weights
        self.g_ema.restore(backup)
    
    def save_checkpoint(self, epoch, filename='checkpoint.pth'):
        """Save model checkpoint"""
        
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_ema_shadow': self.g_ema.shadow,
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
        }
        
        save_path = os.path.join(self.config.save_dir, filename)
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved: {save_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.g_ema.shadow = checkpoint['g_ema_shadow']
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        
        start_epoch = checkpoint['epoch']
        print(f"Checkpoint loaded from epoch {start_epoch}")
        
        return start_epoch
    
    def train(self, dataset, num_epochs=100):
        """Full training loop"""
        
        # Create dataloader
        dataloader = create_dataloader(
            dataset,
            batch_size=self.config.batch_size,
            num_workers=4
        )
        
        # Initialize TensorBoard
        log_dir = os.path.join(self.config.save_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(dataset)}")
        
        start_epoch = 0
        
        # Training loop
        for epoch in range(start_epoch, num_epochs):
            # Train
            train_losses = self.train_epoch(dataloader, epoch)
            
            # Generate samples
            if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
                self.generate_samples(epoch)
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
                self.save_checkpoint(epoch + 1)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{num_epochs}:")
            print(f"  G Loss: {train_losses['g_loss']:.4f}")
            print(f"  D Loss: {train_losses['d_loss']:.4f}")
        
        # Save final model with EMA weights
        self.save_checkpoint(num_epochs, 'final_model.pth')
        
        # Close TensorBoard writer
        if self.writer:
            self.writer.close()
        
        print("\nTraining completed!")


def main():
    parser = argparse.ArgumentParser(description='Train StyleGAN model')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to cartoon images directory')
    args = parser.parse_args()
    
    # Initialize configuration
    config = StyleGANConfig()
    
    # Override with dataset path
    config.dataset_dir = args.dataset
    
    # Initialize trainer
    trainer = StyleGANTrainer(config)
    
    # Create dataset
    dataset = CartoonDataset(
        config.dataset_dir,
        img_size=config.img_size,
        augment=True
    )
    
    # Start training
    trainer.train(dataset, num_epochs=config.epochs)


if __name__ == '__main__':
    main()
