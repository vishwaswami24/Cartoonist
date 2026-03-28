"""
Training script for Pix2Pix model
Image-to-image translation training loop
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

from models.pix2pix import Generator, Discriminator, Pix2PixLoss
from data.dataset import PairedImageDataset, create_dataloader, save_image
from config import Pix2PixConfig


class Pix2PixTrainer:
    """Trainer class for Pix2Pix model"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.generator = Generator(in_channels=3, out_channels=3, ngf=config.ngf).to(self.device)
        self.discriminator = Discriminator(in_channels=3, ndf=config.ndf).to(self.device)
        
        # Initialize loss
        self.criterion = Pix2PixLoss(lambda_l1=config.lambda_l1)
        
        # Initialize optimizers
        self.optimizer_G = optim.Adam(
            self.generator.parameters(), 
            lr=config.learning_rate, 
            betas=(0.5, 0.999)
        )
        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(), 
            lr=config.learning_rate, 
            betas=(0.5, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler_G = optim.lr_scheduler.LambdaLR(
            self.optimizer_G, 
            lr_lambda=lambda epoch: max(0, 1 - (epoch - config.lr_decay_epoch) / config.lr_decay_epoch)
        )
        self.scheduler_D = optim.lr_scheduler.LambdaLR(
            self.optimizer_D, 
            lr_lambda=lambda epoch: max(0, 1 - (epoch - config.lr_decay_epoch) / config.lr_decay_epoch)
        )
        
        # TensorBoard writer
        self.writer = None
        
        # Create save directory
        os.makedirs(config.save_dir, exist_ok=True)
    
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch"""
        
        self.generator.train()
        self.discriminator.train()
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')
        
        total_g_loss = 0
        total_d_loss = 0
        total_l1_loss = 0
        
        for batch_idx, batch in enumerate(progress_bar):
            # Get data
            real_A = batch['A'].to(self.device)
            real_B = batch['B'].to(self.device)
            
            # --- Train Discriminator ---
            self.optimizer_D.zero_grad()
            
            # Generate fake images
            fake_B = self.generator(real_A)
            
            # Discriminator predictions
            pred_real = self.discriminator(real_A, real_B)
            pred_fake = self.discriminator(real_A, fake_B.detach())
            
            # Calculate discriminator loss
            _, d_loss, _ = self.criterion(pred_fake, pred_real, fake_B, real_B)
            d_loss.backward()
            self.optimizer_D.step()
            
            # --- Train Generator ---
            self.optimizer_G.zero_grad()
            
            # Generate fake images
            fake_B = self.generator(real_A)
            
            # Discriminator prediction
            pred_fake = self.discriminator(real_A, fake_B)
            
            # Calculate generator loss
            g_loss, _, l1_loss = self.criterion(pred_fake, pred_real, fake_B, real_B)
            g_loss.backward()
            self.optimizer_G.step()
            
            # Update statistics
            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()
            total_l1_loss += l1_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'G_loss': f'{g_loss.item():.4f}',
                'D_loss': f'{d_loss.item():.4f}',
                'L1_loss': f'{l1_loss.item():.4f}'
            })
            
            # Log to TensorBoard
            if self.writer and batch_idx % 10 == 0:
                step = epoch * len(dataloader) + batch_idx
                self.writer.add_scalar('Loss/G_total', g_loss.item(), step)
                self.writer.add_scalar('Loss/D_total', d_loss.item(), step)
                self.writer.add_scalar('Loss/L1', l1_loss.item(), step)
        
        # Update learning rates
        self.scheduler_G.step()
        self.scheduler_D.step()
        
        # Return average losses
        return {
            'g_loss': total_g_loss / len(dataloader),
            'd_loss': total_d_loss / len(dataloader),
            'l1_loss': total_l1_loss / len(dataloader)
        }
    
    @torch.no_grad()
    def validate(self, dataloader, epoch):
        """Validate on test set"""
        
        self.generator.eval()
        
        # Save some sample outputs
        batch = next(iter(dataloader))
        real_A = batch['A'].to(self.device)
        real_B = batch['B'].to(self.device)
        
        fake_B = self.generator(real_A)
        
        # Create comparison image
        comparison = torch.cat([real_A, fake_B, real_B], dim=0)
        comparison = (comparison + 1) / 2  # Denormalize
        comparison = torch.clamp(comparison, 0, 1)
        
        # Save image
        save_path = os.path.join(self.config.save_dir, f'epoch_{epoch:04d}.png')
        save_image(comparison, save_path, denormalize=False)
        
        if self.writer:
            self.writer.add_image('Validation/Comparison', comparison, epoch)
    
    def save_checkpoint(self, epoch, filename='checkpoint.pth'):
        """Save model checkpoint"""
        
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'scheduler_G_state_dict': self.scheduler_G.state_dict(),
            'scheduler_D_state_dict': self.scheduler_D.state_dict(),
        }
        
        save_path = os.path.join(self.config.save_dir, filename)
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved: {save_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        self.scheduler_G.load_state_dict(checkpoint['scheduler_G_state_dict'])
        self.scheduler_D.load_state_dict(checkpoint['scheduler_D_state_dict'])
        
        start_epoch = checkpoint['epoch']
        print(f"Checkpoint loaded from epoch {start_epoch}")
        
        return start_epoch
    
    def train(self, train_dataset, val_dataset=None, num_epochs=100):
        """Full training loop"""
        
        # Create dataloaders
        train_loader = create_dataloader(
            train_dataset, 
            batch_size=self.config.batch_size,
            num_workers=4
        )
        
        val_loader = None
        if val_dataset:
            val_loader = create_dataloader(
                val_dataset, 
                batch_size=self.config.batch_size,
                shuffle=False
            )
        
        # Initialize TensorBoard
        log_dir = os.path.join(self.config.save_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(train_dataset)}")
        if val_dataset:
            print(f"Validation samples: {len(val_dataset)}")
        
        start_epoch = 0
        
        # Training loop
        for epoch in range(start_epoch, num_epochs):
            # Train
            train_losses = self.train_epoch(train_loader, epoch)
            
            # Validate
            if val_loader and (epoch % 5 == 0 or epoch == num_epochs - 1):
                self.validate(val_loader, epoch)
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
                self.save_checkpoint(epoch + 1)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{num_epochs}:")
            print(f"  G Loss: {train_losses['g_loss']:.4f}")
            print(f"  D Loss: {train_losses['d_loss']:.4f}")
            print(f"  L1 Loss: {train_losses['l1_loss']:.4f}")
        
        # Save final model
        self.save_checkpoint(num_epochs, 'final_model.pth')
        
        # Close TensorBoard writer
        if self.writer:
            self.writer.close()
        
        print("\nTraining completed!")


def main():
    parser = argparse.ArgumentParser(description='Train Pix2Pix model')
    parser.add_argument('--dataset', type=str, required=True, 
                       help='Path to dataset directory')
    parser.add_argument('--config', type=str, default='default',
                       help='Configuration to use')
    args = parser.parse_args()
    
    # Initialize configuration
    config = Pix2PixConfig()
    
    # Override with dataset path
    config.dataset_dir = args.dataset
    
    # Initialize trainer
    trainer = Pix2PixTrainer(config)
    
    # Create datasets
    train_dataset = PairedImageDataset(
        config.dataset_dir, 
        mode='train', 
        img_size=config.img_size
    )
    
    val_dataset = PairedImageDataset(
        config.dataset_dir, 
        mode='test', 
        img_size=config.img_size
    )
    
    # Start training
    trainer.train(
        train_dataset,
        val_dataset,
        num_epochs=config.epochs
    )


if __name__ == '__main__':
    main()
