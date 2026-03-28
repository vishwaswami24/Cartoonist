"""
Data loading utilities for GAN training
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms


class PairedImageDataset(Dataset):
    """Dataset for paired image-to-image translation (Pix2Pix)"""
    
    def __init__(self, root_dir, mode='train', img_size=256):
        """
        Args:
            root_dir: Path to dataset directory with 'A' and 'B' subfolders
            mode: 'train' or 'test'
            img_size: Size to resize images
        """
        self.root_dir = root_dir
        self.mode = mode
        self.img_size = img_size
        
        # Directory structure: root_dir/A (input), root_dir/B (target)
        self.dir_A = os.path.join(root_dir, 'A', mode)
        self.dir_B = os.path.join(root_dir, 'B', mode)
        
        # Get list of images
        self.A_paths = sorted([os.path.join(self.dir_A, f) 
                               for f in os.listdir(self.dir_A) 
                               if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.B_paths = sorted([os.path.join(self.dir_B, f) 
                               for f in os.listdir(self.dir_B) 
                               if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        assert len(self.A_paths) == len(self.B_paths), \
            "Number of images in A and B must match"
        
        # Transforms
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
    
    def __len__(self):
        return len(self.A_paths)
    
    def __getitem__(self, idx):
        # Load images
        A_img = Image.open(self.A_paths[idx]).convert('RGB')
        B_img = Image.open(self.B_paths[idx]).convert('RGB')
        
        # Apply transforms
        A = self.transform(A_img)
        B = self.transform(B_img)
        
        return {'A': A, 'B': B, 'A_path': self.A_paths[idx], 'B_path': self.B_paths[idx]}


class UnpairedImageDataset(Dataset):
    """Dataset for unpaired image datasets (for StyleGAN)"""
    
    def __init__(self, root_dir, img_size=256):
        """
        Args:
            root_dir: Path to directory containing images
            img_size: Size to resize images
        """
        self.root_dir = root_dir
        self.img_size = img_size
        
        # Get list of images
        self.image_paths = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            import glob
            self.image_paths.extend(glob.glob(os.path.join(root_dir, ext)))
        
        self.image_paths = sorted(self.image_paths)
        
        # Transforms
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img = Image.open(self.image_paths[idx]).convert('RGB')
        
        # Apply transforms
        img = self.transform(img)
        
        return {'image': img, 'path': self.image_paths[idx]}


class CartoonDataset(Dataset):
    """Specialized dataset for cartoon face generation"""
    
    def __init__(self, root_dir, img_size=256, augment=True):
        """
        Args:
            root_dir: Path to cartoon images directory
            img_size: Output image size
            augment: Whether to apply data augmentation
        """
        self.root_dir = root_dir
        self.img_size = img_size
        self.augment = augment
        
        # Get all image paths
        self.image_paths = []
        for subdir in os.listdir(root_dir):
            subdir_path = os.path.join(root_dir, subdir)
            if os.path.isdir(subdir_path):
                for f in os.listdir(subdir_path):
                    if f.endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(subdir_path, f))
        
        # If no subdirs, load from root
        if not self.image_paths:
            for f in os.listdir(root_dir):
                if f.endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(root_dir, f))
        
        # Transforms
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        
        return {'image': img, 'path': img_path}


def create_dataloader(dataset, batch_size=8, num_workers=4, shuffle=True):
    """Create DataLoader from dataset"""
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader


def load_image(path, img_size=256):
    """Load a single image for inference"""
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    img = Image.open(path).convert('RGB')
    img = transform(img)
    
    return img.unsqueeze(0)  # Add batch dimension


def save_image(tensor, path, denormalize=True):
    """Save tensor as image"""
    
    if denormalize:
        # Denormalize from [-1, 1] to [0, 1]
        tensor = (tensor + 1) / 2
        tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to PIL Image
    img = transforms.ToPILImage()(tensor.squeeze(0).cpu())
    img.save(path)
