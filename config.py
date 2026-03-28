"""
Configuration file for GAN models and training parameters
"""

class Config:
    """Base configuration class"""
    
    # General settings
    device = 'cuda'  # or 'cpu'
    random_seed = 42
    
    # Image settings
    img_size = 256
    channels = 3
    
    # Training settings
    batch_size = 8
    learning_rate = 0.0002
    epochs = 100
    
    # Model settings
    latent_dim = 100


class Pix2PixConfig(Config):
    """Configuration for Pix2Pix image-to-image translation"""
    
    # Model architecture
    ngf = 64  # Number of generator filters
    ndf = 64  # Number of discriminator filters
    
    # Loss weights
    lambda_l1 = 100.0  # L1 loss weight
    lambda_gan = 1.0   # GAN loss weight
    
    # Dataset path
    dataset_dir = 'datasets/pix2pix'
    save_dir = 'outputs/pix2pix'
    
    # Learning rate
    lr_decay_epoch = 50


class StyleGANConfig(Config):
    """Configuration for StyleGAN cartoon face generation"""
    
    # Model architecture
    z_dim = 512  # Latent vector dimension
    w_dim = 512  # Intermediate latent dimension
    mapping_layers = 8
    
    # Channel multiplier (controls model size)
    channel_multiplier = 1  # Use 1 for smaller model, 2 for original
    
    # Training settings
    r1_reg_weight = 10.0
    g_reg_interval = 4
    
    # Dataset path
    dataset_dir = 'datasets/cartoons'
    save_dir = 'outputs/stylegan'
    
    # EMA (Exponential Moving Average)
    ema_decay = 0.995


class CartoonizationConfig(Pix2PixConfig):
    """Configuration for photo-to-cartoon translation"""
    
    # Override specific settings for cartoonization
    img_size = 512  # Higher resolution for better results
    batch_size = 4  # Smaller batch due to larger images
    
    dataset_dir = 'datasets/cartoonization'
    save_dir = 'outputs/cartoonization'
