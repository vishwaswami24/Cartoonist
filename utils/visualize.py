"""
Visualize model architecture and training progress
"""

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path


def visualize_pix2pix_architecture():
    """Create a diagram of Pix2Pix architecture"""
    
    print("Pix2Pix Architecture:")
    print("=" * 60)
    print("\nGenerator (U-Net):")
    print("""
    Input (256x256)
         ↓
    Encoder Block 1 (256x256, 64 filters)
         ↓
    Encoder Block 2 (128x128, 128 filters)
         ↓
    Encoder Block 3 (64x64, 256 filters)
         ↓
    Encoder Block 4 (32x32, 512 filters)
         ↓
    Encoder Block 5 (16x16, 512 filters)
         ↓
    Encoder Block 6 (8x8, 512 filters)
         ↓
    Encoder Block 7 (4x4, 512 filters)
         ↓
    Bottleneck (2x2, 512 filters)
         ↓
    Decoder Block 7 ← skip connection from Enc 7
         ↓
    Decoder Block 6 ← skip connection from Enc 6
         ↓
    Decoder Block 5 ← skip connection from Enc 5
         ↓
    Decoder Block 4 ← skip connection from Enc 4
         ↓
    Decoder Block 3 ← skip connection from Enc 3
         ↓
    Decoder Block 2 ← skip connection from Enc 2
         ↓
    Decoder Block 1 ← skip connection from Enc 1
         ↓
    Output (256x256, RGB)
    """)
    
    print("\nDiscriminator (PatchGAN):")
    print("""
    Input: Concatenated (Image A + Image B)
         ↓
    Conv Block 1 (256x256, 64 filters)
         ↓
    Conv Block 2 (128x128, 128 filters)
         ↓
    Conv Block 3 (64x64, 256 filters)
         ↓
    Conv Block 4 (32x32, 512 filters)
         ↓
    Output Patch (30x30, 1 channel)
    
    Each patch predicts: "Is this real or fake?"
    """)
    
    print("\nKey Features:")
    print("• Skip connections preserve spatial information")
    print("• PatchGAN discriminates at patch level")
    print("• L1 loss ensures pixel-level accuracy")
    print("• GAN loss ensures realistic outputs")
    print("=" * 60)


def visualize_stylegan_architecture():
    """Create a diagram of StyleGAN architecture"""
    
    print("\nStyleGAN Architecture:")
    print("=" * 60)
    print("\nGenerator:")
    print("""
    Latent Vector z (512 dim)
         ↓
    Mapping Network (8 layers)
         ↓
    Intermediate Latent w (512 dim)
         ↓
    ┌────────────────────────────┐
    │ 4x4 Constant Input          │
    │ + Style Modulation          │
    │ + Noise Injection           │
    └────────────────────────────┘
         ↓
    ┌────────────────────────────┐
    │ Style Block (4x4 → 8x8)    │
    │ - Modulated Convolution     │
    │ - Noise Injection           │
    │ - Activation                │
    └────────────────────────────┘
         ↓
    ┌────────────────────────────┐
    │ Style Block (8x8 → 16x16)  │
    └────────────────────────────┘
         ↓
    ... (progressive upsampling)
         ↓
    ┌────────────────────────────┐
    │ Style Block (128x128→256x256)│
    └────────────────────────────┘
         ↓
    To RGB Layer
         ↓
    Output Image (256x256)
    """)
    
    print("\nDiscriminator:")
    print("""
    Input Image (256x256)
         ↓
    From RGB Layer
         ↓
    Discriminator Block (256x256)
         ↓
    Discriminator Block (128x128)
         ↓
    Discriminator Block (64x64)
         ↓
    ... (progressive downsampling)
         ↓
    Discriminator Block (4x4)
         ↓
    Output Layer
         ↓
    Real/Fake Score (scalar)
    """)
    
    print("\nKey Features:")
    print("• Mapping network transforms latent space")
    print("• Style modulation controls features at each scale")
    print("• Noise injection adds stochastic variation")
    print("• Progressive growing for stable training")
    print("• EMA weights for smoother generation")
    print("=" * 60)


def plot_training_progress(log_dir, save_path='training_progress.png'):
    """Plot training progress from saved checkpoints"""
    
    log_path = Path(log_dir)
    
    if not log_path.exists():
        print(f"Log directory not found: {log_dir}")
        return
    
    # Find all epoch images
    epoch_images = sorted(log_path.glob('epoch_*.png'))
    
    if not epoch_images:
        print("No training progress images found")
        return
    
    print(f"\nFound {len(epoch_images)} training checkpoints")
    
    # Create figure
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    # Show samples from different epochs
    num_show = min(8, len(epoch_images))
    step = max(1, len(epoch_images) // num_show)
    
    for i, idx in enumerate(range(0, len(epoch_images), step)):
        if i >= 8:
            break
        
        img_path = epoch_images[idx]
        img = Image.open(img_path)
        
        axes[i].imshow(img)
        axes[i].set_title(f'Epoch {img_path.stem}')
        axes[i].axis('off')
    
    # Fill remaining axes with black
    for i in range(num_show, 8):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Training progress saved to: {save_path}")
    plt.show()


def visualize_latent_space_interpolation(model_path, output_dir='outputs/visualizations'):
    """Visualize latent space interpolation"""
    
    from models.stylegan import Generator
    from config import StyleGANConfig
    from data.dataset import save_image
    from torchvision.utils import make_grid
    
    print("\nLoading StyleGAN model...")
    config = StyleGANConfig()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Generator(
        z_dim=config.z_dim,
        w_dim=config.w_dim,
        img_resolution=config.img_size,
        channel_multiplier=config.channel_multiplier,
        mapping_layers=config.mapping_layers
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'g_ema_shadow' in checkpoint:
        model.load_state_dict(checkpoint['g_ema_shadow'])
    else:
        model.load_state_dict(checkpoint['generator_state_dict'])
    
    model.eval()
    
    print("Generating interpolation sequence...")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate interpolation between multiple pairs
    num_pairs = 4
    steps_per_pair = 5
    
    with torch.no_grad():
        for pair_idx in range(num_pairs):
            # Sample two random latent vectors
            z1 = torch.randn(1, config.z_dim).to(device)
            z2 = torch.randn(1, config.z_dim).to(device)
            
            # Interpolate
            alphas = torch.linspace(0, 1, steps_per_pair).view(-1, 1, 1).to(device)
            interpolated = z1 * (1 - alphas) + z2 * alphas
            
            # Generate
            generated = model(interpolated.squeeze(1))
            
            # Save grid
            grid = make_grid(generated, nrow=steps_per_pair, padding=2)
            save_image(grid.unsqueeze(0), 
                      f'{output_dir}/interpolation_pair_{pair_idx+1}.png',
                      denormalize=True)
            
            print(f"  Generated interpolation pair {pair_idx+1}/{num_pairs}")
    
    print(f"\nVisualizations saved to: {output_dir}")


def main():
    """Main visualization function"""
    
    print("=" * 60)
    print("GAN Model Visualization")
    print("=" * 60)
    
    # Print architecture diagrams
    visualize_pix2pix_architecture()
    visualize_stylegan_architecture()
    
    # Try to plot training progress if outputs exist
    print("\nChecking for training outputs...")
    
    outputs_dir = Path('outputs')
    if outputs_dir.exists():
        # Check for pix2pix outputs
        pix2pix_dir = outputs_dir / 'pix2pix'
        if pix2pix_dir.exists():
            plot_training_progress(pix2pix_dir, 'outputs/pix2pix_progress.png')
        
        # Check for stylegan outputs
        stylegan_dir = outputs_dir / 'stylegan'
        if stylegan_dir.exists():
            plot_training_progress(stylegan_dir, 'outputs/stylegan_progress.png')
    else:
        print("No outputs directory found. Run training first to see progress.")
    
    print("\n" + "=" * 60)
    print("Visualization complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
