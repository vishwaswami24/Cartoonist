"""
Example: Batch cartoonization of images
This script demonstrates how to process multiple images at once
"""

import os
from pathlib import Path
from demo import DemoApp
import argparse


def batch_cartoonize(input_folder, output_folder, model_path):
    """
    Cartoonize all images in a folder
    
    Args:
        input_folder: Folder containing input photos
        output_folder: Folder to save cartoonized images
        model_path: Path to trained Pix2Pix model
    """
    
    # Initialize demo with model
    print(f"Loading model from: {model_path}")
    demo = DemoApp(pix2pix_checkpoint=model_path)
    
    # Create output folder
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Supported image formats
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    # Get all images
    image_files = [f for f in os.listdir(input_folder) 
                   if any(f.lower().endswith(ext) for ext in image_extensions)]
    
    print(f"\nFound {len(image_files)} images to process\n")
    
    # Process each image
    for i, filename in enumerate(image_files, 1):
        input_path = os.path.join(input_folder, filename)
        output_name = f"cartoon_{filename}"
        output_path = os.path.join(output_folder, output_name)
        
        print(f"[{i}/{len(image_files)}] Processing: {filename}")
        
        try:
            demo.cartoonize(input_path, output_path)
            print(f"  ✓ Saved: {output_path}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print(f"\n✅ Batch processing complete!")
    print(f"Output folder: {output_folder}")


def main():
    parser = argparse.ArgumentParser(description='Batch Cartoonize Images')
    parser.add_argument('--input', type=str, required=True,
                       help='Input folder with photos')
    parser.add_argument('--output', type=str, required=True,
                       help='Output folder for cartoons')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained Pix2Pix model')
    
    args = parser.parse_args()
    
    batch_cartoonize(args.input, args.output, args.model)


if __name__ == '__main__':
    main()
