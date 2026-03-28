"""
Utility scripts for dataset preparation
"""

import os
import argparse
from PIL import Image
from pathlib import Path
import shutil


def resize_images(input_dir, output_dir, size=256):
    """Resize all images in directory to specified size"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    image_extensions = ['.jpg', '.jpeg', '.png']
    count = 0
    
    for filename in os.listdir(input_dir):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            try:
                img_path = os.path.join(input_dir, filename)
                img = Image.open(img_path)
                
                # Resize with high-quality resampling
                img_resized = img.resize((size, size), Image.Resampling.LANCZOS)
                
                # Save
                output_path = os.path.join(output_dir, filename)
                img_resized.save(output_path)
                
                count += 1
                print(f"Resized: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    print(f"\nTotal images processed: {count}")


def create_paired_dataset(dir_a, dir_b, output_dir):
    """Create paired dataset structure from two directories"""
    
    # Create output structure
    os.makedirs(os.path.join(output_dir, 'A', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'B', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'A', 'test'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'B', 'test'), exist_ok=True)
    
    # Get all images
    images_a = sorted([f for f in os.listdir(dir_a) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    images_b = sorted([f for f in os.listdir(dir_b) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    if len(images_a) != len(images_b):
        print(f"Warning: Different number of images ({len(images_a)} vs {len(images_b)})")
        print("Make sure images are properly paired!")
    
    # Split train/test (90/10)
    split_idx = int(len(images_a) * 0.9)
    train_images = images_a[:split_idx]
    test_images = images_a[split_idx:]
    
    # Copy training images
    print("Copying training images...")
    for i, filename in enumerate(train_images):
        src_a = os.path.join(dir_a, filename)
        src_b = os.path.join(dir_b, filename)
        
        dst_a = os.path.join(output_dir, 'A', 'train', filename)
        dst_b = os.path.join(output_dir, 'B', 'train', filename)
        
        if os.path.exists(src_a):
            shutil.copy2(src_a, dst_a)
        if os.path.exists(src_b):
            shutil.copy2(src_b, dst_b)
    
    # Copy test images
    print("Copying test images...")
    for filename in test_images:
        src_a = os.path.join(dir_a, filename)
        src_b = os.path.join(dir_b, filename)
        
        dst_a = os.path.join(output_dir, 'A', 'test', filename)
        dst_b = os.path.join(output_dir, 'B', 'test', filename)
        
        if os.path.exists(src_a):
            shutil.copy2(src_a, dst_a)
        if os.path.exists(src_b):
            shutil.copy2(src_b, dst_b)
    
    print(f"\nDataset created successfully!")
    print(f"Training pairs: {len(train_images)}")
    print(f"Test pairs: {len(test_images)}")


def extract_frames(video_path, output_dir, interval=30):
    """Extract frames from video at specified interval"""
    
    import cv2
    
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        if frame_count % interval == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            img = Image.fromarray(frame_rgb)
            
            # Save
            output_path = os.path.join(output_dir, f'frame_{saved_count:04d}.png')
            img.save(output_path)
            
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    
    print(f"Extracted {saved_count} frames from video")


def create_cartoon_dataset(input_dir, output_dir, min_size=256):
    """Prepare cartoon dataset by filtering and organizing images"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    image_extensions = ['.jpg', '.jpeg', '.png']
    count = 0
    
    for filename in os.listdir(input_dir):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            try:
                img_path = os.path.join(input_dir, filename)
                img = Image.open(img_path)
                
                # Check size
                if img.width >= min_size and img.height >= min_size:
                    # Copy to output
                    dst_path = os.path.join(output_dir, filename)
                    shutil.copy2(img_path, dst_path)
                    count += 1
                    print(f"Added: {filename} ({img.width}x{img.height})")
                else:
                    print(f"Skipped (too small): {filename} ({img.width}x{img.height})")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    print(f"\nTotal images added: {count}")


def main():
    parser = argparse.ArgumentParser(description='Dataset Preparation Utilities')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Resize command
    resize_parser = subparsers.add_parser('resize', help='Resize images')
    resize_parser.add_argument('--input', type=str, required=True)
    resize_parser.add_argument('--output', type=str, required=True)
    resize_parser.add_argument('--size', type=int, default=256)
    
    # Create paired dataset command
    pair_parser = subparsers.add_parser('create-paired', help='Create paired dataset')
    pair_parser.add_argument('--dir-a', type=str, required=True)
    pair_parser.add_argument('--dir-b', type=str, required=True)
    pair_parser.add_argument('--output', type=str, required=True)
    
    # Extract frames command
    extract_parser = subparsers.add_parser('extract-frames', help='Extract frames from video')
    extract_parser.add_argument('--video', type=str, required=True)
    extract_parser.add_argument('--output', type=str, required=True)
    extract_parser.add_argument('--interval', type=int, default=30)
    
    # Prepare cartoon dataset command
    cartoon_parser = subparsers.add_parser('prepare-cartoons', help='Prepare cartoon dataset')
    cartoon_parser.add_argument('--input', type=str, required=True)
    cartoon_parser.add_argument('--output', type=str, required=True)
    cartoon_parser.add_argument('--min-size', type=int, default=256)
    
    args = parser.parse_args()
    
    if args.command == 'resize':
        resize_images(args.input, args.output, args.size)
    
    elif args.command == 'create-paired':
        create_paired_dataset(args.dir_a, args.dir_b, args.output)
    
    elif args.command == 'extract-frames':
        extract_frames(args.video, args.output, args.interval)
    
    elif args.command == 'prepare-cartoons':
        create_cartoon_dataset(args.input, args.output, args.min_size)
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
