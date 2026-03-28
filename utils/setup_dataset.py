"""
Quick Dataset Setup Script
Creates folder structure and provides instructions
"""

import os
from pathlib import Path


def setup_dataset(name="quick_test"):
    """Create dataset folder structure"""
    
    base_path = Path(f"datasets/{name}")
    
    # Create folders
    folders = [
        base_path / "A" / "train",
        base_path / "A" / "test",
        base_path / "B" / "train",
        base_path / "B" / "test",
    ]
    
    print(f"\n📁 Creating dataset structure for '{name}'...\n")
    
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {folder}")
    
    print("\n✅ Dataset folders created successfully!")
    
    # Print instructions
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print(f"""
1. Gather 10-20 paired images

2. Place ORIGINAL photos in:
   📂 datasets/{name}/A/train/

3. Place CARTOON versions in:
   📂 datasets/{name}/B/train/

⚠️  IMPORTANT: Files must have IDENTICAL names!

   ✅ CORRECT:
   A/train/image_01.jpg  ← Original
   B/train/image_01.jpg  ← Cartoon version
   
   ❌ WRONG:
   A/train/photo1.jpg
   B/train/cartoon1.jpg

4. After adding images, train with:
   python train_pix2pix.py --dataset datasets/{name}

5. Training will take 2-3 hours for 100 epochs

6. When done, use your model:
   python -m backend --pix2pix outputs/pix2pix/final_model.pth
""")
    
    print("="*60)
    print("\n💡 Tip: Check FIRST_MODEL_GUIDE.md for detailed instructions!")
    print("="*60 + "\n")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]
    else:
        dataset_name = "quick_test"
    
    setup_dataset(dataset_name)
