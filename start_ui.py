"""
Quick start script for the Cartoonist UI.
Starts the packaged backend and prints frontend instructions.
"""

import os
import subprocess
import sys
import time


def print_banner():
    print("=" * 60)
    print("Cartoonist - UI Quick Start")
    print("=" * 60)
    print()


def check_dependencies():
    """Check if required packages are installed."""
    print("Checking dependencies...")

    try:
        import flask  # noqa: F401
        import flask_cors  # noqa: F401
        print("   [ok] Flask installed")
    except ImportError:
        print("   [missing] Flask not installed")
        print("\nInstall with: pip install flask flask-cors")
        return False

    try:
        import torch  # noqa: F401
        print("   [ok] PyTorch installed")
    except ImportError:
        print("   [missing] PyTorch not installed")
        print("\nInstall with: pip install torch torchvision")
        return False

    print()
    return True


def find_checkpoint():
    """Find an available Pix2Pix checkpoint if one exists."""
    print("Searching for Pix2Pix checkpoints...")

    checkpoints_dir = "checkpoints"
    pix2pix_path = None

    if os.path.exists(checkpoints_dir):
        for file_name in os.listdir(checkpoints_dir):
            lower_name = file_name.lower()
            if file_name.endswith(".pth") and "pix2pix" in lower_name:
                pix2pix_path = os.path.join(checkpoints_dir, file_name)
                print(f"   [ok] Found Pix2Pix checkpoint: {file_name}")
                break
    else:
        print("   [warning] No checkpoints directory found")

    if pix2pix_path is None:
        print("   [info] No local Pix2Pix checkpoint found")

    print()
    return pix2pix_path


def start_backend(pix2pix_path=None, port=5000):
    """Start the packaged Flask backend."""
    print("Starting backend server...")
    print(f"   Port: {port}")
    print(f"   URL: http://localhost:{port}")
    print()

    cmd = [sys.executable, "-m", "backend", "--port", str(port)]

    if pix2pix_path:
        cmd.extend(["--pix2pix", pix2pix_path])

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nBackend stopped.")
    except Exception as error:
        print(f"Error starting backend: {error}")


def print_instructions():
    """Print startup instructions for the frontend."""
    print("=" * 60)
    print("Backend is ready to start")
    print()
    print("Frontend steps:")
    print()
    print("   1. Open a new terminal")
    print("   2. Change into the frontend folder:")
    print("      cd frontend")
    print()
    print("   3. Install dependencies if needed:")
    print("      npm install")
    print()
    print("   4. Start the dev server:")
    print("      npm run dev")
    print()
    print("   5. Open http://localhost:3000")
    print()
    print("=" * 60)
    print()
    print("API endpoints:")
    print("   - Status: http://localhost:5000/api/status")
    print("   - Cartoonize: POST http://localhost:5000/api/cartoonize")
    print()
    print("Press Ctrl+C to stop the backend")
    print("=" * 60)


def main():
    print_banner()

    if not check_dependencies():
        return

    pix2pix_path = find_checkpoint()

    if not pix2pix_path:
        print("Warning: no local Pix2Pix checkpoint found.")
        print("The backend will use the free AnimeGAN fallback instead.")
        print()

    print_instructions()
    print()
    print("Starting backend in 3 seconds...")
    time.sleep(3)
    print()

    start_backend(pix2pix_path)


if __name__ == "__main__":
    main()
