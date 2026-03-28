"""
Free model fallback for the Cartoonist backend.

This loader uses torch.hub to download AnimeGANv2 on first use.
"""

import os
from typing import Callable

import numpy as np
import torch
from PIL import Image

MODEL_CACHE_DIR = os.path.join(os.path.dirname(__file__), 'model_cache')
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
torch.hub.set_dir(MODEL_CACHE_DIR)

ANIMEGAN_REPO = 'bryandlee/animegan2-pytorch:main'


def load_animegan_cartoonizer(device: str):
    """Load a free AnimeGANv2 portrait cartoonizer from torch.hub."""
    model = torch.hub.load(
        ANIMEGAN_REPO,
        'generator',
        pretrained='face_paint_512_v2',
        device=device,
        progress=True,
        trust_repo=True,
    )
    painter = torch.hub.load(
        ANIMEGAN_REPO,
        'face2paint',
        size=512,
        device=device,
        side_by_side=False,
        trust_repo=True,
    )
    model = model.to(device)
    model.eval()
    return model, painter, 'AnimeGANv2 Face Portrait v2'


def cartoonize_with_animegan(model, painter: Callable, image: Image.Image):
    """Run AnimeGANv2 on a PIL image and return a PIL image."""
    output = painter(model, image.convert('RGB'))

    if isinstance(output, Image.Image):
        return output

    if torch.is_tensor(output):
        array = output.detach().cpu().clamp(0, 255).to(torch.uint8).numpy()
        if array.ndim == 3 and array.shape[0] in (1, 3):
            array = np.transpose(array, (1, 2, 0))
        return Image.fromarray(array)

    return Image.fromarray(np.asarray(output).astype(np.uint8))
