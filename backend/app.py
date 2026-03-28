"""
Flask backend for the Cartoonist UI.
Provides only the photo-to-cartoon workflow.
"""

import argparse
import io
import os
from typing import Any, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

from config import Pix2PixConfig
from data.dataset import load_image
from .free_models import cartoonize_with_animegan, load_animegan_cartoonizer

app = Flask(__name__)
CORS(app)

pix2pix_model: Optional[Any] = None
pix2pix_ready = False
pix2pix_source = 'Unavailable'
pix2pix_mode = 'none'
animegan_painter = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def _resolve_state_dict(checkpoint: Any, candidate_keys: Tuple[str, ...]):
    if isinstance(checkpoint, dict):
        for key in candidate_keys:
            if key in checkpoint:
                return checkpoint[key]

    return checkpoint


def load_pix2pix_model(checkpoint_path: Optional[str] = None):
    """Load a local Pix2Pix checkpoint or fall back to a free AnimeGAN model."""
    global animegan_painter, pix2pix_mode, pix2pix_model, pix2pix_ready, pix2pix_source

    pix2pix_model = None
    pix2pix_ready = False
    pix2pix_mode = 'none'
    pix2pix_source = 'Unavailable'
    animegan_painter = None

    if checkpoint_path and os.path.exists(checkpoint_path):
        from models.pix2pix import Generator as Pix2PixGenerator

        config = Pix2PixConfig()
        model = Pix2PixGenerator(
            in_channels=3,
            out_channels=3,
            ngf=config.ngf,
        ).to(device)

        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            state_dict = _resolve_state_dict(checkpoint, ('generator_state_dict', 'state_dict'))
            model.load_state_dict(state_dict)
            model.eval()
            pix2pix_model = model
            pix2pix_ready = True
            pix2pix_mode = 'checkpoint'
            pix2pix_source = f'Local Pix2Pix checkpoint: {os.path.basename(checkpoint_path)}'
            print(f'[ok] Pix2Pix loaded from {checkpoint_path}')
            return True
        except Exception as error:
            print(f'[error] Failed to load Pix2Pix checkpoint: {error}')

    print('[warning] Pix2Pix checkpoint not found. Falling back to a free AnimeGANv2 model.')

    try:
        model, painter, source = load_animegan_cartoonizer(device)
        pix2pix_model = model
        animegan_painter = painter
        pix2pix_ready = True
        pix2pix_mode = 'animegan2'
        pix2pix_source = source
        print(f'[ok] Free cartoonizer loaded: {source}')
        return True
    except Exception as error:
        print(f'[error] Failed to load free cartoonizer: {error}')
        return False


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get backend connectivity and cartoonizer readiness."""
    return jsonify({
        'pix2pix': pix2pix_ready,
        'device': device,
        'cartoon_model': pix2pix_source,
    })


@app.route('/api/cartoonize', methods=['POST'])
def cartoonize():
    """Cartoonize an uploaded image."""
    if not pix2pix_ready or pix2pix_model is None:
        return jsonify({
            'error': 'Cartoon model is not loaded. Start the backend with a trained checkpoint or the free fallback.'
        }), 503

    file = request.files.get('image')
    if file is None or file.filename == '':
        return jsonify({'error': 'No image provided.'}), 400

    try:
        file.stream.seek(0)

        if pix2pix_mode == 'animegan2':
            input_image = Image.open(file.stream).convert('RGB')
            output_image = cartoonize_with_animegan(pix2pix_model, animegan_painter, input_image)
        else:
            input_tensor = load_image(file.stream, img_size=Pix2PixConfig.img_size).to(device)

            pix2pix_model.eval()
            with torch.no_grad():
                output_tensor = pix2pix_model(input_tensor)

            output = (output_tensor + 1) / 2
            output = torch.clamp(output, 0, 1)
            output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
            output = (output * 255).astype(np.uint8)
            output_image = Image.fromarray(output)

        image_io = io.BytesIO()
        output_image.save(image_io, 'PNG')
        image_io.seek(0)
        return send_file(image_io, mimetype='image/png')
    except Exception as error:
        print(f'[error] Cartoonize failed: {error}')
        return jsonify({'error': str(error)}), 500


def main():
    """Initialize and start the server."""
    parser = argparse.ArgumentParser(description='Cartoonist Flask Backend')
    parser.add_argument('--pix2pix', type=str, help='Path to a Pix2Pix checkpoint')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host address')
    parser.add_argument('--port', type=int, default=5000, help='Port number')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()

    print('[startup] Cartoonist backend')
    print(f'[startup] Device: {device}')

    pix2pix_loaded = load_pix2pix_model(args.pix2pix)

    print('[ready] Backend initialized')
    print(f'[ready] Cartoon model loaded: {pix2pix_loaded}')
    print(f'[ready] API available at http://{args.host}:{args.port}/api')

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
