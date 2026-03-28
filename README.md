# Cartoonist

<p align="center">
  <img src="https://skillicons.dev/icons?i=python,flask,pytorch,react,vite" alt="Cartoonist tech stack" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python badge" />
  <img src="https://img.shields.io/badge/Flask-Backend-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask badge" />
  <img src="https://img.shields.io/badge/PyTorch-Inference-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch badge" />
  <img src="https://img.shields.io/badge/React-Frontend-61DAFB?style=for-the-badge&logo=react&logoColor=0A0A0A" alt="React badge" />
  <img src="https://img.shields.io/badge/Vite-Dev%20Server-646CFF?style=for-the-badge&logo=vite&logoColor=white" alt="Vite badge" />
</p>

<p align="center">
  Photo-to-cartoon web app with a React frontend, a packaged Flask backend, and a free AnimeGAN fallback.
</p>

<img width="1915" height="922" alt="Screenshot 2026-03-28 172838" src="https://github.com/user-attachments/assets/76562a2e-9312-46cf-bca1-367fea13b02e" />

## Overview

Cartoonist is a focused photo-to-cartoon app.

- Upload a portrait, selfie, or scene in the browser
- Send the image to a Flask API for processing
- Get back a stylized cartoon result as a PNG
- Run with a free AnimeGANv2 fallback or your own trained Pix2Pix checkpoint

The current app is intentionally single-purpose: photo to cartoon only.

## Features

- Drag-and-drop React UI built with Vite
- Flask API packaged under `backend/`
- Free AnimeGANv2 fallback on startup
- Optional local Pix2Pix checkpoint support
- Before/after preview with one-click download
- Simple local setup for development

## Tech Stack

- Python
- Flask + Flask-CORS
- PyTorch
- React 18
- Vite
- Axios
- React Dropzone

## Project Structure

```text
Cartoonist/
|-- backend/
|   |-- app.py             # Flask app and API routes
|   |-- free_models.py     # Free AnimeGAN fallback loader
|   |-- __init__.py
|   `-- __main__.py        # Enables: python -m backend
|-- frontend/
|   |-- src/
|   |   |-- App.tsx
|   |   |-- App.css
|   |   |-- api.ts
|   |   |-- index.css
|   |   `-- main.tsx
|   `-- package.json
|-- data/                  # Dataset utilities
|-- models/                # Training/inference model definitions
|-- config.py
|-- requirements.txt
|-- start_ui.py            # Helper script to start the packaged backend
`-- train_pix2pix.py       # Train your own Pix2Pix checkpoint
```

## Quick Start

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the backend

```bash
python -m backend --port 5000
```

The API will be available at `http://localhost:5000/api`.

### 3. Start the frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`.

## How The Model Works

Cartoonist supports two backend modes:

### Free mode

If you do not pass a local checkpoint, the backend loads a free AnimeGANv2 portrait model through `torch.hub`.

```bash
python -m backend
```

Notes:

- The first run downloads the fallback model
- The downloaded files are cached under `backend/model_cache/`
- This is the easiest way to get the app working locally

### Local Pix2Pix checkpoint

If you have trained your own paired photo-to-cartoon model, start the backend with it:

```bash
python -m backend --pix2pix outputs/pix2pix/final_model.pth
```

## API

### `GET /api/status`

Returns backend health and model readiness.

Example:

```json
{
  "cartoon_model": "AnimeGANv2 Face Portrait v2",
  "device": "cpu",
  "pix2pix": true
}
```

### `POST /api/cartoonize`

Uploads one image file using the `image` form field and returns a PNG result.

## Training Your Own Pix2Pix Model

Create a paired dataset like this:

```text
datasets/my_dataset/
|-- A/
|   |-- train/
|   `-- test/
`-- B/
    |-- train/
    `-- test/
```

- `A` contains the original photos
- `B` contains the matching cartoon targets
- filenames must match exactly across pairs

Train the model:

```bash
python train_pix2pix.py --dataset datasets/my_dataset
```

Then use the checkpoint in the app:

```bash
python -m backend --pix2pix outputs/pix2pix/final_model.pth
```

## Development Notes

- Frontend dev server: `http://localhost:3000`
- Backend API: `http://localhost:5000`
- Main frontend entry: `frontend/src/main.tsx`
- Main backend entry: `backend/__main__.py`

`start_ui.py` is also available if you want a simple helper that prints the frontend steps and launches the packaged backend for you.

## Legacy Files

Some older research and experimentation files are still in the repo, including `demo.py`, `inference.py`, and `train_stylegan.py`.

They are not part of the current shipped web app. The active product flow in this repository is photo to cartoon through the React UI and the packaged Flask backend.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).

## Acknowledgments

- Pix2Pix: Isola et al., "Image-to-Image Translation with Conditional Adversarial Networks"
- AnimeGANv2 fallback model via PyTorch Hub
