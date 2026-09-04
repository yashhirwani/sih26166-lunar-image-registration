"""
preprocess.py
-------------
Cleans and prepares lunar images before matching.

What it does:
1. Reads an image
2. Converts to grayscale (single channel, no color)
3. Applies CLAHE (fixes uneven brightness caused by sun angle differences)
4. Resizes if needed

Why we need this:
- Lunar images have harsh shadows due to sun angle
- CLAHE makes features visible even in shadowed areas
- Without this, keypoint detection fails in dark regions
"""

import cv2
import numpy as np
from .config_loader import get_config
from .config_loader import get_config


def load_image(image_path):
    """
    Reads an image from disk.
    Supports PNG, JPG, and any standard format.
    Returns the image as a numpy array (grid of numbers).
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    return img


def load_image_from_bytes(image_bytes):
    """
    Reads an image from uploaded bytes (used by Streamlit file uploader).
    Returns grayscale image as numpy array.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not decode uploaded image.")
    return img


def apply_clahe(img, clip_limit=None, tile_size=(8, 8)):
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization).

    What CLAHE does in simple words:
    - Divides the image into small tiles (8x8 by default)
    - In each tile, it stretches the contrast to make features more visible
    - Clips extreme values to avoid noise amplification
    - Merges tiles smoothly

    Why this helps for lunar images:
    - Sun at low angle = half the image in shadow, half very bright
    - CLAHE equalizes both areas so features are visible everywhere
    - Makes SIFT/AKAZE find more keypoints in shadowed regions

    Parameters:
        clip_limit: how aggressively to enhance contrast (2.0 is good default)
        tile_size: size of each tile (8x8 is good default)
    """
    if clip_limit is None:
        clip_limit = get_config()['preprocessing']['clahe_clip_limit']
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    return clahe.apply(img)


def resize_to_max(img, max_size=None):
    """
    Resizes image so its largest dimension is max_size pixels.
    Keeps aspect ratio (doesn't distort the image).

    Why: Very large images are slow to process.
    For demo purposes, 1024px is enough detail.
    """
    if max_size is None:
        max_size = get_config()['preprocessing']['max_size']
    h, w = img.shape
    if max(h, w) <= max_size:
        return img  # already small enough
    scale = max_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def preprocess(img, max_size=None, clip_limit=None):
    """
    Full preprocessing pipeline for one image.
    Steps: resize → CLAHE

    Parameters:
        img: grayscale numpy array
        max_size: maximum image dimension
        clip_limit: CLAHE aggressiveness. If None, reads from config.
                    Can be overridden using sun angle from XML metadata.

    Returns:
        cleaned image ready for keypoint detection
    """
    img = resize_to_max(img, max_size)
    img = apply_clahe(img, clip_limit)
    return img


def preprocess_with_sun_angle(img, sun_elevation_deg, max_size=None):
    """
    Preprocessing that adapts CLAHE based on sun elevation angle.

    This is the illumination-aware preprocessing mode.
    Sun angle comes from the OHRC XML metadata.

    Logic:
    - Sun elevation < 10° → very harsh shadows → aggressive CLAHE (clip=4.0)
    - Sun elevation 10-30° → medium shadows → moderate CLAHE (clip=2.5)
    - Sun elevation > 30° → soft shadows → gentle CLAHE (clip=1.5)

    This directly addresses PS Challenge #1: Illumination Variation.

    Parameters:
        img: grayscale numpy array
        sun_elevation_deg: sun elevation in degrees (from XML metadata)
        max_size: maximum image dimension

    Returns:
        cleaned image
    """
    if sun_elevation_deg < 10:
        clip_limit = 4.0
        print(f"         Sun elevation {sun_elevation_deg:.1f}° → HARD illumination → CLAHE clip=4.0")
    elif sun_elevation_deg < 30:
        clip_limit = 2.5
        print(f"         Sun elevation {sun_elevation_deg:.1f}° → MEDIUM illumination → CLAHE clip=2.5")
    else:
        clip_limit = 1.5
        print(f"         Sun elevation {sun_elevation_deg:.1f}° → EASY illumination → CLAHE clip=1.5")

    return preprocess(img, max_size, clip_limit)


def preprocess_pair(img1, img2, max_size=None):
    """
    Preprocesses both source and reference images.
    Returns both cleaned images.
    """
    img1_clean = preprocess(img1, max_size)
    img2_clean = preprocess(img2, max_size)
    return img1_clean, img2_clean
