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


def apply_shadow_mask(img, shadow_threshold=10):
    """
    Creates a mask that marks shadow regions as invalid.

    Why this helps:
    - Lunar images with low sun angle have large black shadow regions
    - Shadow pixels = 0 or near-0 brightness = no texture = no matchable features
    - Trying to match in shadow regions wastes time and produces wrong matches
    - By masking shadows, we focus matching only on lit, textured regions

    How it works:
    - Pixels below shadow_threshold are marked as shadow (mask=0)
    - Pixels above threshold are marked as valid (mask=255)

    Parameters:
        img: grayscale numpy array
        shadow_threshold: pixel value below which = shadow (default 10)

    Returns:
        mask: binary mask (255=valid, 0=shadow)
        shadow_fraction: fraction of image in shadow (0.0 to 1.0)
    """
    mask = np.where(img > shadow_threshold, 255, 0).astype(np.uint8)

    # Erode mask slightly to avoid matching on shadow boundaries
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=2)

    shadow_fraction = 1.0 - (np.sum(mask > 0) / mask.size)
    return mask, shadow_fraction


def histogram_matching(src, ref):
    """
    Makes source image have similar brightness distribution as reference.

    Why this helps:
    - Two images from different orbits have different overall brightness
    - Different sun angles cause different average brightness levels
    - SIFT/AKAZE compute gradients — similar brightness = more similar gradients = better matching

    How it works:
    - Compute brightness histogram of both images
    - Find mapping from src brightness values to ref brightness values
    - Apply mapping to src image

    Simple example:
    - src has mostly dark pixels (sun angle 5°)
    - ref has mostly medium pixels (sun angle 15°)
    - After matching: src is adjusted to look more like ref in brightness

    Parameters:
        src: source image to adjust
        ref: reference image (target brightness distribution)

    Returns:
        matched: source image with adjusted brightness
    """
    # Compute cumulative histogram for both images
    src_hist, _ = np.histogram(src.flatten(), 256, [0, 256])
    ref_hist, _ = np.histogram(ref.flatten(), 256, [0, 256])

    src_cdf = src_hist.cumsum()
    ref_cdf = ref_hist.cumsum()

    # Normalize CDFs
    src_cdf_norm = src_cdf / src_cdf[-1]
    ref_cdf_norm = ref_cdf / ref_cdf[-1]

    # Build lookup table: for each src value, find closest ref value
    lookup = np.zeros(256, dtype=np.uint8)
    ref_idx = 0
    for src_val in range(256):
        while ref_idx < 255 and ref_cdf_norm[ref_idx] < src_cdf_norm[src_val]:
            ref_idx += 1
        lookup[src_val] = ref_idx

    matched = lookup[src]
    return matched


def preprocess_pair_advanced(img1, img2, max_size=None,
                              src_sun_elevation=None,
                              ref_sun_elevation=None,
                              use_histogram_matching=True,
                              use_shadow_mask=True):
    """
    Advanced preprocessing pipeline for challenging image pairs.

    Steps:
    1. Resize both images
    2. Apply sun-angle-aware CLAHE to each
    3. Histogram matching (make src look like ref in brightness)
    4. Shadow detection

    Parameters:
        img1: source image
        img2: reference image
        max_size: max image dimension
        src_sun_elevation: sun angle of source (from XML)
        ref_sun_elevation: sun angle of reference (from XML)
        use_histogram_matching: whether to apply histogram matching
        use_shadow_mask: whether to detect shadows

    Returns:
        img1_clean: preprocessed source
        img2_clean: preprocessed reference
        mask1: shadow mask for source (or None)
        mask2: shadow mask for reference (or None)
        shadow_info: dict with shadow fraction info
    """
    shadow_info = {}

    # Step 1: Resize
    if max_size is None:
        max_size = get_config()['preprocessing']['max_size']
    img1 = resize_to_max(img1, max_size)
    img2 = resize_to_max(img2, max_size)

    # Step 2: Sun-angle-aware CLAHE
    if src_sun_elevation is not None:
        img1 = apply_clahe(img1,
            clip_limit=4.0 if src_sun_elevation < 10
            else 2.5 if src_sun_elevation < 30 else 1.5)
    else:
        img1 = apply_clahe(img1)

    if ref_sun_elevation is not None:
        img2 = apply_clahe(img2,
            clip_limit=4.0 if ref_sun_elevation < 10
            else 2.5 if ref_sun_elevation < 30 else 1.5)
    else:
        img2 = apply_clahe(img2)

    # Step 3: Histogram matching (make src brightness similar to ref)
    if use_histogram_matching:
        img1 = histogram_matching(img1, img2)
        print("         Histogram matching applied")

    # Step 4: Shadow masking
    mask1 = mask2 = None
    if use_shadow_mask:
        mask1, sf1 = apply_shadow_mask(img1)
        mask2, sf2 = apply_shadow_mask(img2)
        shadow_info['src_shadow_fraction'] = sf1
        shadow_info['ref_shadow_fraction'] = sf2
        print(f"         Shadow fraction — source: {sf1:.1%}, reference: {sf2:.1%}")
        if sf1 > 0.5 or sf2 > 0.5:
            print("         WARNING: >50% image in shadow — matching will be difficult")

    return img1, img2, mask1, mask2, shadow_info


def preprocess_pair(img1, img2, max_size=None):
    """
    Preprocesses both source and reference images.
    Returns both cleaned images.
    """
    img1_clean = preprocess(img1, max_size)
    img2_clean = preprocess(img2, max_size)
    return img1_clean, img2_clean
