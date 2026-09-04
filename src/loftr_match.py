"""
loftr_match.py
--------------
Deep learning based matching using LoFTR (Local Feature TRansformer).

What is LoFTR?
- A pretrained deep learning model
- Finds matching points between two images WITHOUT needing keypoints first
- Works directly on image pixels (detector-free)
- Much better than SIFT for:
  * Very different lighting conditions
  * Low texture regions (flat lunar terrain)
  * Images from different cameras
  * Large viewpoint changes

We use the pretrained 'outdoor' weights which work well for
aerial/satellite imagery including lunar images.

No training needed — we just download and use it.
"""

import cv2
import numpy as np
import torch
import warnings

# Suppress the Python 3.14 torch warning
warnings.filterwarnings('ignore', category=FutureWarning)


def load_loftr(device=None):
    """
    Loads the pretrained LoFTR model.

    First call downloads weights from internet (~40MB) — takes 1-2 minutes.
    After that it's cached and loads instantly.

    Parameters:
        device: 'cuda' for GPU, 'cpu' for CPU. Auto-detects if None.

    Returns:
        matcher: loaded LoFTR model ready to use
        device: the device being used
    """
    from kornia.feature import LoFTR

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"         Loading LoFTR on {device}...")
    matcher = LoFTR(pretrained='outdoor')
    matcher = matcher.to(device)
    matcher.eval()  # set to inference mode (not training mode)

    return matcher, device


def detect_and_match_loftr(img1, img2, matcher=None, device=None, confidence_threshold=0.7):
    """
    Finds matching points between two images using LoFTR.

    How it works (simple):
    - Converts images to tensors (format PyTorch understands)
    - Feeds both images to LoFTR model
    - Model outputs matching point pairs + confidence scores
    - We filter by confidence threshold
    - Convert back to OpenCV format

    Parameters:
        img1: source image (grayscale numpy array)
        img2: reference image (grayscale numpy array)
        matcher: preloaded LoFTR model (loads automatically if None)
        device: 'cpu' or 'cuda'
        confidence_threshold: minimum confidence to keep a match (0.7 = keep confident matches only)

    Returns:
        src_pts: matched point coordinates in image 1
        dst_pts: matched point coordinates in image 2
        confidence: confidence scores for each match
        num_matches: number of matches found
    """
    # Load model if not provided
    if matcher is None:
        matcher, device = load_loftr(device)

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # LoFTR requires images to be multiples of 8 in both dimensions
    h1, w1 = img1.shape
    h2, w2 = img2.shape

    # Pad to nearest multiple of 8
    h1_pad = ((h1 + 7) // 8) * 8
    w1_pad = ((w1 + 7) // 8) * 8
    h2_pad = ((h2 + 7) // 8) * 8
    w2_pad = ((w2 + 7) // 8) * 8

    img1_padded = np.zeros((h1_pad, w1_pad), dtype=np.float32)
    img2_padded = np.zeros((h2_pad, w2_pad), dtype=np.float32)
    img1_padded[:h1, :w1] = img1.astype(np.float32) / 255.0
    img2_padded[:h2, :w2] = img2.astype(np.float32) / 255.0

    # Convert to PyTorch tensors
    # LoFTR expects shape: [batch, channels, height, width]
    tensor1 = torch.from_numpy(img1_padded)[None, None].to(device)
    tensor2 = torch.from_numpy(img2_padded)[None, None].to(device)

    # Run LoFTR
    with torch.no_grad():  # no_grad = faster, less memory, no training
        input_dict = {
            'image0': tensor1,
            'image1': tensor2
        }
        correspondences = matcher(input_dict)

    # Extract results
    mkpts0 = correspondences['keypoints0'].cpu().numpy()  # points in img1
    mkpts1 = correspondences['keypoints1'].cpu().numpy()  # points in img2
    confidence = correspondences['confidence'].cpu().numpy()

    # Filter by confidence
    mask = confidence > confidence_threshold
    mkpts0 = mkpts0[mask]
    mkpts1 = mkpts1[mask]
    confidence = confidence[mask]

    if len(mkpts0) < 4:
        raise ValueError(
            f"LoFTR found only {len(mkpts0)} confident matches "
            f"(threshold={confidence_threshold}). "
            f"Try lowering confidence_threshold or use different images."
        )

    # Convert to OpenCV format
    src_pts = mkpts0.reshape(-1, 1, 2).astype(np.float32)
    dst_pts = mkpts1.reshape(-1, 1, 2).astype(np.float32)

    return src_pts, dst_pts, confidence, len(mkpts0)


def loftr_to_opencv_matches(confidence):
    """
    Creates fake OpenCV match objects from LoFTR confidence scores.
    Needed because our visualizer expects OpenCV match objects.

    Parameters:
        confidence: array of confidence scores from LoFTR

    Returns:
        list of fake DMatch objects compatible with cv2.drawMatches
    """
    matches = []
    for i, conf in enumerate(confidence):
        m = cv2.DMatch()
        m.queryIdx = i
        m.trainIdx = i
        m.distance = float(1.0 - conf)  # convert confidence to distance
        matches.append(m)
    return matches


def create_fake_keypoints(pts):
    """
    Creates fake OpenCV KeyPoint objects from point coordinates.
    Needed because our visualizer expects OpenCV KeyPoint objects.

    Parameters:
        pts: array of (x, y) coordinates

    Returns:
        list of cv2.KeyPoint objects
    """
    keypoints = []
    for pt in pts.reshape(-1, 2):
        kp = cv2.KeyPoint(float(pt[0]), float(pt[1]), 1.0)
        keypoints.append(kp)
    return keypoints
