"""
lightglue_match.py
------------------
Deep learning feature matching using DISK + LightGlue.

What is LightGlue?
- State-of-the-art deep learning matcher (2023, better than SuperGlue)
- Works with DISK keypoint detector
- Specifically designed for challenging matching conditions:
  * Low texture (flat lunar terrain)
  * Illumination changes (different sun angles)
  * Viewpoint changes (different orbits)

What is DISK?
- Deep Image keypoint detector
- Finds more stable keypoints than SIFT on difficult images
- Works well on lunar surface with extreme shadows

Why better than LoFTR for our case:
- LightGlue is faster
- DISK finds better keypoints on crater-dominated terrain
- Combined DISK+LightGlue outperforms LoFTR-outdoor on remote sensing data

No training needed — pretrained weights downloaded automatically.
"""

import cv2
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


def load_lightglue(device=None):
    """
    Loads pretrained DISK + LightGlue models.

    First call downloads weights (~50MB) — takes 1-2 minutes.
    After that loads instantly from cache.

    Parameters:
        device: 'cuda' or 'cpu'. Auto-detects if None.

    Returns:
        extractor: DISK keypoint extractor
        matcher: LightGlue matcher
        device: device being used
    """
    from kornia.feature import DISK, LightGlue

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"         Loading DISK + LightGlue on {device}...")

    # DISK extractor — finds keypoints and descriptors
    extractor = DISK.from_pretrained('depth').to(device)
    extractor.eval()

    # LightGlue matcher — matches keypoints between two images
    matcher = LightGlue(features='disk').to(device)
    matcher.eval()

    print(f"         DISK + LightGlue loaded successfully")
    return extractor, matcher, device


def detect_and_match_lightglue(img1, img2, extractor=None, matcher=None,
                                device=None, max_keypoints=2048,
                                confidence_threshold=0.5):
    """
    Finds matching points using DISK + LightGlue.

    How it works:
    1. DISK finds keypoints in both images (much better than SIFT for hard cases)
    2. LightGlue matches keypoints using attention mechanism
    3. LightGlue also filters matches internally (removes bad ones)
    4. We apply an additional confidence filter

    Parameters:
        img1: source image (grayscale numpy array)
        img2: reference image (grayscale numpy array)
        extractor: preloaded DISK model (loads if None)
        matcher: preloaded LightGlue model (loads if None)
        device: 'cpu' or 'cuda'
        max_keypoints: max keypoints per image (more = slower but better)
        confidence_threshold: min confidence to keep match

    Returns:
        src_pts: matched points in image 1
        dst_pts: matched points in image 2
        confidence: confidence score per match
        num_matches: number of matches found
    """
    if extractor is None or matcher is None:
        extractor, matcher, device = load_lightglue(device)

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # DISK requires images as float tensors [B, C, H, W]
    # Values in range [0, 1]
    def img_to_tensor(img):
        h, w = img.shape[:2]
        # Pad to multiple of 32 (DISK requirement)
        h_pad = ((h + 31) // 32) * 32
        w_pad = ((w + 31) // 32) * 32
        padded = np.zeros((h_pad, w_pad), dtype=np.float32)
        padded[:h, :w] = img.astype(np.float32) / 255.0
        # Convert to [1, 1, H, W] then repeat to [1, 3, H, W] (DISK expects 3 channels)
        t = torch.from_numpy(padded)[None, None].repeat(1, 3, 1, 1).to(device)
        return t, h, w

    t1, h1, w1 = img_to_tensor(img1)
    t2, h2, w2 = img_to_tensor(img2)

    with torch.no_grad():
        # Extract keypoints and descriptors with DISK
        features1 = extractor(t1, n=max_keypoints, pad_if_not_divisible=True)[0]
        features2 = extractor(t2, n=max_keypoints, pad_if_not_divisible=True)[0]

        kps1 = features1.keypoints      # [N, 2] keypoint coordinates
        descs1 = features1.descriptors  # [N, 128] descriptors
        kps2 = features2.keypoints
        descs2 = features2.descriptors

        # Match with LightGlue
        # Prepare input in LightGlue format
        data = {
            'image0': {
                'keypoints': kps1[None],        # [1, N, 2]
                'descriptors': descs1[None],    # [1, N, 128]
                'image_size': torch.tensor([[w1, h1]], device=device).float()
            },
            'image1': {
                'keypoints': kps2[None],
                'descriptors': descs2[None],
                'image_size': torch.tensor([[w2, h2]], device=device).float()
            }
        }

        matches_out = matcher(data)

    # Extract match indices and confidence
    matches = matches_out['matches'][0].cpu().numpy()       # [M, 2]
    confidence = matches_out['matching_scores0'][0].cpu().numpy()  # [N]

    # Get matched keypoint coordinates
    kps1_np = kps1.cpu().numpy()  # [N, 2]
    kps2_np = kps2.cpu().numpy()  # [N, 2]

    # matches[:, 0] = index in kps1, matches[:, 1] = index in kps2
    valid = matches[:, 0] >= 0
    if valid.sum() == 0:
        raise ValueError("LightGlue found no matches")

    matched_kps1 = kps1_np[matches[valid, 0]]  # [M, 2]
    matched_kps2 = kps2_np[matches[valid, 1]]  # [M, 2]
    matched_conf = confidence[matches[valid, 0]]  # [M]

    # Filter by confidence
    conf_mask = matched_conf > confidence_threshold
    if conf_mask.sum() < 4:
        # Lower threshold if too few matches
        conf_mask = matched_conf > 0.3
    if conf_mask.sum() < 4:
        raise ValueError(
            f"LightGlue found only {conf_mask.sum()} confident matches. "
            f"Images may not overlap or are too different."
        )

    final_kps1 = matched_kps1[conf_mask]
    final_kps2 = matched_kps2[conf_mask]
    final_conf = matched_conf[conf_mask]

    # Convert to OpenCV format
    src_pts = final_kps1.reshape(-1, 1, 2).astype(np.float32)
    dst_pts = final_kps2.reshape(-1, 1, 2).astype(np.float32)

    return src_pts, dst_pts, final_conf, len(final_kps1)


def lightglue_to_opencv_keypoints(pts):
    """
    Converts point array to OpenCV KeyPoint list for visualization.
    """
    return [cv2.KeyPoint(float(p[0]), float(p[1]), 1.0)
            for p in pts.reshape(-1, 2)]


def lightglue_to_opencv_matches(confidence):
    """
    Creates OpenCV DMatch objects from confidence scores for visualization.
    """
    matches = []
    for i, conf in enumerate(confidence):
        m = cv2.DMatch()
        m.queryIdx = i
        m.trainIdx = i
        m.distance = float(1.0 - conf)
        matches.append(m)
    return matches
