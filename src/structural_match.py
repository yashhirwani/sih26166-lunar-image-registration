"""
structural_match.py
-------------------
Illumination-invariant feature matching using Phase Congruency.

Why this helps for lunar images:
- SIFT/AKAZE operate on raw pixel intensity gradients.
  When sun angle changes, shadows move → intensity gradients change
  dramatically even though the underlying surface structure is identical.
- Phase Congruency (PC) measures structural significance based on PHASE
  ALIGNMENT across Fourier frequency components — not on intensity magnitude.
  Physical edges and ridges produce consistent phase alignment regardless of
  whether they are bright or dark, illuminated or shadowed.
- This makes PC a much more robust representation for matching images taken
  at different sun angles (the primary challenge in the medium/hard cases).

Method:
  1. Compute PC map from each input image (replaces raw grayscale).
  2. Run the existing SIFT or AKAZE detector/matcher on the PC maps.
  The matching machinery itself is unchanged — only the input representation
  differs. This is the core idea behind the RIFT descriptor family referenced
  in arXiv:2509.04775.

API note (verified against installed phasepack):
  phasecong() returns a 7-element tuple.
  [0] = phase congruency magnitude map M  (H×W float64, range ~0-1)
  [1] = mean phase image
  [2] = orientation map (degrees)
  [3] = feature type (edge +1 / ridge -1 / flat 0)
  [4] = list of PC per orientation
  [5] = list of even/odd filter responses per scale
  [6] = noise threshold scalar
"""

import numpy as np
import cv2
import warnings
warnings.filterwarnings('ignore')


# ── Phase congruency parameters (Kovesi recommended defaults) ─────────────────
_PC_PARAMS = dict(
    nscale=4,          # number of wavelet scales
    norient=6,         # number of orientations (6 = 30° apart)
    minWaveLength=3,   # minimum wavelength of filter in pixels
    mult=2.1,          # scaling factor between successive filter wavelengths
    sigmaOnf=0.55,     # bandwidth of log-Gabor filter
)


def compute_phase_congruency_map(image: np.ndarray) -> np.ndarray:
    """
    Computes a phase congruency magnitude map from a grayscale image.

    Phase congruency highlights structural features (edges, ridges, corners)
    based on phase alignment across frequency components, not intensity
    gradient magnitude.  Shadows and brightness shifts change raw intensity
    but leave phase structure largely intact — making this representation
    robust to the illumination variation challenge in PS166.

    Parameters:
        image : np.ndarray — grayscale (H×W) or BGR (H×W×3) input

    Returns:
        np.ndarray — uint8 (H×W), values 0-255, normalised PC magnitude map
    """
    from phasepack import phasecong

    # Ensure single-channel float input
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_float = image.astype(np.float64)

    # Compute PC — result[0] is the magnitude map M (confirmed by API probe)
    result = phasecong(img_float, **_PC_PARAMS)
    pc_map = result[0]   # shape (H, W), float64

    # Normalise to uint8 for compatibility with SIFT/AKAZE
    pc_min, pc_max = pc_map.min(), pc_map.max()
    pc_norm = (
        (pc_map - pc_min) / (pc_max - pc_min + 1e-8) * 255
    ).astype(np.uint8)

    return pc_norm


def detect_and_match_structural(
    image1: np.ndarray,
    image2: np.ndarray,
    detector: str = "sift",
):
    """
    Matches two images using the existing SIFT or AKAZE detector/matcher
    applied to their Phase Congruency maps instead of raw grayscale.

    This is a drop-in alternative to detect_and_match_sift() /
    detect_and_match_akaze() in match.py — it returns the identical
    (src_pts, dst_pts, kp1, kp2, good_matches) tuple so the pipeline
    router can score it alongside the other candidates with zero changes
    to the scoring or homography estimation code.

    Parameters:
        image1, image2 : grayscale or BGR np.ndarray input images
        detector       : "sift" (default) or "akaze"

    Returns:
        src_pts      : (N, 1, 2) float32 — matched points in image1
        dst_pts      : (N, 1, 2) float32 — matched points in image2
        kp1, kp2     : lists of cv2.KeyPoint
        good_matches : list of cv2.DMatch
    """
    from .match import detect_and_match_sift, detect_and_match_akaze

    # Compute PC maps
    pc1 = compute_phase_congruency_map(image1)
    pc2 = compute_phase_congruency_map(image2)

    print(f"         [PC] map1: min={pc1.min()} max={pc1.max()} "
          f"non-zero={np.count_nonzero(pc1)}/{pc1.size} "
          f"({100*np.count_nonzero(pc1)/pc1.size:.1f}%)")
    print(f"         [PC] map2: min={pc2.min()} max={pc2.max()} "
          f"non-zero={np.count_nonzero(pc2)}/{pc2.size} "
          f"({100*np.count_nonzero(pc2)/pc2.size:.1f}%)")

    if detector == "akaze":
        return detect_and_match_akaze(pc1, pc2)
    return detect_and_match_sift(pc1, pc2)
