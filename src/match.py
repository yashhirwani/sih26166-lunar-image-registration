"""
match.py
--------
Finds matching points between two lunar images.

What it does:
1. Detects keypoints (distinctive spots) in both images
2. Computes descriptors (fingerprints) for each keypoint
3. Matches descriptors between image 1 and image 2
4. Filters out bad matches using Lowe's ratio test

Two methods available:
- SIFT: classic, reliable, good for most cases
- AKAZE: better for illumination changes, faster than SIFT
"""

import cv2
import numpy as np


def detect_and_match_sift(img1, img2, ratio_threshold=0.75):
    """
    Detects keypoints and matches them using SIFT algorithm.

    SIFT = Scale Invariant Feature Transform
    - Finds distinctive spots (keypoints) in both images
    - Creates 128-number fingerprint (descriptor) for each spot
    - Matches fingerprints between images
    - Filters bad matches using ratio test

    Parameters:
        img1: source image (grayscale numpy array)
        img2: reference image (grayscale numpy array)
        ratio_threshold: how strict the matching is (0.75 = keep only clear matches)

    Returns:
        src_pts: matched point coordinates in image 1
        dst_pts: matched point coordinates in image 2
        keypoints1: all keypoints in image 1
        keypoints2: all keypoints in image 2
        good_matches: list of good match objects (for visualization)
    """
    sift = cv2.SIFT_create(nfeatures=5000)

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        raise ValueError(f"Not enough keypoints found. Image1: {len(kp1)}, Image2: {len(kp2)}")

    # BFMatcher = Brute Force Matcher
    # Compares every descriptor in img1 with every descriptor in img2
    # k=2 means return top 2 matches for each keypoint
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    matches = matcher.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    # For each keypoint, we got 2 matches (best and second best)
    # If best match is clearly better than second best → keep it
    # If they're similar in quality → it's ambiguous → discard
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)

    if len(good_matches) < 4:
        raise ValueError(f"Not enough good matches found: {len(good_matches)}. Try different images.")

    # Extract the actual pixel coordinates of matched points
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    return src_pts, dst_pts, kp1, kp2, good_matches


def detect_and_match_akaze(img1, img2, ratio_threshold=0.75):
    """
    Detects keypoints and matches them using AKAZE algorithm.

    AKAZE = Accelerated KAZE
    - Better than SIFT for images with very different illumination
    - Handles shadow variations better
    - Slightly faster than SIFT

    Same parameters and return values as detect_and_match_sift.
    """
    # Try different AKAZE APIs depending on opencv-contrib version
    try:
        akaze = cv2.AKAZE_create()
    except AttributeError:
        try:
            akaze = cv2.xfeatures2d_AKAZE.create()
        except AttributeError:
            # Fall back to KAZE if AKAZE not available
            akaze = cv2.KAZE_create()

    kp1, des1 = akaze.detectAndCompute(img1, None)
    kp2, des2 = akaze.detectAndCompute(img2, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        raise ValueError(f"Not enough keypoints found. Image1: {len(kp1)}, Image2: {len(kp2)}")

    # Use NORM_HAMMING for binary descriptors (AKAZE), NORM_L2 for float descriptors (KAZE)
    norm = cv2.NORM_HAMMING if des1.dtype == np.uint8 else cv2.NORM_L2
    matcher = cv2.BFMatcher(norm)
    matches = matcher.knnMatch(des1, des2, k=2)

    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)

    if len(good_matches) < 4:
        raise ValueError(f"Not enough good matches found: {len(good_matches)}. Try different images.")

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    return src_pts, dst_pts, kp1, kp2, good_matches


def enforce_uniform_distribution(src_pts, dst_pts, good_matches, img_shape, grid_size=8, max_per_cell=5):
    """
    Ensures match points are spread evenly across the image.

    Why ISRO requires this:
    - Without this, all matches cluster around craters or bright spots
    - Clustered matches = bad alignment in regions with no matches
    - Uniform distribution = good alignment everywhere in the image

    How it works:
    - Divides image into a grid (default 8x8 = 64 cells)
    - Keeps only max_per_cell best matches per cell
    - Result: matches spread evenly across entire image

    Parameters:
        src_pts, dst_pts: matched point coordinates
        good_matches: match objects
        img_shape: (height, width) of image
        grid_size: how many rows and columns to divide image into
        max_per_cell: maximum matches allowed per grid cell

    Returns:
        filtered src_pts, dst_pts, good_matches with uniform distribution
    """
    h, w = img_shape
    cell_h = h / grid_size
    cell_w = w / grid_size

    # Dictionary: cell → list of (match_index, match_distance)
    cells = {}
    for i, (pt, m) in enumerate(zip(src_pts.reshape(-1, 2), good_matches)):
        x, y = pt
        cell_row = min(int(y / cell_h), grid_size - 1)
        cell_col = min(int(x / cell_w), grid_size - 1)
        cell_key = (cell_row, cell_col)
        if cell_key not in cells:
            cells[cell_key] = []
        cells[cell_key].append((i, m.distance))

    # Keep only best max_per_cell matches per cell
    kept_indices = []
    for cell_key, cell_matches in cells.items():
        # Sort by distance (lower = better match)
        cell_matches.sort(key=lambda x: x[1])
        for idx, _ in cell_matches[:max_per_cell]:
            kept_indices.append(idx)

    if len(kept_indices) < 4:
        # Not enough after filtering, return original
        return src_pts, dst_pts, good_matches

    kept_indices = sorted(kept_indices)
    filtered_src = src_pts[kept_indices]
    filtered_dst = dst_pts[kept_indices]
    filtered_matches = [good_matches[i] for i in kept_indices]

    return filtered_src, filtered_dst, filtered_matches
