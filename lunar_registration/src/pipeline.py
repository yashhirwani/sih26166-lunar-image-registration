"""
pipeline.py
-----------
The main registration pipeline.
Connects all modules together into one function call.

Usage:
    result = run_pipeline(img1, img2, method='auto')

Returns a dictionary with everything:
    - registered image
    - match visualization
    - checkerboard
    - all metrics
"""

import numpy as np
import cv2
import time

from .preprocess import preprocess_pair
from .match import detect_and_match_sift, detect_and_match_akaze, enforce_uniform_distribution
from .transform import estimate_homography_ransac, warp_image, compute_rmse, subpixel_refinement
from .metrics import compute_all_metrics, format_metrics_report
from .visualize import draw_matches, create_checkerboard, create_side_by_side, create_difference_image


def run_pipeline(img1, img2, method='auto', max_size=1024):
    """
    Complete image registration pipeline.

    Steps:
    1. Preprocess both images (CLAHE)
    2. Detect and match keypoints
    3. Enforce uniform distribution
    4. Estimate homography (RANSAC)
    5. Apply sub-pixel refinement
    6. Warp source image
    7. Compute metrics
    8. Generate visualizations

    Parameters:
        img1: source image (numpy array, grayscale)
        img2: reference image (numpy array, grayscale)
        method: 'sift', 'akaze', or 'auto' (auto picks best)
        max_size: maximum image dimension for processing

    Returns:
        result dict with all outputs
    """
    start_time = time.time()
    result = {}

    # ── Step 1: Preprocess ──────────────────────────────────────────
    img1_clean, img2_clean = preprocess_pair(img1, img2, max_size)
    result['img1_preprocessed'] = img1_clean
    result['img2_preprocessed'] = img2_clean

    # ── Step 2: Detect and match keypoints ─────────────────────────
    # 'auto' tries AKAZE first (better for illumination), falls back to SIFT
    if method == 'auto' or method == 'akaze':
        try:
            src_pts, dst_pts, kp1, kp2, good_matches = detect_and_match_akaze(img1_clean, img2_clean)
            used_method = 'AKAZE'
        except Exception as akaze_err:
            if method == 'akaze':
                # User explicitly requested AKAZE — re-raise so they know it failed
                raise ValueError(f"AKAZE requested but failed: {akaze_err}") from akaze_err
            # Auto mode: fall back to SIFT
            src_pts, dst_pts, kp1, kp2, good_matches = detect_and_match_sift(img1_clean, img2_clean)
            used_method = 'SIFT (AKAZE fallback)'
    elif method == 'sift':
        src_pts, dst_pts, kp1, kp2, good_matches = detect_and_match_sift(img1_clean, img2_clean)
        used_method = 'SIFT'
    else:
        raise ValueError(f"Unknown method: {method}. Use 'sift', 'akaze', or 'auto'")

    result['method'] = used_method
    result['total_matches_before_filter'] = len(good_matches)

    # ── Step 3: Enforce uniform distribution ───────────────────────
    # Makes sure matches are spread evenly across image
    # Required by ISRO PS explicitly
    src_pts_uniform, dst_pts_uniform, good_matches_uniform = enforce_uniform_distribution(
        src_pts, dst_pts, good_matches, img1_clean.shape
    )
    result['total_matches_after_distribution'] = len(good_matches_uniform)

    # ── Step 4: Estimate homography with RANSAC ────────────────────
    M, mask, num_inliers, inlier_ratio = estimate_homography_ransac(
        src_pts_uniform, dst_pts_uniform
    )
    result['homography_matrix'] = M
    result['ransac_mask'] = mask

    # ── Step 5: Warp source image ──────────────────────────────────
    warped_img = warp_image(img1_clean, M, img2_clean.shape)
    result['registered_image'] = warped_img

    # ── Step 6: Sub-pixel refinement ───────────────────────────────
    # Fine-tunes alignment to below 1 pixel accuracy
    try:
        shift, response = subpixel_refinement(
            img2_clean.astype(np.float64),
            warped_img.astype(np.float64)
        )
        result['subpixel_shift'] = shift
        result['subpixel_response'] = float(response)

        # Apply sub-pixel correction if shift is small (< 5 pixels)
        if abs(shift[0]) < 5 and abs(shift[1]) < 5:
            correction = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
            h, w = warped_img.shape
            warped_refined = cv2.warpAffine(warped_img, correction, (w, h))
            result['registered_image_refined'] = warped_refined
        else:
            result['registered_image_refined'] = warped_img
    except Exception:
        result['registered_image_refined'] = warped_img
        result['subpixel_shift'] = (0, 0)

    # ── Step 7: Compute metrics ────────────────────────────────────
    metrics = compute_all_metrics(
        src_pts_uniform, dst_pts_uniform, M, mask, img1_clean.shape
    )
    metrics['processing_time'] = time.time() - start_time
    result['metrics'] = metrics
    result['metrics_report'] = format_metrics_report(metrics, used_method)

    # ── Step 8: Generate visualizations ───────────────────────────
    # Match lines visualization
    result['match_visualization'] = draw_matches(
        img1_clean, kp1, img2_clean, kp2,
        good_matches_uniform, mask
    )

    # Checkerboard view
    result['checkerboard'] = create_checkerboard(
        img2_clean, result['registered_image_refined']
    )

    # Side by side comparison
    result['side_by_side'] = create_side_by_side(
        img2_clean,
        result['registered_image_refined'],
        "Reference Image",
        "Registered Source"
    )

    # Difference image
    result['difference_image'] = create_difference_image(
        img2_clean, result['registered_image_refined']
    )

    return result
