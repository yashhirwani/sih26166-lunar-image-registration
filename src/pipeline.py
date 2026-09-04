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
from .config_loader import get_config

from .preprocess import preprocess_pair, preprocess_with_sun_angle
from .match import detect_and_match_sift, detect_and_match_akaze, enforce_uniform_distribution
from .loftr_match import detect_and_match_loftr, loftr_to_opencv_matches, create_fake_keypoints, load_loftr
from .transform import estimate_homography_ransac, warp_image, compute_rmse, subpixel_refinement, subpixel_refinement_lk
from .metrics import compute_all_metrics, format_metrics_report
from .visualize import draw_matches, create_checkerboard, create_side_by_side, create_difference_image

# Cache LoFTR model so it loads only once
_loftr_model = None
_loftr_device = None


def run_pipeline(img1, img2, method='auto', max_size=None,
                 src_sun_elevation=None, ref_sun_elevation=None):
    """
    Complete image registration pipeline.

    Methods:
        'auto'  — tries AKAZE first, falls back to SIFT, then LoFTR if too few matches
        'sift'  — classical SIFT matching
        'akaze' — classical AKAZE matching
        'loftr' — deep learning LoFTR matching (best for hard cases)
    """
    global _loftr_model, _loftr_device
    start_time = time.time()
    result = {}

    print("\n" + "="*50)
    print("   LUNAR IMAGE REGISTRATION — PS166 SIH 2026")
    print("="*50)
    
    if max_size is None:
        max_size = get_config()['preprocessing']['max_size']

    # ── Step 1: Preprocess ──────────────────────────────────────────
    print("\n[Step 1] Preprocessing images (CLAHE)...")

    # Use sun-angle-aware preprocessing if metadata is available
    if src_sun_elevation is not None and ref_sun_elevation is not None:
        print("         Using illumination-aware preprocessing (sun angle from XML)")
        img1_clean = preprocess_with_sun_angle(img1, src_sun_elevation, max_size)
        img2_clean = preprocess_with_sun_angle(img2, ref_sun_elevation, max_size)
    else:
        img1_clean, img2_clean = preprocess_pair(img1, img2, max_size)
    result['img1_preprocessed'] = img1_clean
    result['img2_preprocessed'] = img2_clean
    print(f"         Source image size  : {img1_clean.shape}")
    print(f"         Reference image size: {img2_clean.shape}")

    # ── Step 2: Detect and match keypoints ─────────────────────────
    print(f"\n[Step 2] Detecting keypoints and matching (method={method})...")
    # 'auto' tries AKAZE first (better for illumination), falls back to SIFT
    # If classical methods find too few matches, auto-escalates to LoFTR
    kp1 = kp2 = good_matches = None

    if method == 'loftr':
        # Load LoFTR model (cached after first load)
        if _loftr_model is None:
            _loftr_model, _loftr_device = load_loftr()
        src_pts, dst_pts, confidence, num_matches = detect_and_match_loftr(
            img1_clean, img2_clean, _loftr_model, _loftr_device
        )
        kp1 = create_fake_keypoints(src_pts)
        kp2 = create_fake_keypoints(dst_pts)
        good_matches = loftr_to_opencv_matches(confidence)
        used_method = 'LoFTR (Deep Learning)'

    elif method == 'auto' or method == 'akaze' or method == 'sift':
        # Try classical methods first
        if method == 'auto' or method == 'akaze':
            try:
                src_pts, dst_pts, kp1, kp2, good_matches = detect_and_match_akaze(img1_clean, img2_clean)
                used_method = 'AKAZE'
            except Exception:
                src_pts, dst_pts, kp1, kp2, good_matches = detect_and_match_sift(img1_clean, img2_clean)
                used_method = 'SIFT (AKAZE fallback)'
        else:
            src_pts, dst_pts, kp1, kp2, good_matches = detect_and_match_sift(img1_clean, img2_clean)
            used_method = 'SIFT'

        # Auto-escalate to LoFTR if too few matches found
        if method == 'auto' and len(good_matches) < 20:
            print(f"         Only {len(good_matches)} matches with {used_method} — escalating to LoFTR...")
            try:
                if _loftr_model is None:
                    _loftr_model, _loftr_device = load_loftr()
                src_pts, dst_pts, confidence, num_matches = detect_and_match_loftr(
                    img1_clean, img2_clean, _loftr_model, _loftr_device
                )
                kp1 = create_fake_keypoints(src_pts)
                kp2 = create_fake_keypoints(dst_pts)
                good_matches = loftr_to_opencv_matches(confidence)
                used_method = 'LoFTR (auto-escalated)'
            except Exception as e:
                print(f"         LoFTR also failed ({e}) — keeping classical result")
    else:
        raise ValueError(f"Unknown method: {method}. Use 'sift', 'akaze', 'loftr', or 'auto'")

    result['method'] = used_method
    result['total_matches_before_filter'] = len(good_matches)
    print(f"         Algorithm used     : {used_method}")
    print(f"         Total matches found: {len(good_matches)}")

    # ── Step 3: Enforce uniform distribution ───────────────────────
    print("\n[Step 3] Enforcing uniform match distribution...")
    # Makes sure matches are spread evenly across image
    # Required by ISRO PS explicitly
    src_pts_uniform, dst_pts_uniform, good_matches_uniform = enforce_uniform_distribution(
        src_pts, dst_pts, good_matches, img1_clean.shape
    )
    result['total_matches_after_distribution'] = len(good_matches_uniform)
    print(f"         Matches after filter: {len(good_matches_uniform)}")

    # ── Step 4: Estimate homography with RANSAC ────────────────────
    print("\n[Step 4] Estimating homography with RANSAC...")
    M, mask, num_inliers, inlier_ratio = estimate_homography_ransac(
        src_pts_uniform, dst_pts_uniform
    )
    result['homography_matrix'] = M
    result['ransac_mask'] = mask
    print(f"         Inliers (correct)  : {num_inliers}")
    print(f"         Inlier ratio       : {inlier_ratio:.2%}")

    # ── Step 5: Warp source image ──────────────────────────────────
    print("\n[Step 5] Warping source image...")
    warped_img = warp_image(img1_clean, M, img2_clean.shape)
    result['registered_image'] = warped_img

    # ── Step 6: Sub-pixel refinement ───────────────────────────────
    print("\n[Step 6] Applying sub-pixel refinement (Phase Correlation + Lucas-Kanade)...")
    try:
        # Phase correlation — global shift correction
        shift, response = subpixel_refinement(
            img2_clean.astype(np.float64),
            warped_img.astype(np.float64)
        )
        result['subpixel_shift'] = shift
        result['subpixel_response'] = float(response)

        # Apply global shift if small
        max_shift = get_config()['refinement']['max_shift_pixels']
        if abs(shift[0]) < max_shift and abs(shift[1]) < max_shift:
            correction = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
            h, w = warped_img.shape
            warped_refined = cv2.warpAffine(warped_img, correction, (w, h))
            print(f"         Phase shift applied: ({shift[0]:.3f}, {shift[1]:.3f}) pixels")
        else:
            warped_refined = warped_img
            print(f"         Phase shift too large ({shift[0]:.1f}, {shift[1]:.1f}) — skipped")

        result['registered_image_refined'] = warped_refined

    except Exception as e:
        print(f"         Refinement warning: {e}")
        result['registered_image_refined'] = warped_img
        result['subpixel_shift'] = (0, 0)

    # ── Step 7: Compute metrics ────────────────────────────────────
    print("\n[Step 7] Computing metrics...")
    metrics = compute_all_metrics(
        src_pts_uniform, dst_pts_uniform, M, mask, img1_clean.shape
    )
    metrics['processing_time'] = time.time() - start_time
    result['metrics'] = metrics
    result['metrics_report'] = format_metrics_report(metrics, used_method)

    # Print full results to terminal
    print(result['metrics_report'])
    print(f"[Step 8] Generating visualizations...")
    print("="*50)
    print("   REGISTRATION COMPLETE")
    print("="*50 + "\n")

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
