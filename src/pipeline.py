"""
pipeline.py
-----------
The main registration pipeline.
Connects all modules together into one function call.

Confidence Router Logic:
1. Try AKAZE (fast, good for similar illumination)
2. If < 20 matches → try SIFT
3. If < 20 matches → escalate to LightGlue (deep learning)
4. If LightGlue also fails → return graceful failure result

Graceful failure:
- Never crashes — always returns a result dict
- result['success'] = True/False
- result['failure_reason'] tells user what went wrong
- result['confidence'] = 'high' / 'medium' / 'low' / 'failed'
"""

import numpy as np
import cv2
import time
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

from .config_loader import get_config
from .preprocess import preprocess_pair, preprocess_with_sun_angle, preprocess_pair_advanced
from .match import detect_and_match_sift, detect_and_match_akaze, enforce_uniform_distribution
from .loftr_match import detect_and_match_loftr, loftr_to_opencv_matches, create_fake_keypoints, load_loftr
from .structural_match import detect_and_match_structural
from .transform import (estimate_homography_ransac, warp_image, subpixel_refinement,
                         estimate_transform_adaptive, warp_image_adaptive, compute_rmse_adaptive)
from .metrics import compute_all_metrics, format_metrics_report, assess_reliability
from .visualize import draw_matches, create_checkerboard, create_side_by_side, create_difference_image

# Cached models — loaded once, reused across calls
_loftr_model = None
_loftr_device = None
_lightglue_extractor = None
_lightglue_matcher = None
_lightglue_device = None


def _try_lightglue(img1, img2, confidence_threshold=0.3):
    """Load and run LightGlue. Returns (src_pts, dst_pts, kp1, kp2, matches) or raises."""
    global _lightglue_extractor, _lightglue_matcher, _lightglue_device
    from .lightglue_match import (detect_and_match_lightglue,
                                   lightglue_to_opencv_keypoints,
                                   lightglue_to_opencv_matches,
                                   load_lightglue)
    if _lightglue_extractor is None:
        _lightglue_extractor, _lightglue_matcher, _lightglue_device = load_lightglue()

    src_pts, dst_pts, confidence, num = detect_and_match_lightglue(
        img1, img2, _lightglue_extractor, _lightglue_matcher, _lightglue_device,
        confidence_threshold=confidence_threshold
    )
    kp1 = lightglue_to_opencv_keypoints(src_pts)
    kp2 = lightglue_to_opencv_keypoints(dst_pts)
    matches = lightglue_to_opencv_matches(confidence)
    return src_pts, dst_pts, kp1, kp2, matches


def _try_loftr(img1, img2):
    """Load and run LoFTR. Returns (src_pts, dst_pts, kp1, kp2, matches) or raises."""
    global _loftr_model, _loftr_device
    if _loftr_model is None:
        _loftr_model, _loftr_device = load_loftr()
    src_pts, dst_pts, confidence, num = detect_and_match_loftr(
        img1, img2, _loftr_model, _loftr_device
    )
    kp1 = create_fake_keypoints(src_pts)
    kp2 = create_fake_keypoints(dst_pts)
    matches = loftr_to_opencv_matches(confidence)
    return src_pts, dst_pts, kp1, kp2, matches


def _score_result(src_pts, dst_pts, M, mask):
    """
    Scores a matching result to pick the best one.
    Higher score = better result.
    Score = inlier_count * inlier_ratio * spatial_score
    """
    if M is None or mask is None or len(mask) == 0:
        return 0.0
    inlier_count = int(mask.sum())
    inlier_ratio = inlier_count / len(mask)
    if inlier_count < 4:
        return 0.0
    # Spatial score
    from .metrics import compute_spatial_distribution_score
    import numpy as np
    inlier_mask = mask.ravel() == 1
    dst_inliers = dst_pts[inlier_mask].reshape(-1, 2)
    # Use a rough image shape estimate
    spatial = compute_spatial_distribution_score(dst_inliers, (1024, 1024))
    return inlier_count * inlier_ratio * spatial


def _run_method_safe(name, fn, *args, **kwargs):
    """
    Runs a matching function safely. Returns (src_pts, dst_pts, kp1, kp2, matches, score) or None.
    Uses estimate_transform_adaptive for scoring so affine is used for small match sets.
    """
    try:
        src_pts, dst_pts, kp1, kp2, matches = fn(*args, **kwargs)
        if len(matches) < 4:
            return None
        # Use adaptive transform for scoring — avoids penalising good matches
        # that happen to be few in number (affine handles these correctly)
        try:
            tr = estimate_transform_adaptive(
                src_pts[:min(len(src_pts), 500)],
                dst_pts[:min(len(dst_pts), 500)],
                reproj_threshold=3.0
            )
            score = _score_result(src_pts, dst_pts, tr['transform_matrix'], tr['inlier_mask'])
        except Exception:
            score = len(matches) * 0.1  # fallback score
        print(f"         [{name}] {len(matches)} matches, score={score:.2f}")
        return src_pts, dst_pts, kp1, kp2, matches, score
    except Exception as e:
        print(f"         [{name}] failed: {e}")
        return None
    global _lightglue_extractor, _lightglue_matcher, _lightglue_device
    from .lightglue_match import (detect_and_match_lightglue,
                                   lightglue_to_opencv_keypoints,
                                   lightglue_to_opencv_matches,
                                   load_lightglue)
    if _lightglue_extractor is None:
        _lightglue_extractor, _lightglue_matcher, _lightglue_device = load_lightglue()

    src_pts, dst_pts, confidence, num = detect_and_match_lightglue(
        img1, img2, _lightglue_extractor, _lightglue_matcher, _lightglue_device
    )
    kp1 = lightglue_to_opencv_keypoints(src_pts)
    kp2 = lightglue_to_opencv_keypoints(dst_pts)
    matches = lightglue_to_opencv_matches(confidence)
    return src_pts, dst_pts, kp1, kp2, matches


def _try_loftr(img1, img2):
    """Load and run LoFTR. Returns (src_pts, dst_pts, kp1, kp2, matches) or raises."""
    global _loftr_model, _loftr_device
    if _loftr_model is None:
        _loftr_model, _loftr_device = load_loftr()
    src_pts, dst_pts, confidence, num = detect_and_match_loftr(
        img1, img2, _loftr_model, _loftr_device
    )
    kp1 = create_fake_keypoints(src_pts)
    kp2 = create_fake_keypoints(dst_pts)
    matches = loftr_to_opencv_matches(confidence)
    return src_pts, dst_pts, kp1, kp2, matches


def _make_failure_result(reason, processing_time=0.0):
    """Returns a standardised failure result dict."""
    return {
        'success': False,
        'failure_reason': reason,
        'confidence': 'failed',
        'method': 'N/A',
        'metrics': {
            'rmse': float('inf'),
            'rmse_x': float('inf'),
            'rmse_y': float('inf'),
            'inlier_count': 0,
            'total_matches': 0,
            'inlier_ratio': 0.0,
            'spatial_score': 0.0,
            'processing_time': processing_time
        },
        'metrics_report': f"\n[REGISTRATION FAILED]\nReason: {reason}\n",
        'total_matches_before_filter': 0,
        'total_matches_after_distribution': 0,
    }


def run_pipeline(img1, img2, method='auto', max_size=None,
                 src_sun_elevation=None, ref_sun_elevation=None,
                 src_metadata=None, ref_metadata=None):
    """
    Complete image registration pipeline with graceful failure handling.

    Parameters:
        img1: source image (numpy array, grayscale)
        img2: reference image (numpy array, grayscale)
        method: 'auto' | 'sift' | 'akaze' | 'loftr' | 'lightglue'
        max_size: max image dimension (default from config)
        src_sun_elevation: sun elevation of source in degrees (from XML)
        ref_sun_elevation: sun elevation of reference in degrees (from XML)
        src_metadata: full metadata dict from XML (optional)
        ref_metadata: full metadata dict from XML (optional)

    Returns:
        result dict with:
            success: True/False
            confidence: 'high' / 'medium' / 'low' / 'failed'
            method: algorithm used
            metrics: dict with RMSE, inlier count, etc.
            registered_image_refined: aligned source image
            match_visualization: image with match lines
            checkerboard: checkerboard blend
            side_by_side: side by side comparison
            difference_image: pixel difference image
    """
    start_time = time.time()

    print("\n" + "=" * 50)
    print("   LUNAR IMAGE REGISTRATION — PS166 SIH 2026")
    print("=" * 50)

    if max_size is None:
        max_size = get_config()['preprocessing']['max_size']

    # ── Step 0: Input validation ────────────────────────────────────
    print("\n[Step 0] Validating inputs...")
    if img1 is None or img2 is None:
        return _make_failure_result("One or both images could not be loaded.")
    if img1.size == 0 or img2.size == 0:
        return _make_failure_result("One or both images are empty.")
    if img1.mean() < 2:
        return _make_failure_result("Source image appears all-black — no features can be detected.")
    if img2.mean() < 2:
        return _make_failure_result("Reference image appears all-black — no features can be detected.")
    print("         Input validation passed")

    # ── Step 1: Preprocess ─────────────────────────────────────────
    print("\n[Step 1] Preprocessing images...")
    try:
        use_advanced = (src_sun_elevation is not None or ref_sun_elevation is not None)
        if use_advanced:
            print("         Using advanced illumination-aware preprocessing")
            img1_clean, img2_clean, mask1, mask2, shadow_info = preprocess_pair_advanced(
                img1, img2, max_size,
                src_sun_elevation=src_sun_elevation,
                ref_sun_elevation=ref_sun_elevation,
                use_histogram_matching=True,
                use_shadow_mask=True
            )
        else:
            img1_clean, img2_clean = preprocess_pair(img1, img2, max_size)
            mask1 = mask2 = None
            shadow_info = {}

        print(f"         Source : {img1_clean.shape}, mean={img1_clean.mean():.1f}")
        print(f"         Reference: {img2_clean.shape}, mean={img2_clean.mean():.1f}")

    except Exception as e:
        return _make_failure_result(f"Preprocessing failed: {e}",
                                    time.time() - start_time)

    # ── Step 2: Feature matching with confidence router ────────────
    print(f"\n[Step 2] Feature matching (method={method})...")

    src_pts = dst_pts = kp1 = kp2 = good_matches = None
    used_method = 'unknown'
    escalation_log = []

    try:
        if method == 'lightglue':
            src_pts, dst_pts, kp1, kp2, good_matches = _try_lightglue(img1_clean, img2_clean)
            used_method = 'DISK + LightGlue'

        elif method == 'loftr':
            src_pts, dst_pts, kp1, kp2, good_matches = _try_loftr(img1_clean, img2_clean)
            used_method = 'LoFTR'

        elif method == 'pc-sift':
            src_pts, dst_pts, kp1, kp2, good_matches = detect_and_match_structural(
                img1_clean, img2_clean, detector='sift')
            used_method = 'PC-SIFT (Phase Congruency)'

        elif method in ('auto', 'akaze', 'sift'):
            # --- Classical tier ---
            if method in ('auto', 'akaze'):
                # AUTO MODE: Run ALL methods, pick the best scoring one
                print("         Running all algorithms — will pick best result...")

                candidates = []

                # Try AKAZE
                r = _run_method_safe('AKAZE', detect_and_match_akaze, img1_clean, img2_clean)
                if r: candidates.append(('AKAZE', r))

                # Try SIFT
                r = _run_method_safe('SIFT', detect_and_match_sift, img1_clean, img2_clean)
                if r: candidates.append(('SIFT', r))

                # Try LightGlue
                r = _run_method_safe('LightGlue', _try_lightglue, img1_clean, img2_clean)
                if r: candidates.append(('LightGlue', r))

                # Try Phase Congruency + SIFT (illumination-invariant structural matching)
                # Particularly useful for medium/hard cases with different sun angles
                r = _run_method_safe('PC-SIFT', detect_and_match_structural,
                                     img1_clean, img2_clean, 'sift')
                if r: candidates.append(('PC-SIFT', r))

                # Try LoFTR only if others are weak
                best_score_so_far = max((c[1][5] for c in candidates), default=0)
                if best_score_so_far < 5.0:
                    r = _run_method_safe('LoFTR', _try_loftr, img1_clean, img2_clean)
                    if r: candidates.append(('LoFTR', r))

                if not candidates:
                    raise ValueError("All algorithms failed to find matches")

                # Pick the best scoring candidate
                best_name, best_result = max(candidates, key=lambda x: x[1][5])
                src_pts, dst_pts, kp1, kp2, good_matches, best_score = best_result
                used_method = best_name
                escalation_log = [f"{n}: score={r[5]:.2f}" for n, r in candidates]
                print(f"         Best algorithm: {best_name} (score={best_score:.2f})")

            else:
                src_pts, dst_pts, kp1, kp2, good_matches = detect_and_match_sift(
                    img1_clean, img2_clean)
                used_method = 'SIFT'
                escalation_log.append(f"SIFT: {len(good_matches)} matches")

            # method == 'sift' explicit case handled above
        else:
            return _make_failure_result(
                f"Unknown method '{method}'. Use: auto, sift, akaze, loftr, lightglue, pc-sift",
                time.time() - start_time)

    except Exception as e:
        return _make_failure_result(
            f"Feature matching failed: {e}. "
            f"Images may not overlap or are from different lunar regions.",
            time.time() - start_time)

    if good_matches is None or len(good_matches) < 4:
        return _make_failure_result(
            f"Not enough matches found ({len(good_matches) if good_matches else 0}). "
            f"Try: (1) different image pair, (2) check images overlap the same lunar region, "
            f"(3) try 'lightglue' algorithm.",
            time.time() - start_time)

    print(f"         Algorithm: {used_method}")
    print(f"         Matches: {len(good_matches)}")
    if escalation_log:
        print(f"         Escalation: {' → '.join(escalation_log)}")

    result = {
        'success': True,
        'method': used_method,
        'escalation_log': escalation_log,
        'total_matches_before_filter': len(good_matches)
    }

    # ── Step 3: Uniform distribution enforcement ───────────────────
    print("\n[Step 3] Enforcing uniform spatial distribution...")
    src_pts_u, dst_pts_u, good_matches_u = enforce_uniform_distribution(
        src_pts, dst_pts, good_matches, img1_clean.shape
    )
    result['total_matches_after_distribution'] = len(good_matches_u)
    print(f"         After distribution filter: {len(good_matches_u)}")

    # ── Step 4: Homography estimation (RANSAC) ─────────────────────
    print("\n[Step 4] Estimating transform (adaptive: homography or affine)...")
    try:
        transform_result = estimate_transform_adaptive(src_pts_u, dst_pts_u)
        M           = transform_result['transform_matrix']
        mask        = transform_result['inlier_mask']
        num_inliers = transform_result['num_inliers']
        inlier_ratio= transform_result['inlier_ratio']
        ttype       = transform_result['transform_type']
    except Exception as e:
        return _make_failure_result(
            f"Geometric estimation failed: {e}. "
            f"Found matches but couldn't estimate transformation — images may be too different.",
            time.time() - start_time)

    result['homography_matrix'] = M
    result['transform_type']    = ttype
    result['ransac_mask']       = mask
    print(f"         Transform type : {ttype.upper()}")
    print(f"         Inliers: {num_inliers} / {len(good_matches_u)}")
    print(f"         Inlier ratio: {inlier_ratio:.2%}")

    if num_inliers < 4:
        return _make_failure_result(
            f"Only {num_inliers} inliers after outlier rejection. "
            f"Matches found but geometrically inconsistent — images likely from different regions "
            f"or pre-alignment insufficient. Try without geo-assist or use a different pair.",
            time.time() - start_time)

    # ── Step 5: Image warping ──────────────────────────────────────
    print("\n[Step 5] Warping source image...")
    try:
        warped_img = warp_image_adaptive(img1_clean, transform_result, img2_clean.shape)
        result['registered_image'] = warped_img
    except Exception as e:
        return _make_failure_result(f"Warping failed: {e}", time.time() - start_time)

    if warped_img.mean() < 5:
        return _make_failure_result(
            "Warped image is mostly black — transform is degenerate. "
            "Try a different algorithm or image pair.",
            time.time() - start_time)

    # ── Step 6: Sub-pixel refinement ──────────────────────────────
    print("\n[Step 6] Sub-pixel refinement (phase correlation)...")
    try:
        shift, response = subpixel_refinement(
            img2_clean.astype(np.float64),
            warped_img.astype(np.float64)
        )
        result['subpixel_shift'] = shift
        result['subpixel_response'] = float(response)

        max_shift = get_config()['refinement']['max_shift_pixels']
        if abs(shift[0]) < max_shift and abs(shift[1]) < max_shift:
            correction = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
            h, w = warped_img.shape
            warped_refined = cv2.warpAffine(warped_img, correction, (w, h))
            print(f"         Shift corrected: ({shift[0]:.3f}, {shift[1]:.3f}) px")
        else:
            warped_refined = warped_img
            print(f"         Shift too large — skipped")

        result['registered_image_refined'] = warped_refined

    except Exception as e:
        result['registered_image_refined'] = warped_img
        result['subpixel_shift'] = (0, 0)
        print(f"         Refinement skipped: {e}")

    # ── Step 7: Metrics ────────────────────────────────────────────
    print("\n[Step 7] Computing evaluation metrics...")
    metrics = compute_all_metrics(
        src_pts_u, dst_pts_u, M, mask, img1_clean.shape,
        transform_result=transform_result
    )
    metrics['processing_time'] = time.time() - start_time
    result['metrics'] = metrics
    result['metrics_report'] = format_metrics_report(metrics, used_method)

    # ── Confidence / reliability — single authoritative source ─────
    # metrics already contains degenerate_fit, confidence, reliability_reason
    # from assess_reliability() called inside compute_all_metrics().
    # Copy them to the top-level result so the UI can access them directly
    # without digging into the nested metrics dict.
    result['confidence']         = metrics['confidence']
    result['degenerate_fit']     = metrics['degenerate_fit']
    result['reliability_reason'] = metrics['reliability_reason']

    print(result['metrics_report'])
    print(f"         Confidence: {result['confidence'].upper()}")
    if result['degenerate_fit']:
        print(f"         ⚠️  DEGENERATE FIT: {result['reliability_reason']}")

    # ── Step 8: Visualizations ─────────────────────────────────────
    print("[Step 8] Generating visualizations...")
    try:
        result['match_visualization'] = draw_matches(
            img1_clean, kp1, img2_clean, kp2, good_matches_u, mask)
        result['checkerboard'] = create_checkerboard(
            img2_clean, result['registered_image_refined'])
        result['side_by_side'] = create_side_by_side(
            img2_clean, result['registered_image_refined'],
            "Reference Image", "Registered Source")
        result['difference_image'] = create_difference_image(
            img2_clean, result['registered_image_refined'])
    except Exception as e:
        print(f"         Visualization warning: {e}")

    print("=" * 50)
    print("   REGISTRATION COMPLETE")
    print("=" * 50 + "\n")

    return result


# ── Cross-source registration (OHRC vs LRO NAC / SELENE) ─────────────────────
# This is a NEW entry point that does NOT modify run_pipeline().
# It handles the GSD normalisation and cropping specific to cross-source
# pairs, then delegates entirely to run_pipeline() for all matching,
# RANSAC, sub-pixel refinement, and confidence gating logic.

def register_cross_source(
    source_image,
    source_meta: dict,
    reference_meta: dict,
    method: str = 'auto',
    max_size: int = 1024,
) -> dict:
    """
    Registers a Chandrayaan-2 source image (OHRC browse PNG or .img patch)
    against an external reference tile (LRO NAC / WAC GeoTIFF loaded by
    lro_loader.load_geotiff_reference()).

    Steps performed here (not repeated inside run_pipeline):
      1. Crop reference tile to the overlap region + 15 % margin.
      2. Normalise both images to the coarser GSD (downsample finer image).
      3. Delegate to run_pipeline() — preprocessing, matcher router, RANSAC,
         sub-pixel refinement, metrics, and confidence/degenerate_fit gating
         from assess_reliability() are all reused without duplication.

    Parameters:
        source_image   : np.ndarray — OHRC image (grayscale uint8)
        source_meta    : dict from ohrc_loader.parse_ohrc_xml() or
                         extract_browse_png_from_zip() — must contain
                         corner lat/lon keys and pixel_resolution_m
        reference_meta : dict from lro_loader.load_geotiff_reference() —
                         must contain image, transform, and corner lat/lon
        method         : matching algorithm ('auto', 'sift', 'akaze',
                         'loftr', 'lightglue')
        max_size       : max image dimension passed to run_pipeline

    Returns:
        result dict from run_pipeline() with two extra keys added:
            cross_source_crop_original_shape : tuple — reference shape before crop
            cross_source_crop_final_shape    : tuple — reference shape after crop
            cross_source_target_res_m        : float — GSD used for both images
    """
    from .overlap import crop_reference_to_overlap, normalize_gsd

    print("\n" + "=" * 50)
    print("   CROSS-SOURCE REGISTRATION (OHRC vs LRO)")
    print("=" * 50)

    # ── Step CS-1: Crop reference to overlap region ────────────────
    original_ref_shape = reference_meta['image'].shape
    try:
        cropped_ref, updated_ref_corners = crop_reference_to_overlap(
            reference_meta, source_meta, margin_fraction=0.15
        )
    except ValueError as e:
        # Return a structured failure so the UI can display it cleanly
        return _make_failure_result(str(e))

    print(f"[CS-1] Crop: {original_ref_shape} → {cropped_ref.shape}")

    # ── Step CS-2: GSD normalisation ──────────────────────────────
    src_res = source_meta.get('pixel_resolution_m',
                source_meta.get('pixel_resolution_m', 0.25))
    ref_res = reference_meta.get('pixel_resolution_m',
                reference_meta.get('resolution_m_per_px', 1.0))

    # Always normalise to the COARSER resolution (larger m/px value)
    target_res = max(src_res, ref_res)

    source_norm  = normalize_gsd(source_image,  src_res, target_res)
    ref_norm     = normalize_gsd(cropped_ref,   ref_res, target_res)

    print(f"[CS-2] Source : {source_image.shape} @ {src_res:.3f} m/px "
          f"→ {source_norm.shape} @ {target_res:.3f} m/px")
    print(f"[CS-2] Ref    : {cropped_ref.shape} @ {ref_res:.3f} m/px "
          f"→ {ref_norm.shape} @ {target_res:.3f} m/px")

    # ── Step CS-3: Delegate entirely to run_pipeline() ─────────────
    # Pass sun elevation from source metadata if available so the
    # illumination-aware CLAHE path activates automatically.
    sun_el = source_meta.get('sun_elevation')

    result = run_pipeline(
        source_norm,
        ref_norm,
        method=method,
        max_size=max_size,
        src_sun_elevation=sun_el,
        ref_sun_elevation=None,       # LRO tiles don't have a sun_elevation field
        src_metadata=source_meta,
        ref_metadata=reference_meta,
    )

    # Attach cross-source provenance to the result
    result['cross_source_crop_original_shape'] = original_ref_shape
    result['cross_source_crop_final_shape']    = cropped_ref.shape
    result['cross_source_target_res_m']        = target_res
    result['cross_source_src_label']           = source_meta.get('source_label', 'OHRC')
    result['cross_source_ref_label']           = reference_meta.get('source_label', 'LRO_reference')

    # ── Step CS-4: Log summary ──────────────────────────────────────
    m = result.get('metrics', {})
    print("\n[CS-4] Cross-source registration summary:")
    print(f"        Crop reduction  : {original_ref_shape} → {cropped_ref.shape}")
    print(f"        Target GSD      : {target_res:.3f} m/px")
    print(f"        Matcher used    : {result.get('method', 'N/A')}")
    print(f"        Inlier count    : {m.get('inlier_count', 'N/A')}")
    print(f"        Inlier ratio    : {m.get('inlier_ratio', 0):.2%}")
    print(f"        RMSE            : {m.get('rmse', 'N/A')}")
    print(f"        Spatial score   : {m.get('spatial_score', 'N/A')}")
    print(f"        Confidence      : {result.get('confidence', 'N/A')}")
    print(f"        Degenerate fit  : {result.get('degenerate_fit', 'N/A')}")

    if not result.get('success', True):
        print(f"        Failure reason  : {result.get('failure_reason', '')}")
    elif result.get('degenerate_fit'):
        print(f"        ⚠️  {result.get('reliability_reason', '')}")

    # Debug info if result is very poor
    if m.get('inlier_count', 0) < 4:
        print("\n[CS-4] DEBUG — very few inliers, likely causes:")
        print(f"        Source corners  : "
              f"UL ({source_meta.get('upper_left_lat', '?'):.4f}, "
              f"{source_meta.get('upper_left_lon', '?'):.4f})")
        print(f"        Ref corners     : "
              f"UL ({reference_meta.get('upper_left_lat', '?'):.4f}, "
              f"{reference_meta.get('upper_left_lon', '?'):.4f})")
        print(f"        Source norm shape : {source_norm.shape}, "
              f"mean={source_norm.mean():.1f}")
        print(f"        Ref norm shape    : {ref_norm.shape}, "
              f"mean={ref_norm.mean():.1f}")

    return result


# ── IIRS hyperspectral registration ──────────────────────────────────────────
# Synthesizes a panchromatic-equivalent band from the IIRS hyperspectral cube,
# then delegates entirely to the existing crop / GSD-normalize / match / RANSAC /
# metrics / confidence-gating pipeline.  No logic is duplicated here.

def register_iirs(
    source_iirs_cube: "np.ndarray",
    iirs_meta: dict,
    reference_image: "np.ndarray",
    reference_meta: dict,
    method: str = 'auto',
    max_size: int = 1024,
    band_range_nm: tuple = (800, 1000),
) -> dict:
    """
    Registers a Chandrayaan-2 IIRS scene against a reference image
    (OHRC browse PNG or LRO NAC GeoTIFF) by synthesizing a single-channel
    panchromatic-equivalent band from the hyperspectral cube.

    Steps performed here (not repeated inside run_pipeline):
      1. Synthesize panchromatic band from IIRS cube via equal-weighted
         band averaging in band_range_nm window.
      2. Crop reference to IIRS footprint + 15% margin.
      3. Normalise both images to the coarser GSD (always downsample finer).
      4. Delegate to run_pipeline() — preprocessing, matcher router, RANSAC,
         sub-pixel refinement, metrics, and Task-1 confidence gating are all
         reused without duplication.

    ⚠️  Approximation note:
    The synthesized panchromatic band is an equal-weighted spectral average,
    not a radiometrically precise OHRC/NAC simulation.  This is labelled
    clearly in iirs_loader.synthesize_panchromatic() and in the UI.

    Parameters:
        source_iirs_cube : np.ndarray (num_bands, num_rows, num_cols)
                           from iirs_loader.load_iirs_cube()
        iirs_meta        : dict from iirs_loader.parse_iirs_label()
                           must contain: wavelengths_nm, resolution_m_per_px,
                           and the corner lat/lon keys used by overlap.py
        reference_image  : np.ndarray (rows, cols) — OHRC or LRO NAC image
        reference_meta   : dict from ohrc_loader or lro_loader — must contain
                           resolution_m_per_px and corner lat/lon keys
        method           : matching algorithm ('auto', 'sift', 'akaze',
                           'loftr', 'lightglue')
        max_size         : max image dimension for run_pipeline()
        band_range_nm    : (min_nm, max_nm) window for band selection in
                           synthesize_panchromatic(); default (800,1000) nm
                           targets the shortest-wavelength IIRS bands

    Returns:
        result dict from run_pipeline() with extra keys:
            iirs_pan_shape          : shape of synthesized pan band
            iirs_pan_value_range    : (min, max) before uint8 normalisation
            iirs_bands_used         : list of band indices selected
            iirs_wavelength_range_nm: (min_nm, max_nm) of selected bands
            iirs_target_res_m       : GSD used for both images
            cross_source_crop_original_shape : reference shape before crop
            cross_source_crop_final_shape    : reference shape after crop
    """
    import numpy as np
    from .iirs_loader import synthesize_panchromatic
    from .overlap import crop_reference_to_overlap, normalize_gsd

    print("\n" + "=" * 50)
    print("   IIRS HYPERSPECTRAL REGISTRATION")
    print("=" * 50)

    # ── Step I-1: Synthesize panchromatic band ─────────────────────
    wavelengths_nm = iirs_meta['wavelengths_nm']

    # Track bands used for result provenance
    band_indices = [
        i for i, wl in enumerate(wavelengths_nm)
        if band_range_nm[0] <= wl <= band_range_nm[1]
    ]
    if not band_indices:
        # Fallback: shortest-wavelength third (mirrors synthesize_panchromatic logic)
        min_wl, max_wl = min(wavelengths_nm), max(wavelengths_nm)
        fallback_max = min_wl + (max_wl - min_wl) / 3
        band_indices = [i for i, wl in enumerate(wavelengths_nm) if wl <= fallback_max]

    pan_band = synthesize_panchromatic(
        source_iirs_cube, wavelengths_nm, band_range_nm
    )

    # Capture value range before normalisation for reporting
    # (synthesize_panchromatic already prints this, but also store in result)
    selected_raw = source_iirs_cube[band_indices, :, :].astype(np.float32)
    raw_min = float(selected_raw.min())
    raw_max = float(selected_raw.max())
    sel_wls = [wavelengths_nm[i] for i in band_indices]

    print(f"[I-1] Pan band shape: {pan_band.shape}  "
          f"value range before norm: [{raw_min:.3f}, {raw_max:.3f}]")

    # ── Step I-2: Crop reference to IIRS footprint ─────────────────
    original_ref_shape = reference_image.shape
    ref_meta_with_image = dict(reference_meta)
    ref_meta_with_image['image'] = reference_image

    try:
        cropped_ref, _ = crop_reference_to_overlap(
            ref_meta_with_image, iirs_meta, margin_fraction=0.15
        )
    except ValueError as e:
        return _make_failure_result(str(e))

    print(f"[I-2] Reference crop: {original_ref_shape} → {cropped_ref.shape}")

    # ── Step I-3: GSD normalisation ────────────────────────────────
    iirs_res = iirs_meta.get('resolution_m_per_px',
                iirs_meta.get('pixel_resolution_m', 80.0))
    ref_res  = reference_meta.get('pixel_resolution_m',
                reference_meta.get('resolution_m_per_px', 0.25))

    target_res = max(iirs_res, ref_res)   # always towards coarser
    source_norm = normalize_gsd(pan_band,   iirs_res, target_res)
    ref_norm    = normalize_gsd(cropped_ref, ref_res,  target_res)

    print(f"[I-3] IIRS pan : {pan_band.shape} @ {iirs_res:.1f} m/px "
          f"→ {source_norm.shape} @ {target_res:.1f} m/px")
    print(f"[I-3] Ref      : {cropped_ref.shape} @ {ref_res:.3f} m/px "
          f"→ {ref_norm.shape} @ {target_res:.1f} m/px")

    # ── Step I-4: Delegate to run_pipeline() ──────────────────────
    result = run_pipeline(
        source_norm,
        ref_norm,
        method=method,
        max_size=max_size,
        src_sun_elevation=None,   # IIRS doesn't expose sun_elevation in same way
        ref_sun_elevation=None,
    )

    # ── Attach IIRS-specific provenance ────────────────────────────
    result['iirs_pan_shape']           = pan_band.shape
    result['iirs_pan_value_range']     = (raw_min, raw_max)
    result['iirs_bands_used']          = band_indices
    result['iirs_wavelength_range_nm'] = (
        (min(sel_wls), max(sel_wls)) if sel_wls else (None, None)
    )
    result['iirs_target_res_m']                  = target_res
    result['cross_source_crop_original_shape']   = original_ref_shape
    result['cross_source_crop_final_shape']      = cropped_ref.shape

    # ── Summary log ────────────────────────────────────────────────
    m = result.get('metrics', {})
    print("\n[IIRS] Registration summary:")
    print(f"       Wavelength range used : {result['iirs_wavelength_range_nm']} nm")
    print(f"       Pan band shape        : {result['iirs_pan_shape']}")
    print(f"       Raw value range       : [{raw_min:.3f}, {raw_max:.3f}]")
    print(f"       Crop reduction        : {original_ref_shape} → {cropped_ref.shape}")
    print(f"       Target GSD            : {target_res:.2f} m/px")
    print(f"       Matcher used          : {result.get('method', 'N/A')}")
    print(f"       Inlier count          : {m.get('inlier_count', 'N/A')}")
    print(f"       Inlier ratio          : {m.get('inlier_ratio', 0):.2%}")
    print(f"       RMSE                  : {m.get('rmse', 'N/A')}")
    print(f"       Confidence            : {result.get('confidence', 'N/A')}")
    print(f"       Degenerate fit        : {result.get('degenerate_fit', 'N/A')}")

    if not result.get('success', True):
        print(f"       Failure reason        : {result.get('failure_reason', '')}")
    elif result.get('degenerate_fit'):
        print(f"       ⚠️  {result.get('reliability_reason', '')}")

    return result


# ── Geo-assisted registration ─────────────────────────────────────────────────
# Uses OHRC .csv ground-coordinate files to compute a coarse pre-alignment
# before feature matching, turning a blind search into a guided refinement.

def register_with_geo_assist(
    source_image: "np.ndarray",
    source_zip_path: str,
    reference_image: "np.ndarray",
    reference_zip_path: str,
    method: str = 'auto',
    max_size: int = 1024,
    max_distance_deg: float = 0.001,
    source_meta: dict = None,
    reference_meta: dict = None,
) -> dict:
    """
    Full pipeline: coarse geo-based pre-alignment using .csv files,
    then standard feature-matching refinement on the pre-aligned pair.

    Falls back to the existing standard run_pipeline() without pre-alignment
    if geo-assist fails for any reason — never blocks a result.

    Composition order for final homography:
        fine_H maps pre-warped source → reference
        coarse_H maps original source → pre-warped source frame
        final_H = fine_H @ coarse_H maps original source → reference

    This composition is verified against the easy case (near-identity
    expected) in Step 4 of this task.

    Parameters:
        source_image        : grayscale np.ndarray
        source_zip_path     : path to source OHRC .zip (for .csv extraction)
        reference_image     : grayscale np.ndarray
        reference_zip_path  : path to reference OHRC .zip (for .csv extraction)
        method              : matching algorithm ('auto', 'sift', etc.)
        max_size            : max image dimension for run_pipeline()
        max_distance_deg    : geo-match tolerance in degrees (~30m default)

    Returns:
        result dict from run_pipeline() plus extra keys:
            coarse_geo_prealignment_used : bool
            coarse_geo_residual_rmse     : float (px) or None
            coarse_geo_n_points          : int
            coarse_geo_homography        : list (3x3) or None
            final_homography_composed    : bool
    """
    import numpy as np
    import cv2
    from .geo_align import (load_csv_from_zip, find_common_ground_points,
                             estimate_coarse_transform, warp_source_coarse)

    print("\n" + "=" * 50)
    print("   GEO-ASSISTED REGISTRATION")
    print("=" * 50)

    geo_used = False
    coarse_H = None
    coarse_rmse = None
    n_geo_pts = 0

    try:
        # ── Step G-1: Load CSV geolocation files ──────────────────
        print("\n[G-1] Loading CSV geolocation files...")
        csv1 = load_csv_from_zip(source_zip_path)
        csv2 = load_csv_from_zip(reference_zip_path)

        if csv1 is None or csv2 is None:
            raise ValueError("Could not find .csv files in one or both zips")

        # ── Step G-2: Find geo-linked pixel correspondences ────────
        print("\n[G-2] Finding geo-linked pixel correspondences...")

        # Scale CSV pixel coordinates from full-res space to browse PNG space.
        # The CSV maps positions in the full-resolution .img file, but we are
        # registering the browse PNG which is a downsampled version.
        # parse_ohrc_xml() already extracted the full-res dimensions — reuse them.
        import numpy as np
        src_full_w = (source_meta or {}).get('samples', 12000)
        src_full_h = (source_meta or {}).get('lines',   101075)
        ref_full_w = (reference_meta or {}).get('samples', 12000)
        ref_full_h = (reference_meta or {}).get('lines',   101075)

        src_browse_h, src_browse_w = source_image.shape[:2]
        ref_browse_h, ref_browse_w = reference_image.shape[:2]

        # Make copies so we don't mutate the originals
        csv1_scaled = csv1.copy()
        csv2_scaled = csv2.copy()

        sx1 = src_browse_w / src_full_w
        sy1 = src_browse_h / src_full_h
        sx2 = ref_browse_w / ref_full_w
        sy2 = ref_browse_h / ref_full_h

        csv1_scaled[:, 2] *= sx1   # pixel_col
        csv1_scaled[:, 3] *= sy1   # scan_line
        csv2_scaled[:, 2] *= sx2
        csv2_scaled[:, 3] *= sy2

        print(f"[G-2] Scale correction:")
        print(f"      Source : full {src_full_w}×{src_full_h} → browse {src_browse_w}×{src_browse_h} "
              f"(sx={sx1:.4f} sy={sy1:.4f})")
        print(f"      Reference: full {ref_full_w}×{ref_full_h} → browse {ref_browse_w}×{ref_browse_h} "
              f"(sx={sx2:.4f} sy={sy2:.4f})")

        img1_pts, img2_pts = find_common_ground_points(
            csv1_scaled, csv2_scaled, max_distance_deg=max_distance_deg
        )
        n_geo_pts = len(img1_pts)

        if n_geo_pts < 4:
            raise ValueError(
                f"Only {n_geo_pts} geo-linked points found within "
                f"{max_distance_deg}° tolerance. "
                f"Images may not overlap geographically, or max_distance_deg is too strict."
            )

        # ── Step G-3: Estimate coarse homography from geo-points ───
        print("\n[G-3] Estimating coarse homography from geo-points...")
        coarse_H, coarse_rmse = estimate_coarse_transform(img1_pts, img2_pts)

        # ── Step G-4: Pre-warp source image ────────────────────────
        print("\n[G-4] Pre-warping source onto reference frame...")
        pre_warped = warp_source_coarse(
            source_image, coarse_H,
            output_shape=(reference_image.shape[1], reference_image.shape[0])
        )

        geo_used = True
        print(f"\n[G-4] ✅ Geo pre-alignment succeeded — "
              f"coarse residual {coarse_rmse:.1f}px — "
              f"running feature matching on pre-aligned pair")

        # ── Step G-5: Standard pipeline on pre-warped pair ─────────
        # Pass sun elevations so preprocessing adapts to lighting conditions
        src_sun = (source_meta or {}).get('sun_elevation')
        ref_sun = (reference_meta or {}).get('sun_elevation')

        result = run_pipeline(
            pre_warped, reference_image,
            method=method, max_size=max_size,
            src_sun_elevation=src_sun,
            ref_sun_elevation=ref_sun,
            src_metadata=source_meta,
            ref_metadata=reference_meta,
        )

        # ── Step G-6: Compose homographies ────────────────────────
        # fine_H maps pre-warped source → reference
        # coarse_H maps original source → pre-warped source frame
        # final_H = fine_H @ coarse_H maps original source → reference
        fine_H = result.get('homography_matrix')
        if fine_H is not None and coarse_H is not None:
            composed_H = fine_H @ coarse_H

            # Sanity check: verify composition by testing on geo-inlier points
            # Project original source geo-points through composed H
            # and compare to reference geo-points
            if n_geo_pts >= 4:
                test_pts = img1_pts[:min(20, n_geo_pts)].reshape(-1, 1, 2).astype(np.float32)
                ref_pts  = img2_pts[:min(20, n_geo_pts)].reshape(-1, 1, 2).astype(np.float32)
                composed_proj = cv2.perspectiveTransform(test_pts, composed_H).reshape(-1, 2)
                composed_err  = np.linalg.norm(composed_proj - ref_pts.reshape(-1, 2), axis=1)
                composed_rmse = float(np.sqrt(np.mean(composed_err ** 2)))
                print(f"[G-6] Composed homography sanity check: "
                      f"RMSE on geo-points = {composed_rmse:.2f}px "
                      f"(should be ≤ coarse residual {coarse_rmse:.1f}px)")

            result['homography_matrix'] = composed_H
            result['final_homography_composed'] = True
        else:
            result['final_homography_composed'] = False

    except Exception as e:
        print(f"\n[geo_assist] ⚠️ Geo pre-alignment failed: {e}")
        print(f"[geo_assist] Falling back to standard pipeline without pre-alignment")
        src_sun = (source_meta or {}).get('sun_elevation')
        ref_sun = (reference_meta or {}).get('sun_elevation')
        result = run_pipeline(
            source_image, reference_image,
            method=method, max_size=max_size,
            src_sun_elevation=src_sun,
            ref_sun_elevation=ref_sun,
            src_metadata=source_meta,
            ref_metadata=reference_meta,
        )
        result['final_homography_composed'] = False

    # ── Attach geo-assist provenance ───────────────────────────────
    result['coarse_geo_prealignment_used'] = geo_used
    result['coarse_geo_residual_rmse']     = coarse_rmse
    result['coarse_geo_n_points']          = n_geo_pts
    result['coarse_geo_homography']        = coarse_H.tolist() if coarse_H is not None else None

    # ── Summary ────────────────────────────────────────────────────
    m = result.get('metrics', {})
    print("\n[GEO] Registration summary:")
    print(f"      Geo pre-alignment used : {geo_used}")
    print(f"      Geo points found       : {n_geo_pts}")
    print(f"      Coarse residual RMSE   : {coarse_rmse:.2f}px" if coarse_rmse else "      Coarse RMSE : N/A")
    print(f"      Final matcher          : {result.get('method', 'N/A')}")
    print(f"      Inlier count           : {m.get('inlier_count', 'N/A')}")
    print(f"      Inlier ratio           : {m.get('inlier_ratio', 0):.2%}")
    print(f"      RMSE (fine)            : {m.get('rmse', 'N/A')}")
    print(f"      Confidence             : {result.get('confidence', 'N/A')}")
    print(f"      Degenerate fit         : {result.get('degenerate_fit', 'N/A')}")
    print(f"      Homography composed    : {result.get('final_homography_composed', False)}")

    return result
