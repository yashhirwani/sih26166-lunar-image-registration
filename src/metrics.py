"""
metrics.py
----------
Calculates all evaluation metrics required by ISRO PS166.

Metrics explained:
- RMSE: how many pixels off the alignment is (lower = better, <1.0 = sub-pixel)
- Inlier Count: how many correct matches were found
- Inlier Ratio: percentage of correct matches (higher = better)
- Spatial Distribution Score: how evenly spread the matches are (higher = better)
"""

import numpy as np

# ── Reliability constants ──────────────────────────────────────────
# Homography has 8 DOF; 4 correspondences = exact fit (RMSE always ~0, meaningless).
MIN_INLIERS_DEGENERATE = 8   # at or below this, fit is statistically unreliable
MIN_INLIERS_RELIABLE   = 10  # below this, cap confidence regardless of RMSE/ratio

# Affine has 6 DOF; 4 correspondences give 2 slack equations → not degenerate
MIN_INLIERS_DEGENERATE_AFFINE = 4
MIN_INLIERS_RELIABLE_AFFINE   = 6


def assess_reliability(
    inlier_count: int,
    inlier_ratio: float,
    spatial_score: float,
    transform_type: str = "homography",
) -> dict:
    """
    Determines whether a registration result is statistically trustworthy,
    independent of how good RMSE looks.

    Thresholds depend on transform_type:
    - homography (8 DOF): degenerate at ≤8 inliers, reliable floor at 10
    - affine     (6 DOF): degenerate at ≤4 inliers, reliable floor at 6
      (4 points give 8 equations for 6 unknowns → 2 slack → RANSAC can reject)

    Returns:
        {
            "degenerate_fit"    : bool
            "confidence"        : str  — "high"|"medium"|"low"|"failed"
            "reliability_reason": str
        }
    """
    if transform_type == "affine":
        min_degen    = MIN_INLIERS_DEGENERATE_AFFINE
        min_reliable = MIN_INLIERS_RELIABLE_AFFINE
        dof_label    = "affine (6 DOF)"
    else:
        min_degen    = MIN_INLIERS_DEGENERATE
        min_reliable = MIN_INLIERS_RELIABLE
        dof_label    = "homography (8 DOF)"

    degenerate = inlier_count <= min_degen

    if degenerate:
        return {
            "degenerate_fit": True,
            "confidence": "failed",
            "reliability_reason": (
                f"Only {inlier_count} inlier(s) found. "
                f"A {dof_label} fit needs more points to be statistically checkable; "
                f"at {inlier_count} the fit may be forced exact and RMSE is not meaningful."
            ),
        }

    if inlier_count < min_reliable:
        return {
            "degenerate_fit": False,
            "confidence": "low",
            "reliability_reason": (
                f"Only {inlier_count} inliers (below the {min_reliable}-point "
                f"reliability floor for {dof_label}). "
                f"Metrics reported but treat cautiously."
            ),
        }

    # Above the floor — base confidence on ratio and spatial distribution
    if inlier_ratio >= 0.5 and spatial_score >= 0.7:
        confidence = "high"
    elif inlier_ratio >= 0.25 and spatial_score >= 0.4:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "degenerate_fit": False,
        "confidence": confidence,
        "reliability_reason": "",
    }


def compute_all_metrics(src_pts, dst_pts, M, mask, img_shape, grid_size=8,
                        transform_result=None):
    """
    Computes all metrics in one call.

    Parameters:
        src_pts          : matched points in source image
        dst_pts          : matched points in reference image
        M                : transform matrix (3×3 homography or 2×3 affine)
        mask             : inlier mask from RANSAC (True = correct match)
        img_shape        : (height, width) of image
        grid_size        : grid size for spatial distribution score
        transform_result : optional dict from estimate_transform_adaptive()
                           if provided, uses compute_rmse_adaptive for correct RMSE
    Returns:
        dictionary with all metrics
    """
    import cv2

    metrics = {}
    ttype = (transform_result or {}).get('transform_type', 'homography')

    # --- RMSE ---
    inlier_mask = mask.ravel() == 1
    src_inliers = src_pts[inlier_mask]
    dst_inliers = dst_pts[inlier_mask]

    if len(src_inliers) > 0 and M is not None:
        if transform_result is not None:
            from .transform import compute_rmse_adaptive
            rmse, rmse_x, rmse_y = compute_rmse_adaptive(src_pts, dst_pts, transform_result)
        else:
            # Legacy path: homography only
            src_transformed = cv2.perspectiveTransform(src_inliers, M)
            diff = src_transformed.reshape(-1, 2) - dst_inliers.reshape(-1, 2)
            squared = np.sum(diff ** 2, axis=1)
            rmse   = float(np.sqrt(np.mean(squared)))
            rmse_x = float(np.sqrt(np.mean(diff[:, 0] ** 2)))
            rmse_y = float(np.sqrt(np.mean(diff[:, 1] ** 2)))

        metrics['rmse']   = rmse
        metrics['rmse_x'] = rmse_x
        metrics['rmse_y'] = rmse_y
    else:
        metrics['rmse'] = metrics['rmse_x'] = metrics['rmse_y'] = float('inf')

    # --- Inlier Count ---
    # Number of correct matches after RANSAC filtering
    metrics['inlier_count'] = int(inlier_mask.sum())

    # --- Total Matches ---
    metrics['total_matches'] = len(mask)

    # --- Inlier Ratio ---
    # inliers / total matches
    # Higher = more reliable matching
    if metrics['total_matches'] > 0:
        metrics['inlier_ratio'] = metrics['inlier_count'] / metrics['total_matches']
    else:
        metrics['inlier_ratio'] = 0.0

    # --- Spatial Distribution Score ---
    # Measures how evenly matches are spread across image
    # Score = 0 to 1 (1 = perfectly uniform, 0 = all in one spot)
    # ISRO explicitly requires uniform distribution
    metrics['spatial_score'] = compute_spatial_distribution_score(
        dst_inliers.reshape(-1, 2), img_shape, grid_size
    )

    # --- Reliability assessment ---
    reliability = assess_reliability(
        inlier_count=metrics['inlier_count'],
        inlier_ratio=metrics['inlier_ratio'],
        spatial_score=metrics['spatial_score'],
        transform_type=ttype,
    )
    metrics['degenerate_fit']     = reliability['degenerate_fit']
    metrics['confidence']         = reliability['confidence']
    metrics['reliability_reason'] = reliability['reliability_reason']
    metrics['transform_type']     = ttype

    return metrics


def compute_spatial_distribution_score(points, img_shape, grid_size=8):
    """
    Measures how evenly match points are distributed across the image.

    How it works:
    - Divides image into grid_size x grid_size cells (default 8x8 = 64 cells)
    - Counts how many match points fall in each cell
    - Calculates entropy of this distribution
    - Normalizes to 0-1 range

    Score = 1.0 → matches perfectly spread everywhere
    Score = 0.0 → all matches in one cell

    Parameters:
        points: array of (x, y) coordinates of match points
        img_shape: (height, width) of image
        grid_size: number of grid rows and columns

    Returns:
        score: float between 0 and 1
    """
    if len(points) == 0:
        return 0.0

    h, w = img_shape
    cell_h = h / grid_size
    cell_w = w / grid_size

    # Count points per cell
    cell_counts = np.zeros((grid_size, grid_size))
    for x, y in points:
        col = min(int(x / cell_w), grid_size - 1)
        row = min(int(y / cell_h), grid_size - 1)
        cell_counts[row, col] += 1

    # Compute entropy
    # Entropy is high when distribution is uniform
    total = cell_counts.sum()
    if total == 0:
        return 0.0

    probs = cell_counts.flatten() / total
    probs = probs[probs > 0]  # remove zeros (log(0) is undefined)

    entropy = -np.sum(probs * np.log(probs))
    max_entropy = np.log(grid_size * grid_size)  # perfect uniform distribution

    if max_entropy == 0:
        return 0.0

    return float(entropy / max_entropy)


def format_metrics_report(metrics, method_name="SIFT"):
    """
    Formats metrics into a readable report string.
    """
    sub_pixel = "✅ YES" if metrics['rmse'] < 1.0 else "❌ NO"
    confidence = metrics.get('confidence', 'N/A').upper()
    degenerate = metrics.get('degenerate_fit', False)
    degen_flag = "⚠️ YES — RMSE not meaningful" if degenerate else "No"
    ttype      = metrics.get('transform_type', 'homography')

    report = f"""
╔══════════════════════════════════════════╗
║        REGISTRATION METRICS REPORT       ║
╠══════════════════════════════════════════╣
║  Algorithm        : {method_name:<22}║
║  Transform        : {ttype:<22}║
║  RMSE             : {metrics['rmse']:<22.4f}║
║  RMSE-X           : {metrics['rmse_x']:<22.4f}║
║  RMSE-Y           : {metrics['rmse_y']:<22.4f}║
║  Sub-pixel        : {sub_pixel:<22}║
║  Inlier Count     : {metrics['inlier_count']:<22}║
║  Total Matches    : {metrics['total_matches']:<22}║
║  Inlier Ratio     : {metrics['inlier_ratio']:<22.4f}║
║  Spatial Score    : {metrics['spatial_score']:<22.4f}║
║  Confidence       : {confidence:<22}║
║  Degenerate Fit   : {degen_flag:<22}║
╚══════════════════════════════════════════╝
"""
    if degenerate:
        report += f"\n⚠️  WARNING: {metrics.get('reliability_reason', '')}\n"
    elif metrics.get('reliability_reason'):
        report += f"\nℹ️  NOTE: {metrics.get('reliability_reason', '')}\n"
    return report
