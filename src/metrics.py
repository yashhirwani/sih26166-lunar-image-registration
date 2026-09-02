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


def compute_all_metrics(src_pts, dst_pts, M, mask, img_shape, grid_size=8):
    """
    Computes all metrics in one call.

    Parameters:
        src_pts: matched points in source image
        dst_pts: matched points in reference image
        M: homography matrix
        mask: inlier mask from RANSAC (True = correct match)
        img_shape: (height, width) of image
        grid_size: grid size for spatial distribution score

    Returns:
        dictionary with all metrics
    """
    import cv2

    metrics = {}

    # --- RMSE ---
    # Root Mean Square Error
    # Measures average alignment error in pixels
    # Sub-pixel = RMSE < 1.0
    inlier_mask = mask.ravel() == 1
    src_inliers = src_pts[inlier_mask]
    dst_inliers = dst_pts[inlier_mask]

    if len(src_inliers) > 0 and M is not None:
        src_transformed = cv2.perspectiveTransform(src_inliers, M)
        diff = src_transformed.reshape(-1, 2) - dst_inliers.reshape(-1, 2)
        squared_distances = np.sum(diff ** 2, axis=1)
        metrics['rmse'] = float(np.sqrt(np.mean(squared_distances)))
        metrics['rmse_x'] = float(np.sqrt(np.mean(diff[:, 0] ** 2)))
        metrics['rmse_y'] = float(np.sqrt(np.mean(diff[:, 1] ** 2)))
    else:
        metrics['rmse'] = float('inf')
        metrics['rmse_x'] = float('inf')
        metrics['rmse_y'] = float('inf')

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

    Parameters:
        metrics: dictionary from compute_all_metrics
        method_name: name of algorithm used

    Returns:
        formatted string report
    """
    sub_pixel = "✅ YES" if metrics['rmse'] < 1.0 else "❌ NO"

    report = f"""
╔══════════════════════════════════════════╗
║        REGISTRATION METRICS REPORT       ║
╠══════════════════════════════════════════╣
║  Algorithm        : {method_name:<22}║
║  RMSE             : {metrics['rmse']:<22.4f}║
║  RMSE-X           : {metrics['rmse_x']:<22.4f}║
║  RMSE-Y           : {metrics['rmse_y']:<22.4f}║
║  Sub-pixel        : {sub_pixel:<22}║
║  Inlier Count     : {metrics['inlier_count']:<22}║
║  Total Matches    : {metrics['total_matches']:<22}║
║  Inlier Ratio     : {metrics['inlier_ratio']:<22.4f}║
║  Spatial Score    : {metrics['spatial_score']:<22.4f}║
╚══════════════════════════════════════════╝
"""
    return report
