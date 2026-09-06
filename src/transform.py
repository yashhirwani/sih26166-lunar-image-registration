"""
transform.py
------------
Estimates the geometric transformation and warps the source image.

What it does:
1. Takes matched point pairs
2. Removes wrong matches using RANSAC (outlier rejection)
3. Estimates homography matrix (the transformation recipe)
4. Applies sub-pixel refinement using phase correlation
5. Warps source image to align with reference image

Terms explained:
- Homography: a 3x3 matrix that describes how to rotate, scale, shift an image
- RANSAC: removes wrong matches by checking geometric consistency
- Warp: actually moving/transforming the image pixels
- Sub-pixel: accuracy better than 1 pixel
"""

import cv2
import numpy as np
from scipy.signal import fftconvolve
from .config_loader import get_config


def estimate_homography_ransac(src_pts, dst_pts, reproj_threshold=None):
    """
    Estimates the transformation matrix using RANSAC.

    RANSAC in simple words:
    - Randomly picks 4 match pairs
    - Calculates what transformation they suggest
    - Checks how many other matches agree with this transformation
    - Repeats many times
    - Keeps the transformation that most matches agree with
    - Matches that agree = inliers (correct)
    - Matches that disagree = outliers (wrong, discarded)

    Parameters:
        src_pts: points in source image
        dst_pts: corresponding points in reference image
        reproj_threshold: how many pixels of error is acceptable (3.0 is standard)

    Returns:
        M: 3x3 homography matrix
        mask: boolean array (True = inlier, False = outlier)
        num_inliers: count of correct matches
        inlier_ratio: fraction of correct matches
    """
    if reproj_threshold is None:
        reproj_threshold = get_config()['ransac']['reproj_threshold']
        
    M, mask = cv2.findHomography(
        src_pts,
        dst_pts,
        cv2.RANSAC,
        reproj_threshold,
        maxIters=2000,
        confidence=0.995
    )

    if M is None:
        raise ValueError("Could not estimate homography. Not enough valid matches.")

    num_inliers = int(mask.sum())
    inlier_ratio = num_inliers / len(mask)

    return M, mask, num_inliers, inlier_ratio


def warp_image(src_img, M, ref_img_shape):
    """
    Applies the homography transformation to warp source image.

    Warping = moving every pixel in source image to its new position
    based on the homography matrix M.

    Result: source image now looks like it was taken from the same
    position as the reference image.

    Parameters:
        src_img: source image to warp
        M: 3x3 homography matrix
        ref_img_shape: (height, width) of reference image

    Returns:
        warped_img: source image after transformation
    """
    h, w = ref_img_shape
    warped = cv2.warpPerspective(src_img, M, (w, h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=0)
    return warped


def subpixel_refinement_lk(img1, img2, src_pts, dst_pts, mask):
    """
    Sub-pixel refinement using Lucas-Kanade optical flow.

    What is Lucas-Kanade?
    - A classical computer vision algorithm
    - For each matched point pair, it looks at a small window around the point
    - Finds the exact sub-pixel position where the two windows best match
    - Much more accurate than global phase correlation for point-wise refinement

    Why better than phase correlation:
    - Phase correlation gives ONE global shift for the whole image
    - Lucas-Kanade gives individual sub-pixel correction for EACH match point
    - Result: much lower RMSE

    Parameters:
        img1: source image (after warping)
        img2: reference image
        src_pts: source match points
        dst_pts: destination match points
        mask: inlier mask from RANSAC

    Returns:
        refined_src_pts: sub-pixel refined source points
        refined_dst_pts: sub-pixel refined destination points
        refined_mask: updated mask
    """
    inlier_mask = mask.ravel() == 1
    src_inliers = src_pts[inlier_mask].reshape(-1, 1, 2)
    dst_inliers = dst_pts[inlier_mask].reshape(-1, 1, 2)

    if len(src_inliers) < 4:
        return src_pts, dst_pts, mask

    # Lucas-Kanade parameters
    lk_params = dict(
        winSize=(21, 21),       # window size around each point
        maxLevel=3,             # pyramid levels (handles larger motions)
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,    # max iterations
            0.001  # epsilon (stop when correction < 0.001 pixels)
        )
    )

    # Run Lucas-Kanade optical flow
    # Finds where each point in img1 moved to in img2
    refined_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        img1, img2,
        src_inliers.astype(np.float32),
        None,
        **lk_params
    )

    if refined_pts is None or status is None:
        return src_pts, dst_pts, mask

    # Keep only points where LK succeeded
    lk_good = status.ravel() == 1
    if lk_good.sum() < 4:
        return src_pts, dst_pts, mask

    return (
        src_inliers[lk_good],
        refined_pts[lk_good],
        np.ones((lk_good.sum(), 1), dtype=np.uint8)
    )


def subpixel_refinement(img1, img2):
    """
    Refines alignment to sub-pixel accuracy using phase correlation.
    """
    f1 = img1.astype(np.float64)
    f2 = img2.astype(np.float64)
    shift, response = cv2.phaseCorrelate(f1, f2)
    return shift, response


def compute_rmse(src_pts, dst_pts, M, mask):
    """
    Computes RMSE (Root Mean Square Error) of the registration.

    RMSE measures how accurate the alignment is.
    - For each inlier match, we transform the source point using M
    - We measure distance between transformed point and expected destination point
    - We average all these distances
    - Sub-pixel = RMSE < 1.0

    Lower RMSE = better alignment.
    Sub-pixel accuracy = RMSE < 1.0 pixel.

    Parameters:
        src_pts: points in source image
        dst_pts: corresponding points in reference image
        M: estimated homography matrix
        mask: inlier mask from RANSAC

    Returns:
        rmse: the RMSE value (float, lower is better)
    """
    if mask is None or M is None:
        return float('inf')

    # Get only inlier points
    inlier_mask = mask.ravel() == 1
    src_inliers = src_pts[inlier_mask]
    dst_inliers = dst_pts[inlier_mask]

    if len(src_inliers) == 0:
        return float('inf')

    # Transform source points using estimated homography
    src_transformed = cv2.perspectiveTransform(src_inliers, M)

    # Compute Euclidean distance between transformed and actual destination points
    diff = src_transformed.reshape(-1, 2) - dst_inliers.reshape(-1, 2)
    squared_distances = np.sum(diff ** 2, axis=1)
    rmse = np.sqrt(np.mean(squared_distances))

    return float(rmse)


def estimate_transform_adaptive(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    degenerate_threshold: int = 8,
    reproj_threshold: float = 3.0,
) -> dict:
    """
    Estimates a geometric transform, automatically choosing between
    homography and affine based on how many matched points are available.

    Why this matters:
    - Homography has 8 degrees of freedom (DOF)
    - With exactly 4 points: 4×2=8 equations for 8 unknowns → always exact fit
    - RMSE is forced to 0.0 — looks perfect but is mathematically meaningless
    - Affine has 6 DOF. With 4 points: 4×2=8 equations for 6 unknowns
    - There are 2 slack equations → RANSAC can actually reject wrong fits
    - RMSE is now a real number that means something

    Decision logic:
    - n_points > degenerate_threshold (8) → use homography (statistically safe)
    - n_points <= degenerate_threshold  → use affine (gives real RMSE even with 4 pts)

    Parameters:
        src_pts             : (N,1,2) or (N,2) float32 source points
        dst_pts             : (N,1,2) or (N,2) float32 destination points
        degenerate_threshold: max points at which homography is degenerate (default 8)
        reproj_threshold    : RANSAC reprojection error threshold in pixels

    Returns dict with keys:
        transform_matrix : np.ndarray — 3×3 for homography, 2×3 for affine
        transform_type   : str — "homography" or "affine"
        inlier_mask      : np.ndarray — boolean mask of inliers
        num_inliers      : int
        inlier_ratio     : float
    """
    # Normalise shape to (N,1,2)
    s = src_pts.reshape(-1, 1, 2).astype(np.float32)
    d = dst_pts.reshape(-1, 1, 2).astype(np.float32)
    n = len(s)

    if n < 4:
        raise ValueError(f"Need at least 4 points, got {n}")

    # ── Try homography first if we have enough points ─────────────
    if n > degenerate_threshold:
        H, mask = cv2.findHomography(
            s, d, cv2.RANSAC, reproj_threshold,
            maxIters=2000, confidence=0.995
        )
        if H is not None and mask is not None:
            n_in = int(mask.sum())
            return {
                "transform_matrix": H,
                "transform_type":   "homography",
                "inlier_mask":      mask,
                "num_inliers":      n_in,
                "inlier_ratio":     n_in / n,
            }

    # ── Fall back to affine ────────────────────────────────────────
    A, mask = cv2.estimateAffinePartial2D(
        s, d,
        method=cv2.RANSAC,
        ransacReprojThreshold=reproj_threshold,
        maxIters=2000,
        confidence=0.995,
    )
    if A is None:
        raise ValueError(
            "Both homography and affine estimation failed. "
            "Check that matched points are not all collinear or identical."
        )

    # cv2.estimateAffinePartial2D returns (N,1) mask
    if mask is None:
        mask = np.ones((n, 1), dtype=np.uint8)

    n_in = int(mask.sum())
    print(f"[estimate_transform_adaptive] AFFINE (6 DOF) — "
          f"{n} input points, {n_in} inliers  "
          f"(homography would be degenerate at ≤{degenerate_threshold} points)")

    return {
        "transform_matrix": A,
        "transform_type":   "affine",
        "inlier_mask":      mask,
        "num_inliers":      n_in,
        "inlier_ratio":     n_in / n,
    }


def warp_image_adaptive(src_img: np.ndarray, transform_result: dict,
                         ref_shape: tuple) -> np.ndarray:
    """
    Warps source image using the result from estimate_transform_adaptive().
    Branches on transform_type to call the correct OpenCV warp function:
      homography → cv2.warpPerspective  (needs 3×3 matrix)
      affine     → cv2.warpAffine       (needs 2×3 matrix)

    Using the wrong warp function for the transform type silently produces
    garbage — this function prevents that.

    Parameters:
        src_img          : source image (H,W) or (H,W,C)
        transform_result : dict from estimate_transform_adaptive()
        ref_shape        : (height, width) of reference image

    Returns:
        warped image same shape as (ref_shape[0], ref_shape[1])
    """
    h, w = ref_shape[0], ref_shape[1]
    M   = transform_result["transform_matrix"]
    ttype = transform_result["transform_type"]

    if ttype == "homography":
        return cv2.warpPerspective(
            src_img, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    elif ttype == "affine":
        return cv2.warpAffine(
            src_img, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    else:
        raise ValueError(f"Unknown transform_type '{ttype}' — expected 'homography' or 'affine'")


def compute_rmse_adaptive(src_pts: np.ndarray, dst_pts: np.ndarray,
                           transform_result: dict) -> tuple:
    """
    Computes RMSE using the correct projection function for the transform type.

    For homography: cv2.perspectiveTransform
    For affine:     cv2.transform (on homogeneous coords)

    Returns:
        (rmse, rmse_x, rmse_y)
    """
    M     = transform_result["transform_matrix"]
    ttype = transform_result["transform_type"]
    mask  = transform_result["inlier_mask"]

    inlier_mask = mask.ravel() == 1
    src_in = src_pts.reshape(-1, 2)[inlier_mask].astype(np.float32)
    dst_in = dst_pts.reshape(-1, 2)[inlier_mask].astype(np.float32)

    if len(src_in) == 0:
        return float('inf'), float('inf'), float('inf')

    if ttype == "homography":
        proj = cv2.perspectiveTransform(
            src_in.reshape(-1, 1, 2), M
        ).reshape(-1, 2)
    else:
        # affine: M is 2×3, apply as [x,y,1] @ M.T
        ones = np.ones((len(src_in), 1), dtype=np.float32)
        src_h = np.hstack([src_in, ones])           # (N,3)
        proj  = (src_h @ M.T).astype(np.float32)   # (N,2)

    diff = proj - dst_in
    rmse   = float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))
    rmse_x = float(np.sqrt(np.mean(diff[:, 0] ** 2)))
    rmse_y = float(np.sqrt(np.mean(diff[:, 1] ** 2)))
    return rmse, rmse_x, rmse_y
