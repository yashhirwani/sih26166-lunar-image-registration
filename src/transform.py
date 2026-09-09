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

# ── Outlier rejection method ────────────────────────────────────────────────
# MAGSAC++ (cv2.USAC_MAGSAC) avoids the fixed inlier/outlier threshold that
# plain RANSAC relies on — it marginalizes over the noise scale instead of
# requiring the caller to guess one, which produces a more accurate inlier
# set (and unlike RANSAC, one that doesn't strongly depend on
# reproj_threshold being "roughly right"). It's supported by OpenCV's USAC
# framework since 4.5.1. We probe it once at import time and fall back to
# classic cv2.RANSAC transparently if the installed OpenCV build lacks it,
# so the pipeline never hard-fails over an outlier-rejection method choice.
_HAS_USAC_MAGSAC = hasattr(cv2, "USAC_MAGSAC")


def _ransac_method_for(kind: str) -> int:
    """Returns the best available outlier-rejection method flag for
    'homography' or 'affine' estimation, preferring MAGSAC++."""
    if _HAS_USAC_MAGSAC:
        return cv2.USAC_MAGSAC
    return cv2.RANSAC


def outlier_method_name() -> str:
    """Human-readable label for whichever method estimation actually used —
    surfaced in the UI so a MAGSAC fallback to RANSAC is never silent."""
    return "MAGSAC++ (USAC_MAGSAC)" if _HAS_USAC_MAGSAC else "RANSAC (MAGSAC++ unavailable in this OpenCV build)"


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
        _ransac_method_for('homography'),
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

    outlier_method = outlier_method_name()

    # ── Try homography first if we have enough points ─────────────
    if n > degenerate_threshold:
        H, mask = cv2.findHomography(
            s, d, _ransac_method_for('homography'), reproj_threshold,
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
                "outlier_method":   outlier_method,
            }

    # ── Fall back to affine (full 6 DOF: rotation, independent x/y scale,
    #    shear, translation — estimateAffine2D, not the 4-DOF similarity
    #    transform of estimateAffinePartial2D) ──────────────────────────
    try:
        A, mask = cv2.estimateAffine2D(
            s, d,
            method=_ransac_method_for('affine'),
            ransacReprojThreshold=reproj_threshold,
            maxIters=2000,
            confidence=0.995,
        )
    except cv2.error:
        # Installed OpenCV build's USAC framework doesn't support this
        # estimator — fall back to classic RANSAC rather than crashing.
        A, mask = cv2.estimateAffine2D(
            s, d,
            method=cv2.RANSAC,
            ransacReprojThreshold=reproj_threshold,
            maxIters=2000,
            confidence=0.995,
        )
        outlier_method = "RANSAC (MAGSAC++ not supported for affine estimation on this OpenCV build)"

    if A is None:
        raise ValueError(
            "Both homography and affine estimation failed. "
            "Check that matched points are not all collinear or identical."
        )

    if mask is None:
        mask = np.ones((n, 1), dtype=np.uint8)

    n_in = int(mask.sum())
    print(f"[estimate_transform_adaptive] AFFINE (6 DOF) -- "
          f"{n} input points, {n_in} inliers, outlier method: {outlier_method} "
          f"(homography would be degenerate at <={degenerate_threshold} points)")

    return {
        "transform_matrix": A,
        "transform_type":   "affine",
        "inlier_mask":      mask,
        "num_inliers":      n_in,
        "inlier_ratio":     n_in / n,
        "outlier_method":   outlier_method,
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


def _project_points(pts: np.ndarray, M: np.ndarray, ttype: str) -> np.ndarray:
    """Projects (N,2) points through a homography (3x3) or affine (2x3) matrix."""
    pts = pts.reshape(-1, 2).astype(np.float32)
    if ttype == "homography":
        return cv2.perspectiveTransform(pts.reshape(-1, 1, 2), M).reshape(-1, 2)
    ones = np.ones((len(pts), 1), dtype=np.float32)
    pts_h = np.hstack([pts, ones])
    return (pts_h @ M.T).astype(np.float32)


def compose_subpixel_shift(transform_result: dict, shift: tuple) -> dict:
    """
    Folds the global phase-correlation shift (applied to the warped image in
    Step 6 of the pipeline) into the transform matrix, so that RMSE computed
    against the returned matrix matches the image that is actually delivered
    (registered_image_refined) instead of the pre-refinement warp.

    The phase-correlation correction is applied downstream as
    cv2.warpAffine(warped_img, [[1,0,shift[0]],[0,1,shift[1]]], ...) —
    i.e. a pure translation added AFTER the fitted transform, in the
    destination/reference pixel frame. This composes the same translation
    into the transform matrix so a point mapped through the returned matrix
    lands where it actually ends up in the refined output image.

    Returns a new dict (does not mutate the input) with:
        transform_matrix          : M with the shift folded in
        transform_matrix_pre_refinement : the original, unshifted matrix
        subpixel_shift_applied    : the (dx, dy) that was composed in
    """
    M = transform_result["transform_matrix"]
    ttype = transform_result["transform_type"]
    dx, dy = float(shift[0]), float(shift[1])

    if ttype == "homography":
        T = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], dtype=np.float64)
        M_refined = T @ M.astype(np.float64)
    else:
        M_refined = M.astype(np.float64).copy()
        M_refined[0, 2] += dx
        M_refined[1, 2] += dy

    out = dict(transform_result)
    out["transform_matrix"] = M_refined.astype(M.dtype) if np.issubdtype(M.dtype, np.floating) else M_refined
    out["transform_matrix_pre_refinement"] = M
    out["subpixel_shift_applied"] = (dx, dy)
    return out


def held_out_rmse(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    transform_result: dict,
    holdout_fraction: float = 0.25,
    min_fit_points: int = 6,
    min_holdout_points: int = 3,
    reproj_threshold: float = 3.0,
    seed: int = 42,
) -> dict:
    """
    Independent accuracy check: re-fits the transform on a random subset of
    the RANSAC inliers and measures residual error on the *held-out* inliers
    that were never used for fitting.

    Why this matters: compute_rmse_adaptive() reports residuals on the same
    points used to estimate the transform — a fit will always look good on
    its own training points, so that number alone cannot support a claim of
    "sub-pixel accuracy achieved". This function gives a genuinely
    out-of-sample estimate.

    Returns a dict:
        available   : bool — False if there aren't enough inliers to split
        reason      : str  — explains why unavailable, if applicable
        rmse, rmse_x, rmse_y : float — held-out residuals (only if available)
        n_fit, n_holdout     : int
    """
    mask = transform_result["inlier_mask"]
    ttype = transform_result["transform_type"]
    inlier_idx = np.where(mask.ravel() == 1)[0]
    n = len(inlier_idx)

    min_needed = min_fit_points + min_holdout_points
    if n < min_needed:
        return {
            "available": False,
            "reason": (
                f"Only {n} inliers — need at least {min_needed} "
                f"({min_fit_points} to fit + {min_holdout_points} held out) "
                f"for an independent accuracy estimate."
            ),
        }

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(inlier_idx)
    n_holdout = max(min_holdout_points, int(round(n * holdout_fraction)))
    n_holdout = min(n_holdout, n - min_fit_points)  # always leave enough to fit
    holdout_idx = shuffled[:n_holdout]
    fit_idx = shuffled[n_holdout:]

    s_all = src_pts.reshape(-1, 2).astype(np.float32)
    d_all = dst_pts.reshape(-1, 2).astype(np.float32)
    s_fit, d_fit = s_all[fit_idx], d_all[fit_idx]
    s_hold, d_hold = s_all[holdout_idx], d_all[holdout_idx]

    # Re-fit on the fit subset only (non-robust — these are already RANSAC
    # inliers, so an exact/least-squares fit is appropriate and avoids the
    # extra randomness of running RANSAC again on a small subset).
    try:
        if ttype == "homography":
            M_fit, _ = cv2.findHomography(s_fit, d_fit, method=0)
        else:
            M_fit, _ = cv2.estimateAffine2D(s_fit, d_fit, method=cv2.LMEDS)
        if M_fit is None:
            raise ValueError("re-fit on held-in subset returned no solution")
    except Exception as e:
        return {"available": False, "reason": f"Held-out re-fit failed: {e}"}

    proj = _project_points(s_hold, M_fit, ttype)
    diff = proj - d_hold
    rmse = float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))
    rmse_x = float(np.sqrt(np.mean(diff[:, 0] ** 2)))
    rmse_y = float(np.sqrt(np.mean(diff[:, 1] ** 2)))

    return {
        "available": True,
        "rmse": rmse,
        "rmse_x": rmse_x,
        "rmse_y": rmse_y,
        "n_fit": int(len(fit_idx)),
        "n_holdout": int(len(holdout_idx)),
    }
