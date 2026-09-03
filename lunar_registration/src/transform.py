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


def estimate_homography_ransac(src_pts, dst_pts, reproj_threshold=3.0):
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


def subpixel_refinement(img1, img2):
    """
    Refines alignment to sub-pixel accuracy using phase correlation.

    What is sub-pixel accuracy?
    - Normal matching finds points accurate to ±1 pixel
    - Sub-pixel means accuracy of 0.1-0.5 pixels
    - ISRO explicitly requires sub-pixel accuracy

    How phase correlation works (simple):
    - Takes two aligned images
    - Uses math (Fourier Transform) to find the tiny remaining shift
    - Returns correction in x and y direction (can be fractional like 0.3 pixels)

    Parameters:
        img1: reference image (float)
        img2: warped source image (float)

    Returns:
        shift: (x_shift, y_shift) in sub-pixel units
    """
    # Convert to float for precise calculation
    f1 = img1.astype(np.float64)
    f2 = img2.astype(np.float64)

    # Use OpenCV's phase correlation
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
