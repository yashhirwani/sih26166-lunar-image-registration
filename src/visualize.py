"""
visualize.py
------------
Creates all visual outputs for the registration results.

What it produces:
1. Match visualization: lines connecting matched points between two images
2. Checkerboard view: alternating tiles showing alignment quality
3. Side by side comparison: before vs after registration
4. Keypoint heatmap: density of match points across image
"""

import cv2
import numpy as np


def draw_matches(img1, kp1, img2, kp2, good_matches, mask=None, max_matches=100):
    """
    Draws lines connecting matched keypoints between two images.

    Green lines = correct matches (inliers)
    Red lines = wrong matches (outliers)

    Parameters:
        img1, img2: the two images
        kp1, kp2: keypoints in each image
        good_matches: list of match objects
        mask: inlier mask (True = inlier/correct, False = outlier/wrong)
        max_matches: maximum lines to draw (too many looks messy)

    Returns:
        match_img: image with match lines drawn
    """
    # Convert grayscale to color so we can draw colored lines
    img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    if mask is not None:
        # Separate inliers (correct) and outliers (wrong)
        inlier_matches = [good_matches[i] for i in range(len(good_matches))
                         if i < len(mask) and mask.ravel()[i] == 1]
        outlier_matches = [good_matches[i] for i in range(len(good_matches))
                          if i < len(mask) and mask.ravel()[i] == 0]

        # Limit number of lines drawn
        inlier_matches = inlier_matches[:max_matches]
        outlier_matches = outlier_matches[:20]

        # Draw inliers in green using drawMatches (thin lines, standard thickness)
        match_img = cv2.drawMatches(
            img1_color, kp1, img2_color, kp2,
            inlier_matches, None,
            matchColor=(0, 255, 0),      # green for inliers
            singlePointColor=(128, 128, 128),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        # Draw outliers in RED with thick bold lines manually using cv2.line
        # cv2.drawMatches doesn't support thickness — we draw manually instead
        w1 = img1_color.shape[1]  # width of image 1 (offset for image 2 points)
        for m in outlier_matches:
            pt1 = tuple(map(int, kp1[m.queryIdx].pt))
            pt2_raw = kp2[m.trainIdx].pt
            pt2 = (int(pt2_raw[0]) + w1, int(pt2_raw[1]))  # offset by image 1 width
            cv2.line(match_img, pt1, pt2, color=(0, 0, 255), thickness=3)
            # Draw bold dots at endpoints
            cv2.circle(match_img, pt1, radius=5, color=(0, 0, 200), thickness=-1)
            cv2.circle(match_img, pt2, radius=5, color=(0, 0, 200), thickness=-1)
    else:
        # No mask, draw all matches in green
        matches_to_draw = good_matches[:max_matches]
        match_img = cv2.drawMatches(
            img1_color, kp1, img2_color, kp2,
            matches_to_draw, None,
            matchColor=(0, 255, 0),
            singlePointColor=(128, 128, 128),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

    return match_img


def create_checkerboard(img1, img2, tile_size=64):
    """
    Creates a checkerboard blend of two images.

    What it shows:
    - Alternating tiles from img1 and img2
    - If images are perfectly aligned, the border between tiles is invisible
    - If misaligned, you see obvious jumps at tile borders

    This is the best way to visually check alignment quality.
    Judges love this visualization.

    Parameters:
        img1: reference image
        img2: registered (warped) source image
        tile_size: size of each checkerboard square in pixels

    Returns:
        checkerboard image
    """
    # Make both images same size
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h, w = min(h1, h2), min(w1, w2)

    img1_crop = img1[:h, :w]
    img2_crop = img2[:h, :w]

    # Convert to color
    if len(img1_crop.shape) == 2:
        img1_color = cv2.cvtColor(img1_crop, cv2.COLOR_GRAY2BGR)
    else:
        img1_color = img1_crop.copy()

    if len(img2_crop.shape) == 2:
        img2_color = cv2.cvtColor(img2_crop, cv2.COLOR_GRAY2BGR)
    else:
        img2_color = img2_crop.copy()

    result = img1_color.copy()

    # Fill alternating tiles from img2
    for row in range(0, h, tile_size):
        for col in range(0, w, tile_size):
            tile_row = row // tile_size
            tile_col = col // tile_size
            if (tile_row + tile_col) % 2 == 0:
                row_end = min(row + tile_size, h)
                col_end = min(col + tile_size, w)
                result[row:row_end, col:col_end] = img2_color[row:row_end, col:col_end]

    return result


def create_side_by_side(img1, img2, label1="Reference", label2="Registered"):
    """
    Creates a side by side comparison of two images with labels.

    Parameters:
        img1: left image (reference)
        img2: right image (registered/warped source)
        label1, label2: text labels for each image

    Returns:
        combined image with both side by side
    """
    # Make same height
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h = max(h1, h2)

    # Convert to color
    if len(img1.shape) == 2:
        img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    else:
        img1_color = img1.copy()

    if len(img2.shape) == 2:
        img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    else:
        img2_color = img2.copy()

    # Pad to same height
    if h1 < h:
        img1_color = cv2.copyMakeBorder(img1_color, 0, h - h1, 0, 0,
                                         cv2.BORDER_CONSTANT, value=0)
    if h2 < h:
        img2_color = cv2.copyMakeBorder(img2_color, 0, h - h2, 0, 0,
                                         cv2.BORDER_CONSTANT, value=0)

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img1_color, label1, (10, 30), font, 0.8, (0, 255, 0), 2)
    cv2.putText(img2_color, label2, (10, 30), font, 0.8, (0, 255, 0), 2)

    # Separator line
    separator = np.zeros((h, 4, 3), dtype=np.uint8)
    separator[:] = (0, 200, 0)

    combined = np.hstack([img1_color, separator, img2_color])
    return combined


def create_difference_image(img1, img2):
    """
    Creates a difference image showing where the two images differ.

    Bright areas = large difference (misalignment)
    Dark areas = small difference (good alignment)

    Parameters:
        img1: reference image
        img2: registered source image

    Returns:
        difference image (amplified for visibility)
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h, w = min(h1, h2), min(w1, w2)

    img1_crop = img1[:h, :w].astype(np.float32)
    img2_crop = img2[:h, :w].astype(np.float32)

    diff = np.abs(img1_crop - img2_crop)

    # Amplify for visibility
    diff_amplified = np.clip(diff * 3, 0, 255).astype(np.uint8)

    # Apply colormap for better visualization
    diff_color = cv2.applyColorMap(diff_amplified, cv2.COLORMAP_JET)

    return diff_color


def numpy_to_bytes(img):
    """
    Converts numpy image array to PNG bytes for Streamlit display.
    """
    _, buffer = cv2.imencode('.png', img)
    return buffer.tobytes()
