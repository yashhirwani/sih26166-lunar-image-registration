"""
geo_align.py
------------
Geo-assisted coarse pre-alignment using OHRC .csv ground-coordinate files.

The problem this solves:
- Every feature matcher (SIFT, LightGlue, PC-SIFT) searches "blind" for
  correspondences — it has no prior knowledge of where a point in image 1
  should land in image 2.
- On South Pole images with heavy shadows and flat repetitive terrain, this
  blind search fails (4-5 inliers, below the reliability floor).
- But we already know the approximate answer: each OHRC zip contains a .csv
  file that maps pixel positions to real lunar lat/lon coordinates. Since
  both images are geolocated, we can compute approximate pixel-to-pixel
  correspondences directly from orbit geometry — no pixel matching needed
  for this rough estimate.
- We use these geo-correspondences to pre-warp the source image roughly onto
  the reference frame, then run standard feature matching on the pre-aligned
  pair. This turns "find a needle in a shadowed haystack" into "confirm and
  refine a needle we already roughly located."

CSV format (confirmed from real OHRC file header):
    Longitude, Latitude, Pixel, Scan
    34.7545467, -84.6487714, 0, 0
    34.7460307, -84.6490892, 100, 0
    ...
Columns are: index 0=Longitude, 1=Latitude, 2=Pixel(col), 3=Scan(row)
Sampling: every ~100 pixels in both directions → ~122K rows per image.
"""

import numpy as np
import csv
import cv2
import zipfile
import io
from pathlib import Path


def load_csv_geolocation(csv_path: str) -> np.ndarray:
    """
    Loads an OHRC .csv ground-coordinates file from disk.

    Confirmed column order (from real OHRC file header):
        Longitude, Latitude, Pixel, Scan
    i.e. col 0=lon, 1=lat, 2=pixel_col, 3=scan_line

    Parameters:
        csv_path : path to the .csv file

    Returns:
        np.ndarray shape (N, 4): [longitude, latitude, pixel_col, scan_line]
    """
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        print(f"[geo_align] CSV header: {header}")
        # Confirm column order matches expectation
        expected = ['longitude', 'latitude', 'pixel', 'scan']
        actual = [h.strip().lower() for h in header]
        if actual != expected:
            print(f"[geo_align] WARNING: unexpected column order {actual} — expected {expected}")
            print(f"[geo_align] Proceeding with confirmed order: lon=0, lat=1, col=2, row=3")

        for row in reader:
            if len(row) < 4:
                continue
            try:
                lon  = float(row[0])
                lat  = float(row[1])
                col  = float(row[2])
                line = float(row[3])
                rows.append([lon, lat, col, line])
            except ValueError:
                continue

    arr = np.array(rows, dtype=np.float64)
    print(f"[geo_align] Loaded {len(arr)} geo-points from {Path(csv_path).name}")
    print(f"[geo_align] Lon range: [{arr[:,0].min():.4f}, {arr[:,0].max():.4f}]  "
          f"Lat range: [{arr[:,1].min():.4f}, {arr[:,1].max():.4f}]")
    print(f"[geo_align] Col range: [{arr[:,2].min():.0f}, {arr[:,2].max():.0f}]  "
          f"Row range: [{arr[:,3].min():.0f}, {arr[:,3].max():.0f}]")
    return arr


def load_csv_from_zip(zip_path: str) -> np.ndarray:
    """
    Extracts and loads the .csv geolocation file directly from an OHRC zip
    without writing to disk.

    Parameters:
        zip_path : path to the OHRC .zip file

    Returns:
        np.ndarray shape (N, 4): [longitude, latitude, pixel_col, scan_line]
        or None if no CSV found in the zip
    """
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for name in zf.namelist():
            if name.endswith('.csv') and 'grd' in name:
                print(f"[geo_align] Found CSV in zip: {name}")
                with zf.open(name) as f:
                    content = f.read().decode('utf-8')

                rows = []
                lines = content.split('\n')
                header = lines[0].strip().split(',')
                print(f"[geo_align] CSV header: {header}")

                for line in lines[1:]:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(',')
                    if len(parts) < 4:
                        continue
                    try:
                        rows.append([float(parts[0]), float(parts[1]),
                                     float(parts[2]), float(parts[3])])
                    except ValueError:
                        continue

                arr = np.array(rows, dtype=np.float64)
                print(f"[geo_align] Loaded {len(arr)} geo-points")
                print(f"[geo_align] Lon range: [{arr[:,0].min():.4f}, {arr[:,0].max():.4f}]  "
                      f"Lat range: [{arr[:,1].min():.4f}, {arr[:,1].max():.4f}]")
                return arr

    print(f"[geo_align] No CSV found in {zip_path}")
    return None


def find_common_ground_points(
    csv1: np.ndarray,
    csv2: np.ndarray,
    max_distance_deg: float = 0.001,
    max_points: int = 500,
) -> tuple:
    """
    Finds pixel correspondences between two images using only geolocation.

    For each sampled point in image 1's CSV, finds the nearest lat/lon point
    in image 2's CSV. If within max_distance_deg, it's a valid geo-linked
    correspondence.

    max_distance_deg=0.001 corresponds to roughly 30m on lunar surface.
    The CSV is sampled every 100 pixels (~25m at 0.25m/px), so 0.001° is
    about 1-2 sampling intervals — tight enough to be accurate but loose
    enough to find overlapping points.

    Parameters:
        csv1, csv2       : (N,4) arrays from load_csv_geolocation
        max_distance_deg : maximum lat/lon distance to consider a match
        max_points       : maximum number of correspondences to return
                           (subsampled if more found, for speed)

    Returns:
        img1_points : (M, 2) float32 array of (col, row) in image 1
        img2_points : (M, 2) float32 array of (col, row) in image 2
    """
    from scipy.spatial import cKDTree

    # Build KD-tree on image 2's lat/lon coordinates
    tree = cKDTree(csv2[:, 0:2])  # columns: lon, lat

    # Query nearest neighbor for each point in image 1
    distances, indices = tree.query(csv1[:, 0:2], k=1)

    # Keep only points within distance threshold
    valid = distances <= max_distance_deg
    n_valid = valid.sum()

    print(f"[geo_align] {n_valid} / {len(csv1)} geo-points matched "
          f"within {max_distance_deg}° tolerance "
          f"(~{max_distance_deg * 30440:.0f}m on lunar surface)")

    if n_valid == 0:
        # Try relaxing threshold and report nearest-neighbor distance stats
        p10, p50, p90 = np.percentile(distances, [10, 50, 90])
        print(f"[geo_align] Distance stats (deg): p10={p10:.5f} p50={p50:.5f} p90={p90:.5f}")
        print(f"[geo_align] Suggestion: try max_distance_deg >= {p10:.4f}")
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    img1_pts = csv1[valid][:, 2:4].astype(np.float32)   # col, scan_line
    img2_pts = csv2[indices[valid]][:, 2:4].astype(np.float32)

    # Report pixel shift distribution (useful diagnostic)
    shifts = img2_pts - img1_pts
    print(f"[geo_align] Pixel shift distribution (img2 - img1):")
    print(f"            col shift:  mean={shifts[:,0].mean():.1f}  "
          f"std={shifts[:,0].std():.1f}  "
          f"range=[{shifts[:,0].min():.0f}, {shifts[:,0].max():.0f}]")
    print(f"            row shift:  mean={shifts[:,1].mean():.1f}  "
          f"std={shifts[:,1].std():.1f}  "
          f"range=[{shifts[:,1].min():.0f}, {shifts[:,1].max():.0f}]")

    # Subsample if too many points (for speed in RANSAC)
    if len(img1_pts) > max_points:
        rng = np.random.default_rng(seed=42)  # fixed seed for reproducibility
        idx = rng.choice(len(img1_pts), max_points, replace=False)
        img1_pts = img1_pts[idx]
        img2_pts = img2_pts[idx]
        print(f"[geo_align] Subsampled to {max_points} points for speed")

    return img1_pts, img2_pts


def estimate_coarse_transform(
    img1_points: np.ndarray,
    img2_points: np.ndarray,
    reproj_threshold: float = 10.0,
) -> tuple:
    """
    Fits an approximate homography from geo-linked pixel correspondences.
    Uses RANSAC to handle noisy geo-points (sparse CSV sampling + flat-plane
    approximation ignores terrain elevation).

    reproj_threshold=10.0 pixels is intentionally loose — the geo-based
    correspondences are not sub-pixel accurate (CSV is sampled every 100px)
    so we expect ~5-15px residual from the coarse estimate alone.

    Parameters:
        img1_points      : (N, 2) float32 — (col, row) in image 1
        img2_points      : (N, 2) float32 — (col, row) in image 2
        reproj_threshold : RANSAC inlier threshold in pixels

    Returns:
        H             : 3×3 homography matrix (float64)
        residual_rmse : RMSE on geo inliers in pixels (diagnostic only)
    """
    if len(img1_points) < 4:
        raise ValueError(
            f"Only {len(img1_points)} geo-linked points — need ≥ 4 for homography. "
            f"Try increasing max_distance_deg in find_common_ground_points()."
        )

    # OpenCV expects (N,1,2) shape
    p1 = img1_points.reshape(-1, 1, 2).astype(np.float32)
    p2 = img2_points.reshape(-1, 1, 2).astype(np.float32)

    H, mask = cv2.findHomography(p1, p2, cv2.RANSAC, reproj_threshold,
                                  maxIters=2000, confidence=0.995)

    if H is None:
        raise ValueError("cv2.findHomography returned None — geo-points are degenerate or collinear")

    n_inliers = int(mask.sum())
    print(f"[geo_align] Homography estimated: {n_inliers}/{len(img1_points)} geo-inliers")

    # Residual RMSE on inliers (diagnostic)
    inlier_mask = mask.ravel() == 1
    in1 = img1_points[inlier_mask].reshape(-1, 1, 2).astype(np.float32)
    in2 = img2_points[inlier_mask].reshape(-1, 1, 2).astype(np.float32)
    projected = cv2.perspectiveTransform(in1, H).reshape(-1, 2)
    residuals = np.linalg.norm(projected - in2.reshape(-1, 2), axis=1)
    residual_rmse = float(np.sqrt(np.mean(residuals ** 2)))

    print(f"[geo_align] Coarse residual RMSE: {residual_rmse:.2f} px "
          f"(expected ~5-50px — feature matching will refine further)")

    return H, residual_rmse


def warp_source_coarse(
    source_image: np.ndarray,
    coarse_H: np.ndarray,
    output_shape: tuple,
) -> np.ndarray:
    """
    Pre-warps the source image using the coarse geo-based homography,
    placing it roughly onto the reference image's coordinate frame.

    Parameters:
        source_image : (H, W) grayscale image
        coarse_H     : 3×3 homography matrix from estimate_coarse_transform
        output_shape : (width, height) of output — should match reference image

    Returns:
        pre-warped source image, same shape as reference
    """
    warped = cv2.warpPerspective(
        source_image, coarse_H, output_shape,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    non_zero_frac = np.count_nonzero(warped) / warped.size
    print(f"[geo_align] Pre-warped source: shape={warped.shape} "
          f"non-zero={non_zero_frac:.1%}")
    if non_zero_frac < 0.05:
        print(f"[geo_align] WARNING: pre-warped image is mostly black — "
              f"coarse homography may be incorrect (check geo-point coverage)")
    return warped
