"""
test_match.py
-------------
Unit tests for src/match.py — AKAZE / SIFT detection+matching and the
uniform spatial distribution filter.

Uses synthetic image pairs (a random textured image and a warped copy of
itself) so no external test-set files are required.

Run with:
    python -m pytest tests/test_match.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import cv2
import pytest

from src.match import (
    detect_and_match_sift,
    detect_and_match_akaze,
    enforce_uniform_distribution,
)


# ── Synthetic image pair generation ─────────────────────────────────

def _make_textured_image(size=512, seed=0):
    """Builds a synthetic image with plenty of distinctive keypoint-friendly
    structure: random blobs + shapes, blurred, so SIFT/AKAZE have something
    to lock onto (a pure random-noise field has no stable local structure)."""
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), dtype=np.uint8)

    # Random circles and rectangles of varying intensity give stable corners
    # and blobs for both SIFT and AKAZE to detect.
    for _ in range(80):
        x, y = rng.integers(20, size - 20, size=2)
        r = rng.integers(4, 20)
        intensity = int(rng.integers(60, 255))
        cv2.circle(img, (int(x), int(y)), int(r), intensity, -1)

    for _ in range(40):
        x1, y1 = rng.integers(0, size - 30, size=2)
        w, h = rng.integers(10, 40, size=2)
        intensity = int(rng.integers(40, 220))
        cv2.rectangle(img, (int(x1), int(y1)),
                      (int(x1 + w), int(y1 + h)), intensity, -1)

    # Slight blur + noise to look more like a natural image
    img = cv2.GaussianBlur(img, (3, 3), 0)
    noise = rng.normal(0, 5, size=img.shape)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


def _make_pair_translated(size=512, seed=0, tx=15, ty=-10):
    img1 = _make_textured_image(size, seed)
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    img2 = cv2.warpAffine(img1, M, (size, size), borderMode=cv2.BORDER_REFLECT)
    return img1, img2


def _make_pair_rotated(size=512, seed=0, angle_deg=8, scale=1.0):
    img1 = _make_textured_image(size, seed)
    center = (size / 2, size / 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, scale)
    img2 = cv2.warpAffine(img1, M, (size, size), borderMode=cv2.BORDER_REFLECT)
    return img1, img2


# ── detect_and_match_sift ───────────────────────────────────────────

def test_sift_runs_without_raising_on_translated_pair():
    img1, img2 = _make_pair_translated()
    src_pts, dst_pts, kp1, kp2, good_matches = detect_and_match_sift(img1, img2)

    assert len(src_pts) == len(dst_pts) == len(good_matches)
    assert len(good_matches) >= 4
    assert src_pts.shape[1:] == (1, 2)
    assert dst_pts.shape[1:] == (1, 2)
    assert all(isinstance(m, cv2.DMatch) for m in good_matches)
    assert all(isinstance(k, cv2.KeyPoint) for k in kp1)
    assert all(isinstance(k, cv2.KeyPoint) for k in kp2)


def test_sift_runs_without_raising_on_rotated_pair():
    img1, img2 = _make_pair_rotated()
    src_pts, dst_pts, kp1, kp2, good_matches = detect_and_match_sift(img1, img2)
    assert len(src_pts) == len(dst_pts) == len(good_matches)
    assert len(good_matches) >= 4


def test_sift_match_indices_are_within_bounds():
    img1, img2 = _make_pair_translated()
    src_pts, dst_pts, kp1, kp2, good_matches = detect_and_match_sift(img1, img2)
    for m in good_matches:
        assert 0 <= m.queryIdx < len(kp1)
        assert 0 <= m.trainIdx < len(kp2)


# ── detect_and_match_akaze ──────────────────────────────────────────

def test_akaze_runs_without_raising_on_translated_pair():
    img1, img2 = _make_pair_translated()
    src_pts, dst_pts, kp1, kp2, good_matches = detect_and_match_akaze(img1, img2)

    assert len(src_pts) == len(dst_pts) == len(good_matches)
    assert len(good_matches) >= 4
    assert src_pts.shape[1:] == (1, 2)
    assert dst_pts.shape[1:] == (1, 2)
    assert all(isinstance(m, cv2.DMatch) for m in good_matches)
    assert all(isinstance(k, cv2.KeyPoint) for k in kp1)
    assert all(isinstance(k, cv2.KeyPoint) for k in kp2)


def test_akaze_runs_without_raising_on_rotated_pair():
    img1, img2 = _make_pair_rotated()
    src_pts, dst_pts, kp1, kp2, good_matches = detect_and_match_akaze(img1, img2)
    assert len(src_pts) == len(dst_pts) == len(good_matches)
    assert len(good_matches) >= 4


def test_akaze_match_indices_are_within_bounds():
    img1, img2 = _make_pair_translated()
    src_pts, dst_pts, kp1, kp2, good_matches = detect_and_match_akaze(img1, img2)
    for m in good_matches:
        assert 0 <= m.queryIdx < len(kp1)
        assert 0 <= m.trainIdx < len(kp2)


def test_akaze_available_from_plain_shell_run():
    """Regression guard: cv2.AKAZE_create() moved to cv2.xfeatures2d in some
    OpenCV 5.x builds. This just confirms the fallback chain in
    detect_and_match_akaze() actually resolves to a usable detector when run
    as a normal pytest process (not just inline in an existing session)."""
    img1, img2 = _make_pair_translated(size=256, seed=42)
    # Should not raise "AKAZE not available in this OpenCV build"
    detect_and_match_akaze(img1, img2)


# ── Not-enough-keypoints edge case ──────────────────────────────────

def test_sift_raises_on_blank_images():
    blank1 = np.zeros((256, 256), dtype=np.uint8)
    blank2 = np.zeros((256, 256), dtype=np.uint8)
    with pytest.raises(ValueError):
        detect_and_match_sift(blank1, blank2)


def test_akaze_raises_on_blank_images():
    blank1 = np.zeros((256, 256), dtype=np.uint8)
    blank2 = np.zeros((256, 256), dtype=np.uint8)
    with pytest.raises(ValueError):
        detect_and_match_akaze(blank1, blank2)


# ── enforce_uniform_distribution ────────────────────────────────────

def test_enforce_uniform_distribution_filters_and_returns_coherent_lengths():
    img1, img2 = _make_pair_translated()
    src_pts, dst_pts, kp1, kp2, good_matches = detect_and_match_sift(img1, img2)

    filtered_src, filtered_dst, filtered_matches = enforce_uniform_distribution(
        src_pts, dst_pts, good_matches, img1.shape, grid_size=8, max_per_cell=5
    )

    assert len(filtered_src) == len(filtered_dst) == len(filtered_matches)
    assert len(filtered_src) <= len(src_pts)
    # Every cell should contribute at most max_per_cell points.
    h, w = img1.shape
    grid_size = 8
    max_per_cell = 5
    cell_h, cell_w = h / grid_size, w / grid_size
    counts = {}
    for pt in filtered_src.reshape(-1, 2):
        x, y = pt
        row = min(int(y / cell_h), grid_size - 1)
        col = min(int(x / cell_w), grid_size - 1)
        counts[(row, col)] = counts.get((row, col), 0) + 1
    assert all(c <= max_per_cell for c in counts.values())


def test_enforce_uniform_distribution_falls_back_when_too_few_survive():
    """If filtering would drop below 4 points, the function returns the
    original unfiltered arrays instead."""
    # Construct a tiny synthetic match set: 3 points, all in different cells,
    # each with max_per_cell=1 -> 3 survive, which is < 4, so it should
    # fall back to returning the originals unchanged.
    src_pts = np.float32([[10, 10], [500, 500], [900, 50]]).reshape(-1, 1, 2)
    dst_pts = src_pts.copy()

    class FakeMatch:
        def __init__(self, distance):
            self.distance = distance

    good_matches = [FakeMatch(1.0), FakeMatch(2.0), FakeMatch(3.0)]

    filtered_src, filtered_dst, filtered_matches = enforce_uniform_distribution(
        src_pts, dst_pts, good_matches, (1024, 1024), grid_size=8, max_per_cell=5
    )
    # Falls back to originals since kept_indices (3) < 4
    assert len(filtered_matches) == 3
    np.testing.assert_array_equal(filtered_src, src_pts)


def test_enforce_uniform_distribution_grid_size_and_max_per_cell_effect():
    """A larger max_per_cell should never keep fewer points than a smaller one
    for the same input."""
    img1, img2 = _make_pair_rotated()
    src_pts, dst_pts, kp1, kp2, good_matches = detect_and_match_sift(img1, img2)

    _, _, matches_strict = enforce_uniform_distribution(
        src_pts, dst_pts, good_matches, img1.shape, grid_size=8, max_per_cell=1
    )
    _, _, matches_loose = enforce_uniform_distribution(
        src_pts, dst_pts, good_matches, img1.shape, grid_size=8, max_per_cell=10
    )
    assert len(matches_loose) >= len(matches_strict)
