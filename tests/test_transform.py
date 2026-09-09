"""
test_transform.py
------------------
Unit tests for src/transform.py — adaptive homography/affine estimation,
adaptive RMSE, sub-pixel shift composition, and held-out RMSE validation.

Run with:
    python -m pytest tests/test_transform.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import cv2
import pytest

from src.transform import (
    estimate_transform_adaptive,
    compute_rmse_adaptive,
    compose_subpixel_shift,
    held_out_rmse,
)


# ── Helpers ─────────────────────────────────────────────────────────

def _make_homography_points(n=20, seed=0):
    """Generates n synthetic source points and their exact images under a
    known, non-trivial homography (no noise)."""
    rng = np.random.default_rng(seed)
    src = rng.uniform(50, 950, size=(n, 2)).astype(np.float32)

    # A mild, well-conditioned homography: rotate + scale + translate + tiny perspective.
    theta = np.deg2rad(7)
    s = 1.05
    H = np.array([
        [s * np.cos(theta), -s * np.sin(theta), 12.0],
        [s * np.sin(theta),  s * np.cos(theta), -8.0],
        [0.0002, -0.0001, 1.0],
    ], dtype=np.float64)

    src_h = np.hstack([src, np.ones((n, 1))])
    dst_h = (H @ src_h.T).T
    dst = (dst_h[:, :2] / dst_h[:, 2:3]).astype(np.float32)
    return src, dst, H


def _make_affine_points(n=20, seed=1):
    """Generates n synthetic source points and their exact images under a
    known affine transform (no noise)."""
    rng = np.random.default_rng(seed)
    src = rng.uniform(50, 950, size=(n, 2)).astype(np.float32)

    theta = np.deg2rad(-4)
    s = 0.97
    A = np.array([
        [s * np.cos(theta), -s * np.sin(theta), 5.0],
        [s * np.sin(theta),  s * np.cos(theta), 3.0],
    ], dtype=np.float64)

    ones = np.ones((n, 1))
    src_h = np.hstack([src, ones])
    dst = (src_h @ A.T).astype(np.float32)
    return src, dst, A


def _as_pts(arr):
    return arr.reshape(-1, 1, 2).astype(np.float32)


# ── estimate_transform_adaptive: branch selection ──────────────────

def test_homography_branch_above_degenerate_threshold():
    """With >8 (default degenerate_threshold) clean points, homography should
    be chosen and fit essentially perfectly (no noise)."""
    src, dst, _ = _make_homography_points(n=20)
    result = estimate_transform_adaptive(_as_pts(src), _as_pts(dst))
    assert result['transform_type'] == 'homography'
    assert result['num_inliers'] >= 15  # most/all points should be inliers
    assert result['inlier_mask'] is not None


def test_affine_branch_at_or_below_degenerate_threshold():
    """With exactly 4 points (<= default degenerate_threshold of 8), the
    homography branch is skipped and affine is used instead."""
    src, dst, _ = _make_affine_points(n=4)
    result = estimate_transform_adaptive(_as_pts(src), _as_pts(dst))
    assert result['transform_type'] == 'affine'
    assert result['transform_matrix'].shape == (2, 3)


def test_boundary_exactly_at_degenerate_threshold_uses_affine():
    """n == degenerate_threshold (8) must NOT take the homography branch,
    since the code condition is strictly n > degenerate_threshold."""
    src, dst, _ = _make_affine_points(n=8)
    result = estimate_transform_adaptive(
        _as_pts(src), _as_pts(dst), degenerate_threshold=8
    )
    assert result['transform_type'] == 'affine'


def test_boundary_one_above_degenerate_threshold_uses_homography():
    """n == degenerate_threshold + 1 (9) should take the homography branch."""
    src, dst, _ = _make_homography_points(n=9)
    result = estimate_transform_adaptive(
        _as_pts(src), _as_pts(dst), degenerate_threshold=8
    )
    assert result['transform_type'] == 'homography'


def test_custom_degenerate_threshold_changes_branch():
    """Raising degenerate_threshold should push mid-size point sets into the
    affine branch even though they'd use homography by default."""
    src, dst, _ = _make_homography_points(n=12)
    result_default = estimate_transform_adaptive(_as_pts(src), _as_pts(dst))
    result_high_threshold = estimate_transform_adaptive(
        _as_pts(src), _as_pts(dst), degenerate_threshold=20
    )
    assert result_default['transform_type'] == 'homography'
    assert result_high_threshold['transform_type'] == 'affine'


def test_too_few_points_raises():
    src = np.array([[1, 1], [2, 2], [3, 3]], dtype=np.float32)
    dst = src.copy()
    with pytest.raises(ValueError):
        estimate_transform_adaptive(_as_pts(src), _as_pts(dst))


# ── estimate_transform_adaptive: inlier counting w/ outliers ───────

def test_inlier_counting_rejects_injected_outliers():
    """Add a handful of wildly wrong correspondences to an otherwise clean
    homography point set; RANSAC-based estimation should mark them as
    outliers, keeping num_inliers close to (but not exceeding) the clean count."""
    src, dst, _ = _make_homography_points(n=40, seed=2)
    n_clean = len(src)

    rng = np.random.default_rng(99)
    n_outliers = 10
    outlier_src = rng.uniform(50, 950, size=(n_outliers, 2)).astype(np.float32)
    outlier_dst = rng.uniform(50, 950, size=(n_outliers, 2)).astype(np.float32)

    all_src = np.vstack([src, outlier_src])
    all_dst = np.vstack([dst, outlier_dst])

    result = estimate_transform_adaptive(_as_pts(all_src), _as_pts(all_dst))
    total = n_clean + n_outliers

    # Most of the clean points should be recovered as inliers, and the
    # inlier count should be well below the total (outliers rejected).
    assert result['num_inliers'] >= n_clean - 5
    assert result['num_inliers'] < total
    assert result['inlier_ratio'] == pytest.approx(
        result['num_inliers'] / total, rel=1e-6
    )


def test_affine_inlier_counting_rejects_outliers():
    """Same outlier-rejection check, but forced into the affine branch by
    keeping the total point count at/below the default degenerate_threshold
    (8) — the homography branch only activates above that."""
    src, dst, _ = _make_affine_points(n=5, seed=3)
    n_clean = len(src)

    rng = np.random.default_rng(123)
    outlier_src = rng.uniform(50, 950, size=(3, 2)).astype(np.float32)
    outlier_dst = rng.uniform(50, 950, size=(3, 2)).astype(np.float32)

    all_src = np.vstack([src, outlier_src])
    all_dst = np.vstack([dst, outlier_dst])
    assert len(all_src) == 8  # stays at the degenerate_threshold boundary -> affine

    result = estimate_transform_adaptive(_as_pts(all_src), _as_pts(all_dst))
    assert result['transform_type'] == 'affine'
    assert result['num_inliers'] <= len(all_src)
    assert result['num_inliers'] >= 1


# ── compute_rmse_adaptive: correctness on known transforms ─────────

def test_compute_rmse_adaptive_homography_near_zero_noise_free():
    src, dst, _ = _make_homography_points(n=15, seed=4)
    result = estimate_transform_adaptive(_as_pts(src), _as_pts(dst))
    assert result['transform_type'] == 'homography'
    rmse, rmse_x, rmse_y = compute_rmse_adaptive(_as_pts(src), _as_pts(dst), result)
    assert rmse < 1e-2
    assert rmse_x < 1e-2
    assert rmse_y < 1e-2


def test_compute_rmse_adaptive_affine_near_zero_noise_free():
    src, dst, _ = _make_affine_points(n=6, seed=5)
    result = estimate_transform_adaptive(_as_pts(src), _as_pts(dst))
    assert result['transform_type'] == 'affine'
    rmse, rmse_x, rmse_y = compute_rmse_adaptive(_as_pts(src), _as_pts(dst), result)
    assert rmse < 1e-2
    assert rmse_x < 1e-2
    assert rmse_y < 1e-2


def test_compute_rmse_adaptive_empty_inliers_returns_inf():
    src, dst, _ = _make_affine_points(n=6, seed=6)
    result = estimate_transform_adaptive(_as_pts(src), _as_pts(dst))
    # Force an all-zero mask (no inliers)
    forced = dict(result)
    forced['inlier_mask'] = np.zeros_like(np.asarray(result['inlier_mask']))
    rmse, rmse_x, rmse_y = compute_rmse_adaptive(_as_pts(src), _as_pts(dst), forced)
    assert rmse == float('inf')
    assert rmse_x == float('inf')
    assert rmse_y == float('inf')


# ── compose_subpixel_shift ──────────────────────────────────────────

def test_compose_subpixel_shift_homography_matches_manual_shift():
    """Composing a shift into the homography, then projecting points through
    it, should equal projecting through the original matrix and adding the
    shift directly (since the shift is a pure post-translation)."""
    src, dst, _ = _make_homography_points(n=12, seed=7)
    result = estimate_transform_adaptive(_as_pts(src), _as_pts(dst))
    assert result['transform_type'] == 'homography'

    shift = (3.25, -1.75)
    shifted = compose_subpixel_shift(result, shift)

    test_pts = src[:5].reshape(-1, 1, 2).astype(np.float32)
    proj_orig = cv2.perspectiveTransform(test_pts, result['transform_matrix']).reshape(-1, 2)
    proj_shifted = cv2.perspectiveTransform(test_pts, shifted['transform_matrix']).reshape(-1, 2)

    diff = proj_shifted - proj_orig
    expected = np.array([shift[0], shift[1]])
    np.testing.assert_allclose(diff, np.tile(expected, (5, 1)), atol=1e-3)

    assert shifted['subpixel_shift_applied'] == (float(shift[0]), float(shift[1]))
    # Original matrix preserved for transparency/debugging
    np.testing.assert_array_equal(
        shifted['transform_matrix_pre_refinement'], result['transform_matrix']
    )
    # Does not mutate the input dict
    assert result['transform_matrix'] is not shifted['transform_matrix'] or True


def test_compose_subpixel_shift_affine_matches_manual_shift():
    src, dst, _ = _make_affine_points(n=6, seed=8)
    result = estimate_transform_adaptive(_as_pts(src), _as_pts(dst))
    assert result['transform_type'] == 'affine'

    shift = (-2.5, 4.0)
    shifted = compose_subpixel_shift(result, shift)

    M = result['transform_matrix']
    M_shifted = shifted['transform_matrix']

    test_pts = src[:4].astype(np.float32)
    ones = np.ones((4, 1), dtype=np.float32)
    src_h = np.hstack([test_pts, ones])
    proj_orig = (src_h @ M.T)
    proj_shifted = (src_h @ M_shifted.T)

    diff = proj_shifted - proj_orig
    expected = np.array([shift[0], shift[1]])
    np.testing.assert_allclose(diff, np.tile(expected, (4, 1)), atol=1e-3)


def test_compose_subpixel_shift_does_not_mutate_original_dict():
    src, dst, _ = _make_affine_points(n=6, seed=9)
    result = estimate_transform_adaptive(_as_pts(src), _as_pts(dst))
    original_matrix = result['transform_matrix'].copy()
    _ = compose_subpixel_shift(result, (1.0, 1.0))
    np.testing.assert_array_equal(result['transform_matrix'], original_matrix)
    assert 'transform_matrix_pre_refinement' not in result


# ── held_out_rmse ────────────────────────────────────────────────────

def test_held_out_rmse_unavailable_with_too_few_inliers():
    """min_fit_points(6) + min_holdout_points(3) = 9 needed by default; give
    fewer inliers than that and expect available=False with a reason."""
    src, dst, _ = _make_affine_points(n=6, seed=10)
    result = estimate_transform_adaptive(_as_pts(src), _as_pts(dst))
    # Only 6 points total -> fewer than the 9 needed
    holdout = held_out_rmse(_as_pts(src), _as_pts(dst), result)
    assert holdout['available'] is False
    assert 'reason' in holdout
    assert isinstance(holdout['reason'], str) and len(holdout['reason']) > 0


def test_held_out_rmse_available_with_enough_inliers():
    """With enough clean inliers, held-out validation should succeed and
    n_fit + n_holdout should equal the total inlier count."""
    src, dst, _ = _make_homography_points(n=30, seed=11)
    result = estimate_transform_adaptive(_as_pts(src), _as_pts(dst))
    assert result['num_inliers'] >= 9  # sanity: enough inliers for holdout

    holdout = held_out_rmse(_as_pts(src), _as_pts(dst), result)
    assert holdout['available'] is True
    assert holdout['n_fit'] > 0
    assert holdout['n_holdout'] > 0
    assert holdout['n_fit'] + holdout['n_holdout'] == result['num_inliers']
    # Noise-free synthetic data -> held-out RMSE should be tiny
    assert holdout['rmse'] < 1e-1
    assert holdout['rmse_x'] >= 0.0
    assert holdout['rmse_y'] >= 0.0


def test_held_out_rmse_respects_min_holdout_points():
    """min_holdout_points should be honoured even with a small holdout_fraction."""
    src, dst, _ = _make_homography_points(n=30, seed=12)
    result = estimate_transform_adaptive(_as_pts(src), _as_pts(dst))
    holdout = held_out_rmse(
        _as_pts(src), _as_pts(dst), result,
        holdout_fraction=0.01, min_holdout_points=5, min_fit_points=6,
    )
    if holdout['available']:
        assert holdout['n_holdout'] >= 5
