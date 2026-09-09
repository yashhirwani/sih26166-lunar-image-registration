"""
test_metrics_extra.py
----------------------
Unit tests for the parts of src/metrics.py not already covered by
tests/test_metrics.py (which is scoped to assess_reliability):
    - compute_spatial_distribution_score
    - compute_coverage_metrics
    - assess_subpixel_claim

Run with:
    python -m pytest tests/test_metrics_extra.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

from src.metrics import (
    compute_spatial_distribution_score,
    compute_coverage_metrics,
    assess_subpixel_claim,
)


# ── compute_spatial_distribution_score ──────────────────────────────

def test_uniform_points_score_high():
    """One point placed in the center of every grid cell -> perfectly
    uniform distribution -> score should be (near) 1.0."""
    grid_size = 8
    img_shape = (800, 800)  # divides evenly by 8 -> 100px cells
    points = []
    cell = 800 / grid_size
    for row in range(grid_size):
        for col in range(grid_size):
            x = col * cell + cell / 2
            y = row * cell + cell / 2
            points.append((x, y))
    points = np.array(points)

    score = compute_spatial_distribution_score(points, img_shape, grid_size)
    assert score > 0.99


def test_clustered_points_score_low():
    """All points crammed into a single grid cell -> entropy is 0 ->
    score should be 0.0."""
    img_shape = (800, 800)
    grid_size = 8
    # All points inside cell (0,0), i.e. x,y in [0, 100)
    rng = np.random.default_rng(0)
    points = rng.uniform(1, 90, size=(50, 2))

    score = compute_spatial_distribution_score(points, img_shape, grid_size)
    assert score == pytest.approx(0.0, abs=1e-9)


def test_uniform_beats_clustered():
    img_shape = (800, 800)
    grid_size = 8
    cell = 800 / grid_size
    uniform_pts = np.array([
        (col * cell + cell / 2, row * cell + cell / 2)
        for row in range(grid_size) for col in range(grid_size)
    ])
    clustered_pts = np.tile(np.array([[10.0, 10.0]]), (64, 1))

    uniform_score = compute_spatial_distribution_score(uniform_pts, img_shape, grid_size)
    clustered_score = compute_spatial_distribution_score(clustered_pts, img_shape, grid_size)
    assert uniform_score > clustered_score


def test_empty_points_returns_zero():
    score = compute_spatial_distribution_score(np.empty((0, 2)), (800, 800), 8)
    assert score == 0.0


# ── compute_coverage_metrics ────────────────────────────────────────

def test_coverage_metrics_known_layout():
    """4 points, each in a distinct corner cell of an 8x8 grid on an 800x800
    image (cell size 100px) -> exactly 4 occupied cells, coverage_fraction
    4/64, and each cell has 1/4 of the points -> max_cell_fraction = 0.25."""
    img_shape = (800, 800)
    grid_size = 8
    points = np.array([
        [10, 10],      # cell (0,0)
        [790, 10],     # cell (0,7)
        [10, 790],     # cell (7,0)
        [790, 790],    # cell (7,7)
    ])

    cov = compute_coverage_metrics(points, img_shape, grid_size)
    assert cov['occupied_cells'] == 4
    assert cov['total_cells'] == 64
    assert cov['coverage_fraction'] == pytest.approx(4 / 64)
    assert cov['max_cell_fraction'] == pytest.approx(0.25)
    # Spread should span nearly the full image (780/800 in both axes)
    assert cov['spatial_spread_x'] == pytest.approx(780 / 800, rel=1e-6)
    assert cov['spatial_spread_y'] == pytest.approx(780 / 800, rel=1e-6)


def test_coverage_metrics_all_points_one_cell():
    img_shape = (800, 800)
    grid_size = 8
    points = np.array([[5, 5], [6, 6], [7, 7], [8, 8]])

    cov = compute_coverage_metrics(points, img_shape, grid_size)
    assert cov['occupied_cells'] == 1
    assert cov['coverage_fraction'] == pytest.approx(1 / 64)
    assert cov['max_cell_fraction'] == pytest.approx(1.0)
    # All points nearly on top of each other -> near-zero spread
    assert cov['spatial_spread_x'] < 0.01
    assert cov['spatial_spread_y'] < 0.01


def test_coverage_metrics_empty_points():
    cov = compute_coverage_metrics(np.empty((0, 2)), (800, 800), 8)
    assert cov['occupied_cells'] == 0
    assert cov['coverage_fraction'] == 0.0
    assert cov['spatial_spread_x'] == 0.0
    assert cov['spatial_spread_y'] == 0.0
    assert cov['max_cell_fraction'] == 0.0
    assert cov['total_cells'] == 64


def test_coverage_metrics_spread_capped_at_one():
    """spatial_spread should never exceed 1.0 even with points at exact
    boundary extremes."""
    img_shape = (100, 100)
    grid_size = 4
    points = np.array([[0, 0], [99, 99]])
    cov = compute_coverage_metrics(points, img_shape, grid_size)
    assert cov['spatial_spread_x'] <= 1.0
    assert cov['spatial_spread_y'] <= 1.0


# ── assess_subpixel_claim ───────────────────────────────────────────

def test_subpixel_claim_degenerate_never_achieved():
    metrics = {
        'degenerate_fit': True,
        'held_out_validation': {'available': True, 'rmse': 0.1, 'n_holdout': 20},
    }
    result = assess_subpixel_claim(metrics)
    assert result['achieved'] is False
    assert result['label'] == 'NOT MEANINGFUL'


def test_subpixel_claim_holdout_unavailable_is_unverified_not_achieved():
    """Even if the fit is non-degenerate, if held-out validation is
    unavailable the claim must be UNVERIFIED, never silently 'achieved'."""
    metrics = {
        'degenerate_fit': False,
        'held_out_validation': {'available': False, 'reason': 'too few inliers'},
    }
    result = assess_subpixel_claim(metrics)
    assert result['achieved'] is False
    assert result['label'] == 'UNVERIFIED'
    assert 'too few inliers' in result['basis']


def test_subpixel_claim_holdout_missing_key_entirely_is_unverified():
    """held_out_validation absent entirely should also resolve to UNVERIFIED,
    not crash or default to achieved."""
    metrics = {'degenerate_fit': False}
    result = assess_subpixel_claim(metrics)
    assert result['achieved'] is False
    assert result['label'] == 'UNVERIFIED'


def test_subpixel_claim_holdout_available_and_below_one_px_achieved():
    metrics = {
        'degenerate_fit': False,
        'held_out_validation': {'available': True, 'rmse': 0.42, 'n_holdout': 15},
    }
    result = assess_subpixel_claim(metrics)
    assert result['achieved'] is True
    assert 'held-out verified' in result['label']
    assert '0.4200' in result['basis']


def test_subpixel_claim_holdout_available_and_at_or_above_one_px_not_achieved():
    metrics = {
        'degenerate_fit': False,
        'held_out_validation': {'available': True, 'rmse': 1.0, 'n_holdout': 15},
    }
    result = assess_subpixel_claim(metrics)
    assert result['achieved'] is False
    assert result['label'] == 'NO'


def test_subpixel_claim_holdout_available_well_above_one_px_not_achieved():
    metrics = {
        'degenerate_fit': False,
        'held_out_validation': {'available': True, 'rmse': 3.7, 'n_holdout': 15},
    }
    result = assess_subpixel_claim(metrics)
    assert result['achieved'] is False
    assert result['label'] == 'NO'
    assert '3.7000' in result['basis']
