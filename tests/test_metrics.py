"""
test_metrics.py
---------------
Unit tests for src/metrics.py — specifically the reliability assessment
and degenerate-fit detection logic.

Run with:
    cd /Users/kajol/Desktop/ps166/lunar_registration
    python -m pytest tests/test_metrics.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.metrics import assess_reliability, MIN_INLIERS_DEGENERATE, MIN_INLIERS_RELIABLE


# ── Core cases from the task specification ────────────────────────

def test_degenerate_fit_detected():
    """4 inliers — exactly at DOF boundary, RMSE is always ~0, not meaningful."""
    result = assess_reliability(inlier_count=4, inlier_ratio=0.267, spatial_score=0.333)
    assert result["degenerate_fit"] is True
    assert result["confidence"] == "failed"
    assert "RMSE is not meaningful" in result["reliability_reason"] or \
           "mathematically forced" in result["reliability_reason"]


def test_low_but_not_degenerate():
    """9 inliers — above degenerate threshold but below reliable floor."""
    result = assess_reliability(inlier_count=9, inlier_ratio=0.5, spatial_score=0.6)
    assert result["degenerate_fit"] is False
    assert result["confidence"] == "low"
    assert str(MIN_INLIERS_RELIABLE) in result["reliability_reason"]


def test_reliable_high_confidence():
    """50 inliers with good ratio and spatial score — high confidence."""
    result = assess_reliability(inlier_count=50, inlier_ratio=0.6, spatial_score=0.8)
    assert result["degenerate_fit"] is False
    assert result["confidence"] == "high"
    assert result["reliability_reason"] == ""


# ── Boundary conditions ───────────────────────────────────────────

def test_exactly_at_degenerate_boundary():
    """Exactly MIN_INLIERS_DEGENERATE inliers = still degenerate."""
    result = assess_reliability(
        inlier_count=MIN_INLIERS_DEGENERATE,
        inlier_ratio=0.8,
        spatial_score=0.9
    )
    assert result["degenerate_fit"] is True
    assert result["confidence"] == "failed"


def test_one_above_degenerate_boundary():
    """MIN_INLIERS_DEGENERATE + 1 = not degenerate, but below reliable floor."""
    result = assess_reliability(
        inlier_count=MIN_INLIERS_DEGENERATE + 1,
        inlier_ratio=0.8,
        spatial_score=0.9
    )
    assert result["degenerate_fit"] is False
    assert result["confidence"] == "low"  # still below MIN_INLIERS_RELIABLE


def test_exactly_at_reliable_floor():
    """Exactly MIN_INLIERS_RELIABLE inliers = no longer low from count alone."""
    result = assess_reliability(
        inlier_count=MIN_INLIERS_RELIABLE,
        inlier_ratio=0.6,
        spatial_score=0.8
    )
    assert result["degenerate_fit"] is False
    # ratio >= 0.5 and spatial >= 0.7 → high
    assert result["confidence"] == "high"


def test_medium_confidence():
    """Enough inliers but moderate ratio and spatial score."""
    result = assess_reliability(inlier_count=20, inlier_ratio=0.3, spatial_score=0.5)
    assert result["degenerate_fit"] is False
    assert result["confidence"] == "medium"


def test_low_confidence_poor_ratio():
    """Enough inliers but very poor ratio."""
    result = assess_reliability(inlier_count=20, inlier_ratio=0.1, spatial_score=0.2)
    assert result["degenerate_fit"] is False
    assert result["confidence"] == "low"


def test_zero_inliers():
    """Zero inliers — degenerate."""
    result = assess_reliability(inlier_count=0, inlier_ratio=0.0, spatial_score=0.0)
    assert result["degenerate_fit"] is True
    assert result["confidence"] == "failed"


def test_single_inlier():
    """1 inlier — degenerate."""
    result = assess_reliability(inlier_count=1, inlier_ratio=1.0, spatial_score=1.0)
    assert result["degenerate_fit"] is True
    assert result["confidence"] == "failed"


def test_return_keys_always_present():
    """All three required keys always present regardless of inlier count."""
    for count in [0, 4, 8, 9, 10, 50, 317]:
        result = assess_reliability(inlier_count=count, inlier_ratio=0.5, spatial_score=0.5)
        assert "degenerate_fit" in result
        assert "confidence" in result
        assert "reliability_reason" in result


def test_easy_case_real_values():
    """Simulate our known easy test case — 317 inliers, 100% ratio."""
    result = assess_reliability(inlier_count=317, inlier_ratio=1.0, spatial_score=0.9991)
    assert result["degenerate_fit"] is False
    assert result["confidence"] == "high"
    assert result["reliability_reason"] == ""


def test_medium_case_real_values():
    """Simulate medium test case — 5 inliers, 35% ratio."""
    result = assess_reliability(inlier_count=5, inlier_ratio=0.357, spatial_score=0.320)
    assert result["degenerate_fit"] is True
    assert result["confidence"] == "failed"


def test_hard_case_real_values():
    """Simulate hard test case — 4 inliers, 26% ratio, RMSE 0.0."""
    result = assess_reliability(inlier_count=4, inlier_ratio=0.267, spatial_score=0.333)
    assert result["degenerate_fit"] is True
    assert result["confidence"] == "failed"


# ── Affine transform_type tests ───────────────────────────────────

def test_affine_4_inliers_not_degenerate():
    """5 inliers with affine (6 DOF) = NOT degenerate — has slack equations."""
    result = assess_reliability(
        inlier_count=5, inlier_ratio=0.5, spatial_score=0.5,
        transform_type="affine"
    )
    assert result["degenerate_fit"] is False
    assert result["confidence"] == "low"   # below reliable floor (6) but not degenerate


def test_affine_degenerate_at_4():
    """4 inliers with affine: degenerate threshold is 4, so exactly 4 = degenerate."""
    result = assess_reliability(
        inlier_count=4, inlier_ratio=0.5, spatial_score=0.5,
        transform_type="affine"
    )
    # MIN_INLIERS_DEGENERATE_AFFINE = 4, so 4 <= 4 → degenerate
    assert result["degenerate_fit"] is True


def test_affine_5_inliers_not_degenerate():
    """5 inliers with affine = not degenerate (above floor of 4)."""
    result = assess_reliability(
        inlier_count=5, inlier_ratio=0.6, spatial_score=0.6,
        transform_type="affine"
    )
    assert result["degenerate_fit"] is False
    assert result["confidence"] == "low"   # below reliable floor (6)


def test_affine_high_confidence():
    """8 inliers with affine + good ratio = high confidence."""
    result = assess_reliability(
        inlier_count=8, inlier_ratio=0.6, spatial_score=0.8,
        transform_type="affine"
    )
    assert result["degenerate_fit"] is False
    assert result["confidence"] == "high"


def test_homography_default_unchanged():
    """Default transform_type='homography' — existing behaviour preserved."""
    result = assess_reliability(inlier_count=4, inlier_ratio=0.5, spatial_score=0.5)
    assert result["degenerate_fit"] is True
    assert result["confidence"] == "failed"
