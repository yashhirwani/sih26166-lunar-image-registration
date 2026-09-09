"""
test_loaders.py
----------------
Smoke tests for image loading: src.preprocess.load_image / preprocess_pair
on the real easy-tier PNGs shipped in data/test_set/, plus a guarded check
of src.lro_loader.load_geotiff_reference against the one real GeoTIFF in
data/reference/.

Loader tests that need fixture files not present in this repo (OHRC raw
.img/.xml label pairs for src/ohrc_loader.py, an IIRS hyperspectral cube for
src/iirs_loader.py) are skipped rather than invented — see the checks at
the top of each section below.

Run with:
    python -m pytest tests/test_loaders.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

from src.preprocess import load_image, load_image_from_bytes, preprocess_pair

REPO_ROOT = os.path.join(os.path.dirname(__file__), '..')
EASY_DIR = os.path.join(REPO_ROOT, 'data', 'test_set', 'easy_same_acquisition')
EASY_SRC = os.path.join(EASY_DIR, 'easy_ncp_20260102T1224107393_p1.png')
EASY_REF = os.path.join(EASY_DIR, 'easy_nrp_20260102T1224107393_p1.png')

GEOTIFF_PATH = os.path.join(REPO_ROOT, 'data', 'reference', 'lro_south_pole.tif')


# ── src.preprocess.load_image ───────────────────────────────────────

@pytest.mark.skipif(not os.path.exists(EASY_SRC), reason=f"missing fixture: {EASY_SRC}")
def test_load_image_easy_src_dtype_and_shape():
    img = load_image(EASY_SRC)
    assert isinstance(img, np.ndarray)
    assert img.dtype == np.uint8
    assert img.ndim == 2  # grayscale, single channel
    assert img.shape[0] > 0 and img.shape[1] > 0


@pytest.mark.skipif(not os.path.exists(EASY_SRC), reason=f"missing fixture: {EASY_SRC}")
def test_load_image_easy_src_non_degenerate_content():
    """Confirms the loaded image isn't blank/degenerate — has real variation
    in pixel values, not all-black or all-one-value."""
    img = load_image(EASY_SRC)
    assert img.mean() > 2.0        # not near-all-black (mirrors pipeline's own check)
    assert img.std() > 1.0         # has actual contrast/texture, not flat
    assert img.max() > img.min()   # not a constant image


@pytest.mark.skipif(not os.path.exists(EASY_REF), reason=f"missing fixture: {EASY_REF}")
def test_load_image_easy_ref_dtype_and_shape():
    img = load_image(EASY_REF)
    assert isinstance(img, np.ndarray)
    assert img.dtype == np.uint8
    assert img.ndim == 2
    assert img.shape[0] > 0 and img.shape[1] > 0


def test_load_image_raises_on_missing_file():
    with pytest.raises(ValueError):
        load_image(os.path.join(REPO_ROOT, 'data', 'test_set', 'does_not_exist.png'))


@pytest.mark.skipif(not os.path.exists(EASY_SRC), reason=f"missing fixture: {EASY_SRC}")
def test_load_image_from_bytes_matches_load_image():
    """load_image_from_bytes should decode to the same content as load_image
    reading the same file directly (round-trip through raw bytes)."""
    with open(EASY_SRC, 'rb') as f:
        raw_bytes = f.read()
    img_from_bytes = load_image_from_bytes(raw_bytes)
    img_from_path = load_image(EASY_SRC)

    assert img_from_bytes.dtype == np.uint8
    assert img_from_bytes.shape == img_from_path.shape
    np.testing.assert_array_equal(img_from_bytes, img_from_path)


def test_load_image_from_bytes_raises_on_garbage():
    with pytest.raises(ValueError):
        load_image_from_bytes(b"not a real image file")


# ── src.preprocess.preprocess_pair ──────────────────────────────────

@pytest.mark.skipif(
    not (os.path.exists(EASY_SRC) and os.path.exists(EASY_REF)),
    reason="missing easy-tier fixture pair",
)
def test_preprocess_pair_on_easy_pair():
    img1 = load_image(EASY_SRC)
    img2 = load_image(EASY_REF)

    img1_clean, img2_clean = preprocess_pair(img1, img2, max_size=1024)

    assert isinstance(img1_clean, np.ndarray)
    assert isinstance(img2_clean, np.ndarray)
    assert img1_clean.dtype == np.uint8
    assert img2_clean.dtype == np.uint8

    # Resized to max_size=1024 on the longest edge (never upsampled if already smaller)
    assert max(img1_clean.shape) <= 1024
    assert max(img2_clean.shape) <= 1024

    # CLAHE-processed content should not be blank
    assert img1_clean.mean() > 2.0
    assert img2_clean.mean() > 2.0
    assert img1_clean.std() > 1.0
    assert img2_clean.std() > 1.0


@pytest.mark.skipif(
    not (os.path.exists(EASY_SRC) and os.path.exists(EASY_REF)),
    reason="missing easy-tier fixture pair",
)
def test_preprocess_pair_respects_max_size_shrink():
    img1 = load_image(EASY_SRC)
    img2 = load_image(EASY_REF)

    img1_small, img2_small = preprocess_pair(img1, img2, max_size=256)
    assert max(img1_small.shape) <= 256
    assert max(img2_small.shape) <= 256


# ── src.lro_loader.load_geotiff_reference (guarded) ─────────────────
# Requires the optional `rasterio` dependency and a real GeoTIFF fixture.
# data/ contains no OHRC raw .img/.xml pair or IIRS hyperspectral cube, so
# ohrc_loader.py / iirs_loader.py are intentionally not tested here rather
# than inventing fixture files that don't exist in this repo.

rasterio = pytest.importorskip(
    "rasterio", reason="rasterio not installed — skipping GeoTIFF loader test"
)


@pytest.mark.skipif(not os.path.exists(GEOTIFF_PATH), reason=f"missing fixture: {GEOTIFF_PATH}")
def test_load_geotiff_reference_smoke():
    from src.lro_loader import load_geotiff_reference

    meta = load_geotiff_reference(GEOTIFF_PATH, source_label="LRO_test")

    assert isinstance(meta['image'], np.ndarray)
    assert meta['image'].ndim == 2
    assert meta['image'].shape[0] > 0 and meta['image'].shape[1] > 0
    assert meta['source_label'] == 'LRO_test'
    assert meta['pixel_resolution_m'] > 0
    assert len(meta['corner_latlon']) == 4
    for lat, lon in meta['corner_latlon']:
        assert -90.0 <= lat <= 90.0
        assert -180.0 <= lon <= 180.0
