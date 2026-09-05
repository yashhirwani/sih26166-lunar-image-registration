"""
iirs_loader.py
--------------
Loads and processes Chandrayaan-2 IIRS (Imaging Infrared Spectrometer) data.

IIRS overview:
- Hyperspectral sensor: each pixel has reflectance values across ~256 narrow
  spectral bands covering ~0.8–5.0 µm (VNIR + SWIR range)
- Spatial resolution: ~80 m/px (coarser than OHRC at 0.25 m/px)
- Cannot be fed to SIFT/AKAZE directly — must first be synthesized into a
  single-channel image comparable to a panchromatic sensor

Implementation status
─────────────────────
✅ IMPLEMENTED (no dependency on real file format):
    synthesize_panchromatic()     — band-averaged pan synthesis, fully functional
    format_iirs_metadata_display() — metadata card for UI
    make_false_color_composite()  — RGB preview from 3 selected bands

⏳ STUBBED (waiting for real label file to fill in):
    parse_iirs_label()   — raises NotImplementedError
    load_iirs_cube()     — raises NotImplementedError

    These two will be completed once the downloaded IIRS label file contents
    are shared so exact XML/PDS3 field names can be confirmed.

Wavelength convention note:
    IIRS wavelengths are typically in micrometers (0.8–5.0 µm). This module
    normalises them to nanometers internally so all band_range_nm parameters
    are consistent with the function signatures. The conversion is applied
    inside parse_iirs_label() — synthesize_panchromatic() always receives nm.
"""

import numpy as np
import xml.etree.ElementTree as ET


# ── Stubbed functions (fill in once real label file is available) ─────────────

def parse_iirs_label(label_path: str) -> dict:
    """
    ⏳ STUBBED — implement once real IIRS label file is available.

    Parses the IIRS .xml/.lbl label file for cube geometry and band info.
    Adapt tag names to match the actual label structure — do NOT guess field
    names blindly. Inspect the downloaded label file and adjust accordingly.

    If the file is PDS4-style XML, use xml.etree.ElementTree (already imported).
    If the file is a classic PDS3 .LBL key=value file, use the pvl library:
        import pvl; label = pvl.load(label_path)

    Wavelength unit check: IIRS wavelengths are often in micrometers (µm).
    Before returning, convert to nanometers if needed:
        if max(wavelengths) < 10:   # still in µm
            wavelengths_nm = [w * 1000 for w in wavelengths]
    Then print the actual range found so the caller can sanity-check:
        print(f"[iirs_loader] Wavelength range: {wavelengths_nm[0]:.1f}–{wavelengths_nm[-1]:.1f} nm")

    Returns dict with keys:
        num_bands          : int
        num_rows           : int
        num_cols           : int
        wavelengths_nm     : list[float]  — center wavelength per band, in nm
        data_type          : str          — numpy dtype string e.g. "uint16"
        interleave         : str          — "BSQ" | "BIL" | "BIP"
        corner_latlon      : list         — [(lat,lon)×4] if in label, else []
        upper_left_lat/lon : float        — extracted from corner_latlon
        upper_right_lat/lon: float
        lower_left_lat/lon : float
        lower_right_lat/lon: float
        resolution_m_per_px: float        — from label; ~80 m/px expected
        pixel_resolution_m : float        — alias of resolution_m_per_px
        source_label       : str          — "IIRS"
    """
    raise NotImplementedError(
        "parse_iirs_label() is not yet implemented. "
        "Share the contents of the downloaded IIRS .xml or .lbl label file "
        "so the correct field names can be confirmed before coding this function."
    )


def load_iirs_cube(data_path: str, meta: dict) -> np.ndarray:
    """
    ⏳ STUBBED — implement once real IIRS label file is available.

    Loads the raw IIRS hyperspectral cube (.qub or .img) using the shape,
    dtype, and interleave information from parse_iirs_label().

    Returns np.ndarray shaped (num_bands, num_rows, num_cols) — BSQ-equivalent
    band-first layout regardless of the file's native interleave.

    Interleave reshaping guide (fill in the correct one once known):
        BSQ (band sequential):
            raw.reshape(num_bands, num_rows, num_cols)           # already band-first
        BIL (band interleaved by line):
            raw.reshape(num_rows, num_bands, num_cols)
            .transpose(1, 0, 2)                                  # → (bands, rows, cols)
        BIP (band interleaved by pixel):
            raw.reshape(num_rows, num_cols, num_bands)
            .transpose(2, 0, 1)                                  # → (bands, rows, cols)
    """
    raise NotImplementedError(
        "load_iirs_cube() is not yet implemented. "
        "This will be filled in once parse_iirs_label() is complete and "
        "the interleave format is confirmed from the real label file."
    )


# ── Fully implemented functions ───────────────────────────────────────────────

def synthesize_panchromatic(
    cube: np.ndarray,
    wavelengths_nm: list,
    band_range_nm: tuple = (800, 1000),
) -> np.ndarray:
    """
    Synthesizes a single-channel panchromatic-equivalent image from the
    IIRS hyperspectral cube by equally-weighted averaging of bands within
    band_range_nm.

    ⚠️  Approximation note:
    This is equal-weighted band averaging, NOT a radiometrically precise
    panchromatic simulation. An ideal implementation would weight each band
    by the target sensor's (OHRC / LRO NAC) spectral quantum efficiency curve,
    which is not publicly available. Equal weighting is an acceptable and
    clearly-labelled approximation for this timeline.

    IIRS wavelength coverage (~0.8–5.0 µm = 800–5000 nm):
    The default band_range_nm=(800, 1000) targets the VNIR boundary where IIRS
    has its shortest wavelengths — the closest analogue to a panchromatic band
    given that IIRS does not cover the visible range (< 800 nm).
    If you need to adjust this, call:
        synthesize_panchromatic(cube, wavelengths_nm, band_range_nm=(900, 2500))
    The function will print the bands selected so you can sanity-check.

    Parameters:
        cube           : np.ndarray, shape (num_bands, num_rows, num_cols)
        wavelengths_nm : list of float, center wavelength per band in nm
        band_range_nm  : (min_nm, max_nm) inclusive band selection window

    Returns:
        np.ndarray, shape (num_rows, num_cols), dtype uint8, values 0-255
    """
    # ── Band selection ─────────────────────────────────────────────
    band_indices = [
        i for i, wl in enumerate(wavelengths_nm)
        if band_range_nm[0] <= wl <= band_range_nm[1]
    ]

    # If nothing found in the requested range, report honestly and fall back
    # to the shortest-wavelength third of available bands
    if not band_indices:
        min_wl = min(wavelengths_nm)
        max_wl = max(wavelengths_nm)
        fallback_max = min_wl + (max_wl - min_wl) / 3
        band_indices = [
            i for i, wl in enumerate(wavelengths_nm)
            if wl <= fallback_max
        ]
        print(
            f"[iirs_loader] ⚠️  No IIRS bands in requested range {band_range_nm} nm. "
            f"Actual wavelength range: {min_wl:.1f}–{max_wl:.1f} nm. "
            f"Falling back to shortest-wavelength third: "
            f"{min_wl:.1f}–{fallback_max:.1f} nm "
            f"({len(band_indices)} bands). "
            f"Adjust band_range_nm to match this product's coverage."
        )
        if not band_indices:
            raise ValueError(
                f"Cannot synthesize panchromatic band: no bands available. "
                f"Wavelength range in cube: {min_wl:.1f}–{max_wl:.1f} nm."
            )
    else:
        selected_wls = [wavelengths_nm[i] for i in band_indices]
        print(
            f"[iirs_loader] synthesize_panchromatic: "
            f"using {len(band_indices)} bands "
            f"({selected_wls[0]:.1f}–{selected_wls[-1]:.1f} nm) "
            f"from range {band_range_nm} nm"
        )

    # ── Equal-weighted average ─────────────────────────────────────
    selected = cube[band_indices, :, :].astype(np.float32)
    pan = np.mean(selected, axis=0)          # (rows, cols), float32

    print(
        f"[iirs_loader] pan band before normalisation: "
        f"min={pan.min():.3f}  max={pan.max():.3f}  "
        f"mean={pan.mean():.3f}  shape={pan.shape}"
    )

    # ── Normalise to uint8 ─────────────────────────────────────────
    pan_min, pan_max = pan.min(), pan.max()
    pan_norm = (
        (pan - pan_min) / (pan_max - pan_min + 1e-8) * 255
    ).astype(np.uint8)

    return pan_norm


def make_false_color_composite(
    cube: np.ndarray,
    wavelengths_nm: list,
    rgb_wavelengths_nm: tuple = (2200, 1600, 1000),
) -> np.ndarray:
    """
    Generates an RGB false-color composite by mapping three IIRS bands to
    R, G, B channels. Default wavelengths (2200, 1600, 1000 nm) highlight
    mineral spectral differences visible in SWIR — useful for showing judges
    that the hyperspectral cube contains meaningful spectral information.

    Parameters:
        cube                 : (bands, rows, cols) array
        wavelengths_nm       : center wavelength per band in nm
        rgb_wavelengths_nm   : (R_nm, G_nm, B_nm) target wavelengths

    Returns:
        np.ndarray, shape (rows, cols, 3), dtype uint8
    """
    def nearest_band(target_nm):
        diffs = [abs(wl - target_nm) for wl in wavelengths_nm]
        return int(np.argmin(diffs))

    channels = []
    for target_nm in rgb_wavelengths_nm:
        idx = nearest_band(target_nm)
        actual_nm = wavelengths_nm[idx]
        band = cube[idx, :, :].astype(np.float32)
        # Percentile stretch for better visual contrast
        p2, p98 = np.percentile(band, 2), np.percentile(band, 98)
        stretched = np.clip((band - p2) / (p98 - p2 + 1e-8) * 255, 0, 255).astype(np.uint8)
        channels.append(stretched)
        print(f"[iirs_loader] false-color channel: target={target_nm} nm → actual={actual_nm:.1f} nm (band {idx})")

    return np.stack(channels, axis=2)   # (rows, cols, 3) RGB


def format_iirs_metadata_display(meta: dict) -> str:
    """
    Formatted metadata card matching the style of ohrc_loader's and
    lro_loader's display functions, for consistency in the UI.

    Works with both a fully-populated meta dict (from parse_iirs_label)
    and a partially-populated one (for UI placeholder state before loader
    functions are implemented).
    """
    num_bands   = meta.get('num_bands', '?')
    num_rows    = meta.get('num_rows', '?')
    num_cols    = meta.get('num_cols', '?')
    res         = meta.get('resolution_m_per_px', meta.get('pixel_resolution_m', 0))
    wls         = meta.get('wavelengths_nm', [])
    interleave  = meta.get('interleave', '?')
    data_type   = meta.get('data_type', '?')

    if wls:
        wl_range = f"{wls[0]:.1f}–{wls[-1]:.1f} nm  ({len(wls)} bands)"
    else:
        wl_range = "unknown (label not yet parsed)"

    ul_lat = meta.get('upper_left_lat', '?')
    ul_lon = meta.get('upper_left_lon', '?')
    lr_lat = meta.get('lower_right_lat', '?')
    lr_lon = meta.get('lower_right_lon', '?')

    return f"""
IIRS Hyperspectral Cube
═══════════════════════════════════════
Sensor            : Chandrayaan-2 IIRS
Spatial Size      : {num_rows} rows × {num_cols} cols
Resolution        : {res:.1f} m/pixel
Bands             : {num_bands}
Wavelength range  : {wl_range}
Interleave        : {interleave}
Data type         : {data_type}

Corner Coordinates (Lunar Lat/Lon)
─────────────────────────────────────
Upper Left        : {ul_lat}, {ul_lon}
Lower Right       : {lr_lat}, {lr_lon}
═══════════════════════════════════════
⚠️  Panchromatic synthesis is an equal-weighted band average,
    not a precise radiometric match to OHRC / LRO NAC response.
"""
