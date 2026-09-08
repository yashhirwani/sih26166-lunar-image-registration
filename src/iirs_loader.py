"""
iirs_loader.py
--------------
Loads and processes Chandrayaan-2 IIRS (Imaging Infrared Spectrometer) data.

IIRS overview:
- Hyperspectral sensor: each pixel has reflectance values across ~256 narrow
  spectral bands covering ~0.71–5.0 µm (VNIR + SWIR range)
- Spatial resolution: ~80 m/px (coarser than OHRC at 0.25 m/px) — GSD lives
  in the companion _d_obs_ product, not in the reflectance .hdr
- Cannot be fed to SIFT/AKAZE directly — must first be synthesized into a
  single-channel image comparable to a panchromatic sensor

Real product format (confirmed from ch2_iir_*_d_rfl_d18_srd):
- Geometry and wavelengths come from a plain-text ENVI .hdr (not XML/PDS4)
- Cube is a headerless .qub (raw binary, ENVI Standard)
- Geolocation (corner lat/lon) is in a separate _d_loc_ product — left as
  None here and not guessed
"""

import re
import numpy as np


# ENVI data type codes: 1=byte, 2=int16, 3=int32, 4=float32, 5=float64, 12=uint16
_ENVI_TYPE_MAP = {
    1: "uint8",
    2: "int16",
    3: "int32",
    4: "float32",
    5: "float64",
    12: "uint16",
}


def parse_iirs_label(hdr_path: str) -> dict:
    """
    Parses an ENVI-format .hdr text file (not XML/PDS4) for IIRS cube geometry.

    This reflectance product's header does not include geolocation or GSD;
    those live in companion _d_loc_ / _d_obs_ products. corner_latlon and
    resolution_m_per_px are returned as None and must not be guessed.
    """
    with open(hdr_path, "r") as f:
        content = f.read()

    def extract_scalar(key, cast=int):
        match = re.search(rf"{key}\s*=\s*(\S+)", content)
        if not match:
            raise ValueError(f"Could not find '{key}' in header file {hdr_path}")
        return cast(match.group(1))

    def extract_wavelength_list(text):
        match = re.search(r"wavelength\s*=\s*\{([^}]+)\}", text, re.DOTALL)
        if not match:
            raise ValueError(f"Could not find wavelength list in header file {hdr_path}")
        return [float(v.strip()) for v in match.group(1).split(",") if v.strip()]

    num_cols = extract_scalar("samples", int)
    num_rows = extract_scalar("lines", int)
    num_bands = extract_scalar("bands", int)
    envi_data_type = extract_scalar("data type", int)
    interleave_match = re.search(r"interleave\s*=\s*(\w+)", content)
    if not interleave_match:
        raise ValueError(f"Could not find 'interleave' in header file {hdr_path}")
    interleave = interleave_match.group(1).lower()
    byte_order = extract_scalar("byte order", int)
    wavelengths_nm = extract_wavelength_list(content)

    if envi_data_type not in _ENVI_TYPE_MAP:
        raise ValueError(
            f"Unhandled ENVI data type code {envi_data_type} — "
            f"extend _ENVI_TYPE_MAP if needed"
        )
    numpy_dtype = _ENVI_TYPE_MAP[envi_data_type]

    if len(wavelengths_nm) != num_bands:
        raise ValueError(
            f"Wavelength count ({len(wavelengths_nm)}) doesn't match band count "
            f"({num_bands}) — check the .hdr file for a parsing error"
        )

    print(
        f"[iirs_loader] Wavelength range: "
        f"{wavelengths_nm[0]:.1f}–{wavelengths_nm[-1]:.1f} nm "
        f"({num_bands} bands, {num_rows}×{num_cols}, {interleave} {numpy_dtype})"
    )

    return {
        "num_bands": num_bands,
        "num_rows": num_rows,
        "num_cols": num_cols,
        "wavelengths_nm": wavelengths_nm,
        "data_type": numpy_dtype,
        "interleave": interleave,
        "byte_order": "little" if byte_order == 0 else "big",
        "corner_latlon": None,  # not in this .hdr — would come from _d_loc_
        "resolution_m_per_px": None,  # not in this .hdr — would come from _d_obs_
        "source_label": "IIRS",
    }


def load_iirs_cube(qub_path: str, meta: dict) -> np.ndarray:
    """
    Loads the raw IIRS hyperspectral cube from a .qub file (raw binary,
    same format as ENVI's .img/.bin — no header inside the .qub itself,
    all geometry comes from the parsed .hdr in meta).

    Returns array shaped (bands, rows, cols) — i.e. (256, num_rows, num_cols) —
    regardless of the file's native interleave, for consistency with the
    rest of this module.
    """
    dtype = np.dtype(meta["data_type"])
    if meta["byte_order"] == "little":
        dtype = dtype.newbyteorder("<")
    else:
        dtype = dtype.newbyteorder(">")

    raw = np.fromfile(qub_path, dtype=dtype)

    expected_size = meta["num_bands"] * meta["num_rows"] * meta["num_cols"]
    if raw.size != expected_size:
        raise ValueError(
            f"File size mismatch: got {raw.size} values, expected {expected_size} "
            f"({meta['num_bands']} bands x {meta['num_rows']} rows x "
            f"{meta['num_cols']} cols). "
            f"Check the .hdr dimensions match the actual .qub file size."
        )

    if meta["interleave"] == "bsq":
        cube = raw.reshape((meta["num_bands"], meta["num_rows"], meta["num_cols"]))
    elif meta["interleave"] == "bil":
        cube = raw.reshape(
            (meta["num_rows"], meta["num_bands"], meta["num_cols"])
        ).transpose(1, 0, 2)
    elif meta["interleave"] == "bip":
        cube = raw.reshape(
            (meta["num_rows"], meta["num_cols"], meta["num_bands"])
        ).transpose(2, 0, 1)
    else:
        raise ValueError(f"Unknown interleave format: {meta['interleave']}")

    return cube


# ── Fully implemented functions ───────────────────────────────────────────────

def synthesize_panchromatic(
    cube: np.ndarray,
    wavelengths_nm: list,
    band_range_nm: tuple = (712, 950),
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

    IIRS wavelength coverage for this product (~0.71–5.0 µm = 712–5009 nm):
    The default band_range_nm=(712, 950) uses the shortest-wavelength
    (most visible-light-like) bands actually present. The shortest center
    wavelength in the real reflectance .hdr is 712.3 nm, not 800 nm.
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

    # ── Normalise to uint8 (percentile stretch) ────────────────────
    # Min-max stretch is crushed by a few saturated outliers (this product
    # has max ≫ mean). A 2–98 percentile stretch matches the false-color
    # preview and yields a viewable grayscale image.
    p2, p98 = np.percentile(pan, 2), np.percentile(pan, 98)
    pan_norm = np.clip(
        (pan - p2) / (p98 - p2 + 1e-8) * 255, 0, 255
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
    res         = meta.get('resolution_m_per_px', meta.get('pixel_resolution_m'))
    wls         = meta.get('wavelengths_nm', [])
    interleave  = meta.get('interleave', '?')
    data_type   = meta.get('data_type', '?')

    if wls:
        wl_range = f"{wls[0]:.1f}–{wls[-1]:.1f} nm  ({len(wls)} bands)"
    else:
        wl_range = "unknown (label not yet parsed)"

    if res is None:
        res_str = "not in this .hdr (see _d_obs_ product)"
    else:
        res_str = f"{res:.1f} m/pixel"

    corners = meta.get('corner_latlon')
    if corners:
        ul_lat, ul_lon = corners[0]
        lr_lat, lr_lon = corners[2]
        ul_str = f"{ul_lat}, {ul_lon}"
        lr_str = f"{lr_lat}, {lr_lon}"
    else:
        ul_str = lr_str = "not in this .hdr (see _d_loc_ product)"

    return f"""
IIRS Hyperspectral Cube
═══════════════════════════════════════
Sensor            : Chandrayaan-2 IIRS
Spatial Size      : {num_rows} rows × {num_cols} cols
Resolution        : {res_str}
Bands             : {num_bands}
Wavelength range  : {wl_range}
Interleave        : {interleave}
Data type         : {data_type}

Corner Coordinates (Lunar Lat/Lon)
─────────────────────────────────────
Upper Left        : {ul_str}
Lower Right       : {lr_str}
═══════════════════════════════════════
⚠️  Panchromatic synthesis is an equal-weighted band average,
    not a precise radiometric match to OHRC / LRO NAC response.
"""
