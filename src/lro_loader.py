"""
lro_loader.py
-------------
Loads georeferenced LRO NAC / WAC / SELENE reference tiles (GeoTIFF).

Why GeoTIFF and not raw PDS3?
- PDS3 .IMG/.LBL format requires a custom label parser — not worth the
  implementation time this close to demo day.
- QuickMap (https://quickmap.lroc.asu.edu) exports GeoTIFF clips with an
  embedded affine transform and CRS, so no label parsing is needed at all.
- rasterio reads GeoTIFF natively in two lines of code.

How to get a tile (human step):
1. Open your OHRC image's XML, note corner lat/lon (already in metadata dict
   produced by ohrc_loader.parse_ohrc_xml()).
2. Go to https://quickmap.lroc.asu.edu
3. Navigate to that region, select LRO NAC or WAC layer.
4. Export / clip to the bounding box covering your OHRC footprint.
5. Save as data/reference/lro_<region_name>.tif

Return dict key names are aligned with ohrc_loader.parse_ohrc_xml() output
so downstream code can treat OHRC and LRO metadata dicts consistently:
  corner coordinates  → upper_left_lat / _lon, upper_right_lat / _lon,
                         lower_left_lat / _lon, lower_right_lat / _lon
  resolution          → pixel_resolution_m
  acquisition context → source_label, crs
"""

import numpy as np

MOON_RADIUS_M = 1_737_400.0  # mean lunar radius in metres


def load_geotiff_reference(path: str, source_label: str = "LRO_reference") -> dict:
    """
    Loads a georeferenced LRO NAC/WAC or SELENE GeoTIFF tile.

    Parameters:
        path         : path to the .tif file
        source_label : human-readable label shown in the UI
                       (e.g. "LRO NAC", "LRO WAC", "SELENE TC")

    Returns dict with keys:
        image               : np.ndarray, uint8 or uint16, single channel
        transform           : rasterio Affine object
        crs                 : CRS string (e.g. "EPSG:4326")
        corner_latlon       : list of 4 (lat, lon) tuples,
                              order: UL, UR, LR, LL  (same as OHRC xml corners)
        upper_left_lat/lon  : float  ─┐
        upper_right_lat/lon : float   │ mirrored from corner_latlon
        lower_left_lat/lon  : float   │ so downstream code works uniformly
        lower_right_lat/lon : float  ─┘
        pixel_resolution_m  : float, ground sample distance in metres/pixel
        resolution_m_per_px : float, alias for pixel_resolution_m
        source_label        : str
        shape               : (height, width) of the loaded image
        dtype               : original numpy dtype before any conversion
    """
    try:
        import rasterio
    except ImportError:
        raise ImportError(
            "rasterio is required for GeoTIFF loading. "
            "Install it with: pip install rasterio"
        )

    with rasterio.open(path) as ds:
        crs = ds.crs
        if crs is None:
            raise ValueError(
                f"GeoTIFF at {path} has no CRS embedded. "
                "Re-export from QuickMap ensuring 'geographic (lat/lon)' is selected."
            )

        # Read band 1 as grayscale. Do NOT reproject the raster — even a
        # projected CRS (Moon polar stereographic) is kept in native pixels.
        raw = ds.read(1)
        original_dtype = raw.dtype

        if raw.dtype != np.uint8:
            raw_min, raw_max = raw.min(), raw.max()
            if raw_max > raw_min:
                image = ((raw.astype(np.float32) - raw_min)
                         / (raw_max - raw_min) * 255).astype(np.uint8)
            else:
                image = np.zeros_like(raw, dtype=np.uint8)
        else:
            image = raw

        transform = ds.transform
        bounds = ds.bounds
        transformer = None
        is_projected = not crs.is_geographic

        if crs.is_geographic:
            corner_latlon = [
                (bounds.top,    bounds.left),
                (bounds.top,    bounds.right),
                (bounds.bottom, bounds.right),
                (bounds.bottom, bounds.left),
            ]
            deg_to_m = (np.pi * MOON_RADIUS_M) / 180.0
            resolution_m = abs(transform.a) * deg_to_m
        else:
            # Projected CRS (e.g. Moon polar stereographic, units in metres).
            # Convert only the four corner points to geographic lat/lon for
            # overlap detection — do NOT reproject the full raster.
            import pyproj
            geo_crs = crs.geodetic_crs
            if geo_crs is None:
                raise ValueError(
                    f"Projected CRS has no geodetic base CRS; cannot convert "
                    f"corners to lat/lon. CRS={crs}"
                )
            transformer = pyproj.Transformer.from_crs(
                crs, geo_crs, always_xy=True
            )
            corners_projected = [
                (bounds.left,  bounds.top),
                (bounds.right, bounds.top),
                (bounds.right, bounds.bottom),
                (bounds.left,  bounds.bottom),
            ]
            corner_latlon = []
            for x, y in corners_projected:
                lon, lat = transformer.transform(x, y)
                corner_latlon.append((float(lat), float(lon)))
            resolution_m = abs(transform.a)  # already metres/pixel

    lats = [c[0] for c in corner_latlon]
    lons = [c[1] for c in corner_latlon]

    meta = {
        "image":               image,
        "transform":           transform,
        "crs":                 str(crs),
        "shape":               image.shape,
        "dtype":               str(original_dtype),

        "corner_latlon":        corner_latlon,
        "upper_left_lat":       corner_latlon[0][0],
        "upper_left_lon":       corner_latlon[0][1],
        "upper_right_lat":      corner_latlon[1][0],
        "upper_right_lon":      corner_latlon[1][1],
        "lower_right_lat":      corner_latlon[2][0],
        "lower_right_lon":      corner_latlon[2][1],
        "lower_left_lat":       corner_latlon[3][0],
        "lower_left_lon":       corner_latlon[3][1],

        "pixel_resolution_m":  resolution_m,
        "resolution_m_per_px": resolution_m,

        "source_label":        source_label,
        "is_projected":        is_projected,
        "pyproj_transformer_to_geographic": transformer,
    }

    print(f"[lro_loader] Loaded: {path}")
    print(f"             Shape  : {image.shape}  dtype_original={original_dtype}")
    print(f"             CRS    : {crs}")
    print(f"             Projected: {is_projected}")
    print(f"             Corner lat/lon (UL, UR, LR, LL): {corner_latlon}")
    print(f"             Geographic bbox: lat [{min(lats):.4f}, {max(lats):.4f}]  "
          f"lon [{min(lons):.4f}, {max(lons):.4f}]")
    print(f"             GSD    : {resolution_m:.3f} m/px")

    return meta


def format_lro_metadata_display(meta: dict) -> str:
    """
    Formats LRO metadata into a readable string for display in the UI.
    Style matches ohrc_loader.format_metadata_display() so both appear
    visually consistent in the Streamlit tabs.
    """
    return f"""
LRO / External Reference Metadata
═══════════════════════════════════════
Source Label      : {meta.get('source_label', 'N/A')}
CRS               : {meta.get('crs', 'N/A')}
Projected CRS     : {meta.get('is_projected', False)}
Image Shape       : {meta.get('shape', 'N/A')}
Original Dtype    : {meta.get('dtype', 'N/A')}

Resolution
─────────────────────────────────────
GSD               : {meta.get('pixel_resolution_m', 0):.2f} m/pixel

Corner Coordinates (Lunar Lat/Lon)
─────────────────────────────────────
Upper Left        : {meta.get('upper_left_lat', 0):.4f}°, {meta.get('upper_left_lon', 0):.4f}°
Upper Right       : {meta.get('upper_right_lat', 0):.4f}°, {meta.get('upper_right_lon', 0):.4f}°
Lower Left        : {meta.get('lower_left_lat', 0):.4f}°, {meta.get('lower_left_lon', 0):.4f}°
Lower Right       : {meta.get('lower_right_lat', 0):.4f}°, {meta.get('lower_right_lon', 0):.4f}°
═══════════════════════════════════════
"""
