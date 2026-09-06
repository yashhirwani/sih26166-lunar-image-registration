"""
app.py
------
Streamlit web UI for the Lunar Image Registration System.
PS166 — SIH 2026

Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import cv2
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))
from src.preprocess import load_image_from_bytes
from src.pipeline import run_pipeline
from src.visualize import numpy_to_bytes

# ── Page Configuration ─────────────────────────────────────────────
st.set_page_config(
    page_title="Lunar Image Registration | PS166",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom Styling ─────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.5em;
        font-weight: bold;
        color: #E8E8E8;
        text-align: center;
        padding: 20px 0;
    }
    .subtitle {
        font-size: 1.1em;
        color: #AAAAAA;
        text-align: center;
        margin-bottom: 30px;
    }
    .metric-card {
        background: #1E1E1E;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #333;
    }
    .metric-value {
        font-size: 2em;
        font-weight: bold;
        color: #00FF88;
    }
    .metric-label {
        color: #AAAAAA;
        font-size: 0.9em;
    }
    .success-badge {
        background: #00FF8822;
        border: 1px solid #00FF88;
        color: #00FF88;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
    }
    .fail-badge {
        background: #FF444422;
        border: 1px solid #FF4444;
        color: #FF4444;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🌙 Lunar Image Registration System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">PS166 — SIH 2026 | Chandrayaan-2 OHRC Image Registration with Sub-pixel Accuracy</div>', unsafe_allow_html=True)
st.divider()

# ── Sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    method = st.selectbox(
        "Algorithm",
        ["auto", "lightglue", "loftr", "pc-sift", "akaze", "sift"],
        help="Auto = best-of-all (AKAZE/SIFT/LightGlue/PC-SIFT/LoFTR). PC-SIFT = illumination-invariant phase congruency — best for different sun angles."
    )

    max_size = st.slider(
        "Max Image Size (px)",
        min_value=256,
        max_value=2048,
        value=1024,
        step=128,
        help="Larger = more accurate but slower"
    )

    st.divider()
    st.markdown("**About Algorithms:**")
    st.markdown("🟡 **Auto** — Smart router: Classical→LightGlue→LoFTR")
    st.markdown("⚡ **LightGlue** — Best for hard cases (DISK+LightGlue)")
    st.markdown("🔵 **AKAZE** — Fast classical, illumination-robust")
    st.markdown("🟢 **SIFT** — Classic, reliable, scale-invariant")
    st.markdown("🔴 **LoFTR** — Dense deep learning matcher")
    st.markdown("🟣 **PC-SIFT** — Phase congruency (illumination-invariant)")

    st.divider()
    st.markdown("**PS Requirements:**")
    st.markdown("✅ Sub-pixel accuracy (RMSE < 1.0)")
    st.markdown("✅ Uniform match distribution")
    st.markdown("✅ RMSE, Inlier count, Inlier ratio")
    st.markdown("✅ Match point visualization")
    st.markdown("✅ Registered output image")
    st.markdown("✅ Illumination-aware preprocessing")
    st.markdown("✅ Real OHRC .img data support")

# ── Main Content ───────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📤 Upload & Run",
    "🛰️ Real OHRC Data",
    "🔗 Match Points",
    "🎯 Registration Result",
    "📊 Metrics"
])

# ── Tab 1: Upload ──────────────────────────────────────────────────
with tab1:
    st.subheader("Upload Images")
    st.markdown("Upload a **Source Image** and a **Reference Image** of the same lunar region.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Source Image** (Moving — will be aligned)")
        source_file = st.file_uploader(
            "Upload source image",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            key="source",
            label_visibility="collapsed"
        )
        if source_file:
            source_bytes = source_file.read()
            source_img = load_image_from_bytes(source_bytes)
            st.image(source_img, caption=f"Source: {source_file.name} | Shape: {source_img.shape}", use_container_width=True)

    with col2:
        st.markdown("**Reference Image** (Fixed — target)")
        reference_file = st.file_uploader(
            "Upload reference image",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            key="reference",
            label_visibility="collapsed"
        )
        if reference_file:
            reference_bytes = reference_file.read()
            reference_img = load_image_from_bytes(reference_bytes)
            st.image(reference_img, caption=f"Reference: {reference_file.name} | Shape: {reference_img.shape}", use_container_width=True)

    st.divider()

    # Run button
    run_btn = st.button(
        "🚀 Run Registration",
        type="primary",
        use_container_width=True,
        disabled=not (source_file and reference_file)
    )

    if not (source_file and reference_file):
        st.info("👆 Upload both images to enable registration")

    # Run pipeline
    if run_btn and source_file and reference_file:
        with st.spinner("🔄 Running registration pipeline..."):
            try:
                result = run_pipeline(
                    source_img, reference_img,
                    method=method,
                    max_size=max_size
                )
                st.session_state['result'] = result

                if not result.get('success', True):
                    # Graceful failure
                    st.error(f"❌ Registration failed: {result.get('failure_reason', 'Unknown error')}")
                    st.markdown("**Suggestions:**")
                    st.markdown("- Make sure both images show the same lunar region")
                    st.markdown("- Try the **LightGlue** algorithm for hard cases")
                    st.markdown("- Check that images are not all-black or corrupted")
                else:
                    # Success
                    confidence = result.get('confidence', 'medium')
                    conf_color = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(confidence, "⚪")
                    st.success(f"✅ Registration complete! Algorithm: **{result['method']}** | Confidence: {conf_color} **{confidence.upper()}**")

                    # Show escalation log if auto-routing happened
                    if result.get('escalation_log'):
                        with st.expander("🔀 Algorithm routing details"):
                            for log in result['escalation_log']:
                                st.text(log)

                    # Quick metrics preview
                    metrics = result['metrics']
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        sub = "✅ Sub-pixel" if metrics['rmse'] < 1.0 else "⚠️ Not sub-pixel"
                        st.metric("RMSE", f"{metrics['rmse']:.4f} px", sub)
                    with c2:
                        st.metric("Inlier Count", metrics['inlier_count'])
                    with c3:
                        st.metric("Inlier Ratio", f"{metrics['inlier_ratio']:.2%}")
                    with c4:
                        st.metric("Spatial Score", f"{metrics['spatial_score']:.4f}")

                    st.info("👆 Check the other tabs for detailed visualizations and metrics")

            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")
                st.markdown("**Possible reasons:**")
                st.markdown("- Images don't overlap (different lunar regions)")
                st.markdown("- Images too different in lighting")
                st.markdown("- Try **LightGlue** algorithm in settings")

# ── Tab 2: Real OHRC Data ──────────────────────────────────────────
with tab2:
    st.subheader("🛰️ Real OHRC Data & Cross-Source Registration")
    st.markdown("Load actual Chandrayaan-2 OHRC data and optionally register against an LRO NAC/WAC GeoTIFF or IIRS hyperspectral reference.")

    # ── Reference source selector ──────────────────────────────────
    ref_source_mode = st.radio(
        "Reference source",
        ["OHRC (Chandrayaan-2 zip)", "LRO NAC / SELENE (GeoTIFF)", "IIRS (hyperspectral)"],
        horizontal=True,
        help="OHRC = two OHRC images.  LRO NAC = OHRC vs external GeoTIFF.  IIRS = hyperspectral cube synthesised to panchromatic then registered."
    )

    st.divider()

    # ── Source OHRC zip (always shown) ────────────────────────────
    # ── Source / Reference inputs ──────────────────────────────────
    # IIRS mode has a completely different layout — show it separately
    if ref_source_mode == "IIRS (hyperspectral)":
        st.info(
            "ℹ️ IIRS is a hyperspectral sensor (~0.8–5.0 µm, ~256 bands, ~80 m/px). "
            "The pipeline synthesizes a panchromatic-equivalent image from the cube "
            "using equal-weighted band averaging (450–750 nm window or nearest available). "
            "This is an approximation, clearly labelled in results."
        )

        iirs_col1, iirs_col2 = st.columns(2)
        with iirs_col1:
            st.markdown("**IIRS cube file path** (.qub or .img):")
            iirs_cube_path = st.text_input(
                "IIRS cube path",
                placeholder="/path/to/iirs_data.qub",
                label_visibility="collapsed",
                key="iirs_cube_path"
            )
            st.markdown("**IIRS label file path** (.xml or .lbl):")
            iirs_label_path = st.text_input(
                "IIRS label path",
                placeholder="/path/to/iirs_label.xml",
                label_visibility="collapsed",
                key="iirs_label_path"
            )
        with iirs_col2:
            st.markdown("**Reference image path** (OHRC zip or LRO GeoTIFF):")
            iirs_ref_path = st.text_input(
                "Reference path",
                placeholder="/path/to/ohrc_reference.zip  or  lro_nac.tif",
                label_visibility="collapsed",
                key="iirs_ref_path"
            )
            iirs_ref_type = st.selectbox(
                "Reference type",
                ["OHRC zip", "LRO GeoTIFF"],
                key="iirs_ref_type"
            )

        iirs_load_btn = st.button(
            "🔬 Load & Preview IIRS Data",
            type="primary",
            use_container_width=True,
            disabled=not (iirs_cube_path and iirs_label_path and iirs_ref_path),
            key="iirs_load_btn"
        )

        if not (iirs_cube_path and iirs_label_path and iirs_ref_path):
            st.info("Enter IIRS cube path, label path, and reference path above.")
            st.markdown("**Note:** `parse_iirs_label()` and `load_iirs_cube()` are currently "
                        "⏳ **stubbed** — they will be completed once the real IIRS label "
                        "file structure is confirmed.")

        if iirs_load_btn and iirs_cube_path and iirs_label_path and iirs_ref_path:
            with st.spinner("Loading IIRS data..."):
                try:
                    from src.iirs_loader import (parse_iirs_label, load_iirs_cube,
                                                  synthesize_panchromatic,
                                                  make_false_color_composite,
                                                  format_iirs_metadata_display)

                    # Parse label and load cube
                    iirs_meta = parse_iirs_label(iirs_label_path)
                    iirs_cube = load_iirs_cube(iirs_cube_path, iirs_meta)

                    st.session_state['iirs_cube']   = iirs_cube
                    st.session_state['iirs_meta']   = iirs_meta
                    st.session_state['iirs_mode']   = True
                    st.session_state['iirs_ref_path'] = iirs_ref_path
                    st.session_state['iirs_ref_type'] = iirs_ref_type

                    st.success(f"✅ IIRS cube loaded: {iirs_cube.shape} (bands × rows × cols)")

                    # ── Metadata card ──────────────────────────────
                    st.code(format_iirs_metadata_display(iirs_meta))

                    # ── Synthesized pan band preview ───────────────
                    pan = synthesize_panchromatic(
                        iirs_cube, iirs_meta['wavelengths_nm']
                    )
                    st.session_state['iirs_pan'] = pan

                    # ── False-color composite preview ──────────────
                    try:
                        fc = make_false_color_composite(
                            iirs_cube, iirs_meta['wavelengths_nm']
                        )
                    except Exception as fc_err:
                        fc = None
                        st.warning(f"False-color composite unavailable: {fc_err}")

                    # Display side by side
                    prev_col1, prev_col2 = st.columns(2)
                    with prev_col1:
                        st.markdown("**Synthesized panchromatic band** (used for registration)")
                        st.image(pan, caption=f"Pan band — shape {pan.shape}, "
                                 f"equal-weighted avg of selected IIRS bands",
                                 use_container_width=True)
                    with prev_col2:
                        if fc is not None:
                            st.markdown("**False-color composite** (3 IIRS bands as RGB)")
                            st.image(fc, caption="False-color — highlights mineral spectral differences",
                                     use_container_width=True)
                        else:
                            st.markdown("**False-color composite** — unavailable")

                    st.session_state['iirs_ready'] = True

                except NotImplementedError as e:
                    st.error(
                        f"⏳ **Loader not yet implemented:** {e}\n\n"
                        "Share the IIRS label file contents so `parse_iirs_label()` "
                        "and `load_iirs_cube()` can be completed."
                    )
                except Exception as e:
                    st.error(f"❌ Failed to load IIRS data: {e}")

        # Register button for IIRS mode
        if st.session_state.get('iirs_ready'):
            if st.button("🚀 Register IIRS vs Reference", type="primary",
                         use_container_width=True, key="iirs_register_btn"):
                with st.spinner("Running IIRS registration..."):
                    try:
                        from src.pipeline import register_iirs
                        from src.ohrc_loader import extract_browse_png_from_zip
                        from src.lro_loader import load_geotiff_reference

                        iirs_cube   = st.session_state['iirs_cube']
                        iirs_meta   = st.session_state['iirs_meta']
                        ref_path    = st.session_state['iirs_ref_path']
                        ref_type    = st.session_state['iirs_ref_type']

                        if ref_type == "OHRC zip":
                            ref_img, ref_meta = extract_browse_png_from_zip(ref_path)
                        else:
                            ref_meta_full = load_geotiff_reference(ref_path)
                            ref_img  = ref_meta_full['image']
                            ref_meta = ref_meta_full

                        result = register_iirs(
                            iirs_cube, iirs_meta,
                            ref_img, ref_meta,
                            method=method,
                            max_size=max_size,
                        )
                        st.session_state['result'] = result

                        if not result.get('success', True):
                            st.error(f"❌ {result.get('failure_reason', 'Registration failed')}")
                        else:
                            metrics    = result['metrics']
                            confidence = result.get('confidence', 'medium')
                            conf_icon  = {"high":"🟢","medium":"🟡","low":"🔴","failed":"❌"}.get(confidence,"⚪")
                            st.success(f"✅ IIRS registration complete! "
                                       f"Algorithm: **{result['method']}** | "
                                       f"Confidence: {conf_icon} **{confidence.upper()}**")
                            if result.get('degenerate_fit'):
                                st.warning(f"⚠️ {result.get('reliability_reason','')}")

                            # IIRS-specific result info
                            wl_range = result.get('iirs_wavelength_range_nm', ('?','?'))
                            st.info(
                                f"📡 IIRS wavelengths used: {wl_range[0]:.0f}–{wl_range[1]:.0f} nm  |  "
                                f"Pan band shape: {result.get('iirs_pan_shape','?')}  |  "
                                f"Target GSD: {result.get('iirs_target_res_m',0):.1f} m/px"
                            )

                            c1, c2, c3, c4 = st.columns(4)
                            with c1: st.metric("RMSE", f"{metrics['rmse']:.4f} px")
                            with c2: st.metric("Inlier Count", metrics['inlier_count'])
                            with c3: st.metric("Inlier Ratio", f"{metrics['inlier_ratio']:.2%}")
                            with c4: st.metric("Spatial Score", f"{metrics['spatial_score']:.4f}")
                            st.info("Check Match Points, Registration Result, and Metrics tabs.")
                    except Exception as e:
                        st.error(f"❌ IIRS registration failed: {e}")

    else:
        # ── Non-IIRS: original OHRC / LRO column layout ───────────
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Source OHRC zip file path:**")
            src_zip = st.text_input(
                "Source zip path",
                placeholder="/Users/kajol/Desktop/ps166/pradan.issdc.gov.in/.../ch2_ohr_ncp_....zip",
                label_visibility="collapsed"
            )

        with col2:
            if ref_source_mode == "OHRC (Chandrayaan-2 zip)":
                st.markdown("**Reference OHRC zip file path:**")
                ref_zip = st.text_input(
                    "Reference zip path",
                    placeholder="/Users/kajol/Desktop/ps166/pradan.issdc.gov.in/.../ch2_ohr_ncp_....zip",
                    label_visibility="collapsed"
                )
                lro_tif_path = None
            else:
                st.markdown("**LRO NAC / SELENE GeoTIFF path:**")
                lro_tif_path = st.text_input(
                    "GeoTIFF path",
                    placeholder="/Users/kajol/Desktop/ps166/lunar_registration/data/reference/lro_south_pole.tif",
                    label_visibility="collapsed"
                )
                ref_zip = None
                st.caption(
                    "Export from https://quickmap.lroc.asu.edu — navigate to your OHRC region, "
                    "select LRO NAC or WAC layer, export as GeoTIFF in geographic (lat/lon) CRS."
                )

    # ── Load button ────────────────────────────────────────────────
    inputs_ready = bool(src_zip and (ref_zip or lro_tif_path))
    load_btn = st.button("🛰️ Load Data", type="primary", use_container_width=True,
                         disabled=not inputs_ready)

    if load_btn and inputs_ready:
        with st.spinner("Loading data..."):
            try:
                from src.ohrc_loader import extract_browse_png_from_zip, format_metadata_display

                # Always load source OHRC
                src_patch, src_meta = extract_browse_png_from_zip(src_zip)

                if ref_source_mode == "OHRC (Chandrayaan-2 zip)":
                    # ── OHRC vs OHRC mode ──────────────────────────
                    ref_patch, ref_meta = extract_browse_png_from_zip(ref_zip)

                    st.session_state['ohrc_source']    = src_patch
                    st.session_state['ohrc_reference'] = ref_patch
                    st.session_state['ohrc_src_meta']  = src_meta
                    st.session_state['ohrc_ref_meta']  = ref_meta
                    st.session_state['cross_source']   = False
                    st.session_state['ohrc_ready']     = True
                    st.session_state['ohrc_src_zip']   = src_zip
                    st.session_state['ohrc_ref_zip']   = ref_zip

                    st.success("✅ Both OHRC images loaded!")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(src_patch,
                                 caption=f"Source {src_patch.shape}",
                                 use_container_width=True)
                        st.code(format_metadata_display(src_meta))
                    with col2:
                        st.image(ref_patch,
                                 caption=f"Reference {ref_patch.shape}",
                                 use_container_width=True)
                        st.code(format_metadata_display(ref_meta))

                    # Illumination analysis
                    src_sun = src_meta.get('sun_elevation', 0)
                    ref_sun = ref_meta.get('sun_elevation', 0)
                    sun_delta = abs(src_sun - ref_sun)
                    st.divider()
                    st.subheader("☀️ Illumination Analysis")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Source Sun Elevation", f"{src_sun:.2f}°",
                                  src_meta.get('illumination_difficulty', '').split('(')[0])
                    with c2:
                        st.metric("Reference Sun Elevation", f"{ref_sun:.2f}°",
                                  ref_meta.get('illumination_difficulty', '').split('(')[0])
                    with c3:
                        st.metric("Sun Angle Delta", f"{sun_delta:.2f}°",
                                  "Hard" if sun_delta > 20 else "Medium" if sun_delta > 5 else "Easy")

                    # Warn if sun angle delta is large
                    if sun_delta > 8:
                        st.warning(
                            f"⚠️ **Large sun angle difference ({sun_delta:.1f}°).** "
                            f"The same surface features look very different between these two images. "
                            f"Feature matching may find few inliers even though the geographic overlap is good. "
                            f"This is the core challenge of PS166 — illumination variation."
                        )

                    # Overlap analysis
                    st.divider()
                    st.subheader("🗺️ Overlap Analysis")
                    from src.overlap import check_overlap_and_warn
                    overlap_info, overlap_warning = check_overlap_and_warn(src_meta, ref_meta)
                    if overlap_warning:
                        st.warning(overlap_warning)
                    else:
                        diff_color = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}.get(
                            overlap_info.get('difficulty', 'medium'), "⚪")
                        st.success(f"{diff_color} {overlap_info.get('recommendation', '')} *(geographic overlap only — actual match quality depends on illumination similarity)*")
                    o1, o2, o3 = st.columns(3)
                    with o1:
                        st.metric("Overlap", f"{overlap_info.get('overlap_fraction', 0):.1%}")
                    with o2:
                        st.metric("Center Distance", f"{overlap_info.get('distance_km', 0):.1f} km")
                    with o3:
                        st.metric("Difficulty", overlap_info.get('difficulty', 'N/A').upper())

                else:
                    # ── OHRC vs LRO GeoTIFF mode ───────────────────
                    from src.lro_loader import load_geotiff_reference, format_lro_metadata_display

                    lro_meta = load_geotiff_reference(lro_tif_path, source_label="LRO NAC")
                    ref_patch = lro_meta['image']

                    st.session_state['ohrc_source']    = src_patch
                    st.session_state['ohrc_reference'] = ref_patch
                    st.session_state['ohrc_src_meta']  = src_meta
                    st.session_state['ohrc_ref_meta']  = lro_meta
                    st.session_state['cross_source']   = True
                    st.session_state['ohrc_ready']     = True

                    st.success("✅ OHRC source + LRO GeoTIFF reference loaded!")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Source — OHRC**")
                        st.image(src_patch,
                                 caption=f"OHRC source {src_patch.shape}",
                                 use_container_width=True)
                        st.code(format_metadata_display(src_meta))
                    with col2:
                        st.markdown("**Reference — LRO NAC GeoTIFF**")
                        # Preview: show centre crop if very large
                        preview = lro_meta['image']
                        if max(preview.shape) > 2048:
                            h, w = preview.shape
                            ch, cw = h // 2, w // 2
                            preview = preview[max(0, ch-512):ch+512,
                                              max(0, cw-512):cw+512]
                        st.image(preview,
                                 caption=f"LRO reference (preview) — full shape {lro_meta['shape']}",
                                 use_container_width=True)
                        st.code(format_lro_metadata_display(lro_meta))

                    # Overlap check for cross-source pair
                    st.divider()
                    st.subheader("🗺️ Overlap Analysis (OHRC vs LRO)")
                    from src.overlap import check_overlap_and_warn
                    overlap_info, overlap_warning = check_overlap_and_warn(src_meta, lro_meta)
                    if overlap_warning:
                        st.warning(overlap_warning)
                    else:
                        diff_color = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}.get(
                            overlap_info.get('difficulty', 'medium'), "⚪")
                        st.success(f"{diff_color} {overlap_info.get('recommendation', '')} *(geographic overlap only — actual match quality depends on illumination)*")
                    o1, o2, o3 = st.columns(3)
                    with o1:
                        st.metric("Overlap", f"{overlap_info.get('overlap_fraction', 0):.1%}")
                    with o2:
                        st.metric("Center Distance", f"{overlap_info.get('distance_km', 0):.1f} km")
                    with o3:
                        st.metric("Difficulty", overlap_info.get('difficulty', 'N/A').upper())

                    # Source resolution info
                    st.divider()
                    src_res = src_meta.get('pixel_resolution_m', 0.25)
                    ref_res = lro_meta.get('pixel_resolution_m', 1.0)
                    r1, r2, r3 = st.columns(3)
                    with r1:
                        st.metric("OHRC GSD", f"{src_res:.2f} m/px")
                    with r2:
                        st.metric("LRO GSD", f"{ref_res:.2f} m/px")
                    with r3:
                        st.metric("Scale ratio", f"{ref_res / src_res:.1f}×")

            except Exception as e:
                st.error(f"❌ Failed to load data: {str(e)}")

    # ── Register button (persists across rerenders) ────────────────
    if st.session_state.get('ohrc_ready'):
        cross = st.session_state.get('cross_source', False)

        # Geo-assist toggle — only shown for OHRC vs OHRC mode
        # (needs .csv from both zips; not available for LRO GeoTIFF)
        use_geo_assist = False
        if not cross:
            use_geo_assist = st.checkbox(
                "🛰️ Use metadata-assisted coarse pre-alignment (uses .csv geolocation from zip)",
                value=True,
                help="Reads the ground-coordinate .csv file from each zip to estimate a rough "
                     "pixel-to-pixel alignment before feature matching. Helps on difficult pairs "
                     "where the two images are offset by many hundreds of pixels."
            )
            if use_geo_assist:
                st.caption(
                    "The .csv maps pixel positions to lunar lat/lon (sampled every ~100 pixels). "
                    "Coarse homography is estimated from geo-correspondences, source is pre-warped, "
                    "then standard feature matching refines to sub-pixel accuracy. "
                    "Falls back to standard pipeline if geo-align fails."
                )

        btn_label = "🚀 Register OHRC vs LRO (cross-source)" if cross else (
            "🚀 Register with Geo-Assist" if use_geo_assist else "🚀 Register Real OHRC Data"
        )

        if st.button(btn_label, type="primary", use_container_width=True):
            with st.spinner("Running registration..."):
                try:
                    src_patch = st.session_state['ohrc_source']
                    ref_patch = st.session_state['ohrc_reference']
                    src_meta  = st.session_state['ohrc_src_meta']
                    ref_meta  = st.session_state['ohrc_ref_meta']
                    src_zip   = st.session_state.get('ohrc_src_zip', '')
                    ref_zip   = st.session_state.get('ohrc_ref_zip', '')

                    if cross:
                        from src.pipeline import register_cross_source
                        result = register_cross_source(
                            source_image=src_patch,
                            source_meta=src_meta,
                            reference_meta=ref_meta,
                            method=method,
                            max_size=max_size,
                        )
                        if 'cross_source_crop_original_shape' in result:
                            st.info(
                                f"📐 Reference crop: "
                                f"{result['cross_source_crop_original_shape']} → "
                                f"{result['cross_source_crop_final_shape']}  |  "
                                f"Target GSD: {result['cross_source_target_res_m']:.2f} m/px"
                            )
                    elif use_geo_assist and src_zip and ref_zip:
                        from src.pipeline import register_with_geo_assist
                        result = register_with_geo_assist(
                            source_image=src_patch,
                            source_zip_path=src_zip,
                            reference_image=ref_patch,
                            reference_zip_path=ref_zip,
                            method=method,
                            max_size=max_size,
                        )
                        # Show geo-assist diagnostics
                        geo_used = result.get('coarse_geo_prealignment_used', False)
                        n_pts    = result.get('coarse_geo_n_points', 0)
                        c_rmse   = result.get('coarse_geo_residual_rmse')
                        if geo_used:
                            st.info(
                                f"🛰️ Geo pre-alignment: **{n_pts}** geo-linked points found  |  "
                                f"Coarse residual: **{c_rmse:.1f} px**  |  "
                                f"Homography composed: {result.get('final_homography_composed', False)}"
                            )
                        else:
                            st.warning(
                                "⚠️ Geo pre-alignment failed — result is from standard pipeline fallback. "
                                "Check that both zip files contain .csv geolocation files."
                            )
                    else:
                        from src.pipeline import run_pipeline
                        result = run_pipeline(
                            src_patch, ref_patch,
                            method=method,
                            max_size=max_size,
                            src_sun_elevation=src_meta.get('sun_elevation'),
                            ref_sun_elevation=ref_meta.get('sun_elevation'),
                            src_metadata=src_meta,
                            ref_metadata=ref_meta,
                        )

                    st.session_state['result'] = result

                    if not result.get('success', True):
                        st.error(f"❌ {result.get('failure_reason', 'Registration failed')}")
                    else:
                        metrics    = result['metrics']
                        confidence = result.get('confidence', 'medium')
                        conf_color = {"high": "🟢", "medium": "🟡",
                                      "low": "🔴", "failed": "❌"}.get(confidence, "⚪")
                        st.success(
                            f"✅ Registration complete!  "
                            f"Algorithm: **{result['method']}**  |  "
                            f"Confidence: {conf_color} **{confidence.upper()}**"
                        )
                        if result.get('degenerate_fit'):
                            st.warning(f"⚠️ {result.get('reliability_reason', '')}")
                        c1, c2, c3, c4 = st.columns(4)
                        with c1:
                            st.metric("RMSE", f"{metrics['rmse']:.4f} px")
                        with c2:
                            st.metric("Inlier Count", metrics['inlier_count'])
                        with c3:
                            st.metric("Inlier Ratio", f"{metrics['inlier_ratio']:.2%}")
                        with c4:
                            st.metric("Spatial Score", f"{metrics['spatial_score']:.4f}")
                        st.info("Check Match Points, Registration Result, and Metrics tabs for full details")

                except Exception as e:
                    st.error(f"❌ Registration failed: {str(e)}")

    elif not inputs_ready:
        st.info("Enter a source OHRC zip path and a reference path above to begin.")
        st.markdown("**Example OHRC zip:**")
        st.code("/Users/kajol/Desktop/ps166/pradan.issdc.gov.in/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/ohr_collection/data/calibrated/20260102/ch2_ohr_ncp_20260102T1224107393_d_img_d18.zip")
        st.markdown("**Example LRO GeoTIFF:**")
        st.code("/Users/kajol/Desktop/ps166/lunar_registration/data/reference/lro_south_pole.tif")

# ── Tab 3: Match Points ────────────────────────────────────────────
with tab3:
    st.subheader("🔗 Match Point Visualization")
    st.markdown("Green lines = correct matches (inliers) | Red lines = wrong matches (outliers)")

    if 'result' in st.session_state:
        result = st.session_state['result']

        if not result.get('success', True) or 'match_visualization' not in result:
            st.warning(f"⚠️ Registration failed — no match visualization available.")
            if result.get('failure_reason'):
                st.error(result['failure_reason'])
        else:
            match_vis = result['match_visualization']
            st.image(match_vis, caption="Match visualization — green=correct, red=wrong", use_container_width=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Matches", result.get('total_matches_before_filter', 0))
            with col2:
                st.metric("After Distribution Filter", result.get('total_matches_after_distribution', 0))
            with col3:
                st.metric("Inliers (Correct)", result['metrics']['inlier_count'])

            st.download_button(
                "⬇️ Download Match Image",
                data=numpy_to_bytes(match_vis),
                file_name="match_visualization.png",
                mime="image/png"
            )
    else:
        st.info("Run the registration first (Tab 1 or Tab 2)")

# ── Tab 4: Registration Result ─────────────────────────────────────
with tab4:
    st.subheader("🎯 Registration Result")

    if 'result' in st.session_state:
        result = st.session_state['result']

        # Guard: if registration failed, no visualizations available
        if not result.get('success', True) or 'side_by_side' not in result:
            st.warning("⚠️ Registration failed — no result image available.")
            if result.get('failure_reason'):
                st.error(result['failure_reason'])
        else:
            view_mode = st.radio(
                "View Mode",
                ["Side by Side", "Checkerboard", "Difference Image"],
                horizontal=True
            )

            if view_mode == "Side by Side":
                st.image(result['side_by_side'],
                        caption="Left: Reference | Right: Registered Source",
                        use_container_width=True)

            elif view_mode == "Checkerboard":
                st.markdown("**Checkerboard view:** alternating tiles from Reference and Registered image.")
                st.markdown("If alignment is perfect, you won't see borders between tiles.")
                st.image(result['checkerboard'],
                        caption="Checkerboard blend — seamless = perfect alignment",
                        use_container_width=True)

            elif view_mode == "Difference Image":
                st.markdown("**Difference image:** bright = misaligned areas, dark = well-aligned areas")
                st.image(result['difference_image'],
                        caption="Difference image — darker is better",
                        use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "⬇️ Download Registered Image",
                    data=numpy_to_bytes(result['registered_image_refined']),
                    file_name="registered_image.png",
                    mime="image/png"
                )
            with col2:
                st.download_button(
                    "⬇️ Download Checkerboard",
                    data=numpy_to_bytes(result['checkerboard']),
                    file_name="checkerboard.png",
                    mime="image/png"
                )
    else:
        st.info("Run the registration first (Tab 1)")

# ── Tab 5: Metrics ─────────────────────────────────────────────────
with tab5:
    st.subheader("📊 Evaluation Metrics")

    if 'result' in st.session_state:
        result = st.session_state['result']

        # Guard for failed results
        if not result.get('success', True) or 'metrics' not in result:
            st.error(f"❌ Registration failed — no metrics available.")
            if result.get('failure_reason'):
                st.error(result['failure_reason'])
        else:
            metrics = result['metrics']

            # ── Reliability / degenerate-fit warning ──────────────────────
            if metrics.get('degenerate_fit'):
                st.warning(f"⚠️ **Result not statistically reliable:** {metrics['reliability_reason']}")
            elif metrics.get('confidence') == 'low' and metrics.get('reliability_reason'):
                st.info(f"ℹ️ **Low-confidence result:** {metrics['reliability_reason']}")

            # ── Confidence badge ───────────────────────────────────────────
            confidence = metrics.get('confidence', result.get('confidence', 'N/A'))
            conf_color = {"high": "success-badge", "medium": "fail-badge",
                          "low": "fail-badge", "failed": "fail-badge"}.get(confidence, "fail-badge")
            conf_icon  = {"high": "🟢", "medium": "🟡", "low": "🔴", "failed": "❌"}.get(confidence, "⚪")
            st.markdown(f'<span class="{conf_color}">{conf_icon} Confidence: {confidence.upper()}</span>',
                        unsafe_allow_html=True)

            # ── Sub-pixel status ───────────────────────────────────────────
            if metrics['rmse'] < 1.0 and not metrics.get('degenerate_fit'):
                st.markdown('<span class="success-badge">✅ Sub-pixel accuracy achieved (RMSE < 1.0)</span>', unsafe_allow_html=True)
            elif metrics.get('degenerate_fit'):
                st.markdown('<span class="fail-badge">⚠️ RMSE not meaningful (degenerate fit)</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="fail-badge">⚠️ Sub-pixel not achieved (RMSE ≥ 1.0)</span>', unsafe_allow_html=True)

            st.divider()

            # ── Metrics grid ───────────────────────────────────────────────
            col1, col2, col3 = st.columns(3)
            with col1:
                rmse_label = "⚠️ Not meaningful" if metrics.get('degenerate_fit') else (
                    "✅ Sub-pixel" if metrics['rmse'] < 1.0 else "❌ Not sub-pixel"
                )
                st.metric("RMSE (pixels)", f"{metrics['rmse']:.4f}", rmse_label)
                st.metric("RMSE-X", f"{metrics['rmse_x']:.4f}")
                st.metric("RMSE-Y", f"{metrics['rmse_y']:.4f}")
            with col2:
                st.metric("Inlier Count", metrics['inlier_count'])
                st.metric("Total Matches", metrics['total_matches'])
                st.metric("Inlier Ratio", f"{metrics['inlier_ratio']:.2%}")
            with col3:
                st.metric("Spatial Distribution Score", f"{metrics['spatial_score']:.4f}")
                st.metric("Processing Time", f"{metrics['processing_time']:.2f}s")
                st.metric("Algorithm Used", result.get('method', 'N/A'))

            st.divider()

            # ── Full report ────────────────────────────────────────────────
            st.subheader("Full Report")
            st.code(result.get('metrics_report', 'No report available'))

            st.download_button(
                "⬇️ Download Metrics Report",
                data=result.get('metrics_report', ''),
                file_name="metrics_report.txt",
                mime="text/plain"
            )

            with st.expander("📐 Transform Matrix (Transformation Parameters)"):
                st.markdown("This matrix describes how the source image was transformed to align with reference:")
                H = result.get('homography_matrix')
                ttype = metrics.get('transform_type', result.get('transform_type', 'unknown'))
                st.caption(f"Transform type: {ttype}")
                st.code(str(H) if H is not None else "Not available")

    else:
        st.info("Run the registration first (Tab 1)")

# ── Footer ─────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align:center; color:#666; font-size:0.8em;'>
PS166 | SIH 2026 | Chandrayaan-2 Lunar Image Registration | ISRO PRADAN Dataset
</div>
""", unsafe_allow_html=True)
