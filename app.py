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
        ["auto", "akaze", "sift"],
        help="Auto = tries AKAZE first (better for illumination), falls back to SIFT"
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
    st.markdown("🔵 **AKAZE** — Better for illumination variation")
    st.markdown("🟢 **SIFT** — Classic, reliable, scale-invariant")
    st.markdown("🟡 **Auto** — Picks best automatically")

    st.divider()
    st.markdown("**PS Requirements:**")
    st.markdown("✅ Sub-pixel accuracy (RMSE < 1.0)")
    st.markdown("✅ Uniform match distribution")
    st.markdown("✅ RMSE, Inlier count, Inlier ratio")
    st.markdown("✅ Match point visualization")
    st.markdown("✅ Registered output image")

# ── Main Content ───────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📤 Upload & Run",
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
                st.success(f"✅ Registration complete! Algorithm used: **{result['method']}**")

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
                st.error(f"❌ Registration failed: {str(e)}")
                st.markdown("**Possible reasons:**")
                st.markdown("- Images don't overlap (different lunar regions)")
                st.markdown("- Images too different in lighting")
                st.markdown("- Try a different algorithm in settings")

# ── Tab 2: Match Points ────────────────────────────────────────────
with tab2:
    st.subheader("🔗 Match Point Visualization")
    st.markdown("Green lines = correct matches (inliers) | Red lines = wrong matches (outliers)")

    if 'result' in st.session_state:
        result = st.session_state['result']
        match_vis = result['match_visualization']
        st.image(match_vis, caption="Match visualization — green=correct, red=wrong", use_container_width=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Matches", result['total_matches_before_filter'])
        with col2:
            st.metric("After Distribution Filter", result['total_matches_after_distribution'])
        with col3:
            st.metric("Inliers (Correct)", result['metrics']['inlier_count'])

        # Download match visualization
        st.download_button(
            "⬇️ Download Match Image",
            data=numpy_to_bytes(match_vis),
            file_name="match_visualization.png",
            mime="image/png"
        )
    else:
        st.info("Run the registration first (Tab 1)")

# ── Tab 3: Registration Result ─────────────────────────────────────
with tab3:
    st.subheader("🎯 Registration Result")

    if 'result' in st.session_state:
        result = st.session_state['result']

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

# ── Tab 4: Metrics ─────────────────────────────────────────────────
with tab4:
    st.subheader("📊 Evaluation Metrics")

    if 'result' in st.session_state:
        result = st.session_state['result']
        metrics = result['metrics']

        # Sub-pixel status
        if metrics['rmse'] < 1.0:
            st.markdown('<span class="success-badge">✅ Sub-pixel accuracy achieved (RMSE < 1.0)</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="fail-badge">⚠️ Sub-pixel not achieved (RMSE ≥ 1.0)</span>', unsafe_allow_html=True)

        st.divider()

        # Metrics grid
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("RMSE (pixels)", f"{metrics['rmse']:.4f}")
            st.metric("RMSE-X", f"{metrics['rmse_x']:.4f}")
            st.metric("RMSE-Y", f"{metrics['rmse_y']:.4f}")
        with col2:
            st.metric("Inlier Count", metrics['inlier_count'])
            st.metric("Total Matches", metrics['total_matches'])
            st.metric("Inlier Ratio", f"{metrics['inlier_ratio']:.2%}")
        with col3:
            st.metric("Spatial Distribution Score", f"{metrics['spatial_score']:.4f}")
            st.metric("Processing Time", f"{metrics['processing_time']:.2f}s")
            st.metric("Algorithm Used", result['method'])

        st.divider()

        # Full report
        st.subheader("Full Report")
        st.code(result['metrics_report'])

        # Download report
        st.download_button(
            "⬇️ Download Metrics Report",
            data=result['metrics_report'],
            file_name="metrics_report.txt",
            mime="text/plain"
        )

        # Homography matrix
        with st.expander("📐 Homography Matrix (Transformation Parameters)"):
            st.markdown("This 3×3 matrix describes exactly how the source image was transformed to align with reference:")
            st.code(str(result['homography_matrix']))

    else:
        st.info("Run the registration first (Tab 1)")

# ── Footer ─────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align:center; color:#666; font-size:0.8em;'>
PS166 | SIH 2026 | Chandrayaan-2 Lunar Image Registration | ISRO PRADAN Dataset
</div>
""", unsafe_allow_html=True)
