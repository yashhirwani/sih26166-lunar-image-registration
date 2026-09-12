"""
LunarAlign — Chandrayaan-2 Image Registration Workbench
PS166 · ISRO · SIH 2026

Scientific image correspondence and registration for OHRC, TMC-2, and IIRS
sensors against multi-modal reference datasets (OHRC, LRO NAC, SELENE).
"""

import streamlit as st
import numpy as np
import cv2
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from src.preprocess import load_image_from_bytes
from src.pipeline import run_pipeline
from src.visualize import numpy_to_bytes

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="LunarAlign · PS166 · ISRO",
    page_icon="⊕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design system CSS ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"],
[data-testid="stHeader"], .stApp, [data-testid="block-container"] {
    background-color: #f5f6f8 !important;
    color: #1a1a1a !important;
    font-family: 'IBM Plex Sans', 'Segoe UI', system-ui, sans-serif !important;
}
[data-testid="stHeader"] { background: #f5f6f8 !important; border-bottom: 1px solid #d8dce4 !important; }
[data-testid="stMain"] > div:first-child { padding-top: 0 !important; }
[data-testid="block-container"] { padding-top: 8px !important; }
.stMainBlockContainer { padding-top: 8px !important; }

[data-testid="stSidebar"] { background-color: #ffffff !important; border-right: 1px solid #d8dce4 !important; }
[data-testid="stSidebar"] * { color: #1a1a1a !important; }

.product-header {
    padding: 20px 0 10px 0;
    border-bottom: 2px solid #003f7f;
    margin-bottom: 20px;
}
.product-meta {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.64em; letter-spacing: 1.5px; color: #5a6271;
    text-transform: uppercase; margin-bottom: 4px;
}
.product-title {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 1.7em; font-weight: 700; color: #003f7f;
    letter-spacing: -0.3px; margin: 0;
}
.product-subtitle {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.71em; color: #3a4255; margin-top: 3px;
}

.sec-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.64em; font-weight: 600; letter-spacing: 1.4px;
    text-transform: uppercase; color: #003f7f;
    border-left: 3px solid #003f7f; padding-left: 7px;
    margin-bottom: 8px; margin-top: 4px;
}

.readout-strip {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72em; background: #1a1a2e; color: #a8d8a8;
    border: 1px solid #003f7f; padding: 8px 14px;
    letter-spacing: 0.6px; margin: 8px 0; border-radius: 2px;
}
.readout-strip b { color: #ffd580; }

.chip {
    display: inline-block; font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68em; font-weight: 600; letter-spacing: 0.8px;
    text-transform: uppercase; padding: 3px 10px 3px 8px;
    border: 1px solid; margin-right: 6px; border-radius: 2px;
}
.chip::before {
    display: inline-block; width: 6px; height: 6px; border-radius: 50%;
    margin-right: 6px; content: ''; vertical-align: middle; margin-bottom: 1px;
}
.chip-high    { background:#e6f4ea; border-color:#1a7f37; color:#1a7f37; }
.chip-high::before   { background:#1a7f37; }
.chip-medium  { background:#fff8e1; border-color:#bf8700; color:#bf8700; }
.chip-medium::before { background:#bf8700; }
.chip-low     { background:#fff0e0; border-color:#e06000; color:#e06000; }
.chip-low::before    { background:#e06000; }
.chip-failed  { background:#fce8e8; border-color:#cf222e; color:#cf222e; }
.chip-failed::before { background:#cf222e; }
.chip-subpx   { background:#e6f4ea; border-color:#1a7f37; color:#1a7f37; }
.chip-subpx::before  { background:#1a7f37; }
.chip-nosubpx { background:#fff0e0; border-color:#e06000; color:#e06000; }
.chip-nosubpx::before{ background:#e06000; }
.chip-degen   { background:#fce8e8; border-color:#cf222e; color:#cf222e; }
.chip-degen::before  { background:#cf222e; }
.chip-info    { background:#e8f0fe; border-color:#003f7f; color:#003f7f; }
.chip-info::before   { background:#003f7f; }

.instr-panel { background:#ffffff; border:1px solid #d0d4db; border-top:3px solid #003f7f; padding:14px 16px; margin-bottom:8px; }
.instr-panel-title { font-family:'IBM Plex Mono',monospace; font-size:0.62em; letter-spacing:1.2px; text-transform:uppercase; color:#5a6271; margin-bottom:10px; padding-bottom:6px; border-bottom:1px solid #e8eaed; }
.instr-row { display:flex; justify-content:space-between; align-items:baseline; padding:5px 0; border-bottom:1px solid #f0f2f5; font-size:0.83em; }
.instr-row:last-child { border-bottom:none; }
.instr-key { color:#5a6271; font-family:'IBM Plex Sans',sans-serif; font-size:0.92em; }
.instr-val { font-family:'IBM Plex Mono',monospace; font-weight:600; color:#1a1a1a; font-size:0.92em; }
.instr-val-good { color:#1a7f37; font-family:'IBM Plex Mono',monospace; font-weight:600; }
.instr-val-warn { color:#bf8700; font-family:'IBM Plex Mono',monospace; font-weight:600; }
.instr-val-bad  { color:#cf222e; font-family:'IBM Plex Mono',monospace; font-weight:600; }

.measurement { font-family:'IBM Plex Mono',monospace; font-size:2.2em; font-weight:600; color:#003f7f; line-height:1.1; letter-spacing:-1px; }
.measurement-label { font-family:'IBM Plex Mono',monospace; font-size:0.62em; letter-spacing:1.2px; text-transform:uppercase; color:#5a6271; margin-bottom:4px; }
.measurement-unit { font-size:0.45em; color:#5a6271; font-weight:400; }
.measurement-good { color:#1a7f37 !important; }
.measurement-warn { color:#bf8700 !important; }
.measurement-bad  { color:#cf222e !important; }

.algo-table { width:100%; border-collapse:collapse; font-size:0.76em; margin-bottom:12px; }
.algo-table th { font-family:'IBM Plex Mono',monospace; font-size:0.7em; letter-spacing:1px; text-transform:uppercase; color:#5a6271; padding:5px 8px; border-bottom:1px solid #d0d4db; text-align:left; }
.algo-table td { padding:5px 8px; border-bottom:1px solid #e8eaed; color:#1a1a1a; vertical-align:top; }
.algo-table td:first-child { font-family:'IBM Plex Mono',monospace; font-weight:600; color:#003f7f; white-space:nowrap; }
.algo-table tr:last-child td { border-bottom:none; }
.algo-table tr.active td { background:#e8f0fe; }

.req-row { display:flex; align-items:baseline; gap:8px; padding:4px 0; border-bottom:1px solid #f0f2f5; font-size:0.76em; }
.req-row:last-child { border-bottom:none; }
.req-check { font-family:'IBM Plex Mono',monospace; font-weight:700; color:#ffffff; background:#1a7f37; padding:1px 5px; font-size:0.8em; flex-shrink:0; }
.req-name { color:#1a1a1a; font-weight:500; }
.req-value { font-family:'IBM Plex Mono',monospace; color:#5a6271; font-size:0.9em; margin-left:auto; }

[data-testid="stButton"] > button[kind="primary"] {
    background:#003f7f !important; border:1px solid #003f7f !important;
    color:#ffffff !important; border-radius:2px !important;
    font-family:'IBM Plex Sans',sans-serif !important;
    font-weight:600 !important; font-size:0.85em !important;
}
[data-testid="stButton"] > button[kind="primary"]:hover { background:#00519e !important; }
[data-testid="stButton"] > button[kind="secondary"] {
    background:#ffffff !important; border:1px solid #d0d4db !important;
    color:#1a1a1a !important; border-radius:2px !important; font-size:0.82em !important;
}

[data-testid="stTabs"] [role="tablist"] { border-bottom:2px solid #d0d4db; gap:0; background:#ffffff; }
[data-testid="stTabs"] [role="tab"] { font-family:'IBM Plex Mono',monospace !important; font-size:0.72em !important; font-weight:600 !important; letter-spacing:0.8px !important; text-transform:uppercase !important; color:#5a6271 !important; padding:10px 18px !important; border-bottom:2px solid transparent !important; margin-bottom:-2px !important; border-radius:0 !important; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] { color:#003f7f !important; border-bottom-color:#003f7f !important; background:#f0f5ff !important; }

[data-testid="stTextInput"] input { font-family:'IBM Plex Mono',monospace !important; font-size:0.78em !important; background:#ffffff !important; border:1px solid #c0c6d0 !important; border-radius:2px !important; color:#1a1a1a !important; padding:8px 10px !important; }
[data-testid="stTextInput"] input:focus { border-color:#003f7f !important; box-shadow:0 0 0 2px rgba(0,63,127,0.15) !important; }

[data-testid="stSelectbox"] > div > div { background:#ffffff !important; border:1px solid #c0c6d0 !important; border-radius:2px !important; font-family:'IBM Plex Mono',monospace !important; font-size:0.82em !important; color:#1a1a1a !important; }

[data-testid="stMetric"] { background:#ffffff !important; border:1px solid #d0d4db !important; border-top:2px solid #003f7f !important; border-radius:0 !important; padding:12px 14px !important; }
[data-testid="stMetricLabel"] { font-family:'IBM Plex Mono',monospace !important; font-size:0.65em !important; letter-spacing:0.9px !important; text-transform:uppercase !important; color:#5a6271 !important; }
[data-testid="stMetricValue"] { font-family:'IBM Plex Mono',monospace !important; font-size:1.6em !important; font-weight:600 !important; color:#003f7f !important; letter-spacing:-0.5px !important; }

[data-testid="stAlert"] { border-radius:2px !important; font-size:0.83em !important; }
[data-testid="stCode"], .stCode > div { background:#1a1a2e !important; border:1px solid #003f7f !important; border-radius:2px !important; font-family:'IBM Plex Mono',monospace !important; font-size:0.76em !important; color:#a8d8a8 !important; }
[data-testid="stExpander"] { border:1px solid #d0d4db !important; border-radius:0 !important; background:#ffffff !important; }
hr { border-color:#d0d4db !important; margin:10px 0 !important; }

[data-testid="stFileDropzone"] { background:#fafbfd !important; border:1px dashed #b0b8c4 !important; border-radius:2px !important; }
[data-testid="stFileDropzone"]:hover { border-color:#003f7f !important; background:#f0f5ff !important; }

/* Theme toggle button - top right */
.theme-toggle-wrap {
    position: fixed;
    top: 10px;
    right: 60px;
    z-index: 999999;
}
.theme-toggle-btn {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72em;
    font-weight: 600;
    background: #ffffff;
    border: 1px solid #d0d4db;
    color: #003f7f;
    padding: 5px 12px;
    cursor: pointer;
    border-radius: 2px;
    letter-spacing: 0.5px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}
.theme-toggle-btn:hover { background: #003f7f; color: #ffffff; border-color: #003f7f; }

.workbench-footer { margin-top:32px; padding:12px 0; border-top:1px solid #d0d4db; font-family:'IBM Plex Mono',monospace; font-size:0.65em; color:#9aa0aa; letter-spacing:0.6px; text-align:center; }
::-webkit-scrollbar { width:5px; height:5px; }
::-webkit-scrollbar-track { background:#f5f6f8; }
::-webkit-scrollbar-thumb { background:#c0c6d0; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ────────────────────────────────────────────────────────
def chip(label: str, kind: str = "info") -> str:
    return f'<span class="chip chip-{kind}">{label}</span>'

def confidence_chip(conf: str) -> str:
    labels = {
        "high":   "HIGH CONFIDENCE",
        "medium": "MEDIUM CONFIDENCE",
        "low":    "LOW CONFIDENCE",
        "failed": "FAILED",
    }
    return chip(labels.get(conf, conf.upper()), conf)

def subpixel_chip(rmse: float, degen: bool) -> str:
    if degen:    return chip("RMSE UNRELIABLE · DEGENERATE FIT", "degen")
    if rmse < 1: return chip(f"SUB-PIXEL · RMSE {rmse:.4f} px", "subpx")
    return chip(f"NOT SUB-PIXEL · RMSE {rmse:.4f} px", "nosubpx")

def irow(key, val, cls="instr-val"):
    return (f'<div class="instr-row">'
            f'<span class="instr-key">{key}</span>'
            f'<span class="{cls}">{val}</span></div>')

def readout(fields: dict) -> str:
    parts = []
    for k, v in fields.items():
        parts.append(f"<b>{k}</b> {v}")
    return (f'<div class="readout-strip">{" · ".join(parts)}</div>')

def sec(label: str) -> str:
    return f'<div class="sec-label">{label}</div>'


# ── Theme toggle logic ─────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = False

# Apply dark theme override if toggled
if st.session_state["dark_mode"]:
    st.markdown("""
    <style>
    html, body,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stHeader"],
    .stApp,
    [data-testid="block-container"] {
        background-color: #0D1117 !important;
        color: #C9D1D9 !important;
    }
    [data-testid="stSidebar"] {
        background-color: #161B22 !important;
        border-right: 1px solid #21262D !important;
    }
    [data-testid="stSidebar"] * { color: #C9D1D9 !important; }
    .status-strip { background: #161B22 !important; border-color: #21262D !important; color: #8B949E !important; }
    .product-title { color: #58A6FF !important; }
    .product-meta, .product-subtitle { color: #8B949E !important; }
    .product-header { border-color: #1F6FEB !important; }
    .sec-label { color: #58A6FF !important; border-color: #1F6FEB !important; }
    .instr-panel { background: #161B22 !important; border-color: #21262D !important; }
    .instr-panel-title { color: #8B949E !important; }
    .instr-row { border-color: #21262D !important; }
    .instr-key { color: #8B949E !important; }
    .instr-val { color: #E6EDF3 !important; }
    [data-testid="stMetric"] { background: #161B22 !important; border-color: #21262D !important; border-top-color: #1F6FEB !important; }
    [data-testid="stMetricValue"] { color: #58A6FF !important; }
    [data-testid="stMetricLabel"] { color: #8B949E !important; }
    [data-testid="stButton"] > button[kind="primary"] { background: #1F6FEB !important; border-color: #1F6FEB !important; }
    [data-testid="stTabs"] [role="tablist"] { background: #161B22 !important; border-color: #21262D !important; }
    [data-testid="stTabs"] [role="tab"] { color: #8B949E !important; }
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] { color: #E6EDF3 !important; border-color: #1F6FEB !important; background: #0D1117 !important; }
    .readout-strip { background: #0D1117 !important; border-color: #1F6FEB !important; }
    .algo-table th { color: #8B949E !important; border-color: #21262D !important; }
    .algo-table td { border-color: #21262D !important; color: #C9D1D9 !important; }
    .algo-table td:first-child { color: #58A6FF !important; }
    .measurement-good { color: #3FB950 !important; }
    .measurement-warn { color: #D29922 !important; }
    .measurement-bad  { color: #F85149 !important; }
    [data-testid="stTextInput"] input { background: #0D1117 !important; border-color: #30363D !important; color: #C9D1D9 !important; }
    [data-testid="stSelectbox"] > div > div { background: #161B22 !important; border-color: #30363D !important; color: #C9D1D9 !important; }
    [data-testid="stFileDropzone"] { background: #161B22 !important; border-color: #30363D !important; }
    [data-testid="stFileDropzone"] * { color: #8B949E !important; }
    .workbench-footer { color: #484F58 !important; border-color: #21262D !important; }
    hr { border-color: #21262D !important; }
    </style>
    """, unsafe_allow_html=True)

# ── Status strip ──────────────────────────────────────────────────
# ── Status strip + header + theme toggle ──────────────────────────
st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
has_result = "result" in st.session_state and st.session_state["result"].get("success", False)
result_conf = st.session_state.get("result", {}).get("confidence", "") if has_result else ""

# Status strip
st.markdown(
    f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.67em;'
    f'letter-spacing:0.7px;color:#5a6271;background:#eceef2;'
    f'border-bottom:1px solid #d0d4db;padding:5px 8px;margin-bottom:0;">'
    f'● SYSTEM: READY'
    + (f' &nbsp;·&nbsp; ● LAST: {result_conf.upper()}' if has_result else '')
    + f'&nbsp;&nbsp;·&nbsp;&nbsp;ALGORITHM: {st.session_state.get("_method","AUTO").upper()}'
    + f'&nbsp;&nbsp;·&nbsp;&nbsp;RES: {st.session_state.get("_max_size",1024)} px'
    + f'</div>',
    unsafe_allow_html=True,
)

# Header — full width, no columns needed
toggle_icon = "☀️" if st.session_state["dark_mode"] else "🌙"
st.markdown(f"""
<div class="product-header" style="display:flex;justify-content:space-between;align-items:center;">
  <div>
    <div class="product-meta">PS166 · ISRO · SIH 2026 · CHANDRAYAAN-2</div>
    <div class="product-title">LunarAlign</div>
    <div class="product-subtitle">
      IMAGE REGISTRATION &amp; CORRESPONDENCE WORKBENCH &nbsp;·&nbsp;
      OHRC · TMC-2 · IIRS · LRO NAC · SELENE
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# Theme toggle — placed in sidebar top for clean layout
# (Streamlit columns can't reliably place buttons top-right of custom HTML blocks)
st.markdown("<br>", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.62em;'
        'letter-spacing:1.4px;text-transform:uppercase;color:#003f7f;'
        'font-weight:700;padding:12px 0 8px 0;border-bottom:1px solid #d0d4db;'
        'margin-bottom:12px">Parameters</div>',
        unsafe_allow_html=True,
    )

    # Single theme toggle at top of sidebar
    _tlabel = "☀️  Light Mode" if st.session_state["dark_mode"] else "🌙  Dark Mode"
    if st.button(_tlabel, key="theme_toggle", use_container_width=True,
                 help="Toggle between light and dark mission-control themes"):
        st.session_state["dark_mode"] = not st.session_state["dark_mode"]
        st.rerun()
    st.markdown('<div style="margin-bottom:8px"></div>', unsafe_allow_html=True)
    method = st.selectbox(
        "Matching algorithm",
        ["auto", "lightglue", "pc-sift", "sift", "loftr", "akaze"],
        index=0,
        help=(
            "auto: Runs all tiers in parallel, selects highest-scoring result.\n"
            "lightglue: DISK keypoints + LightGlue Transformer matcher.\n"
            "pc-sift: Phase congruency maps → SIFT (illumination-invariant).\n"
            "sift: Scale-space feature matching, fast.\n"
            "loftr: Detector-free dense Transformer matching.\n"
            "akaze: Non-linear scale-space, baseline classical."
        ),
    )
    st.session_state["_method"] = method

    max_size = st.slider(
        "Max processing resolution (px)",
        min_value=256, max_value=2048, value=1024, step=128,
        help="Image longest side resampled to this before matching.",
    )
    st.session_state["_max_size"] = max_size

    st.markdown(
        '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.62em;'
        'letter-spacing:1.4px;text-transform:uppercase;color:#003f7f;'
        'font-weight:700;padding:12px 0 6px 0;border-bottom:1px solid #d0d4db;'
        'margin:10px 0 8px 0">Algorithm Reference</div>',
        unsafe_allow_html=True,
    )
    algo_rows = [
        ("auto",      "Multi-tier confidence router. Runs all in parallel."),
        ("lightglue", "DISK + LightGlue Transformer. Best overall accuracy."),
        ("pc-sift",   "Phase congruency → SIFT. Illumination-invariant."),
        ("sift",      "Scale-Invariant Feature Transform. Fast, reliable."),
        ("loftr",     "Detector-free dense Transformer. Low-texture cases."),
        ("akaze",     "Non-linear scale-space. Classical baseline."),
    ]
    rows_html = "".join(
        f'<tr{"  class=\"active\"" if a == method else ""}>'
        f'<td>{a}</td><td style="color:#444;font-family:sans-serif">{d}</td></tr>'
        for a, d in algo_rows
    )
    st.markdown(
        f'<table class="algo-table"><thead><tr>'
        f'<th>NAME</th><th>DESCRIPTION</th>'
        f'</tr></thead><tbody>{rows_html}</tbody></table>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.62em;'
        'letter-spacing:1.4px;text-transform:uppercase;color:#003f7f;'
        'font-weight:700;padding:12px 0 6px 0;border-bottom:1px solid #d0d4db;'
        'margin:10px 0 8px 0">PS166 Compliance</div>',
        unsafe_allow_html=True,
    )
    reqs = [
        ("Sub-pixel RMSE",       "RMSE < 1.0 px"),
        ("Uniform distribution", "8×8 grid enforcement"),
        ("RMSE metric",          "Computed per inlier"),
        ("Inlier count + ratio", "Post-RANSAC stats"),
        ("Match visualisation",  "Inlier/outlier overlay"),
        ("Registered product",   "Warped + downloadable"),
        ("Illumination-aware",   "Sun-angle adaptive"),
        ("Multi-modal support",  "OHRC·TMC2·IIRS·LRO"),
    ]
    req_html = "".join(
        f'<div class="req-row">'
        f'<span class="req-check">✓</span>'
        f'<span class="req-name">{n}</span>'
        f'<span class="req-value">{v}</span>'
        f'</div>'
        for n, v in reqs
    )
    st.markdown(req_html, unsafe_allow_html=True)


# ── Tabs ───────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "INPUT & REGISTRATION",
    "OHRC / LRO NAC / IIRS",
    "CORRESPONDENCE ANALYSIS",
    "REGISTRATION OUTPUT",
    "QUANTITATIVE REPORT",
])


# ══════════════════════════════════════════════════════════════════
# TAB 1 — INPUT & REGISTRATION
# ══════════════════════════════════════════════════════════════════
with tab1:
    st.markdown(sec("Image Input"), unsafe_allow_html=True)
    st.caption(
        "Upload any two single-channel lunar images (PNG / JPG / TIFF). "
        "For OHRC zip files, LRO NAC GeoTIFFs or IIRS cubes use the "
        "OHRC / LRO NAC / IIRS tab."
    )

    c_src, c_ref = st.columns(2, gap="medium")

    with c_src:
        st.markdown('<div class="sec-label">SRC · SOURCE IMAGE · MOVING</div>',
                    unsafe_allow_html=True)
        source_file = st.file_uploader(
            "src", type=["png","jpg","jpeg","tif","tiff"],
            key="source", label_visibility="collapsed",
        )
        if source_file:
            src_bytes  = source_file.read()
            source_img = load_image_from_bytes(src_bytes)
            st.image(source_img, use_container_width=True)
            st.markdown(
                readout({"FILE": source_file.name,
                         "SIZE": f"{source_img.shape[1]}×{source_img.shape[0]} px"}),
                unsafe_allow_html=True,
            )

    with c_ref:
        st.markdown('<div class="sec-label">REF · REFERENCE IMAGE · FIXED</div>',
                    unsafe_allow_html=True)
        reference_file = st.file_uploader(
            "ref", type=["png","jpg","jpeg","tif","tiff"],
            key="reference", label_visibility="collapsed",
        )
        if reference_file:
            ref_bytes     = reference_file.read()
            reference_img = load_image_from_bytes(ref_bytes)
            st.image(reference_img, use_container_width=True)
            st.markdown(
                readout({"FILE": reference_file.name,
                         "SIZE": f"{reference_img.shape[1]}×{reference_img.shape[0]} px"}),
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    run_btn = st.button(
        "▶  EXECUTE REGISTRATION",
        type="primary",
        use_container_width=True,
        disabled=not (source_file and reference_file),
    )
    if not (source_file and reference_file):
        st.markdown(
            '<div style="text-align:center;font-family:\'IBM Plex Mono\',monospace;'
            'font-size:0.68em;color:#9aa0aa;letter-spacing:0.6px;padding:6px 0">'
            'LOAD BOTH IMAGES TO ENABLE REGISTRATION</div>',
            unsafe_allow_html=True,
        )

    if run_btn and source_file and reference_file:
        with st.spinner("Running registration pipeline…"):
            result = run_pipeline(
                source_img, reference_img,
                method=method, max_size=max_size,
            )
        st.session_state["result"] = result

        if not result.get("success", True):
            st.error(f"REGISTRATION FAILED: {result.get('failure_reason','unknown error')}")
            with st.expander("Diagnostic details"):
                st.write(result.get("failure_reason",""))
        else:
            m    = result["metrics"]
            conf = result.get("confidence", "medium")
            degen = result.get("degenerate_fit", False)

            st.markdown(
                confidence_chip(conf) + "&nbsp;" +
                subpixel_chip(m["rmse"], degen) +
                (chip("DEGENERATE FIT", "degen") if degen else ""),
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)

            rc1, rc2, rc3, rc4 = st.columns(4)
            rc1.metric("RMSE",
                       f"{m['rmse']:.4f} px",
                       "sub-pixel ✓" if m["rmse"] < 1.0 and not degen else "≥ 1.0 px")
            rc2.metric("INLIER COUNT", m["inlier_count"])
            rc3.metric("INLIER RATIO", f"{m['inlier_ratio']:.1%}")
            rc4.metric("SPATIAL SCORE", f"{m['spatial_score']:.4f}")

            st.markdown(
                readout({"ALGORITHM": result["method"],
                         "TRANSFORM": m.get("transform_type","homography").upper(),
                         "TIME": f"{m.get('processing_time',0):.2f}s"}),
                unsafe_allow_html=True,
            )
            if result.get("escalation_log"):
                st.caption("Routing: " + " → ".join(result["escalation_log"]))
            if degen and result.get("reliability_reason"):
                st.warning(result["reliability_reason"])
            st.info("→ Results available in CORRESPONDENCE ANALYSIS · REGISTRATION OUTPUT · QUANTITATIVE REPORT")


# ══════════════════════════════════════════════════════════════════
# TAB 2 — OHRC / LRO NAC / IIRS
# ══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(sec("Sensor Mode"), unsafe_allow_html=True)

    ref_source_mode = st.radio(
        "mode",
        ["OHRC vs OHRC", "OHRC vs LRO NAC (GeoTIFF)", "IIRS Hyperspectral"],
        horizontal=True,
        label_visibility="collapsed",
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # ── IIRS ──────────────────────────────────────────────────────
    if ref_source_mode == "IIRS Hyperspectral":
        _app_dir = os.path.dirname(os.path.abspath(__file__))
        _def_hdr = os.path.join(_app_dir,"data","iirs",
            "ch2_iir_ndi_20250729T0936115604_d_rfl_d18_srd.hdr")
        _def_qub = os.path.join(_app_dir,"data","iirs",
            "ch2_iir_ndi_20250729T0936115604_d_rfl_d18_srd.qub")

        st.markdown(sec("IIRS Input — ENVI format (.hdr + .qub)"), unsafe_allow_html=True)
        st.caption(
            "Chandrayaan-2 IIRS L1 reflectance product: 256 spectral bands, 712–5009 nm. "
            "Pipeline synthesises panchromatic-equivalent band (712–950 nm) before matching."
        )
        st.markdown(
            readout({"FORMAT":"ENVI BSQ/BIL/BIP","BANDS":"256",
                     "RANGE":"712–5009 nm","GSD":"~80 m/px"}),
            unsafe_allow_html=True,
        )

        ii1, ii2 = st.columns(2, gap="medium")
        with ii1:
            st.markdown(sec("ENVI Header (.hdr)"), unsafe_allow_html=True)
            iirs_label_path = st.text_input("hdr",value=_def_hdr,
                label_visibility="collapsed",key="iirs_label_path")
            iirs_hdr_upload = st.file_uploader("or upload .hdr",type=["hdr"],
                key="iirs_hdr_upload",label_visibility="collapsed")
            st.markdown(sec("Raw Cube (.qub)"), unsafe_allow_html=True)
            iirs_cube_path = st.text_input("qub",value=_def_qub,
                label_visibility="collapsed",key="iirs_cube_path")
        with ii2:
            st.markdown(sec("Reference Image (for registration)"), unsafe_allow_html=True)
            iirs_ref_path = st.text_input("ref",
                placeholder="OHRC .zip or LRO .tif",
                label_visibility="collapsed",key="iirs_ref_path")
            iirs_ref_type = st.selectbox("Reference type",
                ["OHRC zip","LRO GeoTIFF"],
                key="iirs_ref_type",label_visibility="collapsed")

        if iirs_hdr_upload:
            hdr_tmp = os.path.join(_app_dir,"data","iirs","_uploaded.hdr")
            os.makedirs(os.path.dirname(hdr_tmp),exist_ok=True)
            open(hdr_tmp,"wb").write(iirs_hdr_upload.getvalue())
            iirs_label_path = hdr_tmp

        if st.button("▶  LOAD & PREVIEW IIRS CUBE",type="primary",
                     disabled=not(iirs_cube_path and iirs_label_path),
                     key="iirs_load_btn"):
            with st.spinner("Loading IIRS cube (~3 GB)…"):
                try:
                    from src.iirs_loader import (parse_iirs_label,load_iirs_cube,
                        synthesize_panchromatic,make_false_color_composite,
                        format_iirs_metadata_display)
                    imeta = parse_iirs_label(iirs_label_path)
                    icube = load_iirs_cube(iirs_cube_path,imeta)
                    ipan  = synthesize_panchromatic(icube,imeta["wavelengths_nm"])
                    try:    ifc = make_false_color_composite(icube,imeta["wavelengths_nm"])
                    except: ifc = None
                    st.session_state.update({
                        "iirs_cube":icube,"iirs_meta":imeta,
                        "iirs_pan":ipan,"iirs_fc":ifc,"iirs_ready":True,
                    })
                    wls = imeta["wavelengths_nm"]
                    st.markdown(
                        readout({"SHAPE":str(icube.shape),
                                 "DTYPE":str(icube.dtype),
                                 "WL_RANGE":f"{wls[0]:.1f}–{wls[-1]:.1f} nm",
                                 "BANDS":str(imeta["num_bands"])}),
                        unsafe_allow_html=True,
                    )
                    st.success(f"IIRS cube loaded · {icube.shape[0]} bands · "
                               f"{icube.shape[2]}×{icube.shape[1]} px")
                except Exception as e:
                    st.error(f"Load failed: {e}")

        if st.session_state.get("iirs_ready"):
            from src.iirs_loader import format_iirs_metadata_display
            st.code(format_iirs_metadata_display(st.session_state["iirs_meta"]))
            ipan = st.session_state.get("iirs_pan")
            ifc  = st.session_state.get("iirs_fc")
            ip1, ip2 = st.columns(2,gap="medium")
            with ip1:
                st.markdown(sec("Synthesised Panchromatic (712–950 nm avg)"),
                            unsafe_allow_html=True)
                if ipan is not None:
                    d = ipan
                    if d.shape[0] > 1024:
                        sc = 1024/d.shape[0]
                        d = cv2.resize(d,(max(1,int(d.shape[1]*sc)),1024),cv2.INTER_AREA)
                    st.image(d,use_container_width=True,
                             caption=f"Full: {ipan.shape} · equal-weighted band avg")
            with ip2:
                st.markdown(sec("False-Colour Composite (SWIR)"),
                            unsafe_allow_html=True)
                if ifc is not None:
                    d = ifc
                    if d.shape[0] > 1024:
                        sc = 1024/d.shape[0]
                        d = cv2.resize(d,(max(1,int(d.shape[1]*sc)),1024),cv2.INTER_AREA)
                    st.image(d,use_container_width=True,
                             caption="R=2200nm · G=1600nm · B=1000nm")

            if st.button("▶  REGISTER IIRS vs REFERENCE",type="primary",
                         disabled=not iirs_ref_path,key="iirs_register_btn"):
                with st.spinner("Running IIRS registration…"):
                    try:
                        from src.pipeline import register_iirs
                        from src.ohrc_loader import extract_browse_png_from_zip
                        from src.lro_loader import load_geotiff_reference
                        icube = st.session_state["iirs_cube"]
                        imeta = st.session_state["iirs_meta"]
                        if iirs_ref_type == "OHRC zip":
                            ri,rm = extract_browse_png_from_zip(iirs_ref_path)
                        else:
                            rf = load_geotiff_reference(iirs_ref_path)
                            ri,rm = rf["image"],rf
                        res = register_iirs(icube,imeta,ri,rm,
                                            method=method,max_size=max_size,
                                            band_range_nm=(712,950))
                        st.session_state["result"] = res
                        if not res.get("success",True):
                            st.error(res.get("failure_reason","Registration failed"))
                        else:
                            m = res["metrics"]
                            c = res.get("confidence","medium")
                            st.markdown(confidence_chip(c)+"&nbsp;"+
                                        subpixel_chip(m["rmse"],res.get("degenerate_fit",False)),
                                        unsafe_allow_html=True)
                            cc1,cc2,cc3,cc4 = st.columns(4)
                            cc1.metric("RMSE",f"{m['rmse']:.4f} px")
                            cc2.metric("INLIERS",m["inlier_count"])
                            cc3.metric("INLIER RATIO",f"{m['inlier_ratio']:.1%}")
                            cc4.metric("SPATIAL SCORE",f"{m['spatial_score']:.4f}")
                    except Exception as e:
                        st.error(f"IIRS registration failed: {e}")

    # ── OHRC / LRO ────────────────────────────────────────────────
    else:
        zc1, zc2 = st.columns(2,gap="medium")
        with zc1:
            st.markdown(sec("Source — OHRC zip"), unsafe_allow_html=True)
            src_zip = st.text_input("src_zip",label_visibility="collapsed",
                placeholder="…/ch2_ohr_ncp_YYYYMMDDTHHMMSS_d_img_dNN.zip")
        with zc2:
            if ref_source_mode == "OHRC vs OHRC":
                st.markdown(sec("Reference — OHRC zip"), unsafe_allow_html=True)
                ref_zip      = st.text_input("ref_zip",label_visibility="collapsed",
                    placeholder="…/ch2_ohr_ncp_YYYYMMDDTHHMMSS_d_img_dNN.zip")
                lro_tif_path = None
            else:
                st.markdown(sec("Reference — LRO NAC GeoTIFF"), unsafe_allow_html=True)
                lro_tif_path = st.text_input("lro_path",label_visibility="collapsed",
                    placeholder="…/nac_roi_*.tif")
                ref_zip = None
                st.caption(
                    "Source: quickmap.lroc.asu.edu · Geographic and polar-stereographic CRS both supported"
                )

        inputs_ready = bool(src_zip and (ref_zip or lro_tif_path))
        if st.button("▶  LOAD DATA",type="primary",disabled=not inputs_ready):
            with st.spinner("Loading…"):
                try:
                    from src.ohrc_loader import extract_browse_png_from_zip,format_metadata_display
                    sp,sm = extract_browse_png_from_zip(src_zip)

                    if ref_source_mode == "OHRC vs OHRC":
                        rp,rm = extract_browse_png_from_zip(ref_zip)
                        st.session_state.update({
                            "ohrc_source":sp,"ohrc_reference":rp,
                            "ohrc_src_meta":sm,"ohrc_ref_meta":rm,
                            "ohrc_src_zip":src_zip,"ohrc_ref_zip":ref_zip,
                            "cross_source":False,"ohrc_ready":True,
                        })
                        lc1,lc2 = st.columns(2,gap="medium")
                        with lc1:
                            st.markdown(sec("SRC · OHRC"), unsafe_allow_html=True)
                            st.image(sp,use_container_width=True)
                            st.markdown(
                                readout({
                                    "SIZE":f"{sp.shape[1]}×{sp.shape[0]} px",
                                    "SUN_EL":f"{sm.get('sun_elevation',0):.2f}°",
                                    "ORBIT":str(sm.get('orbit_number','?')),
                                    "RES":f"{sm.get('pixel_resolution_m',0.25):.3f} m/px",
                                }),
                                unsafe_allow_html=True,
                            )
                            st.code(format_metadata_display(sm))
                        with lc2:
                            st.markdown(sec("REF · OHRC"), unsafe_allow_html=True)
                            st.image(rp,use_container_width=True)
                            st.markdown(
                                readout({
                                    "SIZE":f"{rp.shape[1]}×{rp.shape[0]} px",
                                    "SUN_EL":f"{rm.get('sun_elevation',0):.2f}°",
                                    "ORBIT":str(rm.get('orbit_number','?')),
                                    "RES":f"{rm.get('pixel_resolution_m',0.25):.3f} m/px",
                                }),
                                unsafe_allow_html=True,
                            )
                            st.code(format_metadata_display(rm))

                        src_sun   = sm.get("sun_elevation",0)
                        ref_sun   = rm.get("sun_elevation",0)
                        sun_delta = abs(src_sun-ref_sun)

                        st.markdown(sec("Illumination & Overlap Analysis"),
                                    unsafe_allow_html=True)
                        st.markdown(
                            readout({
                                "SRC_SUN_EL":f"{src_sun:.2f}°",
                                "REF_SUN_EL":f"{ref_sun:.2f}°",
                                "DELTA":f"{sun_delta:.2f}°",
                                "DIFFICULTY": ("HIGH" if sun_delta>8 else
                                               "MEDIUM" if sun_delta>3 else "LOW"),
                            }),
                            unsafe_allow_html=True,
                        )
                        if sun_delta > 8:
                            st.warning(
                                f"Sun-angle delta {sun_delta:.1f}° is high. "
                                "Surface features appear significantly different between acquisitions. "
                                "Recommended algorithm: pc-sift (illumination-invariant)."
                            )
                        from src.overlap import check_overlap_and_warn
                        oi,ow = check_overlap_and_warn(sm,rm)
                        if ow: st.warning(ow)
                        else:
                            oa1,oa2,oa3 = st.columns(3)
                            oa1.metric("OVERLAP",f"{oi.get('overlap_fraction',0):.1%}")
                            oa2.metric("CENTRE DIST",f"{oi.get('distance_km',0):.1f} km")
                            oa3.metric("DIFFICULTY",oi.get("difficulty","?").upper())
                            st.caption(oi.get("recommendation",""))

                    else:
                        from src.lro_loader import load_geotiff_reference,format_lro_metadata_display
                        lm = load_geotiff_reference(lro_tif_path,source_label="LRO NAC")
                        st.session_state.update({
                            "ohrc_source":sp,"ohrc_reference":lm["image"],
                            "ohrc_src_meta":sm,"ohrc_ref_meta":lm,
                            "cross_source":True,"ohrc_ready":True,
                        })
                        lc1,lc2 = st.columns(2,gap="medium")
                        with lc1:
                            st.markdown(sec("SRC · OHRC"), unsafe_allow_html=True)
                            st.image(sp,use_container_width=True)
                            st.markdown(
                                readout({"SIZE":f"{sp.shape[1]}×{sp.shape[0]} px",
                                         "SUN_EL":f"{sm.get('sun_elevation',0):.2f}°",
                                         "RES":f"{sm.get('pixel_resolution_m',0.25):.3f} m/px"}),
                                unsafe_allow_html=True,
                            )
                            st.code(format_metadata_display(sm))
                        with lc2:
                            st.markdown(sec("REF · LRO NAC"), unsafe_allow_html=True)
                            pv = lm["image"]
                            if max(pv.shape)>1024:
                                h,w=pv.shape; c=h//2; cw=w//2
                                pv=pv[max(0,c-512):c+512,max(0,cw-512):cw+512]
                            st.image(pv,use_container_width=True,
                                     caption=f"Preview · Full: {lm['shape']}")
                            st.markdown(
                                readout({"SHAPE":str(lm['shape']),
                                         "RES":f"{lm.get('pixel_resolution_m',1):.3f} m/px",
                                         "CRS":("PROJECTED" if lm.get("is_projected") else "GEOGRAPHIC")}),
                                unsafe_allow_html=True,
                            )
                            st.code(format_lro_metadata_display(lm))

                        from src.overlap import check_overlap_and_warn
                        oi,ow = check_overlap_and_warn(sm,lm)
                        st.markdown(sec("Geographic Overlap · OHRC vs LRO"),
                                    unsafe_allow_html=True)
                        if ow: st.warning(ow)
                        else:
                            oa1,oa2,oa3 = st.columns(3)
                            oa1.metric("OVERLAP",f"{oi.get('overlap_fraction',0):.1%}")
                            oa2.metric("CENTRE DIST",f"{oi.get('distance_km',0):.1f} km")
                            oa3.metric("DIFFICULTY",oi.get("difficulty","?").upper())

                        src_r = sm.get("pixel_resolution_m",0.25)
                        ref_r = lm.get("pixel_resolution_m",1.0)
                        st.markdown(
                            readout({"OHRC_GSD":f"{src_r:.3f} m/px",
                                     "LRO_GSD":f"{ref_r:.3f} m/px",
                                     "SCALE_RATIO":f"{ref_r/src_r:.1f}×"}),
                            unsafe_allow_html=True,
                        )

                except Exception as e:
                    st.error(f"Load failed: {e}")

        # ── Register ───────────────────────────────────────────────
        if st.session_state.get("ohrc_ready"):
            st.markdown("<br>", unsafe_allow_html=True)
            cross = st.session_state.get("cross_source",False)
            use_geo = False
            if not cross:
                use_geo = st.checkbox(
                    "Enable geo-assisted coarse pre-alignment (recommended for offset pairs)",
                    value=True,
                    help="Uses .csv pixel↔lat/lon tables from each zip for coarse "
                         "homography before feature matching. Significantly reduces "
                         "search space for large-offset acquisitions.",
                )
            btn_lbl = (
                "▶  REGISTER OHRC vs LRO NAC" if cross
                else "▶  REGISTER WITH GEO PRE-ALIGNMENT" if use_geo
                else "▶  REGISTER"
            )
            if st.button(btn_lbl,type="primary",use_container_width=True):
                with st.spinner("Running registration…"):
                    try:
                        sp  = st.session_state["ohrc_source"]
                        rp  = st.session_state["ohrc_reference"]
                        sm  = st.session_state["ohrc_src_meta"]
                        rm  = st.session_state["ohrc_ref_meta"]
                        sz  = st.session_state.get("ohrc_src_zip","")
                        rz  = st.session_state.get("ohrc_ref_zip","")
                        if cross:
                            from src.pipeline import register_cross_source
                            res = register_cross_source(sp,sm,rm,method=method,max_size=max_size)
                            if "cross_source_crop_original_shape" in res:
                                st.markdown(
                                    readout({"CROP":
                                             f"{res['cross_source_crop_original_shape']} → "
                                             f"{res['cross_source_crop_final_shape']}",
                                             "TARGET_GSD":
                                             f"{res.get('cross_source_target_res_m',0):.3f} m/px"}),
                                    unsafe_allow_html=True,
                                )
                        elif use_geo and sz and rz:
                            from src.pipeline import register_with_geo_assist
                            res = register_with_geo_assist(
                                sp,sz,rp,rz,method=method,max_size=max_size,
                                source_meta=sm,reference_meta=rm,
                            )
                            geo_used  = res.get("coarse_geo_prealignment_used",False)
                            n_pts     = res.get("coarse_geo_n_points",0)
                            c_rmse    = res.get("coarse_geo_residual_rmse")
                            if geo_used:
                                st.markdown(
                                    readout({"GEO_POINTS":str(n_pts),
                                             "COARSE_RMSE":
                                             f"{c_rmse:.1f} px" if c_rmse else "N/A",
                                             "H_COMPOSED":
                                             str(res.get("final_homography_composed",False)).upper()}),
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.warning("Geo pre-alignment failed — standard pipeline used")
                        else:
                            from src.pipeline import run_pipeline
                            res = run_pipeline(sp,rp,method=method,max_size=max_size,
                                               src_sun_elevation=sm.get("sun_elevation"),
                                               ref_sun_elevation=rm.get("sun_elevation"),
                                               src_metadata=sm,ref_metadata=rm)
                        st.session_state["result"] = res
                        if not res.get("success",True):
                            st.error(res.get("failure_reason","Registration failed"))
                        else:
                            m = res["metrics"]
                            c = res.get("confidence","medium")
                            st.markdown(
                                confidence_chip(c)+"&nbsp;"+
                                subpixel_chip(m["rmse"],res.get("degenerate_fit",False)),
                                unsafe_allow_html=True,
                            )
                            if res.get("degenerate_fit"):
                                st.warning(res.get("reliability_reason",""))
                            rc1,rc2,rc3,rc4 = st.columns(4)
                            rc1.metric("RMSE",f"{m['rmse']:.4f} px")
                            rc2.metric("INLIERS",m["inlier_count"])
                            rc3.metric("INLIER RATIO",f"{m['inlier_ratio']:.1%}")
                            rc4.metric("SPATIAL SCORE",f"{m['spatial_score']:.4f}")
                            st.info("→ See CORRESPONDENCE ANALYSIS and REGISTRATION OUTPUT tabs")
                    except Exception as e:
                        st.error(f"Registration failed: {e}")

        elif not inputs_ready:
            st.markdown(
                '<div style="padding:16px;background:#ffffff;border:1px solid #d0d4db;'
                'font-family:\'IBM Plex Mono\',monospace;font-size:0.74em;color:#5a6271;'
                'line-height:1.8">'
                '<span style="color:#003f7f;font-weight:700">OHRC ZIP EXAMPLE</span><br>'
                '…/calibrated/20260102/ch2_ohr_ncp_20260102T1224107393_d_img_d18.zip<br><br>'
                '<span style="color:#003f7f;font-weight:700">LRO NAC GEOTIFF EXAMPLE</span><br>'
                '…/nac_roi/moutnpltlo1/nac_roi_moutnpltlo1_p843s0306.tif'
                '</div>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════
# TAB 3 — CORRESPONDENCE ANALYSIS
# ══════════════════════════════════════════════════════════════════
with tab3:
    if "result" not in st.session_state:
        st.markdown(
            '<div style="padding:48px;text-align:center;font-family:\'IBM Plex Mono\','
            'monospace;font-size:0.72em;color:#9aa0aa;letter-spacing:0.6px">'
            'NO RESULT AVAILABLE · RUN REGISTRATION FIRST</div>',
            unsafe_allow_html=True,
        )
    else:
        res = st.session_state["result"]
        if not res.get("success",True) or "match_visualization" not in res:
            st.error(f"Registration did not produce correspondence data. "
                     f"{res.get('failure_reason','')}")
        else:
            m = res["metrics"]
            st.markdown(sec("Correspondence Statistics"), unsafe_allow_html=True)
            st.markdown(
                readout({
                    "CANDIDATES": str(res.get("total_matches_before_filter",0)),
                    "AFTER_SPATIAL_FILTER": str(res.get("total_matches_after_distribution",0)),
                    "RANSAC_INLIERS": str(m["inlier_count"]),
                    "INLIER_RATIO": f"{m['inlier_ratio']:.1%}",
                    "SPATIAL_SCORE": f"{m['spatial_score']:.4f}",
                }),
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(sec("Match Point Visualisation"), unsafe_allow_html=True)
            st.caption(
                "● Green lines — inlier correspondences (geometrically consistent with estimated transform)   "
                "● Red lines — RANSAC-rejected outliers"
            )
            match_vis = res["match_visualization"]
            st.image(match_vis, use_container_width=True)
            st.download_button(
                "⬇  DOWNLOAD CORRESPONDENCE IMAGE",
                data=numpy_to_bytes(match_vis),
                file_name="lunaralign_correspondences.png",
                mime="image/png",
                use_container_width=True,
            )


# ══════════════════════════════════════════════════════════════════
# TAB 4 — REGISTRATION OUTPUT
# ══════════════════════════════════════════════════════════════════
with tab4:
    if "result" not in st.session_state:
        st.markdown(
            '<div style="padding:48px;text-align:center;font-family:\'IBM Plex Mono\','
            'monospace;font-size:0.72em;color:#9aa0aa;letter-spacing:0.6px">'
            'NO RESULT AVAILABLE · RUN REGISTRATION FIRST</div>',
            unsafe_allow_html=True,
        )
    else:
        res = st.session_state["result"]
        if not res.get("success",True) or "side_by_side" not in res:
            st.error(f"Registration failed. {res.get('failure_reason','')}")
        else:
            view_mode = st.radio(
                "view",
                ["Side by Side","Checkerboard","Difference Map","Composite"],
                horizontal=True,
                label_visibility="collapsed",
            )
            captions = {
                "Side by Side":  "Left: Reference · Right: Registered source",
                "Checkerboard":  "Alternating 64 px tiles · seamless borders = correct alignment",
                "Difference Map":"Pixel-wise absolute difference · dark = aligned · bright = residual error",
                "Composite":     "Registered OHRC placed on LRO NAC background · blue border = OHRC strip boundary",
            }
            key_map = {
                "Side by Side":  "side_by_side",
                "Checkerboard":  "checkerboard",
                "Difference Map":"difference_image",
                "Composite":     "composite",
            }
            st.markdown(
                f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.72em;'
                f'color:#5a6271;margin-bottom:10px">{captions[view_mode]}</div>',
                unsafe_allow_html=True,
            )
            st.image(res[key_map[view_mode]], use_container_width=True)
            dc1,dc2,dc3 = st.columns(3)
            with dc1:
                st.download_button("⬇  DOWNLOAD REGISTERED IMAGE",
                    data=numpy_to_bytes(res["registered_image_refined"]),
                    file_name="lunaralign_registered.png",mime="image/png",
                    use_container_width=True)
            with dc2:
                st.download_button("⬇  DOWNLOAD CHECKERBOARD",
                    data=numpy_to_bytes(res["checkerboard"]),
                    file_name="lunaralign_checkerboard.png",mime="image/png",
                    use_container_width=True)
            with dc3:
                if res.get("composite") is not None:
                    st.download_button("⬇  DOWNLOAD COMPOSITE",
                        data=numpy_to_bytes(res["composite"]),
                        file_name="lunaralign_composite.png",mime="image/png",
                        use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# TAB 5 — QUANTITATIVE REPORT
# ══════════════════════════════════════════════════════════════════
with tab5:
    if "result" not in st.session_state:
        st.markdown(
            '<div style="padding:48px;text-align:center;font-family:\'IBM Plex Mono\','
            'monospace;font-size:0.72em;color:#9aa0aa;letter-spacing:0.6px">'
            'NO RESULT AVAILABLE · RUN REGISTRATION FIRST</div>',
            unsafe_allow_html=True,
        )
    else:
        res  = st.session_state["result"]
        if not res.get("success",True) or "metrics" not in res:
            st.error(f"Registration failed. {res.get('failure_reason','')}")
        else:
            m     = res["metrics"]
            conf  = m.get("confidence", res.get("confidence","medium"))
            degen = m.get("degenerate_fit",False)
            ttype = m.get("transform_type","homography")

            # Status row
            st.markdown(
                confidence_chip(conf) + "&nbsp;" +
                subpixel_chip(m["rmse"], degen) + "&nbsp;" +
                chip(ttype.upper(), "info"),
                unsafe_allow_html=True,
            )
            if degen:
                st.warning(m.get("reliability_reason",""))
            elif m.get("reliability_reason"):
                st.info(m["reliability_reason"])

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Large measurement displays ─────────────────────────
            rm1, rm2, rm3, rm4 = st.columns(4)
            rmse_cls = ("measurement-good" if m["rmse"]<1.0 and not degen
                        else "measurement-bad" if degen else "measurement-warn")
            with rm1:
                st.markdown(
                    f'<div class="measurement-label">RMSE</div>'
                    f'<div class="measurement {rmse_cls}">'
                    f'{m["rmse"]:.4f}<span class="measurement-unit"> px</span></div>',
                    unsafe_allow_html=True,
                )
            with rm2:
                st.markdown(
                    f'<div class="measurement-label">INLIER COUNT</div>'
                    f'<div class="measurement {"measurement-good" if m["inlier_count"]>=10 else "measurement-warn"}">'
                    f'{m["inlier_count"]}</div>',
                    unsafe_allow_html=True,
                )
            with rm3:
                ratio_cls = ("measurement-good" if m["inlier_ratio"]>=0.7
                             else "measurement-warn" if m["inlier_ratio"]>=0.3
                             else "measurement-bad")
                st.markdown(
                    f'<div class="measurement-label">INLIER RATIO</div>'
                    f'<div class="measurement {ratio_cls}">'
                    f'{m["inlier_ratio"]*100:.1f}<span class="measurement-unit"> %</span></div>',
                    unsafe_allow_html=True,
                )
            with rm4:
                spat_cls = ("measurement-good" if m["spatial_score"]>=0.7
                            else "measurement-warn")
                st.markdown(
                    f'<div class="measurement-label">SPATIAL SCORE</div>'
                    f'<div class="measurement {spat_cls}">'
                    f'{m["spatial_score"]:.4f}</div>',
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Instrument panels ──────────────────────────────────
            ip1, ip2, ip3 = st.columns(3, gap="medium")

            def _cls(v, good, warn):
                return "instr-val-good" if v>=good else "instr-val-warn" if v>=warn else "instr-val-bad"

            with ip1:
                st.markdown(
                    f'<div class="instr-panel">'
                    f'<div class="instr-panel-title">ACCURACY</div>'
                    + irow("RMSE", f"{m['rmse']:.4f} px",
                           "instr-val-good" if m["rmse"]<1 and not degen
                           else "instr-val-bad" if degen else "instr-val-warn")
                    + irow("RMSE-X", f"{m['rmse_x']:.4f} px")
                    + irow("RMSE-Y", f"{m['rmse_y']:.4f} px")
                    + irow("SUB-PIXEL",
                           "YES" if m["rmse"]<1 and not degen else "NO",
                           "instr-val-good" if m["rmse"]<1 and not degen else "instr-val-bad")
                    + irow("DEGENERATE FIT",
                           "YES" if degen else "NO",
                           "instr-val-bad" if degen else "instr-val-good")
                    + '</div>',
                    unsafe_allow_html=True,
                )

            with ip2:
                st.markdown(
                    f'<div class="instr-panel">'
                    f'<div class="instr-panel-title">CORRESPONDENCE</div>'
                    + irow("INLIER COUNT", str(m["inlier_count"]),
                           _cls(m["inlier_count"],10,6))
                    + irow("TOTAL MATCHES", str(m["total_matches"]))
                    + irow("INLIER RATIO", f"{m['inlier_ratio']:.4f}",
                           _cls(m["inlier_ratio"],0.7,0.3))
                    + irow("SPATIAL SCORE", f"{m['spatial_score']:.4f}",
                           _cls(m["spatial_score"],0.7,0.4))
                    + irow("DISTRIBUTION", "8×8 GRID ENFORCED")
                    + '</div>',
                    unsafe_allow_html=True,
                )

            with ip3:
                st.markdown(
                    f'<div class="instr-panel">'
                    f'<div class="instr-panel-title">PROCESSING</div>'
                    + irow("ALGORITHM", res.get("method","N/A"))
                    + irow("TRANSFORM", ttype.upper())
                    + irow("CONFIDENCE", conf.upper(),
                           "instr-val-good" if conf=="high"
                           else "instr-val-warn" if conf=="medium"
                           else "instr-val-bad")
                    + irow("PROC TIME", f"{m.get('processing_time',0):.2f} s")
                    + '</div>',
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # ── AI Explanation ─────────────────────────────────────
            st.markdown('<div class="sec-label">AI Result Explanation</div>',
                        unsafe_allow_html=True)
            with st.spinner("Generating explanation…"):
                try:
                    from src.ai_explain import explain_result
                    src_meta = st.session_state.get("ohrc_src_meta")
                    ref_meta = st.session_state.get("ohrc_ref_meta")
                    explanation = explain_result(res, src_meta, ref_meta)
                    st.markdown(
                        f'<div style="background:#ffffff;border:1px solid #d0d4db;'
                        f'border-left:4px solid #003f7f;padding:16px 18px;'
                        f'font-size:0.88em;line-height:1.7;color:#1a1a1a;'
                        f'font-family:\'IBM Plex Sans\',sans-serif;">'
                        f'{explanation}</div>',
                        unsafe_allow_html=True,
                    )
                except Exception as e:
                    st.caption(f"AI explanation unavailable: {e}")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(sec("Full Metrics Report"), unsafe_allow_html=True)
            st.code(res.get("metrics_report","No report available"), language="text")

            with st.expander("Transform matrix"):
                H = res.get("homography_matrix")
                st.caption(f"Transform type: {ttype}  ·  "
                           f"{'3×3 homography' if ttype=='homography' else '2×3 affine'}")
                st.code(str(H) if H is not None else "Not available", language="text")

            st.download_button(
                "⬇  DOWNLOAD METRICS REPORT (.txt)",
                data=res.get("metrics_report",""),
                file_name="lunaralign_report.txt",
                mime="text/plain",
                use_container_width=True,
            )


# ── Footer ─────────────────────────────────────────────────────────
st.markdown(
    '<div class="workbench-footer">'
    'LUNARALIGN · PS166 · ISRO · SIH 2026 · '
    'CHANDRAYAAN-2 IMAGE REGISTRATION WORKBENCH · '
    'OHRC · TMC-2 · IIRS · LRO NAC · SELENE'
    '</div>',
    unsafe_allow_html=True,
)
