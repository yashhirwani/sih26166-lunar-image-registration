"""
app.py
------
Streamlit web UI for the Lunar Image Registration System — "LUNAR-ALIGN"
mission-console theme. PS166 — SIH 2026.

Run with: streamlit run app.py
"""

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import cv2
import sys
import os
import io
import json
import base64
import zipfile

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))
from src.preprocess import load_image_from_bytes, load_image
from src.pipeline import run_pipeline
from src.visualize import numpy_to_bytes
from src.metrics import assess_subpixel_claim

APP_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Bundled demo cases ────────────────────────────────────────────────
# Only the "easy" tier has a CONFIRMED genuine overlap (same acquisition,
# two browse-render variants — see data/test_set/README.md). Medium/hard
# tiers are candidate pairs with UNCONFIRMED overlap (no footprint metadata
# to verify them) — they are labelled as such everywhere they're shown so
# nobody mistakes them for verified ground truth.
DEMO_CASES = {
    "easy": {
        "label": "Easy — Same Acquisition",
        "confirmed": True,
        "description": "Same OHRC acquisition, two browse-render variants. "
                        "Near-identical illumination/geometry — pipeline sanity check.",
        "source": os.path.join(APP_DIR, "data", "test_set", "easy_same_acquisition",
                                "easy_ncp_20260102T1224107393_p1.png"),
        "reference": os.path.join(APP_DIR, "data", "test_set", "easy_same_acquisition",
                                   "easy_nrp_20260102T1224107393_p1.png"),
    },
    "medium": {
        "label": "Medium — Adjacent Orbit (unconfirmed overlap)",
        "confirmed": False,
        "description": "Consecutive-frame captures, same orbit pass. Overlap is a CANDIDATE, "
                        "not verified against footprint metadata — some crops may not share terrain.",
        "source": os.path.join(APP_DIR, "data", "test_set", "medium_adjacent_orbit",
                                "medium_20210405T0047199117_p1.png"),
        "reference": os.path.join(APP_DIR, "data", "test_set", "medium_adjacent_orbit",
                                   "medium_20210405T0047199239_p1.png"),
    },
}

STEP_DEFS = [
    (1, "01", "Mission Brief"),
    (2, "02", "Ingestion & Validation"),
    (3, "03", "Algorithm Config"),
    (4, "04", "Execution Pipeline"),
    (5, "05", "Results Dashboard"),
]
STEP_LABELS = [f"{code}   {label}" for _, code, label in STEP_DEFS]


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def get_deep_model_status():
    """Checks whether torch/kornia are importable and whether LightGlue/LoFTR
    weights are already cached locally (so the UI can warn *before* a run
    that a method may need a first-time download, rather than failing
    mid-pipeline)."""
    status = {"torch": False, "kornia": False, "cuda": False,
              "lightglue_cached": False, "loftr_cached": False}
    try:
        import torch
        status["torch"] = True
        status["cuda"] = torch.cuda.is_available()
    except ImportError:
        pass
    try:
        import kornia  # noqa: F401
        status["kornia"] = True
    except ImportError:
        pass
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints")
    if os.path.isdir(cache_dir):
        files = os.listdir(cache_dir)
        status["lightglue_cached"] = any("lightglue" in f.lower() or "disk" in f.lower() for f in files)
        status["loftr_cached"] = any("loftr" in f.lower() for f in files)
    return status


def image_sanity_warnings(img, label):
    """Lightweight upload validation — catches the most common 'this will
    just fail deep in the pipeline' mistakes before the user hits Run."""
    warnings = []
    if img is None:
        return [f"{label}: could not be read as an image."]
    if img.size == 0:
        warnings.append(f"{label}: image is empty.")
        return warnings
    if img.mean() < 2:
        warnings.append(f"{label}: appears almost entirely black — no features will be detectable.")
    if img.mean() > 253:
        warnings.append(f"{label}: appears almost entirely white/blown out.")
    h, w = img.shape[:2]
    if min(h, w) < 64:
        warnings.append(f"{label}: very small ({w}x{h}px) — registration needs more spatial detail.")
    return warnings


def build_result_bundle_zip(result):
    """Packs the key outputs (registered image, checkerboard, match viz,
    heatmap, metrics report, raw metrics JSON) into one downloadable ZIP —
    'downloadable result bundle' instead of clicking N separate buttons."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        image_keys = {
            "registered_image.png": "registered_image_refined",
            "checkerboard.png": "checkerboard",
            "side_by_side.png": "side_by_side",
            "difference_image.png": "difference_image",
            "match_visualization.png": "match_visualization",
            "spatial_heatmap.png": "spatial_heatmap",
        }
        for fname, key in image_keys.items():
            if key in result and result[key] is not None:
                zf.writestr(fname, numpy_to_bytes(result[key]))
        if result.get("metrics_report"):
            zf.writestr("metrics_report.txt", result["metrics_report"])
        metrics = dict(result.get("metrics", {}))
        metrics.pop("coverage", None)  # contains a numpy array — drop from JSON, it's in the report
        safe_metrics = {k: (v if not hasattr(v, "tolist") else v.tolist())
                        for k, v in metrics.items() if not isinstance(v, dict)}
        zf.writestr("metrics.json", json.dumps(safe_metrics, indent=2, default=str))
    buf.seek(0)
    return buf.getvalue()


def build_metrics_report_pdf(result, label="Registration"):
    """Renders a clean one-page PDF report of the run — the deliverable a
    judge/reviewer would actually want to save or print, rather than a raw
    console-text dump. White background (practical for printing), a single
    accent color for identity."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle)
    from reportlab.lib.enums import TA_LEFT
    import datetime

    metrics = result.get("metrics", {})
    holdout = metrics.get("held_out_validation", {})
    coverage = metrics.get("coverage", {})
    subpixel = assess_subpixel_claim(metrics)
    accent = colors.HexColor("#0e7490")

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleX", parent=styles["Title"], textColor=accent, spaceAfter=2)
    h2_style = ParagraphStyle("H2X", parent=styles["Heading2"], textColor=accent,
                               spaceBefore=14, spaceAfter=6)
    body_style = ParagraphStyle("BodyX", parent=styles["BodyText"], alignment=TA_LEFT, leading=14)
    small_style = ParagraphStyle("SmallX", parent=styles["BodyText"], fontSize=8.5,
                                  textColor=colors.HexColor("#666666"))

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=20 * mm, bottomMargin=18 * mm,
                             leftMargin=20 * mm, rightMargin=20 * mm)
    story = []

    story.append(Paragraph("LUNAR-ALIGN — Registration Report", title_style))
    story.append(Paragraph(f"{label} &middot; PS166, SIH 2026", small_style))
    story.append(Paragraph(datetime.datetime.now().strftime("Generated %Y-%m-%d %H:%M"), small_style))
    story.append(Spacer(1, 10 * mm))

    if not result.get("success", True):
        story.append(Paragraph("Registration FAILED", h2_style))
        story.append(Paragraph(result.get("failure_reason", "Unknown error"), body_style))
        doc.build(story)
        buf.seek(0)
        return buf.getvalue()

    story.append(Paragraph("Summary", h2_style))
    summary_rows = [
        ["Algorithm", result.get("method", "N/A")],
        ["Confidence", result.get("confidence", "N/A").upper()],
        ["Transform", metrics.get("transform_type", "unknown")],
        ["Outlier rejection", result.get("outlier_method", "unknown")],
        ["Sub-pixel accuracy", f'{subpixel["label"]} — {subpixel["basis"]}'],
    ]
    t = Table(summary_rows, colWidths=[45 * mm, 115 * mm])
    t.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("TEXTCOLOR", (0, 0), (0, -1), accent),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor("#dddddd")),
    ]))
    story.append(t)

    story.append(Paragraph("Metrics", h2_style))
    metric_rows = [["Metric", "Value"],
                   ["RMSE — fit points", f"{metrics.get('rmse', float('nan')):.4f} px"],
                   ["RMSE — held-out (independent)",
                    f"{holdout['rmse']:.4f} px (n={holdout['n_holdout']} unseen)" if holdout.get("available")
                    else f"unavailable ({holdout.get('reason', 'n/a')})"],
                   ["Inlier count / total matches", f"{metrics.get('inlier_count', '?')} / {metrics.get('total_matches', '?')}"],
                   ["Inlier ratio", f"{metrics.get('inlier_ratio', 0):.2%}"],
                   ["Spatial distribution score", f"{metrics.get('spatial_score', 0):.4f}"],
                   ["Spatial coverage", f"{coverage.get('coverage_fraction', 0):.1%} "
                                        f"({coverage.get('occupied_cells','?')}/{coverage.get('total_cells','?')} cells)"],
                   ["Degenerate fit", "YES — metrics not meaningful" if metrics.get("degenerate_fit") else "No"],
                   ["Processing time", f"{metrics.get('processing_time', 0):.2f}s"]]
    t2 = Table(metric_rows, colWidths=[65 * mm, 95 * mm])
    t2.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef6f8")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor("#dddddd")),
    ]))
    story.append(t2)

    if metrics.get("degenerate_fit") or metrics.get("reliability_reason"):
        story.append(Paragraph("Reliability Note", h2_style))
        story.append(Paragraph(metrics.get("reliability_reason", ""), body_style))

    story.append(Spacer(1, 10 * mm))
    story.append(Paragraph(
        "A same-data fit RMSE below 1.0px is not sufficient evidence of sub-pixel accuracy on its own — "
        "the fit always looks good on the points used to build it. The held-out RMSE (computed on inliers "
        "never used to fit the transform) is the number that should be trusted.", small_style))

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()


def render_subpixel_badge(metrics):
    """Single source of truth for the sub-pixel claim in the UI — mirrors
    src.metrics.assess_subpixel_claim so the app can never show 'sub-pixel
    achieved' off a same-data RMSE alone or on a degenerate fit."""
    claim = assess_subpixel_claim(metrics)
    if claim["achieved"]:
        st.markdown(f'<span class="success-badge">Sub-pixel accuracy achieved</span> '
                     f'<span style="color:#7f8db3">— {claim["basis"]}</span>', unsafe_allow_html=True)
    elif metrics.get("degenerate_fit"):
        st.markdown('<span class="fail-badge">RMSE not meaningful (degenerate fit)</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="fail-badge">Sub-pixel {claim["label"].lower()}</span> '
                     f'<span style="color:#7f8db3">— {claim["basis"]}</span>', unsafe_allow_html=True)


def image_to_data_uri(img):
    return "data:image/png;base64," + base64.b64encode(numpy_to_bytes(img)).decode("ascii")


def go_to_step(n):
    """Navigates the sidebar stepper programmatically (demo cards, Next/Back
    buttons, etc). Only touches the plain 'step' key, never the sidebar
    radio's own widget key — Streamlit raises StreamlitAPIException if a
    widget-owned session_state key is written after that widget has already
    been instantiated in the current script run, which is exactly what
    happens here (the sidebar renders before these step-content callers).
    render_sidebar() derives the radio's `key` from `step` itself, so
    changing `step` here and rerunning is enough to force it to re-init."""
    st.session_state["step"] = n
    st.rerun()


def render_metric_card_grid(cards):
    """cards: list of dicts — label, value, unit, badge, badge_variant,
    sub_label, sub_value, sub_variant. Rendered as one HTML grid so all
    cards share identical height regardless of content length.

    Built as a single unbroken line (no embedded newlines/indentation) —
    Streamlit's markdown parser treats a blank/whitespace-only line inside
    an HTML block (e.g. from an empty conditional substitution) as a
    CommonMark indented-code-block break, which silently turns the rest of
    that block into literal code text instead of rendered HTML.
    """
    parts = ['<div class="mc-grid">']
    for c in cards:
        badge_html = (f'<span class="mc-badge mc-badge-{c.get("badge_variant","cyan")}">{c["badge"]}</span>'
                       if c.get("badge") else "")
        sub_html = ""
        if c.get("sub_label"):
            sub_variant = c.get("sub_variant", "muted")
            sub_html = (f'<div class="mc-sub mc-sub-{sub_variant}">{c["sub_label"]}: '
                        f'<b>{c.get("sub_value","")}</b></div>')
        parts.append(
            '<div class="mc-card">'
            f'<div class="mc-top-row"><span class="mc-label">{c["label"]}</span>{badge_html}</div>'
            f'<div class="mc-value-row"><span class="mc-value">{c["value"]}</span>'
            f'<span class="mc-unit">{c.get("unit","")}</span></div>'
            f'{sub_html}'
            '</div>'
        )
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def render_caveat_banner(title, message):
    st.markdown(
        '<div class="caveat-banner">'
        '<div class="caveat-icon">&#9888;</div>'
        f'<div><div class="caveat-title">{title}</div>'
        f'<div class="caveat-message">{message}</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )


def render_split_slider(reference_img, registered_img, height=520):
    """A real drag-to-reveal split comparison — not a fade/blend. Renders
    an isolated HTML/JS component (no dependency on any JS library)."""
    ref_uri = image_to_data_uri(reference_img)
    reg_uri = image_to_data_uri(registered_img)
    html = f"""
    <div id="split-box" style="position:relative;width:100%;height:{height}px;overflow:hidden;
         border-radius:10px;border:1px solid #253049;cursor:ew-resize;user-select:none;background:#000;
         font-family:'JetBrains Mono',monospace;">
      <img src="{reg_uri}" style="position:absolute;top:0;left:0;width:100%;height:100%;
           object-fit:contain;object-position:center;background:#000;" draggable="false">
      <div id="split-top" style="position:absolute;top:0;left:0;width:50%;height:100%;overflow:hidden;">
        <img id="split-top-img" src="{ref_uri}" style="position:absolute;top:0;left:0;height:100%;
             object-fit:contain;object-position:center;background:#000;" draggable="false">
      </div>
      <div id="split-divider" style="position:absolute;top:0;left:50%;width:2px;height:100%;
           background:#22d3ee;box-shadow:0 0 10px #22d3ee;transform:translateX(-1px);">
        <div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);width:30px;height:30px;
             border-radius:50%;background:#0a0e17;border:2px solid #22d3ee;display:flex;align-items:center;
             justify-content:center;color:#22d3ee;font-size:13px;">&#8596;</div>
      </div>
      <div style="position:absolute;bottom:10px;left:10px;background:rgba(5,7,12,0.8);color:#22d3ee;
           font-size:11px;padding:4px 10px;border-radius:4px;letter-spacing:0.08em;border:1px solid #1c2433;">REFERENCE</div>
      <div style="position:absolute;bottom:10px;right:10px;background:rgba(5,7,12,0.8);color:#a78bfa;
           font-size:11px;padding:4px 10px;border-radius:4px;letter-spacing:0.08em;border:1px solid #1c2433;">REGISTERED</div>
    </div>
    <script>
    (function() {{
        const box = document.getElementById('split-box');
        const top = document.getElementById('split-top');
        const topImg = document.getElementById('split-top-img');
        const divider = document.getElementById('split-divider');
        let dragging = false;
        function fitImgWidth() {{ topImg.style.width = box.offsetWidth + 'px'; }}
        fitImgWidth();
        window.addEventListener('resize', fitImgWidth);
        function setPos(clientX) {{
            const rect = box.getBoundingClientRect();
            let x = Math.max(0, Math.min(rect.width, clientX - rect.left));
            const pct = (x / rect.width) * 100;
            top.style.width = pct + '%';
            divider.style.left = pct + '%';
        }}
        box.addEventListener('mousedown', (e) => {{ dragging = true; setPos(e.clientX); }});
        window.addEventListener('mouseup', () => dragging = false);
        window.addEventListener('mousemove', (e) => {{ if (dragging) setPos(e.clientX); }});
        box.addEventListener('touchstart', (e) => {{ dragging = true; setPos(e.touches[0].clientX); }}, {{passive:true}});
        window.addEventListener('touchend', () => dragging = false);
        box.addEventListener('touchmove', (e) => {{ if (dragging) setPos(e.touches[0].clientX); }}, {{passive:true}});
    }})();
    </script>
    """
    components.html(html, height=height + 16)


CONF_ICON = {"high": "\U0001F7E2", "medium": "\U0001F7E1", "low": "\U0001F534", "failed": "❌"}


# ═══════════════════════════════════════════════════════════════════════
# Page config + theme
# ═══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="LUNAR-ALIGN | SPEC-REG // PS166",
    page_icon="\U0001F311",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');

:root {
  --bg-void: #05070c;
  --bg-panel: #0b0f18;
  --bg-panel-alt: #10151f;
  --border-subtle: #1c2433;
  --border-strong: #2a3548;
  --cyan: #22d3ee;
  --violet: #a78bfa;
  --green: #34d399;
  --red: #f87171;
  --amber: #fbbf24;
  --text-primary: #e7ecf5;
  --text-secondary: #8291ab;
  --text-tertiary: #4b5670;
  --mono: 'JetBrains Mono', Consolas, monospace;
  --sans: 'Inter', -apple-system, sans-serif;
}

html, body, [data-testid="stAppViewContainer"], .main, [data-testid="stHeader"] {
  background-color: var(--bg-void) !important;
  color: var(--text-primary);
  font-family: var(--sans);
}
[data-testid="stHeader"] { background: transparent !important; }
.block-container { padding-top: 1.2rem; max-width: 1400px; }

h1, h2, h3, h4 { color: var(--text-primary) !important; font-family: var(--sans); letter-spacing: -0.01em; }
h2 { font-size: 1.35rem !important; }
h3 { font-size: 1.05rem !important; color: var(--cyan) !important; text-transform: uppercase; letter-spacing: 0.06em; }
p, li, span, label, div { color: var(--text-primary); }
.stCaption, [data-testid="stCaptionContainer"] { color: var(--text-secondary) !important; }

/* ── Sidebar ─────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background-color: var(--bg-panel) !important;
  border-right: 1px solid var(--border-subtle);
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1rem; }
.sb-logo { font-family: var(--mono); font-size: 1.05rem; font-weight: 700; color: var(--cyan);
  letter-spacing: 0.02em; padding: 0 0.2rem 0.4rem 0.2rem; }
.sb-logo-sub { font-size: 0.68rem; color: var(--text-tertiary); font-weight: 400; letter-spacing: 0.08em; }
.sb-section-label { font-family: var(--mono); font-size: 0.65rem; color: var(--text-tertiary);
  letter-spacing: 0.12em; margin: 0.6rem 0 0.3rem 0.2rem; text-transform: uppercase; }
.sb-datum { font-family: var(--mono); font-size: 0.68rem; color: var(--text-tertiary); line-height: 1.6;
  padding: 0 0.2rem; }
.sb-divider { border-top: 1px solid var(--border-subtle); margin: 0.7rem 0; }

/* Sidebar radio restyled as a stepper nav list */
[data-testid="stSidebar"] div[role="radiogroup"] { gap: 2px; }
[data-testid="stSidebar"] div[role="radiogroup"] > label {
  display: flex; align-items: center; padding: 0.55rem 0.6rem; border-radius: 7px;
  border: 1px solid transparent; margin-bottom: 2px; transition: all 0.15s ease;
}
[data-testid="stSidebar"] div[role="radiogroup"] > label:hover { background: rgba(34,211,238,0.06); }
[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
  background: rgba(34,211,238,0.10); border-color: rgba(34,211,238,0.35);
}
[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child { display: none !important; }
[data-testid="stSidebar"] div[role="radiogroup"] p {
  font-family: var(--mono) !important; font-size: 0.78rem !important; color: var(--text-secondary) !important;
  letter-spacing: 0.02em;
}
[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) p { color: var(--cyan) !important; font-weight: 600 !important; }

/* ── Buttons ─────────────────────────────────────────────────── */
.stButton > button, .stDownloadButton > button, [data-testid="stPopoverButton"] {
  background: var(--bg-panel-alt) !important; color: var(--text-primary) !important;
  border: 1px solid var(--border-strong) !important; border-radius: 7px !important;
  font-family: var(--mono) !important; font-size: 0.78rem !important; letter-spacing: 0.03em;
  text-transform: uppercase; padding: 0.5rem 1rem !important; transition: all 0.15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover, [data-testid="stPopoverButton"]:hover {
  border-color: var(--cyan) !important; color: var(--cyan) !important; background: rgba(34,211,238,0.06) !important;
}
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, rgba(34,211,238,0.16), rgba(167,139,250,0.16)) !important;
  border-color: var(--cyan) !important; color: var(--cyan) !important;
}
.stButton > button[kind="primary"]:hover { box-shadow: 0 0 16px rgba(34,211,238,0.25); }
/* Popover flyout panel — matches the rest of the dark theme */
[data-testid="stPopoverBody"] {
  background: var(--bg-panel) !important; border: 1px solid var(--border-strong) !important;
}

/* ── Radio-as-pill-buttons (comparison switcher) ────────────────── */
.st-key-pill_switcher div[role="radiogroup"] { display: flex; flex-wrap: wrap; gap: 8px; }
.st-key-pill_switcher div[role="radiogroup"] > label {
  border: 1px solid var(--border-strong); border-radius: 20px; padding: 0.45rem 1rem !important;
  background: var(--bg-panel-alt); font-family: var(--mono);
}
.st-key-pill_switcher div[role="radiogroup"] > label > div:first-child { display: none !important; }
.st-key-pill_switcher div[role="radiogroup"] > label p { font-size: 0.72rem !important; letter-spacing: 0.04em; text-transform: uppercase; color: var(--text-secondary) !important; }
.st-key-pill_switcher div[role="radiogroup"] > label:has(input:checked) {
  border-color: var(--violet); background: rgba(167,139,250,0.12);
}
.st-key-pill_switcher div[role="radiogroup"] > label:has(input:checked) p { color: var(--violet) !important; font-weight: 600 !important; }

/* ── Top bar ─────────────────────────────────────────────────── */
.topbar { display: flex; justify-content: space-between; align-items: center;
  padding: 0.9rem 1.2rem; background: var(--bg-panel); border: 1px solid var(--border-subtle);
  border-radius: 10px; margin-bottom: 0.8rem; }
.tb-left { display: flex; align-items: baseline; gap: 0.6rem; flex-wrap: wrap; }
.tb-logo { font-family: var(--mono); font-weight: 700; color: var(--cyan); font-size: 1rem; letter-spacing: 0.03em; }
.tb-sep { color: var(--text-tertiary); }
.tb-sub { font-family: var(--mono); font-size: 0.78rem; color: var(--text-secondary); letter-spacing: 0.04em; }
.tb-right { display: flex; align-items: center; gap: 1rem; }
.status-pill { font-family: var(--mono); font-size: 0.68rem; padding: 0.3rem 0.7rem; border-radius: 20px;
  letter-spacing: 0.08em; border: 1px solid; }
.status-pill-cyan { color: var(--cyan); border-color: var(--cyan); background: rgba(34,211,238,0.08); }
.status-pill-muted { color: var(--text-tertiary); border-color: var(--border-strong); background: transparent; }
.tb-runtime { text-align: right; font-family: var(--mono); }
.tb-runtime-label { display: block; font-size: 0.6rem; color: var(--text-tertiary); letter-spacing: 0.1em; }
.tb-runtime-value { display: block; font-size: 0.85rem; color: var(--amber); font-weight: 600; }

/* ── Metric cards ────────────────────────────────────────────── */
.mc-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin: 0.6rem 0 1rem 0; }
.mc-card { background: var(--bg-panel); border: 1px solid var(--border-subtle); border-radius: 10px;
  padding: 1rem 1.1rem; transition: border-color 0.15s ease; }
.mc-card:hover { border-color: var(--border-strong); }
.mc-top-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.55rem; }
.mc-label { font-family: var(--mono); font-size: 0.66rem; color: var(--text-tertiary); letter-spacing: 0.1em; text-transform: uppercase; }
.mc-badge { font-family: var(--mono); font-size: 0.6rem; padding: 0.15rem 0.5rem; border-radius: 10px;
  letter-spacing: 0.06em; border: 1px solid; text-transform: uppercase; }
.mc-badge-cyan { color: var(--cyan); border-color: var(--cyan); background: rgba(34,211,238,0.08); }
.mc-badge-violet { color: var(--violet); border-color: var(--violet); background: rgba(167,139,250,0.08); }
.mc-badge-green { color: var(--green); border-color: var(--green); background: rgba(52,211,153,0.08); }
.mc-badge-amber { color: var(--amber); border-color: var(--amber); background: rgba(251,191,36,0.08); }
.mc-badge-red { color: var(--red); border-color: var(--red); background: rgba(248,113,113,0.08); }
.mc-value-row { display: flex; align-items: baseline; gap: 0.35rem; }
.mc-value { font-family: var(--mono); font-size: 1.9rem; font-weight: 700; color: var(--text-primary); }
.mc-unit { font-family: var(--mono); font-size: 0.85rem; color: var(--text-tertiary); }
.mc-sub { font-size: 0.72rem; color: var(--text-tertiary); margin-top: 0.4rem; }
.mc-sub b { color: var(--text-secondary); }
.mc-sub-green b { color: var(--green) !important; }
.mc-sub-amber b { color: var(--amber) !important; }
.mc-sub-red b { color: var(--red) !important; }

/* ── Caveat banner ───────────────────────────────────────────── */
.caveat-banner { display: flex; gap: 0.7rem; background: rgba(248,113,113,0.06); border: 1px solid rgba(248,113,113,0.4);
  border-left: 3px solid var(--red); border-radius: 8px; padding: 0.8rem 1rem; margin: 0.6rem 0 1rem 0; }
.caveat-icon { color: var(--red); font-size: 1.1rem; }
.caveat-title { font-family: var(--mono); font-size: 0.72rem; color: var(--red); letter-spacing: 0.06em;
  text-transform: uppercase; font-weight: 700; margin-bottom: 0.2rem; }
.caveat-message { font-size: 0.82rem; color: var(--text-secondary); line-height: 1.5; }

/* ── Confirmation badges (demo cards etc.) ──────────────────────── */
.success-badge { background: rgba(52,211,153,0.10); border: 1px solid var(--green); color: var(--green);
  padding: 3px 11px; border-radius: 16px; font-family: var(--mono); font-size: 0.72rem; letter-spacing: 0.03em; }
.fail-badge { background: rgba(248,113,113,0.10); border: 1px solid var(--red); color: var(--red);
  padding: 3px 11px; border-radius: 16px; font-family: var(--mono); font-size: 0.72rem; letter-spacing: 0.03em; }

/* ── Generic panels / widgets ────────────────────────────────── */
[data-testid="stMetric"] { background: var(--bg-panel); border: 1px solid var(--border-subtle);
  border-radius: 8px; padding: 0.7rem 0.9rem; }
[data-testid="stMetricLabel"] { font-family: var(--mono) !important; font-size: 0.68rem !important;
  color: var(--text-tertiary) !important; letter-spacing: 0.08em; text-transform: uppercase; }
[data-testid="stMetricValue"] { font-family: var(--mono) !important; color: var(--cyan) !important; }
[data-testid="stExpander"] { background: var(--bg-panel); border: 1px solid var(--border-subtle) !important; border-radius: 8px !important; }
.stAlert { border-radius: 8px !important; font-family: var(--sans); }
[data-testid="stFileUploader"] { background: var(--bg-panel); border: 1px dashed var(--border-strong); border-radius: 8px; padding: 0.5rem; }

/* ── Form widgets — BaseWeb components default to a light theme and need
   explicit dark overrides; without these, selectboxes/text inputs render
   as light-gray boxes with barely-legible text against the dark page. ── */
[data-baseweb="select"] > div, [data-baseweb="base-input"], [data-baseweb="input"] {
  background: var(--bg-panel-alt) !important; border-color: var(--border-strong) !important;
}
[data-baseweb="select"] input, [data-baseweb="select"] div, [data-baseweb="input"] input {
  color: var(--text-primary) !important;
}
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input, [data-testid="stTextArea"] textarea {
  background: var(--bg-panel-alt) !important; color: var(--text-primary) !important;
  border-color: var(--border-strong) !important;
}
/* Dropdown menu portal (options list) — rendered outside the normal tree */
[data-baseweb="popover"] [data-baseweb="menu"], ul[data-testid="stSelectboxVirtualDropdown"] {
  background: var(--bg-panel) !important; border: 1px solid var(--border-strong) !important;
}
[data-baseweb="menu"] li, [data-testid="stSelectboxVirtualDropdown"] li { color: var(--text-primary) !important; }
[data-baseweb="menu"] li:hover, [data-testid="stSelectboxVirtualDropdown"] li:hover {
  background: rgba(34,211,238,0.10) !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] { background: transparent !important; }
code, .stCode, pre { font-family: var(--mono) !important; }
hr { border-color: var(--border-subtle) !important; }
.step-card { background: var(--bg-panel); border: 1px solid var(--border-subtle); border-radius: 10px; padding: 1.1rem 1.3rem; margin-bottom: 0.8rem; }
.footer-strip { text-align: center; color: var(--text-tertiary); font-family: var(--mono); font-size: 0.68rem;
  letter-spacing: 0.08em; padding: 1.5rem 0 0.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# Session state defaults
# ═══════════════════════════════════════════════════════════════════════

_defaults = {
    "step": 1,
    "method": "auto", "max_size": 1024,
    "ingestion_mode": None,
    "source_img": None, "reference_img": None,
    "src_meta": None, "ref_meta": None,
    "src_zip": "", "ref_zip": "",
    "use_geo_assist": True,
    "result": None, "last_run_label": None,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ═══════════════════════════════════════════════════════════════════════
# Top bar
# ═══════════════════════════════════════════════════════════════════════

def render_top_bar():
    result = st.session_state.get("result")
    status_variant = "cyan" if result else "muted"
    status_txt = "ACTIVE" if result else "IDLE"
    runtime_html = ""
    if result and result.get("success") and "metrics" in result:
        t = result["metrics"].get("processing_time")
        if t:
            gpu = get_deep_model_status().get("cuda")
            runtime_html = (f'<div class="tb-runtime"><span class="tb-runtime-label">WALL RUNTIME</span>'
                             f'<span class="tb-runtime-value">&#9889;{t:.2f}s ({"GPU" if gpu else "CPU"})</span></div>')
    # Built as one unbroken line — see render_metric_card_grid's docstring for
    # why: a blank line from an empty conditional substitution (runtime_html)
    # would otherwise break Streamlit's raw-HTML passthrough mid-block.
    st.markdown(
        '<div class="topbar"><div class="tb-left">'
        '<span class="tb-logo">&#9670; LUNAR-ALIGN</span>'
        '<span class="tb-sep">//</span>'
        '<span class="tb-sub">SPEC-REG // CHANDRAYAAN-2 OHRC / LRO-NAC // PS166</span>'
        f'</div><div class="tb-right">{runtime_html}'
        f'<span class="status-pill status-pill-{status_variant}">{status_txt}</span>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    bcol1, bcol2, bcol3, bcol4 = st.columns(4)
    with bcol1:
        if st.button("↺ New Registration", use_container_width=True):
            for k, v in _defaults.items():
                st.session_state[k] = v
            go_to_step(1)
    with bcol2:
        if result and result.get("success"):
            st.download_button("⬇ Export Bundle (.zip)", data=build_result_bundle_zip(result),
                                file_name="registration_result_bundle.zip", mime="application/zip",
                                use_container_width=True, key="topbar_export")
        else:
            st.button("⬇ Export Bundle (.zip)", use_container_width=True, disabled=True)
    with bcol3:
        if result and result.get("success"):
            st.download_button("⬇ Report (.pdf)",
                                data=build_metrics_report_pdf(result, st.session_state.get("last_run_label", "Registration")),
                                file_name="registration_report.pdf", mime="application/pdf",
                                use_container_width=True, key="topbar_report_pdf")
        else:
            st.button("⬇ Report (.pdf)", use_container_width=True, disabled=True)
    with bcol4:
        with st.popover("\U0001F4D8 System Docs", use_container_width=True):
            st.markdown(
                "**LUNAR-ALIGN / SPEC-REG** — PS166, SIH 2026.\n\n"
                "Registers Chandrayaan-2 OHRC imagery (same-sensor or cross-sensor vs LRO NAC/IIRS) "
                "with MAGSAC++ outlier rejection and independently held-out-verified sub-pixel RMSE.\n\n"
                "See `README.md` and `demo/README.md` in the repo for full documentation, "
                "known limitations, and the benchmark suite."
            )


# ═══════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.markdown('<div class="sb-logo">&#9670; LUNAR-ALIGN<br><span class="sb-logo-sub">SPEC-REG // V1.0</span></div>',
                     unsafe_allow_html=True)
        st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

        # Keyed on the current step so that a programmatic jump (go_to_step,
        # called from step-content buttons rendered after this sidebar)
        # forces a fresh widget instance reflecting the new step, instead of
        # Streamlit reusing stale sticky state under a fixed key — and
        # without ever writing to this widget's own session_state key from
        # outside its own instantiation (which Streamlit disallows).
        current_step = st.session_state["step"]
        choice = st.radio("nav", STEP_LABELS, index=current_step - 1,
                           key=f"step_nav_{current_step}", label_visibility="collapsed")
        chosen_step = STEP_DEFS[STEP_LABELS.index(choice)][0]
        if chosen_step != current_step:
            st.session_state["step"] = chosen_step
            st.rerun()

        st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sb-section-label">Deep-model availability</div>', unsafe_allow_html=True)
        _dm = get_deep_model_status()
        core_ok = _dm["torch"] and _dm["kornia"]
        st.markdown(f'<div class="sb-datum">{"&#9679;" if core_ok else "&#9675;"} torch/kornia '
                     f'{"(" + ("GPU" if _dm["cuda"] else "CPU") + ")" if core_ok else "unavailable"}<br>'
                     f'{"&#9679;" if _dm["lightglue_cached"] else "&#9675;"} LightGlue weights cached<br>'
                     f'{"&#9679;" if _dm["loftr_cached"] else "&#9675;"} LoFTR weights cached</div>',
                     unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# Step 1 — Mission Brief
# ═══════════════════════════════════════════════════════════════════════

def step_mission_brief():
    # Built as ONE st.markdown call (not an opening '<div>' call followed by
    # separate st.subheader/st.markdown calls closed with a later '</div>')
    # — each Streamlit call renders into its own isolated DOM container, so
    # a tag opened in one call is never actually the parent of content from
    # a later call; the open '<div>' rendered as an empty, unstyled box and
    # the content after it got no card styling at all.
    st.markdown(
        '<div class="step-card">'
        '<h3 style="margin-top:0">Mission Brief</h3>'
        '<p><b>LUNAR-ALIGN</b> registers a <b>source</b> lunar image onto a <b>reference</b> image — '
        'same-sensor OHRC&harr;OHRC, or cross-sensor OHRC&harr;LRO/IIRS — and reports sub-pixel '
        'accuracy metrics with <b>independent held-out validation</b>, not just a same-data fit residual. '
        'MAGSAC++ handles outlier rejection; post-RANSAC spatial-coverage checks and a heatmap '
        'confirm matches are uniformly distributed, not clustered.</p>'
        '<p><b>Flow:</b> Ingestion &amp; Validation (load or upload a pair) &rarr; Algorithm Config '
        '(pick a matcher) &rarr; Execution Pipeline (run) &rarr; Results Dashboard. '
        'Or launch a bundled demo case below for a one-click run.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### Bundled demo cases")
    st.caption("Ship with the repo — no external data needed. Only the Easy tier has a **confirmed** genuine "
               "ground overlap; see the badge on each card.")

    demo_cols = st.columns(len(DEMO_CASES))
    for col, (case_id, case) in zip(demo_cols, DEMO_CASES.items()):
        with col:
            st.markdown(f"**{case['label']}**")
            badge_cls = "success-badge" if case["confirmed"] else "fail-badge"
            badge_txt = "CONFIRMED overlap" if case["confirmed"] else "UNCONFIRMED overlap"
            st.markdown(f'<span class="{badge_cls}">{badge_txt}</span>', unsafe_allow_html=True)
            st.caption(case["description"])
            case_files_exist = os.path.exists(case["source"]) and os.path.exists(case["reference"])
            if not case_files_exist:
                st.warning("Demo files not found — skip this card.")
            elif st.button(f"▶ Load &amp; Run", key=f"demo_{case_id}", use_container_width=True):
                with st.spinner(f"Loading and registering '{case['label']}'..."):
                    demo_src = load_image(case["source"])
                    demo_ref = load_image(case["reference"])
                    demo_result = run_pipeline(demo_src, demo_ref,
                                                method=st.session_state["method"],
                                                max_size=st.session_state["max_size"])
                    st.session_state["result"] = demo_result
                    st.session_state["last_run_label"] = case["label"]
                    st.session_state["ingestion_mode"] = "demo"
                go_to_step(5)

    st.button("Continue → Ingestion &amp; Validation", type="primary", on_click=lambda: go_to_step(2))


# ═══════════════════════════════════════════════════════════════════════
# Step 2 — Ingestion & Validation
# ═══════════════════════════════════════════════════════════════════════

def step_ingestion():
    st.subheader("Ingestion &amp; Validation")
    mode = st.radio("Ingestion mode", ["Quick Upload", "Advanced: Real OHRC / LRO / IIRS"],
                     horizontal=True, label_visibility="collapsed")

    if mode == "Quick Upload":
        st.markdown("Upload a **Source** image and a **Reference** image of the same lunar region.")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Source** (moving — will be aligned)")
            source_file = st.file_uploader("source", type=["png", "jpg", "jpeg", "tif", "tiff"],
                                            key="source_upload", label_visibility="collapsed")
            if source_file:
                simg = load_image_from_bytes(source_file.read())
                st.image(simg, caption=f"{source_file.name} | {simg.shape}", use_container_width=True)
                for w in image_sanity_warnings(simg, "Source"):
                    st.warning(w)
                st.session_state["source_img"] = simg
        with col2:
            st.markdown("**Reference** (fixed — target)")
            reference_file = st.file_uploader("reference", type=["png", "jpg", "jpeg", "tif", "tiff"],
                                               key="reference_upload", label_visibility="collapsed")
            if reference_file:
                rimg = load_image_from_bytes(reference_file.read())
                st.image(rimg, caption=f"{reference_file.name} | {rimg.shape}", use_container_width=True)
                for w in image_sanity_warnings(rimg, "Reference"):
                    st.warning(w)
                st.session_state["reference_img"] = rimg

        if st.session_state["source_img"] is not None and st.session_state["reference_img"] is not None:
            st.session_state["ingestion_mode"] = "simple"
            st.success("Both images loaded — continue to Algorithm Config.")
            st.button("Continue → Algorithm Config", type="primary", on_click=lambda: go_to_step(3))
        else:
            st.info("Upload both images to continue.")

    else:
        _render_advanced_ingestion()


def _render_advanced_ingestion():
    st.info(
        "**Advanced / local-file mode.** Reads files by path from the machine running the app "
        "(OHRC .zip bundles, LRO GeoTIFFs, IIRS .hdr/.qub cubes are too large/structured for a "
        "simple browser upload). These flows load *and* register in one step — the result lands "
        "straight on the Results Dashboard."
    )
    ref_source_mode = st.radio(
        "Reference source",
        ["OHRC (Chandrayaan-2 zip)", "LRO NAC / SELENE (GeoTIFF)", "IIRS (hyperspectral)"],
        horizontal=True,
        help="OHRC = two OHRC images.  LRO NAC = OHRC vs external GeoTIFF.  IIRS = hyperspectral cube "
             "synthesised to panchromatic then registered."
    )
    method = st.session_state["method"]
    max_size = st.session_state["max_size"]
    st.divider()

    if ref_source_mode == "IIRS (hyperspectral)":
        _default_iirs_hdr = os.path.join(APP_DIR, "data", "iirs", "ch2_iir_ndi_20250729T0936115604_d_rfl_d18_srd.hdr")
        _default_iirs_qub = os.path.join(APP_DIR, "data", "iirs", "ch2_iir_ndi_20250729T0936115604_d_rfl_d18_srd.qub")

        st.info(
            "IIRS reflectance product: ENVI `.hdr` + headerless `.qub` (256 bands, 712.3-5009.7 nm). "
            "The pipeline synthesizes a panchromatic-equivalent image by equal-weighted averaging of "
            "bands in 712-950 nm — an approximation, labelled clearly in results. Cross-sensor matching "
            "remains **experimental** and not benchmark-validated on confirmed data."
        )

        iirs_col1, iirs_col2 = st.columns(2)
        with iirs_col1:
            st.markdown("**IIRS ENVI header** (`.hdr`):")
            iirs_label_path = st.text_input("IIRS header path", value=_default_iirs_hdr, key="iirs_label_path")
            iirs_hdr_upload = st.file_uploader("Or pick a .hdr file", type=["hdr"], key="iirs_hdr_upload")
            st.markdown("**IIRS cube** (`.qub`, ~3 GB — path recommended over upload):")
            iirs_cube_path = st.text_input("IIRS cube path", value=_default_iirs_qub, key="iirs_cube_path")
        with iirs_col2:
            st.markdown("**Reference image path** (required to register):")
            iirs_ref_path = st.text_input("Reference path",
                                           placeholder="/path/to/ohrc_reference.zip  or  lro_nac.tif",
                                           key="iirs_ref_path")
            iirs_ref_type = st.selectbox("Reference type", ["OHRC zip", "LRO GeoTIFF"], key="iirs_ref_type")

        if iirs_hdr_upload is not None:
            hdr_tmp = os.path.join(APP_DIR, "data", "iirs", "_uploaded.hdr")
            os.makedirs(os.path.dirname(hdr_tmp), exist_ok=True)
            with open(hdr_tmp, "wb") as _f:
                _f.write(iirs_hdr_upload.getvalue())
            iirs_label_path = hdr_tmp

        iirs_load_btn = st.button("Load &amp; Preview IIRS Data", type="primary", use_container_width=True,
                                   disabled=not (iirs_cube_path and iirs_label_path), key="iirs_load_btn")

        if iirs_load_btn and iirs_cube_path and iirs_label_path:
            with st.spinner("Loading IIRS cube (~3 GB)..."):
                try:
                    from src.iirs_loader import (parse_iirs_label, load_iirs_cube,
                                                  synthesize_panchromatic, make_false_color_composite)
                    iirs_meta = parse_iirs_label(iirs_label_path)
                    iirs_cube = load_iirs_cube(iirs_cube_path, iirs_meta)
                    pan = synthesize_panchromatic(iirs_cube, iirs_meta['wavelengths_nm'])
                    try:
                        fc = make_false_color_composite(iirs_cube, iirs_meta['wavelengths_nm'])
                    except Exception as fc_err:
                        fc = None
                        st.warning(f"False-color composite unavailable: {fc_err}")
                    st.session_state['iirs_cube'] = iirs_cube
                    st.session_state['iirs_meta'] = iirs_meta
                    st.session_state['iirs_pan'] = pan
                    st.session_state['iirs_fc'] = fc
                    st.session_state['iirs_ready'] = True
                    st.success(f"IIRS cube loaded: {iirs_cube.shape} (bands x rows x cols), dtype={iirs_cube.dtype}")
                except Exception as e:
                    st.error(f"Failed to load IIRS data: {e}")

        if st.session_state.get('iirs_ready'):
            from src.iirs_loader import format_iirs_metadata_display
            st.code(format_iirs_metadata_display(st.session_state['iirs_meta']))
            pan, fc = st.session_state.get('iirs_pan'), st.session_state.get('iirs_fc')
            pc1, pc2 = st.columns(2)
            with pc1:
                st.markdown("**Synthesized panchromatic band**")
                if pan is not None:
                    disp = pan
                    if pan.shape[0] > 2048:
                        scale = 2048 / pan.shape[0]
                        disp = cv2.resize(pan, (max(1, int(pan.shape[1]*scale)), 2048), interpolation=cv2.INTER_AREA)
                    st.image(disp, caption=f"full shape {pan.shape}", use_container_width=True)
            with pc2:
                if fc is not None:
                    st.markdown("**False-color composite**")
                    fc_disp = fc
                    if fc.shape[0] > 2048:
                        scale = 2048 / fc.shape[0]
                        fc_disp = cv2.resize(fc, (max(1, int(fc.shape[1]*scale)), 2048), interpolation=cv2.INTER_AREA)
                    st.image(fc_disp, caption="False-color SWIR composite", use_container_width=True)

            if st.button("▶ Register IIRS vs Reference", type="primary", use_container_width=True,
                         key="iirs_register_btn", disabled=not bool(iirs_ref_path)):
                with st.spinner("Running IIRS registration..."):
                    try:
                        from src.pipeline import register_iirs
                        from src.ohrc_loader import extract_browse_png_from_zip
                        from src.lro_loader import load_geotiff_reference

                        if iirs_ref_type == "OHRC zip":
                            ref_img, ref_meta = extract_browse_png_from_zip(iirs_ref_path)
                        else:
                            ref_meta_full = load_geotiff_reference(iirs_ref_path)
                            ref_img, ref_meta = ref_meta_full['image'], ref_meta_full

                        result = register_iirs(
                            st.session_state['iirs_cube'], st.session_state['iirs_meta'],
                            ref_img, ref_meta, method=method, max_size=max_size, band_range_nm=(712, 950),
                        )
                        st.session_state['result'] = result
                        st.session_state['ingestion_mode'] = 'iirs'
                        st.session_state['last_run_label'] = "IIRS vs " + iirs_ref_type
                        go_to_step(5)
                    except Exception as e:
                        st.error(f"IIRS registration failed: {e}")

    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Source OHRC zip file path:**")
            src_zip = st.text_input("Source zip path", key="src_zip_input", label_visibility="collapsed",
                                     placeholder="/path/to/ch2_ohr_ncp_....zip")
        with col2:
            if ref_source_mode == "OHRC (Chandrayaan-2 zip)":
                st.markdown("**Reference OHRC zip file path:**")
                ref_zip = st.text_input("Reference zip path", key="ref_zip_input", label_visibility="collapsed",
                                         placeholder="/path/to/ch2_ohr_ncp_....zip")
                lro_tif_path = None
            else:
                st.markdown("**LRO NAC / SELENE GeoTIFF path:**")
                lro_tif_path = st.text_input("GeoTIFF path", key="lro_tif_input", label_visibility="collapsed",
                                              placeholder="/path/to/lro_south_pole.tif")
                ref_zip = None
                st.caption("Export from quickmap.lroc.asu.edu — navigate to your OHRC region, "
                           "select LRO NAC/WAC, export as GeoTIFF in geographic (lat/lon) CRS.")

        inputs_ready = bool(src_zip and (ref_zip or lro_tif_path))
        load_btn = st.button("Load Data", type="primary", use_container_width=True, disabled=not inputs_ready)

        if load_btn and inputs_ready:
            with st.spinner("Loading data..."):
                try:
                    from src.ohrc_loader import extract_browse_png_from_zip, format_metadata_display
                    from src.overlap import check_overlap_and_warn
                    src_patch, src_meta = extract_browse_png_from_zip(src_zip)

                    if ref_source_mode == "OHRC (Chandrayaan-2 zip)":
                        ref_patch, ref_meta = extract_browse_png_from_zip(ref_zip)
                        st.session_state.update({
                            'ohrc_source': src_patch, 'ohrc_reference': ref_patch,
                            'ohrc_src_meta': src_meta, 'ohrc_ref_meta': ref_meta,
                            'cross_source': False, 'ohrc_ready': True,
                            'ohrc_src_zip': src_zip, 'ohrc_ref_zip': ref_zip,
                        })
                        st.success("Both OHRC images loaded")
                        c1, c2 = st.columns(2)
                        with c1:
                            st.image(src_patch, caption=f"Source {src_patch.shape}", use_container_width=True)
                            st.code(format_metadata_display(src_meta))
                        with c2:
                            st.image(ref_patch, caption=f"Reference {ref_patch.shape}", use_container_width=True)
                            st.code(format_metadata_display(ref_meta))

                        src_sun, ref_sun = src_meta.get('sun_elevation', 0), ref_meta.get('sun_elevation', 0)
                        sun_delta = abs(src_sun - ref_sun)
                        st.markdown("**Illumination Analysis**")
                        i1, i2, i3 = st.columns(3)
                        i1.metric("Source Sun Elevation", f"{src_sun:.2f}deg")
                        i2.metric("Reference Sun Elevation", f"{ref_sun:.2f}deg")
                        i3.metric("Sun Angle Delta", f"{sun_delta:.2f}deg",
                                  "Hard" if sun_delta > 20 else "Medium" if sun_delta > 5 else "Easy")
                        if sun_delta > 8:
                            st.warning(f"Large sun angle difference ({sun_delta:.1f}deg) — matching may find "
                                       f"few inliers even with good geographic overlap.")

                        overlap_info, overlap_warning = check_overlap_and_warn(src_meta, ref_meta)
                        st.markdown("**Overlap Analysis**")
                        if overlap_warning:
                            st.warning(overlap_warning)
                        else:
                            st.success(overlap_info.get('recommendation', ''))
                        o1, o2, o3 = st.columns(3)
                        o1.metric("Overlap", f"{overlap_info.get('overlap_fraction', 0):.1%}")
                        o2.metric("Center Distance", f"{overlap_info.get('distance_km', 0):.1f} km")
                        o3.metric("Difficulty", overlap_info.get('difficulty', 'N/A').upper())

                    else:
                        from src.lro_loader import load_geotiff_reference, format_lro_metadata_display
                        lro_meta = load_geotiff_reference(lro_tif_path, source_label="LRO NAC")
                        ref_patch = lro_meta['image']
                        st.session_state.update({
                            'ohrc_source': src_patch, 'ohrc_reference': ref_patch,
                            'ohrc_src_meta': src_meta, 'ohrc_ref_meta': lro_meta,
                            'cross_source': True, 'ohrc_ready': True,
                        })
                        st.success("OHRC source + LRO GeoTIFF reference loaded")
                        c1, c2 = st.columns(2)
                        with c1:
                            st.image(src_patch, caption=f"OHRC source {src_patch.shape}", use_container_width=True)
                            st.code(format_metadata_display(src_meta))
                        with c2:
                            preview = lro_meta['image']
                            if max(preview.shape) > 2048:
                                h, w = preview.shape
                                ch, cw = h // 2, w // 2
                                preview = preview[max(0, ch-512):ch+512, max(0, cw-512):cw+512]
                            st.image(preview, caption=f"LRO reference (preview) — full {lro_meta['shape']}",
                                     use_container_width=True)
                            st.code(format_lro_metadata_display(lro_meta))

                        overlap_info, overlap_warning = check_overlap_and_warn(src_meta, lro_meta)
                        st.markdown("**Overlap Analysis (OHRC vs LRO)**")
                        if overlap_warning:
                            st.warning(overlap_warning)
                        else:
                            st.success(overlap_info.get('recommendation', ''))
                        o1, o2, o3 = st.columns(3)
                        o1.metric("Overlap", f"{overlap_info.get('overlap_fraction', 0):.1%}")
                        o2.metric("Center Distance", f"{overlap_info.get('distance_km', 0):.1f} km")
                        o3.metric("Difficulty", overlap_info.get('difficulty', 'N/A').upper())
                        st.caption("Cross-sensor registration is experimental — not yet benchmark-validated.")

                except Exception as e:
                    st.error(f"Failed to load data: {str(e)}")

        if st.session_state.get('ohrc_ready'):
            cross = st.session_state.get('cross_source', False)
            use_geo_assist = False
            if not cross:
                use_geo_assist = st.checkbox(
                    "Use metadata-assisted coarse pre-alignment (uses .csv geolocation from zip)",
                    value=True, key="geo_assist_checkbox")

            btn_label = ("▶ Register OHRC vs LRO (cross-source)" if cross else
                         ("▶ Register with Geo-Assist" if use_geo_assist else "▶ Register Real OHRC Data"))

            if st.button(btn_label, type="primary", use_container_width=True, key="advanced_register_btn"):
                with st.spinner("Running registration..."):
                    try:
                        src_patch = st.session_state['ohrc_source']
                        ref_patch = st.session_state['ohrc_reference']
                        src_meta = st.session_state['ohrc_src_meta']
                        ref_meta = st.session_state['ohrc_ref_meta']
                        src_zip_s = st.session_state.get('ohrc_src_zip', '')
                        ref_zip_s = st.session_state.get('ohrc_ref_zip', '')

                        if cross:
                            from src.pipeline import register_cross_source
                            result = register_cross_source(src_patch, src_meta, ref_meta,
                                                             method=method, max_size=max_size)
                            st.session_state['ingestion_mode'] = 'ohrc_lro'
                        elif use_geo_assist and src_zip_s and ref_zip_s:
                            from src.pipeline import register_with_geo_assist
                            result = register_with_geo_assist(
                                src_patch, src_zip_s, ref_patch, ref_zip_s,
                                method=method, max_size=max_size)
                            st.session_state['ingestion_mode'] = 'geo_assist'
                        else:
                            result = run_pipeline(
                                src_patch, ref_patch, method=method, max_size=max_size,
                                src_sun_elevation=src_meta.get('sun_elevation'),
                                ref_sun_elevation=ref_meta.get('sun_elevation'),
                                src_metadata=src_meta, ref_metadata=ref_meta)
                            st.session_state['ingestion_mode'] = 'ohrc_ohrc'

                        st.session_state['result'] = result
                        st.session_state['last_run_label'] = "Advanced OHRC/LRO registration"
                        go_to_step(5)
                    except Exception as e:
                        st.error(f"Registration failed: {str(e)}")
        elif not inputs_ready:
            st.info("Enter a source OHRC zip path and a reference path above to begin.")


# ═══════════════════════════════════════════════════════════════════════
# Step 3 — Algorithm Config
# ═══════════════════════════════════════════════════════════════════════

def step_algorithm_config():
    st.subheader("Algorithm Config")

    # Explicit value= + manual write-back rather than relying on key= to pick
    # up the pre-set st.session_state default — on this codebase's Streamlit
    # version that documented key-binding pattern was observed to silently
    # ignore the pre-set value on a widget's first-ever instantiation
    # (slider rendered at min_value instead of the pre-set default).
    algo_options = ["auto", "lightglue", "loftr", "pc-sift", "akaze", "sift"]
    col1, col2 = st.columns(2)
    with col1:
        chosen_method = st.selectbox(
            "Algorithm", algo_options,
            index=algo_options.index(st.session_state.get("method", "auto")),
            help="auto = best-of-all ensemble (AKAZE/SIFT/LightGlue/PC-SIFT/LoFTR). "
                 "pc-sift = illumination-invariant phase congruency."
        )
        st.session_state["method"] = chosen_method
    with col2:
        chosen_max_size = st.slider(
            "Max Image Size (px)", min_value=256, max_value=2048, step=128,
            value=st.session_state.get("max_size", 1024),
            help="Larger = more accurate but slower.")
        st.session_state["max_size"] = chosen_max_size

    st.markdown("**About Algorithms**")
    algo_cols = st.columns(3)
    algo_info = [
        ("auto", "Smart router: runs AKAZE/SIFT/LightGlue/PC-SIFT (+LoFTR if weak), picks the best-scoring result."),
        ("lightglue", "DISK+LightGlue — deep learning, best for hard/illumination-varied cases."),
        ("akaze", "Fast classical, illumination-robust."),
        ("sift", "Classic, reliable, scale-invariant."),
        ("loftr", "Dense deep-learning matcher, detector-free."),
        ("pc-sift", "Phase congruency — illumination-invariant structural matching."),
    ]
    for i, (name, desc) in enumerate(algo_info):
        with algo_cols[i % 3]:
            st.markdown(f"**{name}**")
            st.caption(desc)

    st.divider()
    ready = st.session_state.get("source_img") is not None and st.session_state.get("reference_img") is not None
    if not ready and st.session_state.get("ingestion_mode") != "simple":
        st.info("Load a pair in **Ingestion &amp; Validation** (Quick Upload) to continue with the standard "
                "execution step — or use the Advanced flows, which register inline.")
    st.button("Continue → Execution Pipeline", type="primary", on_click=lambda: go_to_step(4))


# ═══════════════════════════════════════════════════════════════════════
# Step 4 — Execution Pipeline
# ═══════════════════════════════════════════════════════════════════════

def step_execution():
    st.subheader("Execution Pipeline")

    mode = st.session_state.get("ingestion_mode")
    if mode in ("ohrc_lro", "geo_assist", "ohrc_ohrc", "iirs", "demo") and st.session_state.get("result"):
        st.success(f"Registration already executed during ingestion ({mode}). View it on the Results Dashboard.")
        st.button("View Results →", type="primary", on_click=lambda: go_to_step(5))
        return

    source_img = st.session_state.get("source_img")
    reference_img = st.session_state.get("reference_img")

    if source_img is None or reference_img is None:
        st.warning("No source/reference pair ingested yet.")
        st.button("← Back to Ingestion", on_click=lambda: go_to_step(2))
        return

    c1, c2 = st.columns(2)
    c1.image(source_img, caption=f"Source | {source_img.shape}", use_container_width=True)
    c2.image(reference_img, caption=f"Reference | {reference_img.shape}", use_container_width=True)

    st.markdown(f"**Config:** algorithm `{st.session_state['method']}` &middot; "
                f"max size `{st.session_state['max_size']}px`")

    if st.button("▶ RUN REGISTRATION", type="primary", use_container_width=True, key="exec_run_btn"):
        with st.spinner("Running registration pipeline..."):
            try:
                result = run_pipeline(source_img, reference_img,
                                       method=st.session_state["method"],
                                       max_size=st.session_state["max_size"])
                st.session_state["result"] = result
                st.session_state["last_run_label"] = "Quick Upload registration"
                if not result.get("success", True):
                    st.error(f"Registration failed: {result.get('failure_reason', 'Unknown error')}")
                    st.markdown("**Suggestions:** confirm both images show the same lunar region, "
                                "try the `lightglue` algorithm, check images aren't all-black.")
                else:
                    if result.get("escalation_log"):
                        with st.expander("Algorithm routing details"):
                            for log in result["escalation_log"]:
                                st.text(log)
                    st.success("Registration complete.")
                    go_to_step(5)
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")


# ═══════════════════════════════════════════════════════════════════════
# Step 5 — Results Dashboard
# ═══════════════════════════════════════════════════════════════════════

def step_results_dashboard():
    result = st.session_state.get("result")
    label = st.session_state.get("last_run_label") or "Registration"

    if not result:
        st.info("No result yet — run a registration from Execution Pipeline, Ingestion, or a demo case on "
                "Mission Brief.")
        return

    if not result.get("success", True):
        st.error(f"Registration failed: {result.get('failure_reason', 'Unknown error')}")
        st.markdown("**Suggestions:** confirm both images show the same lunar region, try `lightglue`, "
                    "check images aren't all-black or corrupted.")
        return

    metrics = result["metrics"]
    holdout = metrics.get("held_out_validation", {})
    coverage = metrics.get("coverage", {})
    subpixel = assess_subpixel_claim(metrics)
    confidence = result.get("confidence", "medium")

    st.subheader(f"Results Dashboard — {label}")
    st.markdown(f'{CONF_ICON.get(confidence,"")} **{result.get("method","N/A")}** &middot; '
                f'confidence **{confidence.upper()}** &middot; transform **{metrics.get("transform_type","?")}** '
                f'&middot; outlier rejection **{result.get("outlier_method","?")}**')

    # ── Caveat banner ──────────────────────────────────────────
    if metrics.get("degenerate_fit"):
        render_caveat_banner("DEGENERATE FIT — METRICS NOT MEANINGFUL", metrics.get("reliability_reason", ""))
    elif not holdout.get("available"):
        render_caveat_banner(
            "DEPENDENT RESIDUAL METRIC ONLY",
            f"In-sample fit RMSE ({metrics['rmse']:.2f} px) shown below, but an independent held-out RMSE "
            f"could not be computed ({holdout.get('reason', 'too few inliers to split')}). "
            f"Sub-pixel claim is flagged conditional until an independent check is available."
        )
    elif not subpixel["achieved"]:
        render_caveat_banner("SUB-PIXEL ACCURACY NOT ACHIEVED",
                              f"Held-out RMSE {holdout['rmse']:.2f}px &ge; 1.0px — {subpixel['basis']}.")

    # ── Metric cards ───────────────────────────────────────────
    outlier_variant = "cyan" if "MAGSAC" in str(result.get("outlier_method", "")) else "amber"
    residual_sub = (f"n={holdout['n_holdout']} unseen" if holdout.get("available") else "unavailable")
    dof_label = "Projective (8-DOF)" if metrics.get("transform_type") == "homography" else "Affine (6-DOF)"
    condition_badge = "DEGENERATE" if metrics.get("degenerate_fit") else "WELL-COND."
    condition_variant = "red" if metrics.get("degenerate_fit") else "green"
    max_cell = coverage.get("max_cell_fraction", 0)
    bias_txt = "SOME BIAS" if max_cell > 0.25 else "NONE DETECTED"
    bias_variant = "amber" if max_cell > 0.25 else "green"

    render_metric_card_grid([
        {
            "label": "Residual Error / RMSE", "value": f"{metrics['rmse']:.2f}", "unit": "px",
            "badge": "SUB-PIXEL" if subpixel["achieved"] else "CHECK", "badge_variant": "green" if subpixel["achieved"] else "amber",
            "sub_label": "Held-out RMSE", "sub_value": (f"{holdout['rmse']:.2f} px" if holdout.get("available") else "unavailable"),
            "sub_variant": "green" if holdout.get("available") and holdout["rmse"] < 1.0 else "amber",
        },
        {
            "label": "Inlier Retention Ratio", "value": f"{metrics['inlier_ratio']*100:.2f}", "unit": "%",
            "badge": result.get("outlier_method", "RANSAC").split(" ")[0], "badge_variant": outlier_variant,
            "sub_label": "Valid correspondences", "sub_value": f"{metrics['inlier_count']} / {metrics['total_matches']}",
        },
        {
            "label": "Spatial Dispersion", "value": f"{metrics['spatial_score']:.3f}", "unit": "/ 1.0",
            "badge": "ENTROPY 2D", "badge_variant": "violet",
            "sub_label": "Quadrant bias", "sub_value": f"{bias_txt} ({max_cell:.0%})", "sub_variant": bias_variant,
        },
        {
            "label": "Model &amp; Conditioning", "value": dof_label.split(" ")[0], "unit": dof_label.split(" ")[1],
            "badge": condition_badge, "badge_variant": condition_variant,
            "sub_label": "Coverage", "sub_value": f"{coverage.get('coverage_fraction', 0):.0%} of grid cells",
        },
    ])

    st.divider()

    # ── Comparison view switcher ─────────────────────────────────
    # st.container(key=...) — not a '<div>' opened in one st.markdown() call
    # and closed in another — is the correct way to scope CSS to a widget's
    # container in Streamlit; each st.markdown() call renders into its own
    # isolated DOM node, so a tag spanning two separate calls never actually
    # wraps the widget in between (see step_mission_brief's fix for the same
    # bug). st.container(key="pill_switcher") emits a stable `st-key-pill_switcher`
    # class on the real wrapping element, which the CSS below targets.
    with st.container(key="pill_switcher"):
        view = st.radio("view", ["Split-Slider Overlay", "Checkerboard Blend", "Differential Error Map",
                                  "Match Vectors", "Spatial Heatmap"],
                         horizontal=True, label_visibility="collapsed", key="dashboard_view_mode")

    if view == "Split-Slider Overlay":
        ref_img = result.get("reference_image")
        reg_img = result.get("registered_image_refined")
        if ref_img is None:
            sbs = result["side_by_side"]
            half_w = sbs.shape[1] // 2
            ref_img, reg_img = sbs[:, :half_w - 2], sbs[:, half_w + 2:]
        render_split_slider(ref_img, reg_img)
        st.caption("Drag the divider (or click anywhere) to compare reference vs registered source. "
                   "A good registration shows no feature shift as you sweep across.")

    elif view == "Checkerboard Blend":
        st.image(result["checkerboard"], caption="Alternating reference/registered tiles — seamless = perfect alignment",
                  use_container_width=True)

    elif view == "Differential Error Map":
        st.image(result["difference_image"], caption="Bright = misaligned, dark = well-aligned",
                  use_container_width=True)

    elif view == "Match Vectors":
        st.image(result["match_visualization"], caption="Green = inlier match, red = outlier",
                  use_container_width=True)
        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("Total Matches (pre-filter)", result.get("total_matches_before_filter", 0))
        mcol2.metric("After Distribution Filter", result.get("total_matches_after_distribution", 0))
        mcol3.metric("Inliers (correct)", metrics["inlier_count"])

    else:  # Spatial Heatmap
        if coverage.get("max_cell_fraction", 0) > 0.4:
            st.warning("One grid cell holds over 40% of all inliers — the fit may be dominated by a "
                       "single region despite an acceptable overall spatial score.")
        st.image(result["spatial_heatmap"],
                  caption="Inlier density per grid cell (numbers = inlier count per cell)",
                  use_container_width=True)
        hc1, hc2, hc3, hc4 = st.columns(4)
        hc1.metric("Occupied Cells", f"{coverage.get('occupied_cells','?')}/{coverage.get('total_cells','?')}")
        hc2.metric("Coverage", f"{coverage.get('coverage_fraction',0):.1%}")
        hc3.metric("Spread X x Y", f"{coverage.get('spatial_spread_x',0):.0%} x {coverage.get('spatial_spread_y',0):.0%}")
        hc4.metric("Max Cell Share", f"{max_cell:.0%}")


# ═══════════════════════════════════════════════════════════════════════
# Router
# ═══════════════════════════════════════════════════════════════════════

render_top_bar()
render_sidebar()

_step = st.session_state["step"]
if _step == 1:
    step_mission_brief()
elif _step == 2:
    step_ingestion()
elif _step == 3:
    step_algorithm_config()
elif _step == 4:
    step_execution()
else:
    step_results_dashboard()

st.markdown('<div class="footer-strip">PS166 // SIH 2026 // CHANDRAYAAN-2 LUNAR IMAGE REGISTRATION // ISRO PRADAN DATASET</div>',
            unsafe_allow_html=True)
