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
import base64
import html
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
if sys.platform == "win32":
    # src/pipeline.py prints Unicode (em-dash, box drawing) which crashes on the
    # default cp1252 Windows console encoding — force UTF-8 stdout/stderr.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
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

if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = True

# ── Theme tokens ───────────────────────────────────────────────────
THEMES = {
    "dark": {
        "bg": "#151719", "surface": "#1C2023", "surface-2": "#202529", "surface-3": "#252A2E",
        "input": "#202529", "image-bg": "#171A1C",
        "sidebar": "#191C1F", "blue-dark": "#294C6B",
        "border": "#343A3F", "border-strong": "#464D53",
        "text": "#F1F2EF", "text-muted": "#B5BAB7", "text-dim": "#858C89",
        "blue": "#3F6F9B", "blue-hover": "#5685AE", "blue-text": "#5685AE", "blue-soft": "rgba(63,111,155,0.12)",
        "green": "#4E9A6B", "amber": "#B4874A", "red": "#B85C59",
        "shadow": "0 1px 2px rgba(0,0,0,0.30)",
        "tex-opacity": "0.30",
    },
    "light": {
        "bg": "#F3F4F2", "surface": "#FFFFFF", "surface-2": "#FAFAF8", "surface-3": "#ECEEEB",
        "input": "#FFFFFF", "image-bg": "#1C211F",
        "border": "#D7DAD6", "border-strong": "#B8BCB8",
        "text": "#1C211F", "text-muted": "#626A66", "text-dim": "#89918D",
        "blue": "#174A7C", "blue-hover": "#2D6598", "blue-text": "#174A7C", "blue-soft": "rgba(23,74,124,0.07)",
        "green": "#287A4B", "amber": "#A56A18", "red": "#A94442",
        "shadow": "0 1px 2px rgba(28,33,31,0.06)",
        "tex-opacity": "0.10",
    },
}
_theme = THEMES["dark"] if st.session_state["dark_mode"] else THEMES["light"]


@st.cache_data(show_spinner=False)
def _lunar_texture_b64(path: str) -> str:
    """Darkened strip of the local LRO south-pole mosaic, used as a subtle header texture."""
    try:
        im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if im is None:
            return ""
        if im.ndim == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGRA2GRAY if im.shape[2] == 4 else cv2.COLOR_BGR2GRAY)
        im = cv2.normalize(im.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX)
        im = im[: int(im.shape[0] * 0.56)]
        w = 1100
        im = cv2.resize(im, (w, int(im.shape[0] * w / im.shape[1])), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", (im * 0.85).clip(0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 70])
        return base64.b64encode(buf.tobytes()).decode() if ok else ""
    except Exception:
        return ""


_tex = _lunar_texture_b64(os.path.join(os.path.dirname(os.path.abspath(__file__)), "lro_south_pole (1).tif"))
_root_vars = "\n".join(f"  --{k}: {v};" for k, v in _theme.items())
_lunar_css = f'url("data:image/jpeg;base64,{_tex}")' if _tex else "none"

# ── Design system CSS ──────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

:root {{
{_root_vars}
  --lunar: {_lunar_css};
  --sans: 'IBM Plex Sans', 'Segoe UI', system-ui, sans-serif;
  --mono: 'IBM Plex Mono', ui-monospace, Consolas, monospace;
  --r: 6px;
}}

/* ── Canvas ──────────────────────────────────────────────────────── */
html, body, .stApp {{ background: var(--bg) !important; color: var(--text) !important; font-family: var(--sans) !important; }}
.stApp {{ background: var(--bg) !important; }}
[data-testid="stAppViewContainer"], [data-testid="stMain"] {{ background: transparent !important; }}
[data-testid="stHeader"] {{ background: transparent !important; height: 2.4rem !important; }}
[data-testid="stAppDeployButton"], [data-testid="stDecoration"] {{ display: none !important; }}
[data-testid="stToolbar"] button, [data-testid="stMainMenuButton"] {{ color: var(--text-dim) !important; }}
.stMainBlockContainer, [data-testid="stMainBlockContainer"] {{ padding: 0.9rem 2rem 1.5rem !important; max-width: 1640px !important; }}
[data-testid="stVerticalBlock"] {{ gap: 0.8rem; }}
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {{ color: var(--text); }}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{ color: var(--text-muted) !important; font-size: 13px !important; }}
::-webkit-scrollbar {{ width: 8px; height: 8px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: var(--border-strong); border-radius: 4px; }}

/* ── Header ──────────────────────────────────────────────────────── */
.hdr {{
  position: relative; overflow: hidden; isolation: isolate;
  display: flex; justify-content: space-between; align-items: center; gap: 24px;
  padding: 22px 28px; margin-bottom: 16px; border: 1px solid var(--border); border-radius: var(--r);
  background: var(--surface);
}}
.hdr::before {{
  content: ""; position: absolute; inset: 0 0 0 38%; z-index: -1; opacity: var(--tex-opacity);
  background: var(--lunar) right center / cover no-repeat;
  -webkit-mask-image: linear-gradient(90deg, transparent, #000 55%);
  mask-image: linear-gradient(90deg, transparent, #000 55%);
}}
.hdr-meta {{ font-size: 11.5px; font-weight: 500; letter-spacing: 1.6px; text-transform: uppercase; color: var(--text-muted); }}
.hdr-title {{ font-size: 32px; font-weight: 600; letter-spacing: -0.6px; line-height: 1.15; color: var(--text); margin: 4px 0 2px; }}
.hdr-title span {{ color: var(--blue-text); }}
.hdr-sub {{ font-size: 12.5px; font-weight: 500; letter-spacing: 1.2px; text-transform: uppercase; color: var(--text-muted); }}
.hdr-status {{ text-align: right; flex-shrink: 0; }}
.hdr-ready {{ display: inline-flex; align-items: center; gap: 8px; font-size: 13px; font-weight: 600; letter-spacing: .8px; text-transform: uppercase; color: var(--text); }}
.hdr-ready i {{ width: 7px; height: 7px; border-radius: 50%; background: var(--green); }}
.hdr-line {{ font-size: 12.5px; color: var(--text-muted); margin-top: 4px; }}
.hdr-line b {{ font-weight: 500; color: var(--text); }}

/* ── Tabs: segmented navigation ──────────────────────────────────── */
[data-testid="stTabs"] [role="tablist"] {{
  display: grid !important; grid-template-columns: 1.45fr 1.05fr 1.2fr 0.85fr 0.85fr; gap: 4px;
  padding: 4px !important; margin-bottom: 14px; border: 1px solid var(--border) !important; border-radius: var(--r);
  background: var(--surface) !important; overflow: visible !important;
}}
[data-testid="stTabs"] [role="tablist"]::after, .react-aria-SelectionIndicator,
[data-testid="stTabsScrollRight"], [data-testid="stTabsScrollLeft"] {{ display: none !important; }}
[data-testid="stTab"] {{
  justify-content: flex-start !important; height: auto !important; margin: 0 !important; padding: 10px 12px !important;
  border: none !important; border-radius: 4px !important; background: transparent !important; transition: background .15s;
}}
[data-testid="stTab"]:hover {{ background: var(--surface-3) !important; }}
[data-testid="stTab"] [data-testid="stMarkdownContainer"] p {{
  font-family: var(--sans) !important; font-size: 12.5px !important; font-weight: 600 !important; letter-spacing: .3px;
  text-transform: uppercase; color: var(--text-muted) !important; margin: 0 !important; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}}
[data-testid="stTab"] strong {{ font-family: var(--mono); font-weight: 500; color: var(--text-dim); margin-right: 6px; }}
[data-testid="stTab"][aria-selected="true"] {{ background: var(--blue-dark) !important; border: 1px solid var(--blue) !important; box-shadow: var(--shadow); }}
[data-testid="stTab"][aria-selected="true"] p, [data-testid="stTab"][aria-selected="true"] strong {{ color: var(--text) !important; }}
[data-testid="stTabPanel"] {{ padding: 0 !important; }}

/* ── Section intro ───────────────────────────────────────────────── */
.intro {{ display: flex; justify-content: space-between; align-items: flex-end; gap: 20px; margin: 2px 0 2px; }}
.intro h2 {{ font-family: var(--sans); font-size: 20px !important; font-weight: 600 !important; letter-spacing: -0.2px; color: var(--text) !important; margin: 0 !important; padding: 0 !important; }}
.intro p {{ font-size: 13.5px; color: var(--text-muted) !important; margin: 3px 0 0 !important; }}
.intro-aside {{ font-size: 12.5px; color: var(--text-dim); white-space: nowrap; }}

/* ── Panels ──────────────────────────────────────────────────────── */
div[class*="st-key-panel_"] {{
  background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: var(--r) !important;
  padding: 18px 20px 20px !important; gap: 0.85rem !important;
}}
div.st-key-row_input [data-testid="stHorizontalBlock"] {{ align-items: stretch !important; }}
div.st-key-row_input [data-testid="stColumn"] > [data-testid="stVerticalBlock"] {{ height: 100%; }}
div.st-key-row_input [data-testid="stLayoutWrapper"]:has(> div[class*="st-key-panel_"]) {{ flex: 1 1 auto; }}
div.st-key-row_input div[class*="st-key-panel_"] {{ flex: 1 1 auto; }}

.ph {{ display: flex; justify-content: space-between; align-items: baseline; gap: 12px; }}
.ph-title {{ font-size: 15px; font-weight: 600; color: var(--text); }}
.ph-sub {{ font-size: 13px; color: var(--text-muted); margin-top: 1px; }}
.state {{ display: inline-flex; align-items: center; gap: 7px; font-size: 12.5px; font-weight: 500; color: var(--text-muted); white-space: nowrap; }}
.state i {{ width: 7px; height: 7px; border-radius: 50%; background: var(--text-dim); }}
.state.ok i {{ background: var(--green); }}
.state.warn i {{ background: var(--amber); }}
.state.bad i {{ background: var(--red); }}
.state.ok {{ color: var(--text); }}
.label {{ font-size: 11.5px; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; color: var(--text-muted); }}

/* image metadata */
.meta {{ display: grid; grid-template-columns: minmax(0, 1.5fr) minmax(0, 0.8fr) minmax(0, 0.9fr); gap: 12px; padding-top: 12px; border-top: 1px solid var(--border); }}
.meta div {{ min-width: 0; }}
.meta b {{ display: block; font-family: var(--mono); font-size: 13.5px; font-weight: 500; color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.meta span {{ font-size: 12px; color: var(--text-muted); }}
.fname {{ font-family: var(--mono); font-size: 12px; color: var(--text-dim); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-top: -4px; }}

/* ── Images ──────────────────────────────────────────────────────── */
[data-testid="stImage"] img {{ border-radius: 4px; max-height: 76vh; object-fit: contain; }}
div.st-key-panel_src [data-testid="stElementContainer"]:has([data-testid="stImage"]),
div.st-key-panel_ref [data-testid="stElementContainer"]:has([data-testid="stImage"]),
div.st-key-panel_src [data-testid="stFullScreenFrame"] > div, div.st-key-panel_ref [data-testid="stFullScreenFrame"] > div {{ width: 100% !important; }}
div.st-key-panel_src [data-testid="stImage"], div.st-key-panel_ref [data-testid="stImage"],
div.st-key-panel_src [data-testid="stImageContainer"], div.st-key-panel_ref [data-testid="stImageContainer"] {{
  width: 100% !important; display: flex !important; justify-content: center; align-items: center;
}}
div.st-key-panel_src [data-testid="stImageContainer"], div.st-key-panel_ref [data-testid="stImageContainer"] {{
  height: clamp(220px, 25vh, 280px); background: var(--image-bg); border-radius: 4px;
}}
div.st-key-panel_src img, div.st-key-panel_ref img {{
  width: auto !important; max-width: 100% !important; height: clamp(220px, 25vh, 280px) !important; max-height: clamp(220px, 25vh, 280px) !important; object-fit: contain !important; border-radius: 0;
}}
div.st-key-panel_match [data-testid="stImageContainer"], div.st-key-panel_output [data-testid="stImageContainer"] {{
  background: var(--image-bg); border-radius: 4px; display: flex; justify-content: center;
}}

/* ── Uploader ────────────────────────────────────────────────────── */
[data-testid="stFileUploaderDropzone"] {{
  display: flex !important; flex-direction: column !important; align-items: center !important; justify-content: center !important;
  gap: 14px !important; min-height: 220px; padding: 24px !important; border-radius: 4px !important;
  background: var(--surface-2) !important; border: 1px dashed var(--border-strong) !important; transition: border-color .15s, background .15s;
}}
[data-testid="stFileUploaderDropzone"]:hover {{ border-color: var(--blue-text) !important; background: var(--surface-3) !important; }}
[data-testid="stFileUploaderDropzone"] > span {{ order: 2; margin: 0 !important; }}
[data-testid="stFileUploaderDropzoneInstructions"] {{ order: 1; margin: 0 !important; flex: 0 0 auto !important; display: flex !important; flex-direction: column !important; align-items: center !important; text-align: center; }}
[data-testid="stFileUploaderDropzoneInstructions"] span, [data-testid="stFileUploaderDropzoneInstructions"] small {{ font-family: var(--sans) !important; font-size: 12.5px !important; color: var(--text-muted) !important; }}
[data-testid="stFileUploaderDropzoneInstructions"]::before {{ display: block; font-size: 15px; font-weight: 600; color: var(--text); margin-bottom: 4px; content: "Upload image"; }}
div.st-key-panel_src [data-testid="stFileUploaderDropzoneInstructions"]::before {{ content: "Upload source image"; }}
div.st-key-panel_ref [data-testid="stFileUploaderDropzoneInstructions"]::before {{ content: "Upload reference image"; }}
[data-testid="stFileUploaderDropzone"]:has([data-testid="stFileChips"]) {{
  min-height: 0; flex-direction: row !important; justify-content: space-between !important; padding: 8px 10px !important; border-style: solid !important; background: var(--surface-2) !important;
}}
[data-testid="stFileChips"] {{ flex: 1 1 auto; min-width: 0; }}
[data-testid="stFileChip"] {{ background: transparent !important; border: none !important; box-shadow: none !important; padding-left: 0 !important; }}
[data-testid="stFileChip"] * {{ color: var(--text-muted) !important; font-family: var(--sans) !important; font-size: 12px !important; }}
[data-testid="stFileChip"] > div:first-child {{ background: var(--surface-3) !important; }}
[data-testid="stFileChipName"] {{ color: var(--text) !important; font-size: 13px !important; }}
[data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-borderlessIcon"] {{ color: var(--text-muted) !important; border: 1px solid var(--border) !important; border-radius: 4px !important; }}
[data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-borderlessIcon"] * {{ color: var(--text-muted) !important; }}

/* ── Buttons ─────────────────────────────────────────────────────── */
[data-testid="stBaseButton-primary"] {{
  background: var(--blue) !important; border: 1px solid var(--blue-hover) !important; border-radius: 4px !important; color: #fff !important;
  box-shadow: var(--shadow) !important; transition: background .15s;
}}
[data-testid="stBaseButton-primary"] p {{ font-family: var(--sans) !important; font-weight: 600 !important; color: #fff !important; letter-spacing: .3px; }}
[data-testid="stBaseButton-primary"]:hover {{ background: var(--blue-hover) !important; border-color: var(--blue-hover) !important; }}
[data-testid="stBaseButton-primary"]:disabled {{ background: var(--surface-3) !important; border-color: var(--border) !important; box-shadow: none !important; }}
[data-testid="stBaseButton-primary"]:disabled p {{ color: var(--text-dim) !important; }}
[data-testid="stBaseButton-secondary"] {{ background: var(--surface-2) !important; border: 1px solid var(--border-strong) !important; border-radius: 4px !important; color: var(--text) !important; }}
[data-testid="stBaseButton-secondary"] p {{ color: var(--text) !important; font-family: var(--sans) !important; font-size: 13px !important; font-weight: 500 !important; }}
[data-testid="stBaseButton-secondary"]:hover {{ border-color: var(--text-dim) !important; background: var(--surface-3) !important; }}
[data-testid="stDownloadButton"] button {{ min-height: 40px; }}

/* ── Form controls ───────────────────────────────────────────────── */
[data-testid="stWidgetLabel"] p {{ font-family: var(--sans) !important; font-size: 13px !important; font-weight: 500 !important; color: var(--text) !important; }}
[data-testid="stSelectbox"] [role="group"] {{ background: var(--input) !important; border: 1px solid var(--border-strong) !important; border-radius: 4px !important; }}
[data-testid="stSelectbox"] [role="group"]:focus-within {{ border-color: var(--blue-text) !important; }}
[data-testid="stSelectbox"] input {{ background: transparent !important; color: var(--text) !important; -webkit-text-fill-color: var(--text) !important; font-family: var(--mono) !important; font-size: 13.5px !important; }}
[data-testid="stSelectbox"] [role="group"] button, [data-testid="stSelectbox"] [role="group"] svg {{ color: var(--text-muted) !important; background: transparent !important; }}
[role="listbox"] {{ background: var(--surface-3) !important; border: 1px solid var(--border-strong) !important; }}
[role="option"] {{ color: var(--text) !important; font-family: var(--mono) !important; font-size: 13.5px !important; background: transparent !important; }}
[role="option"][data-focused], [role="option"][aria-selected="true"], [role="option"]:hover {{ background: var(--blue-soft) !important; }}
[role="option"] * {{ color: inherit !important; }}
[data-testid="stTextInputRootElement"], [data-baseweb="input"], [data-baseweb="base-input"] {{ background: var(--input) !important; border-color: var(--border-strong) !important; border-radius: 4px !important; }}
[data-testid="stTextInputRootElement"] input {{ background: transparent !important; color: var(--text) !important; -webkit-text-fill-color: var(--text) !important; font-family: var(--mono) !important; font-size: 13px !important; }}
[data-testid="stTextInputRootElement"] input::placeholder {{ color: var(--text-dim) !important; -webkit-text-fill-color: var(--text-dim) !important; }}
[data-testid="stTextInputRootElement"]:focus-within {{ border-color: var(--blue-text) !important; }}
[data-testid="stSlider"] [role="slider"] {{ background: var(--blue-text) !important; box-shadow: none !important; }}
[data-testid="stSliderThumbValue"] {{ color: var(--text) !important; font-family: var(--mono) !important; font-size: 12.5px !important; }}
[data-testid="stSliderTickBar"], [data-testid="stSliderTickBar"] * {{ color: var(--text-dim) !important; font-family: var(--mono) !important; font-size: 11px !important; }}
[data-testid="stRadio"] [role="radiogroup"] {{ gap: 0 !important; display: inline-flex !important; flex-wrap: wrap; border: 1px solid var(--border); border-radius: var(--r); padding: 3px; background: var(--surface-2); }}
[data-testid="stRadio"] [role="radiogroup"] > label {{ margin: 0 !important; padding: 7px 14px !important; border-radius: 4px; transition: background .15s; }}
[data-testid="stRadioOption"] > div > div > div:first-child:not([data-testid]) {{ display: none !important; }}
[data-testid="stRadio"] [role="radiogroup"] > label[data-selected="true"] {{ background: var(--blue-soft); box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--blue-text) 45%, transparent); }}
[data-testid="stRadio"] [role="radiogroup"] > label[data-selected="true"] p {{ color: var(--text) !important; }}
[data-testid="stRadio"] [role="radiogroup"] > label:hover {{ background: var(--surface-3); }}
[data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) {{ background: var(--blue-soft); box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--blue-text) 45%, transparent); }}
[data-testid="stRadio"] [role="radiogroup"] > label p {{ font-family: var(--sans) !important; font-size: 13px !important; font-weight: 500 !important; color: var(--text-muted) !important; }}
[data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) p {{ color: var(--text) !important; }}
[data-testid="stCheckbox"] label p {{ color: var(--text) !important; font-size: 13.5px !important; }}
[data-testid="stTooltipContent"], [data-baseweb="tooltip"] > div {{ background: var(--surface-3) !important; color: var(--text) !important; border: 1px solid var(--border-strong) !important; font-size: 12px !important; }}
[data-testid="stTooltipIcon"] svg {{ color: var(--text-dim) !important; stroke: var(--text-dim) !important; }}

/* ── Feedback, code, expander, native metric ─────────────────────── */
[data-testid="stAlertContainer"] {{ background: var(--surface-2) !important; border: 1px solid var(--border) !important; border-left: 3px solid var(--border-strong) !important; border-radius: 4px !important; }}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]) {{ border-left-color: var(--amber) !important; }}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]) {{ border-left-color: var(--red) !important; }}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]) {{ border-left-color: var(--green) !important; }}
[data-testid="stAlertContainer"] p, [data-testid="stAlertContainer"] div {{ color: var(--text) !important; font-size: 13.5px !important; }}
[data-testid="stCode"] pre, [data-testid="stCode"] code, [data-testid="stCode"] > div {{ background: var(--surface-2) !important; color: var(--text) !important; font-family: var(--mono) !important; font-size: 12.5px !important; }}
[data-testid="stCode"] pre {{ border: 1px solid var(--border) !important; border-radius: 4px !important; }}
[data-testid="stExpander"] details {{ background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 4px !important; }}
[data-testid="stExpander"] summary {{ background: transparent !important; }}
[data-testid="stExpander"] summary p {{ font-size: 13.5px !important; font-weight: 500 !important; color: var(--text) !important; }}
[data-testid="stExpander"] svg {{ color: var(--text-muted) !important; }}
[data-testid="stMetric"] {{ background: transparent !important; border: none !important; border-left: 1px solid var(--border) !important; padding: 4px 16px !important; }}
[data-testid="stMetricLabel"] p {{ font-size: 12px !important; font-weight: 500 !important; color: var(--text-muted) !important; }}
[data-testid="stMetricValue"], [data-testid="stMetricValue"] div {{ font-family: var(--mono) !important; font-size: 26px !important; font-weight: 500 !important; color: var(--text) !important; }}
[data-testid="stSpinner"], [data-testid="stSpinner"] * {{ color: var(--text-muted) !important; font-size: 13px !important; }}
hr {{ border-color: var(--border) !important; }}

/* ── Measurements (divided rows, not cards) ──────────────────────── */
.mrow {{ display: grid; grid-template-columns: repeat(var(--cols, 4), minmax(0, 1fr)); border-top: 1px solid var(--border); }}
.m {{ padding: 16px 20px 16px; border-bottom: 1px solid var(--border); border-left: 1px solid var(--border); min-width: 0; }}
.m:nth-child(4n+1) {{ border-left: none; padding-left: 0; }}
.m-l {{ font-size: 12px; font-weight: 600; letter-spacing: .9px; text-transform: uppercase; color: var(--text-muted); }}
.m-v {{ font-family: var(--mono); font-size: 32px; font-weight: 500; letter-spacing: -1px; line-height: 1.2; color: var(--text); margin-top: 6px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.m-v small {{ font-size: 15px; font-weight: 400; letter-spacing: 0; color: var(--text-muted); margin-left: 5px; }}
.m-v.txt {{ font-family: var(--sans); font-size: 22px; font-weight: 600; letter-spacing: -0.2px; padding: 5px 0 4px; }}
.m-n {{ font-size: 12.5px; color: var(--text-muted); margin-top: 3px; display: flex; align-items: center; gap: 6px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.m-n i {{ width: 6px; height: 6px; border-radius: 50%; background: var(--text-dim); flex-shrink: 0; }}
.m.good .m-n i {{ background: var(--green); }}
.m.warn .m-n i {{ background: var(--amber); }}
.m.bad  .m-n i {{ background: var(--red); }}
.m.good .m-v.txt {{ color: var(--green); }}
.m.warn .m-v.txt {{ color: var(--amber); }}
.m.bad  .m-v.txt {{ color: var(--red); }}
.mrow.big .m-v {{ font-size: 40px; }}
.mbar {{ height: 3px; border-radius: 2px; background: var(--surface-3); margin-top: 8px; overflow: hidden; max-width: 180px; }}
.mbar i {{ display: block; height: 100%; background: var(--text-muted); }}
.mrow.eight {{ --cols: 8; }}
.mrow.eight .m:nth-child(4n+1) {{ border-left: 1px solid var(--border); padding-left: 20px; }}
.mrow.eight .m:first-child {{ border-left: none; padding-left: 0; }}
@media (max-width: 1700px) {{
  .mrow.eight {{ --cols: 4; }}
  .mrow.eight .m:nth-child(4n+1) {{ border-left: none; padding-left: 0; }}
}}

.result-head {{ display: flex; justify-content: space-between; align-items: baseline; gap: 16px; margin-bottom: 6px; }}
.result-status {{ display: flex; align-items: center; gap: 10px; font-size: 14px; font-weight: 600; color: var(--text); }}
.result-status i {{ width: 8px; height: 8px; border-radius: 50%; background: var(--green); }}
.result-status.bad i {{ background: var(--red); }}
.result-status span {{ font-weight: 400; color: var(--text-muted); }}
.result-status span b {{ font-family: var(--mono); font-weight: 500; color: var(--text); }}
.empty {{ padding: 6px 0 2px; }}
.empty b {{ display: block; font-size: 15px; font-weight: 500; color: var(--text); }}
.empty span {{ font-size: 13.5px; color: var(--text-muted); }}

.log {{ border: 1px solid var(--border); border-radius: 4px; background: var(--surface-2); }}
.log-h {{ display: flex; justify-content: space-between; padding: 9px 14px; border-bottom: 1px solid var(--border); font-size: 12px; font-weight: 600; letter-spacing: .9px; text-transform: uppercase; color: var(--text-muted); }}
.log-h span {{ font-weight: 400; letter-spacing: 0; text-transform: none; }}
.log ol {{ margin: 0; padding: 8px 14px 10px 40px; font-family: var(--mono); font-size: 12.5px; color: var(--text); line-height: 1.75; }}
.log li::marker {{ color: var(--text-dim); }}

/* ── Report sections ─────────────────────────────────────────────── */
.instr-panel {{ background: var(--surface); border: 1px solid var(--border); border-radius: var(--r); padding: 16px 20px; height: 100%; }}
.instr-panel-title {{ font-size: 12px; font-weight: 600; letter-spacing: .9px; text-transform: uppercase; color: var(--text-muted); padding-bottom: 8px; margin-bottom: 2px; border-bottom: 1px solid var(--border); }}
.instr-row {{ display: flex; justify-content: space-between; align-items: baseline; gap: 12px; padding: 8px 0; border-bottom: 1px solid var(--border); font-size: 13.5px; }}
.instr-row:last-child {{ border-bottom: none; padding-bottom: 0; }}
.instr-key {{ color: var(--text-muted); }}
.instr-val {{ font-family: var(--mono); font-weight: 500; color: var(--text); text-align: right; }}
.instr-val-good {{ color: var(--green) !important; font-family: var(--mono); font-weight: 500; }}
.instr-val-warn {{ color: var(--amber) !important; font-family: var(--mono); font-weight: 500; }}
.instr-val-bad  {{ color: var(--red) !important; font-family: var(--mono); font-weight: 500; }}
.ai-explain-box {{ font-size: 14px; line-height: 1.75; color: var(--text); }}

/* ── Shared small components (also used by the data-source view) ─── */
.sec-label {{ font-size: 12px; font-weight: 600; letter-spacing: .9px; text-transform: uppercase; color: var(--text-muted); margin: 8px 0 6px; }}
.readout-strip {{ font-family: var(--mono); font-size: 12.5px; color: var(--text); background: var(--surface-2); border: 1px solid var(--border); border-radius: 4px; padding: 9px 14px; margin: 8px 0; }}
.readout-strip b {{ font-family: var(--sans); font-size: 11.5px; font-weight: 600; letter-spacing: .6px; color: var(--text-muted); margin-right: 4px; }}
.chip {{ display: inline-flex; align-items: center; gap: 7px; font-size: 12.5px; font-weight: 500; padding: 4px 10px; margin: 0 6px 4px 0; border: 1px solid var(--border); border-radius: 4px; color: var(--text); background: var(--surface); }}
.chip::before {{ content: ""; width: 6px; height: 6px; border-radius: 50%; background: var(--text-dim); }}
.chip-high::before, .chip-subpx::before {{ background: var(--green); }}
.chip-medium::before, .chip-low::before, .chip-nosubpx::before {{ background: var(--amber); }}
.chip-failed::before, .chip-degen::before {{ background: var(--red); }}
.chip-info::before {{ background: var(--blue-text); }}
.panel-box {{ padding: 14px 18px; background: var(--surface); border: 1px solid var(--border); border-radius: var(--r); font-family: var(--mono); font-size: 12.5px; color: var(--text-muted); line-height: 1.8; }}
.panel-box .panel-box-heading {{ font-family: var(--sans); font-size: 12px; font-weight: 600; letter-spacing: .8px; color: var(--text); }}

/* ── Sidebar ─────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {{ background: var(--surface) !important; border-right: 1px solid var(--border) !important; }}
[data-testid="stSidebarHeader"] {{ height: 2.4rem !important; min-height: 0 !important; padding-bottom: 0 !important; }}
[data-testid="stSidebarUserContent"] {{ padding: 0 1.25rem 1.5rem !important; }}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{ gap: 0.7rem; }}
[data-testid="stSidebarCollapseButton"] span {{ color: var(--text-dim) !important; }}
.sb-h {{ font-size: 11.5px; font-weight: 600; letter-spacing: 1.1px; text-transform: uppercase; color: var(--text-muted); padding-top: 16px; margin-top: 6px; border-top: 1px solid var(--border); }}
.sb-h.first {{ border-top: none; padding-top: 0; margin-top: 0; }}
.algo-r {{ padding: 8px 0 8px 12px; border-left: 2px solid transparent; }}
.algo-r b {{ font-family: var(--mono); font-size: 13px; font-weight: 500; color: var(--text); }}
.algo-r div {{ font-size: 12.5px; color: var(--text-muted); line-height: 1.45; margin-top: 1px; }}
.algo-r.active {{ border-left-color: var(--blue-text); background: var(--blue-soft); border-radius: 0 4px 4px 0; }}
.algo-r.active b {{ color: var(--blue-text); }}
.req-row {{ display: flex; align-items: baseline; gap: 9px; padding: 5px 0; font-size: 13px; }}
.req-check {{ color: var(--green); font-weight: 600; flex-shrink: 0; }}
.req-name {{ color: var(--text); }}
.req-value {{ margin-left: auto; font-size: 12px; color: var(--text-dim); text-align: right; }}

/* ── Workspace composition (01 Input & registration) ─────────────── */
.hero {{ text-align: center; margin: -4px 0 0; }}
.hero h2 {{ font-family: var(--sans); font-size: 21px !important; font-weight: 600 !important; letter-spacing: -0.3px; color: var(--text) !important; margin: 0 !important; padding: 0 !important; }}
.hero p {{ font-size: 14px; color: var(--text-muted) !important; margin: 2px 0 0 !important; }}

div.st-key-panel_src, div.st-key-panel_ref {{ padding: 16px 18px 16px !important; gap: 12px !important; background: var(--surface) !important; }}
.ih {{ display: flex; justify-content: space-between; align-items: center; gap: 12px; }}
.ih-t {{ display: flex; align-items: center; gap: 10px; font-size: 16px; font-weight: 600; color: var(--text); }}
.ih-role {{ font-size: 10.5px; font-weight: 600; letter-spacing: 1.1px; text-transform: uppercase; color: var(--text-muted); background: var(--surface-3); box-shadow: inset 2px 0 0 var(--blue-text); padding: 2px 8px; border-radius: 3px; }}
.ih-d {{ font-size: 13px; color: var(--text-muted); margin-top: 1px; }}
.imeta {{ display: flex; align-items: baseline; gap: 18px; min-width: 0; font-size: 12.5px; color: var(--text-muted); }}
.imeta span {{ white-space: nowrap; }}
.imeta b {{ font-family: var(--mono); font-size: 13px; font-weight: 500; color: var(--text); }}
.imeta-f {{ margin-left: auto; min-width: 0; overflow: hidden; text-overflow: ellipsis; font-family: var(--mono); font-size: 12px; color: var(--text-dim); }}
div.st-key-panel_src [data-testid="stFileUploaderDropzone"], div.st-key-panel_ref [data-testid="stFileUploaderDropzone"] {{ min-height: clamp(220px, 25vh, 280px); }}
div.st-key-panel_src [data-testid="stFileUploaderDropzone"]:has([data-testid="stFileChips"]),
div.st-key-panel_ref [data-testid="stFileUploaderDropzone"]:has([data-testid="stFileChips"]) {{ min-height: 0; }}
div.st-key-panel_src [data-testid="stImageContainer"], div.st-key-panel_ref [data-testid="stImageContainer"] {{ height: clamp(220px, 25vh, 280px) !important; }}
div.st-key-panel_src img, div.st-key-panel_ref img {{ height: clamp(220px, 25vh, 280px) !important; max-height: clamp(220px, 25vh, 280px) !important; }}
div.st-key-panel_src [data-testid="stFileUploaderDropzone"]:has([data-testid="stFileChips"]),
div.st-key-panel_ref [data-testid="stFileUploaderDropzone"]:has([data-testid="stFileChips"]) {{ padding: 4px 8px !important; }}
div.st-key-panel_result [data-testid="stExpander"] {{ margin-top: 4px; }}
[data-testid="stSidebar"] .req-row {{ flex-wrap: wrap; row-gap: 0; }}
[data-testid="stSidebar"] .req-value {{ width: 100%; margin-left: 22px; text-align: left; font-size: 11.5px; }}

div.st-key-row_input [data-testid="stColumn"]:nth-child(2) [data-testid="stVerticalBlock"] {{ justify-content: center; height: 100%; }}
.flow {{ display: flex; flex-direction: column; align-items: center; gap: 8px; }}
.flow-arrow {{ width: 40px; height: 40px; border-radius: 50%; display: grid; place-items: center; font-size: 18px; color: var(--text-muted); background: var(--surface); border: 1px solid var(--border-strong); }}
.flow-label {{ font-size: 10.5px; font-weight: 600; letter-spacing: .9px; text-transform: uppercase; color: var(--text-dim); text-align: center; line-height: 1.35; }}

div.st-key-action {{ gap: 0 !important; margin: 2px 0 0; }}
div.st-key-action [data-testid="stVerticalBlock"] {{ gap: 6px !important; }}
div.st-key-action [data-testid="stBaseButton-primary"] {{ min-height: 52px !important; border-radius: 6px !important; }}
div.st-key-action [data-testid="stBaseButton-primary"]:disabled {{
  background: color-mix(in srgb, var(--blue) 38%, var(--surface)) !important;
  border-color: color-mix(in srgb, var(--blue) 55%, var(--surface)) !important;
}}
div.st-key-action [data-testid="stBaseButton-primary"]:disabled p {{ color: rgba(255,255,255,0.62) !important; }}
div.st-key-action [data-testid="stBaseButton-primary"] p {{ font-size: 15px !important; letter-spacing: 1px !important; white-space: nowrap; }}
.action-note {{ text-align: center; font-size: 13.5px; color: var(--text); }}
.action-note span {{ display: block; font-size: 12.5px; color: var(--text-muted); margin-top: 1px; }}

div.st-key-panel_result {{ padding: 16px 22px !important; gap: 12px !important; }}
.rs-empty {{ display: flex; justify-content: space-between; align-items: center; gap: 24px; }}
.rs-title {{ display: flex; align-items: center; gap: 9px; font-size: 18px; font-weight: 600; color: var(--text); margin-top: 3px; }}
.rs-title i {{ width: 8px; height: 8px; border-radius: 50%; background: var(--green); }}
.rs-title.wait i {{ background: transparent; border: 1.5px solid var(--text-dim); }}
.rs-title.bad i {{ background: var(--red); }}
.rs-sub {{ font-size: 13.5px; color: var(--text-muted); margin-top: 2px; }}
.rs-will {{ font-size: 12.5px; line-height: 1.6; color: var(--text-dim); text-align: right; }}
.rs-head {{ display: flex; justify-content: space-between; align-items: flex-end; gap: 16px; }}
.rs-meta {{ text-align: right; font-size: 12.5px; color: var(--text-muted); }}
.rs-meta b {{ display: block; font-family: var(--mono); font-size: 15px; font-weight: 500; color: var(--text); }}
.mrow.secondary {{ border-top: none; background: var(--surface-2); border-radius: 4px; }}
.mrow.secondary .m, .mrow.secondary .m:nth-child(4n+1) {{ border-bottom: none; padding: 11px 18px; }}
.mrow.secondary .m-v.txt {{ font-size: 17px; padding: 2px 0 0; }}

[data-testid="stSidebar"] {{ background: var(--sidebar) !important; }}
[data-testid="stSidebar"][aria-expanded="true"] {{ min-width: 272px !important; max-width: 272px !important; }}
.hdr {{ padding: 14px 24px !important; margin-bottom: 12px !important; }}
.hdr-title {{ font-size: 26px !important; margin: 2px 0 1px !important; }}

/* ── Footer ──────────────────────────────────────────────────────── */
.foot {{ display: flex; justify-content: space-between; flex-wrap: wrap; gap: 12px; margin-top: 28px; padding-top: 14px; border-top: 1px solid var(--border); font-size: 12.5px; color: var(--text-dim); }}
.foot b {{ font-weight: 600; color: var(--text-muted); }}
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

def esc(v) -> str:
    return html.escape(str(v))

def intro(title: str, desc: str, aside: str = "") -> str:
    return (f'<div class="intro"><div><h2>{title}</h2><p>{desc}</p></div>'
            f'<div class="intro-aside">{aside}</div></div>')

def panel_head(title: str, sub: str = "", right: str = "") -> str:
    sub_html = f'<div class="ph-sub">{sub}</div>' if sub else ""
    return f'<div class="ph"><div><div class="ph-title">{title}</div>{sub_html}</div>{right}</div>'

def state(text: str, kind: str = "") -> str:
    return f'<span class="state {kind}"><i></i>{text}</span>'

def metric(label, value, unit="", note="", status="", text=False, bar=None) -> str:
    unit_html = f"<small>{unit}</small>" if unit else ""
    note_html = f'<div class="m-n"><i></i>{note}</div>' if note else ""
    bar_html = (f'<div class="mbar"><i style="width:{max(0.0, min(1.0, bar)) * 100:.1f}%"></i></div>'
                if bar is not None else "")
    return (f'<div class="m {status}"><div class="m-l">{label}</div>'
            f'<div class="m-v{" txt" if text else ""}">{value}{unit_html}</div>{bar_html}{note_html}</div>')

def metric_row(items, extra_cls: str = "") -> str:
    return f'<div class="mrow {extra_cls}">{"".join(items)}</div>'

def meta_row(items) -> str:
    return '<div class="meta">' + "".join(f"<div><b>{esc(v)}</b><span>{l}</span></div>" for l, v in items) + "</div>"

def image_head(title: str, role: str, desc: str, loaded) -> str:
    return (f'<div class="ih"><div><div class="ih-t">{title}<span class="ih-role">{role}</span></div>'
            f'<div class="ih-d">{desc}</div></div>'
            f'{state("Loaded", "ok") if loaded else state("Not loaded")}</div>')

def image_meta(img, upload) -> str:
    return (f'<div class="imeta"><span><b>{img.shape[1]} × {img.shape[0]}</b> px</span>'
            f'<span><b>{bit_depth_label(img.dtype)}</b></span>'
            f'<span><b>{size_label(upload.size)}</b></span>'
            f'<span class="imeta-f" title="{esc(upload.name)}">{esc(upload.name)}</span></div>')

def log_list(lines) -> str:
    return '<div class="log"><ol>' + "".join(f"<li>{esc(l)}</li>" for l in lines) + "</ol></div>"

def routing_log(lines, right: str = "") -> str:
    rows = "".join(f"<li>{esc(l)}</li>" for l in lines)
    return f'<div class="log"><div class="log-h">Algorithm routing log<span>{right}</span></div><ol>{rows}</ol></div>'

def bit_depth_label(dtype) -> str:
    return {"uint8": "8-bit", "uint16": "16-bit", "float32": "32-bit float",
            "float64": "64-bit float"}.get(str(dtype), str(dtype))

def size_label(n_bytes: int) -> str:
    return f"{n_bytes / 1_048_576:.2f} MB" if n_bytes >= 1_048_576 else f"{n_bytes / 1024:.0f} KB"

def rmse_note(rmse: float, degen: bool) -> str:
    if degen:      return "Unreliable fit"
    if rmse < 0.5: return "Excellent"
    if rmse < 1.0: return "Sub-pixel"
    if rmse < 2.0: return "Acceptable"
    return "High residual"

def awaiting_view(title: str, desc: str, key: str) -> None:
    st.markdown(intro(title, desc), unsafe_allow_html=True)
    with st.container(key=f"panel_wait_{key}"):
        st.markdown(
            '<div class="empty"><b>No registration result yet.</b>'
            '<span>Run a registration in 01 Input &amp; Registration or 02 Data Sources to populate this view.</span></div>',
            unsafe_allow_html=True,
        )


# ── Sidebar (rendered first so the header reflects current settings) ─
with st.sidebar:
    st.markdown('<div class="sb-h first">Parameters</div>', unsafe_allow_html=True)
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

    st.markdown('<div class="sb-h">Algorithm reference</div>', unsafe_allow_html=True)
    algo_rows = [
        ("auto",      "Multi-tier confidence router. Runs all in parallel."),
        ("lightglue", "DISK + LightGlue Transformer. Best overall accuracy."),
        ("pc-sift",   "Phase congruency → SIFT. Illumination-invariant."),
        ("sift",      "Scale-Invariant Feature Transform. Fast, reliable."),
        ("loftr",     "Detector-free dense Transformer. Low-texture cases."),
        ("akaze",     "Non-linear scale-space. Classical baseline."),
    ]
    st.markdown(
        "".join(
            f'<div class="algo-r{" active" if a == method else ""}"><b>{a}</b><div>{d}</div></div>'
            for a, d in algo_rows
        ),
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sb-h">PS166 compliance</div>', unsafe_allow_html=True)
    reqs = [
        ("Sub-pixel RMSE",       "< 1.0 px"),
        ("Uniform distribution", "8×8 grid"),
        ("RMSE metric",          "Per inlier"),
        ("Inlier count + ratio", "Post-RANSAC"),
        ("Match visualisation",  "Inlier / outlier"),
        ("Registered product",   "Downloadable"),
        ("Illumination-aware",   "Sun angle"),
        ("Multi-modal support",  "OHRC · IIRS · LRO"),
    ]
    st.markdown(
        "".join(
            f'<div class="req-row"><span class="req-check">✓</span>'
            f'<span class="req-name">{n}</span><span class="req-value">{v}</span></div>'
            for n, v in reqs
        ),
        unsafe_allow_html=True,
    )


# ── Header (slot filled again at end of run so it reflects this run) ─
def header_html() -> str:
    return """
<div class="hdr">
  <div>
    <div class="hdr-meta">PS166 · ISRO · SIH 2026 · Chandrayaan-2</div>
    <div class="hdr-title">Lunar<span>Align</span></div>
    <div class="hdr-sub">Lunar image registration &amp; correspondence workbench</div>
  </div>
  <div class="hdr-status">
    <div class="hdr-ready"><i></i>System ready</div>
    <div class="hdr-line">Mode: <b>Image registration</b></div>
  </div>
</div>
"""


st.markdown(header_html(), unsafe_allow_html=True)


# ── Tabs ───────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "**01** Input & Registration",
    "**02** Data Sources",
    "**03** Correspondence",
    "**04** Output",
    "**05** Report",
])


# ══════════════════════════════════════════════════════════════════
# TAB 1 — INPUT & REGISTRATION
# ══════════════════════════════════════════════════════════════════
with tab1:
    st.markdown(
        '<div class="hero"><h2>Input &amp; registration</h2>'
        '<p>Upload two lunar images. The source image is aligned to the reference frame.</p></div>',
        unsafe_allow_html=True,
    )

    source_file = reference_file = source_img = reference_img = None
    with st.container(key="row_input"):
        c_src, c_flow, c_ref = st.columns([1, 0.1, 1], gap="small")

        with c_src:
            with st.container(key="panel_src"):
                src_head = st.empty()
                source_file = st.file_uploader(
                    "src", type=["png", "jpg", "jpeg", "tif", "tiff"],
                    key="source", label_visibility="collapsed",
                )
                if source_file:
                    src_bytes  = source_file.read()
                    source_img = load_image_from_bytes(src_bytes)
                    st.image(source_img, use_container_width=True)
                    st.markdown(image_meta(source_img, source_file), unsafe_allow_html=True)
                src_head.markdown(image_head("Source image", "Moving", "The image to be aligned", source_file),
                                  unsafe_allow_html=True)

        with c_flow:
            st.markdown('<div class="flow"><div class="flow-arrow">→</div><div class="flow-label">aligned<br>to</div></div>',
                        unsafe_allow_html=True)

        with c_ref:
            with st.container(key="panel_ref"):
                ref_head = st.empty()
                reference_file = st.file_uploader(
                    "ref", type=["png", "jpg", "jpeg", "tif", "tiff"],
                    key="reference", label_visibility="collapsed",
                )
                if reference_file:
                    ref_bytes     = reference_file.read()
                    reference_img = load_image_from_bytes(ref_bytes)
                    st.image(reference_img, use_container_width=True)
                    st.markdown(image_meta(reference_img, reference_file), unsafe_allow_html=True)
                ref_head.markdown(image_head("Reference image", "Fixed", "The target frame", reference_file),
                                  unsafe_allow_html=True)

    both = bool(source_file and reference_file)
    with st.container(key="action"):
        _a1, a2, _a3 = st.columns([1, 1.1, 1])
        with a2:
            run_btn = st.button(
                "▶  EXECUTE REGISTRATION",
                type="primary",
                use_container_width=True,
                disabled=not both,
            )
            if both:
                status_line = f"Ready · {esc(method)} · max {max_size} px"
            elif source_file:
                status_line = "Waiting for the reference image"
            elif reference_file:
                status_line = "Waiting for the source image"
            else:
                status_line = "Waiting for source and reference images"
            st.markdown(
                f'<div class="action-note">Align the source image to the reference frame.<span>{status_line}</span></div>',
                unsafe_allow_html=True,
            )

    if run_btn and source_file and reference_file:
        with st.spinner("Running registration pipeline…"):
            result = run_pipeline(
                source_img, reference_img,
                method=method, max_size=max_size,
            )
        st.session_state["result"] = result

    # ── Registration result ───────────────────────────────────────
    result = st.session_state.get("result")
    with st.container(key="panel_result"):
        if result is None:
            st.markdown(
                '<div class="rs-empty"><div><div class="label">Registration result</div>'
                '<div class="rs-title wait"><i></i>Waiting for registration</div>'
                '<div class="rs-sub">Upload a source and reference image to generate alignment metrics.</div></div>'
                '<div class="rs-will">RMSE · Inliers · Inlier ratio<br>Spatial score · Confidence · Transform</div></div>',
                unsafe_allow_html=True,
            )

        elif not result.get("success", True):
            st.markdown(
                '<div class="rs-head"><div><div class="label">Registration result</div>'
                '<div class="rs-title bad"><i></i>Registration failed</div>'
                f'<div class="rs-sub">{esc(result.get("failure_reason", "unknown error"))}</div></div></div>',
                unsafe_allow_html=True,
            )
            if result.get("escalation_log"):
                with st.expander(f'Algorithm routing log · {len(result["escalation_log"])} entries'):
                    st.markdown(log_list(result["escalation_log"]), unsafe_allow_html=True)
            with st.expander("Diagnostic details"):
                st.write(result.get("failure_reason", ""))

        else:
            m     = result["metrics"]
            conf  = result.get("confidence", "medium")
            degen = result.get("degenerate_fit", False)
            ttype = m.get("transform_type", "homography")
            total = m.get("total_matches", m["inlier_count"])
            subpx = m["rmse"] < 1.0 and not degen
            ratio_status = "good" if m["inlier_ratio"] >= 0.7 else "warn" if m["inlier_ratio"] >= 0.3 else "bad"
            conf_status = {"high": "good", "medium": "warn"}.get(conf, "bad")

            st.markdown(
                '<div class="rs-head"><div><div class="label">Registration result</div>'
                '<div class="rs-title"><i></i>Registration complete</div></div>'
                f'<div class="rs-meta"><b>{esc(result.get("method", ""))}</b>'
                f'<span>{m.get("processing_time", 0):.2f} s processing</span></div></div>',
                unsafe_allow_html=True,
            )
            st.markdown(metric_row([
                metric("RMSE", f'{m["rmse"]:.4f}', "px", rmse_note(m["rmse"], degen),
                       "good" if subpx else "bad" if degen else "warn"),
                metric("Inliers", f'{m["inlier_count"]}', f"/ {total}", "RANSAC inliers", ratio_status,
                       bar=m["inlier_ratio"]),
                metric("Inlier ratio", f'{m["inlier_ratio"] * 100:.1f}', "%",
                       "Strong" if m["inlier_ratio"] >= 0.7 else "Moderate" if m["inlier_ratio"] >= 0.3 else "Weak",
                       ratio_status),
                metric("Spatial score", f'{m["spatial_score"]:.3f}', "",
                       "Well distributed" if m["spatial_score"] >= 0.7 else "Partial coverage" if m["spatial_score"] >= 0.4 else "Clustered",
                       "good" if m["spatial_score"] >= 0.7 else "warn"),
            ], "big"), unsafe_allow_html=True)
            st.markdown(metric_row([
                metric("Confidence", esc(conf.capitalize()), "", "", conf_status, text=True),
                metric("Sub-pixel", "Verified" if subpx else ("Unreliable" if degen else "Not met"), "", "",
                       "good" if subpx else "bad" if degen else "warn", text=True),
                metric("Method", esc(result.get("method", "N/A")), "", "", "", text=True),
                metric("Transform", esc(ttype.capitalize()), "", "", "", text=True),
            ], "secondary"), unsafe_allow_html=True)

            if degen and result.get("reliability_reason"):
                st.warning(result["reliability_reason"])

            log = result.get("escalation_log") or []
            if log:
                with st.expander(f"Algorithm routing log · {len(log)} entries"):
                    st.markdown(log_list(log), unsafe_allow_html=True)
            st.caption("Detailed analysis: 03 Correspondence · 04 Output · 05 Report")


# ══════════════════════════════════════════════════════════════════
# TAB 2 — OHRC / LRO NAC / IIRS
# ══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(intro(
        "Data sources",
        "Register mission products: OHRC browse zips, LRO NAC GeoTIFF references and IIRS hyperspectral cubes.",
        "OHRC zip · GeoTIFF · ENVI .hdr / .qub",
    ), unsafe_allow_html=True)
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
                '<div class="panel-box">'
                '<span class="panel-box-heading">OHRC ZIP EXAMPLE</span><br>'
                '…/calibrated/20260102/ch2_ohr_ncp_20260102T1224107393_d_img_d18.zip<br><br>'
                '<span class="panel-box-heading">LRO NAC GEOTIFF EXAMPLE</span><br>'
                '…/nac_roi/moutnpltlo1/nac_roi_moutnpltlo1_p843s0306.tif'
                '</div>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════
# TAB 3 — CORRESPONDENCE ANALYSIS
# ══════════════════════════════════════════════════════════════════
with tab3:
    if "result" not in st.session_state:
        awaiting_view("Correspondence analysis", "Feature matches between the source and reference frames.", "03")
    else:
        res = st.session_state["result"]
        if not res.get("success",True) or "match_visualization" not in res:
            st.error(f"Registration did not produce correspondence data. "
                     f"{res.get('failure_reason','')}")
        else:
            m = res["metrics"]
            total_before = res.get("total_matches_before_filter", 0)
            total_after  = res.get("total_matches_after_distribution", m.get("total_matches", 0))
            total        = m.get("total_matches", m["inlier_count"])
            outliers     = max(0, total - m["inlier_count"])

            st.markdown(intro(
                "Correspondence analysis",
                "Candidate matches, spatial filtering and RANSAC verification for the current registration.",
                f'Method: {esc(res.get("method", ""))}',
            ), unsafe_allow_html=True)
            with st.container(key="panel_match_stats"):
                st.markdown(metric_row([
                    metric("Matches", str(total_before), "", f"{total_after} after spatial filter"),
                    metric("Inliers", str(m["inlier_count"]), "", "Geometrically consistent", "good", bar=m["inlier_ratio"]),
                    metric("Outliers", str(outliers), "", "Rejected by RANSAC", "warn" if outliers else ""),
                    metric("Inlier ratio", f'{m["inlier_ratio"] * 100:.1f}', "%",
                           f'Spatial score {m["spatial_score"]:.3f}',
                           "good" if m["inlier_ratio"] >= 0.7 else "warn"),
                ]), unsafe_allow_html=True)

            with st.container(key="panel_match"):
                st.markdown(panel_head("Match visualisation", "Correspondences between reference and registered source"),
                            unsafe_allow_html=True)
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
        awaiting_view("Registration output", "Visual verification of the registered product.", "04")
    else:
        res = st.session_state["result"]
        if not res.get("success",True) or "side_by_side" not in res:
            st.error(f"Registration failed. {res.get('failure_reason','')}")
        else:
            st.markdown(intro(
                "Registration output",
                "Inspect the registered product against the reference frame.",
                f'Transform: {esc(res["metrics"].get("transform_type", "homography"))}',
            ), unsafe_allow_html=True)
            with st.container(key="panel_output"):
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
                st.caption(captions[view_mode])
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
        awaiting_view("Quantitative report", "Accuracy, correspondence and processing measurements.", "05")
    else:
        res  = st.session_state["result"]
        if not res.get("success",True) or "metrics" not in res:
            st.error(f"Registration failed. {res.get('failure_reason','')}")
        else:
            m     = res["metrics"]
            conf  = m.get("confidence", res.get("confidence","medium"))
            degen = m.get("degenerate_fit",False)
            ttype = m.get("transform_type","homography")
            subpx = m["rmse"] < 1.0 and not degen

            st.markdown(intro(
                "Quantitative report",
                "Accuracy, correspondence and processing measurements for the current registration.",
                confidence_chip(conf) + subpixel_chip(m["rmse"], degen) + chip(ttype.upper(), "info"),
            ), unsafe_allow_html=True)

            if degen:
                st.warning(m.get("reliability_reason",""))
            elif m.get("reliability_reason"):
                st.info(m["reliability_reason"])

            with st.container(key="panel_headline"):
                st.markdown(metric_row([
                    metric("RMSE", f'{m["rmse"]:.4f}', "px", rmse_note(m["rmse"], degen),
                           "good" if subpx else "bad" if degen else "warn"),
                    metric("Inlier ratio", f'{m["inlier_ratio"] * 100:.1f}', "%",
                           "Strong" if m["inlier_ratio"] >= 0.7 else "Moderate" if m["inlier_ratio"] >= 0.3 else "Weak",
                           "good" if m["inlier_ratio"] >= 0.7 else "warn" if m["inlier_ratio"] >= 0.3 else "bad"),
                    metric("Spatial score", f'{m["spatial_score"]:.4f}', "",
                           "Well distributed" if m["spatial_score"] >= 0.7 else "Partial coverage",
                           "good" if m["spatial_score"] >= 0.7 else "warn"),
                    metric("Inliers", str(m["inlier_count"]), f'/ {m["total_matches"]}', "RANSAC inliers",
                           "good" if m["inlier_count"] >= 10 else "warn", bar=m["inlier_ratio"]),
                ], "big"), unsafe_allow_html=True)

            def _cls(v, good, warn):
                return "instr-val-good" if v>=good else "instr-val-warn" if v>=warn else "instr-val-bad"

            geo_used = res.get("coarse_geo_prealignment_used", False)
            if geo_used:
                geo_txt = f'Used · {res.get("coarse_geo_n_points", 0)} points'
                if res.get("coarse_geo_residual_rmse") is not None:
                    geo_txt += f' · {res["coarse_geo_residual_rmse"]:.1f} px'
            else:
                geo_txt = "Not used"

            ip1, ip2, ip3 = st.columns(3, gap="medium")
            with ip1:
                st.markdown(
                    '<div class="instr-panel"><div class="instr-panel-title">Accuracy</div>'
                    + irow("RMSE", f"{m['rmse']:.4f} px",
                           "instr-val-good" if subpx else "instr-val-bad" if degen else "instr-val-warn")
                    + irow("RMSE-X", f"{m['rmse_x']:.4f} px")
                    + irow("RMSE-Y", f"{m['rmse_y']:.4f} px")
                    + irow("Inlier ratio", f"{m['inlier_ratio']:.4f}", _cls(m["inlier_ratio"],0.7,0.3))
                    + irow("Spatial score", f"{m['spatial_score']:.4f}", _cls(m["spatial_score"],0.7,0.4))
                    + irow("Sub-pixel", "Yes" if subpx else "No", "instr-val-good" if subpx else "instr-val-bad")
                    + irow("Degenerate fit", "Yes" if degen else "No", "instr-val-bad" if degen else "instr-val-good")
                    + '</div>',
                    unsafe_allow_html=True,
                )
            with ip2:
                st.markdown(
                    '<div class="instr-panel"><div class="instr-panel-title">Correspondence</div>'
                    + irow("Total matches", str(res.get("total_matches_before_filter", m["total_matches"])))
                    + irow("Filtered matches", str(res.get("total_matches_after_distribution", m["total_matches"])))
                    + irow("RANSAC candidates", str(m["total_matches"]))
                    + irow("Inliers", str(m["inlier_count"]), _cls(m["inlier_count"],10,6))
                    + irow("Distribution", "8×8 grid enforced")
                    + '</div>',
                    unsafe_allow_html=True,
                )
            with ip3:
                st.markdown(
                    '<div class="instr-panel"><div class="instr-panel-title">Processing</div>'
                    + irow("Algorithm", esc(res.get("method","N/A")))
                    + irow("Transform", ttype.capitalize())
                    + irow("Confidence", conf.upper(),
                           "instr-val-good" if conf=="high"
                           else "instr-val-warn" if conf=="medium"
                           else "instr-val-bad")
                    + irow("Processing time", f"{m.get('processing_time',0):.2f} s")
                    + irow("Geo pre-alignment", esc(geo_txt))
                    + '</div>',
                    unsafe_allow_html=True,
                )

            # ── AI Explanation ─────────────────────────────────────
            with st.container(key="panel_ai"):
                st.markdown(panel_head("AI interpretation", "Generated summary of the measurements above"),
                            unsafe_allow_html=True)
                with st.spinner("Generating explanation…"):
                    try:
                        from src.ai_explain import explain_result
                        src_meta = st.session_state.get("ohrc_src_meta")
                        ref_meta = st.session_state.get("ohrc_ref_meta")
                        explanation = explain_result(res, src_meta, ref_meta)
                        st.markdown(
                            f'<div class="ai-explain-box">{esc(explanation).replace(chr(10), "<br>")}</div>',
                            unsafe_allow_html=True,
                        )
                    except Exception as e:
                        st.caption(f"AI explanation unavailable: {e}")

            rp1, rp2 = st.columns([1.35, 1], gap="medium")
            with rp1:
                with st.container(key="panel_report"):
                    st.markdown(panel_head("Full metrics report"), unsafe_allow_html=True)
                    st.code(res.get("metrics_report","No report available"), language="text")
            with rp2:
                with st.container(key="panel_transform"):
                    st.markdown(panel_head("Transform", f"{'3×3 homography' if ttype=='homography' else '2×3 affine'}"),
                                unsafe_allow_html=True)
                    with st.expander("Transform matrix", expanded=True):
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

# Theme toggle is rendered last: calling st.rerun() before the uploaders render
# would make Streamlit drop the uploaded files.
with st.sidebar:
    st.markdown('<div class="sb-h">Display</div>', unsafe_allow_html=True)
    _tlabel = "Switch to light mode" if st.session_state["dark_mode"] else "Switch to dark mode"
    if st.button(_tlabel, key="theme_toggle", use_container_width=True):
        st.session_state["dark_mode"] = not st.session_state["dark_mode"]
        st.rerun()

st.markdown(
    '<div class="foot"><div><b>LunarAlign</b> · PS166 · Smart India Hackathon 2026</div>'
    '<div>Indian Space Research Organisation · Chandrayaan-2</div></div>',
    unsafe_allow_html=True,
)
