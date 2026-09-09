# SIH26166 — Lunar Image Registration System
**PS166 | SIH 2026 | ISRO Chandrayaan-2 OHRC Image Registration**

## What this project does
This software takes two lunar images (from Chandrayaan-2 OHRC, with experimental support for LRO NAC and IIRS) and aligns them on top of each other. It handles different lighting conditions, camera angles, and zoom levels using a mix of classical and deep-learning feature matching.

---

## Quick start (Windows / PowerShell)

```powershell
git clone <repo-url>
cd "SIH FINAL\Project"

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

This was developed and tested with **Python 3.13 on Windows**. The commands above should work the same on macOS/Linux (`source venv/bin/activate` instead of `venv\Scripts\activate`), though that hasn't been specifically tested here.

### opencv gotcha — read this before installing manually

`requirements.txt` lists only `opencv-contrib-python`. **Never** additionally `pip install opencv-python` in this environment — having both packages installed at once causes a conflict where `opencv-contrib-python`'s extra modules (e.g. AKAZE) silently disappear from the `cv2` namespace, even though import succeeds with no error. If matching mysteriously loses AKAZE support, check `pip list` for both packages and remove the plain `opencv-python` one.

### Deep-learning weights (LoFTR / LightGlue) — first-run download

`kornia` + `torch` + `torchvision` power the LoFTR and DISK+LightGlue matchers. The first time either is used, it downloads pretrained weights via `torch.hub` into your local cache:

- Windows: `%USERPROFILE%\.cache\torch\hub\checkpoints`
- macOS/Linux: `~/.cache/torch/hub/checkpoints`

After that first download, the pipeline runs fully offline. The `models/` directory in the repo is just a placeholder (`.gitkeep`) — weights are **not** vendored in the repo.

Known issue: LoFTR's weight host (`cmp.felk.cvut.cz`) has been observed to fail with a TLS certificate error on some networks. If that happens, LightGlue and the classical methods (SIFT, AKAZE, PC-SIFT) still work fine — only LoFTR is affected.

---

## How to use the app

1. Upload a Source image (the image to be aligned)
2. Upload a Reference image (the target image)
3. Run registration
4. Check the result tabs — matched keypoints, aligned/warped output, and metrics (RMSE, inlier count/ratio, spatial coverage)

---

## Project structure

```
Project/                        ← repo root
├── app.py                      — Streamlit UI, the only entrypoint (run: streamlit run app.py)
├── config.yaml                 — tunable thresholds (preprocessing, matching, RANSAC, refinement)
├── requirements.txt
├── src/                        — the actual pipeline code
│   ├── config_loader.py        — loads & caches config.yaml
│   ├── console.py              — UTF-8 console setup for Windows terminals
│   ├── exceptions.py           — RegistrationError exception hierarchy used across the pipeline
│   ├── geo_align.py            — geo-assisted coarse pre-alignment from OHRC .csv geolocation files
│   ├── iirs_loader.py          — IIRS hyperspectral cube loader + panchromatic synthesis (experimental)
│   ├── lightglue_match.py      — DISK + LightGlue deep-learning matcher
│   ├── loftr_match.py          — LoFTR (Local Feature TRansformer) detector-free matcher
│   ├── lro_loader.py           — LRO NAC/WAC GeoTIFF reference tile loader (via rasterio)
│   ├── match.py                — SIFT / AKAZE keypoint detection + descriptor matching
│   ├── metrics.py              — RMSE, inlier count/ratio, spatial distribution, reliability assessment
│   ├── ohrc_loader.py          — Chandrayaan-2 OHRC .img + XML metadata loader
│   ├── overlap.py              — checks whether two OHRC footprints overlap before matching
│   ├── pipeline.py             — connects all modules into the end-to-end registration pipeline
│   ├── preprocess.py           — CLAHE, histogram matching, shadow masking, resizing
│   ├── structural_match.py     — phase-congruency (PC-SIFT) illumination-invariant matching
│   ├── transform.py            — homography/affine estimation, MAGSAC++/RANSAC, warping, sub-pixel refinement, held-out RMSE
│   └── visualize.py            — match-line, checkerboard, side-by-side, and heatmap visualizations
├── generate_demo_cases.py       — regenerates the frozen demo/case*/ artifacts from bundled test data
├── scripts/
│   └── run_benchmark.py        — CLI benchmark sweep across methods/pairs, writes demo/benchmark_results.*
├── tests/                      — pytest suite (71 tests): test_metrics.py, test_transform.py, test_match.py,
│                                   test_metrics_extra.py, test_loaders.py — run: python -m pytest tests/ -v
├── data/
│   ├── patches/                — OHRC browse-image patches for manual testing
│   ├── reference/               — LRO reference tiles (GeoTIFF) for cross-sensor testing
│   └── test_set/                — curated easy/medium/hard registration test pairs (see below)
├── models/                     — placeholder only (.gitkeep); DL weights are downloaded on first run, not vendored
├── logs/                       — runtime logs
├── demo/                       — frozen demo cases (case1_easy_ohrc_ohrc/ is the confirmed one) + benchmark_results.md/.json — see demo/README.md
└── lunar_registration/         — STALE leftover directory, not the real app (see note below)
```

> **Note on `lunar_registration/`:** an earlier restructuring moved code into `lunar_registration/src/`, but that directory now contains only compiled `__pycache__` files — no actual source. It is dead and should not be treated as the app location. The real, only entrypoint is `app.py` at the repo root, run via `streamlit run app.py`. This leftover directory has not been deleted in case anything still references it; don't add new code there.

---

## Algorithms implemented

- **Preprocessing:** CLAHE contrast enhancement, histogram matching, shadow masking, resizing
- **Feature matching:** SIFT, AKAZE, LoFTR (deep, detector-free), DISK+LightGlue (deep), PC-SIFT / phase congruency (illumination-invariant)
- **Outlier rejection:** MAGSAC++ (OpenCV's `USAC_MAGSAC`) for both homography and affine estimation, with automatic fallback to classic RANSAC if the installed OpenCV build doesn't support MAGSAC++
- **Transform estimation:** adaptive homography/affine selection, phase-correlation sub-pixel refinement
- **Metrics:** fit RMSE and an independent held-out RMSE, inlier count/ratio, spatial distribution score, post-RANSAC spatial coverage + heatmap visualization, degenerate-fit detection
- **Data loading:** OHRC `.img`/XML, LRO GeoTIFF, IIRS hyperspectral (experimental)
- **Geo-assistance:** coarse pre-alignment using OHRC CSV geolocation files, footprint-overlap pre-check

---

## Test data

`data/test_set/` (see `data/test_set/README.md` for full detail) has three tiers:

- **Easy (`easy_same_acquisition/`)** — one **confirmed** overlapping pair (same acquisition, two render variants). Use this as the pipeline sanity check; it should register almost trivially.
- **Medium (`medium_adjacent_orbit/`)** — two pairs from temporally adjacent frames on the same orbit pass. Overlap is **not confirmed** — these are candidates only, without footprint metadata to verify true ground overlap. Visually screen before treating a pair as valid.
- **Hard (`hard_cross_date/`)** — four single crops from different dates/seasons. Overlap between any two is **not confirmed**; treat as candidates to visually screen for shared terrain, not verified ground truth.

`data/reference/` holds LRO NAC reference tiles (GeoTIFF, via `lro_loader.py`) for cross-sensor experiments. `data/patches/` holds additional OHRC browse-image patches for manual testing.

---

## Known limitations

- **Cross-sensor matching** (OHRC vs LRO NAC, IIRS panchromatic synthesis) is experimental and has not been benchmark-validated.
- **RMSE claims need the held-out number, not the fit number.** A fit RMSE computed on the same points used to estimate the transform is not sufficient evidence of accuracy — it can look good even for a bad fit. The pipeline also computes an independent held-out RMSE (`held_out_rmse` in `src/transform.py`); that's the number to trust.
- **LoFTR may be unavailable offline** on first run if its weight host is unreachable (see TLS note above) — the other matchers are unaffected.
- **Medium/hard test tiers have unconfirmed ground-truth overlap** — don't cite registration results on them as validated accuracy without visually confirming the pair actually shares terrain.

---

## Requirements

- Python 3.13 (developed/tested on Windows; should also work on macOS/Linux with the same commands)
- See `requirements.txt` for the full dependency list (opencv-contrib-python, numpy, scipy, matplotlib, pillow, streamlit, kornia, torch, torchvision, pyyaml, pytest, rasterio, pyproj, phasepack)
- No GPU required — CPU works for the classical methods and is usable (if slower) for the deep-learning matchers

---

## Team
SIH 2026 | PS166 | ISRO
