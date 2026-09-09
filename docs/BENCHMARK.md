# Benchmark / Evaluation Suite

This document explains how to run `scripts/run_benchmark.py`, what the three
test-set tiers mean (and, critically, what is and isn't verified about them),
and shows results from an actual run against this repository's code.

## Running it

```bash
# Fast smoke test: the one confirmed pair, method=auto. Finishes in well
# under a minute (excluding any first-time model download).
python scripts/run_benchmark.py --quick

# Default sweep: easy pair + all medium-tier pairs, methods = auto, sift,
# akaze, lightglue. Takes several minutes (see "Actual results" below).
python scripts/run_benchmark.py

# Add the hard cross-date tier (all pairwise combinations, clearly labelled
# unconfirmed):
python scripts/run_benchmark.py --include-hard

# Restrict to specific methods:
python scripts/run_benchmark.py --methods sift akaze

# Try LoFTR explicitly (its pretrained weights currently fail to download
# in this environment -- see "Known issue: LoFTR" below -- so it's excluded
# from the default method list, but you can opt in):
python scripts/run_benchmark.py --methods auto sift akaze lightglue loftr

# Full option list:
python scripts/run_benchmark.py --help
```

Every run writes three things:
- a plain-ASCII results table printed to the console,
- `demo/benchmark_results.json` — full per-run details (all metrics, every
  field described below),
- `demo/benchmark_results.md` — the same table in Markdown.

Both output paths can be overridden with `--output-json` / `--output-md`.

A single pair/method failure (a bad match, a model that won't load, a
degenerate fit) is caught and recorded as a row with `status: failed` or
`status: error` — it never aborts the rest of the sweep.

## Test-set tiers and what is actually confirmed

The full picture lives in `data/test_set/README.md`; the short version, which
this benchmark script enforces in every row it emits (`overlap_confirmation`
field / `CONFIRM` column):

| Tier | Folder | Overlap status | What it actually is |
|---|---|---|---|
| Easy | `easy_same_acquisition/` | **CONFIRMED** | One genuine pair: two browse-render variants (`ncp` vs `nrp`) of the *same acquisition* — same scene, same illumination, different rendering stretch. This is the only pair in the repo with a verified genuine overlap. |
| Medium | `medium_adjacent_orbit/` | **UNCONFIRMED** | Two timestamp groups, each captured moments apart on the same orbit pass, each with `p1`/`p2` crops at the same vertical position. No footprint/lat-lon metadata exists to confirm the two frames in a group actually image the same ground — only that they are temporally adjacent in the same pass. The benchmark compares `p1`-vs-`p1` and `p2`-vs-`p2` across each group's two timestamps. |
| Hard | `hard_cross_date/` | **UNCONFIRMED** | Four single images from different dates (2021, 2022, and Dec 2025 - Mar 2026) with *no known pairing at all*. `--include-hard` runs every pairwise combination (6 pairs) and labels every result "candidate, unconfirmed overlap — visual similarity only, not verified terrain." |

**Do not read the medium/hard numbers below as proof the pipeline correctly
registered two genuinely overlapping images.** They show how the pipeline
*behaves* on those pairs (including correctly refusing/flagging pairs that
don't overlap) — not verified registration accuracy. Only the easy-tier row
is backed by a confirmed ground truth.

## Known issue: LoFTR

`method='loftr'` downloads its pretrained weights from
`cmp.felk.cvut.cz` on first use. In this environment that host's TLS
certificate chain is broken, so the download fails and every LoFTR run
returns a graceful `status: failed` result (or, if the failure happens
outside `run_pipeline`'s own try/except, is caught by the benchmark script
itself and recorded as `status: error`) — it never crashes the sweep. LoFTR
is excluded from the default `--methods` list for this reason; pass
`--methods ... loftr` explicitly if you want to see the current failure (or
if the certificate issue is fixed in your environment).

## Known quirk: `--methods akaze` runs the full ensemble

`src/pipeline.py`'s method router treats `method='akaze'` the same as
`method='auto'` — both take the "run every classical algorithm and score
the best one" branch (`if method in ('auto', 'akaze')`). This means an
explicit `akaze` run in this benchmark takes about as long as `auto` (it
actually runs AKAZE + SIFT + LightGlue + PC-SIFT internally and picks the
winner) and, on this test set, both settled on **SIFT** as the picked
result rather than AKAZE. That is existing behaviour in `src/pipeline.py`,
which is out of scope for this benchmark workstream to change — noted here
so the timing and "method used" columns aren't surprising.

## Actual results

### Default sweep (easy + medium, methods = auto / sift / akaze / lightglue)

Run: `python scripts/run_benchmark.py` — **20/20 runs succeeded**, 351.2s
total wall time (~5.9 minutes), on this machine with LightGlue weights
already cached locally.

| Pair | Tier | Overlap | Method | Confidence | Degenerate | RMSE fit (px) | RMSE held-out (px) | Inliers | Inlier ratio | Coverage | Spatial score | Time (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| easy_20260102T1224107393 | easy | confirmed | auto | high | no | 0.0194 | 0.0195 | 317 | 100.0% | 100.0% | 0.999 | 26.49 |
| easy_20260102T1224107393 | easy | confirmed | sift | high | no | 0.0194 | 0.0195 | 317 | 100.0% | 100.0% | 0.999 | 0.75 |
| easy_20260102T1224107393 | easy | confirmed | akaze | high | no | 0.0194 | 0.0195 | 317 | 100.0% | 100.0% | 0.999 | 24.23 |
| easy_20260102T1224107393 | easy | confirmed | lightglue | high | no | 0.1139 | 0.1165 | 308 | 100.0% | 96.9% | 0.992 | 16.83 |
| medium_...117_vs_239_p1 | medium | UNCONFIRMED | auto | high | no | 0.0000 | 0.0000 | 320 | 100.0% | 100.0% | 1.000 | 32.19 |
| medium_...117_vs_239_p1 | medium | UNCONFIRMED | sift | high | no | 0.0000 | 0.0000 | 320 | 100.0% | 100.0% | 1.000 | 0.74 |
| medium_...117_vs_239_p1 | medium | UNCONFIRMED | akaze | high | no | 0.0000 | 0.0000 | 320 | 100.0% | 100.0% | 1.000 | 25.16 |
| medium_...117_vs_239_p1 | medium | UNCONFIRMED | lightglue | high | no | 0.0000 | 0.0000 | 316 | 100.0% | 100.0% | 0.998 | 12.63 |
| medium_...117_vs_239_p2 | medium | UNCONFIRMED | auto | high | no | 0.0000 | 0.0000 | 320 | 100.0% | 100.0% | 1.000 | 25.23 |
| medium_...117_vs_239_p2 | medium | UNCONFIRMED | sift | high | no | 0.0000 | 0.0000 | 320 | 100.0% | 100.0% | 1.000 | 0.69 |
| medium_...117_vs_239_p2 | medium | UNCONFIRMED | akaze | high | no | 0.0000 | 0.0000 | 320 | 100.0% | 100.0% | 1.000 | 25.21 |
| medium_...117_vs_239_p2 | medium | UNCONFIRMED | lightglue | high | no | 0.0000 | 0.0000 | 307 | 100.0% | 100.0% | 0.996 | 12.07 |
| medium_...072_vs_189_p1 | medium | UNCONFIRMED | auto | high | no | 0.0000 | 0.0000 | 320 | 100.0% | 100.0% | 1.000 | 34.61 |
| medium_...072_vs_189_p1 | medium | UNCONFIRMED | sift | high | no | 0.0000 | 0.0000 | 320 | 100.0% | 100.0% | 1.000 | 0.84 |
| medium_...072_vs_189_p1 | medium | UNCONFIRMED | akaze | high | no | 0.0000 | 0.0000 | 320 | 100.0% | 100.0% | 1.000 | 33.82 |
| medium_...072_vs_189_p1 | medium | UNCONFIRMED | lightglue | high | no | 0.0000 | 0.0000 | 308 | 100.0% | 96.9% | 0.992 | 13.86 |
| medium_...072_vs_189_p2 | medium | UNCONFIRMED | auto | high | no | 0.0000 | 0.0000 | 320 | 100.0% | 100.0% | 1.000 | 24.81 |
| medium_...072_vs_189_p2 | medium | UNCONFIRMED | sift | high | no | 0.0000 | 0.0000 | 320 | 100.0% | 100.0% | 1.000 | 0.74 |
| medium_...072_vs_189_p2 | medium | UNCONFIRMED | akaze | high | no | 0.0000 | 0.0000 | 320 | 100.0% | 100.0% | 1.000 | 26.52 |
| medium_...072_vs_189_p2 | medium | UNCONFIRMED | lightglue | high | no | 0.0000 | 0.0000 | 303 | 100.0% | 95.3% | 0.988 | 13.20 |

Full machine-readable version: `demo/benchmark_results.json`. Same table as
Markdown: `demo/benchmark_results.md`.

**Reading these numbers honestly:** the easy-tier row (RMSE ~0.02px,
high confidence) is the one result backed by a confirmed genuine overlap —
that's the real "pipeline sanity check" number. The medium-tier rows show
essentially the same near-perfect RMSE (0.0000px, 100% inlier ratio) as the
easy tier. That is *not* evidence the pipeline nails hard registration
cases — it's a strong signal that these particular `p1`-vs-`p1` /
`p2`-vs-`p2` same-crop-position frame pairs are visually near-identical
(consistent with being seconds apart on the same pass), i.e. an easier case
than the "adjacent orbit" label suggests, not a validated difficult one.
Per `data/test_set/README.md`, there is still no footprint metadata to
confirm true ground overlap for this tier, so these rows are reported as
unconfirmed candidates regardless of how clean the numbers look.

### Hard tier (cross-date, `--include-hard`, method=sift only for speed)

Run: `python scripts/run_benchmark.py --include-hard --methods sift` — all
6 pairwise combinations of the 4 hard-tier images, 11 total rows (easy +
medium + hard), 7.5s wall time.

| Pair | Overlap | Confidence | Degenerate | Inliers | Inlier ratio | Result |
|---|---|---|---|---|---|---|
| hard_20251227..._vs_20260130... | UNCONFIRMED | failed | **YES** | 4 | 44.4% | "succeeded" but flagged degenerate — RMSE not meaningful |
| hard_20251227..._vs_20260331... | UNCONFIRMED | -- | -- | -- | -- | graceful failure: only 3 inliers after RANSAC |
| hard_20251227..._vs_20220713... | UNCONFIRMED | -- | -- | -- | -- | graceful failure: only 3 inliers after RANSAC |
| hard_20260130..._vs_20260331... | UNCONFIRMED | failed | **YES** | 4 | 20.0% | "succeeded" but flagged degenerate — RMSE not meaningful |
| hard_20260130..._vs_20220713... | UNCONFIRMED | -- | -- | -- | -- | graceful failure: only 3 inliers after RANSAC |
| hard_20260331..._vs_20220713... | UNCONFIRMED | -- | -- | -- | -- | graceful failure: only 3 inliers after RANSAC |

This is exactly the behaviour you want from an honest pipeline on images
with no known shared terrain: 4 of 6 pairs fail outright (too few RANSAC
inliers to fit any transform), and the 2 that technically "succeed" are
correctly caught by `degenerate_fit=True` / `confidence='failed'` — i.e.
the pipeline itself is telling you not to trust those results, exactly
because 4 inliers on an 8-DOF homography is a mathematically forced exact
fit, not evidence of a real match. No hard-tier row should ever be read as
a validated registration.

## Result fields (JSON)

Each entry in `demo/benchmark_results.json["results"]` has:

- `pair`, `tier`, `overlap_confirmation` — which pair, its tier, and the
  tier's confirmation status (verbatim, so it can never be misquoted).
- `method`, `method_used` — the method requested vs. the one the pipeline
  actually picked (relevant for `auto`/`akaze`, see "Known quirk" above).
- `status` — `ok` / `failed` (graceful pipeline failure) / `error`
  (exception caught by the benchmark script itself) / `skipped` (missing
  image file).
- `confidence`, `degenerate_fit`, `transform_type`.
- `rmse_fit`, `rmse_x_fit`, `rmse_y_fit` — same-data RMSE (see
  `assess_subpixel_claim` in `src/metrics.py` for why this alone can't
  support a "sub-pixel achieved" claim).
- `rmse_holdout`, `holdout_available`, `holdout_n_fit`, `holdout_n_holdout`
  — the independent, out-of-sample RMSE from `held_out_rmse()`.
- `inlier_count`, `total_matches`, `inlier_ratio`, `coverage_fraction`,
  `spatial_score`.
- `processing_time` (seconds for that single pipeline call).
- `failure_reason` / `error` — present only on non-`ok` rows.
