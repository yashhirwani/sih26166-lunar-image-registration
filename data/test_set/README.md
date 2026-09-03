# Curated Test Set — from `browse_previews_all.zip`

## What I found in your original zip
- 57 PNG files, but only **26 genuinely unique acquisitions**.
- `d18` and `d32` files are **byte-for-byte identical** in every case checked — these are not scale variants, just a duplicate naming suffix. I kept only one copy (`d18`).
- `ncp` vs `nrp` for the same timestamp are **not identical but very close** (mean pixel diff ~2.6/255) — these look like two slightly different browse-rendering stretches of the **same acquisition**, not different times or locations. Good for an easy sanity-check pair, not a real registration challenge.
- All images are large narrow strips (1200 × ~9000–10000 px), not small patches — I cropped 1024×1024 patches (matching your `max_size=1024` config) from each so they're ready to feed directly into the pipeline.

## Folder structure

```
test_set/
├── easy_same_acquisition/     ← 1 pair: ncp vs nrp, same timestamp
├── medium_adjacent_orbit/     ← 2 pairs: consecutive-frame captures, same orbit pass
├── hard_cross_date/           ← 4 single images from different dates/seasons
└── README.md
```

### Tier 1 — Easy (`easy_same_acquisition/`)
`easy_ncp_20260102T1224107393_p1.png` ↔ `easy_nrp_20260102T1224107393_p1.png`
Same acquisition, two render variants. Near-identical illumination/geometry — use this as your **pipeline sanity check** (should register almost trivially, RMSE very low). Good for confirming Steps 0–11 run cleanly before testing anything harder.

### Tier 2 — Medium (`medium_adjacent_orbit/`)
Two pairs, each from consecutive-frame captures ~seconds apart on the same orbit pass:
- `medium_20210405T0047199117_*` ↔ `medium_20210405T0047199239_*`
- `medium_20210405T0245288072_*` ↔ `medium_20210405T0245288189_*`

Each has 2 crops (`p1`, `p2`) taken from different vertical positions along the strip. **Caveat: true ground overlap between these consecutive frames is not confirmed** — I don't have footprint/lat-lon metadata to verify it, only that they're temporally adjacent in the same pass. Visually inspect `p1`/`p2` crops from both timestamps before assuming a given pair overlaps; try a few combinations if the first doesn't share terrain.

### Tier 3 — Hard (`hard_cross_date/`)
Four single crops from clearly different dates/seasons (2020s spread across 2021, 2022, and Dec 2025–Mar 2026):
- `hard_20251227T1027178560_p1.png`
- `hard_20260130T1908101751_p1.png`
- `hard_20260331T1105235288_p1.png`
- `hard_20220713T1223352508_p1.png`

**Important caveat:** since OHRC targets narrow, specific sites per orbit, I cannot confirm any two of these show the *same* lunar location without footprint metadata (lat/lon/target ID from the accompanying XML/label files, which weren't in this zip). Treat these as candidates to **visually screen for shared terrain** (matching crater shapes/patterns) before using as a genuine cross-date illumination-variant test pair — don't assume any two are a valid pair by default.

## Recommended next step
If you have the corresponding `.xml`/label files from Pradan (they usually ship alongside the browse PNGs), share them — footprint coordinates would let me confirm real overlapping pairs programmatically instead of by visual guesswork, which would upgrade Tier 2/3 from "candidates" to "confirmed test pairs."
