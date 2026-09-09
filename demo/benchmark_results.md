# Benchmark Results

Generated: 2026-09-09T09:45:11.136184+00:00

Overlap confirmation status is authoritative per data/test_set/README.md:
only `easy_same_acquisition` is a CONFIRMED genuine overlap. All
`medium_adjacent_orbit` and `hard_cross_date` rows below are candidates
with UNCONFIRMED overlap and must not be read as verified ground truth.

Methods: ['auto', 'sift', 'akaze', 'lightglue']  |  Max size: 1024  |  Total wall time: 351.2s

Summary: 20 ok, 0 failed (graceful), 0 error (exception), 0 skipped (missing file)

| Pair | Tier | Overlap status | Method | Status | Confidence | Degenerate | RMSE fit (px) | RMSE held-out (px) | Inliers | Inlier ratio | Coverage | Spatial score | Time (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| easy_20260102T1224107393 | easy_same_acquisition | confirmed | auto | ok | high | no | 0.0194 | 0.0195 | 317 | 100.00% | 100.0% | 0.999 | 26.49 |
| easy_20260102T1224107393 | easy_same_acquisition | confirmed | sift | ok | high | no | 0.0194 | 0.0195 | 317 | 100.00% | 100.0% | 0.999 | 0.75 |
| easy_20260102T1224107393 | easy_same_acquisition | confirmed | akaze | ok | high | no | 0.0194 | 0.0195 | 317 | 100.00% | 100.0% | 0.999 | 24.23 |
| easy_20260102T1224107393 | easy_same_acquisition | confirmed | lightglue | ok | high | no | 0.1139 | 0.1165 | 308 | 100.00% | 96.9% | 0.992 | 16.83 |
| medium_20210405T0047199117_vs_20210405T0047199239_p1 | medium_adjacent_orbit | UNCONFIRMED (candidate) | auto | ok | high | no | 0.0000 | 0.0000 | 320 | 100.00% | 100.0% | 1.000 | 32.19 |
| medium_20210405T0047199117_vs_20210405T0047199239_p1 | medium_adjacent_orbit | UNCONFIRMED (candidate) | sift | ok | high | no | 0.0000 | 0.0000 | 320 | 100.00% | 100.0% | 1.000 | 0.74 |
| medium_20210405T0047199117_vs_20210405T0047199239_p1 | medium_adjacent_orbit | UNCONFIRMED (candidate) | akaze | ok | high | no | 0.0000 | 0.0000 | 320 | 100.00% | 100.0% | 1.000 | 25.16 |
| medium_20210405T0047199117_vs_20210405T0047199239_p1 | medium_adjacent_orbit | UNCONFIRMED (candidate) | lightglue | ok | high | no | 0.0000 | 0.0000 | 316 | 100.00% | 100.0% | 0.998 | 12.63 |
| medium_20210405T0047199117_vs_20210405T0047199239_p2 | medium_adjacent_orbit | UNCONFIRMED (candidate) | auto | ok | high | no | 0.0000 | 0.0000 | 320 | 100.00% | 100.0% | 1.000 | 25.23 |
| medium_20210405T0047199117_vs_20210405T0047199239_p2 | medium_adjacent_orbit | UNCONFIRMED (candidate) | sift | ok | high | no | 0.0000 | 0.0000 | 320 | 100.00% | 100.0% | 1.000 | 0.69 |
| medium_20210405T0047199117_vs_20210405T0047199239_p2 | medium_adjacent_orbit | UNCONFIRMED (candidate) | akaze | ok | high | no | 0.0000 | 0.0000 | 320 | 100.00% | 100.0% | 1.000 | 25.21 |
| medium_20210405T0047199117_vs_20210405T0047199239_p2 | medium_adjacent_orbit | UNCONFIRMED (candidate) | lightglue | ok | high | no | 0.0000 | 0.0000 | 307 | 100.00% | 100.0% | 0.996 | 12.07 |
| medium_20210405T0245288072_vs_20210405T0245288189_p1 | medium_adjacent_orbit | UNCONFIRMED (candidate) | auto | ok | high | no | 0.0000 | 0.0000 | 320 | 100.00% | 100.0% | 1.000 | 34.61 |
| medium_20210405T0245288072_vs_20210405T0245288189_p1 | medium_adjacent_orbit | UNCONFIRMED (candidate) | sift | ok | high | no | 0.0000 | 0.0000 | 320 | 100.00% | 100.0% | 1.000 | 0.84 |
| medium_20210405T0245288072_vs_20210405T0245288189_p1 | medium_adjacent_orbit | UNCONFIRMED (candidate) | akaze | ok | high | no | 0.0000 | 0.0000 | 320 | 100.00% | 100.0% | 1.000 | 33.82 |
| medium_20210405T0245288072_vs_20210405T0245288189_p1 | medium_adjacent_orbit | UNCONFIRMED (candidate) | lightglue | ok | high | no | 0.0000 | 0.0000 | 308 | 100.00% | 96.9% | 0.992 | 13.86 |
| medium_20210405T0245288072_vs_20210405T0245288189_p2 | medium_adjacent_orbit | UNCONFIRMED (candidate) | auto | ok | high | no | 0.0000 | 0.0000 | 320 | 100.00% | 100.0% | 1.000 | 24.81 |
| medium_20210405T0245288072_vs_20210405T0245288189_p2 | medium_adjacent_orbit | UNCONFIRMED (candidate) | sift | ok | high | no | 0.0000 | 0.0000 | 320 | 100.00% | 100.0% | 1.000 | 0.74 |
| medium_20210405T0245288072_vs_20210405T0245288189_p2 | medium_adjacent_orbit | UNCONFIRMED (candidate) | akaze | ok | high | no | 0.0000 | 0.0000 | 320 | 100.00% | 100.0% | 1.000 | 26.52 |
| medium_20210405T0245288072_vs_20210405T0245288189_p2 | medium_adjacent_orbit | UNCONFIRMED (candidate) | lightglue | ok | high | no | 0.0000 | 0.0000 | 303 | 100.00% | 95.3% | 0.988 | 13.20 |
