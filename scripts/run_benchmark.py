#!/usr/bin/env python
"""
run_benchmark.py
-----------------
CLI benchmark / evaluation runner for the OHRC image-registration pipeline
(src/pipeline.py::run_pipeline).

Runs the pipeline over the curated test set in data/test_set/ across a
configurable set of matching methods, and writes:
  - a plain-ASCII results table to the console
  - the same table + full per-run details to demo/benchmark_results.json
  - a human-readable Markdown report to demo/benchmark_results.md

IMPORTANT — overlap confirmation status (read data/test_set/README.md):
  - easy_same_acquisition/   : CONFIRMED genuine overlap (same acquisition,
                                two browse-render variants of one scene).
  - medium_adjacent_orbit/   : UNCONFIRMED. Consecutive-frame captures on the
                                same orbit pass, but there is no footprint
                                metadata to verify true ground overlap.
  - hard_cross_date/         : UNCONFIRMED. Four single images from different
                                dates with no known pairing at all — any pair
                                formed from them is a visual-similarity
                                candidate only, not a verified terrain match.

Every result row carries its tier's confirmation status verbatim so nobody
mistakes a "candidate, unconfirmed overlap" result for verified ground truth.
This matters because this tooling targets an ISRO hackathon judge audience —
see the project owner's explicit instruction not to imply verified ground
truth for medium/hard tiers anywhere in this script's output.

Usage:
    python scripts/run_benchmark.py                    # default sweep
    python scripts/run_benchmark.py --quick             # fast smoke test
    python scripts/run_benchmark.py --methods sift akaze
    python scripts/run_benchmark.py --include-hard
    python scripts/run_benchmark.py --methods loftr     # opt-in, may fail
    python scripts/run_benchmark.py --max-size 768
    python scripts/run_benchmark.py --help

Exit code is always 0 unless the script itself crashes outside a per-run
try/except (it shouldn't) — a failed registration run is recorded as a
result row, not a script failure.
"""

import argparse
import contextlib
import io
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_ROOT)

from src.console import setup_utf8_console
setup_utf8_console()

from src.preprocess import load_image
from src.pipeline import run_pipeline


DATA_DIR = os.path.join(REPO_ROOT, 'data', 'test_set')
DEMO_DIR = os.path.join(REPO_ROOT, 'demo')

ALL_METHODS = ['auto', 'sift', 'akaze', 'loftr', 'lightglue', 'pc-sift']
DEFAULT_METHODS = ['auto', 'sift', 'akaze', 'lightglue']

CONFIRMED = "CONFIRMED"
UNCONFIRMED_MEDIUM = "UNCONFIRMED (candidate, unconfirmed overlap)"
UNCONFIRMED_HARD = (
    "UNCONFIRMED (candidate, unconfirmed overlap -- "
    "visual similarity only, not verified terrain)"
)


# ── Pair definitions ────────────────────────────────────────────────

class Pair:
    """One image pair to benchmark, with its tier and confirmation status."""

    def __init__(self, name, tier, confirmation, path1, path2):
        self.name = name
        self.tier = tier
        self.confirmation = confirmation
        self.path1 = path1
        self.path2 = path2

    def exists(self):
        return os.path.exists(self.path1) and os.path.exists(self.path2)


def _p(tier_dir, filename):
    return os.path.join(DATA_DIR, tier_dir, filename)


def build_easy_pairs():
    """The one confirmed pair: same acquisition, two browse-render variants."""
    return [
        Pair(
            name="easy_20260102T1224107393",
            tier="easy_same_acquisition",
            confirmation=CONFIRMED,
            path1=_p("easy_same_acquisition", "easy_ncp_20260102T1224107393_p1.png"),
            path2=_p("easy_same_acquisition", "easy_nrp_20260102T1224107393_p1.png"),
        )
    ]


def build_medium_pairs():
    """
    Two timestamp groups on the same orbit pass, each with p1/p2 crops.
    Per the README: 'each has 2 crops (p1, p2) taken from different vertical
    positions along the strip' -- the sensible same-crop-position comparison
    is p1-of-group-A vs p1-of-group-B (and p2 vs p2), i.e. the SAME crop
    position across the two nearly-simultaneous frames. True ground overlap
    between the two frames is explicitly UNCONFIRMED per the README (no
    footprint metadata available) -- these are candidates only.
    """
    groups = [
        ("20210405T0047199117", "20210405T0047199239"),
        ("20210405T0245288072", "20210405T0245288189"),
    ]
    pairs = []
    for ts_a, ts_b in groups:
        for crop in ("p1", "p2"):
            f_a = f"medium_{ts_a}_{crop}.png"
            f_b = f"medium_{ts_b}_{crop}.png"
            pairs.append(Pair(
                name=f"medium_{ts_a}_vs_{ts_b}_{crop}",
                tier="medium_adjacent_orbit",
                confirmation=UNCONFIRMED_MEDIUM,
                path1=_p("medium_adjacent_orbit", f_a),
                path2=_p("medium_adjacent_orbit", f_b),
            ))
    return pairs


def build_hard_pairs():
    """
    All pairwise combinations of the 4 cross-date single images. No two of
    these are known to share terrain -- every resulting pair is a visual
    similarity candidate only, never treated as ground truth.
    """
    files = [
        "hard_20251227T1027178560_p1.png",
        "hard_20260130T1908101751_p1.png",
        "hard_20260331T1105235288_p1.png",
        "hard_20220713T1223352508_p1.png",
    ]
    pairs = []
    for i in range(len(files)):
        for j in range(i + 1, len(files)):
            name_i = files[i].replace("hard_", "").replace("_p1.png", "")
            name_j = files[j].replace("hard_", "").replace("_p1.png", "")
            pairs.append(Pair(
                name=f"hard_{name_i}_vs_{name_j}",
                tier="hard_cross_date",
                confirmation=UNCONFIRMED_HARD,
                path1=_p("hard_cross_date", files[i]),
                path2=_p("hard_cross_date", files[j]),
            ))
    return pairs


# ── Running one pipeline call safely ────────────────────────────────

def run_one(pair: Pair, method: str, max_size: int) -> dict:
    """
    Runs run_pipeline() for one (pair, method) combination. Never raises --
    any exception (including LoFTR's TLS/download failure) is caught and
    turned into a result row with status='error'.
    """
    row = {
        "pair": pair.name,
        "tier": pair.tier,
        "overlap_confirmation": pair.confirmation,
        "method": method,
        "max_size": max_size,
    }

    if not pair.exists():
        row["status"] = "skipped"
        row["error"] = f"missing image file(s): {pair.path1} / {pair.path2}"
        row["processing_time"] = 0.0
        return row

    try:
        img1 = load_image(pair.path1)
        img2 = load_image(pair.path2)
    except Exception as e:
        row["status"] = "error"
        row["error"] = f"failed to load images: {e}"
        row["processing_time"] = 0.0
        return row

    t0 = time.time()
    log_buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(log_buffer):
            result = run_pipeline(img1, img2, method=method, max_size=max_size)
        elapsed = time.time() - t0

        row["processing_time"] = elapsed
        row["success"] = bool(result.get("success"))

        if not result.get("success"):
            row["status"] = "failed"
            row["failure_reason"] = result.get("failure_reason", "unknown")
            row["confidence"] = result.get("confidence", "failed")
            return row

        metrics = result.get("metrics", {}) or {}
        holdout = metrics.get("held_out_validation") or {}
        coverage = metrics.get("coverage") or {}

        row["status"] = "ok"
        row["method_used"] = result.get("method")
        row["confidence"] = result.get("confidence")
        row["degenerate_fit"] = result.get("degenerate_fit")
        row["transform_type"] = result.get("transform_type")
        row["rmse_fit"] = metrics.get("rmse")
        row["rmse_x_fit"] = metrics.get("rmse_x")
        row["rmse_y_fit"] = metrics.get("rmse_y")
        row["rmse_holdout"] = holdout.get("rmse") if holdout.get("available") else None
        row["holdout_available"] = bool(holdout.get("available"))
        row["holdout_n_fit"] = holdout.get("n_fit")
        row["holdout_n_holdout"] = holdout.get("n_holdout")
        row["inlier_count"] = metrics.get("inlier_count")
        row["total_matches"] = metrics.get("total_matches")
        row["inlier_ratio"] = metrics.get("inlier_ratio")
        row["coverage_fraction"] = coverage.get("coverage_fraction")
        row["spatial_score"] = metrics.get("spatial_score")
        return row

    except Exception as e:
        elapsed = time.time() - t0
        row["status"] = "error"
        row["processing_time"] = elapsed
        row["error"] = f"{type(e).__name__}: {e}"
        row["log_tail"] = "\n".join(log_buffer.getvalue().splitlines()[-15:])
        return row


# ── Console table rendering (plain ASCII only) ──────────────────────

def _fmt(v, spec=""):
    if v is None:
        return "--"
    if isinstance(v, float):
        try:
            return format(v, spec or ".3f")
        except (ValueError, TypeError):
            return str(v)
    return str(v)


def render_console_table(rows):
    headers = [
        "PAIR", "TIER", "CONFIRM", "METHOD", "STATUS", "CONF",
        "DEGEN", "RMSE-FIT", "RMSE-HOLD", "INLIERS", "RATIO",
        "COVER", "SPATIAL", "TIME(s)"
    ]

    def confirm_short(c):
        return "confirmed" if c == CONFIRMED else "UNCONFIRMED"

    def status_short(row):
        st = row.get("status")
        if st == "ok":
            return "[OK]"
        if st == "failed":
            return "[FAIL]"
        if st == "skipped":
            return "[SKIP]"
        return "[ERROR]"

    data_rows = []
    for r in rows:
        data_rows.append([
            r.get("pair", "--"),
            r.get("tier", "--"),
            confirm_short(r.get("overlap_confirmation", "")),
            r.get("method", "--"),
            status_short(r),
            _fmt(r.get("confidence")) if r.get("status") == "ok" else "--",
            "YES" if r.get("degenerate_fit") else ("no" if r.get("status") == "ok" else "--"),
            _fmt(r.get("rmse_fit"), ".4f"),
            _fmt(r.get("rmse_holdout"), ".4f") if r.get("holdout_available") else "unavail",
            _fmt(r.get("inlier_count"), "d") if r.get("inlier_count") is not None else "--",
            _fmt(r.get("inlier_ratio"), ".2%") if r.get("inlier_ratio") is not None else "--",
            _fmt(r.get("coverage_fraction"), ".1%") if r.get("coverage_fraction") is not None else "--",
            _fmt(r.get("spatial_score"), ".3f"),
            _fmt(r.get("processing_time"), ".2f"),
        ])

    col_widths = [len(h) for h in headers]
    for row in data_rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    def fmt_row(cells):
        return " | ".join(str(c).ljust(col_widths[i]) for i, c in enumerate(cells))

    lines = []
    lines.append(fmt_row(headers))
    lines.append("-+-".join("-" * w for w in col_widths))
    for row in data_rows:
        lines.append(fmt_row(row))
    return "\n".join(lines)


def render_markdown_table(rows):
    headers = [
        "Pair", "Tier", "Overlap status", "Method", "Status", "Confidence",
        "Degenerate", "RMSE fit (px)", "RMSE held-out (px)", "Inliers",
        "Inlier ratio", "Coverage", "Spatial score", "Time (s)"
    ]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]

    for r in rows:
        status = r.get("status", "error")
        confirm = "confirmed" if r.get("overlap_confirmation") == CONFIRMED else "UNCONFIRMED (candidate)"
        cells = [
            r.get("pair", "--"),
            r.get("tier", "--"),
            confirm,
            r.get("method", "--"),
            status,
            _fmt(r.get("confidence")) if status == "ok" else "--",
            ("yes" if r.get("degenerate_fit") else "no") if status == "ok" else "--",
            _fmt(r.get("rmse_fit"), ".4f") if status == "ok" else "--",
            (_fmt(r.get("rmse_holdout"), ".4f") if r.get("holdout_available") else "unavailable") if status == "ok" else "--",
            _fmt(r.get("inlier_count")) if status == "ok" else "--",
            _fmt(r.get("inlier_ratio"), ".2%") if status == "ok" and r.get("inlier_ratio") is not None else "--",
            _fmt(r.get("coverage_fraction"), ".1%") if status == "ok" and r.get("coverage_fraction") is not None else "--",
            _fmt(r.get("spatial_score"), ".3f") if status == "ok" else "--",
            _fmt(r.get("processing_time"), ".2f"),
        ]
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the OHRC registration pipeline (src/pipeline.run_pipeline) "
            "across the curated test set in data/test_set/. Only the easy tier has "
            "a confirmed ground-truth overlap -- medium/hard results are always "
            "labelled as unconfirmed candidates, never presented as verified."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/run_benchmark.py\n"
            "  python scripts/run_benchmark.py --quick\n"
            "  python scripts/run_benchmark.py --methods sift akaze\n"
            "  python scripts/run_benchmark.py --include-hard\n"
            "  python scripts/run_benchmark.py --methods auto sift akaze lightglue loftr\n"
        ),
    )
    parser.add_argument(
        "--methods", nargs="+", default=DEFAULT_METHODS,
        choices=ALL_METHODS,
        help=(
            "Matching methods to run. Default excludes 'loftr' because its "
            "pretrained weights currently fail to download in this environment "
            "(broken TLS chain at cmp.felk.cvut.cz) -- pass it explicitly to try "
            "it anyway; a failure is caught and reported per-run, not fatal."
        ),
    )
    parser.add_argument(
        "--include-hard", action="store_true",
        help=(
            "Also run all pairwise combinations of the hard_cross_date/ images. "
            "These have NO confirmed pairing at all -- results are always labelled "
            "'candidate, unconfirmed overlap -- visual similarity only'."
        ),
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Fast smoke test: only the confirmed easy pair, method=auto. Ignores --methods/--include-hard.",
    )
    parser.add_argument(
        "--max-size", type=int, default=1024,
        help="Max image dimension passed to run_pipeline (matches config.yaml default).",
    )
    parser.add_argument(
        "--output-json", default=os.path.join(DEMO_DIR, "benchmark_results.json"),
        help="Path to write full JSON results.",
    )
    parser.add_argument(
        "--output-md", default=os.path.join(DEMO_DIR, "benchmark_results.md"),
        help="Path to write the human-readable Markdown report.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    print("=" * 78)
    print("  OHRC REGISTRATION PIPELINE BENCHMARK")
    print("=" * 78)

    if args.quick:
        pairs = build_easy_pairs()
        methods = ["auto"]
        print("Mode: --quick (easy pair only, method=auto)")
    else:
        pairs = build_easy_pairs() + build_medium_pairs()
        if args.include_hard:
            pairs += build_hard_pairs()
        methods = args.methods
        print(f"Mode: full sweep -- {len(pairs)} pair(s) x {len(methods)} method(s) "
              f"= {len(pairs) * len(methods)} run(s)")
        if args.include_hard:
            print("Hard tier INCLUDED -- all results from it are unconfirmed candidates.")
        else:
            print("Hard tier excluded (pass --include-hard to add it).")

    print(f"Methods: {methods}")
    print(f"Max size: {args.max_size}")
    print()

    missing = [p for p in pairs if not p.exists()]
    if missing:
        print("WARNING: the following pairs reference missing files and will be "
              "recorded as skipped:")
        for p in missing:
            print(f"  - {p.name}: {p.path1} / {p.path2}")
        print()

    rows = []
    total_runs = len(pairs) * len(methods)
    run_idx = 0
    overall_start = time.time()

    for pair in pairs:
        for method in methods:
            run_idx += 1
            print(f"[{run_idx}/{total_runs}] {pair.name} ({pair.tier}) "
                  f"method={method} ... ", end="", flush=True)
            t0 = time.time()
            try:
                row = run_one(pair, method, args.max_size)
            except Exception as e:
                # Absolute last-resort catch -- run_one() already catches its
                # own exceptions, but nothing here may ever crash the sweep.
                row = {
                    "pair": pair.name, "tier": pair.tier,
                    "overlap_confirmation": pair.confirmation,
                    "method": method, "status": "error",
                    "error": f"unexpected top-level failure: {e}",
                    "processing_time": time.time() - t0,
                }
                traceback.print_exc()
            rows.append(row)

            status = row.get("status", "error")
            elapsed = row.get("processing_time", 0.0)
            if status == "ok":
                print(f"[OK] conf={row.get('confidence')} "
                      f"rmse_fit={_fmt(row.get('rmse_fit'), '.4f')} "
                      f"inliers={row.get('inlier_count')} "
                      f"({elapsed:.1f}s)")
            elif status == "failed":
                print(f"[FAIL] {row.get('failure_reason', '')[:70]} ({elapsed:.1f}s)")
            elif status == "skipped":
                print(f"[SKIP] {row.get('error', '')}")
            else:
                print(f"[ERROR] {row.get('error', '')[:100]} ({elapsed:.1f}s)")

    total_elapsed = time.time() - overall_start
    print()
    print("=" * 78)
    print("  RESULTS TABLE")
    print("=" * 78)
    print(render_console_table(rows))
    print()
    print(f"Total wall time: {total_elapsed:.1f}s for {total_runs} run(s)")

    n_ok = sum(1 for r in rows if r.get("status") == "ok")
    n_failed = sum(1 for r in rows if r.get("status") == "failed")
    n_error = sum(1 for r in rows if r.get("status") == "error")
    n_skipped = sum(1 for r in rows if r.get("status") == "skipped")
    print(f"Summary: {n_ok} ok, {n_failed} failed (graceful), "
          f"{n_error} error (exception), {n_skipped} skipped (missing file)")
    print()
    print("REMINDER: only the easy_same_acquisition pair has a CONFIRMED overlap.")
    print("All medium/hard tier results above are candidates with UNCONFIRMED")
    print("overlap -- see the 'overlap_confirmation' field / CONFIRM column.")

    # ── Write outputs ────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_md)), exist_ok=True)

    json_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "args": {
            "methods": methods,
            "include_hard": args.include_hard,
            "quick": args.quick,
            "max_size": args.max_size,
        },
        "summary": {
            "total_runs": total_runs,
            "ok": n_ok,
            "failed": n_failed,
            "error": n_error,
            "skipped": n_skipped,
            "total_wall_time_s": total_elapsed,
        },
        "results": rows,
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2, default=str)
    print(f"\nWrote JSON results to: {args.output_json}")

    md_lines = [
        "# Benchmark Results",
        "",
        f"Generated: {json_payload['generated_at_utc']}",
        "",
        "Overlap confirmation status is authoritative per data/test_set/README.md:",
        "only `easy_same_acquisition` is a CONFIRMED genuine overlap. All",
        "`medium_adjacent_orbit` and `hard_cross_date` rows below are candidates",
        "with UNCONFIRMED overlap and must not be read as verified ground truth.",
        "",
        f"Methods: {methods}  |  Max size: {args.max_size}  |  "
        f"Total wall time: {total_elapsed:.1f}s",
        "",
        f"Summary: {n_ok} ok, {n_failed} failed (graceful), "
        f"{n_error} error (exception), {n_skipped} skipped (missing file)",
        "",
        render_markdown_table(rows),
        "",
    ]
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"Wrote Markdown report to: {args.output_md}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
