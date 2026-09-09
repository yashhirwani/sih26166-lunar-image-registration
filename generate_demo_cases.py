"""
generate_demo_cases.py
-----------------------
Generates frozen demo artifacts under demo/ for the project's real,
currently-available test pairs, clearly labelling confirmation status.
See demo/README.md for how to interpret each case.

Run from the project root:
    python generate_demo_cases.py
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.console import setup_utf8_console
setup_utf8_console()

import cv2
from src.preprocess import load_image
from src.pipeline import run_pipeline, register_cross_source
from src.lro_loader import load_geotiff_reference

PROJECT = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.join(PROJECT, "demo")


def save_case(case_dir, result, label, confirmed, notes):
    os.makedirs(case_dir, exist_ok=True)
    meta = {
        "label": label,
        "overlap_confirmed": confirmed,
        "notes": notes,
        "success": result.get("success", True),
    }
    if not result.get("success", True):
        meta["failure_reason"] = result.get("failure_reason")
        with open(os.path.join(case_dir, "result.json"), "w") as f:
            json.dump(meta, f, indent=2, default=str)
        print(f"  [FAILED] {label}: {result.get('failure_reason')}")
        return

    metrics = result["metrics"]
    holdout = metrics.get("held_out_validation", {})
    meta.update({
        "method": result.get("method"),
        "confidence": result.get("confidence"),
        "degenerate_fit": result.get("degenerate_fit"),
        "transform_type": result.get("transform_type"),
        "outlier_method": result.get("outlier_method"),
        "rmse_fit_px": metrics.get("rmse"),
        "rmse_held_out_px": holdout.get("rmse") if holdout.get("available") else None,
        "held_out_available": holdout.get("available"),
        "inlier_count": metrics.get("inlier_count"),
        "inlier_ratio": metrics.get("inlier_ratio"),
        "spatial_score": metrics.get("spatial_score"),
        "coverage_fraction": metrics.get("coverage", {}).get("coverage_fraction"),
        "processing_time_s": metrics.get("processing_time"),
    })
    with open(os.path.join(case_dir, "result.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    with open(os.path.join(case_dir, "metrics_report.txt"), "w") as f:
        f.write(result.get("metrics_report", ""))

    for fname, key in [
        ("registered_image.png", "registered_image_refined"),
        ("checkerboard.png", "checkerboard"),
        ("side_by_side.png", "side_by_side"),
        ("difference_image.png", "difference_image"),
        ("match_visualization.png", "match_visualization"),
        ("spatial_heatmap.png", "spatial_heatmap"),
    ]:
        if key in result and result[key] is not None:
            cv2.imwrite(os.path.join(case_dir, fname), result[key])

    print(f"  [OK] {label}: method={result.get('method')} confidence={result.get('confidence')} "
          f"rmse_fit={metrics.get('rmse'):.4f} held_out={holdout.get('rmse') if holdout.get('available') else 'n/a'}")


def main():
    print("=" * 60)
    print("Generating demo/ artifacts from real bundled test data")
    print("=" * 60)

    # ── Case 1: Easy OHRC<->OHRC — CONFIRMED overlap ──────────────
    print("\n[1/3] Easy OHRC<->OHRC (confirmed overlap)...")
    base = os.path.join(PROJECT, "data", "test_set", "easy_same_acquisition")
    img1 = load_image(os.path.join(base, "easy_ncp_20260102T1224107393_p1.png"))
    img2 = load_image(os.path.join(base, "easy_nrp_20260102T1224107393_p1.png"))
    result = run_pipeline(img1, img2, method="auto")
    save_case(
        os.path.join(DEMO, "case1_easy_ohrc_ohrc"), result,
        "Easy: OHRC vs OHRC, same acquisition",
        confirmed=True,
        notes="Two browse-render variants of the SAME OHRC acquisition (near-identical "
              "illumination/geometry). This is the only pair in the repo with a CONFIRMED "
              "genuine ground overlap. See data/test_set/README.md.",
    )

    # ── Case 2: "Illumination-varied" candidate — UNCONFIRMED overlap ──
    print("\n[2/3] Candidate illumination/orbit-varied OHRC<->OHRC (UNCONFIRMED overlap)...")
    base2 = os.path.join(PROJECT, "data", "test_set", "medium_adjacent_orbit")
    img3 = load_image(os.path.join(base2, "medium_20210405T0047199117_p1.png"))
    img4 = load_image(os.path.join(base2, "medium_20210405T0047199239_p1.png"))
    result2 = run_pipeline(img3, img4, method="auto")
    save_case(
        os.path.join(DEMO, "case2_medium_candidate_UNCONFIRMED"), result2,
        "Candidate: OHRC vs OHRC, adjacent orbit crops",
        confirmed=False,
        notes="NOT a verified overlap - no footprint/lat-lon metadata was available to confirm "
              "these two consecutive-frame crops actually share terrain (see "
              "data/test_set/README.md, Tier 2). Included to show pipeline behavior "
              "(including graceful low-confidence/failure reporting) on a harder, unverified "
              "pair - do not present this result as proof of accuracy on illumination-varied data.",
    )

    # ── Case 3: Cross-sensor OHRC vs LRO — plumbing demo, NOT a verified overlap ──
    print("\n[3/3] Cross-sensor OHRC vs LRO GeoTIFF (plumbing demo, NOT geographically confirmed)...")
    try:
        lro_meta = load_geotiff_reference(
            os.path.join(PROJECT, "data", "reference", "lro_south_pole.tif"),
            source_label="LRO NAC")
        src_img = load_image(os.path.join(base, "easy_ncp_20260102T1224107393_p1.png"))
        # This OHRC browse PNG carries no embedded lat/lon (no accompanying XML in this repo),
        # so we cannot claim any geographic overlap with the LRO tile - source_meta below is
        # a placeholder just to exercise the register_cross_source() code path end-to-end.
        src_meta = {
            "pixel_resolution_m": 0.25,
            "source_label": "OHRC (no XML metadata available - location unknown)",
        }
        result3 = register_cross_source(src_img, src_meta, lro_meta, method="auto")
        save_case(
            os.path.join(DEMO, "case3_cross_sensor_PLUMBING_DEMO"), result3,
            "Cross-sensor: OHRC vs LRO GeoTIFF (code-path demo only)",
            confirmed=False,
            notes="NOT a geographically verified pair. The bundled OHRC browse PNG has no "
                  "accompanying XML, so its true lunar location is unknown to this pipeline - "
                  "it cannot be matched against the LRO tile's known footprint. This case exists "
                  "only to demonstrate that register_cross_source() runs end-to-end (GSD "
                  "normalisation, crop, delegation to run_pipeline) and reports its confidence "
                  "honestly rather than crashing. Cross-sensor matching remains EXPERIMENTAL and "
                  "unvalidated on confirmed data - see README Known Limitations.",
        )
    except Exception as e:
        print(f"  [FAILED] Cross-sensor plumbing demo raised: {e}")
        os.makedirs(os.path.join(DEMO, "case3_cross_sensor_PLUMBING_DEMO"), exist_ok=True)
        with open(os.path.join(DEMO, "case3_cross_sensor_PLUMBING_DEMO", "result.json"), "w") as f:
            json.dump({"success": False, "exception": str(e),
                       "notes": "register_cross_source() raised - see exception."}, f, indent=2)

    print("\nDone. See demo/README.md for how to interpret these.")


if __name__ == "__main__":
    main()
