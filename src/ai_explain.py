"""
ai_explain.py
-------------
Uses Groq API to generate plain-English explanations of registration results.
The explanation is tailored to the actual numbers and context so every result
gets a meaningful, specific description — not a generic template.
"""

import os

GROQ_API_KEY = "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"
GROQ_MODEL   = "groq/compound-mini"


def explain_result(result: dict, source_meta: dict = None,
                   ref_meta: dict = None) -> str:
    """
    Generates a plain-English explanation of a registration result using Groq.

    Parameters:
        result      : pipeline result dict (metrics, confidence, method, etc.)
        source_meta : OHRC XML metadata for source image (optional)
        ref_meta    : OHRC XML metadata for reference image (optional)

    Returns:
        str — 3-5 sentence plain English explanation
    """
    try:
        from groq import Groq
    except ImportError:
        return "Install groq: pip install groq"

    m     = result.get("metrics", {})
    conf  = result.get("confidence", "unknown")
    degen = result.get("degenerate_fit", False)
    method = result.get("method", "unknown")
    rmse   = m.get("rmse", 0)
    inliers = m.get("inlier_count", 0)
    ratio   = m.get("inlier_ratio", 0)
    spatial = m.get("spatial_score", 0)
    ttype   = m.get("transform_type", "homography")
    proc_t  = m.get("processing_time", 0)

    # Build context from metadata if available
    meta_context = ""
    if source_meta and ref_meta:
        src_sun = source_meta.get("sun_elevation", "unknown")
        ref_sun = ref_meta.get("sun_elevation", "unknown")
        delta   = abs(float(src_sun) - float(ref_sun)) if src_sun != "unknown" else None
        src_orbit = source_meta.get("orbit_number", "unknown")
        ref_orbit = ref_meta.get("orbit_number", "unknown")
        meta_context = f"""
Source image: orbit {src_orbit}, sun elevation {src_sun}°
Reference image: orbit {ref_orbit}, sun elevation {ref_sun}°
Sun angle difference: {f'{delta:.1f}°' if delta else 'unknown'}
"""
    elif source_meta:
        src_sun = source_meta.get("sun_elevation", "unknown")
        meta_context = f"Source image sun elevation: {src_sun}°"

    # Geo-assist context
    geo_context = ""
    if result.get("coarse_geo_prealignment_used"):
        geo_pts   = result.get("coarse_geo_n_points", 0)
        geo_rmse  = result.get("coarse_geo_residual_rmse", 0)
        geo_context = (f"GPS-based pre-alignment was used: "
                       f"{geo_pts} GPS correspondences found, "
                       f"coarse alignment residual {geo_rmse:.1f} pixels.")

    # Cross-source context
    cross_context = ""
    if result.get("cross_source_crop_original_shape"):
        orig = result["cross_source_crop_original_shape"]
        final = result["cross_source_crop_final_shape"]
        gsd   = result.get("cross_source_target_res_m", 0)
        cross_context = (f"This was a cross-source registration (OHRC vs LRO NAC). "
                         f"Reference was cropped from {orig} to {final} pixels. "
                         f"Both images normalised to {gsd:.2f} m/pixel.")

    prompt = f"""You are an expert in lunar image registration and remote sensing. 
Explain this Chandrayaan-2 image registration result in 3-4 simple sentences that a 
non-expert can understand. Be specific about the numbers. Explain what the numbers mean 
in plain English. Be honest about failures — don't sugarcoat bad results.
Do not use bullet points. Write as flowing sentences.

REGISTRATION RESULT:
- Algorithm used: {method}
- Transform type: {ttype}
- RMSE: {rmse:.4f} pixels {"(sub-pixel accuracy achieved)" if rmse < 1.0 and not degen else "(not sub-pixel)" if not degen else "(UNRELIABLE — degenerate fit)"}
- Inlier count: {inliers} correct matches found
- Inlier ratio: {ratio:.1%} of matches were correct
- Spatial distribution score: {spatial:.4f} (how evenly spread the matches are, 1.0 = perfect)
- Confidence: {conf.upper()}
- Degenerate fit: {"YES — too few inliers, RMSE is mathematically forced and not meaningful" if degen else "NO — result is statistically valid"}
- Processing time: {proc_t:.2f} seconds
{meta_context}
{geo_context}
{cross_context}

Write a clear, honest explanation of what this result means. 
If confidence is HIGH, explain why it's good. 
If FAILED or degenerate, explain clearly why it failed and what it means.
Mention the actual numbers naturally in your explanation.
Keep it under 100 words."""

    try:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"AI explanation unavailable: {e}"
