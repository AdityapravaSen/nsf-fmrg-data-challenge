"""10_geometry_descriptor_requirements_audit.py

Phase II exploratory requirements audit.

Purpose: identify what information any future geometry descriptor must preserve.
This is requirements analysis, not a descriptor implementation, production
extractor, classifier, or ML model. Track 21 remains sealed.
"""

from __future__ import annotations

from datetime import datetime
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import Dict, List, Tuple

import json
import math
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


warnings.filterwarnings("ignore", category=RuntimeWarning)

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from nsf_fmrg_data import load_wyko_asc


EXP07_PATH = REPO_ROOT / "scripts" / "07_heightmap_2d_object_identity_audit.py"
exp07 = SourceFileLoader("exp07_heightmap_2d_object_identity_audit", str(EXP07_PATH)).load_module()


SEALED_TRACKS = {21}
TRACK_IDS = [8, 10, 14]
assert not (set(TRACK_IDS) & SEALED_TRACKS), "Track 21 is sealed and must not be analyzed."

CENTRAL_Y_MIN_MM = float(exp07.CENTRAL_EXCLUSION_Y_MIN_MM)
CENTRAL_Y_MAX_MM = float(exp07.CENTRAL_EXCLUSION_Y_MAX_MM)
MIN_CENTRAL_FINITE_FRACTION = 0.80
BOUNDARY_EXCLUSION_MM = 2.0

EXP04_DIR = REPO_ROOT / "processed_data" / "run_outputs" / "04_heightmap_prior_sensitivity_20260713_194307" / "tables"
EXP05_DIR = REPO_ROOT / "processed_data" / "run_outputs" / "05_heightmap_persistence_corridor_20260713_201054" / "tables"
EXP08_DIR = REPO_ROOT / "processed_data" / "run_outputs" / "08_heightmap_baseline_cross_section_audit_20260714_222110" / "tables"
EXP09_DIR = REPO_ROOT / "processed_data" / "run_outputs" / "09_heightmap_cross_section_shape_coherence_audit_20260714_223253" / "tables"


def robust_mad(values: np.ndarray, scale: bool = True) -> float:
    return float(exp07.robust_mad(values, scale=scale))


def finite_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    return exp07.finite_runs(mask)


def safe_percentile(values: np.ndarray, q: float) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    return float(np.nanpercentile(v, q))


def safe_iqr(values: np.ndarray) -> float:
    return safe_percentile(values, 75) - safe_percentile(values, 25)


def count_prominent_local_maxima(y: np.ndarray, z: np.ndarray, threshold_um: float) -> int:
    valid = np.isfinite(z)
    count = 0
    for a, b in finite_runs(valid):
        if (b - a) < 5:
            continue
        zz = z[a:b]
        for k in range(1, len(zz) - 1):
            if zz[k] >= zz[k - 1] and zz[k] >= zz[k + 1] and zz[k] > threshold_um:
                count += 1
    return int(count)


def compute_track_property_metrics(track_id: int) -> pd.DataFrame:
    height_dir = REPO_ROOT / "data" / "raw" / "height_maps"
    hm = load_wyko_asc(height_dir, track_id, crop_to_common=True)
    x_mm = np.asarray(hm["x_actual_mm"], dtype=float)
    y_mm = np.asarray(hm["y_mm"], dtype=float)
    Z_mm = np.asarray(hm["Z_mm"], dtype=float)
    Z_det, _coef, _dmeta = exp07.robust_plane_fit_substrate_focused(
        Z_mm,
        x_mm,
        y_mm,
        y_excl=exp07.SUBSTRATE_EXCLUSION_Y_MM,
        stride_x=40,
        stride_y=2,
        max_iter=3,
    )
    fields, _fmeta = exp07.build_fields(Z_det, y_mm)

    central = (y_mm >= CENTRAL_Y_MIN_MM) & (y_mm <= CENTRAL_Y_MAX_MM)
    central_y = y_mm[central]
    left_half = central & (y_mm <= 0.5 * (CENTRAL_Y_MIN_MM + CENTRAL_Y_MAX_MM))
    right_half = central & (y_mm > 0.5 * (CENTRAL_Y_MIN_MM + CENTRAL_Y_MAX_MM))
    finite = np.asarray(fields["finite_mask"], dtype=bool)
    signed = np.asarray(fields["signed_residual_um"], dtype=float)
    z_um = np.asarray(fields["detrended_z_um"], dtype=float)
    grad = np.asarray(fields["gradient_activity_um_per_mm"], dtype=float)
    n_sub = np.asarray(fields["n_substrate_by_x"], dtype=float)
    spread_um = np.asarray(fields["spread_um_by_x"], dtype=float)

    rows: List[Dict[str, object]] = []
    x_min = float(np.nanmin(x_mm))
    x_max = float(np.nanmax(x_mm))
    for j, x in enumerate(x_mm):
        valid_col = finite[:, j]
        c_valid = valid_col & central & np.isfinite(signed[:, j])
        c = signed[c_valid, j]
        zc = z_um[c_valid, j]
        yc = y_mm[c_valid]
        g = grad[c_valid, j]
        if c.size == 0:
            continue
        centered = c - float(np.nanmedian(c))
        shape_scale = robust_mad(centered, scale=True)
        peak_idx = int(np.nanargmax(c)) if c.size else 0
        peak_y = float(yc[peak_idx]) if c.size else float("nan")
        pos_mask = c > 0
        positive_support_width = float(np.nanmax(yc[pos_mask]) - np.nanmin(yc[pos_mask])) if np.sum(pos_mask) >= 2 else 0.0
        roughness = robust_mad(np.diff(c), scale=True) if c.size >= 3 else float("nan")
        if c.size >= 5:
            curvature_proxy = robust_mad(np.diff(c, n=2), scale=True)
        else:
            curvature_proxy = float("nan")
        left_vals = signed[valid_col & left_half & np.isfinite(signed[:, j]), j]
        right_vals = signed[valid_col & right_half & np.isfinite(signed[:, j]), j]
        asym = (float(np.nanmedian(left_vals)) - float(np.nanmedian(right_vals))) if left_vals.size and right_vals.size else float("nan")
        scale_for_peaks = max(shape_scale if np.isfinite(shape_scale) else 0.0, 1.0)
        local_maxima = count_prominent_local_maxima(yc, c, float(np.nanmedian(c) + scale_for_peaks))
        rows.append(
            {
                "track_id": int(track_id),
                "x_mm": float(x),
                "x_index": int(j),
                "finite_fraction": float(np.mean(valid_col)),
                "central_finite_fraction": float(np.sum(c_valid) / max(1, int(np.sum(central)))),
                "baseline_support_count": float(n_sub[j]),
                "fallback_baseline_required": bool(n_sub[j] < exp07.MIN_SUBSTRATE_SAMPLES_PER_COLUMN),
                "is_within_boundary_exclusion": bool((x < x_min + BOUNDARY_EXCLUSION_MM) or (x > x_max - BOUNDARY_EXCLUSION_MM)),
                "eligible_core": bool(
                    float(np.sum(c_valid) / max(1, int(np.sum(central)))) >= MIN_CENTRAL_FINITE_FRACTION
                    and n_sub[j] >= exp07.MIN_SUBSTRATE_SAMPLES_PER_COLUMN
                    and not ((x < x_min + BOUNDARY_EXCLUSION_MM) or (x > x_max - BOUNDARY_EXCLUSION_MM))
                ),
                "overall_height_central_median_um": float(np.nanmedian(c)),
                "overall_height_central_mean_um": float(np.nanmean(c)),
                "amplitude_iqr_um": safe_iqr(c),
                "amplitude_mad_um": robust_mad(c, scale=True),
                "amplitude_peak_to_trough_um": float(np.nanmax(c) - np.nanmin(c)),
                "positive_fraction": float(np.mean(c > 0)),
                "fraction_above_plus1_mad": float(np.mean(c > spread_um[j])) if np.isfinite(spread_um[j]) else float("nan"),
                "fraction_below_minus1_mad": float(np.mean(c < -spread_um[j])) if np.isfinite(spread_um[j]) else float("nan"),
                "positive_support_width_proxy_mm": positive_support_width,
                "peak_position_y_mm": peak_y,
                "peak_position_offset_from_corridor_center_mm": float(peak_y - 0.5 * (CENTRAL_Y_MIN_MM + CENTRAL_Y_MAX_MM)),
                "left_minus_right_median_asymmetry_um": asym,
                "roughness_diff_mad_um": roughness,
                "curvature_second_diff_mad_um": curvature_proxy,
                "gradient_median_um_per_mm": float(np.nanmedian(g)) if np.any(np.isfinite(g)) else float("nan"),
                "multi_peak_count_proxy": local_maxima,
                "shape_scale_mad_um": shape_scale,
                "near_flat_shape": bool((not np.isfinite(shape_scale)) or shape_scale < 0.75),
                "detrended_central_median_um": float(np.nanmedian(zc)),
            }
        )
    return pd.DataFrame(rows)


def summarize_property_by_track(metrics: pd.DataFrame) -> pd.DataFrame:
    props = [
        "overall_height_central_median_um",
        "amplitude_iqr_um",
        "amplitude_mad_um",
        "amplitude_peak_to_trough_um",
        "positive_fraction",
        "fraction_above_plus1_mad",
        "positive_support_width_proxy_mm",
        "peak_position_y_mm",
        "peak_position_offset_from_corridor_center_mm",
        "left_minus_right_median_asymmetry_um",
        "roughness_diff_mad_um",
        "curvature_second_diff_mad_um",
        "gradient_median_um_per_mm",
        "multi_peak_count_proxy",
        "shape_scale_mad_um",
    ]
    rows: List[Dict[str, object]] = []
    core = metrics[metrics["eligible_core"]].copy()
    for track_id, group in core.groupby("track_id"):
        for prop in props:
            vals = group[prop].to_numpy(dtype=float)
            rows.append(
                {
                    "track_id": int(track_id),
                    "property": prop,
                    "n": int(np.sum(np.isfinite(vals))),
                    "p10": safe_percentile(vals, 10),
                    "p50": safe_percentile(vals, 50),
                    "p90": safe_percentile(vals, 90),
                    "iqr": safe_iqr(vals),
                    "mean": float(np.nanmean(vals)) if np.any(np.isfinite(vals)) else float("nan"),
                    "std": float(np.nanstd(vals)) if np.any(np.isfinite(vals)) else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def load_prior_evidence() -> Dict[str, object]:
    threshold = pd.read_csv(EXP04_DIR / "threshold_sweep_summary.csv")
    exp05 = pd.read_csv(EXP05_DIR / "threshold_set_robustness_summary.csv")
    exp08 = pd.read_csv(EXP08_DIR / "track_level_distribution_summary.csv")
    exp09 = pd.read_csv(EXP09_DIR / "track_shape_coherence_summary.csv")
    low09 = pd.read_csv(EXP09_DIR / "low_coherence_longitudinal_regimes.csv")
    between09 = pd.read_csv(EXP09_DIR / "between_track_coherence_summary.csv")
    return {
        "exp04_threshold_selected_min": int(threshold["n_selected"].min()),
        "exp04_threshold_selected_max": int(threshold["n_selected"].max()),
        "exp04_threshold_unstable_note": "18/24 representative cases unstable from Progress.md and Experiment 04 outputs",
        "exp05_track8_remove_low_p95_width_shift_mm": float(exp05[(exp05.track_id == 8) & (exp05.compare_set == "REMOVE_LOW")]["p95_abs_width_shift_mm"].iloc[0]),
        "exp05_track14_remove_low_p95_width_shift_mm": float(exp05[(exp05.track_id == 14) & (exp05.compare_set == "REMOVE_LOW")]["p95_abs_width_shift_mm"].iloc[0]),
        "exp08_track_summary": exp08.to_dict(orient="records"),
        "exp09_track_summary": exp09.to_dict(orient="records"),
        "exp09_low_regimes": low09.to_dict(orient="records"),
        "exp09_between_summary": between09.to_dict(orient="records"),
    }


def build_requirements_matrix(track_summary: pd.DataFrame, prior: Dict[str, object]) -> pd.DataFrame:
    def stat(prop: str, track: int, col: str = "p50") -> float:
        row = track_summary[(track_summary.track_id == track) & (track_summary.property == prop)]
        return float(row[col].iloc[0]) if not row.empty else float("nan")

    t8_shape = prior["exp09_track_summary"][0]["within_pairwise_similarity_p50"]
    t10_shape = prior["exp09_track_summary"][1]["within_pairwise_similarity_p50"]
    t14_shape = prior["exp09_track_summary"][2]["within_pairwise_similarity_p50"]
    low_regime_count = len(prior["exp09_low_regimes"])
    threshold_range = prior["exp04_threshold_selected_max"] - prior["exp04_threshold_selected_min"]
    rows = [
        {
            "property": "offset_normalized_cross_section_shape",
            "description": "Central-corridor profile shape after robust vertical centering and amplitude normalization.",
            "observed_variability": f"Within-track median shape similarity: T8={t8_shape:.3f}, T10={t10_shape:.3f}, T14={t14_shape:.3f}; {low_regime_count} low-coherence regimes detected in T10.",
            "robustness": "High for T8/T14; moderate and regime-dependent for T10; explicitly removes baseline and amplitude effects.",
            "information_content": "High: captures shoulder/curvature/multi-feature morphology that scalar height or width loses.",
            "dependency_on_preprocessing": "Requires inherited central audit corridor and finite-support-aware normalization, but not a thresholded component.",
            "likely_usefulness": "High for downstream target representation because it preserves recurring morphology and regime changes.",
            "stability_along_x": "Stable for T8/T14; sub-regime stable for T10.",
            "baseline_sensitivity": "Low after centering; fallback-baseline columns excluded in Experiment 09 primary pool.",
            "finite_support_sensitivity": "Moderate; depends on central finite support and shared-support thresholds.",
            "threshold_sensitivity": "Low; not based on threshold-defined components.",
            "cross_track_consistency": "Partly consistent; T8-T14 between-track similarity is high but T10 differs in long-range coherence.",
            "physical_interpretability": "Moderate: interpretable as shape state, not as a named physical boundary.",
            "ml_prediction_target_suitability": "Strong candidate requirement; target would likely need vector or functional representation.",
            "rating": "Essential",
            "evidence": "Experiment 09 normalized-shape audit and low-coherence regime table.",
        },
        {
            "property": "vertical_height_offset_or_central_elevation",
            "description": "Central median signed residual relative to substrate baseline.",
            "observed_variability": f"Median central residuals from Experiment 08: T8={prior['exp08_track_summary'][0]['central_median_signed_residual_um_p50']:.3f} µm, T10={prior['exp08_track_summary'][1]['central_median_signed_residual_um_p50']:.3f} µm, T14={prior['exp08_track_summary'][2]['central_median_signed_residual_um_p50']:.3f} µm.",
            "robustness": "Moderate; Experiment 08 found no broad track-level baseline median pathology, but individual fallback cases remain suspicious.",
            "information_content": "Moderate to high; encodes signed elevation state that normalization deliberately removes.",
            "dependency_on_preprocessing": "Depends directly on substrate baseline definition and finite substrate support.",
            "likely_usefulness": "Useful as a descriptor component, but unsafe as the sole target.",
            "stability_along_x": "Variable along x; event-strength distributions differ by track.",
            "baseline_sensitivity": "Moderate to high for columns with fallback baseline or weak substrate support.",
            "finite_support_sensitivity": "Moderate.",
            "threshold_sensitivity": "Low if measured continuously; high if converted to positive components.",
            "cross_track_consistency": "Not consistent as a sole representation.",
            "physical_interpretability": "High as signed elevation relative to substrate, but not equivalent to full processed width.",
            "ml_prediction_target_suitability": "Useful scalar companion target; insufficient alone.",
            "rating": "Useful",
            "evidence": "Experiment 08 baseline audit and Experiment 07 signed-residual fields.",
        },
        {
            "property": "amplitude_or_relief",
            "description": "Central profile relief, e.g. IQR/MAD/peak-to-trough amplitude.",
            "observed_variability": f"Median amplitude IQR from current audit: T8={stat('amplitude_iqr_um',8):.3f} µm, T10={stat('amplitude_iqr_um',10):.3f} µm, T14={stat('amplitude_iqr_um',14):.3f} µm.",
            "robustness": "Moderate; robust summaries avoid single spikes but remain sensitive to finite support and local events.",
            "information_content": "High; separates near-flat profiles from high-relief events and complements normalized shape.",
            "dependency_on_preprocessing": "Uses detrended/residual profile but less dependent on absolute baseline after centering if expressed as relief.",
            "likely_usefulness": "Likely useful as an independent degree of freedom because Experiment 09 removed it intentionally.",
            "stability_along_x": "Meaningfully variable along x.",
            "baseline_sensitivity": "Low to moderate for relief metrics; higher for signed amplitude thresholds.",
            "finite_support_sensitivity": "Moderate.",
            "threshold_sensitivity": "Low if continuous; high if thresholded.",
            "cross_track_consistency": "Variable but measurable for all tracks.",
            "physical_interpretability": "High as local relief magnitude.",
            "ml_prediction_target_suitability": "Useful and likely necessary with shape.",
            "rating": "Essential",
            "evidence": "Current property metrics plus Experiment 09 normalization removing amplitude.",
        },
        {
            "property": "asymmetry_and_peak_position",
            "description": "Left-right imbalance and y-location of dominant local maximum within inherited central corridor.",
            "observed_variability": f"Median peak y: T8={stat('peak_position_y_mm',8):.3f}, T10={stat('peak_position_y_mm',10):.3f}, T14={stat('peak_position_y_mm',14):.3f} mm; median asymmetry: T8={stat('left_minus_right_median_asymmetry_um',8):.3f}, T10={stat('left_minus_right_median_asymmetry_um',10):.3f}, T14={stat('left_minus_right_median_asymmetry_um',14):.3f} µm.",
            "robustness": "Moderate; peak location can be unstable in multi-peak or low-amplitude profiles, asymmetry medians are more robust.",
            "information_content": "High for distinguishing tilted, shouldered, or laterally shifted shapes.",
            "dependency_on_preprocessing": "Depends on inherited central corridor and finite support; less tied to thresholded extraction.",
            "likely_usefulness": "Useful; should be preserved by any vector/functional descriptor.",
            "stability_along_x": "Variable; can encode process-state changes and regimes.",
            "baseline_sensitivity": "Low after within-profile centering for shape; moderate for signed asymmetry magnitude.",
            "finite_support_sensitivity": "Moderate to high if one side is poorly supported.",
            "threshold_sensitivity": "Low if continuous.",
            "cross_track_consistency": "Not invariant; variation is informative.",
            "physical_interpretability": "Moderate to high as lateral morphology descriptors.",
            "ml_prediction_target_suitability": "Useful if validity/support flags accompany it.",
            "rating": "Useful",
            "evidence": "Current asymmetry/peak metrics and Experiment 09 shape-coherence audit.",
        },
        {
            "property": "roughness_curvature_and_multi_peak_structure",
            "description": "Local derivative roughness, curvature proxy, and count of prominent local maxima within the central corridor.",
            "observed_variability": f"Median roughness diff MAD: T8={stat('roughness_diff_mad_um',8):.3f}, T10={stat('roughness_diff_mad_um',10):.3f}, T14={stat('roughness_diff_mad_um',14):.3f} µm; median multi-peak proxy: T8={stat('multi_peak_count_proxy',8):.1f}, T10={stat('multi_peak_count_proxy',10):.1f}, T14={stat('multi_peak_count_proxy',14):.1f}.",
            "robustness": "Moderate to low; derivative and peak-count proxies are sensitive to noise and sampling, but robust summaries reduce this.",
            "information_content": "High for shoulders, secondary peaks, and complex morphology noted in Experiment 06.",
            "dependency_on_preprocessing": "Sensitive to smoothing choices; current audit avoids selecting a final smoothing/tuning rule.",
            "likely_usefulness": "Useful as information to preserve, but raw derivative counts may be unsuitable as standalone targets.",
            "stability_along_x": "Variable; likely captures local events and regimes.",
            "baseline_sensitivity": "Low for derivative shape, moderate for peak thresholds.",
            "finite_support_sensitivity": "High around missing gaps.",
            "threshold_sensitivity": "Moderate if peak-count threshold is used.",
            "cross_track_consistency": "Variable; morphology complexity differs.",
            "physical_interpretability": "Moderate; should be described as shape complexity, not assigned process mechanism.",
            "ml_prediction_target_suitability": "Useful as part of richer descriptor; individual counts are optional/fragile.",
            "rating": "Useful",
            "evidence": "Experiment 06 multi-structure morphology and current derivative/peak proxies.",
        },
        {
            "property": "threshold_defined_width_or_single_boundary_interval",
            "description": "Width derived from thresholded positive or absolute components or direct boundary selection.",
            "observed_variability": f"Experiment 04 selected count varied by {threshold_range} cases across thresholds; Experiment 05 REMOVE_LOW p95 width shifts were {prior['exp05_track8_remove_low_p95_width_shift_mm']:.3f} mm for T8 and {prior['exp05_track14_remove_low_p95_width_shift_mm']:.3f} mm for T14.",
            "robustness": "Low under current evidence.",
            "information_content": "Potentially useful if named narrowly, but discards multi-structure and shape information.",
            "dependency_on_preprocessing": "High: threshold set, seed, baseline, finite support, and association rules all matter.",
            "likely_usefulness": "Unsuitable as sole descriptor at this stage.",
            "stability_along_x": "Unstable in prior experiments, especially Track 10.",
            "baseline_sensitivity": "High if based on signed threshold crossings.",
            "finite_support_sensitivity": "High.",
            "threshold_sensitivity": "High by construction and executed evidence.",
            "cross_track_consistency": "Poor; Track 10 collapsed in Experiment 05.",
            "physical_interpretability": "Low unless descriptor is narrowly named, e.g. raised-crown width.",
            "ml_prediction_target_suitability": "Unsuitable as a primary deterministic target now.",
            "rating": "Unsuitable",
            "evidence": "Experiments 04--06 threshold sensitivity, corridor collapse, and object-correspondence confounds.",
        },
        {
            "property": "finite_support_and_validity_state",
            "description": "Finite support, fallback-baseline status, missingness contact, and profile validity/ambiguity flags.",
            "observed_variability": "Experiment 09 eligible fractions: T8=0.935, T10=0.775, T14=0.719; fallback fractions all-x: T8=0.021, T10=0.131, T14=0.154.",
            "robustness": "High as measurement metadata; not a geometry target but required context.",
            "information_content": "Essential for distinguishing absent evidence from morphology.",
            "dependency_on_preprocessing": "Directly reflects measurement support and baseline eligibility criteria.",
            "likely_usefulness": "Essential as validity/confidence covariates or masks.",
            "stability_along_x": "Variable and informative about measurement reliability.",
            "baseline_sensitivity": "High relevance; determines fallback logic.",
            "finite_support_sensitivity": "It is the finite-support measure.",
            "threshold_sensitivity": "Low if recorded directly.",
            "cross_track_consistency": "Different across tracks; must be preserved.",
            "physical_interpretability": "Measurement reliability, not process morphology.",
            "ml_prediction_target_suitability": "Essential as mask/validity metadata, not as target geometry.",
            "rating": "Essential",
            "evidence": "Experiments 07--09 finite-support and fallback-baseline outputs.",
        },
    ]
    return pd.DataFrame(rows)


def answer_questions(requirements: pd.DataFrame, prior: Dict[str, object]) -> Dict[str, object]:
    essential = requirements[requirements["rating"] == "Essential"]["property"].tolist()
    useful = requirements[requirements["rating"] == "Useful"]["property"].tolist()
    unsuitable = requirements[requirements["rating"] == "Unsuitable"]["property"].tolist()
    return {
        "question_1_stable_properties": [
            "Offset/amplitude-normalized shape is stable for Tracks 8 and 14 and locally stable in Track 10 sub-regimes.",
            "Finite-support/validity metadata is reproducible and should be treated as stable measurement context rather than morphology.",
        ],
        "question_2_process_state_variation_properties": [
            "Amplitude/relief, signed central elevation, asymmetry, peak position, roughness/curvature, and multi-peak structure vary meaningfully along x.",
            "Track 10 low-coherence regimes from Experiment 09 indicate local morphology regimes rather than one global template.",
        ],
        "question_3_unstable_or_preprocessing_dependent_properties": unsuitable
        + ["raw gradient-transition boundaries", "strongest-activity seed identity", "fallback-baseline-dependent signed amplitudes"],
        "question_4_scalar_vs_multidimensional": "One scalar is not sufficient. Executed evidence requires preserving at least normalized shape, amplitude/relief, signed elevation, asymmetry/position/shape-complexity, longitudinal regime/coherence, and validity/finite-support metadata.",
        "question_5_descriptor_must_preserve": essential + useful,
        "essential_properties": essential,
        "useful_properties": useful,
        "unsuitable_properties": unsuitable,
        "preliminary_descriptor_design_recommendations": [
            "Use a representation capable of preserving profile shape, not only scalar width or height.",
            "Keep amplitude and signed elevation as separate degrees of freedom because Experiment 09 normalization removes them while Experiment 08 shows they still vary.",
            "Include validity, finite-support, and fallback-baseline flags with any target representation.",
            "Allow longitudinal sub-regimes; do not force one global template for all x positions, especially Track 10.",
            "Do not choose PCA, splines, neural embeddings, or any specific encoding yet; current evidence only defines requirements.",
        ],
    }


def plot_requirement_ratings(out_path: Path, requirements: pd.DataFrame) -> None:
    order = ["Essential", "Useful", "Optional", "Unsuitable"]
    counts = requirements["rating"].value_counts().reindex(order, fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(counts.index, counts.values, color=["tab:green", "tab:blue", "tab:gray", "tab:red"])
    ax.set_title("Experiment 10 requirements matrix rating counts")
    ax.set_ylabel("number of candidate properties")
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_property_distributions(out_path: Path, metrics: pd.DataFrame) -> None:
    core = metrics[metrics["eligible_core"]].copy()
    props = [
        ("overall_height_central_median_um", "central median residual (µm)"),
        ("amplitude_iqr_um", "amplitude IQR (µm)"),
        ("left_minus_right_median_asymmetry_um", "asymmetry (µm)"),
        ("roughness_diff_mad_um", "roughness proxy (µm)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, (prop, label) in zip(axes.ravel(), props):
        data = [core.loc[core.track_id == tid, prop].dropna().to_numpy(dtype=float) for tid in TRACK_IDS]
        ax.boxplot(data, labels=[f"T{tid}" for tid in TRACK_IDS], showfliers=False)
        ax.set_title(label)
        ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_prior_coherence(out_path: Path, prior: Dict[str, object]) -> None:
    exp09 = pd.DataFrame(prior["exp09_track_summary"])
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(exp09))
    ax.bar(x - 0.18, exp09["within_pairwise_similarity_p50"], width=0.36, label="within pairwise median")
    ax.bar(x + 0.18, exp09["similarity_to_medoid_p50"], width=0.36, label="to-medoid median")
    ax.set_xticks(x)
    ax.set_xticklabels([f"T{int(t)}" for t in exp09["track_id"]])
    ax.set_ylim(0, 1)
    ax.set_ylabel("normalized-shape similarity")
    ax.set_title("Experiment 09 shape-coherence evidence")
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPO_ROOT / "processed_data" / "run_outputs" / f"10_geometry_descriptor_requirements_audit_{run_tag}"
    table_dir = out_dir / "tables"
    fig_dir = out_dir / "figures"
    for d in (table_dir, fig_dir):
        d.mkdir(parents=True, exist_ok=True)

    prior = load_prior_evidence()
    metrics = pd.concat([compute_track_property_metrics(tid) for tid in TRACK_IDS], ignore_index=True)
    track_summary = summarize_property_by_track(metrics)
    requirements = build_requirements_matrix(track_summary, prior)
    qa = answer_questions(requirements, prior)

    metrics.to_csv(table_dir / "candidate_property_metrics_by_x.csv", index=False)
    track_summary.to_csv(table_dir / "candidate_property_track_summary.csv", index=False)
    requirements.to_csv(table_dir / "requirements_matrix.csv", index=False)
    (table_dir / "requirements_questions_answers.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    (table_dir / "prior_experiment_evidence_summary.json").write_text(json.dumps(prior, indent=2), encoding="utf-8")

    plot_requirement_ratings(fig_dir / "requirements_rating_counts.png", requirements)
    plot_property_distributions(fig_dir / "candidate_property_distributions_by_track.png", metrics)
    plot_prior_coherence(fig_dir / "experiment09_shape_coherence_summary.png", prior)

    metadata = {
        "experiment": "10_geometry_descriptor_requirements_audit",
        "out_dir": str(out_dir),
        "tracks": TRACK_IDS,
        "sealed_tracks": sorted(SEALED_TRACKS),
        "purpose": "Requirements analysis for future geometry descriptors; no descriptor implementation, extractor, classifier, or ML model.",
        "preprocessing": "Reuses Experiment 07 loading, substrate-focused detrending, baseline, residual, and finite-support conventions.",
        "central_corridor_y_mm": [CENTRAL_Y_MIN_MM, CENTRAL_Y_MAX_MM],
        "central_corridor_note": "Inherited audit region for comparability only; not asserted as final boundary.",
        "source_evidence_dirs": {
            "experiment_04": str(EXP04_DIR),
            "experiment_05": str(EXP05_DIR),
            "experiment_08": str(EXP08_DIR),
            "experiment_09": str(EXP09_DIR),
        },
        "outputs": {
            "tables": sorted([p.name for p in table_dir.iterdir()]),
            "figures": sorted([p.name for p in fig_dir.iterdir()]),
        },
    }
    (table_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Wrote Experiment 10 outputs to: {out_dir}")
    print("Requirement ratings:")
    print(requirements[["property", "rating"]].to_string(index=False))


if __name__ == "__main__":
    main()