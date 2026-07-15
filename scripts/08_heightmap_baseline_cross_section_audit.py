"""08_heightmap_baseline_cross_section_audit.py

Diagnostic audit only: baseline / y-cross-section inspection following
Experiment 07 preprocessing exactly.

This script does not implement a new extractor, regime classifier, or final
geometry target. It does not inspect Track 21.
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


# Reuse Experiment 07 preprocessing functions/constants directly to avoid
# silently redefining loading, detrending, baseline, residual, or gradient logic.
EXP07_PATH = REPO_ROOT / "scripts" / "07_heightmap_2d_object_identity_audit.py"
exp07 = SourceFileLoader("exp07_heightmap_2d_object_identity_audit", str(EXP07_PATH)).load_module()


SEALED_TRACKS = {21}
TRACK_IDS = [8, 10, 14]
assert not (set(TRACK_IDS) & SEALED_TRACKS), "Track 21 is sealed and must not be analyzed."

CENTRAL_Y_MIN_MM = float(exp07.CENTRAL_EXCLUSION_Y_MIN_MM)
CENTRAL_Y_MAX_MM = float(exp07.CENTRAL_EXCLUSION_Y_MAX_MM)
BOUNDARY_EXCLUSION_MM = 2.0
MIN_TOTAL_FINITE_FRACTION = 0.40
MIN_CENTRAL_FINITE_FRACTION = 0.80
REPRESENTATIVE_QUANTILES = {"quiet": 0.10, "moderate": 0.50, "strong": 0.90}


def robust_mad(values: np.ndarray, scale: bool = True) -> float:
    return float(exp07.robust_mad(values, scale=scale))


def safe_percentile(values: np.ndarray, q: float) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    return float(np.nanpercentile(v, q))


def compute_cross_section_metrics(
    track_id: int,
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    fields: Dict[str, np.ndarray],
) -> pd.DataFrame:
    central = (y_mm >= CENTRAL_Y_MIN_MM) & (y_mm <= CENTRAL_Y_MAX_MM)
    substrate = ~central
    rows: List[Dict[str, object]] = []

    signed = np.asarray(fields["signed_residual_um"], dtype=float)
    z_um = np.asarray(fields["detrended_z_um"], dtype=float)
    finite = np.asarray(fields["finite_mask"], dtype=bool)
    baseline_um = np.asarray(fields["baseline_um_by_x"], dtype=float)
    spread_um = np.asarray(fields["spread_um_by_x"], dtype=float)
    n_sub = np.asarray(fields["n_substrate_by_x"], dtype=float)

    for j, x in enumerate(x_mm):
        valid_col = finite[:, j]
        total_finite_fraction = float(np.mean(valid_col))
        central_valid = valid_col & central & np.isfinite(signed[:, j])
        substrate_valid = valid_col & substrate & np.isfinite(signed[:, j])
        central_finite_fraction = float(np.sum(central_valid) / max(1, int(np.sum(central))))
        substrate_finite_fraction = float(np.sum(substrate_valid) / max(1, int(np.sum(substrate))))

        cr = signed[central_valid, j]
        sr = signed[substrate_valid, j]
        spread = float(spread_um[j]) if np.isfinite(spread_um[j]) else float("nan")
        if cr.size:
            frac_pos = float(np.mean(cr > 0.0))
            frac_above_1 = float(np.mean(cr > spread)) if np.isfinite(spread) else float("nan")
            frac_above_2 = float(np.mean(cr > 2.0 * spread)) if np.isfinite(spread) else float("nan")
            frac_below_1 = float(np.mean(cr < -spread)) if np.isfinite(spread) else float("nan")
            event_strength = float(np.nanmedian(np.abs(cr) / max(spread, 1e-9))) if np.isfinite(spread) else float("nan")
        else:
            frac_pos = frac_above_1 = frac_above_2 = frac_below_1 = event_strength = float("nan")

        rows.append(
            {
                "track_id": int(track_id),
                "x_mm": float(x),
                "x_index": int(j),
                "is_within_boundary_exclusion": bool(
                    (x < float(np.nanmin(x_mm)) + BOUNDARY_EXCLUSION_MM)
                    or (x > float(np.nanmax(x_mm)) - BOUNDARY_EXCLUSION_MM)
                ),
                "finite_fraction": total_finite_fraction,
                "central_finite_fraction": central_finite_fraction,
                "substrate_finite_fraction": substrate_finite_fraction,
                "estimated_substrate_baseline_um": float(baseline_um[j]),
                "substrate_spread_mad_um": spread,
                "n_substrate_samples_for_baseline": float(n_sub[j]),
                "central_median_signed_residual_um": float(np.nanmedian(cr)) if cr.size else float("nan"),
                "central_mean_signed_residual_um": float(np.nanmean(cr)) if cr.size else float("nan"),
                "central_positive_fraction": frac_pos,
                "central_fraction_above_plus1_mad": frac_above_1,
                "central_fraction_above_plus2_mad": frac_above_2,
                "central_fraction_below_minus1_mad": frac_below_1,
                "central_residual_iqr_um": (safe_percentile(cr, 75) - safe_percentile(cr, 25)) if cr.size else float("nan"),
                "central_residual_max_um": float(np.nanmax(cr)) if cr.size else float("nan"),
                "central_residual_min_um": float(np.nanmin(cr)) if cr.size else float("nan"),
                "substrate_residual_median_after_baseline_um": float(np.nanmedian(sr)) if sr.size else float("nan"),
                "substrate_residual_mad_after_baseline_um": robust_mad(sr, scale=True) if sr.size else float("nan"),
                "event_strength_central_median_abs_residual_over_mad": event_strength,
                "detrended_z_central_median_um": float(np.nanmedian(z_um[central_valid, j])) if np.any(central_valid) else float("nan"),
                "detrended_z_substrate_median_um": float(np.nanmedian(z_um[substrate_valid, j])) if np.any(substrate_valid) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def select_representative_positions(metrics: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    eligible = metrics[
        metrics["event_strength_central_median_abs_residual_over_mad"].notna()
        & metrics["central_median_signed_residual_um"].notna()
        & (metrics["finite_fraction"] >= MIN_TOTAL_FINITE_FRACTION)
        & (metrics["central_finite_fraction"] >= MIN_CENTRAL_FINITE_FRACTION)
        & (~metrics["is_within_boundary_exclusion"])
    ].copy()
    selection_mode = "strict"
    if len(eligible) < 3:
        eligible = metrics[
            metrics["event_strength_central_median_abs_residual_over_mad"].notna()
            & metrics["central_median_signed_residual_um"].notna()
            & (metrics["central_finite_fraction"] >= 0.50)
            & (~metrics["is_within_boundary_exclusion"])
        ].copy()
        selection_mode = "relaxed_central_finite_fraction_0p50"
    if len(eligible) < 3:
        eligible = metrics[
            metrics["event_strength_central_median_abs_residual_over_mad"].notna()
            & metrics["central_median_signed_residual_um"].notna()
        ].copy()
        selection_mode = "fallback_all_valid_event_strength"

    eligible = eligible.sort_values("event_strength_central_median_abs_residual_over_mad").reset_index(drop=True)
    eligible["event_strength_rank"] = np.arange(1, len(eligible) + 1)
    eligible["event_strength_percentile_rank"] = (eligible["event_strength_rank"] - 1) / max(1, len(eligible) - 1)
    eligible["selection_mode"] = selection_mode

    selected_rows: List[pd.Series] = []
    used_indices: set[int] = set()
    for label, q in REPRESENTATIVE_QUANTILES.items():
        if eligible.empty:
            continue
        target_value = float(np.nanquantile(eligible["event_strength_central_median_abs_residual_over_mad"], q))
        distances = np.abs(eligible["event_strength_central_median_abs_residual_over_mad"] - target_value)
        order = list(np.argsort(distances.to_numpy()))
        chosen_pos = order[0]
        for candidate_pos in order:
            original_idx = int(eligible.loc[candidate_pos, "x_index"])
            if original_idx not in used_indices:
                chosen_pos = candidate_pos
                break
        chosen = eligible.loc[chosen_pos].copy()
        chosen["event_class"] = label
        chosen["target_quantile"] = float(q)
        chosen["target_event_strength_value"] = target_value
        chosen["selection_abs_distance_to_target"] = float(distances.iloc[chosen_pos])
        used_indices.add(int(chosen["x_index"]))
        selected_rows.append(chosen)

    selected = pd.DataFrame(selected_rows)
    class_order = {"quiet": 0, "moderate": 1, "strong": 2}
    if not selected.empty:
        selected["event_class_order"] = selected["event_class"].map(class_order)
        selected = selected.sort_values("event_class_order").drop(columns=["event_class_order"]).reset_index(drop=True)
    return selected, eligible


def plot_selected_cross_section(
    out_path: Path,
    track_id: int,
    event_class: str,
    x_mm_value: float,
    x_index: int,
    y_mm: np.ndarray,
    fields: Dict[str, np.ndarray],
    diagnostics: pd.Series,
) -> None:
    z_um = fields["detrended_z_um"][:, x_index]
    residual_um = fields["signed_residual_um"][:, x_index]
    finite = fields["finite_mask"][:, x_index]
    baseline_um = float(fields["baseline_um_by_x"][x_index])

    fig, axes = plt.subplots(2, 1, figsize=(9, 7.2), sharex=True)

    ax = axes[0]
    ax.plot(y_mm[finite], z_um[finite], color="black", lw=1.2, label="detrended z(y)")
    ax.axhline(baseline_um, color="tab:blue", ls="--", lw=1.1, label="estimated substrate baseline")
    ax.axvspan(CENTRAL_Y_MIN_MM, CENTRAL_Y_MAX_MM, color="gold", alpha=0.16, label="central exclusion/prior band")
    if np.any(~finite):
        ymin, ymax = ax.get_ylim()
        ax.scatter(y_mm[~finite], np.full(np.sum(~finite), ymin), s=5, color="0.65", label="NaN y samples")
    ax.set_ylabel("detrended z (µm)")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best", fontsize=8)

    ax = axes[1]
    ax.plot(y_mm[finite], residual_um[finite], color="tab:purple", lw=1.2, label="signed residual r(y)")
    ax.axhline(0.0, color="black", ls="--", lw=1.0, label="zero residual")
    ax.axvspan(CENTRAL_Y_MIN_MM, CENTRAL_Y_MAX_MM, color="gold", alpha=0.16, label="central exclusion/prior band")
    if np.any(~finite):
        ymin, ymax = ax.get_ylim()
        ax.scatter(y_mm[~finite], np.full(np.sum(~finite), ymin), s=5, color="0.65", label="NaN y samples")
    ax.set_xlabel("y (mm)")
    ax.set_ylabel("signed residual (µm)")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best", fontsize=8)

    fig.suptitle(
        f"Track {track_id} {event_class.upper()} cross-section | x={x_mm_value:.3f} mm\n"
        f"event={diagnostics['event_strength_central_median_abs_residual_over_mad']:.3f}, "
        f"central median={diagnostics['central_median_signed_residual_um']:.3f} µm, "
        f"central positive fraction={diagnostics['central_positive_fraction']:.3f}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_track_comparison(
    out_path: Path,
    track_id: int,
    selected: pd.DataFrame,
    y_mm: np.ndarray,
    fields: Dict[str, np.ndarray],
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)
    colors = {"quiet": "tab:blue", "moderate": "tab:orange", "strong": "tab:red"}
    for _, row in selected.iterrows():
        label = str(row["event_class"])
        j = int(row["x_index"])
        finite = fields["finite_mask"][:, j]
        axes[0].plot(y_mm[finite], fields["detrended_z_um"][finite, j], color=colors[label], lw=1.1, label=f"{label} x={row['x_mm']:.2f}")
        axes[0].axhline(float(fields["baseline_um_by_x"][j]), color=colors[label], ls="--", lw=0.7, alpha=0.8)
        axes[1].plot(y_mm[finite], fields["signed_residual_um"][finite, j], color=colors[label], lw=1.1, label=f"{label} x={row['x_mm']:.2f}")
    for ax in axes:
        ax.axvspan(CENTRAL_Y_MIN_MM, CENTRAL_Y_MAX_MM, color="gold", alpha=0.14)
        ax.grid(True, alpha=0.2)
        ax.legend(loc="best", fontsize=8)
    axes[0].set_ylabel("detrended z (µm)")
    axes[0].set_title(f"Track {track_id}: selected quiet/moderate/strong cross-sections")
    axes[1].axhline(0.0, color="black", ls="--", lw=1.0)
    axes[1].set_ylabel("signed residual (µm)")
    axes[1].set_xlabel("y (mm)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def summarize_track_distribution(metrics: pd.DataFrame) -> Dict[str, object]:
    valid = metrics[
        metrics["event_strength_central_median_abs_residual_over_mad"].notna()
        & metrics["central_median_signed_residual_um"].notna()
        & (metrics["central_finite_fraction"] >= MIN_CENTRAL_FINITE_FRACTION)
        & (~metrics["is_within_boundary_exclusion"])
    ].copy()
    if valid.empty:
        valid = metrics.dropna(subset=["event_strength_central_median_abs_residual_over_mad", "central_median_signed_residual_um"]).copy()

    def dist(prefix: str, col: str) -> Dict[str, float]:
        v = valid[col].to_numpy(dtype=float)
        return {
            f"{prefix}_p10": safe_percentile(v, 10),
            f"{prefix}_p50": safe_percentile(v, 50),
            f"{prefix}_p90": safe_percentile(v, 90),
            f"{prefix}_mean": float(np.nanmean(v)) if np.any(np.isfinite(v)) else float("nan"),
        }

    out: Dict[str, object] = {
        "track_id": int(metrics["track_id"].iloc[0]),
        "n_x_total": int(len(metrics)),
        "n_x_valid_for_distribution": int(len(valid)),
        "valid_fraction_for_distribution": float(len(valid) / max(1, len(metrics))),
        "median_finite_fraction_valid_x": float(np.nanmedian(valid["finite_fraction"])),
        "median_central_finite_fraction_valid_x": float(np.nanmedian(valid["central_finite_fraction"])),
    }
    out.update(dist("central_median_signed_residual_um", "central_median_signed_residual_um"))
    out.update(dist("central_positive_fraction", "central_positive_fraction"))
    out.update(dist("central_fraction_above_plus1_mad", "central_fraction_above_plus1_mad"))
    out.update(dist("event_strength", "event_strength_central_median_abs_residual_over_mad"))
    out.update(dist("substrate_residual_median_after_baseline_um", "substrate_residual_median_after_baseline_um"))
    return out


def classify_evidence(summary_df: pd.DataFrame) -> Tuple[str, Dict[str, float]]:
    s = summary_df.set_index("track_id")
    t8_med = float(s.loc[8, "central_median_signed_residual_um_p50"])
    t10_med = float(s.loc[10, "central_median_signed_residual_um_p50"])
    t14_med = float(s.loc[14, "central_median_signed_residual_um_p50"])
    t8_pos = float(s.loc[8, "central_positive_fraction_p50"])
    t10_pos = float(s.loc[10, "central_positive_fraction_p50"])
    t14_pos = float(s.loc[14, "central_positive_fraction_p50"])
    t8_sub_abs = abs(float(s.loc[8, "substrate_residual_median_after_baseline_um_p50"] or 0.0))
    t10_sub_abs = abs(float(s.loc[10, "substrate_residual_median_after_baseline_um_p50"] or 0.0))
    t14_sub_abs = abs(float(s.loc[14, "substrate_residual_median_after_baseline_um_p50"] or 0.0))
    t8_event = float(s.loc[8, "event_strength_p50"])
    t10_event = float(s.loc[10, "event_strength_p50"])
    t14_event = float(s.loc[14, "event_strength_p50"])

    evidence = {
        "track8_minus_mean_track10_14_central_median_um": t8_med - 0.5 * (t10_med + t14_med),
        "track8_minus_mean_track10_14_positive_fraction": t8_pos - 0.5 * (t10_pos + t14_pos),
        "max_abs_substrate_residual_median_after_baseline_um": max(t8_sub_abs, t10_sub_abs, t14_sub_abs),
        "track8_minus_mean_track10_14_event_strength": t8_event - 0.5 * (t10_event + t14_event),
    }

    baseline_pathology = evidence["max_abs_substrate_residual_median_after_baseline_um"] > 1.0
    geometry_distinction = (
        evidence["track8_minus_mean_track10_14_central_median_um"] > 2.0
        and evidence["track8_minus_mean_track10_14_positive_fraction"] > 0.15
        and not baseline_pathology
    )
    if geometry_distinction:
        verdict = "supports_genuine_track8_vs_track10_14_signed_geometry_distinction"
    elif baseline_pathology:
        verdict = "baseline_induced_distinction_plausible_or_suspicious"
    else:
        verdict = "ambiguous"
    return verdict, evidence


def main() -> None:
    height_dir = REPO_ROOT / "data" / "raw" / "height_maps"
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPO_ROOT / "processed_data" / "run_outputs" / f"08_heightmap_baseline_cross_section_audit_{run_tag}"
    fig_dir = out_dir / "figures"
    table_dir = out_dir / "tables"
    for d in (fig_dir, table_dir):
        d.mkdir(parents=True, exist_ok=True)

    all_metrics: List[pd.DataFrame] = []
    all_selected: List[pd.DataFrame] = []
    all_eligible: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, object]] = []
    figures: List[str] = []
    preprocessing_meta: Dict[str, object] = {}

    for track_id in TRACK_IDS:
        hm = load_wyko_asc(height_dir, track_id, crop_to_common=True)
        x_mm = np.asarray(hm["x_actual_mm"], dtype=float)
        y_mm = np.asarray(hm["y_mm"], dtype=float)
        Z_mm = np.asarray(hm["Z_mm"], dtype=float)

        Z_det, _coef, detrend_meta = exp07.robust_plane_fit_substrate_focused(
            Z_mm,
            x_mm,
            y_mm,
            y_excl=exp07.SUBSTRATE_EXCLUSION_Y_MM,
            stride_x=40,
            stride_y=2,
            max_iter=3,
        )
        fields, field_meta = exp07.build_fields(Z_det, y_mm)
        preprocessing_meta[str(track_id)] = {
            "source_file": hm["file"],
            "detrend_meta": detrend_meta,
            "field_meta": field_meta,
        }

        metrics = compute_cross_section_metrics(track_id, x_mm, y_mm, fields)
        selected, eligible = select_representative_positions(metrics)
        all_metrics.append(metrics)
        all_selected.append(selected)
        all_eligible.append(eligible.assign(track_id=track_id))
        summary_rows.append(summarize_track_distribution(metrics))

        for _, row in selected.iterrows():
            p = fig_dir / f"track_{track_id}_{row['event_class']}_x_{row['x_mm']:.3f}_cross_section.png"
            plot_selected_cross_section(p, track_id, str(row["event_class"]), float(row["x_mm"]), int(row["x_index"]), y_mm, fields, row)
            figures.append(str(p.relative_to(out_dir)))
        p = fig_dir / f"track_{track_id}_quiet_moderate_strong_comparison.png"
        plot_track_comparison(p, track_id, selected, y_mm, fields)
        figures.append(str(p.relative_to(out_dir)))

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    selected_df = pd.concat(all_selected, ignore_index=True)
    eligible_df = pd.concat(all_eligible, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)
    verdict, verdict_evidence = classify_evidence(summary_df)

    metrics_path = table_dir / "all_x_cross_section_metrics.csv"
    selected_path = table_dir / "selected_cross_section_diagnostics.csv"
    eligible_path = table_dir / "selection_candidate_pool.csv"
    summary_path = table_dir / "track_level_distribution_summary.csv"
    metadata_path = table_dir / "run_metadata.json"
    verdict_path = table_dir / "baseline_distinction_audit_result.json"

    metrics_df.to_csv(metrics_path, index=False)
    selected_df.to_csv(selected_path, index=False)
    eligible_df.to_csv(eligible_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    verdict_payload = {"audit_result": verdict, "evidence": verdict_evidence}
    verdict_path.write_text(json.dumps(verdict_payload, indent=2), encoding="utf-8")

    metadata = {
        "experiment": "08_heightmap_baseline_cross_section_audit",
        "out_dir": str(out_dir),
        "tracks": TRACK_IDS,
        "sealed_tracks": sorted(SEALED_TRACKS),
        "reused_experiment07_script": str(EXP07_PATH),
        "baseline_calculation": {
            "description": "Exact Experiment 07 logic: substrate-focused plane detrending; per-column substrate baseline and MAD are computed from finite y samples outside the central exclusion band; columns with insufficient substrate support fall back to global outside-band substrate median/MAD; signed residual equals detrended z minus the per-column baseline.",
            "central_exclusion_y_mm": [CENTRAL_Y_MIN_MM, CENTRAL_Y_MAX_MM],
            "min_substrate_samples_per_column": int(exp07.MIN_SUBSTRATE_SAMPLES_PER_COLUMN),
            "min_substrate_spread_um": float(exp07.MIN_SUBSTRATE_SPREAD_UM),
        },
        "selection_rule": {
            "event_strength_statistic": "median(abs(signed_residual_um) / substrate_spread_mad_um) within finite samples in y=[0.65,1.35] mm",
            "quiet_quantile": REPRESENTATIVE_QUANTILES["quiet"],
            "moderate_quantile": REPRESENTATIVE_QUANTILES["moderate"],
            "strong_quantile": REPRESENTATIVE_QUANTILES["strong"],
            "eligibility": {
                "minimum_total_finite_fraction": MIN_TOTAL_FINITE_FRACTION,
                "minimum_central_finite_fraction": MIN_CENTRAL_FINITE_FRACTION,
                "boundary_exclusion_mm_each_end": BOUNDARY_EXCLUSION_MM,
            },
            "tie_breaking": "nearest event-strength to target quantile among eligible x columns; selected x indices are unique within each track",
        },
        "preprocessing_meta_by_track": preprocessing_meta,
        "outputs": {
            "figures": figures,
            "tables": [
                str(metrics_path.relative_to(out_dir)),
                str(selected_path.relative_to(out_dir)),
                str(eligible_path.relative_to(out_dir)),
                str(summary_path.relative_to(out_dir)),
                str(verdict_path.relative_to(out_dir)),
                str(metadata_path.relative_to(out_dir)),
            ],
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Wrote Experiment 08 outputs to: {out_dir}")
    print("Selected x positions:")
    for track_id, group in selected_df.groupby("track_id"):
        items = ", ".join(f"{r.event_class}={r.x_mm:.3f} mm" for r in group.itertuples(index=False))
        print(f"  Track {track_id}: {items}")
    print(f"Audit result: {verdict}")


if __name__ == "__main__":
    main()