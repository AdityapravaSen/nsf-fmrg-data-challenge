"""09_heightmap_cross_section_shape_coherence_audit.py

Exploratory diagnostic audit only.

Question: do central-corridor y-cross-sections recur longitudinally after
removing vertical offset and amplitude effects?

This is not target extraction, not classification, and not a production
algorithm. Track 21 remains sealed.
"""

from __future__ import annotations

from datetime import datetime
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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

# Shared, non-track-specific audit configuration.
BOUNDARY_EXCLUSION_MM = 2.0
MIN_TOTAL_FINITE_FRACTION = 0.40
MIN_CENTRAL_FINITE_FRACTION = 0.80
MIN_BASELINE_SUPPORT_COUNT = int(exp07.MIN_SUBSTRATE_SAMPLES_PER_COLUMN)
MIN_NORMALIZATION_SUPPORT_FRACTION = 0.70
MIN_SHARED_SUPPORT_FRACTION = 0.60
MIN_SHAPE_SCALE_UM = 0.75
MAX_WITHIN_PAIRWISE_PROFILES = 900
MAX_BETWEEN_PAIRWISE_PROFILES_PER_TRACK = 550
LOW_COHERENCE_SIMILARITY_THRESHOLD = 0.50
LOW_COHERENCE_MIN_SPAN_MM = 1.0
TRACK10_AUDIT_X_MM = 83.244


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


def safe_mean(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.nanmean(v)) if v.size else float("nan")


def choose_evenly_spaced_indices(n: int, max_n: int) -> np.ndarray:
    if n <= max_n:
        return np.arange(n, dtype=int)
    return np.unique(np.linspace(0, n - 1, max_n).round().astype(int))


def interpolate_profile_without_gap_bridging(
    y_native: np.ndarray,
    values: np.ndarray,
    valid: np.ndarray,
    y_grid: np.ndarray,
) -> np.ndarray:
    """Interpolate only inside contiguous finite native runs.

    Missing gaps remain NaN; no zero-fill or cross-gap interpolation occurs.
    """

    out = np.full_like(y_grid, np.nan, dtype=float)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(values)
    for a, b in finite_runs(valid):
        if (b - a) < 2:
            continue
        yg = y_native[a:b]
        vg = values[a:b]
        inside = (y_grid >= float(yg[0])) & (y_grid <= float(yg[-1]))
        if np.any(inside):
            out[inside] = np.interp(y_grid[inside], yg, vg)
    return out


def normalize_shape(profile_um: np.ndarray) -> Tuple[np.ndarray, Dict[str, object]]:
    valid = np.isfinite(profile_um)
    support_fraction = float(np.mean(valid))
    if np.sum(valid) < 5:
        return np.full_like(profile_um, np.nan), {
            "normalization_status": "insufficient_support",
            "shape_support_fraction": support_fraction,
            "shape_center_um": float("nan"),
            "shape_scale_um": float("nan"),
            "is_nonflat": False,
        }
    center = float(np.nanmedian(profile_um[valid]))
    centered = profile_um - center
    scale = robust_mad(centered[valid], scale=True)
    if not np.isfinite(scale) or scale < MIN_SHAPE_SCALE_UM:
        return np.full_like(profile_um, np.nan), {
            "normalization_status": "near_flat_or_too_low_scale",
            "shape_support_fraction": support_fraction,
            "shape_center_um": center,
            "shape_scale_um": float(scale) if np.isfinite(scale) else float("nan"),
            "is_nonflat": False,
        }
    norm = centered / scale
    norm[~valid] = np.nan
    return norm, {
        "normalization_status": "ok",
        "shape_support_fraction": support_fraction,
        "shape_center_um": center,
        "shape_scale_um": float(scale),
        "is_nonflat": True,
    }


def shape_similarity(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, int]:
    shared = np.isfinite(a) & np.isfinite(b)
    n_shared = int(np.sum(shared))
    shared_fraction = float(n_shared / max(1, len(a)))
    if shared_fraction < MIN_SHARED_SUPPORT_FRACTION or n_shared < 5:
        return float("nan"), shared_fraction, n_shared
    aa = a[shared]
    bb = b[shared]
    aa = aa - np.nanmean(aa)
    bb = bb - np.nanmean(bb)
    denom = float(np.sqrt(np.nansum(aa * aa) * np.nansum(bb * bb)))
    if denom <= 0 or not np.isfinite(denom):
        return float("nan"), shared_fraction, n_shared
    return float(np.nansum(aa * bb) / denom), shared_fraction, n_shared


def build_profile_table_and_matrix(
    track_id: int,
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    fields: Dict[str, np.ndarray],
    y_grid: np.ndarray,
) -> Tuple[pd.DataFrame, np.ndarray]:
    central_native = (y_mm >= CENTRAL_Y_MIN_MM) & (y_mm <= CENTRAL_Y_MAX_MM)
    substrate_native = ~central_native
    z_um = np.asarray(fields["detrended_z_um"], dtype=float)
    finite = np.asarray(fields["finite_mask"], dtype=bool)
    n_sub = np.asarray(fields["n_substrate_by_x"], dtype=float)

    normalized_profiles: List[np.ndarray] = []
    rows: List[Dict[str, object]] = []
    x_min = float(np.nanmin(x_mm))
    x_max = float(np.nanmax(x_mm))

    for j, x in enumerate(x_mm):
        valid_col = finite[:, j]
        central_valid = valid_col & central_native & np.isfinite(z_um[:, j])
        total_finite_fraction = float(np.mean(valid_col))
        central_finite_fraction = float(np.sum(central_valid) / max(1, int(np.sum(central_native))))
        baseline_support_count = float(n_sub[j])
        fallback_required = bool(baseline_support_count < MIN_BASELINE_SUPPORT_COUNT)
        substrate_support_fraction = float(np.sum(valid_col & substrate_native) / max(1, int(np.sum(substrate_native))))
        in_boundary_exclusion = bool((x < x_min + BOUNDARY_EXCLUSION_MM) or (x > x_max - BOUNDARY_EXCLUSION_MM))
        eligible = bool(
            (total_finite_fraction >= MIN_TOTAL_FINITE_FRACTION)
            and (central_finite_fraction >= MIN_CENTRAL_FINITE_FRACTION)
            and (baseline_support_count >= MIN_BASELINE_SUPPORT_COUNT)
            and (not in_boundary_exclusion)
        )

        interp = interpolate_profile_without_gap_bridging(y_mm[central_native], z_um[central_native, j], central_valid[central_native], y_grid)
        norm, nmeta = normalize_shape(interp)
        normalization_support_ok = bool(nmeta["shape_support_fraction"] >= MIN_NORMALIZATION_SUPPORT_FRACTION)
        nonflat = bool(nmeta["is_nonflat"] and normalization_support_ok)
        if not nonflat:
            norm = np.full_like(y_grid, np.nan, dtype=float)

        rows.append(
            {
                "track_id": int(track_id),
                "x_mm": float(x),
                "x_index": int(j),
                "finite_fraction": total_finite_fraction,
                "central_corridor_finite_fraction": central_finite_fraction,
                "substrate_side_finite_fraction": substrate_support_fraction,
                "baseline_support_count": baseline_support_count,
                "fallback_baseline_required_exp08_style": fallback_required,
                "is_within_boundary_exclusion": in_boundary_exclusion,
                "eligible_primary_shape_pool": eligible,
                "shape_support_fraction_on_common_grid": float(nmeta["shape_support_fraction"]),
                "normalization_status": nmeta["normalization_status"],
                "shape_center_um_median": nmeta["shape_center_um"],
                "shape_scale_um_mad": nmeta["shape_scale_um"],
                "nonflat_normalized_profile": bool(nonflat),
                "included_in_similarity_analysis": bool(eligible and nonflat),
            }
        )
        normalized_profiles.append(norm)

    return pd.DataFrame(rows), np.vstack(normalized_profiles)


def pairwise_similarity_records(
    profiles: np.ndarray,
    x_values: np.ndarray,
    track_id: int,
    sample_indices: np.ndarray,
    context: str,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for pos_i in range(len(sample_indices)):
        i = int(sample_indices[pos_i])
        for pos_j in range(pos_i + 1, len(sample_indices)):
            j = int(sample_indices[pos_j])
            sim, shared_frac, n_shared = shape_similarity(profiles[i], profiles[j])
            if np.isfinite(sim):
                rows.append(
                    {
                        "track_id": int(track_id),
                        "context": context,
                        "x_i_mm": float(x_values[i]),
                        "x_j_mm": float(x_values[j]),
                        "x_separation_mm": float(abs(x_values[j] - x_values[i])),
                        "shape_similarity_corr": sim,
                        "shared_support_fraction": shared_frac,
                        "n_shared_grid_points": int(n_shared),
                    }
                )
    return pd.DataFrame(rows)


def summarize_similarity_distribution(df: pd.DataFrame, prefix: str) -> Dict[str, float]:
    vals = df["shape_similarity_corr"].to_numpy(dtype=float) if not df.empty else np.array([], dtype=float)
    return {
        f"{prefix}_n_pairs": int(np.sum(np.isfinite(vals))),
        f"{prefix}_similarity_p10": safe_percentile(vals, 10),
        f"{prefix}_similarity_p25": safe_percentile(vals, 25),
        f"{prefix}_similarity_p50": safe_percentile(vals, 50),
        f"{prefix}_similarity_p75": safe_percentile(vals, 75),
        f"{prefix}_similarity_p90": safe_percentile(vals, 90),
        f"{prefix}_similarity_mean": safe_mean(vals),
    }


def separation_summary(df: pd.DataFrame, bins: Iterable[float]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    bins_arr = np.asarray(list(bins), dtype=float)
    labels = [f"{bins_arr[i]:.1f}-{bins_arr[i+1]:.1f}" for i in range(len(bins_arr) - 1)]
    tmp = df.copy()
    tmp["separation_bin_mm"] = pd.cut(tmp["x_separation_mm"], bins=bins_arr, labels=labels, include_lowest=True)
    rows = []
    for key, group in tmp.groupby("separation_bin_mm", observed=True):
        vals = group["shape_similarity_corr"].to_numpy(dtype=float)
        rows.append(
            {
                "track_id": int(group["track_id"].iloc[0]),
                "separation_bin_mm": str(key),
                "n_pairs": int(len(group)),
                "similarity_p10": safe_percentile(vals, 10),
                "similarity_p50": safe_percentile(vals, 50),
                "similarity_p90": safe_percentile(vals, 90),
                "similarity_mean": safe_mean(vals),
            }
        )
    return pd.DataFrame(rows)


def choose_medoid(profiles: np.ndarray, included_indices: np.ndarray, x_values: np.ndarray, track_id: int) -> Tuple[int, pd.DataFrame, Dict[str, float]]:
    sample_local = choose_evenly_spaced_indices(len(included_indices), MAX_WITHIN_PAIRWISE_PROFILES)
    sample_indices = included_indices[sample_local]
    pair_df = pairwise_similarity_records(profiles, x_values, track_id, sample_indices, "within_track_medoid_sample")
    if pair_df.empty:
        return int(included_indices[0]), pair_df, {"medoid_sample_size": int(len(sample_indices)), "medoid_median_similarity": float("nan")}

    medians: Dict[int, List[float]] = {int(i): [] for i in sample_indices}
    for row in pair_df.itertuples(index=False):
        # Recover indices from x values via nearest sample entry; sample x values are unique enough at native spacing.
        i = int(sample_indices[int(np.nanargmin(np.abs(x_values[sample_indices] - row.x_i_mm)))])
        j = int(sample_indices[int(np.nanargmin(np.abs(x_values[sample_indices] - row.x_j_mm)))])
        medians[i].append(float(row.shape_similarity_corr))
        medians[j].append(float(row.shape_similarity_corr))
    medoid_idx = max(medians, key=lambda k: np.nanmedian(medians[k]) if medians[k] else -np.inf)
    return int(medoid_idx), pair_df, {
        "medoid_sample_size": int(len(sample_indices)),
        "medoid_median_similarity": float(np.nanmedian(medians[medoid_idx])) if medians[medoid_idx] else float("nan"),
    }


def similarity_to_representative(profiles: np.ndarray, profile_table: pd.DataFrame, medoid_idx: int) -> pd.DataFrame:
    medoid = profiles[medoid_idx]
    rows: List[Dict[str, object]] = []
    for row in profile_table.itertuples(index=False):
        if not bool(row.included_in_similarity_analysis):
            sim = shared = float("nan")
            n_shared = 0
        else:
            sim, shared, n_shared = shape_similarity(profiles[int(row.x_index)], medoid)
        rows.append(
            {
                "track_id": int(row.track_id),
                "x_mm": float(row.x_mm),
                "x_index": int(row.x_index),
                "included_in_similarity_analysis": bool(row.included_in_similarity_analysis),
                "similarity_to_track_medoid": sim,
                "shared_support_fraction_to_medoid": shared,
                "n_shared_grid_points_to_medoid": int(n_shared),
                "medoid_x_index": int(medoid_idx),
            }
        )
    return pd.DataFrame(rows)


def detect_low_coherence_regimes(sim_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for track_id, group in sim_df[sim_df["included_in_similarity_analysis"]].groupby("track_id"):
        g = group.sort_values("x_mm").reset_index(drop=True)
        low = g["similarity_to_track_medoid"].to_numpy(dtype=float) < LOW_COHERENCE_SIMILARITY_THRESHOLD
        for a, b in finite_runs(low):
            x0 = float(g.loc[a, "x_mm"])
            x1 = float(g.loc[b - 1, "x_mm"])
            span = max(0.0, x1 - x0)
            if span < LOW_COHERENCE_MIN_SPAN_MM:
                continue
            vals = g.loc[a : b - 1, "similarity_to_track_medoid"].to_numpy(dtype=float)
            rows.append(
                {
                    "track_id": int(track_id),
                    "regime_id": f"track_{track_id}_lowcoherence_{len(rows)+1:02d}",
                    "x_min_mm": x0,
                    "x_max_mm": x1,
                    "x_span_mm": span,
                    "n_profiles": int(b - a),
                    "median_similarity_to_medoid": safe_percentile(vals, 50),
                    "min_similarity_to_medoid": safe_percentile(vals, 0),
                    "threshold_used": float(LOW_COHERENCE_SIMILARITY_THRESHOLD),
                }
            )
    return pd.DataFrame(rows)


def between_track_pairwise(
    profiles_by_track: Dict[int, np.ndarray],
    included_by_track: Dict[int, np.ndarray],
    x_by_track: Dict[int, np.ndarray],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    pairs = [(8, 10), (8, 14), (10, 14)]
    for ta, tb in pairs:
        ia = included_by_track[ta][choose_evenly_spaced_indices(len(included_by_track[ta]), MAX_BETWEEN_PAIRWISE_PROFILES_PER_TRACK)]
        ib = included_by_track[tb][choose_evenly_spaced_indices(len(included_by_track[tb]), MAX_BETWEEN_PAIRWISE_PROFILES_PER_TRACK)]
        for i in ia:
            for j in ib:
                sim, shared, n_shared = shape_similarity(profiles_by_track[ta][int(i)], profiles_by_track[tb][int(j)])
                if np.isfinite(sim):
                    rows.append(
                        {
                            "track_a": int(ta),
                            "track_b": int(tb),
                            "x_a_mm": float(x_by_track[ta][int(i)]),
                            "x_b_mm": float(x_by_track[tb][int(j)]),
                            "shape_similarity_corr": sim,
                            "shared_support_fraction": shared,
                            "n_shared_grid_points": int(n_shared),
                        }
                    )
    return pd.DataFrame(rows)


def audit_track10_case(profile_table: pd.DataFrame, sim_df: pd.DataFrame, profiles: np.ndarray, medoid_idx: int) -> Dict[str, object]:
    t10 = profile_table[profile_table["track_id"] == 10].copy()
    idx = int((t10["x_mm"] - TRACK10_AUDIT_X_MM).abs().idxmin())
    row = profile_table.loc[idx].to_dict()
    sim_row = sim_df[(sim_df["track_id"] == 10) & (sim_df["x_index"] == row["x_index"])].iloc[0].to_dict()
    direct_sim, direct_shared, direct_n = shape_similarity(profiles[int(row["x_index"])], profiles[int(medoid_idx)])
    return {
        "target_x_mm": float(TRACK10_AUDIT_X_MM),
        "nearest_x_mm": float(row["x_mm"]),
        "nearest_x_index": int(row["x_index"]),
        "eligible_primary_shape_pool": bool(row["eligible_primary_shape_pool"]),
        "fallback_baseline_required_exp08_style": bool(row["fallback_baseline_required_exp08_style"]),
        "nonflat_normalized_profile": bool(row["nonflat_normalized_profile"]),
        "included_in_similarity_analysis": bool(row["included_in_similarity_analysis"]),
        "shape_scale_um_mad": float(row["shape_scale_um_mad"]),
        "shape_support_fraction_on_common_grid": float(row["shape_support_fraction_on_common_grid"]),
        "similarity_to_track10_medoid_primary_pool": float(sim_row["similarity_to_track_medoid"]),
        "shared_support_fraction_to_medoid_primary_pool": float(sim_row["shared_support_fraction_to_medoid"]),
        "direct_similarity_to_track10_medoid_after_normalization": float(direct_sim),
        "direct_shared_support_fraction_to_medoid": float(direct_shared),
        "direct_n_shared_grid_points_to_medoid": int(direct_n),
        "note": "Shape similarity is computed after median-centering and MAD normalization, so it is insensitive to the absolute residual amplitude flagged in Experiment 08.",
    }


def classify_evidence(track_summary: pd.DataFrame, low_regimes: pd.DataFrame, sep_df: pd.DataFrame) -> Tuple[int, str, Dict[str, object]]:
    medians = track_summary["within_pairwise_similarity_p50"].to_numpy(dtype=float)
    p10s = track_summary["within_pairwise_similarity_p10"].to_numpy(dtype=float)
    medoid_medians = track_summary["similarity_to_medoid_p50"].to_numpy(dtype=float)
    nonflat_fracs = track_summary["nonflat_fraction_of_eligible"].to_numpy(dtype=float)

    evidence = {
        "min_within_pairwise_median": safe_percentile(medians, 0),
        "min_within_pairwise_p10": safe_percentile(p10s, 0),
        "min_similarity_to_medoid_median": safe_percentile(medoid_medians, 0),
        "min_nonflat_fraction_of_eligible": safe_percentile(nonflat_fracs, 0),
        "n_low_coherence_regimes": int(len(low_regimes)),
    }

    if (
        evidence["min_within_pairwise_median"] >= 0.75
        and evidence["min_similarity_to_medoid_median"] >= 0.75
        and evidence["min_nonflat_fraction_of_eligible"] >= 0.70
        and evidence["n_low_coherence_regimes"] == 0
    ):
        return 1, "strong recurring within-track cross-sectional shape coherence is present", evidence
    if evidence["min_within_pairwise_median"] < 0.35 and evidence["min_similarity_to_medoid_median"] < 0.45:
        return 3, "normalized cross-sectional shape is too heterogeneous for a template/corridor formulation", evidence
    if evidence["n_low_coherence_regimes"] > 0 or evidence["min_within_pairwise_p10"] < 0.0:
        return 2, "coherence exists only in longitudinal sub-regimes", evidence
    return 4, "result remains ambiguous", evidence


def plot_profile_overlays(out_path: Path, track_id: int, y_grid: np.ndarray, profiles: np.ndarray, included_indices: np.ndarray, medoid_idx: int) -> None:
    sample = included_indices[choose_evenly_spaced_indices(len(included_indices), min(180, len(included_indices)))]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for idx in sample:
        ax.plot(y_grid, profiles[int(idx)], color="0.55", alpha=0.12, lw=0.7)
    ax.plot(y_grid, profiles[int(medoid_idx)], color="tab:red", lw=2.2, label="medoid normalized profile")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_title(f"Track {track_id}: normalized central-corridor profile overlay")
    ax.set_xlabel("y in inherited central audit corridor (mm)")
    ax.set_ylabel("median-centered / MAD-normalized shape")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_similarity_vs_x(out_path: Path, track_id: int, sim_df: pd.DataFrame) -> None:
    g = sim_df[(sim_df["track_id"] == track_id) & sim_df["included_in_similarity_analysis"]].sort_values("x_mm")
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(g["x_mm"], g["similarity_to_track_medoid"], lw=1.1, color="tab:blue")
    ax.axhline(LOW_COHERENCE_SIMILARITY_THRESHOLD, color="tab:red", ls="--", lw=1.0, label="low-coherence threshold")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title(f"Track {track_id}: similarity to medoid normalized profile vs x")
    ax.set_xlabel("physical x (mm)")
    ax.set_ylabel("correlation to medoid")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_similarity_vs_separation(out_path: Path, pair_df: pd.DataFrame, track_id: int) -> None:
    g = pair_df[pair_df["track_id"] == track_id]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    if not g.empty:
        sample = g.iloc[choose_evenly_spaced_indices(len(g), min(6000, len(g)))]
        ax.scatter(sample["x_separation_mm"], sample["shape_similarity_corr"], s=3, alpha=0.12, color="tab:purple")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title(f"Track {track_id}: within-track shape similarity vs x separation")
    ax.set_xlabel("x separation (mm)")
    ax.set_ylabel("pairwise normalized-shape correlation")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_between_track_comparison(out_path: Path, within_summary: pd.DataFrame, between_df: pd.DataFrame) -> None:
    labels: List[str] = []
    data: List[np.ndarray] = []
    for row in within_summary.itertuples(index=False):
        labels.append(f"within T{row.track_id}")
        data.append(np.array([row.within_pairwise_similarity_p10, row.within_pairwise_similarity_p50, row.within_pairwise_similarity_p90], dtype=float))
    between_rows = []
    for (ta, tb), group in between_df.groupby(["track_a", "track_b"]):
        vals = group["shape_similarity_corr"].to_numpy(dtype=float)
        between_rows.append((f"T{ta}-T{tb}", safe_percentile(vals, 10), safe_percentile(vals, 50), safe_percentile(vals, 90)))
    for label, p10, p50, p90 in between_rows:
        labels.append(f"between {label}")
        data.append(np.array([p10, p50, p90], dtype=float))

    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(labels))
    p10 = [d[0] for d in data]
    p50 = [d[1] for d in data]
    p90 = [d[2] for d in data]
    ax.vlines(x, p10, p90, color="0.45", lw=3, alpha=0.7, label="p10-p90")
    ax.scatter(x, p50, color="tab:red", zorder=3, label="median")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("normalized-shape correlation")
    ax.set_title("Within-track vs between-track shape coherence")
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_track10_case(out_path: Path, y_grid: np.ndarray, profiles: np.ndarray, profile_table: pd.DataFrame, audit: Dict[str, object], medoid_idx: int) -> None:
    idx = int(audit["nearest_x_index"])
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(y_grid, profiles[medoid_idx], color="tab:red", lw=2.2, label=f"Track 10 medoid x={profile_table.loc[medoid_idx, 'x_mm']:.3f} mm")
    if np.any(np.isfinite(profiles[idx])):
        ax.plot(y_grid, profiles[idx], color="tab:blue", lw=1.7, label=f"Track 10 audit x={audit['nearest_x_mm']:.3f} mm")
    ax.axhline(0.0, color="black", ls="--", lw=0.8)
    ax.set_title(
        f"Track 10 x≈83.244 normalized-shape audit | "
        f"direct similarity={audit['direct_similarity_to_track10_medoid_after_normalization']:.3f}"
    )
    ax.set_xlabel("y in inherited central audit corridor (mm)")
    ax.set_ylabel("median-centered / MAD-normalized shape")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    height_dir = REPO_ROOT / "data" / "raw" / "height_maps"
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPO_ROOT / "processed_data" / "run_outputs" / f"09_heightmap_cross_section_shape_coherence_audit_{run_tag}"
    fig_dir = out_dir / "figures"
    table_dir = out_dir / "tables"
    for d in (fig_dir, table_dir):
        d.mkdir(parents=True, exist_ok=True)

    y_grid = np.linspace(CENTRAL_Y_MIN_MM, CENTRAL_Y_MAX_MM, 176)

    profiles_by_track: Dict[int, np.ndarray] = {}
    x_by_track: Dict[int, np.ndarray] = {}
    included_by_track: Dict[int, np.ndarray] = {}
    profile_tables: List[pd.DataFrame] = []
    pair_tables: List[pd.DataFrame] = []
    medoid_rows: List[Dict[str, object]] = []
    sim_to_medoid_tables: List[pd.DataFrame] = []
    sep_tables: List[pd.DataFrame] = []
    preprocessing_meta: Dict[str, object] = {}
    figure_paths: List[str] = []
    medoid_by_track: Dict[int, int] = {}

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
        profile_table, norm_profiles = build_profile_table_and_matrix(track_id, x_mm, y_mm, fields, y_grid)
        included = profile_table.index[profile_table["included_in_similarity_analysis"]].to_numpy(dtype=int)
        if included.size < 2:
            raise RuntimeError(f"Track {track_id} has fewer than two included normalized profiles.")

        medoid_idx, medoid_pair_df, medoid_meta = choose_medoid(norm_profiles, included, x_mm, track_id)
        sim_df = similarity_to_representative(norm_profiles, profile_table, medoid_idx)
        sample_indices = included[choose_evenly_spaced_indices(len(included), MAX_WITHIN_PAIRWISE_PROFILES)]
        pair_df = pairwise_similarity_records(norm_profiles, x_mm, track_id, sample_indices, "within_track_even_sample")
        sep_df = separation_summary(pair_df, bins=[0, 1, 2, 5, 10, 20, 40, 80, 100])

        profiles_by_track[track_id] = norm_profiles
        x_by_track[track_id] = x_mm
        included_by_track[track_id] = included
        medoid_by_track[track_id] = medoid_idx
        profile_tables.append(profile_table)
        sim_to_medoid_tables.append(sim_df)
        pair_tables.append(pair_df)
        sep_tables.append(sep_df)
        preprocessing_meta[str(track_id)] = {"source_file": hm["file"], "detrend_meta": detrend_meta, "field_meta": field_meta}

        medoid_rows.append(
            {
                "track_id": int(track_id),
                "medoid_x_index": int(medoid_idx),
                "medoid_x_mm": float(x_mm[medoid_idx]),
                **medoid_meta,
            }
        )

        p = fig_dir / f"track_{track_id}_normalized_profile_overlay_and_medoid.png"
        plot_profile_overlays(p, track_id, y_grid, norm_profiles, included, medoid_idx)
        figure_paths.append(str(p.relative_to(out_dir)))
        p = fig_dir / f"track_{track_id}_similarity_to_medoid_vs_x.png"
        plot_similarity_vs_x(p, track_id, sim_df)
        figure_paths.append(str(p.relative_to(out_dir)))
        p = fig_dir / f"track_{track_id}_similarity_vs_x_separation.png"
        plot_similarity_vs_separation(p, pair_df, track_id)
        figure_paths.append(str(p.relative_to(out_dir)))

    profile_df = pd.concat(profile_tables, ignore_index=True)
    sim_to_medoid_df = pd.concat(sim_to_medoid_tables, ignore_index=True)
    within_pair_df = pd.concat(pair_tables, ignore_index=True)
    sep_df_all = pd.concat(sep_tables, ignore_index=True)
    medoid_df = pd.DataFrame(medoid_rows)
    low_regime_df = detect_low_coherence_regimes(sim_to_medoid_df)
    between_df = between_track_pairwise(profiles_by_track, included_by_track, x_by_track)

    summary_rows: List[Dict[str, object]] = []
    for track_id in TRACK_IDS:
        pt = profile_df[profile_df["track_id"] == track_id]
        sim = sim_to_medoid_df[(sim_to_medoid_df["track_id"] == track_id) & sim_to_medoid_df["included_in_similarity_analysis"]]
        pair = within_pair_df[within_pair_df["track_id"] == track_id]
        sim_vals = sim["similarity_to_track_medoid"].to_numpy(dtype=float)
        row = {
            "track_id": int(track_id),
            "n_x_total": int(len(pt)),
            "n_eligible_x_columns": int(pt["eligible_primary_shape_pool"].sum()),
            "eligible_fraction": float(pt["eligible_primary_shape_pool"].mean()),
            "n_nonflat_normalized_profiles": int(pt["included_in_similarity_analysis"].sum()),
            "nonflat_fraction_of_total": float(pt["included_in_similarity_analysis"].mean()),
            "nonflat_fraction_of_eligible": float(pt["included_in_similarity_analysis"].sum() / max(1, int(pt["eligible_primary_shape_pool"].sum()))),
            "fallback_required_fraction_all_x": float(pt["fallback_baseline_required_exp08_style"].mean()),
            "fallback_required_fraction_eligible": float(pt.loc[pt["eligible_primary_shape_pool"], "fallback_baseline_required_exp08_style"].mean()) if int(pt["eligible_primary_shape_pool"].sum()) else float("nan"),
            "similarity_to_medoid_p10": safe_percentile(sim_vals, 10),
            "similarity_to_medoid_p50": safe_percentile(sim_vals, 50),
            "similarity_to_medoid_p90": safe_percentile(sim_vals, 90),
            "similarity_to_medoid_mean": safe_mean(sim_vals),
        }
        row.update(summarize_similarity_distribution(pair, "within_pairwise"))
        summary_rows.append(row)
    track_summary_df = pd.DataFrame(summary_rows)

    between_summary_rows = []
    for (ta, tb), group in between_df.groupby(["track_a", "track_b"]):
        vals = group["shape_similarity_corr"].to_numpy(dtype=float)
        between_summary_rows.append(
            {
                "track_a": int(ta),
                "track_b": int(tb),
                "n_pairs": int(len(group)),
                "between_similarity_p10": safe_percentile(vals, 10),
                "between_similarity_p50": safe_percentile(vals, 50),
                "between_similarity_p90": safe_percentile(vals, 90),
                "between_similarity_mean": safe_mean(vals),
            }
        )
    between_summary_df = pd.DataFrame(between_summary_rows)

    track10_audit = audit_track10_case(profile_df, sim_to_medoid_df, profiles_by_track[10], medoid_by_track[10])
    p = fig_dir / "track_10_x83p244_normalized_shape_comparison.png"
    plot_track10_case(p, y_grid, profiles_by_track[10], profile_df[profile_df["track_id"] == 10].reset_index(drop=True), track10_audit, medoid_by_track[10])
    figure_paths.append(str(p.relative_to(out_dir)))
    p = fig_dir / "within_vs_between_track_coherence_comparison.png"
    plot_between_track_comparison(p, track_summary_df, between_df)
    figure_paths.append(str(p.relative_to(out_dir)))

    conclusion_id, conclusion_text, conclusion_evidence = classify_evidence(track_summary_df, low_regime_df, sep_df_all)

    profile_df.to_csv(table_dir / "profile_eligibility_and_normalization.csv", index=False)
    sim_to_medoid_df.to_csv(table_dir / "similarity_to_representative.csv", index=False)
    within_pair_df.to_csv(table_dir / "within_track_pairwise_similarity_sample.csv", index=False)
    sep_df_all.to_csv(table_dir / "similarity_vs_x_separation_summary.csv", index=False)
    medoid_df.to_csv(table_dir / "representative_medoid_profiles.csv", index=False)
    low_regime_df.to_csv(table_dir / "low_coherence_longitudinal_regimes.csv", index=False)
    between_df.to_csv(table_dir / "between_track_pairwise_similarity_sample.csv", index=False)
    between_summary_df.to_csv(table_dir / "between_track_coherence_summary.csv", index=False)
    track_summary_df.to_csv(table_dir / "track_shape_coherence_summary.csv", index=False)
    np.savez_compressed(
        table_dir / "normalized_profiles_by_track.npz",
        y_grid_mm=y_grid,
        **{f"track_{tid}_profiles": profiles_by_track[tid] for tid in TRACK_IDS},
        **{f"track_{tid}_x_mm": x_by_track[tid] for tid in TRACK_IDS},
    )
    (table_dir / "track10_x83p244_shape_audit.json").write_text(json.dumps(track10_audit, indent=2), encoding="utf-8")
    (table_dir / "evidence_level_conclusion.json").write_text(
        json.dumps(
            {
                "conclusion_id": int(conclusion_id),
                "conclusion_text": conclusion_text,
                "evidence": conclusion_evidence,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    metadata = {
        "experiment": "09_heightmap_cross_section_shape_coherence_audit",
        "out_dir": str(out_dir),
        "tracks": TRACK_IDS,
        "sealed_tracks": sorted(SEALED_TRACKS),
        "reused_experiment07_script": str(EXP07_PATH),
        "central_corridor_y_mm": [CENTRAL_Y_MIN_MM, CENTRAL_Y_MAX_MM],
        "central_corridor_note": "Inherited from prior experiments for audit comparability only; not claimed as a final target boundary.",
        "eligibility_rules": {
            "minimum_total_finite_fraction": MIN_TOTAL_FINITE_FRACTION,
            "minimum_central_corridor_finite_fraction": MIN_CENTRAL_FINITE_FRACTION,
            "minimum_substrate_side_baseline_support_count": MIN_BASELINE_SUPPORT_COUNT,
            "boundary_exclusion_mm_each_end": BOUNDARY_EXCLUSION_MM,
        },
        "normalization_rules": {
            "interpolation": "Restrict to inherited central corridor; interpolate onto shared physical y grid only within contiguous finite native y-runs; leave unsupported gaps as NaN.",
            "vertical_offset_removal": "Subtract robust median of supported central-corridor profile on shared y grid.",
            "amplitude_normalization": "Divide by 1.4826*MAD of median-centered supported central-corridor profile.",
            "minimum_shape_scale_um": MIN_SHAPE_SCALE_UM,
            "minimum_normalization_support_fraction": MIN_NORMALIZATION_SUPPORT_FRACTION,
        },
        "similarity_rules": {
            "metric": "Pearson correlation of normalized profiles over mutually supported y-grid locations.",
            "minimum_shared_support_fraction": MIN_SHARED_SUPPORT_FRACTION,
            "max_within_pairwise_profiles": MAX_WITHIN_PAIRWISE_PROFILES,
            "max_between_pairwise_profiles_per_track": MAX_BETWEEN_PAIRWISE_PROFILES_PER_TRACK,
        },
        "preprocessing_meta_by_track": preprocessing_meta,
        "figures": figure_paths,
        "tables": sorted([p.name for p in table_dir.iterdir()]),
    }
    (table_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Wrote Experiment 09 outputs to: {out_dir}")
    for row in track_summary_df.itertuples(index=False):
        print(
            f"Track {row.track_id}: eligible={row.n_eligible_x_columns}/{row.n_x_total}, "
            f"nonflat={row.n_nonflat_normalized_profiles}, within median={row.within_pairwise_similarity_p50:.3f}, "
            f"medoid median={row.similarity_to_medoid_p50:.3f}"
        )
    print(f"Conclusion {conclusion_id}: {conclusion_text}")


if __name__ == "__main__":
    main()