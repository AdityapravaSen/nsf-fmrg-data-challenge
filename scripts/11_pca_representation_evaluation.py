"""11_pca_representation_evaluation.py

Evaluate one candidate geometry representation: PCA on normalized central-
corridor cross-sectional profiles.

This is not a model, not an extractor, and not a comparison against other
descriptor families. Track 21 remains sealed.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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
EXP09_DIR = REPO_ROOT / "processed_data" / "run_outputs" / "09_heightmap_cross_section_shape_coherence_audit_20260714_223253" / "tables"
EXP10_DIR = REPO_ROOT / "processed_data" / "run_outputs" / "10_geometry_descriptor_requirements_audit_20260714_225429" / "tables"

TRACK_IDS = [8, 10, 14]
SEALED_TRACKS = {21}
assert not (set(TRACK_IDS) & SEALED_TRACKS), "Track 21 is sealed and must not be analyzed."

PCA_GRID_SUPPORT_THRESHOLD = 0.95
PCA_ROW_COMPLETE_ON_RETAINED_GRID = True
VARIANCE_TARGETS = [0.80, 0.90, 0.95, 0.99]
RECONSTRUCTION_KS = [1, 2, 3, 5]
N_SCORE_COMPONENTS_TO_SAVE = 10
N_COMPONENTS_TO_INTERPRET = 5


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


def load_experiment09_profiles() -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    npz = np.load(EXP09_DIR / "normalized_profiles_by_track.npz")
    profile_table = pd.read_csv(EXP09_DIR / "profile_eligibility_and_normalization.csv")
    y_grid = np.asarray(npz["y_grid_mm"], dtype=float)

    matrices = []
    rows = []
    for track_id in TRACK_IDS:
        profiles = np.asarray(npz[f"track_{track_id}_profiles"], dtype=float)
        x_mm = np.asarray(npz[f"track_{track_id}_x_mm"], dtype=float)
        meta = profile_table[profile_table["track_id"] == track_id].reset_index(drop=True).copy()
        if len(meta) != profiles.shape[0]:
            raise RuntimeError(f"Metadata/profile length mismatch for Track {track_id}.")
        included = meta["included_in_similarity_analysis"].to_numpy(dtype=bool)
        selected = meta[included].copy()
        selected["source_profile_row_in_track"] = np.flatnonzero(included)
        selected["x_mm_from_npz"] = x_mm[included]
        matrices.append(profiles[included])
        rows.append(selected)
    X = np.vstack(matrices)
    meta_all = pd.concat(rows, ignore_index=True)
    return X, y_grid, meta_all


def make_pca_ready_matrix(X: np.ndarray, y_grid: np.ndarray, meta: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, Dict[str, object]]:
    feature_support = np.isfinite(X).mean(axis=0)
    keep_features = feature_support >= PCA_GRID_SUPPORT_THRESHOLD
    Xf = X[:, keep_features]
    if PCA_ROW_COMPLETE_ON_RETAINED_GRID:
        keep_rows = np.isfinite(Xf).all(axis=1)
    else:
        keep_rows = np.isfinite(Xf).mean(axis=1) >= 0.99
    X_ready = Xf[keep_rows]
    meta_ready = meta.loc[keep_rows].copy().reset_index(drop=True)
    y_ready = y_grid[keep_features]
    if not np.isfinite(X_ready).all():
        raise RuntimeError("PCA-ready matrix still contains NaNs; no filling is allowed.")
    info = {
        "input_profiles": int(X.shape[0]),
        "input_y_grid_points": int(X.shape[1]),
        "retained_y_grid_points": int(X_ready.shape[1]),
        "retained_profiles": int(X_ready.shape[0]),
        "feature_support_threshold": float(PCA_GRID_SUPPORT_THRESHOLD),
        "row_rule": "all retained y-grid points finite; no NaN filling",
        "retained_profile_fraction": float(X_ready.shape[0] / max(1, X.shape[0])),
        "retained_y_grid_fraction": float(X_ready.shape[1] / max(1, X.shape[1])),
        "retained_profiles_by_track": {str(tid): int(np.sum(meta_ready["track_id"] == tid)) for tid in TRACK_IDS},
    }
    return X_ready, y_ready, meta_ready, info


def compute_pca_svd(X: np.ndarray) -> Dict[str, np.ndarray]:
    mean = X.mean(axis=0)
    Xc = X - mean[None, :]
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    n = X.shape[0]
    eigenvalues = (S**2) / max(1, n - 1)
    evr = eigenvalues / np.sum(eigenvalues)
    scores = Xc @ Vt.T
    return {"mean": mean, "components": Vt, "singular_values": S, "eigenvalues": eigenvalues, "explained_variance_ratio": evr, "scores": scores}


def reconstruct(pca: Dict[str, np.ndarray], scores: np.ndarray, k: int) -> np.ndarray:
    k = int(min(k, pca["components"].shape[0]))
    return pca["mean"][None, :] + scores[:, :k] @ pca["components"][:k, :]


def reconstruction_error_table(X: np.ndarray, meta: pd.DataFrame, pca: Dict[str, np.ndarray]) -> pd.DataFrame:
    max_components = pca["components"].shape[0]
    ks = sorted(set(RECONSTRUCTION_KS + [10, 15, 20, 30, 50, max_components]))
    ks = [k for k in ks if k <= max_components]
    rows: List[Dict[str, object]] = []
    scores = pca["scores"]
    for k in ks:
        Xhat = reconstruct(pca, scores, k)
        rmse_by_profile = np.sqrt(np.mean((X - Xhat) ** 2, axis=1))
        rows.append(
            {
                "group": "all_tracks",
                "track_id": "all",
                "n_components": int(k),
                "n_profiles": int(len(rmse_by_profile)),
                "rmse_p10": safe_percentile(rmse_by_profile, 10),
                "rmse_p50": safe_percentile(rmse_by_profile, 50),
                "rmse_p90": safe_percentile(rmse_by_profile, 90),
                "rmse_mean": safe_mean(rmse_by_profile),
            }
        )
        for track_id in TRACK_IDS:
            mask = meta["track_id"].to_numpy(dtype=int) == track_id
            vals = rmse_by_profile[mask]
            rows.append(
                {
                    "group": f"track_{track_id}",
                    "track_id": int(track_id),
                    "n_components": int(k),
                    "n_profiles": int(len(vals)),
                    "rmse_p10": safe_percentile(vals, 10),
                    "rmse_p50": safe_percentile(vals, 50),
                    "rmse_p90": safe_percentile(vals, 90),
                    "rmse_mean": safe_mean(vals),
                }
            )
    return pd.DataFrame(rows)


def variance_tables(pca: Dict[str, np.ndarray]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    evr = pca["explained_variance_ratio"]
    cumulative = np.cumsum(evr)
    variance = pd.DataFrame(
        {
            "component": np.arange(1, len(evr) + 1, dtype=int),
            "explained_variance": pca["eigenvalues"],
            "explained_variance_ratio": evr,
            "cumulative_explained_variance": cumulative,
        }
    )
    rows = []
    for target in VARIANCE_TARGETS:
        n = int(np.searchsorted(cumulative, target, side="left") + 1)
        rows.append({"variance_target": float(target), "n_components_required": n, "achieved_cumulative_variance": float(cumulative[n - 1])})
    return variance, pd.DataFrame(rows)


def choose_representative_indices(meta: pd.DataFrame) -> Dict[str, int]:
    medoids = pd.read_csv(EXP09_DIR / "representative_medoid_profiles.csv")
    chosen: Dict[str, int] = {}
    for row in medoids.itertuples(index=False):
        tid = int(row.track_id)
        target_x = float(row.medoid_x_mm)
        subset = meta[meta["track_id"] == tid]
        idx = int((subset["x_mm"] - target_x).abs().idxmin())
        chosen[f"track_{tid}_medoid"] = idx

    low = pd.read_csv(EXP09_DIR / "low_coherence_longitudinal_regimes.csv")
    for row in low.itertuples(index=False):
        tid = int(row.track_id)
        xmid = 0.5 * (float(row.x_min_mm) + float(row.x_max_mm))
        subset = meta[meta["track_id"] == tid]
        idx = int((subset["x_mm"] - xmid).abs().idxmin())
        chosen[str(row.regime_id)] = idx
    return chosen


def representative_error_table(X: np.ndarray, meta: pd.DataFrame, pca: Dict[str, np.ndarray], reps: Dict[str, int]) -> pd.DataFrame:
    rows = []
    ks = RECONSTRUCTION_KS + [pca["components"].shape[0]]
    for label, idx in reps.items():
        for k in ks:
            xhat = reconstruct(pca, pca["scores"][[idx], :], k)[0]
            rmse = float(np.sqrt(np.mean((X[idx] - xhat) ** 2)))
            rows.append(
                {
                    "representative_id": label,
                    "track_id": int(meta.loc[idx, "track_id"]),
                    "x_mm": float(meta.loc[idx, "x_mm"]),
                    "n_components": int(k),
                    "rmse": rmse,
                }
            )
    return pd.DataFrame(rows)


def track10_regime_preservation(X: np.ndarray, meta: pd.DataFrame, pca: Dict[str, np.ndarray], reps: Dict[str, int]) -> pd.DataFrame:
    medoid_idx = reps["track_10_medoid"]
    rows = []
    regime_items = [(label, idx) for label, idx in reps.items() if label.startswith("track_10_lowcoherence")]
    for label, idx in regime_items:
        for k in [3, 5, pca["components"].shape[0]]:
            x_med = reconstruct(pca, pca["scores"][[medoid_idx], :], k)[0]
            x_reg = reconstruct(pca, pca["scores"][[idx], :], k)[0]
            corr = float(np.corrcoef(x_med, x_reg)[0, 1])
            rows.append(
                {
                    "regime_id": label,
                    "regime_x_mm": float(meta.loc[idx, "x_mm"]),
                    "track10_medoid_x_mm": float(meta.loc[medoid_idx, "x_mm"]),
                    "n_components": int(k),
                    "correlation_to_track10_medoid_after_reconstruction": corr,
                    "remains_distinct_from_medoid_under_threshold_0p5": bool(corr < 0.5),
                }
            )
    return pd.DataFrame(rows)


def interpret_components(pca: Dict[str, np.ndarray], y: np.ndarray, n: int = N_COMPONENTS_TO_INTERPRET) -> pd.DataFrame:
    rows = []
    center = 0.5 * (float(y[0]) + float(y[-1]))
    left = y <= center
    right = y > center
    mid = (y >= center - 0.10) & (y <= center + 0.10)
    edges = ~mid
    for i in range(min(n, pca["components"].shape[0])):
        c = pca["components"][i]
        left_mean = float(np.mean(c[left]))
        right_mean = float(np.mean(c[right]))
        mid_mean = float(np.mean(c[mid])) if np.any(mid) else float("nan")
        edge_mean = float(np.mean(c[edges])) if np.any(edges) else float("nan")
        sign_changes = int(np.sum(np.diff(np.sign(c)) != 0))
        if abs(left_mean - right_mean) > 0.12:
            qualitative = "possible left-right asymmetry / lateral shape contrast"
        elif abs(mid_mean - edge_mean) > 0.12:
            qualitative = "possible center-versus-shoulder contrast"
        elif sign_changes >= 3:
            qualitative = "multi-lobed component; interpretation unclear beyond shape complexity"
        else:
            qualitative = "smooth broad shape variation; exact physical interpretation unclear"
        rows.append(
            {
                "component": int(i + 1),
                "explained_variance_ratio": float(pca["explained_variance_ratio"][i]),
                "left_mean_loading": left_mean,
                "right_mean_loading": right_mean,
                "center_mean_loading": mid_mean,
                "edge_mean_loading": edge_mean,
                "sign_change_count": sign_changes,
                "qualitative_interpretation": qualitative,
                "caution": "PCA signs are arbitrary; interpretation is qualitative and should not be over-read.",
            }
        )
    return pd.DataFrame(rows)


def assessment_against_requirements(variance_thresholds: pd.DataFrame, recon: pd.DataFrame, track10: pd.DataFrame) -> Dict[str, object]:
    n95 = int(variance_thresholds.loc[np.isclose(variance_thresholds["variance_target"], 0.95), "n_components_required"].iloc[0])
    err5 = recon[(recon["group"] == "all_tracks") & (recon["n_components"] == 5)]["rmse_p50"].iloc[0]
    t10_distinct_5 = track10[track10["n_components"] == 5]["remains_distinct_from_medoid_under_threshold_0p5"].mean()
    return {
        "normalized_shape_requirement": "PCA directly represents offset/amplitude-normalized shape from Experiment 09.",
        "amplitude_or_relief_requirement": "Not satisfied by PCA scores alone because profiles were amplitude-normalized before PCA; amplitude must remain a companion descriptor.",
        "finite_support_validity_requirement": "Not satisfied by PCA scores alone; PCA-ready subset excludes missing values and must retain support/eligibility metadata.",
        "signed_elevation_requirement": "Not satisfied by PCA scores alone because vertical offset was removed before PCA; signed elevation must remain companion metadata.",
        "shape_compression_observation": f"95% variance requires {n95} components; median 5-component reconstruction RMSE is {err5:.4f} normalized units.",
        "track10_regime_preservation_observation": f"Fraction of Track 10 low-coherence representatives remaining below correlation 0.5 to the Track 10 medoid with 5 PCs: {t10_distinct_5:.3f}.",
        "overall_assessment": "PCA is plausible for the normalized-shape component of a descriptor, but it does not by itself satisfy Experiment 10 because amplitude, signed elevation, finite-support validity, and regime metadata are external requirements.",
    }


def plot_variance(out_path: Path, variance: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(variance["component"].iloc[:20], variance["explained_variance_ratio"].iloc[:20], alpha=0.7, label="individual")
    ax.plot(variance["component"].iloc[:20], variance["cumulative_explained_variance"].iloc[:20], marker="o", color="tab:red", label="cumulative")
    ax.set_xlabel("principal component")
    ax.set_ylabel("explained variance ratio")
    ax.set_title("PCA explained variance")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_reconstruction_error(out_path: Path, errors: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for group, g in errors.groupby("group"):
        if group == "all_tracks" or group.startswith("track_"):
            ax.plot(g["n_components"], g["rmse_p50"], marker="o", lw=1.2, label=group)
    ax.set_xlabel("retained principal components")
    ax.set_ylabel("median reconstruction RMSE (normalized units)")
    ax.set_title("PCA reconstruction error")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_components(out_path: Path, y: np.ndarray, pca: Dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for i in range(min(5, pca["components"].shape[0])):
        ax.plot(y, pca["components"][i], lw=1.2, label=f"PC{i+1} ({pca['explained_variance_ratio'][i]:.2%})")
    ax.axhline(0, color="black", ls="--", lw=0.8)
    ax.set_xlabel("y in inherited central corridor (mm)")
    ax.set_ylabel("loading")
    ax.set_title("Leading PCA component loadings")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_representative_reconstructions(out_dir: Path, X: np.ndarray, y: np.ndarray, meta: pd.DataFrame, pca: Dict[str, np.ndarray], reps: Dict[str, int]) -> List[str]:
    made = []
    for label, idx in reps.items():
        if not label.endswith("medoid"):
            continue
        ks = RECONSTRUCTION_KS + [pca["components"].shape[0]]
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.plot(y, X[idx], color="black", lw=2.0, label="original normalized profile")
        for k in ks:
            xhat = reconstruct(pca, pca["scores"][[idx], :], k)[0]
            name = "all retained" if k == pca["components"].shape[0] else f"{k} PC"
            ax.plot(y, xhat, lw=1.0, label=name)
        ax.axhline(0, color="0.4", ls="--", lw=0.7)
        ax.set_title(f"{label}: PCA reconstruction at x={meta.loc[idx, 'x_mm']:.3f} mm")
        ax.set_xlabel("y in inherited central corridor (mm)")
        ax.set_ylabel("normalized profile")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        p = out_dir / f"{label}_pca_reconstructions.png"
        fig.savefig(p, dpi=180)
        plt.close(fig)
        made.append(str(p))
    return made


def plot_track10_regime_reconstructions(out_path: Path, X: np.ndarray, y: np.ndarray, meta: pd.DataFrame, pca: Dict[str, np.ndarray], reps: Dict[str, int]) -> None:
    items = [(label, idx) for label, idx in reps.items() if label == "track_10_medoid" or label.startswith("track_10_lowcoherence")]
    items = items[:7]
    fig, axes = plt.subplots(len(items), 1, figsize=(8, 2.0 * len(items)), sharex=True)
    if len(items) == 1:
        axes = [axes]
    for ax, (label, idx) in zip(axes, items):
        ax.plot(y, X[idx], color="black", lw=1.4, label="original")
        ax.plot(y, reconstruct(pca, pca["scores"][[idx], :], 5)[0], color="tab:red", lw=1.1, label="5 PC reconstruction")
        ax.set_title(f"{label} | x={meta.loc[idx, 'x_mm']:.3f} mm", fontsize=9)
        ax.grid(True, alpha=0.2)
    axes[-1].set_xlabel("y in inherited central corridor (mm)")
    axes[0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPO_ROOT / "processed_data" / "run_outputs" / f"11_pca_representation_evaluation_{run_tag}"
    table_dir = out_dir / "tables"
    fig_dir = out_dir / "figures"
    for d in (table_dir, fig_dir):
        d.mkdir(parents=True, exist_ok=True)

    X, y_grid, meta = load_experiment09_profiles()
    X_ready, y_ready, meta_ready, ready_info = make_pca_ready_matrix(X, y_grid, meta)
    pca = compute_pca_svd(X_ready)
    variance, thresholds = variance_tables(pca)
    errors = reconstruction_error_table(X_ready, meta_ready, pca)
    reps = choose_representative_indices(meta_ready)
    rep_errors = representative_error_table(X_ready, meta_ready, pca, reps)
    track10_regime = track10_regime_preservation(X_ready, meta_ready, pca, reps)
    pc_interp = interpret_components(pca, y_ready)
    req_assessment = assessment_against_requirements(thresholds, errors, track10_regime)

    # Save score table for first components only to keep file compact and auditable.
    score_cols = {f"pc{i+1}_score": pca["scores"][:, i] for i in range(min(N_SCORE_COMPONENTS_TO_SAVE, pca["scores"].shape[1]))}
    score_df = pd.concat([meta_ready.reset_index(drop=True), pd.DataFrame(score_cols)], axis=1)

    meta_ready.to_csv(table_dir / "pca_ready_profile_metadata.csv", index=False)
    variance.to_csv(table_dir / "pca_variance_summary.csv", index=False)
    thresholds.to_csv(table_dir / "pca_component_count_thresholds.csv", index=False)
    errors.to_csv(table_dir / "reconstruction_error_by_components.csv", index=False)
    rep_errors.to_csv(table_dir / "representative_reconstruction_errors.csv", index=False)
    track10_regime.to_csv(table_dir / "track10_regime_preservation.csv", index=False)
    pc_interp.to_csv(table_dir / "pc_qualitative_interpretation.csv", index=False)
    score_df.to_csv(table_dir / "pca_profile_scores_first10.csv", index=False)
    np.savez_compressed(
        table_dir / "pca_model_arrays.npz",
        y_grid_mm=y_ready,
        mean_profile=pca["mean"],
        components=pca["components"],
        explained_variance_ratio=pca["explained_variance_ratio"],
        scores_first10=pca["scores"][:, : min(N_SCORE_COMPONENTS_TO_SAVE, pca["scores"].shape[1])],
    )
    (table_dir / "assessment_against_experiment10_requirements.json").write_text(json.dumps(req_assessment, indent=2), encoding="utf-8")

    plot_variance(fig_dir / "pca_explained_variance.png", variance)
    plot_reconstruction_error(fig_dir / "pca_reconstruction_error_curve.png", errors)
    plot_components(fig_dir / "leading_pca_components.png", y_ready, pca)
    made_recon = plot_representative_reconstructions(fig_dir, X_ready, y_ready, meta_ready, pca, reps)
    plot_track10_regime_reconstructions(fig_dir / "track10_regime_pca_reconstructions.png", X_ready, y_ready, meta_ready, pca, reps)

    metadata = {
        "experiment": "11_pca_representation_evaluation",
        "out_dir": str(out_dir),
        "tracks": TRACK_IDS,
        "sealed_tracks": sorted(SEALED_TRACKS),
        "purpose": "Evaluate PCA only as a candidate implementation for normalized cross-sectional shape representation.",
        "source_experiment09_dir": str(EXP09_DIR),
        "source_experiment10_dir": str(EXP10_DIR),
        "pca_ready_matrix": ready_info,
        "pca_method": "NumPy SVD on feature-mean-centered, Experiment-09 normalized profiles; no NaN filling; complete rows over retained high-support y-grid only.",
        "reconstruction_components_evaluated": RECONSTRUCTION_KS + [int(pca["components"].shape[0])],
        "outputs": {
            "tables": sorted([p.name for p in table_dir.iterdir()]),
            "figures": sorted([p.name for p in fig_dir.iterdir()]),
        },
    }
    (table_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Wrote Experiment 11 outputs to: {out_dir}")
    print("Variance targets:")
    print(thresholds.to_string(index=False))
    print("Assessment:")
    print(req_assessment["overall_assessment"])


if __name__ == "__main__":
    main()