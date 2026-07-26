"""Experiment 30: core-width learnability diagnostic.

Locked diagnostic: test whether the already-computed Experiment 29
local_core_width_mm target contains learnable signal from thermal physics
features under development-only Leave-One-Track-Out validation.

Track 21 remains sealed. This script reads only development-track rows from the
multimodal CSV prefix expected for Tracks 8, 10, and 14, and never evaluates or
uses Track 21 for scaling, training, validation, or diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import csv
import json
import math
import platform
import subprocess
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_ARTIFACT_PATH = (
    REPO_ROOT
    / "processed_data"
    / "run_outputs"
    / "29_dominant_component_core_geometry_20260724_174420"
    / "tables"
    / "anchor_aggregated_width.csv"
)
DATASET_PATH = REPO_ROOT / "processed_data" / "final_multimodal_dataset.csv"
OUTPUT_ROOT = REPO_ROOT / "processed_data" / "run_outputs"

EXPERIMENT_NAME = "30_core_width_learnability_diagnostic"
DEVELOPMENT_TRACKS = (8, 10, 14)
DEVELOPMENT_TRACK_SET = set(DEVELOPMENT_TRACKS)
SEALED_TRACKS = {21}
EXPECTED_ROWS_PER_DEVELOPMENT_TRACK = 400
EXPECTED_DEVELOPMENT_ROWS = EXPECTED_ROWS_PER_DEVELOPMENT_TRACK * len(DEVELOPMENT_TRACKS)
assert DEVELOPMENT_TRACK_SET == {8, 10, 14}
assert not (DEVELOPMENT_TRACK_SET & SEALED_TRACKS), "Track 21 is sealed and cannot be a development track."

FEATURE_BASE_COLUMNS = ["peak_temp", "sqrt_mp_area", "mp_length"]
WINDOW_OFFSETS = (-2, -1, 0, 1, 2)
WINDOW_FEATURE_COLUMNS = [f"{feature}_t{offset:+d}" for offset in WINDOW_OFFSETS for feature in FEATURE_BASE_COLUMNS]
PROHIBITED_MODEL_COLUMNS = {
    "x_position_mm",
    "x_norm",
    "frame_index",
    "track_id",
    "sem_tile_index",
    "substrate_roughness_variance",
    "substrate_mean_intensity",
}
FOLDS = [
    {"fold": "holdout_8", "train_tracks": [10, 14], "val_track": 8},
    {"fold": "holdout_10", "train_tracks": [8, 14], "val_track": 10},
    {"fold": "holdout_14", "train_tracks": [8, 10], "val_track": 14},
]

warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass(frozen=True)
class SampleBuildDiagnostics:
    target_rows_total: int
    target_rows_dev: int
    target_anchor_valid_true: int
    target_finite_valid: int
    joined_valid_targets: int
    samples_after_windowing: int
    discarded_nonfinite_target: int
    discarded_missing_join: int
    discarded_incomplete_window: int
    discarded_nonfinite_features: int
    samples_by_track: Dict[int, int]
    valid_targets_by_track: Dict[int, int]


def git_text(args: Sequence[str]) -> str:
    for git_cmd in ("git", "/usr/bin/git"):
        try:
            return subprocess.check_output([git_cmd, *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            continue
    return "unavailable"


def make_json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def read_development_dataset_prefix(dataset_path: Path) -> pd.DataFrame:
    """Read only the expected development-track prefix from the multimodal CSV.

    The repository dataset is organized with development tracks 8, 10, and 14 in
    the first 1200 data rows. To keep Track 21 sealed, this function stops after
    those 1200 rows and asserts that exactly Tracks 8, 10, and 14 were read.
    """

    required = {"track_id", "frame_index", "x_position_mm", "peak_temp", "mp_area_px", "mp_length"}
    rows: List[Dict[str, float | int]] = []
    with dataset_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        missing = sorted(required - fieldnames)
        if missing:
            raise RuntimeError(f"Dataset missing required columns: {missing}")
        for row_number, row in enumerate(reader, start=1):
            if row_number > EXPECTED_DEVELOPMENT_ROWS:
                break
            track_id = int(float(row["track_id"]))
            frame_index = int(float(row["frame_index"]))
            if track_id not in DEVELOPMENT_TRACK_SET:
                raise RuntimeError(
                    f"Encountered non-development track {track_id} within the locked development prefix; "
                    "stopping to preserve Track 21 sealing."
                )
            mp_area_px = float(row["mp_area_px"])
            sqrt_mp_area = math.sqrt(mp_area_px) if math.isfinite(mp_area_px) and mp_area_px >= 0 else float("nan")
            rows.append(
                {
                    "track_id": track_id,
                    "frame_index": frame_index,
                    "x_position_mm": float(row["x_position_mm"]),
                    "peak_temp": float(row["peak_temp"]),
                    "sqrt_mp_area": sqrt_mp_area,
                    "mp_length": float(row["mp_length"]),
                }
            )

    df = pd.DataFrame(rows)
    if len(df) != EXPECTED_DEVELOPMENT_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_DEVELOPMENT_ROWS} development rows; read {len(df)}.")
    observed_tracks = set(int(t) for t in df["track_id"].unique())
    if observed_tracks != DEVELOPMENT_TRACK_SET:
        raise RuntimeError(f"Expected development tracks {sorted(DEVELOPMENT_TRACK_SET)}; observed {sorted(observed_tracks)}.")
    counts = df.groupby("track_id").size().to_dict()
    for track_id in DEVELOPMENT_TRACKS:
        if int(counts.get(track_id, 0)) != EXPECTED_ROWS_PER_DEVELOPMENT_TRACK:
            raise RuntimeError(f"Track {track_id}: expected 400 dataset rows; observed {counts.get(track_id, 0)}.")
    if 21 in observed_tracks:
        raise RuntimeError("Track 21 was loaded, violating the sealed-track guard.")
    return df.sort_values(["track_id", "frame_index"]).reset_index(drop=True)


def read_target_artifact(target_path: Path) -> pd.DataFrame:
    required = {"track_id", "frame_index", "local_core_width_mm", "anchor_valid"}
    df = pd.read_csv(target_path)
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Target artifact missing required columns: {missing}")
    observed_tracks = set(int(t) for t in df["track_id"].unique())
    if observed_tracks != DEVELOPMENT_TRACK_SET:
        raise RuntimeError(f"Target artifact track set mismatch: observed {sorted(observed_tracks)}.")
    if observed_tracks & SEALED_TRACKS:
        raise RuntimeError("Target artifact unexpectedly contains sealed Track 21 rows.")
    return df


def build_centered_window_samples(dataset_df: pd.DataFrame, target_df: pd.DataFrame) -> Tuple[pd.DataFrame, SampleBuildDiagnostics]:
    valid_target_mask = target_df["anchor_valid"].astype(bool)
    valid_target_df = target_df.loc[valid_target_mask].copy()
    finite_target_mask = np.isfinite(valid_target_df["local_core_width_mm"].to_numpy(dtype=float))
    discarded_nonfinite_target = int((~finite_target_mask).sum())
    valid_target_df = valid_target_df.loc[finite_target_mask].copy()

    dataset_index = {
        (int(row.track_id), int(row.frame_index)): row
        for row in dataset_df.itertuples(index=False)
    }

    records: List[Dict[str, float | int | str]] = []
    discarded_missing_join = 0
    discarded_incomplete_window = 0
    discarded_nonfinite_features = 0

    for target_row in valid_target_df.itertuples(index=False):
        track_id = int(target_row.track_id)
        frame_index = int(target_row.frame_index)
        if track_id not in DEVELOPMENT_TRACK_SET:
            raise RuntimeError(f"Non-development track encountered during sample construction: {track_id}")
        anchor_key = (track_id, frame_index)
        anchor_data = dataset_index.get(anchor_key)
        if anchor_data is None:
            discarded_missing_join += 1
            continue

        flattened: List[float] = []
        complete_window = True
        for offset in WINDOW_OFFSETS:
            neighbor_key = (track_id, frame_index + offset)
            neighbor = dataset_index.get(neighbor_key)
            if neighbor is None:
                complete_window = False
                break
            flattened.extend([float(neighbor.peak_temp), float(neighbor.sqrt_mp_area), float(neighbor.mp_length)])
        if not complete_window:
            discarded_incomplete_window += 1
            continue
        if not np.isfinite(np.asarray(flattened, dtype=float)).all():
            discarded_nonfinite_features += 1
            continue

        rec: Dict[str, float | int | str] = {
            "track_id": track_id,
            "frame_index": frame_index,
            "x_position_mm": float(anchor_data.x_position_mm),
            "actual_local_core_width_mm": float(target_row.local_core_width_mm),
        }
        for col, value in zip(WINDOW_FEATURE_COLUMNS, flattened):
            rec[col] = float(value)
        records.append(rec)

    samples = pd.DataFrame(records)
    if samples.empty:
        raise RuntimeError("No valid centered-window samples were constructed.")
    observed_tracks = set(int(t) for t in samples["track_id"].unique())
    if observed_tracks != DEVELOPMENT_TRACK_SET:
        raise RuntimeError(f"Windowed sample track set mismatch: observed {sorted(observed_tracks)}.")
    if PROHIBITED_MODEL_COLUMNS & set(WINDOW_FEATURE_COLUMNS):
        raise RuntimeError(f"Prohibited model columns entered feature matrix: {sorted(PROHIBITED_MODEL_COLUMNS & set(WINDOW_FEATURE_COLUMNS))}")

    diagnostics = SampleBuildDiagnostics(
        target_rows_total=int(len(target_df)),
        target_rows_dev=int(len(target_df[target_df["track_id"].isin(DEVELOPMENT_TRACKS)])),
        target_anchor_valid_true=int(valid_target_mask.sum()),
        target_finite_valid=int(len(valid_target_df)),
        joined_valid_targets=int(len(valid_target_df) - discarded_missing_join),
        samples_after_windowing=int(len(samples)),
        discarded_nonfinite_target=discarded_nonfinite_target,
        discarded_missing_join=discarded_missing_join,
        discarded_incomplete_window=discarded_incomplete_window,
        discarded_nonfinite_features=discarded_nonfinite_features,
        samples_by_track={int(k): int(v) for k, v in samples.groupby("track_id").size().to_dict().items()},
        valid_targets_by_track={int(k): int(v) for k, v in valid_target_df.groupby("track_id").size().to_dict().items()},
    )
    return samples.sort_values(["track_id", "frame_index"]).reset_index(drop=True), diagnostics


def pearson_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    if float(np.std(y_true)) == 0.0 or float(np.std(y_pred)) == 0.0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "n_valid_samples": int(len(y_true)),
        "target_mean": float(np.mean(y_true)),
        "target_std": float(np.std(y_true, ddof=1)) if len(y_true) > 1 else 0.0,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "median_absolute_error": float(median_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "pearson_r": pearson_corr(y_true, y_pred),
    }


def run_loto(samples: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    fold_rows: List[Dict[str, object]] = []
    prediction_frames: List[pd.DataFrame] = []
    scaler_fit_summary: Dict[str, object] = {}

    if set(int(t) for t in samples["track_id"].unique()) != DEVELOPMENT_TRACK_SET:
        raise RuntimeError("Sample set is not exactly Tracks 8, 10, and 14.")

    for fold in FOLDS:
        fold_name = str(fold["fold"])
        train_tracks = [int(t) for t in fold["train_tracks"]]
        val_track = int(fold["val_track"])
        if set(train_tracks + [val_track]) != DEVELOPMENT_TRACK_SET:
            raise RuntimeError(f"Fold {fold_name} does not use exactly the development tracks.")
        if val_track in train_tracks:
            raise RuntimeError(f"Fold {fold_name} has validation track in training tracks.")

        train_df = samples[samples["track_id"].isin(train_tracks)].copy()
        val_df = samples[samples["track_id"] == val_track].copy()
        if train_df.empty or val_df.empty:
            raise RuntimeError(f"Fold {fold_name}: empty train or validation split.")

        x_train = train_df[WINDOW_FEATURE_COLUMNS].to_numpy(dtype=float)
        y_train = train_df["actual_local_core_width_mm"].to_numpy(dtype=float)
        x_val = val_df[WINDOW_FEATURE_COLUMNS].to_numpy(dtype=float)
        y_val = val_df["actual_local_core_width_mm"].to_numpy(dtype=float)

        if not np.isfinite(x_train).all() or not np.isfinite(x_val).all() or not np.isfinite(y_train).all() or not np.isfinite(y_val).all():
            raise RuntimeError(f"Fold {fold_name}: nonfinite train/validation arrays.")

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_val_scaled = scaler.transform(x_val)
        model = BayesianRidge()
        default_params = BayesianRidge().get_params(deep=True)
        if model.get_params(deep=True) != default_params:
            raise RuntimeError("BayesianRidge parameters differ from sklearn defaults.")
        model.fit(x_train_scaled, y_train)
        y_pred = model.predict(x_val_scaled)
        if not np.isfinite(y_pred).all():
            raise RuntimeError(f"Fold {fold_name}: predictions contain nonfinite values.")

        metrics = compute_metrics(y_val, y_pred)
        fold_rows.append(
            {
                "fold": fold_name,
                "train_tracks": "+".join(str(t) for t in train_tracks),
                "val_track": val_track,
                **metrics,
            }
        )

        pred_df = val_df[["track_id", "frame_index", "x_position_mm", "actual_local_core_width_mm"]].copy()
        pred_df.insert(0, "fold", fold_name)
        pred_df["predicted_local_core_width_mm"] = y_pred
        pred_df["residual"] = pred_df["actual_local_core_width_mm"] - pred_df["predicted_local_core_width_mm"]
        prediction_frames.append(pred_df)

        scaler_fit_summary[fold_name] = {
            "train_tracks": train_tracks,
            "val_track": val_track,
            "n_train_samples": int(len(train_df)),
            "n_val_samples": int(len(val_df)),
            "scaler_n_features_in": int(scaler.n_features_in_),
            "scaler_fit_track_ids": sorted(int(t) for t in train_df["track_id"].unique()),
        }

    fold_metrics = pd.DataFrame(fold_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    y_true_pooled = predictions["actual_local_core_width_mm"].to_numpy(dtype=float)
    y_pred_pooled = predictions["predicted_local_core_width_mm"].to_numpy(dtype=float)
    pooled_row = {"scope": "pooled_oof_concatenated", **compute_metrics(y_true_pooled, y_pred_pooled)}
    positive_folds = int((fold_metrics["r2"] > 0).sum())
    pooled_r2 = float(pooled_row["r2"])
    if pooled_r2 > 0 and positive_folds >= 1:
        final_decision = "SIGNAL DETECTED"
    elif positive_folds >= 1 and pooled_r2 <= 0:
        final_decision = "WEAK SIGNAL"
    else:
        final_decision = "NO SIGNAL"
    pooled_row["positive_r2_folds"] = positive_folds
    pooled_row["final_decision"] = final_decision
    pooled_metrics = pd.DataFrame([pooled_row])

    expected_predictions = int(fold_metrics["n_valid_samples"].sum())
    if len(predictions) != expected_predictions:
        raise RuntimeError(f"Prediction row count mismatch: {len(predictions)} != {expected_predictions}")
    return fold_metrics, pooled_metrics, predictions, scaler_fit_summary


def plot_actual_vs_predicted(predictions: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=False, sharey=False)
    for ax, (fold_name, group) in zip(axes, predictions.groupby("fold", sort=False)):
        actual = group["actual_local_core_width_mm"].to_numpy(dtype=float)
        pred = group["predicted_local_core_width_mm"].to_numpy(dtype=float)
        ax.scatter(actual, pred, s=18, alpha=0.75)
        low = float(min(actual.min(), pred.min()))
        high = float(max(actual.max(), pred.max()))
        ax.plot([low, high], [low, high], color="black", linewidth=1, linestyle="--")
        ax.set_title(fold_name)
        ax.set_xlabel("Actual local_core_width_mm")
        ax.set_ylabel("Predicted local_core_width_mm")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_residual_vs_x(predictions: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=False, sharey=True)
    for ax, (fold_name, group) in zip(axes, predictions.groupby("fold", sort=False)):
        ax.axhline(0.0, color="black", linewidth=1, linestyle="--")
        ax.scatter(group["x_position_mm"], group["residual"], s=18, alpha=0.75)
        ax.set_title(fold_name)
        ax.set_xlabel("x_position_mm")
        ax.set_ylabel("Residual actual - predicted (mm)")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def write_outputs(
    output_dir: Path,
    fold_metrics: pd.DataFrame,
    pooled_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    diagnostics: SampleBuildDiagnostics,
    scaler_fit_summary: Dict[str, object],
) -> None:
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    metadata_dir = output_dir / "metadata"
    for path in (tables_dir, figures_dir, metadata_dir):
        path.mkdir(parents=True, exist_ok=True)

    fold_metrics.to_csv(tables_dir / "loto_fold_metrics.csv", index=False)
    pooled_metrics.to_csv(tables_dir / "loto_pooled_metrics.csv", index=False)
    predictions.to_csv(tables_dir / "per_fold_predictions.csv", index=False)
    plot_actual_vs_predicted(predictions, figures_dir / "actual_vs_predicted_per_fold.png")
    plot_residual_vs_x(predictions, figures_dir / "residual_vs_x_per_fold.png")

    final_decision = str(pooled_metrics.iloc[0]["final_decision"])
    metadata = {
        "experiment_name": EXPERIMENT_NAME,
        "timestamp": output_dir.name.replace(f"{EXPERIMENT_NAME}_", ""),
        "git_branch": git_text(["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_commit": git_text(["rev-parse", "HEAD"]),
        "python_version": sys.version,
        "platform": platform.platform(),
        "target_artifact_path": str(TARGET_ARTIFACT_PATH),
        "dataset_path": str(DATASET_PATH),
        "target_column": "local_core_width_mm",
        "target_filter": "anchor_valid == True and finite local_core_width_mm; no smoothing/modification/bridging/interpolation",
        "model": "sklearn.linear_model.BayesianRidge",
        "model_parameters": BayesianRidge().get_params(deep=True),
        "feature_list": FEATURE_BASE_COLUMNS,
        "window_feature_columns": WINDOW_FEATURE_COLUMNS,
        "window_definition": {
            "type": "centered",
            "offsets": list(WINDOW_OFFSETS),
            "n_frames": 5,
            "flattening": "for each offset t-2..t+2, append peak_temp, sqrt_mp_area, mp_length",
            "same_track_required": True,
        },
        "scaler": "sklearn.preprocessing.StandardScaler fit separately on each training fold only",
        "scaler_fit_summary": scaler_fit_summary,
        "folds": FOLDS,
        "valid_sample_counts": {
            "by_track": diagnostics.samples_by_track,
            "total": diagnostics.samples_after_windowing,
            "by_fold_validation": {str(r.fold): int(r.n_valid_samples) for r in fold_metrics.itertuples(index=False)},
        },
        "alignment_counts": diagnostics.__dict__,
        "track21_sealed_status": {
            "sealed_tracks": sorted(SEALED_TRACKS),
            "loaded_or_used": False,
            "guard": "Read only the first 1200 multimodal data rows and asserted exactly development tracks {8,10,14}; target artifact asserted to contain only {8,10,14}.",
        },
        "prohibited_inputs_status": {
            "sem_features_used": False,
            "coordinate_or_identity_metadata_used_as_model_features": False,
            "model_feature_columns": WINDOW_FEATURE_COLUMNS,
            "prohibited_model_columns": sorted(PROHIBITED_MODEL_COLUMNS),
        },
        "validation_checks": {
            "target_came_directly_from_exp29_artifact": True,
            "no_target_smoothing_or_modification": True,
            "centered_windows_pm2_frames": True,
            "windows_cross_track_boundaries": False,
            "bayesian_ridge_default_hyperparameters": True,
            "predictions_finite": bool(np.isfinite(predictions["predicted_local_core_width_mm"].to_numpy(dtype=float)).all()),
            "pooled_metrics_from_concatenated_oof_predictions": True,
            "existing_exp28_29_scripts_modified_by_this_script": False,
        },
        "final_decision_rule": {
            "SIGNAL DETECTED": "pooled OOF R2 > 0 AND at least 1 of 3 folds has R2 > 0",
            "WEAK SIGNAL": "at least 1 fold has R2 > 0 BUT pooled OOF R2 <= 0",
            "NO SIGNAL": "all 3 folds have R2 <= 0",
        },
        "final_decision": final_decision,
    }
    (metadata_dir / "run_metadata.json").write_text(json.dumps(make_json_safe(metadata), indent=2), encoding="utf-8")


def main() -> int:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / f"{EXPERIMENT_NAME}_{timestamp}"

    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Target artifact: {TARGET_ARTIFACT_PATH}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Development tracks: {list(DEVELOPMENT_TRACKS)}")
    print("Track 21 sealed: it is not loaded, used, evaluated, or used for scaling.")

    target_df = read_target_artifact(TARGET_ARTIFACT_PATH)
    dataset_df = read_development_dataset_prefix(DATASET_PATH)
    samples, diagnostics = build_centered_window_samples(dataset_df, target_df)
    fold_metrics, pooled_metrics, predictions, scaler_fit_summary = run_loto(samples)
    write_outputs(output_dir, fold_metrics, pooled_metrics, predictions, diagnostics, scaler_fit_summary)

    print("\nTarget/sample alignment counts:")
    print(json.dumps(make_json_safe(diagnostics.__dict__), indent=2))
    print("\nPer-fold metrics:")
    print(fold_metrics.to_string(index=False))
    print("\nPooled OOF metrics:")
    print(pooled_metrics.to_string(index=False))
    print(f"\nOutput directory: {output_dir}")
    print(f"Final locked classification: {pooled_metrics.iloc[0]['final_decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
