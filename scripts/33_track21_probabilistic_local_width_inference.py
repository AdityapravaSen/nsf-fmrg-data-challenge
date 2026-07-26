"""Experiment 33: final Track 21 probabilistic local-width inference.

This is deployment of the frozen Experiment 32 methodology, not a new model
selection experiment. The final BayesianRidge model is trained on all eligible
Experiment 32 development samples from Tracks 8, 10, and 14, then applied to
Track 21 thermal predictor windows only.

Track 21 target/Wyko geometry remains sealed. No Track 21 target columns are
loaded for inference, and no Track 21 evaluation metrics are computed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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
from scipy.stats import norm
from sklearn.linear_model import BayesianRidge
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
EXP32_OUTPUT_DIR = REPO_ROOT / "processed_data" / "run_outputs" / "32_probabilistic_local_width_evaluation_20260725_203529"
EXP32_METADATA_PATH = EXP32_OUTPUT_DIR / "metadata" / "run_metadata.json"
OUTPUT_ROOT = REPO_ROOT / "processed_data" / "run_outputs"

EXPERIMENT_NAME = "33_track21_probabilistic_local_width_inference"
DEVELOPMENT_TRACKS = (8, 10, 14)
DEVELOPMENT_TRACK_SET = set(DEVELOPMENT_TRACKS)
TRACK21_ID = 21
SEALED_TARGET_GEOMETRY_TRACKS = {21}
EXPECTED_ROWS_PER_DEVELOPMENT_TRACK = 400
EXPECTED_DEVELOPMENT_ROWS = EXPECTED_ROWS_PER_DEVELOPMENT_TRACK * len(DEVELOPMENT_TRACKS)
EXPECTED_TRACK21_ROWS = 400
EXPECTED_EXPERIMENT32_SAMPLE_COUNTS = {8: 383, 10: 354, 14: 377}
EXPECTED_EXPERIMENT32_TOTAL_TRAINING_SAMPLES = 1114

FEATURE_BASE_COLUMNS = ["peak_temp", "sqrt_mp_area", "mp_length"]
WINDOW_OFFSETS = (-2, -1, 0, 1, 2)
WINDOW_FEATURE_COLUMNS = [f"{feature}_t{offset:+d}" for offset in WINDOW_OFFSETS for feature in FEATURE_BASE_COLUMNS]
NOMINAL_LEVELS = (0.50, 0.80, 0.90, 0.95)
ALLOWED_MULTIMODAL_COLUMNS = ["track_id", "frame_index", "x_position_mm", "peak_temp", "mp_area_px", "mp_length"]
PROHIBITED_TRACK21_TARGET_COLUMNS = {
    "local_core_width_mm",
    "smoothed_macro_width_mm",
    "pc1",
    "pc2",
    "pc3",
    "pc4",
    "pc5",
    "amplitude_um",
    "signed_elevation_um",
    "heightmap_x_mm",
    "heightmap_x_delta_mm",
    "heightmap_x_index",
    "finite_fraction",
    "central_corridor_finite_fraction",
    "substrate_side_finite_fraction",
    "baseline_support_count",
    "shape_support_fraction_on_common_grid",
    "retained_pca_grid_finite_fraction",
    "shape_center_um_median",
    "shape_scale_um_mad",
}
assert DEVELOPMENT_TRACK_SET == {8, 10, 14}
assert TRACK21_ID not in DEVELOPMENT_TRACK_SET

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


@dataclass(frozen=True)
class Track21WindowDiagnostics:
    track21_rows_available: int
    prediction_windows_generated: int
    excluded_incomplete_window: int
    excluded_nonfinite_features: int
    excluded_rows: List[Dict[str, object]]


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
    if isinstance(obj, np.ndarray):
        return make_json_safe(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def summary_stats(values: np.ndarray) -> Dict[str, float | int]:
    arr = np.asarray(values, dtype=float)
    if not np.isfinite(arr).all():
        raise RuntimeError("Cannot summarize nonfinite values.")
    return {
        "n": int(arr.size),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
    }


def read_multimodal_allowed_columns(dataset_path: Path) -> pd.DataFrame:
    header = pd.read_csv(dataset_path, nrows=0).columns.tolist()
    missing = sorted(set(ALLOWED_MULTIMODAL_COLUMNS) - set(header))
    if missing:
        raise RuntimeError(f"Dataset missing required predictor/metadata columns: {missing}")
    loaded_forbidden = sorted(set(ALLOWED_MULTIMODAL_COLUMNS) & PROHIBITED_TRACK21_TARGET_COLUMNS)
    if loaded_forbidden:
        raise RuntimeError(f"Allowed multimodal columns unexpectedly include target-derived columns: {loaded_forbidden}")
    df = pd.read_csv(dataset_path, usecols=ALLOWED_MULTIMODAL_COLUMNS)
    df.insert(0, "source_row_index", np.arange(len(df), dtype=int))
    return df


def read_development_dataset_prefix(multimodal_allowed: pd.DataFrame) -> pd.DataFrame:
    dev = multimodal_allowed.iloc[:EXPECTED_DEVELOPMENT_ROWS].copy()
    if len(dev) != EXPECTED_DEVELOPMENT_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_DEVELOPMENT_ROWS} development rows; observed {len(dev)}.")
    observed = set(int(t) for t in dev["track_id"].unique())
    if observed != DEVELOPMENT_TRACK_SET:
        raise RuntimeError(f"Development prefix track mismatch: observed {sorted(observed)}.")
    if TRACK21_ID in observed:
        raise RuntimeError("Track 21 predictor rows appeared in development training prefix.")
    counts = dev.groupby("track_id").size().to_dict()
    for track_id in DEVELOPMENT_TRACKS:
        if int(counts.get(track_id, 0)) != EXPECTED_ROWS_PER_DEVELOPMENT_TRACK:
            raise RuntimeError(f"Track {track_id}: expected 400 development rows, observed {counts.get(track_id, 0)}.")
    return prepare_predictor_frame(dev)


def read_track21_predictor_rows(multimodal_allowed: pd.DataFrame) -> pd.DataFrame:
    track21 = multimodal_allowed[multimodal_allowed["track_id"].astype(int) == TRACK21_ID].copy()
    if len(track21) != EXPECTED_TRACK21_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_TRACK21_ROWS} Track 21 predictor rows; observed {len(track21)}.")
    observed = set(int(t) for t in track21["track_id"].unique())
    if observed != {TRACK21_ID}:
        raise RuntimeError(f"Track 21 predictor row selection mismatch: observed {sorted(observed)}.")
    return prepare_predictor_frame(track21)


def prepare_predictor_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["track_id"] = out["track_id"].astype(float).astype(int)
    out["frame_index"] = out["frame_index"].astype(float).astype(int)
    out["x_position_mm"] = out["x_position_mm"].astype(float)
    out["peak_temp"] = out["peak_temp"].astype(float)
    out["mp_area_px"] = out["mp_area_px"].astype(float)
    out["mp_length"] = out["mp_length"].astype(float)
    area = out["mp_area_px"].to_numpy(dtype=float)
    out["sqrt_mp_area"] = np.where(np.isfinite(area) & (area >= 0.0), np.sqrt(area), np.nan)
    return out.sort_values(["track_id", "frame_index", "source_row_index"]).reset_index(drop=True)


def read_target_artifact(target_path: Path) -> pd.DataFrame:
    required = {"track_id", "frame_index", "local_core_width_mm", "anchor_valid"}
    df = pd.read_csv(target_path)
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Target artifact missing required columns: {missing}")
    observed = set(int(t) for t in df["track_id"].unique())
    if observed != DEVELOPMENT_TRACK_SET:
        raise RuntimeError(f"Target artifact must contain only development tracks; observed {sorted(observed)}.")
    if observed & SEALED_TARGET_GEOMETRY_TRACKS:
        raise RuntimeError("Target artifact unexpectedly contains Track 21 target rows.")
    return df


def build_development_training_samples(dataset_df: pd.DataFrame, target_df: pd.DataFrame) -> Tuple[pd.DataFrame, SampleBuildDiagnostics]:
    valid_target_mask = target_df["anchor_valid"].astype(bool)
    valid_target_df = target_df.loc[valid_target_mask].copy()
    finite_target_mask = np.isfinite(valid_target_df["local_core_width_mm"].to_numpy(dtype=float))
    discarded_nonfinite_target = int((~finite_target_mask).sum())
    valid_target_df = valid_target_df.loc[finite_target_mask].copy()

    dataset_index = {
        (int(row.track_id), int(row.frame_index)): row
        for row in dataset_df.itertuples(index=False)
    }
    records: List[Dict[str, object]] = []
    discarded_missing_join = 0
    discarded_incomplete_window = 0
    discarded_nonfinite_features = 0

    for target_row in valid_target_df.itertuples(index=False):
        track_id = int(target_row.track_id)
        frame_index = int(target_row.frame_index)
        if track_id not in DEVELOPMENT_TRACK_SET:
            raise RuntimeError(f"Non-development target encountered: {track_id}")
        anchor_data = dataset_index.get((track_id, frame_index))
        if anchor_data is None:
            discarded_missing_join += 1
            continue
        flattened: List[float] = []
        complete_window = True
        for offset in WINDOW_OFFSETS:
            neighbor = dataset_index.get((track_id, frame_index + offset))
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
        rec: Dict[str, object] = {
            "track_id": track_id,
            "source_row_index": int(anchor_data.source_row_index),
            "frame_index": frame_index,
            "x_position_mm": float(anchor_data.x_position_mm),
            "y_train": float(target_row.local_core_width_mm),
        }
        for col, value in zip(WINDOW_FEATURE_COLUMNS, flattened):
            rec[col] = float(value)
        records.append(rec)

    samples = pd.DataFrame(records).sort_values(["track_id", "frame_index"]).reset_index(drop=True)
    if samples.empty:
        raise RuntimeError("No development training samples were constructed.")
    samples_by_track = {int(k): int(v) for k, v in samples.groupby("track_id").size().to_dict().items()}
    if samples_by_track != EXPECTED_EXPERIMENT32_SAMPLE_COUNTS:
        raise RuntimeError(f"Training sample counts changed from Experiment 32: {samples_by_track}")
    if len(samples) != EXPECTED_EXPERIMENT32_TOTAL_TRAINING_SAMPLES:
        raise RuntimeError(f"Expected {EXPECTED_EXPERIMENT32_TOTAL_TRAINING_SAMPLES} training samples; observed {len(samples)}.")

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
        samples_by_track=samples_by_track,
        valid_targets_by_track={int(k): int(v) for k, v in valid_target_df.groupby("track_id").size().to_dict().items()},
    )
    return samples, diagnostics


def build_track21_inference_windows(track21_df: pd.DataFrame) -> Tuple[pd.DataFrame, Track21WindowDiagnostics]:
    if set(int(t) for t in track21_df["track_id"].unique()) != {TRACK21_ID}:
        raise RuntimeError("Track 21 inference frame contains non-Track-21 rows.")
    dataset_index = {
        (int(row.track_id), int(row.frame_index)): row
        for row in track21_df.itertuples(index=False)
    }
    records: List[Dict[str, object]] = []
    excluded: List[Dict[str, object]] = []
    excluded_incomplete = 0
    excluded_nonfinite = 0

    for row in track21_df.sort_values(["frame_index", "source_row_index"]).itertuples(index=False):
        track_id = int(row.track_id)
        frame_index = int(row.frame_index)
        flattened: List[float] = []
        missing_neighbors: List[int] = []
        for offset in WINDOW_OFFSETS:
            neighbor = dataset_index.get((track_id, frame_index + offset))
            if neighbor is None:
                missing_neighbors.append(frame_index + offset)
            else:
                flattened.extend([float(neighbor.peak_temp), float(neighbor.sqrt_mp_area), float(neighbor.mp_length)])
        if missing_neighbors:
            excluded_incomplete += 1
            excluded.append(
                {
                    "source_row_index": int(row.source_row_index),
                    "track_id": track_id,
                    "frame_index": frame_index,
                    "x_position_mm": float(row.x_position_mm),
                    "reason": "incomplete_centered_5_frame_window",
                    "missing_neighbor_frame_indices": missing_neighbors,
                }
            )
            continue
        if not np.isfinite(np.asarray(flattened, dtype=float)).all():
            excluded_nonfinite += 1
            excluded.append(
                {
                    "source_row_index": int(row.source_row_index),
                    "track_id": track_id,
                    "frame_index": frame_index,
                    "x_position_mm": float(row.x_position_mm),
                    "reason": "nonfinite_predictor_feature_in_centered_window",
                }
            )
            continue
        rec: Dict[str, object] = {
            "track_id": track_id,
            "source_row_index": int(row.source_row_index),
            "frame_index": frame_index,
            "x_position_mm": float(row.x_position_mm),
        }
        for col, value in zip(WINDOW_FEATURE_COLUMNS, flattened):
            rec[col] = float(value)
        records.append(rec)

    windows = pd.DataFrame(records).sort_values(["frame_index", "source_row_index"]).reset_index(drop=True)
    windows.insert(0, "prediction_index", np.arange(len(windows), dtype=int))
    diagnostics = Track21WindowDiagnostics(
        track21_rows_available=int(len(track21_df)),
        prediction_windows_generated=int(len(windows)),
        excluded_incomplete_window=excluded_incomplete,
        excluded_nonfinite_features=excluded_nonfinite,
        excluded_rows=excluded,
    )
    if diagnostics.track21_rows_available != diagnostics.prediction_windows_generated + len(diagnostics.excluded_rows):
        raise RuntimeError("Track 21 row accounting failed.")
    return windows, diagnostics


def add_prediction_intervals(pred: pd.DataFrame) -> pd.DataFrame:
    out = pred.copy()
    for level in NOMINAL_LEVELS:
        pct = int(round(level * 100))
        z = float(norm.ppf((1.0 + level) / 2.0))
        out[f"predictive_{pct}_lower_mm"] = out["predictive_mean_local_core_width_mm"] - z * out["predictive_std_local_core_width_mm"]
        out[f"predictive_{pct}_upper_mm"] = out["predictive_mean_local_core_width_mm"] + z * out["predictive_std_local_core_width_mm"]
        out[f"predictive_{pct}_interval_width_mm"] = out[f"predictive_{pct}_upper_mm"] - out[f"predictive_{pct}_lower_mm"]
    return out


def train_and_predict(training_samples: pd.DataFrame, track21_windows: pd.DataFrame):
    if set(int(t) for t in training_samples["track_id"].unique()) != DEVELOPMENT_TRACK_SET:
        raise RuntimeError("Training samples are not exactly Tracks 8, 10, and 14.")
    if set(int(t) for t in track21_windows["track_id"].unique()) != {TRACK21_ID}:
        raise RuntimeError("Inference windows are not exactly Track 21.")
    x_train = training_samples[WINDOW_FEATURE_COLUMNS].to_numpy(dtype=float)
    y_train = training_samples["y_train"].to_numpy(dtype=float)
    x_track21 = track21_windows[WINDOW_FEATURE_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(x_train).all() or not np.isfinite(y_train).all() or not np.isfinite(x_track21).all():
        raise RuntimeError("Training or Track 21 feature arrays contain nonfinite values.")
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_track21_scaled = scaler.transform(x_track21)
    model = BayesianRidge()
    if model.get_params(deep=True) != BayesianRidge().get_params(deep=True):
        raise RuntimeError("BayesianRidge parameters differ from sklearn defaults.")
    model.fit(x_train_scaled, y_train)
    y_mean, y_std = model.predict(x_track21_scaled, return_std=True)
    if not np.isfinite(y_mean).all() or not np.isfinite(y_std).all() or not np.all(y_std > 0):
        raise RuntimeError("Track 21 predictive mean/std validation failed.")
    return scaler, model, y_mean, y_std


def plot_track21_predictions(pred: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    x = pred["x_position_mm"].to_numpy(dtype=float)
    mean = pred["predictive_mean_local_core_width_mm"].to_numpy(dtype=float)
    lo = pred["predictive_90_lower_mm"].to_numpy(dtype=float)
    hi = pred["predictive_90_upper_mm"].to_numpy(dtype=float)
    ax.fill_between(x, lo, hi, alpha=0.25, label="90% predictive interval")
    ax.plot(x, mean, color="C0", linewidth=1.8, label="predictive mean")
    ax.set_xlabel("Track 21 physical x position (mm)")
    ax.set_ylabel("Predicted local_core_width_mm")
    ax.set_title("Track 21 probabilistic local-width inference (no ground truth overlay)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def write_outputs(
    output_dir: Path,
    predictions: pd.DataFrame,
    predictive_mean: np.ndarray,
    predictive_std: np.ndarray,
    scaler: StandardScaler,
    model: BayesianRidge,
    training_samples: pd.DataFrame,
    training_diagnostics: SampleBuildDiagnostics,
    track21_diagnostics: Track21WindowDiagnostics,
    validation_checks: Dict[str, object],
) -> None:
    predictions_dir = output_dir / "predictions"
    arrays_dir = output_dir / "arrays"
    models_dir = output_dir / "models"
    metadata_dir = output_dir / "metadata"
    figures_dir = output_dir / "figures"
    for path in (predictions_dir, arrays_dir, models_dir, metadata_dir, figures_dir):
        path.mkdir(parents=True, exist_ok=True)

    predictions.to_csv(predictions_dir / "track21_probabilistic_local_width_predictions.csv", index=False)
    np.save(arrays_dir / "track21_predictive_mean.npy", predictive_mean)
    np.save(arrays_dir / "track21_predictive_std.npy", predictive_std)
    np.savez(
        arrays_dir / "track21_predictive_distribution.npz",
        predictive_mean=predictive_mean,
        predictive_std=predictive_std,
        prediction_index=predictions["prediction_index"].to_numpy(dtype=int),
        source_row_index=predictions["source_row_index"].to_numpy(dtype=int),
        frame_index=predictions["frame_index"].to_numpy(dtype=int),
        x_position_mm=predictions["x_position_mm"].to_numpy(dtype=float),
    )

    model_payload = {
        "class": "sklearn.linear_model.BayesianRidge",
        "parameters": model.get_params(deep=True),
        "alpha": float(model.alpha_),
        "lambda": float(model.lambda_),
        "intercept": float(model.intercept_),
        "coef": model.coef_.astype(float).tolist(),
        "sigma_shape": list(model.sigma_.shape),
        "sigma_diag": np.diag(model.sigma_).astype(float).tolist(),
        "feature_order": WINDOW_FEATURE_COLUMNS,
    }
    (models_dir / "bayesian_ridge_parameters.json").write_text(json.dumps(make_json_safe(model_payload), indent=2), encoding="utf-8")

    scaler_payload = {
        "class": "sklearn.preprocessing.StandardScaler",
        "fit_tracks_only": list(DEVELOPMENT_TRACKS),
        "n_features_in": int(scaler.n_features_in_),
        "feature_names": WINDOW_FEATURE_COLUMNS,
        "mean": scaler.mean_.astype(float).tolist(),
        "scale": scaler.scale_.astype(float).tolist(),
        "var": scaler.var_.astype(float).tolist(),
    }
    (models_dir / "feature_scaler_stats.json").write_text(json.dumps(make_json_safe(scaler_payload), indent=2), encoding="utf-8")

    mean_summary = summary_stats(predictive_mean)
    std_summary = summary_stats(predictive_std)
    interval_summary = {
        f"mean_{int(level * 100)}pct_interval_width_mm": float(predictions[f"predictive_{int(level * 100)}_interval_width_mm"].mean())
        for level in NOMINAL_LEVELS
    }

    training_metadata = {
        "training_tracks": list(DEVELOPMENT_TRACKS),
        "training_sample_count": int(len(training_samples)),
        "training_sample_counts_by_track": {int(k): int(v) for k, v in training_samples.groupby("track_id").size().to_dict().items()},
        "target_artifact_path": str(TARGET_ARTIFACT_PATH),
        "target_column": "local_core_width_mm",
        "target_filter": "anchor_valid == True and finite local_core_width_mm; no smoothing/modification/bridging/interpolation",
        "feature_order": WINDOW_FEATURE_COLUMNS,
        "training_diagnostics": training_diagnostics.__dict__,
        "scaler_fit_on_development_only": True,
        "model_fit_on_development_only": True,
        "track21_used_for_training": False,
    }
    (metadata_dir / "training_metadata.json").write_text(json.dumps(make_json_safe(training_metadata), indent=2), encoding="utf-8")

    track21_metadata = {
        "track_id": TRACK21_ID,
        "track21_predictor_rows_available": track21_diagnostics.track21_rows_available,
        "track21_predictions_generated": track21_diagnostics.prediction_windows_generated,
        "prediction_ordering": "sorted by frame_index then source_row_index; prediction_index assigned after sorting",
        "predictor_columns_loaded_from_multimodal_csv": ALLOWED_MULTIMODAL_COLUMNS,
        "target_derived_columns_not_loaded_for_track21_inference": sorted(PROHIBITED_TRACK21_TARGET_COLUMNS),
        "excluded_incomplete_window": track21_diagnostics.excluded_incomplete_window,
        "excluded_nonfinite_features": track21_diagnostics.excluded_nonfinite_features,
        "excluded_rows": track21_diagnostics.excluded_rows,
        "predictive_mean_summary": mean_summary,
        "predictive_std_summary": std_summary,
        "interval_width_summary": interval_summary,
        "track21_wyko_geometry_loaded": False,
        "track21_targets_loaded": False,
        "track21_metrics_computed": False,
    }
    (metadata_dir / "track21_inference_metadata.json").write_text(json.dumps(make_json_safe(track21_metadata), indent=2), encoding="utf-8")

    run_metadata = {
        "experiment_name": EXPERIMENT_NAME,
        "stage_type": "final deployment/inference of frozen Experiment 32 methodology; not model selection",
        "timestamp": output_dir.name.replace(f"{EXPERIMENT_NAME}_", ""),
        "git_branch": git_text(["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_commit": git_text(["rev-parse", "HEAD"]),
        "git_status_short_at_run": git_text(["status", "--short"]),
        "python_version": sys.version,
        "platform": platform.platform(),
        "dataset_path": str(DATASET_PATH),
        "experiment32_output_dir": str(EXP32_OUTPUT_DIR),
        "experiment32_metadata_path": str(EXP32_METADATA_PATH),
        "experiment32_final_decision": "PASS",
        "methodology": {
            "target": "local_core_width_mm from Experiment 29 artifact",
            "model": "sklearn.linear_model.BayesianRidge default hyperparameters",
            "uncertainty": "BayesianRidge native posterior predictive std from predict(return_std=True)",
            "features": FEATURE_BASE_COLUMNS,
            "feature_order": WINDOW_FEATURE_COLUMNS,
            "window_offsets": list(WINDOW_OFFSETS),
            "scaler": "StandardScaler fit on all eligible development samples only",
            "no_uncertainty_recalibration_or_inflation": True,
        },
        "track21_sealing_status": {
            "track21_predictor_information_used_for_final_inference": True,
            "track21_target_geometry_loaded": False,
            "track21_wyko_height_maps_loaded": False,
            "track21_local_core_width_loaded": False,
            "track21_target_derived_columns_loaded_for_inference": False,
            "track21_evaluation_metrics_computed": False,
        },
        "prediction_artifacts": {
            "csv": str(output_dir / "predictions" / "track21_probabilistic_local_width_predictions.csv"),
            "mean_npy": str(output_dir / "arrays" / "track21_predictive_mean.npy"),
            "std_npy": str(output_dir / "arrays" / "track21_predictive_std.npy"),
            "npz": str(output_dir / "arrays" / "track21_predictive_distribution.npz"),
        },
        "validation_checks": validation_checks,
        "predictive_mean_summary": mean_summary,
        "predictive_std_summary": std_summary,
        "interval_width_summary": interval_summary,
    }
    (metadata_dir / "run_metadata.json").write_text(json.dumps(make_json_safe(run_metadata), indent=2), encoding="utf-8")

    plot_track21_predictions(predictions, figures_dir / "track21_predictive_mean_90pct_interval.png")


def validate_exports(output_dir: Path, predictions: pd.DataFrame, predictive_mean: np.ndarray, predictive_std: np.ndarray) -> Dict[str, object]:
    csv_path = output_dir / "predictions" / "track21_probabilistic_local_width_predictions.csv"
    mean_path = output_dir / "arrays" / "track21_predictive_mean.npy"
    std_path = output_dir / "arrays" / "track21_predictive_std.npy"
    npz_path = output_dir / "arrays" / "track21_predictive_distribution.npz"
    csv_df = pd.read_csv(csv_path)
    mean_npy = np.load(mean_path)
    std_npy = np.load(std_path)
    npz = np.load(npz_path)
    checks = {
        "track21_target_wyko_geometry_never_loaded": True,
        "training_contains_only_tracks_8_10_14": True,
        "track21_used_only_for_predictor_inference": True,
        "scaler_fit_uses_development_data_only": True,
        "model_fit_uses_development_data_only": True,
        "feature_order_exactly_matches_experiment32": True,
        "all_exported_predictive_means_finite": bool(np.isfinite(predictive_mean).all()),
        "all_exported_predictive_stds_finite": bool(np.isfinite(predictive_std).all()),
        "all_exported_predictive_stds_strictly_positive": bool((predictive_std > 0).all()),
        "csv_and_npy_predictive_mean_match": bool(np.allclose(csv_df["predictive_mean_local_core_width_mm"].to_numpy(dtype=float), mean_npy)),
        "csv_and_npy_predictive_std_match": bool(np.allclose(csv_df["predictive_std_local_core_width_mm"].to_numpy(dtype=float), std_npy)),
        "npz_mean_matches_csv_and_npy": bool(np.allclose(npz["predictive_mean"], csv_df["predictive_mean_local_core_width_mm"].to_numpy(dtype=float)) and np.allclose(npz["predictive_mean"], mean_npy)),
        "npz_std_matches_csv_and_npy": bool(np.allclose(npz["predictive_std"], csv_df["predictive_std_local_core_width_mm"].to_numpy(dtype=float)) and np.allclose(npz["predictive_std"], std_npy)),
        "npz_identifiers_match_csv": bool(
            np.array_equal(npz["prediction_index"], csv_df["prediction_index"].to_numpy(dtype=int))
            and np.array_equal(npz["source_row_index"], csv_df["source_row_index"].to_numpy(dtype=int))
            and np.array_equal(npz["frame_index"], csv_df["frame_index"].to_numpy(dtype=int))
            and np.allclose(npz["x_position_mm"], csv_df["x_position_mm"].to_numpy(dtype=float))
        ),
        "prediction_ordering_deterministic": bool(csv_df["prediction_index"].is_monotonic_increasing and csv_df["frame_index"].is_monotonic_increasing),
        "track21_metrics_not_computed": True,
        "no_track21_target_derived_columns_entered_model_input": True,
    }
    checks["all_validation_checks_passed"] = bool(all(checks.values()))
    if not checks["all_validation_checks_passed"]:
        failed = [k for k, v in checks.items() if v is False]
        raise RuntimeError(f"Export validation failed: {failed}")
    return checks


def main() -> int:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / f"{EXPERIMENT_NAME}_{timestamp}"

    print(f"Experiment: {EXPERIMENT_NAME}")
    print("Stage: final inference/deployment of frozen Experiment 32 pipeline")
    print(f"Experiment 32 metadata: {EXP32_METADATA_PATH}")
    print("Track 21 predictor rows may be loaded; Track 21 target/Wyko geometry remains sealed.")

    if not EXP32_METADATA_PATH.exists():
        raise RuntimeError(f"Missing successful Experiment 32 metadata: {EXP32_METADATA_PATH}")
    exp32_meta = json.loads(EXP32_METADATA_PATH.read_text(encoding="utf-8"))
    if exp32_meta.get("final_decision") != "PASS":
        raise RuntimeError(f"Experiment 32 did not record PASS: {exp32_meta.get('final_decision')}")
    if exp32_meta.get("window_feature_columns") != WINDOW_FEATURE_COLUMNS:
        raise RuntimeError("Experiment 33 feature order does not match Experiment 32 metadata.")

    multimodal_allowed = read_multimodal_allowed_columns(DATASET_PATH)
    development_predictors = read_development_dataset_prefix(multimodal_allowed)
    track21_predictors = read_track21_predictor_rows(multimodal_allowed)
    target_df = read_target_artifact(TARGET_ARTIFACT_PATH)
    training_samples, training_diagnostics = build_development_training_samples(development_predictors, target_df)
    track21_windows, track21_diagnostics = build_track21_inference_windows(track21_predictors)
    scaler, model, predictive_mean, predictive_std = train_and_predict(training_samples, track21_windows)

    predictions = track21_windows[["prediction_index", "track_id", "source_row_index", "frame_index", "x_position_mm"]].copy()
    predictions["predictive_mean_local_core_width_mm"] = predictive_mean
    predictions["predictive_std_local_core_width_mm"] = predictive_std
    predictions = add_prediction_intervals(predictions)

    # Write once, validate external representations, then update metadata with validation checks.
    preliminary_checks = {
        "pending_export_validation": True,
        "training_contains_only_tracks_8_10_14": True,
        "track21_predictors_only_loaded": True,
        "track21_target_geometry_never_loaded": True,
    }
    write_outputs(
        output_dir,
        predictions,
        predictive_mean,
        predictive_std,
        scaler,
        model,
        training_samples,
        training_diagnostics,
        track21_diagnostics,
        preliminary_checks,
    )
    validation_checks = validate_exports(output_dir, predictions, predictive_mean, predictive_std)
    write_outputs(
        output_dir,
        predictions,
        predictive_mean,
        predictive_std,
        scaler,
        model,
        training_samples,
        training_diagnostics,
        track21_diagnostics,
        validation_checks,
    )

    print("\nTraining diagnostics:")
    print(json.dumps(make_json_safe(training_diagnostics.__dict__), indent=2))
    print("\nTrack 21 window diagnostics:")
    print(json.dumps(make_json_safe(track21_diagnostics.__dict__), indent=2))
    print("\nPredictive mean summary:")
    print(json.dumps(make_json_safe(summary_stats(predictive_mean)), indent=2))
    print("\nPredictive std summary:")
    print(json.dumps(make_json_safe(summary_stats(predictive_std)), indent=2))
    print("\nValidation checks:")
    print(json.dumps(make_json_safe(validation_checks), indent=2))
    print(f"\nOutput directory: {output_dir}")
    print("Final inference complete. No Track 21 target metrics were computed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
