"""Experiment 32: probabilistic local core-width evaluation.

Locked implementation of the revised Experiment 32 protocol:
BayesianRidge native posterior predictive distribution for Experiment 29
local_core_width_mm, evaluated by development-track LOTO against a naive
training-only Gaussian baseline.

This script intentionally stops after Experiment 32 development evaluation.
It does not train a final all-development model and does not predict Track 21.
Track 21 rows are not read from final_multimodal_dataset.csv during evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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
from scipy.stats import norm
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
PROTOCOL_PATH = REPO_ROOT / "docs" / "experiment_32_methodological_audit.md"
OUTPUT_ROOT = REPO_ROOT / "processed_data" / "run_outputs"

EXPERIMENT_NAME = "32_probabilistic_local_width_evaluation"
DEVELOPMENT_TRACKS = (8, 10, 14)
DEVELOPMENT_TRACK_SET = set(DEVELOPMENT_TRACKS)
SEALED_TRACKS = {21}
EXPECTED_ROWS_PER_DEVELOPMENT_TRACK = 400
EXPECTED_DEVELOPMENT_ROWS = EXPECTED_ROWS_PER_DEVELOPMENT_TRACK * len(DEVELOPMENT_TRACKS)
EXPECTED_EXPERIMENT30_SAMPLE_COUNTS = {8: 383, 10: 354, 14: 377}
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
    "smoothed_macro_width_mm",
}
FOLDS = [
    {"fold": "holdout_8", "train_tracks": [10, 14], "val_track": 8},
    {"fold": "holdout_10", "train_tracks": [8, 14], "val_track": 10},
    {"fold": "holdout_14", "train_tracks": [8, 10], "val_track": 14},
]
NOMINAL_LEVELS = (0.50, 0.80, 0.90, 0.95)

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
    if isinstance(obj, np.ndarray):
        return make_json_safe(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def read_development_dataset_prefix(dataset_path: Path) -> pd.DataFrame:
    """Read only locked development rows from the multimodal CSV.

    The current repository dataset stores Tracks 8, 10, and 14 in the first
    1200 data rows. This function stops before row 1201, then asserts that only
    development tracks were read. This is the same anti-Track-21 sealing
    semantics used by Experiment 30.
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
                    f"Encountered non-development track {track_id} within locked development prefix; "
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
            "y_true": float(target_row.local_core_width_mm),
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

    samples_by_track = {int(k): int(v) for k, v in samples.groupby("track_id").size().to_dict().items()}
    if samples_by_track != EXPECTED_EXPERIMENT30_SAMPLE_COUNTS:
        raise RuntimeError(
            "Centered-window sample counts do not match Experiment 30 semantics: "
            f"observed {samples_by_track}, expected {EXPECTED_EXPERIMENT30_SAMPLE_COUNTS}."
        )

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
    return samples.sort_values(["track_id", "frame_index"]).reset_index(drop=True), diagnostics


def pearson_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    if float(np.std(y_true)) == 0.0 or float(np.std(y_pred)) == 0.0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def deterministic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float | int]:
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


def gaussian_crps(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if not np.all(sigma > 0):
        raise RuntimeError("Gaussian CRPS requires strictly positive sigma.")
    z = (y_true - mu) / sigma
    return sigma * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - (1.0 / math.sqrt(math.pi)))


def gaussian_nll(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if not np.all(sigma > 0):
        raise RuntimeError("Gaussian NLL requires strictly positive sigma.")
    return 0.5 * np.log(2.0 * math.pi * sigma**2) + 0.5 * ((y_true - mu) / sigma) ** 2


def add_interval_columns(df: pd.DataFrame, prefix: str, mu_col: str, sigma_col: str) -> pd.DataFrame:
    out = df.copy()
    for level in NOMINAL_LEVELS:
        pct = int(round(level * 100))
        z = float(norm.ppf((1.0 + level) / 2.0))
        out[f"{prefix}_lower_{pct}"] = out[mu_col] - z * out[sigma_col]
        out[f"{prefix}_upper_{pct}"] = out[mu_col] + z * out[sigma_col]
        out[f"{prefix}_covered_{pct}"] = (out["y_true"] >= out[f"{prefix}_lower_{pct}"]) & (out["y_true"] <= out[f"{prefix}_upper_{pct}"])
        out[f"{prefix}_interval_width_{pct}"] = out[f"{prefix}_upper_{pct}"] - out[f"{prefix}_lower_{pct}"]
    return out


def probabilistic_metrics(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> Dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if not np.isfinite(y_true).all() or not np.isfinite(mu).all() or not np.isfinite(sigma).all():
        raise RuntimeError("Probabilistic metric inputs contain nonfinite values.")
    if not np.all(sigma > 0):
        raise RuntimeError("Probabilistic metric inputs contain non-positive sigma.")
    row: Dict[str, float | int] = {
        "n_valid_samples": int(len(y_true)),
        "crps": float(np.mean(gaussian_crps(y_true, mu, sigma))),
        "nll": float(np.mean(gaussian_nll(y_true, mu, sigma))),
        "mean_sigma": float(np.mean(sigma)),
        "median_sigma": float(np.median(sigma)),
        "min_sigma": float(np.min(sigma)),
        "max_sigma": float(np.max(sigma)),
    }
    for level in NOMINAL_LEVELS:
        pct = int(round(level * 100))
        z = float(norm.ppf((1.0 + level) / 2.0))
        lower = mu - z * sigma
        upper = mu + z * sigma
        row[f"coverage_{pct}"] = float(np.mean((y_true >= lower) & (y_true <= upper)))
        row[f"mean_interval_width_{pct}"] = float(np.mean(upper - lower))
    return row


def evaluate_decision(per_fold_prob: pd.DataFrame, pooled_prob: pd.DataFrame) -> Dict[str, object]:
    model_fold = per_fold_prob[per_fold_prob["distribution"] == "bayesian_ridge_native"].copy()
    naive_fold = per_fold_prob[per_fold_prob["distribution"] == "naive_training_gaussian"].copy()
    merged = model_fold.merge(naive_fold, on="fold", suffixes=("_model", "_naive"))
    fold_crps_wins = int((merged["crps_model"] < merged["crps_naive"]).sum())
    model_90_all = model_fold["coverage_90"].to_numpy(dtype=float)
    pooled_model_crps = float(pooled_prob.loc[pooled_prob["distribution"] == "bayesian_ridge_native", "crps"].iloc[0])
    pooled_naive_crps = float(pooled_prob.loc[pooled_prob["distribution"] == "naive_training_gaussian", "crps"].iloc[0])

    pass_criteria = {
        "pooled_model_crps_lt_pooled_naive_crps": bool(pooled_model_crps < pooled_naive_crps),
        "model_crps_lt_naive_crps_on_at_least_2_of_3_folds": bool(fold_crps_wins >= 2),
        "model_90pct_coverage_ge_75pct_on_all_3_folds": bool(np.all(model_90_all >= 0.75)),
    }
    conditional_criteria = {
        "model_crps_lt_naive_crps_on_at_least_1_of_3_folds": bool(fold_crps_wins >= 1),
        "model_90pct_coverage_ge_65pct_on_all_3_folds": bool(np.all(model_90_all >= 0.65)),
        "pooled_model_crps_lt_pooled_naive_crps_or_90pct_coverage_ge_75pct_on_at_least_2_of_3_folds": bool(
            (pooled_model_crps < pooled_naive_crps) or (int(np.sum(model_90_all >= 0.75)) >= 2)
        ),
    }
    fail_triggers = {
        "model_crps_ge_naive_crps_on_all_3_folds": bool(fold_crps_wins == 0),
        "model_90pct_coverage_lt_65pct_on_any_fold": bool(np.any(model_90_all < 0.65)),
    }

    if any(fail_triggers.values()):
        decision = "FAIL"
    elif all(pass_criteria.values()):
        decision = "PASS"
    elif all(conditional_criteria.values()):
        decision = "CONDITIONAL PASS"
    else:
        decision = "FAIL"

    return {
        "decision": decision,
        "fold_crps_wins_model_vs_naive": fold_crps_wins,
        "pooled_model_crps": pooled_model_crps,
        "pooled_naive_crps": pooled_naive_crps,
        "pooled_crps_relative_improvement": float((pooled_naive_crps - pooled_model_crps) / pooled_naive_crps),
        "model_90pct_coverages_by_fold": {str(r.fold): float(r.coverage_90) for r in model_fold.itertuples(index=False)},
        "pass_criteria": pass_criteria,
        "conditional_pass_criteria": conditional_criteria,
        "fail_triggers": fail_triggers,
    }


def run_loto(samples: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object], Dict[str, object]]:
    deterministic_rows: List[Dict[str, object]] = []
    prob_rows: List[Dict[str, object]] = []
    prediction_frames: List[pd.DataFrame] = []
    fold_metadata: Dict[str, object] = {}

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
        y_train = train_df["y_true"].to_numpy(dtype=float)
        x_val = val_df[WINDOW_FEATURE_COLUMNS].to_numpy(dtype=float)
        y_val = val_df["y_true"].to_numpy(dtype=float)
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

        # Frozen held-out predictive distribution. Held-out labels are not used in
        # scaling, model fitting, y_pred/y_std generation, or baseline parameters.
        y_pred, y_std = model.predict(x_val_scaled, return_std=True)
        baseline_mean = float(np.mean(y_train))
        baseline_std = float(np.std(y_train, ddof=1))
        baseline_mu = np.full_like(y_val, baseline_mean, dtype=float)
        baseline_sigma = np.full_like(y_val, baseline_std, dtype=float)

        if not np.isfinite(y_pred).all() or not np.isfinite(y_std).all() or not np.all(y_std > 0):
            raise RuntimeError(f"Fold {fold_name}: BayesianRidge predictions/std values failed finite positive validation.")
        if not math.isfinite(baseline_std) or baseline_std <= 0:
            raise RuntimeError(f"Fold {fold_name}: invalid training-only baseline std {baseline_std}.")

        det = deterministic_metrics(y_val, y_pred)
        deterministic_rows.append(
            {
                "fold": fold_name,
                "train_tracks": "+".join(str(t) for t in train_tracks),
                "val_track": val_track,
                **det,
            }
        )
        prob_rows.append(
            {
                "fold": fold_name,
                "train_tracks": "+".join(str(t) for t in train_tracks),
                "val_track": val_track,
                "distribution": "bayesian_ridge_native",
                **probabilistic_metrics(y_val, y_pred, y_std),
            }
        )
        prob_rows.append(
            {
                "fold": fold_name,
                "train_tracks": "+".join(str(t) for t in train_tracks),
                "val_track": val_track,
                "distribution": "naive_training_gaussian",
                **probabilistic_metrics(y_val, baseline_mu, baseline_sigma),
            }
        )

        pred_df = val_df[["track_id", "frame_index", "x_position_mm", "y_true"]].copy()
        pred_df.insert(0, "fold", fold_name)
        pred_df.insert(1, "train_tracks", "+".join(str(t) for t in train_tracks))
        pred_df["y_pred"] = y_pred
        pred_df["y_std"] = y_std
        pred_df["residual"] = pred_df["y_true"] - pred_df["y_pred"]
        pred_df["baseline_mean"] = baseline_mean
        pred_df["baseline_std"] = baseline_std
        pred_df["model_crps"] = gaussian_crps(y_val, y_pred, y_std)
        pred_df["baseline_crps"] = gaussian_crps(y_val, baseline_mu, baseline_sigma)
        pred_df["model_nll"] = gaussian_nll(y_val, y_pred, y_std)
        pred_df["baseline_nll"] = gaussian_nll(y_val, baseline_mu, baseline_sigma)
        pred_df = add_interval_columns(pred_df, "model", "y_pred", "y_std")
        pred_df = add_interval_columns(pred_df, "baseline", "baseline_mean", "baseline_std")
        prediction_frames.append(pred_df)

        fold_metadata[fold_name] = {
            "train_tracks": train_tracks,
            "val_track": val_track,
            "n_train_samples": int(len(train_df)),
            "n_val_samples": int(len(val_df)),
            "training_target_mean_for_naive_baseline": baseline_mean,
            "training_target_sample_std_ddof1_for_naive_baseline": baseline_std,
            "scaler": {
                "class": "sklearn.preprocessing.StandardScaler",
                "fit_tracks_only": train_tracks,
                "n_features_in": int(scaler.n_features_in_),
                "feature_names": WINDOW_FEATURE_COLUMNS,
                "mean": scaler.mean_.astype(float).tolist(),
                "scale": scaler.scale_.astype(float).tolist(),
                "var": scaler.var_.astype(float).tolist(),
            },
            "model": {
                "class": "sklearn.linear_model.BayesianRidge",
                "parameters": model.get_params(deep=True),
                "alpha": float(model.alpha_),
                "lambda": float(model.lambda_),
                "intercept": float(model.intercept_),
                "coef": model.coef_.astype(float).tolist(),
                "sigma_shape": list(model.sigma_.shape),
                "sigma_diag": np.diag(model.sigma_).astype(float).tolist(),
            },
            "prediction_freeze_validation": {
                "heldout_targets_used_for_scaling_fitting_or_uncertainty": False,
                "y_pred_finite": bool(np.isfinite(y_pred).all()),
                "y_std_finite": bool(np.isfinite(y_std).all()),
                "y_std_strictly_positive": bool(np.all(y_std > 0)),
            },
        }

    per_fold_deterministic = pd.DataFrame(deterministic_rows)
    per_fold_prob = pd.DataFrame(prob_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)

    y_true_pooled = predictions["y_true"].to_numpy(dtype=float)
    y_pred_pooled = predictions["y_pred"].to_numpy(dtype=float)
    y_std_pooled = predictions["y_std"].to_numpy(dtype=float)
    baseline_mu_pooled = predictions["baseline_mean"].to_numpy(dtype=float)
    baseline_sigma_pooled = predictions["baseline_std"].to_numpy(dtype=float)

    pooled_det = pd.DataFrame([{"scope": "pooled_oof_concatenated", **deterministic_metrics(y_true_pooled, y_pred_pooled)}])
    pooled_prob = pd.DataFrame(
        [
            {"scope": "pooled_oof_concatenated", "distribution": "bayesian_ridge_native", **probabilistic_metrics(y_true_pooled, y_pred_pooled, y_std_pooled)},
            {"scope": "pooled_oof_concatenated", "distribution": "naive_training_gaussian", **probabilistic_metrics(y_true_pooled, baseline_mu_pooled, baseline_sigma_pooled)},
        ]
    )

    coverage_rows: List[Dict[str, object]] = []
    for _, row in per_fold_prob.iterrows():
        for level in NOMINAL_LEVELS:
            pct = int(round(level * 100))
            coverage_rows.append(
                {
                    "scope": row["fold"],
                    "distribution": row["distribution"],
                    "nominal_coverage": level,
                    "observed_coverage": float(row[f"coverage_{pct}"]),
                    "mean_interval_width": float(row[f"mean_interval_width_{pct}"]),
                    "n_valid_samples": int(row["n_valid_samples"]),
                }
            )
    for _, row in pooled_prob.iterrows():
        for level in NOMINAL_LEVELS:
            pct = int(round(level * 100))
            coverage_rows.append(
                {
                    "scope": "pooled_oof_concatenated",
                    "distribution": row["distribution"],
                    "nominal_coverage": level,
                    "observed_coverage": float(row[f"coverage_{pct}"]),
                    "mean_interval_width": float(row[f"mean_interval_width_{pct}"]),
                    "n_valid_samples": int(row["n_valid_samples"]),
                }
            )
    coverage_table = pd.DataFrame(coverage_rows)

    model_fold = per_fold_prob[per_fold_prob["distribution"] == "bayesian_ridge_native"]
    naive_fold = per_fold_prob[per_fold_prob["distribution"] == "naive_training_gaussian"]
    crps_comparison = model_fold[["fold", "val_track", "crps"]].merge(
        naive_fold[["fold", "crps"]], on="fold", suffixes=("_model", "_naive")
    )
    crps_comparison["model_beats_naive"] = crps_comparison["crps_model"] < crps_comparison["crps_naive"]
    crps_comparison["relative_improvement"] = (crps_comparison["crps_naive"] - crps_comparison["crps_model"]) / crps_comparison["crps_naive"]
    pooled_model_crps = float(pooled_prob.loc[pooled_prob["distribution"] == "bayesian_ridge_native", "crps"].iloc[0])
    pooled_naive_crps = float(pooled_prob.loc[pooled_prob["distribution"] == "naive_training_gaussian", "crps"].iloc[0])
    crps_comparison = pd.concat(
        [
            crps_comparison,
            pd.DataFrame(
                [
                    {
                        "fold": "pooled_oof_concatenated",
                        "val_track": "pooled",
                        "crps_model": pooled_model_crps,
                        "crps_naive": pooled_naive_crps,
                        "model_beats_naive": pooled_model_crps < pooled_naive_crps,
                        "relative_improvement": (pooled_naive_crps - pooled_model_crps) / pooled_naive_crps,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    decision = evaluate_decision(per_fold_prob, pooled_prob)

    expected_predictions = int(per_fold_deterministic["n_valid_samples"].sum())
    if len(predictions) != expected_predictions:
        raise RuntimeError(f"Prediction row count mismatch: {len(predictions)} != {expected_predictions}")
    if not np.isfinite(predictions[["y_true", "y_pred", "y_std", "baseline_mean", "baseline_std"]].to_numpy(dtype=float)).all():
        raise RuntimeError("Final predictions contain nonfinite target, prediction, or std values.")
    if not (predictions["y_std"].to_numpy(dtype=float) > 0).all():
        raise RuntimeError("Final BayesianRidge std values are not all strictly positive.")
    if not (predictions["baseline_std"].to_numpy(dtype=float) > 0).all():
        raise RuntimeError("Final baseline std values are not all strictly positive.")

    return per_fold_deterministic, pooled_det, per_fold_prob, pooled_prob, predictions, coverage_table, crps_comparison, fold_metadata, decision


def plot_calibration(coverage_table: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    model = coverage_table[coverage_table["distribution"] == "bayesian_ridge_native"]
    for scope, group in model.groupby("scope", sort=False):
        if scope == "pooled_oof_concatenated":
            ax.plot(group["nominal_coverage"], group["observed_coverage"], marker="o", linewidth=2.5, label=scope)
        else:
            ax.plot(group["nominal_coverage"], group["observed_coverage"], marker="o", alpha=0.75, label=scope)
    ax.plot([0.45, 1.0], [0.45, 1.0], color="black", linestyle="--", linewidth=1, label="ideal")
    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(0.45, 1.0)
    ax.set_xlabel("Nominal central interval coverage")
    ax.set_ylabel("Observed coverage")
    ax.set_title("Experiment 32 BayesianRidge native calibration")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_intervals(predictions: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
    for ax, (fold_name, group) in zip(axes, predictions.groupby("fold", sort=False)):
        g = group.sort_values("x_position_mm")
        ax.fill_between(g["x_position_mm"].to_numpy(dtype=float), g["model_lower_90"].to_numpy(dtype=float), g["model_upper_90"].to_numpy(dtype=float), alpha=0.22, label="model 90% interval")
        ax.plot(g["x_position_mm"], g["y_pred"], color="C0", linewidth=1.5, label="model mean")
        ax.scatter(g["x_position_mm"], g["y_true"], s=10, color="black", alpha=0.65, label="held-out target")
        ax.plot(g["x_position_mm"], g["baseline_mean"], color="C3", linewidth=1.0, linestyle="--", label="naive mean")
        ax.set_title(fold_name)
        ax.set_ylabel("local_core_width_mm")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("x_position_mm")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_crps_by_fold(crps_comparison: pd.DataFrame, out_path: Path) -> None:
    fold_only = crps_comparison[crps_comparison["fold"] != "pooled_oof_concatenated"].copy()
    x = np.arange(len(fold_only))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, fold_only["crps_model"], width, label="BayesianRidge native")
    ax.bar(x + width / 2, fold_only["crps_naive"], width, label="Naive training Gaussian")
    ax.set_xticks(x)
    ax.set_xticklabels(fold_only["fold"].tolist())
    ax.set_ylabel("Gaussian CRPS (lower is better)")
    ax.set_title("Experiment 32 CRPS comparison")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_residual_standardized(predictions: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, (fold_name, group) in zip(axes, predictions.groupby("fold", sort=False)):
        z = group["residual"].to_numpy(dtype=float) / group["y_std"].to_numpy(dtype=float)
        ax.hist(z, bins=25, alpha=0.75, density=True)
        grid = np.linspace(-4, 4, 300)
        ax.plot(grid, norm.pdf(grid), color="black", linestyle="--", linewidth=1, label="N(0,1)")
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set_title(fold_name)
        ax.set_xlabel("standardized residual")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("density")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def write_outputs(
    output_dir: Path,
    per_fold_deterministic: pd.DataFrame,
    pooled_deterministic: pd.DataFrame,
    per_fold_prob: pd.DataFrame,
    pooled_prob: pd.DataFrame,
    predictions: pd.DataFrame,
    coverage_table: pd.DataFrame,
    crps_comparison: pd.DataFrame,
    diagnostics: SampleBuildDiagnostics,
    fold_metadata: Dict[str, object],
    decision: Dict[str, object],
) -> None:
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    metadata_dir = output_dir / "metadata"
    frozen_dir = output_dir / "frozen_predictions"
    for path in (tables_dir, figures_dir, metadata_dir, frozen_dir):
        path.mkdir(parents=True, exist_ok=True)

    per_fold_prob.to_csv(tables_dir / "per_fold_probabilistic_metrics.csv", index=False)
    pooled_prob.to_csv(tables_dir / "pooled_probabilistic_metrics.csv", index=False)
    per_fold_deterministic.to_csv(tables_dir / "per_fold_deterministic_metrics.csv", index=False)
    pooled_deterministic.to_csv(tables_dir / "pooled_deterministic_metrics.csv", index=False)
    predictions.to_csv(tables_dir / "oof_predictions.csv", index=False)
    coverage_table.to_csv(tables_dir / "coverage_table.csv", index=False)
    crps_comparison.to_csv(tables_dir / "crps_comparison_table.csv", index=False)

    for fold_name, group in predictions.groupby("fold", sort=False):
        frozen_cols = ["fold", "train_tracks", "track_id", "frame_index", "x_position_mm", "y_pred", "y_std", "baseline_mean", "baseline_std"]
        group[frozen_cols].to_csv(frozen_dir / f"{fold_name}_frozen_predictive_distribution.csv", index=False)

    (metadata_dir / "model_scaler_metadata_by_fold.json").write_text(json.dumps(make_json_safe(fold_metadata), indent=2), encoding="utf-8")
    (metadata_dir / "final_decision.json").write_text(json.dumps(make_json_safe(decision), indent=2), encoding="utf-8")

    plot_calibration(coverage_table, figures_dir / "calibration_nominal_vs_observed.png")
    plot_intervals(predictions, figures_dir / "predictive_intervals_90pct_by_fold.png")
    plot_crps_by_fold(crps_comparison, figures_dir / "crps_comparison_by_fold.png")
    plot_residual_standardized(predictions, figures_dir / "standardized_residual_histograms.png")

    metadata = {
        "experiment_name": EXPERIMENT_NAME,
        "timestamp": output_dir.name.replace(f"{EXPERIMENT_NAME}_", ""),
        "protocol_path": str(PROTOCOL_PATH),
        "protocol_title": "Experiment 32 — Methodological Audit and Revised Protocol",
        "git_branch": git_text(["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_commit": git_text(["rev-parse", "HEAD"]),
        "git_status_short_at_run": git_text(["status", "--short"]),
        "python_version": sys.version,
        "platform": platform.platform(),
        "target_artifact_path": str(TARGET_ARTIFACT_PATH),
        "dataset_path": str(DATASET_PATH),
        "target_column": "local_core_width_mm",
        "target_source": "Experiment 29 completed artifact anchor_aggregated_width.csv",
        "target_filter": "anchor_valid == True and finite local_core_width_mm; no smoothing/modification/bridging/interpolation",
        "features": FEATURE_BASE_COLUMNS,
        "window_feature_columns": WINDOW_FEATURE_COLUMNS,
        "window_definition": {
            "type": "centered",
            "offsets": list(WINDOW_OFFSETS),
            "n_frames": 5,
            "flattening": "for each offset t-2..t+2, append peak_temp, sqrt_mp_area, mp_length",
            "same_track_required": True,
            "semantics_reused_from_experiment_30": True,
        },
        "model": "sklearn.linear_model.BayesianRidge",
        "model_parameters": BayesianRidge().get_params(deep=True),
        "uncertainty_model": "BayesianRidge native posterior predictive std from predict(return_std=True) only",
        "baseline": "Normal(mean_train, sample_std_train_ddof1) fit on training samples only per fold",
        "scaler": "sklearn.preprocessing.StandardScaler fit separately on each training fold only",
        "folds": FOLDS,
        "valid_sample_counts": {
            "by_track": diagnostics.samples_by_track,
            "total": diagnostics.samples_after_windowing,
            "expected_experiment30_by_track": EXPECTED_EXPERIMENT30_SAMPLE_COUNTS,
            "matches_experiment30_semantics": diagnostics.samples_by_track == EXPECTED_EXPERIMENT30_SAMPLE_COUNTS,
        },
        "alignment_counts": diagnostics.__dict__,
        "track21_geometry_loaded": False,
        "track21_targets_loaded": False,
        "track21_rows_read_from_multimodal_dataset": False,
        "track21_used_for_training": False,
        "track21_used_for_calibration": False,
        "track21_used_for_evaluation": False,
        "track21_sealed_status": {
            "sealed_tracks": sorted(SEALED_TRACKS),
            "loaded_or_used": False,
            "guard": "Read only the first 1200 multimodal data rows and asserted exactly development tracks {8,10,14}; target artifact asserted to contain only {8,10,14}.",
        },
        "anti_leakage_status": {
            "scaler_fit_on_training_folds_only": True,
            "model_fit_on_training_folds_only": True,
            "baseline_parameters_fit_on_training_folds_only": True,
            "heldout_labels_used_only_after_prediction_freeze_for_metrics": True,
            "no_residual_inflation": True,
            "no_heldout_residual_calibration": True,
            "no_conformal_adjustment": True,
            "no_sigma_scaling": True,
        },
        "prohibited_inputs_status": {
            "smoothed_macro_width_mm_used": False,
            "sem_features_used": False,
            "coordinate_or_identity_metadata_used_as_model_features": False,
            "track_specific_thresholds_or_hacks_used": False,
            "new_target_created": False,
            "target_parameters_tuned": False,
            "model_hyperparameters_tuned": False,
            "model_feature_columns": WINDOW_FEATURE_COLUMNS,
            "prohibited_model_columns": sorted(PROHIBITED_MODEL_COLUMNS),
        },
        "validation_checks": {
            "predictions_finite": bool(np.isfinite(predictions["y_pred"].to_numpy(dtype=float)).all()),
            "std_values_finite": bool(np.isfinite(predictions["y_std"].to_numpy(dtype=float)).all()),
            "std_values_strictly_positive": bool((predictions["y_std"].to_numpy(dtype=float) > 0).all()),
            "baseline_std_values_finite": bool(np.isfinite(predictions["baseline_std"].to_numpy(dtype=float)).all()),
            "baseline_std_values_strictly_positive": bool((predictions["baseline_std"].to_numpy(dtype=float) > 0).all()),
            "sample_counts_match_experiment30": diagnostics.samples_by_track == EXPECTED_EXPERIMENT30_SAMPLE_COUNTS,
            "pooled_metrics_from_concatenated_oof_predictions": True,
            "existing_experiment29_or_30_artifacts_modified": False,
        },
        "decision_rule": {
            "PASS": [
                "pooled model CRPS < pooled naive CRPS",
                "model CRPS < naive CRPS on >= 2/3 folds",
                "90% coverage >= 75% on all three folds",
            ],
            "CONDITIONAL PASS": [
                "model CRPS < naive CRPS on >= 1/3 folds",
                "90% coverage >= 65% on all three folds",
                "pooled model CRPS < pooled naive CRPS OR 90% coverage >= 75% on >= 2/3 folds",
            ],
            "FAIL": [
                "model CRPS >= naive CRPS on all three folds",
                "90% coverage < 65% on any fold",
            ],
        },
        "final_decision": decision["decision"],
    }
    (metadata_dir / "run_metadata.json").write_text(json.dumps(make_json_safe(metadata), indent=2), encoding="utf-8")


def main() -> int:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / f"{EXPERIMENT_NAME}_{timestamp}"

    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Protocol: {PROTOCOL_PATH}")
    print(f"Target artifact: {TARGET_ARTIFACT_PATH}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Development tracks: {list(DEVELOPMENT_TRACKS)}")
    print("Track 21 sealed: geometry/targets not loaded; Track 21 rows not read from multimodal dataset.")

    if not PROTOCOL_PATH.exists():
        raise RuntimeError(f"Required Experiment 32 protocol document is missing: {PROTOCOL_PATH}")
    target_df = read_target_artifact(TARGET_ARTIFACT_PATH)
    dataset_df = read_development_dataset_prefix(DATASET_PATH)
    samples, diagnostics = build_centered_window_samples(dataset_df, target_df)
    (
        per_fold_deterministic,
        pooled_deterministic,
        per_fold_prob,
        pooled_prob,
        predictions,
        coverage_table,
        crps_comparison,
        fold_metadata,
        decision,
    ) = run_loto(samples)
    write_outputs(
        output_dir,
        per_fold_deterministic,
        pooled_deterministic,
        per_fold_prob,
        pooled_prob,
        predictions,
        coverage_table,
        crps_comparison,
        diagnostics,
        fold_metadata,
        decision,
    )

    print("\nTarget/sample alignment counts:")
    print(json.dumps(make_json_safe(diagnostics.__dict__), indent=2))
    print("\nPer-fold deterministic metrics:")
    print(per_fold_deterministic.to_string(index=False))
    print("\nPooled deterministic metrics:")
    print(pooled_deterministic.to_string(index=False))
    print("\nPer-fold probabilistic metrics:")
    print(per_fold_prob.to_string(index=False))
    print("\nPooled probabilistic metrics:")
    print(pooled_prob.to_string(index=False))
    print("\nCRPS comparison:")
    print(crps_comparison.to_string(index=False))
    print("\nDecision rule evaluation:")
    print(json.dumps(make_json_safe(decision), indent=2))
    print(f"\nOutput directory: {output_dir}")
    print(f"Final Experiment 32 decision: {decision['decision']}")
    print("Experiment 32 stops here; no Track 21 blind inference was run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
