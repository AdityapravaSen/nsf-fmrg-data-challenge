"""25_phase5_loto_coordinate_augmented_ridge.py

Controlled experimental-branch LOTO test: coordinate-augmented Ridge.

Hypothesis:
    The frozen Ridge SEM-only model may underperform because it lacks a known
    longitudinal process-position variable. This script tests whether adding one
    fixed, physically normalized coordinate feature improves development-track
    Leave-One-Track-Out generalization.

This is not a Track 21 evaluation script. Track 21 is not loaded, inspected, or
used. No shared Phase III or Phase IV infrastructure is modified.

Only scientific change relative to the frozen Ridge SEM-only LOTO baseline:
    Add x_norm = (x_position_mm - 60.0) / 40.0

The coordinate is appended once per feature window, using the metadata row for
that window. This metadata row corresponds to the final frame in the window, which
is also the prediction/target alignment frame used by Phase3TargetAligner.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from scripts.phase3_data_loader import FeaturePreprocessor
from ml.targets import Phase3TargetAligner
from ml.metrics import evaluate_regression


FINAL_DATASET_PATH = REPO_ROOT / "processed_data" / "final_multimodal_dataset.csv"
OUTPUT_ROOT = REPO_ROOT / "processed_data" / "phase5" / "coordinate_augmented_ridge"
RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = OUTPUT_ROOT / f"25_phase5_loto_coordinate_augmented_ridge_{RUN_TAG}"

WINDOW_SIZE = 5
TARGET_GROUP = "pca_shape"
TARGET_NAMES = ["pc1", "pc2", "pc3", "pc4", "pc5"]
RIDGE_ALPHA = 1.0
X_CENTER_MM = 60.0
X_HALF_RANGE_MM = 40.0

FOLDS = [
    {"fold": "holdout_8", "train_tracks": [10, 14], "val_track": 8},
    {"fold": "holdout_10", "train_tracks": [8, 14], "val_track": 10},
    {"fold": "holdout_14", "train_tracks": [8, 10], "val_track": 14},
]

FROZEN_BASELINE = {
    "model": "ridge_regression",
    "feature_group": "sem_only",
    "mae": 1.275480,
    "rmse": 1.655831,
    "median_absolute_error": 1.013160,
    "r2": -0.282801,
}

warnings.filterwarnings("ignore", category=FutureWarning)


def x_norm_from_meta(meta_df: pd.DataFrame) -> np.ndarray:
    """Return fixed physical x-coordinate normalization for prediction frame.

    The metadata row produced by FeaturePreprocessor.create_sequence_windows
    corresponds to the final frame of each five-frame window. That same row is
    used by Phase3TargetAligner to align the PCA target. Therefore this x_norm
    value represents the prediction location, not all five frames in the window.
    """

    x = pd.to_numeric(meta_df["x_position_mm"], errors="raise").to_numpy(dtype=float)
    x_norm = (x - X_CENTER_MM) / X_HALF_RANGE_MM
    if not np.isfinite(x_norm).all():
        raise ValueError("x_norm contains non-finite values.")
    return x_norm.reshape(-1, 1)


def flatten_sequence_features(x_seq: np.ndarray) -> np.ndarray:
    """Flatten SEM-only sequence windows exactly as in the frozen Ridge baseline."""

    if x_seq.ndim != 3:
        raise ValueError(f"Expected X sequence array with 3 dimensions; got shape {x_seq.shape}.")
    return x_seq.reshape(len(x_seq), -1)


def frozen_sem_feature_columns() -> list[str]:
    """Return frozen SEM-only feature columns from FeaturePreprocessor."""

    schema = FeaturePreprocessor()
    return list(schema.sem_features)


def build_phase3_arrays(
    feature_columns: list[str],
    train_tracks: list[int],
    val_track: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Build train/validation arrays for one LOTO fold."""

    preprocessor = FeaturePreprocessor()
    preprocessor.all_features = list(feature_columns)

    train_data, val_data = preprocessor.load_and_scale(
        csv_path=FINAL_DATASET_PATH,
        train_tracks=train_tracks,
        eval_tracks=[val_track],
    )
    x_train_seq, train_meta = preprocessor.create_sequence_windows(train_data, window_size=WINDOW_SIZE)
    x_val_seq, val_meta = preprocessor.create_sequence_windows(val_data, window_size=WINDOW_SIZE)

    aligner = Phase3TargetAligner(dataset_path=FINAL_DATASET_PATH)
    y_train, _train_target_meta = aligner.align(train_meta, target_group=TARGET_GROUP, return_metadata=True)
    y_val, _val_target_meta = aligner.align(val_meta, target_group=TARGET_GROUP, return_metadata=True)

    return x_train_seq, x_val_seq, y_train, y_val, train_meta, val_meta


def build_coordinate_augmented_features(x_seq: np.ndarray, meta_df: pd.DataFrame) -> np.ndarray:
    """Append one x_norm coordinate to the flattened SEM-only window features."""

    x_flat = flatten_sequence_features(x_seq)
    x_coord = x_norm_from_meta(meta_df)
    if len(x_flat) != len(x_coord):
        raise ValueError(f"Feature/coordinate row mismatch: {len(x_flat)} vs {len(x_coord)}")
    return np.hstack([x_flat, x_coord])


def validate_predictions(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> None:
    """Validate prediction shape and finite values."""

    if y_true.shape != y_pred.shape:
        raise ValueError(f"{label} prediction shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")
    if not np.isfinite(y_pred).all():
        bad_count = int((~np.isfinite(y_pred)).sum())
        raise ValueError(f"{label} predictions contain {bad_count} non-finite values.")


def metric_row(metrics: dict) -> dict[str, float]:
    """Extract overall scalar metrics from an evaluation payload."""

    overall = metrics["overall"]
    return {
        "mae": float(overall["mae"]),
        "rmse": float(overall["rmse"]),
        "median_absolute_error": float(overall["median_absolute_error"]),
        "r2": float(overall["r2"]),
    }


def run_loto_coordinate_augmented_ridge() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run LOTO validation for coordinate-augmented SEM-only Ridge."""

    feature_columns = frozen_sem_feature_columns()
    fold_rows = []
    pooled_true = []
    pooled_pred = []

    for fold in FOLDS:
        fold_name = str(fold["fold"])
        train_tracks = list(fold["train_tracks"])
        val_track = int(fold["val_track"])
        print("\n" + "=" * 88)
        print(f"Fold: {fold_name} | train={train_tracks} | validate={val_track}")
        print(f"Feature columns: {feature_columns} + ['x_norm']")

        x_train_seq, x_val_seq, y_train, y_val, train_meta, val_meta = build_phase3_arrays(feature_columns, train_tracks, val_track)
        x_train_aug = build_coordinate_augmented_features(x_train_seq, train_meta)
        x_val_aug = build_coordinate_augmented_features(x_val_seq, val_meta)

        print(f"Shapes: X_train={x_train_aug.shape}, Y_train={y_train.shape}, X_val={x_val_aug.shape}, Y_val={y_val.shape}")

        model = Ridge(alpha=RIDGE_ALPHA, random_state=42)
        model.fit(x_train_aug, y_train)
        y_val_pred = model.predict(x_val_aug)
        validate_predictions(y_val, y_val_pred, fold_name)
        metrics = evaluate_regression(y_val, y_val_pred, target_names=TARGET_NAMES)

        row = {
            "fold": fold_name,
            "train_tracks": "+".join(str(t) for t in train_tracks),
            "val_track": val_track,
            "feature_group": "sem_only_plus_x_norm",
            "model": "ridge_regression",
            "n_val_samples": int(len(y_val)),
            **metric_row(metrics),
        }
        fold_rows.append(row)
        pooled_true.append(y_val)
        pooled_pred.append(y_val_pred)

        print(
            f"  ridge_regression  MAE={row['mae']:.6f}, RMSE={row['rmse']:.6f}, "
            f"MedianAE={row['median_absolute_error']:.6f}, R2={row['r2']:.6f}"
        )

    fold_df = pd.DataFrame(fold_rows)
    y_true = np.vstack(pooled_true)
    y_pred = np.vstack(pooled_pred)
    pooled_metrics = evaluate_regression(y_true, y_pred, target_names=TARGET_NAMES)
    aggregate_df = pd.DataFrame(
        [
            {
                "model": "ridge_regression",
                "feature_group": "sem_only_plus_x_norm",
                "n_total_val_samples": int(len(y_true)),
                **metric_row(pooled_metrics),
                "mean_fold_mae": float(fold_df["mae"].mean()),
                "std_fold_mae": float(fold_df["mae"].std(ddof=0)),
                "mean_fold_rmse": float(fold_df["rmse"].mean()),
                "std_fold_rmse": float(fold_df["rmse"].std(ddof=0)),
                "mean_fold_r2": float(fold_df["r2"].mean()),
                "std_fold_r2": float(fold_df["r2"].std(ddof=0)),
            }
        ]
    )
    return fold_df, aggregate_df


def write_outputs(fold_df: pd.DataFrame, aggregate_df: pd.DataFrame) -> None:
    """Write experiment outputs to a timestamped Phase 5 directory."""

    table_dir = OUTPUT_DIR / "tables"
    metadata_dir = OUTPUT_DIR / "metadata"
    table_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    fold_df.to_csv(table_dir / "loto_fold_metrics.csv", index=False)
    aggregate_df.to_csv(table_dir / "loto_aggregate_metrics.csv", index=False)

    metadata = {
        "experiment": "coordinate_augmented_ridge_loto",
        "out_dir": str(OUTPUT_DIR),
        "dataset_path": str(FINAL_DATASET_PATH),
        "model": "Ridge Regression",
        "alpha": RIDGE_ALPHA,
        "base_feature_group": "sem_only",
        "base_feature_columns": frozen_sem_feature_columns(),
        "added_feature": "x_norm = (x_position_mm - 60.0) / 40.0",
        "coordinate_frame": "final frame metadata row of each five-frame window; same row used for PCA target alignment",
        "window_size": WINDOW_SIZE,
        "target_group": TARGET_GROUP,
        "target_names": TARGET_NAMES,
        "folds": FOLDS,
        "frozen_baseline": FROZEN_BASELINE,
        "only_methodological_change": "one appended coordinate feature x_norm; no target, model class, alpha, windowing, or Track 21 usage changes",
        "track21_used": False,
    }
    (metadata_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def print_summary(fold_df: pd.DataFrame, aggregate_df: pd.DataFrame) -> None:
    """Print fold and aggregate comparison to frozen baseline."""

    print("\n" + "=" * 88)
    print("Coordinate-augmented Ridge LOTO fold metrics")
    print(fold_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    row = aggregate_df.iloc[0]
    print("\nAggregate pooled LOTO metrics")
    print(
        f"  MAE={row['mae']:.6f}, RMSE={row['rmse']:.6f}, "
        f"MedianAE={row['median_absolute_error']:.6f}, R2={row['r2']:.6f}"
    )
    print("\nFrozen Ridge SEM-only baseline")
    print(
        f"  MAE={FROZEN_BASELINE['mae']:.6f}, RMSE={FROZEN_BASELINE['rmse']:.6f}, "
        f"MedianAE={FROZEN_BASELINE['median_absolute_error']:.6f}, R2={FROZEN_BASELINE['r2']:.6f}"
    )
    print("\nDelta vs frozen baseline (negative MAE/RMSE/MedianAE is better; positive R2 is better)")
    print(f"  ΔMAE={row['mae'] - FROZEN_BASELINE['mae']:.6f}")
    print(f"  ΔRMSE={row['rmse'] - FROZEN_BASELINE['rmse']:.6f}")
    print(f"  ΔMedianAE={row['median_absolute_error'] - FROZEN_BASELINE['median_absolute_error']:.6f}")
    print(f"  ΔR2={row['r2'] - FROZEN_BASELINE['r2']:.6f}")
    print(f"\nOutputs written to: {OUTPUT_DIR}")


def main() -> int:
    print("Phase 5 experimental branch: coordinate-augmented Ridge LOTO")
    print(f"Dataset: {FINAL_DATASET_PATH}")
    print("Track 21 is not loaded, inspected, or used.")
    print("Only methodological change: append x_norm to frozen SEM-only flattened windows.")

    fold_df, aggregate_df = run_loto_coordinate_augmented_ridge()
    write_outputs(fold_df, aggregate_df)
    print_summary(fold_df, aggregate_df)

    if len(fold_df) != len(FOLDS):
        raise RuntimeError(f"Expected {len(FOLDS)} fold rows; got {len(fold_df)}.")
    print("\nCoordinate-augmented Ridge LOTO experiment complete: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
