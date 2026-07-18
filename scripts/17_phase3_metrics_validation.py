"""17_phase3_metrics_validation.py

Validation script for reusable Phase III regression metrics.

This script exercises the metrics module using the existing Phase III data
pipeline. It creates simple example predictions for infrastructure validation
only; it does not train models or interpret model quality.
"""

from __future__ import annotations

from pathlib import Path
import sys
import warnings

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from scripts.phase3_data_loader import FeaturePreprocessor
from ml.targets import Phase3TargetAligner
from ml.datasets import Phase3TorchDataset
from ml.metrics import (
    evaluate_by_track,
    evaluate_regression,
    mean_absolute_error,
    median_absolute_error,
    r2_score,
    root_mean_squared_error,
)


FINAL_DATASET_PATH = REPO_ROOT / "processed_data" / "final_multimodal_dataset.csv"
WINDOW_SIZE = 5
TRAIN_TRACKS = [8, 10]
VAL_TRACKS = [14]
TARGET_GROUP = "pca_shape"
TARGET_NAMES = ["pc1", "pc2", "pc3", "pc4", "pc5"]

warnings.filterwarnings("ignore", category=FutureWarning)


def make_constant_prediction(y_train: np.ndarray, n_rows: int) -> np.ndarray:
    """Create a simple mean-target prediction for validation only."""

    mean_target = np.mean(y_train, axis=0, keepdims=True)
    return np.repeat(mean_target, repeats=n_rows, axis=0)


def print_metric_summary(label: str, metrics: dict) -> None:
    """Print a compact metric summary."""

    overall = metrics["overall"]
    print(f"\n{label}")
    print(f"  n_samples: {metrics['n_samples']}")
    print(f"  n_targets: {metrics['n_targets']}")
    print(f"  MAE: {overall['mae']:.6f}")
    print(f"  RMSE: {overall['rmse']:.6f}")
    print(f"  Median AE: {overall['median_absolute_error']:.6f}")
    print(f"  R2: {overall['r2']:.6f}")
    for target, target_metrics in metrics["per_target"].items():
        print(
            f"  {target}: "
            f"MAE={target_metrics['mae']:.6f}, "
            f"RMSE={target_metrics['rmse']:.6f}, "
            f"MedianAE={target_metrics['median_absolute_error']:.6f}, "
            f"R2={target_metrics['r2']:.6f}"
        )


def exercise_validation_checks(y_true: np.ndarray, y_pred: np.ndarray, meta_df) -> None:
    """Confirm representative validation failures raise clear exceptions."""

    checks = [
        ("shape mismatch", lambda: evaluate_regression(y_true, y_pred[:, :-1], target_names=TARGET_NAMES[:-1])),
        ("non-finite target", lambda: evaluate_regression(_with_nan(y_true), y_pred, target_names=TARGET_NAMES)),
        ("metadata length mismatch", lambda: evaluate_by_track(y_true, y_pred, meta_df.iloc[:-1], target_names=TARGET_NAMES)),
    ]

    print("\nValidation failure checks")
    for label, func in checks:
        try:
            func()
        except (TypeError, ValueError) as exc:
            print(f"  {label}: PASS ({exc})")
        else:
            raise AssertionError(f"Expected validation failure did not occur for: {label}")


def _with_nan(values: np.ndarray) -> np.ndarray:
    out = values.copy()
    out[0, 0] = np.nan
    return out


def main() -> int:
    print("Phase III metrics validation")
    print(f"Dataset: {FINAL_DATASET_PATH}")
    print(f"Train tracks: {TRAIN_TRACKS}")
    print(f"Validation tracks: {VAL_TRACKS}")
    print(f"Target group: {TARGET_GROUP}")

    # ==========================================
    # STEP 1: BUILD EXISTING PHASE III DATA PATH
    # ==========================================
    preprocessor = FeaturePreprocessor()
    train_data, val_data = preprocessor.load_and_scale(
        csv_path=FINAL_DATASET_PATH,
        train_tracks=TRAIN_TRACKS,
        eval_tracks=VAL_TRACKS,
    )
    x_train_seq, train_meta = preprocessor.create_sequence_windows(train_data, window_size=WINDOW_SIZE)
    x_val_seq, val_meta = preprocessor.create_sequence_windows(val_data, window_size=WINDOW_SIZE)

    aligner = Phase3TargetAligner(dataset_path=FINAL_DATASET_PATH)
    y_train, train_target_meta = aligner.align(train_meta, target_group=TARGET_GROUP, return_metadata=True)
    y_val, val_target_meta = aligner.align(val_meta, target_group=TARGET_GROUP, return_metadata=True)

    train_dataset = Phase3TorchDataset(x_train_seq, y_train, train_target_meta)
    val_dataset = Phase3TorchDataset(x_val_seq, y_val, val_target_meta)

    print("\nPipeline outputs")
    print(f"  train dataset size: {len(train_dataset)}")
    print(f"  validation dataset size: {len(val_dataset)}")
    print(f"  y_train shape: {y_train.shape}")
    print(f"  y_val shape: {y_val.shape}")

    # ==========================================
    # STEP 2: CREATE SIMPLE EXAMPLE PREDICTIONS
    # ==========================================
    y_train_pred = make_constant_prediction(y_train, len(y_train))
    y_val_pred = make_constant_prediction(y_train, len(y_val))

    # ==========================================
    # STEP 3: EXERCISE INDIVIDUAL METRIC FUNCTIONS
    # ==========================================
    print("\nIndividual metric functions on validation split")
    print(f"  MAE: {mean_absolute_error(y_val, y_val_pred):.6f}")
    print(f"  RMSE: {root_mean_squared_error(y_val, y_val_pred):.6f}")
    print(f"  Median AE: {median_absolute_error(y_val, y_val_pred):.6f}")
    print(f"  R2: {r2_score(y_val, y_val_pred):.6f}")

    # ==========================================
    # STEP 4: STRUCTURED EVALUATION
    # ==========================================
    train_metrics = evaluate_regression(y_train, y_train_pred, target_names=TARGET_NAMES)
    val_metrics = evaluate_regression(y_val, y_val_pred, target_names=TARGET_NAMES)
    print_metric_summary("Training structured metrics", train_metrics)
    print_metric_summary("Validation structured metrics", val_metrics)

    # ==========================================
    # STEP 5: TRACK-AWARE EVALUATION
    # ==========================================
    train_by_track = evaluate_by_track(y_train, y_train_pred, train_target_meta, target_names=TARGET_NAMES)
    val_by_track = evaluate_by_track(y_val, y_val_pred, val_target_meta, target_names=TARGET_NAMES)

    print("\nTrack-aware metrics")
    for track_id, payload in train_by_track["by_track"].items():
        overall = payload["overall"]
        print(f"  train track {track_id}: n={payload['n_samples']}, MAE={overall['mae']:.6f}, RMSE={overall['rmse']:.6f}")
    for track_id, payload in val_by_track["by_track"].items():
        overall = payload["overall"]
        print(f"  validation track {track_id}: n={payload['n_samples']}, MAE={overall['mae']:.6f}, RMSE={overall['rmse']:.6f}")

    # ==========================================
    # STEP 6: VALIDATION FAILURE CHECKS
    # ==========================================
    exercise_validation_checks(y_val, y_val_pred, val_target_meta)

    print("\nPhase III metrics validation complete: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
