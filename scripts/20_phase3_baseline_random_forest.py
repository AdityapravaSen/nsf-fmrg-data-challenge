"""20_phase3_baseline_random_forest.py

Third Phase III baseline modeling experiment.

This script trains a multi-output Random Forest Regressor to predict the frozen
PCA shape targets (PC1--PC5) from Phase III feature windows. It mirrors the
Linear Regression and Ridge Regression baseline experiments as closely as
possible while using a simple nonlinear tree-ensemble model.

This is an experiment script, not a reusable modeling framework. It does not
modify preprocessing, target alignment, metrics, descriptors, or datasets.
"""

from __future__ import annotations

from pathlib import Path
import sys
import warnings

import numpy as np
from sklearn.ensemble import RandomForestRegressor


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from scripts.phase3_data_loader import FeaturePreprocessor
from ml.targets import Phase3TargetAligner
from ml.metrics import evaluate_by_track, evaluate_regression


FINAL_DATASET_PATH = REPO_ROOT / "processed_data" / "final_multimodal_dataset.csv"
WINDOW_SIZE = 5
TRAIN_TRACKS = [8, 10]
VAL_TRACKS = [14]
TARGET_GROUP = "pca_shape"
TARGET_NAMES = ["pc1", "pc2", "pc3", "pc4", "pc5"]

RF_CONFIG = {
    "n_estimators": 300,
    "random_state": 42,
    "min_samples_leaf": 2,
    "n_jobs": -1,
}

LINEAR_REGRESSION_VALIDATION_BASELINE = {
    "thermal_only": {"mae": 2.535383, "rmse": 2.924938, "median_absolute_error": 2.407038, "r2": -6.554562},
    "sem_only": {"mae": 0.936175, "rmse": 1.206902, "median_absolute_error": 0.722373, "r2": -0.152840},
    "thermal_sem": {"mae": 2.464486, "rmse": 2.863417, "median_absolute_error": 2.381854, "r2": -7.072938},
}

RIDGE_REGRESSION_VALIDATION_BASELINE = {
    "thermal_only": {"mae": 2.024324, "rmse": 2.383099, "median_absolute_error": 1.963877, "r2": -2.268337},
    "sem_only": {"mae": 0.935437, "rmse": 1.205774, "median_absolute_error": 0.722716, "r2": -0.149425},
    "thermal_sem": {"mae": 2.069955, "rmse": 2.428553, "median_absolute_error": 2.007077, "r2": -2.437421},
}

warnings.filterwarnings("ignore", category=FutureWarning)


def flatten_sequence_features(x_seq: np.ndarray) -> np.ndarray:
    """Flatten sequence windows for classical scikit-learn models only."""

    if x_seq.ndim != 3:
        raise ValueError(f"Expected X sequence array with 3 dimensions; got shape {x_seq.shape}.")
    return x_seq.reshape(len(x_seq), -1)


def build_feature_group_config() -> dict[str, list[str]]:
    """Return feature-column groups from the stable FeaturePreprocessor schema."""

    schema = FeaturePreprocessor()
    return {
        "thermal_only": list(schema.thermal_features),
        "sem_only": list(schema.sem_features),
        "thermal_sem": list(schema.thermal_features + schema.sem_features),
    }


def build_phase3_arrays(feature_columns: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, object, object]:
    """Build train/validation feature windows and aligned PCA targets."""

    preprocessor = FeaturePreprocessor()
    preprocessor.all_features = list(feature_columns)

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

    return x_train_seq, x_val_seq, y_train, y_val, train_target_meta, val_target_meta


def validate_predictions(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> None:
    """Validate prediction array shape before metric evaluation."""

    if y_true.shape != y_pred.shape:
        raise ValueError(f"{label} prediction shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")
    if not np.isfinite(y_pred).all():
        bad_count = int((~np.isfinite(y_pred)).sum())
        raise ValueError(f"{label} predictions contain {bad_count} non-finite values.")


def print_overall_metrics(prefix: str, metrics: dict) -> None:
    """Print one-line overall metric summary."""

    overall = metrics["overall"]
    print(
        f"  {prefix}: "
        f"MAE={overall['mae']:.6f}, "
        f"RMSE={overall['rmse']:.6f}, "
        f"MedianAE={overall['median_absolute_error']:.6f}, "
        f"R2={overall['r2']:.6f}"
    )


def print_track_metrics(prefix: str, by_track: dict) -> None:
    """Print compact per-track metrics."""

    for track_id in by_track["track_order"]:
        payload = by_track["by_track"][str(track_id)]
        overall = payload["overall"]
        print(
            f"    {prefix} Track {track_id}: "
            f"n={payload['n_samples']}, "
            f"MAE={overall['mae']:.6f}, "
            f"RMSE={overall['rmse']:.6f}, "
            f"MedianAE={overall['median_absolute_error']:.6f}, "
            f"R2={overall['r2']:.6f}"
        )


def describe_change(candidate_value: float, baseline_value: float, lower_is_better: bool = True, tolerance: float = 1.0e-6) -> str:
    """Describe Random Forest performance relative to a prior baseline."""

    delta = candidate_value - baseline_value
    if abs(delta) <= tolerance:
        return "similar"
    if lower_is_better:
        return "improved" if delta < 0 else "worsened"
    return "improved" if delta > 0 else "worsened"


def run_feature_group_experiment(group_name: str, feature_columns: list[str]) -> dict[str, object]:
    """Run one Random Forest Regression baseline for one feature group."""

    print("\n" + "=" * 78)
    print(f"Feature group: {group_name}")
    print(f"Feature columns ({len(feature_columns)}): {feature_columns}")
    print(f"Random Forest config: {RF_CONFIG}")

    x_train_seq, x_val_seq, y_train, y_val, train_meta, val_meta = build_phase3_arrays(feature_columns)
    x_train_flat = flatten_sequence_features(x_train_seq)
    x_val_flat = flatten_sequence_features(x_val_seq)

    print("Shapes")
    print(f"  X_train_seq:  {x_train_seq.shape}")
    print(f"  X_train_flat: {x_train_flat.shape}")
    print(f"  Y_train:      {y_train.shape}")
    print(f"  X_val_seq:    {x_val_seq.shape}")
    print(f"  X_val_flat:   {x_val_flat.shape}")
    print(f"  Y_val:        {y_val.shape}")

    model = RandomForestRegressor(**RF_CONFIG)
    model.fit(x_train_flat, y_train)

    y_train_pred = model.predict(x_train_flat)
    y_val_pred = model.predict(x_val_flat)
    validate_predictions(y_train, y_train_pred, "training")
    validate_predictions(y_val, y_val_pred, "validation")

    train_metrics = evaluate_regression(y_train, y_train_pred, target_names=TARGET_NAMES)
    val_metrics = evaluate_regression(y_val, y_val_pred, target_names=TARGET_NAMES)
    train_by_track = evaluate_by_track(y_train, y_train_pred, train_meta, target_names=TARGET_NAMES)
    val_by_track = evaluate_by_track(y_val, y_val_pred, val_meta, target_names=TARGET_NAMES)

    print("Overall metrics")
    print_overall_metrics("train", train_metrics)
    print_overall_metrics("validation", val_metrics)

    print("Per-track metrics")
    print_track_metrics("train", train_by_track)
    print_track_metrics("validation", val_by_track)

    return {
        "group_name": group_name,
        "feature_columns": feature_columns,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "train_by_track": train_by_track,
        "val_by_track": val_by_track,
        "train_shape": x_train_flat.shape,
        "val_shape": x_val_flat.shape,
        "target_shape_train": y_train.shape,
        "target_shape_val": y_val.shape,
    }


def print_comparison_summary(results: list[dict[str, object]]) -> None:
    """Print validation metrics and descriptive comparison to prior baselines."""

    print("\n" + "=" * 78)
    print("Random Forest Regression baseline comparison: validation split")
    print("Target: PCA shape (PC1--PC5)")
    print("Split: train Tracks 8+10, validation Track 14")
    print(f"Random Forest config: {RF_CONFIG}")
    print("\nFeature group                 MAE        RMSE       MedianAE   R2")
    print("-" * 78)
    for result in results:
        overall = result["val_metrics"]["overall"]
        print(
            f"{result['group_name']:<28} "
            f"{overall['mae']:>9.6f}  "
            f"{overall['rmse']:>9.6f}  "
            f"{overall['median_absolute_error']:>9.6f}  "
            f"{overall['r2']:>9.6f}"
        )

    print("\nDescriptive comparison to previous validation baselines")
    print("Feature group                 vs Linear MAE    vs Ridge MAE     vs Linear R2     vs Ridge R2")
    print("-" * 94)
    for result in results:
        group_name = result["group_name"]
        rf = result["val_metrics"]["overall"]
        linear = LINEAR_REGRESSION_VALIDATION_BASELINE[group_name]
        ridge = RIDGE_REGRESSION_VALIDATION_BASELINE[group_name]
        print(
            f"{group_name:<28} "
            f"{describe_change(rf['mae'], linear['mae'], lower_is_better=True):<16} "
            f"{describe_change(rf['mae'], ridge['mae'], lower_is_better=True):<16} "
            f"{describe_change(rf['r2'], linear['r2'], lower_is_better=False):<16} "
            f"{describe_change(rf['r2'], ridge['r2'], lower_is_better=False):<16}"
        )


def main() -> int:
    print("Phase III baseline experiment: multi-output Random Forest Regression")
    print(f"Dataset: {FINAL_DATASET_PATH}")
    print(f"Train tracks: {TRAIN_TRACKS}")
    print(f"Validation tracks: {VAL_TRACKS}")
    print(f"Window size: {WINDOW_SIZE}")
    print(f"Target group: {TARGET_GROUP} ({TARGET_NAMES})")
    print(f"Random Forest config: {RF_CONFIG}")
    print("Flattening: X_flat = X_seq.reshape(len(X_seq), -1) inside this experiment only")

    feature_groups = build_feature_group_config()
    results = []
    for group_name, feature_columns in feature_groups.items():
        results.append(run_feature_group_experiment(group_name, feature_columns))

    print_comparison_summary(results)

    if len(results) != 3:
        raise RuntimeError(f"Expected 3 feature-group experiments; completed {len(results)}.")

    print("\nPhase III baseline Random Forest Regression experiment complete: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
