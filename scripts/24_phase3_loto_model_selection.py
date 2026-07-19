"""24_phase3_loto_model_selection.py

Leave-One-Track-Out (LOTO) development validation for final Phase III model
selection.

This script is a fixed-configuration model-selection audit using only existing
development tracks. It evaluates the already-completed classical baseline model
families across all three development-track holdouts.

It does not inspect Track 21, tune hyperparameters, add features, modify
preprocessing, or modify target alignment.
"""

from __future__ import annotations

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge


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
WINDOW_SIZE = 5
TARGET_GROUP = "pca_shape"
TARGET_NAMES = ["pc1", "pc2", "pc3", "pc4", "pc5"]

FOLDS = [
    {"fold": "holdout_8", "train_tracks": [10, 14], "val_track": 8},
    {"fold": "holdout_10", "train_tracks": [8, 14], "val_track": 10},
    {"fold": "holdout_14", "train_tracks": [8, 10], "val_track": 14},
]

RIDGE_ALPHA = 1.0
RF_CONFIG = {
    "n_estimators": 300,
    "random_state": 42,
    "min_samples_leaf": 2,
    "n_jobs": -1,
}

warnings.filterwarnings("ignore", category=FutureWarning)


def flatten_sequence_features(x_seq: np.ndarray) -> np.ndarray:
    """Flatten sequence windows for fixed classical baseline models."""

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


def build_model_config() -> dict[str, object]:
    """Return fixed classical baseline models."""

    return {
        "linear_regression": LinearRegression(),
        "ridge_regression": Ridge(alpha=RIDGE_ALPHA),
        "random_forest": RandomForestRegressor(**RF_CONFIG),
    }


def build_phase3_arrays(
    feature_columns: list[str],
    train_tracks: list[int],
    val_track: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build train/validation arrays for one LOTO fold and feature group."""

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

    return x_train_seq, x_val_seq, y_train, y_val


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


def run_loto_validation() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all LOTO folds, models, and feature groups."""

    feature_groups = build_feature_group_config()
    fold_rows = []
    pooled_predictions: dict[tuple[str, str], dict[str, list[np.ndarray]]] = {}

    for fold in FOLDS:
        fold_name = str(fold["fold"])
        train_tracks = list(fold["train_tracks"])
        val_track = int(fold["val_track"])
        print("\n" + "=" * 88)
        print(f"Fold: {fold_name} | train={train_tracks} | validate={val_track}")

        for feature_group, feature_columns in feature_groups.items():
            print("\n" + "-" * 88)
            print(f"Feature group: {feature_group}")
            print(f"Feature columns ({len(feature_columns)}): {feature_columns}")

            x_train_seq, x_val_seq, y_train, y_val = build_phase3_arrays(feature_columns, train_tracks, val_track)
            x_train_flat = flatten_sequence_features(x_train_seq)
            x_val_flat = flatten_sequence_features(x_val_seq)

            print(f"Shapes: X_train={x_train_flat.shape}, Y_train={y_train.shape}, X_val={x_val_flat.shape}, Y_val={y_val.shape}")

            for model_name, model in build_model_config().items():
                model.fit(x_train_flat, y_train)
                y_val_pred = model.predict(x_val_flat)
                validate_predictions(y_val, y_val_pred, f"{fold_name}/{feature_group}/{model_name}")
                metrics = evaluate_regression(y_val, y_val_pred, target_names=TARGET_NAMES)
                row = {
                    "fold": fold_name,
                    "train_tracks": "+".join(str(t) for t in train_tracks),
                    "val_track": val_track,
                    "feature_group": feature_group,
                    "model": model_name,
                    "n_val_samples": int(len(y_val)),
                    **metric_row(metrics),
                }
                fold_rows.append(row)

                key = (model_name, feature_group)
                if key not in pooled_predictions:
                    pooled_predictions[key] = {"true": [], "pred": []}
                pooled_predictions[key]["true"].append(y_val)
                pooled_predictions[key]["pred"].append(y_val_pred)

                print(
                    f"  {model_name:<18} "
                    f"MAE={row['mae']:.6f}, RMSE={row['rmse']:.6f}, "
                    f"MedianAE={row['median_absolute_error']:.6f}, R2={row['r2']:.6f}"
                )

    fold_df = pd.DataFrame(fold_rows)

    aggregate_rows = []
    for (model_name, feature_group), payload in pooled_predictions.items():
        y_true = np.vstack(payload["true"])
        y_pred = np.vstack(payload["pred"])
        pooled_metrics = evaluate_regression(y_true, y_pred, target_names=TARGET_NAMES)
        subset = fold_df[(fold_df["model"] == model_name) & (fold_df["feature_group"] == feature_group)]
        aggregate_rows.append(
            {
                "model": model_name,
                "feature_group": feature_group,
                "n_total_val_samples": int(len(y_true)),
                **metric_row(pooled_metrics),
                "mean_fold_mae": float(subset["mae"].mean()),
                "std_fold_mae": float(subset["mae"].std(ddof=0)),
                "mean_fold_rmse": float(subset["rmse"].mean()),
                "std_fold_rmse": float(subset["rmse"].std(ddof=0)),
                "mean_fold_r2": float(subset["r2"].mean()),
                "std_fold_r2": float(subset["r2"].std(ddof=0)),
            }
        )

    aggregate_df = pd.DataFrame(aggregate_rows).sort_values(["mae", "rmse", "median_absolute_error"], ascending=True).reset_index(drop=True)
    return fold_df, aggregate_df


def print_aggregate_summary(aggregate_df: pd.DataFrame) -> None:
    """Print aggregate LOTO comparison."""

    print("\n" + "=" * 88)
    print("Aggregate pooled LOTO validation metrics")
    print("Model               Feature group     MAE       RMSE      MedianAE  R2        mean_fold_MAE  std_fold_MAE")
    print("-" * 112)
    for row in aggregate_df.itertuples(index=False):
        print(
            f"{str(row.model):<19} "
            f"{str(row.feature_group):<17} "
            f"{float(row.mae):>8.6f}  "
            f"{float(row.rmse):>8.6f}  "
            f"{float(row.median_absolute_error):>8.6f}  "
            f"{float(row.r2):>8.6f}  "
            f"{float(row.mean_fold_mae):>13.6f}  "
            f"{float(row.std_fold_mae):>12.6f}"
        )

    winner = aggregate_df.iloc[0]
    print("\nRecommended by pooled LOTO MAE:")
    print(f"  model={winner['model']}, feature_group={winner['feature_group']}, MAE={winner['mae']:.6f}, RMSE={winner['rmse']:.6f}, R2={winner['r2']:.6f}")


def main() -> int:
    print("Phase III Leave-One-Track-Out model-selection audit")
    print(f"Dataset: {FINAL_DATASET_PATH}")
    print(f"Development tracks: {[8, 10, 14]}")
    print(f"Target group: {TARGET_GROUP} ({TARGET_NAMES})")
    print("Models: Linear Regression, Ridge Regression, Random Forest Regression")
    print(f"Ridge alpha: {RIDGE_ALPHA}")
    print(f"Random Forest config: {RF_CONFIG}")
    print("Track 21 is not loaded, inspected, or used.")

    fold_df, aggregate_df = run_loto_validation()
    print_aggregate_summary(aggregate_df)

    if len(fold_df) != len(FOLDS) * 3 * 3:
        raise RuntimeError(f"Expected {len(FOLDS) * 3 * 3} fold result rows; got {len(fold_df)}.")
    if len(aggregate_df) != 9:
        raise RuntimeError(f"Expected 9 aggregate rows; got {len(aggregate_df)}.")

    print("\nPhase III LOTO model-selection audit complete: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
