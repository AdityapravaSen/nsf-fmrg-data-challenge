"""22_phase3_random_forest_feature_importance.py

Random Forest feature-importance analysis for Phase III.

This script analyzes the fitted Random Forest baseline using the built-in
feature_importances_ attribute. It reuses the same preprocessing, target
alignment, feature groups, flattening convention, and Random Forest configuration
from the Phase III Random Forest baseline experiment.

This is an analysis experiment, not a new predictive model or reusable
interpretability framework.
"""

from __future__ import annotations

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


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
TRAIN_TRACKS = [8, 10]
VAL_TRACKS = [14]
TARGET_GROUP = "pca_shape"
TARGET_NAMES = ["pc1", "pc2", "pc3", "pc4", "pc5"]
TOP_N = 15

RF_CONFIG = {
    "n_estimators": 300,
    "random_state": 42,
    "min_samples_leaf": 2,
    "n_jobs": -1,
}

warnings.filterwarnings("ignore", category=FutureWarning)


def flatten_sequence_features(x_seq: np.ndarray) -> np.ndarray:
    """Flatten sequence windows using the Random Forest baseline convention."""

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


def build_phase3_arrays(feature_columns: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    y_train, _train_target_meta = aligner.align(train_meta, target_group=TARGET_GROUP, return_metadata=True)
    y_val, _val_target_meta = aligner.align(val_meta, target_group=TARGET_GROUP, return_metadata=True)

    return x_train_seq, x_val_seq, y_train, y_val


def flattened_feature_metadata(feature_columns: list[str], window_size: int) -> pd.DataFrame:
    """Build metadata for flattened sequence-feature positions."""

    rows = []
    for timestep in range(window_size):
        for feature_name in feature_columns:
            rows.append(
                {
                    "flat_index": len(rows),
                    "timestep": timestep,
                    "feature_name": feature_name,
                    "flattened_feature": f"t{timestep}_{feature_name}",
                    "modality": infer_modality(feature_name),
                }
            )
    return pd.DataFrame(rows)


def infer_modality(feature_name: str) -> str:
    """Infer modality from the stable Phase III feature naming convention."""

    if feature_name.startswith("substrate_"):
        return "SEM"
    return "Thermal"


def importance_table(importances: np.ndarray, feature_columns: list[str]) -> pd.DataFrame:
    """Return a ranked feature-importance table with timestep metadata."""

    meta = flattened_feature_metadata(feature_columns, WINDOW_SIZE)
    if len(importances) != len(meta):
        raise ValueError(f"Importance length does not match flattened feature count: {len(importances)} vs {len(meta)}.")
    table = meta.copy()
    table["importance"] = np.asarray(importances, dtype=float)
    table = table.sort_values("importance", ascending=False).reset_index(drop=True)
    table["rank"] = np.arange(1, len(table) + 1, dtype=int)
    return table[["rank", "flattened_feature", "feature_name", "timestep", "modality", "importance"]]


def temporal_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Aggregate feature importance by timestep."""

    out = table.groupby("timestep", sort=True)["importance"].sum().reset_index()
    out = out.rename(columns={"importance": "total_importance"})
    return out


def modality_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Aggregate feature importance by modality."""

    out = table.groupby("modality", sort=True)["importance"].sum().reset_index()
    out = out.rename(columns={"importance": "total_importance"})
    total = float(out["total_importance"].sum())
    out["percent_importance"] = 100.0 * out["total_importance"] / total if total > 0 else np.nan
    return out


def validate_importances(importances: np.ndarray, expected_features: int, label: str) -> None:
    """Validate Random Forest impurity importance vector."""

    if importances.shape != (expected_features,):
        raise ValueError(f"{label} importance shape mismatch: got {importances.shape}, expected {(expected_features,)}.")
    if not np.isfinite(importances).all():
        raise ValueError(f"{label} importances contain non-finite values.")
    total = float(np.sum(importances))
    if not np.isclose(total, 1.0, rtol=1.0e-6, atol=1.0e-8):
        raise ValueError(f"{label} importances should sum to 1.0; got {total}.")


def print_top_features(table: pd.DataFrame, n: int = TOP_N) -> None:
    """Print top ranked flattened features."""

    print(f"Top {n} flattened features")
    print("  Rank  Timestep  Feature                                      Modality   Importance")
    for row in table.head(n).itertuples(index=False):
        print(
            f"  {int(row.rank):>4}  "
            f"{int(row.timestep):>8}  "
            f"{str(row.feature_name):<44} "
            f"{str(row.modality):<8} "
            f"{float(row.importance):>10.6f}"
        )


def print_temporal_summary(summary: pd.DataFrame) -> None:
    """Print timestep-level importance summary."""

    print("Temporal importance summary")
    for row in summary.itertuples(index=False):
        print(f"  timestep {int(row.timestep)}: total_importance={float(row.total_importance):.6f}")


def print_modality_summary(summary: pd.DataFrame) -> None:
    """Print modality-level importance summary."""

    print("Modality importance summary")
    for row in summary.itertuples(index=False):
        print(
            f"  {str(row.modality)}: "
            f"total_importance={float(row.total_importance):.6f}, "
            f"percent={float(row.percent_importance):.2f}%"
        )


def run_feature_group_analysis(group_name: str, feature_columns: list[str]) -> dict[str, object]:
    """Fit the Random Forest baseline and analyze built-in feature importances."""

    print("\n" + "=" * 78)
    print(f"Feature group: {group_name}")
    print(f"Feature columns ({len(feature_columns)}): {feature_columns}")
    print(f"Random Forest config: {RF_CONFIG}")

    x_train_seq, x_val_seq, y_train, y_val = build_phase3_arrays(feature_columns)
    x_train_flat = flatten_sequence_features(x_train_seq)
    x_val_flat = flatten_sequence_features(x_val_seq)

    model = RandomForestRegressor(**RF_CONFIG)
    model.fit(x_train_flat, y_train)
    y_val_pred = model.predict(x_val_flat)
    val_metrics = evaluate_regression(y_val, y_val_pred, target_names=TARGET_NAMES)

    importances = np.asarray(model.feature_importances_, dtype=float)
    validate_importances(importances, expected_features=x_train_flat.shape[1], label=group_name)

    table = importance_table(importances, feature_columns)
    time_summary = temporal_summary(table)
    mod_summary = modality_summary(table)

    print("Validation metric context")
    overall = val_metrics["overall"]
    print(
        f"  MAE={overall['mae']:.6f}, RMSE={overall['rmse']:.6f}, "
        f"MedianAE={overall['median_absolute_error']:.6f}, R2={overall['r2']:.6f}"
    )
    print_top_features(table)
    print_temporal_summary(time_summary)
    if group_name == "thermal_sem":
        print_modality_summary(mod_summary)

    strongest = table.iloc[0]
    strongest_timestep = int(time_summary.sort_values("total_importance", ascending=False).iloc[0]["timestep"])
    print(
        "Descriptive summary: "
        f"largest flattened-feature importance is {strongest['feature_name']} at timestep {int(strongest['timestep'])}; "
        f"highest aggregate timestep importance occurs at timestep {strongest_timestep}."
    )

    return {
        "group_name": group_name,
        "feature_columns": feature_columns,
        "importance_table": table,
        "temporal_summary": time_summary,
        "modality_summary": mod_summary,
        "val_metrics": val_metrics,
    }


def print_pca_target_discussion() -> None:
    """Print the target-specific interpretation limitation for built-in RF importance."""

    print("\nPCA-target discussion")
    print(
        "  scikit-learn RandomForestRegressor exposes one aggregate feature_importances_ "
        "vector for the fitted multi-output model. This implementation does not naturally "
        "separate importance by PC1--PC5 without fitting additional target-specific models. "
        "Therefore the current analysis reports aggregate importance across the five PCA targets."
    )


def main() -> int:
    print("Phase III Random Forest feature importance analysis")
    print(f"Dataset: {FINAL_DATASET_PATH}")
    print(f"Train tracks: {TRAIN_TRACKS}")
    print(f"Validation tracks: {VAL_TRACKS}")
    print(f"Window size: {WINDOW_SIZE}")
    print(f"Target group: {TARGET_GROUP} ({TARGET_NAMES})")
    print(f"Random Forest config: {RF_CONFIG}")
    print("Importance source: RandomForestRegressor.feature_importances_")

    feature_groups = build_feature_group_config()
    results = []
    for group_name, feature_columns in feature_groups.items():
        results.append(run_feature_group_analysis(group_name, feature_columns))

    if len(results) != 3:
        raise RuntimeError(f"Expected 3 feature-group analyses; completed {len(results)}.")

    print_pca_target_discussion()

    combined = next(result for result in results if result["group_name"] == "thermal_sem")
    combined_modality = combined["modality_summary"]
    thermal_pct = float(combined_modality.loc[combined_modality["modality"] == "Thermal", "percent_importance"].iloc[0])
    sem_pct = float(combined_modality.loc[combined_modality["modality"] == "SEM", "percent_importance"].iloc[0])
    print("\nOverall descriptive summary")
    print(f"  Combined model modality importance: Thermal={thermal_pct:.2f}%, SEM={sem_pct:.2f}%.")
    print("  Importance values are impurity-based Random Forest importances and should be treated as baseline descriptors.")

    print("\nPhase III Random Forest feature importance analysis complete: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
