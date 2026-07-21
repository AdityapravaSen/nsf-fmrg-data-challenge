"""25_phase4_baseline_evaluation.py

Phase IV frozen baseline evaluation runner.

Stage 1 implementation: preflight only.

This script prepares the sealed Track 21 evaluation by validating that the
frozen Phase III inputs, schema, selected model configuration, PCA model arrays,
and output locations are available and internally consistent.

This stage intentionally does not fit scalers, create feature windows, align
training targets, train Ridge Regression, generate Track 21 predictions,
reconstruct PCA profiles, or compute metrics.

Frozen final baseline configuration:
- Model: Ridge Regression
- alpha: 1.0
- Feature group: SEM-only
- Target group: PCA shape (pc1--pc5)
- Training tracks: 8, 10, 14
- Sealed inference track: 21
- Window size: 5

Track 21 is inspected only for inference feature-row availability and schema
validity. Track 21 target labels are not used during this preflight stage.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import sys
from typing import Dict, List

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


# =============================================================================
# Frozen Phase IV configuration
# =============================================================================

SCRIPT_NAME = "25_phase4_baseline_evaluation"
RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")

DATASET_PATH = REPO_ROOT / "processed_data" / "final_multimodal_dataset.csv"
PCA_MODEL_PATH = (
    REPO_ROOT
    / "processed_data"
    / "run_outputs"
    / "11_pca_representation_evaluation_20260714_230834"
    / "tables"
    / "pca_model_arrays.npz"
)

OUTPUT_ROOT = REPO_ROOT / "processed_data" / "phase4" / "baseline_evaluation"
OUTPUT_DIR = OUTPUT_ROOT / f"{SCRIPT_NAME}_{RUN_TAG}"

TRAIN_TRACKS = [8, 10, 14]
INFERENCE_TRACK = 21
WINDOW_SIZE = 5

MODEL_NAME = "Ridge Regression"
RIDGE_ALPHA = 1.0
FEATURE_GROUP = "sem_only"
TARGET_GROUP = "pca_shape"

IDENTITY_COLUMNS = ["track_id", "frame_index", "x_position_mm"]
TARGET_COLUMNS = ["pc1", "pc2", "pc3", "pc4", "pc5"]
VALIDITY_COLUMNS = ["pca_ready"]
EXPECTED_TRACKS = sorted(TRAIN_TRACKS + [INFERENCE_TRACK])


class PreflightError(RuntimeError):
    """Raised when Phase IV preflight validation fails."""


def create_output_directories(output_dir: Path) -> Dict[str, Path]:
    """Create the Phase IV output directory structure for this run."""

    subdirs = {
        "root": output_dir,
        "predictions": output_dir / "predictions",
        "reconstructions": output_dir / "reconstructions",
        "metrics": output_dir / "metrics",
        "metadata": output_dir / "metadata",
        "models": output_dir / "models",
        "reports": output_dir / "reports",
    }
    for path in subdirs.values():
        path.mkdir(parents=True, exist_ok=False)
    return subdirs


def frozen_feature_columns() -> List[str]:
    """Return the frozen SEM-only feature order from FeaturePreprocessor."""

    preprocessor = FeaturePreprocessor()
    features = list(preprocessor.sem_features)
    expected = ["substrate_roughness_variance", "substrate_mean_intensity"]
    if features != expected:
        raise PreflightError(f"SEM-only feature order changed: expected {expected}, observed {features}")
    return features


def require_columns(df: pd.DataFrame, columns: List[str], label: str) -> None:
    """Require that a dataframe contains the requested columns."""

    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise PreflightError(f"{label} is missing required columns: {missing}")


def require_finite(df: pd.DataFrame, columns: List[str], label: str) -> None:
    """Require finite numeric values in selected dataframe columns."""

    for col in columns:
        values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        bad = ~np.isfinite(values)
        if bool(bad.any()):
            examples = df.loc[bad, IDENTITY_COLUMNS + [col]].head(5).to_dict(orient="records")
            raise PreflightError(
                f"{label} contains {int(bad.sum())} non-finite values in {col!r}. "
                f"Example rows: {examples}"
            )


def validate_unique_identity(df: pd.DataFrame) -> None:
    """Validate that canonical row identity is unique."""

    duplicated = df.duplicated(IDENTITY_COLUMNS, keep=False)
    if bool(duplicated.any()):
        examples = df.loc[duplicated, IDENTITY_COLUMNS].head(10).to_dict(orient="records")
        raise PreflightError(f"Dataset has duplicated identity rows. Example duplicates: {examples}")


def load_dataset(path: Path) -> pd.DataFrame:
    """Load and minimally validate the frozen multimodal dataset."""

    if not path.exists():
        raise FileNotFoundError(f"Missing frozen dataset: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise PreflightError(f"Frozen dataset is empty: {path}")
    return df


def validate_dataset_schema(df: pd.DataFrame, feature_columns: List[str]) -> None:
    """Validate dataset columns, identities, and required track presence."""

    required = sorted(set(IDENTITY_COLUMNS + feature_columns + TARGET_COLUMNS + VALIDITY_COLUMNS))
    require_columns(df, required, "final_multimodal_dataset.csv")
    validate_unique_identity(df)

    observed_tracks = sorted(int(v) for v in df["track_id"].dropna().unique())
    missing_tracks = [track for track in EXPECTED_TRACKS if track not in observed_tracks]
    if missing_tracks:
        raise PreflightError(f"Dataset is missing required tracks: {missing_tracks}; observed tracks: {observed_tracks}")

    missing_identity = df[IDENTITY_COLUMNS].isna().any(axis=1)
    if bool(missing_identity.any()):
        raise PreflightError(f"Dataset contains {int(missing_identity.sum())} rows with missing identity values.")


def development_training_rows(df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    """Return supervised development rows for final model fitting."""

    dev = df[df["track_id"].astype(int).isin(TRAIN_TRACKS)].copy()
    pca_ready = dev["pca_ready"].map(lambda value: bool(value) if pd.notna(value) else False)
    dev = dev[pca_ready].copy()
    if dev.empty:
        raise PreflightError("No development rows remain after pca_ready == True filtering.")

    present_tracks = sorted(int(v) for v in dev["track_id"].dropna().unique())
    missing_tracks = [track for track in TRAIN_TRACKS if track not in present_tracks]
    if missing_tracks:
        raise PreflightError(f"No pca_ready development rows for tracks: {missing_tracks}")

    require_finite(dev, feature_columns, "development training rows")
    require_finite(dev, TARGET_COLUMNS, "development training targets")
    return dev


def track21_inference_rows(df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    """Return Track 21 rows for inference without target-label filtering."""

    track21 = df[df["track_id"].astype(int) == INFERENCE_TRACK].copy()
    if track21.empty:
        raise PreflightError("No Track 21 rows found for inference.")
    if len(track21) < WINDOW_SIZE:
        raise PreflightError(f"Track 21 has {len(track21)} rows, fewer than WINDOW_SIZE={WINDOW_SIZE}.")

    require_finite(track21, feature_columns, "Track 21 inference rows")
    require_finite(track21, IDENTITY_COLUMNS, "Track 21 metadata")
    return track21


def validate_pca_model(path: Path) -> Dict[str, object]:
    """Validate frozen PCA model arrays needed for normalized profile reconstruction."""

    if not path.exists():
        raise FileNotFoundError(f"Missing frozen PCA model arrays: {path}")
    with np.load(path) as pca:
        required_arrays = ["mean_profile", "components", "y_grid_mm"]
        missing = [name for name in required_arrays if name not in pca.files]
        if missing:
            raise PreflightError(f"PCA model arrays are missing required arrays: {missing}")
        mean_profile = np.asarray(pca["mean_profile"], dtype=float)
        components = np.asarray(pca["components"], dtype=float)
        y_grid_mm = np.asarray(pca["y_grid_mm"], dtype=float)

    if mean_profile.ndim != 1:
        raise PreflightError(f"PCA mean_profile must be 1D; got shape {mean_profile.shape}")
    if components.ndim != 2:
        raise PreflightError(f"PCA components must be 2D; got shape {components.shape}")
    if components.shape[0] < len(TARGET_COLUMNS):
        raise PreflightError(f"PCA components has {components.shape[0]} rows; at least {len(TARGET_COLUMNS)} required.")
    if components.shape[1] != mean_profile.shape[0]:
        raise PreflightError(
            f"PCA component width {components.shape[1]} does not match mean_profile length {mean_profile.shape[0]}."
        )
    if y_grid_mm.shape[0] != mean_profile.shape[0]:
        raise PreflightError(f"PCA y_grid length {y_grid_mm.shape[0]} does not match mean_profile length {mean_profile.shape[0]}.")
    if not np.isfinite(mean_profile).all() or not np.isfinite(components[: len(TARGET_COLUMNS)]).all() or not np.isfinite(y_grid_mm).all():
        raise PreflightError("PCA model arrays contain non-finite values.")

    return {
        "mean_profile_length": int(mean_profile.shape[0]),
        "components_shape": list(components.shape),
        "y_grid_length": int(y_grid_mm.shape[0]),
    }


def write_stage1_metadata(output_paths: Dict[str, Path], payload: Dict[str, object]) -> Path:
    """Write preflight metadata for auditability."""

    path = output_paths["metadata"] / "stage1_preflight_metadata.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def print_console_summary(
    df: pd.DataFrame,
    dev: pd.DataFrame,
    track21: pd.DataFrame,
    feature_columns: List[str],
    pca_summary: Dict[str, object],
    output_paths: Dict[str, Path],
    metadata_path: Path,
) -> None:
    """Print a clear Stage 1 preflight summary."""

    expected_dev_windows = sum(max(0, len(g) - WINDOW_SIZE + 1) for _track, g in dev.groupby("track_id"))
    expected_track21_windows = max(0, len(track21) - WINDOW_SIZE + 1)

    print("Phase IV frozen baseline evaluation — Stage 1 preflight")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Rows loaded: {len(df)}")
    print(f"Tracks observed: {sorted(int(v) for v in df['track_id'].dropna().unique())}")
    print(f"Frozen model: {MODEL_NAME} (alpha={RIDGE_ALPHA})")
    print(f"Frozen feature group: {FEATURE_GROUP}")
    print(f"Frozen feature columns: {feature_columns}")
    print(f"Frozen target columns: {TARGET_COLUMNS}")
    print(f"Training tracks: {TRAIN_TRACKS}")
    print(f"Inference track: {INFERENCE_TRACK}")
    print(f"Window size: {WINDOW_SIZE}")
    print()
    print("Development training rows after pca_ready filtering:")
    for track_id, group in dev.groupby("track_id", sort=True):
        print(f"  Track {int(track_id)}: {len(group)} rows")
    print(f"  Expected development windows: {expected_dev_windows}")
    print()
    print("Track 21 inference rows:")
    print(f"  Rows preserved without target filtering: {len(track21)}")
    print(f"  Expected Track 21 windows: {expected_track21_windows}")
    print()
    print("Frozen PCA model arrays:")
    print(f"  PCA path: {PCA_MODEL_PATH}")
    print(f"  Mean profile length: {pca_summary['mean_profile_length']}")
    print(f"  Components shape: {pca_summary['components_shape']}")
    print(f"  y-grid length: {pca_summary['y_grid_length']}")
    print()
    print(f"Output directory: {output_paths['root']}")
    print(f"Stage 1 metadata: {metadata_path}")
    print()
    print("Stage 1 complete: all preflight checks passed.")
    print("No scaler fitting, model fitting, Track 21 prediction, reconstruction, or metrics were performed.")


def flatten_windows(x_seq: np.ndarray) -> np.ndarray:
    """Flatten sequence windows for the frozen Ridge Regression baseline."""

    if x_seq.ndim != 3:
        raise PreflightError(f"Expected 3D sequence array; got shape {x_seq.shape}.")
    samples, window_size, n_features = x_seq.shape
    if window_size != WINDOW_SIZE:
        raise PreflightError(f"Window size mismatch: expected {WINDOW_SIZE}, got {window_size}.")
    if n_features != len(frozen_feature_columns()):
        raise PreflightError(f"Feature dimension mismatch: expected {len(frozen_feature_columns())}, got {n_features}.")
    return x_seq.reshape(samples, window_size * n_features)


def execute_blind_inference(
    dev: pd.DataFrame,
    track21: pd.DataFrame,
    feature_columns: List[str],
    output_paths: Dict[str, Path],
) -> Dict[str, object]:
    """Execute frozen Stage 2/3 blind inference using the proven Ridge workflow."""

    print("\n" + "=" * 60)
    print("STAGE 2: Preprocessing & Sequence Generation")
    print("=" * 60)

    preprocessor = FeaturePreprocessor()
    preprocessor.all_features = list(feature_columns)

    train_df = dev.copy()
    eval_df = track21.copy()

    print(f"Train split: {len(train_df)} valid rows (Tracks: {TRAIN_TRACKS})")
    print(f"Eval split:  {len(eval_df)} inference rows (Track: {INFERENCE_TRACK})")

    preprocessor.scaler.fit(train_df[preprocessor.all_features])
    preprocessor.is_fitted = True

    train_df.loc[:, preprocessor.all_features] = preprocessor.scaler.transform(train_df[preprocessor.all_features])
    eval_df.loc[:, preprocessor.all_features] = preprocessor.scaler.transform(eval_df[preprocessor.all_features])

    require_finite(train_df, feature_columns, "scaled development training rows")
    require_finite(eval_df, feature_columns, "scaled Track 21 inference rows")

    print("\nCreating 5-frame sequence windows...")
    x_train_seq, train_meta = preprocessor.create_sequence_windows(train_df, window_size=WINDOW_SIZE)
    x_track21_seq, track21_meta = preprocessor.create_sequence_windows(eval_df, window_size=WINDOW_SIZE)

    x_train = flatten_windows(x_train_seq)
    x_track21 = flatten_windows(x_track21_seq)

    print(f"X_train_seq shape: {x_train_seq.shape}")
    print(f"X_train_flat shape: {x_train.shape}")
    print(f"X_track21_seq shape: {x_track21_seq.shape}")
    print(f"X_track21_flat shape: {x_track21.shape}")
    print(f"train_meta rows: {len(train_meta)}")
    print(f"track21_meta rows: {len(track21_meta)}")

    if len(train_meta) != len(x_train):
        raise PreflightError("Training metadata length does not match flattened training windows.")
    if len(track21_meta) != len(x_track21):
        raise PreflightError("Track 21 metadata length does not match flattened Track 21 windows.")

    print("\n" + "=" * 60)
    print("STAGE 3: Target Alignment, Ridge Fitting, and Blind Prediction")
    print("=" * 60)

    aligner = Phase3TargetAligner(dataset_path=DATASET_PATH)
    y_train = aligner.align(meta_df=train_meta, target_group=TARGET_GROUP, return_metadata=False)

    if y_train.shape != (len(x_train), len(TARGET_COLUMNS)):
        raise PreflightError(f"Training target shape mismatch: got {y_train.shape}, expected {(len(x_train), len(TARGET_COLUMNS))}.")
    if not np.isfinite(y_train).all():
        raise PreflightError("Training targets contain non-finite values after alignment.")

    print(f"Fitting Ridge Regression (alpha={RIDGE_ALPHA}) on {x_train.shape[0]} development samples...")
    model = Ridge(alpha=1.0, random_state=42)
    model.fit(x_train, y_train)

    print(f"Predicting Track {INFERENCE_TRACK} (blind inference)...")
    y_pred = model.predict(x_track21)
    expected_prediction_shape = (len(x_track21), len(TARGET_COLUMNS))
    if y_pred.shape != expected_prediction_shape:
        raise PreflightError(f"Prediction shape mismatch: got {y_pred.shape}, expected {expected_prediction_shape}.")
    if not np.isfinite(y_pred).all():
        raise PreflightError("Track 21 predictions contain non-finite values.")

    print("\n" + "=" * 60)
    print("STAGE 4: Prediction Export")
    print("=" * 60)
    print("Track 21 is a blind test set. Ground truth PCA targets are unavailable locally.")
    print("Skipping local metric calculations and exporting prediction CSV.")

    output_df = track21_meta[IDENTITY_COLUMNS].copy()
    for i, col in enumerate(TARGET_COLUMNS):
        output_df[f"predicted_{col}"] = y_pred[:, i]
    output_df["model_name"] = MODEL_NAME
    output_df["ridge_alpha"] = RIDGE_ALPHA
    output_df["feature_group"] = FEATURE_GROUP
    output_df["window_size"] = WINDOW_SIZE

    prediction_path = output_paths["predictions"] / "track21_predictions.csv"
    output_df.to_csv(prediction_path, index=False)

    development_metadata_path = output_paths["metadata"] / "final_development_training_metadata.csv"
    track21_metadata_path = output_paths["metadata"] / "track21_inference_metadata.csv"
    train_meta.to_csv(development_metadata_path, index=False)
    track21_meta.to_csv(track21_metadata_path, index=False)

    print(f"Prediction CSV: {prediction_path}")
    print(f"Development metadata CSV: {development_metadata_path}")
    print(f"Track 21 metadata CSV: {track21_metadata_path}")

    return {
        "stage": "stage_2_3_4_blind_inference_complete",
        "train_rows": int(len(train_df)),
        "track21_rows": int(len(eval_df)),
        "train_windows": int(len(train_meta)),
        "track21_windows": int(len(track21_meta)),
        "x_train_seq_shape": list(x_train_seq.shape),
        "x_train_flat_shape": list(x_train.shape),
        "x_track21_seq_shape": list(x_track21_seq.shape),
        "x_track21_flat_shape": list(x_track21.shape),
        "y_train_shape": list(y_train.shape),
        "y_pred_shape": list(y_pred.shape),
        "prediction_csv": str(prediction_path),
        "development_metadata_csv": str(development_metadata_path),
        "track21_metadata_csv": str(track21_metadata_path),
        "metrics_status": "skipped_track21_targets_unavailable",
        "track21_policy": {
            "used_for_scaler_fitting": False,
            "used_for_model_fitting": False,
            "used_for_hyperparameter_tuning": False,
            "target_labels_required_for_prediction": False,
        },
    }


def write_run_metadata(output_paths: Dict[str, Path], payload: Dict[str, object]) -> Path:
    """Write consolidated run metadata."""

    path = output_paths["metadata"] / "run_metadata.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def main() -> int:
    feature_columns = frozen_feature_columns()
    output_paths = create_output_directories(OUTPUT_DIR)

    df = load_dataset(DATASET_PATH)
    validate_dataset_schema(df, feature_columns)
    dev = development_training_rows(df, feature_columns)
    track21 = track21_inference_rows(df, feature_columns)
    pca_summary = validate_pca_model(PCA_MODEL_PATH)

    payload: Dict[str, object] = {
        "script": SCRIPT_NAME,
        "stage": "stage_1_preflight_only",
        "run_tag": RUN_TAG,
        "dataset_path": str(DATASET_PATH),
        "pca_model_path": str(PCA_MODEL_PATH),
        "output_dir": str(OUTPUT_DIR),
        "model": {"name": MODEL_NAME, "alpha": RIDGE_ALPHA},
        "feature_group": FEATURE_GROUP,
        "feature_columns": feature_columns,
        "target_group": TARGET_GROUP,
        "target_columns": TARGET_COLUMNS,
        "train_tracks": TRAIN_TRACKS,
        "inference_track": INFERENCE_TRACK,
        "window_size": WINDOW_SIZE,
        "dataset_rows": int(len(df)),
        "development_rows_after_pca_ready_filter": int(len(dev)),
        "track21_rows_preserved_without_target_filtering": int(len(track21)),
        "expected_track21_windows": int(max(0, len(track21) - WINDOW_SIZE + 1)),
        "pca_summary": pca_summary,
        "track21_policy": {
            "used_for_scaler_fitting": False,
            "used_for_model_fitting": False,
            "used_for_hyperparameter_tuning": False,
            "target_labels_required_for_stage1": False,
        },
    }
    metadata_path = write_stage1_metadata(output_paths, payload)
    print_console_summary(df, dev, track21, feature_columns, pca_summary, output_paths, metadata_path)

    inference_payload = execute_blind_inference(dev, track21, feature_columns, output_paths)
    payload.update(inference_payload)
    run_metadata_path = write_run_metadata(output_paths, payload)

    print("\nPhase IV frozen baseline blind inference complete: PASS")
    print(f"Run metadata: {run_metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
