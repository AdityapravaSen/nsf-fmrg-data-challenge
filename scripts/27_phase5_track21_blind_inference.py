"""27_phase5_track21_blind_inference.py

Final blind inference for sealed Track 21 using the current Phase 5 pipeline.

This script trains the frozen Phase 5 model on development tracks 8, 10, and 14
and generates unlabeled Track 21 predictions. Track 21 target-related columns may
exist in the dataset, but they are never aligned, inspected, used for fitting, or
used for evaluation.

Frozen Phase 5 configuration:
- Model: BayesianRidge
- Features: thermal physics drivers from FeaturePreprocessor
  (peak_temp, sqrt_mp_area, mp_length)
- Target group: smoothed_macro_width
- Target column: smoothed_macro_width_mm
- Training tracks: 8, 10, 14
- Blind inference track: 21
- Window size: 5
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from scripts.phase3_data_loader import FeaturePreprocessor
from ml.targets import Phase3TargetAligner


SCRIPT_NAME = "27_phase5_track21_blind_inference"
RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")

DATASET_PATH = REPO_ROOT / "processed_data" / "final_multimodal_dataset.csv"
OUTPUT_ROOT = REPO_ROOT / "processed_data" / "phase5" / "track21_blind_inference"
OUTPUT_DIR = OUTPUT_ROOT / f"{SCRIPT_NAME}_{RUN_TAG}"

TRAIN_TRACKS = [8, 10, 14]
INFERENCE_TRACK = 21
WINDOW_SIZE = 5

MODEL_NAME = "BayesianRidge"
TARGET_GROUP = "smoothed_macro_width"
TARGET_COLUMN = "smoothed_macro_width_mm"
PREDICTION_COLUMN = "predicted_smoothed_macro_width_mm"

IDENTITY_COLUMNS = ["track_id", "frame_index", "x_position_mm"]
BASE_FEATURE_COLUMNS = ["peak_temp", "mp_area_px", "mp_length"]
EXPECTED_TRACKS = sorted(TRAIN_TRACKS + [INFERENCE_TRACK])


class BlindInferenceError(RuntimeError):
    """Raised when Track 21 blind inference validation fails."""


def create_output_directories(output_dir: Path) -> Dict[str, Path]:
    """Create timestamped output directories for final inference artifacts."""

    subdirs = {
        "root": output_dir,
        "predictions": output_dir / "predictions",
        "arrays": output_dir / "arrays",
        "metadata": output_dir / "metadata",
        "models": output_dir / "models",
    }
    for path in subdirs.values():
        path.mkdir(parents=True, exist_ok=False)
    return subdirs


def require_columns(df: pd.DataFrame, columns: List[str], label: str) -> None:
    """Require that a dataframe contains selected columns."""

    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise BlindInferenceError(f"{label} is missing required columns: {missing}")


def require_finite(df: pd.DataFrame, columns: List[str], label: str) -> None:
    """Require finite numeric values in selected dataframe columns."""

    for col in columns:
        values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        bad = ~np.isfinite(values)
        if bool(bad.any()):
            examples = df.loc[bad, [c for c in IDENTITY_COLUMNS if c in df.columns] + [col]].head(5).to_dict(orient="records")
            raise BlindInferenceError(
                f"{label} contains {int(bad.sum())} non-finite values in {col!r}. "
                f"Example rows: {examples}"
            )


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the frozen multimodal dataset."""

    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise BlindInferenceError(f"Dataset is empty: {path}")
    return df


def validate_dataset(df: pd.DataFrame) -> None:
    """Validate dataset schema and track availability before inference."""

    require_columns(df, IDENTITY_COLUMNS + BASE_FEATURE_COLUMNS + [TARGET_COLUMN], "final_multimodal_dataset.csv")

    duplicated = df.duplicated(IDENTITY_COLUMNS, keep=False)
    if bool(duplicated.any()):
        examples = df.loc[duplicated, IDENTITY_COLUMNS].head(10).to_dict(orient="records")
        raise BlindInferenceError(f"Dataset has duplicated identity rows. Example duplicates: {examples}")

    missing_identity = df[IDENTITY_COLUMNS].isna().any(axis=1)
    if bool(missing_identity.any()):
        raise BlindInferenceError(f"Dataset contains {int(missing_identity.sum())} rows with missing identity values.")

    observed_tracks = sorted(int(v) for v in df["track_id"].dropna().unique())
    missing_tracks = [track for track in EXPECTED_TRACKS if track not in observed_tracks]
    if missing_tracks:
        raise BlindInferenceError(f"Dataset is missing required tracks: {missing_tracks}; observed tracks: {observed_tracks}")

    dev = df[df["track_id"].astype(int).isin(TRAIN_TRACKS)].copy()
    track21 = df[df["track_id"].astype(int) == INFERENCE_TRACK].copy()
    if dev.empty:
        raise BlindInferenceError("No development rows found for training tracks.")
    if track21.empty:
        raise BlindInferenceError("No Track 21 rows found for blind inference.")
    if len(track21) < WINDOW_SIZE:
        raise BlindInferenceError(f"Track 21 has {len(track21)} rows, fewer than WINDOW_SIZE={WINDOW_SIZE}.")

    require_finite(dev, BASE_FEATURE_COLUMNS + [TARGET_COLUMN], "development training rows")
    require_finite(track21, BASE_FEATURE_COLUMNS, "Track 21 inference rows")
    require_finite(track21, IDENTITY_COLUMNS, "Track 21 metadata")


def validate_preprocessor_schema(preprocessor: FeaturePreprocessor) -> List[str]:
    """Validate the frozen Phase 5 thermal physics feature schema."""

    expected_features = ["peak_temp", "sqrt_mp_area", "mp_length"]
    feature_columns = list(preprocessor.all_features)
    if feature_columns != expected_features:
        raise BlindInferenceError(
            f"FeaturePreprocessor feature schema changed: expected {expected_features}, observed {feature_columns}"
        )
    return feature_columns


def flatten_windows(x_seq: np.ndarray, *, label: str) -> np.ndarray:
    """Flatten sequence windows for the Phase 5 Bayesian Ridge model."""

    if x_seq.ndim != 3:
        raise BlindInferenceError(f"{label} must be a 3D sequence array; got shape {x_seq.shape}.")
    samples, window_size, n_features = x_seq.shape
    if samples == 0:
        raise BlindInferenceError(f"{label} contains zero sequence windows.")
    if window_size != WINDOW_SIZE:
        raise BlindInferenceError(f"{label} window size mismatch: expected {WINDOW_SIZE}, got {window_size}.")
    if n_features != 3:
        raise BlindInferenceError(f"{label} feature dimension mismatch: expected 3, got {n_features}.")
    x_flat = x_seq.reshape(samples, window_size * n_features)
    if not np.isfinite(x_flat).all():
        raise BlindInferenceError(f"{label} flattened feature matrix contains non-finite values.")
    return x_flat


def validate_metadata(meta_df: pd.DataFrame, expected_rows: int, label: str) -> None:
    """Validate sequence metadata row count and identity columns."""

    if len(meta_df) != expected_rows:
        raise BlindInferenceError(f"{label} metadata row count mismatch: expected {expected_rows}, got {len(meta_df)}")
    require_columns(meta_df, IDENTITY_COLUMNS, label)
    require_finite(meta_df, IDENTITY_COLUMNS, label)
    duplicated = meta_df.duplicated(IDENTITY_COLUMNS, keep=False)
    if bool(duplicated.any()):
        examples = meta_df.loc[duplicated, IDENTITY_COLUMNS].head(10).to_dict(orient="records")
        raise BlindInferenceError(f"{label} metadata contains duplicate identity rows. Example duplicates: {examples}")


def execute_blind_inference(output_paths: Dict[str, Path]) -> Dict[str, object]:
    """Train on development tracks and predict sealed Track 21."""

    print("=" * 72)
    print("PHASE 5 FINAL BLIND INFERENCE — TRACK 21")
    print("=" * 72)
    print(f"Dataset: {DATASET_PATH}")
    print(f"Training tracks: {TRAIN_TRACKS}")
    print(f"Blind inference track: {INFERENCE_TRACK}")
    print(f"Model: {MODEL_NAME}")
    print(f"Target group: {TARGET_GROUP}")
    print("Track 21 target-related columns are ignored completely.")

    df = load_dataset(DATASET_PATH)
    validate_dataset(df)

    preprocessor = FeaturePreprocessor()
    feature_columns = validate_preprocessor_schema(preprocessor)
    print(f"Feature columns: {feature_columns}")

    print("\nStage 1: Feature engineering and leakage-safe scaling")
    train_df, track21_df = preprocessor.load_and_scale(
        csv_path=DATASET_PATH,
        train_tracks=TRAIN_TRACKS,
        eval_tracks=[INFERENCE_TRACK],
    )

    if train_df.empty:
        raise BlindInferenceError("Training dataframe is empty after preprocessing.")
    if track21_df.empty:
        raise BlindInferenceError("Track 21 dataframe is empty after preprocessing.")
    require_finite(train_df, feature_columns, "scaled development training features")
    require_finite(track21_df, feature_columns, "scaled Track 21 inference features")

    print("\nStage 2: Sequence generation")
    x_train_seq, train_meta = preprocessor.create_sequence_windows(train_df, window_size=WINDOW_SIZE)
    x_track21_seq, track21_meta = preprocessor.create_sequence_windows(track21_df, window_size=WINDOW_SIZE)
    x_train = flatten_windows(x_train_seq, label="X_train_seq")
    x_track21 = flatten_windows(x_track21_seq, label="X_track21_seq")
    validate_metadata(train_meta, len(x_train), "development training metadata")
    validate_metadata(track21_meta, len(x_track21), "Track 21 inference metadata")

    expected_track21_windows = max(0, len(track21_df) - WINDOW_SIZE + 1)
    if len(track21_meta) != expected_track21_windows:
        raise BlindInferenceError(
            f"Track 21 prediction count mismatch: expected {expected_track21_windows}, got {len(track21_meta)}"
        )

    print(f"X_train_seq shape: {x_train_seq.shape}")
    print(f"X_train_flat shape: {x_train.shape}")
    print(f"X_track21_seq shape: {x_track21_seq.shape}")
    print(f"X_track21_flat shape: {x_track21.shape}")

    print("\nStage 3: Training target alignment")
    aligner = Phase3TargetAligner(dataset_path=DATASET_PATH)
    y_train = aligner.align(meta_df=train_meta, target_group=TARGET_GROUP, return_metadata=False).ravel()
    if y_train.shape != (len(x_train),):
        raise BlindInferenceError(f"Training target shape mismatch: got {y_train.shape}, expected {(len(x_train),)}")
    if not np.isfinite(y_train).all():
        raise BlindInferenceError("Training targets contain non-finite values after alignment.")
    print(f"Y_train shape: {y_train.shape}")

    print("\nStage 4: Bayesian Ridge fitting and Track 21 prediction")
    model = BayesianRidge()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_track21)
    if y_pred.shape != (len(x_track21),):
        raise BlindInferenceError(f"Prediction shape mismatch: got {y_pred.shape}, expected {(len(x_track21),)}")
    if not np.isfinite(y_pred).all():
        raise BlindInferenceError("Track 21 predictions contain non-finite values.")

    print("\nStage 5: Export predictions and audit metadata")
    prediction_df = track21_meta[IDENTITY_COLUMNS].copy()
    prediction_df[PREDICTION_COLUMN] = y_pred
    prediction_df["model_name"] = MODEL_NAME
    prediction_df["feature_group"] = "thermal_physics"
    prediction_df["target_group"] = TARGET_GROUP
    prediction_df["window_size"] = WINDOW_SIZE

    csv_path = output_paths["predictions"] / "track21_blind_predictions.csv"
    npy_path = output_paths["arrays"] / "track21_blind_predictions.npy"
    npz_path = output_paths["arrays"] / "track21_blind_prediction_bundle.npz"
    train_meta_path = output_paths["metadata"] / "development_training_metadata.csv"
    track21_meta_path = output_paths["metadata"] / "track21_inference_metadata.csv"
    scaler_stats_path = output_paths["models"] / "feature_scaler_stats.json"
    model_params_path = output_paths["models"] / "bayesian_ridge_parameters.json"

    prediction_df.to_csv(csv_path, index=False)
    np.save(npy_path, y_pred)
    np.savez(
        npz_path,
        predictions=y_pred,
        track_id=prediction_df["track_id"].to_numpy(dtype=int),
        frame_index=prediction_df["frame_index"].to_numpy(dtype=int),
        x_position_mm=prediction_df["x_position_mm"].to_numpy(dtype=float),
    )
    train_meta.to_csv(train_meta_path, index=False)
    track21_meta.to_csv(track21_meta_path, index=False)

    scaler_stats = {
        "feature_columns": feature_columns,
        "mean": preprocessor.scaler.mean_.astype(float).tolist(),
        "scale": preprocessor.scaler.scale_.astype(float).tolist(),
        "var": preprocessor.scaler.var_.astype(float).tolist(),
        "n_samples_seen": int(preprocessor.scaler.n_samples_seen_),
        "fit_tracks": TRAIN_TRACKS,
        "track21_used_for_scaler_fit": False,
    }
    scaler_stats_path.write_text(json.dumps(scaler_stats, indent=2), encoding="utf-8")

    model_params = {
        "model_name": MODEL_NAME,
        "coef": np.asarray(model.coef_, dtype=float).tolist(),
        "intercept": float(model.intercept_),
        "alpha": float(model.alpha_),
        "lambda": float(model.lambda_),
        "n_iter": int(model.n_iter_),
    }
    if getattr(model, "sigma_", None) is not None:
        model_params["sigma_shape"] = list(np.asarray(model.sigma_).shape)
    model_params_path.write_text(json.dumps(model_params, indent=2), encoding="utf-8")

    if not csv_path.exists() or not npy_path.exists() or not npz_path.exists():
        raise BlindInferenceError("One or more prediction exports were not written successfully.")
    written_df = pd.read_csv(csv_path)
    if len(written_df) != len(prediction_df):
        raise BlindInferenceError(f"Written CSV row count mismatch: expected {len(prediction_df)}, got {len(written_df)}")
    require_columns(written_df, IDENTITY_COLUMNS + [PREDICTION_COLUMN], "written prediction CSV")
    require_finite(written_df, IDENTITY_COLUMNS + [PREDICTION_COLUMN], "written prediction CSV")

    print(f"Prediction CSV: {csv_path}")
    print(f"Prediction NPY: {npy_path}")
    print(f"Prediction NPZ bundle: {npz_path}")
    print(f"Run output directory: {output_paths['root']}")

    return {
        "script": SCRIPT_NAME,
        "run_tag": RUN_TAG,
        "dataset_path": str(DATASET_PATH),
        "output_dir": str(output_paths["root"]),
        "model": MODEL_NAME,
        "feature_group": "thermal_physics",
        "feature_columns": feature_columns,
        "target_group": TARGET_GROUP,
        "target_column": TARGET_COLUMN,
        "prediction_column": PREDICTION_COLUMN,
        "train_tracks": TRAIN_TRACKS,
        "inference_track": INFERENCE_TRACK,
        "window_size": WINDOW_SIZE,
        "dataset_rows": int(len(df)),
        "train_rows": int(len(train_df)),
        "track21_rows": int(len(track21_df)),
        "train_windows": int(len(train_meta)),
        "track21_windows": int(len(track21_meta)),
        "expected_track21_windows": int(expected_track21_windows),
        "x_train_seq_shape": list(x_train_seq.shape),
        "x_train_flat_shape": list(x_train.shape),
        "x_track21_seq_shape": list(x_track21_seq.shape),
        "x_track21_flat_shape": list(x_track21.shape),
        "y_train_shape": list(y_train.shape),
        "y_pred_shape": list(y_pred.shape),
        "prediction_csv": str(csv_path),
        "prediction_npy": str(npy_path),
        "prediction_npz": str(npz_path),
        "development_metadata_csv": str(train_meta_path),
        "track21_metadata_csv": str(track21_meta_path),
        "scaler_stats_json": str(scaler_stats_path),
        "model_parameters_json": str(model_params_path),
        "prediction_summary": {
            "min": float(np.min(y_pred)),
            "max": float(np.max(y_pred)),
            "mean": float(np.mean(y_pred)),
            "std": float(np.std(y_pred)),
        },
        "track21_policy": {
            "treated_as_unlabeled_inference_only": True,
            "used_for_scaler_fitting": False,
            "used_for_target_alignment": False,
            "used_for_model_fitting": False,
            "used_for_validation_or_evaluation": False,
            "target_related_columns_ignored": True,
        },
    }


def main() -> int:
    output_paths = create_output_directories(OUTPUT_DIR)
    metadata = execute_blind_inference(output_paths)
    metadata_path = output_paths["metadata"] / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nFinal Track 21 blind inference complete: PASS")
    print(f"Run metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
