"""16_phase3_dataset_validation.py

End-to-end Phase III dataset integration validation.

This script demonstrates the current stable data path:

FeaturePreprocessor -> Phase3TargetAligner -> Phase3TorchDataset

It does not train models, recompute descriptors, modify Phase I/II outputs, or
inspect Track 21 targets.
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


FINAL_DATASET_PATH = REPO_ROOT / "processed_data" / "final_multimodal_dataset.csv"
WINDOW_SIZE = 5
TRAIN_TRACKS = [8, 10]
VAL_TRACKS = [14]
TARGET_GROUP = "pca_shape"
SAMPLE_INDICES = [0, 1, -1]

warnings.filterwarnings("ignore", category=FutureWarning)


def verify_metadata_order(label: str, source_meta, dataset: Phase3TorchDataset) -> bool:
    """Confirm dataset metadata preserves the incoming metadata order exactly."""

    required_cols = ["track_id", "frame_index", "x_position_mm"]
    source = source_meta[required_cols].reset_index(drop=True).copy()
    stored = dataset.meta_df[required_cols].reset_index(drop=True).copy()
    for col in required_cols:
        source[col] = np.asarray(source[col], dtype=float)
        stored[col] = np.asarray(stored[col], dtype=float)
    passed = bool(np.allclose(source.to_numpy(), stored.to_numpy(), rtol=0.0, atol=1.0e-12))
    print(f"  {label} metadata order preserved: {passed}")
    if not passed:
        raise ValueError(f"{label} metadata ordering changed during dataset construction.")
    return passed


def print_dataset_summary(label: str, dataset: Phase3TorchDataset) -> None:
    """Print dataset size and sample tensor shapes."""

    x0, y0, meta0 = dataset[0]
    track_counts = {int(k): int(v) for k, v in dataset.meta_df["track_id"].value_counts().sort_index().items()}

    print(f"\n{label} dataset")
    print(f"  size: {len(dataset)}")
    print(f"  feature tensor shape: {tuple(x0.shape)}")
    print(f"  target tensor shape: {tuple(y0.shape)}")
    print(f"  first metadata: {meta0}")
    print(f"  track counts: {track_counts}")


def inspect_samples(label: str, dataset: Phase3TorchDataset) -> None:
    """Retrieve a few sample items and print compact diagnostics."""

    print(f"\n{label} sample retrieval")
    for index in SAMPLE_INDICES:
        resolved_index = index if index >= 0 else len(dataset) + index
        x_item, y_item, meta_item = dataset[index]
        print(
            f"  index {resolved_index}: "
            f"x_shape={tuple(x_item.shape)}, "
            f"y_shape={tuple(y_item.shape)}, "
            f"track={int(meta_item['track_id'])}, "
            f"frame={int(meta_item['frame_index'])}, "
            f"x_mm={float(meta_item['x_position_mm']):.3f}"
        )


def main() -> int:
    print("Phase III PyTorch dataset integration validation")
    print(f"Dataset: {FINAL_DATASET_PATH}")
    print(f"Train tracks: {TRAIN_TRACKS}")
    print(f"Validation tracks: {VAL_TRACKS}")
    print(f"Window size: {WINDOW_SIZE}")
    print(f"Target group: {TARGET_GROUP}")

    # ==========================================
    # STEP 1: FEATURE PREPROCESSING
    # ==========================================
    preprocessor = FeaturePreprocessor()
    train_data, val_data = preprocessor.load_and_scale(
        csv_path=FINAL_DATASET_PATH,
        train_tracks=TRAIN_TRACKS,
        eval_tracks=VAL_TRACKS,
    )
    x_train_seq, train_meta = preprocessor.create_sequence_windows(train_data, window_size=WINDOW_SIZE)
    x_val_seq, val_meta = preprocessor.create_sequence_windows(val_data, window_size=WINDOW_SIZE)

    # ==========================================
    # STEP 2: TARGET ALIGNMENT
    # ==========================================
    aligner = Phase3TargetAligner(dataset_path=FINAL_DATASET_PATH)
    y_train, train_target_meta = aligner.align(train_meta, target_group=TARGET_GROUP, return_metadata=True)
    y_val, val_target_meta = aligner.align(val_meta, target_group=TARGET_GROUP, return_metadata=True)

    print("\nAligned arrays")
    print(f"  X_train_seq shape: {x_train_seq.shape}")
    print(f"  Y_train shape: {y_train.shape}")
    print(f"  train_meta rows: {len(train_meta)}")
    print(f"  X_val_seq shape: {x_val_seq.shape}")
    print(f"  Y_val shape: {y_val.shape}")
    print(f"  val_meta rows: {len(val_meta)}")
    print(f"  train target NaN count: {int(np.isnan(y_train).sum())}")
    print(f"  validation target NaN count: {int(np.isnan(y_val).sum())}")

    # ==========================================
    # STEP 3: TORCH DATASET PACKAGING
    # ==========================================
    train_dataset = Phase3TorchDataset(x_train_seq, y_train, train_target_meta)
    val_dataset = Phase3TorchDataset(x_val_seq, y_val, val_target_meta)

    print_dataset_summary("Training", train_dataset)
    print_dataset_summary("Validation", val_dataset)

    print("\nMetadata ordering checks")
    verify_metadata_order("Training", train_meta, train_dataset)
    verify_metadata_order("Validation", val_meta, val_dataset)

    inspect_samples("Training", train_dataset)
    inspect_samples("Validation", val_dataset)

    print("\nPhase III dataset integration validation complete: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
