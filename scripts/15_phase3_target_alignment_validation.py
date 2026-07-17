"""15_phase3_target_alignment_validation.py

Small Phase III validation script for target alignment only.

This script demonstrates that Phase3TargetAligner can consume metadata from
FeaturePreprocessor.create_sequence_windows and align frozen geometry targets
from processed_data/final_multimodal_dataset.csv.

It does not build models, create PyTorch tensors, recompute descriptors, or
modify Phase I/II outputs.
"""

from __future__ import annotations

from pathlib import Path
import sys
import warnings

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from scripts.phase3_data_loader import FeaturePreprocessor
from ml.targets import Phase3TargetAligner


FINAL_DATASET_PATH = REPO_ROOT / "processed_data" / "final_multimodal_dataset.csv"
WINDOW_SIZE = 5
TRAIN_TRACKS = [8, 10]
VAL_TRACKS = [14]
TARGET_GROUPS = ["pca_shape", "amplitude", "signed_elevation"]

warnings.filterwarnings("ignore", category=FutureWarning)


def print_alignment_summary(label: str, target_group: str, y, aligned_meta: pd.DataFrame, expected_len: int) -> None:
    target_cols = [col for col in aligned_meta.columns if col in ["pc1", "pc2", "pc3", "pc4", "pc5", "amplitude_um", "signed_elevation_um"]]
    nan_counts = {col: int(aligned_meta[col].isna().sum()) for col in target_cols}
    track_counts = {int(k): int(v) for k, v in aligned_meta["track_id"].value_counts().sort_index().items()}

    print(f"\n{label} / {target_group}")
    print(f"  Y shape: {y.shape}")
    print(f"  metadata rows: {len(aligned_meta)}")
    print(f"  expected metadata rows: {expected_len}")
    print(f"  length match: {len(y) == expected_len == len(aligned_meta)}")
    print(f"  track counts: {track_counts}")
    print(f"  target NaN counts: {nan_counts}")


def main() -> int:
    print("Phase III target-alignment validation")
    print(f"Dataset: {FINAL_DATASET_PATH}")
    print(f"Train tracks: {TRAIN_TRACKS}")
    print(f"Validation tracks: {VAL_TRACKS}")
    print(f"Window size: {WINDOW_SIZE}")

    preprocessor = FeaturePreprocessor()
    train_data, val_data = preprocessor.load_and_scale(
        csv_path=FINAL_DATASET_PATH,
        train_tracks=TRAIN_TRACKS,
        eval_tracks=VAL_TRACKS,
    )
    x_train_seq, train_meta = preprocessor.create_sequence_windows(train_data, window_size=WINDOW_SIZE)
    x_val_seq, val_meta = preprocessor.create_sequence_windows(val_data, window_size=WINDOW_SIZE)

    print("\nFeature sequence outputs")
    print(f"  X_train_seq shape: {x_train_seq.shape}")
    print(f"  train_meta rows: {len(train_meta)}")
    print(f"  X_val_seq shape: {x_val_seq.shape}")
    print(f"  val_meta rows: {len(val_meta)}")

    aligner = Phase3TargetAligner(dataset_path=FINAL_DATASET_PATH)
    aligner.validate_dataset()
    aligner.validate_meta(train_meta)
    aligner.validate_meta(val_meta)
    print("\nValidation summary")
    print("  dataset validation: PASS")
    print("  train_meta validation: PASS")
    print("  val_meta validation: PASS")

    for target_group in TARGET_GROUPS:
        y_train, train_target_meta = aligner.align(train_meta, target_group=target_group, return_metadata=True)
        y_val, val_target_meta = aligner.align(val_meta, target_group=target_group, return_metadata=True)

        print_alignment_summary("train", target_group, y_train, train_target_meta, len(train_meta))
        print_alignment_summary("validation", target_group, y_val, val_target_meta, len(val_meta))

    print("\nTarget alignment validation complete: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
