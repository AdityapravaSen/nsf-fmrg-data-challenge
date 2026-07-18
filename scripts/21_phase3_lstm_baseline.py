"""21_phase3_lstm_baseline.py

First Phase III neural-network baseline modeling experiment.

This script trains a simple one-layer LSTM regressor to predict the frozen PCA
shape targets (PC1--PC5) from Phase III feature windows. Unlike the previous
Linear Regression, Ridge Regression, and Random Forest baselines, this experiment
uses the sequence representation directly and does not flatten temporal windows.

This is an experiment script, not a reusable training framework. It does not
modify preprocessing, target alignment, metrics, descriptors, or datasets.
"""

from __future__ import annotations

from pathlib import Path
import random
import sys
import warnings

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from scripts.phase3_data_loader import FeaturePreprocessor
from ml.datasets import Phase3TorchDataset
from ml.targets import Phase3TargetAligner
from ml.metrics import evaluate_by_track, evaluate_regression


FINAL_DATASET_PATH = REPO_ROOT / "processed_data" / "final_multimodal_dataset.csv"
WINDOW_SIZE = 5
TRAIN_TRACKS = [8, 10]
VAL_TRACKS = [14]
TARGET_GROUP = "pca_shape"
TARGET_NAMES = ["pc1", "pc2", "pc3", "pc4", "pc5"]

SEED = 42
HIDDEN_SIZE = 48
NUM_LAYERS = 1
BATCH_SIZE = 32
EPOCHS = 120
LEARNING_RATE = 1.0e-3
WEIGHT_DECAY = 0.0

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

RANDOM_FOREST_VALIDATION_BASELINE = {
    "thermal_only": {"mae": 1.892585, "rmse": 2.115979, "median_absolute_error": 1.821794, "r2": -1.635473},
    "sem_only": {"mae": 1.314589, "rmse": 1.664540, "median_absolute_error": 1.135141, "r2": -0.848344},
    "thermal_sem": {"mae": 1.427145, "rmse": 1.747150, "median_absolute_error": 1.245874, "r2": -0.946050},
}

warnings.filterwarnings("ignore", category=FutureWarning)


class LSTMRegressor(nn.Module):
    """Simple one-layer LSTM regressor using the final hidden state."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _sequence, (hidden, _cell) = self.lstm(x)
        last_hidden = hidden[-1]
        return self.output(last_hidden)


def set_reproducible_seed(seed: int) -> None:
    """Set random seeds for reproducible baseline execution."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def build_feature_group_config() -> dict[str, list[str]]:
    """Return feature-column groups from the stable FeaturePreprocessor schema."""

    schema = FeaturePreprocessor()
    return {
        "thermal_only": list(schema.thermal_features),
        "sem_only": list(schema.sem_features),
        "thermal_sem": list(schema.thermal_features + schema.sem_features),
    }


def build_phase3_datasets(feature_columns: list[str]) -> tuple[Phase3TorchDataset, Phase3TorchDataset, object, object]:
    """Build train/validation torch datasets and aligned metadata."""

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

    train_dataset = Phase3TorchDataset(x_train_seq, y_train, train_target_meta)
    val_dataset = Phase3TorchDataset(x_val_seq, y_val, val_target_meta)
    return train_dataset, val_dataset, train_target_meta, val_target_meta


def validate_predictions(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> None:
    """Validate prediction array shape before metric evaluation."""

    if y_true.shape != y_pred.shape:
        raise ValueError(f"{label} prediction shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")
    if not np.isfinite(y_pred).all():
        bad_count = int((~np.isfinite(y_pred)).sum())
        raise ValueError(f"{label} predictions contain {bad_count} non-finite values.")


def train_lstm(model: nn.Module, train_loader: DataLoader, device: torch.device) -> list[float]:
    """Train the LSTM baseline with a fixed simple training loop."""

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    losses: list[float] = []
    model.train()

    for _epoch in range(EPOCHS):
        epoch_loss = 0.0
        n_samples = 0
        for x_batch, y_batch, _meta in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            prediction = model(x_batch)
            loss = criterion(prediction, y_batch)
            loss.backward()
            optimizer.step()

            batch_size = int(x_batch.shape[0])
            epoch_loss += float(loss.item()) * batch_size
            n_samples += batch_size
        losses.append(epoch_loss / max(1, n_samples))
    return losses


def predict(model: nn.Module, dataset: Phase3TorchDataset, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Generate predictions and return targets for one dataset."""

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    y_true_parts = []
    y_pred_parts = []
    model.eval()
    with torch.no_grad():
        for x_batch, y_batch, _meta in loader:
            x_batch = x_batch.to(device)
            y_pred = model(x_batch).cpu().numpy()
            y_pred_parts.append(y_pred)
            y_true_parts.append(y_batch.cpu().numpy())
    return np.vstack(y_true_parts), np.vstack(y_pred_parts)


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
    """Describe LSTM performance relative to a prior baseline."""

    delta = candidate_value - baseline_value
    if abs(delta) <= tolerance:
        return "similar"
    if lower_is_better:
        return "improved" if delta < 0 else "worsened"
    return "improved" if delta > 0 else "worsened"


def run_feature_group_experiment(group_name: str, feature_columns: list[str], device: torch.device) -> dict[str, object]:
    """Run one LSTM baseline for one feature group."""

    print("\n" + "=" * 78)
    print(f"Feature group: {group_name}")
    print(f"Feature columns ({len(feature_columns)}): {feature_columns}")

    set_reproducible_seed(SEED)
    train_dataset, val_dataset, train_meta, val_meta = build_phase3_datasets(feature_columns)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )

    sample_x, sample_y, _sample_meta = train_dataset[0]
    input_size = int(sample_x.shape[-1])
    output_size = int(sample_y.shape[-1])
    model = LSTMRegressor(input_size=input_size, hidden_size=HIDDEN_SIZE, output_size=output_size, num_layers=NUM_LAYERS).to(device)

    print("Shapes")
    print(f"  X_train_seq:  {train_dataset.X.shape}")
    print(f"  Y_train:      {train_dataset.Y.shape}")
    print(f"  X_val_seq:    {val_dataset.X.shape}")
    print(f"  Y_val:        {val_dataset.Y.shape}")
    print("Model")
    print(f"  input_size={input_size}, hidden_size={HIDDEN_SIZE}, output_size={output_size}, num_layers={NUM_LAYERS}")
    print(f"  epochs={EPOCHS}, batch_size={BATCH_SIZE}, learning_rate={LEARNING_RATE}")

    losses = train_lstm(model, train_loader, device)
    y_train, y_train_pred = predict(model, train_dataset, device)
    y_val, y_val_pred = predict(model, val_dataset, device)
    validate_predictions(y_train, y_train_pred, "training")
    validate_predictions(y_val, y_val_pred, "validation")

    train_metrics = evaluate_regression(y_train, y_train_pred, target_names=TARGET_NAMES)
    val_metrics = evaluate_regression(y_val, y_val_pred, target_names=TARGET_NAMES)
    train_by_track = evaluate_by_track(y_train, y_train_pred, train_meta, target_names=TARGET_NAMES)
    val_by_track = evaluate_by_track(y_val, y_val_pred, val_meta, target_names=TARGET_NAMES)

    print("Training loss")
    print(f"  first_epoch_mse={losses[0]:.6f}, final_epoch_mse={losses[-1]:.6f}")
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
        "train_shape": train_dataset.X.shape,
        "val_shape": val_dataset.X.shape,
        "target_shape_train": train_dataset.Y.shape,
        "target_shape_val": val_dataset.Y.shape,
        "losses": losses,
    }


def print_comparison_summary(results: list[dict[str, object]]) -> None:
    """Print validation metrics and descriptive comparison to prior baselines."""

    print("\n" + "=" * 78)
    print("LSTM baseline comparison: validation split")
    print("Target: PCA shape (PC1--PC5)")
    print("Split: train Tracks 8+10, validation Track 14")
    print(f"Architecture: one-layer LSTM, hidden_size={HIDDEN_SIZE}")
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
    print("Feature group                 vs Linear MAE    vs Ridge MAE     vs RF MAE        vs Linear R2     vs Ridge R2      vs RF R2")
    print("-" * 126)
    for result in results:
        group_name = result["group_name"]
        lstm = result["val_metrics"]["overall"]
        linear = LINEAR_REGRESSION_VALIDATION_BASELINE[group_name]
        ridge = RIDGE_REGRESSION_VALIDATION_BASELINE[group_name]
        forest = RANDOM_FOREST_VALIDATION_BASELINE[group_name]
        print(
            f"{group_name:<28} "
            f"{describe_change(lstm['mae'], linear['mae'], lower_is_better=True):<16} "
            f"{describe_change(lstm['mae'], ridge['mae'], lower_is_better=True):<16} "
            f"{describe_change(lstm['mae'], forest['mae'], lower_is_better=True):<16} "
            f"{describe_change(lstm['r2'], linear['r2'], lower_is_better=False):<16} "
            f"{describe_change(lstm['r2'], ridge['r2'], lower_is_better=False):<16} "
            f"{describe_change(lstm['r2'], forest['r2'], lower_is_better=False):<16}"
        )


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Phase III baseline experiment: one-layer LSTM Regression")
    print(f"Dataset: {FINAL_DATASET_PATH}")
    print(f"Train tracks: {TRAIN_TRACKS}")
    print(f"Validation tracks: {VAL_TRACKS}")
    print(f"Window size: {WINDOW_SIZE}")
    print(f"Target group: {TARGET_GROUP} ({TARGET_NAMES})")
    print(f"Device: {device}")
    print("Sequence handling: uses X_seq directly; no flattening")

    feature_groups = build_feature_group_config()
    results = []
    for group_name, feature_columns in feature_groups.items():
        results.append(run_feature_group_experiment(group_name, feature_columns, device))

    print_comparison_summary(results)

    if len(results) != 3:
        raise RuntimeError(f"Expected 3 feature-group experiments; completed {len(results)}.")

    print("\nPhase III baseline LSTM experiment complete: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
