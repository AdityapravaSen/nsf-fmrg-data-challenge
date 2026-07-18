"""Phase III regression evaluation utilities.

This module provides reusable NumPy-based regression metrics for Phase III
modeling experiments. It is intentionally independent of PyTorch, model classes,
feature preprocessing, and target alignment.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd


ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]]


def _as_2d_float_array(values: ArrayLike, label: str) -> np.ndarray:
    """Convert input values to a finite 2D floating-point NumPy array."""

    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError(f"{label} must be non-empty.")
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{label} must be a 1D or 2D array; got shape {arr.shape}.")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{label} must have at least one sample and one target; got shape {arr.shape}.")
    if not np.isfinite(arr).all():
        bad_count = int((~np.isfinite(arr)).sum())
        raise ValueError(f"{label} contains {bad_count} non-finite values.")
    return arr


def validate_regression_inputs(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Validate and normalize regression targets and predictions.

    Parameters
    ----------
    y_true : array-like
        Ground-truth target values. May be 1D for single-output regression or
        2D for multi-output regression.
    y_pred : array-like
        Predicted target values with the same shape as `y_true`.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Validated `(y_true, y_pred)` arrays, both represented as 2D float arrays
        with shape `(n_samples, n_targets)`.

    Raises
    ------
    ValueError
        If either input is empty, has unsupported dimensionality, contains
        non-finite values, or if shapes/sample counts/target counts do not
        match.
    """

    true = _as_2d_float_array(y_true, "y_true")
    pred = _as_2d_float_array(y_pred, "y_pred")

    if true.shape != pred.shape:
        raise ValueError(f"y_true and y_pred must have identical shapes; got {true.shape} and {pred.shape}.")
    if true.shape[0] != pred.shape[0]:
        raise ValueError(f"Sample counts must match; got {true.shape[0]} and {pred.shape[0]}.")
    if true.shape[1] != pred.shape[1]:
        raise ValueError(f"Target counts must match; got {true.shape[1]} and {pred.shape[1]}.")
    return true, pred


def mean_absolute_error(y_true: ArrayLike, y_pred: ArrayLike, multioutput: str = "uniform_average") -> Union[float, np.ndarray]:
    """Compute mean absolute error.

    Parameters
    ----------
    y_true, y_pred : array-like
        Ground-truth and predicted values with identical shape.
    multioutput : {'uniform_average', 'raw_values'}, optional
        If `uniform_average`, return a scalar average across targets. If
        `raw_values`, return one value per target.

    Returns
    -------
    float or numpy.ndarray
        MAE value or per-target MAE values.
    """

    true, pred = validate_regression_inputs(y_true, y_pred)
    values = np.mean(np.abs(true - pred), axis=0)
    return _aggregate_multioutput(values, multioutput)


def root_mean_squared_error(y_true: ArrayLike, y_pred: ArrayLike, multioutput: str = "uniform_average") -> Union[float, np.ndarray]:
    """Compute root mean squared error."""

    true, pred = validate_regression_inputs(y_true, y_pred)
    values = np.sqrt(np.mean((true - pred) ** 2, axis=0))
    return _aggregate_multioutput(values, multioutput)


def median_absolute_error(y_true: ArrayLike, y_pred: ArrayLike, multioutput: str = "uniform_average") -> Union[float, np.ndarray]:
    """Compute median absolute error."""

    true, pred = validate_regression_inputs(y_true, y_pred)
    values = np.median(np.abs(true - pred), axis=0)
    return _aggregate_multioutput(values, multioutput)


def r2_score(y_true: ArrayLike, y_pred: ArrayLike, multioutput: str = "uniform_average") -> Union[float, np.ndarray]:
    """Compute coefficient of determination, R².

    Targets with zero variance receive `nan` for their raw R² value. Aggregate
    R² uses `nanmean` so constant targets do not dominate multi-output results.
    """

    true, pred = validate_regression_inputs(y_true, y_pred)
    ss_res = np.sum((true - pred) ** 2, axis=0)
    ss_tot = np.sum((true - np.mean(true, axis=0)) ** 2, axis=0)
    values = np.full(true.shape[1], np.nan, dtype=float)
    non_constant = ss_tot > 0
    values[non_constant] = 1.0 - (ss_res[non_constant] / ss_tot[non_constant])
    return _aggregate_multioutput(values, multioutput)


def _aggregate_multioutput(values: np.ndarray, multioutput: str) -> Union[float, np.ndarray]:
    """Aggregate or return per-target metric values."""

    if multioutput == "raw_values":
        return values
    if multioutput == "uniform_average":
        return float(np.nanmean(values))
    raise ValueError("multioutput must be either 'uniform_average' or 'raw_values'.")


def _target_names(n_targets: int, target_names: Optional[Sequence[str]] = None) -> List[str]:
    """Build validated target names for metric dictionaries."""

    if target_names is None:
        return [f"target_{i}" for i in range(n_targets)]
    names = list(target_names)
    if len(names) != n_targets:
        raise ValueError(f"target_names length must match number of targets; got {len(names)} and {n_targets}.")
    return [str(name) for name in names]


def evaluate_regression(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    target_names: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """Compute a structured regression metric summary.

    Parameters
    ----------
    y_true, y_pred : array-like
        Ground-truth and predicted values. Inputs may be single-output or
        multi-output, but shapes must match exactly.
    target_names : sequence of str, optional
        Optional names used for per-target metric dictionaries.

    Returns
    -------
    dict
        Structured metric payload containing sample/target counts, overall
        metrics, and one per-target metrics dictionary for each target column.
    """

    true, pred = validate_regression_inputs(y_true, y_pred)
    names = _target_names(true.shape[1], target_names)

    mae_raw = mean_absolute_error(true, pred, multioutput="raw_values")
    rmse_raw = root_mean_squared_error(true, pred, multioutput="raw_values")
    medae_raw = median_absolute_error(true, pred, multioutput="raw_values")
    r2_raw = r2_score(true, pred, multioutput="raw_values")

    per_target: Dict[str, Dict[str, float]] = {}
    for idx, name in enumerate(names):
        per_target[name] = {
            "mae": float(mae_raw[idx]),
            "rmse": float(rmse_raw[idx]),
            "median_absolute_error": float(medae_raw[idx]),
            "r2": float(r2_raw[idx]) if np.isfinite(r2_raw[idx]) else float("nan"),
        }

    return {
        "n_samples": int(true.shape[0]),
        "n_targets": int(true.shape[1]),
        "overall": {
            "mae": float(mean_absolute_error(true, pred)),
            "rmse": float(root_mean_squared_error(true, pred)),
            "median_absolute_error": float(median_absolute_error(true, pred)),
            "r2": float(r2_score(true, pred)),
        },
        "per_target": per_target,
    }


def evaluate_by_track(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    meta_df: pd.DataFrame,
    target_names: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """Compute regression metrics grouped by track ID.

    Parameters
    ----------
    y_true, y_pred : array-like
        Ground-truth and predicted arrays with identical shape.
    meta_df : pandas.DataFrame
        Metadata dataframe aligned row-for-row to `y_true` and `y_pred`. It must
        contain `track_id`.
    target_names : sequence of str, optional
        Optional target names passed through to `evaluate_regression`.

    Returns
    -------
    dict
        Structured payload containing overall metrics and a `by_track` mapping
        keyed by track ID strings in sorted track order.
    """

    true, pred = validate_regression_inputs(y_true, y_pred)
    if not isinstance(meta_df, pd.DataFrame):
        raise TypeError("meta_df must be a pandas DataFrame.")
    if "track_id" not in meta_df.columns:
        raise ValueError("meta_df must contain a 'track_id' column.")
    if len(meta_df) != true.shape[0]:
        raise ValueError(f"len(meta_df) must match number of samples; got {len(meta_df)} and {true.shape[0]}.")
    if meta_df["track_id"].isna().any():
        raise ValueError("meta_df contains missing track_id values.")

    track_values = pd.to_numeric(meta_df["track_id"], errors="raise").astype(int).to_numpy()
    tracks = sorted(int(track) for track in np.unique(track_values))

    by_track: Dict[str, Dict[str, object]] = {}
    for track in tracks:
        mask = track_values == track
        by_track[str(track)] = evaluate_regression(true[mask], pred[mask], target_names=target_names)

    return {
        "overall": evaluate_regression(true, pred, target_names=target_names),
        "by_track": by_track,
        "track_order": tracks,
    }
