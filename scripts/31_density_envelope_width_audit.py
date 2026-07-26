"""Experiment 31: finite-support density-envelope width audit.

Two locked stages:
31A: extract and audit density_envelope_width_mm from development-track Wyko
     finite-support density profiles.
31B: if and only if 31A returns PASS or CONDITIONAL PASS, run the same centered
     thermal-window BayesianRidge learnability diagnostic used in Experiment 30,
     with the additional locked baselines and decision gates.

Track 21 Wyko geometry remains sealed throughout this script.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import csv
import json
import math
import platform
import subprocess
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.stats import spearmanr
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from nsf_fmrg_data import load_wyko_asc  # organizer loader, unchanged


warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

EXPERIMENT_NAME = "31_density_envelope_width_audit"
DATASET_PATH = REPO_ROOT / "processed_data" / "final_multimodal_dataset.csv"
HEIGHT_DIR = REPO_ROOT / "data" / "raw" / "height_maps"
EXP29_ANCHOR_PATH = (
    REPO_ROOT
    / "processed_data"
    / "run_outputs"
    / "29_dominant_component_core_geometry_20260724_174420"
    / "tables"
    / "anchor_aggregated_width.csv"
)
EXP30_FOLD_METRICS_PATH = (
    REPO_ROOT
    / "processed_data"
    / "run_outputs"
    / "30_core_width_learnability_diagnostic_20260724_204318"
    / "tables"
    / "loto_fold_metrics.csv"
)
OUTPUT_ROOT = REPO_ROOT / "processed_data" / "run_outputs"

TRACK_IDS = (8, 10, 14)
TRACK_SET = set(TRACK_IDS)
SEALED_TRACKS = {21}
EXPECTED_ROWS_PER_TRACK = 400
EXPECTED_DEV_ROWS = EXPECTED_ROWS_PER_TRACK * len(TRACK_IDS)
assert TRACK_SET == {8, 10, 14}
assert not (TRACK_SET & SEALED_TRACKS), "Track 21 is sealed and cannot be used."

KERNEL_HALF_WIDTH_PX = 25
KERNEL_WIDTH_PX = 2 * KERNEL_HALF_WIDTH_PX + 1
DENSITY_THRESHOLD_TAU = 0.30
AGG_HALF_WINDOW_MM = 0.10
MIN_VALID_NATIVE_COLUMNS_PER_ANCHOR = 25

FEATURE_BASE_COLUMNS = ["peak_temp", "sqrt_mp_area", "mp_length"]
WINDOW_OFFSETS = (-2, -1, 0, 1, 2)
WINDOW_FEATURE_COLUMNS = [f"{feature}_t{offset:+d}" for offset in WINDOW_OFFSETS for feature in FEATURE_BASE_COLUMNS]
PROHIBITED_MODEL_COLUMNS = {
    "track_id",
    "frame_index",
    "x_position_mm",
    "x_norm",
    "sem_tile_index",
    "substrate_roughness_variance",
    "substrate_mean_intensity",
}
FOLDS = [
    {"fold": "holdout_8", "train_tracks": [10, 14], "val_track": 8},
    {"fold": "holdout_10", "train_tracks": [8, 14], "val_track": 10},
    {"fold": "holdout_14", "train_tracks": [8, 10], "val_track": 14},
]

EXP30_LOCKED_RESULTS = {
    "classification": "NO SIGNAL",
    "fold_r2": {"holdout_8": -0.45898073836909115, "holdout_10": -0.0953594323266227, "holdout_14": -0.3385542277651039},
    "fold_pearson_r": {"holdout_8": 0.03577194377811633, "holdout_10": 0.05539879478148959, "holdout_14": 0.0012387372657617834},
    "pooled_oof_r2": 0.09820896156464198,
}


@dataclass(frozen=True)
class TrackData:
    track_id: int
    source_file: str
    Z_mm: np.ndarray
    x_mm: np.ndarray
    y_mm: np.ndarray


@dataclass(frozen=True)
class StageBSampleDiagnostics:
    eligible_targets: int
    samples_after_windowing: int
    discarded_incomplete_window: int
    discarded_nonfinite_features: int
    samples_by_track: Dict[int, int]


def git_text(args: Sequence[str]) -> str:
    for git_cmd in ("git", "/usr/bin/git"):
        try:
            return subprocess.check_output([git_cmd, *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            continue
    return "unavailable"


def make_json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def safe_stat(values: Sequence[float], func, default: float = float("nan")) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return default
    return float(func(arr))


def finite_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    idx = np.flatnonzero(np.asarray(mask, dtype=bool))
    if idx.size == 0:
        return []
    breaks = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[idx[0], idx[breaks + 1]]
    stops = np.r_[idx[breaks] + 1, idx[-1] + 1]
    return [(int(a), int(b)) for a, b in zip(starts, stops)]


def load_track(track_id: int) -> TrackData:
    if track_id in SEALED_TRACKS:
        raise RuntimeError("Track 21 Wyko geometry is sealed and must not be loaded.")
    if track_id not in TRACK_SET:
        raise RuntimeError(f"Unsupported non-development track requested: {track_id}")
    hm = load_wyko_asc(HEIGHT_DIR, track_id, crop_to_common=True)
    Z_mm = np.asarray(hm["Z_mm"], dtype=float)
    x_mm = np.asarray(hm["x_actual_mm"], dtype=float)
    y_mm = np.asarray(hm["y_mm"], dtype=float)
    if Z_mm.shape != (len(y_mm), len(x_mm)):
        raise RuntimeError(f"Track {track_id}: Z shape {Z_mm.shape} does not match coordinates.")
    if not (np.isfinite(x_mm).all() and np.isfinite(y_mm).all()):
        raise RuntimeError(f"Track {track_id}: nonfinite coordinate array.")
    return TrackData(track_id=track_id, source_file=str(hm["file"]), Z_mm=Z_mm, x_mm=x_mm, y_mm=y_mm)


def read_development_dataset_prefix(dataset_path: Path) -> pd.DataFrame:
    required = {"track_id", "frame_index", "x_position_mm", "peak_temp", "mp_area_px", "mp_length"}
    rows: List[Dict[str, float | int]] = []
    with dataset_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise RuntimeError(f"Dataset missing required columns: {missing}")
        for row_number, row in enumerate(reader, start=1):
            if row_number > EXPECTED_DEV_ROWS:
                break
            track_id = int(float(row["track_id"]))
            if track_id not in TRACK_SET:
                raise RuntimeError(f"Non-development track {track_id} encountered in first {EXPECTED_DEV_ROWS} rows.")
            mp_area_px = float(row["mp_area_px"])
            rows.append(
                {
                    "track_id": track_id,
                    "frame_index": int(float(row["frame_index"])),
                    "x_position_mm": float(row["x_position_mm"]),
                    "peak_temp": float(row["peak_temp"]),
                    "sqrt_mp_area": math.sqrt(mp_area_px) if math.isfinite(mp_area_px) and mp_area_px >= 0 else float("nan"),
                    "mp_length": float(row["mp_length"]),
                }
            )
    df = pd.DataFrame(rows)
    if len(df) != EXPECTED_DEV_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_DEV_ROWS} development rows; observed {len(df)}.")
    observed = set(int(t) for t in df["track_id"].unique())
    if observed != TRACK_SET:
        raise RuntimeError(f"Expected exactly tracks {sorted(TRACK_SET)}; observed {sorted(observed)}.")
    if observed & SEALED_TRACKS:
        raise RuntimeError("Track 21 appeared in development prefix.")
    counts = df.groupby("track_id").size().to_dict()
    for track_id in TRACK_IDS:
        if int(counts.get(track_id, 0)) != EXPECTED_ROWS_PER_TRACK:
            raise RuntimeError(f"Track {track_id}: expected 400 rows; observed {counts.get(track_id, 0)}.")
    return df.sort_values(["track_id", "frame_index"]).reset_index(drop=True)


def read_anchors(dataset_df: pd.DataFrame) -> pd.DataFrame:
    anchors = dataset_df[["track_id", "frame_index", "x_position_mm"]].copy()
    anchors = anchors.sort_values(["track_id", "x_position_mm", "frame_index"]).reset_index(drop=True)
    anchors["anchor_index"] = anchors.groupby("track_id").cumcount().astype(int)
    return anchors[["track_id", "anchor_index", "frame_index", "x_position_mm"]]


def extract_density_width(track: TrackData) -> pd.DataFrame:
    finite_mask = np.isfinite(track.Z_mm).astype(float)
    density = ndimage.uniform_filter1d(
        finite_mask,
        size=KERNEL_WIDTH_PX,
        axis=0,
        mode="constant",
        cval=0.0,
        origin=0,
    )
    rows: List[Dict[str, object]] = []
    n_y = len(track.y_mm)
    for j, x in enumerate(track.x_mm):
        rho = density[:, j]
        support = rho >= DENSITY_THRESHOLD_TAU
        runs = finite_runs(support)
        if not runs:
            rows.append(
                {
                    "track_id": track.track_id,
                    "x_index": int(j),
                    "x_position_mm": float(x),
                    "density_envelope_width_mm": float("nan"),
                    "left_boundary_mm": float("nan"),
                    "right_boundary_mm": float("nan"),
                    "envelope_start_y_index": -1,
                    "envelope_stop_y_index_exclusive": -1,
                    "envelope_length_px": 0,
                    "n_threshold_runs": 0,
                    "boundary_touching": False,
                    "column_finite_fraction": float(np.mean(finite_mask[:, j])),
                    "max_density": float(np.nanmax(rho)),
                }
            )
            continue
        lengths = np.asarray([b - a for a, b in runs], dtype=int)
        order = np.argsort(-lengths, kind="stable")
        a, b = runs[int(order[0])]
        left = float(track.y_mm[a])
        right = float(track.y_mm[b - 1])
        rows.append(
            {
                "track_id": track.track_id,
                "x_index": int(j),
                "x_position_mm": float(x),
                "density_envelope_width_mm": float(right - left),
                "left_boundary_mm": left,
                "right_boundary_mm": right,
                "envelope_start_y_index": int(a),
                "envelope_stop_y_index_exclusive": int(b),
                "envelope_length_px": int(b - a),
                "n_threshold_runs": int(len(runs)),
                "boundary_touching": bool(a == 0 or b == n_y),
                "column_finite_fraction": float(np.mean(finite_mask[:, j])),
                "max_density": float(np.nanmax(rho)),
            }
        )
    return pd.DataFrame(rows)


def aggregate_to_anchors(native: pd.DataFrame, anchors: pd.DataFrame, x_mm: np.ndarray) -> pd.DataFrame:
    track_id = int(native["track_id"].iloc[0])
    widths = native["density_envelope_width_mm"].to_numpy(dtype=float)
    boundary_touching = native["boundary_touching"].to_numpy(dtype=bool)
    track_anchors = anchors[anchors["track_id"] == track_id].sort_values("anchor_index")
    x_min = float(np.nanmin(x_mm))
    x_max = float(np.nanmax(x_mm))
    rows: List[Dict[str, object]] = []
    for arow in track_anchors.itertuples(index=False):
        xa = float(arow.x_position_mm)
        in_window = np.abs(x_mm - xa) <= AGG_HALF_WINDOW_MM
        valid = in_window & np.isfinite(widths)
        n_window = int(np.sum(in_window))
        n_valid = int(np.sum(valid))
        window_intersects_domain = bool((xa + AGG_HALF_WINDOW_MM >= x_min) and (xa - AGG_HALF_WINDOW_MM <= x_max))
        rows.append(
            {
                "track_id": track_id,
                "anchor_index": int(arow.anchor_index),
                "frame_index": int(arow.frame_index),
                "x_position_mm": xa,
                "window_intersects_native_x_domain": window_intersects_domain,
                "outside_native_x_domain": bool(not window_intersects_domain),
                "density_envelope_width_mm": safe_stat(widths[valid], np.nanmedian),
                "window_native_column_count": n_window,
                "window_valid_column_count": n_valid,
                "window_valid_column_fraction": float(n_valid / n_window) if n_window else 0.0,
                "window_boundary_touching_fraction": float(np.mean(boundary_touching[valid])) if n_valid else float("nan"),
                "anchor_valid": bool(n_valid >= MIN_VALID_NATIVE_COLUMNS_PER_ANCHOR),
            }
        )
    return pd.DataFrame(rows)


def immediate_valid_adjacent_jumps(df: pd.DataFrame, value_col: str, index_col: str) -> Tuple[np.ndarray, int, int]:
    ordered = df.sort_values(index_col).reset_index(drop=True)
    vals = ordered[value_col].to_numpy(dtype=float)
    idx = ordered[index_col].to_numpy(dtype=int)
    adjacent_index = np.diff(idx) == 1
    finite_pair = np.isfinite(vals[:-1]) & np.isfinite(vals[1:])
    used = adjacent_index & finite_pair
    blocked_invalid_or_gap = int(np.sum(adjacent_index & ~finite_pair))
    nonadjacent = int(np.sum(~adjacent_index))
    return np.abs(np.diff(vals)[used]), blocked_invalid_or_gap, nonadjacent


def build_stage_a_summaries(native_all: pd.DataFrame, anchor_all: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, object]] = []
    criteria_rows: List[Dict[str, object]] = []
    exp29 = pd.read_csv(EXP29_ANCHOR_PATH)
    for track_id in TRACK_IDS:
        native = native_all[native_all["track_id"] == track_id].sort_values("x_index")
        anchors = anchor_all[anchor_all["track_id"] == track_id].sort_values("anchor_index")
        valid_native = native["density_envelope_width_mm"].to_numpy(dtype=float)
        valid_anchor_vals = anchors.loc[anchors["anchor_valid"].astype(bool), "density_envelope_width_mm"].to_numpy(dtype=float)
        native_jumps, native_blocked, native_nonadj = immediate_valid_adjacent_jumps(native, "density_envelope_width_mm", "x_index")
        anchor_for_jump = anchors.copy()
        anchor_for_jump.loc[~anchor_for_jump["anchor_valid"].astype(bool), "density_envelope_width_mm"] = np.nan
        anchor_jumps, anchor_blocked, anchor_nonadj = immediate_valid_adjacent_jumps(anchor_for_jump, "density_envelope_width_mm", "anchor_index")
        inside_domain = anchors["window_intersects_native_x_domain"].astype(bool)
        zero_valid_inside = int(((anchors["window_valid_column_count"] == 0) & inside_domain).sum())
        exp29_vals = exp29.loc[(exp29["track_id"] == track_id) & (exp29["anchor_valid"].astype(bool)), "local_core_width_mm"].to_numpy(dtype=float)
        mean_val = safe_stat(valid_anchor_vals, np.nanmean)
        std_val = safe_stat(valid_anchor_vals, lambda a: np.nanstd(a, ddof=1) if len(a) > 1 else 0.0)
        row = {
            "track_id": track_id,
            "native_column_count": int(len(native)),
            "native_valid_count": int(np.isfinite(valid_native).sum()),
            "native_valid_fraction": float(np.isfinite(valid_native).mean()),
            "anchor_count": int(len(anchors)),
            "anchor_valid_count": int(anchors["anchor_valid"].sum()),
            "anchor_valid_fraction": float(anchors["anchor_valid"].mean()),
            "mean_density_envelope_width_mm": mean_val,
            "std_density_envelope_width_mm": std_val,
            "cov_density_envelope_width": float(std_val / mean_val) if np.isfinite(mean_val) and mean_val != 0 else float("nan"),
            "native_abs_adjacent_jump_p95_mm": safe_stat(native_jumps, lambda a: np.nanpercentile(a, 95)),
            "native_abs_adjacent_jump_n_pairs": int(native_jumps.size),
            "native_adjacent_pairs_blocked_by_invalid": native_blocked,
            "anchor_abs_adjacent_jump_p95_mm": safe_stat(anchor_jumps, lambda a: np.nanpercentile(a, 95)),
            "anchor_abs_adjacent_jump_n_pairs": int(anchor_jumps.size),
            "anchor_adjacent_pairs_blocked_by_invalid": anchor_blocked,
            "anchor_nonadjacent_pairs_seen": anchor_nonadj,
            "zero_valid_column_anchors_inside_domain": zero_valid_inside,
            "boundary_touching_fraction_native_valid": float(native.loc[np.isfinite(valid_native), "boundary_touching"].mean()) if np.isfinite(valid_native).any() else float("nan"),
            "exp29_mean_local_core_width_mm": safe_stat(exp29_vals, np.nanmean),
            "exp29_std_local_core_width_mm": safe_stat(exp29_vals, lambda a: np.nanstd(a, ddof=1) if len(a) > 1 else 0.0),
        }
        rows.append(row)
        criteria = [
            ("SA1", "Native valid extraction fraction >= 95%", row["native_valid_fraction"], ">= 0.95", row["native_valid_fraction"] >= 0.95),
            ("SA2", "Anchor valid fraction >= 92%", row["anchor_valid_fraction"], ">= 0.92", row["anchor_valid_fraction"] >= 0.92),
            ("SA3", "p95 absolute adjacent valid-anchor width jump <= 0.20 mm", row["anchor_abs_adjacent_jump_p95_mm"], "<= 0.20", np.isfinite(row["anchor_abs_adjacent_jump_p95_mm"]) and row["anchor_abs_adjacent_jump_p95_mm"] <= 0.20),
            ("SA4", "Mean density-envelope width finite and > 0", row["mean_density_envelope_width_mm"], "> 0 and finite", np.isfinite(row["mean_density_envelope_width_mm"]) and row["mean_density_envelope_width_mm"] > 0),
            ("SA5", "Within-track standard deviation > 0", row["std_density_envelope_width_mm"], "> 0", np.isfinite(row["std_density_envelope_width_mm"]) and row["std_density_envelope_width_mm"] > 0),
            ("SA6", "Zero-valid-column anchors inside usable/native x-domain <= 5", row["zero_valid_column_anchors_inside_domain"], "<= 5", row["zero_valid_column_anchors_inside_domain"] <= 5),
        ]
        for code, desc, observed, threshold, passed in criteria:
            criteria_rows.append({"track_id": track_id, "criterion": code, "description": desc, "observed_value": observed, "threshold": threshold, "pass": bool(passed)})
    summary = pd.DataFrame(rows)
    criteria_df = pd.DataFrame(criteria_rows)
    mean_by_track = {int(r.track_id): float(r.mean_density_envelope_width_mm) for r in summary.itertuples(index=False)}
    ordering = sorted(mean_by_track, key=mean_by_track.get)
    cross = pd.DataFrame(
        [
            {
                "diagnostic": "cross_track_mean_ordering",
                "ordering_low_to_high": " <= ".join(f"T{t}" for t in ordering),
                "matches_T14_le_T10_le_T8": bool(mean_by_track[14] <= mean_by_track[10] <= mean_by_track[8]),
                "used_for_stage_a_pass_fail": False,
                "track8_mean": mean_by_track[8],
                "track10_mean": mean_by_track[10],
                "track14_mean": mean_by_track[14],
            }
        ]
    )
    return summary, criteria_df, cross


def decide_stage_a(criteria_df: pd.DataFrame) -> Tuple[str, str]:
    non_sa3 = criteria_df[criteria_df["criterion"] != "SA3"]
    sa3 = criteria_df[criteria_df["criterion"] == "SA3"]
    if bool(non_sa3["pass"].all()) and bool(sa3["pass"].all()):
        return "PASS", "All Stage A criteria SA1-SA6 passed for all development tracks."
    if not bool(non_sa3["pass"].all()):
        failed = non_sa3.loc[~non_sa3["pass"], ["track_id", "criterion"]].to_dict("records")
        return "FAIL", f"At least one non-SA3 criterion failed: {failed}"
    if bool((sa3["observed_value"].astype(float) < 0.30).all()):
        return "CONDITIONAL PASS", "Only SA3 failed, and every SA3 p95 adjacent-anchor jump was < 0.30 mm."
    failed = sa3.loc[sa3["observed_value"].astype(float) >= 0.30, ["track_id", "observed_value"]].to_dict("records")
    return "FAIL", f"At least one SA3 p95 adjacent-anchor jump was >= 0.30 mm: {failed}"


def pearson_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or float(np.std(y_true)) == 0.0 or float(np.std(y_pred)) == 0.0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def spearman_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or float(np.std(y_true)) == 0.0 or float(np.std(y_pred)) == 0.0:
        return float("nan")
    rho = spearmanr(y_true, y_pred, nan_policy="omit").correlation
    return float(rho) if rho is not None and np.isfinite(rho) else float("nan")


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, include_pred_stats: bool = True) -> Dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    out: Dict[str, float | int] = {
        "n": int(len(y_true)),
        "target_mean": float(np.mean(y_true)),
        "target_std": float(np.std(y_true, ddof=1)) if len(y_true) > 1 else 0.0,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "median_absolute_error": float(median_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "pearson_r": pearson_corr(y_true, y_pred),
        "spearman_rho": spearman_corr(y_true, y_pred),
    }
    if include_pred_stats:
        out["prediction_mean"] = float(np.mean(y_pred))
        out["prediction_std"] = float(np.std(y_pred, ddof=1)) if len(y_pred) > 1 else 0.0
    return out


def build_stage_b_samples(anchor_targets: pd.DataFrame, dataset_df: pd.DataFrame) -> Tuple[pd.DataFrame, StageBSampleDiagnostics]:
    eligible = anchor_targets[anchor_targets["anchor_valid"].astype(bool)].copy()
    eligible = eligible[np.isfinite(eligible["density_envelope_width_mm"].to_numpy(dtype=float))].copy()
    data_index = {(int(r.track_id), int(r.frame_index)): r for r in dataset_df.itertuples(index=False)}
    records: List[Dict[str, object]] = []
    discarded_incomplete = 0
    discarded_nonfinite = 0
    for row in eligible.itertuples(index=False):
        track_id = int(row.track_id)
        frame_index = int(row.frame_index)
        vals: List[float] = []
        complete = True
        for offset in WINDOW_OFFSETS:
            neighbor = data_index.get((track_id, frame_index + offset))
            if neighbor is None:
                complete = False
                break
            vals.extend([float(neighbor.peak_temp), float(neighbor.sqrt_mp_area), float(neighbor.mp_length)])
        if not complete:
            discarded_incomplete += 1
            continue
        if not np.isfinite(np.asarray(vals, dtype=float)).all():
            discarded_nonfinite += 1
            continue
        rec: Dict[str, object] = {
            "track_id": track_id,
            "frame_index": frame_index,
            "x_position_mm": float(row.x_position_mm),
            "actual_density_envelope_width_mm": float(row.density_envelope_width_mm),
        }
        for col, val in zip(WINDOW_FEATURE_COLUMNS, vals):
            rec[col] = float(val)
        records.append(rec)
    samples = pd.DataFrame(records).sort_values(["track_id", "frame_index"]).reset_index(drop=True)
    if set(int(t) for t in samples["track_id"].unique()) != TRACK_SET:
        raise RuntimeError("Stage B samples are not exactly Tracks 8, 10, and 14.")
    if PROHIBITED_MODEL_COLUMNS & set(WINDOW_FEATURE_COLUMNS):
        raise RuntimeError("Prohibited metadata columns entered Stage B model feature matrix.")
    diag = StageBSampleDiagnostics(
        eligible_targets=int(len(eligible)),
        samples_after_windowing=int(len(samples)),
        discarded_incomplete_window=discarded_incomplete,
        discarded_nonfinite_features=discarded_nonfinite,
        samples_by_track={int(k): int(v) for k, v in samples.groupby("track_id").size().to_dict().items()},
    )
    return samples, diag


def run_stage_b(samples: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    fold_rows: List[Dict[str, object]] = []
    train_baseline_rows: List[Dict[str, object]] = []
    oracle_baseline_rows: List[Dict[str, object]] = []
    pred_frames: List[pd.DataFrame] = []
    scaler_summary: Dict[str, object] = {}
    for fold in FOLDS:
        fold_name = str(fold["fold"])
        train_tracks = [int(t) for t in fold["train_tracks"]]
        val_track = int(fold["val_track"])
        train_df = samples[samples["track_id"].isin(train_tracks)].copy()
        val_df = samples[samples["track_id"] == val_track].copy()
        x_train = train_df[WINDOW_FEATURE_COLUMNS].to_numpy(dtype=float)
        y_train = train_df["actual_density_envelope_width_mm"].to_numpy(dtype=float)
        x_val = val_df[WINDOW_FEATURE_COLUMNS].to_numpy(dtype=float)
        y_val = val_df["actual_density_envelope_width_mm"].to_numpy(dtype=float)
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_val_scaled = scaler.transform(x_val)
        model = BayesianRidge()
        default_params = BayesianRidge().get_params(deep=True)
        if model.get_params(deep=True) != default_params:
            raise RuntimeError("BayesianRidge is not using sklearn defaults.")
        model.fit(x_train_scaled, y_train)
        y_pred = model.predict(x_val_scaled)
        if not np.isfinite(y_pred).all():
            raise RuntimeError(f"Fold {fold_name}: model predictions are nonfinite.")
        fold_rows.append({"fold": fold_name, "train_tracks": "+".join(str(t) for t in train_tracks), "val_track": val_track, **regression_metrics(y_val, y_pred)})
        train_mean_pred = np.full_like(y_val, float(np.mean(y_train)), dtype=float)
        oracle_mean_pred = np.full_like(y_val, float(np.mean(y_val)), dtype=float)
        train_baseline_rows.append(
            {
                "fold": fold_name,
                "baseline": "training_fold_mean_deployable",
                "train_tracks": "+".join(str(t) for t in train_tracks),
                "val_track": val_track,
                "prediction_value": float(np.mean(y_train)),
                **regression_metrics(y_val, train_mean_pred),
            }
        )
        oracle_baseline_rows.append(
            {
                "fold": fold_name,
                "baseline": "held_out_track_oracle_mean_diagnostic_non_deployable",
                "val_track": val_track,
                "prediction_value": float(np.mean(y_val)),
                "mae": float(mean_absolute_error(y_val, oracle_mean_pred)),
                "rmse": float(math.sqrt(mean_squared_error(y_val, oracle_mean_pred))),
                "n": int(len(y_val)),
                "target_mean": float(np.mean(y_val)),
                "target_std": float(np.std(y_val, ddof=1)) if len(y_val) > 1 else 0.0,
            }
        )
        pred = val_df[["track_id", "frame_index", "x_position_mm", "actual_density_envelope_width_mm"]].copy()
        pred.insert(0, "fold", fold_name)
        pred["predicted_density_envelope_width_mm"] = y_pred
        pred["residual"] = pred["actual_density_envelope_width_mm"] - pred["predicted_density_envelope_width_mm"]
        pred["actual_centered_by_track_mean"] = pred["actual_density_envelope_width_mm"] - float(np.mean(y_val))
        pred["predicted_centered_by_track_mean"] = pred["predicted_density_envelope_width_mm"] - float(np.mean(y_pred))
        pred_frames.append(pred)
        scaler_summary[fold_name] = {
            "train_tracks": train_tracks,
            "val_track": val_track,
            "n_train_samples": int(len(train_df)),
            "n_val_samples": int(len(val_df)),
            "scaler_fit_track_ids": sorted(int(t) for t in train_df["track_id"].unique()),
            "scaler_n_features_in": int(scaler.n_features_in_),
        }
    fold_metrics = pd.DataFrame(fold_rows)
    train_baselines = pd.DataFrame(train_baseline_rows)
    oracle_baselines = pd.DataFrame(oracle_baseline_rows)
    predictions = pd.concat(pred_frames, ignore_index=True)
    y_true = predictions["actual_density_envelope_width_mm"].to_numpy(dtype=float)
    y_pred = predictions["predicted_density_envelope_width_mm"].to_numpy(dtype=float)
    pooled = pd.DataFrame([{"scope": "pooled_oof_concatenated", **regression_metrics(y_true, y_pred)}])
    positive = fold_metrics[fold_metrics["r2"] > 0].copy()
    signal_gate = False
    if float(pooled.iloc[0]["r2"]) > 0 and len(positive) >= 1:
        oracle_lookup = oracle_baselines.set_index("fold")
        for row in positive.itertuples(index=False):
            oracle_mae = float(oracle_lookup.loc[row.fold, "mae"])
            if float(row.pearson_r) > 0.10 and float(row.mae) < oracle_mae:
                signal_gate = True
                break
    if signal_gate:
        final = "SIGNAL DETECTED"
    elif len(positive) >= 1:
        final = "WEAK SIGNAL"
    else:
        final = "NO SIGNAL"
    pooled["positive_r2_folds"] = int(len(positive))
    pooled["final_decision"] = final
    return fold_metrics, train_baselines, oracle_baselines, pooled, predictions, scaler_summary


def plot_stage_a(fig_dir: Path, native_all: pd.DataFrame, anchor_all: pd.DataFrame) -> List[str]:
    figures: List[str] = []
    exp29 = pd.read_csv(EXP29_ANCHOR_PATH)
    for track_id in TRACK_IDS:
        native = native_all[native_all["track_id"] == track_id]
        anchors = anchor_all[anchor_all["track_id"] == track_id]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(native["x_position_mm"], native["density_envelope_width_mm"], lw=0.7)
        ax.set_title(f"Track {track_id}: native density-envelope width vs x")
        ax.set_xlabel("x_position_mm")
        ax.set_ylabel("density_envelope_width_mm")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = fig_dir / f"track_{track_id}_native_density_width_vs_x.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        figures.append(str(path))

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(anchors["x_position_mm"], anchors["density_envelope_width_mm"], marker=".", ms=3, lw=0.8, label="Exp31 density envelope")
        e29 = exp29[exp29["track_id"] == track_id]
        ax.plot(e29["x_position_mm"], e29["local_core_width_mm"], marker=".", ms=2, lw=0.7, alpha=0.75, label="Exp29 local_core_width")
        ax.set_title(f"Track {track_id}: anchor widths vs x")
        ax.set_xlabel("x_position_mm")
        ax.set_ylabel("width (mm)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = fig_dir / f"track_{track_id}_anchor_density_width_vs_exp29.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        figures.append(str(path))

    fig, ax = plt.subplots(figsize=(8, 4))
    data = [anchor_all.loc[(anchor_all["track_id"] == t) & (anchor_all["anchor_valid"]), "density_envelope_width_mm"].dropna() for t in TRACK_IDS]
    ax.boxplot(data, labels=[f"T{t}" for t in TRACK_IDS])
    ax.set_ylabel("density_envelope_width_mm")
    ax.set_title("Density-envelope width distributions by track")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = fig_dir / "density_width_distribution_by_track.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    figures.append(str(path))
    return figures


def plot_density_profile_examples(fig_dir: Path, tracks: Dict[int, TrackData]) -> List[str]:
    figures: List[str] = []
    for track_id, track in tracks.items():
        finite_mask = np.isfinite(track.Z_mm).astype(float)
        density = ndimage.uniform_filter1d(finite_mask, size=KERNEL_WIDTH_PX, axis=0, mode="constant", cval=0.0)
        x_targets = [30.0, 50.0, 70.0, 90.0]
        fig, axes = plt.subplots(1, len(x_targets), figsize=(16, 4), sharey=True)
        for ax, xt in zip(axes, x_targets):
            j = int(np.nanargmin(np.abs(track.x_mm - xt)))
            rho = density[:, j]
            support = rho >= DENSITY_THRESHOLD_TAU
            ax.plot(rho, track.y_mm, label="rho")
            ax.axvline(DENSITY_THRESHOLD_TAU, color="black", linestyle="--", lw=1, label="tau=0.30")
            if support.any():
                runs = finite_runs(support)
                lengths = np.asarray([b - a for a, b in runs])
                a, b = runs[int(np.argsort(-lengths, kind="stable")[0])]
                ax.axhspan(track.y_mm[a], track.y_mm[b - 1], color="tab:orange", alpha=0.25, label="selected envelope")
            ax.set_title(f"x={track.x_mm[j]:.1f} mm")
            ax.set_xlabel("finite density")
            ax.grid(True, alpha=0.3)
        axes[0].set_ylabel("y_mm")
        fig.suptitle(f"Track {track_id}: representative density profiles")
        fig.tight_layout()
        path = fig_dir / f"track_{track_id}_representative_density_profiles.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        figures.append(str(path))
    return figures


def plot_stage_b(fig_dir: Path, predictions: pd.DataFrame) -> List[str]:
    figures: List[str] = []
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (fold, group) in zip(axes, predictions.groupby("fold", sort=False)):
        actual = group["actual_density_envelope_width_mm"].to_numpy(dtype=float)
        pred = group["predicted_density_envelope_width_mm"].to_numpy(dtype=float)
        ax.scatter(actual, pred, s=18, alpha=0.75)
        lo, hi = min(actual.min(), pred.min()), max(actual.max(), pred.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_title(fold)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = fig_dir / "stage_b_actual_vs_predicted_per_fold.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    figures.append(str(path))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, (fold, group) in zip(axes, predictions.groupby("fold", sort=False)):
        ax.axhline(0, color="k", linestyle="--", lw=1)
        ax.scatter(group["x_position_mm"], group["residual"], s=18, alpha=0.75)
        ax.set_title(fold)
        ax.set_xlabel("x_position_mm")
        ax.set_ylabel("residual")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = fig_dir / "stage_b_residual_vs_x_per_fold.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    figures.append(str(path))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, (fold, group) in zip(axes, predictions.groupby("fold", sort=False)):
        g = group.sort_values("x_position_mm")
        ax.plot(g["x_position_mm"], g["actual_density_envelope_width_mm"], lw=1.0, label="actual")
        ax.plot(g["x_position_mm"], g["predicted_density_envelope_width_mm"], lw=1.0, label="predicted")
        ax.set_title(fold)
        ax.set_xlabel("x_position_mm")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("density_envelope_width_mm")
    axes[-1].legend()
    fig.tight_layout()
    path = fig_dir / "stage_b_spatial_profile_per_fold.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    figures.append(str(path))

    fig, ax = plt.subplots(figsize=(5, 5))
    actual = predictions["actual_density_envelope_width_mm"].to_numpy(dtype=float)
    pred = predictions["predicted_density_envelope_width_mm"].to_numpy(dtype=float)
    ax.scatter(actual, pred, s=16, alpha=0.65)
    lo, hi = min(actual.min(), pred.min()), max(actual.max(), pred.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_title("Pooled OOF parity")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = fig_dir / "stage_b_pooled_oof_parity.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    figures.append(str(path))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(predictions["actual_centered_by_track_mean"], predictions["predicted_centered_by_track_mean"], s=16, alpha=0.65)
    ax.axhline(0, color="k", lw=1)
    ax.axvline(0, color="k", lw=1)
    ax.set_title("Diagnostic within-track centered covariance")
    ax.set_xlabel("Actual minus held-out-track mean")
    ax.set_ylabel("Prediction minus prediction track mean")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = fig_dir / "stage_b_within_track_centered_parity_diagnostic.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    figures.append(str(path))
    return figures


def write_metadata(
    metadata_dir: Path,
    run_tag: str,
    source_files: Dict[str, str],
    stage_a_decision: str,
    stage_a_reason: str,
    stage_b_decision: str | None,
    stage_b_diag: StageBSampleDiagnostics | None,
    scaler_summary: Dict[str, object] | None,
) -> None:
    metadata = {
        "experiment": EXPERIMENT_NAME,
        "timestamp": run_tag,
        "git_branch": git_text(["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_commit": git_text(["rev-parse", "HEAD"]),
        "python_version": sys.version,
        "platform": platform.platform(),
        "tracks_analyzed": list(TRACK_IDS),
        "sealed_tracks": sorted(SEALED_TRACKS),
        "track21_geometry_loaded": False,
        "track21_usage": "Track 21 Wyko geometry was not loaded or inspected; Track 21 was not used in Stage A or Stage B.",
        "source_data_files": source_files,
        "anchor_source": str(DATASET_PATH),
        "anchor_coordinate": "x_position_mm",
        "stage_a_target": {
            "name": "density_envelope_width_mm",
            "kernel_half_width_px": KERNEL_HALF_WIDTH_PX,
            "kernel_width_px": KERNEL_WIDTH_PX,
            "density_threshold_tau": DENSITY_THRESHOLD_TAU,
            "tau_description": "predeclared operational density threshold motivated by the observed finite-support regime; not a uniquely physically established constant",
            "boundary_mode": "zero-padding at y boundaries via scipy.ndimage.uniform_filter1d(mode='constant', cval=0.0)",
            "multiple_run_rule": "select longest contiguous rho >= tau run; stable first occurrence tie-break",
            "aggregation_half_window_mm": AGG_HALF_WINDOW_MM,
            "minimum_valid_native_columns_per_anchor": MIN_VALID_NATIVE_COLUMNS_PER_ANCHOR,
            "parameters_tuned": False,
            "uses_thermal_features": False,
        },
        "stage_a_decision": stage_a_decision,
        "stage_a_decision_reason": stage_a_reason,
        "stage_b_ran": stage_b_decision is not None,
        "stage_b_decision": stage_b_decision,
        "stage_b_sample_diagnostics": stage_b_diag.__dict__ if stage_b_diag else None,
        "stage_b_model": "sklearn.linear_model.BayesianRidge with sklearn default hyperparameters" if stage_b_decision else None,
        "stage_b_model_parameters": BayesianRidge().get_params(deep=True) if stage_b_decision else None,
        "stage_b_features": FEATURE_BASE_COLUMNS if stage_b_decision else None,
        "stage_b_window_definition": {
            "type": "centered five-frame window",
            "offsets": list(WINDOW_OFFSETS),
            "flattened_columns": WINDOW_FEATURE_COLUMNS,
            "same_track_required": True,
        } if stage_b_decision else None,
        "stage_b_scaler": "StandardScaler fit independently on each training fold only" if stage_b_decision else None,
        "stage_b_scaler_summary": scaler_summary,
        "leakage_safeguards": {
            "track21_geometry_never_loaded": True,
            "track21_never_used_stage_a": True,
            "track21_never_used_stage_b": True,
            "target_parameters_fixed_before_modeling": True,
            "h_and_tau_never_tuned": True,
            "target_extraction_never_uses_thermal_features": True,
            "track_id_never_model_feature": True,
            "x_position_mm_never_model_feature": True,
            "sem_features_never_model_features": True,
            "centered_windows_never_cross_track_boundaries": True,
            "validation_targets_never_used_for_fitting": True,
            "oracle_held_out_track_mean_baseline_is_diagnostic_only": True,
            "experiment30_no_signal_classification_unchanged": True,
        },
        "experiment30_locked_comparison": EXP30_LOCKED_RESULTS,
    }
    (metadata_dir / "run_metadata.json").write_text(json.dumps(make_json_safe(metadata), indent=2), encoding="utf-8")


def main() -> int:
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_ROOT / f"{EXPERIMENT_NAME}_{run_tag}"
    table_dir = out_dir / "tables"
    fig_dir = out_dir / "figures"
    metadata_dir = out_dir / "metadata"
    for d in (table_dir, fig_dir, metadata_dir):
        d.mkdir(parents=True, exist_ok=False)

    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Output directory: {out_dir}")
    print(f"Locked Stage A parameters: h={KERNEL_HALF_WIDTH_PX}, tau={DENSITY_THRESHOLD_TAU}, aggregation=±{AGG_HALF_WINDOW_MM} mm, min columns={MIN_VALID_NATIVE_COLUMNS_PER_ANCHOR}")
    print("Track 21 geometry sealed: not loaded, not inspected, not used.")

    dataset_df = read_development_dataset_prefix(DATASET_PATH)
    anchors = read_anchors(dataset_df)
    tracks: Dict[int, TrackData] = {}
    source_files: Dict[str, str] = {}
    native_tables: List[pd.DataFrame] = []
    anchor_tables: List[pd.DataFrame] = []
    for track_id in TRACK_IDS:
        track = load_track(track_id)
        tracks[track_id] = track
        source_files[str(track_id)] = track.source_file
        native = extract_density_width(track)
        anchor_df = aggregate_to_anchors(native, anchors, track.x_mm)
        if len(anchor_df) != EXPECTED_ROWS_PER_TRACK:
            raise RuntimeError(f"Track {track_id}: expected 400 anchor rows; observed {len(anchor_df)}.")
        native_tables.append(native)
        anchor_tables.append(anchor_df)
        print(f"Stage A extracted Track {track_id}: native_rows={len(native)}, anchor_rows={len(anchor_df)}, source={track.source_file}")

    native_all = pd.concat(native_tables, ignore_index=True)
    anchor_all = pd.concat(anchor_tables, ignore_index=True)
    stage_a_summary, stage_a_criteria, cross_diag = build_stage_a_summaries(native_all, anchor_all)
    stage_a_decision, stage_a_reason = decide_stage_a(stage_a_criteria)

    native_all.to_csv(table_dir / "native_density_width.csv", index=False)
    anchor_all.to_csv(table_dir / "anchor_aggregated_density_width.csv", index=False)
    stage_a_summary.to_csv(table_dir / "stage_a_track_summary.csv", index=False)
    stage_a_criteria.to_csv(table_dir / "stage_a_criteria.csv", index=False)
    cross_diag.to_csv(table_dir / "stage_a_cross_track_diagnostics.csv", index=False)
    figures = plot_stage_a(fig_dir, native_all, anchor_all)
    figures.extend(plot_density_profile_examples(fig_dir, tracks))

    print("\nStage A summary:")
    print(stage_a_summary.to_string(index=False))
    print("\nStage A criteria:")
    print(stage_a_criteria.to_string(index=False))
    print("\nCross-track diagnostic:")
    print(cross_diag.to_string(index=False))
    print(f"\nStage A decision: {stage_a_decision} — {stage_a_reason}")

    stage_b_decision: str | None = None
    stage_b_diag: StageBSampleDiagnostics | None = None
    scaler_summary: Dict[str, object] | None = None
    if stage_a_decision in {"PASS", "CONDITIONAL PASS"}:
        samples, stage_b_diag = build_stage_b_samples(anchor_all, dataset_df)
        fold_metrics, train_baselines, oracle_baselines, pooled_metrics, predictions, scaler_summary = run_stage_b(samples)
        stage_b_decision = str(pooled_metrics.iloc[0]["final_decision"])
        samples.to_csv(table_dir / "stage_b_model_samples.csv", index=False)
        fold_metrics.to_csv(table_dir / "stage_b_loto_fold_model_metrics.csv", index=False)
        train_baselines.to_csv(table_dir / "stage_b_training_mean_baseline_metrics.csv", index=False)
        oracle_baselines.to_csv(table_dir / "stage_b_oracle_mean_baseline_metrics.csv", index=False)
        pooled_metrics.to_csv(table_dir / "stage_b_loto_pooled_metrics.csv", index=False)
        predictions.to_csv(table_dir / "stage_b_per_fold_predictions.csv", index=False)
        figures.extend(plot_stage_b(fig_dir, predictions))
        print("\nStage B sample diagnostics:")
        print(json.dumps(make_json_safe(stage_b_diag.__dict__), indent=2))
        print("\nStage B model metrics:")
        print(fold_metrics.to_string(index=False))
        print("\nStage B training-mean baseline metrics:")
        print(train_baselines.to_string(index=False))
        print("\nStage B oracle held-out-track mean baseline metrics (diagnostic/non-deployable):")
        print(oracle_baselines.to_string(index=False))
        print("\nStage B pooled OOF metrics:")
        print(pooled_metrics.to_string(index=False))
        print(f"\nStage B final classification: {stage_b_decision}")
    else:
        print("\nStage A failed; Stage B was not run.")

    (metadata_dir / "figures.json").write_text(json.dumps([str(Path(p).relative_to(out_dir)) for p in figures], indent=2), encoding="utf-8")
    write_metadata(metadata_dir, run_tag, source_files, stage_a_decision, stage_a_reason, stage_b_decision, stage_b_diag, scaler_summary)
    print(f"\nWrote outputs to: {out_dir}")
    print("Track 21 geometry loaded: no")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
