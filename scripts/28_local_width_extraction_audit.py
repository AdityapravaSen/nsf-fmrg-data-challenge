"""Experiment 28: raw local width extraction from Wyko finite-support topology.

This standalone audit tests Candidate 1 from the local-width hypothesis:
for every native Wyko x-column, define the local width as the y-extent of
the longest contiguous finite (non-NaN) run. The primary extraction is raw:
no minimum run threshold, no NaN-gap bridging, no smoothing, no height
thresholding, and no detrending-derived boundary selection.

Guardrails:
- Track 21 is sealed and is not loaded or inspected.
- Only Tracks 8, 10, and 14 are analyzed.
- No ML model is trained.
- Existing pipelines and prior artifacts are not modified.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import json
import math
import platform
import subprocess
import sys
import warnings
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
	sys.path.append(str(SRC_DIR))

from nsf_fmrg_data import load_wyko_asc  # organizer loader, unchanged


warnings.filterwarnings("ignore", category=RuntimeWarning)

SEALED_TRACKS = {21}
TRACK_IDS = [8, 10, 14]
assert not (set(TRACK_IDS) & SEALED_TRACKS), "Track 21 is sealed and must not be analyzed."

REPRESENTATIVE_X_MM = [30.0, 50.0, 70.0, 90.0]
MIN_RUN_THRESHOLDS_PX = [20, 50, 100, 150]
JUMP_THRESHOLDS_MM = [0.05, 0.10, 0.20, 0.50]
AMBIGUOUS_RATIO_THRESHOLD = 1.50
MAX_PLOT_X_PIXELS = 1800
CENTRAL_EXCLUSION_Y_MM = (0.65, 1.35)


@dataclass(frozen=True)
class TrackArrays:
	track_id: int
	source_file: str
	Z_mm: np.ndarray
	Z_det_mm: np.ndarray
	x_mm: np.ndarray
	y_mm: np.ndarray
	detrend_meta: Dict[str, object]


def finite_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
	"""Return half-open y-index intervals for contiguous finite support.

	Physically, each interval is one connected set of Wyko-valid measurements
	along a single x cross-section. No gaps are bridged.
	"""

	idx = np.flatnonzero(np.asarray(mask, dtype=bool))
	if idx.size == 0:
		return []
	breaks = np.where(np.diff(idx) > 1)[0]
	starts = np.r_[idx[0], idx[breaks + 1]]
	stops = np.r_[idx[breaks] + 1, idx[-1] + 1]
	return [(int(a), int(b)) for a, b in zip(starts, stops)]


def robust_plane_fit_substrate_focused(
	Z_mm: np.ndarray,
	x_mm: np.ndarray,
	y_mm: np.ndarray,
	y_excl: Tuple[float, float] = CENTRAL_EXCLUSION_Y_MM,
	stride_x: int = 40,
	stride_y: int = 2,
	max_iter: int = 3,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, object]]:
	"""Detrend finite height values while preserving NaN topology exactly.

	The detrended field is used only for diagnostic visualization and
	morphology comparison. Candidate 1 boundaries are always selected from the
	raw finite mask, not from detrended values.
	"""

	y0, y1 = y_excl
	xs = x_mm[::stride_x]
	ys = y_mm[::stride_y]
	Zs = Z_mm[::stride_y, ::stride_x]
	Xs, Ys = np.meshgrid(xs, ys)
	z = Zs.ravel()
	A = np.c_[Xs.ravel(), Ys.ravel(), np.ones(Xs.size)]
	valid = np.isfinite(z)
	outside = (Ys.ravel() < y0) | (Ys.ravel() > y1)
	keep = valid & outside

	meta: Dict[str, object] = {
		"method": "substrate-focused robust plane fit, diagnostics only",
		"stride_x": int(stride_x),
		"stride_y": int(stride_y),
		"excluded_y_min_mm": float(y0),
		"excluded_y_max_mm": float(y1),
		"fit_samples_initial": int(np.sum(keep)),
	}
	if np.sum(keep) < 100:
		meta["fit_failed"] = True
		return Z_mm.copy(), None, meta

	coef = None
	for _ in range(max_iter):
		coef, *_ = np.linalg.lstsq(A[keep], z[keep], rcond=None)
		resid = z - A @ coef
		rv = resid[keep]
		lo, hi = np.nanpercentile(rv, [5, 95])
		keep_new = keep & (resid >= lo) & (resid <= hi)
		if np.sum(keep_new) < 100:
			break
		keep = keep_new

	assert coef is not None
	plane = coef[0] * x_mm[None, :] + coef[1] * y_mm[:, None] + coef[2]
	Z_det = Z_mm - plane
	meta.update(
		{
			"fit_failed": False,
			"fit_samples_final": int(np.sum(keep)),
			"coef_x": float(coef[0]),
			"coef_y": float(coef[1]),
			"coef_c": float(coef[2]),
		}
	)
	return Z_det, np.asarray(coef, dtype=float), meta


def safe_stat(values: Sequence[float], func, default: float = float("nan")) -> float:
	arr = np.asarray(values, dtype=float)
	arr = arr[np.isfinite(arr)]
	if arr.size == 0:
		return default
	return float(func(arr))


def describe_distribution(values: Sequence[float], prefix: str) -> Dict[str, float]:
	"""Return audit-friendly descriptive statistics for finite values."""

	arr = np.asarray(values, dtype=float)
	arr = arr[np.isfinite(arr)]
	if arr.size == 0:
		return {f"{prefix}_{k}": float("nan") for k in ["n", "mean", "std", "median", "iqr", "min", "max", "p05", "p95"]}
	q25, q75 = np.nanpercentile(arr, [25, 75])
	return {
		f"{prefix}_n": int(arr.size),
		f"{prefix}_mean": float(np.nanmean(arr)),
		f"{prefix}_std": float(np.nanstd(arr, ddof=1)) if arr.size > 1 else 0.0,
		f"{prefix}_median": float(np.nanmedian(arr)),
		f"{prefix}_iqr": float(q75 - q25),
		f"{prefix}_min": float(np.nanmin(arr)),
		f"{prefix}_max": float(np.nanmax(arr)),
		f"{prefix}_p05": float(np.nanpercentile(arr, 5)),
		f"{prefix}_p95": float(np.nanpercentile(arr, 95)),
	}


def describe_jumps(values: Sequence[float], prefix: str) -> Dict[str, float]:
	arr = np.asarray(values, dtype=float)
	good = np.isfinite(arr[:-1]) & np.isfinite(arr[1:])
	jumps = np.abs(np.diff(arr)[good])
	out = {f"{prefix}_n_adjacent_pairs": int(jumps.size)}
	if jumps.size == 0:
		for k in ["median", "p90", "p95", "p99", "max"]:
			out[f"{prefix}_{k}"] = float("nan")
		for th in JUMP_THRESHOLDS_MM:
			out[f"{prefix}_fraction_gt_{th:g}_mm"] = float("nan")
		return out
	out.update(
		{
			f"{prefix}_median": float(np.nanmedian(jumps)),
			f"{prefix}_p90": float(np.nanpercentile(jumps, 90)),
			f"{prefix}_p95": float(np.nanpercentile(jumps, 95)),
			f"{prefix}_p99": float(np.nanpercentile(jumps, 99)),
			f"{prefix}_max": float(np.nanmax(jumps)),
		}
	)
	for th in JUMP_THRESHOLDS_MM:
		out[f"{prefix}_fraction_gt_{th:g}_mm"] = float(np.mean(jumps > th))
	return out


def extract_longest_runs(track_id: int, Z_mm: np.ndarray, x_mm: np.ndarray, y_mm: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
	"""Extract raw Candidate 1 at every native x-position.

	The selected run is the longest contiguous finite y-run in the raw Wyko
	column. Ties are resolved deterministically by the first occurrence in y.
	"""

	rows: List[Dict[str, object]] = []
	run_rows: List[Dict[str, object]] = []
	n_y = int(Z_mm.shape[0])
	for j, x in enumerate(x_mm):
		finite = np.isfinite(Z_mm[:, j])
		runs = finite_runs(finite)
		lengths = np.array([b - a for a, b in runs], dtype=int)
		total_finite = int(np.sum(finite))
		if lengths.size:
			order = np.argsort(-lengths, kind="stable")
			selected_idx = int(order[0])
			second_len = int(lengths[order[1]]) if len(order) > 1 else 0
			a, b = runs[selected_idx]
			left = float(y_mm[a])
			right = float(y_mm[b - 1])
			width = float(right - left)
			longest_len = int(b - a)
			selected_run_rank = 1
		else:
			selected_idx = -1
			second_len = 0
			left = right = width = float("nan")
			longest_len = 0
			selected_run_rank = 0

		for rank, idx_run in enumerate(np.argsort(-lengths, kind="stable"), start=1):
			a, b = runs[int(idx_run)]
			run_rows.append(
				{
					"track_id": int(track_id),
					"x_index": int(j),
					"x_position_mm": float(x),
					"run_rank_by_length": int(rank),
					"is_selected_longest_run": bool(int(idx_run) == selected_idx),
					"run_start_y_index": int(a),
					"run_stop_y_index_exclusive": int(b),
					"run_left_boundary_mm": float(y_mm[a]),
					"run_right_boundary_mm": float(y_mm[b - 1]),
					"run_width_mm": float(y_mm[b - 1] - y_mm[a]),
					"run_length_px": int(b - a),
				}
			)

		rows.append(
			{
				"track_id": int(track_id),
				"x_index": int(j),
				"x_position_mm": float(x),
				"left_boundary_mm": left,
				"right_boundary_mm": right,
				"local_width_mm": width,
				"run_length_px": int(longest_len),
				"total_finite_px": int(total_finite),
				"finite_fraction": float(total_finite / n_y),
				"number_of_finite_runs": int(len(runs)),
				"second_longest_run_px": int(second_len),
				"longest_over_total_finite": float(longest_len / total_finite) if total_finite > 0 else float("nan"),
				"longest_over_second_longest": float(longest_len / second_len) if second_len > 0 else float("inf") if longest_len > 0 else float("nan"),
				"selected_run_rank": int(selected_run_rank),
				"zero_finite_column": bool(total_finite == 0),
				"ambiguous_comparable_runs": bool(second_len > 0 and longest_len / second_len <= AMBIGUOUS_RATIO_THRESHOLD),
			}
		)
	return pd.DataFrame(rows), pd.DataFrame(run_rows)


def load_track(track_id: int, height_dir: Path) -> TrackArrays:
	if track_id in SEALED_TRACKS:
		raise ValueError("Track 21 is sealed and must not be loaded.")
	hm = load_wyko_asc(height_dir, track_id, crop_to_common=True)
	Z_mm = np.asarray(hm["Z_mm"], dtype=float)
	x_mm = np.asarray(hm["x_actual_mm"], dtype=float)
	y_mm = np.asarray(hm["y_mm"], dtype=float)
	Z_det, _coef, dmeta = robust_plane_fit_substrate_focused(Z_mm, x_mm, y_mm)
	raw_mask = np.isfinite(Z_mm)
	det_mask = np.isfinite(Z_det)
	diff = raw_mask != det_mask
	dmeta.update(
		{
			"nan_mask_invariance_diff_pixels": int(np.sum(diff)),
			"nan_mask_invariance_diff_fraction": float(np.mean(diff)),
			"candidate1_boundaries_use_detrended_data": False,
		}
	)
	return TrackArrays(track_id, str(hm["file"]), Z_mm, Z_det, x_mm, y_mm, dmeta)


def track_load_summary(track: TrackArrays) -> Dict[str, object]:
	dx = np.diff(track.x_mm)
	dy = np.diff(track.y_mm)
	finite_fraction = float(np.mean(np.isfinite(track.Z_mm)))
	return {
		"track_id": int(track.track_id),
		"source_file": track.source_file,
		"shape_y": int(track.Z_mm.shape[0]),
		"shape_x": int(track.Z_mm.shape[1]),
		"x_min_mm": float(np.nanmin(track.x_mm)),
		"x_max_mm": float(np.nanmax(track.x_mm)),
		"y_min_mm": float(np.nanmin(track.y_mm)),
		"y_max_mm": float(np.nanmax(track.y_mm)),
		"x_spacing_median_mm": float(np.nanmedian(dx)),
		"x_spacing_min_mm": float(np.nanmin(dx)),
		"x_spacing_max_mm": float(np.nanmax(dx)),
		"y_spacing_median_mm": float(np.nanmedian(dy)),
		"y_spacing_min_mm": float(np.nanmin(dy)),
		"y_spacing_max_mm": float(np.nanmax(dy)),
		"native_x_resolution_used": True,
		"finite_fraction": finite_fraction,
		"nan_fraction": float(1.0 - finite_fraction),
		"nan_mask_invariance_diff_pixels": int(track.detrend_meta["nan_mask_invariance_diff_pixels"]),
		"nan_mask_invariance_diff_fraction": float(track.detrend_meta["nan_mask_invariance_diff_fraction"]),
	}


def build_track_summary(native: pd.DataFrame, load_row: Dict[str, object]) -> Dict[str, object]:
	out = dict(load_row)
	for col in [
		"local_width_mm",
		"run_length_px",
		"total_finite_px",
		"finite_fraction",
		"number_of_finite_runs",
		"second_longest_run_px",
		"longest_over_total_finite",
		"longest_over_second_longest",
	]:
		vals = native[col].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
		out.update(describe_distribution(vals, col))
	out["zero_finite_column_fraction"] = float(native["zero_finite_column"].mean())
	out["ambiguous_comparable_runs_fraction"] = float(native["ambiguous_comparable_runs"].mean())
	out["primary_extraction_min_run_threshold_px"] = 0
	out["primary_extraction_bridges_nan_gaps"] = False
	out["primary_extraction_smooths_width_or_boundaries"] = False
	out["primary_extraction_uses_height_threshold"] = False
	return out


def topology_summary(native: pd.DataFrame) -> Dict[str, object]:
	row: Dict[str, object] = {"track_id": int(native["track_id"].iloc[0])}
	for col in ["number_of_finite_runs", "run_length_px", "second_longest_run_px", "longest_over_total_finite", "longest_over_second_longest"]:
		row.update(describe_distribution(native[col].replace([np.inf, -np.inf], np.nan), col))
	row["comparable_runs_ratio_threshold"] = float(AMBIGUOUS_RATIO_THRESHOLD)
	row["comparable_runs_fraction"] = float(native["ambiguous_comparable_runs"].mean())
	row["columns_with_second_run_fraction"] = float((native["second_longest_run_px"] > 0).mean())
	return row


def adjacent_jump_summary(native: pd.DataFrame) -> Dict[str, object]:
	row: Dict[str, object] = {"track_id": int(native["track_id"].iloc[0])}
	row.update(describe_jumps(native["local_width_mm"], "width_jump_mm"))
	row.update(describe_jumps(native["left_boundary_mm"], "left_boundary_jump_mm"))
	row.update(describe_jumps(native["right_boundary_mm"], "right_boundary_jump_mm"))
	return row


def minimum_run_sensitivity(native: pd.DataFrame) -> pd.DataFrame:
	rows: List[Dict[str, object]] = []
	track_id = int(native["track_id"].iloc[0])
	for th in MIN_RUN_THRESHOLDS_PX:
		retained = native["run_length_px"].to_numpy() >= th
		rejected_x = native.loc[~retained, "x_position_mm"].to_numpy(dtype=float)
		row: Dict[str, object] = {
			"track_id": track_id,
			"hypothetical_min_run_px": int(th),
			"retained_column_fraction": float(np.mean(retained)),
			"rejected_column_fraction": float(np.mean(~retained)),
			"rejected_x_min_mm": safe_stat(rejected_x, np.nanmin),
			"rejected_x_max_mm": safe_stat(rejected_x, np.nanmax),
			"rejected_x_median_mm": safe_stat(rejected_x, np.nanmedian),
			"primary_target_changed_by_this_filter": False,
		}
		row.update(describe_distribution(native.loc[retained, "local_width_mm"], "retained_width_mm"))
		row.update(describe_distribution(native.loc[~retained, "local_width_mm"], "rejected_width_mm"))
		rows.append(row)
	return pd.DataFrame(rows)


def nearest_x_index(x_mm: np.ndarray, x0: float) -> int:
	return int(np.nanargmin(np.abs(np.asarray(x_mm, dtype=float) - float(x0))))


def choose_representative_cases(native: pd.DataFrame, x_mm: np.ndarray) -> pd.DataFrame:
	cases: List[Dict[str, object]] = []

	def add_case(label: str, idx: int) -> None:
		if idx < 0 or idx >= len(native):
			return
		cases.append({"case_label": label, "x_index": int(idx), "requested_x_mm": float(native.iloc[idx]["x_position_mm"])})

	for x0 in REPRESENTATIVE_X_MM:
		add_case(f"nearest_requested_x_{x0:g}_mm", nearest_x_index(x_mm, x0))

	valid_width = native["local_width_mm"].to_numpy(dtype=float)
	if np.any(np.isfinite(valid_width)):
		add_case("largest_width", int(np.nanargmax(valid_width)))
		positive = np.where(np.isfinite(valid_width) & (valid_width > 0))[0]
		if positive.size:
			add_case("smallest_nonzero_width", int(positive[np.nanargmin(valid_width[positive])]))

	width = native["local_width_mm"].to_numpy(dtype=float)
	good = np.isfinite(width[:-1]) & np.isfinite(width[1:])
	if np.any(good):
		jumps = np.full(len(width) - 1, np.nan)
		jumps[good] = np.abs(np.diff(width)[good])
		add_case("largest_adjacent_width_jump_left_column", int(np.nanargmax(jumps)))
		add_case("largest_adjacent_width_jump_right_column", int(np.nanargmax(jumps) + 1))

	dom = native["longest_over_total_finite"].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
	if np.any(np.isfinite(dom)):
		add_case("lowest_longest_over_total_finite", int(np.nanargmin(dom)))

	ratio = native["longest_over_second_longest"].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
	if np.any(np.isfinite(ratio)):
		add_case("most_comparable_first_second_runs", int(np.nanargmin(ratio)))

	case_df = pd.DataFrame(cases).drop_duplicates(subset=["x_index"]).reset_index(drop=True)
	case_df["track_id"] = int(native["track_id"].iloc[0])
	merged = case_df.merge(native, on=["track_id", "x_index"], how="left", suffixes=("", "_native"))
	return merged


def classify_morphology(track: TrackArrays, native: pd.DataFrame, cases: pd.DataFrame) -> pd.DataFrame:
	"""Compare finite-support boundaries with independent detrended gradients.

	This diagnostic does not move boundaries. It measures local |dz/dy| near the
	selected run edges and assigns an interpretive category for representative
	cases.
	"""

	rows: List[Dict[str, object]] = []
	y = track.y_mm
	dy = float(np.nanmedian(np.diff(y)))
	for _, case in cases.iterrows():
		j = int(case["x_index"])
		z = track.Z_det_mm[:, j]
		finite = np.isfinite(z)
		grad = np.full_like(z, np.nan, dtype=float)
		for a, b in finite_runs(finite):
			if b - a >= 3:
				grad[a:b] = np.abs(np.gradient(z[a:b], y[a:b])) * 1e3
		finite_grad = grad[np.isfinite(grad)]
		left = float(case["left_boundary_mm"])
		right = float(case["right_boundary_mm"])
		a = int(np.nanargmin(np.abs(y - left))) if np.isfinite(left) else -1
		b = int(np.nanargmin(np.abs(y - right))) if np.isfinite(right) else -1
		edge_window_px = max(2, int(round(0.02 / dy)))
		edge_vals = []
		for idx in [a, b]:
			if idx >= 0:
				lo = max(0, idx - edge_window_px)
				hi = min(len(y), idx + edge_window_px + 1)
				edge_vals.extend(list(grad[lo:hi][np.isfinite(grad[lo:hi])]))
		edge_grad = float(np.nanmax(edge_vals)) if edge_vals else float("nan")
		if finite_grad.size and np.isfinite(edge_grad):
			pct = float(100.0 * np.mean(finite_grad <= edge_grad))
		else:
			pct = float("nan")
		has_nan_outside = bool((a > 0 and not finite[a - 1]) or (b + 1 < len(finite) and not finite[b + 1])) if a >= 0 and b >= 0 else False
		if not np.isfinite(edge_grad):
			category = "inconclusive_no_finite_gradient"
		elif pct >= 90.0:
			category = "coincides_with_strong_height_gradient"
		elif has_nan_outside:
			category = "abrupt_missing_support_transition_with_weak_height_gradient"
		else:
			category = "boundary_lies_in_smooth_or_low_gradient_region"
		rows.append(
			{
				"track_id": int(track.track_id),
				"case_label": case["case_label"],
				"x_index": int(j),
				"x_position_mm": float(case["x_position_mm"]),
				"boundary_edge_gradient_max_um_per_mm": edge_grad,
				"boundary_edge_gradient_percentile_within_profile": pct,
				"finite_support_abrupt_transition_adjacent_to_boundary": has_nan_outside,
				"morphology_category": category,
				"diagnostic_only_boundaries_not_adjusted": True,
			}
		)
	return pd.DataFrame(rows)


def compare_macro_width(native_all: pd.DataFrame) -> pd.DataFrame:
	path = REPO_ROOT / "processed_data" / "final_multimodal_dataset.csv"
	if not path.exists():
		return pd.DataFrame()
	cols = pd.read_csv(path, nrows=0).columns.tolist()
	if "smoothed_macro_width_mm" not in cols or "track_id" not in cols:
		return pd.DataFrame()
	x_col = "heightmap_x_mm" if "heightmap_x_mm" in cols else "x_position_mm"
	records: List[Dict[str, float]] = []
	with path.open("r", newline="", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		for raw in reader:
			try:
				track_id = int(float(raw["track_id"]))
			except Exception:
				continue
			if track_id not in TRACK_IDS:
				continue
			records.append(
				{
					"track_id": float(track_id),
					x_col: float(raw[x_col]) if raw.get(x_col, "") != "" else float("nan"),
					"smoothed_macro_width_mm": float(raw["smoothed_macro_width_mm"])
					if raw.get("smoothed_macro_width_mm", "") != ""
					else float("nan"),
				}
			)
	df = pd.DataFrame(records)
	if df.empty:
		return pd.DataFrame()
	rows: List[Dict[str, object]] = []
	for track_id, group in df.groupby("track_id"):
		native = native_all[native_all["track_id"] == int(track_id)].sort_values("x_position_mm")
		x_native = native["x_position_mm"].to_numpy(dtype=float)
		w_native = native["local_width_mm"].to_numpy(dtype=float)
		matched = []
		macro = []
		dx = []
		for _, r in group.iterrows():
			if not np.isfinite(r[x_col]) or not np.isfinite(r["smoothed_macro_width_mm"]):
				continue
			j = nearest_x_index(x_native, float(r[x_col]))
			matched.append(float(w_native[j]))
			macro.append(float(r["smoothed_macro_width_mm"]))
			dx.append(float(x_native[j] - r[x_col]))
		matched_arr = np.asarray(matched, dtype=float)
		macro_arr = np.asarray(macro, dtype=float)
		good = np.isfinite(matched_arr) & np.isfinite(macro_arr)
		diff = matched_arr[good] - macro_arr[good]
		if np.sum(good) >= 2 and np.nanstd(matched_arr[good]) > 0 and np.nanstd(macro_arr[good]) > 0:
			corr = float(np.corrcoef(matched_arr[good], macro_arr[good])[0, 1])
		else:
			corr = float("nan")
		row: Dict[str, object] = {
			"track_id": int(track_id),
			"macro_width_source": str(path),
			"coordinate_column_used": x_col,
			"n_matched": int(np.sum(good)),
			"nearest_match_abs_dx_median_mm": safe_stat(np.abs(dx), np.nanmedian),
			"nearest_match_abs_dx_max_mm": safe_stat(np.abs(dx), np.nanmax),
			"correlation_native_candidate1_vs_smoothed_macro": corr,
			"mae_abs_difference_mm": safe_stat(np.abs(diff), np.nanmean),
			"abs_difference_median_mm": safe_stat(np.abs(diff), np.nanmedian),
			"abs_difference_p95_mm": safe_stat(np.abs(diff), lambda a: np.nanpercentile(a, 95)),
			"candidate1_mean_mm_at_macro_grid": safe_stat(matched_arr[good], np.nanmean),
			"candidate1_std_mm_at_macro_grid": safe_stat(matched_arr[good], np.nanstd),
			"smoothed_macro_mean_mm": safe_stat(macro_arr[good], np.nanmean),
			"smoothed_macro_std_mm": safe_stat(macro_arr[good], np.nanstd),
			"used_only_as_independent_reference": True,
		}
		rows.append(row)
	return pd.DataFrame(rows)


def downsample_step(n_x: int) -> int:
	return max(1, int(math.ceil(n_x / MAX_PLOT_X_PIXELS)))


def plot_track_diagnostics(fig_dir: Path, track: TrackArrays, native: pd.DataFrame, cases: pd.DataFrame) -> List[str]:
	made: List[str] = []
	x = track.x_mm
	y = track.y_mm

	fig, ax = plt.subplots(figsize=(13, 3.2))
	ax.plot(native["x_position_mm"], native["local_width_mm"], lw=0.7)
	ax.set_title(f"Track {track.track_id}: raw Candidate 1 local width at native Wyko x-resolution")
	ax.set_xlabel("x (mm)")
	ax.set_ylabel("local_width_mm")
	fig.tight_layout()
	p = fig_dir / f"track_{track.track_id}_local_width_vs_x.png"
	fig.savefig(p, dpi=180)
	plt.close(fig)
	made.append(str(p))

	fig, ax = plt.subplots(figsize=(13, 3.2))
	ax.plot(native["x_position_mm"], native["left_boundary_mm"], lw=0.7, label="left boundary")
	ax.plot(native["x_position_mm"], native["right_boundary_mm"], lw=0.7, label="right boundary")
	ax.set_title(f"Track {track.track_id}: selected longest finite-run boundaries")
	ax.set_xlabel("x (mm)")
	ax.set_ylabel("y boundary (mm)")
	ax.legend()
	fig.tight_layout()
	p = fig_dir / f"track_{track.track_id}_boundaries_vs_x.png"
	fig.savefig(p, dpi=180)
	plt.close(fig)
	made.append(str(p))

	fig, axes = plt.subplots(4, 1, figsize=(13, 8), sharex=True)
	axes[0].plot(native["x_position_mm"], native["run_length_px"], lw=0.6)
	axes[0].set_ylabel("longest run px")
	axes[1].plot(native["x_position_mm"], native["number_of_finite_runs"], lw=0.6)
	axes[1].set_ylabel("# finite runs")
	axes[2].plot(native["x_position_mm"], native["longest_over_total_finite"], lw=0.6)
	axes[2].set_ylabel("longest/total")
	ratio = native["longest_over_second_longest"].replace([np.inf, -np.inf], np.nan)
	axes[3].plot(native["x_position_mm"], ratio, lw=0.6)
	axes[3].set_ylabel("longest/second")
	axes[3].set_xlabel("x (mm)")
	fig.suptitle(f"Track {track.track_id}: finite-run topology diagnostics")
	fig.tight_layout()
	p = fig_dir / f"track_{track.track_id}_finite_run_topology_support_diagnostics.png"
	fig.savefig(p, dpi=180)
	plt.close(fig)
	made.append(str(p))

	step = downsample_step(len(x))
	finite_ds = np.isfinite(track.Z_mm[:, ::step]).astype(float)
	x_ds = x[::step]
	fig, ax = plt.subplots(figsize=(14, 4.3))
	ax.imshow(
		finite_ds,
		origin="lower",
		aspect="auto",
		extent=[float(x_ds[0]), float(x_ds[-1]), float(y[0]), float(y[-1])],
		interpolation="nearest",
		cmap="gray_r",
		vmin=0,
		vmax=1,
	)
	ax.plot(native["x_position_mm"], native["left_boundary_mm"], color="tab:blue", lw=0.8, label="left boundary")
	ax.plot(native["x_position_mm"], native["right_boundary_mm"], color="tab:red", lw=0.8, label="right boundary")
	ax.set_title(f"Track {track.track_id}: 2D finite/NaN topology with raw longest-run boundary overlay")
	ax.set_xlabel("x (mm)")
	ax.set_ylabel("y (mm)")
	ax.legend(loc="upper right")
	fig.tight_layout()
	p = fig_dir / f"track_{track.track_id}_2d_nan_topology_boundary_overlay.png"
	fig.savefig(p, dpi=200)
	plt.close(fig)
	made.append(str(p))

	n_cases = len(cases)
	ncols = 2
	nrows = int(math.ceil(n_cases / ncols))
	fig, axes = plt.subplots(nrows, ncols, figsize=(13, max(3.2, 2.8 * nrows)), squeeze=False)
	z_lim = np.nanpercentile(np.abs(track.Z_det_mm[np.isfinite(track.Z_det_mm)]) * 1e3, 99) if np.any(np.isfinite(track.Z_det_mm)) else 20.0
	z_lim = float(min(max(z_lim, 5.0), 100.0))
	for ax, (_, case) in zip(axes.ravel(), cases.iterrows()):
		j = int(case["x_index"])
		z_um = track.Z_det_mm[:, j] * 1e3
		finite = np.isfinite(track.Z_mm[:, j])
		ax.plot(y, z_um, color="0.25", lw=0.9, label="detrended height (diagnostic)")
		ax.fill_between(y, -z_lim, z_lim, where=finite, color="tab:green", alpha=0.08, label="finite support")
		if np.isfinite(case["left_boundary_mm"]):
			ax.axvspan(float(case["left_boundary_mm"]), float(case["right_boundary_mm"]), color="tab:orange", alpha=0.16, label="selected longest run")
			ax.axvline(float(case["left_boundary_mm"]), color="tab:blue", lw=1.0)
			ax.axvline(float(case["right_boundary_mm"]), color="tab:red", lw=1.0)
		ax.set_title(f"{case['case_label']}\nx={case['x_position_mm']:.3f} mm, w={case['local_width_mm']:.3f} mm", fontsize=8)
		ax.set_xlabel("y (mm)")
		ax.set_ylabel("z detrended (µm)")
		ax.set_ylim(-z_lim, z_lim)
	for ax in axes.ravel()[n_cases:]:
		ax.axis("off")
	handles, labels = axes[0, 0].get_legend_handles_labels()
	if handles:
		fig.legend(handles[:3], labels[:3], loc="upper right", fontsize=8)
	fig.suptitle(f"Track {track.track_id}: representative cross-sections; detrending did not select boundaries")
	fig.tight_layout(rect=[0, 0, 0.98, 0.96])
	p = fig_dir / f"track_{track.track_id}_representative_cross_sections.png"
	fig.savefig(p, dpi=180)
	plt.close(fig)
	made.append(str(p))
	return made


def status_from_evidence(criterion: str, row: Dict[str, object], macro: Optional[pd.DataFrame], morph: Optional[pd.DataFrame]) -> Tuple[str, str, str, str]:
	"""Reasoned PASS/CONCERN/FAIL/INCONCLUSIVE judgments for falsification."""

	tid = int(row["track_id"])
	if criterion == "longitudinal width continuity":
		p95 = row.get("width_jump_mm_p95", np.nan)
		frac02 = row.get("width_jump_mm_fraction_gt_0.2_mm", np.nan)
		evidence = f"width adjacent-jump p95={p95:.4g} mm; fraction >0.2 mm={frac02:.4g}"
		if np.isfinite(frac02) and frac02 > 0.20:
			return "FAIL", evidence, "Frequent large native-column width jumps indicate structure hopping or severe fragmentation.", "Width trace is not longitudinally coherent."
		if np.isfinite(frac02) and frac02 > 0.05:
			return "CONCERN", evidence, "Large jumps occur at a nontrivial fraction of adjacent native columns.", "Continuity is questionable."
		return "PASS", evidence, "Adjacent width jumps are rare at descriptive 0.2 mm scale.", "Raw width is locally stable by this diagnostic."
	if criterion == "boundary continuity":
		l = row.get("left_boundary_jump_mm_fraction_gt_0.2_mm", np.nan)
		r = row.get("right_boundary_jump_mm_fraction_gt_0.2_mm", np.nan)
		evidence = f"left/right boundary jump fraction >0.2 mm = {l:.4g}/{r:.4g}"
		if np.nanmax([l, r]) > 0.20:
			return "FAIL", evidence, "At least one boundary frequently jumps across y, suggesting object-identity changes.", "Boundary trajectories are discontinuous."
		if np.nanmax([l, r]) > 0.05:
			return "CONCERN", evidence, "Boundary jumps are present but not dominant.", "Boundary coherence needs visual review."
		return "PASS", evidence, "Boundary jumps above 0.2 mm are uncommon.", "Boundary trajectories are comparatively continuous."
	if criterion == "physical plausibility":
		med = row.get("local_width_mm_median", np.nan)
		p95 = row.get("local_width_mm_p95", np.nan)
		evidence = f"width median={med:.4g} mm; p95={p95:.4g} mm; y range=0-{row.get('y_max_mm', np.nan):.4g} mm"
		if np.isfinite(p95) and p95 < 0.3:
			return "FAIL", evidence, "Extracted widths are too narrow to represent the expected macro track surface.", "Finite run may be an internal strip."
		if np.isfinite(p95) and p95 > row.get("y_max_mm", np.inf):
			return "FAIL", evidence, "Width exceeds the measured y-domain.", "Implementation or interpretation error."
		return "CONCERN", evidence, "Widths are bounded by the narrow Wyko y-domain and are far smaller than existing macro-width values.", "May measure measurable surface core, not macro width."
	if criterion == "morphology agreement":
		m = morph[morph["track_id"] == tid] if morph is not None and not morph.empty else pd.DataFrame()
		counts = m["morphology_category"].value_counts().to_dict() if not m.empty else {}
		evidence = f"representative morphology categories={counts}"
		if not counts:
			return "INCONCLUSIVE", evidence, "No morphology diagnostics were available.", "Cannot assess."
		strong = counts.get("coincides_with_strong_height_gradient", 0)
		if strong >= max(1, len(m) // 2):
			return "PASS", evidence, "Many representative finite-support boundaries align with strong detrended height gradients.", "Evidence channels partially agree."
		if counts.get("abrupt_missing_support_transition_with_weak_height_gradient", 0) > 0:
			return "CONCERN", evidence, "Boundaries often mark measurement-validity transitions rather than height-gradient transitions.", "May be measurement support rather than physical edge."
		return "CONCERN", evidence, "Limited representative agreement with height-gradient morphology.", "Visual inspection needed."
	if criterion == "robustness across Tracks 8, 10, 14":
		valid = 1.0 - row.get("zero_finite_column_fraction", np.nan)
		evidence = f"track {tid} finite-run-valid column fraction={valid:.4g}"
		if np.isfinite(valid) and valid < 0.50:
			return "FAIL", evidence, "Less than half of native x-columns yield any finite run.", "Track robustness failure."
		return "PASS", evidence, "All or nearly all columns have a raw longest finite run.", "Candidate can be computed across this track."
	if criterion == "minimum-run sensitivity":
		evidence = f"see minimum_run_sensitivity.csv for track {tid} retained fractions and retained-width stats"
		frac150 = row.get("min_run_150_retained_fraction", np.nan)
		if np.isfinite(frac150) and frac150 < 0.50:
			return "CONCERN", evidence, "A high hypothetical minimum-run filter would reject many columns.", "Primary remains unfiltered, but support is sensitivity-prone."
		return "PASS", evidence, "Hypothetical filters retain most columns for this track.", "No-threshold primary is not dominated by tiny runs."
	if criterion == "internal NaN fragmentation":
		comp = row.get("ambiguous_comparable_runs_fraction", np.nan)
		nr_med = row.get("number_of_finite_runs_median", np.nan)
		evidence = f"median finite runs={nr_med:.4g}; comparable longest/second fraction={comp:.4g}"
		if np.isfinite(comp) and comp > 0.20:
			return "FAIL", evidence, "Largest and second-largest runs are often comparable, creating object-identity ambiguity.", "Candidate 1 likely hops or splits."
		if np.isfinite(comp) and comp > 0.05:
			return "CONCERN", evidence, "Comparable secondary runs occur at meaningful frequency.", "Ambiguous cases need review."
		return "PASS", evidence, "Dominant run is usually clearly larger than the second run.", "Fragmentation is present but not usually ambiguous."
	if criterion == "relationship to existing macro-width":
		m = macro[macro["track_id"] == tid] if macro is not None and not macro.empty else pd.DataFrame()
		if m.empty:
			return "INCONCLUSIVE", "no macro-width comparison available", "smoothed_macro_width_mm unavailable.", "No reference."
		r = m.iloc[0]
		evidence = f"corr={r['correlation_native_candidate1_vs_smoothed_macro']:.4g}; MAE={r['mae_abs_difference_mm']:.4g} mm; means candidate/macro={r['candidate1_mean_mm_at_macro_grid']:.4g}/{r['smoothed_macro_mean_mm']:.4g} mm"
		if np.isfinite(r["mae_abs_difference_mm"]) and r["mae_abs_difference_mm"] > 1.0:
			return "CONCERN", evidence, "Candidate 1 width differs substantially from existing macro-width reference.", "Likely different physical quantity."
		return "PASS", evidence, "Candidate 1 is numerically close to macro-width reference.", "Reference agreement is acceptable."
	if criterion == "detrending/NaN-mask invariance":
		diff = row.get("nan_mask_invariance_diff_pixels", np.nan)
		frac = row.get("nan_mask_invariance_diff_fraction", np.nan)
		evidence = f"raw-vs-detrended finite-mask differences={diff} pixels ({frac:.4g})"
		if diff == 0:
			return "PASS", evidence, "Detrending preserved NaN topology exactly.", "Candidate 1 is invariant to detrending mask."
		return "FAIL", evidence, "Detrending changed validity topology unexpectedly.", "Implementation contradiction."
	if criterion == "evidence-channel agreement":
		evidence = "integrates continuity, morphology, fragmentation, and macro-width comparison diagnostics"
		hard_fail = row.get("ambiguous_comparable_runs_fraction", 0) > 0.20 or row.get("width_jump_mm_fraction_gt_0.2_mm", 0) > 0.20
		if hard_fail:
			return "FAIL", evidence, "One or more core evidence channels indicate Candidate 1 is unstable.", "Do not accept as target."
		return "CONCERN", evidence, "Some channels support computability, but physical meaning vs measurable support remains unresolved.", "Further investigation required before target development."
	return "INCONCLUSIVE", "criterion not implemented", "No judgment.", "No judgment."


def build_falsification_summary(
	track_summary: pd.DataFrame,
	adjacent: pd.DataFrame,
	topology: pd.DataFrame,
	sensitivity: pd.DataFrame,
	macro: pd.DataFrame,
	morphology: pd.DataFrame,
) -> pd.DataFrame:
	merged = track_summary.merge(adjacent, on="track_id", how="left").merge(topology, on="track_id", how="left", suffixes=("", "_topology"))
	for th in MIN_RUN_THRESHOLDS_PX:
		keep = sensitivity[sensitivity["hypothetical_min_run_px"] == th][["track_id", "retained_column_fraction"]].rename(
			columns={"retained_column_fraction": f"min_run_{th}_retained_fraction"}
		)
		merged = merged.merge(keep, on="track_id", how="left")
	criteria = [
		"longitudinal width continuity",
		"boundary continuity",
		"physical plausibility",
		"morphology agreement",
		"robustness across Tracks 8, 10, 14",
		"minimum-run sensitivity",
		"internal NaN fragmentation",
		"relationship to existing macro-width",
		"detrending/NaN-mask invariance",
		"evidence-channel agreement",
	]
	rows: List[Dict[str, object]] = []
	for _, r in merged.iterrows():
		row = r.to_dict()
		for criterion in criteria:
			status, evidence, explanation, qualitative = status_from_evidence(criterion, row, macro, morphology)
			rows.append(
				{
					"track_id": int(row["track_id"]),
					"criterion": criterion,
					"status": status,
					"quantitative_evidence": evidence,
					"qualitative_evidence": qualitative,
					"explanation": explanation,
				}
			)
	return pd.DataFrame(rows)


def git_text(args: Sequence[str]) -> str:
	for git_cmd in ("git", "/usr/bin/git"):
		try:
			return subprocess.check_output([git_cmd, *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
		except Exception:
			continue
	git_dir = REPO_ROOT / ".git"
	head_path = git_dir / "HEAD"
	try:
		head = head_path.read_text(encoding="utf-8").strip()
		if args == ["rev-parse", "--abbrev-ref", "HEAD"] and head.startswith("ref: refs/heads/"):
			return head.replace("ref: refs/heads/", "", 1)
		if args == ["rev-parse", "HEAD"]:
			if head.startswith("ref: "):
				ref_path = git_dir / head.replace("ref: ", "", 1)
				if ref_path.exists():
					return ref_path.read_text(encoding="utf-8").strip()
			return head
	except Exception:
		pass
	return "unavailable"


def main() -> None:
	height_dir = REPO_ROOT / "data" / "raw" / "height_maps"
	run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
	out_dir = REPO_ROOT / "processed_data" / "run_outputs" / f"28_local_width_extraction_audit_{run_tag}"
	table_dir = out_dir / "tables"
	fig_dir = out_dir / "figures"
	metadata_dir = out_dir / "metadata"
	log_dir = out_dir / "logs"
	for d in [table_dir, fig_dir, metadata_dir, log_dir]:
		d.mkdir(parents=True, exist_ok=False)

	native_tables: List[pd.DataFrame] = []
	run_tables: List[pd.DataFrame] = []
	load_rows: List[Dict[str, object]] = []
	track_rows: List[Dict[str, object]] = []
	adjacent_rows: List[Dict[str, object]] = []
	topology_rows: List[Dict[str, object]] = []
	sensitivity_tables: List[pd.DataFrame] = []
	case_tables: List[pd.DataFrame] = []
	morphology_tables: List[pd.DataFrame] = []
	figures: List[str] = []
	source_files: Dict[str, str] = {}
	detrend_meta: Dict[str, object] = {}

	log_lines = ["Experiment 28 raw Candidate 1 local-width extraction audit", f"output_dir={out_dir}"]
	for track_id in TRACK_IDS:
		track = load_track(track_id, height_dir)
		source_files[str(track_id)] = track.source_file
		detrend_meta[str(track_id)] = track.detrend_meta
		load_row = track_load_summary(track)
		load_rows.append(load_row)
		native, runs = extract_longest_runs(track_id, track.Z_mm, track.x_mm, track.y_mm)
		native_tables.append(native)
		run_tables.append(runs)
		track_rows.append(build_track_summary(native, load_row))
		adjacent_rows.append(adjacent_jump_summary(native))
		topology_rows.append(topology_summary(native))
		sensitivity_tables.append(minimum_run_sensitivity(native))
		cases = choose_representative_cases(native, track.x_mm)
		morph = classify_morphology(track, native, cases)
		case_tables.append(cases)
		morphology_tables.append(morph)
		figures.extend(plot_track_diagnostics(fig_dir, track, native, cases))
		log_lines.append(f"Track {track_id}: shape={track.Z_mm.shape}, finite_fraction={np.mean(np.isfinite(track.Z_mm)):.6f}, source={track.source_file}")

	native_all = pd.concat(native_tables, ignore_index=True)
	runs_all = pd.concat(run_tables, ignore_index=True)
	load_summary = pd.DataFrame(load_rows)
	track_summary = pd.DataFrame(track_rows)
	adjacent_summary = pd.DataFrame(adjacent_rows)
	topology_summary_df = pd.DataFrame(topology_rows)
	sensitivity = pd.concat(sensitivity_tables, ignore_index=True)
	representative_cases = pd.concat(case_tables, ignore_index=True)
	morphology = pd.concat(morphology_tables, ignore_index=True)
	macro = compare_macro_width(native_all)
	falsification = build_falsification_summary(track_summary, adjacent_summary, topology_summary_df, sensitivity, macro, morphology)

	native_all.to_csv(table_dir / "native_local_width.csv", index=False)
	runs_all.to_csv(table_dir / "all_finite_runs_by_column.csv", index=False)
	load_summary.to_csv(table_dir / "native_load_summary.csv", index=False)
	track_summary.to_csv(table_dir / "track_summary.csv", index=False)
	adjacent_summary.to_csv(table_dir / "adjacent_jump_summary.csv", index=False)
	topology_summary_df.to_csv(table_dir / "finite_run_topology_summary.csv", index=False)
	sensitivity.to_csv(table_dir / "minimum_run_sensitivity.csv", index=False)
	representative_cases.to_csv(table_dir / "representative_cases.csv", index=False)
	morphology.to_csv(table_dir / "morphology_comparison.csv", index=False)
	macro.to_csv(table_dir / "macro_width_comparison.csv", index=False)
	falsification.to_csv(table_dir / "falsification_summary.csv", index=False)

	metadata = {
		"experiment": "28_local_width_extraction_audit",
		"timestamp": run_tag,
		"git_branch": git_text(["rev-parse", "--abbrev-ref", "HEAD"]),
		"git_commit": git_text(["rev-parse", "HEAD"]),
		"python_version": sys.version,
		"platform": platform.platform(),
		"source_data_files": source_files,
		"tracks_analyzed": TRACK_IDS,
		"sealed_tracks": sorted(SEALED_TRACKS),
		"track21_untouched_confirmation": "Track 21 was neither loaded nor inspected by this script.",
		"organizer_loader": "src/nsf_fmrg_data.py::load_wyko_asc(crop_to_common=True)",
		"extraction_definition": {
			"candidate": "Candidate 1 — longest contiguous finite y-run at each native Wyko x-column",
			"primary_min_run_threshold_px": 0,
			"bridges_nan_gaps": False,
			"smooths_boundaries_or_width": False,
			"uses_height_threshold": False,
			"uses_detrending_to_define_boundaries": False,
			"tie_breaking": "stable first occurrence in y among equal-length runs",
		},
		"diagnostic_thresholds": {
			"minimum_run_sensitivity_px": MIN_RUN_THRESHOLDS_PX,
			"jump_thresholds_mm": JUMP_THRESHOLDS_MM,
			"ambiguous_longest_over_second_ratio_leq": AMBIGUOUS_RATIO_THRESHOLD,
		},
		"detrending_diagnostic_meta_by_track": detrend_meta,
		"outputs": {
			"tables": sorted([p.name for p in table_dir.glob("*.csv")]),
			"figures": [str(Path(p).relative_to(out_dir)) for p in figures],
		},
	}
	(metadata_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
	(log_dir / "experiment.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

	print(f"Wrote Experiment 28 outputs to: {out_dir}")
	print(f"Native width rows: {len(native_all)}")
	print(f"All finite-run rows: {len(runs_all)}")
	print(f"Figures: {len(figures)}")
	print("Track 21 untouched: yes")


if __name__ == "__main__":
	main()
