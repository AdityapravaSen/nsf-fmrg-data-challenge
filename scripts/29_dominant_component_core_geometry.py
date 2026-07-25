"""Experiment 29: dominant 2D finite-component local core geometry.

Target-validation only. This script implements the locked Experiment 29
protocol without predictive modeling, target tuning, morphology cleanup, or
Track 21 geometry access.

The proposed target is local_core_width_mm: the median y-extent of the
dominant longitudinal 4-connected finite Wyko-mask component in a fixed
±0.10 mm window around each thermal anchor.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
	sys.path.append(str(SRC_DIR))

from nsf_fmrg_data import load_wyko_asc  # organizer loader, unchanged


warnings.filterwarnings("ignore", category=RuntimeWarning)

SEALED_TRACKS = {21}
TRACK_IDS = [8, 10, 14]
EXPECTED_ANCHORS_PER_TRACK = 400
assert not (set(TRACK_IDS) & SEALED_TRACKS), "Track 21 is sealed and must not be analyzed."

CONNECTIVITY_4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
AGG_HALF_WINDOW_MM = 0.10
VALID_ANCHOR_MIN_COLUMNS = 25
MAX_PLOT_X_PIXELS = 1800
JUMP_THRESHOLDS_MM = [0.05, 0.10, 0.20, 0.50]
MORPH_SAMPLE_ANCHORS = 20
CENTRAL_EXCLUSION_Y_MM = (0.65, 1.35)


@dataclass(frozen=True)
class TrackData:
	track_id: int
	source_file: str
	Z_mm: np.ndarray
	x_mm: np.ndarray
	y_mm: np.ndarray


@dataclass(frozen=True)
class DominantComponentResult:
	labels: np.ndarray
	dominant_label: int
	n_components_total: int
	pixel_counts: np.ndarray
	x_extents: np.ndarray
	dominant_support_min_x_index: int
	dominant_support_max_x_index: int


def git_text(args: Sequence[str]) -> str:
	for git_cmd in ("git", "/usr/bin/git"):
		try:
			return subprocess.check_output([git_cmd, *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
		except Exception:
			continue
	git_dir = REPO_ROOT / ".git"
	try:
		head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
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


def safe_stat(values: Sequence[float], func, default: float = float("nan")) -> float:
	arr = np.asarray(values, dtype=float)
	arr = arr[np.isfinite(arr)]
	if arr.size == 0:
		return default
	return float(func(arr))


def describe_distribution(values: Sequence[float], prefix: str) -> Dict[str, float]:
	arr = np.asarray(values, dtype=float)
	arr = arr[np.isfinite(arr)]
	keys = ["n", "mean", "std", "median", "iqr", "p05", "p95"]
	if arr.size == 0:
		return {f"{prefix}_{k}": float("nan") for k in keys}
	q25, q75 = np.nanpercentile(arr, [25, 75])
	return {
		f"{prefix}_n": int(arr.size),
		f"{prefix}_mean": float(np.nanmean(arr)),
		f"{prefix}_std": float(np.nanstd(arr, ddof=1)) if arr.size > 1 else 0.0,
		f"{prefix}_median": float(np.nanmedian(arr)),
		f"{prefix}_iqr": float(q75 - q25),
		f"{prefix}_p05": float(np.nanpercentile(arr, 5)),
		f"{prefix}_p95": float(np.nanpercentile(arr, 95)),
	}


def load_thermal_anchors_or_raise(dataset_path: Path, metadata_dir: Optional[Path] = None) -> pd.DataFrame:
	"""Read locked thermal-anchor coordinates and stop on schema/data contradictions.

	The amended protocol requires exactly 400 thermal-anchor rows per development
	track. x_position_mm is the canonical physical anchor coordinate;
	heightmap_x_mm is retained only as provenance metadata when present.
	"""

	required = {"track_id", "frame_index", "x_position_mm"}
	records: List[Dict[str, object]] = []
	with dataset_path.open("r", newline="", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		fieldnames = set(reader.fieldnames or [])
		missing_cols = sorted(required - fieldnames)
		if missing_cols:
			raise RuntimeError(f"final_multimodal_dataset.csv missing required columns: {missing_cols}")
		for row_idx, row in enumerate(reader):
			try:
				track_id = int(float(row["track_id"]))
			except Exception:
				continue
			if track_id in SEALED_TRACKS:
				continue
			if track_id not in TRACK_IDS:
				continue
			try:
				frame_index = int(float(row["frame_index"]))
			except Exception:
				frame_index = -1
			raw_x = row.get("x_position_mm", "")
			try:
				x_val = float(raw_x) if raw_x != "" else float("nan")
			except Exception:
				x_val = float("nan")
			raw_hm_x = row.get("heightmap_x_mm", "")
			try:
				hm_x = float(raw_hm_x) if raw_hm_x != "" else float("nan")
			except Exception:
				hm_x = float("nan")
			records.append(
				{
					"track_id": track_id,
					"anchor_index": None,
					"frame_index": frame_index,
					"x_position_mm": x_val,
					"heightmap_x_mm": hm_x,
					"source_row_index": row_idx,
				}
			)

	anchors = pd.DataFrame(records)
	discrepancies: List[Dict[str, object]] = []
	if set(anchors["track_id"].unique()) != set(TRACK_IDS):
		discrepancies.append({"issue": "development_track_set_mismatch", "observed_tracks": sorted(anchors["track_id"].unique().tolist())})
	for track_id in TRACK_IDS:
		group = anchors[anchors["track_id"] == track_id].copy()
		n_rows = int(len(group))
		n_finite_x = int(np.isfinite(group["x_position_mm"].to_numpy(dtype=float)).sum()) if n_rows else 0
		if n_rows != EXPECTED_ANCHORS_PER_TRACK:
			discrepancies.append({"track_id": track_id, "issue": "anchor_row_count_mismatch", "observed": n_rows, "expected": EXPECTED_ANCHORS_PER_TRACK})
		if n_finite_x != EXPECTED_ANCHORS_PER_TRACK:
			bad_rows = group.loc[~np.isfinite(group["x_position_mm"].to_numpy(dtype=float)), "source_row_index"].astype(int).tolist()
			discrepancies.append(
				{
					"track_id": track_id,
					"issue": "nonfinite_x_position_mm_anchor_coordinates",
					"finite_observed": n_finite_x,
					"expected_finite": EXPECTED_ANCHORS_PER_TRACK,
					"source_row_indices": bad_rows[:25],
					"n_bad_rows": len(bad_rows),
				}
			)
	if discrepancies:
		payload = {
			"experiment": "29_dominant_component_core_geometry",
			"status": "STOPPED_BEFORE_WYKO_GEOMETRY_LOADING",
			"reason": "Amended locked protocol requires exactly 400 finite x_position_mm thermal anchors per development track.",
			"dataset_path": str(dataset_path),
			"discrepancies": discrepancies,
			"track21_geometry_loaded": False,
		}
		if metadata_dir is not None:
			(metadata_dir / "data_discrepancy.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
		raise RuntimeError(json.dumps(payload, indent=2))

	anchors = anchors.sort_values(["track_id", "x_position_mm", "source_row_index"]).reset_index(drop=True)
	anchors["anchor_index"] = anchors.groupby("track_id").cumcount().astype(int)
	assert len(anchors) == EXPECTED_ANCHORS_PER_TRACK * len(TRACK_IDS)
	assert np.all(np.isfinite(anchors["x_position_mm"].to_numpy(dtype=float)))
	return anchors[["track_id", "anchor_index", "frame_index", "x_position_mm", "heightmap_x_mm", "source_row_index"]]


def load_track(track_id: int, height_dir: Path) -> TrackData:
	if track_id in SEALED_TRACKS:
		raise RuntimeError("Track 21 is sealed and must not be loaded.")
	if track_id not in TRACK_IDS:
		raise RuntimeError(f"Unsupported development track: {track_id}")
	hm = load_wyko_asc(height_dir, track_id, crop_to_common=True)
	Z_mm = np.asarray(hm["Z_mm"], dtype=float)
	x_mm = np.asarray(hm["x_actual_mm"], dtype=float)
	y_mm = np.asarray(hm["y_mm"], dtype=float)
	if Z_mm.shape != (len(y_mm), len(x_mm)):
		raise RuntimeError(f"Track {track_id}: Z shape does not match coordinate lengths.")
	if not (np.all(np.isfinite(x_mm)) and np.all(np.isfinite(y_mm))):
		raise RuntimeError(f"Track {track_id}: nonfinite Wyko coordinate array.")
	return TrackData(track_id, str(hm["file"]), Z_mm, x_mm, y_mm)


def find_dominant_component(finite_mask: np.ndarray) -> DominantComponentResult:
	labels, n_components = ndimage.label(finite_mask, structure=CONNECTIVITY_4)
	pixel_counts = np.bincount(labels.ravel(), minlength=n_components + 1).astype(np.int64)
	x_extents = np.zeros(n_components + 1, dtype=np.int64)
	support_min = np.full(n_components + 1, np.iinfo(np.int64).max, dtype=np.int64)
	support_max = np.full(n_components + 1, -1, dtype=np.int64)
	for j in range(labels.shape[1]):
		u = np.unique(labels[:, j])
		u = u[u > 0]
		if u.size:
			x_extents[u] += 1
			support_min[u] = np.minimum(support_min[u], j)
			support_max[u] = np.maximum(support_max[u], j)
	if n_components < 1:
		raise RuntimeError("No finite 2D components found.")
	candidate_labels = np.arange(1, n_components + 1, dtype=int)
	order = sorted(candidate_labels, key=lambda lab: (-int(x_extents[lab]), -int(pixel_counts[lab]), int(lab)))
	dominant_label = int(order[0])
	return DominantComponentResult(
		labels=labels,
		dominant_label=dominant_label,
		n_components_total=int(n_components),
		pixel_counts=pixel_counts,
		x_extents=x_extents,
		dominant_support_min_x_index=int(support_min[dominant_label]),
		dominant_support_max_x_index=int(support_max[dominant_label]),
	)


def native_component_width(track: TrackData, comp: DominantComponentResult) -> pd.DataFrame:
	rows: List[Dict[str, object]] = []
	labels = comp.labels
	lab = comp.dominant_label
	for j, x in enumerate(track.x_mm):
		yy = np.flatnonzero(labels[:, j] == lab)
		if yy.size == 0:
			rows.append(
				{
					"track_id": track.track_id,
					"x_index": int(j),
					"x_position_mm": float(x),
					"component_absent": True,
					"left_boundary_mm": float("nan"),
					"right_boundary_mm": float("nan"),
					"raw_component_width_mm": float("nan"),
					"component_pixel_count_in_column": 0,
					"component_hole_fraction": float("nan"),
				}
			)
			continue
		left_idx = int(np.min(yy))
		right_idx = int(np.max(yy))
		span = right_idx - left_idx + 1
		rows.append(
			{
				"track_id": track.track_id,
				"x_index": int(j),
				"x_position_mm": float(x),
				"component_absent": False,
				"left_boundary_mm": float(track.y_mm[left_idx]),
				"right_boundary_mm": float(track.y_mm[right_idx]),
				"raw_component_width_mm": float(track.y_mm[right_idx] - track.y_mm[left_idx]),
				"component_pixel_count_in_column": int(yy.size),
				"component_hole_fraction": float(1.0 - (yy.size / span)),
			}
		)
	return pd.DataFrame(rows)


def aggregate_to_anchors(native: pd.DataFrame, anchors: pd.DataFrame, x_mm: np.ndarray) -> pd.DataFrame:
	rows: List[Dict[str, object]] = []
	raw_width = native["raw_component_width_mm"].to_numpy(dtype=float)
	left = native["left_boundary_mm"].to_numpy(dtype=float)
	right = native["right_boundary_mm"].to_numpy(dtype=float)
	track_id = int(native["track_id"].iloc[0])
	track_anchors = anchors[anchors["track_id"] == track_id].sort_values("anchor_index")
	x_min = float(np.nanmin(x_mm))
	x_max = float(np.nanmax(x_mm))
	for _, arow in track_anchors.iterrows():
		xa = float(arow["x_position_mm"])
		in_window = np.abs(x_mm - xa) <= AGG_HALF_WINDOW_MM
		valid = in_window & np.isfinite(raw_width)
		count = int(np.sum(valid))
		n_window = int(np.sum(in_window))
		window_intersects_domain = bool((xa + AGG_HALF_WINDOW_MM >= x_min) and (xa - AGG_HALF_WINDOW_MM <= x_max))
		rows.append(
			{
				"track_id": track_id,
				"anchor_index": int(arow["anchor_index"]),
				"frame_index": int(arow["frame_index"]),
				"x_position_mm": xa,
				"heightmap_x_mm": float(arow["heightmap_x_mm"]) if np.isfinite(float(arow["heightmap_x_mm"])) else float("nan"),
				"source_row_index": int(arow["source_row_index"]),
				"window_intersects_native_x_domain": window_intersects_domain,
				"outside_native_x_domain": bool(not window_intersects_domain),
				"local_core_width_mm": safe_stat(raw_width[valid], np.nanmedian),
				"left_boundary_median_mm": safe_stat(left[valid], np.nanmedian),
				"right_boundary_median_mm": safe_stat(right[valid], np.nanmedian),
				"window_valid_column_count": count,
				"window_valid_column_fraction": float(count / n_window) if n_window else 0.0,
				"anchor_valid": bool(count >= VALID_ANCHOR_MIN_COLUMNS),
			}
		)
	return pd.DataFrame(rows)


def robust_plane_fit_diagnostic(Z_mm: np.ndarray, x_mm: np.ndarray, y_mm: np.ndarray) -> np.ndarray:
	y0, y1 = CENTRAL_EXCLUSION_Y_MM
	xs = x_mm[::40]
	ys = y_mm[::2]
	Zs = Z_mm[::2, ::40]
	Xs, Ys = np.meshgrid(xs, ys)
	z = Zs.ravel()
	A = np.c_[Xs.ravel(), Ys.ravel(), np.ones(Xs.size)]
	valid = np.isfinite(z)
	outside = (Ys.ravel() < y0) | (Ys.ravel() > y1)
	keep = valid & outside
	if np.sum(keep) < 100:
		return Z_mm.copy()
	coef = None
	for _ in range(3):
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
	return Z_mm - plane


def boundary_morphology_percentile(track: TrackData, native: pd.DataFrame, anchors: pd.DataFrame) -> pd.DataFrame:
	track_anchors = anchors[(anchors["track_id"] == track.track_id) & anchors["anchor_valid"]].sort_values("x_position_mm")
	if track_anchors.empty:
		return pd.DataFrame()
	idx = np.unique(np.linspace(0, len(track_anchors) - 1, min(MORPH_SAMPLE_ANCHORS, len(track_anchors))).round().astype(int))
	sample = track_anchors.iloc[idx]
	Z_det = robust_plane_fit_diagnostic(track.Z_mm, track.x_mm, track.y_mm)
	y = track.y_mm
	rows: List[Dict[str, object]] = []
	left = native["left_boundary_mm"].to_numpy(dtype=float)
	right = native["right_boundary_mm"].to_numpy(dtype=float)
	for _, arow in sample.iterrows():
		xa = float(arow["x_position_mm"])
		cols = np.flatnonzero((np.abs(track.x_mm - xa) <= AGG_HALF_WINDOW_MM) & np.isfinite(left) & np.isfinite(right))
		col_percentiles: List[float] = []
		for j in cols:
			z = Z_det[:, j]
			finite = np.isfinite(z)
			if np.sum(finite) < 3:
				continue
			grad = np.full_like(z, np.nan, dtype=float)
			finite_idx = np.flatnonzero(finite)
			breaks = np.where(np.diff(finite_idx) > 1)[0]
			starts = np.r_[finite_idx[0], finite_idx[breaks + 1]]
			stops = np.r_[finite_idx[breaks] + 1, finite_idx[-1] + 1]
			for a, b in zip(starts, stops):
				if b - a >= 3:
					grad[a:b] = np.abs(np.gradient(z[a:b], y[a:b])) * 1e3
			finite_grad = grad[np.isfinite(grad)]
			if finite_grad.size == 0:
				continue
			li = int(np.nanargmin(np.abs(y - left[j])))
			ri = int(np.nanargmin(np.abs(y - right[j])))
			edge_vals: List[float] = []
			for edge_idx in (li, ri):
				lo = max(0, edge_idx - 5)
				hi = min(len(y), edge_idx + 6)
				vals = grad[lo:hi]
				edge_vals.extend(vals[np.isfinite(vals)].tolist())
			if not edge_vals:
				continue
			edge_max = float(np.nanmax(edge_vals))
			col_percentiles.append(float(100.0 * np.mean(finite_grad <= edge_max)))
		rows.append(
			{
				"track_id": track.track_id,
				"anchor_index": int(arow["anchor_index"]),
				"x_position_mm": xa,
				"n_columns_evaluated": int(len(col_percentiles)),
				"mean_boundary_edge_gradient_percentile": safe_stat(col_percentiles, np.nanmean),
			}
		)
	return pd.DataFrame(rows)


def component_summary(track: TrackData, comp: DominantComponentResult) -> Dict[str, object]:
	labels = np.arange(1, comp.n_components_total + 1, dtype=int)
	sorted_labels = sorted(labels, key=lambda lab: (-int(comp.x_extents[lab]), -int(comp.pixel_counts[lab]), int(lab)))
	dom = comp.dominant_label
	second = sorted_labels[1] if len(sorted_labels) > 1 else None
	total_finite = int(np.sum(comp.pixel_counts[1:]))
	return {
		"track_id": track.track_id,
		"dominant_label": int(dom),
		"n_components_total": int(comp.n_components_total),
		"dominant_x_extent_columns": int(comp.x_extents[dom]),
		"dominant_x_extent_fraction_of_scan": float(comp.x_extents[dom] / track.Z_mm.shape[1]),
		"dominant_pixel_count": int(comp.pixel_counts[dom]),
		"dominant_over_second_x_extent_ratio": float(comp.x_extents[dom] / comp.x_extents[second]) if second is not None and comp.x_extents[second] > 0 else float("inf"),
		"dominant_over_second_pixel_ratio": float(comp.pixel_counts[dom] / comp.pixel_counts[second]) if second is not None and comp.pixel_counts[second] > 0 else float("inf"),
		"finite_pixels_inside_dominant_fraction": float(comp.pixel_counts[dom] / total_finite) if total_finite else float("nan"),
		"dominant_support_min_x_index": int(comp.dominant_support_min_x_index),
		"dominant_support_max_x_index": int(comp.dominant_support_max_x_index),
	}


def per_column_topology_summary(native: pd.DataFrame) -> Dict[str, object]:
	row: Dict[str, object] = {"track_id": int(native["track_id"].iloc[0])}
	row.update(describe_distribution(native["raw_component_width_mm"], "raw_component_width_mm"))
	row.update(describe_distribution(native["component_pixel_count_in_column"], "component_pixel_count_in_column"))
	row.update(describe_distribution(native["component_hole_fraction"], "component_hole_fraction"))
	row["fraction_component_absent"] = float(native["component_absent"].mean())
	return row


def anchor_coverage_summary(anchor_df: pd.DataFrame) -> Dict[str, object]:
	row: Dict[str, object] = {"track_id": int(anchor_df["track_id"].iloc[0])}
	inside_domain = anchor_df["window_intersects_native_x_domain"].astype(bool)
	row["anchor_valid_fraction"] = float(anchor_df["anchor_valid"].mean())
	row["total_anchor_count"] = int(len(anchor_df))
	row["outside_domain_anchor_count"] = int((~inside_domain).sum())
	row["domain_intersecting_anchor_count"] = int(inside_domain.sum())
	row["zero_valid_column_anchor_count"] = int(((anchor_df["window_valid_column_count"] == 0) & inside_domain).sum())
	row["zero_valid_column_anchor_count_all_anchors"] = int((anchor_df["window_valid_column_count"] == 0).sum())
	row.update(describe_distribution(anchor_df["window_valid_column_count"], "window_valid_column_count"))
	return row


def adjacent_anchor_jump_summary(anchor_df: pd.DataFrame) -> Dict[str, object]:
	width = anchor_df.sort_values("anchor_index")["local_core_width_mm"].to_numpy(dtype=float)
	valid = np.isfinite(width[:-1]) & np.isfinite(width[1:])
	jumps = np.abs(np.diff(width)[valid])
	row: Dict[str, object] = {"track_id": int(anchor_df["track_id"].iloc[0]), "n_adjacent_valid_pairs": int(jumps.size)}
	for k, q in [("median", 50), ("p90", 90), ("p95", 95), ("p99", 99)]:
		row[f"abs_delta_local_core_width_mm_{k}"] = safe_stat(jumps, lambda a, qq=q: np.nanpercentile(a, qq))
	row["abs_delta_local_core_width_mm_max"] = safe_stat(jumps, np.nanmax)
	for th in JUMP_THRESHOLDS_MM:
		row[f"fraction_gt_{th:g}_mm"] = float(np.mean(jumps > th)) if jumps.size else float("nan")
	return row


def cross_track_comparability(anchor_all: pd.DataFrame) -> pd.DataFrame:
	rows: List[Dict[str, object]] = []
	intervals: Dict[int, Tuple[float, float]] = {}
	for track_id, group in anchor_all.groupby("track_id"):
		vals = group.loc[group["anchor_valid"], "local_core_width_mm"].to_numpy(dtype=float)
		vals = vals[np.isfinite(vals)]
		p05 = safe_stat(vals, lambda a: np.nanpercentile(a, 5))
		p95 = safe_stat(vals, lambda a: np.nanpercentile(a, 95))
		intervals[int(track_id)] = (p05, p95)
		q25 = safe_stat(vals, lambda a: np.nanpercentile(a, 25))
		q75 = safe_stat(vals, lambda a: np.nanpercentile(a, 75))
		rows.append(
			{
				"track_id": int(track_id),
				"mean_local_core_width_mm": safe_stat(vals, np.nanmean),
				"std_local_core_width_mm": safe_stat(vals, lambda a: np.nanstd(a, ddof=1) if len(a) > 1 else 0.0),
				"iqr_local_core_width_mm": float(q75 - q25) if np.isfinite(q25) and np.isfinite(q75) else float("nan"),
				"central90_low_mm": p05,
				"central90_high_mm": p95,
			}
		)
	overlap_count = 0
	for a, b in [(8, 10), (8, 14), (10, 14)]:
		lo = max(intervals[a][0], intervals[b][0])
		hi = min(intervals[a][1], intervals[b][1])
		if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
			overlap_count += 1
	df = pd.DataFrame(rows)
	df["pairwise_central90_overlap_count_of_3"] = int(overlap_count)
	return df


def build_falsification_decision(
	dominant_summary: pd.DataFrame,
	native_all: pd.DataFrame,
	topology: pd.DataFrame,
	anchor_all: pd.DataFrame,
	coverage: pd.DataFrame,
	jumps: pd.DataFrame,
	cross: pd.DataFrame,
	morph: pd.DataFrame,
) -> Tuple[pd.DataFrame, str]:
	rows: List[Dict[str, object]] = []

	def add(track_id: object, category: str, criterion: str, observed: object, threshold: str, passed: bool) -> None:
		rows.append({"track_id": track_id, "category": category, "criterion": criterion, "observed_value": observed, "threshold": threshold, "pass": bool(passed)})

	for _, r in dominant_summary.iterrows():
		tid = int(r.track_id)
		add(tid, "A", "A1_n_components_total_ge_2", int(r.n_components_total), ">= 2", int(r.n_components_total) >= 2)
		add(tid, "A", "A2_dominant_x_extent_fraction_ge_0.95", float(r.dominant_x_extent_fraction_of_scan), ">= 0.95", float(r.dominant_x_extent_fraction_of_scan) >= 0.95)
		add(tid, "A", "A3_dominant_over_second_x_extent_ratio_ge_3", float(r.dominant_over_second_x_extent_ratio), ">= 3.0", float(r.dominant_over_second_x_extent_ratio) >= 3.0)

	for _, r in dominant_summary.iterrows():
		tid = int(r.track_id)
		native = native_all[native_all["track_id"] == tid].sort_values("x_index")
		span = native[(native["x_index"] >= int(r.dominant_support_min_x_index)) & (native["x_index"] <= int(r.dominant_support_max_x_index))]
		frac_absent_inside = float(span["component_absent"].mean()) if len(span) else float("nan")
		add(tid, "B", "B1_fraction_component_absent_inside_x_support_le_0.05", frac_absent_inside, "<= 0.05", np.isfinite(frac_absent_inside) and frac_absent_inside <= 0.05)
	for _, r in topology.iterrows():
		add(int(r.track_id), "B", "B2_median_component_hole_fraction_le_0.35", float(r.component_hole_fraction_median), "<= 0.35", float(r.component_hole_fraction_median) <= 0.35)
	for _, r in jumps.iterrows():
		val = float(r.abs_delta_local_core_width_mm_p95)
		add(int(r.track_id), "B", "B3_p95_adjacent_anchor_delta_width_le_0.20", val, "<= 0.20 mm", np.isfinite(val) and val <= 0.20)

	for _, r in cross.iterrows():
		tid = int(r.track_id)
		add(tid, "C", "C1_mean_local_core_width_defined_and_gt_0", float(r.mean_local_core_width_mm), "finite and > 0", np.isfinite(r.mean_local_core_width_mm) and r.mean_local_core_width_mm > 0)
		add(tid, "C", "C2_within_track_std_gt_0", float(r.std_local_core_width_mm), "> 0", np.isfinite(r.std_local_core_width_mm) and r.std_local_core_width_mm > 0)
	overlap_count = int(cross["pairwise_central90_overlap_count_of_3"].iloc[0]) if len(cross) else 0
	add("GLOBAL", "C", "C3_at_least_2_pairwise_central90_overlaps", overlap_count, ">= 2 of 3", overlap_count >= 2)

	for _, r in coverage.iterrows():
		add(int(r.track_id), "D", "D1_anchor_valid_fraction_ge_0.90", float(r.anchor_valid_fraction), ">= 0.90", float(r.anchor_valid_fraction) >= 0.90)
		add(int(r.track_id), "D", "D2_zero_valid_column_anchor_count_inside_domain_eq_0", int(r.zero_valid_column_anchor_count), "== 0 among anchors whose ±0.10 mm window intersects native Wyko x-domain", int(r.zero_valid_column_anchor_count) == 0)

	morph_summary = morph.groupby("track_id")["mean_boundary_edge_gradient_percentile"].mean().reset_index() if not morph.empty else pd.DataFrame(columns=["track_id", "mean_boundary_edge_gradient_percentile"])
	for track_id in TRACK_IDS:
		vals = morph_summary[morph_summary["track_id"] == track_id]
		val = float(vals["mean_boundary_edge_gradient_percentile"].iloc[0]) if len(vals) else float("nan")
		add(track_id, "E", "E1_mean_boundary_edge_gradient_percentile_ge_80", val, ">= 80%", np.isfinite(val) and val >= 80.0)
	medians = {int(r.track_id): float(anchor_all[(anchor_all["track_id"] == int(r.track_id)) & anchor_all["anchor_valid"]]["local_core_width_mm"].median()) for _, r in cross.iterrows()}
	ordering_pass = bool(medians.get(8, -np.inf) > medians.get(10, np.inf) and medians.get(8, -np.inf) > medians.get(14, np.inf) and abs(medians.get(10, np.nan) - medians.get(14, np.nan)) <= max(0.20, 0.25 * np.nanmean([medians.get(10, np.nan), medians.get(14, np.nan)])))
	add("GLOBAL", "E", "E2_median_order_track8_highest_tracks10_14_comparable_lower", medians, "Track 8 highest; Tracks 10/14 comparable lower", ordering_pass)

	df = pd.DataFrame(rows)
	a_fail = int((df[(df["category"] == "A") & (~df["pass"])]).shape[0])
	c_fail = int((df[(df["category"] == "C") & (~df["pass"])]).shape[0])
	bde_fail = int((df[df["category"].isin(["B", "D", "E"]) & (~df["pass"])]).shape[0])
	per_track_fail = df[(df["track_id"].isin(TRACK_IDS)) & (~df["pass"])].groupby("track_id").size().to_dict()
	max_track_fail = max(per_track_fail.values()) if per_track_fail else 0
	if a_fail > 0 or c_fail > 0 or bde_fail >= 7 or max_track_fail >= 4:
		decision = "REJECT TARGET"
	elif a_fail == 0 and c_fail == 0 and bde_fail <= 2 and max_track_fail <= 2:
		decision = "ACCEPT TARGET FOR LEARNABILITY TESTING"
	elif a_fail == 0 and c_fail == 0 and 3 <= bde_fail <= 6:
		decision = "INVESTIGATE FURTHER"
	else:
		decision = "REJECT TARGET"
	df["final_global_decision"] = decision
	df["n_fail_B_D_E"] = bde_fail
	df["max_failures_single_track"] = int(max_track_fail)
	return df, decision


def downsample_step(n_x: int) -> int:
	return max(1, int(math.ceil(n_x / MAX_PLOT_X_PIXELS)))


def plot_track_figures(fig_dir: Path, track: TrackData, comp: DominantComponentResult, native: pd.DataFrame, anchors: pd.DataFrame) -> List[str]:
	made: List[str] = []
	step = downsample_step(track.Z_mm.shape[1])
	x_ds = track.x_mm[::step]
	y = track.y_mm
	dom = comp.dominant_label

	label_view = np.zeros_like(comp.labels[:, ::step], dtype=float)
	label_ds = comp.labels[:, ::step]
	label_view[label_ds > 0] = 1.0
	label_view[label_ds == dom] = 2.0
	fig, ax = plt.subplots(figsize=(14, 4.2))
	ax.imshow(label_view, origin="lower", aspect="auto", extent=[float(x_ds[0]), float(x_ds[-1]), float(y[0]), float(y[-1])], interpolation="nearest", cmap="viridis")
	ax.set_title(f"Track {track.track_id}: finite mask labeled components (dominant highlighted)")
	ax.set_xlabel("x (mm)")
	ax.set_ylabel("y (mm)")
	fig.tight_layout()
	p = fig_dir / f"track_{track.track_id}_validity_mask_labeled_components.png"
	fig.savefig(p, dpi=180)
	plt.close(fig)
	made.append(str(p))

	fig, ax = plt.subplots(figsize=(14, 4.2))
	ax.imshow((comp.labels[:, ::step] == dom).astype(float), origin="lower", aspect="auto", extent=[float(x_ds[0]), float(x_ds[-1]), float(y[0]), float(y[-1])], interpolation="nearest", cmap="gray_r")
	ax.plot(native["x_position_mm"], native["left_boundary_mm"], lw=0.8, color="tab:blue", label="left")
	ax.plot(native["x_position_mm"], native["right_boundary_mm"], lw=0.8, color="tab:red", label="right")
	ax.legend()
	ax.set_title(f"Track {track.track_id}: dominant component boundaries")
	ax.set_xlabel("x (mm)")
	ax.set_ylabel("y (mm)")
	fig.tight_layout()
	p = fig_dir / f"track_{track.track_id}_dominant_component_boundaries.png"
	fig.savefig(p, dpi=180)
	plt.close(fig)
	made.append(str(p))

	fig, ax = plt.subplots(figsize=(13, 3.2))
	ax.plot(native["x_position_mm"], native["raw_component_width_mm"], lw=0.7)
	ax.set_title(f"Track {track.track_id}: native dominant-component width vs x")
	ax.set_xlabel("x (mm)")
	ax.set_ylabel("raw_component_width_mm")
	fig.tight_layout()
	p = fig_dir / f"track_{track.track_id}_native_component_width_vs_x.png"
	fig.savefig(p, dpi=180)
	plt.close(fig)
	made.append(str(p))

	fig, ax = plt.subplots(figsize=(13, 3.2))
	ax.plot(anchors["x_position_mm"], anchors["local_core_width_mm"], marker=".", ms=2, lw=0.8)
	ax.set_title(f"Track {track.track_id}: anchor-aggregated local_core_width_mm")
	ax.set_xlabel("x_position_mm")
	ax.set_ylabel("local_core_width_mm")
	fig.tight_layout()
	p = fig_dir / f"track_{track.track_id}_anchor_aggregated_width_vs_x.png"
	fig.savefig(p, dpi=180)
	plt.close(fig)
	made.append(str(p))

	fig, ax = plt.subplots(figsize=(13, 3.2))
	ax.plot(anchors["x_position_mm"], anchors["left_boundary_median_mm"], marker=".", ms=2, lw=0.8, label="left median")
	ax.plot(anchors["x_position_mm"], anchors["right_boundary_median_mm"], marker=".", ms=2, lw=0.8, label="right median")
	ax.legend()
	ax.set_title(f"Track {track.track_id}: boundary medians vs x")
	ax.set_xlabel("x_position_mm")
	ax.set_ylabel("boundary median (mm)")
	fig.tight_layout()
	p = fig_dir / f"track_{track.track_id}_boundary_medians_vs_x.png"
	fig.savefig(p, dpi=180)
	plt.close(fig)
	made.append(str(p))

	labels = np.arange(1, comp.n_components_total + 1, dtype=int)
	order = sorted(labels, key=lambda lab: (-int(comp.x_extents[lab]), -int(comp.pixel_counts[lab]), int(lab)))[:30]
	fig, ax = plt.subplots(figsize=(10, 4))
	ax.bar(np.arange(len(order)), comp.x_extents[order], label="x extent columns")
	ax.set_xticks(np.arange(len(order)))
	ax.set_xticklabels([str(int(o)) for o in order], rotation=90, fontsize=7)
	ax.set_title(f"Track {track.track_id}: component size ranking by x extent")
	ax.set_xlabel("component label rank")
	ax.set_ylabel("unique x columns intersected")
	fig.tight_layout()
	p = fig_dir / f"track_{track.track_id}_component_size_ranking.png"
	fig.savefig(p, dpi=180)
	plt.close(fig)
	made.append(str(p))
	return made


def plot_global_figures(fig_dir: Path, anchor_all: pd.DataFrame) -> List[str]:
	made: List[str] = []
	fig, ax = plt.subplots(figsize=(8, 4.5))
	data = [anchor_all[(anchor_all["track_id"] == t) & anchor_all["anchor_valid"]]["local_core_width_mm"].dropna().to_numpy() for t in TRACK_IDS]
	ax.boxplot(data, labels=[str(t) for t in TRACK_IDS], showfliers=False)
	ax.set_title("Cross-track local_core_width_mm distributions")
	ax.set_xlabel("track_id")
	ax.set_ylabel("local_core_width_mm")
	fig.tight_layout()
	p = fig_dir / "cross_track_width_distributions.png"
	fig.savefig(p, dpi=180)
	plt.close(fig)
	made.append(str(p))

	fig, ax = plt.subplots(figsize=(8, 4.5))
	coverage = anchor_all.groupby("track_id")["anchor_valid"].mean().reindex(TRACK_IDS)
	ax.bar([str(t) for t in TRACK_IDS], coverage.to_numpy())
	ax.axhline(0.90, color="tab:red", ls="--", lw=1.0, label="D1 threshold")
	ax.set_ylim(0, 1.05)
	ax.set_title("Cross-track anchor coverage")
	ax.set_xlabel("track_id")
	ax.set_ylabel("anchor_valid fraction")
	ax.legend()
	fig.tight_layout()
	p = fig_dir / "cross_track_anchor_coverage.png"
	fig.savefig(p, dpi=180)
	plt.close(fig)
	made.append(str(p))
	return made


def main() -> None:
	run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
	out_dir = REPO_ROOT / "processed_data" / "run_outputs" / f"29_dominant_component_core_geometry_{run_tag}"
	table_dir = out_dir / "tables"
	fig_dir = out_dir / "figures"
	metadata_dir = out_dir / "metadata"
	for d in (table_dir, fig_dir, metadata_dir):
		d.mkdir(parents=True, exist_ok=False)

	dataset_path = REPO_ROOT / "processed_data" / "final_multimodal_dataset.csv"
	anchors = load_thermal_anchors_or_raise(dataset_path, metadata_dir=metadata_dir)

	native_tables: List[pd.DataFrame] = []
	anchor_tables: List[pd.DataFrame] = []
	component_rows: List[Dict[str, object]] = []
	topology_rows: List[Dict[str, object]] = []
	coverage_rows: List[Dict[str, object]] = []
	jump_rows: List[Dict[str, object]] = []
	morph_tables: List[pd.DataFrame] = []
	figures: List[str] = []
	source_files: Dict[str, str] = {}

	height_dir = REPO_ROOT / "data" / "raw" / "height_maps"
	for track_id in TRACK_IDS:
		track = load_track(track_id, height_dir)
		source_files[str(track_id)] = track.source_file
		finite_mask = np.isfinite(track.Z_mm)
		comp = find_dominant_component(finite_mask)
		native = native_component_width(track, comp)
		anchor_df = aggregate_to_anchors(native, anchors, track.x_mm)
		if len(anchor_df) != EXPECTED_ANCHORS_PER_TRACK:
			raise RuntimeError(f"Track {track_id}: expected 400 anchor rows, observed {len(anchor_df)}")
		native_tables.append(native)
		anchor_tables.append(anchor_df)
		component_rows.append(component_summary(track, comp))
		topology_rows.append(per_column_topology_summary(native))
		coverage_rows.append(anchor_coverage_summary(anchor_df))
		jump_rows.append(adjacent_anchor_jump_summary(anchor_df))
		morph_tables.append(boundary_morphology_percentile(track, native, anchor_df))
		figures.extend(plot_track_figures(fig_dir, track, comp, native, anchor_df))

	native_all = pd.concat(native_tables, ignore_index=True)
	anchor_all = pd.concat(anchor_tables, ignore_index=True)
	if len(anchor_all) != EXPECTED_ANCHORS_PER_TRACK * len(TRACK_IDS):
		raise RuntimeError(f"Expected 1200 anchor rows, observed {len(anchor_all)}")

	dominant_summary = pd.DataFrame(component_rows)
	topology_summary = pd.DataFrame(topology_rows)
	coverage_summary = pd.DataFrame(coverage_rows)
	jump_summary = pd.DataFrame(jump_rows)
	cross = cross_track_comparability(anchor_all)
	morph = pd.concat(morph_tables, ignore_index=True) if morph_tables else pd.DataFrame()
	falsification, decision = build_falsification_decision(dominant_summary, native_all, topology_summary, anchor_all, coverage_summary, jump_summary, cross, morph)
	figures.extend(plot_global_figures(fig_dir, anchor_all))

	native_all.to_csv(table_dir / "native_component_width.csv", index=False)
	anchor_all.to_csv(table_dir / "anchor_aggregated_width.csv", index=False)
	dominant_summary.to_csv(table_dir / "dominant_component_summary.csv", index=False)
	topology_summary.to_csv(table_dir / "per_column_topology_summary.csv", index=False)
	coverage_summary.to_csv(table_dir / "anchor_coverage_summary.csv", index=False)
	jump_summary.to_csv(table_dir / "adjacent_anchor_jump_summary.csv", index=False)
	cross.to_csv(table_dir / "cross_track_comparability.csv", index=False)
	morph.to_csv(table_dir / "boundary_morphology_audit.csv", index=False)
	falsification.to_csv(table_dir / "falsification_decision.csv", index=False)

	metadata = {
		"experiment": "29_dominant_component_core_geometry",
		"timestamp": run_tag,
		"git_branch": git_text(["rev-parse", "--abbrev-ref", "HEAD"]),
		"git_commit": git_text(["rev-parse", "HEAD"]),
		"python_version": sys.version,
		"platform": platform.platform(),
		"tracks_analyzed": TRACK_IDS,
		"sealed_tracks": sorted(SEALED_TRACKS),
		"track21_untouched_confirmation": "Track 21 Wyko geometry was neither loaded nor inspected.",
		"source_data_files": source_files,
		"anchor_source": str(dataset_path),
		"anchor_columns_used": ["track_id", "frame_index", "x_position_mm"],
		"anchor_coordinate_semantics": {
			"x_position_mm": "canonical physical Experiment 29 thermal-anchor coordinate used for all 400 anchors per development track",
			"heightmap_x_mm": "Experiment 12 derived nearest-native-Wyko mapping retained only as provenance metadata; not used as the Experiment 29 anchor definition",
			"outside_domain_anchor_policy": "anchors with no native Wyko support inside ±0.10 mm remain represented with NaN width/boundaries, window_valid_column_count=0, window_valid_column_fraction=0, and anchor_valid=False",
			"d2_semantics": "Among anchors whose ±0.10 mm aggregation window intersects the native Wyko x-domain, zero anchors may have window_valid_column_count == 0; outside-domain anchors are explicitly reported but are not D2 failures.",
		},
		"prohibited_columns_used": [],
		"locked_parameters": {
			"connectivity": "4-connected",
			"morphological_operations": "none",
			"dominant_component_primary": "maximum unique x-columns intersected",
			"dominant_component_tie_break": "total component pixel count",
			"aggregation_half_window_mm": AGG_HALF_WINDOW_MM,
			"aggregation": "nanmedian",
			"valid_anchor_min_columns": VALID_ANCHOR_MIN_COLUMNS,
			"height_threshold": "none",
			"detrending_for_extraction": "none",
		},
		"final_decision": decision,
		"outputs": {
			"tables": sorted(p.name for p in table_dir.glob("*.csv")),
			"figures": sorted(p.name for p in fig_dir.glob("*.png")),
		},
	}
	(metadata_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

	print(f"Wrote Experiment 29 outputs to: {out_dir}")
	print(f"Final decision: {decision}")
	print("Track 21 untouched: yes")


if __name__ == "__main__":
	main()
