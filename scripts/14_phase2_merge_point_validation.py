"""14_phase2_merge_point_validation.py

Phase 2 merge-point validation only.

This script validates the already-merged multimodal table produced by
Experiment 13. It does not modify descriptor extraction, reopen prior geometry
definition experiments, or start Phase 3 modeling.

Inputs:
- processed_data/final_multimodal_dataset.csv
- processed_data/phase1_unified_master.csv

Outputs:
- processed_data/run_outputs/14_phase2_merge_point_validation_<timestamp>/
  - validation_summary.json
  - anchor_alignment_checks.csv
  - unexpected_nan_rows.csv, if any unexpected NaNs are found
  - plots/track_{8,10,14}_phase2_validation.png

Track 21 remains sealed. The validation scope is Tracks 8, 10, and 14 only;
Track 21 is not plotted or used for diagnostics.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "processed_data"
RUN_OUTPUTS_DIR = PROCESSED_DIR / "run_outputs"
FINAL_DATASET_PATH = PROCESSED_DIR / "final_multimodal_dataset.csv"
MASTER_PATH = PROCESSED_DIR / "phase1_unified_master.csv"

TRACK_IDS = [8, 10, 14]
SEALED_TRACKS = {21}
EXPECTED_ROWS_PER_TRACK = 400
JOIN_KEYS = ["track_id", "frame_index", "x_position_mm"]
PHASE1_COLUMNS = [
	"track_id",
	"frame_index",
	"x_position_mm",
	"peak_temp",
	"mean_temp",
	"mp_area_px",
	"mp_centroid_x",
	"mp_centroid_y",
	"mp_length",
	"mp_width",
	"sem_tile_index",
	"x_start_mm",
	"x_end_mm",
	"substrate_roughness_variance",
	"substrate_mean_intensity",
]
PLOT_COLUMNS = ["mp_area_px", "substrate_roughness_variance", "pc1", "amplitude_um"]
PC_COLUMNS = ["pc1", "pc2", "pc3", "pc4", "pc5"]
ANCHOR_TOLERANCE_MM = 1.0e-9
TRACK10_REGIME_WINDOW_MM = (78.0, 97.0)


class ValidationError(RuntimeError):
	"""Raised when a required Phase 2 validation invariant fails."""


def require_columns(df: pd.DataFrame, required: List[str], label: str) -> None:
	missing = [c for c in required if c not in df.columns]
	if missing:
		raise ValidationError(f"{label} is missing required columns: {missing}")


def bool_series(series: pd.Series) -> pd.Series:
	return series.map(lambda value: bool(value) if pd.notna(value) else False).astype(bool)


def finite_mad(values: pd.Series) -> float:
	arr = np.asarray(values.dropna(), dtype=float)
	if arr.size == 0:
		return float("nan")
	med = float(np.nanmedian(arr))
	mad = float(1.4826 * np.nanmedian(np.abs(arr - med)))
	return mad if mad > 0 else float("nan")


def read_validation_inputs() -> Tuple[pd.DataFrame, pd.DataFrame]:
	if not FINAL_DATASET_PATH.exists():
		raise FileNotFoundError(f"Missing merged final dataset: {FINAL_DATASET_PATH}")
	if not MASTER_PATH.exists():
		raise FileNotFoundError(f"Missing Phase 1 master table: {MASTER_PATH}")

	final = pd.read_csv(FINAL_DATASET_PATH)
	master = pd.read_csv(MASTER_PATH)
	required_final = sorted(set(PHASE1_COLUMNS + PLOT_COLUMNS + PC_COLUMNS + [
		"heightmap_x_mm",
		"heightmap_x_delta_mm",
		"heightmap_x_index",
		"is_within_heightmap_x_coverage",
		"amplitude_um",
		"signed_elevation_um",
		"eligible",
		"nonflat",
		"pca_ready",
		"regime_id",
		"finite_fraction",
		"central_corridor_finite_fraction",
		"substrate_side_finite_fraction",
		"baseline_support_count",
		"fallback_baseline_required",
		"is_within_boundary_exclusion",
		"shape_support_fraction_on_common_grid",
		"retained_pca_grid_finite_fraction",
		"normalization_status",
		"shape_center_um_median",
		"shape_scale_um_mad",
		"_descriptor_matched",
	]))
	require_columns(final, required_final, "final_multimodal_dataset.csv")
	require_columns(master, PHASE1_COLUMNS, "phase1_unified_master.csv")
	return final, master


def development_subset(df: pd.DataFrame) -> pd.DataFrame:
	return df[df["track_id"].astype(int).isin(TRACK_IDS)].copy()


def duplicate_key_count(df: pd.DataFrame) -> int:
	return int(df.duplicated(JOIN_KEYS, keep=False).sum())


def validate_row_counts(dev_final: pd.DataFrame, dev_master: pd.DataFrame) -> Dict[str, Dict[str, object]]:
	payload: Dict[str, Dict[str, object]] = {}
	for track_id in TRACK_IDS:
		n_final = int((dev_final["track_id"].astype(int) == track_id).sum())
		n_master = int((dev_master["track_id"].astype(int) == track_id).sum())
		payload[str(track_id)] = {
			"final_rows": n_final,
			"phase1_rows": n_master,
			"expected_rows": EXPECTED_ROWS_PER_TRACK,
			"final_pass": bool(n_final == EXPECTED_ROWS_PER_TRACK),
			"phase1_pass": bool(n_master == EXPECTED_ROWS_PER_TRACK),
		}
	return payload


def build_anchor_alignment_checks(dev_final: pd.DataFrame, dev_master: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
	checks: List[Dict[str, object]] = []
	for track_id in TRACK_IDS:
		f = dev_final[dev_final["track_id"].astype(int) == track_id][JOIN_KEYS].sort_values(JOIN_KEYS).reset_index(drop=True)
		m = dev_master[dev_master["track_id"].astype(int) == track_id][JOIN_KEYS].sort_values(JOIN_KEYS).reset_index(drop=True)
		same_length = len(f) == len(m)
		frame_match = bool(same_length and f["frame_index"].equals(m["frame_index"]))
		if same_length:
			dx = pd.to_numeric(f["x_position_mm"], errors="coerce") - pd.to_numeric(m["x_position_mm"], errors="coerce")
			max_abs_dx = float(np.nanmax(np.abs(dx))) if len(dx) else 0.0
			exact_x_match = bool(max_abs_dx <= ANCHOR_TOLERANCE_MM)
			mismatched_rows = int((np.abs(dx) > ANCHOR_TOLERANCE_MM).sum())
		else:
			max_abs_dx = float("nan")
			exact_x_match = False
			mismatched_rows = -1
		checks.append(
			{
				"track_id": int(track_id),
				"phase1_rows": int(len(m)),
				"final_rows": int(len(f)),
				"same_row_count": bool(same_length),
				"frame_index_sequence_match": frame_match,
				"max_abs_x_position_delta_mm": max_abs_dx,
				"x_position_alignment_tolerance_mm": ANCHOR_TOLERANCE_MM,
				"x_position_sequence_match": exact_x_match,
				"mismatched_x_rows": mismatched_rows,
				"anchor_alignment_pass": bool(same_length and frame_match and exact_x_match),
			}
		)
	checks_df = pd.DataFrame(checks)
	payload = {
		"duplicate_key_rows_final_dev_tracks": duplicate_key_count(dev_final),
		"duplicate_key_rows_phase1_dev_tracks": duplicate_key_count(dev_master),
		"all_tracks_anchor_aligned": bool(checks_df["anchor_alignment_pass"].all()),
		"max_abs_x_position_delta_mm_all_tracks": float(checks_df["max_abs_x_position_delta_mm"].max()),
	}
	return checks_df, payload


def expected_nan_mask(df: pd.DataFrame, col: str) -> pd.Series:
	outside = ~bool_series(df["is_within_heightmap_x_coverage"])
	pca_not_ready = ~bool_series(df["pca_ready"])
	no_central_values = pd.to_numeric(df["central_corridor_finite_fraction"], errors="coerce").fillna(0.0).eq(0.0)
	normalization_status = df["normalization_status"].astype(str)

	if col in PHASE1_COLUMNS:
		return pd.Series(False, index=df.index)
	if col in ["heightmap_x_mm", "heightmap_x_delta_mm"]:
		return outside
	if col in PC_COLUMNS:
		return pca_not_ready
	if col in ["amplitude_um", "signed_elevation_um"]:
		return outside | no_central_values
	if col in [
		"finite_fraction",
		"central_corridor_finite_fraction",
		"substrate_side_finite_fraction",
		"baseline_support_count",
		"shape_support_fraction_on_common_grid",
		"retained_pca_grid_finite_fraction",
	]:
		return outside
	if col in ["shape_center_um_median", "shape_scale_um_mad"]:
		return outside | normalization_status.eq("insufficient_support")
	return pd.Series(False, index=df.index)


def validate_unexpected_nans(dev_final: pd.DataFrame) -> Tuple[Dict[str, object], pd.DataFrame]:
	columns_to_check = [
		*PHASE1_COLUMNS,
		"heightmap_x_mm",
		"heightmap_x_delta_mm",
		*PC_COLUMNS,
		"amplitude_um",
		"signed_elevation_um",
		"eligible",
		"nonflat",
		"pca_ready",
		"regime_id",
		"finite_fraction",
		"central_corridor_finite_fraction",
		"substrate_side_finite_fraction",
		"baseline_support_count",
		"fallback_baseline_required",
		"is_within_boundary_exclusion",
		"shape_support_fraction_on_common_grid",
		"retained_pca_grid_finite_fraction",
		"normalization_status",
		"shape_center_um_median",
		"shape_scale_um_mad",
		"_descriptor_matched",
	]
	rows: List[Dict[str, object]] = []
	by_column: Dict[str, Dict[str, int]] = {}
	for col in columns_to_check:
		is_nan = dev_final[col].isna()
		allowed = expected_nan_mask(dev_final, col)
		unexpected = is_nan & ~allowed
		by_column[col] = {
			"nan_count": int(is_nan.sum()),
			"expected_nan_count": int((is_nan & allowed).sum()),
			"unexpected_nan_count": int(unexpected.sum()),
		}
		for row in dev_final.loc[unexpected, ["track_id", "frame_index", "x_position_mm", "normalization_status", "pca_ready", "is_within_heightmap_x_coverage"]].itertuples(index=False):
			rows.append(
				{
					"track_id": int(row.track_id),
					"frame_index": int(row.frame_index),
					"x_position_mm": float(row.x_position_mm),
					"column": col,
					"normalization_status": str(row.normalization_status),
					"pca_ready": bool(row.pca_ready),
					"is_within_heightmap_x_coverage": bool(row.is_within_heightmap_x_coverage),
				}
			)
	unexpected_df = pd.DataFrame(rows)
	payload = {
		"columns_checked": columns_to_check,
		"by_column": by_column,
		"unexpected_nan_cells": int(len(unexpected_df)),
		"unexpected_nan_pass": bool(unexpected_df.empty),
		"allowed_descriptor_validity_states": {
			"pc1_to_pc5": "NaN is expected when pca_ready is False.",
			"amplitude_um_and_signed_elevation_um": "NaN is expected only outside height-map x coverage or when the central corridor has zero finite support.",
			"heightmap_x_mm_and_heightmap_x_delta_mm": "NaN is expected only outside height-map x coverage.",
			"shape_center_um_median_and_shape_scale_um_mad": "NaN is expected outside coverage or when normalization_status is insufficient_support.",
		},
	}
	return payload, unexpected_df


def summarize_descriptor_coverage(dev_final: pd.DataFrame) -> Dict[str, object]:
	payload: Dict[str, object] = {}
	for track_id, group in dev_final.groupby("track_id", sort=True):
		track_payload = {
			"n_rows": int(len(group)),
			"descriptor_matched_rows": int(bool_series(group["_descriptor_matched"]).sum()),
			"within_heightmap_x_coverage_rows": int(bool_series(group["is_within_heightmap_x_coverage"]).sum()),
			"eligible_rows": int(bool_series(group["eligible"]).sum()),
			"nonflat_rows": int(bool_series(group["nonflat"]).sum()),
			"pca_ready_rows": int(bool_series(group["pca_ready"]).sum()),
			"normalization_status_counts": {str(k): int(v) for k, v in group["normalization_status"].value_counts(dropna=False).items()},
			"pc1_finite_rows": int(group["pc1"].notna().sum()),
			"amplitude_um_finite_rows": int(group["amplitude_um"].notna().sum()),
		}
		payload[str(int(track_id))] = track_payload
	return payload


def validate_track10_regime_visibility(dev_final: pd.DataFrame) -> Dict[str, object]:
	track10 = dev_final[dev_final["track_id"].astype(int) == 10].copy()
	lo, hi = TRACK10_REGIME_WINDOW_MM
	pre = track10[(track10["x_position_mm"] >= 70.0) & (track10["x_position_mm"] < lo)]
	window = track10[(track10["x_position_mm"] >= lo) & (track10["x_position_mm"] <= hi)]
	post = track10[(track10["x_position_mm"] > hi) & (track10["x_position_mm"] <= 100.0)]

	lowcoherence = track10[track10["regime_id"].astype(str).str.contains("lowcoherence", na=False)]
	window_lowcoherence = lowcoherence[(lowcoherence["x_position_mm"] >= lo) & (lowcoherence["x_position_mm"] <= hi)]

	binned_rows: List[Dict[str, object]] = []
	for start in np.arange(76.0, 100.0, 2.0):
		stop = float(start + 2.0)
		g = track10[(track10["x_position_mm"] >= start) & (track10["x_position_mm"] < stop)]
		binned_rows.append(
			{
				"x_start_mm": float(start),
				"x_stop_mm": stop,
				"n_rows": int(len(g)),
				"pc1_finite_rows": int(g["pc1"].notna().sum()),
				"pc1_median": float(g["pc1"].median()) if g["pc1"].notna().any() else None,
				"amplitude_um_median": float(g["amplitude_um"].median()) if g["amplitude_um"].notna().any() else None,
			}
		)

	pc1_scale = finite_mad(track10["pc1"])
	amp_scale = finite_mad(track10["amplitude_um"])
	pre_pc1 = float(pre["pc1"].median()) if pre["pc1"].notna().any() else float("nan")
	window_pc1 = float(window["pc1"].median()) if window["pc1"].notna().any() else float("nan")
	pre_amp = float(pre["amplitude_um"].median()) if pre["amplitude_um"].notna().any() else float("nan")
	window_amp = float(window["amplitude_um"].median()) if window["amplitude_um"].notna().any() else float("nan")
	pc1_effect = float((window_pc1 - pre_pc1) / pc1_scale) if np.isfinite(pc1_scale) else float("nan")
	amp_effect = float((window_amp - pre_amp) / amp_scale) if np.isfinite(amp_scale) else float("nan")
	finite_window_bins = [r for r in binned_rows if lo <= r["x_start_mm"] <= hi and r["pc1_finite_rows"] > 0]
	high_pc1_bins = [r for r in finite_window_bins if r["pc1_median"] is not None and float(r["pc1_median"]) > pre_pc1]

	pass_flag = bool(
		len(window) > 0
		and int(window["pc1"].notna().sum()) >= 20
		and len(window_lowcoherence) > 0
		and np.isfinite(pc1_effect)
		and abs(pc1_effect) >= 0.25
		and len(high_pc1_bins) >= 3
	)
	return {
		"track_id": 10,
		"target_window_mm": [lo, hi],
		"pre_window_mm": [70.0, lo],
		"post_window_mm": [hi, 100.0],
		"window_rows": int(len(window)),
		"pre_window_rows": int(len(pre)),
		"post_window_rows": int(len(post)),
		"window_pc1_finite_rows": int(window["pc1"].notna().sum()),
		"pre_pc1_median": pre_pc1,
		"window_pc1_median": window_pc1,
		"pc1_robust_effect_vs_pre_mad_units": pc1_effect,
		"pre_amplitude_um_median": pre_amp,
		"window_amplitude_um_median": window_amp,
		"amplitude_robust_effect_vs_pre_mad_units": amp_effect,
		"lowcoherence_rows_in_78_97_mm": int(len(window_lowcoherence)),
		"lowcoherence_regime_ids_in_window": sorted(set(str(v) for v in window_lowcoherence["regime_id"].dropna().unique())),
		"pc1_bins_above_pre_median_in_window": int(len(high_pc1_bins)),
		"binned_2mm_descriptor_medians_76_100mm": binned_rows,
		"visible_descriptor_behavior_pass": pass_flag,
		"interpretation": "Track 10 is considered visibly represented when the 78-97 mm window has finite PC1 support, lowcoherence regime labels, and a robust PC1 shift relative to the 70-78 mm pre-window.",
	}


def plot_track_validation(dev_final: pd.DataFrame, out_dir: Path) -> List[str]:
	plots_dir = out_dir / "plots"
	plots_dir.mkdir(parents=True, exist_ok=True)
	plot_paths: List[str] = []
	specs = [
		("mp_area_px", "Thermal melt-pool area (px)"),
		("substrate_roughness_variance", "SEM substrate roughness variance"),
		("pc1", "PC1 geometry descriptor"),
		("amplitude_um", "Amplitude (µm)"),
	]
	for track_id in TRACK_IDS:
		group = dev_final[dev_final["track_id"].astype(int) == track_id].sort_values("x_position_mm")
		fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(11, 10), sharex=True, constrained_layout=True)
		for ax, (col, ylabel) in zip(axes, specs):
			ax.plot(group["x_position_mm"], group[col], marker="o", markersize=2.5, linewidth=1.0)
			ax.set_ylabel(ylabel)
			ax.grid(True, alpha=0.25)
			if track_id == 10:
				ax.axvspan(TRACK10_REGIME_WINDOW_MM[0], TRACK10_REGIME_WINDOW_MM[1], color="tab:orange", alpha=0.12, label="78-97 mm regime window")
				ax.axvline(TRACK10_REGIME_WINDOW_MM[0], color="tab:orange", linestyle="--", linewidth=1.0)
				ax.axvline(TRACK10_REGIME_WINDOW_MM[1], color="tab:orange", linestyle="--", linewidth=1.0)
		axes[0].set_title(f"Phase 2 merge-point validation: Track {track_id}")
		axes[-1].set_xlabel("x_position_mm")
		if track_id == 10:
			axes[0].legend(loc="best")
		path = plots_dir / f"track_{track_id}_phase2_validation.png"
		fig.savefig(path, dpi=180)
		plt.close(fig)
		plot_paths.append(str(path))
	return plot_paths


def aggregate_pass(summary: Dict[str, object]) -> bool:
	return bool(
		all(v["final_pass"] and v["phase1_pass"] for v in summary["row_counts_by_track"].values())
		and summary["anchor_alignment"]["all_tracks_anchor_aligned"]
		and summary["anchor_alignment"]["duplicate_key_rows_final_dev_tracks"] == 0
		and summary["anchor_alignment"]["duplicate_key_rows_phase1_dev_tracks"] == 0
		and summary["nan_validation"]["unexpected_nan_pass"]
		and summary["track10_regime_visibility"]["visible_descriptor_behavior_pass"]
	)


def main() -> int:
	run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
	out_dir = RUN_OUTPUTS_DIR / f"14_phase2_merge_point_validation_{run_tag}"
	out_dir.mkdir(parents=True, exist_ok=True)

	final, master = read_validation_inputs()
	dev_final = development_subset(final)
	dev_master = development_subset(master)

	row_counts = validate_row_counts(dev_final, dev_master)
	anchor_checks, anchor_payload = build_anchor_alignment_checks(dev_final, dev_master)
	nan_payload, unexpected_nan_df = validate_unexpected_nans(dev_final)
	descriptor_coverage = summarize_descriptor_coverage(dev_final)
	track10_visibility = validate_track10_regime_visibility(dev_final)
	plot_paths = plot_track_validation(dev_final, out_dir)

	anchor_checks_path = out_dir / "anchor_alignment_checks.csv"
	anchor_checks.to_csv(anchor_checks_path, index=False)
	unexpected_nan_path = out_dir / "unexpected_nan_rows.csv"
	if not unexpected_nan_df.empty:
		unexpected_nan_df.to_csv(unexpected_nan_path, index=False)

	summary: Dict[str, object] = {
		"experiment": "14_phase2_merge_point_validation",
		"scope": "Phase 2 validation only; no descriptor extraction changes and no Phase 3 modeling.",
		"out_dir": str(out_dir),
		"inputs": {
			"final_multimodal_dataset": str(FINAL_DATASET_PATH),
			"phase1_unified_master": str(MASTER_PATH),
		},
		"tracks_validated": TRACK_IDS,
		"sealed_tracks": sorted(SEALED_TRACKS),
		"sealed_track_policy": "Track 21 is held out and is not plotted or used for diagnostics in this validation script.",
		"final_dataset_schema": list(final.columns),
		"row_counts_by_track": row_counts,
		"anchor_alignment": anchor_payload,
		"anchor_alignment_checks_csv": str(anchor_checks_path),
		"nan_validation": nan_payload,
		"unexpected_nan_rows_csv": str(unexpected_nan_path) if not unexpected_nan_df.empty else None,
		"descriptor_coverage_by_track": descriptor_coverage,
		"track10_regime_visibility": track10_visibility,
		"plots": plot_paths,
	}
	summary["overall_validation_pass"] = aggregate_pass(summary)

	summary_path = out_dir / "validation_summary.json"
	summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

	print(f"Wrote validation summary: {summary_path}")
	print(f"Wrote anchor alignment checks: {anchor_checks_path}")
	print("Wrote plots:")
	for path in plot_paths:
		print(f"  {path}")
	print(f"Overall validation pass: {summary['overall_validation_pass']}")
	return 0 if summary["overall_validation_pass"] else 1


if __name__ == "__main__":
	try:
		raise SystemExit(main())
	except ValidationError as exc:
		print(f"ERROR: {exc}", file=sys.stderr)
		raise SystemExit(1)
