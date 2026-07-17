"""13_merge_geometry_descriptors.py

Integration step: merge Experiment 12 geometry descriptors into the unified
multimodal anchor table.

Inputs:
- processed_data/phase1_unified_master.csv
- latest processed_data/run_outputs/12_geometry_descriptor_implementation_*/tables/
  geometry_descriptors_track_{8,10,14}.csv

Outputs:
- processed_data/final_multimodal_dataset.csv
- processed_data/run_outputs/13_descriptor_merge_<timestamp>/merge_validation_summary.json
- processed_data/run_outputs/13_descriptor_merge_<timestamp>/descriptor_merge_statistics.csv

Track 21 remains sealed and is not processed.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import json
import sys

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "processed_data"
RUN_OUTPUTS_DIR = PROCESSED_DIR / "run_outputs"
MASTER_PATH = PROCESSED_DIR / "phase1_unified_master.csv"
FINAL_DATASET_PATH = PROCESSED_DIR / "final_multimodal_dataset.csv"

SEALED_TRACKS = {21}
TRACK_IDS = [8, 10, 14]
JOIN_KEYS = ["track_id", "frame_index", "x_position_mm"]
DESCRIPTOR_RUN_GLOB = "12_geometry_descriptor_implementation_*"


class MergeValidationError(RuntimeError):
    """Raised when a required one-to-one merge invariant fails."""


def latest_descriptor_tables_dir() -> Path:
    candidates = sorted(
        [p for p in RUN_OUTPUTS_DIR.glob(DESCRIPTOR_RUN_GLOB) if (p / "tables").is_dir()],
        key=lambda p: p.name,
    )
    if not candidates:
        raise FileNotFoundError("No Experiment 12 descriptor output directory found.")
    return candidates[-1] / "tables"


def require_columns(df: pd.DataFrame, required: List[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise MergeValidationError(f"{label} is missing required columns: {missing}")


def duplicate_key_rows(df: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    return df[df.duplicated(keys, keep=False)].sort_values(keys).copy()


def bool_series(series: pd.Series) -> pd.Series:
    """Convert a possibly object/NaN boolean column to bool without pandas downcast warnings."""

    return series.map(lambda v: bool(v) if pd.notna(v) else False).astype(bool)


def validate_no_track21(df: pd.DataFrame, label: str) -> None:
    if "track_id" not in df.columns:
        raise MergeValidationError(f"{label} has no track_id column.")
    sealed_mask = df["track_id"].astype(int).isin(SEALED_TRACKS)
    if bool(sealed_mask.any()):
        raise MergeValidationError(f"{label} contains sealed Track 21 rows; aborting without merge.")


def validate_master_tracks_without_inspecting_sealed(master: pd.DataFrame) -> None:
    observed = sorted(int(v) for v in master["track_id"].dropna().unique())
    allowed = set(TRACK_IDS) | SEALED_TRACKS
    unexpected = sorted(set(observed) - allowed)
    if unexpected:
        raise MergeValidationError(f"Master table contains unsupported tracks {unexpected}; expected only {sorted(allowed)}.")


def read_master() -> pd.DataFrame:
    if not MASTER_PATH.exists():
        raise FileNotFoundError(f"Missing master table: {MASTER_PATH}")
    master = pd.read_csv(MASTER_PATH)
    require_columns(master, JOIN_KEYS, "phase1_unified_master.csv")
    validate_master_tracks_without_inspecting_sealed(master)
    return master


def read_descriptor_tables(tables_dir: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    missing_files: List[str] = []
    for track_id in TRACK_IDS:
        p = tables_dir / f"geometry_descriptors_track_{track_id}.csv"
        if not p.exists():
            missing_files.append(str(p))
            continue
        df = pd.read_csv(p)
        require_columns(df, JOIN_KEYS, p.name)
        validate_no_track21(df, p.name)
        bad_tracks = sorted(set(int(v) for v in df["track_id"].dropna().unique()) - {track_id})
        if bad_tracks:
            raise MergeValidationError(f"{p.name} contains unexpected track IDs: {bad_tracks}")
        frames.append(df)
    if missing_files:
        raise FileNotFoundError(f"Missing descriptor files: {missing_files}")
    descriptors = pd.concat(frames, ignore_index=True)
    require_columns(descriptors, JOIN_KEYS, "combined descriptor table")
    return descriptors


def descriptor_columns(descriptors: pd.DataFrame) -> List[str]:
    return [c for c in descriptors.columns if c not in JOIN_KEYS]


def build_descriptor_statistics(merged: pd.DataFrame, descriptor_cols: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    pc_cols = [c for c in ["pc1", "pc2", "pc3", "pc4", "pc5"] if c in merged.columns]

    for track_id, group in merged.groupby("track_id", sort=True):
        row: Dict[str, object] = {
            "track_id": int(track_id),
            "n_rows": int(len(group)),
            "descriptor_matched_rows": int(group["_descriptor_matched"].sum()),
            "descriptor_match_fraction": float(group["_descriptor_matched"].mean()) if len(group) else float("nan"),
        }
        if "eligible" in group.columns:
            eligible = bool_series(group["eligible"])
            row["eligible_rows"] = int(eligible.sum())
            row["eligible_fraction"] = float(eligible.mean())
        if "nonflat" in group.columns:
            nonflat = bool_series(group["nonflat"])
            row["nonflat_rows"] = int(nonflat.sum())
            row["nonflat_fraction"] = float(nonflat.mean())
        if "pca_ready" in group.columns:
            pca_ready = bool_series(group["pca_ready"])
            row["pca_ready_rows"] = int(pca_ready.sum())
            row["pca_ready_fraction"] = float(pca_ready.mean())
        if "is_within_heightmap_x_coverage" in group.columns:
            in_x = bool_series(group["is_within_heightmap_x_coverage"])
            row["outside_heightmap_x_coverage_rows"] = int((~in_x).sum())
        if "fallback_baseline_required" in group.columns:
            fallback = bool_series(group["fallback_baseline_required"])
            row["fallback_baseline_rows"] = int(fallback.sum())
        if pc_cols:
            row["rows_with_all_pc_scores"] = int(group[pc_cols].notna().all(axis=1).sum())
            row["rows_with_any_nan_pc_score"] = int(group[pc_cols].isna().any(axis=1).sum())
        for col in descriptor_cols:
            row[f"nan_count__{col}"] = int(group[col].isna().sum())
        rows.append(row)

    total: Dict[str, object] = {
        "track_id": "all",
        "n_rows": int(len(merged)),
        "descriptor_matched_rows": int(merged["_descriptor_matched"].sum()),
        "descriptor_match_fraction": float(merged["_descriptor_matched"].mean()) if len(merged) else float("nan"),
    }
    if "eligible" in merged.columns:
        eligible = bool_series(merged["eligible"])
        total["eligible_rows"] = int(eligible.sum())
        total["eligible_fraction"] = float(eligible.mean())
    if "nonflat" in merged.columns:
        nonflat = bool_series(merged["nonflat"])
        total["nonflat_rows"] = int(nonflat.sum())
        total["nonflat_fraction"] = float(nonflat.mean())
    if "pca_ready" in merged.columns:
        pca_ready = bool_series(merged["pca_ready"])
        total["pca_ready_rows"] = int(pca_ready.sum())
        total["pca_ready_fraction"] = float(pca_ready.mean())
    if "is_within_heightmap_x_coverage" in merged.columns:
        in_x = bool_series(merged["is_within_heightmap_x_coverage"])
        total["outside_heightmap_x_coverage_rows"] = int((~in_x).sum())
    if "fallback_baseline_required" in merged.columns:
        fallback = bool_series(merged["fallback_baseline_required"])
        total["fallback_baseline_rows"] = int(fallback.sum())
    if pc_cols:
        total["rows_with_all_pc_scores"] = int(merged[pc_cols].notna().all(axis=1).sum())
        total["rows_with_any_nan_pc_score"] = int(merged[pc_cols].isna().any(axis=1).sum())
    for col in descriptor_cols:
        total[f"nan_count__{col}"] = int(merged[col].isna().sum())
    rows.append(total)
    return pd.DataFrame(rows)


def validate_and_merge(master: pd.DataFrame, descriptors: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    master_dupes = duplicate_key_rows(master, JOIN_KEYS)
    if not master_dupes.empty:
        raise MergeValidationError(f"Master table has duplicated join keys; first duplicate rows: {master_dupes.head(5).to_dict(orient='records')}")

    master_dev = master[master["track_id"].astype(int).isin(TRACK_IDS)].copy()
    sealed_row_count = int(master["track_id"].astype(int).isin(SEALED_TRACKS).sum())

    descriptor_dupes = duplicate_key_rows(descriptors, JOIN_KEYS)
    if not descriptor_dupes.empty:
        raise MergeValidationError(
            f"Descriptor table has duplicated join keys; first duplicate rows: {descriptor_dupes.head(5).to_dict(orient='records')}"
        )

    unexpected = descriptors.merge(master_dev[JOIN_KEYS], on=JOIN_KEYS, how="left", indicator=True)
    unexpected = unexpected[unexpected["_merge"] == "left_only"]
    if not unexpected.empty:
        raise MergeValidationError(
            f"Found {len(unexpected)} descriptor rows with no master anchor; first rows: {unexpected[JOIN_KEYS].head(10).to_dict(orient='records')}"
        )

    desc_cols = descriptor_columns(descriptors)
    merged = master.merge(descriptors, on=JOIN_KEYS, how="left", validate="one_to_one", indicator=True)
    unmatched = merged[(merged["track_id"].astype(int).isin(TRACK_IDS)) & (merged["_merge"] != "both")]
    if not unmatched.empty:
        raise MergeValidationError(
            f"Found {len(unmatched)} development-track master anchors without exactly one descriptor match; first rows: {unmatched[JOIN_KEYS].head(10).to_dict(orient='records')}"
        )

    row_count_before = int(len(master))
    row_count_after = int(len(merged))
    if row_count_after != row_count_before:
        raise MergeValidationError(f"Row count changed after merge: before={row_count_before}, after={row_count_after}")

    merged["_descriptor_matched"] = merged["_merge"].eq("both")
    merged = merged.drop(columns=["_merge"])

    final_dupes = duplicate_key_rows(merged, JOIN_KEYS)
    if not final_dupes.empty:
        raise MergeValidationError(f"Merged table has duplicated join keys; first duplicate rows: {final_dupes.head(5).to_dict(orient='records')}")

    pc_cols = [c for c in ["pc1", "pc2", "pc3", "pc4", "pc5"] if c in merged.columns]
    summary: Dict[str, object] = {
        "row_count_before_merge": row_count_before,
        "row_count_after_merge": row_count_after,
        "row_count_preserved": bool(row_count_before == row_count_after),
        "master_duplicate_key_rows": int(len(master_dupes)),
        "descriptor_duplicate_key_rows": int(len(descriptor_dupes)),
        "unexpected_descriptor_rows": int(len(unexpected)),
        "unmatched_development_master_anchor_rows": int(len(unmatched)),
        "sealed_track_rows_preserved_without_descriptor_merge": sealed_row_count,
        "every_development_thermal_anchor_matched_exactly_once": True,
        "sealed_track_policy": "Track 21 rows in the pre-existing master table are carried through unmodified with descriptor fields left NaN; no Track 21 descriptor file is read or generated.",
        "no_duplicated_rows_after_merge": bool(final_dupes.empty),
        "descriptor_columns_appended": desc_cols,
        "descriptor_nan_counts": {col: int(merged[col].isna().sum()) for col in desc_cols},
        "descriptor_coverage_by_track": {},
    }
    for track_id, group in merged[merged["track_id"].astype(int).isin(TRACK_IDS)].groupby("track_id", sort=True):
        track_payload: Dict[str, object] = {
            "n_rows": int(len(group)),
            "matched_rows": int(group["_descriptor_matched"].sum()),
            "match_fraction": float(group["_descriptor_matched"].mean()) if len(group) else float("nan"),
        }
        if "eligible" in group.columns:
            track_payload["eligible_rows"] = int(bool_series(group["eligible"]).sum())
        if "nonflat" in group.columns:
            track_payload["nonflat_rows"] = int(bool_series(group["nonflat"]).sum())
        if "pca_ready" in group.columns:
            track_payload["pca_ready_rows"] = int(bool_series(group["pca_ready"]).sum())
        if pc_cols:
            track_payload["rows_with_all_pc_scores"] = int(group[pc_cols].notna().all(axis=1).sum())
        summary["descriptor_coverage_by_track"][str(int(track_id))] = track_payload
    return merged, summary


def main() -> int:
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RUN_OUTPUTS_DIR / f"13_descriptor_merge_{run_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    descriptor_tables_dir = latest_descriptor_tables_dir()
    master = read_master()
    descriptors = read_descriptor_tables(descriptor_tables_dir)
    desc_cols = descriptor_columns(descriptors)
    merged, summary = validate_and_merge(master, descriptors)
    stats = build_descriptor_statistics(merged, desc_cols)

    merged.to_csv(FINAL_DATASET_PATH, index=False)
    stats_path = out_dir / "descriptor_merge_statistics.csv"
    summary_path = out_dir / "merge_validation_summary.json"
    stats.to_csv(stats_path, index=False)

    summary.update(
        {
            "experiment": "13_descriptor_merge",
            "out_dir": str(out_dir),
            "master_input": str(MASTER_PATH),
            "descriptor_tables_dir": str(descriptor_tables_dir),
            "final_dataset": str(FINAL_DATASET_PATH),
            "join_keys": JOIN_KEYS,
            "tracks_processed": TRACK_IDS,
            "sealed_tracks": sorted(SEALED_TRACKS),
            "outputs": [str(FINAL_DATASET_PATH), str(summary_path), str(stats_path)],
            "implementation_assumptions": [
                "Experiment 12 descriptor CSVs are already frozen and validated; this script only merges them.",
                "The latest timestamped Experiment 12 run directory is used as the descriptor source.",
                "Join keys are track_id, frame_index, and x_position_mm exactly as stored in CSV files.",
                "Track 21 descriptor inputs are never read or generated.",
                "If Track 21 rows are present in the pre-existing master table, they are preserved unmodified with descriptor columns left NaN; descriptor one-to-one validation is applied only to Tracks 8, 10, and 14.",
            ],
            "final_dataset_schema": list(merged.columns),
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote final dataset: {FINAL_DATASET_PATH}")
    print(f"Wrote validation summary: {summary_path}")
    print(f"Wrote merge statistics: {stats_path}")
    print(f"Rows before merge: {summary['row_count_before_merge']}")
    print(f"Rows after merge: {summary['row_count_after_merge']}")
    print("One-to-one merge validation: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MergeValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
