"""Phase III target alignment utilities.

This module aligns frozen geometry targets from
processed_data/final_multimodal_dataset.csv with metadata produced by the
Phase III feature preprocessor.

It does not recompute descriptors, perform feature engineering, or depend on
PyTorch. Exact joins are performed only on the canonical row identity:
track_id, frame_index, and x_position_mm.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


class Phase3TargetAligner:
    """Align frozen Phase III geometry targets to feature-window metadata.

    The aligner consumes metadata rows produced by the Phase III feature
    preprocessor and performs an exact, one-to-one merge against the frozen
    multimodal dataset. It preserves incoming metadata order and returns NumPy
    target arrays suitable for downstream modeling code without importing or
    depending on PyTorch.

    The class intentionally does not recompute geometry descriptors, perform
    feature engineering, create rolling windows, or use nearest-neighbor joins.
    """

    def __init__(
        self,
        dataset_path: Optional[Union[str, Path]] = None,
        dataset: Optional[pd.DataFrame] = None,
        join_keys: Sequence[str] = ("track_id", "frame_index", "x_position_mm"),
    ) -> None:
        """Create a Phase III target aligner.

        Parameters
        ----------
        dataset_path : str or pathlib.Path, optional
            Path to the frozen `final_multimodal_dataset.csv`. If omitted and
            `dataset` is also omitted, the repository-default path
            `processed_data/final_multimodal_dataset.csv` is resolved relative
            to this module.
        dataset : pandas.DataFrame, optional
            Already-loaded frozen multimodal dataset. This is useful for tests
            or pipelines that want to avoid repeated file I/O. The dataframe is
            copied on construction.
        join_keys : sequence of str, optional
            Exact row-identity columns used to align metadata to target rows.
            The Phase III contract uses `track_id`, `frame_index`, and
            `x_position_mm`.

        Raises
        ------
        ValueError
            If both `dataset_path` and `dataset` are provided, or if the loaded
            dataset violates the expected schema or uniqueness contract.
        FileNotFoundError
            If `dataset_path` is supplied or resolved but does not exist.
        """
        # ==========================================
        # TASK 1: STORE THE JOIN CONTRACT
        # ==========================================
        self.join_keys = list(join_keys)
        self.dataset_path = Path(dataset_path) if dataset_path is not None else None

        # ==========================================
        # TASK 2: DEFINE SUPPORTED TARGET GROUPS
        # ==========================================
        self.target_groups: Dict[str, Dict[str, List[str]]] = {
            "pca_shape": {
                "targets": ["pc1", "pc2", "pc3", "pc4", "pc5"],
                "validity": [
                    "eligible",
                    "nonflat",
                    "pca_ready",
                    "normalization_status",
                    "shape_support_fraction_on_common_grid",
                    "retained_pca_grid_finite_fraction",
                    "_descriptor_matched",
                ],
            },
            "amplitude": {
                "targets": ["amplitude_um"],
                "validity": [
                    "eligible",
                    "is_within_heightmap_x_coverage",
                    "central_corridor_finite_fraction",
                    "finite_fraction",
                    "baseline_support_count",
                    "fallback_baseline_required",
                    "_descriptor_matched",
                ],
            },
            "signed_elevation": {
                "targets": ["signed_elevation_um"],
                "validity": [
                    "eligible",
                    "is_within_heightmap_x_coverage",
                    "central_corridor_finite_fraction",
                    "finite_fraction",
                    "baseline_support_count",
                    "fallback_baseline_required",
                    "_descriptor_matched",
                ],
            },
            "smoothed_macro_width": {
                "targets": ["smoothed_macro_width_mm"],
                "validity": [
                    "eligible", 
                    "is_within_heightmap_x_coverage"
                ],
            },
        }

        # ==========================================
        # TASK 3: LOAD OR ACCEPT THE FROZEN DATASET
        # ==========================================
        if dataset_path is not None and dataset is not None:
            raise ValueError("Provide either dataset_path or dataset, not both.")

        if dataset is not None:
            self.dataset = dataset.copy()
        else:
            if self.dataset_path is None:
                self.dataset_path = self._default_dataset_path()
            if not self.dataset_path.exists():
                raise FileNotFoundError(f"Dataset file does not exist: {self.dataset_path}")
            self.dataset = pd.read_csv(self.dataset_path)

        self.validate_dataset()

    def _default_dataset_path(self) -> Path:
        """Resolve the repository-default final multimodal dataset path."""

        repo_root = Path(__file__).resolve().parents[2]
        return repo_root / "processed_data" / "final_multimodal_dataset.csv"

    def _all_target_columns(self) -> List[str]:
        cols: List[str] = []
        for spec in self.target_groups.values():
            cols.extend(spec["targets"])
        return sorted(set(cols))

    def _all_validity_columns(self) -> List[str]:
        cols: List[str] = []
        for spec in self.target_groups.values():
            cols.extend(spec["validity"])
        return sorted(set(cols))

    def _target_columns(self, target_group: str) -> List[str]:
        if target_group not in self.target_groups:
            available = sorted(self.target_groups.keys())
            raise ValueError(f"Unsupported target_group={target_group!r}. Available groups: {available}")
        return list(self.target_groups[target_group]["targets"])

    def _validity_columns(self, target_group: str) -> List[str]:
        if target_group not in self.target_groups:
            available = sorted(self.target_groups.keys())
            raise ValueError(f"Unsupported target_group={target_group!r}. Available groups: {available}")
        return list(self.target_groups[target_group]["validity"])

    def validate_dataset(self) -> bool:
        """Validate the frozen final multimodal dataset contract.

        Validation checks that the dataset is loaded, contains all join keys,
        supported target columns, and relevant validity columns, and has unique
        row identity over the configured join keys. This method is schema and
        integrity validation only; it does not derive, fill, or recompute any
        descriptor fields.

        Returns
        -------
        bool
            `True` when validation succeeds.

        Raises
        ------
        ValueError
            If the dataset is empty, required columns are missing, join-key
            values are missing, or join keys are duplicated.
        """

        if self.dataset is None or len(self.dataset) == 0:
            raise ValueError("Dataset is empty or was not loaded.")

        required_cols = sorted(set(self.join_keys + self._all_target_columns() + self._all_validity_columns()))
        missing = [col for col in required_cols if col not in self.dataset.columns]
        if missing:
            raise ValueError(f"Dataset is missing required columns: {missing}")

        missing_keys = self.dataset[self.join_keys].isna().any(axis=1)
        if bool(missing_keys.any()):
            raise ValueError(f"Dataset contains {int(missing_keys.sum())} rows with missing join-key values.")

        duplicate_keys = self.dataset.duplicated(self.join_keys, keep=False)
        if bool(duplicate_keys.any()):
            examples = self.dataset.loc[duplicate_keys, self.join_keys].head(5).to_dict(orient="records")
            raise ValueError(f"Dataset join keys are not unique. Example duplicates: {examples}")

        # This module intentionally consumes existing descriptor columns only.
        # It does not import or call any descriptor-extraction code.
        return True

    def validate_meta(self, meta_df: pd.DataFrame) -> bool:
        """Validate feature-window metadata before target alignment.

        The metadata must be the handshake output from the feature pipeline: one
        row per feature sequence, where each row identifies the final frame in
        the rolling window. Duplicate metadata rows are rejected in Version 1 so
        every sequence maps to exactly one target row.

        Parameters
        ----------
        meta_df : pandas.DataFrame
            Metadata dataframe containing the configured join keys. For the
            Phase III contract these are `track_id`, `frame_index`, and
            `x_position_mm`.

        Returns
        -------
        bool
            `True` when validation succeeds.

        Raises
        ------
        TypeError
            If `meta_df` is not a pandas dataframe.
        ValueError
            If metadata is empty, missing required join keys, contains missing
            join-key values, contains duplicate join-key rows, or has join-key
            dtypes incompatible with the loaded dataset.
        """

        if not isinstance(meta_df, pd.DataFrame):
            raise TypeError("meta_df must be a pandas DataFrame.")
        if len(meta_df) == 0:
            raise ValueError("meta_df is empty.")

        missing = [col for col in self.join_keys if col not in meta_df.columns]
        if missing:
            raise ValueError(f"meta_df is missing required metadata columns: {missing}")

        missing_keys = meta_df[self.join_keys].isna().any(axis=1)
        if bool(missing_keys.any()):
            raise ValueError(f"meta_df contains {int(missing_keys.sum())} rows with missing join-key values.")

        duplicate_keys = meta_df.duplicated(self.join_keys, keep=False)
        if bool(duplicate_keys.any()):
            examples = meta_df.loc[duplicate_keys, self.join_keys].head(5).to_dict(orient="records")
            raise ValueError(f"Duplicate metadata rows are not allowed. Example duplicates: {examples}")

        for key in self.join_keys:
            if key not in self.dataset.columns:
                raise ValueError(f"Dataset is missing join key {key!r}.")
            try:
                if key == "x_position_mm":
                    pd.to_numeric(meta_df[key])
                    pd.to_numeric(self.dataset[key])
                else:
                    meta_df[key].astype(self.dataset[key].dtype, copy=False)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Join-key dtype is not compatible for column {key!r}.") from exc

        return True

    def _prepare_join_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for key in self.join_keys:
            if key == "track_id" or key == "frame_index":
                out[key] = pd.to_numeric(out[key], errors="raise").astype(int)
            elif key == "x_position_mm":
                out[key] = pd.to_numeric(out[key], errors="raise").astype(float)
        return out

    def _validate_target_values(self, aligned: pd.DataFrame, target_group: str, target_cols: Sequence[str]) -> None:
        missing_targets = aligned[list(target_cols)].isna().any(axis=1)
        if bool(missing_targets.any()):
            examples = aligned.loc[missing_targets, self.join_keys + list(target_cols)].head(5).to_dict(orient="records")
            raise ValueError(
                f"Aligned target group {target_group!r} contains {int(missing_targets.sum())} rows with missing target values. "
                f"Example rows: {examples}"
            )

        if target_group == "pca_shape" and "pca_ready" in aligned.columns:
            not_ready = ~aligned["pca_ready"].map(lambda value: bool(value) if pd.notna(value) else False)
            if bool(not_ready.any()):
                examples = aligned.loc[not_ready, self.join_keys + ["pca_ready"]].head(5).to_dict(orient="records")
                raise ValueError(f"pca_shape alignment requires pca_ready == True. Example invalid rows: {examples}")

    def align(
        self,
        meta_df: pd.DataFrame,
        target_group: str = "pca_shape",
        return_metadata: bool = True,
    ) -> Union[np.ndarray, Tuple[np.ndarray, pd.DataFrame]]:
        """Align target values to incoming feature-window metadata.

        This method performs an exact one-to-one merge using the configured join
        keys. It never uses nearest-neighbor matching, `merge_asof`, or
        interpolation. The order of `meta_df` is preserved exactly in both the
        returned NumPy array and optional aligned metadata dataframe.

        Parameters
        ----------
        meta_df : pandas.DataFrame
            Metadata produced by FeaturePreprocessor.create_sequence_windows.
            It must contain track_id, frame_index, and x_position_mm.
        target_group : str
            Built-in target group to align. Supported values are
            `pca_shape`, `amplitude`, and `signed_elevation`.
        return_metadata : bool
            If True, also return the aligned metadata and validity columns.

        Returns
        -------
        numpy.ndarray or tuple[numpy.ndarray, pandas.DataFrame]
            If `return_metadata` is False, returns only `Y`. If
            `return_metadata` is True, returns `(Y, aligned_meta)`.

            `Y` has shape `(n_samples, 5)` for `pca_shape` and
            `(n_samples, 1)` for scalar target groups. `aligned_meta` contains
            the join keys, requested target columns, and target-group-specific
            validity columns in the same row order as `Y`.

        Raises
        ------
        TypeError
            If `meta_df` is not a pandas dataframe.
        ValueError
            If metadata validation fails, `target_group` is unsupported, the
            exact merge does not match every metadata row exactly once, ordering
            is not preserved, requested target values are missing, or
            `pca_shape` rows are not `pca_ready`.
        """

        self.validate_meta(meta_df)
        target_cols = self._target_columns(target_group)
        validity_cols = self._validity_columns(target_group)

        meta = self._prepare_join_frame(meta_df[self.join_keys]).copy()
        meta["_input_order"] = np.arange(len(meta), dtype=int)

        dataset_cols = list(dict.fromkeys(self.join_keys + target_cols + validity_cols))
        dataset_subset = self._prepare_join_frame(self.dataset[dataset_cols]).copy()

        aligned = meta.merge(
            dataset_subset,
            on=self.join_keys,
            how="left",
            validate="one_to_one",
            indicator=True,
            sort=False,
        )

        if len(aligned) != len(meta):
            raise ValueError(f"Row count changed during target alignment: before={len(meta)}, after={len(aligned)}")

        unmatched = aligned[aligned["_merge"] != "both"]
        if not unmatched.empty:
            examples = unmatched[self.join_keys].head(5).to_dict(orient="records")
            raise ValueError(f"Every metadata row must match exactly one dataset row. Unmatched examples: {examples}")

        aligned = aligned.sort_values("_input_order").reset_index(drop=True)
        expected_order = np.arange(len(aligned), dtype=int)
        if not np.array_equal(aligned["_input_order"].to_numpy(dtype=int), expected_order):
            raise ValueError("Metadata ordering was not preserved during target alignment.")

        self._validate_target_values(aligned, target_group, target_cols)

        y = aligned[target_cols].to_numpy(dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        if not return_metadata:
            return y

        metadata_cols = list(dict.fromkeys(self.join_keys + target_cols + validity_cols))
        aligned_meta = aligned[metadata_cols].copy()
        return y, aligned_meta
