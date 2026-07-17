"""Phase III PyTorch dataset utilities.

This module provides the final reusable data interface before baseline model
training. It packages already-preprocessed feature arrays, already-aligned target
arrays, and their metadata into a thin torch.utils.data.Dataset wrapper.

It does not perform feature preprocessing, target alignment, descriptor
recomputation, or model training.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class Phase3TorchDataset(Dataset):
    """Thin PyTorch Dataset wrapper for aligned Phase III samples.

    Parameters
    ----------
    X : array-like
        Preprocessed feature array. For sequence models this is expected to have
        shape `(n_samples, window_size, n_features)`.
    Y : array-like
        Aligned target array with first dimension equal to `n_samples`.
    meta_df : pandas.DataFrame
        Metadata aligned to `X` and `Y` in canonical sample order. It must
        contain `track_id`, `frame_index`, and `x_position_mm`.
    feature_dtype : torch.dtype, optional
        Torch dtype used for feature tensors.
    target_dtype : torch.dtype, optional
        Torch dtype used for target tensors.

    Notes
    -----
    This class assumes upstream modules have already performed feature
    preprocessing and target alignment. It validates the alignment contract and
    preserves metadata ordering exactly.
    """

    def __init__(
        self,
        X,
        Y,
        meta_df: pd.DataFrame,
        feature_dtype: torch.dtype = torch.float32,
        target_dtype: torch.dtype = torch.float32,
    ) -> None:
        # ==========================================
        # TASK 1: STORE ARRAYS AND METADATA
        # ==========================================
        self.required_metadata_cols: List[str] = ["track_id", "frame_index", "x_position_mm"]
        self.feature_dtype = feature_dtype
        self.target_dtype = target_dtype

        self.X = np.asarray(X)
        self.Y = np.asarray(Y)
        self.meta_df = meta_df.copy() if isinstance(meta_df, pd.DataFrame) else meta_df

        # ==========================================
        # TASK 2: VALIDATE THE SAMPLE CONTRACT
        # ==========================================
        self.validate()

    def validate(self) -> bool:
        """Validate feature, target, and metadata alignment.

        Returns
        -------
        bool
            `True` when validation succeeds.

        Raises
        ------
        TypeError
            If `meta_df` is not a pandas dataframe.
        ValueError
            If lengths disagree, metadata columns are missing, metadata contains
            missing join-key values, duplicate metadata rows are present, or
            feature/target arrays have insufficient dimensions.
        """

        if not isinstance(self.meta_df, pd.DataFrame):
            raise TypeError("meta_df must be a pandas DataFrame.")

        if self.X.ndim < 2:
            raise ValueError(f"X must have at least 2 dimensions; got shape {self.X.shape}.")
        if self.Y.ndim < 1:
            raise ValueError(f"Y must have at least 1 dimension; got shape {self.Y.shape}.")

        n_x = int(self.X.shape[0])
        n_y = int(self.Y.shape[0])
        n_meta = int(len(self.meta_df))

        if n_x != n_y:
            raise ValueError(f"len(X) must equal len(Y); got len(X)={n_x}, len(Y)={n_y}.")
        if n_y != n_meta:
            raise ValueError(f"len(Y) must equal len(meta_df); got len(Y)={n_y}, len(meta_df)={n_meta}.")
        if n_meta == 0:
            raise ValueError("Dataset metadata is empty.")

        missing_cols = [col for col in self.required_metadata_cols if col not in self.meta_df.columns]
        if missing_cols:
            raise ValueError(f"meta_df is missing required metadata columns: {missing_cols}")

        missing_keys = self.meta_df[self.required_metadata_cols].isna().any(axis=1)
        if bool(missing_keys.any()):
            raise ValueError(f"meta_df contains {int(missing_keys.sum())} rows with missing metadata keys.")

        duplicate_keys = self.meta_df.duplicated(self.required_metadata_cols, keep=False)
        if bool(duplicate_keys.any()):
            examples = self.meta_df.loc[duplicate_keys, self.required_metadata_cols].head(5).to_dict(orient="records")
            raise ValueError(f"Duplicate metadata rows are not allowed. Example duplicates: {examples}")

        return True

    def __len__(self) -> int:
        """Return the number of aligned samples."""

        return int(self.X.shape[0])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
        """Return one aligned feature/target sample and its metadata.

        Parameters
        ----------
        index : int
            Sample index in canonical metadata order.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, dict]
            Feature tensor, target tensor, and metadata dictionary for the same
            sample index.
        """

        x_tensor = torch.as_tensor(self.X[index], dtype=self.feature_dtype)
        y_tensor = torch.as_tensor(self.Y[index], dtype=self.target_dtype)
        metadata = self.meta_df.iloc[int(index)][self.required_metadata_cols].to_dict()
        return x_tensor, y_tensor, metadata
