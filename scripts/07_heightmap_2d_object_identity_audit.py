"""07_heightmap_2d_object_identity_audit.py

Experiment 07A: 2D object-identity audit artifact generation.

This script is exploratory diagnostics only. It does not select a final
processed-track object and does not output a final processed-track width.

Repository guardrails:
- Do not create virtual environments.
- Do not inspect Track 21.
- Use Tracks 8, 10, and 14 with one shared configuration.
- Preserve NaNs and finite-support topology.
- Do not globally fill or interpolate missing profilometry.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import json
import math
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


# =========================
# Shared configuration
# =========================

SEALED_TRACKS = {21}
TRACK_IDS = [8, 10, 14]
assert not (set(TRACK_IDS) & SEALED_TRACKS), "Track 21 is sealed and must not be analyzed."

CENTRAL_EXCLUSION_Y_MIN_MM = 0.65
CENTRAL_EXCLUSION_Y_MAX_MM = 1.35
SUBSTRATE_EXCLUSION_Y_MM = (CENTRAL_EXCLUSION_Y_MIN_MM, CENTRAL_EXCLUSION_Y_MAX_MM)

MIN_SUBSTRATE_SAMPLES_PER_COLUMN = 50
MIN_SUBSTRATE_SPREAD_UM = 0.75

# Shared diagnostic saliency levels in normalized units. These levels are not
# selected as winners; all are inventoried to expose sensitivity.
SALIENCY_LEVELS = [2.0, 3.0, 4.0]
PRIMARY_OVERLAY_LEVEL = 3.0

# Shared neutral component filters to keep the inventory auditable rather than
# pixel-noise dominated. These are not track-specific.
MIN_COMPONENT_PIXELS = 150
MIN_COMPONENT_X_SPAN_MM = 0.20
CONNECTIVITY_2D = np.ones((3, 3), dtype=bool)

# Plotting only: cap x-resolution in figures; analysis uses native arrays.
MAX_PLOT_X_PIXELS = 1800


@dataclass(frozen=True)
class ChannelSpec:
    name: str
    display_name: str
    polarity: str


CHANNEL_SPECS = [
    ChannelSpec("signed_positive", "signed positive residual", "positive"),
    ChannelSpec("signed_negative", "signed negative residual", "negative"),
    ChannelSpec("absolute_residual", "absolute residual magnitude", "absolute"),
    ChannelSpec("gradient_activity", "y-gradient magnitude", "absolute"),
]


# =========================
# Numeric utilities
# =========================


def robust_mad(values: np.ndarray, scale: bool = True) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    med = float(np.nanmedian(v))
    mad = float(np.nanmedian(np.abs(v - med)))
    return float(1.4826 * mad if scale else mad)


def finite_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    idx = np.flatnonzero(np.asarray(mask, dtype=bool))
    if idx.size == 0:
        return []
    breaks = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[idx[0], idx[breaks + 1]]
    stops = np.r_[idx[breaks] + 1, idx[-1] + 1]
    return [(int(a), int(b)) for a, b in zip(starts, stops)]


def safe_percentile(values: np.ndarray, q: float, default: float = float("nan")) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return default
    return float(np.nanpercentile(v, q))


def contiguous_true_run_count(mask: np.ndarray) -> int:
    return len(finite_runs(np.asarray(mask, dtype=bool)))


def interval_iou_mm(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return float(inter / union) if union > 0 else 0.0


# =========================
# Detrending and fields
# =========================


def robust_plane_fit_substrate_focused(
    Z_mm: np.ndarray,
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    y_excl: Tuple[float, float],
    stride_x: int = 40,
    stride_y: int = 2,
    max_iter: int = 3,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, float]]:
    """Fit and subtract a plane using downsampled finite substrate points."""

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

    meta = {
        "stride_x": float(stride_x),
        "stride_y": float(stride_y),
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


def substrate_baseline_by_x(Z_det_mm: np.ndarray, y_mm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """Per-column substrate baseline/spread outside the central exclusion band.

    No z-values are filled. Only the 1D baseline/spread vectors receive an
    explicitly recorded fallback when a column has insufficient substrate support.
    """

    y0, y1 = SUBSTRATE_EXCLUSION_Y_MM
    outside = (y_mm < y0) | (y_mm > y1)
    Z_sub = Z_det_mm[outside, :]
    finite_sub = np.isfinite(Z_sub)

    baseline = np.full(Z_det_mm.shape[1], np.nan, dtype=float)
    spread = np.full(Z_det_mm.shape[1], np.nan, dtype=float)
    n_sub = finite_sub.sum(axis=0).astype(int)

    for j in range(Z_det_mm.shape[1]):
        if n_sub[j] < MIN_SUBSTRATE_SAMPLES_PER_COLUMN:
            continue
        vals = Z_sub[:, j]
        vals = vals[np.isfinite(vals)]
        med = float(np.nanmedian(vals))
        mad = robust_mad(vals, scale=True)
        baseline[j] = med
        spread[j] = max(float(mad), MIN_SUBSTRATE_SPREAD_UM * 1e-3)

    good = np.isfinite(baseline) & np.isfinite(spread)
    fallback_used = ~good
    global_vals = Z_sub[np.isfinite(Z_sub)]
    global_base = float(np.nanmedian(global_vals)) if global_vals.size else float("nan")
    global_spread = robust_mad(global_vals, scale=True) if global_vals.size else float("nan")
    if not np.isfinite(global_spread):
        global_spread = MIN_SUBSTRATE_SPREAD_UM * 1e-3
    global_spread = max(float(global_spread), MIN_SUBSTRATE_SPREAD_UM * 1e-3)

    baseline[fallback_used] = global_base
    spread[fallback_used] = global_spread

    meta = {
        "per_column_baseline_fraction": float(np.mean(good)),
        "fallback_column_fraction": float(np.mean(fallback_used)),
        "global_fallback_baseline_um": float(global_base * 1e3) if np.isfinite(global_base) else float("nan"),
        "global_fallback_spread_um": float(global_spread * 1e3),
        "min_substrate_samples_per_column": int(MIN_SUBSTRATE_SAMPLES_PER_COLUMN),
    }
    return baseline, spread, n_sub, meta


def gradient_magnitude_y_um_per_mm(Z_det_mm: np.ndarray, y_mm: np.ndarray) -> np.ndarray:
    """Compute |dz/dy| in finite y-runs for each x-column without filling NaNs."""

    grad = np.full_like(Z_det_mm, np.nan, dtype=float)
    for j in range(Z_det_mm.shape[1]):
        col = Z_det_mm[:, j]
        valid = np.isfinite(col)
        for a, b in finite_runs(valid):
            if (b - a) >= 5:
                grad[a:b, j] = np.gradient(col[a:b], y_mm[a:b]) * 1e3
    return np.abs(grad)


def build_fields(Z_det_mm: np.ndarray, y_mm: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    baseline, spread, n_sub, base_meta = substrate_baseline_by_x(Z_det_mm, y_mm)
    z_um = Z_det_mm * 1e3
    signed_resid_um = (Z_det_mm - baseline[None, :]) * 1e3
    spread_um = spread * 1e3
    abs_resid_um = np.abs(signed_resid_um)
    grad_um_per_mm = gradient_magnitude_y_um_per_mm(Z_det_mm, y_mm)
    finite_mask = np.isfinite(Z_det_mm)

    grad_scale = robust_mad(grad_um_per_mm[np.isfinite(grad_um_per_mm)], scale=True)
    if not np.isfinite(grad_scale) or grad_scale <= 0:
        grad_scale = 1.0

    fields = {
        "detrended_z_um": z_um,
        "signed_residual_um": signed_resid_um,
        "absolute_residual_um": abs_resid_um,
        "gradient_activity_um_per_mm": grad_um_per_mm,
        "finite_mask": finite_mask,
        "spread_um_by_x": spread_um,
        "baseline_um_by_x": baseline * 1e3,
        "n_substrate_by_x": n_sub.astype(float),
        "signed_positive_z": signed_resid_um / spread_um[None, :],
        "signed_negative_z": (-signed_resid_um) / spread_um[None, :],
        "absolute_residual_z": abs_resid_um / spread_um[None, :],
        "gradient_activity_z": grad_um_per_mm / grad_scale,
    }
    meta: Dict[str, object] = {**base_meta, "gradient_scale_um_per_mm": float(grad_scale)}
    return fields, meta


# =========================
# Structure inventory
# =========================


def channel_saliency_field(fields: Dict[str, np.ndarray], spec: ChannelSpec) -> np.ndarray:
    return np.asarray(fields[f"{spec.name}_z"], dtype=float)


def nan_contact_for_component(component_mask: np.ndarray, finite_mask: np.ndarray) -> Dict[str, object]:
    dilated = ndimage.binary_dilation(component_mask, structure=CONNECTIVITY_2D)
    ring = dilated & (~component_mask)
    touches_nan = bool(np.any(ring & (~finite_mask)))
    yy, xx = np.where(component_mask)
    touches_profile_edge = bool(
        yy.size > 0
        and (np.any(yy == 0) or np.any(yy == component_mask.shape[0] - 1) or np.any(xx == 0) or np.any(xx == component_mask.shape[1] - 1))
    )
    if touches_nan and touches_profile_edge:
        summary = "touches_nan_and_profile_edge"
    elif touches_nan:
        summary = "touches_nan_boundary"
    elif touches_profile_edge:
        summary = "touches_profile_edge"
    else:
        summary = "no_immediate_nan_contact"
    return {"touches_nan_boundary": touches_nan, "touches_profile_edge": touches_profile_edge, "summary": summary}


def objective_morphology_description(x_span_mm: float, y_span_mm: float, fragmentation_count: int, nan_summary: str) -> str:
    aspect = x_span_mm / max(y_span_mm, 1e-9)
    parts = []
    if x_span_mm >= 5.0 and aspect >= 10.0:
        parts.append("longitudinal_band_like")
    elif x_span_mm < 1.0 and y_span_mm < 0.25:
        parts.append("compact_patch")
    elif y_span_mm >= 0.75:
        parts.append("broad_y_extent")
    else:
        parts.append("intermediate_patch")
    if fragmentation_count > 1:
        parts.append("fragmented_x_support")
    if "nan" in nan_summary:
        parts.append("nan_adjacent")
    return ";".join(parts)


def component_records_for_channel(
    track_id: int,
    structure_counter_start: int,
    spec: ChannelSpec,
    level: float,
    fields: Dict[str, np.ndarray],
    x_mm: np.ndarray,
    y_mm: np.ndarray,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], int, np.ndarray]:
    sal = channel_saliency_field(fields, spec)
    finite_mask = np.asarray(fields["finite_mask"], dtype=bool)
    mask = finite_mask & np.isfinite(sal) & (sal >= level)

    labels, n_labels = ndimage.label(mask, structure=CONNECTIVITY_2D)
    objects = ndimage.find_objects(labels)

    inventory: List[Dict[str, object]] = []
    topology: List[Dict[str, object]] = []
    retained_mask = np.zeros(mask.shape, dtype=bool)
    structure_counter = structure_counter_start

    for label_value, slc in enumerate(objects, start=1):
        if slc is None:
            continue
        ys, xs = slc
        local = labels[ys, xs] == label_value
        pixel_count = int(np.sum(local))
        if pixel_count < MIN_COMPONENT_PIXELS:
            continue

        yy_local, xx_local = np.where(local)
        yy = yy_local + ys.start
        xx = xx_local + xs.start

        x_min = float(x_mm[int(np.min(xx))])
        x_max = float(x_mm[int(np.max(xx))])
        y_min = float(y_mm[int(np.min(yy))])
        y_max = float(y_mm[int(np.max(yy))])
        x_span = max(0.0, x_max - x_min)
        y_span = max(0.0, y_max - y_min)
        if x_span < MIN_COMPONENT_X_SPAN_MM:
            continue

        unique_x = np.unique(xx)
        x_support = np.zeros(int(np.max(xx) - np.min(xx) + 1), dtype=bool)
        x_support[unique_x - int(np.min(xx))] = True
        fragmentation_count = contiguous_true_run_count(x_support)
        possible_cols = max(1, int(np.max(xx) - np.min(xx) + 1))
        x_support_fraction = float(len(unique_x) / possible_cols)

        bbox_finite = finite_mask[ys, xs]
        finite_support_fraction = float(pixel_count / max(1, int(np.sum(bbox_finite))))

        component_mask = np.zeros(mask.shape, dtype=bool)
        component_mask[ys, xs] = local
        nan_contact = nan_contact_for_component(component_mask, finite_mask)
        retained_mask |= component_mask

        structure_id = f"structure_{structure_counter:04d}"
        structure_counter += 1
        morph_desc = objective_morphology_description(x_span, y_span, fragmentation_count, nan_contact["summary"])

        inventory.append(
            {
                "track_id": int(track_id),
                "structure_id": structure_id,
                "evidence_channel": spec.name,
                "saliency_level": float(level),
                "x_min_mm": x_min,
                "x_max_mm": x_max,
                "x_span_mm": float(x_span),
                "x_support_fraction": x_support_fraction,
                "y_min_mm": y_min,
                "y_max_mm": y_max,
                "median_y_mm": float(np.nanmedian(y_mm[yy])),
                "fragmentation_count": int(fragmentation_count),
                "finite_support_fraction": finite_support_fraction,
                "nan_contact_summary": str(nan_contact["summary"]),
                "morphology_description": morph_desc,
                "pixel_count": int(pixel_count),
                "label_value_internal": int(label_value),
            }
        )
        topology.append(
            {
                "track_id": int(track_id),
                "structure_id": structure_id,
                "evidence_channel": spec.name,
                "saliency_level": float(level),
                "touches_nan_boundary": bool(nan_contact["touches_nan_boundary"]),
                "touches_profile_edge": bool(nan_contact["touches_profile_edge"]),
                "nan_contact_summary": str(nan_contact["summary"]),
                "begins_near_finite_support_transition": bool(nan_contact["touches_nan_boundary"] or nan_contact["touches_profile_edge"]),
                "ends_near_finite_support_transition": bool(nan_contact["touches_nan_boundary"] or nan_contact["touches_profile_edge"]),
                "missingness_could_explain_apparent_termination": bool(nan_contact["touches_nan_boundary"] or nan_contact["touches_profile_edge"]),
                "bbox_finite_fraction": float(np.mean(bbox_finite)),
                "component_pixels": int(pixel_count),
            }
        )

    return inventory, topology, structure_counter, retained_mask


def audit_structures_for_track(
    track_id: int,
    fields: Dict[str, np.ndarray],
    x_mm: np.ndarray,
    y_mm: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[Tuple[str, float], np.ndarray]]:
    inventory_all: List[Dict[str, object]] = []
    topology_all: List[Dict[str, object]] = []
    overlay_masks: Dict[Tuple[str, float], np.ndarray] = {}
    counter = 1

    for spec in CHANNEL_SPECS:
        for level in SALIENCY_LEVELS:
            inv, topo, counter, retained_mask = component_records_for_channel(
                track_id=track_id,
                structure_counter_start=counter,
                spec=spec,
                level=float(level),
                fields=fields,
                x_mm=x_mm,
                y_mm=y_mm,
            )
            inventory_all.extend(inv)
            topology_all.extend(topo)
            overlay_masks[(spec.name, float(level))] = retained_mask

    return pd.DataFrame(inventory_all), pd.DataFrame(topology_all), overlay_masks


def build_channel_summary(inventory: pd.DataFrame) -> pd.DataFrame:
    if inventory.empty:
        return pd.DataFrame(
            columns=["track_id", "structure_id", "channel", "spatial_extent_summary", "shared_x_support", "overlap_metrics_if_meaningful"]
        )

    rows: List[Dict[str, object]] = []
    for _, row in inventory.iterrows():
        same_track = inventory[(inventory["track_id"] == row["track_id"]) & (inventory["structure_id"] != row["structure_id"])]
        other_channels = same_track[same_track["evidence_channel"] != row["evidence_channel"]]
        best = None
        best_score = -1.0
        for _, other in other_channels.iterrows():
            x_overlap = max(0.0, min(row["x_max_mm"], other["x_max_mm"]) - max(row["x_min_mm"], other["x_min_mm"]))
            y_overlap = max(0.0, min(row["y_max_mm"], other["y_max_mm"]) - max(row["y_min_mm"], other["y_min_mm"]))
            shared_x = x_overlap / max(row["x_span_mm"], other["x_span_mm"], 1e-9)
            bbox_iou = interval_iou_mm(row["x_min_mm"], row["x_max_mm"], other["x_min_mm"], other["x_max_mm"]) * interval_iou_mm(
                row["y_min_mm"], row["y_max_mm"], other["y_min_mm"], other["y_max_mm"]
            )
            score = shared_x + bbox_iou
            if score > best_score:
                best_score = score
                best = {
                    "other_structure_id": other["structure_id"],
                    "other_channel": other["evidence_channel"],
                    "other_saliency_level": float(other["saliency_level"]),
                    "shared_x_fraction_bbox": float(shared_x),
                    "x_overlap_mm": float(x_overlap),
                    "y_overlap_mm": float(y_overlap),
                    "bbox_overlap_index": float(bbox_iou),
                }
        extent = f"x=[{row['x_min_mm']:.3f},{row['x_max_mm']:.3f}] mm; y=[{row['y_min_mm']:.3f},{row['y_max_mm']:.3f}] mm"
        rows.append(
            {
                "track_id": int(row["track_id"]),
                "structure_id": row["structure_id"],
                "channel": row["evidence_channel"],
                "spatial_extent_summary": extent,
                "shared_x_support": json.dumps(best if best is not None else {}, sort_keys=True),
                "overlap_metrics_if_meaningful": json.dumps(best if best is not None else {}, sort_keys=True),
            }
        )
    return pd.DataFrame(rows)


# =========================
# Plotting
# =========================


def downsample_x_for_plot(arr: np.ndarray, x_mm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    step = max(1, int(math.ceil(len(x_mm) / MAX_PLOT_X_PIXELS)))
    return arr[:, ::step], x_mm[::step], step


def plot_field(
    out_path: Path,
    track_id: int,
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    field: np.ndarray,
    title: str,
    cmap_name: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    colorbar_label: str = "",
) -> None:
    arr, xp, _ = downsample_x_for_plot(np.asarray(field, dtype=float), x_mm)
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color="0.75")
    masked = np.ma.masked_invalid(arr)

    fig, ax = plt.subplots(figsize=(14, 4.2))
    im = ax.imshow(
        masked,
        origin="lower",
        aspect="auto",
        extent=[float(xp[0]), float(xp[-1]), float(y_mm[0]), float(y_mm[-1])],
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.axhspan(CENTRAL_EXCLUSION_Y_MIN_MM, CENTRAL_EXCLUSION_Y_MAX_MM, color="cyan", alpha=0.07, lw=0)
    ax.set_title(f"Track {track_id}: {title}")
    ax.set_xlabel("physical x (mm)")
    ax.set_ylabel("y (mm)")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.012)
    cb.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_topology(out_path: Path, track_id: int, x_mm: np.ndarray, y_mm: np.ndarray, finite_mask: np.ndarray) -> None:
    arr, xp, _ = downsample_x_for_plot(finite_mask.astype(float), x_mm)
    fig, ax = plt.subplots(figsize=(14, 4.2))
    im = ax.imshow(
        arr,
        origin="lower",
        aspect="auto",
        extent=[float(xp[0]), float(xp[-1]), float(y_mm[0]), float(y_mm[-1])],
        interpolation="nearest",
        cmap="gray_r",
        vmin=0,
        vmax=1,
    )
    ax.axhspan(CENTRAL_EXCLUSION_Y_MIN_MM, CENTRAL_EXCLUSION_Y_MAX_MM, color="cyan", alpha=0.07, lw=0)
    ax.set_title(f"Track {track_id}: finite-support / NaN topology (white=finite, black=NaN)")
    ax.set_xlabel("physical x (mm)")
    ax.set_ylabel("y (mm)")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.012)
    cb.set_label("finite support")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_overlay(
    out_path: Path,
    track_id: int,
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    background: np.ndarray,
    background_title: str,
    masks: List[Tuple[np.ndarray, str, str]],
    cmap_name: str,
    vmin: Optional[float],
    vmax: Optional[float],
    colorbar_label: str,
) -> None:
    bg, xp, step = downsample_x_for_plot(background, x_mm)
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color="0.75")
    fig, ax = plt.subplots(figsize=(14, 4.2))
    im = ax.imshow(
        np.ma.masked_invalid(bg),
        origin="lower",
        aspect="auto",
        extent=[float(xp[0]), float(xp[-1]), float(y_mm[0]), float(y_mm[-1])],
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    for mask, color, label in masks:
        if mask is None or not np.any(mask):
            continue
        md = mask[:, ::step].astype(float)
        ax.contour(xp, y_mm, md, levels=[0.5], colors=[color], linewidths=0.6)
        ax.plot([], [], color=color, lw=1.5, label=label)
    ax.axhspan(CENTRAL_EXCLUSION_Y_MIN_MM, CENTRAL_EXCLUSION_Y_MAX_MM, color="cyan", alpha=0.07, lw=0)
    ax.set_title(f"Track {track_id}: candidate structure overlay on {background_title}")
    ax.set_xlabel("physical x (mm)")
    ax.set_ylabel("y (mm)")
    if masks:
        ax.legend(loc="upper right", fontsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.012)
    cb.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def make_track_figures(
    fig_dir: Path,
    track_id: int,
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    fields: Dict[str, np.ndarray],
    overlay_masks: Dict[Tuple[str, float], np.ndarray],
) -> List[str]:
    made: List[str] = []
    z_lim = max(5.0, min(80.0, safe_percentile(np.abs(fields["detrended_z_um"]), 99, 20.0)))
    resid_lim = max(5.0, min(80.0, safe_percentile(np.abs(fields["signed_residual_um"]), 99, 20.0)))
    abs_lim = max(5.0, min(80.0, safe_percentile(fields["absolute_residual_um"], 99, 20.0)))
    grad_lim = max(50.0, min(1000.0, safe_percentile(fields["gradient_activity_um_per_mm"], 99, 200.0)))

    specs = [
        ("detrended_z", fields["detrended_z_um"], "detrended z(x,y)", "coolwarm", -z_lim, z_lim, "µm"),
        ("signed_residual", fields["signed_residual_um"], "signed residual relative to substrate baseline", "coolwarm", -resid_lim, resid_lim, "µm"),
        ("absolute_residual", fields["absolute_residual_um"], "absolute residual magnitude", "magma", 0.0, abs_lim, "µm"),
        ("gradient_activity", fields["gradient_activity_um_per_mm"], "y-gradient magnitude/activity", "viridis", 0.0, grad_lim, "µm/mm"),
    ]
    for stem, arr, title, cmap, vmin, vmax, cbar in specs:
        p = fig_dir / f"track_{track_id}_{stem}.png"
        plot_field(p, track_id, x_mm, y_mm, arr, title, cmap, vmin, vmax, cbar)
        made.append(str(p))

    p = fig_dir / f"track_{track_id}_nan_topology.png"
    plot_topology(p, track_id, x_mm, y_mm, fields["finite_mask"])
    made.append(str(p))

    level = float(PRIMARY_OVERLAY_LEVEL)
    p = fig_dir / f"track_{track_id}_overlay_signed_residual_structures.png"
    plot_overlay(
        p,
        track_id,
        x_mm,
        y_mm,
        fields["signed_residual_um"],
        f"signed residual (level={level:g})",
        [
            (overlay_masks.get(("signed_positive", level)), "red", "signed_positive"),
            (overlay_masks.get(("signed_negative", level)), "blue", "signed_negative"),
        ],
        "coolwarm",
        -resid_lim,
        resid_lim,
        "µm",
    )
    made.append(str(p))

    p = fig_dir / f"track_{track_id}_overlay_absolute_residual_structures.png"
    plot_overlay(
        p,
        track_id,
        x_mm,
        y_mm,
        fields["absolute_residual_um"],
        f"absolute residual (level={level:g})",
        [(overlay_masks.get(("absolute_residual", level)), "yellow", "absolute_residual")],
        "magma",
        0.0,
        abs_lim,
        "µm",
    )
    made.append(str(p))

    p = fig_dir / f"track_{track_id}_overlay_gradient_structures.png"
    plot_overlay(
        p,
        track_id,
        x_mm,
        y_mm,
        fields["gradient_activity_um_per_mm"],
        f"gradient field (level={level:g})",
        [(overlay_masks.get(("gradient_activity", level)), "cyan", "gradient_activity")],
        "viridis",
        0.0,
        grad_lim,
        "µm/mm",
    )
    made.append(str(p))
    return made


# =========================
# Main
# =========================


def main() -> None:
    height_dir = REPO_ROOT / "data" / "raw" / "height_maps"
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPO_ROOT / "processed_data" / "run_outputs" / f"07_heightmap_2d_object_identity_audit_{run_tag}"
    fig_dir = out_dir / "figures"
    table_dir = out_dir / "tables"
    diag_dir = out_dir / "diagnostics"
    for d in (fig_dir, table_dir, diag_dir):
        d.mkdir(parents=True, exist_ok=True)

    all_inventory: List[pd.DataFrame] = []
    all_topology: List[pd.DataFrame] = []
    field_summary_rows: List[Dict[str, object]] = []
    figures_made: List[str] = []
    detrend_meta: Dict[str, Dict[str, object]] = {}
    field_meta_by_track: Dict[str, Dict[str, object]] = {}

    for track_id in TRACK_IDS:
        hm = load_wyko_asc(height_dir, track_id, crop_to_common=True)
        x_mm = np.asarray(hm["x_actual_mm"], dtype=float)
        y_mm = np.asarray(hm["y_mm"], dtype=float)
        Z_mm = np.asarray(hm["Z_mm"], dtype=float)

        Z_det, _coef, dmeta = robust_plane_fit_substrate_focused(
            Z_mm,
            x_mm,
            y_mm,
            y_excl=SUBSTRATE_EXCLUSION_Y_MM,
            stride_x=40,
            stride_y=2,
            max_iter=3,
        )
        fields, fmeta = build_fields(Z_det, y_mm)
        inventory, topology, overlay_masks = audit_structures_for_track(track_id, fields, x_mm, y_mm)

        all_inventory.append(inventory)
        all_topology.append(topology)
        detrend_meta[str(track_id)] = dmeta
        field_meta_by_track[str(track_id)] = fmeta
        figures_made.extend(make_track_figures(fig_dir, track_id, x_mm, y_mm, fields, overlay_masks))

        finite = np.asarray(fields["finite_mask"], dtype=bool)
        field_summary_rows.append(
            {
                "track_id": int(track_id),
                "source_file": hm["file"],
                "n_y": int(len(y_mm)),
                "n_x": int(len(x_mm)),
                "x_min_mm": float(np.nanmin(x_mm)),
                "x_max_mm": float(np.nanmax(x_mm)),
                "y_min_mm": float(np.nanmin(y_mm)),
                "y_max_mm": float(np.nanmax(y_mm)),
                "finite_fraction": float(np.mean(finite)),
                "detrended_z_p01_um": safe_percentile(fields["detrended_z_um"], 1),
                "detrended_z_p50_um": safe_percentile(fields["detrended_z_um"], 50),
                "detrended_z_p99_um": safe_percentile(fields["detrended_z_um"], 99),
                "signed_residual_p01_um": safe_percentile(fields["signed_residual_um"], 1),
                "signed_residual_p50_um": safe_percentile(fields["signed_residual_um"], 50),
                "signed_residual_p99_um": safe_percentile(fields["signed_residual_um"], 99),
                "absolute_residual_p99_um": safe_percentile(fields["absolute_residual_um"], 99),
                "gradient_activity_p99_um_per_mm": safe_percentile(fields["gradient_activity_um_per_mm"], 99),
                **{f"field_meta_{k}": v for k, v in fmeta.items() if isinstance(v, (int, float, bool, str))},
            }
        )

    inventory_df = pd.concat(all_inventory, ignore_index=True) if all_inventory else pd.DataFrame()
    topology_df = pd.concat(all_topology, ignore_index=True) if all_topology else pd.DataFrame()
    channel_summary_df = build_channel_summary(inventory_df)
    field_summary_df = pd.DataFrame(field_summary_rows)

    inventory_path = table_dir / "structure_inventory.csv"
    channel_summary_path = table_dir / "structure_channel_summary.csv"
    topology_path = table_dir / "support_topology_summary.csv"
    field_summary_path = table_dir / "field_summary.csv"
    metadata_path = table_dir / "run_metadata.json"

    inventory_df.to_csv(inventory_path, index=False)
    channel_summary_df.to_csv(channel_summary_path, index=False)
    topology_df.to_csv(topology_path, index=False)
    field_summary_df.to_csv(field_summary_path, index=False)

    metadata = {
        "experiment": "07A_heightmap_2d_object_identity_audit",
        "out_dir": str(out_dir),
        "tracks": TRACK_IDS,
        "sealed_tracks": sorted(SEALED_TRACKS),
        "common_x_window_loader_crop": True,
        "uses_organizer_loader": "src/nsf_fmrg_data.py::load_wyko_asc",
        "central_exclusion_y_mm": [CENTRAL_EXCLUSION_Y_MIN_MM, CENTRAL_EXCLUSION_Y_MAX_MM],
        "saliency_levels": SALIENCY_LEVELS,
        "primary_overlay_level": PRIMARY_OVERLAY_LEVEL,
        "min_component_pixels": MIN_COMPONENT_PIXELS,
        "min_component_x_span_mm": MIN_COMPONENT_X_SPAN_MM,
        "channels": [asdict(s) for s in CHANNEL_SPECS],
        "detrend_meta_by_track": detrend_meta,
        "field_meta_by_track": field_meta_by_track,
        "outputs": {
            "figures": [str(Path(p).relative_to(out_dir)) for p in figures_made],
            "tables": [
                str(inventory_path.relative_to(out_dir)),
                str(channel_summary_path.relative_to(out_dir)),
                str(topology_path.relative_to(out_dir)),
                str(field_summary_path.relative_to(out_dir)),
                str(metadata_path.relative_to(out_dir)),
            ],
        },
        "notes": [
            "No Track 21 loading or inspection.",
            "No final processed-track object or width is selected.",
            "NaNs are preserved in fields and plotted as invalid/masked values.",
            "Candidate structures are neutral connected regions per evidence channel and saliency level.",
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Wrote Experiment 07A outputs to: {out_dir}")
    print(f"Figures: {len(figures_made)}")
    print(f"Inventory rows: {len(inventory_df)}")
    print(f"Topology rows: {len(topology_df)}")


if __name__ == "__main__":
    main()