"""05_heightmap_persistence_corridor_experiment.py

Exploratory experiment (NOT production) for NSF Future Manufacturing Data Challenge:

Goal
----
Reduce brittleness to a single binary threshold by:
1) generating candidate elevated components across multiple thresholds,
2) grouping them into cross-threshold "families" per x-location,
3) selecting a globally coherent x-direction corridor via dynamic programming
   (Viterbi-style) instead of greedy per-x selection.

Hard constraints (repo policy / user request)
-------------------------------------------
- Do NOT create virtual environments.
- Do NOT run / execute notebooks.
- Do NOT modify organizer-provided starter files.
- Do NOT load or analyze Track 21 (SEALED).
- Use only tracks: 8, 10, 14.
- Use organizer height-map loader from src/nsf_fmrg_data.py.
- Use substrate-focused detrending logic derived from prior notebook work.

Outputs
-------
Writes under:
  processed_data/run_outputs/05_heightmap_persistence_corridor_<timestamp>/

Produces:
- tables/*.csv, *.json
- figures/*.png
- diagnostics/*.png (per-location plots for most sensitive locations)

Execution
---------
/opt/homebrew/opt/python@3.11/bin/python3.11 scripts/05_heightmap_persistence_corridor_experiment.py

"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import json
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Allow running from repo root without installing as a package.
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from nsf_fmrg_data import load_wyko_asc  # organizer loader (do not modify)

warnings.filterwarnings("ignore", category=RuntimeWarning)


# =========================
# Configuration
# =========================

SEALED_TRACKS = {21}
TRACK_IDS = [8, 10, 14]
assert not (set(TRACK_IDS) & SEALED_TRACKS), "Track 21 is sealed and must not be analyzed."

# Representative x positions used in notebook 04
REQUESTED_X_MM = np.array([25, 35, 45, 55, 65, 75, 85, 95], dtype=float)

# Multi-threshold levels (shared across tracks)
THRESH_MULTIPLIERS_FULL = [1.5, 2.0, 2.5, 3.0, 3.5]
THRESHOLD_SETS: Dict[str, List[float]] = {
    "FULL": THRESH_MULTIPLIERS_FULL,
    "REMOVE_LOW": [2.0, 2.5, 3.0, 3.5],
    "REMOVE_HIGH": [1.5, 2.0, 2.5, 3.0],
    "INNER": [2.0, 2.5, 3.0],
}

# Coarse analysis x grid.
# Choose 0.2 mm so that:
# - it is ~50x coarser than native (~0.004 mm), avoiding per-column solving,
# - still resolves gradual width/boundary trends without overwhelming compute.
ANALYSIS_X_STEP_MM = 0.2

# Local x window half-width for robust aggregation.
# With 0.2 mm grid, using +/- 0.20 mm includes ~100 native columns (0.40 / 0.004).
# This gives robust stats while remaining local in x.
X_WINDOW_HALF_WIDTH_MM = 0.20

# Detrending: substrate-focused plane fit excluding a central y band.
CENTRAL_EXCLUSION_Y_MIN_MM = 0.65
CENTRAL_EXCLUSION_Y_MAX_MM = 1.35

# Robust baseline/spread computed from "substrate" samples outside exclusion.
MIN_BASELINE_SAMPLES = 200
MIN_SPREAD_UM = 0.75  # same floor as notebook 03/04 config

# Component extraction constraints
MIN_COMPONENT_SAMPLES = 8
MIN_COMPONENT_WIDTH_MM = 0.035
MAX_COMPONENT_WIDTH_MM = 1.10

# Family building (within a given x)
FAMILY_IOU_MIN = 0.10
FAMILY_CONTAINMENT_BONUS = 0.20
FAMILY_CENTROID_DIST_MAX_MM = 0.25

# Corridor scoring (node evidence)
NODE_PERSISTENCE_WEIGHT = 2.0
NODE_HEIGHT_MEDIAN_WEIGHT = 0.7
NODE_HEIGHT_PEAK_WEIGHT = 0.3
NODE_STABILITY_WEIGHT = 1.0
NODE_COVERAGE_WEIGHT = 0.8

# Corridor scoring (transition costs)
TRANSITION_CENTROID_L1_WEIGHT = 2.0
TRANSITION_LEFT_L1_WEIGHT = 1.0
TRANSITION_RIGHT_L1_WEIGHT = 1.0
TRANSITION_WIDTH_L1_WEIGHT = 0.5

# Gap handling
MAX_GAP_STEPS = 5  # allow skipping up to this many x grid steps
GAP_STEP_PENALTY = 1.5
GAP_OPEN_PENALTY = 1.0

# Evidence gating
MIN_PERSISTENCE_FRACTION_FOR_NODE = 0.5
MIN_FINITE_FRACTION_FOR_NODE = 0.10

# Agreement threshold (CONTROL vs corridor)
AGREE_CENTER_TOL_MM = 0.10
AGREE_WIDTH_REL_TOL = 0.20


# =========================
# Utilities
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


def interval_iou(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    a0, a1 = a
    b0, b1 = b
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    if union <= 0:
        return 0.0
    return inter / union


def interval_contains(outer: Tuple[float, float], inner: Tuple[float, float]) -> bool:
    return outer[0] <= inner[0] and inner[1] <= outer[1]


def safe_log1p(x: float) -> float:
    return float(np.log1p(max(0.0, x)))


# =========================
# Detrending (substrate-focused)
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
    """Fit and subtract a plane using downsampled points, excluding the central y band.

    Mirrors the notebook-03/04 approach:
    - Use a sparse grid (stride_x, stride_y)
    - Exclude y in [y_excl[0], y_excl[1]] to reduce track-region leakage
    - Robustify with percentile trimming iterations

    Returns:
      Z_det_mm, coef (ax, ay, c), metadata
    """

    y0, y1 = y_excl
    xs = x_mm[::stride_x]
    ys = y_mm[::stride_y]
    Zs = Z_mm[::stride_y, ::stride_x]

    Xs, Ys = np.meshgrid(xs, ys)
    z = Zs.ravel()
    A = np.c_[Xs.ravel(), Ys.ravel(), np.ones(Xs.size)]

    valid = np.isfinite(z)
    ymask = (Ys.ravel() < y0) | (Ys.ravel() > y1)
    keep = valid & ymask

    meta = {
        "stride_x": float(stride_x),
        "stride_y": float(stride_y),
        "excluded_y_min_mm": float(y0),
        "excluded_y_max_mm": float(y1),
        "fit_samples_used": float(np.sum(keep)),
    }

    if np.sum(keep) < 100:
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

    plane = coef[0] * x_mm[None, :] + coef[1] * y_mm[:, None] + coef[2]
    Z_det = Z_mm - plane

    meta.update({"coef_x": float(coef[0]), "coef_y": float(coef[1]), "coef_c": float(coef[2])})
    return Z_det, np.asarray(coef, dtype=float), meta


# =========================
# Local x-window aggregation
# =========================


def aggregate_profile_in_x_window(
    Z_det_mm: np.ndarray,
    x_actual_mm: np.ndarray,
    y_mm: np.ndarray,
    x0_mm: float,
    half_width_mm: float,
) -> Dict[str, np.ndarray]:
    """Aggregate a robust 1D profile z(y) at analysis location x0.

    - Select native columns within [x0-half_width, x0+half_width]
    - For each y-row, robustly aggregate across x using median over finite samples
    - Preserve missingness: output finite counts and fractions; no NaN filling.
    """

    x_mask = (x_actual_mm >= x0_mm - half_width_mm) & (x_actual_mm <= x0_mm + half_width_mm)
    cols = np.flatnonzero(x_mask)

    if cols.size == 0:
        z_med = np.full_like(y_mm, np.nan, dtype=float)
        n_finite = np.zeros_like(y_mm, dtype=int)
        frac_finite = np.zeros_like(y_mm, dtype=float)
        return {
            "cols": cols,
            "z_med_mm": z_med,
            "n_finite": n_finite,
            "frac_finite": frac_finite,
        }

    Z_win = Z_det_mm[:, cols]  # shape (Ny, Ncols)
    finite = np.isfinite(Z_win)
    n_finite = np.sum(finite, axis=1).astype(int)
    frac_finite = n_finite / float(cols.size)

    z_med = np.nanmedian(Z_win, axis=1)

    return {
        "cols": cols,
        "z_med_mm": z_med.astype(float),
        "n_finite": n_finite,
        "frac_finite": frac_finite.astype(float),
    }


# =========================
# Component candidates per threshold
# =========================


@dataclass
class Component:
    thr_mult: float
    y_min: float
    y_max: float
    centroid: float
    width: float
    n_finite: int
    median_above_um: float
    peak_above_um: float


def compute_baseline_and_spread_um(
    y_mm: np.ndarray,
    z_med_mm: np.ndarray,
    y_excl: Tuple[float, float],
    min_samples: int,
    min_spread_um: float,
) -> Tuple[float, float, str, int]:
    """Compute local baseline and robust spread from substrate region outside central exclusion band."""

    y0, y1 = y_excl
    finite = np.isfinite(z_med_mm)
    substrate_mask = finite & ((y_mm < y0) | (y_mm > y1))

    vals_um = z_med_mm[substrate_mask] * 1e3
    n = int(vals_um.size)
    if n >= min_samples:
        baseline = float(np.nanmedian(vals_um))
        spread = float(max(robust_mad(vals_um), min_spread_um))
        return baseline, spread, "substrate_outside_exclusion", n

    # Fallback: all finite values.
    vals_um_all = z_med_mm[finite] * 1e3
    n_all = int(vals_um_all.size)
    if n_all == 0:
        return float("nan"), float("nan"), "no_finite", 0

    baseline = float(np.nanmedian(vals_um_all))
    spread = float(max(robust_mad(vals_um_all), min_spread_um))
    return baseline, spread, "all_finite_fallback", n_all


def candidate_components_for_threshold(
    y_mm: np.ndarray,
    z_med_mm: np.ndarray,
    baseline_um: float,
    spread_um: float,
    thr_mult: float,
) -> List[Component]:
    """Extract finite-run-preserving y-components for one threshold."""

    if not np.isfinite(baseline_um) or not np.isfinite(spread_um):
        return []

    z_um = z_med_mm * 1e3
    above = (z_um - baseline_um) > (thr_mult * spread_um)
    finite = np.isfinite(z_um)
    mask = above & finite

    comps: List[Component] = []
    for a, b in finite_runs(mask):
        if (b - a) < MIN_COMPONENT_SAMPLES:
            continue
        y0 = float(y_mm[a])
        y1 = float(y_mm[b - 1])
        width = float(y1 - y0)
        if width < MIN_COMPONENT_WIDTH_MM or width > MAX_COMPONENT_WIDTH_MM:
            continue

        seg = (slice(a, b),)
        z_seg = z_um[seg]
        z_above = z_seg - baseline_um

        nfin = int(np.sum(np.isfinite(z_seg)))
        if nfin < MIN_COMPONENT_SAMPLES:
            continue

        # centroid: use y weighted by positive height above baseline where finite
        w = np.clip(z_above, 0.0, None)
        wsum = float(np.nansum(w))
        if wsum > 0:
            centroid = float(np.nansum(y_mm[a:b] * w) / wsum)
        else:
            centroid = float(0.5 * (y0 + y1))

        comps.append(
            Component(
                thr_mult=float(thr_mult),
                y_min=y0,
                y_max=y1,
                centroid=centroid,
                width=float(y1 - y0),
                n_finite=nfin,
                median_above_um=float(np.nanmedian(z_above[np.isfinite(z_above)])) if nfin else float("nan"),
                peak_above_um=float(np.nanmax(z_above[np.isfinite(z_above)])) if nfin else float("nan"),
            )
        )

    return comps


# =========================
# Cross-threshold families (per x)
# =========================


@dataclass
class Family:
    family_id: int
    x_mm: float
    components: List[Component]
    thresholds_present: List[float]

    # representative interval and summary statistics
    rep_y_min: float
    rep_y_max: float
    rep_centroid: float
    rep_width: float

    persistence_fraction: float
    thr_low: float
    thr_high: float

    centroid_spread: float
    width_spread: float
    median_height_um: float
    peak_height_um: float


def build_families_for_x(
    x_mm: float,
    comps_by_thr: Dict[float, List[Component]],
    thresh_list: Sequence[float],
) -> List[Family]:
    """Associate components across thresholds by y-interval overlap and containment.

    Matching rule (transparent):
    - Build all components sorted by descending threshold (high → low)
    - Seed families from high-threshold components
    - For each lower-threshold component, match to best existing family if:
        * IoU >= FAMILY_IOU_MIN OR containment holds (with bonus)
        * centroid distance <= FAMILY_CENTROID_DIST_MAX_MM
      else start a new family.

    Representative interval:
    - use component at the median threshold (by multiplier) within the family;
      if missing, use the component with max persistence-weighted peak height.
    """

    all_comps: List[Component] = []
    for t in sorted(thresh_list, reverse=True):
        all_comps.extend(comps_by_thr.get(float(t), []))

    families: List[List[Component]] = []

    def comp_interval(c: Component) -> Tuple[float, float]:
        return (c.y_min, c.y_max)

    for c in all_comps:
        best_j = None
        best_score = -1e9
        for j, fam in enumerate(families):
            # compare against the closest-threshold component already in family if possible
            # (but for simplicity, compare against family representative proxy: last added)
            ref = fam[-1]
            iou = interval_iou(comp_interval(ref), comp_interval(c))
            contain = interval_contains(comp_interval(ref), comp_interval(c)) or interval_contains(
                comp_interval(c), comp_interval(ref)
            )
            centroid_dist = abs(ref.centroid - c.centroid)
            if centroid_dist > FAMILY_CENTROID_DIST_MAX_MM:
                continue
            if (iou < FAMILY_IOU_MIN) and (not contain):
                continue
            score = iou + (FAMILY_CONTAINMENT_BONUS if contain else 0.0) - 0.5 * (centroid_dist / FAMILY_CENTROID_DIST_MAX_MM)
            if score > best_score:
                best_score = score
                best_j = j
        if best_j is None:
            families.append([c])
        else:
            families[best_j].append(c)

    out: List[Family] = []
    for fid, fam in enumerate(families):
        thr_present = sorted({float(c.thr_mult) for c in fam})
        persistence = len(thr_present) / float(len(thresh_list)) if len(thresh_list) else 0.0

        # pick representative component
        thr_sorted = sorted(thresh_list)
        mid_thr = thr_sorted[len(thr_sorted) // 2]
        rep = None
        for c in fam:
            if float(c.thr_mult) == float(mid_thr):
                rep = c
                break
        if rep is None:
            rep = max(fam, key=lambda cc: (cc.peak_above_um, cc.median_above_um))

        centroids = np.array([c.centroid for c in fam], dtype=float)
        widths = np.array([c.width for c in fam], dtype=float)
        centroid_spread = float(np.nanmax(centroids) - np.nanmin(centroids)) if centroids.size else float("nan")
        width_spread = float(np.nanmax(widths) - np.nanmin(widths)) if widths.size else float("nan")

        med_heights = np.array([c.median_above_um for c in fam], dtype=float)
        peak_heights = np.array([c.peak_above_um for c in fam], dtype=float)
        median_height = float(np.nanmedian(med_heights)) if med_heights.size else float("nan")
        peak_height = float(np.nanmax(peak_heights)) if peak_heights.size else float("nan")

        out.append(
            Family(
                family_id=int(fid),
                x_mm=float(x_mm),
                components=list(fam),
                thresholds_present=thr_present,
                rep_y_min=float(rep.y_min),
                rep_y_max=float(rep.y_max),
                rep_centroid=float(rep.centroid),
                rep_width=float(rep.width),
                persistence_fraction=float(persistence),
                thr_low=float(min(thr_present)) if thr_present else float("nan"),
                thr_high=float(max(thr_present)) if thr_present else float("nan"),
                centroid_spread=float(centroid_spread),
                width_spread=float(width_spread),
                median_height_um=float(median_height),
                peak_height_um=float(peak_height),
            )
        )

    return out


# =========================
# Corridor optimization (dynamic programming)
# =========================


@dataclass
class Node:
    x_idx: int
    x_mm: float
    family: Family
    finite_fraction: float
    node_score: float


def compute_node_score(f: Family, finite_fraction: float) -> float:
    """Evidence score for a node (family at x).

    No strong center/width priors.
    Emphasis:
    - persistence across thresholds
    - height evidence (median + peak)
    - stability (low centroid/width spread)
    - finite coverage
    """

    persistence = f.persistence_fraction
    height_med = safe_log1p(f.median_height_um)
    height_peak = safe_log1p(f.peak_height_um)

    # stability: penalize large spreads (mm). Using exp(-k*spread).
    k = 6.0
    stab = math.exp(-k * max(0.0, f.centroid_spread)) * math.exp(-k * max(0.0, f.width_spread))

    cov = max(0.0, min(1.0, finite_fraction))

    return (
        NODE_PERSISTENCE_WEIGHT * persistence
        + NODE_HEIGHT_MEDIAN_WEIGHT * height_med
        + NODE_HEIGHT_PEAK_WEIGHT * height_peak
        + NODE_STABILITY_WEIGHT * stab
        + NODE_COVERAGE_WEIGHT * cov
    )


def transition_cost(a: Node, b: Node) -> float:
    dyc = abs(a.family.rep_centroid - b.family.rep_centroid)
    dyl = abs(a.family.rep_y_min - b.family.rep_y_min)
    dyr = abs(a.family.rep_y_max - b.family.rep_y_max)
    dw = abs(a.family.rep_width - b.family.rep_width)

    return (
        TRANSITION_CENTROID_L1_WEIGHT * dyc
        + TRANSITION_LEFT_L1_WEIGHT * dyl
        + TRANSITION_RIGHT_L1_WEIGHT * dyr
        + TRANSITION_WIDTH_L1_WEIGHT * dw
    )


def solve_corridor_dp(
    nodes_by_x: List[List[Node]],
    x_mm: np.ndarray,
) -> Tuple[List[Optional[Node]], float]:
    """Viterbi-style best path with optional gaps.

    We allow "missing" positions by not selecting any node at that x.
    Gaps are represented implicitly by transitions that skip up to MAX_GAP_STEPS.
    """

    nX = len(nodes_by_x)

    # DP state: best score ending at a given node at x.
    best_score: List[Dict[int, float]] = [dict() for _ in range(nX)]
    backptr: List[Dict[int, Tuple[int, int]]] = [dict() for _ in range(nX)]
    # maps node_index -> (prev_x, prev_node_index)

    for xi in range(nX):
        nodes = nodes_by_x[xi]
        for ni, node in enumerate(nodes):
            # start a new path here
            best = node.node_score
            best_prev = None

            # connect from previous nodes within gap
            for xj in range(max(0, xi - MAX_GAP_STEPS), xi):
                gap_steps = xi - xj
                if not best_score[xj]:
                    continue
                for nj, prev_val in best_score[xj].items():
                    prev_node = nodes_by_x[xj][nj]
                    cost = transition_cost(prev_node, node)
                    gap_pen = GAP_OPEN_PENALTY + GAP_STEP_PENALTY * (gap_steps - 1)
                    cand = prev_val + node.node_score - cost - gap_pen
                    if cand > best:
                        best = cand
                        best_prev = (xj, nj)

            best_score[xi][ni] = best
            if best_prev is not None:
                backptr[xi][ni] = best_prev

    # best terminal node
    end_x = None
    end_n = None
    end_score = -1e18
    for xi in range(nX):
        for ni, val in best_score[xi].items():
            if val > end_score:
                end_score = val
                end_x, end_n = xi, ni

    path: List[Optional[Node]] = [None for _ in range(nX)]
    if end_x is None:
        return path, float("nan")

    xi, ni = end_x, end_n
    while True:
        path[xi] = nodes_by_x[xi][ni]
        if ni not in backptr[xi]:
            break
        xi, ni = backptr[xi][ni]

    return path, float(end_score)


# =========================
# CONTROL artifacts (Notebook 04)
# =========================


def load_control_outputs(project_dir: Path) -> pd.DataFrame:
    """Load notebook-04 CONTROL outputs from saved artifacts (no rerun)."""

    # Hard-code the known run directory (from provided workspace tree)
    nb04_dir = project_dir / "processed_data" / "run_outputs" / "04_heightmap_prior_sensitivity_20260713_194307"
    p = nb04_dir / "tables" / "control_outputs.csv"
    if not p.exists():
        raise FileNotFoundError(f"CONTROL artifact not found: {p}")
    df = pd.read_csv(p)
    # ensure consistent dtypes
    df["track_id"] = df["track_id"].astype(int)
    df["requested_x_mm"] = df["requested_x_mm"].astype(float)
    return df


# =========================
# Plotting helpers
# =========================


def plot_track_overview(
    out_path: Path,
    track_id: int,
    x_grid: np.ndarray,
    y_mm: np.ndarray,
    finite_frac_yx: np.ndarray,
    families_by_x: List[List[Family]],
    corridor_path: List[Optional[Node]],
    title: str,
) -> None:
    """Per-track x-y overview:
    - background: finite coverage fraction (aggregated profile)
    - all family representative intervals
    - selected corridor intervals
    """

    fig, ax = plt.subplots(figsize=(12, 4))

    # Background: finite coverage fraction per (y,x) for aggregated profile.
    im = ax.imshow(
        finite_frac_yx,
        aspect="auto",
        origin="lower",
        extent=[x_grid[0], x_grid[-1], y_mm[0], y_mm[-1]],
        cmap="Greys",
        vmin=0.0,
        vmax=1.0,
        alpha=0.35,
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("finite fraction in x-window")

    # All families
    for xi, fams in enumerate(families_by_x):
        x = x_grid[xi]
        for fam in fams:
            ax.plot([x, x], [fam.rep_y_min, fam.rep_y_max], color="tab:blue", alpha=0.25, linewidth=2)

    # Selected corridor
    for node in corridor_path:
        if node is None:
            continue
        x = node.x_mm
        ax.plot([x, x], [node.family.rep_y_min, node.family.rep_y_max], color="tab:red", alpha=0.9, linewidth=3)

    ax.set_title(title)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_series_with_gaps(out_path: Path, x: np.ndarray, y: np.ndarray, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(x, y, color="tab:red", linewidth=1)
    ax.scatter(x[np.isfinite(y)], y[np.isfinite(y)], s=8, color="tab:red")
    ax.set_title(title)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_sensitive_location(
    out_path: Path,
    y_mm: np.ndarray,
    z_med_um: np.ndarray,
    baseline_um: float,
    spread_um: float,
    comps_by_thr: Dict[float, List[Component]],
    families: List[Family],
    chosen_family_ids: Dict[str, Optional[int]],
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(y_mm, z_med_um, color="k", linewidth=1, label="aggregated profile (median over x-window)")
    if np.isfinite(baseline_um):
        ax.axhline(baseline_um, color="gray", linestyle="--", linewidth=1, label="baseline")
    if np.isfinite(spread_um):
        for t in sorted(comps_by_thr.keys()):
            ax.axhline(baseline_um + t * spread_um, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

    # components
    colors = {1.5: "#1f77b4", 2.0: "#2ca02c", 2.5: "#ff7f0e", 3.0: "#d62728", 3.5: "#9467bd"}
    for t, comps in comps_by_thr.items():
        for c in comps:
            ax.axvspan(c.y_min, c.y_max, color=colors.get(float(t), "tab:blue"), alpha=0.08)

    # families rep intervals
    for fam in families:
        ax.axvspan(fam.rep_y_min, fam.rep_y_max, color="tab:blue", alpha=0.10)

    # chosen families per threshold set
    for set_name, fid in chosen_family_ids.items():
        if fid is None:
            continue
        fam = next((f for f in families if f.family_id == fid), None)
        if fam is None:
            continue
        ax.plot([fam.rep_centroid], [baseline_um + fam.peak_height_um], marker="o", label=f"chosen ({set_name})")

    ax.set_title(title)
    ax.set_xlabel("y (mm)")
    ax.set_ylabel("height (µm) relative to plane")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# =========================
# Main experiment per track & threshold set
# =========================


def run_for_track_and_thresholds(
    project_dir: Path,
    out_dir: Path,
    track_id: int,
    thresh_list: Sequence[float],
    tag: str,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Run the corridor method for one track and a given threshold set."""

    height_dir = project_dir / "data" / "raw" / "height_maps"
    data = load_wyko_asc(height_dir, track_id, crop_to_common=True)
    Z_mm = data["Z_mm"]
    x_actual = data["x_actual_mm"]
    y_mm = data["y_mm"]

    # Detrend
    Z_det, coef, det_meta = robust_plane_fit_substrate_focused(
        Z_mm,
        x_actual,
        y_mm,
        y_excl=(CENTRAL_EXCLUSION_Y_MIN_MM, CENTRAL_EXCLUSION_Y_MAX_MM),
    )

    # Coarse x grid within available range
    x0 = float(np.nanmin(x_actual))
    x1 = float(np.nanmax(x_actual))
    x_grid = np.arange(x0, x1 + 1e-9, ANALYSIS_X_STEP_MM, dtype=float)

    # Precompute aggregated profiles and finite coverage matrices
    z_med_mm_yx = np.empty((len(y_mm), len(x_grid)), dtype=float)
    finite_frac_yx = np.empty((len(y_mm), len(x_grid)), dtype=float)
    finite_overall = np.empty(len(x_grid), dtype=float)

    agg_cache: List[Dict[str, np.ndarray]] = []
    for i, xg in enumerate(x_grid):
        agg = aggregate_profile_in_x_window(Z_det, x_actual, y_mm, xg, X_WINDOW_HALF_WIDTH_MM)
        agg_cache.append(agg)
        z_med_mm_yx[:, i] = agg["z_med_mm"]
        finite_frac_yx[:, i] = agg["frac_finite"]
        finite_overall[i] = float(np.mean(agg["frac_finite"]))

    # Candidate components + families per x
    families_by_x: List[List[Family]] = []
    nodes_by_x: List[List[Node]] = []

    baseline_um_list = np.empty(len(x_grid), dtype=float)
    spread_um_list = np.empty(len(x_grid), dtype=float)
    baseline_src_list: List[str] = []
    baseline_n_list = np.empty(len(x_grid), dtype=int)

    for xi, xg in enumerate(x_grid):
        z_med_mm = agg_cache[xi]["z_med_mm"]

        baseline_um, spread_um, baseline_src, baseline_n = compute_baseline_and_spread_um(
            y_mm,
            z_med_mm,
            y_excl=(CENTRAL_EXCLUSION_Y_MIN_MM, CENTRAL_EXCLUSION_Y_MAX_MM),
            min_samples=MIN_BASELINE_SAMPLES,
            min_spread_um=MIN_SPREAD_UM,
        )
        baseline_um_list[xi] = baseline_um
        spread_um_list[xi] = spread_um
        baseline_src_list.append(baseline_src)
        baseline_n_list[xi] = baseline_n

        comps_by_thr: Dict[float, List[Component]] = {}
        for t in thresh_list:
            comps_by_thr[float(t)] = candidate_components_for_threshold(y_mm, z_med_mm, baseline_um, spread_um, float(t))

        families = build_families_for_x(float(xg), comps_by_thr, thresh_list)
        families_by_x.append(families)

        nodes: List[Node] = []
        for fam in families:
            ff = float(finite_overall[xi])
            # gate nodes
            if fam.persistence_fraction < MIN_PERSISTENCE_FRACTION_FOR_NODE:
                continue
            if ff < MIN_FINITE_FRACTION_FOR_NODE:
                continue
            score = compute_node_score(fam, finite_fraction=ff)
            nodes.append(Node(x_idx=int(xi), x_mm=float(xg), family=fam, finite_fraction=ff, node_score=float(score)))

        nodes_by_x.append(nodes)

    # Solve corridor
    corridor_path, path_score = solve_corridor_dp(nodes_by_x, x_grid)

    # Create per-x output table
    rows = []
    for xi, xg in enumerate(x_grid):
        node = corridor_path[xi]
        fams = families_by_x[xi]
        if node is None:
            rows.append(
                {
                    "track_id": int(track_id),
                    "x_mm": float(xg),
                    "left_boundary_mm": float("nan"),
                    "right_boundary_mm": float("nan"),
                    "width_mm": float("nan"),
                    "persistence_fraction": float("nan"),
                    "centroid_threshold_spread_mm": float("nan"),
                    "width_threshold_spread_mm": float("nan"),
                    "finite_fraction": float(finite_overall[xi]),
                    "node_score": float("nan"),
                    "path_score": float(path_score),
                    "n_competing_families": int(len(fams)),
                    "corridor_valid": False,
                    "threshold_set": str(tag),
                }
            )
        else:
            f = node.family
            rows.append(
                {
                    "track_id": int(track_id),
                    "x_mm": float(xg),
                    "left_boundary_mm": float(f.rep_y_min),
                    "right_boundary_mm": float(f.rep_y_max),
                    "width_mm": float(f.rep_width),
                    "persistence_fraction": float(f.persistence_fraction),
                    "centroid_threshold_spread_mm": float(f.centroid_spread),
                    "width_threshold_spread_mm": float(f.width_spread),
                    "finite_fraction": float(node.finite_fraction),
                    "node_score": float(node.node_score),
                    "path_score": float(path_score),
                    "n_competing_families": int(len(fams)),
                    "corridor_valid": True,
                    "threshold_set": str(tag),
                }
            )

    out_df = pd.DataFrame(rows)

    # Diagnostics plots
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_track_overview(
        fig_dir / f"track_{track_id}_{tag}_A_overview.png",
        track_id,
        x_grid,
        y_mm,
        finite_frac_yx,
        families_by_x,
        corridor_path,
        title=f"Track {track_id} | threshold-set {tag} | families + corridor (background=finite fraction)",
    )

    plot_series_with_gaps(
        fig_dir / f"track_{track_id}_{tag}_B_width_vs_x.png",
        x_grid,
        out_df["width_mm"].to_numpy(float),
        title=f"Track {track_id} | threshold-set {tag} | corridor width vs x",
        ylabel="width (mm)",
    )

    plot_series_with_gaps(
        fig_dir / f"track_{track_id}_{tag}_C_persistence_vs_x.png",
        x_grid,
        out_df["persistence_fraction"].to_numpy(float),
        title=f"Track {track_id} | threshold-set {tag} | persistence fraction vs x",
        ylabel="persistence fraction",
    )

    meta = {
        "track_id": int(track_id),
        "threshold_set": str(tag),
        "thresh_list": [float(t) for t in thresh_list],
        "analysis_x_step_mm": float(ANALYSIS_X_STEP_MM),
        "x_window_half_width_mm": float(X_WINDOW_HALF_WIDTH_MM),
        "central_exclusion_y": [float(CENTRAL_EXCLUSION_Y_MIN_MM), float(CENTRAL_EXCLUSION_Y_MAX_MM)],
        "detrend_meta": det_meta,
        "path_score": float(path_score),
        "n_x": int(len(x_grid)),
        "finite_overall_summary": {
            "min": float(np.nanmin(finite_overall)),
            "p05": float(np.nanpercentile(finite_overall, 5)),
            "median": float(np.nanmedian(finite_overall)),
            "p95": float(np.nanpercentile(finite_overall, 95)),
            "max": float(np.nanmax(finite_overall)),
        },
    }

    # Save baseline/spread per x
    baseline_df = pd.DataFrame(
        {
            "track_id": int(track_id),
            "threshold_set": str(tag),
            "x_mm": x_grid,
            "finite_fraction": finite_overall,
            "baseline_um": baseline_um_list,
            "spread_um": spread_um_list,
            "baseline_source": baseline_src_list,
            "baseline_n": baseline_n_list,
        }
    )
    baseline_df.to_csv(out_dir / "tables" / f"track_{track_id}_{tag}_baseline_and_coverage.csv", index=False)

    return out_df, meta


# =========================
# Representative-case comparison & robustness
# =========================


def nearest_x_rows(df: pd.DataFrame, requested_x: float) -> pd.Series:
    j = int(np.nanargmin(np.abs(df["x_mm"].to_numpy(float) - requested_x)))
    return df.iloc[j]


def compare_control_vs_corridor(
    control_df: pd.DataFrame,
    corridor_df: pd.DataFrame,
    track_id: int,
    requested_x_list: Sequence[float],
    threshold_set: str,
) -> pd.DataFrame:
    rows = []
    csub = control_df[(control_df.track_id == track_id)].copy()
    csub = csub[csub["config_name"] == "CONTROL"]

    for xreq in requested_x_list:
        crow = csub[np.isclose(csub["requested_x_mm"], float(xreq))].iloc[0]
        prow = nearest_x_rows(corridor_df, float(xreq))

        control_status = str(crow["extraction_status"])
        corr_valid = bool(prow["corridor_valid"])
        corr_status = "selected" if corr_valid else "no_valid_component"

        c_left = float(crow.get("segmentation_left_mm", np.nan))
        c_right = float(crow.get("segmentation_right_mm", np.nan))
        c_width = float(crow.get("segmentation_width_mm", np.nan))
        c_cent = float(crow.get("segmentation_centroid_mm", np.nan))

        p_left = float(prow.get("left_boundary_mm", np.nan))
        p_right = float(prow.get("right_boundary_mm", np.nan))
        p_width = float(prow.get("width_mm", np.nan))
        p_cent = float(0.5 * (p_left + p_right)) if np.isfinite(p_left) and np.isfinite(p_right) else float("nan")

        agree = False
        if (control_status == "selected") and corr_valid and np.isfinite(c_cent) and np.isfinite(p_cent):
            agree_center = abs(c_cent - p_cent) <= AGREE_CENTER_TOL_MM
            agree_width = (
                abs(c_width - p_width) / max(c_width, 1e-9) <= AGREE_WIDTH_REL_TOL
                if np.isfinite(c_width) and np.isfinite(p_width)
                else False
            )
            agree = bool(agree_center and agree_width)

        rows.append(
            {
                "track_id": int(track_id),
                "requested_x_mm": float(xreq),
                "threshold_set": str(threshold_set),
                "control_status": control_status,
                "corridor_status": corr_status,
                "control_left_mm": c_left,
                "control_right_mm": c_right,
                "control_width_mm": c_width,
                "corridor_left_mm": p_left,
                "corridor_right_mm": p_right,
                "corridor_width_mm": p_width,
                "center_delta_mm": float(p_cent - c_cent) if np.isfinite(p_cent) and np.isfinite(c_cent) else float("nan"),
                "width_delta_mm": float(p_width - c_width) if np.isfinite(p_width) and np.isfinite(c_width) else float("nan"),
                "spatial_agreement": bool(agree),
                "corridor_resolves_control_invalid": bool((control_status != "selected") and corr_valid),
                "corridor_rejects_control_selected": bool((control_status == "selected") and (not corr_valid)),
            }
        )

    return pd.DataFrame(rows)


# =========================
# Main
# =========================


def main() -> int:
    project_dir = REPO_ROOT
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = project_dir / "processed_data" / "run_outputs" / f"05_heightmap_persistence_corridor_{run_tag}"
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "diagnostics").mkdir(parents=True, exist_ok=True)

    # Save config
    config = {
        "tracks": TRACK_IDS,
        "sealed_tracks": sorted(list(SEALED_TRACKS)),
        "requested_x_mm": REQUESTED_X_MM.tolist(),
        "analysis_x_step_mm": ANALYSIS_X_STEP_MM,
        "x_window_half_width_mm": X_WINDOW_HALF_WIDTH_MM,
        "threshold_sets": THRESHOLD_SETS,
        "central_exclusion_y_mm": [CENTRAL_EXCLUSION_Y_MIN_MM, CENTRAL_EXCLUSION_Y_MAX_MM],
        "baseline": {"min_samples": MIN_BASELINE_SAMPLES, "min_spread_um": MIN_SPREAD_UM},
        "component": {
            "min_samples": MIN_COMPONENT_SAMPLES,
            "min_width_mm": MIN_COMPONENT_WIDTH_MM,
            "max_width_mm": MAX_COMPONENT_WIDTH_MM,
        },
        "family_matching": {
            "iou_min": FAMILY_IOU_MIN,
            "containment_bonus": FAMILY_CONTAINMENT_BONUS,
            "centroid_dist_max_mm": FAMILY_CENTROID_DIST_MAX_MM,
            "rep_interval": "component at median threshold if present else max-peak component",
        },
        "corridor_scoring": {
            "node_weights": {
                "persistence": NODE_PERSISTENCE_WEIGHT,
                "height_median": NODE_HEIGHT_MEDIAN_WEIGHT,
                "height_peak": NODE_HEIGHT_PEAK_WEIGHT,
                "stability": NODE_STABILITY_WEIGHT,
                "coverage": NODE_COVERAGE_WEIGHT,
            },
            "transition_weights": {
                "centroid_l1": TRANSITION_CENTROID_L1_WEIGHT,
                "left_l1": TRANSITION_LEFT_L1_WEIGHT,
                "right_l1": TRANSITION_RIGHT_L1_WEIGHT,
                "width_l1": TRANSITION_WIDTH_L1_WEIGHT,
            },
            "gap": {
                "max_gap_steps": MAX_GAP_STEPS,
                "gap_open_penalty": GAP_OPEN_PENALTY,
                "gap_step_penalty": GAP_STEP_PENALTY,
            },
            "gating": {
                "min_persistence_fraction": MIN_PERSISTENCE_FRACTION_FOR_NODE,
                "min_finite_fraction": MIN_FINITE_FRACTION_FOR_NODE,
            },
        },
        "agreement": {"center_tol_mm": AGREE_CENTER_TOL_MM, "width_rel_tol": AGREE_WIDTH_REL_TOL},
    }
    (out_dir / "tables" / "config.json").write_text(json.dumps(config, indent=2))

    control_df = load_control_outputs(project_dir)

    all_corridor_rows = []
    all_meta = []
    all_rep_comp = []

    for set_name, thresh_list in THRESHOLD_SETS.items():
        for track_id in TRACK_IDS:
            df, meta = run_for_track_and_thresholds(project_dir, out_dir, track_id, thresh_list, tag=set_name)
            df.to_csv(out_dir / "tables" / f"track_{track_id}_{set_name}_corridor.csv", index=False)
            all_corridor_rows.append(df)
            all_meta.append(meta)

            rep_df = compare_control_vs_corridor(control_df, df, track_id, REQUESTED_X_MM, threshold_set=set_name)
            rep_df.to_csv(out_dir / "tables" / f"track_{track_id}_{set_name}_control_vs_corridor_rep_cases.csv", index=False)
            all_rep_comp.append(rep_df)

    # Combined tables
    corridor_all = pd.concat(all_corridor_rows, ignore_index=True)
    corridor_all.to_csv(out_dir / "tables" / "corridor_all_tracks_all_threshold_sets.csv", index=False)

    rep_all = pd.concat(all_rep_comp, ignore_index=True)
    rep_all.to_csv(out_dir / "tables" / "control_vs_corridor_rep_cases_all.csv", index=False)

    (out_dir / "tables" / "run_metadata.json").write_text(json.dumps(all_meta, indent=2))

    # Threshold-set robustness summary
    # For each track and x, compare corridor outputs between FULL and the other sets.
    base = corridor_all[corridor_all.threshold_set == "FULL"].copy()

    rob_rows = []
    for track_id in TRACK_IDS:
        base_t = base[base.track_id == track_id]
        for set_name in ["REMOVE_LOW", "REMOVE_HIGH", "INNER"]:
            comp = corridor_all[(corridor_all.track_id == track_id) & (corridor_all.threshold_set == set_name)]
            # align by x (same grid by construction)
            merged = base_t.merge(comp, on=["track_id", "x_mm"], suffixes=("_full", f"_{set_name.lower()}"))

            # valid agreement
            valid_full = merged["corridor_valid_full"].to_numpy(bool)
            valid_alt = merged[f"corridor_valid_{set_name.lower()}"].to_numpy(bool)
            both_valid = valid_full & valid_alt

            center_full = 0.5 * (merged["left_boundary_mm_full"].to_numpy(float) + merged["right_boundary_mm_full"].to_numpy(float))
            center_alt = 0.5 * (
                merged[f"left_boundary_mm_{set_name.lower()}"].to_numpy(float)
                + merged[f"right_boundary_mm_{set_name.lower()}"].to_numpy(float)
            )

            width_full = merged["width_mm_full"].to_numpy(float)
            width_alt = merged[f"width_mm_{set_name.lower()}"].to_numpy(float)

            center_delta = np.full_like(center_full, np.nan, dtype=float)
            width_delta = np.full_like(width_full, np.nan, dtype=float)
            center_delta[both_valid] = center_alt[both_valid] - center_full[both_valid]
            width_delta[both_valid] = width_alt[both_valid] - width_full[both_valid]

            rob_rows.append(
                {
                    "track_id": int(track_id),
                    "compare_set": str(set_name),
                    "n_x": int(len(merged)),
                    "n_valid_full": int(np.sum(valid_full)),
                    "n_valid_alt": int(np.sum(valid_alt)),
                    "n_both_valid": int(np.sum(both_valid)),
                    "median_abs_center_shift_mm": float(np.nanmedian(np.abs(center_delta))),
                    "p95_abs_center_shift_mm": float(np.nanpercentile(np.abs(center_delta[np.isfinite(center_delta)]), 95))
                    if np.any(np.isfinite(center_delta))
                    else float("nan"),
                    "median_abs_width_shift_mm": float(np.nanmedian(np.abs(width_delta))),
                    "p95_abs_width_shift_mm": float(np.nanpercentile(np.abs(width_delta[np.isfinite(width_delta)]), 95))
                    if np.any(np.isfinite(width_delta))
                    else float("nan"),
                }
            )

    rob_df = pd.DataFrame(rob_rows)
    rob_df.to_csv(out_dir / "tables" / "threshold_set_robustness_summary.csv", index=False)

    # pick 10 most sensitive locations by max abs center shift across comparisons (within each track)
    sens_rows = []
    for track_id in TRACK_IDS:
        base_t = base[base.track_id == track_id].copy()
        base_t = base_t.sort_values("x_mm")
        x = base_t["x_mm"].to_numpy(float)

        shifts = []
        for set_name in ["REMOVE_LOW", "REMOVE_HIGH", "INNER"]:
            comp = corridor_all[(corridor_all.track_id == track_id) & (corridor_all.threshold_set == set_name)].sort_values("x_mm")
            merged = base_t.merge(comp, on=["track_id", "x_mm"], suffixes=("_full", "_alt"))
            valid = merged["corridor_valid_full"].to_numpy(bool) & merged["corridor_valid_alt"].to_numpy(bool)
            center_full = 0.5 * (merged["left_boundary_mm_full"].to_numpy(float) + merged["right_boundary_mm_full"].to_numpy(float))
            center_alt = 0.5 * (merged["left_boundary_mm_alt"].to_numpy(float) + merged["right_boundary_mm_alt"].to_numpy(float))
            d = np.full_like(center_full, 0.0)
            d[valid] = np.abs(center_alt[valid] - center_full[valid])
            shifts.append(d)

        if shifts:
            max_shift = np.max(np.vstack(shifts), axis=0)
        else:
            max_shift = np.zeros_like(x)

        # choose top 10 by shift, but require FULL valid
        full_valid = base_t["corridor_valid"].to_numpy(bool)
        max_shift[~full_valid] = -1.0
        top_idx = np.argsort(max_shift)[::-1][:10]
        for i in top_idx:
            if max_shift[i] < 0:
                continue
            sens_rows.append({"track_id": int(track_id), "x_mm": float(x[i]), "max_abs_center_shift_mm": float(max_shift[i])})

    sens_df = pd.DataFrame(sens_rows).sort_values(["track_id", "max_abs_center_shift_mm"], ascending=[True, False])
    sens_df.to_csv(out_dir / "tables" / "top10_threshold_set_sensitive_locations.csv", index=False)

    # Minimal diagnostic plots for sensitive locations (D)
    # To keep this script bounded, we plot aggregated profile + threshold lines + components + family reps.
    # We do NOT rerun per-threshold-set selection here; we annotate which family is chosen at that x.
    diag_dir = out_dir / "diagnostics"

    for track_id in TRACK_IDS:
        height_dir = project_dir / "data" / "raw" / "height_maps"
        data = load_wyko_asc(height_dir, track_id, crop_to_common=True)
        Z_mm = data["Z_mm"]
        x_actual = data["x_actual_mm"]
        y_mm = data["y_mm"]

        Z_det, _, _ = robust_plane_fit_substrate_focused(
            Z_mm,
            x_actual,
            y_mm,
            y_excl=(CENTRAL_EXCLUSION_Y_MIN_MM, CENTRAL_EXCLUSION_Y_MAX_MM),
        )

        sens_t = sens_df[sens_df.track_id == track_id].copy()
        for _, r in sens_t.iterrows():
            x0 = float(r["x_mm"])
            agg = aggregate_profile_in_x_window(Z_det, x_actual, y_mm, x0, X_WINDOW_HALF_WIDTH_MM)
            z_med_mm = agg["z_med_mm"]
            z_med_um = z_med_mm * 1e3

            baseline_um, spread_um, _, _ = compute_baseline_and_spread_um(
                y_mm,
                z_med_mm,
                y_excl=(CENTRAL_EXCLUSION_Y_MIN_MM, CENTRAL_EXCLUSION_Y_MAX_MM),
                min_samples=MIN_BASELINE_SAMPLES,
                min_spread_um=MIN_SPREAD_UM,
            )

            # components/families on FULL threshold list
            comps_by_thr = {
                float(t): candidate_components_for_threshold(y_mm, z_med_mm, baseline_um, spread_um, float(t))
                for t in THRESH_MULTIPLIERS_FULL
            }
            fams = build_families_for_x(x0, comps_by_thr, THRESH_MULTIPLIERS_FULL)

            chosen_family_ids: Dict[str, Optional[int]] = {}
            for set_name in THRESHOLD_SETS.keys():
                df = pd.read_csv(out_dir / "tables" / f"track_{track_id}_{set_name}_corridor.csv")
                prow = nearest_x_rows(df, x0)
                if bool(prow["corridor_valid"]):
                    # find best match family by overlap with chosen interval
                    interval = (float(prow["left_boundary_mm"]), float(prow["right_boundary_mm"]))
                    best = None
                    best_iou = -1.0
                    for fam in fams:
                        iou = interval_iou(interval, (fam.rep_y_min, fam.rep_y_max))
                        if iou > best_iou:
                            best_iou = iou
                            best = fam.family_id
                    chosen_family_ids[set_name] = best
                else:
                    chosen_family_ids[set_name] = None

            plot_sensitive_location(
                diag_dir / f"track_{track_id}_x{int(round(x0))}_sensitive.png",
                y_mm,
                z_med_um,
                baseline_um,
                spread_um,
                comps_by_thr,
                fams,
                chosen_family_ids,
                title=f"Track {track_id} @ x≈{x0:.2f} mm | sensitive location",
            )

    # Global CONTROL vs corridor comparison plot/table already saved; add a compact robustness plot.
    fig_dir = out_dir / "figures"
    for track_id in TRACK_IDS:
        # width comparison across sets
        fig, ax = plt.subplots(figsize=(12, 3))
        for set_name in THRESHOLD_SETS.keys():
            df = pd.read_csv(out_dir / "tables" / f"track_{track_id}_{set_name}_corridor.csv")
            ax.plot(df["x_mm"], df["width_mm"], label=set_name, linewidth=1)
        ax.set_title(f"Track {track_id} | width vs x across threshold sets")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("width (mm)")
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=4, fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / f"track_{track_id}_F_width_across_threshold_sets.png", dpi=180)
        plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
