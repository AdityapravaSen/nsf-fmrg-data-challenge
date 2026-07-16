"""06_heightmap_geometry_definition_audit.py

READ-ONLY SCIENTIFIC AUDIT (NOT production target extraction)
------------------------------------------------------------

Purpose
-------
Audit whether the project is using the wrong *geometric definition* of the
processed-track boundary in Wyko/Bruker height maps.

We compare four fundamentally different width definitions on shared profiles
aggregated from the height map at:
- representative x positions: [25,35,45,55,65,75,85,95] mm
- dense audit grid: 1.0 mm spacing over the common 20–100 mm interval

Constraints / Guardrails
------------------------
- Do NOT run notebooks.
- Do NOT create virtual environments.
- Use Homebrew Python 3.11 explicitly when executing.
- Do NOT load Track 21 (sealed).
- Use organizer loader unchanged: src/nsf_fmrg_data.py::load_wyko_asc.
- Use the shared substrate-focused detrending assumption from prior work.
- Do not interpolate across long NaN runs.
- Preserve explicit validity / failure reasons.

Outputs
-------
Writes under:
  processed_data/run_outputs/06_heightmap_geometry_definition_audit_<timestamp>/

Includes:
- tables/*.csv, *.json
- figures/*.png
- diagnostics/*.png

Execution
---------
/opt/homebrew/opt/python@3.11/bin/python3.11 scripts/06_heightmap_geometry_definition_audit.py

"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import json
import math
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Allow running from repo root without installing as a package.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from nsf_fmrg_data import load_wyko_asc  # organizer loader (unchanged)

warnings.filterwarnings("ignore", category=RuntimeWarning)


# =========================
# Configuration
# =========================

SEALED_TRACKS = {21}
TRACK_IDS = [8, 10, 14]
assert not (set(TRACK_IDS) & SEALED_TRACKS), "Track 21 is sealed and must not be analyzed."

REP_X_MM = np.array([25, 35, 45, 55, 65, 75, 85, 95], dtype=float)
TRACK10_FORENSIC_X_MM = np.array([35, 45, 55, 65, 75, 85, 95], dtype=float)

DENSE_STEP_MM = 1.0

# Shared local x-window configuration for aggregation (must be shared across tracks).
X_WINDOW_HALF_WIDTH_MM = 0.20

# Substrate-focused detrending configuration (from prior work)
CENTRAL_EXCLUSION_Y_MIN_MM = 0.65
CENTRAL_EXCLUSION_Y_MAX_MM = 1.35

# Substrate baseline/spread estimation must use samples outside the central exclusion band.
SUBSTRATE_Y_EXCLUSION = (CENTRAL_EXCLUSION_Y_MIN_MM, CENTRAL_EXCLUSION_Y_MAX_MM)

# Profile processing
MIN_RUN_SAMPLES_FOR_SMOOTHING = 15
SMOOTH_WINDOW_SAMPLES = 9  # odd

# Substrate statistics
MIN_SUBSTRATE_SAMPLES = 200
MIN_SPREAD_UM = 0.75

# Threshold levels for methods A/B (do NOT select winner)
THRESH_MULTIPLIERS = [1.5, 2.0, 2.5, 3.0, 3.5]

# Method C (gradient) settings
EDGE_SEARCH_MARGIN_MM = 0.10
EDGE_MIN_STRENGTH_UM_PER_MM = 50.0

# Method D (substrate-return) settings
RETURN_WINDOW_MM = 0.12
RETURN_MIN_PERSIST_MM = 0.10
RETURN_Z_K = 2.0
RETURN_GRAD_K = 2.0
RETURN_SEED_TOPQ = 0.10
RETURN_MAX_SEARCH_MM = 1.20

# y-side regions used for local substrate estimates
SIDE_REGION_WIDTH_MM = 0.55

# Candidate / seed definitions
MIN_CANDIDATE_SAMPLES = 8
MIN_CANDIDATE_WIDTH_MM = 0.035
MAX_CANDIDATE_WIDTH_MM = 1.10

SEED_ACTIVITY_SMOOTH_WINDOW_SAMPLES = 9
SEED_ACTIVITY_MIN_RUN_SAMPLES = 15
SEED_ACTIVITY_TOPQ = 0.10
SEED_PROMINENCE_MIN_Z = 2.5


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


def longest_run(runs: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    if not runs:
        return None
    return max(runs, key=lambda ab: ab[1] - ab[0])


def interval_iou(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    a0, a1 = a
    b0, b1 = b
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    if union <= 0:
        return 0.0
    return inter / union


def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def rolling_mean(run: np.ndarray, window: int) -> np.ndarray:
    w = int(window)
    if w < 3:
        w = 3
    if w % 2 == 0:
        w += 1
    pad = w // 2
    padded = np.pad(run, pad_width=pad, mode="edge")
    kernel = np.ones(w, dtype=float) / w
    return np.convolve(padded, kernel, mode="valid")


def smooth_within_finite_runs(z: np.ndarray, valid: np.ndarray, window: int, min_run: int) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    out = np.full_like(z, np.nan, dtype=float)
    for a, b in finite_runs(valid):
        run = z[a:b]
        if (b - a) >= int(min_run):
            out[a:b] = rolling_mean(run, window)
        else:
            out[a:b] = run
    return out


def summarize_interval(z: np.ndarray, valid: np.ndarray, a: int, b: int) -> Tuple[int, float, float]:
    """Return (n_finite, median_abs, peak_abs) on [a,b)."""
    v = np.asarray(valid[a:b], dtype=bool)
    zz = np.asarray(z[a:b], dtype=float)
    zz = zz[v & np.isfinite(zz)]
    if zz.size == 0:
        return 0, float("nan"), float("nan")
    absz = np.abs(zz)
    return int(zz.size), float(np.nanmedian(absz)), float(np.nanmax(absz))


def interval_contains_y(a_mm: float, b_mm: float, y_mm: float) -> bool:
    lo = min(a_mm, b_mm)
    hi = max(a_mm, b_mm)
    return (y_mm >= lo) and (y_mm <= hi)


def interval_overlap_mm(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    a0, a1 = a
    b0, b1 = b
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    return float(inter)


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
    """Fit and subtract a plane using downsampled points, excluding a central y band."""

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
    x_center_mm: float,
    x_half_width_mm: float,
) -> Dict[str, np.ndarray]:
    """Aggregate nearby columns into a single y-z profile.

    Strategy: robust column aggregation via median across selected x-columns.
    - preserves NaNs: median ignores NaNs; output can still be NaN if all NaN
    - stores finite support per y
    - returns per-y counts of finite samples used
    """

    x_actual_mm = np.asarray(x_actual_mm, dtype=float)
    y_mm = np.asarray(y_mm, dtype=float)
    Z_det_mm = np.asarray(Z_det_mm, dtype=float)

    x0 = float(x_center_mm - x_half_width_mm)
    x1 = float(x_center_mm + x_half_width_mm)
    cols = np.flatnonzero((x_actual_mm >= x0) & (x_actual_mm <= x1))

    if cols.size == 0:
        z_med = np.full((len(y_mm),), np.nan, dtype=float)
        n_finite = np.zeros((len(y_mm),), dtype=int)
    else:
        Zw = Z_det_mm[:, cols]
        n_finite = np.sum(np.isfinite(Zw), axis=1).astype(int)
        z_med = np.nanmedian(Zw, axis=1)

    valid = np.isfinite(z_med)

    return {
        "y_mm": y_mm,
        "z_mm": z_med,
        "valid": valid,
        "n_finite_per_y": n_finite,
        "x_cols": cols.astype(int),
        "x_window_mm": np.array([x0, x1], dtype=float),
        "x_center_mm": np.array([float(x_center_mm)], dtype=float),
    }


# =========================
# Substrate estimation
# =========================


def substrate_stats(z_mm: np.ndarray, valid: np.ndarray) -> Tuple[float, float, int]:
    z = np.asarray(z_mm, dtype=float)
    v = np.asarray(valid, dtype=bool)
    zv = z[v & np.isfinite(z)]
    if zv.size == 0:
        return float("nan"), float("nan"), 0
    baseline = float(np.nanmedian(zv))
    spread = robust_mad(zv, scale=True)
    spread = max(spread, MIN_SPREAD_UM * 1e-3)
    return baseline, float(spread), int(zv.size)


def compute_substrate_estimates(profile: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Compute substrate-focused baseline/spread.

    Global substrate uses finite samples OUTSIDE the central exclusion band.
    Falls back to all finite samples only if insufficient support.

    Left/right substrate are estimated in broad outer y regions (independent).
    """

    y = np.asarray(profile["y_mm"], dtype=float)
    z = np.asarray(profile["z_mm"], dtype=float)
    valid = np.asarray(profile["valid"], dtype=bool)

    # whole-profile median for confound auditing (not used as substrate)
    whole_profile_median = float(np.nanmedian(z[valid])) if np.any(valid) else float("nan")

    y0, y1 = SUBSTRATE_Y_EXCLUSION
    outside = (y < y0) | (y > y1)

    sub_mask = valid & outside
    n_sub = int(np.sum(sub_mask & np.isfinite(z)))

    fallback_used = False
    if n_sub < MIN_SUBSTRATE_SAMPLES:
        # fallback to all finite samples (explicitly recorded)
        fallback_used = True
        sub_mask = valid

    global_baseline, global_spread, n_global = substrate_stats(z, sub_mask)

    y_min = float(np.nanmin(y))
    y_max = float(np.nanmax(y))
    left_region = (y >= y_min) & (y <= y_min + SIDE_REGION_WIDTH_MM)
    right_region = (y >= y_max - SIDE_REGION_WIDTH_MM) & (y <= y_max)

    left_baseline, left_spread, n_left = substrate_stats(z, valid & left_region)
    right_baseline, right_spread, n_right = substrate_stats(z, valid & right_region)

    lr_diff_mm = left_baseline - right_baseline if (math.isfinite(left_baseline) and math.isfinite(right_baseline)) else float("nan")

    return {
        "whole_profile_median_mm": whole_profile_median,
        "substrate_baseline_fallback": bool(fallback_used),
        "substrate_global_baseline_mm": global_baseline,
        "substrate_global_spread_mm": global_spread,
        "substrate_n_global": float(n_global),
        "substrate_left_baseline_mm": left_baseline,
        "substrate_left_spread_mm": left_spread,
        "substrate_left_region_y_min_mm": float(y_min),
        "substrate_left_region_y_max_mm": float(y_min + SIDE_REGION_WIDTH_MM),
        "substrate_n_left": float(n_left),
        "substrate_right_baseline_mm": right_baseline,
        "substrate_right_spread_mm": right_spread,
        "substrate_right_region_y_min_mm": float(y_max - SIDE_REGION_WIDTH_MM),
        "substrate_right_region_y_max_mm": float(y_max),
        "substrate_n_right": float(n_right),
        "substrate_left_minus_right_um": (lr_diff_mm * 1e3) if np.isfinite(lr_diff_mm) else float("nan"),
    }


# =========================
# Method A/B: thresholded regions
# =========================


def contiguous_regions(mask: np.ndarray) -> List[Tuple[int, int]]:
    return finite_runs(mask)


def region_to_bounds(region: Tuple[int, int], y_mm: np.ndarray) -> Tuple[float, float, float, float]:
    a, b = region
    y0 = float(y_mm[a])
    y1 = float(y_mm[b - 1])
    width = max(0.0, y1 - y0)
    centroid = 0.5 * (y0 + y1)
    return y0, y1, float(width), float(centroid)


def extract_candidates_A(profile: Dict[str, np.ndarray], sub: Dict[str, float], k: float) -> List[Dict]:
    y = profile["y_mm"]
    z = profile["z_mm"]
    valid = profile["valid"]

    baseline = sub["substrate_global_baseline_mm"]
    spread = sub["substrate_global_spread_mm"]
    thr = float(baseline + k * spread)
    mask = valid & np.isfinite(z) & (z > thr)
    regions = contiguous_regions(mask)

    resid_um = (z - baseline) * 1e3

    out: List[Dict] = []
    cid = 0
    for a, b in regions:
        y0, y1, w, c = region_to_bounds((a, b), y)
        if w < MIN_CANDIDATE_WIDTH_MM or w > MAX_CANDIDATE_WIDTH_MM:
            continue
        n, med_abs, peak_abs = summarize_interval(resid_um, np.isfinite(resid_um), a, b)
        if n < MIN_CANDIDATE_SAMPLES:
            continue
        out.append(
            {
                "method": "A",
                "threshold_k": float(k),
                "thr_um": float(thr * 1e3),
                "candidate_id": int(cid),
                "y_min_mm": float(y0),
                "y_max_mm": float(y1),
                "width_mm": float(w),
                "centroid_mm": float(c),
                "n_samples": int(n),
                "median_abs_residual_um": float(med_abs),
                "peak_abs_residual_um": float(peak_abs),
                "finite_fraction": float(np.mean(profile["valid"][a:b])),
            }
        )
        cid += 1
    return out


def extract_candidates_B(profile: Dict[str, np.ndarray], sub: Dict[str, float], k: float) -> List[Dict]:
    y = profile["y_mm"]
    z = profile["z_mm"]
    valid = profile["valid"]

    baseline = sub["substrate_global_baseline_mm"]
    spread = sub["substrate_global_spread_mm"]
    thr_abs = float(k * spread)
    resid = z - baseline
    mask = valid & np.isfinite(resid) & (np.abs(resid) > thr_abs)
    regions = contiguous_regions(mask)

    resid_um = resid * 1e3

    out: List[Dict] = []
    cid = 0
    for a, b in regions:
        y0, y1, w, c = region_to_bounds((a, b), y)
        if w < MIN_CANDIDATE_WIDTH_MM or w > MAX_CANDIDATE_WIDTH_MM:
            continue
        n, med_abs, peak_abs = summarize_interval(resid_um, np.isfinite(resid_um), a, b)
        if n < MIN_CANDIDATE_SAMPLES:
            continue
        inside = resid_um[a:b]
        inside_v = np.isfinite(inside)
        frac_pos = float(np.mean(inside[inside_v] > 0.0)) if np.any(inside_v) else float("nan")
        frac_neg = float(np.mean(inside[inside_v] < 0.0)) if np.any(inside_v) else float("nan")
        out.append(
            {
                "method": "B",
                "threshold_k": float(k),
                "thr_abs_um": float(thr_abs * 1e3),
                "candidate_id": int(cid),
                "y_min_mm": float(y0),
                "y_max_mm": float(y1),
                "width_mm": float(w),
                "centroid_mm": float(c),
                "n_samples": int(n),
                "median_abs_residual_um": float(med_abs),
                "peak_abs_residual_um": float(peak_abs),
                "finite_fraction": float(np.mean(profile["valid"][a:b])),
                "frac_pos": frac_pos,
                "frac_neg": frac_neg,
            }
        )
        cid += 1
    return out


# =========================
# Method C: gradient transitions
# =========================


def gradient_transition_candidates(
    y: np.ndarray,
    z_mm: np.ndarray,
    valid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """Return (z_s, dzdy, candidates) where candidates are local transition points.

    Candidates are local maxima in |gradient| within finite runs.
    Records signed gradient, strength, and robust z-score vs gradient activity.
    """

    z_s = smooth_within_finite_runs(z_mm, valid, SMOOTH_WINDOW_SAMPLES, MIN_RUN_SAMPLES_FOR_SMOOTHING)
    dzdy = np.full_like(z_s, np.nan, dtype=float)
    for a, b in finite_runs(np.isfinite(z_s)):
        if (b - a) < 5:
            continue
        dzdy[a:b] = np.gradient(z_s[a:b], y[a:b])

    g_abs = np.abs(dzdy)
    g_scale = robust_mad(dzdy[np.isfinite(dzdy)], scale=True)
    if not np.isfinite(g_scale) or g_scale <= 0:
        g_scale = 1e-6

    # find local maxima in |grad| within finite runs
    cand_list: List[Dict] = []
    run_id = 0
    for a, b in finite_runs(np.isfinite(g_abs)):
        if (b - a) < 5:
            run_id += 1
            continue
        for i in range(a + 1, b - 1):
            if not np.isfinite(g_abs[i]):
                continue
            if (g_abs[i] >= g_abs[i - 1]) and (g_abs[i] >= g_abs[i + 1]):
                strength_um_per_mm = float(g_abs[i] * 1e3)
                zscore = float(g_abs[i] / g_scale)
                cand_list.append(
                    {
                        "candidate_id": int(len(cand_list)),
                        "run_id": int(run_id),
                        "y_mm": float(y[i]),
                        "grad_um_per_mm": float(dzdy[i] * 1e3),
                        "abs_grad_um_per_mm": strength_um_per_mm,
                        "grad_strength_z": zscore,
                    }
                )
        run_id += 1

    return z_s, dzdy, cand_list


def method_C_transitions(profile: Dict[str, np.ndarray], seed: Dict[str, float]) -> Dict[str, object]:
    y = np.asarray(profile["y_mm"], dtype=float)
    z = np.asarray(profile["z_mm"], dtype=float)
    valid = np.asarray(profile["valid"], dtype=bool)

    z_s, dzdy, cand_list = gradient_transition_candidates(y, z, valid)
    if not cand_list:
        return {"status": "no_candidates", "left_status": "no_candidates", "right_status": "no_candidates", "width_mm": float("nan")}

    # Apply a conservative inside-margin filter
    y_lo = float(y[0] + EDGE_SEARCH_MARGIN_MM)
    y_hi = float(y[-1] - EDGE_SEARCH_MARGIN_MM)
    cand_list = [c for c in cand_list if (c["y_mm"] >= y_lo and c["y_mm"] <= y_hi)]
    if not cand_list:
        return {"status": "no_candidates", "left_status": "no_candidates", "right_status": "no_candidates", "width_mm": float("nan")}

    seed_y0 = float(seed.get("seed_y_min_mm", float("nan")))
    seed_y1 = float(seed.get("seed_y_max_mm", float("nan")))
    if not (np.isfinite(seed_y0) and np.isfinite(seed_y1)):
        return {"status": "no_seed", "left_status": "no_seed", "right_status": "no_seed", "width_mm": float("nan")}

    left_cands = [c for c in cand_list if c["y_mm"] <= seed_y0]
    right_cands = [c for c in cand_list if c["y_mm"] >= seed_y1]

    def score_transition(c: Dict, side: str) -> float:
        # Rank primarily by robust strength, then proximity to seed edge
        strength = float(c["grad_strength_z"])
        if side == "left":
            dist = float(seed_y0 - c["y_mm"])
        else:
            dist = float(c["y_mm"] - seed_y1)
        return strength - 0.5 * dist

    def pick_side(cands: List[Dict], side: str) -> Tuple[Dict[str, object], List[Dict]]:
        if not cands:
            return ({f"{side}_status": "no_candidate", f"{side}_reason": "no_transition_on_side"}, [])
        scored = sorted([(score_transition(c, side), c) for c in cands], key=lambda t: t[0], reverse=True)
        best_score, best = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else float("-inf")
        margin = float(best_score - second_score) if np.isfinite(second_score) else float("inf")
        ambiguous = bool(len(scored) > 1 and margin < 0.5)
        status = "ok" if best["abs_grad_um_per_mm"] >= EDGE_MIN_STRENGTH_UM_PER_MM else "weak"
        return (
            {
                f"{side}_status": status,
                f"{side}_y_mm": float(best["y_mm"]),
                f"{side}_abs_grad_um_per_mm": float(best["abs_grad_um_per_mm"]),
                f"{side}_grad_strength_z": float(best["grad_strength_z"]),
                f"{side}_n_candidates": int(len(scored)),
                f"{side}_margin": float(margin) if np.isfinite(margin) else float("nan"),
                f"{side}_ambiguous": bool(ambiguous),
                f"{side}_reason": "ok" if status != "no_candidate" else "no_candidate",
            },
            [c for _, c in scored],
        )

    out: Dict[str, object] = {
        "status": "ok",
        "n_total_candidates": int(len(cand_list)),
    }
    left_out, left_ranked = pick_side(left_cands, "left")
    right_out, right_ranked = pick_side(right_cands, "right")
    out.update(left_out)
    out.update(right_out)
    out["all_candidates"] = cand_list
    out["left_ranked"] = left_ranked
    out["right_ranked"] = right_ranked

    if ("left_y_mm" in out) and ("right_y_mm" in out) and np.isfinite(out["left_y_mm"]) and np.isfinite(out["right_y_mm"]):
        out["width_mm"] = float(max(0.0, float(out["right_y_mm"]) - float(out["left_y_mm"])))
    else:
        out["width_mm"] = float("nan")

    return out


# =========================
# Method D: substrate-return boundaries
# =========================


def run_mean_abs(z: np.ndarray, a: int, b: int) -> float:
    zz = z[a:b]
    zz = zz[np.isfinite(zz)]
    return float(np.nanmean(np.abs(zz))) if zz.size else float("nan")


def run_mean_abs_grad(g: np.ndarray, a: int, b: int) -> float:
    gg = g[a:b]
    gg = gg[np.isfinite(gg)]
    return float(np.nanmean(np.abs(gg))) if gg.size else float("nan")


def pick_disturbance_seed(y: np.ndarray, resid: np.ndarray, valid: np.ndarray) -> Optional[int]:
    """Seed = index of high absolute residual; diagnostic, not tuned."""
    r = np.abs(resid)
    cand = valid & np.isfinite(r)
    if not np.any(cand):
        return None
    rv = r[cand]
    if rv.size < 10:
        return int(np.nanargmax(r))
    q = float(np.nanquantile(rv, 1.0 - RETURN_SEED_TOPQ))
    top = cand & (r >= q)
    if not np.any(top):
        return int(np.nanargmax(r))
    # pick median y among the top residual points
    y_top = y[top]
    y0 = float(np.nanmedian(y_top))
    return int(np.nanargmin(np.abs(y - y0)))

def disturbance_activity_signal(
    y: np.ndarray,
    z_s_mm: np.ndarray,
    valid: np.ndarray,
    baseline_mm: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Return (abs_resid_um, abs_grad_um_per_mm, activity, resid_scale_um, grad_scale_um_per_mm)."""

    resid_um = (z_s_mm - baseline_mm) * 1e3

    g = np.full_like(z_s_mm, np.nan, dtype=float)
    for a, b in finite_runs(np.isfinite(z_s_mm)):
        if (b - a) < 5:
            continue
        g[a:b] = np.gradient(z_s_mm[a:b], y[a:b])
    abs_grad_um_per_mm = np.abs(g) * 1e3

    resid_scale_um = robust_mad(resid_um[np.isfinite(resid_um)], scale=True)
    if not np.isfinite(resid_scale_um) or resid_scale_um <= 0:
        resid_scale_um = 1e-6

    grad_scale_um_per_mm = robust_mad(abs_grad_um_per_mm[np.isfinite(abs_grad_um_per_mm)], scale=True)
    if not np.isfinite(grad_scale_um_per_mm) or grad_scale_um_per_mm <= 0:
        grad_scale_um_per_mm = 1e-6

    abs_resid = np.abs(resid_um)
    activity = (abs_resid / resid_scale_um) + 0.5 * (abs_grad_um_per_mm / grad_scale_um_per_mm)
    activity[~(valid & np.isfinite(activity))] = np.nan
    activity_s = smooth_within_finite_runs(
        activity,
        np.isfinite(activity),
        SEED_ACTIVITY_SMOOTH_WINDOW_SAMPLES,
        SEED_ACTIVITY_MIN_RUN_SAMPLES,
    )
    return abs_resid, abs_grad_um_per_mm, activity_s, float(resid_scale_um), float(grad_scale_um_per_mm)


def find_seed_intervals(
    y: np.ndarray,
    activity: np.ndarray,
    valid: np.ndarray,
) -> List[Dict]:
    """Identify contiguous disturbance intervals from an activity signal.

    Threshold is a robust upper-quantile with a minimum z-score floor.
    """

    cand = valid & np.isfinite(activity)
    if not np.any(cand):
        return []
    av = activity[cand]
    if av.size < 10:
        thr = float(np.nanmax(av))
    else:
        thr = float(np.nanquantile(av, 1.0 - SEED_ACTIVITY_TOPQ))
    thr = max(thr, SEED_PROMINENCE_MIN_Z)

    mask = cand & (activity >= thr)
    intervals = []
    for a, b in finite_runs(mask):
        y0 = float(y[a])
        y1 = float(y[b - 1])
        width = float(max(0.0, y1 - y0))
        if width <= 0:
            continue
        integ = float(np.nansum(activity[a:b]))
        intervals.append(
            {
                "interval_id": int(len(intervals)),
                "a": int(a),
                "b": int(b),
                "y_min_mm": y0,
                "y_max_mm": y1,
                "width_mm": width,
                "centroid_mm": float(0.5 * (y0 + y1)),
                "integrated_activity": integ,
                "n_samples": int(np.sum(np.isfinite(activity[a:b]))),
            }
        )
    return intervals


def select_contiguous_seed(
    y: np.ndarray,
    z_mm: np.ndarray,
    valid: np.ndarray,
    baseline_mm: float,
) -> Dict[str, float]:
    """Select a contiguous disturbance seed interval.

    Returns a dict with ambiguity metadata.
    """

    z_s = smooth_within_finite_runs(z_mm, valid, SMOOTH_WINDOW_SAMPLES, MIN_RUN_SAMPLES_FOR_SMOOTHING)
    _, _, activity, resid_scale_um, grad_scale_um_per_mm = disturbance_activity_signal(y, z_s, valid, baseline_mm)
    intervals = find_seed_intervals(y, activity, np.isfinite(activity))
    if not intervals:
        return {
            "seed_status": "no_seed",
            "seed_reason": "no_disturbance_intervals",
        }

    ranked = sorted(intervals, key=lambda d: (d["integrated_activity"], d["width_mm"]), reverse=True)
    best = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None
    margin = float(best["integrated_activity"] - second["integrated_activity"]) if second is not None else float("inf")
    ambiguous = bool(second is not None and margin / max(best["integrated_activity"], 1e-9) < 0.15)

    return {
        "seed_status": "ok" if not ambiguous else "ambiguous_seed",
        "seed_reason": "ok" if not ambiguous else "competing_seed_intervals",
        "seed_y_min_mm": float(best["y_min_mm"]),
        "seed_y_max_mm": float(best["y_max_mm"]),
        "seed_centroid_mm": float(best["centroid_mm"]),
        "seed_integrated_activity": float(best["integrated_activity"]),
        "n_competing_seed_intervals": float(len(ranked)),
        "seed_margin_to_second_best": float(margin) if np.isfinite(margin) else float("nan"),
        "seed_ambiguous": bool(ambiguous),
        "seed_activity_threshold_z": float(max(float(np.nanquantile(activity[np.isfinite(activity)], 1.0 - SEED_ACTIVITY_TOPQ)), SEED_PROMINENCE_MIN_Z))
        if np.any(np.isfinite(activity))
        else float("nan"),
        "seed_resid_scale_um": float(resid_scale_um),
        "seed_grad_scale_um_per_mm": float(grad_scale_um_per_mm),
    }


def substrate_return(profile: Dict[str, np.ndarray], sub: Dict[str, float], seed: Dict[str, float]) -> Dict[str, object]:
    y = np.asarray(profile["y_mm"], dtype=float)
    z = np.asarray(profile["z_mm"], dtype=float)
    valid = np.asarray(profile["valid"], dtype=bool)

    seed_status = seed.get("seed_status", "no_seed")
    if seed_status == "no_seed":
        return {"status": "no_seed", "failure_category": "no_seed"}
    if seed_status == "ambiguous_seed":
        return {"status": "ambiguous_seed", "failure_category": "ambiguous_seed", **seed}

    seed_y0 = float(seed["seed_y_min_mm"])
    seed_y1 = float(seed["seed_y_max_mm"])

    z_s = smooth_within_finite_runs(z, valid, SMOOTH_WINDOW_SAMPLES, MIN_RUN_SAMPLES_FOR_SMOOTHING)

    # gradient within runs
    g = np.full_like(z_s, np.nan)
    for a, b in finite_runs(np.isfinite(z_s)):
        if (b - a) < 5:
            continue
        g[a:b] = np.gradient(z_s[a:b], y[a:b])
    abs_grad_um_per_mm = np.abs(g) * 1e3

    grad_scale_um_per_mm = robust_mad(abs_grad_um_per_mm[np.isfinite(abs_grad_um_per_mm)], scale=True)
    if not np.isfinite(grad_scale_um_per_mm) or grad_scale_um_per_mm <= 0:
        grad_scale_um_per_mm = 1e-6

    spread_um = float(sub["substrate_global_spread_mm"] * 1e3)
    resid_thr_um = float(RETURN_Z_K * spread_um)
    grad_thr_um_per_mm = float(RETURN_GRAD_K * grad_scale_um_per_mm)

    base_global = float(sub["substrate_global_baseline_mm"])
    base_L = float(sub["substrate_left_baseline_mm"]) if np.isfinite(sub["substrate_left_baseline_mm"]) else base_global
    base_R = float(sub["substrate_right_baseline_mm"]) if np.isfinite(sub["substrate_right_baseline_mm"]) else base_global

    def idx_for_y(yy: float) -> int:
        return int(np.nanargmin(np.abs(y - yy)))

    seed_i0 = idx_for_y(seed_y0)
    seed_i1 = idx_for_y(seed_y1)
    if seed_i1 < seed_i0:
        seed_i0, seed_i1 = seed_i1, seed_i0

    if not np.all(np.isfinite(z_s[seed_i0 : seed_i1 + 1])):
        return {"status": "missingness_or_nan_gap", "failure_category": "missingness_or_nan_gap", **seed}

    def window_indices(center_idx: int, win_mm: float) -> Optional[Tuple[int, int]]:
        yc = float(y[center_idx])
        y_lo = yc - 0.5 * win_mm
        y_hi = yc + 0.5 * win_mm
        a = int(np.searchsorted(y, y_lo, side="left"))
        b = int(np.searchsorted(y, y_hi, side="right"))
        a = max(a, 0)
        b = min(b, len(y))
        if b <= a:
            return None
        if not np.all(np.isfinite(z_s[a:b])):
            return None
        return a, b

    def persist_indices(center_idx: int, side: str, persist_mm: float) -> Optional[Tuple[int, int]]:
        yc = float(y[center_idx])
        if side == "left":
            y_lo = yc - persist_mm
            y_hi = yc
        else:
            y_lo = yc
            y_hi = yc + persist_mm
        a = int(np.searchsorted(y, y_lo, side="left"))
        b = int(np.searchsorted(y, y_hi, side="right"))
        a = max(a, 0)
        b = min(b, len(y))
        if b <= a:
            return None
        if not np.all(np.isfinite(z_s[a:b])):
            return None
        return a, b

    def eval_return_window(a: int, b: int, baseline_mm: float) -> Dict[str, float]:
        resid_um = (z_s[a:b] - baseline_mm) * 1e3
        n = int(np.sum(np.isfinite(resid_um)))
        if n == 0:
            return {
                "n_finite": 0.0,
                "mean_resid_um": float("nan"),
                "max_abs_resid_um": float("nan"),
                "mean_abs_grad_um_per_mm": float("nan"),
            }
        return {
            "n_finite": float(n),
            "mean_resid_um": float(np.nanmean(resid_um)),
            "max_abs_resid_um": float(np.nanmax(np.abs(resid_um))),
            "mean_abs_grad_um_per_mm": float(np.nanmean(abs_grad_um_per_mm[a:b])),
        }

    def find_return(side: str) -> Dict[str, object]:
        if side == "left":
            baseline_mm = base_L
            start_idx = seed_i0
            step = -1
            limit_y = seed_y0 - float(RETURN_MAX_SEARCH_MM)
            seed_edge_y = seed_y0
        else:
            baseline_mm = base_R
            start_idx = seed_i1
            step = 1
            limit_y = seed_y1 + float(RETURN_MAX_SEARCH_MM)
            seed_edge_y = seed_y1

        i = int(start_idx)
        while 0 <= i < len(y):
            yi = float(y[i])
            if (side == "left" and yi < limit_y) or (side == "right" and yi > limit_y):
                return {"status": "no_substrate_like_return", "failure_category": "no_substrate_like_return"}

            widx = window_indices(i, float(RETURN_WINDOW_MM))
            pidx = persist_indices(i, side, float(RETURN_MIN_PERSIST_MM))
            if widx is None or pidx is None:
                i += step
                continue

            aw, bw = widx
            ap, bp = pidx
            stats_w = eval_return_window(aw, bw, baseline_mm)
            stats_p = eval_return_window(ap, bp, baseline_mm)

            if stats_w["n_finite"] < 5:
                i += step
                continue

            cond_w = (stats_w["max_abs_resid_um"] <= resid_thr_um) and (stats_w["mean_abs_grad_um_per_mm"] <= grad_thr_um_per_mm)
            cond_p = (stats_p["max_abs_resid_um"] <= resid_thr_um) and (stats_p["mean_abs_grad_um_per_mm"] <= grad_thr_um_per_mm)
            if cond_w and cond_p:
                dist_from_seed_edge = float(abs(yi - seed_edge_y))
                out = {
                    "status": "ok",
                    "boundary_y_mm": float(yi),
                    "baseline_used_um": float(baseline_mm * 1e3),
                    "residual_threshold_um": float(resid_thr_um),
                    "gradient_threshold_um_per_mm": float(grad_thr_um_per_mm),
                    "return_window_mm": float(RETURN_WINDOW_MM),
                    "persist_window_mm": float(RETURN_MIN_PERSIST_MM),
                    "distance_from_seed_edge_mm": dist_from_seed_edge,
                    **{f"return_{k}": float(v) for k, v in stats_w.items()},
                }
                return out

            i += step

        return {"status": "edge_of_profile", "failure_category": "edge_of_profile"}

    left = find_return("left")
    right = find_return("right")

    status = "ok" if (left.get("status") == "ok" and right.get("status") == "ok") else "partial"
    out: Dict[str, object] = {
        "status": status,
        "failure_category": None if status == "ok" else "partial",
        **seed,
        "left_status": left.get("status"),
        "right_status": right.get("status"),
        "left_failure_category": left.get("failure_category"),
        "right_failure_category": right.get("failure_category"),
        "left_y_mm": safe_float(left.get("boundary_y_mm")),
        "right_y_mm": safe_float(right.get("boundary_y_mm")),
    }
    if np.isfinite(out["left_y_mm"]) and np.isfinite(out["right_y_mm"]):
        out["width_mm"] = float(max(0.0, out["right_y_mm"] - out["left_y_mm"]))
    else:
        out["width_mm"] = float("nan")

    # attach evidence
    for prefix, obj in [("left", left), ("right", right)]:
        for k, v in obj.items():
            if k in ("status", "failure_category", "boundary_y_mm"):
                continue
            out[f"{prefix}_{k}"] = v
        if obj.get("failure_category") is not None:
            out[f"{prefix}_failure_reason"] = obj.get("failure_category")
        else:
            out[f"{prefix}_failure_reason"] = "ok"

    return out


# =========================
# Plotting
# =========================


def plot_profile_diagnostics(
    out_path: Path,
    track_id: int,
    x_mm: float,
    profile: Dict[str, np.ndarray],
    sub: Dict[str, float],
    A: Dict[str, object],
    B: Dict[str, object],
    C: Dict[str, object],
    D: Dict[str, object],
):
    y = profile["y_mm"]
    z = profile["z_mm"]
    valid = profile["valid"]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(y[valid], z[valid] * 1e3, lw=1.5, color="black", label="aggregated profile (μm)")

    # Mark invalid regions
    if (~valid).any():
        ax.scatter(y[~valid], np.zeros_like(y[~valid]), s=6, color="lightgray", label="invalid y (NaN)")

    # corrected substrate baselines
    ax.axhline(sub["substrate_global_baseline_mm"] * 1e3, color="tab:blue", ls="--", lw=1.3, label="substrate baseline (global)")
    if np.isfinite(sub["substrate_left_baseline_mm"]):
        ax.axhline(sub["substrate_left_baseline_mm"] * 1e3, color="tab:cyan", ls=":", lw=1.1, label="substrate baseline (left)")
    if np.isfinite(sub["substrate_right_baseline_mm"]):
        ax.axhline(sub["substrate_right_baseline_mm"] * 1e3, color="tab:purple", ls=":", lw=1.1, label="substrate baseline (right)")

    # central exclusion band visual guide
    ax.axvspan(CENTRAL_EXCLUSION_Y_MIN_MM, CENTRAL_EXCLUSION_Y_MAX_MM, color="tab:blue", alpha=0.06, label="central exclusion band")

    # Method A/B: plot ALL candidates across thresholds as translucent spans
    def plot_candidates(cands: List[Dict], color: str, label: str):
        shown = False
        for c in cands:
            y0 = c["y_min_mm"]
            y1 = c["y_max_mm"]
            ax.axvspan(y0, y1, color=color, alpha=0.08)
            if not shown:
                ax.plot([], [], color=color, lw=6, alpha=0.25, label=label)
                shown = True

    plot_candidates(A.get("all_candidates", []) if isinstance(A, dict) else [], "tab:green", "A candidates (all k)")
    plot_candidates(B.get("all_candidates", []) if isinstance(B, dict) else [], "tab:orange", "B candidates (all k)")

    # Method C: show all transition candidates and selected edges
    if isinstance(C, dict) and ("all_candidates" in C):
        for c in C.get("all_candidates", []):
            ax.axvline(c["y_mm"], color="tab:red", lw=0.6, alpha=0.25)
        ax.plot([], [], color="tab:red", lw=1.5, alpha=0.6, label="C transition candidates")

    if isinstance(C, dict) and C.get("left_status") in ("ok", "weak"):
        ax.axvline(C.get("left_y_mm"), color="tab:red", ls="-.", lw=1.4, label="C left selected")
    if isinstance(C, dict) and C.get("right_status") in ("ok", "weak"):
        ax.axvline(C.get("right_y_mm"), color="tab:red", ls="-.", lw=1.4, label="C right selected")

    # Method D: seed interval and returns
    if isinstance(D, dict) and np.isfinite(D.get("seed_y_min_mm", float("nan"))) and np.isfinite(D.get("seed_y_max_mm", float("nan"))):
        ax.axvspan(D["seed_y_min_mm"], D["seed_y_max_mm"], color="tab:gray", alpha=0.20, label="D seed interval")
    if isinstance(D, dict) and np.isfinite(D.get("left_y_mm", float("nan"))):
        ax.axvline(D["left_y_mm"], color="tab:brown", ls="-", lw=1.8, label="D left return")
    if isinstance(D, dict) and np.isfinite(D.get("right_y_mm", float("nan"))):
        ax.axvline(D["right_y_mm"], color="tab:brown", ls="-", lw=1.8, label="D right return")

    ax.set_title(f"Track {track_id} | x={x_mm:.1f} mm | geometry definitions")
    ax.set_xlabel("y (mm)")
    ax.set_ylabel("z (μm) (detrended)")

    txt = (
        f"finite frac={np.mean(valid):.2f} | "
        f"LR substrate Δ={sub['substrate_left_minus_right_um'] if np.isfinite(sub['substrate_left_minus_right_um']) else np.nan:.2f} μm | "
        f"fallback={bool(sub['substrate_baseline_fallback'])}"
    )
    ax.text(0.01, 0.01, txt, transform=ax.transAxes, fontsize=9, va="bottom", ha="left")

    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# =========================
# Main experiment
# =========================


def main():
    project_dir = REPO_ROOT
    height_dir = project_dir / "data" / "raw" / "height_maps"

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = project_dir / "processed_data" / "run_outputs" / f"06_heightmap_geometry_definition_audit_{run_tag}"
    table_dir = out_dir / "tables"
    fig_dir = out_dir / "figures"
    diag_dir = out_dir / "diagnostics"
    for d in (table_dir, fig_dir, diag_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Load + detrend per track
    track_data = {}
    detrend_meta = {}
    for track_id in TRACK_IDS:
        hm = load_wyko_asc(height_dir, track_id, crop_to_common=True)
        Z = hm["Z_mm"]
        x = hm["x_actual_mm"]
        y = hm["y_mm"]

        Z_det, coef, meta = robust_plane_fit_substrate_focused(
            Z,
            x,
            y,
            y_excl=(CENTRAL_EXCLUSION_Y_MIN_MM, CENTRAL_EXCLUSION_Y_MAX_MM),
            stride_x=40,
            stride_y=2,
            max_iter=3,
        )

        track_data[track_id] = {"hm": hm, "Z_det": Z_det}
        detrend_meta[str(track_id)] = meta

    # Build dense x grid shared within each track (native x differs slightly)
    per_track_dense_x = {}
    for track_id in TRACK_IDS:
        x = track_data[track_id]["hm"]["x_actual_mm"]
        x0 = float(np.nanmin(x))
        x1 = float(np.nanmax(x))
        # clamp to [20,100] but rely on loader crop
        dense = np.arange(math.ceil(x0), math.floor(x1) + 1e-9, DENSE_STEP_MM, dtype=float)
        per_track_dense_x[track_id] = dense

    records_profiles = []
    records_baseline_compare = []
    records_candidates = []
    records_associations = []
    records_methodC = []
    records_methodD = []
    records_track10_forensic = []

    rep_cases = {(track_id, float(x_mm)) for track_id in TRACK_IDS for x_mm in REP_X_MM}

    # Process profiles for representative cases and dense grid
    for track_id in TRACK_IDS:
        hm = track_data[track_id]["hm"]
        Z_det = track_data[track_id]["Z_det"]
        x_actual = hm["x_actual_mm"]
        y = hm["y_mm"]

        # union grid: representative + dense
        xs = sorted(set([float(v) for v in REP_X_MM] + [float(v) for v in per_track_dense_x[track_id]]))

        for x_mm in xs:
            prof = aggregate_profile_in_x_window(
                Z_det_mm=Z_det,
                x_actual_mm=x_actual,
                y_mm=y,
                x_center_mm=x_mm,
                x_half_width_mm=X_WINDOW_HALF_WIDTH_MM,
            )
            sub = compute_substrate_estimates(prof)

            finite_frac = float(np.mean(prof["valid"]))
            lr_flag = bool(
                np.isfinite(sub["substrate_left_minus_right_um"])
                and np.isfinite(sub["substrate_global_spread_mm"])
                and (abs(sub["substrate_left_minus_right_um"]) > 3.0 * sub["substrate_global_spread_mm"] * 1e3)
            )

            records_baseline_compare.append(
                {
                    "track_id": track_id,
                    "x_mm": x_mm,
                    "whole_profile_median_um": float(sub["whole_profile_median_mm"] * 1e3) if np.isfinite(sub["whole_profile_median_mm"]) else float("nan"),
                    "substrate_baseline_um": float(sub["substrate_global_baseline_mm"] * 1e3) if np.isfinite(sub["substrate_global_baseline_mm"]) else float("nan"),
                    "difference_um": float((sub["whole_profile_median_mm"] - sub["substrate_global_baseline_mm"]) * 1e3)
                    if (np.isfinite(sub["whole_profile_median_mm"]) and np.isfinite(sub["substrate_global_baseline_mm"]))
                    else float("nan"),
                    "substrate_spread_um": float(sub["substrate_global_spread_mm"] * 1e3) if np.isfinite(sub["substrate_global_spread_mm"]) else float("nan"),
                    "fallback_used": bool(sub["substrate_baseline_fallback"]),
                }
            )

            baseline_mm = float(sub["substrate_global_baseline_mm"]) if np.isfinite(sub["substrate_global_baseline_mm"]) else float("nan")
            seed = select_contiguous_seed(prof["y_mm"], prof["z_mm"], prof["valid"], baseline_mm)

            A_all: List[Dict] = []
            B_all: List[Dict] = []
            for k in THRESH_MULTIPLIERS:
                A_all.extend(extract_candidates_A(prof, sub, float(k)))
                B_all.extend(extract_candidates_B(prof, sub, float(k)))

            C = method_C_transitions(prof, seed)
            D = substrate_return(prof, sub, seed)

            def associate_candidates(cands: List[Dict]) -> Dict[str, object]:
                if seed.get("seed_status") not in ("ok",):
                    return {"assoc_status": "no_seed", "assoc_reason": seed.get("seed_reason"), "selected_candidate_id": float("nan")}
                y0 = float(seed["seed_y_min_mm"])
                y1 = float(seed["seed_y_max_mm"])
                seed_c = float(seed["seed_centroid_mm"])
                if not cands:
                    return {"assoc_status": "no_candidates", "assoc_reason": "no_candidates", "selected_candidate_id": float("nan")}

                containing = [c for c in cands if interval_contains_y(c["y_min_mm"], c["y_max_mm"], seed_c)]
                if containing:
                    best = min(containing, key=lambda c: abs(c["centroid_mm"] - seed_c))
                    reason = "contains_seed_centroid"
                else:
                    best = min(cands, key=lambda c: abs(c["centroid_mm"] - seed_c))
                    reason = "nearest_centroid"

                overlap = interval_overlap_mm((best["y_min_mm"], best["y_max_mm"]), (y0, y1))
                return {
                    "assoc_status": "ok",
                    "assoc_reason": reason,
                    "selected_candidate_id": int(best["candidate_id"]),
                    "selected_threshold_k": float(best["threshold_k"]),
                    "selected_y_min_mm": float(best["y_min_mm"]),
                    "selected_y_max_mm": float(best["y_max_mm"]),
                    "selected_width_mm": float(best["width_mm"]),
                    "selected_centroid_mm": float(best["centroid_mm"]),
                    "centroid_dist_from_seed_mm": float(abs(best["centroid_mm"] - seed_c)),
                    "overlap_with_seed_interval_mm": float(overlap),
                }

            A_assoc = associate_candidates(A_all)
            B_assoc = associate_candidates(B_all)

            # save profile record (every audited x)
            records_profiles.append(
                {
                    "track_id": track_id,
                    "x_mm": x_mm,
                    "x_window_left_mm": float(prof["x_window_mm"][0]),
                    "x_window_right_mm": float(prof["x_window_mm"][1]),
                    "n_x_cols": int(len(prof["x_cols"])),
                    "finite_fraction": finite_frac,
                    "lr_substrate_disagree": lr_flag,
                    **sub,
                }
            )

            for c in A_all + B_all:
                records_candidates.append({"track_id": track_id, "x_mm": x_mm, **c})

            records_associations.append(
                {
                    "track_id": track_id,
                    "x_mm": x_mm,
                    **seed,
                    **{f"A_{k}": v for k, v in A_assoc.items()},
                    **{f"B_{k}": v for k, v in B_assoc.items()},
                }
            )

            records_methodC.append(
                {
                    "track_id": track_id,
                    "x_mm": x_mm,
                    "status": C.get("status") if isinstance(C, dict) else None,
                    "left_status": C.get("left_status") if isinstance(C, dict) else None,
                    "right_status": C.get("right_status") if isinstance(C, dict) else None,
                    "left_y_mm": safe_float(C.get("left_y_mm")) if isinstance(C, dict) else float("nan"),
                    "right_y_mm": safe_float(C.get("right_y_mm")) if isinstance(C, dict) else float("nan"),
                    "width_mm": safe_float(C.get("width_mm")) if isinstance(C, dict) else float("nan"),
                    "left_n_candidates": safe_float(C.get("left_n_candidates")) if isinstance(C, dict) else float("nan"),
                    "right_n_candidates": safe_float(C.get("right_n_candidates")) if isinstance(C, dict) else float("nan"),
                    "left_margin": safe_float(C.get("left_margin")) if isinstance(C, dict) else float("nan"),
                    "right_margin": safe_float(C.get("right_margin")) if isinstance(C, dict) else float("nan"),
                    "left_ambiguous": bool(C.get("left_ambiguous", False)) if isinstance(C, dict) else False,
                    "right_ambiguous": bool(C.get("right_ambiguous", False)) if isinstance(C, dict) else False,
                    "left_reason": C.get("left_reason") if isinstance(C, dict) else None,
                    "right_reason": C.get("right_reason") if isinstance(C, dict) else None,
                }
            )
            if isinstance(C, dict):
                for c in C.get("all_candidates", []):
                    records_methodC.append({"track_id": track_id, "x_mm": x_mm, "status": "candidate", **c})

            records_methodD.append({"track_id": track_id, "x_mm": x_mm, **D})

            # Representative-case diagnostics plot
            if (track_id, float(x_mm)) in rep_cases:
                p = diag_dir / f"rep_profile_track_{track_id}_x_{x_mm:.1f}mm.png"
                plot_profile_diagnostics(
                    p,
                    track_id,
                    x_mm,
                    prof,
                    sub,
                    {"all_candidates": A_all},
                    {"all_candidates": B_all},
                    C,
                    D,
                )

            # Track-10 forensic table rows (requested x only)
            if track_id == 10 and any(abs(x_mm - v) < 1e-6 for v in TRACK10_FORENSIC_X_MM):
                baseline_mm = sub["substrate_global_baseline_mm"]
                resid_um = (prof["z_mm"] - baseline_mm) * 1e3
                v = prof["valid"] & np.isfinite(resid_um)
                if np.any(v):
                    abs_resid = np.abs(resid_um[v])
                    mag_med = float(np.nanmedian(abs_resid))
                    pos_frac = float(np.mean(resid_um[v] > 0.0))
                    neg_frac = float(np.mean(resid_um[v] < 0.0))
                    sign_class = "mixed"
                    if pos_frac > 0.75:
                        sign_class = "primarily_positive"
                    elif neg_frac > 0.75:
                        sign_class = "primarily_negative"
                else:
                    mag_med = float("nan")
                    pos_frac = float("nan")
                    neg_frac = float("nan")
                    sign_class = "unknown"

                A_counts = {k: 0 for k in THRESH_MULTIPLIERS}
                B_counts = {k: 0 for k in THRESH_MULTIPLIERS}
                for c in A_all:
                    A_counts[float(c["threshold_k"])] += 1
                for c in B_all:
                    B_counts[float(c["threshold_k"])] += 1

                seed_ok = seed.get("seed_status") == "ok"
                if seed_ok:
                    sy0, sy1 = float(seed["seed_y_min_mm"]), float(seed["seed_y_max_mm"])
                    frag_by_k = {}
                    for k in THRESH_MULTIPLIERS:
                        cands_k = [c for c in A_all if np.isclose(c["threshold_k"], k)]
                        n_overlap = sum(interval_overlap_mm((c["y_min_mm"], c["y_max_mm"]), (sy0, sy1)) > 0 for c in cands_k)
                        frag_by_k[k] = int(n_overlap)
                else:
                    frag_by_k = {k: -1 for k in THRESH_MULTIPLIERS}

                records_track10_forensic.append(
                    {
                        "track_id": 10,
                        "x_mm": x_mm,
                        "finite_fraction": finite_frac,
                        "localized_disturbed_region_identifiable": bool(seed.get("seed_status") in ("ok", "ambiguous_seed")),
                        "disturbance_sign_class": sign_class,
                        "frac_positive": pos_frac,
                        "frac_negative": neg_frac,
                        "median_abs_residual_um": mag_med,
                        "lr_substrate_consistent": (not lr_flag),
                        "lr_substrate_diff_um": sub.get("substrate_left_minus_right_um"),
                        "seed_status": seed.get("seed_status"),
                        "seed_ambiguous": bool(seed.get("seed_ambiguous", False)),
                        "seed_n_competing": safe_float(seed.get("n_competing_seed_intervals")),
                        "seed_integrated_activity": safe_float(seed.get("seed_integrated_activity")),
                        "A_candidates_k1p5": A_counts[1.5],
                        "A_candidates_k2p0": A_counts[2.0],
                        "A_candidates_k2p5": A_counts[2.5],
                        "A_candidates_k3p0": A_counts[3.0],
                        "A_candidates_k3p5": A_counts[3.5],
                        "B_candidates_k1p5": B_counts[1.5],
                        "B_candidates_k2p0": B_counts[2.0],
                        "B_candidates_k2p5": B_counts[2.5],
                        "B_candidates_k3p0": B_counts[3.0],
                        "B_candidates_k3p5": B_counts[3.5],
                        "A_seed_overlap_count_k1p5": frag_by_k[1.5],
                        "A_seed_overlap_count_k2p0": frag_by_k[2.0],
                        "A_seed_overlap_count_k2p5": frag_by_k[2.5],
                        "A_seed_overlap_count_k3p0": frag_by_k[3.0],
                        "A_seed_overlap_count_k3p5": frag_by_k[3.5],
                        "D_left_return_ok": bool(D.get("left_status") == "ok"),
                        "D_right_return_ok": bool(D.get("right_status") == "ok"),
                        "D_left_failure": D.get("left_failure_category"),
                        "D_right_failure": D.get("right_failure_category"),
                        "C_left_ambiguous": bool(C.get("left_ambiguous", False)) if isinstance(C, dict) else False,
                        "C_right_ambiguous": bool(C.get("right_ambiguous", False)) if isinstance(C, dict) else False,
                        "primary_uncertainty_source": "baseline_inconsistency"
                        if lr_flag
                        else ("multiple_competing_disturbances" if bool(seed.get("seed_ambiguous", False)) else "ok"),
                    }
                )

    df_profiles = pd.DataFrame.from_records(records_profiles)
    df_baseline = pd.DataFrame.from_records(records_baseline_compare)
    df_candidates = pd.DataFrame.from_records(records_candidates)
    df_assoc = pd.DataFrame.from_records(records_associations)
    df_C = pd.DataFrame.from_records(records_methodC)
    df_D = pd.DataFrame.from_records(records_methodD)
    df_t10 = pd.DataFrame.from_records(records_track10_forensic)

    df_profiles.to_csv(table_dir / "profiles_substrate_estimates.csv", index=False)
    df_baseline.to_csv(table_dir / "whole_profile_median_vs_substrate_baseline.csv", index=False)
    df_candidates.to_csv(table_dir / "AB_all_candidates.csv", index=False)
    df_assoc.to_csv(table_dir / "seed_and_AB_seed_associations.csv", index=False)
    df_C.to_csv(table_dir / "methodC_transitions_and_selection.csv", index=False)
    df_D.to_csv(table_dir / "methodD_seed_return_evidence.csv", index=False)
    df_t10.to_csv(table_dir / "track10_forensic_table.csv", index=False)

    # Representative-case multi-threshold sensitivity summaries for A/B
    rep_assoc = df_assoc[df_assoc["x_mm"].isin(REP_X_MM)].copy()

    rep_rows = []
    for track_id, x_mm in rep_assoc[["track_id", "x_mm"]].itertuples(index=False):
        seed_row = rep_assoc[(rep_assoc.track_id == track_id) & (rep_assoc.x_mm == x_mm)].iloc[0].to_dict()
        seed_status = seed_row.get("seed_status")

        row = {"track_id": int(track_id), "x_mm": float(x_mm), "seed_status": seed_status, "seed_ambiguous": bool(seed_row.get("seed_ambiguous", False))}

        for method in ["A", "B"]:
            cands = df_candidates[(df_candidates.track_id == track_id) & (df_candidates.x_mm == x_mm) & (df_candidates.method == method)]
            n_total = int(len(cands))
            row[f"{method}_n_total_candidates"] = n_total

            lefts = []
            rights = []
            widths = []
            n_thr_with_assoc = 0
            if seed_status == "ok":
                syc = float(seed_row["seed_centroid_mm"])
                for k in THRESH_MULTIPLIERS:
                    ck = cands[np.isclose(cands.threshold_k.astype(float), float(k))]
                    if ck.empty:
                        continue
                    containing = ck[(ck.y_min_mm <= syc) & (syc <= ck.y_max_mm)]
                    if not containing.empty:
                        sel = containing.iloc[(containing.centroid_mm - syc).abs().argmin()]
                    else:
                        sel = ck.iloc[(ck.centroid_mm - syc).abs().argmin()]
                    n_thr_with_assoc += 1
                    lefts.append(float(sel.y_min_mm))
                    rights.append(float(sel.y_max_mm))
                    widths.append(float(sel.width_mm))

            row[f"{method}_n_thresholds_with_seed_assoc"] = int(n_thr_with_assoc)
            if n_thr_with_assoc > 0:
                row[f"{method}_left_spread_mm"] = float(np.nanmax(lefts) - np.nanmin(lefts))
                row[f"{method}_right_spread_mm"] = float(np.nanmax(rights) - np.nanmin(rights))
                row[f"{method}_width_spread_mm"] = float(np.nanmax(widths) - np.nanmin(widths))
                row[f"{method}_left_median_mm"] = float(np.nanmedian(lefts))
                row[f"{method}_right_median_mm"] = float(np.nanmedian(rights))
                row[f"{method}_width_median_mm"] = float(np.nanmedian(widths))
            else:
                row[f"{method}_left_spread_mm"] = float("nan")
                row[f"{method}_right_spread_mm"] = float("nan")
                row[f"{method}_width_spread_mm"] = float("nan")
                row[f"{method}_left_median_mm"] = float("nan")
                row[f"{method}_right_median_mm"] = float("nan")
                row[f"{method}_width_median_mm"] = float("nan")

        c_row = df_C[(df_C.track_id == track_id) & (df_C.x_mm == x_mm) & (df_C.status != "candidate")].iloc[0].to_dict()
        d_row = df_D[(df_D.track_id == track_id) & (df_D.x_mm == x_mm)].iloc[0].to_dict()
        row.update({"C_left_mm": safe_float(c_row.get("left_y_mm")), "C_right_mm": safe_float(c_row.get("right_y_mm")), "C_width_mm": safe_float(c_row.get("width_mm"))})
        row.update({"D_left_mm": safe_float(d_row.get("left_y_mm")), "D_right_mm": safe_float(d_row.get("right_y_mm")), "D_width_mm": safe_float(d_row.get("width_mm")), "D_status": d_row.get("status")})

        rep_rows.append(row)

    df_rep = pd.DataFrame(rep_rows)
    df_rep.to_csv(table_dir / "rep_AB_threshold_sensitivity_summary.csv", index=False)

    rep_for_cmp = df_rep.copy()
    rep_for_cmp["A_left_mm"] = rep_for_cmp["A_left_median_mm"]
    rep_for_cmp["A_right_mm"] = rep_for_cmp["A_right_median_mm"]
    rep_for_cmp["A_width_mm"] = rep_for_cmp["A_width_median_mm"]
    rep_for_cmp["B_left_mm"] = rep_for_cmp["B_left_median_mm"]
    rep_for_cmp["B_right_mm"] = rep_for_cmp["B_right_median_mm"]
    rep_for_cmp["B_width_mm"] = rep_for_cmp["B_width_median_mm"]

    methods = ["A", "B", "C", "D"]
    pair_rows = []
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            m1, m2 = methods[i], methods[j]

            left1 = rep_for_cmp[f"{m1}_left_mm"]
            left2 = rep_for_cmp[f"{m2}_left_mm"]
            right1 = rep_for_cmp[f"{m1}_right_mm"]
            right2 = rep_for_cmp[f"{m2}_right_mm"]
            w1 = rep_for_cmp[f"{m1}_width_mm"]
            w2 = rep_for_cmp[f"{m2}_width_mm"]

            both = np.isfinite(left1) & np.isfinite(left2) & np.isfinite(right1) & np.isfinite(right2)
            n_both = int(np.sum(both))
            if n_both == 0:
                pair_rows.append(
                    {
                        "method1": m1,
                        "method2": m2,
                        "both_valid": 0,
                        "median_abs_left_diff_mm": float("nan"),
                        "median_abs_right_diff_mm": float("nan"),
                        "median_abs_width_diff_mm": float("nan"),
                        "p95_abs_width_diff_mm": float("nan"),
                        "median_iou": float("nan"),
                    }
                )
                continue

            dL = np.abs(left1[both] - left2[both])
            dR = np.abs(right1[both] - right2[both])
            dW = np.abs(w1[both] - w2[both])

            ious = []
            for idx in rep_for_cmp[both].index:
                a = (float(left1.loc[idx]), float(right1.loc[idx]))
                b = (float(left2.loc[idx]), float(right2.loc[idx]))
                ious.append(interval_iou(a, b))
            ious = np.array(ious, dtype=float)

            pair_rows.append(
                {
                    "method1": m1,
                    "method2": m2,
                    "both_valid": n_both,
                    "median_abs_left_diff_mm": float(np.nanmedian(dL)),
                    "median_abs_right_diff_mm": float(np.nanmedian(dR)),
                    "median_abs_width_diff_mm": float(np.nanmedian(dW)),
                    "p95_abs_width_diff_mm": float(np.nanpercentile(dW, 95)),
                    "median_iou": float(np.nanmedian(ious)),
                }
            )

    df_pairs = pd.DataFrame(pair_rows)
    df_pairs.to_csv(table_dir / "pairwise_agreement_rep_multithreshold_AB.csv", index=False)

    df_rep_grid_rows = []
    for track_id in TRACK_IDS:
        for x_mm in per_track_dense_x[track_id]:
            assoc_row = df_assoc[(df_assoc.track_id == track_id) & (np.isclose(df_assoc.x_mm, x_mm))]
            if assoc_row.empty:
                continue
            assoc = assoc_row.iloc[0].to_dict()
            seed_status = assoc.get("seed_status")
            out_row = {"track_id": int(track_id), "x_mm": float(x_mm), "seed_status": seed_status}

            for method in ["A", "B"]:
                cands = df_candidates[(df_candidates.track_id == track_id) & (np.isclose(df_candidates.x_mm, x_mm)) & (df_candidates.method == method)]
                if seed_status != "ok" or cands.empty:
                    out_row[f"{method}_left_mm"] = float("nan")
                    out_row[f"{method}_right_mm"] = float("nan")
                    out_row[f"{method}_width_mm"] = float("nan")
                    continue
                syc = float(assoc["seed_centroid_mm"])
                lefts = []
                rights = []
                for k in THRESH_MULTIPLIERS:
                    ck = cands[np.isclose(cands.threshold_k.astype(float), float(k))]
                    if ck.empty:
                        continue
                    containing = ck[(ck.y_min_mm <= syc) & (syc <= ck.y_max_mm)]
                    if not containing.empty:
                        sel = containing.iloc[(containing.centroid_mm - syc).abs().argmin()]
                    else:
                        sel = ck.iloc[(ck.centroid_mm - syc).abs().argmin()]
                    lefts.append(float(sel.y_min_mm))
                    rights.append(float(sel.y_max_mm))
                if lefts:
                    out_row[f"{method}_left_mm"] = float(np.nanmedian(lefts))
                    out_row[f"{method}_right_mm"] = float(np.nanmedian(rights))
                    out_row[f"{method}_width_mm"] = float(out_row[f"{method}_right_mm"] - out_row[f"{method}_left_mm"])
                else:
                    out_row[f"{method}_left_mm"] = float("nan")
                    out_row[f"{method}_right_mm"] = float("nan")
                    out_row[f"{method}_width_mm"] = float("nan")

            c_row = df_C[(df_C.track_id == track_id) & (np.isclose(df_C.x_mm, x_mm)) & (df_C.status != "candidate")]
            d_row = df_D[(df_D.track_id == track_id) & (np.isclose(df_D.x_mm, x_mm))]
            if not c_row.empty:
                c = c_row.iloc[0]
                out_row["C_left_mm"] = safe_float(c.get("left_y_mm"))
                out_row["C_right_mm"] = safe_float(c.get("right_y_mm"))
                out_row["C_width_mm"] = safe_float(c.get("width_mm"))
            if not d_row.empty:
                d = d_row.iloc[0]
                out_row["D_left_mm"] = safe_float(d.get("left_y_mm"))
                out_row["D_right_mm"] = safe_float(d.get("right_y_mm"))
                out_row["D_width_mm"] = safe_float(d.get("width_mm"))

            df_rep_grid_rows.append(out_row)

    df_grid = pd.DataFrame(df_rep_grid_rows)
    df_grid.to_csv(table_dir / "densegrid_ABCD_representative_intervals.csv", index=False)

    for track_id in TRACK_IDS:
        df = df_grid[df_grid.track_id == track_id].sort_values("x_mm")
        fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        for ax, kind in zip(axes, ["width", "left", "right"]):
            ax.set_title(f"Track {track_id}: {kind} vs x (1.0 mm grid; A/B are multi-threshold seed-associated medians)")
            ax.set_ylabel("mm")
            ax.grid(True, alpha=0.2)
        axes[-1].set_xlabel("x (mm)")
        for method, color in [("A", "tab:green"), ("B", "tab:orange"), ("C", "tab:red"), ("D", "tab:brown")]:
            axes[0].plot(df["x_mm"], df[f"{method}_width_mm"], lw=1.2, color=color, label=method)
            axes[1].plot(df["x_mm"], df[f"{method}_left_mm"], lw=1.2, color=color)
            axes[2].plot(df["x_mm"], df[f"{method}_right_mm"], lw=1.2, color=color)
        axes[0].legend(loc="upper right", fontsize=9, ncol=4)
        fig.tight_layout()
        fig.savefig(fig_dir / f"track_{track_id}_ABCD_panels_width_left_right_vs_x.png", dpi=200)
        plt.close(fig)

    wide = rep_for_cmp.set_index(["track_id", "x_mm"])
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            m1, m2 = methods[i], methods[j]
            d = wide[f"{m1}_width_mm"] - wide[f"{m2}_width_mm"]
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.axhline(0.0, color="black", lw=1)
            ax.plot(np.arange(len(d)), d.values, marker="o", lw=0.8)
            ax.set_title(f"Rep cases width difference: {m1} - {m2} (A/B are multi-threshold medians)")
            ax.set_ylabel("Δ width (mm)")
            ax.set_xticks(np.arange(len(d)))
            ax.set_xticklabels([f"T{tid}@{x:.0f}" for (tid, x) in d.index], rotation=45, ha="right")
            ax.grid(True, alpha=0.2)
            fig.tight_layout()
            fig.savefig(fig_dir / f"rep_width_diff_{m1}_minus_{m2}.png", dpi=200)
            plt.close(fig)

    d_only = df_D[df_D.track_id.isin(TRACK_IDS)].copy()
    left_fail = d_only["left_failure_category"].fillna("(none)").value_counts()
    right_fail = d_only["right_failure_category"].fillna("(none)").value_counts()
    fail_df = pd.DataFrame({"left_failure_count": left_fail, "right_failure_count": right_fail}).fillna(0).astype(int)
    fail_df.to_csv(table_dir / "methodD_failure_reason_counts.csv")

    # Save metadata / config
    (table_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "tracks": TRACK_IDS,
                "sealed_tracks": sorted(SEALED_TRACKS),
                "rep_x_mm": REP_X_MM.tolist(),
                "dense_step_mm": DENSE_STEP_MM,
                "x_window_half_width_mm": X_WINDOW_HALF_WIDTH_MM,
                "detrend": {
                    "central_exclusion_y_mm": [CENTRAL_EXCLUSION_Y_MIN_MM, CENTRAL_EXCLUSION_Y_MAX_MM],
                },
                "threshold_multipliers": THRESH_MULTIPLIERS,
                "methodC": {
                    "edge_search_margin_mm": EDGE_SEARCH_MARGIN_MM,
                    "edge_min_strength_um_per_mm": EDGE_MIN_STRENGTH_UM_PER_MM,
                },
                "methodD": {
                    "return_window_mm": RETURN_WINDOW_MM,
                    "return_min_persist_mm": RETURN_MIN_PERSIST_MM,
                    "return_z_k": RETURN_Z_K,
                    "return_grad_k": RETURN_GRAD_K,
                    "return_seed_topq": RETURN_SEED_TOPQ,
                    "return_max_search_mm": RETURN_MAX_SEARCH_MM,
                },
                "substrate": {
                    "global_exclusion_band_y_mm": [SUBSTRATE_Y_EXCLUSION[0], SUBSTRATE_Y_EXCLUSION[1]],
                    "min_substrate_samples": MIN_SUBSTRATE_SAMPLES,
                    "min_spread_um": MIN_SPREAD_UM,
                    "side_region_width_mm": SIDE_REGION_WIDTH_MM,
                },
                "detrend_meta_by_track": detrend_meta,
            },
            indent=2,
        )
    )

    print(f"Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
