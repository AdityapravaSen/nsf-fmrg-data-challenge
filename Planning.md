Here's a phased breakdown with dependencies mapped out, split two ways. I've tried to keep each person owning a coherent vertical slice rather than randomly split tasks, so you're not constantly blocked on each other.

## Phase 0 — Setup (both, do together first)

| Task | Branch | Who |
|---|---|---|
| Download Zenodo data, verify folder layout matches README | — (no branch, just local setup) | Both |
| Run `01_starter_code_loading_and_visualization.ipynb` and `02_starter_code_loading_and_visualization.ipynb` end-to-end, confirm it works on your machines | — | Both |

Don't skip this — if the notebook doesn't run cleanly for both of you first, everything downstream wastes time debugging environment issues instead of the actual project.

---

## Phase 1 — Data Understanding & Alignment (blocks everything else)

**Person A — thermal + SEM side**
- Branch: `adi`
- Verify `extract_final_thermal_frames` laser on/off detection actually looks right on all 4 tracks (visually, frame-by-frame)
- Confirm SEM tile → x-position mapping is correct (remember: reversed numbering)
- Write a function that maps thermal frame index ↔ SEM tile ↔ common x-coordinate — this is the "Rosetta stone" the rest of the project depends on

**Person B — height map side**
- Branch: `nabarun`
- Verify `robust_plane_detrend` output looks sane on all 4 tracks
- Write the actual **target extraction**: from detrended `Z_mm`, extract per-x-position width, boundary position, contour deviation, roughness (this is the big missing piece from the README)

**Merge point:** Both branches PR into `main` once each person's alignment/extraction is validated with plots. Do this before Phase 2 — you need a shared "aligned dataset" object both of you build models against.

---

## Phase 2 — Dataset Assembly (shared, short)

- Branch: `feature/dataset-builder`
- Whoever finishes Phase 1 first starts this; combine into one script/module that outputs a clean tabular/array dataset: `(thermal_features_or_frames, sem_features, target_width_or_distribution)` per x-position per track
- This is the contract between "data" and "modeling" — write it once you're both confident the data is right

---

## Phase 3 — Modeling (split by approach, parallel)

**Person A — thermal-based model**
- Branch: `feature/thermal-model`
- Build a baseline model using only thermal features → predict target
- Start simple (hand-crafted features + gradient boosting/regression) before jumping to CNN/3D-CNN

**Person B — SEM + fusion model**
- Branch: `feature/sem-fusion-model`
- Build SEM-only baseline, then a fusion model combining thermal + SEM
- Add the probabilistic head (predict mean + uncertainty, not just a point estimate) — this is what makes it match the "probabilistic" framing in the README

**Merge point:** Compare thermal-only vs SEM-only vs fusion on held-out track(s) — this comparison is itself a nice portfolio result.

---

## Phase 4 — Validation & Writeup (both)

- Branch: `feature/leave-one-track-out-cv`
- Implement leave-one-track-out cross-validation (whoever isn't finishing Phase 3 modeling can start this while the other wraps up)
- Branch: `feature/results-writeup`
- Both: notebook/report comparing models, visualizing predictions vs ground truth, error analysis

---

## Quick reference: branch naming convention

```
feature/<short-description>   → new functionality
fix/<short-description>       → bug fixes
experiment/<short-description> → model experiments that might not pan out
```

Keep `experiment/*` branches for wilder model attempts — no obligation to merge those into `main`, useful for trying things without cluttering the main line of work.
