# Project Phases & Layout

The overarching goal is to predict local geometric variation of the final laser track using in-situ thermal images and surrounding substrate context. Nabarun's pivot requires the project to be structured into three distinct phases.

* Phase 1: Data Understanding, Alignment, & Scientific Discovery (Completed).

* Phase 2: Geometry Representation Design & Dataset Unification (Current).

* Phase 3: Multimodal Learning & Predictive Modeling (Future).

# Phase 0 — Setup (both, do together first)

| Task | Branch | Who |
|---|---|---|
| Download Zenodo data, verify folder layout matches README | — (no branch, just local setup) | Both |
| Run `01_starter_code_loading_and_visualization.ipynb`, `02_starter_code_loading_and_visualization.ipynb` end-to-end, confirm it works on your machines | — | Both |

---

# Phase 1 — Data Understanding & Alignment (blocks everything else)

**Person A — thermal + SEM side**
- Verify `extract_final_thermal_frames` laser on/off detection actually looks right on all 4 tracks (visually, frame-by-frame)
- Confirm SEM tile → x-position mapping is correct (remember: reversed numbering)
- Write a function that maps thermal frame index ↔ SEM tile ↔ common x-coordinate — this is the "Rosetta stone" the rest of the project depends on

**Person B — height map side**
- Verify `robust_plane_detrend` output looks sane on all 4 tracks
- Write the actual **target extraction**: from detrended `Z_mm`, extract per-x-position width, boundary position, contour deviation, roughness (this is the big missing piece from the README)

**Merge point:** Both branches PR into `main` once each person's alignment/extraction is validated with plots. Do this before Phase 2 — you need a shared "aligned dataset" object both of you build models against.

# Phase 2 Plan: Dataset Assembly

## Where things actually stand

Phase 1 is done on both sides, but asymmetrically:

- **Adi (thermal + SEM):** Finished cleanly per the original Phase 1 scope — laser on/off validated, melt-pool features extracted, SEM reverse-tile mapping solved, and a merged `phase1_unified_master.csv` produced on a shared x-axis (20→100 mm, 400 thermal-frame anchors, 0.2 mm/frame).
- **Nabarun (height maps):** Went well beyond simple "extract width" — Experiments 03–11 disproved the naive positive-elevation width assumption, diagnosed threshold instability, ran a 2D object-identity audit, and landed on a **requirements-driven descriptor** (`Experiment 10`) validated with a first implementation candidate: PCA on normalized shape (`Experiment 11`).

## Goal of Phase 2

One clean, merged dataset: for every track (8, 10, 14 — **21 stays sealed**) and every shared x-position, a row containing thermal features, SEM features, and the geometry descriptor. This is the contract between "data" and "modeling" (Phase 3).

## Decisions to lock before building

| Decision | Options | Recommendation |
|---|---|---|
| Descriptor contents | width scalar vs. vector | Per Exp10/11: **PCA shape scores (first 5) + amplitude/relief + signed elevation + validity/finite-support flags + regime tag** — not a single width number |
| Merge grid | Adi's 0.2 mm thermal-frame grid vs. Nabarun's various audit grids (0.2/1.0/native ~0.004 mm) | Standardize on **Adi's 400-frame thermal anchor grid**, since it's the coarsest shared modality resolution and Phase 3 models will consume it anyway |
| Track 10 regimes | Force one global descriptor vs. allow regime tags | Allow regime tag (per Exp09 low-coherence intervals) — do not force one template |
| PCA component count | 3 vs 5 vs variance-target-driven | 5, per Exp11 findings, unless variance table says otherwise |

## Task split

### Nabarun — "Experiment 12: Descriptor Implementation"
Branch: `feature/descriptor-implementation`

- Freeze the descriptor definition (stop iterating on requirements — that phase is closed)
- Implement one function: given `track_id` + arbitrary `x_mm`, return `{pc1..pc5, amplitude_um, signed_elevation_um, eligible, nonflat, regime_id}`
- **Critical:** must evaluate at Adi's exact 400 thermal-frame x-anchors per track, not just the old representative x-list (25/35/.../95 mm) or dense 1 mm grid
- Emit `geometry_descriptors_track_{id}.csv` — one row per anchor, explicit NaN/ineligible flags preserved (no silent filling)
- Keep Track 21 sealed; one shared config across 8/10/14, no per-track tuning

### Adi — Dataset Merge
Branch: `feature/dataset-builder` (per original Planning.md)

- Confirm `phase1_unified_master.csv` x-anchors line up with what Nabarun will emit (flag any drift immediately — don't silently interpolate)
- Once descriptor CSVs land, join per track on `x_mm` (exact join preferred over `merge_asof` here since anchors should now match exactly)
- Produce `phase2_unified_dataset.csv`: thermal + SEM + geometry descriptor per track per anchor

### Both — Merge-point validation
- Plot thermal melt-pool area, SEM roughness, PC1 score, and amplitude vs. x for all 3 tracks on one figure per track
- Specifically check that Track 10's regime transitions (from Exp09, ~78–97 mm) show up as a visible discontinuity in the merged view too — that's your first real multimodal sanity check
- Confirm no Track 21 file was touched anywhere in the merge

## Deliverables checklist

- [ ] `geometry_descriptors_track_8/10/14.csv`
- [ ] `phase2_unified_dataset.csv`
- [ ] Validation plots (per-track, 4-panel: thermal / SEM / PC1 / amplitude vs x)
- [ ] Short note confirming anchor-grid alignment had no unexplained mismatches
- [ ] PR merge into `main` before starting Phase 3

## Guardrails carried forward (don't relitigate these)

- No virtual environments; use Homebrew Python 3.11 for Nabarun's side
- Track 21 sealed until method is fully frozen
- No global NaN filling or cross-gap interpolation
- One shared configuration across tracks — no per-track tuning
- Don't re-open the "which boundary definition is best" question — that's closed; Phase 2 is about *implementing* what Experiment 10 already specified, not re-auditing it
