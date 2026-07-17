# Phase 3 Plan — Multimodal Predictive Modeling

**Project:** NSF Future Manufacturing Data Challenge  
**Phase:** Phase III  
**Status at start:** Phase I and Phase II complete  
**Primary dataset:** final_multimodal_dataset.csv  
**Sealed evaluation track:** Track 21  
**Planning assumption:** Geometry descriptor, descriptor implementation, descriptor integration, and merge validation are frozen.

---

## 1. Current repository state

The repository has completed the data-alignment and geometry-representation phases required before predictive modeling.

### Completed work

#### Phase I — data understanding and multimodal alignment

Person A completed the thermal and SEM alignment workflow.

The repository now contains an aligned thermal/SEM master table:

```text
processed_data/phase1_unified_master.csv
```

This table provides one row per thermal-frame anchor position, with:

- `track_id`
- `frame_index`
- `x_position_mm`
- thermal melt-pool features
- SEM substrate-context features
- common physical x-axis alignment

The shared x-axis spans approximately 20–100 mm, with 400 thermal-frame anchors per track.

#### Phase II — geometry descriptor design and integration

Person B completed the height-map geometry descriptor workstream through Experiment 14.

The completed Phase II state includes:

- geometry descriptor definition frozen,
- PCA representation selected for normalized cross-sectional shape,
- descriptor implementation complete,
- descriptor rows generated for Tracks 8, 10, and 14,
- descriptors merged into the multimodal anchor table,
- merge validation complete,
- Track 21 preserved as sealed.

The integrated dataset is:

```text
processed_data/final_multimodal_dataset.csv
```

The Phase II merge validation confirmed:

- 1600 total rows,
- 400 rows per track,
- Tracks 8, 10, 14 have descriptor fields populated according to validity rules,
- Track 21 rows are preserved but geometry descriptor fields remain uncomputed / `NaN`,
- no duplicated join keys,
- one-to-one descriptor merge for development tracks,
- anchor alignment with phase1_unified_master.csv.

### What remains

The remaining project work is Phase III:

> Build, evaluate, and interpret baseline multimodal predictive models that use thermal and SEM inputs to predict the frozen local geometry descriptors.

Phase III should not reopen the geometry-definition question. The objective is now modeling, evaluation, reporting, and final challenge preparation.

---

## 2. Phase 3 objective

The scientific objective of Phase III is to determine how well local post-process geometry descriptors can be predicted from aligned in-situ and pre-/near-surface observations.

The modeling problem should be framed as:

```text
Inputs
  ↓
Prediction
  ↓
Evaluation
```

### Inputs

Input features come from the frozen multimodal table and should be restricted to information available before or during deposition.

Primary input groups:

1. **Thermal melt-pool features**
   - `peak_temp`
   - `mean_temp`
   - `mp_area_px`
   - `mp_centroid_x`
   - `mp_centroid_y`
   - `mp_length`
   - `mp_width`

2. **SEM substrate-context features**
   - `sem_tile_index`
   - `x_start_mm`
   - `x_end_mm`
   - `substrate_roughness_variance`
   - `substrate_mean_intensity`

3. **Coordinate / anchor metadata**
   - `track_id`
   - `frame_index`
   - `x_position_mm`

Coordinate columns may be used for grouping, splitting, plotting, alignment checks, or controlled baseline comparisons. They should not be used in a way that leaks track identity into an unrealistic prediction setting unless explicitly reported as a coordinate-aware baseline.

### Prediction

The prediction targets are the frozen geometry descriptor fields generated during Phase II.

Primary target groups:

1. **Normalized shape descriptor**
   - `pc1`
   - `pc2`
   - `pc3`
   - `pc4`
   - `pc5`

2. **Amplitude / signed geometry descriptors**
   - `amplitude_um`
   - `signed_elevation_um`

3. **Validity-aware modeling metadata**
   - `eligible`
   - `nonflat`
   - `pca_ready`
   - `finite_fraction`
   - `central_corridor_finite_fraction`
   - `substrate_side_finite_fraction`
   - `baseline_support_count`
   - `fallback_baseline_required`
   - `is_within_boundary_exclusion`
   - `shape_support_fraction_on_common_grid`
   - `retained_pca_grid_finite_fraction`
   - `normalization_status`
   - `shape_center_um_median`
   - `shape_scale_um_mad`

The descriptor validity fields are not new descriptors. They are part of the frozen Phase II dataset contract and must be preserved during preprocessing, model training, evaluation, and reporting.

### Evaluation

Evaluation should answer:

1. Can thermal + SEM features predict normalized profile shape?
2. Can thermal + SEM features predict local amplitude and signed elevation?
3. Are predictions reliable only where descriptors are valid?
4. Does performance differ across Tracks 8, 10, and 14?
5. Does the finalized frozen pipeline generalize to sealed Track 21?

The final evaluation on Track 21 should occur only after preprocessing, feature handling, model class selection, target handling, and reporting templates are frozen using Tracks 8, 10, and 14.

---

## 3. Dataset contract

Phase III must use the frozen merged dataset:

```text
processed_data/final_multimodal_dataset.csv
```

Do not recompute geometry descriptors during Phase III.

Do not modify:

- descriptor definition,
- PCA representation,
- target validity rules,
- Phase II merge logic,
- Track 21 sealed protocol.

### Join keys

The canonical row identity is:

- `track_id`
- `frame_index`
- `x_position_mm`

These keys must remain unique.

Any derived output table should preserve these columns so predictions can be audited against the original dataset.

### Feature columns

Recommended Phase III feature groups:

#### Thermal feature columns

- `peak_temp`
- `mean_temp`
- `mp_area_px`
- `mp_centroid_x`
- `mp_centroid_y`
- `mp_length`
- `mp_width`

#### SEM feature columns

- `sem_tile_index`
- `x_start_mm`
- `x_end_mm`
- `substrate_roughness_variance`
- `substrate_mean_intensity`

#### Anchor / grouping columns

- `track_id`
- `frame_index`
- `x_position_mm`

These should be retained for splitting, grouping, plotting, and leakage checks.

### Target columns

Primary prediction targets:

- `pc1`
- `pc2`
- `pc3`
- `pc4`
- `pc5`
- `amplitude_um`
- `signed_elevation_um`

Validity and metadata fields to preserve:

- `heightmap_x_mm`
- `heightmap_x_delta_mm`
- `heightmap_x_index`
- `is_within_heightmap_x_coverage`
- `eligible`
- `nonflat`
- `pca_ready`
- `regime_id`
- `finite_fraction`
- `central_corridor_finite_fraction`
- `substrate_side_finite_fraction`
- `baseline_support_count`
- `fallback_baseline_required`
- `is_within_boundary_exclusion`
- `shape_support_fraction_on_common_grid`
- `retained_pca_grid_finite_fraction`
- `normalization_status`
- `shape_center_um_median`
- `shape_scale_um_mad`
- `_descriptor_matched`

### Descriptor validity

Not every row has valid target values. This is expected and scientifically meaningful.

Phase III should distinguish between:

1. rows with valid amplitude / signed-elevation targets,
2. rows with valid PCA targets,
3. rows outside height-map support,
4. rows with insufficient normalization support,
5. rows where descriptors are intentionally unavailable.

Required rule:

> Do not silently fill target values that are missing because the descriptor is invalid.

For PCA target modeling, rows should be filtered using `pca_ready` and the non-null state of `pc1`–`pc5`.

For amplitude and signed-elevation modeling, rows should be filtered using the validity state implied by non-null target values and relevant support metadata.

### Track 21 protocol

Track 21 remains sealed until final evaluation.

Current final_multimodal_dataset.csv includes Track 21 feature rows but no descriptor targets.

Phase III should treat Track 21 as follows:

1. Track 21 may be carried through feature preprocessing only if preprocessing rules are already frozen.
2. Track 21 must not influence:
   - feature selection,
   - imputation strategy,
   - scaling decisions,
   - target transformations,
   - validation design,
   - model selection,
   - hyperparameter decisions,
   - threshold decisions,
   - plot interpretation,
   - success criteria.
3. Track 21 predictions should be generated only after the full modeling pipeline is frozen.
4. If Track 21 geometry targets are later produced or revealed for evaluation, they must be evaluated once using the frozen pipeline.

### Training and evaluation data

Development data:

- Track 8
- Track 10
- Track 14

Sealed evaluation data:

- Track 21

The recommended development validation strategy should use only Tracks 8, 10, and 14.

Track 21 is not a validation track.

### Expected preprocessing

Preprocessing must be shared across all models and collaborators.

Expected preprocessing responsibilities:

- preserve row identity,
- preserve validity fields,
- define target-specific valid-row masks,
- handle feature missingness explicitly,
- standardize numeric features using training data only,
- avoid target leakage,
- avoid using descriptor validity fields as model inputs unless explicitly building a separate validity model,
- keep one preprocessing pipeline for all development tracks,
- save preprocessing metadata for reproducibility.

---

## 4. Recommended task split

The Phase III split should preserve the successful Phase II pattern:

- one collaborator owns feature-side preparation and interpretability,
- the other owns target/evaluation/reporting,
- both collaborate on baseline models and final scientific conclusions.

This minimizes merge conflicts because each person primarily works in different modules and output folders.

---

## Person A — feature engineering, preprocessing, and interpretability

### Primary responsibilities

Person A owns the input side of the modeling problem.

Tasks:

- define the canonical feature matrix from final_multimodal_dataset.csv,
- group thermal, SEM, and coordinate metadata columns,
- audit feature missingness,
- validate feature ranges by track,
- define feature preprocessing rules,
- implement leakage checks for feature columns,
- produce feature-distribution diagnostics,
- produce feature-correlation diagnostics,
- compare thermal-only, SEM-only, and thermal+SEM feature sets,
- lead input-side interpretability,
- prepare feature-importance and attribution summaries.

### Independent outputs

Person A can independently produce:

- feature schema document,
- feature validation report,
- preprocessing design note,
- feature missingness report,
- feature distribution plots,
- feature correlation plots,
- thermal-only / SEM-only / combined feature-set definitions.

### Reasoning

This matches Person A’s Phase I ownership of thermal and SEM alignment. It keeps the person most familiar with the input modalities responsible for preserving their physical interpretation during modeling.

---

## Person B — target engineering, evaluation framework, and prediction reporting

### Primary responsibilities

Person B owns the target and evaluation side of the modeling problem.

Tasks:

- define valid-row masks for each target group,
- define target groups:
  - PCA shape targets,
  - amplitude target,
  - signed-elevation target,
- preserve descriptor validity metadata,
- design track-aware validation protocol,
- define prediction tables,
- define metric tables,
- generate target-distribution diagnostics,
- evaluate model predictions by track and by validity state,
- produce prediction-vs-truth plots,
- prepare Track 21 reporting template,
- maintain consistency with Phase II descriptor assumptions.

### Independent outputs

Person B can independently produce:

- target schema document,
- target validity report,
- target missingness report,
- target distribution plots,
- evaluation protocol document,
- metrics template,
- prediction-output schema,
- Track 21 reporting template.

### Reasoning

This matches Person B’s Phase II ownership of geometry descriptors. It ensures descriptor validity and target semantics remain intact during modeling.

---

## Shared — baseline models, final evaluation, and paper integration

Shared tasks:

- agree on frozen preprocessing contract,
- agree on valid-row masks,
- agree on validation protocol,
- select baseline model families,
- run baseline comparisons,
- review evaluation metrics,
- freeze final pipeline before Track 21,
- generate final Track 21 predictions,
- prepare final figures and paper text.

Shared decisions should happen only at defined checkpoints, not continuously during implementation.

Recommended checkpoints:

1. feature/target schema freeze,
2. preprocessing freeze,
3. validation protocol freeze,
4. baseline model comparison review,
5. final pipeline freeze,
6. Track 21 evaluation,
7. paper figure selection.

---

## 5. Repository organization

Recommended Phase III structure:

```text
src/
  ml/
    datasets.py
    preprocessing.py
    targets.py
    models.py
    metrics.py
    plots.py
    evaluate.py
    train.py

scripts/
  15_phase3_dataset_audit.py
  16_phase3_baseline_modeling.py
  17_phase3_model_comparison.py
  18_phase3_track21_prediction.py
  19_phase3_final_reporting.py

processed_data/
  phase3/
    datasets/
    predictions/
    metrics/
    plots/
    reports/
    models/

paper/
  figures/
    phase3/
```

This is only a recommended organization. No code should be added until the collaborators agree on the plan.

### Purpose of each recommended module

#### `src/ml/datasets.py`

Dataset loading, schema checks, track filtering, row identity preservation.

#### `src/ml/preprocessing.py`

Shared preprocessing pipeline for feature columns.

#### `src/ml/targets.py`

Target grouping, valid-row masks, descriptor-validity handling.

#### `src/ml/models.py`

Baseline model wrappers and model registry.

#### `src/ml/metrics.py`

Regression metrics, grouped metrics, validity-aware metric summaries.

#### `src/ml/plots.py`

Standard plots for diagnostics, predictions, residuals, and final paper figures.

#### `src/ml/evaluate.py`

Reusable evaluation logic.

#### `src/ml/train.py`

Training orchestration for development-track models.

### Recommended output organization

```text
processed_data/phase3/datasets/
```

For frozen modeling tables and schema summaries.

```text
processed_data/phase3/predictions/
```

For per-row prediction CSV files preserving `track_id`, `frame_index`, and `x_position_mm`.

```text
processed_data/phase3/metrics/
```

For metric tables by target, model, feature set, track, and validation split.

```text
processed_data/phase3/plots/
```

For diagnostic and final evaluation figures.

```text
processed_data/phase3/reports/
```

For Markdown / JSON summaries of each modeling run.

```text
processed_data/phase3/models/
```

For saved trained model artifacts, if needed.

---

## 6. Experimental protocol

### Training data

Development training and validation should use only:

- Track 8
- Track 10
- Track 14

Track 21 must not be used for development.

### Validation strategy

Use track-aware validation.

Recommended validation principle:

> Evaluate whether a model trained on some tracks can generalize to a held-out development track.

With only three development tracks, the most important validation design is leave-one-track-out evaluation across Tracks 8, 10, and 14.

This avoids treating dense neighboring x positions as independent samples.

Within-track random splits should not be the primary reported result because adjacent rows are spatially correlated and can overstate generalization.

### Track 21 protocol

Track 21 is used only after:

- feature columns are frozen,
- target masks are frozen,
- preprocessing is frozen,
- model families are frozen,
- validation strategy is frozen,
- reporting format is frozen,
- no further tuning decisions remain.

Track 21 reporting should include:

- prediction file,
- preprocessing metadata,
- model metadata,
- feature schema,
- target schema,
- if target truth is available later, final metrics computed once.

### Baseline models

Baseline models should be simple, auditable, and scientifically interpretable.

Recommended baseline families:

1. mean / constant predictor,
2. coordinate-aware naive baseline,
3. linear regression-style baseline,
4. regularized linear baseline,
5. tree-based baseline,
6. multi-output baseline for PC1–PC5,
7. separate single-target baselines for amplitude and signed elevation.

No hyperparameters should be chosen in this planning document.

The goal is to establish a fair baseline suite, not to optimize a complex model prematurely.

### Feature-set comparisons

At minimum, compare:

1. thermal-only features,
2. SEM-only features,
3. thermal + SEM features,
4. optional coordinate-aware diagnostic baseline.

Coordinate-aware baselines must be clearly labeled as diagnostic because they may encode track position rather than physical sensor causality.

### Evaluation metrics

Recommended metrics for continuous targets:

- MAE,
- RMSE,
- median absolute error,
- $R^2$ where scientifically interpretable,
- target-standardized error where useful,
- per-track metrics,
- pooled development metrics,
- valid-row counts used for each metric.

For PCA targets:

- per-PC error,
- multi-output aggregate error across PC1–PC5,
- explained-variance-weighted aggregate error,
- reconstruction-oriented summary if the frozen PCA metadata supports it.

For amplitude and signed elevation:

- MAE in micrometers,
- RMSE in micrometers,
- signed residual bias,
- residual spread by track.

For validity-aware reporting:

- number of eligible rows,
- number of `pca_ready` rows,
- number of rows excluded from each target evaluation,
- metric denominators by target and track.

### Required plots

#### Dataset and target diagnostics

- feature distributions by track,
- target distributions by track,
- missingness / validity heatmap,
- PCA target distributions,
- amplitude and signed-elevation distributions,
- feature-target correlation overview.

#### Model diagnostics

- predicted vs true scatter by target,
- residual vs x-position,
- residual by track,
- residual by validity state,
- prediction trajectories along x for each track,
- comparison of thermal-only, SEM-only, and combined models,
- feature-importance / interpretability plots where supported.

#### Final paper figures

Candidate final paper figures:

- modeling dataset schematic,
- feature/target contract diagram,
- validation protocol diagram,
- per-track prediction trajectory figure,
- predicted vs observed descriptor figure,
- model comparison summary,
- Track 21 prediction figure.

### Required reports

Each modeling run should produce:

- run configuration summary,
- input dataset path,
- feature columns,
- target columns,
- valid-row masks,
- train/evaluation tracks,
- preprocessing summary,
- model family,
- metrics table,
- plots,
- prediction file path,
- notes on excluded rows and validity states.

### Success criteria

Phase III is successful if the team produces:

1. a reproducible frozen modeling pipeline,
2. baseline predictions for all frozen descriptor targets,
3. track-aware development validation,
4. clear comparison of feature groups,
5. validity-aware metrics,
6. interpretable model diagnostics,
7. sealed Track 21 predictions generated only after pipeline freeze,
8. final figures and tables suitable for the challenge report / paper.

Success should not require perfect prediction. The goal is defensible, reproducible, leakage-free predictive modeling.

---

## 7. Deliverables

### Dataset deliverables

- Phase III modeling dataset summary
- feature schema
- target schema
- valid-row mask summary
- preprocessing metadata
- train/evaluation split metadata

### Model deliverables

- baseline model artifacts, if saved
- model configuration summaries
- feature-set comparison results
- target-specific model results
- multi-output PCA model results
- amplitude model results
- signed-elevation model results

### Prediction deliverables

Prediction CSV files preserving:

- `track_id`
- `frame_index`
- `x_position_mm`
- target name
- ground truth when available
- prediction
- residual when available
- model name
- feature set
- validity state

Required prediction outputs:

- development-track cross-validation predictions,
- final development-trained predictions,
- Track 21 predictions after freeze.

### Metric deliverables

- metrics by target,
- metrics by track,
- metrics by feature set,
- metrics by model,
- metrics by validity mask,
- aggregate metric summary.

### Plot deliverables

- feature diagnostic plots,
- target diagnostic plots,
- prediction-vs-truth plots,
- residual plots,
- x-position trajectory plots,
- feature-importance plots,
- final paper-ready figures.

### Report deliverables

- Phase III dataset audit report,
- baseline modeling report,
- model comparison report,
- final Track 21 prediction report,
- final challenge / paper figure list.

---

## 8. Dependencies

### Person A can complete independently

Person A can begin immediately on:

- feature-column inventory,
- feature schema draft,
- feature missingness audit,
- feature distribution plots,
- feature correlation plots,
- preprocessing design,
- feature-set definitions.

These tasks require only the merged dataset and do not require new target decisions.

### Person B can complete independently

Person B can begin immediately on:

- target-column inventory,
- descriptor validity audit,
- valid-row mask definitions,
- target missingness report,
- target distribution plots,
- evaluation metric design,
- prediction-output schema,
- Track 21 reporting template.

These tasks rely on the frozen Phase II descriptor contract and do not require new feature engineering.

### Tasks requiring both collaborators

The following require joint agreement:

- final feature schema freeze,
- final target mask freeze,
- preprocessing pipeline freeze,
- validation protocol freeze,
- baseline model list,
- model comparison interpretation,
- final Track 21 pipeline freeze,
- final paper figures and claims.

### Tasks blocked until Phase II merge into main

If Phase II has not yet been merged into `main`, the following are blocked on that merge:

- building Phase III modules against `main`,
- freezing the Phase III dataset path on `main`,
- running final modeling scripts from `main`,
- generating official prediction files,
- generating official metric tables,
- generating final paper figures,
- Track 21 prediction workflow.

The following are not blocked by the merge and can proceed from the completed Phase II branch:

- planning,
- schema review,
- feature inventory,
- target inventory,
- draft plotting specifications,
- draft evaluation protocol,
- draft report templates.

---

## 9. Guardrails

Phase III must carry forward the project rules established during Phases I and II.

### Geometry and descriptor guardrails

- Do not modify frozen descriptors.
- Do not recompute descriptors unless explicitly performing a reproducibility check.
- Do not redefine the target.
- Do not introduce new geometric descriptors.
- Do not reopen Experiments 03–14.
- Preserve descriptor validity fields.
- Preserve explicit `NaN` states.
- Do not silently fill invalid target values.
- Do not treat missing descriptor values as ordinary numeric zeros.

### Track 21 guardrails

- Track 21 remains sealed until final evaluation.
- Do not use Track 21 for feature selection.
- Do not use Track 21 for preprocessing decisions.
- Do not use Track 21 for model selection.
- Do not use Track 21 for hyperparameter tuning.
- Do not inspect Track 21 target descriptors before final evaluation.
- Generate Track 21 predictions only after the pipeline is frozen.

### Modeling guardrails

- Use one shared preprocessing pipeline.
- Fit preprocessing only on training data.
- Avoid data leakage from target or validity fields into feature matrices.
- Avoid per-track tuning.
- Avoid track-specific thresholds or hacks.
- Report all row exclusions.
- Report metric denominators.
- Preserve row identity in every output.
- Use track-aware validation as the primary development evaluation.
- Do not treat neighboring x positions as statistically independent samples.
- Keep experiments reproducible and auditable.

### Collaboration guardrails

- Person A owns feature-side changes.
- Person B owns target/evaluation-side changes.
- Shared files should be minimal and agreed upon before editing.
- Outputs should be written into timestamped or clearly named Phase III folders.
- Final claims should be based on frozen validation results, not ad hoc visual inspection.

---

## 10. Recommended Phase III start sequence

### Step 1 — Freeze schemas

Person A freezes the feature schema.  
Person B freezes the target and validity schema.  
Both approve the combined dataset contract.

### Step 2 — Freeze preprocessing and masks

Person A defines preprocessing.  
Person B defines valid-row target masks.  
Both confirm no leakage.

### Step 3 — Run development-track validation

Use Tracks 8, 10, and 14 only.

Primary evaluation should be track-aware.

### Step 4 — Compare baselines

Compare thermal-only, SEM-only, and thermal+SEM models.

Report per-target and per-track metrics.

### Step 5 — Freeze final pipeline

No further changes after this checkpoint except bug fixes that are documented.

### Step 6 — Generate Track 21 predictions

Apply the frozen pipeline to Track 21 feature rows.

Do not tune after seeing Track 21 outputs.

### Step 7 — Produce final report and figures

Summarize:

- dataset contract,
- modeling protocol,
- validation results,
- Track 21 predictions,
- limitations,
- scientific interpretation.