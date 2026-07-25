# Scientific Strategy Report: Post-Experiment 30 Diagnosis

**Date:** 2026-07-24  
**Branch:** `nabarun-exp2-local-width`  
**Status:** Pre-registered stop condition reached — NO SIGNAL classification  
**Author:** Scientific strategy analysis  

---

## 1. Executive Conclusion

After 30 experiments across five project phases, the accumulated evidence supports one central finding:

> **Thermal melt-pool features carry process-level (between-track) information about final geometry but contain no detectable local (within-track) predictive signal for any width target tested so far.**

The positive pooled OOF R² = +0.098 from Experiment 30 is **entirely explained by between-track mean separation** — the model partially captures which *process condition* produced the track, not where within the track a spatial fluctuation occurs. All three within-track Pearson correlations are ≈ 0 (0.036, 0.055, 0.001).

However, **the target itself may be a substantial contributor to this failure**. Our `local_core_width_mm` target (Experiment 29) was formally rejected on geometric grounds before learnability testing. It suffers from p95 adjacent-anchor width jumps of 0.40–0.49 mm — spatial noise that exceeds the physical smoothness expected from a ~1 mm-scale deposited track. Analysis of the raw Wyko data reveals that a **finite-support density approach** produces a fundamentally smoother signal (p95 adjacent-column jumps of 0.07–0.08 mm vs. 0.22–0.31 mm for Candidate 1), consistent with the external team's reported success.

**The recommended next step is Experiment 31: a pre-modeling target validity audit of a finite-support density-envelope width**, extracted from columnar finite-fraction profiles without connected-component boundary decisions. This addresses the most information-rich hypothesis remaining: that our target extraction noise, not predictor poverty, is the primary reason for negative within-track R².

---

## 2. Repository Evidence Reviewed

### Primary Sources

| Source | Type | Key Content |
|--------|------|-------------|
| [Progress_nabarun.md](file:///Users/rick/Desktop/TAMU/NSF%20Challenge/docs/Progress_nabarun.md) | Progress log | Full experimental history, Exps 03–29 |
| [Progress_adi.md](file:///Users/rick/Desktop/TAMU/NSF%20Challenge/docs/Progress_adi.md) | Progress log | Thermal/SEM pipeline, Phase I–IV |
| [PhaseIII_Baseline_Results_Summary.md](file:///Users/rick/Desktop/TAMU/NSF%20Challenge/docs/PhaseIII_Baseline_Results_Summary.md) | Results summary | PCA target modeling results |
| [PhaseIII_LOTO_Model_Selection.md](file:///Users/rick/Desktop/TAMU/NSF%20Challenge/docs/PhaseIII_LOTO_Model_Selection.md) | Model selection | LOTO across 9 model×feature configs |
| [PhaseIII_Pipeline_Freeze.md](file:///Users/rick/Desktop/TAMU/NSF%20Challenge/docs/PhaseIII_Pipeline_Freeze.md) | Frozen protocol | Ridge SEM-only baseline contract |
| [28_local_width_extraction_audit.py](file:///Users/rick/Desktop/TAMU/NSF%20Challenge/scripts/28_local_width_extraction_audit.py) | Script | Candidate 1 longest-finite-run audit |
| [29_dominant_component_core_geometry.py](file:///Users/rick/Desktop/TAMU/NSF%20Challenge/scripts/29_dominant_component_core_geometry.py) | Script | 2D dominant component core geometry |
| [30_core_width_learnability_diagnostic.py](file:///Users/rick/Desktop/TAMU/NSF%20Challenge/scripts/30_core_width_learnability_diagnostic.py) | Script | BayesianRidge learnability diagnostic |

### Run Output Artifacts Inspected

| Run | Status | Key Outputs |
|-----|--------|-------------|
| `28_..._20260723_172003` | Complete | 10 falsification criteria, native_local_width.csv (60K rows), all_finite_runs_by_column.csv (120MB) |
| `29_..._20260724_152546` | Stopped (coord discrepancy) | data_discrepancy.json only |
| `29_..._20260724_174420` | **Complete (canonical)** | 18 criteria evaluated, REJECT TARGET, anchor_aggregated_width.csv, native_component_width.csv |
| `30_..._20260724_204318` | Complete | loto_fold_metrics.csv, loto_pooled_metrics.csv, per_fold_predictions.csv (1114 rows), 2 diagnostic PNGs |

### Verified Data Artifacts
- `final_multimodal_dataset.csv` (1600 rows × 43 columns, Tracks 8/10/14/21)
- Experiment 29 `anchor_aggregated_width.csv` (1200 rows × 14 columns)
- Experiment 29 `native_component_width.csv` (59,840 rows)
- Experiment 28 `native_local_width.csv` (59,840 rows)
- Experiment 30 `per_fold_predictions.csv` (1,114 rows)
- `scientific_audit_memo.md` — **does not exist**
- `scientific_review_report.md` — **does not exist**

---

## 3. What Experiments 28–30 Established

### Experiment 28: Candidate 1 Longest Finite Run Audit

**Definition:** For each native Wyko column, extract the longest contiguous finite (non-NaN) y-run and measure its physical y-extent as `local_width_mm`.

**Key findings:**
- Computability was high (94.5–97.9% of native columns)
- Extracted widths (T8: 0.83 mm, T10: 0.49 mm, T14: 0.48 mm) are far narrower than macro width (2.68–3.70 mm) — Candidate 1 measures the *measurable surface core*, not the full track
- High spatial noise: p95 adjacent-column width jumps of 0.22–0.31 mm
- 5–12% of columns have ambiguous primary/secondary run selection ($L_1/L_2 ≤ 1.5$)
- Boundaries aligned with height gradient structure in 70% of sampled cases
- **Verdict:** CONCERN across all tracks — computable but not accepted as target

### Experiment 29: Dominant 2D Component Core Geometry

**Definition:** Identify the largest 4-connected finite component in 2D Wyko heightmap, measure its column-wise y-extent, aggregate with nanmedian in ±0.10 mm windows at thermal anchor positions → `local_core_width_mm`.

**Key findings (canonical run `_174420`):**
- Dominant component covers 88–96% of x-scan, with dominance ratios 40–76× over next largest
- **REJECT TARGET** — 7 of 18 pre-registered criteria failed:
  - A2: Track 10 dominant x-extent = 88.38% (threshold ≥ 95%) — **FAIL**
  - B3: p95 anchor-anchor width jumps 0.40–0.49 mm (threshold ≤ 0.20 mm) — **FAIL all tracks**
  - D1: Track 10 valid-anchor fraction = 88.5% (threshold ≥ 90%) — **FAIL**
  - D2: 14–46 zero-valid-column anchors inside domain per track — **FAIL all tracks**

**Critical nuance:** At the *native column level*, the dominant component width has very low spatial noise (p95 adjacent jumps 0.036–0.040 mm). The high anchor-level jumps arise because:
1. The ±0.10 mm aggregation window sits on coverage gaps
2. The component has true local width variation of ~0.19–0.24 mm std
3. Coverage holes produce NaN anchors, and the jumps between valid neighbors amplify apparent spatial noise

**Positive evidence despite rejection:**
- Boundary gradient alignment ≥ 93% (physically meaningful edges)
- Systematic cross-track ordering: T8 (1.16) > T10 (0.94) > T14 (0.84) — monotonic with laser power

### Experiment 30: Core Width Learnability Diagnostic

**Setup:** BayesianRidge (default), 15 features (3 thermal × 5-frame windows), LOTO on Tracks 8/10/14, 1114 valid samples, Track 21 sealed.

**Results:**

| Fold | N | Target Mean | Target Std | MAE | RMSE | R² | Pearson r |
|------|---|------------|------------|-----|------|----|-----------|
| holdout_8 | 383 | 1.164 | 0.178 | 0.144 | 0.215 | **−0.459** | 0.036 |
| holdout_10 | 354 | 0.939 | 0.196 | 0.160 | 0.205 | **−0.095** | 0.055 |
| holdout_14 | 377 | 0.844 | 0.235 | 0.172 | 0.271 | **−0.339** | 0.001 |
| **Pooled** | **1114** | **0.984** | **0.245** | **0.158** | **0.233** | **+0.098** | **0.445** |

**Pre-registered decision:** NO SIGNAL (0 of 3 folds with R² > 0)

**Diagnostic observations from prediction data:**
- Predicted values collapse to narrow ranges: T8 predictions span [0.97, 1.08], T14 span [0.58, 0.85] — prediction std is 0.017–0.042 vs actual std of 0.18–0.24
- The model outputs near-constant predictions per fold with negligible spatial variation
- Residual-vs-x plots show structured spatial error patterns

---

## 4. What Signal Appears Learnable

### A. Between-track / process-condition variation — **LEARNABLE (moderate confidence)**

**Evidence for:**
- Thermal feature means are strongly separated between tracks: `peak_temp` has 62.7% between-track variance, `sqrt_mp_area` has 95.6%, `mp_length` has 96.3%
- Track-level target means are monotonically ordered with thermal features: T14 (lowest peak_temp, smallest melt pool) → T8 (highest peak_temp, largest melt pool) aligns with T14 (thinnest core width) → T8 (widest core width)
- BayesianRidge achieves pooled R² = +0.098 and Pearson r = 0.445 — partial track-level separation
- This ordering is physically consistent: higher laser power → larger melt pool → wider deposited track

**Evidence against:**
- Only 3 development tracks — the between-track "function" has only 3 support points
- Predictions partially anti-correlate with track mean positions (model predicts training mean, which is biased opposite to held-out track mean)
- Generalization to Track 21 (different process condition) is untested

**Confidence:** Moderate (70%). The physics is reasonable but 3 data points cannot establish a robust process-width relationship.

### B. Within-track low-frequency spatial variation — **UNKNOWN (insufficient evidence)**

**Evidence for:**
- 10–36% of within-track target variance resides in the 50-point smooth component (equivalent to 10 mm spatial scale)
- Target autocorrelation at lag-1 (0.2 mm) ranges from 0.40 to 0.61 — the target is not white noise, it has spatial structure
- Track 14 has the strongest low-frequency component (36% of variance in 10-pt smooth)

**Evidence against:**
- All within-track Pearson r ≈ 0 in Experiment 30 — thermal features predict none of this structure
- `sqrt_mp_area` and `mp_length` have >95% between-track variance and <5% within-track variance, making them nearly constant within a track
- `peak_temp` has 37% within-track variance but its within-track spatial pattern may not correlate with width spatial pattern
- **We have never tested a different target extraction that would preserve this component more cleanly**

**Confidence:** Low-moderate (40%). The low-frequency component exists in the target but we cannot determine from Experiment 30 alone whether it is physically caused by thermal variations or by an independent mechanism.

### C. Within-track high-frequency / local variation — **LIKELY UNLEARNABLE from available predictors**

**Evidence for:** None.

**Evidence against:**
- 65–85% of within-track target variance is in the high-frequency residual (above 10-pt smooth)
- Thermal features are nearly constant within a track (1–3% CoV for `sqrt_mp_area` and `mp_length`)
- Pearson r ≈ 0 within every track
- The high-frequency variation in `local_core_width_mm` may largely be extraction noise from the dominant-component approach (the target was formally rejected on geometric grounds)

**Confidence:** High (85%). The available thermal predictors lack the spatial bandwidth to predict column-to-column width fluctuations.

### D. Measurement / extraction noise — **SUBSTANTIAL (high confidence)**

**Evidence for:**
- Experiment 29 failed the B3 continuity criterion (p95 anchor jumps 0.40–0.49 mm vs 0.20 mm threshold)
- 5–12% of columns have ambiguous component selection (Experiment 28)
- Track 10 has 46 zero-valid-column anchors inside the Wyko x-domain
- Adjacent-anchor width jumps of 0.40+ mm occur despite the native component width having p95 jumps of only 0.036–0.040 mm → the aggregation procedure amplifies noise

**Evidence against:**
- The native column-level width is very smooth (p95 jumps 0.036–0.040 mm)
- Boundary edges align with 93–96th percentile height gradients — the underlying physical measurement is real

**Confidence:** High (90%). The target carries genuine physical information but the current extraction pipeline introduces substantial measurement noise at the anchor aggregation level.

### E. Deterministic coordinate trends — **WEAK / UNCERTAIN**

**Evidence for:**
- Adding `x_position_mm` (or `x_norm`) as a feature did not improve PCA target prediction (Experiment 35: R² −0.269 vs −0.283 without)
- However, we never tested coordinate features with a width target (only with PCA shape)

**Evidence against:**
- Within-track width trends may exist but would be confounded with process startup/shutdown effects that differ between tracks

**Confidence:** Low (30%). Coordinate trends might exist for some target definitions but cannot be reliably separated from process-condition effects with only 3 tracks.

### F. Substrate-related variation — **WEAK (moderate confidence)**

**Evidence for:**
- SEM-only models generalized better than thermal models for PCA shape targets (Ridge SEM-only: MAE 1.28, R² −0.28 vs Ridge Thermal-only: MAE 1.86, R² −1.39)
- RF feature importance showed 49.6% SEM, 50.4% thermal in combined models
- SEM captures pre-existing substrate conditions that persist through deposition

**Evidence against:**
- SEM features are coarse (2 features × ~7.5 mm tiles → ~11 unique SEM values per track)
- SEM-only R² was still negative (−0.28) — better than thermal but still no positive generalization
- Experiment 30 deliberately excluded SEM to isolate thermal signal
- For width (not shape) targets, substrate influence on boundary position is physically less clear

**Confidence:** Moderate (50%). Substrate context may matter for shape but evidence for its influence on track width is indirect.

---

## 5. What Signal Appears Unlearnable

| Component | Status | Basis |
|-----------|--------|-------|
| Within-track local width fluctuations from thermal features alone | **Unlearnable** | 0/3 folds R² > 0, Pearson r ≈ 0 in all folds (Exp 30) |
| PCA cross-section shape from any predictor combination | **Unlearnable** | Negative R² across all 9 model×feature configs in LOTO (Phase III) |
| PCA shape + coordinate augmentation | **Unlearnable** | R² = −0.269 (Exp 35) |
| `smoothed_macro_width_mm` from thermal physics features | **Unlearnable** | Mean fold R² = −0.042, all folds negative |
| Column-level stochastic NaN fragmentation pattern | **Unlearnable** | 20–29 NaN runs per column, governed by surface optics not thermal history |

---

## 6. Explanation of Positive Pooled R² vs. Negative Fold R²

### The R² Paradox — Complete Decomposition

The pooled OOF R² of +0.098 is computed as:
$$R^2_{\text{pooled}} = 1 - \frac{\text{SS}_{\text{res,pooled}}}{\text{SS}_{\text{tot,pooled}}}$$

where $\text{SS}_{\text{tot,pooled}}$ is computed against the **global** mean (0.984 mm), not the per-track mean.

**Variance decomposition of the pooled target:**

| Component | SS | Fraction |
|-----------|----:|---------|
| Between-track ($\sum n_k (m_k - \bar{m})^2$) | 20.46 | **30.5%** |
| Within-track ($\sum n_k s_k^2$) | 46.56 | 69.5% |
| **Total** | **67.02** | 100% |

30.5% of the pooled target variance is between-track mean separation — a component the model can partially capture through the thermal feature distribution without predicting any local spatial variation.

**How the model exploits this:**
The thermal features (especially `sqrt_mp_area` and `mp_length`, which have >95% between-track variance) carry a strong fingerprint of which track/process condition produced the data. When BayesianRidge is trained on two tracks and evaluated on a third, the held-out track's features differ systematically from the training distribution. The model's linear mapping projects this feature shift into a **partially correct offset** toward the held-out track's mean width.

Specifically:
- `peak_temp`, `sqrt_mp_area`, `mp_length` all have the same cross-track ordering as `local_core_width_mm` (T14 < T10 < T8)
- The model learns a positive linear association from training data
- When applied to a held-out track with different feature means, the prediction shifts in the correct direction for the between-track component

However, **within each track**, the prediction is essentially constant (prediction std = 0.017–0.042 vs actual std = 0.178–0.235). The within-track predictions carry no spatial information.

**Mathematical verification:**
A constant-per-fold predictor (training mean) would achieve pooled R² = −0.40 — *worse* than the actual model. So the model does better than blindly predicting the training mean. But this improvement comes entirely from the thermal features' between-track offset, not from within-track spatial prediction.

The model captures approximately:
$$\frac{R^2_{\text{pooled}}}{f_{\text{between}}} = \frac{0.098}{0.305} = 32.2\%$$
of the between-track variance, and **0%** of the within-track variance.

### Which Metric Should Govern?

For a **spatially varying local target**, **per-track R²** (or equivalently, held-out-track R²) is the scientifically correct metric. The pooled R² is misleading because:

1. It conflates process-condition identification with spatial prediction
2. A trivially constant prediction that merely identifies the process regime can achieve positive pooled R²
3. The competition asks for *local geometric variation*, not process-level classification

**Conclusion:** The NO SIGNAL classification is correct. The positive pooled R² does not indicate local predictive capability.

---

## 7. Target-Space Reassessment Table

| # | Target | Classification | Rationale |
|---|--------|---------------|-----------|
| 1 | Raw PCA / profile shape (pc1–pc5) | **ALREADY FALSIFIED** | Negative R² across all 9 model×feature configs in Phase III LOTO. 5 models, 3 feature groups, all failed. |
| 2 | `smoothed_macro_width_mm` | **ALREADY FALSIFIED** | All 3 fold R² negative (mean −0.042). Additionally, the ordering is *anti-correlated* with thermal features (T8 narrowest at 2.68 mm, T14 widest at 3.70 mm, while thermal features rank T8 > T14). Track 21 column is a constant dummy (5.733). Non-reproducible from code. |
| 3 | Largest contiguous finite-run width (Candidate 1) | **ALREADY FALSIFIED** | Experiment 28 showed 6–15% adjacent jumps > 0.2 mm, 5–12% ambiguous column selection, median width 0.48–0.83 mm inconsistent with macro width. Superseded by Experiment 29. |
| 4 | Dominant-component `local_core_width_mm` | **ALREADY FALSIFIED** | Experiment 29: REJECT TARGET (7/18 criteria failed). Experiment 30: NO SIGNAL (0/3 folds R² > 0). Failed on both geometric validity and learnability. |
| 5 | Gap-bridged finite-support envelope | **SCIENTIFICALLY DANGEROUS** | Gap-bridging parameters (bridge length, minimum run) are tunable knobs with no physical anchor. Risk of target engineering toward positive R² without physical justification. No pre-registered definition exists. |
| 6 | **NaN-density / finite-fraction transition width** | **HIGH PRIORITY** | Finite-fraction per column has dramatically lower spatial noise (p95 adj jumps 0.07–0.08 mm vs 0.22–0.31 mm for Candidate 1). Does not require connected-component boundary decisions. Consistent with external team clue. Anchor-aggregated means (T8: 1.23, T10: 0.96, T14: 0.95) preserve cross-track ordering. Must be validated before modeling. |
| 7 | Left/right boundary position separately | **WEAKLY SUPPORTED** | Experiment 28 showed left-boundary jumps (4–11%) and right-boundary jumps (4–10%). Separating boundaries doesn't resolve the fundamental extraction noise. May be useful as a diagnostic but not as a primary target. |
| 8 | Boundary deviation from track-level baseline | **STILL PLAUSIBLE** | Subtracting a per-track smooth baseline would isolate local fluctuations. But with only 3 tracks, the baseline estimation procedure could overfit. Risk of removing the between-track signal that is the only learnable component. |
| 9 | Low-frequency local width component | **STILL PLAUSIBLE** | 10–36% of within-track variance is in the 50-pt smooth component. But extracting this requires choosing a smoothing scale, which is a free parameter. Would need to demonstrate that thermal features predict this component specifically. |
| 10 | Probabilistic width / distribution descriptors | **STILL PLAUSIBLE** | Predicting a width distribution rather than a point estimate could capture the uncertainty inherent in the measurement. But this requires a well-defined target distribution, which we don't have yet. |
| 11 | Macro + residual decomposition | **SCIENTIFICALLY DANGEROUS** | Decomposing into "macro trend + residual" requires defining the macro component, which introduces the same free parameters as smoothing. The residual would likely be dominated by extraction noise. |
| 12 | Finite-fraction profile (columnar density) as a continuous target | **STILL PLAUSIBLE** | Instead of converting density to a width scalar, predict the finite-fraction value itself. This is well-defined, noise-resistant, and physically meaningful (probability of successful optical measurement). But may not correspond to what the competition means by "geometric variation." |

---

## 8. Assessment of the Other Team's NaN-Density Clue

### What They Reportedly Did

Used a "NaN-valley" or "density-valley" method to extract local width from Wyko heightmaps. Achieved MAE ≈ 0.119 mm with BayesianRidge using 3 thermal physics features ("linear3"), beating their constant baseline of MAE ≈ 0.204 mm.

### Critical Analysis

**Does density aggregation suppress disconnected finite fragments in a physically meaningful way?**

Yes. Our Experiment 28 data shows that a simple `finite_fraction × FOV` measure has p95 adjacent-column jumps of 0.07–0.08 mm — roughly **4× smoother** than Candidate 1 (0.22–0.31 mm). This is because:
- Connected-component selection requires a binary boundary decision at each column
- When NaN gaps split the main body, the selected component can jump between fragments
- Density aggregation averages over *all* finite pixels, naturally suppressing fragment-switching noise

**Does it estimate a macroscopic support envelope rather than instantaneous connected-component width?**

Likely yes. The density approach implicitly measures "what fraction of the y-FOV contains measurable surface" rather than "how wide is the single largest connected structure." This is inherently more robust because it doesn't require solving the component-identity problem that Experiments 06, 07, and 10 identified as fundamentally ill-posed.

**Would it preserve genuine spatial variation or mostly collapse toward a track-level mean?**

Our data shows it preserves meaningful spatial variation:
- Density-width CoV: T8 = 0.126, T10 = 0.247, T14 = 0.207
- Local-core-width CoV: T8 = 0.153, T10 = 0.209, T14 = 0.280
- The CoVs are comparable — density-width is not collapsing to a constant

**Is there repository evidence that supports it?**

Yes, multiple lines:
1. Experiment 28's `finite_fraction` column data already exists at native column resolution
2. The fraction varies spatially within each track (std 0.08–0.13 across columns)
3. The cross-track ordering matches thermal features (T8 > T10 ≈ T14)
4. Experiment 10 concluded that finite-support and validity metadata were *essential* descriptor properties — density is precisely this

**Could their reported MAE simply be low because their target variance is small?**

Possibly. Their constant baseline MAE ≈ 0.204 suggests a target std of roughly 0.25–0.30 mm (since MAE ≈ 0.8 × std for normal data). Our `local_core_width_mm` has similar pooled std (0.245 mm). So variance magnitudes are comparable. The question is whether their BayesianRidge MAE of 0.119 represents genuine within-track prediction or merely better between-track interpolation. Without their per-fold R² breakdown, we cannot tell.

**What metrics would we require?**

We would require:
1. Per-fold (held-out-track) R² > 0 for at least 1 of 3 folds
2. Within-track Pearson r significantly above zero
3. MAE improvement over the held-out-track mean baseline (not just the pooled mean)
4. No target parameter tuning after seeing predictive metrics

**How would we prevent copying a method merely because of reported numbers?**

By treating target extraction as a **pre-modeling validity audit** (Experiment 31) before any learnability diagnostic (Experiment 32). The density-width target must pass geometric validity criteria independently of its downstream R².

---

## 9. Appropriate Evaluation Metrics

### Primary Metric: Per-Fold Held-Out-Track R²

For a spatially varying local target, the model must demonstrate that it predicts variation *within* an unseen track, not merely between tracks. Per-fold R² directly measures this.

> [!IMPORTANT]
> If the within-track target variance is very small relative to measurement noise, R² can be negative even for a model that captures real signal. In this case, we should also examine improvement over a per-track constant baseline.

### Secondary Metrics

| Metric | Purpose | When Critical |
|--------|---------|---------------|
| **Per-fold MAE** | Absolute error scale, interpretable in mm | Always |
| **Per-fold MAE relative to target std** (MAE/σ) | Normalized predictive skill | When targets have different variances |
| **Per-fold Pearson r** | Whether predictions covary with actuals within a track | Always — R² can be negative due to bias while r > 0 indicates some signal |
| **Improvement over held-out-track mean baseline** | Does the model beat simply predicting the held-out track mean? | Critical for distinguishing between-track from within-track signal |
| **Pooled OOF R²** | Summary across all folds | Useful as context but **must not be the primary metric** |
| **Pooled OOF Pearson r** | Same caveat as pooled R² — inflated by between-track separation | Secondary |

### Metrics for Target Validity (Pre-Modeling)

| Metric | Purpose |
|--------|---------|
| p95 adjacent-anchor width jump | Spatial continuity of target |
| Anchor coverage fraction | Completeness of target extraction |
| Cross-track mean ordering | Physical plausibility |
| Boundary gradient alignment | Physical meaningfulness |
| Within-track CoV | Non-trivial variation preserved |

### On R² in Low-Variance Settings

If a target has extremely small within-track variance (e.g., std < 0.05 mm), then even physically real predictions can yield R² < 0 because:
$$R^2 = 1 - \frac{\text{MSE}}{\sigma^2_{\text{target}}}$$

When $\sigma^2_{\text{target}}$ is small, any prediction error (even from noise in the *target* itself) dominates. In such cases:
- Positive Pearson r with p < 0.05 would indicate signal even with R² < 0
- MAE improvement over baseline would be a better success criterion
- The target may be too close to a constant to be meaningfully "varying"

---

## 10. Identifiability Analysis

### Predictor Capability by Spatial Scale

| Predictor Source | Between-Track | Low-Frequency Within-Track (>5 mm) | High-Frequency Within-Track (<2 mm) | Physical Basis |
|-----------------|:---:|:---:|:---:|---------------|
| **Thermal melt-pool** (`peak_temp`, `sqrt_mp_area`, `mp_length`) | **Strong** (>95% between-track variance in area/length) | **Very weak** (<5% within-track variance in area/length; 37% for peak_temp but correlation with width undemonstrated) | **None** (thermal features nearly constant within track; spatial resolution 0.2 mm per frame) | Melt pool size set by laser power and scan speed (process-level parameters), with minor thermal fluctuations from substrate variability |
| **SEM substrate** (roughness_variance, mean_intensity) | **Moderate** (generalizes better than thermal for PCA shape) | **Coarse** (~7.5 mm tile resolution, only ~11 unique values per track) | **None** (resolution too coarse) | Substrate surface condition prior to deposition |
| **Scan coordinate** (`x_position_mm`) | **No** (prohibited as identity proxy) | **Possible** in principle (could capture startup/shutdown effects) but confounded with process condition | **No** | Position encodes physical location but not independent physical cause |
| **Temporal context** (5-frame windows) | **No additional** (within-track temporal variation is minimal) | **Negligible** | **None** | Adjacent frames sample 0.2 mm apart, nearly identical melt pool state |
| **Process condition** (laser power, scan speed) | **Strong** in principle but not available as explicit features; encoded in thermal feature means | **No** (constant within a track) | **No** | Determines the track-level process regime |

### Identifiability Assessment

> [!CAUTION]
> **The available predictors have almost no within-track spatial bandwidth.** `sqrt_mp_area` and `mp_length` have within-track CoV of 1.5–2.0%, meaning they are essentially constant along a single track. `peak_temp` has more within-track variation (CoV 2.0–5.3%) but there is no evidence that its spatial pattern correlates with local width spatial patterns.

**Conclusion:** Local width fluctuations at scales below ~5 mm are **fundamentally not identifiable** from the available thermal predictors. The predictors carry process-level information (which track/condition) but essentially no local spatial information.

**What remains identifiable:**
1. **Track-level mean width** — identifiable from thermal feature means (3 support points, low confidence)
2. **Low-frequency trends** — *possibly* identifiable if peak_temp spatial patterns correlate with width patterns, but this is undemonstrated
3. **Substrate-correlated variation** — *possibly* identifiable from SEM features at ~7.5 mm resolution

**Competition-valid prediction under identifiability constraints:**

The most scientifically honest prediction may be:
- A per-track mean prediction based on thermal features (process-level calibration)
- Plus a small spatial adjustment from SEM substrate context
- With explicit predictive uncertainty reflecting the unidentifiable within-track variance

This would be a *well-calibrated probabilistic prediction* rather than a point estimate that pretends to predict local fluctuations it cannot.

**However:** Before accepting this pessimistic interpretation, we must verify whether the **target extraction noise** is masking a real within-track signal. If a cleaner target (density-envelope width) has lower extraction noise and higher spatial autocorrelation, the same thermal features might show weak but positive within-track correlation that was previously obscured.

---

## 11. Single Recommended Next Strategy

### Experiment 31: Finite-Support Density-Envelope Width Audit

**Rationale:** The accumulated evidence points to target extraction quality as the most actionable remaining hypothesis. We have:

1. **Three falsified target formulations** where the common failure mode is negative within-track R²
2. **Evidence that the current target (`local_core_width_mm`) has high extraction noise** (p95 anchor jumps 0.40–0.49 mm, formally rejected on geometric grounds)
3. **Evidence from our own data that finite-fraction density is 4× smoother** than connected-component extraction (p95 column jumps 0.07–0.08 mm vs 0.22–0.31 mm)
4. **External evidence** that a density-based target achieves substantially lower MAE (0.119 vs our 0.158) with the same predictor family

**Why this is the highest-information next step:**
- If a density-envelope target passes geometric validity criteria AND shows positive within-track R², it simultaneously resolves the target quality question and the learnability question
- If it passes validity but fails learnability, it narrows the problem to predictor poverty (identifiability limit) and we can proceed to a competition submission with a well-calibrated process-level model
- If it fails validity, it eliminates the density hypothesis and focuses attention on the identifiability limit

**Hypothesis tested:** The failure of within-track R² in Experiments 30 and earlier is primarily caused by target extraction noise (disconnected-component boundary decisions) rather than by fundamental predictor poverty.

**Falsification criterion:** If a geometrically valid density-envelope target also produces NO SIGNAL in a learnability diagnostic, then the hypothesis is falsified and we accept that thermal predictors carry no within-track local information for any width target.

---

## 12. Exact Pre-Registered Protocol

### Experiment 31: Finite-Support Density-Envelope Width Audit

**This experiment has TWO STAGES. Stage A is target extraction and validity audit. Stage B is a learnability diagnostic using the SAME protocol as Experiment 30, applied to the new target. Stage B runs ONLY if Stage A passes.**

---

#### STAGE A: Target Extraction and Geometric Validity

**Input files:**
- Raw Wyko heightmaps: `data/Track 8/Heightmap_8.ASC`, `data/Track 10/Heightmap_10.ASC`, `data/Track 14/Heightmap_14.ASC`
- Thermal anchor positions: `processed_data/final_multimodal_dataset.csv` (column `x_position_mm`, first 1200 rows, tracks {8, 10, 14} only)
- **Track 21 heightmap: NOT loaded. NOT inspected. Sealed.**

**Target definition: `density_envelope_width_mm`**

For each native Wyko column $j$ at position $x_j$:

1. Compute the binary finite mask: $f(y_i, x_j) = \mathbb{1}[\text{isfinite}(Z(y_i, x_j))]$
2. Compute a 1D density profile by convolving $f$ with a uniform kernel of fixed half-width $h$ pixels:
   $$\rho(y_i, x_j) = \frac{1}{2h+1} \sum_{k=-h}^{h} f(y_{i+k}, x_j)$$
   (with boundary zero-padding)
3. Define the support envelope as the contiguous y-region where $\rho(y_i, x_j) \geq \tau$
   - If multiple contiguous regions exceed $\tau$, select the longest
4. Measure `column_density_width_mm` = $y_{\text{right}} - y_{\text{left}}$ of the selected region
5. Handle edge cases:
   - If no column position has $\rho \geq \tau$: `column_density_width_mm = NaN`
   - If the envelope touches both y-domain boundaries: flag but retain (Wyko FOV truncation)

**Locked parameters (fixed BEFORE any modeling, justified physically):**

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Kernel half-width $h$ | 25 pixels (~0.10 mm) | This is approximately 10% of the Wyko y-FOV (1.907 mm). It smooths over individual NaN gaps (which are typically 1–5 pixels from surface texture) without blurring the macroscopic support boundary. Matches the ±0.10 mm aggregation window used in Experiment 29. |
| Density threshold $\tau$ | 0.30 | A column with 30% or more of its local neighborhood being finite is considered "within the support envelope." This is conservative — the overall finite fractions are 48–63%, so threshold is well below the bulk density. Set at approximately half the minimum track-level finite fraction. |
| Aggregation half-window | 0.10 mm | Matches Experiment 29 for comparability. Aggregation uses nanmedian of `column_density_width_mm` values within $x_{\text{anchor}} \pm 0.10$ mm. |
| Minimum valid columns for anchor | 25 | Matches Experiment 29 for comparability. |

**Why these parameters are NOT tuned:** They are set from physical considerations (NaN gap scale, Wyko FOV, Experiment 29 window) before examining any downstream predictive metric. They will not be changed after seeing Stage B results.

**Stage A Outputs:**
1. `native_density_width.csv` — column-level extraction for all 3 dev tracks (~60K rows)
2. `anchor_aggregated_density_width.csv` — anchor-level aggregation (1200 rows)
3. Diagnostic figures: density profiles, boundaries vs x, actual-vs-predicted overlay with Exp 29 width

**Stage A Geometric Validity Criteria (adapted from Experiment 29):**

| Criterion | Threshold | Scope |
|-----------|-----------|-------|
| SA1: Valid extraction fraction (non-NaN column width) | ≥ 95% of native columns | Per track |
| SA2: Anchor valid fraction | ≥ 92% (higher than Exp 29's 90% due to expected better coverage) | Per track |
| SA3: p95 adjacent-anchor width jump | **≤ 0.20 mm** (same as Exp 29 locked threshold) | Per track |
| SA4: Mean density-width > 0 and finite | Required | Per track |
| SA5: Within-track std > 0 | Required (non-trivial variation) | Per track |
| SA6: Cross-track mean ordering consistent with Exp 29 | T14 ≤ T10 ≤ T8 (matching thermal feature ordering) | Global |
| SA7: Zero-valid-column anchors inside domain | ≤ 5 per track | Per track |

**Stage A Decision Rule:**
- **PASS:** All criteria SA1–SA7 satisfied for all tracks → proceed to Stage B
- **CONDITIONAL PASS:** SA3 failed but p95 < 0.30 mm (improvement over Exp 29's 0.40–0.49) → proceed to Stage B with explicit documentation of remaining noise
- **FAIL:** Any other criterion failure → STOP. Report findings. Do not proceed to Stage B.

---

#### STAGE B: Learnability Diagnostic

**Runs ONLY if Stage A passes (PASS or CONDITIONAL PASS).**

**Target:** `density_envelope_width_mm` from Stage A, anchor-valid rows only, unsmoothed.

**Predictors:** `peak_temp`, `sqrt_mp_area`, `mp_length` — centered 5-frame windows (15 features). Identical to Experiment 30.

**Model:** `sklearn.linear_model.BayesianRidge` with default hyperparameters. Identical to Experiment 30.

**Scaling:** `StandardScaler` fit on training fold only. Identical to Experiment 30.

**Validation:** LOTO across development Tracks 8, 10, 14. Identical to Experiment 30.

**Track 21:** Completely sealed. Not loaded. Not used for scaling, target extraction, or evaluation.

**Metrics computed per fold and pooled:**
- MAE, RMSE, Median AE, R², Pearson r, target mean, target std, n

**Additional diagnostic metrics (not in Experiment 30):**
- Per-fold MAE improvement over held-out-track mean baseline: $\Delta\text{MAE} = \text{MAE}_{\text{track-mean}} - \text{MAE}_{\text{model}}$
- Per-fold Spearman rank correlation (more robust than Pearson for non-normal targets)

**Baselines computed per fold:**
- Held-out-track mean predictor (R² = 0 by definition; serves as reference MAE)
- Grand-mean (pooled training mean) predictor

**Plots:**
1. Actual vs predicted per fold (with identity line)
2. Residual vs x_position_mm per fold
3. Target spatial profile vs predicted spatial profile per fold (overlay on same axes)

**Stage B Decision Rule (identical structure to Experiment 30):**

- **SIGNAL DETECTED:** Pooled OOF R² > 0 **AND** ≥ 1 individual fold R² > 0
- **WEAK SIGNAL:** ≥ 1 individual fold R² > 0 **BUT** pooled OOF R² ≤ 0
- **NO SIGNAL:** All 3 individual fold R² ≤ 0

**Additional acceptance gate for SIGNAL DETECTED:**
- At least 1 fold must have Pearson r > 0.10 (to ensure the R² is not from intercept correction alone)
- The fold with R² > 0 must have MAE strictly less than the held-out-track mean baseline MAE

---

## 13. Acceptance / Rejection Criteria

### If Stage A PASSES and Stage B returns SIGNAL DETECTED:

1. Accept `density_envelope_width_mm` as the competition target
2. Proceed to a BayesianRidge submission pipeline trained on all 3 dev tracks
3. Generate Track 21 blind predictions with calibrated uncertainty
4. Document the target extraction protocol for the competition report

### If Stage A PASSES but Stage B returns NO SIGNAL:

1. Accept that thermal predictors carry no within-track local signal for *any* physically meaningful width target
2. The competition submission should be a process-level calibration model:
   - Predict a constant width per track based on thermal feature means
   - Report calibrated uncertainty intervals reflecting within-track variance
3. This is scientifically honest — the predictors identify the process regime but not local fluctuations

### If Stage A FAILS:

1. The density-envelope approach does not produce a geometrically valid target
2. Two interpretations: (a) the density parameters need adjustment, or (b) Wyko finite-support topology genuinely cannot define a stable width
3. **Fallback:** Submit the process-level BayesianRidge model from the `smoothed_macro_width_mm` experiment (MAE ≈ 0.449, R² ≈ −0.042), acknowledging its limitations
4. Alternative: attempt a `finite_fraction` target directly (predicting the fraction, not converting to width) as a final experiment if time permits

---

## 14. Leakage and Target-Engineering Safeguards

1. **Track 21 sealed:** Heightmap not loaded in either stage. Target extraction uses only Tracks 8, 10, 14.

2. **Parameters locked before modeling:** $h = 25$ px, $\tau = 0.30$, aggregation window ±0.10 mm, min valid columns 25. These are fixed in the script before Stage B. No parameter sweeps.

3. **No target parameter tuning after seeing R²:** If Stage A passes, Stage B uses the exact target from Stage A without modification. If Stage B returns NO SIGNAL, we do not go back and adjust $h$ or $\tau$ to get positive R².

4. **No retrospective reclassification:** The decision rules are locked. NO SIGNAL means NO SIGNAL.

5. **track_id never used as a feature.** `x_position_mm` never used as a feature.

6. **Comparison to Experiment 30 baseline:** The same model, features, and validation protocol ensure that any difference in R² is attributable to the target change, not to modeling choices.

7. **One-shot protocol:** This is a single experiment with pre-registered stopping rules. We do not iterate on the target after seeing results.

---

## 15. Expected Information Gain and Deadline Feasibility

### Implementation Effort

| Task | Estimated Time |
|------|---------------|
| Write Experiment 31 script (Stage A + B) | 2–3 hours |
| Execute Stage A (native column extraction + aggregation) | 15–30 minutes compute |
| Review Stage A results and go/no-go decision | 30 minutes |
| Execute Stage B (if Stage A passes) | 5–10 minutes compute |
| Review Stage B results and final decision | 30 minutes |
| Generate competition submission artifacts (if SIGNAL DETECTED) | 1–2 hours |
| **Total** | **4–7 hours** |

### Expected Information Gain

| Scenario | Probability | Information Gained |
|----------|:-----------:|-------------------|
| Stage A PASS + Stage B SIGNAL DETECTED | 20–30% | High: we have a validated target and a working model. Competition submission proceeds with confidence. |
| Stage A PASS + Stage B NO SIGNAL | 40–50% | High: definitively establishes that thermal predictors lack within-track signal for any clean width target. Competition submission uses process-level model with honest uncertainty. |
| Stage A FAIL | 20–30% | Moderate: density approach eliminated. Narrows to process-level submission. |

### Probability of Material Competition Improvement

- **If SIGNAL DETECTED:** ~25% probability. Would improve from process-level constant prediction (MAE ~0.15–0.20) to spatially varying prediction with genuine within-track skill.
- **Overall probability of material improvement:** 5–8% (25% chance of signal × 20–30% chance of SIGNAL DETECTED).

### Time for Fallback

If Experiment 31 fails entirely (Stage A FAIL or Stage B NO SIGNAL), we still have time to:
1. Submit the existing BayesianRidge process-level model (already implemented)
2. Or submit a simple `density_envelope_width_mm` target with process-level BayesianRidge prediction (even without positive within-track R², the model provides a reasonable central estimate)

The fallback requires minimal additional work (1–2 hours) because the prediction pipeline already exists from the `smoothed_macro_width_mm` experiment.

---

## 16. What Happens If This Experiment Succeeds

If Experiment 31 returns **SIGNAL DETECTED** (Stage A passes, Stage B has ≥1 fold R² > 0 with Pearson r > 0.10):

1. **Lock the target extraction:** `density_envelope_width_mm` with $h=25$, $\tau=0.30$ becomes the competition target
2. **Extract Track 21 target** using the identical extraction protocol (geometry only — no model labels leak)
3. **Train final model** on all 3 development tracks using BayesianRidge with the same 15 thermal features
4. **Generate blind Track 21 predictions** with calibrated uncertainty (BayesianRidge provides posterior variance)
5. **Write competition report** documenting:
   - The experimental trajectory from PCA through density-envelope
   - The NO SIGNAL results for previous targets (scientific honesty)
   - The SIGNAL DETECTED result for density-envelope
   - The physical interpretation: thermal features identify process-level width regime, density-envelope captures physically meaningful local variation with low extraction noise

---

## 17. What Happens If This Experiment Fails

If Experiment 31 returns **NO SIGNAL** (or Stage A fails):

1. **Accept the identifiability limit:** Thermal predictors carry process-level information only
2. **Competition submission strategy:**
   - Train BayesianRidge on the best available clean target (density-envelope width if Stage A passed, otherwise `local_core_width_mm`)
   - Report predictions as: per-track mean estimate ± calibrated uncertainty
   - The uncertainty intervals should reflect the within-track variance that the model cannot predict
3. **Competition report emphasizes:**
   - Rigorous falsification of 4+ target formulations
   - Clear identification of the between-track vs within-track signal decomposition
   - Process-level calibration as the scientifically honest prediction
   - The identifiability limit as a genuine scientific finding (not all local geometry is predictable from the available in-situ measurements)
4. **No further target experiments:** We do not iterate after a second NO SIGNAL result

---

## 18. Final GO / NO-GO Recommendation

### **GO — Proceed with Experiment 31.**

**Rationale:**

1. **High information density:** This single experiment resolves the most important remaining question (target quality vs predictor poverty) regardless of outcome
2. **Low implementation cost:** 4–7 hours total, well within deadline constraints
3. **Pre-registered stopping:** No risk of unbounded iteration — the protocol has locked accept/reject criteria
4. **Builds on existing infrastructure:** Uses the same data, same model, same validation framework as Experiment 30
5. **Worst-case outcome is still useful:** Even NO SIGNAL gives us a defensible competition submission with honest uncertainty quantification
6. **Best-case outcome is substantially better:** SIGNAL DETECTED would be the first positive within-track generalization result in the project's history

**The experiment should be implemented and executed as a single atomic unit with no intermediate design changes.**

> [!IMPORTANT]
> **Do not begin implementation until this protocol is reviewed and approved.** The protocol is locked once approved. No modifications are permitted after seeing Stage A results (except the pre-registered CONDITIONAL PASS → Stage B pathway).
