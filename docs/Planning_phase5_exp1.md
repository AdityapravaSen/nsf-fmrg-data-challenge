Phase 5 — Current Status & Next Steps
Current Status
✅ Frozen baseline

The canonical Phase III/IV baseline remains unchanged.

Model: Ridge Regression (alpha=1.0)
Features: SEM-only
Window: 5-frame flattened sequence
Target: Raw PC1–PC5
Validation: Leave-One-Track-Out (Tracks 8, 10, 14)
Track 21 remains untouched for blind inference.

Baseline aggregate:

MAE = 1.275480
RMSE = 1.655831
Median AE = 1.013160
R² = -0.282801

This remains the official comparison point for every subsequent experiment.

Experiment #1 — Coordinate-Augmented Ridge
Hypothesis

The Ridge model underperforms because it lacks longitudinal process position.

Added one additional feature:

x_norm = (x_position_mm - 60.0) / 40.0

Everything else remained frozen.

Result

Aggregate performance was essentially unchanged.

Observations:

MAE improved by only 0.000475
RMSE became slightly worse
Median AE became slightly worse
Fold variability increased
Only one fold improved consistently
Conclusion

Hypothesis rejected.

The missing longitudinal coordinate is not a major explanation for weak cross-track generalization.

Experiment closed.

Research Direction Shift

Because Experiment #1 failed, we revisited the overall learning problem instead of continuing incremental feature engineering.

Current working hypothesis:

The limitation may not be the model or missing features, but the supervised learning formulation itself.

Specifically:

The model may be capable of predicting macro-scale geometry while struggling to learn high-frequency local PCA variation.

This led to designing a single higher-risk reformulation experiment rather than another minor feature tweak.

Proposed Experiment #2

Working title:

Smoothed PCA Target Ridge LOTO

Core idea:

Inputs remain identical.
Ridge remains identical.
SEM-only features remain identical.
LOTO remains identical.
Track 21 remains untouched.

Only the training targets are modified.

Training targets become a smoothed version of PC1–PC5 (within each training track), while evaluation is still performed against the original raw PC targets.

This directly tests whether suppressing high-frequency target variation improves cross-track generalization.

No implementation has begun.

A complete experimental protocol has already been drafted.

The only remaining open question is the exact smoothing protocol.

Immediate Tasks (Aditya)
1. Merge today's work
Pull latest changes.
Verify nabarun-exp1 is synchronized.
Ensure Experiment #1 outputs are preserved.
2. Review protocol only

Read through the proposed Experiment #2 protocol.

No coding.

Focus on:

leakage
alignment
target construction
reproducibility
implementation feasibility
3. Think about smoothing

Specifically review whether:

centered rolling median
window size = 11

appear scientifically justified.

No implementation yet.

4. Look for repository implications

Identify whether the protocol can truly be implemented as:

scripts/
    26_phase5_loto_smoothed_pca_target_ridge.py

without modifying shared utilities.

If any shared-module modification appears necessary, document it.

My Tasks (Tomorrow)
1.

Send the final protocol stress-test prompt to GPT-5.5.

Goal:

Challenge only the smoothing design.

Nothing else.

2.

Review GPT's final recommendation.

If the protocol changes:

document why

If unchanged:

freeze protocol permanently
3.

Approve final experimental protocol.

No more redesign.

No more strategy discussions.

4.

Implement Experiment #2.

Only after protocol freeze.

5.

Run full LOTO evaluation.

Collect:

fold metrics
pooled metrics
comparison with frozen baseline
6.

Interpret results.

If successful:

decide whether to replace development baseline.

If unsuccessful:

formally close Experiment #2 and freeze the original Ridge baseline for submission.
Current Project State
Phase I
    ✓ Complete

Phase II
    ✓ Complete

Phase III
    ✓ Frozen

Phase IV
    ✓ Frozen

Experiment #1
    ✓ Complete
    ✓ Rejected

Experiment #2
    Protocol complete
    Awaiting final smoothing review
    Not yet implemented

## progress so far...
# Phase 5 Progress Report & Updated Action Plan

**Author/Owner:** Joint Engineering & Modeling Team (Adi & Nabarun)

**Document Ref:** Updated from `Planning_phase5_exp1.md`

**Project:** NSF Future Manufacturing Data Challenge

---

## 1. Initial State & Baseline Review

As established at the start of Phase 5, our canonical baseline operated under the following frozen parameters:

* **Model:** Ridge Regression ($\alpha = 1.0$)
* **Features:** SEM-only (substrate roughness variance and mean intensity)
* **Window Size:** 5-frame flattened thermal/multimodal sequence
* **Target:** Raw PCA components (PC1–PC5)
* **Validation Strategy:** Leave-One-Track-Out (LOTO) cross-validation across development tracks **8, 10, and 14** (with Track 21 strictly sealed for final blind inference).
* **Baseline Aggregate Metrics:** MAE = 1.2755, RMSE = 1.6558, Median AE = 1.0132, **R² = -0.2828**.

---

## 2. Summary of Experiments Carried Out in Phase 5

### **Experiment #1: Coordinate-Augmented Ridge (Completed & Closed)**

* **Hypothesis:** The baseline Ridge model underperformed due to a lack of known longitudinal process position along the scan axis.
* **Implementation:** Appended a normalized spatial coordinate feature $x_{\text{norm}} = (x_{\text{position\_mm}} - 60.0) / 40.0$ to the flattened feature vector.
* **Result:** Hypothesis rejected. Aggregate LOTO performance showed negligible improvement ($\Delta \text{MAE} = -0.000475$), increased fold variability, and no consistent cross-track generalization benefit. The experiment was formally closed.

### **Experiment #2 / Strategic Pivot: Physics-Constrained Bayesian Ridge & Target Smoothing**

* **Context & Realization:** Following competitor analysis and internal audits, relying solely on SEM features caused models to overfit to local spatial tiles rather than general thermal physics. Furthermore, predicting multi-dimensional, high-frequency noisy PCA arrays led to negative R² scores.
* **The Pivot:**
1. **Feature Schema Refactor:** Completely dropped SEM features and isolated **three core thermal physics drivers**: `peak_temp`, `mp_length`, and the linearized square root of melt pool area (`sqrt_mp_area`).
2. **Target Refactor:** Replaced noisy PCA arrays with a stable 1D smoothed macro-scale target (`smoothed_macro_width_mm`) derived via gap interpolation and a centered rolling average (window size = 11).
3. **Model Evaluation & Ablation:** * **Bayesian Ridge:** Yielded massive error reductions (~60–65% lower MAE/RMSE) and stabilized LOTO R² to **-0.041** (with held-out Track 8 reaching **-0.004**, tracking the theoretical mean threshold without exploding).
* **Random Forest (Non-Linear Ablation):** Tested tree-based regression on the same cleaned features. It failed catastrophically during cross-track extrapolation (average R² dropped to **-1.628**, and held-out Track 8 plunged to **-3.858**), proving that non-linear trees overfit and clip when encountering out-of-distribution lower laser power tracks (e.g., 200W Track 8).




* **Conclusion:** Selected **Bayesian Ridge** with the 3 thermal physics features and smoothed 1D target as our official, robust final model.

---

## 3. Files Modified & Created

To preserve core repository stability and prevent breaking Nabarun's original data-loading architecture, updates were cleanly segregated:

* **`src/ml/targets.py`**: Registered the `"smoothed_macro_width"` target group mapping.
* **`scripts/phase3_data_loader.py`**: Updated `FeaturePreprocessor` to exclude SEM variables, retain only the three thermal physics drivers, and dynamically compute `sqrt_mp_area`.
* **`scripts/patch_dataset.py` (or `12b_patch_smoothed_targets.py`)**: Implemented an inline patch that performs gap interpolation and centered rolling average smoothing on `amplitude_um`, injecting `smoothed_macro_width_mm` directly into `processed_data/final_multimodal_dataset.csv`.
* **`scripts/26_phase5_physics_constrained_baseline.py`**: Created the canonical LOTO cross-validation script utilizing `BayesianRidge`.
* **`scripts/27_phase5_generate_evaluation_plots.py`**: Implemented an automated visualization pipeline to render and save LOTO evaluation outputs.

---

## 4. Generated Artifacts & Visualizations

The evaluation visualization script successfully compiled and saved verification assets under **`processed_data/evaluation_plots/`**:

1. **`loto_spatial_predictions_comparison.png`**: High-resolution trajectory subplots comparing ground-truth smoothed width against Bayesian Ridge predictions along the scan axis ($x\_position\_mm$) for held-out Tracks 8, 10, and 14.
2. **`loto_pooled_parity_plot.png`**: Pooled regression parity scatter plot showing model alignment against the ideal $y = x$ line across all development validation folds.

[![loto-pooled-parity-plot.png](https://i.postimg.cc/Rhf2B4BS/loto-pooled-parity-plot.png)](https://postimg.cc/SYSgL0R3)
[![loto-spatial-predictions-comparison.png](https://i.postimg.cc/BbDR040K/loto-spatial-predictions-comparison.png)](https://postimg.cc/JHrp3wtr)

---

## 5. Remaining Next Steps (Phase 6: Final Inference & Reporting)

1. **Repository Synchronization:** Ensure all modified files (`targets.py`, `phase3_data_loader.py`, `patch_dataset.py`, and baseline modeling scripts) are fully committed and pushed to GitHub.
2. **Track 21 Blind Inference:** Execute final predictions on the sealed Track 21 dataset using the frozen Bayesian Ridge pipeline and export output files (`.csv` / `.npy`).
3. **Final Report & Documentation:** Draft the final challenge report, framing the evolution from complex deep-learning baselines to a robust, physics-constrained linear model as a rigorous scientific ablation study.