# Phase IV Plan: Sealed Evaluation & Scientific Validation

> **Guiding Principle for Phase IV:**  
> *Every future change must answer a specific scientific question.* 
> We are shifting from "Can we build a scientifically defensible pipeline?" to "What does this pipeline actually tell us about the physics?"

## Roadmap Overview
- **IV-A:** Frozen evaluation (Guaranteed)
- **IV-B:** Scientific analysis (Guaranteed)
- **IV-C:** Decision checkpoint (Guaranteed)
- **IV-D:** Version 2 development (Optional - only if justified)

---

## Task Split & Responsibilities

### 👤 Pipeline Owner (You)
**Phase IV-A: Frozen Evaluation** (Goal: Execute the frozen pipeline with zero modifications)

- [ ] **Freeze the Repository:** Create a `phase4_track21_baseline` branch or tag. Archive metrics, predictions, configs, plots, and the commit hash. This is "Frozen Baseline v1" and must never be altered.
- [ ] **Run Track 21 Preprocessing:** Complete data prep for Track 21.
- [ ] **Generate Track 21 Predictions:** Run the exact frozen pipeline to produce `predictions.csv` (or equivalent).
- [ ] **Compute Evaluation Metrics:** Run the existing metrics pipeline (MAE, RMSE, Median AE, R², etc.). Do not invent new metrics.
- [ ] **Generate Visualizations:** Create plots for the final report (e.g., predicted vs. actual PC coefficients, reconstructed profile overlays, residual histograms, error along x, profile comparisons).
- [ ] **Archive Outputs:** Save everything in a timestamped `phase4_track21_baseline/` directory.
- [ ] **Produce Phase IV Report:** Write a lightweight `PhaseIV_Baseline_Evaluation.md` containing the frozen config, metrics, figures, and objective facts/observations. **No conclusions or interpretations yet.**

### 👤 Scientific Reviewer (Adi)
**Parallel Work during IV-A** (Goal: Interpret results and prepare deliverables)

- [ ] **Independent Review:** Inspect Track 21 outputs with fresh eyes. Identify where it fails, which descriptors fail, if SEM outperforms thermal, and if error drifts.
- [ ] **Report Skeleton:** Begin drafting `Final_Report.pdf` (Executive Summary, Method, Results, Discussion, Limitations, Future Work).
- [ ] **Presentation Deck:** Start building the Challenge Presentation slides immediately. 
- [ ] **Systematic Error Identification:** Compare feature groups and prepare scientific interpretations of the pipeline's failures.

### 👥 Together (You + Adi)
**Phase IV-B: Scientific Analysis & Phase IV-C: Decision Checkpoint**

- [ ] **Conduct Scientific Analysis (IV-B):** Review outputs together. Answer questions like:
  - Does Track 21 resemble Track 8 or 10?
  - Which PCs fail and is the failure localized?
  - Is there systematic drift or error growth along x?
  - How do Thermal-only vs. SEM-only vs. Combined features perform?
- [ ] **Hold Decision Checkpoint Meeting (IV-C):** Answer three critical questions:
  1. *Is Frozen Baseline already submission quality?* (If YES: Submit. If NO: Continue).
  2. *Can we explain the failure?* (Must be a physics/data explanation, e.g., "insufficient temporal context," not just "high MAE").
  3. *Is there ONE hypothesis worth testing?* (If NO: Submit. If YES: Proceed to Phase IV-D).

---

## Phase IV-D: Version 2 Development (Optional)
*Only initiated if a clear hypothesis is identified in Phase IV-C.*

- [ ] **Test ONE Hypothesis:** Focus on a single, specific scientific question (e.g., "Increase temporal window" or "Incorporate long-range thermal features"). 
- [ ] **No Model Shopping:** Do not randomly try new algorithms (e.g., XGBoost, CNN, Transformer) without a concrete scientific hypothesis driving the change.

---

## 🎯 Immediate Checklist (Today's Handoff Goal)
To ensure Adi has concrete material to review tonight, aim to complete these items today:

- [ ] Frozen Phase III branch/tag created.
- [ ] Track 21 preprocessing completed.
- [ ] Track 21 predictions generated for all frozen baselines.
- [ ] Metrics computed.
- [ ] Visualizations generated.
- [ ] Outputs archived in a timestamped `phase4_track21_baseline/` directory.
- [ ] A short `PhaseIV_Baseline_Evaluation.md` written with the metrics and a few objective observations (no interpretation yet).