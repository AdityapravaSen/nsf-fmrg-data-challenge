# Phase IV Plan: Sealed Evaluation & Scientific Validation

> **Guiding Principle for Phase IV:**
> *Every future change must answer a specific scientific question.*
> We are shifting from *"Can we build a scientifically defensible pipeline?"* to *"What does this pipeline actually tell us about the physics?"* 

## Roadmap Overview

* **IV-A:** Frozen evaluation (Guaranteed)
* **IV-B:** Scientific analysis (Guaranteed)
* **IV-C:** Decision checkpoint (Guaranteed)
* **IV-D:** Version 2 development (Optional — only if scientifically justified)

---

# Current Phase IV Progress

## ✅ Completed

* [x] Phase III pipeline frozen.
* [x] Dedicated Phase IV inference-wrapper architecture finalized.
* [x] `scripts/25_phase4_baseline_evaluation.py` created.
* [x] Stage 1 implementation completed.
* [x] Stage 1 preflight executed successfully.
* [x] Dataset, Track 21, PCA model, and configuration successfully validated.
* [x] Timestamped Phase IV output directory creation implemented.
* [x] Stage 1 metadata generation implemented.

## 🚧 In Progress

* [ ] Stage 2 — preprocessing, scaling, and sequence generation.
* [ ] Stage 3 — model fitting and Track 21 prediction.
* [ ] Stage 4 — reconstruction, metrics, and final outputs.

---

# Task Split & Responsibilities

## 👤 Nabarun (Pipeline & Evaluation)

### Phase IV-A: Frozen Evaluation

Remaining work:

* [ ] Complete Stage 2 of `25_phase4_baseline_evaluation.py`
* [ ] Complete Stage 3 (Ridge fitting + Track 21 prediction)
* [ ] Complete Stage 4 (PCA reconstruction + metrics + exports)
* [ ] Execute the complete frozen baseline evaluation.
* [ ] Archive all outputs into the timestamped Phase IV directory.
* [ ] Produce a lightweight `PhaseIV_Baseline_Evaluation.md` containing only:

  * frozen configuration
  * objective metrics
  * generated figures
  * factual observations

(No interpretation.)

---

## 👤 Aditya (Independent Parallel Work)

### Documentation

* [ ] Continue building the final report structure.
* [ ] Prepare the presentation deck.
* [ ] Organize figures/placeholders for:

  * methodology
  * preprocessing
  * feature engineering
  * model selection
  * LOTO validation
  * Phase IV evaluation
* [ ] Verify that all experiment numbers, filenames, and references are consistent throughout the report.

### Pipeline Audit

Without waiting for Track 21 predictions:

* [ ] Independently review the frozen Phase III pipeline.
* [ ] Review the frozen model-selection results.
* [ ] Review the finalized evaluation methodology.
* [ ] Identify any documentation gaps or reproducibility issues.
* [ ] Create a checklist of figures and tables still required for the final report.

### Reproducibility

* [ ] Verify that every reported experiment has:

  * corresponding code
  * output directory
  * metrics
  * configuration
  * reproducible naming

* [ ] Prepare a list of any missing artifacts before final submission.

---

# 👥 Together

After the frozen baseline finishes:

## Phase IV-B — Scientific Analysis

Review together:

* performance on Track 21
* reconstruction quality
* residual behaviour
* descriptor performance
* localized failure modes
* drift along the scan direction
* comparison with historical LOTO performance

---

## Phase IV-C — Decision Checkpoint

Answer only three questions:

1. Is the frozen baseline already submission quality?
2. Can observed failures be explained scientifically?
3. Is there exactly one scientifically motivated hypothesis worth testing?

If the answer to (3) is **No**, the frozen baseline becomes the final submission.

---

# Phase IV-D (Optional)

Only if justified:

* [ ] Test exactly one scientific hypothesis.
* [ ] No model shopping.
* [ ] No architecture exploration.
* [ ] No parameter fishing.

Every modification must answer a specific scientific question.

---

# 🎯 Immediate Handoff Status

## Completed today (Nabarun)

* [x] Phase IV architecture finalized.
* [x] Evaluation wrapper approach finalized.
* [x] Stage 1 implementation completed.
* [x] Stage 1 preflight executed successfully.
* [x] All dataset and configuration validation checks passed.

## Tomorrow (Nabarun)

* [ ] Stage 2 implementation.
* [ ] Stage 3 implementation.
* [ ] Stage 4 implementation.
* [ ] Execute frozen Track 21 evaluation.

## Tonight (Aditya)

* [ ] Continue report writing.
* [ ] Continue presentation development.
* [ ] Audit reproducibility of all experiments.
* [ ] Review the frozen methodology and prepare any documentation/questions that can be addressed independently of the Track 21 results.

---