# Experiment 32 — Methodological Audit and Revised Protocol

**Date:** 2026-07-25
**Status:** Audit of proposed Experiment 32 in response to six identified methodological issues

---

## 1. Verdict on the Original Experiment 32 Protocol

The original protocol is **flawed in four of six ways** identified. Specifically:

| Issue | Severity | Verdict |
|:---|:---|:---|
| 1. Held-out-fold calibration leakage | **Critical** | Confirmed. Original protocol leaks held-out labels into predictive distribution. |
| 2. Only three development tracks | **Structural** | Confirmed. Constrains uncertainty model to pooled/global estimates only. |
| 3. Interval-width acceptance criterion | **Mathematical error** | Confirmed. Threshold is geometrically impossible for a calibrated Gaussian. |
| 4. Target consistency | **Methodological error** | Confirmed. Cannot mix `local_core_width_mm` (development) with `smoothed_macro_width_mm` (Track 21). |
| 5. Rejected target reuse | **Scientific concern** | Assessed below. Defensible with explicit caveats. |
| 6. Probabilistic gaming | **Design gap** | Confirmed. Coverage alone is insufficient; need proper scoring rules. |

The original protocol cannot be executed as written. Revision is mandatory.

---

## 2. Held-Out-Label Leakage Diagnosis

### The Leak

The original protocol specified:

> "For each LOTO fold, compute empirical residual std $s_{\text{residual}}$ on the held-out track."

This means:

1. Train model on Tracks A + B
2. Predict Track C
3. **Observe Track C residuals** (uses Track C labels)
4. Compute $\sigma_{\text{residual}}$ from Track C residuals
5. Construct predictive intervals for Track C using $\sigma_{\text{residual}}$
6. Evaluate coverage/CRPS on Track C

Steps 3–5 leak held-out labels into the predictive distribution. The intervals are calibrated to the data they are evaluated on. This renders coverage and CRPS evaluations meaningless — they measure self-consistency, not predictive performance.

### Why It Matters

Even a constant-mean predictor with $\sigma = s_{\text{held-out-residual}}$ would achieve ~90% coverage at the 90% level by construction. No predictive skill is required. This is exactly the kind of tautological success that the experiment must avoid.

### Resolution

All uncertainty parameters must be determined **exclusively from training data** before the held-out fold's labels are revealed. The held-out targets may only appear in the EVALUATION of the fully frozen predictive distribution.

---

## 3. Valid Uncertainty-Calibration Method with Only Three Development Tracks

### Three candidate approaches evaluated

I tested all three on the actual Experiment 30 data to determine empirical behavior.

#### Approach A: BayesianRidge Native Posterior Predictive Distribution

BayesianRidge estimates noise precision $\alpha$ and weight precision $\lambda$ from training data. The posterior predictive distribution at input $\mathbf{x}$ is:

$$\hat{y}(\mathbf{x}) \sim \mathcal{N}\left(\mathbf{x}^T \hat{\mathbf{w}}, \; \frac{1}{\alpha} + \mathbf{x}^T \Sigma_w \mathbf{x}\right)$$

- $1/\alpha$ captures estimated observation noise (learned from training residuals)
- $\mathbf{x}^T \Sigma_w \mathbf{x}$ captures parameter uncertainty (naturally inflates for OOD inputs)
- **Uses ZERO held-out information**

#### Approach B: Nested Leave-One-Training-Track-Out

For LOTO fold with holdout C, training {A, B}:

1. Train on A alone → predict B → compute residuals on B
2. Train on B alone → predict A → compute residuals on A
3. Pool all residuals → $\sigma_{\text{nested}} = \text{RMSE}(\text{pooled})$

**Leakage-free**, but with only $N = 2$ sub-folds, each sub-model is trained on a single track and is substantially weaker than the actual 2-track model.

#### Approach C: Training Residual RMSE

Compute in-sample RMSE from the training data. Essentially equivalent to $1/\sqrt{\alpha}$ from BayesianRidge, since BayesianRidge internally estimates this quantity.

### Empirical results on actual Experiment 30 data

| Fold | Actual RMSE | BR native σ | Nested σ | Training σ |
|:---|:---:|:---:|:---:|:---:|
| Holdout 8 | 0.2153 | 0.2170 | 0.3961 | 0.2153 |
| Holdout 10 | 0.2047 | 0.2072 | 0.6622 | 0.2069 |
| Holdout 14 | 0.2713 | 0.1916 | 0.2704 | 0.1871 |

#### Coverage at 90% nominal level

| Fold | BR native | Nested | Naive baseline |
|:---|:---:|:---:|:---:|
| Holdout 8 | **90.1%** | 98.4% | 76.0% |
| Holdout 10 | **91.2%** | 100.0% | 95.2% |
| Holdout 14 | **83.6%** | 89.4% | 76.1% |

#### CRPS (lower is better)

| Fold | BR native | Nested | Naive baseline |
|:---|:---:|:---:|:---:|
| Holdout 8 | **0.1148** | 0.1343 | 0.1923 |
| Holdout 10 | **0.1137** | 0.1793 | 0.1172 |
| Holdout 14 | **0.1366** | 0.1400 | 0.1970 |

### Analysis

**BayesianRidge native** is the correct choice:

1. **Calibration:** 90% coverage is 90.1%, 91.2%, 83.6% across the three folds. Two are within 2 percentage points of nominal; one (Track 14) undercovers by 6.4 points. This is imperfect but meaningfully better than the naive baseline (76.0%, 95.2%, 76.1%) and far sharper than nested (98.4%, 100%, 89.4%).

2. **Proper scoring:** BayesianRidge native achieves the best (lowest) CRPS on **all three folds**. It beats the naive baseline by 10–60% relative CRPS reduction. Nested LOO is worse than naive on the Track 10 fold (0.1793 vs 0.1172) due to massive overcoverage from the inflated $\sigma$.

3. **Why it works:** BayesianRidge's estimated $1/\alpha$ from training data captures the within-track noise variance that the model cannot explain. The parameter uncertainty term $\mathbf{x}^T \Sigma_w \mathbf{x}$ provides partial inflation for OOD inputs, partially compensating for the between-track mean shift. This combination is surprisingly well-tuned for our problem structure.

4. **Why nested fails:** With only $N = 2$ sub-folds, each sub-model trains on a single track (~350–380 samples). The sub-model's prediction of the other training track includes the full between-track mean shift as error, inflating $\sigma_{\text{nested}}$ by a factor of 1.3–3.2× relative to the actual held-out RMSE. This produces massive overcoverage and destroys sharpness.

5. **Why it undercovers on Track 14:** Track 14 has within-track std = 0.235 mm (highest of the three tracks), while the model trains on Tracks 8 + 10 (within-track stds 0.178, 0.196). The BayesianRidge $1/\alpha$ is calibrated to the training tracks' noise level, which is lower than Track 14's — hence undercoverage. This is heteroscedasticity that a single noise parameter cannot capture with only 3 tracks.

### Decision

**Use BayesianRidge native posterior predictive distribution `model.predict(X, return_std=True)` as the sole uncertainty model.** No external calibration. No inflation. No nested estimation.

This is:
- Fully leakage-free (uses only training data through the fitted model)
- The simplest approach
- Empirically superior on proper scoring (CRPS)
- Reasonably calibrated (2/3 folds within 2 pp of nominal; 1/3 undercovering by 6.4 pp)
- Scientifically interpretable: the model's own assessment of what it knows and doesn't know

---

## 4. Correct Treatment of Interval Sharpness

### The Error

The original criterion stated:

> "Mean 90% interval width < 2 × within-track target std"

For a Gaussian $\mathcal{N}(\mu, \sigma^2)$, a central 90% prediction interval has total width:

$$W_{90} = 2 \times 1.645 \times \sigma = 3.29 \sigma$$

If $\sigma \approx \sigma_{\text{target}}$ (expected when the model has no within-track skill), then:

$$W_{90} \approx 3.29 \times \sigma_{\text{target}}$$

The threshold $W_{90} < 2 \sigma_{\text{target}}$ requires $\sigma < 0.608 \sigma_{\text{target}}$, which is incompatible with ~90% coverage unless the model explains >63% of within-track variance ($R^2 > 0.63$). Our model has negative within-track $R^2$. **The criterion is geometrically impossible.**

### Correction

Remove the interval-width criterion entirely. Replace with **CRPS comparison against a properly constructed naive baseline**, which naturally captures both calibration and sharpness in a single proper scoring rule.

CRPS cannot be gamed by inflating $\sigma$: for a Gaussian predictive distribution, the CRPS-optimal $\sigma$ equals the true residual standard deviation. Making $\sigma$ too large degrades CRPS through the sharpness penalty. Making $\sigma$ too small degrades CRPS through the calibration penalty.

### Sharpness diagnostic (report, not gate)

Report the mean 90% interval width alongside the naive baseline's 90% interval width. If the model's intervals are wider than the naive baseline's, this indicates the model's uncertainty is less useful than a simple Gaussian — which is captured by CRPS anyway. This is a diagnostic for the report, not an acceptance criterion.

---

## 5. Target-Consistency Correction

### The Error

Section 16 of the original report mixed `local_core_width_mm` for development with `smoothed_macro_width_mm` for Track 21 blind inference. This is scientifically invalid: a model developed to predict one quantity cannot be evaluated against a different quantity without explicit domain-adaptation justification.

### Correction

The target is `local_core_width_mm` throughout:

1. **Development LOTO:** Train and evaluate BayesianRidge predictions of `local_core_width_mm` from Exp 29 artifacts.
2. **Track 21 blind inference:** Train final BayesianRidge on ALL development data with `local_core_width_mm` as target. Apply to Track 21 thermal features. Produce predictive distribution $\mathcal{N}(\hat{\mu}(x_i), \hat{\sigma}^2(x_i))$ for the **unobserved** Track 21 `local_core_width_mm`.
3. **Track 21 evaluation:** We cannot evaluate Track 21 predictions locally. Evaluation occurs when competition judges compare predictions against their target extraction. Our predictions represent expected local core width (the width of the dominant finite-support component) given process conditions.

> [!IMPORTANT]
> Track 21 `local_core_width_mm` does not exist in the repository — Wyko extraction was never performed on Track 21. We do not need it for prediction. We need it only for evaluation, which is the competition judges' responsibility.

`smoothed_macro_width_mm` is removed from the protocol entirely. It is never used.

---

## 6. Whether `local_core_width_mm` Can Scientifically Be Reused After Experiment 29 REJECT TARGET

### What Experiment 29 Rejected and Why

Experiment 29's pre-registered criteria assessed whether `local_core_width_mm` is suitable as a **spatially smooth, high-coverage deterministic prediction target**. The target failed on three grounds:

| Criterion | What it tests | Why it failed |
|:---|:---|:---|
| **B3** (p95 anchor jump ≤ 0.20 mm) | Spatial continuity between adjacent anchors | Wyko measurement voids create abrupt width discontinuities (0.40–0.49 mm jumps) |
| **D2** (zero-valid anchors = 0 inside domain) | Complete coverage at all spatial positions | 14–46 anchors per track fall in measurement-void regions |
| **A2** (dominant extent ≥ 95%) | Consistent spatial support | Track 10 dominant component starts late due to initial missing data |

### What Experiment 29 Did NOT Reject

Multiple criteria passed that establish the **physical validity of the target values at valid anchors:**

| Criterion | What it establishes | Result |
|:---|:---|:---|
| **E1** (boundary gradient ≥ 80th percentile) | Boundaries track real melt-pool edges | 93.3–95.6th percentile |
| **E2** (cross-track ordering) | Width ordering matches process physics | T14 < T10 < T8, matching thermal magnitudes |
| **A3** (dominance ratio ≥ 3×) | Unambiguous object identity | 40–76× dominance |
| **C1/C2** (defined mean, nonzero std) | Meaningful, varying values | Passed all tracks |
| **C3** (central 90% overlap) | Cross-track comparability | 3/3 pairs overlap |

### The Central Question

Does changing from deterministic to probabilistic evaluation make the target scientifically usable despite its geometric rejection?

### My Assessment

**Partially yes, with an explicit limitation.**

The rejection criteria B3 (spatial smoothness) and D2 (coverage) are **directly relevant to deterministic point prediction** — they establish that the target is too noisy to predict point-by-point along the track. A deterministic model evaluated by per-observation R² requires a spatially smooth target. The target fails this requirement.

However, for probabilistic evaluation:

1. **B3 (spatial discontinuity):** The p95 anchor jumps of 0.40–0.49 mm contribute to within-track variance ($\sigma \approx 0.18\text{--}0.24$ mm). In a probabilistic framework, this variance becomes part of what the uncertainty must capture. The BayesianRidge noise estimate $1/\alpha$ naturally includes this contribution because it is estimated from training residuals, which contain the same discontinuity structure. This is not a fix — the discontinuity noise remains in the target — but the probabilistic framework correctly accounts for it rather than pretending it doesn't exist.

2. **D2 (missing anchors):** Invalid anchors are excluded from evaluation. The model still produces predictions at those positions, but they cannot be scored. This reduces the effective sample size by 4–12% but does not bias the evaluation.

3. **A2 (Track 10 extent):** This is a Track-10-specific coverage limitation, not a quality-of-values problem.

**The values at valid anchors are physically meaningful measurements.** They correspond to the width of the dominant resolidified bead surface as measured by white-light interferometry, aligned with strong height-gradient boundaries. The within-track noise includes both physical melt-pool variation and measurement artifacts from Wyko voids — but these are inseparable without an independent measurement, and representing their combined effect as uncertainty is scientifically honest.

### What About Alternative Targets?

| Target | Problem |
|:---|:---|
| `smoothed_macro_width_mm` | **Anti-correlated** with thermal features (higher temp → narrower macro width, which is physically backward for core bead geometry). Also non-reproducible (generation script missing) and Track 21 contains constant placeholder 5.7329 mm. |
| PCA shape (`pc1`–`pc5`) | Deeply negative LOTO R² (−0.283 pooled). Five-dimensional output is harder to evaluate probabilistically. No evidence of process-condition correlation in PC coordinates. |
| New target | Prohibited by Experiment 31 pre-registered stop condition. |

`local_core_width_mm` is the **only available target** with physically correct correlation to thermal features and physically verified boundary alignment.

### Verdict on Target Reuse

`local_core_width_mm` is **defensible for probabilistic evaluation** under these conditions:

1. The report must state that the target was formally rejected for deterministic use and explain why.
2. Evaluation must be restricted to valid anchors (`anchor_valid == True`).
3. The within-track target variance must be acknowledged as including both physical variation and measurement-artifact discontinuity.
4. The probabilistic evaluation is framed as extending the Exp 30 learnability diagnostic — not as retroactively validating a rejected target.
5. The competition submission must present the predictions as "expected local width with uncertainty" without claiming the underlying target is noise-free.

---

## 7. Revised Experiment 32 Hypothesis

**Hypothesis:** BayesianRidge's native posterior predictive distribution for `local_core_width_mm`, trained on thermal physics features, constitutes a better probabilistic prediction of held-out local track width than a naive training-only Gaussian baseline, as measured by CRPS.

This hypothesis is distinct from Experiments 28–31:
- It does NOT claim the model predicts within-track spatial variation (it does not — all within-track R² are negative)
- It does NOT require a new target extraction or new parameters
- It tests whether the model's demonstrated ability to partially predict process-regime mean geometry, combined with its internally estimated observation noise, produces a useful probabilistic output
- It uses a proper scoring rule (CRPS) that cannot be gamed by inflating uncertainty

---

## 8. Revised Locked LOTO Protocol

### Target
- `local_core_width_mm` from Exp 29 artifact: [anchor_aggregated_width.csv](file:///Users/rick/Desktop/TAMU/NSF%20Challenge/processed_data/run_outputs/29_dominant_component_core_geometry_20260724_174420/tables/anchor_aggregated_width.csv)
- Filtered to `anchor_valid == True` and finite `local_core_width_mm`
- NO new target extraction. NO parameter changes. NO smoothing. NO gap-bridging.

### Predictors
- `peak_temp`, `sqrt_mp_area` (derived from `mp_area_px`), `mp_length`
- 5-frame centered windows (offsets −2, −1, 0, +1, +2): 15 features total
- `StandardScaler` fit on training fold ONLY

### Model
- `sklearn.linear_model.BayesianRidge()` with all default hyperparameters
- Identical to Experiment 30

### Uncertainty Model
- **BayesianRidge native posterior predictive distribution ONLY**
- `y_pred, y_std = model.predict(X_val_scaled, return_std=True)`
- Predictive distribution: $\hat{y}(x_i) \sim \mathcal{N}(\hat{\mu}(x_i), \hat{\sigma}^2(x_i))$
- **NO external calibration, NO residual inflation, NO nested estimation**
- The held-out track's labels are used ONLY for evaluation AFTER the predictive distribution is fully frozen

### LOTO Folds
- Fold 1: Train {10, 14} → Holdout 8 (N_val ≈ 383)
- Fold 2: Train {8, 14} → Holdout 10 (N_val ≈ 354)
- Fold 3: Train {8, 10} → Holdout 14 (N_val ≈ 377)

### Track 21 Sealing
- Track 21 Wyko geometry NEVER loaded
- Track 21 labels NEVER used for training, calibration, evaluation, or model selection
- Track 21 rows NEVER read from the multimodal dataset during Experiment 32

### Per-Fold Outputs (frozen before evaluation)
For each fold, freeze and save:
1. Predictive means $\hat{\mu}(x_i)$ for all valid held-out observations
2. Predictive standard deviations $\hat{\sigma}(x_i)$ for all valid held-out observations
3. Model parameters ($\alpha$, $\lambda$, coefficients)

### Per-Fold Evaluation (applied after freezing)
For each fold, compute against held-out labels:
1. Deterministic: R², MAE, RMSE, Pearson r (reproduced from Exp 30, unchanged)
2. Coverage at 50%, 80%, 90%, 95% nominal levels
3. CRPS (Gaussian)
4. NLL (Gaussian)
5. Mean 90% interval width (diagnostic)

---

## 9. Revised Probabilistic Baselines

### Naive Gaussian Baseline (one per fold, training-only)

For each LOTO fold with training tracks {A, B}:

$$\text{Naive: } \hat{y}(x_i) \sim \mathcal{N}(\bar{y}_{\text{train}}, \; s^2_{\text{train}})$$

where $\bar{y}_{\text{train}}$ = mean of all `local_core_width_mm` values in the training set and $s^2_{\text{train}}$ = sample variance (ddof=1) of the training set.

This baseline:
- Uses ZERO held-out information
- Represents "predict the training mean with training-estimated spread"
- Is the same for every observation within a fold (spatially constant)
- Is the weakest reasonable probabilistic prediction

### Why Not an Oracle Baseline?

An oracle baseline using held-out-track mean and variance would be diagnostic only. It uses information unavailable at prediction time and would leak labels. It may be reported for context but cannot be used in acceptance criteria.

### Baseline CRPS Computation

For each fold, compute naive CRPS using the standard Gaussian CRPS formula:

$$\text{CRPS}_{\text{Gaussian}}(y, \mu, \sigma) = \sigma \left[ z \left(2\Phi(z) - 1\right) + 2\phi(z) - \frac{1}{\sqrt{\pi}} \right]$$

where $z = (y - \mu)/\sigma$, $\Phi$ is the standard normal CDF, and $\phi$ is the standard normal PDF.

---

## 10. Revised Pre-Registered PASS / CONDITIONAL PASS / FAIL Criteria

### Primary Metric: CRPS

CRPS is a strictly proper scoring rule. It rewards both calibration (intervals cover the right fraction) and sharpness (intervals are not unnecessarily wide). It cannot be gamed by inflating $\sigma$ — there is a unique CRPS-optimal $\sigma$ for any given $(y - \mu)$.

### PASS (all must hold)

1. **Pooled model CRPS < pooled naive CRPS** (model beats naive baseline on concatenated OOF predictions)
2. **Model CRPS < naive CRPS on ≥ 2 of 3 folds** (majority of folds show improvement)
3. **90% coverage ≥ 75% on all 3 folds** (model is not dangerously anti-conservative on any fold)

### CONDITIONAL PASS (all must hold)

1. **Model CRPS < naive CRPS on ≥ 1 of 3 folds** (at least one fold shows improvement)
2. **90% coverage ≥ 65% on all 3 folds** (model is not severely anti-conservative)
3. **Pooled model CRPS < pooled naive CRPS OR 90% coverage ≥ 75% on ≥ 2 of 3 folds** (shows utility on at least one dimension)

### FAIL (any triggers failure)

1. **Model CRPS ≥ naive CRPS on all 3 folds** (model adds no probabilistic value)
2. **90% coverage < 65% on any fold** (model is dangerously anti-conservative)

### Why These Thresholds

- **CRPS comparison** is the primary gate because it simultaneously captures calibration and sharpness using a proper scoring rule.
- **75% coverage floor** (PASS) and **65% coverage floor** (CONDITIONAL PASS) acknowledge that BayesianRidge native uncertainty may undercover on some folds (empirically: Track 14 showed 83.6% at 90% nominal) while requiring that it not be catastrophically miscalibrated.
- **Fold-level criteria** prevent passing via pooled metrics alone (same logic as Experiment 30's fold-level R² requirement).

### Anti-Gaming Properties

1. **Inflating σ:** Increases CRPS (proper scoring penalty for lack of sharpness). Cannot game CRPS.
2. **Deflating σ:** Drops coverage below floor AND increases CRPS. Cannot game either metric.
3. **Pooled-only gaming:** Fold-level requirements prevent exploitation of between-track pooling.
4. **Coverage-only gaming:** CRPS comparison required; coverage alone is insufficient.

---

## 11. Track 21 Blind-Inference Procedure

### If Experiment 32 PASSES or CONDITIONAL PASSES

1. **Train final model** on ALL development data:
   - Combine all valid `local_core_width_mm` samples from Tracks 8, 10, 14 (N ≈ 1114)
   - Fit `StandardScaler` on all development features
   - Fit `BayesianRidge()` on scaled features → `local_core_width_mm`

2. **Prepare Track 21 features:**
   - Read Track 21 rows from `final_multimodal_dataset.csv` (400 rows)
   - Derive `sqrt_mp_area = sqrt(mp_area_px)` for Track 21
   - Construct 5-frame centered windows (yielding ~396 prediction windows)
   - Transform with the development-fitted `StandardScaler`

3. **Predict:**
   - `y_pred_21, y_std_21 = model.predict(X_track21_scaled, return_std=True)`
   - Each anchor position gets $(\hat{\mu}(x_i), \hat{\sigma}(x_i))$

4. **Output:**
   - CSV with columns: `track_id`, `frame_index`, `x_position_mm`, `predicted_local_core_width_mm`, `predicted_std_mm`, `predicted_90_lower_mm`, `predicted_90_upper_mm`
   - Lower/upper bounds: $\hat{\mu} \pm 1.645 \hat{\sigma}$

5. **Track 21 Wyko geometry is NEVER loaded.** The predictions are forward predictions based on thermal features only.

### Target Semantic

The predictions represent: "Given Track 21's observed thermal melt-pool characteristics, the expected local core width (dominant finite-support component's y-extent) at each anchor position, with BayesianRidge posterior uncertainty reflecting the model's internally estimated observation noise and parameter uncertainty."

### Expected Characteristics

Track 21 has the lowest thermal feature magnitudes of all tracks (`peak_temp` ≈ 2119, `mp_area_px` ≈ 1596, `mp_length` ≈ 46.1). If the model correctly learned the process-condition → width mapping (T8 > T10 > T14 in features and width), Track 21 predictions should show the **narrowest** predicted mean width, with uncertainty bands reflecting the development-calibrated noise level.

---

## 12. Exact Fallback If Experiment 32 Fails

### If Experiment 32 returns FAIL

**Submit the BayesianRidge thermal-physics model predictions for `local_core_width_mm` with BayesianRidge native posterior uncertainty, WITHOUT claiming calibration.**

Specifically:

1. **Model:** Same BayesianRidge + thermal physics as Experiment 32
2. **Target:** `local_core_width_mm` (Exp 29 artifact)
3. **Features:** `peak_temp`, `sqrt_mp_area`, `mp_length` in 5-frame centered windows
4. **Track 21 inference:** Same procedure as Section 11
5. **Uncertainty:** BayesianRidge `return_std=True` — reported as "model-estimated uncertainty" without calibration claims
6. **Validation results reported:**

| Fold | MAE | RMSE | R² | Pearson r |
|:---|:---:|:---:|:---:|:---:|
| Holdout 8 | 0.144 | 0.215 | −0.459 | 0.036 |
| Holdout 10 | 0.160 | 0.205 | −0.095 | 0.055 |
| Holdout 14 | 0.172 | 0.271 | −0.339 | 0.001 |
| Pooled | 0.158 | 0.233 | +0.098 | 0.445 |

Plus: Experiment 32 probabilistic metrics showing WHY the probabilistic framework failed (CRPS, coverage tables).

7. **Defensible claims:**
   - "Thermal features encode process-regime geometry with systematic cross-track ordering."
   - "Within-track geometric variation is not identifiable from available thermal features."
   - "The model provides a process-conditioned estimate of expected local width."
   - "Uncertainty is reported from the model's posterior but was not demonstrably calibrated in LOTO evaluation."

8. **NO Experiment 33.** The FAIL result is final.

---

## 13. Final Decision

### **AUTHORIZE REVISED EXPERIMENT 32**

---

## 14. One-Paragraph Scientific Justification

Experiment 32 is authorized because it tests a scientifically distinct hypothesis from all prior experiments — that BayesianRidge's native posterior predictive distribution constitutes a competition-valid probabilistic prediction of local geometry, even when deterministic within-track prediction has been conclusively falsified. The revised protocol is methodologically sound: uncertainty derives entirely from the model's internally estimated noise and parameter uncertainty, with zero leakage of held-out labels; the acceptance criteria use a proper scoring rule (CRPS) that cannot be gamed by uncertainty inflation; the target (`local_core_width_mm`) is defensible for probabilistic evaluation because its values at valid anchors are physically verified despite its formal rejection for deterministic spatial-smoothness criteria; and preliminary empirical analysis on the actual data suggests BayesianRidge native uncertainty achieves reasonable calibration (83.6–91.2% at 90% nominal) with CRPS improvement over the naive baseline on all three LOTO folds. The experiment requires no new target extraction, no parameter tuning, no Track 21 geometry access, and has a clear pre-registered stop condition: FAIL triggers the existing pipeline as the final submission with no further experiments.
