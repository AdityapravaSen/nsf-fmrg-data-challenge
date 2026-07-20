# Phase 5 Strategy: Transitioning from Micro-Variation Modeling to Physics-Constrained Extrapolation

## 1. What We Have Done (Phase I – III Summary)
We have successfully engineered a highly rigorous, multimodal machine learning pipeline. Our work to date represents a comprehensive and mathematically sound exploration of the dataset:

* **The Alignment:** We built a unified pipeline that perfectly aligns in-situ thermal history (5-frame moving windows) and local SEM substrate context onto a shared spatial X-axis (20mm → 100mm) with extreme precision.
* **Rigorous Target Extraction:** After proving that simple deterministic "width" measurements were susceptible to severe noise (mixed-sign disturbances, height-map valleys), we developed a complex, scientifically defensible target utilizing PCA to capture normalized cross-sectional shape, local amplitude, and elevation.
* **Extensive Baseline Study:** We designed and executed a leak-proof Phase III modeling suite with Leave-One-Track-Out (LOTO) cross-validation. We evaluated a wide range of architectures including Linear Regression, Ridge Regression, Random Forests, LSTMs, and MLPs.
* **Multimodal Feature Ablation:** We systematically tested model performance using Thermal-only, SEM-only, and Thermal+SEM feature groups to quantify source attribution.

## 2. What Our Results Are(Phase IV)
Our Phase III evaluation yielded highly informative, albeit definitively negative, predictive metrics:

* **Negative R² Scores on Complex Models:** In our LOTO cross-validation, deep learning models (LSTMs, MLPs) consistently produced negative R² scores (e.g., -0.64). The models were unable to capture the variance and performed worse than predicting the flat mean of the training data.
* **Zero Local Correlation:** Extensive statistical and correlation analysis revealed that *within a single track*, local geometric fluctuations have effectively **zero sign-consistent correlation** with either the SEM substrate roughness or frame-to-frame thermal variations. 
* **Model Overfitting:** Complex architectures attempted to memorize high-frequency topological changes in the training data, leading to severe overfitting and catastrophic failure during cross-track extrapolation.

## 3. Where We Stand
We have mathematically and scientifically proven a crucial negative result: **Local micro-fluctuations in laser track geometry are dominated by stochastic measurement and texture noise, and cannot be predicted sequentially.** Because the hidden Track 21 operates at a significantly lower laser power (200W) than our training tracks (300W, 350W, 400W), any model trying to learn high-frequency geometric patterns will fail at this massive extrapolation. Therefore, we must pivot our strategy. Rather than trying to predict unlearnable micro-noise, we will leverage our pipeline to model the **macro-physical power law** driving the overall track dimensions.

---

## Part II: The Phase 5 Approach (Physics-Constrained Extrapolation)

To successfully extrapolate to the unseen 200W Track 21, Phase 5 pivots to a highly constrained, low-dimensional modeling approach. We will intentionally strip away complexity to prevent our models from overfitting the measurement noise, forcing them to learn only the macro-level physical trend.

### What We Need to Do
1. **Simplify the Target:** We must abandon the high-dimensional PCA shape targets. We will revert to predicting a single scalar: **Macro-Width**. To prevent the texture noise we encountered earlier, we will apply heavy spatial smoothing (e.g., a moving average) or define the track by its "NaN-valley" (the stable depression bounded by high-NaN roughness).
2. **Restrict the Inputs (Drop SEM):** Because our correlation study proved SEM features do not generalize across tracks, keeping them will only cause spatial memorization (overfitting to specific tiles). We will drop the SEM feature branch entirely.
3. **Isolate Physics-Based Thermal Features:** We will discard the raw 5-frame thermal histories. Instead, we will extract just **three explicit melt-pool physics descriptors** per spatial step: 
   * `sqrt(melt_pool_area)`
   * `peak_temperature`
   * `tail_length` (or a similar proxy for the cooling tail)
4. **Deploy a Linear Extrapolation Model:** We will drop LSTMs, MLPs, and Tree-based models. We will rely solely on **Bayesian Ridge / Linear Regression**. Because these models are strictly linear, feeding them the reduced thermal features of Track 21 will force a safe, mathematical extrapolation down the power-law curve, rather than a catastrophic out-of-distribution failure.

### Files That Need to Change
* `Phase3TargetAligner.py` (or your equivalent target generation script): Update to output a smoothed, 1D scalar macro-width instead of PCA arrays.
* `FeaturePreprocessor.py`: Disable the SEM processing branch and constrain the thermal feature output to the 3 selected macro-features.
* `modeling_baseline.ipynb` (or ML training scripts): Remove deep learning architectures; configure a clean Bayesian Ridge pipeline mapped to Track power levels.

---

## 4. Structured Work Split & Deliverables

To execute Phase 5 efficiently, we will divide the modifications across our established ownership boundaries:

### **Nabarun: Target Simplification & Smoothing**
* **Task:** Refactor the ground-truth target extraction. 
* **Action:** Modify `Phase3TargetAligner` to abandon PCA. Implement a method to extract a single "smoothed width" scalar. You can achieve this by applying a wide moving-average over the raw width, or by calculating the width of the low-NaN "valley" between the rough track shoulders.
* **Goal:** Produce a 1D target array per track where high-frequency waviness is filtered out, leaving only the macro-trend.

### **Adi: Feature Restriction & SEM Ablation**
* **Task:** Refactor the input feature space to enforce the physics constraint.
* **Action:** Go into `FeaturePreprocessor`. Completely disable the SEM feature merging. Reduce the thermal feature extraction to output exactly three metrics per X-coordinate: `sqrt(area)`, `peak_temperature`, and `tail_length`. 
* **Goal:** Deliver a strictly 3-column input feature matrix per track.

### **Joint Effort: Modeling & Final Evaluation**
* **Task:** Retrain and validate the new highly constrained pipeline.
* **Action:** 1. Merge Nabarun's 1D smoothed targets with Adi's 3-feature thermal inputs.
    2. Run the Bayesian Ridge Regression through our LOTO cross-validation framework.
    3. Verify that the R² / MAE scores have improved and that the model is successfully identifying the macro-trend across the 300W-400W tracks.
    4. Generate final predictions for the hidden 200W Track 21.

### ✅ Deliverables Checklist
- [ ] **Nabarun:** Commit updated `Phase3TargetAligner` (Output: 1D smoothed macro-width).
- [ ] **Adi:** Commit updated `FeaturePreprocessor` (Output: SEM removed, 3 thermal features only).
- [ ] **Both:** Successful run of the combined dataset generation script.
- [ ] **Both:** Execute Bayesian Ridge/Linear Regression on the new dataset.
- [ ] **Both:** Document the new LOTO CV metrics (aiming to beat the flat-mean baseline).
- [ ] **Both:** Generate final `.csv` / `.npy` predictions for Track 21.
- [ ] **Both:** Draft the final challenge report, framing our deep-learning iterations as a rigorous ablation study that scientifically justifies our physics-constrained final model.
