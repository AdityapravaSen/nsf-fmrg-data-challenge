# Phase IV Results & Scientific Evaluation
**Project:** NSF Future Manufacturing Data Challenge  
**Phase:** Phase IV (Sealed Evaluation & Scientific Validation)  
**Execution Lead:** Aditya  

---

## 1. Execution of Phase IV-A: Frozen Evaluation
Following the successful completion of the Stage 1 preflight checks by Nabarun, the remainder of the Phase IV-A frozen baseline evaluation was executed successfully.

### 1.1 Technical Implementation
The execution script (`26_phase4_model_execution.py`) was modularized to isolate the training and inference pipeline from the preflight checks. The following stages from the Phase IV plan were completed:
* **Stage 2 (Preprocessing & Sequences):** The `FeaturePreprocessor` was initialized with the frozen **SEM-only** feature set (`substrate_roughness_variance`, `substrate_mean_intensity`). To accommodate the sealed nature of Track 21, the `pca_ready` validity filter was strictly applied to the training tracks (8, 10, 14) but bypassed for Track 21 to allow full inference.
* **Stage 3 (Model Fitting & Prediction):** A **Ridge Regression ($\alpha = 1.0$)** model was fitted on the 5-frame sequence windows from the development tracks.
* **Stage 4 (Reconstruction & Export):** Track 21 was successfully evaluated. Because the challenge organizers withheld the ground truth PCA targets (populated entirely with `NaN`s), the pipeline performed **Blind Inference**. Local metric calculations (MAE, RMSE, R²) were programmatically skipped to prevent errors.

### 1.2 Generated Artifacts
* The final submission file, `track21_predictions.csv`, was successfully generated and archived in the timestamped Phase IV output directory. It contains the predicted `pc1` through `pc5` geometry descriptors mapped perfectly to the `x_position_mm` axis.

---

## 2. Execution of Phase IV-B: Scientific Analysis
Because objective metrics could not be calculated locally, scientific analysis relied on visual and physical interpretation of the predicted spatial trajectories.

### 2.1 Visualization Suite
A dedicated visualization script (`27_phase4_visualization_blind.py`) was developed to plot the predicted PCA trajectories for Track 21 along the longitudinal axis (20 mm to 100 mm). The resulting figure (`track21_blind_trajectories.png`) serves as the primary artifact for evaluating model behavior.

### 2.2 Physical Interpretation & Failure Modes
Visual inspection of the blind Track 21 predictions revealed a critical scientific observation:
* **Stable, Low-Variance Estimator:** The Ridge Regression predictions for all five PCA components are nearly flat across the entire 80 mm span (e.g., `PC1` hovers consistently around -0.77). 
* **Physical Explanation:** By relying strictly on SEM-only features (which capture macro-level substrate roughness that averages out over long distances) and applying a linear penalty to variance ($\alpha = 1.0$), the model acts as a highly constrained estimator. It predicts the "average" stable geometry learned from the development tracks.
* **Conclusion on Generalization:** The model successfully bounds the PCA shape descriptors to realistic physical ranges, preventing the catastrophic hallucination of peaks and valleys that a Random Forest + Thermal model would have produced. However, it underfits the highly dynamic, high-frequency longitudinal fluctuations characteristic of the LPBF process. 

---

## 3. Execution of Phase IV-C: Decision Checkpoint
Based on the scientific analysis, we evaluated the three checkpoint questions outlined in the Phase IV Plan:

1. **Is the frozen baseline already submission quality?** * **Yes.** While it underfits high-frequency variations, it is a mathematically honest, leak-proof, and highly stable baseline that reflects the extreme difficulty of cross-track generalization.
2. **Can observed failures be explained scientifically?**
   * **Yes.** The LOTO validation in Phase III proved that thermal features overfit to track-specific biases. The "flatness" of Track 21 is the expected physical result of applying a regularized linear model to generalized substrate textures.
3. **Is there exactly one scientifically motivated hypothesis worth testing?**
   * **No.** Any attempt to swap models or add thermal features at this stage to artificially "force" variation into the Track 21 predictions would violate the integrity of the sealed test set (Data Leakage). 

**Decision:** The frozen Ridge Regression (SEM-only) baseline is accepted as the final, official submission. Phase IV-D (Optional Development) is formally declined.

---

## 4. Final Deliverables
* ✅ **Submission File:** `track21_predictions.csv` is ready for upload to the NSF Challenge portal.
* ✅ **Visual Evidence:** `track21_blind_trajectories.png` is ready for inclusion in the final report.
* ✅ **Pipeline Integrity:** The codebase remains completely free of test-set data leakage.

[![track21-blind-trajectories.png](https://i.postimg.cc/63LXZQCY/track21-blind-trajectories.png)](https://postimg.cc/HcxRDdZy)