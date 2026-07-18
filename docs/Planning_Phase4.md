Here is the detailed planning and structure for Phase IV. You can save this as `Planning_Phase4.md` in your repository. It outlines the strict rules of this final phase and distributes the remaining tasks between you and Nabarun to get the final challenge paper ready.

---

# Phase IV Plan — Final Evaluation & Reporting

**Project:** NSF Future Manufacturing Data Challenge

**Phase:** Phase IV (Final Execution)

**Status at start:** Phases I, II, and III Complete. Pipeline is officially frozen.

**Sealed evaluation track:** Track 21

**Winning Baseline:** Random Forest Regression (SEM-only features) predicting PCA shape (`pc1`--`pc5`).

---

## 🚨 The Prime Directive of Phase IV

Phase IV is strictly an **execution and reporting** phase.

* **NO** model architecture changes.
* **NO** hyperparameter tuning (Random Forest remains strictly at `n_estimators=300`, `min_samples_leaf=2`).
* **NO** feature engineering or threshold tweaking.
* If Track 21 performs poorly, we report the poor performance and discuss why (generalization limitations). We do not go back and change the model.

---

## 📅 Step-by-Step Execution Plan

### Step 1: The Sealed Evaluation (Unsealing Track 21)

This step executes the frozen pipeline exactly once to generate the final challenge outputs.

* **Train:** Fit the frozen `FeaturePreprocessor` and `RandomForestRegressor` on the combined data of all three development tracks (Tracks 8, 10, and 14).
* **Predict:** Run inference on Track 21 feature sequences.
* **Export:** Save the raw outputs to `processed_data/track21_final_predictions.csv`, maintaining the strict `track_id`, `frame_index`, and `x_position_mm` identities.

### Step 2: Final Metric Computation

Compute the exact same metrics used in the Phase III LOTO validation to allow for a direct comparison between development generalization and sealed generalization.

* Calculate overall **MAE, RMSE, Median AE, and R²** for Track 21.
* Calculate per-target metrics (breakdown for `pc1` through `pc5`).

### Step 3: Publication-Quality Visualizations

Generate the visual evidence required for the final NSF challenge paper.

* **Trajectory Plots:** Plot the predicted vs. actual PCA shape descriptors along the `x_position_mm` axis for Track 21. (Crucial to show if the model captures spatial trends).
* **Residual/Error Plots:** Scatter plots of predicted vs. observed values to visualize bias or variance.
* **Interpretability / Feature Importance:** Generate a bar chart showing the final Random Forest feature importances for the SEM-only features across the 5-frame rolling window.

### Step 4: Final Challenge Report Assembly

Synthesize the scientific journey from Phase I to Phase IV into the final paper.

* **Data Alignment (Phase I):** Explain the "Rosetta Stone" X-axis alignment of Thermal and SEM modalities.
* **Target Definition (Phase II):** Defend why a simple "width" was discarded in favor of normalized cross-sectional shape (PCA) and local amplitude.
* **Model Selection (Phase III):** Present the LOTO cross-validation tables. Explain why `sem_only` generalized better than `thermal_plus_sem` across disparate physical tracks.
* **Results & Limitations (Phase IV):** Present the Track 21 results. Openly discuss the difficulty of cross-track generalization in laser powder bed fusion (LPBF) processes.

---

## 👥 Work Distribution

### Person A (Adi) — Execution & Visualization

* [ ] Write and execute `script/25_phase4_track21_evaluation.py`.
* [ ] Export `track21_final_predictions.csv`.
* [ ] Generate the spatial trajectory plots (Predicted vs Actual along the X-axis).
* [ ] Generate the final Random Forest Feature Importance charts.

### Person B (Nabarun) — Metrics & Report Drafting

* [ ] Run the final metric computation on Track 21 (MAE, RMSE, Median AE, R²).
* [ ] Generate residual/error distribution plots.
* [ ] Draft the **Methodology** section of the final report (documenting the PCA geometry math and alignment).
* [ ] Draft the **Results** section, directly comparing LOTO averages to Track 21 sealed results.

### Joint Tasks

* [ ] Final review of all figures and metric tables.
* [ ] Compile the final PDF report.
* [ ] Submit repository and paper to the challenge portal.