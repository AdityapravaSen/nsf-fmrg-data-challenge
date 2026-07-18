# Phase III Pipeline Freeze

## 1. Pipeline scope

This document records the frozen Phase III pipeline for sealed Track 21 evaluation.

The pipeline represents the finalized development workflow after completion of the Phase III baseline experimentation stage. It defines the dataset split, preprocessing contract, target-alignment procedure, selected baseline model, and evaluation protocol that will be used for the sealed evaluation.

After this freeze point, no preprocessing, feature engineering, model selection, hyperparameter choice, or evaluation methodology should change unless a genuine implementation bug, numerical error, or reproducibility failure is discovered.

---

## 2. Dataset split

The development split used for model development is fixed as follows.

Development training tracks:

- Track 8
- Track 10

Development validation track:

- Track 14

Sealed evaluation track:

- Track 21

Track 21 has not been used for model selection, feature engineering, preprocessing decisions, hyperparameter selection, threshold selection, interpretation, or reporting-template design.

Track 21 remains sealed until final evaluation with the frozen pipeline.

---

## 3. Frozen preprocessing

The frozen preprocessing pipeline uses the completed Phase III infrastructure.

### Feature preprocessing

Feature preprocessing is performed by `FeaturePreprocessor` in:

`/scripts/phase3_data_loader.py`

The preprocessing contract is:

1. load the frozen merged dataset:

   `processed_data/final_multimodal_dataset.csv`

2. filter to rows where `pca_ready == True` for PCA-shape prediction;
3. split data by physical track;
4. fit the feature scaler using training tracks only;
5. transform training and validation features using the training-fitted scaler;
6. preserve metadata columns:

   - `track_id`
   - `frame_index`
   - `x_position_mm`

No Track 21 rows are used to fit preprocessing parameters.

### Rolling window generation

The canonical Phase III feature representation is a five-frame rolling window:

```text
(samples, window_size, features)
```

with:

```text
window_size = 5
```

The metadata row associated with each rolling window corresponds to the final frame in that window.

The metadata ordering defines the canonical sample ordering for feature-target alignment and evaluation.

### Metadata and PCA target alignment

Target alignment is performed by `Phase3TargetAligner` in:

`src/ml/targets.py`

The target-alignment contract is:

1. consume `train_meta` or `val_meta` from the feature preprocessing pipeline;
2. perform an exact one-to-one merge against:

   `processed_data/final_multimodal_dataset.csv`

3. join only on:

   - `track_id`
   - `frame_index`
   - `x_position_mm`

4. preserve incoming metadata ordering exactly;
5. return NumPy target arrays.

No nearest-neighbor matching, `merge_asof`, interpolation, target filling, or descriptor recomputation is allowed.

The frozen PCA target group is:

- `pc1`
- `pc2`
- `pc3`
- `pc4`
- `pc5`

### Dataset packaging

PyTorch-compatible dataset packaging is provided by `Phase3TorchDataset` in:

`src/ml/datasets.py`

This wrapper packages already-preprocessed features, already-aligned targets, and aligned metadata. It does not perform preprocessing, target alignment, feature engineering, or model training.

### Evaluation protocol

Evaluation is performed using the reusable regression metrics in:

`src/ml/metrics.py`

The same metric definitions are used for all models and will be used for Track 21 evaluation.

---

## 4. Candidate models evaluated

The completed Phase III baseline study evaluated the following model families.

### Linear Regression

Linear Regression established the simplest flattened-feature baseline. It tested whether a linear mapping from feature windows to PCA shape scores was sufficient.

### Ridge Regression

Ridge Regression tested whether L2 regularization improved the same flattened-feature representation.

### Random Forest Regression

Random Forest Regression tested nonlinear tree-based partitioning and provided the strongest nonlinear baseline behavior in the completed development evaluation.

### LSTM Regression

The LSTM baseline tested whether preserving the five-frame temporal ordering improved prediction relative to flattened-window baselines.

### MLP Regression

The MLP baseline tested generic nonlinear neural-network function approximation while using the same flattened feature representation as the classical baselines.

No additional model families are included in the frozen baseline selection.

---

## 5. Final baseline selection

The selected baseline model for sealed evaluation is:
`Random Forest Regression`

The selected feature group is:
`SEM-only`

The selected target group is:
`PCA shape: pc1--pc5`

### Rationale

The final baseline selection was validated using Leave-One-Track-Out (LOTO) cross-validation across all development tracks (Tracks 8, 10, and 14). This ensured the selected model and feature configuration robustly generalized across track holdouts, rather than overfitting to a single validation split.

**LOTO Cross-Validation Average Results:**

| Feature group | Average MAE | Average RMSE | Average Median AE | Average R² |
|---|---:|---:|---:|---:|
| SEM-only | 1.4711 | 2.5553 | 1.2951 | -0.6407 |
| Thermal + SEM | 1.4968 | 2.6132 | 1.3132 | -0.6208 |

Based on the LOTO evaluation, the **SEM-only** feature set produced the strongest average generalization performance (lower MAE, RMSE, and Median AE) across unseen tracks. Adding thermal features degraded held-out error metrics, indicating overfitting to the training tracks.

The Random Forest configuration remains strictly frozen as:

- `n_estimators = 300`
- `min_samples_leaf = 2`
- `random_state = 42`
- `n_jobs = -1`

No hyperparameter tuning will be performed using Track 21.

---

## 6. Frozen evaluation protocol

The evaluation metrics are fixed as:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Median Absolute Error
- R² score

For multi-output PCA targets, metrics are reported as:

- overall metrics across PC1--PC5;
- per-target metrics where appropriate;
- per-track metrics where target truth is available.

The development validation procedure is fixed as:

- train on Tracks 8 and 10;
- validate on Track 14;
- keep Track 21 sealed until final evaluation.

The Track 21 evaluation will use the same:

- feature schema;
- preprocessing logic;
- target-alignment contract, if Track 21 targets become available;
- model configuration;
- metric definitions;
- reporting format.

No Track 21 result will be used to modify preprocessing, feature selection, model selection, hyperparameters, or interpretation rules.

---

## 7. Frozen repository components

The following Phase III components are frozen for sealed evaluation:

### Preprocessing

- `FeaturePreprocessor`
- feature schema
- training-only scaler fitting
- five-frame rolling window construction
- metadata contract

### Target alignment

- `Phase3TargetAligner`
- exact merge on `track_id`, `frame_index`, and `x_position_mm`
- PCA target group `pc1`--`pc5`
- ordering preservation rules

### Dataset generation

- `Phase3TorchDataset`
- aligned feature/target/metadata packaging

### Evaluation helpers

- regression metrics in `src/ml/metrics.py`
- overall metrics
- per-target metrics
- per-track metrics

### Baseline model selection

- selected model family: Random Forest Regression
- selected feature group: SEM-only
- selected target group: PCA shape (`pc1`--`pc5`)
- fixed Random Forest configuration listed above

---

## 8. Conditions under which changes are allowed

After this pipeline freeze, changes are allowed only for:

- implementation bugs;
- numerical errors;
- reproducibility failures;
- file-path or execution issues that prevent the frozen pipeline from running as documented.

Changes are not allowed because:

- Track 21 performs better than expected;
- Track 21 performs worse than expected;
- another model appears promising after the freeze point;
- a different feature group would improve sealed evaluation performance;
- a new interpretation becomes attractive after seeing sealed results.

Any allowed bug fix must be documented explicitly and should preserve the scientific intent of the frozen pipeline.

---

## 9. Final declaration

The Phase III pipeline described in this document is considered frozen for sealed Track 21 evaluation.

The next project stage is final execution and reporting, not additional model development.
