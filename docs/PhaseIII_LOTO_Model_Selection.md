# Phase III Leave-One-Track-Out Model Selection

## 1. Experimental motivation

Before opening the sealed Track 21 dataset, the final model-selection decision was re-evaluated using only the existing development tracks. The motivation was to determine whether the previous validation result, which used Track 14 as the only held-out development track, was sufficiently stable to justify freezing the final model.

The central question was:

> Which fixed baseline model and feature group are most scientifically defensible when every available development track is used once as the held-out validation track?

This experiment was not a new model-development phase. It did not introduce new models, new features, new preprocessing, hyperparameter tuning, target changes, or descriptor changes.

Track 21 remained sealed throughout the experiment.

---

## 2. Validation methodology

Leave-One-Track-Out (LOTO) validation was selected as the most appropriate development-only protocol because only three development tracks are available:

- Track 8
- Track 10
- Track 14

LOTO uses each development track once as the held-out validation track while training on the other two tracks. This directly tests whether the model and feature-group ranking is stable across development-track identity.

This is stronger than selecting a final model from only the Track 14 validation split because it avoids basing the final sealed-evaluation choice on a single development holdout.

The exact folds were:

| Fold | Training tracks | Validation track |
|---|---|---:|
| Fold 1 | 10 + 14 | 8 |
| Fold 2 | 8 + 14 | 10 |
| Fold 3 | 8 + 10 | 14 |

Fold 3 reproduces the earlier development evaluation.

---

## 3. Implementation details

The validation was implemented in:

`/scripts/24_phase3_loto_model_selection.py`

The script reused the frozen Phase III pipeline:

- `FeaturePreprocessor`
- `Phase3TargetAligner`
- `metrics.py`

Feature windows were flattened exactly as in the classical baseline experiments:

```text
X_flat = X_seq.reshape(len(X_seq), -1)
```

The PCA target group remained fixed:

- `pc1`
- `pc2`
- `pc3`
- `pc4`
- `pc5`

The evaluated feature groups were:

- Thermal-only
- SEM-only
- Thermal + SEM

The evaluated model families were fixed to the completed classical baselines:

- Linear Regression
- Ridge Regression
- Random Forest Regression

The model configurations were unchanged:

- Linear Regression: scikit-learn default `LinearRegression`
- Ridge Regression: `alpha = 1.0`
- Random Forest Regression:
  - `n_estimators = 300`
  - `min_samples_leaf = 2`
  - `random_state = 42`
  - `n_jobs = -1`

No LSTM or MLP models were rerun because the purpose was final classical baseline selection and because the prior neural baselines did not improve over the strongest classical results.

---

## 4. Complete per-fold results

### Fold 1 — hold out Track 8

Training tracks: Track 10 + Track 14  
Validation track: Track 8

| Model | Feature group | MAE | RMSE | Median AE | R² |
|---|---|---:|---:|---:|---:|
| Linear Regression | Thermal-only | 9.517956 | 9.902829 | 9.773650 | -41.545012 |
| Ridge Regression | Thermal-only | 1.958351 | 2.337153 | 1.800233 | -2.540169 |
| Random Forest | Thermal-only | 2.440904 | 2.702575 | 2.235486 | -2.413885 |
| Linear Regression | SEM-only | 1.422109 | 1.801268 | 1.172455 | -0.702514 |
| Ridge Regression | SEM-only | 1.409180 | 1.780858 | 1.168536 | -0.664097 |
| Random Forest | SEM-only | 1.576244 | 1.914569 | 1.472387 | -0.866498 |
| Linear Regression | Thermal + SEM | 7.421927 | 7.796190 | 7.599627 | -26.664144 |
| Ridge Regression | Thermal + SEM | 1.999139 | 2.388481 | 1.810182 | -2.792254 |
| Random Forest | Thermal + SEM | 1.447471 | 1.773884 | 1.317011 | -0.719256 |

### Fold 2 — hold out Track 10

Training tracks: Track 8 + Track 14  
Validation track: Track 10

| Model | Feature group | MAE | RMSE | Median AE | R² |
|---|---|---:|---:|---:|---:|
| Linear Regression | Thermal-only | 1.605906 | 1.987627 | 1.398354 | -0.231355 |
| Ridge Regression | Thermal-only | 1.506097 | 1.905360 | 1.261482 | -0.173048 |
| Random Forest | Thermal-only | 1.656080 | 2.052410 | 1.442676 | -0.200828 |
| Linear Regression | SEM-only | 1.428511 | 1.834271 | 1.212041 | -0.105083 |
| Ridge Regression | SEM-only | 1.428602 | 1.834341 | 1.212699 | -0.104930 |
| Random Forest | SEM-only | 1.516358 | 1.900757 | 1.260320 | -0.179235 |
| Linear Regression | Thermal + SEM | 1.677111 | 2.065982 | 1.477324 | -0.294469 |
| Ridge Regression | Thermal + SEM | 1.579029 | 1.984375 | 1.351539 | -0.242208 |
| Random Forest | Thermal + SEM | 1.607646 | 2.000458 | 1.370770 | -0.176433 |

### Fold 3 — hold out Track 14

Training tracks: Track 8 + Track 10  
Validation track: Track 14

| Model | Feature group | MAE | RMSE | Median AE | R² |
|---|---|---:|---:|---:|---:|
| Linear Regression | Thermal-only | 2.535383 | 2.924938 | 2.407038 | -6.554562 |
| Ridge Regression | Thermal-only | 2.024324 | 2.383099 | 1.963877 | -2.268337 |
| Random Forest | Thermal-only | 1.892585 | 2.115979 | 1.821794 | -1.635473 |
| Linear Regression | SEM-only | 0.936175 | 1.206902 | 0.722373 | -0.152840 |
| Ridge Regression | SEM-only | 0.935437 | 1.205774 | 0.722716 | -0.149425 |
| Random Forest | SEM-only | 1.314589 | 1.664540 | 1.135141 | -0.848344 |
| Linear Regression | Thermal + SEM | 2.464486 | 2.863417 | 2.381854 | -7.072938 |
| Ridge Regression | Thermal + SEM | 2.069955 | 2.428553 | 2.007077 | -2.437421 |
| Random Forest | Thermal + SEM | 1.427145 | 1.747150 | 1.245874 | -0.946050 |

---

## 5. Aggregate comparison

Aggregate metrics were computed by pooling the validation predictions from all three LOTO folds for each fixed model and feature-group combination.

| Rank | Model | Feature group | Pooled MAE | Pooled RMSE | Pooled Median AE | Pooled R² | Mean fold MAE | Std fold MAE |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | Ridge Regression | SEM-only | 1.275480 | 1.655831 | 1.013160 | -0.282801 | 1.257740 | 0.228041 |
| 2 | Linear Regression | SEM-only | 1.281532 | 1.665863 | 1.018918 | -0.299170 | 1.262265 | 0.230595 |
| 3 | Random Forest | Thermal + SEM | 1.482236 | 1.835332 | 1.292172 | -0.434097 | 1.494087 | 0.080726 |
| 4 | Random Forest | SEM-only | 1.484451 | 1.843827 | 1.290224 | -0.475172 | 1.469064 | 0.111933 |
| 5 | Ridge Regression | Thermal-only | 1.862703 | 2.308224 | 1.592285 | -1.385506 | 1.829590 | 0.230324 |
| 6 | Ridge Regression | Thermal + SEM | 1.913079 | 2.357842 | 1.673070 | -1.617500 | 1.882708 | 0.216671 |
| 7 | Random Forest | Thermal-only | 2.080956 | 2.396453 | 1.899643 | -1.118968 | 1.996523 | 0.328724 |
| 8 | Linear Regression | Thermal + SEM | 4.510950 | 5.635740 | 3.528340 | -10.481394 | 3.854508 | 2.542944 |
| 9 | Linear Regression | Thermal-only | 5.463471 | 7.048425 | 3.428166 | -15.117279 | 4.553082 | 3.531144 |

The best aggregate model by pooled MAE, pooled RMSE, and pooled R² was:

```text
Ridge Regression + SEM-only features
```

Linear Regression + SEM-only was effectively tied, with only slightly worse aggregate metrics. The difference between Ridge SEM-only and Linear SEM-only was small:

- MAE difference: 0.006052
- RMSE difference: 0.010032
- R² difference: 0.016369

Random Forest models ranked below SEM-only Ridge and SEM-only Linear in aggregate LOTO validation.

---

## 6. Stability analysis

### Model ranking stability

The model ranking was not perfectly stable in every fold and feature group. However, the broad pattern was consistent enough to support final selection.

The SEM-only linear family was consistently strong across all held-out tracks:

| Fold | Best SEM-only model | MAE |
|---|---|---:|
| Holdout Track 8 | Ridge Regression | 1.409180 |
| Holdout Track 10 | Linear Regression | 1.428511 |
| Holdout Track 14 | Ridge Regression | 0.935437 |

Ridge and Linear were effectively tied for SEM-only features. Ridge was best in two of the three folds, but the difference in Fold 2 was negligible and favored Linear by only 0.000091 MAE.

Random Forest did not consistently dominate. It performed best for the combined thermal + SEM feature group in the Track 8 and Track 14 holdouts, but it did not produce the best overall aggregate LOTO performance.

### Feature-group ranking stability

SEM-only features were the most stable and strongest aggregate feature group.

The single best model-feature combination in each fold was:

| Fold | Best model-feature combination | MAE |
|---|---|---:|
| Holdout Track 8 | Ridge Regression + SEM-only | 1.409180 |
| Holdout Track 10 | Linear Regression + SEM-only | 1.428511 |
| Holdout Track 14 | Ridge Regression + SEM-only | 0.935437 |

This provides direct evidence that SEM-only features generalize most consistently across the available development tracks.

Thermal-only and combined thermal + SEM models showed larger instability, especially for ordinary Linear Regression. Ridge regularization greatly reduced the failures of thermal-only and combined linear models, but did not outperform SEM-only Ridge.

### Track-specific behavior

Holdout Track 8 exposed the largest failures for unregularized linear models with thermal features:

- Linear Regression + Thermal-only: MAE = 9.517956
- Linear Regression + Thermal + SEM: MAE = 7.421927

These failures were largely mitigated by Ridge regularization but remained evidence that thermal feature windows can create unstable extrapolation under ordinary least squares.

Holdout Track 10 produced the most compressed model ranking. Several models achieved MAE between approximately 1.43 and 1.68.

Holdout Track 14 reproduced the previous development evaluation and again favored SEM-only Ridge/Linear over Random Forest.

### Previous Track 14-only conclusion

The earlier Track 14-only result suggested that SEM-only Ridge/Linear were strongest. LOTO validation supports that conclusion. The Track 14-only ranking was not merely an artifact of using Track 14 as the sole validation track.

However, the previous pipeline-freeze document's selection of Random Forest + SEM-only is not supported by the full LOTO evidence. Random Forest remains useful as an interpretable nonlinear baseline, but it is not the strongest final predictive baseline under aggregate LOTO validation.

---

## 7. Final recommendation

Based on all development evidence, the recommended model and feature group for sealed Track 21 evaluation is:

```text
Ridge Regression + SEM-only features
```

The recommendation is based on:

1. best pooled LOTO MAE;
2. best pooled LOTO RMSE;
3. best pooled LOTO R²;
4. consistent SEM-only performance across all three held-out development tracks;
5. near-tie behavior between Ridge and Linear, with Ridge slightly preferred by aggregate metrics and regularization stability;
6. poor stability of unregularized thermal-feature linear models;
7. Random Forest not outperforming SEM-only Ridge/Linear in aggregate LOTO validation.

This recommendation changes the current pipeline-freeze model selection. The previous freeze selected Random Forest + SEM-only, but the full LOTO development evidence supports Ridge Regression + SEM-only instead.

The change is scientifically justified because it is based only on development tracks and uses fixed models, fixed features, fixed preprocessing, fixed targets, and fixed metrics. Track 21 remains sealed and did not influence this decision.

---

## 8. Limitations

The LOTO audit strengthens the model-selection decision but does not eliminate all limitations.

Important limitations remain:

1. Only three development tracks are available, so each validation fold is based on one physical track.
2. Validation R² remains negative for every aggregate model-feature combination, including the selected model.
3. The selected SEM-only model may not capture all physically meaningful thermal-process information, even though it generalizes best across the development tracks.
4. The target group evaluated here is PCA shape only; amplitude and signed elevation were not used for final model selection.
5. Track 21 remains sealed, so all conclusions are still development-track conclusions.
6. The dataset contains dense neighboring x positions, but model selection is track-aware rather than row-random to reduce overstatement of generalization.

These limitations should be reported directly in the final challenge report.

---

## 9. Final development-stage conclusion

Leave-One-Track-Out validation was the appropriate final development-stage experiment because it used all available development tracks without touching Track 21 and directly tested model-selection stability.

The experiment indicates that the final sealed-evaluation pipeline should be updated from:

```text
Random Forest + SEM-only
```

to:

```text
Ridge Regression + SEM-only
```

No additional model classes, feature engineering, preprocessing changes, or hyperparameter tuning are justified before sealed evaluation.

The next step is to update the pipeline-freeze decision accordingly, then evaluate Track 21 once using the frozen final selection.
