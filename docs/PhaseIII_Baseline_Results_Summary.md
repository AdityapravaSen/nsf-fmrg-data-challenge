# Phase III Baseline Results Summary

## 1. Experimental objective

Phase III evaluated whether aligned in-situ thermal features and SEM-derived substrate-context features can predict frozen post-process geometry descriptors. The prediction target for the baseline study was the normalized cross-sectional shape descriptor represented by five PCA scores:

- `pc1`
- `pc2`
- `pc3`
- `pc4`
- `pc5`

The experiments used the completed Phase III data pipeline:

1. load the frozen multimodal dataset,
2. preprocess thermal and SEM features without leakage,
3. construct five-frame rolling feature windows,
4. align frozen PCA targets using the metadata contract,
5. evaluate predictions with common regression metrics.

The development evaluation used Tracks 8 and 10 for training and Track 14 as the held-out validation track. Track 21 remained sealed and was not used for model development, model selection, preprocessing decisions, or interpretation.

The goal of this baseline study was not to optimize a final model, but to establish reproducible, interpretable reference performance across simple model families and feature groups.

---

## 2. Experimental progression

The baseline experiments were designed to test increasingly flexible modeling assumptions while keeping the dataset, targets, feature groups, and evaluation protocol fixed.

- **Linear Regression** established the simplest flattened-feature baseline and tested whether a linear mapping from the feature windows to PCA shape scores was sufficient.
- **Ridge Regression** tested whether L2 regularization improved the same flattened-feature representation.
- **Random Forest Regression** tested whether nonlinear tree-based partitioning improved prediction relative to linear models.
- **LSTM Regression** tested whether preserving the five-frame temporal ordering improved prediction relative to flattened-window baselines.
- **Random Forest Feature Importance** inspected the strongest tree-based baseline using built-in impurity-based feature importances.
- **MLP Regression** tested generic nonlinear function approximation while using the same flattened feature representation as the classical baselines.

No hyperparameter optimization, model persistence, plotting, Track 21 evaluation, or descriptor recomputation was performed during this baseline sequence.

---

## 3. Overall performance

Validation metrics below are computed on held-out Track 14. Tracks 8 and 10 were used for training.

| Experiment | Feature group | MAE | RMSE | Median AE | R² |
|---|---|---:|---:|---:|---:|
| Linear Regression | Thermal-only | 2.535383 | 2.924938 | 2.407038 | -6.554562 |
| Linear Regression | SEM-only | 0.936175 | 1.206902 | 0.722373 | -0.152840 |
| Linear Regression | Thermal + SEM | 2.464486 | 2.863417 | 2.381854 | -7.072938 |
| Ridge Regression (`alpha=1.0`) | Thermal-only | 2.024324 | 2.383099 | 1.963877 | -2.268337 |
| Ridge Regression (`alpha=1.0`) | SEM-only | 0.935437 | 1.205774 | 0.722716 | -0.149425 |
| Ridge Regression (`alpha=1.0`) | Thermal + SEM | 2.069955 | 2.428553 | 2.007077 | -2.437421 |
| Random Forest Regression | Thermal-only | 1.892585 | 2.115979 | 1.821794 | -1.635473 |
| Random Forest Regression | SEM-only | 1.314589 | 1.664540 | 1.135141 | -0.848344 |
| Random Forest Regression | Thermal + SEM | 1.427145 | 1.747150 | 1.245874 | -0.946050 |
| LSTM Regression | Thermal-only | 3.093022 | 3.289154 | 3.125284 | -4.971182 |
| LSTM Regression | SEM-only | 1.441860 | 1.792883 | 1.183707 | -0.978544 |
| LSTM Regression | Thermal + SEM | 2.015820 | 2.293431 | 1.916156 | -1.907598 |
| MLP Regression | Thermal-only | 5.094667 | 5.474975 | 5.015851 | -15.362952 |
| MLP Regression | SEM-only | 1.378703 | 1.705670 | 1.186022 | -0.804784 |
| MLP Regression | Thermal + SEM | 4.685141 | 5.155220 | 4.652672 | -13.933672 |

The strongest validation MAE and RMSE were obtained by the SEM-only Linear/Ridge baselines, with Ridge slightly improving over Linear. Among nonlinear models, Random Forest provided the strongest overall baseline behavior, particularly for thermal-only and combined feature sets.

---

## 4. Principal findings

The baseline study supports the following evidence-based findings.

First, Ridge Regression improved over ordinary Linear Regression for all three feature groups. The improvement was small for SEM-only features but substantial for thermal-only and combined features. This indicates that regularization helped stabilize the flattened feature representation.

Second, Random Forest Regression improved substantially over the linear baselines for thermal-only and thermal + SEM feature groups. This suggests that nonlinear structure is present in the relationship between the feature windows and PCA geometry targets. However, Random Forest did not outperform the best SEM-only linear or Ridge models on the held-out validation track.

Third, the LSTM baseline did not improve validation performance despite preserving temporal ordering within the five-frame windows. The LSTM performed worse than Random Forest for all feature groups and worse than Ridge for all feature groups. This does not rule out temporal effects in general, but it does not support the current one-layer LSTM as a stronger baseline for these five-frame windows.

Fourth, the MLP baseline underperformed Random Forest despite also being nonlinear. This suggests that the Random Forest improvement cannot be attributed simply to generic nonlinear function approximation. In this development setting, tree-based partitioning was more effective than the tested feed-forward neural network.

Fifth, the Random Forest feature-importance analysis indicated that both thermal and SEM modalities contribute meaningfully in the combined model. The combined model assigned approximately balanced total impurity importance to the two modalities:

| Modality | Percent importance |
|---|---:|
| Thermal | 50.43% |
| SEM | 49.57% |

Sixth, temporal importance in the combined Random Forest model was distributed across the five-frame window. The highest aggregate importance occurred at timestep 0, but timestep 4 also contributed materially. The pattern does not strongly support the conclusion that sequence modeling over the current five-frame window is necessary.

---

## 5. Cross-track generalization

The development split used Tracks 8 and 10 for training and Track 14 for validation. This split evaluates cross-track generalization rather than within-track interpolation.

The Random Forest model fit the two training tracks consistently, especially for the combined thermal + SEM feature group:

| Feature group | Track | Split | MAE | RMSE | Median AE | R² |
|---|---:|---|---:|---:|---:|---:|
| Thermal + SEM | 8 | train | 0.448607 | 0.600540 | 0.340279 | 0.682730 |
| Thermal + SEM | 10 | train | 0.448670 | 0.602157 | 0.359260 | 0.729262 |

Validation performance on Track 14 was substantially worse:

| Feature group | Track | Split | MAE | RMSE | Median AE | R² |
|---|---:|---|---:|---:|---:|---:|
| Thermal-only | 14 | validation | 1.892585 | 2.115979 | 1.821794 | -1.635473 |
| SEM-only | 14 | validation | 1.314589 | 1.664540 | 1.135141 | -0.848344 |
| Thermal + SEM | 14 | validation | 1.427145 | 1.747150 | 1.245874 | -0.946050 |

These results show a substantial gap between training-track performance and held-out-track performance. The combined feature set produced the strongest Random Forest training performance, whereas SEM-only features produced the strongest Random Forest validation performance on Track 14.

Across all model families, SEM-only features gave the best held-out validation metrics. This is an observation from the current development split only and should not be interpreted as a universal conclusion about the dataset or the sealed track.

---

## 6. Interpretation

The baseline study indicates that the Phase III prediction task is not well described by a simple unregularized linear model. Ridge Regression improved over Linear Regression, especially for thermal-only and combined feature groups, indicating that regularization is beneficial for the flattened window representation.

The Random Forest results show that nonlinear interactions can improve performance, particularly for thermal-only and combined feature sets. However, the MLP results show that nonlinear function approximation alone was not sufficient to match the Random Forest. In the current data regime, the tree-based model appears to be a more effective nonlinear baseline than the tested neural-network models.

The LSTM results do not provide evidence that preserving temporal ordering within the five-frame windows improves prediction. The Random Forest feature-importance analysis also did not show a temporal-importance pattern that clearly favors sequence modeling. This suggests that, for the present window size and feature set, temporal structure may not be the dominant factor limiting prediction performance.

The modality comparison is more nuanced. Random Forest feature importance in the combined model was approximately balanced between thermal and SEM inputs, suggesting that both modalities contain predictive information. At the same time, SEM-only models achieved the strongest held-out Track 14 validation metrics. This may indicate that SEM-derived substrate context is more stable across the current development tracks, while thermal features may improve training-track fit but generalize less reliably in the present split.

The strong degradation from training tracks to validation Track 14 remains the central limitation of the baseline study. The current results should therefore be interpreted primarily as development-track baseline evidence, not as final evidence of sealed-track performance.

---

## 7. Baseline Study Conclusions

- The Phase III baseline study successfully evaluated linear, regularized linear, tree-based nonlinear, sequence neural-network, and feed-forward neural-network models.
- The target for all baseline experiments was the frozen PCA shape descriptor (`pc1`--`pc5`).
- Ridge Regression improved over ordinary Linear Regression for all tested feature groups.
- Random Forest Regression was the strongest nonlinear baseline and improved substantially for thermal-only and combined feature sets.
- SEM-only features produced the best held-out Track 14 validation metrics among the completed baseline experiments.
- Thermal + SEM features produced the strongest Random Forest training-track performance but did not produce the best held-out validation performance.
- The LSTM baseline did not improve validation performance despite preserving temporal ordering.
- The MLP baseline underperformed Random Forest, indicating that generic nonlinear neural-network approximation did not replicate the tree-based baseline behavior.
- Random Forest feature importance indicated meaningful contributions from both thermal and SEM modalities in the combined model.
- Cross-track generalization from Tracks 8 and 10 to Track 14 remains the primary limitation observed in the development evaluation.

---

## 8. Recommended next project phase

Baseline experimentation is complete. The project should now transition from model exploration to finalization.

Recommended next steps are:

1. freeze the final model and feature-set decision for sealed evaluation;
2. freeze the preprocessing, target-alignment, and reporting protocol;
3. evaluate the sealed Track 21 dataset once using the frozen pipeline;
4. generate publication-quality metric tables and figures;
5. prepare the final challenge report and paper-ready discussion.

Additional baseline architectures are not recommended at this stage unless future diagnostics provide a specific scientific reason to reopen model exploration.

Track 21 should remain sealed until the final pipeline and reporting protocol are frozen.
