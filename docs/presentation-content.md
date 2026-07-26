# Final Presentation Content & Script

**Target Duration:** 10 Minutes (approx. 1 minute per slide)
**Target Awards:** Most Innovative Approach, Best Presentation Design

---

## Slide 1: Title Slide
**Visual Elements:** 
- A sleek, high-contrast dark theme background (premium design).
- Title: **From Deterministic Failure to Probabilistic Certainty**
- Subtitle: Predicting Local Geometric Variation from In-Situ Thermal Dynamics
- Presenter Names / Team Name

**Talking Points (Script):**
> "Good afternoon judges. Today we are presenting our solution for predicting local geometric variation of laser tracks. Rather than presenting a standard black-box model, we want to take you through our scientific journey. A journey that started with a hypothesis, hit a mathematical wall, and led to a fundamental pivot in how we understand in-situ thermal data."

---

## Slide 2: The Core Challenge
**Visual Elements:** 
- A simple graphic showing the laser scan path (X-axis) and the thermal melt-pool.
- Two bullet points:
  1. The Goal: Map in-situ thermal features to final processed track geometry.
  2. The Challenge: Extreme spatial noise in Wyko profilometry measurements.

**Talking Points (Script):**
> "Our primary goal was to take the continuous sequence of thermal images recorded during the laser scan and predict the final geometry of the track. The track is not a perfect line; it fluctuates. Our challenge was to map these in-situ thermal features—peak temperature, melt pool area, and length—to those high-frequency spatial fluctuations."

---

## Slide 3: Data & Target Alignment
**Visual Elements:** 
- A 3-step diagram (e.g., Chevron arrows): `In-Situ Thermal (0.2mm resolution)` -> `Unified X-Axis (20mm to 100mm)` -> `Wyko Heightmap`.
- Focus on the term: **Unified Spatial Anchoring**.

**Talking Points (Script):**
> "To build any model, we first needed perfect alignment. We established a unified physical X-axis from 20 to 100 millimeters. This allowed us to spatially synchronize the discrete thermal frames directly with the dense, continuous Wyko profilometry data, ensuring every target prediction was grounded in true physical space."

---

## Slide 4: The False Summit (Deterministic Failure)
**Visual Elements:** 
- A bold, red "FAILED" stamp or similar striking visual over a table showing negative $R^2$ values.
- Key text: "Hypothesis Falsified: Exact local width cannot be deterministically mapped."

**Talking Points (Script):**
> "Initially, we did what most teams likely attempted: we tried to predict the exact, point-estimate width of the track using complex feature engineering, including SEM substrate data. The result? Our Leave-One-Track-Out validation showed negative R-squared values across the board. The model suffered from severe spatial overfitting because the thermal sensors simply lack the spatial bandwidth to predict sub-millimeter extraction noise."

---

## Slide 5: The Identifiability Pivot (Our Innovation)
**Visual Elements:** 
- A clean comparison graphic. 
- Left side (Crossed out): "Predicting a point estimate (Deterministic)".
- Right side (Highlighted): "Predicting a probability envelope (Probabilistic)".

**Talking Points (Script):**
> "This failure wasn't a dead end; it was our biggest breakthrough. We realized that while the sensors lack local spatial bandwidth, they perfectly capture the macro-thermal process regime. This led to our central innovation: The Identifiability Pivot. Instead of fighting the mathematical impossibility of deterministic prediction, we pivoted to probabilistic bounding—predicting the probability distribution of the track boundaries."

---

## Slide 6: Refined Feature Engineering
**Visual Elements:** 
- A visual representation of a "Rolling Window" spanning 5 frames ($t-2$ to $t+2$).
- The three core features: `peak_temp`, `sqrt_mp_area`, `mp_length`.

**Talking Points (Script):**
> "To power this probabilistic approach, we stripped away the noisy, overfitting features like SEM data. We focused purely on the physics: peak temperature, melt pool area, and length. By constructing a centered 5-frame rolling window for every point, we captured the transient thermal state of the melt pool as it moved across the substrate."

---

## Slide 7: Bayesian Ridge Architecture
**Visual Elements:** 
- A simple math or architecture block: `StandardScaler` -> `BayesianRidge`.
- Key equations (simplified): Outputting $\mu$ (mean) and $\sigma^2$ (variance).

**Talking Points (Script):**
> "For our model architecture, we selected Bayesian Ridge Regression. We chose this specifically because it natively outputs a posterior predictive distribution—both a mean and a variance—without requiring artificial post-hoc calibration. It learns the inherent uncertainty in the process directly from the data."

---

## Slide 8: Validation & Quantitative Results
**Visual Elements:** 
- Copy the `crps_comparison_by_fold.png` plot from the Jupyter Notebook outputs (comparing Bayesian Ridge native to Naive baseline).
- Big callout text: **28.3% CRPS Improvement**.

**Talking Points (Script):**
> "We validated our approach using strict Leave-One-Track-Out cross-validation. We evaluated our model against a naive training-only Gaussian baseline using the Continuous Ranked Probability Score, or CRPS. Our Bayesian approach achieved a 28.3% relative improvement over the baseline. More importantly, it achieved near-perfect empirical coverage at the 90% nominal target level."

---

## Slide 9: Final Blind Inference (Track 21)
**Visual Elements:** 
- Copy the large `track21_predictive_mean_90pct_interval.png` plot from the Jupyter Notebook. 
- Ensure the blue envelope (90% interval) is highly visible.

**Talking Points (Script):**
> "This culminates in our final blind inference on Track 21. As you can see, the model outputs a smooth, dynamic predictive mean (the blue line) surrounded by a continuous 90% confidence spatial envelope (the shaded area). This proves that while we cannot deterministically pinpoint every micro-fluctuation, we can safely and reliably bound the geometric variation."

---

## Slide 10: Conclusion & Impact
**Visual Elements:** 
- Three summary bullet points focusing on real-world impact.
  1. Rigorous hypothesis falsification prevents false confidence.
  2. Probabilistic bounding is safer for manufacturing control.
  3. Simple, physically grounded features outperform complex, overfit architectures.

**Talking Points (Script):**
> "In conclusion, our most innovative approach was knowing when to abandon a flawed deterministic hypothesis. For future real-world manufacturing, knowing the reliable bounds of variation (the uncertainty envelope) is far more valuable and safe than relying on an overfit, false point-estimate. Thank you, and we welcome your questions."
