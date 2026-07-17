### first things first
1. Repo forked and cloned
2. dataset downloaded
3. 01_starter_code and 02_starter_code ran and checked
4. 2 branches created
    1. Adi
    2. Nabarun

# Planning and Structure
## Phase 1 — Data Understanding & Alignment (blocks everything else)
## Adi — thermal + SEM side

### Goal of this phase
>Produce one reliable function/module that, for a given track, returns a common x-axis (20→100mm) with, at each position:

* the corresponding thermal frame(s) (melt-pool-centered),
* the corresponding SEM tile/patch (substrate context, track region flagged/excluded),
* everything indexed consistently so Person B's height-map targets line up frame-for-frame.

This is the "Rosetta stone" — get it right once, both of you build on it.

# Phase 1 Progress Report: Data Understanding & Alignment
**Branch:** `Adi` | **Focus:** Thermal & SEM Modality Alignment

## 🎯 Phase Objective
The primary goal of Phase 1 was to establish the "Rosetta Stone" of the dataset: a strictly aligned, common physical X-axis (20.1 mm to 99.9 mm) that unifies the disparate temporal and spatial resolutions of the thermal camera and the SEM imagery. This foundational alignment is required before any machine learning models can map thermal features to final track geometry (Phase 2+).

---

## ✅ Key Achievements & Steps Completed

### 1. Laser On/Off Validation & Frame Extraction
* **Action:** Ran adaptive thresholding (`detect_laser_on_interval`) on global frame intensities (99.5th percentile) for Tracks 8, 10, 14, and 21.
* **Validation:** Successfully isolated exactly **400 active frames** per track. 
* **Physics Check:** At a travel speed of 10 mm/s (0.2 mm/frame), 400 frames perfectly equate to the required **80 mm active window** (spanning from ~20.1 mm to ~99.9 mm).

### 2. Melt-Pool Feature Engineering (Challenge Goal 1)
* **Action:** Built a processing pipeline (`03_phase1_melt_pool_feature_extraction.ipynb`) using `skimage` to segment the melt pool per frame.
* **Methodology:** Applied **Otsu's thresholding** to clean, non-zero pixel arrays to dynamically segment the melt pool from the substrate.
* **Features Extracted:** Area (pixels), Centroid (X, Y), Major Axis Length, Minor Axis Width, Peak Temperature, and Mean Region Temperature.
* **Key Insight:** Analyzed centroid shifts and confirmed that the Stratonics thermal camera is mounted **co-axially**. The melt pool remains relatively stationary near the center of the 400x400 frame while the substrate moves beneath it.

### 3. SEM Substrate Spatial Mapping
* **Action:** Built a spatial mapping script (`03_phase1_melt_pool_feature_extraction.ipynb`) to align Zeiss SEM substrate imagery with the thermal X-axis.
* **Challenges Solved:**
    * Corrected the **reverse-coordinate scan direction** (SEM Tile 01 starts at 100mm and moves backwards to 20mm).
    * Implemented **dynamic step-size calculation** based on file counts (e.g., 100mm / 13 or 14 tiles = ~7.1 to 7.6 mm Field of View).
    * Resolved a missing file anomaly for Track 14 caused by a local naming error.
* **Methodology:** Masked out the central 30% processed track to isolate pure, unannotated substrate. Extracted local pixel variance as a proxy for **substrate roughness**.

### 4. Modality Unification (The Rosetta Stone)
* **Action:** Merged the high-resolution Thermal data (0.2 mm steps) with the coarse-resolution SEM data (~7.5 mm steps).
* **Methodology:** Utilized `pandas.merge_asof` (Nearest Neighbor Interpolation) sorted strictly by `x_position_mm` to staple the nearest valid SEM substrate roughness value to every single thermal frame without distorting physical truth.

### 5. Visual Quality Control (QC)
* **Action:** Generated twin-axis validation plots comparing Continuous Thermal Melt Pool Area alongside Binned/Stepped SEM Substrate Roughness over the shared 20mm -> 100mm X-axis.
* **Result:** Confirmed perfect spatial bounds and signal alignment.

---

## 📁 Artifacts Generated
All scripts executed successfully, producing the following assets in `/processed_data/`:
1. `phase1_thermal_features.csv` *(1,600 rows: 4 tracks x 400 frames)*
2. `phase1_sem_features.csv` *(Coarse physical bounds and midpoint mapping)*
3. `phase1_unified_master.csv` *(The final, merged foundational dataset)*

[![download.png](https://i.postimg.cc/JhMh7Bd2/download.png)](https://postimg.cc/tsvb27b3)

## 🚀 Next Steps (Handoff to Phase 2)
The unified dataset is ready for **Nabarun**. 
**Phase 2 Goal:** Load the Bruker/Wyko height-map arrays, slice them at these exact `x_position_mm` anchors, compute the final geometric target variables (local track width, boundary irregularity), and append them directly to `phase1_unified_master.csv`.

# Progress Log: Feature Engineering & Data Alignment (Person A / Adi)

**Role:** Thermal & SEM Feature Engineering, Data Preprocessing, Model Input Pipeline  
**Current Phase:** Phase III (Predictive Modeling - Feature Pipeline Complete)  
**Sealed Track Policy:** Track 21 is strictly held out.  

---

*The goal of the current phase is to prepare the structured data for Deep Learning models. Following the strict division of labor, I own the feature inputs ($X$), while Person B owns the targets ($Y$).*

**1. Schema Definition & Filtering:**
* Defined the strict input schema (7 thermal features, 2 SEM features).
* Implemented a filter to dynamically drop rows where `pca_ready == False`, ensuring the model only trains on physically valid geometry regions.

**2. Leakage-Proof Scaling:**
* Built the `FeaturePreprocessor` class with a custom Scikit-Learn `StandardScaler`.
* **Crucial Rule Maintained:** Implemented Leave-One-Track-Out (LOTO) splitting. The scaler is fitted *only* on the training tracks (e.g., 8 and 10) and safely transformed the validation track (14) to completely prevent spatial data leakage. 

**3. Thermal History Windowing:**
* Heat diffuses and builds up over time. I wrote a transformation method `create_sequence_windows` to convert the flat tabular data into 3D sequential arrays.
* The feature data is now formatted as `(Samples, Window_Size, Features)` (e.g., Shape: `423, 5, 9`), making it natively compatible with PyTorch 1D-CNNs and LSTMs.

---

## 🚀 Immediate Next Steps / Blockers
* **Handoff to Person B:** I have generated a `meta_df` alongside my 3D windowed arrays. This dataframe contains the exact `track_id` and `x_position_mm` of the *last* frame in every sequence window.
* **Waiting on Target Extraction:** Person B will use my `meta_df` to slice and align the target variables ($Y$). 
* **Joint Task:** Once Person B pushes the target extraction script, we will combine our tensors into a PyTorch `DataLoader` and begin training the baseline models.