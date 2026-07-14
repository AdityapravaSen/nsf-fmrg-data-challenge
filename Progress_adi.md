### first things first
1. Repo forked and cloned
2. dataset downloaded
3. 01_starter_code and 02_starter code ran and checked
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

### Step-by-step
#### 1. Validate laser on/off detection (sanity check, ~30 min)

* Run detect_laser_on_interval on all 4 tracks, plot score vs frame index with the threshold line and the detected on_start/on_stop marked.
* Confirm visually: does the detected window actually correspond to the laser being on? Check 1-2 raw frames just before/after the boundary.
* Flag any track where detection looks wrong — better to catch this now than 3 weeks in.

#### 2. Build melt-pool descriptors per frame (this is Goal 1 of the actual challenge)
For each of the 400 extracted thermal frames per track, compute:

* melt pool region (threshold-based segmentation — pick a temperature/intensity cutoff, or Otsu)
* size (pixel area → mm²), centroid, major/minor axis length (shape/aspect ratio)
* peak temperature, mean temperature in pool, temperature gradient at pool boundary
* asymmetry (e.g. compare leading vs trailing half of pool relative to scan direction)
* cooling-tail length (how far behind the pool centroid does intensity stay elevated above background)
* frame-to-frame change (Δcentroid position, Δsize) — this doubles as a scan-speed/consistency check: does centroid step size match the expected 0.2mm/frame?

>This produces a per-frame feature vector — this is the actual thermal descriptor table the model will eventually consume, not just raw frames.

#### 3. SEM tile → x-position mapping + substrate feature extraction

* Map each SEM tile to its physical x-range (remember: tile 01 = 100mm side, so reverse-numbered vs. thermal/height convention — need to flip to match common x-axis direction)
* Since tiles are coarser resolution than the 400 thermal frames, decide the binning: which SEM tile(s) correspond to which thermal-frame x-range
* Extract substrate-only texture features from each tile (excluding processed track region — you'll need a way to identify that region, e.g. a fixed exclusion band, or intensity-based segmentation if the track is visually distinct in SEM)
* Candidate substrate features: local roughness/texture (e.g. GLCM contrast, local standard deviation), porosity-like defects, grain pattern irregularity — these become the "substrate condition" side of Goal 4

#### 4. Unify onto common x-grid

* Pick a common x-grid resolution (probably the thermal frame resolution, ~0.2mm steps, since it's the finest of the three)
* For each x-bin: thermal frame(s) + interpolated/nearest SEM tile features
* Output as one clean data structure (I'd suggest a pandas DataFrame or a saved .npz/.parquet per track — Person B's height-map targets will get merged onto this same x-grid in Phase 2)

#### 5. Visual QC (non-negotiable before merging)

* Plot: thermal melt-pool centroid trajectory overlaid on x-position, peak temp vs x, SEM substrate roughness vs x — for all 4 tracks, side by side
* This is what you show your friend before merging into main — "here's proof the alignment holds"