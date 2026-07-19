# NSF Future Manufacturing Data Challenge

## Height-Map Target Extraction --- Progress and Scientific Decision Log

**Workstream:** Bruker/Wyko height-map target extraction (Person B)\
**Status:** Exploratory method development\
**Development tracks:** 8, 10, 14\
**Sealed track:** 21 --- do not load or inspect during method
development\
**Last completed major experiment:** Experiment 14 --- Phase 2
merge-point validation\
**Current scientific status:** Experiments 03--14 completed; Phase III engineering infrastructure and baseline modeling completed; LOTO validation completed (Track 21 remains sealed)\
**Current project phase:** Phase III pipeline frozen for sealed Track 21 evaluation\
**Current focus:** Execute the frozen sealed Track 21 evaluation and finalize reporting

------------------------------------------------------------------------

# 1. Purpose

This is the cumulative scientific and engineering record for the
height-map workstream. It records what was tested, the numerical
evidence obtained, assumptions discovered, failed hypotheses, and
current decisions.

The central principle is:

> Define a physically defensible geometric target before training a
> predictive model.

A sophisticated multimodal model cannot rescue an arbitrary or unstable
target definition.

------------------------------------------------------------------------

# Project phases

## Phase I (Completed) --- scientific discovery (Experiments 03--10)

Goals:

• Understand height-map morphology

• Identify failure modes

• Eliminate unstable assumptions

• Establish descriptor requirements

## Phase II (Completed) --- geometry representation design (Experiments
11--14)

Goals:

• Compare descriptor implementations

• Select a physically meaningful prediction target

## Phase III (Current) --- multimodal predictive modeling

Goals:

• Predict local geometry descriptors from thermal and SEM observations

• Validate the final baseline selection using development-only Leave-One-Track-Out (LOTO) across Tracks 8, 10, and 14 (completed; Track 21 remained sealed)

• Evaluate once on sealed Track 21 using the frozen pipeline

------------------------------------------------------------------------

# 2. Challenge and data context

The NSF Future Manufacturing Data Challenge concerns probabilistic
prediction of local geometric variation in single directed energy
deposition (DED) laser tracks using:

1.  thermal videos from a Stratonics ThermaViz melt-pool sensor;
2.  SEM images from a Zeiss EVO MA10;
3.  Bruker/Wyko full-field height maps from a white-light 3D optical
    profilometer.

Relevant conventions:

-   common physical analysis interval: approximately 20--100 mm;
-   height-map x and y: mm;
-   height-map z: nm;
-   raw ASC local x=0 corresponds to the physical 100 mm side;
-   organizer code remaps height maps to increasing 20--100 mm physical
    x order;
-   native height-map x spacing is approximately 0.004 mm;
-   thermal frame rate is 50 fps;
-   scan speed is 10 mm/s;
-   consecutive thermal frames correspond to approximately 0.2 mm laser
    travel.

The coordinate reversal is important for eventual multimodal alignment.

------------------------------------------------------------------------

# 3. Collaboration split

## Person A --- thermal + SEM

Responsibilities:

-   validate thermal laser on/off extraction;
-   validate thermal physical x positions;
-   confirm reversed SEM tile numbering;
-   map thermal frame index, SEM tile, and common x coordinate;
-   establish multimodal alignment.

## Person B --- height maps

Responsibilities:

-   validate profilometry loading and detrending;
-   characterize missingness and data quality;
-   define the local geometric target;
-   extract defensible boundaries, width, and/or other descriptors;
-   produce an aligned target representation for downstream modeling.

This report covers Person B's work.

------------------------------------------------------------------------

# 4. Repository and environment setup

The team fork is configured as `origin`; the organizer repository is
configured as `upstream`. Normal pushes to `origin` therefore update the
team fork.

The dataset was placed in the repository's expected `data/` layout and
the organizer starter notebook was validated.

The intended Python interpreter is:

`/opt/homebrew/opt/python@3.11/bin/python3.11`

The local workflow intentionally avoids virtual environments.

A repository-level file was added:

`.github/copilot-instructions.md`

Its purpose is to give coding agents persistent constraints, including:

-   do not create virtual environments;
-   use the existing Homebrew Python 3.11 interpreter;
-   do not modify organizer files without explicit instruction;
-   do not inspect Track 21;
-   do not commit or push unless explicitly requested.

A VS Code/Copilot behavior was discovered: notebook tooling can trigger
environment creation when a notebook has no selected kernel even after
the agent reads repository instructions. The workaround is to manually
select Homebrew Python 3.11 before notebook-agent execution. Later
experiments therefore prefer direct script execution using the absolute
interpreter path.

------------------------------------------------------------------------

# 5. Starter-code validation

The organizer workflow successfully loaded:

-   thermal data;
-   thermal frames in physical units;
-   SEM tiles;
-   Bruker/Wyko height maps.

An early `NameError: Z_detrended is not defined` was caused by notebook
execution order/kernel state, not bad data. Once prerequisite cells were
executed correctly, the notebook ran.

Example Track 8 thermal validation:

-   raw shape: `(929, 400, 400)`;
-   extracted shape: `(400, 400, 400)`;
-   physical x coverage: approximately 20.1--99.9 mm.

The inspected Track 8 SEM example returned 13 tiles.

This established that paths, repository layout, and organizer loaders
were operational.

------------------------------------------------------------------------

# 6. Core scientific question

The difficult problem is not loading the height map. It is:

> What exactly constitutes the local DED track boundary in profilometry?

The initial working assumption was:

> The deposited track is a positive elevated component above the local
> substrate baseline.

That leads naturally to:

1.  detrend;
2.  estimate substrate baseline/spread;
3.  threshold above baseline;
4.  find connected y-components;
5.  score candidates;
6.  select left/right boundaries;
7.  compute width.

Experiments 03--05 increasingly questioned whether this assumption
describes the full processed-track geometry.

------------------------------------------------------------------------

# 7. Experiment 03 --- exploratory target extraction

## Artifact

`notebooks/03_heightmap_target_extraction_exploration.ipynb`

Purpose: understand the profilometry and construct a first local
extractor on Tracks 8, 10, and 14.

Track 21 remained sealed.

## Detrending

The organizer provided robust global plane detrending. Exploration
suggested that fitting substrate while excluding the presumed central
track region often produced more representative substrate behavior.

The exploratory substrate-focused plane fit excluded:

-   y = 0.65--1.35 mm

This is an important caveat: the exclusion region itself is a structural
prior. Removing a later center-scoring term does not remove all
assumptions about track location.

## Initial extraction formulation

For each x cross-section:

1.  estimate robust substrate baseline;
2.  estimate robust spread;
3.  define `threshold = baseline + k * spread`;
4.  mark finite samples above threshold;
5.  preserve contiguous finite runs;
6.  identify connected components;
7.  reject implausible components;
8.  score candidates;
9.  reject low-score/ambiguous cases;
10. optionally refine selected boundaries with local gradients.

Initial threshold:

-   `k = 2.5`

Candidate sanity filters:

-   minimum component samples: 6;
-   minimum width: 0.035 mm;
-   maximum width: 1.10 mm.

## Candidate scoring

Four terms were used.

### Height evidence

Weight: 2.0.

### Preferred-width score

-   preferred width: 0.32 mm;
-   width scale: 0.30 mm;
-   weight: 1.0.

### Center score

-   prior band: 0.65--1.35 mm;
-   weight: 1.5.

### x-continuity

-   continuity scale: 0.18 mm;
-   weight: 0.6.

## Rejection and ambiguity

The extractor allowed explicit failure states rather than forcing a
target:

-   `selected`;
-   `ambiguous`;
-   `low_score`;
-   `no_valid_component`.

Ambiguity ratio:

-   0.85.

This conservative behavior was intentional: explicit missingness is
preferable to a confidently wrong training target.

## Gradient refinement

For selected components, local gradient refinement used:

-   smoothing window: 9 samples;
-   minimum finite run: 15 samples;
-   edge search margin: 0.08 mm;
-   minimum edge strength: 50 µm/mm.

No global NaN filling was used.

## Main observations

The profilometry was not a clean raised-bead segmentation problem.
Observed issues included:

-   fragmented finite support;
-   NaN gaps capable of splitting genuine structures;
-   multiple elevated structures;
-   ambiguous cross-sections;
-   substantial track-to-track quality differences.

Track 10 appeared especially fragmented.

Two major concerns emerged:

1.  center/width priors might manufacture the selected component;
2.  the MAD threshold might determine existence and identity of
    components.

Experiment 04 tested these concerns.

------------------------------------------------------------------------

# 8. Experiment 04 --- prior/parameter sensitivity audit

## Artifact

`notebooks/04_heightmap_prior_sensitivity.ipynb`

Completed run:

`processed_data/run_outputs/04_heightmap_prior_sensitivity_20260713_194307/`

The notebook executed completely. A later Copilot/TAMU request ended in
HTTP 524, but the computational work had already finished with no
notebook Python errors.

## Questions

1.  Does the center prior materially determine selections?
2.  Does preferred width 0.32 mm steer predicted widths?
3.  Which score terms dominate?
4.  How sensitive is extraction to the MAD multiplier?
5.  Should the current extractor be refined or reformulated?

## CONTROL reproduction

Notebook 04 exactly reproduced Notebook 03's substrate-focused CONTROL.

  Track     Selected   Ambiguous   No valid
  ------- ---------- ----------- ----------
  8                4           1          3
  10               2           3          3
  14               4           0          4

Total selected:

-   **10/24**

Boundary differences from Notebook 03 were numerically negligible.

## Center-prior sensitivity

Configurations included:

-   CONTROL;
-   no center score;
-   weak center;
-   broad center prior;
-   data-derived center prior.

Derived pooled bands:

-   broad: approximately 0.368--1.750 mm;
-   data-derived: approximately 0.656--1.462 mm.

Result:

-   center ablation changed status in only **1/24** cases;
-   **zero selected-component switches** occurred among cases that
    remained selected.

Interpretation:

The center score mostly nudged borderline status decisions. It was not
choosing different components in robust selected cases.

## Width-prior sensitivity

Configurations included:

-   CONTROL;
-   no width score;
-   broad width preference;
-   data-derived width;
-   preferred-width sweep.

The pooled candidate-width median was approximately:

-   **0.123 mm**

Despite this large difference from the hand-selected 0.32 mm preference:

-   no-width-score changed status in **1/24**;
-   broad width preference changed status in **1/24**;
-   data-derived width changed status in **3/24**;
-   selected-component switches remained **zero**.

Preferred-width sweep:

  Preferred width     Selected
  ----------------- ----------
  0.20 mm                    8
  0.26 mm                    9
  0.32 mm                   10
  0.40 mm                   11
  0.50 mm                    9

Robust selections did not show strong monotonic width tracking.

Interpretation:

The 0.32 mm preference was not simply dragging output widths toward 0.32
mm.

## Score-term ablations

-   no height score: 9 same-selected; 1 selected→nonselected; 1
    nonselected→selected;
-   no width score: 9 same-selected; 1 selected→nonselected;
-   no center score: 10 same-selected; 1 nonselected→selected;
-   no continuity score: 9 same-selected; 1 selected→nonselected;
-   weak center/width priors: all 10 CONTROL selections preserved and 2
    nonselected cases promoted.

No individual scoring prior explained the dominant instability.

## Threshold sensitivity --- decisive result

Threshold sweep:

  MAD multiplier     Selected   Ambiguous   Invalid
  ---------------- ---------- ----------- ---------
  1.5×                     16           5         3
  2.0×                     14           4         6
  2.5×                     10           4        10
  3.0×                      8           4        12
  3.5×                      5           5        14

Key result:

-   **18/24 representative cases were unstable across the threshold
    sweep.**

Some cases selected **3--5 materially different components** across
thresholds.

The threshold therefore controlled:

1.  whether a candidate existed;
2.  selected/ambiguous/invalid status;
3.  sometimes which component represented the track.

## Stability classes

Across 24 representative cases:

-   consistently invalid: 13;
-   stable: 9;
-   highly sensitive: 1;
-   moderately sensitive: 1.

## Experiment 04 conclusion

The hypothesis that center and 0.32 mm width priors were the main
problem was largely disproved.

The dominant failure mode was:

`single threshold → binary mask → connected components → score`

The recommendation became:

> Reformulate component selection rather than tune a single "best" MAD
> multiplier.

------------------------------------------------------------------------

# 9. Experiment 05 --- multi-threshold persistence + x corridor

## Motivation

Experiment 04 showed single-threshold brittleness.

Hypothesis 1:

> A physical structure should persist across several reasonable
> thresholds.

Hypothesis 2:

> The track is spatially coherent along x and should not be solved as
> fully independent cross-sections.

## Artifact and execution

Script:

`scripts/05_heightmap_persistence_corridor_experiment.py`

Executed with:

`/opt/homebrew/opt/python@3.11/bin/python3.11 scripts/05_heightmap_persistence_corridor_experiment.py`

Successful output:

`processed_data/run_outputs/05_heightmap_persistence_corridor_20260713_201054/`

## Analysis grid

-   x spacing: **0.2 mm**

This is roughly 50× coarser than native x spacing.

## Local aggregation

At each analysis x:

-   half-window: 0.2 mm;
-   total x-window: approximately 0.4 mm;
-   robust aggregation: per-y median across nearby native columns.

Rules:

-   preserve finite masks;
-   no global NaN filling;
-   no interpolation across long missing regions;
-   explicitly track finite support.

## Multi-threshold candidates

FULL threshold set:

-   \[1.5, 2.0, 2.5, 3.0, 3.5\] × robust spread.

Each candidate stored:

-   y_min;
-   y_max;
-   centroid;
-   width;
-   finite sample count;
-   median height above baseline;
-   peak height above baseline.

No threshold was canonical.

## Cross-threshold families

Components at a fixed x were grouped primarily by y-space overlap.

Implemented matching:

-   interval IoU ≥ 0.1 or containment;
-   centroid distance ≤ 0.25 mm.

Components were not matched solely by similar width.

Representative interval:

-   median-threshold component if available;
-   otherwise maximum-peak component.

## Node evidence

A family became a node only if:

-   persistence fraction ≥ 0.5;
-   finite fraction ≥ 0.1.

Node score combined:

-   persistence, weight 2.0;
-   log median height, 0.7;
-   log peak height, 0.3;
-   cross-threshold stability, 1.0;
-   finite coverage, 0.8.

The strong preferred-width and narrow center-score priors were removed.

## Corridor optimization

Persistent families at successive x positions were treated as
graph/sequence nodes.

A Viterbi-style dynamic-programming path maximized global evidence.

Transition penalties:

-   centroid displacement: weight 2.0;
-   left-boundary displacement: 1.0;
-   right-boundary displacement: 1.0;
-   width change: 0.5.

Gaps:

-   maximum gap steps: 5;
-   gap-open penalty: 1.0;
-   extra skipped-step penalty: 1.5.

Gradient refinement was deliberately excluded to isolate candidate
selection.

## Representative-case comparison with Experiment 04 CONTROL

Representative x positions:

-   25, 35, 45, 55, 65, 75, 85, 95 mm.

### Track 8

-   35 mm: both selected, but poor spatial agreement; corridor much
    narrower.
-   45 mm: CONTROL ambiguous → corridor selected.
-   65/75/85 mm: CONTROL selected → corridor invalid.

### Track 10

CONTROL selected 55 and 75 mm.

The FULL corridor rejected both and remained invalid almost everywhere.

### Track 14

-   25 mm: CONTROL invalid → corridor selected.
-   35/55/65/85 mm: CONTROL selected → corridor invalid.

The corridor resolved a few prior failures but rejected many CONTROL
selections. More valid cases were explicitly not treated as the success
criterion.

## Threshold-set robustness

Threshold sets:

-   FULL: \[1.5, 2.0, 2.5, 3.0, 3.5\]
-   REMOVE_LOW: \[2.0, 2.5, 3.0, 3.5\]
-   REMOVE_HIGH: \[1.5, 2.0, 2.5, 3.0\]
-   INNER: \[2.0, 2.5, 3.0\]

### Track 8

FULL valid locations:

-   **97**

REMOVE_HIGH and INNER were identical where jointly valid in reported
shifts.

REMOVE_LOW:

-   median absolute center shift ≈ 0.004 mm;
-   median absolute width shift ≈ **0.068 mm**;
-   p95 absolute width shift ≈ **0.233 mm**.

### Track 10

FULL valid locations:

-   **1**

The method effectively collapsed on this track.

### Track 14

FULL valid locations:

-   **38**

REMOVE_HIGH and INNER again showed zero reported shifts where jointly
valid.

REMOVE_LOW:

-   median absolute width shift ≈ **0.048 mm**;
-   p95 absolute width shift ≈ **0.138 mm**.

## Experiment 05 conclusion

The low 1.5×MAD threshold materially affected family width.

Therefore the persistence method remained dependent on threshold-set
composition.

The scientific concern is:

> We may have replaced dependence on one arbitrary threshold with
> dependence on an arbitrary set of thresholds.

Experiment 05 did not clearly reduce the fundamental brittleness
identified in Experiment 04.

## Implementation-level failure modes

The executed audit identified:

1.  persistence ≥ 0.5 and finite_fraction ≥ 0.1 may over-gate nodes;
2.  matching against the last family component may fragment real
    structures;
3.  low-threshold inclusion materially affects widths;
4.  the corridor appears too conservative rather than too eager to
    bridge gaps.

Suggested technical revisions included:

-   family-level interval matching;
-   soft rather than hard persistence gating;
-   explicit null nodes.

These revisions are technically sensible.

**Current decision: do not tune the corridor yet.**

------------------------------------------------------------------------

------------------------------------------------------------------------

# 10. Scientific interpretation after Experiments 03--05

The evidence after the first three extraction experiments formed the
following chain:

1.  Experiment 03 built a positive-height connected-component extractor.
2.  The initial concern was that hand-coded center and preferred-width
    priors might manufacture the selected component.
3.  Experiment 04 showed those explicit scoring priors were
    comparatively weak drivers.
4.  The dominant instability was the threshold used to define positive
    components.
5.  Experiment 05 replaced a single threshold with multi-threshold
    persistence and x-direction corridor optimization.
6.  Widths still depended materially on threshold-set composition,
    especially inclusion of 1.5×MAD.
7.  Track 10 produced only one valid FULL-corridor location.

This motivated a more fundamental question:

> Are we optimizing a segmentation algorithm before validating the
> geometric quantity being segmented?

All methods through Experiment 05 implicitly treated the relevant track
geometry as a positive elevated structure above substrate. That
assumption had not been validated.

The possible processed morphology could instead include:

-   a raised crown;
-   depressed or remelted margins;
-   mixed-sign disturbance;
-   asymmetric geometry;
-   a broader disturbed surface than the positive crown;
-   multiple structural transitions;
-   boundaries defined by return to substrate-like behavior.

This led to Experiment 06.

------------------------------------------------------------------------

# 11. Experiment 06 --- geometry-definition audit

## Purpose

Experiment 06 was designed to challenge the assumption:

> full local processed-track width = width of a positive elevated
> component above substrate.

The experiment compared four conceptual geometric definitions on Tracks
8, 10, and 14 while keeping Track 21 sealed.

Artifact:

`scripts/06_heightmap_geometry_definition_audit.py`

The latest corrected Experiment 06 run and its complete tables,
diagnostics, and dense-grid figures were subsequently reviewed directly
rather than relying only on the coding agent's summary.

## Method A --- positive excursion

Method A represents the original conceptual assumption:

> the relevant region is significantly above the substrate baseline.

The corrected audit retained thresholded positive-excursion candidates
across:

-   1.5× robust spread;
-   2.0×;
-   2.5×;
-   3.0×;
-   3.5×.

This method is now interpreted as a possible **raised-crown
descriptor**, not automatically as full processed-track width.

## Method B --- absolute substrate deviation

Method B uses absolute residual magnitude relative to substrate and
therefore allows:

-   positive elevation;
-   negative depression;
-   mixed-sign disturbance.

Conceptually, this asks whether the processed region is better described
as surface disturbance rather than positive bead elevation.

## Method C --- profile-transition boundaries

Method C attempts to identify structural transitions in the y-z profile.

The corrected implementation:

-   operates within finite runs;
-   computes local gradient-transition candidates;
-   retains multiple transition candidates;
-   selects left and right transitions relative to a shared disturbance
    seed.

The executed diagnostics later revealed that Method C's selected
boundaries can jump violently across the y-domain from one x location to
the next.

## Method D --- substrate-return boundaries

Method D identifies a contiguous disturbance seed and searches
independently left and right for persistent substrate-like return.

The return logic uses local residual and gradient/activity evidence.

This was initially considered the most scientifically promising
definition because it could, in principle, include mixed-sign disturbed
morphology.

The executed audit later showed that Method D's interpretation is
inseparable from the identity of the disturbance seed supplied to it.

------------------------------------------------------------------------

# 12. Experiment 06 initial-run confound and correction

The first Experiment 06 execution used:

`global baseline = median(z over all finite y)`

for primary residual interpretation.

This was rejected as a scientific confound.

The disturbed region itself could influence the quantity being called
the substrate baseline. In particular, an approximately balanced
positive/negative residual split in Track 10 could be partially
manufactured by centering the profile around its own median.

The primary audit was therefore corrected to use the established
substrate-focused assumption:

-   exclude y = 0.65--1.35 mm from global substrate estimation;
-   estimate baseline and robust spread from finite samples outside that
    band;
-   preserve minimum-sample fallback logic;
-   explicitly record fallback to whole-profile finite samples when
    required;
-   independently estimate broad left and right local substrate
    statistics.

The corrected implementation also fixed z-unit naming so baseline
differences are represented in micrometers rather than incorrectly using
an `_mm` suffix.

This correction established an important methodological rule:

> The reference called "substrate" must be estimated from intended
> substrate evidence, and fallback must be explicit.

The initial Track 10 mixed-sign interpretation was treated as
provisional until the corrected run.

------------------------------------------------------------------------

# 13. Experiment 06 candidate-selection corrections

The first Experiment 06 implementation also selected the widest
contiguous Method A or B region.

This was rejected because it confounded:

1.  the geometric definition;
2.  the component-selection strategy.

The corrected audit retained all valid contiguous A/B candidates at each
threshold.

Candidate records included geometry and residual evidence such as:

-   y minimum and maximum;
-   width;
-   centroid;
-   sample count;
-   residual magnitude;
-   finite support;
-   positive/negative composition for Method B.

A/B candidates were then diagnostically associated with the disturbance
seed used by Method D.

This correction removed the explicit "widest region wins" assumption.

However, direct inspection of the actual implementation later exposed a
deeper object-correspondence problem in the seed-association rule
itself.

------------------------------------------------------------------------

# 14. Experiment 06 corrected seed formulation

The first Method D seed used the median y position among the strongest
absolute-residual points.

This was rejected because spatially separated disturbances could
synthesize a seed location between actual structures.

The corrected implementation instead created a contiguous disturbance
activity signal.

The activity signal combined normalized:

-   absolute residual magnitude;
-   gradient activity.

Conceptually:

`activity = normalized absolute residual + 0.5 × normalized absolute gradient`

High-activity contiguous intervals were generated and ranked primarily
by integrated activity.

The selected seed was therefore an actual observed contiguous interval.

The corrected implementation recorded:

-   seed interval;
-   seed centroid;
-   integrated activity;
-   number of competing seed intervals;
-   margin to the second-best seed;
-   seed ambiguity.

This was a genuine improvement over a synthetic point seed.

However, the later scientific review identified a new implicit
assumption:

> strongest integrated local residual/gradient activity = processed
> track.

That assumption has not been validated.

A compact sharp negative valley or high-gradient defect can potentially
defeat a broader, smoother processed structure.

------------------------------------------------------------------------

# 15. Experiment 06 direct artifact review --- major finding

After the corrected Experiment 06 run, the actual script, tables, all 24
representative diagnostic profiles, and dense-grid track panels were
inspected directly.

The most important observation was not simply that Methods A/B/C/D
disagree.

The major observation was:

> The methods can lock onto different physical structures in the same
> profile.

The profiles visibly contain combinations of:

-   broad central humps;
-   deep narrow negative valleys;
-   secondary peaks;
-   asymmetric shoulders;
-   outer disturbances;
-   sharp spikes or abrupt transitions;
-   mixed positive/negative morphology;
-   missing or fragmented support.

Therefore, low pairwise interval overlap cannot automatically be
interpreted as four geometric definitions disagreeing about the boundary
of one object.

In some cases, the methods appear to be measuring different objects.

This changes the scientific problem from:

> Which boundary definition is best?

to:

> Which spatial structure is the physical processed track in the first
> place?

The current blocker is **object identity**.

------------------------------------------------------------------------

# 16. Experiment 06 object-correspondence confound

## Shared disturbance seed

The intended conceptual comparison was:

> Compare four substantially different definitions of processed-track
> geometry.

The implemented comparison was not fully independent.

Method C is conditioned on the disturbance seed:

-   left transitions must lie left of the seed;
-   right transitions must lie right of the seed.

Method D directly searches outward from the same seed.

Methods A and B are then associated back to that seed.

Therefore the implementation is closer to:

> Select an activity-defined object, then ask four methods how they
> relate to that object.

This is not equivalent to independently asking four definitions to
identify the processed-track object.

## Circularity concern

The seed activity itself contains gradient evidence.

Method C then audits gradient transitions relative to that seed.

Thus Method C is partially conditioned on an object selected using
evidence related to the evidence Method C is supposed to evaluate.

This does not make all Method C outputs meaningless, but it weakens the
interpretation of the A/B/C/D comparison as an independent
geometry-definition experiment.

## Scientific consequence

The Experiment 06 pairwise disagreement statistics are not clean
evidence that four definitions disagree about the same physical track.

They are partly contaminated by unresolved object correspondence.

------------------------------------------------------------------------

# 17. Method C implementation audit

Direct code and diagnostic review identified several problems.

## Transition candidates

Method C identifies local maxima in absolute gradient magnitude.

Signed gradient is recorded, but selection is driven primarily by
absolute transition strength.

Therefore the selected edge is not required to have a morphology-aware
transition direction.

A sharp valley wall can compete with a true outer boundary.

## Seed-relative selection

Left and right candidates are defined relative to the shared disturbance
seed.

If the seed corresponds to the wrong physical structure, Method C is
solving the wrong local edge problem before transition scoring even
begins.

## Score scaling

The transition score combines normalized gradient strength with a
distance-from-seed penalty.

The practical concern is that:

-   gradient strength is a robust normalized activity quantity;
-   distance is measured in millimeters.

A very sharp transition can overwhelm the relatively small distance
penalty.

The dense-grid figures are consistent with this failure mode.

## Observed behavior

Across the dense x-grid, Method C boundaries can jump across a large
fraction of the approximately 0--2 mm y-domain.

Widths can oscillate from very narrow to greater than approximately 1.5
mm.

This is not plausible as smoothly evolving local geometry of one single
physical track.

The current interpretation is:

> Method C frequently hops between unrelated strong structural
> transitions.

The problem is deeper than simply "add prominence."

------------------------------------------------------------------------

# 18. A/B seed-association audit

The corrected A/B implementation preserves candidate multiplicity, which
is scientifically preferable to widest-region selection.

However, the diagnostic seed association uses the following logic:

1.  prefer a candidate containing the seed centroid;
2.  otherwise choose the nearest candidate centroid.

The implementation does not require:

-   minimum interval overlap;
-   maximum association distance;
-   membership in the same disturbance complex.

Overlap is recorded after association rather than required for
acceptance.

Therefore a positive-excursion candidate on a broad hump can be
associated with a seed on a distinct sharp valley simply because it is
the nearest available positive candidate.

Scientific consequence:

> Low A/B/C/D IoU may partly indicate broken object correspondence
> rather than true disagreement in boundary definition.

This is a critical distinction.

------------------------------------------------------------------------

# 19. Reinterpretation of Track 10

Experiments 03--05 treated Track 10 primarily as a difficult or
fragmented case.

Experiment 05's FULL persistence corridor produced only:

-   **1 valid location**

The initial Experiment 06 run suggested approximately balanced positive
and negative residual morphology, but that interpretation was
temporarily questioned because of the whole-profile-median baseline
confound.

The corrected substrate-focused audit and direct visual review shifted
the interpretation again.

Current conclusion:

> Track 10 contains substantial geometric structure; prior extractor
> collapse cannot be explained simply as missing profilometry.

The more plausible issue is that Track 10 does not consistently present
one obvious positive elevated component matching the assumptions of
Experiments 03--05.

The 1D profiles and dense-grid behavior show competing mixed-sign and
structurally sharp features.

However, Experiment 06 does **not yet establish** which of those
structures is the physical processed track.

Therefore the strongest defensible Track 10 conclusion is:

> Track 10 is evidence against equating "persistent positive elevated
> component" with the full processed-track object, but it is not yet
> evidence for one replacement boundary definition.

------------------------------------------------------------------------

# 20. Scientific evidence chain after Experiments 03--10

The current cumulative interpretation is:

## Experiment 03

A positive-elevation threshold/component/scoring extractor can produce
plausible targets for some profiles, but many profiles are ambiguous or
invalid.

## Experiment 04

The explicit center score and preferred-width score are not the dominant
drivers of robust selections.

The dominant instability is the threshold defining positive components.

Across the tested threshold sweep:

-   selected count changed from 16 to 5;
-   18/24 representative cases were unstable.

## Experiment 05

Replacing one threshold with multi-threshold persistence and x-direction
corridor optimization did not clearly solve the problem.

Widths remained materially dependent on threshold-set composition.

Track 10 nearly collapsed.

## Experiment 06

The positive-elevation assumption was challenged against:

-   absolute substrate deviation;
-   profile transitions;
-   substrate-return geometry.

The corrected audit revealed complex mixed-sign and multi-structure
morphology.

Direct artifact review then showed that the four methods are not
guaranteed to measure the same object.

The shared strongest-activity seed and seed-association logic partially
confound the comparison.

## Current synthesis

The unresolved blocker is not primarily:

-   center-prior strength;
-   preferred width;
-   one MAD multiplier;
-   one threshold set;
-   one continuity weight;
-   one gradient prominence parameter.

The unresolved blocker is:

> **Object identity: which spatial structure in the profilometry
> corresponds to the physical processed track?**

Experiment 08 showed that baseline correction successfully removed
systematic substrate offset but could not fully explain the observed
differences among tracks.

Experiment 09 demonstrated that normalized cross-sectional geometry is
highly coherent within Tracks 8 and 14, whereas Track 10 contains
multiple longitudinal morphological regimes rather than one globally
coherent geometry.

The combination of Experiments 08 and 09 shifts the scientific
interpretation away from "baseline artifact versus geometry" and toward
understanding longitudinal process evolution and local geometric state
changes.

Experiment 10 shifted the project into a new phase.

Rather than continuing to optimize boundary extraction, the project now
focuses on designing a geometric representation suitable for multimodal
prediction.

The audit showed that normalized profile shape, local amplitude, and
measurement-validity information are essential components of any future
descriptor, whereas threshold-defined width is not sufficiently robust
to serve as the primary prediction target.

------------------------------------------------------------------------

# 21. New 2D spatial-coherence hypothesis

The height map is a 2D x-y surface with height z.

The native x spacing is approximately 0.004 mm, and adjacent x columns
are highly correlated.

A manufactured single DED track is a longitudinal physical process.

This motivates a new hypothesis:

> The physical processed-track object may need to be identified as a
> spatially coherent 2D structure across x-y before local left/right
> boundaries are extracted.

This is fundamentally different from Experiment 05.

Experiment 05:

1.  generated threshold-defined 1D components independently at x
    locations;
2.  then optimized continuity among those pre-defined candidates.

The new possible formulation is:

1.  inspect coherent 2D morphology first;
2.  establish object identity across adjacent x positions;
3.  only then derive local boundaries or descriptors.

The dense-grid Method C/D boundary jumps provide strong motivation for
this audit because they appear inconsistent with one continuously
evolving physical object.

However, 2D continuity is **not yet accepted as the answer**.

Continuity can produce algorithmic stability without proving physical
validity.

The following three requirements must remain separate:

1.  algorithmic stability;
2.  spatial continuity;
3.  physical validity.

A method must not be called physically correct merely because it draws a
smooth corridor.

------------------------------------------------------------------------

# 22. Risks and alternatives to the 2D hypothesis

The next scientific review must actively attempt to falsify the
2D-object hypothesis.

Possible alternatives include:

## Multiple physically meaningful structures

The profilometry may contain:

-   raised crown;
-   depression/remelt signatures;
-   shoulders;
-   roughness transitions;

that are all physically meaningful but should not be collapsed into one
"full width."

## Raised crown may be identifiable while full processed width is not

Method A may be a defensible descriptor if named accurately:

> raised crown width

rather than:

> full processed-track width.

## Negative morphology may require separate descriptors

Negative valleys or depressions may be meaningful process outcomes but
not part of one width interval.

## Boundaries may be intrinsically uncertain

The scientifically correct target may be:

-   a probability distribution over left/right boundaries;
-   boundary confidence;
-   a disturbed-region probability field;

rather than one deterministic interval at every x.

## Profilometry alone may be insufficient for semantic identity

Thermal and/or SEM alignment may be required before naming one coherent
profilometry structure "the processed track."

## Detrending itself may influence apparent continuity

The substrate-focused exclusion band:

-   y = 0.65--1.35 mm

is itself a structural assumption.

Any apparent coherent central structure must be checked against this
source of prior information.

------------------------------------------------------------------------

# 23. Current project decision

Geometry descriptor definition is frozen.

Descriptor implementation is complete.

Multimodal dataset integration is complete.

Phase II validation is complete.

Track 21 remains sealed.

The next phase of the project is baseline multimodal predictive modeling
using the frozen dataset.

------------------------------------------------------------------------

# 24. Methodological principles established so far

## Define the target before ML

A sophisticated predictive model cannot rescue an arbitrary target.

## Do not maximize valid-target count

More extracted widths are not automatically better.

Explicit missingness or uncertainty is preferable to confident false
labels.

## Preserve missingness

-   do not globally fill NaNs;
-   do not interpolate across long missing regions;
-   do not silently bridge unsupported boundaries.

Missingness is measurement information.

## Avoid per-track tuning

Use one shared configuration across Tracks 8, 10, and 14.

Per-track constants risk encoding track identity.

## Keep Track 21 sealed

Track 21 should only be inspected after the method/configuration is
frozen for held-out evaluation.

## Audit hidden priors

Priors can enter through:

-   center scores;
-   preferred widths;
-   detrending exclusion bands;
-   substrate definitions;
-   candidate bounds;
-   seed definitions;
-   object-association rules;
-   continuity assumptions.

Removing an explicit score does not make a method assumption-free.

## Do not confuse object detection with boundary extraction

Before comparing boundary definitions, establish that the methods refer
to the same physical object.

## Do not confuse stability, continuity, and physical validity

A stable algorithm may track the wrong object.

A smooth 2D corridor may be physically meaningless.

Physical interpretation requires evidence beyond numerical regularity.

## Dense x columns are not independent

Native x spacing is approximately 0.004 mm.

Adjacent columns should not be treated as independent samples in target
extraction or downstream probabilistic modeling.

## Name the target honestly

Potential defensible targets may include:

-   raised crown width;
-   full disturbed-region width;
-   left/right boundary distributions;
-   centerline;
-   signed height descriptors;
-   negative-depression descriptors;
-   roughness/activity descriptors;
-   validity or uncertainty quantities.

The target name must match the actual extracted quantity.

------------------------------------------------------------------------

# 25. Important files and artifacts

## Repository/context

-   `README.md`
-   dataset README/instructions
-   `src/nsf_fmrg_data.py`
-   `notebooks/01_starter_code_loading_and_visualization.ipynb`

## Persistent coding-agent instructions

-   `.github/copilot-instructions.md`

## Experiment 03

-   `notebooks/03_heightmap_target_extraction_exploration.ipynb`

Relevant run-output pattern:

-   `processed_data/run_outputs/03_heightmap_target_extraction_exploration_*`

## Experiment 04

-   `notebooks/04_heightmap_prior_sensitivity.ipynb`

Completed output:

-   `processed_data/run_outputs/04_heightmap_prior_sensitivity_20260713_194307/`

Important artifacts include:

-   `key_metrics.json`
-   `deliverable_summary.json`
-   `visualA_selection_status_table.csv`
-   `threshold_sweep_unstable_cases.csv`
-   `control_outputs.csv`
-   `top10_most_sensitive_cases.csv`
-   `control_selected_components.csv`
-   center-prior derivation outputs
-   width-prior derivation outputs
-   preferred-width sweep summaries
-   threshold-sweep summaries

## Experiment 05

-   `scripts/05_heightmap_persistence_corridor_experiment.py`

Completed output:

-   `processed_data/run_outputs/05_heightmap_persistence_corridor_20260713_201054/`

Important artifacts include:

-   `config.json`
-   `threshold_set_robustness_summary.csv`
-   `control_vs_corridor_rep_cases_all.csv`
-   `top10_threshold_set_sensitive_locations.csv`
-   diagnostics and per-location figures

## Experiment 06

-   `scripts/06_heightmap_geometry_definition_audit.py`

Relevant artifacts:

-   latest corrected timestamped Experiment 06 run directory;
-   candidate tables for Methods A and B;
-   substrate-baseline comparison tables;
-   seed and seed-ambiguity outputs;
-   Method C transition-candidate outputs;
-   Method D return-evidence and failure outputs;
-   pairwise method-comparison tables;
-   Track 10 forensic tables;
-   24 representative diagnostic profiles;
-   dense-grid Track 8, 10, and 14 method-comparison panels.

The exact latest corrected Experiment 06 run-directory name should
remain recorded from the repository itself when the next reviewer
inspects the workspace.

------------------------------------------------------------------------

# 26. Key numerical results at a glance

## Experiment 04 CONTROL

Total selected:

-   **10/24**

Track 8:

-   4 selected / 1 ambiguous / 3 invalid

Track 10:

-   2 selected / 3 ambiguous / 3 invalid

Track 14:

-   4 selected / 0 ambiguous / 4 invalid

## Center-prior audit

-   status changes: **1/24**
-   selected-component switches: **0**

## Width-prior audit

-   no-width-score status changes: **1/24**
-   selected-component switches: **0**

## Single-threshold audit

-   1.5×MAD → 16 selected
-   2.0×MAD → 14 selected
-   2.5×MAD → 10 selected
-   3.0×MAD → 8 selected
-   3.5×MAD → 5 selected

Threshold-unstable cases:

-   **18/24**

## Experiment 04 stability classification

-   consistently invalid: 13
-   stable: 9
-   highly sensitive: 1
-   moderately sensitive: 1

## Experiment 05 FULL valid corridor locations

-   Track 8: **97**
-   Track 10: **1**
-   Track 14: **38**

## Experiment 05 REMOVE_LOW width sensitivity

Track 8:

-   median absolute width shift ≈ 0.068 mm
-   p95 ≈ 0.233 mm

Track 14:

-   median absolute width shift ≈ 0.048 mm
-   p95 ≈ 0.138 mm

## Experiment 06 qualitative high-level result

The principal completed finding is not a preferred A/B/C/D winner.

It is:

-   complex mixed-sign and multi-structure morphology is present;
-   Method C/D dense-grid boundaries can jump implausibly across the
    y-domain;
-   the methods are not guaranteed to measure the same physical
    structure;
-   shared seed conditioning and A/B seed association confound pairwise
    interpretation;
-   object identity remains unresolved.

Pairwise disagreement from Experiment 06 should therefore not be treated
as clean ground-truth evidence that one boundary definition is superior.

------------------------------------------------------------------------

# 27. Ideas tested and current disposition

  ----------------------------------------------------------------------------
  Idea                    Current status          Evidence/reason
  ----------------------- ----------------------- ----------------------------
  Organizer global        Questioned              Substrate-focused detrending
  detrending only                                 appeared more representative
                                                  in exploratory cases

  Positive elevated       Not supported as a      Threshold instability, Track
  component = full track  general assumption      10 collapse,
                                                  mixed-sign/multi-structure
                                                  morphology

  Positive excursion as   Still plausible         May identify a narrower,
  raised-crown descriptor                         honestly named geometric
                                                  quantity

  Narrow center score     Largely disproved as    1/24 status changes; 0
  drives extraction       dominant issue          component switches

  0.32 mm preferred width Largely disproved as    Minimal ablation effect; no
  drives output           dominant issue          strong width tracking

  Tune one MAD multiplier Rejected                18/24 unstable; selected
                                                  count 16→5

  Multi-threshold         Insufficient as         Width remains threshold-set
  persistence             implemented             dependent

  Experiment 05           Paused                  Operates on already-defined
  x-corridor                                      1D candidates; object
                                                  identity unresolved

  Absolute substrate      Unresolved              Captures mixed-sign
  deviation = full track                          disturbance but may identify
                                                  multiple structures

  Strongest local         Not validated /         Sharp compact disturbances
  activity = track        currently distrusted    can defeat broad smoother
                                                  structures

  Gradient transitions    Not supported as        Method C jumps among
  directly define edges   currently implemented   unrelated strong transitions

  Substrate-return around Unresolved              Return logic may be
  strongest seed                                  reasonable, but object
                                                  identity depends on seed

  Compare A/B/C/D via     Confounded              Methods are not fully
  shared seed                                     independent; correspondence
                                                  may be broken

  2D spatial coherence    Next formulation audit  Motivated by longitudinal
  before boundaries                               process physics and
                                                  implausible 1D jumps

  Full processed-track    Unresolved              May require multimodal
  width from profilometry                         semantic evidence or
  alone                                           probabilistic target
  ----------------------------------------------------------------------------

------------------------------------------------------------------------

# 28. Immediate next step and update checklist

**Phase III --- Baseline multimodal predictive modeling**

Current status:

- Phase III feature preprocessing is implemented.
- Phase III target alignment is implemented.
- The feature/target metadata contract is established.
- Baseline model integration and evaluation have been completed.

Immediate objectives:

- **(Completed)** validate final baseline selection using development-only LOTO across Tracks 8, 10, and 14 (Track 21 remained sealed)
- **(Completed)** freeze the final baseline configuration and reporting protocol for sealed evaluation
- evaluate on held-out Track 21 only using the frozen pipeline
- prepare publication-quality figures and tables,
- refine the final challenge report.

------------------------------------------------------------------------

# 29. Executive summary

The height-map work began with a thresholded positive-elevation
component extractor. Experiment 04 showed that explicit center and
preferred-width priors were not the dominant problem. The dominant
instability was the threshold defining positive components: 18 of 24
representative cases were unstable, and selected count changed from 16
to 5 across a reasonable MAD-multiplier sweep.

Experiment 05 replaced a single threshold with multi-threshold
persistence and x-direction corridor optimization. This did not clearly
solve the problem. Widths remained materially dependent on threshold-set
composition, and Track 10 produced only one valid FULL-corridor
location.

Experiment 06 then challenged the positive-elevation definition against
absolute substrate deviation, profile transitions, and substrate-return
boundaries. An initial whole-profile-median baseline confound was
identified and corrected using substrate-focused baseline estimation.
Widest-region selection for A/B was also removed, and Method D received
a contiguous activity-based seed.

Direct review of the corrected script and executed diagnostics exposed a
deeper issue. The profilometry contains multiple substantial structures
within individual profiles: broad humps, sharp negative valleys,
secondary peaks, shoulders, and mixed-sign disturbances. The A/B/C/D
methods are not guaranteed to measure the same physical object. Method C
and D are conditioned on a shared activity-defined seed, while A/B
candidates are associated back to that seed using containment or
nearest-centroid logic without a strict same-object acceptance
criterion. Consequently, low pairwise IoU can partly reflect broken
object correspondence rather than clean disagreement over the boundary
of one track.

The dense-grid Method C and D boundaries also jump implausibly across
large portions of the y-domain, consistent with algorithms hopping
between unrelated structural features.

The scientific focus has now shifted from extraction to representation.

Experiment 10 established that threshold-defined width is not
sufficiently robust to serve as the principal prediction target.
Instead, the accumulated evidence indicates that any useful geometric
descriptor must preserve normalized cross-sectional shape, local
amplitude, longitudinal process-state variation, and
measurement-validity information.

# IGNORE THIS AS THIS HAS ALREADY BEEN DONE
<!-- The next stage of the project will therefore compare alternative
descriptor implementations while keeping these scientifically derived
requirements fixed. Only after selecting a defensible descriptor will
multimodal machine-learning models be developed. -->

Following descriptor selection, Phase III engineering infrastructure was completed and baseline multimodal models were evaluated on the development tracks. A subsequent Leave-One-Track-Out (LOTO) study across Tracks 8, 10, and 14 established Ridge Regression (alpha = 1.0) with SEM-only features as the final frozen baseline configuration for sealed Track 21 evaluation. Track 21 remained sealed throughout this process.

------------------------------------------------------------------------

# 33. Experiment 10 --- geometry descriptor requirements audit

## Purpose

Following Experiments 03--09, the primary scientific question shifted
from boundary extraction toward representation design.

Rather than asking which extraction algorithm is best, Experiment 10
asked:

> **What information must any useful local geometry descriptor
> preserve?**

This experiment intentionally did **not** compare PCA, splines, or other
mathematical encodings. Instead, it established the scientific
requirements that any future descriptor should satisfy.

## Major findings

The executed audit classified candidate geometric properties according
to their usefulness as downstream prediction targets.

### Essential

-   Offset- and amplitude-normalized cross-sectional shape.
-   Local amplitude (relief).
-   Finite-support and validity state.

### Useful

-   Signed elevation.
-   Peak position.
-   Profile asymmetry.
-   Roughness / curvature.
-   Multi-peak structure.

### Unsuitable

-   Threshold-defined width.
-   Single deterministic boundary interval.
-   Raw gradient-transition boundaries.

This is consistent with the instability observed in Experiments 04--06.

## Scientific interpretation

Experiment 10 represents a transition from geometric extraction toward
representation design.

The accumulated evidence indicates that a single scalar quantity such as
width or maximum height cannot adequately represent the local
morphology.

Instead, the executed evidence suggests that a useful prediction target
must preserve:

-   normalized profile shape,
-   local amplitude,
-   longitudinal regime information,
-   finite-support quality,
-   baseline-validity information.

Importantly, Experiment 10 deliberately stopped short of recommending
PCA or any other mathematical encoding. The experiment established
requirements rather than implementations.

------------------------------------------------------------------------

# 30. Experiment 07A --- 2D object-identity audit artifact generation

## Purpose

Experiment 07 was split into phases. Phase 07A was limited to
implementation and artifact generation only.

The goal was to create a reliable exploratory audit pipeline for later
scientific inspection of 2D x-y height-map morphology. No scientific
interpretation, final geometry target, processed-track object selection,
or final width extraction was performed in this phase.

Artifact:

`scripts/07_heightmap_2d_object_identity_audit.py`

Executed with:

`/opt/homebrew/opt/python@3.11/bin/python3.11 scripts/07_heightmap_2d_object_identity_audit.py`

Successful output:

`processed_data/run_outputs/07_heightmap_2d_object_identity_audit_20260714_212755/`

Tracks analyzed:

-   8;
-   10;
-   14.

Track 21 remained sealed.

## Guardrails preserved

The script uses:

-   the organizer loader `src/nsf_fmrg_data.py::load_wyko_asc`;
-   the common 20--100 mm physical x window from the loader;
-   the same substrate-focused detrending context used in Experiments
    05 and 06, excluding y = 0.65--1.35 mm for plane fitting;
-   one shared configuration across Tracks 8, 10, and 14;
-   no per-track tuning.

The script preserves:

-   original NaNs;
-   finite-support topology;
-   physical x/y coordinates.

It does not:

-   globally fill NaNs;
-   interpolate through missing regions;
-   bridge unsupported areas;
-   select a final object;
-   output a processed-track width.

## 2D fields generated

For each development track, the pipeline generated diagnostics for five
separate evidence channels:

1.  detrended z(x,y);
2.  signed residual relative to substrate baseline;
3.  absolute residual magnitude;
4.  y-gradient magnitude/activity;
5.  finite-support / NaN topology.

These evidence channels were deliberately kept separate. They were not
collapsed into one final activity score.

## Neutral candidate-structure inventory

The pipeline generated exploratory candidate structures from separate
2D evidence channels using shared saliency levels:

-   2.0;
-   3.0;
-   4.0.

The candidate structures are neutral connected regions and are labeled
with generic identifiers such as `structure_0001`, `structure_0002`,
and so on.

They are **not** labeled as tracks, boundaries, crowns, or final
processed regions.

The inventory records spatial extent, x-support, y-location,
fragmentation, finite-support fraction, and NaN-contact information so
that later review can evaluate object identity and missingness without
forcing a target.

## Generated figures

For each of Tracks 8, 10, and 14, the run produced:

-   detrended z figure;
-   signed residual figure;
-   absolute residual figure;
-   gradient activity figure;
-   NaN topology figure;
-   candidate structure overlay on signed residual;
-   candidate structure overlay on absolute residual;
-   candidate structure overlay on gradient field.

Total figures generated:

-   **24**

## Generated tables

The run produced machine-readable tables under the run directory's
`tables/` folder:

-   `structure_inventory.csv`
-   `structure_channel_summary.csv`
-   `support_topology_summary.csv`
-   `field_summary.csv`
-   `run_metadata.json`

The executed run reported:

-   structure inventory rows: **840**;
-   support-topology rows: **840**.

## Current status after 07A

Experiment 07A created the diagnostic artifact set needed for the next
phase of review.

No conclusions were drawn from these artifacts in 07A. The next step is
to inspect the generated 2D figures and tables scientifically before
deciding whether the 2D object-identity hypothesis is supported,
unclear, or rejected.

Observations for Track 8:

Strong continuous positive residual corridor spanning nearly the full 20–100 mm range.
Localized high-saliency peaks are embedded within this corridor rather than replacing it.
Experiment 07 currently inventories salient substructures, not the full corridor.
Gradient activity appears diffuse and may be less informative than residual magnitude for defining candidate geometry.
Hypothesis: the persistent support corridor may be a more fundamental geometric entity than isolated peaks.

------------------------------------------------------------------------

# 31. Experiment 08 --- baseline cross-section audit

## Purpose

The purpose of Experiment 08 was to determine whether the apparent
visual differences between Tracks 8, 10, and 14 arose primarily from:

-   baseline-estimation artifacts,
-   amplitude differences,
-   or genuine geometric differences.

Rather than manually selecting representative cross-sections, quiet,
moderate, and strong events were selected objectively using an
event-strength statistic computed from the signed residual within the
central corridor.

## Selection methodology

Event strength was defined using:

`median(|signed residual| / substrate MAD)`

computed on finite samples with finite-support requirements and edge
exclusion.

Selections were then defined as:

-   quiet = nearest 10th percentile;
-   moderate = nearest 50th percentile;
-   strong = nearest 90th percentile.

## Major findings

-   The substrate residual median after baseline correction was
    essentially 0 µm for all three tracks.
-   There was no evidence of systematic track-wide baseline bias.
-   Track 10 quiet case required fallback baseline estimation because
    of insufficient substrate support.
-   Several selected cross-sections had relatively low finite support.

## Scientific interpretation

Experiment 08 did not provide decisive evidence that Tracks 8, 10 and 14
possess fundamentally different signed residual geometry after baseline
correction.

Instead, the audit remained formally ambiguous.

However, Track 10 continued to exhibit isolated suspicious
cross-sections that could not be explained solely by systematic baseline
offset.

------------------------------------------------------------------------

# 32. Experiment 09 --- longitudinal shape-coherence audit

## Purpose

Determine whether normalized cross-sectional geometry remains coherent
along each track after removing vertical offset and amplitude scaling.

## Method

-   Normalize every cross-section.
-   Compare profile shapes.
-   Compute pairwise similarity.
-   Derive medoid profile.
-   Analyze coherence versus x separation.

## Major findings

### Track 8

-   very high coherence
-   median similarity ≈ 0.916

### Track 14

-   high coherence
-   median similarity ≈ 0.847

### Track 10

-   coherent over short distances
-   several distinct low-coherence longitudinal regions
-   morphology changes substantially between approximately 78--97 mm

The previously suspicious Track 10 quiet section remained an outlier
even after normalization.

## Interpretation

Experiment 09 demonstrates that removing baseline offset and amplitude
differences does not eliminate the observed longitudinal structure.

Instead,

-   Tracks 8 and 14 exhibit strong longitudinal morphological
    consistency.
-   Track 10 consists of multiple coherent local regimes separated by
    genuine morphological transitions.

This indicates that Track 10's unusual behavior cannot be attributed
solely to baseline estimation artifacts.

------------------------------------------------------------------------

# Experiment 11 --- PCA representation evaluation

## Purpose

Experiment 11 evaluated PCA only as a representation for the
offset- and amplitude-normalized cross-sectional shape component.

The goal was not to define the complete geometry descriptor, but to
determine whether PCA preserves the longitudinal morphology discovered
in Experiment 09 while providing a compact representation.

## Major findings

-   Approximately 71.5% of the variance is explained by PC1.
-   Approximately 83.4% is explained by the first two PCs.
-   About 90% is explained by six PCs.
-   Five PCs were chosen as a practical descriptor compromise.
-   Reconstruction error decreases smoothly as more PCs are retained.
-   Track 10 low-coherence regimes remain distinguishable after
    reconstruction.

PCA preserves normalized shape but intentionally removes amplitude and
vertical offset by construction.

Therefore amplitude, signed elevation, finite-support validity, and
regime metadata remain companion descriptor fields.

## Scientific conclusion

PCA is accepted as the normalized shape representation but not as the
complete geometry descriptor.

------------------------------------------------------------------------

# Experiment 12 --- geometry descriptor implementation

## Purpose

Experiment 12 transitioned from descriptor definition and evaluation
into implementation.

The descriptor definition from Experiment 10 and the PCA decision from
Experiment 11 were frozen.

Geometry descriptors are evaluated at Adi's thermal-frame anchor
positions taken from:

`processed_data/phase1_unified_master.csv`

## Implementation

The reusable API:

`extract_geometry_descriptor(track_id, x_position_mm)`

was implemented.

The descriptor now returns:

-   PC1--PC5
-   amplitude
-   signed elevation
-   eligibility
-   nonflat flag
-   regime ID

along with validity metadata.

Descriptors are evaluated only at the thermal x positions.

One descriptor row exists for every thermal anchor.

Explicit NaNs are preserved.

No interpolation is performed.

Track 21 remains sealed.

One shared configuration is used across Tracks 8, 10, and 14.

The implementation generated:

-   `geometry_descriptors_track_8.csv`
-   `geometry_descriptors_track_10.csv`
-   `geometry_descriptors_track_14.csv`

## Scientific significance

The project has now moved from asking:

> "What should the geometry descriptor be?"

to:

> "Producing the descriptor at every aligned thermal observation."

This is the first implementation-ready representation that can be merged
with thermal and SEM features for multimodal learning.

Experiment 11 concluded that PCA is an appropriate representation for the normalized cross-sectional shape component. Five principal components preserve the dominant bead geometry while maintaining distinct Track 10 low-coherence regimes. PCA alone is insufficient as a complete descriptor because amplitude, signed elevation, and validity metadata are intentionally excluded by normalization and must be retained as companion descriptor components.

------------------------------------------------------------------------

# Experiment 13 --- merge geometry descriptors

## Purpose

Experiment 13 integrated the frozen geometry descriptors produced in
Experiment 12 into the aligned multimodal master table at the thermal
anchor grid.

This experiment is strictly a dataset-integration step:

- the Experiment 12 descriptor definition is not modified;
- descriptors are merged, not recomputed;
- Track 21 remains sealed (no Track 21 descriptor file is read,
    generated, or inspected).

## Files created

- `scripts/13_merge_geometry_descriptors.py`
- `processed_data/final_multimodal_dataset.csv`
- `processed_data/merge_validation_summary.json`
- `processed_data/descriptor_merge_statistics.csv`

## Execution

Executed with:

`/opt/homebrew/opt/python@3.11/bin/python3.11 scripts/13_merge_geometry_descriptors.py`

Execution status: **successful**.

## Validation results

- Rows before merge: **1600**
- Rows after merge: **1600**
- Duplicate join-key rows after merge: **0**
- One-to-one merge validation: **passed** for Tracks 8, 10, and 14
- Tracks 8/10/14 descriptor anchors matched exactly once.
- Track 21 master rows were preserved unmodified, with descriptor fields
    remaining **NaN** (no Track 21 descriptors merged).

## Descriptor fields added

The merged dataset appends geometry descriptor fields including:

- normalized shape: **PC1--PC5**
- local amplitude: **amplitude**
- signed elevation: **signed elevation**
- validity and readiness flags (eligibility / nonflat / PCA-ready and
    related finite-support and baseline-support metadata)
- regime metadata (regime identifier)

## Scientific meaning

The geometry descriptor has now been integrated into the multimodal
dataset at the thermal anchor grid. The project can transition from
descriptor construction to merged-dataset validation.

------------------------------------------------------------------------

# Experiment 14 --- Phase 2 merge-point validation

## Purpose

Experiment 14 validated the completed multimodal dataset after geometry
descriptor integration.

The purpose was **not** to modify descriptor extraction or redefine the
geometry descriptor.

Instead, the experiment verified that the merged dataset satisfies the
Phase II requirements before any machine-learning models are developed.

## Implementation

Validation was performed using:

`scripts/14_phase2_merge_point_validation.py`

The validation:

- inspected only Tracks 8, 10 and 14,
- preserved the sealed Track 21 protocol,
- generated validation plots,
- checked anchor alignment,
- verified descriptor coverage,
- verified merge correctness.

The experiment intentionally did **not** modify descriptor extraction or
geometry definitions.

## Validation performed

The script verified:

- exactly 400 rows for each development track,
- anchor alignment with `phase1_unified_master.csv`,
- zero duplicate join keys,
- zero unexpected descriptor rows,
- one-to-one merge correctness,
- descriptor validity statistics,
- descriptor NaN patterns,
- preservation of Track 21 as a sealed evaluation track.

## Generated outputs

The run produced a timestamped run-output directory containing a
validation summary JSON, an anchor-alignment report, and per-track
validation plots.

## Major findings

- anchor alignment passed with zero x-position mismatch,
- exactly one descriptor row exists per thermal anchor,
- Track 21 remained untouched,
- descriptor validity patterns are consistent with Experiment 12
    eligibility rules,
- Track 10's previously identified low-coherence regime remains visible
    within approximately 78--97 mm,
- no integration inconsistencies were identified.

## Scientific significance

Experiment 14 completes Phase II.

The project now possesses:

- aligned thermal features,
- aligned SEM features,
- aligned geometry descriptors,
- one merged multimodal dataset,
- validated anchor alignment.

The project is now ready to transition into multimodal predictive
modeling.

------------------------------------------------------------------------

# Phase III Engineering Infrastructure

## Purpose

After Experiment 14, the project transitioned from geometry-target
engineering into Phase III multimodal predictive modeling.

This section records engineering infrastructure built on the frozen
scientific decisions from Experiments 03--14. It does **not** introduce a
new geometry experiment, modify the descriptor definition, recompute
descriptors, or reopen any Phase I/II scientific conclusions.

Track 21 remains sealed.

## Collaboration split

Person A owns the feature-side Phase III pipeline.

Person B owns the target-side Phase III pipeline.

The two workstreams intentionally communicate only through row identity
metadata:

- `track_id`
- `frame_index`
- `x_position_mm`

The metadata corresponds to the final frame of each rolling feature
window. Its row order defines the canonical Phase III sample order.

## Person A feature preprocessing pipeline

Person A completed a reusable feature preprocessing pipeline implemented
in:

`scripts/phase3_data_loader.py`

The pipeline:

- loads thermal and SEM feature columns from
    `processed_data/final_multimodal_dataset.csv`;
- filters to physically valid PCA-ready rows for the current sequence
    modeling workflow;
- standardizes feature columns using the training split only;
- applies the fitted scaler to validation data without refitting;
- constructs rolling temporal feature windows;
- returns NumPy feature arrays and metadata tables.

The primary outputs are:

- `X_train_seq`
- `X_val_seq`
- `train_meta`
- `val_meta`

The feature arrays have shape:

`(samples, window_size, features)`

The metadata tables preserve the canonical row identity of the target
frame for each feature window.

## Person B target-alignment module

Person B implemented the reusable Phase III target-alignment module in:

`src/ml/targets.py`

The public class is:

`Phase3TargetAligner`

This module is strictly an alignment utility. It does not:

- recompute descriptors;
- redesign geometry;
- recreate rolling windows;
- perform feature engineering;
- modify `processed_data/final_multimodal_dataset.csv`;
- import PyTorch or create data loaders.

The module consumes `train_meta` or `val_meta`, performs an exact
one-to-one merge against:

`processed_data/final_multimodal_dataset.csv`

and returns NumPy target arrays.

Supported target groups in the initial implementation are:

- PCA shape: `pc1`--`pc5`;
- amplitude: `amplitude_um`;
- signed elevation: `signed_elevation_um`.

The join keys are exactly:

- `track_id`
- `frame_index`
- `x_position_mm`

No nearest-neighbor matching, `merge_asof`, interpolation, or target
filling is performed.

## Alignment validation

Validation was performed using:

`scripts/15_phase3_target_alignment_validation.py`

Execution used the repository-standard interpreter:

`/opt/homebrew/opt/python@3.11/bin/python3.11 scripts/15_phase3_target_alignment_validation.py`

Execution status: **successful**.

The validation confirmed that the target-alignment module:

- loads the frozen multimodal dataset;
- validates the dataset schema;
- validates metadata schema;
- rejects ambiguous duplicate metadata rows;
- performs exact one-to-one joins;
- preserves row count;
- preserves metadata ordering;
- returns NumPy arrays;
- keeps target alignment independent from PyTorch.

Training outputs:

- `X_train_seq`: `(423, 5, 9)`
- `train_meta`: 423 rows
- `Y_train` for PCA shape: `(423, 5)`

Validation outputs:

- `X_val_seq`: `(175, 5, 9)`
- `val_meta`: 175 rows
- `Y_val` for PCA shape: `(175, 5)`

Amplitude and signed elevation also aligned successfully.

All supported target groups produced zero target NaNs in the validated
feature-window metadata.

The metadata ordering was preserved exactly.

## Engineering milestones

- Phase III feature preprocessing interface established.
- Phase III target-alignment interface established.
- Feature/target metadata contract established.
- NumPy interface finalized.
- Ready for baseline model integration.

## Architecture established

The Phase III engineering interface is now:

```text
FeaturePreprocessor
    |
    v
X_train_seq
train_meta

    |

Phase3TargetAligner

    |
    v

Y_train
Y_val
```

The two modules intentionally remain independent. Neither module imports
the other. Their only shared contract is the metadata table containing:

- `track_id`
- `frame_index`
- `x_position_mm`

This preserves the scientific separation between input-side feature
preprocessing and target-side descriptor alignment.

## Current interpretation

This work is engineering infrastructure, not a new scientific
experiment.

The geometry descriptor remains unchanged.

The frozen descriptor implementation and merge validation from
Experiments 12--14 remain the scientific basis for Phase III modeling.

Track 21 remains sealed. The Phase III engineering infrastructure and baseline modeling pipeline were completed on the development tracks before the final development-only LOTO model-selection validation was performed.

The project is now ready to execute the frozen sealed evaluation protocol on Track 21.

## Phase III Baseline Modeling Results

The first Phase III baseline modeling scripts have now been executed
using the frozen feature preprocessing, target alignment, and metric
infrastructure.

These experiments predict the PCA shape target group (`pc1`--`pc5`) from
flattened rolling feature windows. The validation split is held-out Track
14, with Tracks 8 and 10 used for training.

No hyperparameter tuning, model persistence, plotting, or Track 21
evaluation was performed.

Validation metrics are summarized below.

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

The Random Forest baseline used:

- `n_estimators = 300`
- `min_samples_leaf = 2`
- `random_state = 42`

The current results are descriptive baseline results only. They validate
that the full Phase III predictive modeling path runs end-to-end for
linear, regularized linear, and nonlinear tree-based regressors.

The LSTM baseline differs from the first three baselines because it uses
the sequence windows directly rather than flattening each window into a
single feature vector. This preserves temporal ordering within the
five-frame window and allows the model to represent a hidden sequential
state before predicting the five PCA shape scores.

The MLP baseline completes the initial baseline set by testing nonlinear
function approximation on the same flattened feature representation used
by the classical models. This separates, at a baseline level, the effect
of a generic nonlinear neural network from the tree-based partitioning
used by the Random Forest and from the temporal-state representation used
by the LSTM.

## Random Forest Feature Importance Analysis

The Random Forest baseline was inspected using the built-in impurity-based
`feature_importances_` values from `sklearn.ensemble.RandomForestRegressor`.
No SHAP, permutation importance, partial dependence, or additional
interpretability framework was introduced.

The analysis used the same Random Forest configuration as the baseline
model:

- `n_estimators = 300`
- `min_samples_leaf = 2`
- `random_state = 42`

The purpose was to describe which flattened sequence-window inputs the
Random Forest relied on most strongly, not to tune the model.

Principal observations:

- In the thermal-only model, the highest-ranked feature was
    `mp_length` at timestep 2. Mean temperature, melt-pool length,
    melt-pool area, and centroid coordinates also appeared among the
    most important thermal features.
- In the SEM-only model, the largest importance was assigned to
    `substrate_mean_intensity` at timestep 0, followed by
    `substrate_roughness_variance` at timestep 0.
- In the combined thermal + SEM model, the top individual flattened
    features were early SEM-context features, especially
    `substrate_mean_intensity` and `substrate_roughness_variance` at
    early timesteps.
- The combined model's total impurity importance was approximately
    balanced by modality: Thermal **50.43%**, SEM **49.57%**.
- Temporal importance in the combined model was largest at timestep 0
    and then distributed across the remaining window, with timestep 4
    also contributing materially.

For the combined model, the timestep-level importance totals were:

| Timestep | Total importance |
|---:|---:|
| 0 | 0.326981 |
| 1 | 0.202160 |
| 2 | 0.146687 |
| 3 | 0.123726 |
| 4 | 0.200446 |

The current scikit-learn multi-output Random Forest exposes one aggregate
feature-importance vector for the full five-output prediction problem. It
does not naturally separate importance by PC1--PC5 without fitting
additional target-specific models. The current analysis therefore reports
aggregate feature importance across the five PCA targets.

These results suggest that future modeling comparisons should preserve
the thermal-only, SEM-only, and combined feature-set structure. The
balanced combined-modality importance also argues against discarding
either modality before stronger validation evidence is available.

## Phase III experimental progression

The baseline modeling sequence was intentionally incremental.

- Linear Regression established the simplest flattened-feature baseline
    for predicting the five PCA shape targets.
- Ridge Regression tested whether L2 regularization improved the same
    flattened feature representation without changing the model class
    substantially.
- Random Forest Regression tested whether nonlinear tree-based
    partitioning improved prediction relative to linear models.
- The LSTM baseline tested whether preserving the five-frame temporal
    ordering improved prediction relative to flattened-window baselines.
- The Random Forest feature-importance analysis inspected the strongest
    tree-based baseline using built-in impurity importances.
- The MLP baseline tested nonlinear function approximation while using
    the same flattened representation as the classical baselines.

No experiment in this baseline sequence modified the frozen descriptor,
the Phase III preprocessing contract, or the target-alignment contract.

## Principal Phase III findings so far

The completed baseline experiments support the following observations.

- Ridge Regression improved over ordinary Linear Regression for all three
    feature groups on the held-out Track 14 validation split.
- Random Forest Regression produced the strongest overall baseline
    validation results among the tested model families when considering
    the best-performing feature group for each model family.
- The Random Forest SEM-only model produced the lowest validation MAE and
    RMSE among the completed baselines.
- The LSTM baseline did not improve validation performance despite
    preserving temporal ordering within the five-frame feature window.
- The MLP baseline performed substantially worse than Random Forest for
    thermal-only and combined feature sets, despite being a nonlinear
    neural-network model operating on the same flattened representation.
- Random Forest feature importance indicated that predictive information
    is distributed across both thermal and SEM modalities in the combined
    feature set.
- Temporal importance in the Random Forest analysis was distributed
    across the five-frame window rather than concentrated in a pattern
    that strongly supports sequence modeling for the current window size.

These are observations from the current development-track validation
only. They should not be treated as final claims about Track 21.

## Cross-track generalization observations

The development split used Tracks 8 and 10 for training and Track 14 as
the held-out validation track.

The Random Forest baseline produced similar training performance on
Tracks 8 and 10. For the combined thermal + SEM Random Forest model,
the training-track metrics were nearly identical:

| Track | Split | MAE | RMSE | Median AE | R² |
|---:|---|---:|---:|---:|---:|
| 8 | train | 0.448607 | 0.600540 | 0.340279 | 0.682730 |
| 10 | train | 0.448670 | 0.602157 | 0.359260 | 0.729262 |

Validation performance degraded substantially on Track 14:

| Feature group | Track | Split | MAE | RMSE | Median AE | R² |
|---|---:|---|---:|---:|---:|---:|
| Thermal-only | 14 | validation | 1.892585 | 2.115979 | 1.821794 | -1.635473 |
| SEM-only | 14 | validation | 1.314589 | 1.664540 | 1.135141 | -0.848344 |
| Thermal + SEM | 14 | validation | 1.427145 | 1.747150 | 1.245874 | -0.946050 |

Among the tested Random Forest feature groups, SEM-only features gave the
strongest validation performance on Track 14. Thermal + SEM features gave
the strongest training performance on Tracks 8 and 10.

The current evidence therefore indicates that model fit is much stronger
on the training tracks than on the held-out development track. This is a
cross-track generalization observation, not a final Track 21 conclusion.

## Current interpretation

The baseline study indicates that nonlinear relationships are important
for this prediction problem, but the form of nonlinearity matters.

Ridge Regression improved the linear baseline, suggesting that
regularization helped the flattened feature representation. Random Forest
then improved substantially over the linear baselines for thermal-only
and combined feature sets, indicating that tree-based nonlinear
partitioning was useful for the current development split.

The LSTM result does not support a claim that preserving five-frame
temporal order improves prediction in the current setup. The MLP result
also does not support a claim that generic nonlinear neural-network
function approximation is sufficient to match the Random Forest baseline
on this small development set.

The SEM-only feature set produced the best held-out Track 14 validation
metrics among the completed baselines, whereas combined thermal + SEM
features fit the training tracks best. This suggests that SEM-derived
substrate context is important for cross-track validation in the present
experiments, while the combined feature set may be more susceptible to
overfitting the development training tracks.

These interpretations remain provisional until the final pipeline is
frozen and evaluated once on Track 21.

## Leave-One-Track-Out (LOTO) model-selection validation (development-only)

The original baseline experiments used a fixed development holdout (train: Tracks 8 + 10; validate: Track 14). Because only three development tracks were available (Tracks 8, 10, and 14), a subsequent Leave-One-Track-Out (LOTO) validation was performed as a more robust development-only model-selection procedure. Track 21 remained sealed throughout. The pooled LOTO results showed that **Ridge Regression (alpha = 1.0)** with **SEM-only** features provided the strongest aggregate generalization performance, so Ridge Regression + SEM-only became the frozen baseline configuration for the sealed Track 21 evaluation.

## Current Phase III project status

The Phase III modeling workflow now has:

- preprocessing infrastructure completed;
- target alignment completed;
- dataset packaging completed;
- regression evaluation infrastructure completed;
- baseline model experiments completed;
- Random Forest feature-importance analysis completed.

The strongest current baseline is the Random Forest family. Within that
family, SEM-only features produced the best held-out Track 14 validation
metrics, while the combined thermal + SEM model produced the strongest
training-track fit.

The strongest initial nonlinear baseline on the original Track 14 holdout split was the Random Forest family. Final model selection was subsequently validated using development-only LOTO across Tracks 8, 10, and 14, and the frozen baseline configuration is now **Ridge Regression (alpha = 1.0)** with **SEM-only** features for PCA shape targets (`pc1`--`pc5`).

Track 21 remains intentionally sealed.

## Current future work

Future work should now focus on finalization rather than adding more
model classes.

Immediate next steps are:

- execute the frozen sealed-evaluation run on Track 21 (without modifying preprocessing, alignment, feature selection, model configuration, or reporting);
- generate Track 21 predictions and report results using the frozen reporting protocol;
- prepare publication-quality figures and summary tables;
- refine the final challenge report and paper text.

Additional architectures should not be added unless later diagnostics
provide a specific scientific reason to do so.

Track 21 remains sealed.
