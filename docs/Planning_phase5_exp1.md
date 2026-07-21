Phase 5 — Current Status & Next Steps
Current Status
✅ Frozen baseline

The canonical Phase III/IV baseline remains unchanged.

Model: Ridge Regression (alpha=1.0)
Features: SEM-only
Window: 5-frame flattened sequence
Target: Raw PC1–PC5
Validation: Leave-One-Track-Out (Tracks 8, 10, 14)
Track 21 remains untouched for blind inference.

Baseline aggregate:

MAE = 1.275480
RMSE = 1.655831
Median AE = 1.013160
R² = -0.282801

This remains the official comparison point for every subsequent experiment.

Experiment #1 — Coordinate-Augmented Ridge
Hypothesis

The Ridge model underperforms because it lacks longitudinal process position.

Added one additional feature:

x_norm = (x_position_mm - 60.0) / 40.0

Everything else remained frozen.

Result

Aggregate performance was essentially unchanged.

Observations:

MAE improved by only 0.000475
RMSE became slightly worse
Median AE became slightly worse
Fold variability increased
Only one fold improved consistently
Conclusion

Hypothesis rejected.

The missing longitudinal coordinate is not a major explanation for weak cross-track generalization.

Experiment closed.

Research Direction Shift

Because Experiment #1 failed, we revisited the overall learning problem instead of continuing incremental feature engineering.

Current working hypothesis:

The limitation may not be the model or missing features, but the supervised learning formulation itself.

Specifically:

The model may be capable of predicting macro-scale geometry while struggling to learn high-frequency local PCA variation.

This led to designing a single higher-risk reformulation experiment rather than another minor feature tweak.

Proposed Experiment #2

Working title:

Smoothed PCA Target Ridge LOTO

Core idea:

Inputs remain identical.
Ridge remains identical.
SEM-only features remain identical.
LOTO remains identical.
Track 21 remains untouched.

Only the training targets are modified.

Training targets become a smoothed version of PC1–PC5 (within each training track), while evaluation is still performed against the original raw PC targets.

This directly tests whether suppressing high-frequency target variation improves cross-track generalization.

No implementation has begun.

A complete experimental protocol has already been drafted.

The only remaining open question is the exact smoothing protocol.

Immediate Tasks (Aditya)
1. Merge today's work
Pull latest changes.
Verify nabarun-exp1 is synchronized.
Ensure Experiment #1 outputs are preserved.
2. Review protocol only

Read through the proposed Experiment #2 protocol.

No coding.

Focus on:

leakage
alignment
target construction
reproducibility
implementation feasibility
3. Think about smoothing

Specifically review whether:

centered rolling median
window size = 11

appear scientifically justified.

No implementation yet.

4. Look for repository implications

Identify whether the protocol can truly be implemented as:

scripts/
    26_phase5_loto_smoothed_pca_target_ridge.py

without modifying shared utilities.

If any shared-module modification appears necessary, document it.

My Tasks (Tomorrow)
1.

Send the final protocol stress-test prompt to GPT-5.5.

Goal:

Challenge only the smoothing design.

Nothing else.

2.

Review GPT's final recommendation.

If the protocol changes:

document why

If unchanged:

freeze protocol permanently
3.

Approve final experimental protocol.

No more redesign.

No more strategy discussions.

4.

Implement Experiment #2.

Only after protocol freeze.

5.

Run full LOTO evaluation.

Collect:

fold metrics
pooled metrics
comparison with frozen baseline
6.

Interpret results.

If successful:

decide whether to replace development baseline.

If unsuccessful:

formally close Experiment #2 and freeze the original Ridge baseline for submission.
Current Project State
Phase I
    ✓ Complete

Phase II
    ✓ Complete

Phase III
    ✓ Frozen

Phase IV
    ✓ Frozen

Experiment #1
    ✓ Complete
    ✓ Rejected

Experiment #2
    Protocol complete
    Awaiting final smoothing review
    Not yet implemented