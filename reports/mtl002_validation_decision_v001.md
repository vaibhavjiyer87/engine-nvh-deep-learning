# MTL-002 Validation Decision — v001

## Experiment

MTL-002 is the final planned model-development experiment in the
engine operating-state project.

It combines:

- a frequency-aware log-mel CNN representation, and
- the same 41 engineered FEAT-001 acoustic descriptors used by
  the strong conventional RPM and torque baselines.

The objective was to determine whether learned representations
provide incremental predictive value when combined with
domain-engineered NVH features.

## Training

Best checkpoint epoch: 14

Early stopping was used according to the frozen specification.

## RPM validation

BASE-002 shared-population MAE:
38.39 RPM

MTL-002 MAE:
112.16 RPM

Relative change:
+192.19%

## Torque validation

TBASE-002 shared-population MAE:
23.32 Nm

MTL-002 MAE:
27.30 Nm

Relative change:
+17.05%

## Frozen success criterion

MTL-002 must:

1. match or outperform the conventional benchmark on both target MAEs;
2. improve at least one target MAE by 2% or more.

## Decision

`not_selected_for_test`

This is the final planned model-development experiment regardless
of the validation outcome.
