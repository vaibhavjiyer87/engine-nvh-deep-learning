# SPLIT-001 Dataset Split Decision

## Decision

The A_full_set source WAV files are divided into training,
validation, and test sets before audio-window generation.

The target fractions are:

- Training: 70%
- Validation: 15%
- Test: 15%

The resulting source-file counts are:

- Training: 536
- Validation: 115
- Test: 116

## Grouping and leakage control

The original WAV file is the grouping unit.

All model windows subsequently generated from one source WAV file
must inherit the split assignment of that source file.

Windows from the same source file are therefore prohibited from
appearing in multiple dataset splits.

## Stratification

The split was stratified using joint quantile bins based on:

- Mean RPM
- Mean torque

The final stratification used 4 bins per
variable.

This was done to maintain broadly similar operating-condition
coverage across training, validation, and test sets.

## Randomization

Primary random seed: 42

Secondary random seed: 43

## Test-set policy

The SPLIT-001 test set is frozen.

It will not be used for:

- Hyperparameter tuning
- Model selection
- Feature-selection decisions
- Early stopping

It is reserved for final model evaluation.

## Limitation

SPLIT-001 tests generalization to unseen source WAV files within
A_full_set. It does not yet represent generalization to an unseen
engine-synthesis family.

A future project stage should evaluate cross-subset generalization
using additional Procedural Engine Sounds subsets.
