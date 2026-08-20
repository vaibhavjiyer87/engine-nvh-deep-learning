# CNN-002 Validation Decision — v001

## Experiment

CNN-002 evaluates SmoothL1Loss while retaining the CNN-001 data,
log-mel representation, model architecture, split, optimizer and
training protocol.

The primary model-selection metric remains validation MAE.

## Validation comparison

| Model | Validation MAE [RPM] |
|---|---:|
| BASE-002 | 37.62 |
| CNN-001 | 60.19 |
| CNN-002 | 101.62 |

CNN-002 additional validation metrics:

- RMSE: 128.24 RPM
- R²: 0.994053
- P95 absolute error: 243.60 RPM

## Decision

CNN-002 does not improve upon the frozen BASE-002 benchmark on the
primary selection metric, validation MAE. CNN-002 is therefore not
approved for test-set evaluation. The frozen test split remains
untouched while the next CNN iteration is developed.

## Status

Decision: `not_selected_for_test`
