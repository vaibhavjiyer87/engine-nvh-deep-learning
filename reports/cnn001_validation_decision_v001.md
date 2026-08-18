# CNN-001 Validation Decision — v001

## Objective

CNN-001 evaluates whether a convolutional neural network operating
directly on log-mel spectrograms can improve steady-state RPM
prediction relative to the conventional BASE-002 Random Forest.

## Primary selection metric

Validation MAE was frozen as the primary selection metric before
CNN-001 evaluation.

## Aggregate validation results

| Metric | BASE-002 | CNN-001 |
|---|---:|---:|
| MAE [RPM] | 37.62 | 60.19 |
| RMSE [RPM] | 127.39 | 76.36 |
| R² | 0.994132 | 0.997892 |
| P95 absolute error [RPM] | 186.70 | 145.33 |

## Decision

CNN-001 does not replace BASE-002 because its validation MAE is
higher than the frozen BASE-002 benchmark.

BASE-002 therefore remains the preferred RPM baseline according to
the pre-defined selection policy.

## Important secondary result

CNN-001 nevertheless demonstrates substantially lower RMSE, higher
R², and lower P95 absolute error.

This indicates that the two models have materially different error
distributions. The CNN appears to control larger errors more
effectively, while the Random Forest achieves lower average absolute
error.

Sample-level validation diagnostics were therefore performed before
defining the next CNN experiment.

## Test-set status

CNN-001 has not been evaluated on the frozen test set.

The test set remains untouched because CNN-001 was not selected by
the primary validation metric.

## Next model iteration

CNN-002 will preserve:

- PREP-001
- SPLIT-001
- SAMPLE-MANIFEST-001
- the CNN log-mel representation
- CNN-001 architecture
- optimizer
- learning rate
- training population

The primary experimental change will be the regression loss.

CNN-002 will replace MSELoss with SmoothL1Loss so that the effect of
the loss function on validation MAE can be evaluated independently
from changes in network architecture.

The frozen primary comparison metric remains validation MAE.
