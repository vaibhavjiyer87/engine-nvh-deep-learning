# Single-Target RPM Model Selection — Final v001

## Objective

Estimate steady-state engine RPM from acoustic measurements using
the frozen PREP-001 / SPLIT-001 modeling population.

## Primary model-selection criterion

Validation MAE was frozen as the primary selection metric before
deep-learning model development.

## Candidates evaluated

Four formal RPM models were compared:

1. BASE-002 — Random Forest using FEAT-001 handcrafted features
2. CNN-001 — conventional 2D CNN using log-mel spectrograms
3. CNN-002 — CNN loss-function experiment
4. CNN-003 — frequency-aware CNN architecture

## Validation results

| Model | MAE [RPM] | RMSE [RPM] | R² | P95 abs. error [RPM] |
|---|---:|---:|---:|---:|
| BASE-002 | 37.62 | 127.39 | 0.994132 | 186.70 |
| CNN-001 | 60.19 | 76.36 | 0.997892 | 145.33 |
| CNN-002 | 101.62 | 128.24 | 0.994053 | 243.60 |
| CNN-003 | 62.73 | 83.58 | 0.997474 | 175.21 |

## Final decision

BASE-002 remains the selected single-target RPM model because it
achieves the lowest validation MAE.

CNN-003 produced a validation MAE of
62.73 RPM compared with
37.62 RPM for BASE-002.

CNN-003 therefore increased the primary validation error by
approximately 66.75% relative to BASE-002.

No CNN candidate was evaluated on the test split because none
surpassed the frozen BASE-002 validation benchmark.

## Engineering interpretation

The experiments indicate that, for this dataset and current
one-second acoustic representation, handcrafted spectral/time
features combined with Random Forest regression are more effective
for precise single-target RPM estimation than the tested
end-to-end log-mel CNN approaches.

CNN-001 demonstrated that the deep representation could control
some larger errors effectively, but its mean absolute error remained
inferior. Changing the regression loss in CNN-002 and preserving
absolute frequency structure more explicitly in CNN-003 did not
overcome that deficit.

Further CNN tuning is therefore stopped at this stage to avoid
iterative optimization against the validation set.

## Next modeling phase

The project now moves to torque prediction and RPM + torque
multi-task learning.

A conventional torque benchmark will be established first. A
multi-task neural model will then be evaluated to determine whether
shared acoustic representations provide value when RPM and torque
are learned jointly.
