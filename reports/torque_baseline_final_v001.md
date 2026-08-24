# Final Torque Baseline — v001

## Population

TORQUE-SPEC-001 contains 11,303 steady-state windows:

- Train: 7,739
- Validation: 1,749
- Test: 1,815

Minimum retention relative to the RPM-steady population was 96.03%.
The split retention spread was 1.24 percentage points.

## Selected model

TBASE-002 — RandomForestRegressor using the frozen 41 FEAT-001
acoustic predictors.

## Validation performance

- MAE: 23.32 Nm
- RMSE: 41.57 Nm
- R²: 0.950266
- P95 absolute error: 104.45 Nm

TBASE-002 was selected using validation MAE before the test set was
evaluated.

## Frozen test performance

- MAE: 20.74 Nm
- RMSE: 41.36 Nm
- R²: 0.945178
- P95 absolute error: 92.56 Nm

## Status

TBASE-002 is now the frozen conventional torque benchmark.

The test result is final evaluation evidence and will not be used to
tune subsequent models.

## Next phase

RPM and torque will now be modeled jointly using MTL-001.

MTL-001 will be developed and selected using training and validation
data only. Its test split will remain untouched until the multi-task
model-selection decision has been frozen.
