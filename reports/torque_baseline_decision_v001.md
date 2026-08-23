# Torque Baseline Decision — v001

## Modeling population

TORQUE-SPEC-001 defines torque-model eligibility using RPM-steady
windows that also satisfy the torque-stability criterion:

torque_range_nm <= max(20 Nm, 0.05 * abs(torque_mean_nm))

The resulting modeling population contains 11,303 windows:

- Train: 7,739
- Validation: 1,749
- Test: 1,815

The minimum split retention relative to the RPM-eligible population
is 96.03%, with only a 1.24 percentage-point spread between splits.

This indicates that the torque-stability rule retains a large and
balanced modeling population.

## Validation models

TBASE-001:
StandardScaler + Ridge regression

TBASE-002:
RandomForestRegressor

Both models use the same 41 FEAT-001 acoustic predictors.

## Validation performance

| Model | MAE [Nm] | RMSE [Nm] | R² | P95 abs. error [Nm] |
|---|---:|---:|---:|---:|
| TBASE-001 Ridge | 76.61 | 92.61 | 0.7532 | 176.58 |
| TBASE-002 Random Forest | 23.32 | 41.57 | 0.9503 | 104.45 |

## Decision

TBASE-002 is selected as the preferred torque baseline because it
achieves substantially lower validation MAE, which is the frozen
primary model-selection metric.

The test split has not yet been evaluated.

TBASE-002 is approved for one-time frozen test evaluation in the
next session.

## Next phase

After the one-time TBASE-002 test evaluation, the torque baseline
will be frozen and used as the benchmark for RPM + torque
multi-task learning.
