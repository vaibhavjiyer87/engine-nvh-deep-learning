# CNN-001 Validation Review — v001

## Status

CNN-001 has completed training and validation evaluation.

The frozen primary selection metric for the RPM modeling workflow is
validation MAE. CNN-001 therefore has not yet been approved for
frozen test-set evaluation.

## Validation comparison

| Metric | BASE-002 | CNN-001 |
|---|---:|---:|
| MAE [RPM] | 37.62 | 60.19 |
| RMSE [RPM] | 127.39 | 76.36 |
| R² | 0.994132 | 0.997892 |
| P95 absolute error [RPM] | 186.70 | 145.33 |

## Current interpretation

BASE-002 remains superior according to the frozen primary metric,
validation MAE.

CNN-001 nevertheless produces lower RMSE, higher R², and lower P95
absolute error. This suggests that the two models may have
substantially different error distributions.

A likely hypothesis is that BASE-002 achieves very small errors for
many samples but experiences a smaller number of large errors,
whereas CNN-001 may produce more broadly distributed moderate errors.

This hypothesis has not yet been accepted and will be tested using
sample-level validation residual analysis.

## Test-set policy

CNN-001 has not been evaluated on the frozen test set.

The test set will remain untouched while CNN validation-stage
diagnostics and any subsequent CNN model iteration are performed.

## Next analysis

The next session will compare BASE-002 and CNN-001 validation errors
at sample level, including:

- absolute-error quantiles;
- residual behavior versus RPM;
- performance by RPM operating band;
- largest-error samples.

The results will guide the definition of CNN-002.
