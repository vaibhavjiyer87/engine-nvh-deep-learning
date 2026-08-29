# Final Modeling Conclusion — Engine Operating State Project

## Modeling sequence

The project evaluated the following model families:

1. BASE-001 — Ridge RPM regression
2. BASE-002 — Random Forest RPM regression
3. CNN-001 — conventional log-mel CNN
4. CNN-002 — alternative CNN regression loss
5. CNN-003 — frequency-aware CNN
6. TBASE-001 — Ridge torque regression
7. TBASE-002 — Random Forest torque regression
8. MTL-001 — shared CNN RPM + torque model
9. MTL-002 — hybrid CNN + engineered-feature multi-task model

## Conventional benchmarks

RPM benchmark:
BASE-002

Torque benchmark:
TBASE-002

## Deep-learning investigation

The CNN experiments tested whether learned spectral representations
could replace or improve upon engineered NVH acoustic descriptors.

MTL-001 additionally tested whether joint RPM and torque supervision
could improve the learned representation.

MTL-002 was the final planned experiment and tested whether learned
CNN features could add incremental information when fused with the
same engineered descriptors used by the conventional baselines.

## MTL-002 validation result

Best epoch:
14

RPM benchmark MAE:
38.39 RPM

MTL-002 RPM MAE:
112.16 RPM

Torque benchmark MAE:
23.32 Nm

MTL-002 torque MAE:
27.30 Nm

MTL-002 decision:
not_selected_for_test

## Final interpretation

None of the tested deep-learning approaches surpassed the frozen conventional benchmarks under the predefined validation criteria. The results support BASE-002 and TBASE-002 as the preferred models for this dataset. The experiments show that the tested learned log-mel representations did not provide sufficient incremental predictive value over carefully engineered NVH features.

## Model-development status

Model development for this project is now CLOSED.

No CNN-004, MTL-003, hyperparameter sweep, or further validation-set
optimization is planned.

The remaining work is project documentation, visualization,
portfolio packaging and lessons learned.
