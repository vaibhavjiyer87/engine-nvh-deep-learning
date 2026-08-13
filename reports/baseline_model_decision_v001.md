# Baseline RPM Model Decision — v001

## Objective

The objective of this stage is to establish a reproducible
conventional machine-learning baseline for predicting engine RPM
from one-second acoustic windows before development of the
deep-learning model.

## Modeling lineage

The baseline experiments use the frozen project artifacts:

- PREP-001
- SPLIT-001
- SAMPLE-MANIFEST-001
- ORDER-001
- FEAT-001

Only windows marked as steady-state and eligible under
`model_eligible_rpm_v001` are included in the first RPM model.

## Predictor definition

FEAT-001 contains audio-derived time-domain, spectral,
frequency-band, and MFCC features.

Ground-truth RPM, torque, dataset identifiers, split labels,
and RPM-dependent ORDER-001 features are excluded from the
predictor matrix.

## BASE-001 — Ridge Regression

BASE-001 uses a StandardScaler followed by Ridge regression.

The selected Ridge regularization parameter was:

- alpha = 10.0

Validation performance:

- MAE = 151.47 RPM
- RMSE = 201.73 RPM
- R² = 0.9853
- P95 absolute error = 399.02 RPM

## BASE-002 — Random Forest

BASE-002 uses a Random Forest regressor with the validation-selected
configuration:

- n_estimators = 300
- max_depth = 12
- min_samples_leaf = 2
- random_state = 42

Validation performance:

- MAE = 37.62 RPM
- RMSE = 127.39 RPM
- R² = 0.9941
- P95 absolute error = 186.70 RPM

## Baseline selection

BASE-002 was selected as the preferred baseline using validation
performance before evaluating the frozen test set.

The Random Forest produced the stronger validation performance and
is therefore designated as the conventional benchmark for subsequent
deep-learning experiments.

## Frozen BASE-002 test performance

After model selection was completed, BASE-002 was evaluated once on
the frozen test split.

Test performance:

- MAE = 39.80 RPM
- RMSE = 172.22 RPM
- R² = 0.9903
- P95 absolute error = 120.99 RPM

These test results were not used for additional model tuning.

## Feature importance

The five highest Random Forest feature-importance rankings are:

- spectral_centroid_hz
- band_energy_1000_2000_hz
- spectral_rolloff_85_hz
- dominant_frequency_hz
- zero_crossing_rate

Feature importance is interpreted as an indication of which FEAT-001
descriptors were most useful to the Random Forest. It does not by
itself establish physical causality.

## Engineering error analysis

Test errors are additionally evaluated as a function of engine-speed
band using RPM-band boundaries derived from the training population.

This analysis is used to identify operating regions where the
baseline performs comparatively well or poorly without using the
test set for further optimization.

## Limitations

The current model is restricted to steady-state windows from the
procedurally generated A_full_set dataset.

Performance on the frozen test split represents generalization to
unseen source WAV files within this dataset, not generalization to
a different engine, synthesis family, vehicle installation, or
measured production-vehicle recording.

The current feature set also consists of conventional handcrafted
audio descriptors. It does not learn its own time-frequency
representation directly from the acoustic signal.

## Next step

BASE-002 is retained as the conventional RPM-prediction benchmark.

The next stage will develop CNN-001 using a log-mel spectrogram
representation derived from the same PREP-001 model windows.
CNN-001 will be compared against BASE-002 using the same frozen
data split and engineering evaluation metrics.
