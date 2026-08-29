# Final Engineering Interpretation

## Main result

The strongest models in this project were conventional Random
Forest regressors trained on 41 engineered acoustic descriptors.

For the single-target RPM problem, BASE-002 achieved a validation
MAE of 37.62 RPM.

For torque, TBASE-002 achieved a validation MAE of
23.32 Nm.

The tested deep-learning models did not surpass these frozen
benchmarks.

## What the CNN experiments showed

CNN-001 established that a log-mel CNN can learn a strong
relationship between acoustic content and engine RPM. Its
validation error distribution also showed reduced large-error
behavior relative to the Random Forest in some metrics.

However, the frozen primary selection metric was MAE, and CNN-001
produced 60.19 RPM MAE compared with
37.62 RPM for BASE-002.

CNN-002 tested whether changing the regression loss would improve
this behavior. It did not produce a winning validation result.

CNN-003 explicitly preserved more absolute frequency structure in
the network, motivated by the physical relationship between RPM
and engine-order frequency. Its validation MAE was
62.73 RPM and therefore also failed to beat BASE-002.

## What the multi-task experiments showed

MTL-001 tested whether RPM and torque could benefit from a shared
learned acoustic representation.

MTL-002 then tested a stronger hybrid hypothesis: whether the CNN
could contribute incremental information when fused with the same
engineered descriptors already used successfully by the Random
Forest models.

On the shared validation population, MTL-002 produced:

- RPM MAE: 112.16 RPM
- Torque MAE: 27.30 Nm

The corresponding frozen conventional benchmarks were:

- RPM: 38.39 RPM
- Torque: 23.32 Nm

MTL-002 was therefore worse on both targets and was not evaluated
on the test set.

## Engineering interpretation

The results do not imply that CNNs are generally inappropriate for
NVH prediction.

They indicate that, for this dataset and modeling formulation,
the engineered acoustic representation is already highly
informative.

The FEAT-001 descriptors explicitly summarize quantities such as
spectral shape, energy distribution, dominant frequency,
time-domain amplitude behavior and MFCC statistics. These
quantities are closely related to the stationary acoustic
signatures that change with operating state.

A Random Forest can exploit nonlinear combinations of these
features efficiently without needing to learn the entire acoustic
representation from scratch.

Several factors may also limit the observed benefit of deep
learning:

1. The dataset is synthetic and the first project uses only the
   selected A_full_set subset.

2. One-second overlapping windows increase the number of samples
   but do not provide the same statistical independence as an
   equivalent number of unrelated recordings.

3. The first models intentionally focus on steady-state operating
   conditions.

4. The test design represents unseen source WAV files within the
   same dataset subset, not unseen real engines or synthesis
   families.

5. The CNN architectures were deliberately controlled rather than
   subjected to an open-ended hyperparameter search. This protects
   validation integrity but means the study does not claim that
   every possible deep architecture would underperform.

## Final modeling decision

BASE-002 remains the selected RPM model.

TBASE-002 remains the selected torque model.

No additional CNN or multi-task tuning is justified within this
project.

The project therefore closes with a useful engineering conclusion:
model complexity should be justified by measured improvement, not
by the expectation that deep learning must outperform a carefully
constructed conventional baseline.
