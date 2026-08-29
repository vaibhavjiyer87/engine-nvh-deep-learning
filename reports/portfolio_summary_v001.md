# Portfolio Summary

## Project

Engine Operating-State Prediction from Acoustic Signals

## Engineering objective

Develop a machine-learning workflow capable of estimating engine RPM
and torque from engine-acoustic measurements and determine whether
deep-learning representations provide measurable benefit over
traditional NVH feature engineering.

## Technical scope

The project included:

- dataset ingestion and audit
- leakage-safe source-file splitting
- signal preprocessing
- engine-order analysis
- acoustic feature engineering
- log-mel spectrogram generation
- conventional regression
- CNN modeling
- multi-task learning
- hybrid learned + engineered feature fusion
- crash-safe training recovery
- frozen validation/test model-selection rules
- reproducible artifact and experiment management

## Final selected models

### RPM

BASE-002

Random Forest using 41 FEAT-001 engineered acoustic descriptors.

Validation MAE:

**37.62 RPM**

### Torque

TBASE-002

Random Forest using the same FEAT-001 representation.

Validation MAE:

**23.32 Nm**

## Deep-learning result

The project evaluated:

- conventional log-mel CNNs
- alternative regression loss
- frequency-aware CNN architecture
- RPM + torque multi-task learning
- hybrid CNN + engineered-feature fusion

The final hybrid model, MTL-002, achieved:

- RPM validation MAE: **112.16 RPM**
- Torque validation MAE: **27.30 Nm**

The corresponding conventional benchmarks on the same shared
validation population were:

- RPM: **38.39 RPM**
- Torque: **23.32 Nm**

MTL-002 therefore failed its frozen validation-selection criterion
and was not evaluated on the test split.

## Engineering conclusion

For this dataset and one-second steady-state acoustic formulation,
domain-informed acoustic descriptors combined with Random Forest
regression provided better predictive accuracy than the tested
end-to-end and hybrid deep-learning architectures.

The project demonstrates that model complexity should be justified
by measurable improvement rather than assumed to be beneficial.

## Resume-style bullets

- Developed a version-controlled NVH machine-learning pipeline for
  engine RPM and torque estimation from acoustic measurements,
  including leakage-safe source-level splitting, signal
  preprocessing, feature engineering and model evaluation.

- Implemented and compared Random Forest regression, log-mel CNNs,
  frequency-aware CNNs, multi-task neural networks and hybrid
  engineered/learned representations under a strict
  validation/test protocol.

- Demonstrated that 41 domain-informed acoustic descriptors with
  Random Forest regression outperformed the tested deep-learning
  architectures, achieving approximately
  **37.6 RPM** validation MAE for RPM and
  **23.3 Nm** for torque.

## 60-second interview explanation

I wanted to test whether engine operating state could be inferred
directly from acoustic data using deep learning.

I first created strong conventional baselines using NVH-inspired
time-domain, spectral, band-energy and MFCC features.

I then progressed through conventional CNNs, a frequency-aware CNN,
multi-task RPM and torque learning, and finally a hybrid network that
combined learned spectrogram representations with the engineered
features.

The key result was that the Random Forest baselines remained more
accurate.

Rather than continuing to tune neural networks against the
validation set until one happened to win, I stopped model
development according to predefined criteria.

That made the main engineering lesson of the project very clear:
use the simplest modeling approach that the evidence supports.
