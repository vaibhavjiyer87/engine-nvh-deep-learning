# Reproducibility and Project Architecture

## Project objective

Estimate engine RPM and torque from acoustic measurements while
comparing conventional NVH feature engineering with deep-learning
representations.

The project intentionally uses a staged model-development process
rather than assuming that the most complex model will perform best.

## Pipeline lineage

Raw 4-channel WAV files

→ Raw file manifest

→ PREP-001

→ SPLIT-001

→ SAMPLE-MANIFEST-001

→ ORDER-001

→ FEAT-001

→ Conventional RPM / torque baselines

→ Log-mel CNN models

→ Multi-task CNN models

→ Hybrid CNN + engineered-feature model

→ Frozen model-selection decisions


## Storage architecture

### GitHub repository

GitHub contains compact, reproducible project artifacts:

- notebooks
- Python source modules
- frozen YAML specifications
- split definitions
- artifact fingerprint records
- compact result tables
- selected figures
- engineering decision reports
- portfolio documentation

### Persistent external storage

Large artifacts are stored outside GitHub:

- raw WAV files
- large feature caches
- HDF5 log-mel cache
- trained model binaries
- live model-recovery checkpoints

Canonical persistent project path:

`/content/drive/MyDrive/NVH_DeepLearning/01_EngineOperatingState`


## Frozen preprocessing — PREP-001

PREP-001 defines:

- 1.0 second analysis windows
- 50% overlap
- 48 kHz source sampling rate
- 16 kHz model sampling rate
- mono average of the two engine-audio channels
- no per-window amplitude normalization
- primary RPM target: `rpm_mean`
- secondary torque target: `torque_mean_nm`


## Split protocol — SPLIT-001

The split is performed at the original source-WAV level.

All windows created from one parent WAV inherit the same split.

This prevents overlapping windows from the same source recording
from appearing in training, validation and test simultaneously.

Approximate split:

- 70% train
- 15% validation
- 15% test


## Model-selection protocol

### Training data

Used to:

- fit model parameters
- calculate normalization statistics
- fit feature transformations

### Validation data

Used to:

- compare model candidates
- select checkpoints
- make model-development decisions

### Test data

Used only after a model passes its frozen validation-selection rule.

Rejected CNN and multi-task models were therefore not evaluated on
the test split.


## FEAT-001 engineered acoustic representation

FEAT-001 contains 41 predictors.

### Time-domain features

- RMS
- peak absolute amplitude
- crest factor
- zero-crossing rate

### Spectral features

- spectral centroid
- spectral bandwidth
- spectral rolloff
- spectral flatness
- dominant frequency

### Band-energy features

Six frequency-band energy descriptors.

### MFCC features

- 13 MFCC means
- 13 MFCC standard deviations


## Log-mel representation

The CNN models use log-mel spectrograms generated with:

- sampling rate: 16 kHz
- FFT size: 1024
- window length: 1024
- hop length: 256
- mel bands: 64
- frequency range: 20–8000 Hz


## Model-development sequence

### RPM models

- BASE-001 — Ridge regression
- BASE-002 — Random Forest
- CNN-001 — conventional log-mel CNN
- CNN-002 — alternative CNN regression loss
- CNN-003 — frequency-aware CNN

### Torque models

- TBASE-001 — Ridge regression
- TBASE-002 — Random Forest

### Multi-task models

- MTL-001 — shared RPM + torque CNN
- MTL-002 — hybrid log-mel CNN + FEAT-001 representation


## Final selected models

### RPM

BASE-002

Random Forest using FEAT-001.

### Torque

TBASE-002

Random Forest using FEAT-001.


## Reproduction workflow

Run notebooks in numerical order.

Frozen configuration files should be treated as immutable experiment
specifications.

If an experimental specification changes, create a new version
rather than modifying a previously evaluated configuration.

Large persistent artifacts must be restored from external storage
before running notebooks that depend on them.


## Known limitations

- The dataset is synthetic.
- The first project focuses on the `A_full_set` subset.
- One-second overlapping windows are statistically correlated.
- The initial modeling phase focuses on steady-state operation.
- The test split contains unseen source WAV files within the same
  dataset subset rather than unseen real engines.
- Deep-learning experiments were deliberately controlled rather
  than subjected to an open-ended hyperparameter search.


## Engineering conclusion

For this dataset and modeling formulation, carefully engineered
acoustic descriptors combined with Random Forest regression
outperformed the tested CNN, multi-task and hybrid deep-learning
architectures.

The project therefore supports using model complexity only when it
produces measurable engineering benefit.
