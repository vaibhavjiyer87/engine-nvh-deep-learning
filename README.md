# Deep-Learning Prediction of Engine Operating State and Order Content

## Project objective
Develop and validate a deep-learning model that predicts engine RPM,
torque, and selected engine-order amplitudes from acoustic recordings.

## Initial project scope
- Input: engine-audio signal
- Outputs: RPM, torque, and selected order amplitudes
- Platform: Google Colab
- Framework: PyTorch
- Storage: Google Drive
- Version control: GitHub

## Planned workflow
1. Acquire and audit the dataset
2. Develop signal-processing and order-analysis methods
3. Build conventional baseline models
4. Train a spectrogram-based CNN
5. Validate on unseen operating conditions
6. Add order-amplitude prediction

## Current status (11 Aug 2026, 5am PST)
- [x] GitHub repository initialized
- [x] Google Drive storage configured
- [x] Google Colab environment verified
- [x] Dataset provenance documented
- [x] A_full_set acquired
- [x] Raw WAV structure validated
- [x] Raw-file manifest generated
- [x] PREP-001 preprocessing specification frozen
- [x] SPLIT-001 train/validation/test split frozen
- [x] Sample-level manifest generated
- [ ] Signal and order-domain validation completed
- [ ] Baseline audio features generated
- [ ] Baseline RPM model trained

## Dataset

- Dataset name and citation
This project uses Version 1.0 of the Procedural Engine Sounds Dataset, created by Robin Doerfler. The dataset contains procedurally generated engine sounds with sample-accurate engine-speed and torque annotations.

- Primary dataset citation:
Doerfler, R. (2025). Procedural Engine Sounds Dataset (Version 1.0). Zenodo. DOI: 10.5281/zenodo.16883336.

- Associated publication:
Doerfler, R., & Wyse, L. (2026). “Analysis-Driven Procedural Generation of an Engine Sound Dataset with Embedded Control Annotations.” Proceedings of the 34th European Signal Processing Conference (EUSIPCO 2026). Preprint: arXiv 2603.07584.

- Initial subset used
The initial phase of this project uses the A_full_set subset. Starting with one subset keeps data processing and model development manageable while preserving a meaningful range of engine-speed and torque conditions.

- Dataset size
Subset:	A_full_set
Number of WAV files:	767
Approximate audio duration:	2.46 hours
Approximate storage size:	3.16 GB
Audio format:	WAV
Sampling rate:	48,000 Hz
Bit depth:	16 bit
Number of channels:	4

The project-generated raw-file manifest contains one row for each of the 767 source WAV files. Exact file-level durations, RPM statistics, torque statistics, and quality-control results are stored in raw_file_manifest_v001.csv.

- Four-channel signal structure
Each WAV file contains four synchronized channels:

1	Left-channel procedural engine sound
2	Right-channel procedural engine sound
3	Engine-speed annotation, stored as RPM multiplied by 0.0001
4	Engine-torque annotation, stored as Nm multiplied by 0.001

The physical annotations are recovered as follows:
Engine speed: RPM = Channel 3 × 10,000
Engine torque: Torque [Nm] = Channel 4 × 1,000

Because the annotations are stored as continuous signals, RPM and torque are synchronized with the audio at the sample level rather than being represented by only one label per recording.

- Synthetic-data limitation
The recordings are generated using procedural audio-synthesis methods rather than collected directly from production vehicles. The synthesis process was developed using harmonic characteristics extracted from real engine recordings, but the resulting signals may not capture the full variability of real automotive measurements.

Important limitations include:
The dataset may not represent every engine architecture, combustion behavior, or exhaust configuration.
Some generated engine and exhaust combinations are fictional.
The recordings do not include all real-world measurement effects, such as microphone-position variation, cabin transfer paths, road noise, wind noise, accessory noise, reflections, sensor noise, and environmental contamination.
Models trained only on this dataset may experience a synthetic-to-real domain gap when applied to measured vehicle data.

The dataset is therefore appropriate for developing and validating the initial signal-processing and deep-learning workflow, but final automotive applicability will require validation and fine-tuning using measured engine or vehicle data.

- License
The dataset is distributed under the Creative Commons Attribution–NonCommercial 4.0 International license, abbreviated as CC BY-NC 4.0.
Use of the dataset requires attribution to the original author, and the dataset cannot be used for commercial purposes without obtaining additional permission. The dataset citation and license terms should be reviewed before distributing derived datasets or trained models.

- Data-storage policy
The raw audio files are not stored in this GitHub repository.
Instead, the unmodified dataset is downloaded from its official source and stored separately in Google Drive. This approach is used because:
The initial subset is approximately 3.16 GB and is unsuitable for normal Git version control.
Storing large binary audio files would unnecessarily increase repository size and cloning time.
Keeping raw data separate prevents accidental modification or duplication of the official dataset.
Users can obtain the data directly from the authoritative source and verify its current license and documentation.
GitHub is reserved for source code, notebooks, configuration files, manifests, selected figures, and compact result tables.

The expected local Google Drive location is:
NVH_DeepLearning/01_EngineOperatingState/data/raw/procedural_engine_sounds/dataset/audio/A_full_set/

The original dataset folder structure and filenames are preserved. Instructions for obtaining and organizing the raw data are provided in data/README.md.

## Dataset splitting

The initial dataset split, `SPLIT-001`, is defined at the original
WAV-file level before window segmentation.

The split uses approximately:

- 70% training
- 15% validation
- 15% test

RPM and torque operating-condition coverage is balanced using joint


## Model-ready sample definition

`SAMPLE-MANIFEST-001` defines the first model-ready sample
population.

Each sample represents a one-second audio window with 50% overlap.
Every sample retains window-level RPM and torque statistics and
inherits the train, validation, or test assignment of its parent
source WAV file.

Transient windows are retained in the manifest, while the first RPM
regression model uses windows marked as steady-state.

The manifest is generated under:

- `PREP-001`
- `SPLIT-001`

The full sample manifest is stored externally in Google Drive.
Its reproducibility record and SHA-256 checksum are stored in
`data/sample_manifest_v001_record.yaml`.
stratification.

All windows from the same source recording remain in the same split
to prevent data leakage.

See `configs/split_v001.yaml` for the frozen specification.

<!-- PORTFOLIO_CASE_STUDY_START -->

# Engine Operating-State Prediction from Acoustic Signals

## Engineering question

Can engine RPM and torque be estimated directly from engine-noise
recordings, and do learned deep representations improve upon
traditional NVH feature engineering?

## Dataset

The project uses the Procedural Engine Sounds Dataset and initially
focuses on the `A_full_set` subset.

The source WAV files contain four channels:

- left engine audio,
- right engine audio,
- scaled engine speed,
- scaled engine torque.

Raw audio is not stored in this repository.

## Modeling pipeline

The project follows a versioned engineering workflow:

`PREP-001 → SPLIT-001 → SAMPLE-MANIFEST-001 → ORDER-001 → FEAT-001`

Two acoustic representations were evaluated:

1. **FEAT-001** — 41 engineered time, spectral, band-energy and
   MFCC predictors.
2. **Log-mel spectrograms** — used as input to CNN and multi-task
   deep-learning models.

All model selection was performed using validation data.

Test data was evaluated only after a model had satisfied its frozen
validation-selection rule.

## Main results

### RPM

The selected conventional model was **BASE-002**, a Random Forest
using FEAT-001.

Validation MAE:

**37.62 RPM**

The controlled CNN experiments did not improve upon this metric.

### Torque

The selected torque model was **TBASE-002**, also a Random Forest
using FEAT-001.

Validation MAE:

**23.32 Nm**

### Final deep-learning experiment

MTL-002 combined:

- a frequency-aware log-mel CNN representation, and
- the same 41 engineered FEAT-001 descriptors.

On the shared validation population:

| Target | Conventional benchmark | MTL-002 |
|---|---:|---:|
| RPM MAE | 38.39 RPM | 112.16 RPM |
| Torque MAE | 23.32 Nm | 27.30 Nm |

MTL-002 did not satisfy the frozen validation criterion and was not
evaluated on the test split.

## Main engineering conclusion

For this synthetic dataset, sample population and one-second
steady-state formulation, carefully engineered NVH acoustic
features combined with Random Forest regression provided better
predictive accuracy than the tested end-to-end and hybrid
deep-learning architectures.

This does **not** imply that CNNs are generally inferior for NVH.
Instead, it demonstrates why strong conventional baselines,
controlled experiments and strict validation/test separation are
necessary before adding model complexity.

## Model-development sequence

FEAT-001 + Random Forest
→ strong conventional benchmark
→ CNN-001 conventional log-mel CNN
→ CNN-002 loss-function experiment
→ CNN-003 frequency-aware CNN
→ MTL-001 shared RPM + torque CNN
→ MTL-002 hybrid CNN + engineered features
→ conventional engineered-feature models remain preferred

## Final selected models

- **RPM:** BASE-002 — Random Forest + FEAT-001
- **Torque:** TBASE-002 — Random Forest + FEAT-001

Further CNN tuning was intentionally stopped after the final planned
MTL-002 experiment.

## Reproducibility

See:

`docs/reproducibility_and_architecture.md`

for the project lineage, storage strategy, experiment protocol and
reproduction workflow.

<!-- PORTFOLIO_CASE_STUDY_END -->
