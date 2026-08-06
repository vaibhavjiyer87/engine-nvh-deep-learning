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

## Current status (06 Aug 2026, 6am PST)
- [x] GitHub repository initialized
- [x] Google Drive storage configured
- [x] Google Colab environment verified
- [x] Dataset provenance documented
- [x] A_full_set acquired
- [x] Raw WAV structure validated
- [x] Raw-file manifest generated
- [ ] Processed sample manifest generated
- [ ] Train/validation/test split defined
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
