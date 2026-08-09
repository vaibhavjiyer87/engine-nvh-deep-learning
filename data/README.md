## Dataset location
The full dataset is not stored in this GitHub repository.
After obtaining the dataset from its official source, place the unmodified files under:

Google Drive:
NVH_DeepLearning/01_EngineOperatingState/data/raw/

The original directory structure should be preserved.

## Dataset review and documentation
Dataset name: Procedural Engine Sounds Dataset
Subset: "A_full_set"
Author: Robin Doerfler
Source: rdoerfler/procedural-engine-sounds
Version "1.0" 
Download date: "2026-08-03"
Primary_DOI: 10.5281/zenodo.16883336
License: CC BY-NC 4.0 license
Citation requirements
Approximate size: 24.5GB ("A_full_set" is 3.16GB)
Available labels: 


## Dataset split

`data/splits/file_split_v001.csv` contains the frozen SPLIT-001
assignment for each source WAV file.

The split is performed before audio segmentation.

All processed windows generated from a source WAV file inherit the
same train, validation, or test assignment. This prevents overlapping
windows from the same recording from appearing in multiple splits.

The split methodology is defined in: `configs/split_v001.yaml`
Audio file format
Sampling rates
Whether RPM is instantaneous or constant
Whether torque is measured, simulated or procedural
Whether engine identity is available
Any restrictions on redistribution
