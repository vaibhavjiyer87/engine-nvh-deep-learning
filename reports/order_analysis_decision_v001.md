# ORDER-001 Signal and Order Analysis Decision

## Purpose

This analysis validates the physical consistency of the model-ready acoustic windows defined by `SAMPLE-MANIFEST-001` before feature generation or machine-learning model development.

The objectives are to:

* Verify that a sample referenced by the sample manifest can be reconstructed correctly from the original four-channel WAV file.
* Confirm consistency between the RPM and torque values stored in the manifest and the embedded source annotations.
* Inspect representative steady-state samples in the time and frequency domains.
* Compare measured spectral content with theoretical engine-order frequencies.
* Establish a reproducible initial method for extracting engine-order amplitudes.

## Input specifications

The analysis uses the following frozen upstream artifacts:

* Preprocessing specification: `PREP-001`
* Dataset split: `SPLIT-001`
* Sample manifest: `SAMPLE-MANIFEST-001`

Exploratory signal- and order-analysis decisions are based only on steady-state samples assigned to the training set. Validation and test samples are not used to design the order-extraction method.

## Sample reconstruction

Each model sample is reconstructed from its source WAV file using the source start and end sample indices stored in `SAMPLE-MANIFEST-001`.

The two acoustic channels are averaged according to `PREP-001`, and the resulting mono signal is resampled from 48 kHz to 16 kHz for model-oriented analysis.

The embedded RPM and torque annotation channels remain at their original source resolution when calculating operating-state statistics.

The reusable implementation is stored in:

`src/data_loader.py`

## Representative operating conditions

Representative steady-state training samples were selected across the RPM–torque operating envelope rather than being manually selected based on visually favorable spectra.

The selected samples are recorded in:

`results/tables/signal_order_analysis/representative_samples_v001.csv`

This selection provides examples spanning low, medium, and high engine speeds together with lower and higher torque conditions.

## Time-domain validation

For each representative sample, the acoustic waveform, RPM trace, and torque trace were inspected over the one-second PREP-001 window.

The reconstructed RPM and torque statistics were checked against the corresponding values in `SAMPLE-MANIFEST-001`. This confirms that the manifest coordinates and source-channel decoding are consistent with the original WAV data.

## Frequency-domain analysis

A Hann-windowed FFT is applied to each one-second, 16-kHz acoustic sample after removal of the signal mean.

The one-second analysis duration provides an FFT-bin spacing of approximately 1 Hz.

The theoretical frequency associated with rotational order (N) is calculated using:

`f_order = N × RPM / 60`

The initial candidate orders investigated are:

* 0.5X
* 1.0X
* 1.5X
* 2.0X
* 3.0X
* 4.0X

These candidate orders are exploratory engineering features and are not yet assumed to be equally useful predictors.

## STFT validation

Short-time Fourier transform analysis is used to examine whether tonal structures remain reasonably stable within samples classified as steady-state.

The initial STFT configuration uses:

* Hann window
* 1024 samples per segment
* 768 samples overlap
* 16-kHz sample rate

For steady-state samples, expected order frequencies are treated as approximately constant over the one-second window. Transient windows will require a different time-varying order-tracking approach and are outside the scope of `ORDER-001`.

## Order-amplitude extraction

`ORDER-001` does not use only the FFT bin nearest the theoretical order frequency.

Instead, a local frequency search is performed around each expected order location. The search width accounts for:

1. The FFT frequency resolution.
2. The RPM variation measured within the sample.

The search-band half-width is defined using at least two FFT bins together with an additional frequency allowance derived from the observed RPM range.

The maximum spectral amplitude within this local search band is retained together with:

* The theoretical order frequency.
* The measured peak frequency.
* The difference between measured and theoretical frequencies.
* The extracted order amplitude in dBFS.

Initial extraction results are stored in:

`results/tables/signal_order_analysis/order_probe_results_v001.csv`

## Decision

The `ORDER-001` methodology is accepted as the initial reproducible order-analysis approach for this project.

It provides a physics-informed method for locating spectral energy associated with expected rotational orders while allowing for finite FFT resolution and small RPM variation within steady-state windows.

The extracted order amplitudes will be treated as derived features rather than being added to the frozen `SAMPLE-MANIFEST-001`.

Individual order usefulness will be evaluated during subsequent baseline feature and model analysis rather than assumed from the exploratory spectra alone.

## Limitations

`ORDER-001` is designed for steady-state samples. It is not a complete order-tracking algorithm for rapidly changing RPM.

The dataset is procedurally generated, and observed order behavior should therefore not be interpreted as proof that the same spectral relationships will transfer directly to measured production-vehicle recordings.

The current analysis also uses mean RPM over each one-second window to establish theoretical order locations. More advanced time-varying order tracking can be introduced in a future project version if transient operation becomes a modeling objective.

## Next step

The next project phase will define `FEAT-001` and generate conventional audio and NVH-oriented features, including time-domain, spectral, frequency-band, MFCC, and validated `ORDER-001` features.

These features will support the first formal RPM-prediction baseline experiments before development of the deep-learning model.
