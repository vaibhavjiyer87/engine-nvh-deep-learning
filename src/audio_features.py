import numpy as np
import librosa


BAND_EDGES_HZ = [
    (0, 250),
    (250, 500),
    (500, 1000),
    (1000, 2000),
    (2000, 4000),
    (4000, 8000),
]


def extract_time_features(audio):
    """Extract basic time-domain features."""

    audio = np.asarray(audio, dtype=np.float64)

    rms = np.sqrt(
        np.mean(audio ** 2)
    )

    peak_abs = np.max(
        np.abs(audio)
    )

    crest_factor = (
        peak_abs / rms
        if rms > 1e-12
        else 0.0
    )

    signs = np.signbit(audio)

    zero_crossing_rate = np.mean(
        signs[1:] != signs[:-1]
    )

    return {
        "rms": float(rms),
        "peak_abs": float(peak_abs),
        "crest_factor": float(crest_factor),
        "zero_crossing_rate":
            float(zero_crossing_rate),
    }


def extract_spectral_features(
    audio,
    sample_rate_hz,
):
    """Extract basic frequency-domain descriptors."""

    audio = np.asarray(
        audio,
        dtype=np.float64,
    )

    centered = (
        audio - np.mean(audio)
    )

    n = len(centered)

    window = np.hanning(n)

    spectrum = np.fft.rfft(
        centered * window
    )

    magnitude = np.abs(
        spectrum
    )

    power = magnitude ** 2

    frequencies = np.fft.rfftfreq(
        n,
        d=1.0 / sample_rate_hz,
    )

    power_sum = np.sum(power)

    if power_sum <= 1e-20:
        centroid = 0.0
        bandwidth = 0.0
        rolloff = 0.0
        dominant_frequency = 0.0

    else:
        centroid = np.sum(
            frequencies * power
        ) / power_sum

        bandwidth = np.sqrt(
            np.sum(
                (
                    frequencies
                    - centroid
                ) ** 2
                * power
            )
            / power_sum
        )

        cumulative_power = (
            np.cumsum(power)
        )

        rolloff_index = np.searchsorted(
            cumulative_power,
            0.85 * power_sum,
        )

        rolloff_index = min(
            rolloff_index,
            len(frequencies) - 1,
        )

        rolloff = frequencies[
            rolloff_index
        ]

        # Ignore DC for dominant tone.
        if len(power) > 1:
            dominant_index = (
                1
                + np.argmax(
                    power[1:]
                )
            )
        else:
            dominant_index = 0

        dominant_frequency = (
            frequencies[
                dominant_index
            ]
        )

    positive_power = (
        power + 1e-20
    )

    spectral_flatness = (
        np.exp(
            np.mean(
                np.log(
                    positive_power
                )
            )
        )
        /
        np.mean(
            positive_power
        )
    )

    return {
        "spectral_centroid_hz":
            float(centroid),

        "spectral_bandwidth_hz":
            float(bandwidth),

        "spectral_rolloff_85_hz":
            float(rolloff),

        "spectral_flatness":
            float(spectral_flatness),

        "dominant_frequency_hz":
            float(dominant_frequency),
    }


def extract_band_energy_features(
    audio,
    sample_rate_hz,
    bands=BAND_EDGES_HZ,
):
    """Calculate relative FFT energy in defined bands."""

    audio = np.asarray(
        audio,
        dtype=np.float64,
    )

    centered = (
        audio - np.mean(audio)
    )

    n = len(centered)

    window = np.hanning(n)

    spectrum = np.fft.rfft(
        centered * window
    )

    power = (
        np.abs(spectrum) ** 2
    )

    frequencies = np.fft.rfftfreq(
        n,
        d=1.0 / sample_rate_hz,
    )

    total_power = max(
        np.sum(power),
        1e-20,
    )

    features = {}

    for low_hz, high_hz in bands:

        mask = (
            (frequencies >= low_hz)
            &
            (frequencies < high_hz)
        )

        band_power = np.sum(
            power[mask]
        )

        name = (
            f"band_energy_"
            f"{low_hz}_"
            f"{high_hz}_hz"
        )

        features[name] = float(
            band_power
            / total_power
        )

    return features


def extract_mfcc_features(
    audio,
    sample_rate_hz,
    n_mfcc=13,
):
    """Extract MFCC mean and standard deviation."""

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate_hz,
        n_mfcc=n_mfcc,
    )

    features = {}

    for index in range(
        n_mfcc
    ):

        coefficient = (
            index + 1
        )

        features[
            f"mfcc_{coefficient:02d}_mean"
        ] = float(
            np.mean(
                mfcc[index]
            )
        )

        features[
            f"mfcc_{coefficient:02d}_std"
        ] = float(
            np.std(
                mfcc[index]
            )
        )

    return features


def extract_baseline_features(
    audio,
    sample_rate_hz,
):
    """Extract complete FEAT-001 audio feature set."""

    features = {}

    features.update(
        extract_time_features(
            audio
        )
    )

    features.update(
        extract_spectral_features(
            audio,
            sample_rate_hz,
        )
    )

    features.update(
        extract_band_energy_features(
            audio,
            sample_rate_hz,
        )
    )

    features.update(
        extract_mfcc_features(
            audio,
            sample_rate_hz,
        )
    )

    return features
