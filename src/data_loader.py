import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def load_manifest_window(
    sample_row,
    raw_root,
    source_sample_rate_hz,
    target_sample_rate_hz,
    audio_left_index,
    audio_right_index,
    rpm_channel_index,
    torque_channel_index,
    rpm_scale_factor,
    torque_scale_factor_nm,
):
    """
    Reconstruct one model window from SAMPLE-MANIFEST-001.

    Parameters
    ----------
    sample_row
        Pandas Series or dictionary-like object containing the
        sample-manifest row.

    raw_root
        Root directory containing the original dataset.

    source_sample_rate_hz
        Expected source WAV sampling rate.

    target_sample_rate_hz
        Target model sampling rate.

    audio_left_index, audio_right_index
        Zero-based acoustic channel indices.

    rpm_channel_index, torque_channel_index
        Zero-based embedded annotation-channel indices.

    rpm_scale_factor
        Scale factor converting the embedded RPM channel to RPM.

    torque_scale_factor_nm
        Scale factor converting the embedded torque channel to Nm.

    Returns
    -------
    dict
        Dictionary containing source-rate mono audio,
        model-rate mono audio, RPM trace, and torque trace.
    """

    wav_path = (
        raw_root
        / sample_row["relative_raw_path"]
    )

    if not wav_path.exists():
        raise FileNotFoundError(
            f"Source WAV not found: {wav_path}"
        )

    signal, sample_rate = sf.read(
        wav_path,
        always_2d=True,
    )

    if sample_rate != source_sample_rate_hz:
        raise ValueError(
            f"Expected source sample rate "
            f"{source_sample_rate_hz} Hz, "
            f"found {sample_rate} Hz."
        )

    start_sample = int(
        sample_row[
            "source_start_sample"
        ]
    )

    end_sample = int(
        sample_row[
            "source_end_sample_exclusive"
        ]
    )

    window = signal[
        start_sample:end_sample,
        :
    ]

    # --------------------------------------------------------
    # Acoustic channels
    # --------------------------------------------------------

    left = window[
        :,
        audio_left_index
    ]

    right = window[
        :,
        audio_right_index
    ]

    mono_source = (
        left + right
    ) / 2.0

    # --------------------------------------------------------
    # Embedded operating-state annotations
    # --------------------------------------------------------

    rpm_trace = (
        window[
            :,
            rpm_channel_index
        ]
        * rpm_scale_factor
    )

    torque_trace_nm = (
        window[
            :,
            torque_channel_index
        ]
        * torque_scale_factor_nm
    )

    # --------------------------------------------------------
    # Resample acoustic signal only
    # --------------------------------------------------------

    gcd = np.gcd(
        source_sample_rate_hz,
        target_sample_rate_hz,
    )

    up = (
        target_sample_rate_hz
        // gcd
    )

    down = (
        source_sample_rate_hz
        // gcd
    )

    mono_model = resample_poly(
        mono_source,
        up,
        down,
    )

    return {
        "mono_source":
            mono_source,

        "mono_model":
            mono_model,

        "rpm_trace":
            rpm_trace,

        "torque_trace_nm":
            torque_trace_nm,

        "source_sample_rate_hz":
            source_sample_rate_hz,

        "model_sample_rate_hz":
            target_sample_rate_hz,
    }
