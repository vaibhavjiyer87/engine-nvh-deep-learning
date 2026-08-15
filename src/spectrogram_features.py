import numpy as np
import librosa


def compute_log_mel(
    audio,
    sample_rate_hz,
    n_fft,
    win_length,
    hop_length,
    n_mels,
    fmin_hz,
    fmax_hz,
):
    """
    Compute a log-mel power spectrogram for one mono audio window.

    No per-window amplitude normalization is applied.
    """

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    if audio.ndim != 1:
        raise ValueError(
            "Audio must be a one-dimensional mono signal."
        )

    if not np.isfinite(audio).all():
        raise ValueError(
            "Audio contains NaN or infinite values."
        )

    mel_power = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate_hz,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin_hz,
        fmax=fmax_hz,
        power=2.0,
        center=False,
    )

    # Convert absolute power to dB.
    # top_db=None avoids per-window dynamic-range clipping.
    log_mel = librosa.power_to_db(
        mel_power,
        ref=1.0,
        amin=1e-10,
        top_db=None,
    )

    return log_mel.astype(
        np.float32
    )
