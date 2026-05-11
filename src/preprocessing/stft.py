import numpy as np


def compute_spectrogram(window):
    """
    Iz enega okna naredi spektrogram.

    Args:
        window: numpy array (W, 3) — eno okno, 3 kanali

    Returns:
        numpy array (freq_bins, 3) — surove magnitude
    """
    window_result = []
    for axis in range(3):

        signal_1d = window[:, axis]

        hann = np.hanning(window.shape[0])
        signal_1d = signal_1d * hann

        freq_bins = np.fft.rfft(signal_1d)
        magnitude = np.abs(freq_bins)

        window_result.append(magnitude)

    return np.stack(window_result, axis=1)


def normalize_spectrogram(spec):
    """
    Normalizira spektrogram na vrednosti 0–255 z log skaliranjem.

    Args:
        spec: numpy array (freq_bins, 3) — surove magnitude

    Returns:
        numpy array (freq_bins, 3) dtype=uint8
    """

    S_log = 10 * np.log10(spec + 1e-10)

    s_min = np.min(S_log)
    s_max = np.max(S_log)

    if s_min == s_max:
        return np.zeros_like(spec, dtype=np.uint8)

    S_norm = (S_log - s_min) / (s_max - s_min) * 255

    S_out = np.clip(S_norm, 0, 255).astype(np.uint8)

    return S_out


def compute_spectrograms(windows):
    """
    Naredi spektrograme za vsa okna.

    Args:
        windows: numpy array (M, W, 3)

    Returns:
        numpy array (M, freq_bins, 3) dtype=uint8
    """
    result = []
    for window in windows:
        spectogram = compute_spectrogram(window)
        spectogram = normalize_spectrogram(spectogram)
        result.append(spectogram)

    return np.stack(result, axis=0)
