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
        return np.zeros_like(spec,dtype=np.uint8)
    
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
        result.append(spectogram)
    
    return np.stack(result,axis=0)    

def group_spectrograms(spectrograms, segment_length):
    """
    Združi zaporedne spektrograme v 2D spektrograme primerne za CNN.

    Iz zaporednih 1D spektrogramov oblike (freq_bins, 3) sestavi
    2D spektrograme oblike (freq_bins, segment_length, 3), kjer
    x os predstavlja čas, y os pa frekvenco.

    Args:
        spectrograms: numpy array (M, freq_bins, 3) — izhod compute_spectrograms()
        segment_length: int — število zaporednih spektrogramov v enem 2D spektrogramu

    Returns:
        numpy array (N, freq_bins, segment_length, 3) dtype=uint8
        kjer je N = M // segment_length
    """
    segment = []
    final_spectrogram = []
    
    for spectrogram in spectrograms:
        segment.append(spectrogram)
        if len(segment) == segment_length:
            stacked = np.stack(segment,axis=1)
            final_spectrogram.append(stacked)
            segment = []
            
    return np.array(final_spectrogram)