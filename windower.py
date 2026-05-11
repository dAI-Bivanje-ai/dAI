import numpy as np


def window_signal(signal, window_size, step):
    """
    Razreže signal na okna.

    Args:
        signal: numpy array (N, 3)
        W:      dolžina okna v vzorcih
        S:      korak med okni v vzorcih

    Returns:
        numpy array (M, W, 3)
    """
    
    signal_length = signal.shape[0]
    number_windows = int(np.floor(((signal_length - window_size) / step) + 1))
    windows = []
    
    for i in range(number_windows):
        start = i * step
        end = start + window_size
        windows.append(signal[start:end])
    
    return np.array(windows)


def window_signal_seconds(signal, Fvz, T_window=2.0, prekrivanje=0.5):
    """
    Ovojnica ki sprejme sekunde namesto vzorcev.
    Izračuna W in S ter pokliče window_signal().
    """
    W = T_window * Fvz
    S = W * (1 - prekrivanje)
    
    W = round(W)
    S = round(S)
    return window_signal(signal,W,S)
        