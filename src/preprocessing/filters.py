from scipy import signal


def bandpass_mic(signal, fvz=8000.0, low_hz=80.0, filter_order=4):
    """
    Band-pass Butterworth filter za mikrofon signal.

    Odstrani:
    - pod 80 Hz: vibracije telesa, dotiki z mizo -> neuporabno za nas
    - nad 3500 Hz: elektronski šum senzorja -> neuporabno za nas

    Ohrani:
    - 80-3500 Hz: govor in koristni zvok

    Uporablja sosfiltfilt (zero-phase, numerično stabilen).

    Args:
        signal: numpy array (N,) — linearni PCM signal (po alaw_decode)
        fvz: vzorčevalna frekvenca [Hz], privzeto 8000.0
        low_hz: spodnja mejna frekvenca [Hz], privzeto 80.0
        high_hz: zgornja mejna frekvenca [Hz], privzeto 3500.0
        filter_order: red Butterworth filtra, privzeto 4

    Returns:
        numpy array (N,) — filtriran signal
    """
    high_hz = fvz // 2
    Wn = [low_hz, high_hz]
    sos = signal.butter(filter_order, Wn, "bandpass", output="sos", fs=fvz)
    return signal.sosfiltfilt(sos, signal)
