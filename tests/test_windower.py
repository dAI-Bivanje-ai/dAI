import numpy as np

from src.preprocessing.windower import window_signal, window_signal_seconds


def make_signal(n):
    """
    Naredi testni signal oblike (n, 3), kjer je vsaka vrstica [i, i, i].

    Tako lahko v testih enostavno preverimo, katero okno pokriva katere vzorce.
    """
    return np.arange(n).repeat(3).reshape(n, 3)


def test_window_signal_stevilo_in_oblika_oken():
    signal = make_signal(10)

    windows = window_signal(signal, window_size=4, step=2)

    # floor((10 - 4) / 2 + 1) = 4 okna, vsako 4 vzorce, 3 osi
    assert windows.shape == (4, 4, 3)


def test_window_signal_pravilno_rezanje():
    signal = make_signal(10)

    windows = window_signal(signal, window_size=4, step=2)

    # Prvo okno se zacne pri vzorcu 0, drugo pri 2 (korak = 2).
    assert windows[0][0][0] == 0
    assert windows[1][0][0] == 2


def test_window_signal_brez_prekrivanja():
    signal = make_signal(10)

    # korak == dolzina okna -> okna se ne prekrivajo
    windows = window_signal(signal, window_size=5, step=5)

    assert windows.shape == (2, 5, 3)
    assert windows[0][0][0] == 0
    assert windows[1][0][0] == 5


def test_window_signal_seconds_izracuna_dolzino_okna():
    # 100 Hz, okno 2 s  okno mora biti dolgo 200 vzorcev
    signal = make_signal(400)

    windows = window_signal_seconds(signal, Fvz=100, T_window=2.0, prekrivanje=0.5)

    assert windows.shape[1] == 200
