import numpy as np

from src.preprocessing.alaw import alaw_decode, alaw_decode_all


def test_alaw_decode_nicla():
    assert alaw_decode(0) == 0.0


def test_alaw_decode_je_v_obmocju():
    # Dekodirana vrednost mora vedno ostati v obmocju [-1, 1].
    for sample in [-128, -64, -1, 0, 1, 64, 127]:
        decoded = alaw_decode(sample)
        assert -1.0 <= decoded <= 1.0


def test_alaw_decode_je_simetricen():
    # Pozitiven in negativen vzorec morata dati nasprotno predznacen rezultat.
    assert alaw_decode(64) == -alaw_decode(-64)


def test_alaw_decode_je_monoton():
    # Vecji vhod -> vecja dekodirana vrednost.
    assert alaw_decode(10) < alaw_decode(50) < alaw_decode(120)


def test_alaw_decode_all_vrne_array_prave_dolzine():
    samples = np.array([-128, -1, 0, 1, 127], dtype=np.int8)

    decoded = alaw_decode_all(samples)

    assert isinstance(decoded, np.ndarray)
    assert decoded.shape == (5,)
    assert np.all(decoded >= -1.0)
    assert np.all(decoded <= 1.0)
