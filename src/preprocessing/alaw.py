import math
import numpy as np


A = 87.7
A_INV = 1.0 / A
ONE_PLUS_LN_A = 1.0 + math.log(A)

sign = lambda x: math.copysign(1, x)


def alaw_decode(sample):
    """
    Dekodira en A-law vzorec v linearni PCM.

    Implementacija inverza A-law kompresijske formule (ITU-T G.711).
    Vhodni int8 vzorec normaliziramo na [-1, 1] pred dekodiranjem.

    Args:
        sample: int8 — A-law kodiran vzorec

    Returns:
        float — linearni PCM vzorec v območju [-1, 1]
    """
    # normaliziramo na območje -1 do 1
    sample /= 128
    abs_sample = math.fabs(sample)

    if abs_sample < 1 / ONE_PLUS_LN_A:
        y = abs_sample * ONE_PLUS_LN_A / A
    else:
        y = math.exp(abs_sample * ONE_PLUS_LN_A - 1) / A

    return sign(sample) * y


def alaw_decode_all(samples):
    """
    Dekodira cel array A-law vzorcev v linearni PCM.

    Args:
        samples: numpy array (N,) dtype=int8 — A-law kodirani vzorci

    Returns:
        numpy array (N,) dtype=float64 — linearni PCM vzorci v [-1, 1]
    """
    result = []
    for sample in samples:
        step = alaw_decode(sample)
        result.append(step)

    return np.array(result)
