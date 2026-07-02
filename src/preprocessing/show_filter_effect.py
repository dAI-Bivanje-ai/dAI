"""
Skripta za vizualizacijo učinka band-pass filtra na mikrofonske podatke.

Prebere en .bin posnetek, dekodira A-law mikrofonski signal, ga filtrira
z band-pass filtrom in primerja original s filtriranim signalom v časovni
domeni, v spektru moči in na STFT spektrogramu. Rezultate shrani kot sliki.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Korenski direktorij projekta (dve mapi nad to datoteko).
ROOT_DIR = Path(__file__).resolve().parents[2]

from src.data_logger.data_logger import DataLogger
from src.visualization.data_visualizer import pripravi_pakete, sestavi_podatke_mic
from src.preprocessing.alaw import alaw_decode_all
from src.preprocessing.filters import bandpass_mic
from src.preprocessing.windower import window_signal_seconds
from src.preprocessing.stft import compute_spectrograms_1d

# Nastavitve: vhodna datoteka, vzorčna frekvenca mikrofona in parametri STFT.
BIN_FILE = ROOT_DIR / "podatki" / "mic_podatki" / "pogovor_05.bin"
MIC_FVZ = 8000.0
FFT_W = 256
FFT_OVERLAP = 0.5

# Časovni korak med zaporednimi STFT okni (v sekundah).
TIME_STEP = (FFT_W * (1 - FFT_OVERLAP)) / MIC_FVZ

# Preberi binarno datoteko in jo razparsiraj v pakete.
logger = DataLogger()
raw_packets = logger.parse_file(str(BIN_FILE))
paketi = pripravi_pakete(raw_packets)

# Sestavi mikrofonski signal iz paketov, ga A-law dekodiraj in filtriraj.
fvz_mic, mic_raw = sestavi_podatke_mic(paketi)
original = alaw_decode_all(mic_raw)
filtered = bandpass_mic(original, fvz=fvz_mic)

# Časovna os za prikaz signala v sekundah.
t = np.arange(len(original)) / fvz_mic

# Izračunaj spekter moči (FFT) za original in filtriran signal v decibelih.
N = len(original)
freqs = np.fft.rfftfreq(N, d=1.0 / fvz_mic)
power_orig = 20 * np.log10(np.abs(np.fft.rfft(original)) + 1e-10)
power_filt = 20 * np.log10(np.abs(np.fft.rfft(filtered)) + 1e-10)

# Razdeli oba signala na prekrivajoča se okna za STFT.
windows_orig = window_signal_seconds(
    original, fvz_mic, T_window=0.032, prekrivanje=FFT_OVERLAP
)
windows_filt = window_signal_seconds(
    filtered, fvz_mic, T_window=0.032, prekrivanje=FFT_OVERLAP
)

# Izračunaj spektrograma in ju pretvori v decibelsko skalo.
S_orig = compute_spectrograms_1d(windows_orig)
S_filt = compute_spectrograms_1d(windows_filt)
S_orig_db = 10 * np.log10(S_orig.T + 1e-10)
S_filt_db = 10 * np.log10(S_filt.T + 1e-10)

# Obseg osi za spektrogram: čas na x osi, frekvenca na y osi.
extent = (0, S_orig.shape[0] * TIME_STEP, 0, fvz_mic / 2)

# Slika 1: časovna domena in spekter moči.
fig1, axes1 = plt.subplots(2, 2, figsize=(16, 8))
fig1.suptitle("Učinek band-pass filtra (80–3500 Hz)", fontsize=14)

# Zgoraj levo: originalni signal v časovni domeni.
axes1[0, 0].plot(t, original, color="gray", linewidth=0.5)
axes1[0, 0].set_title("Originalni signal")
axes1[0, 0].set_xlabel("Čas [s]")
axes1[0, 0].set_ylabel("Amplituda")

# Zgoraj desno: filtriran signal v časovni domeni.
axes1[0, 1].plot(t, filtered, color="steelblue", linewidth=0.5)
axes1[0, 1].set_title("Filtriran signal")
axes1[0, 1].set_xlabel("Čas [s]")
axes1[0, 1].set_ylabel("Amplituda")

# Spodaj levo: spekter moči obeh signalov z označenima mejama filtra.
ax_spec = axes1[1, 0]
ax_spec.semilogx(freqs[1:], power_orig[1:], color="gray", label="Original")
ax_spec.semilogx(freqs[1:], power_filt[1:], color="steelblue", label="Filtriran")
ax_spec.axvline(80, color="red", linestyle="--", linewidth=1, label="80 Hz")
ax_spec.axvline(3500, color="red", linestyle="--", linewidth=1, label="3500 Hz")
ax_spec.set_xlabel("Frekvenca [Hz]")
ax_spec.set_ylabel("Moč [dB]")
ax_spec.set_title("Spekter moči")
ax_spec.legend(loc="upper right")
ax_spec.set_xlim(10, fvz_mic / 2)

# Spodaj desno: prazno, prostor namenoma skrit.
axes1[1, 1].axis("off")

# Shrani prvo sliko v mapo models.
fig1.tight_layout()
out_path1 = ROOT_DIR / "models" / "filter_effect.png"
out_path1.parent.mkdir(exist_ok=True)
fig1.savefig(out_path1, dpi=150)
print(f"Shranjeno: {out_path1}")

# Slika 2: STFT spektrograma pred filtriranjem in po njem.
# Skupna meja barvne skale, da sta spektrograma neposredno primerljiva.
vmin = min(S_orig_db.min(), S_filt_db.min())
vmax = max(S_orig_db.max(), S_filt_db.max())

fig2, axes2 = plt.subplots(1, 2, figsize=(16, 5))
fig2.suptitle("STFT spektrogram pred filtriranjem in po njem", fontsize=14)

# Levo: spektrogram originalnega signala.
im1 = axes2[0].imshow(
    S_orig_db,
    origin="lower",
    aspect="auto",
    extent=extent,
    cmap="inferno",
    vmin=vmin,
    vmax=vmax,
)
axes2[0].set_title("Spektrogram — original")
axes2[0].set_xlabel("Čas [s]")
axes2[0].set_ylabel("Frekvenca [Hz]")
fig2.colorbar(im1, ax=axes2[0], label="Moč [dB]")

# Desno: spektrogram filtriranega signala.
im2 = axes2[1].imshow(
    S_filt_db,
    origin="lower",
    aspect="auto",
    extent=extent,
    cmap="inferno",
    vmin=vmin,
    vmax=vmax,
)
axes2[1].set_title("Spektrogram — filtriran")
axes2[1].set_xlabel("Čas [s]")
axes2[1].set_ylabel("Frekvenca [Hz]")
fig2.colorbar(im2, ax=axes2[1], label="Moč [dB]")

# Shrani drugo sliko in prikaži obe okni.
fig2.tight_layout()
out_path2 = ROOT_DIR / "models" / "filter_effect_spectrograms.png"
fig2.savefig(out_path2, dpi=150)
print(f"Shranjeno: {out_path2}")

plt.show()
