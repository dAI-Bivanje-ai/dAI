"""
Zagonska tocka aplikacije dAI.

Ta datoteka samo zazene realtime GUI (src/realtime/gui_rt.py). Obstaja zato,
da lahko PyInstaller zapakira aplikacijo iz korena projekta, kjer so dostopni
vsi moduli iz paketa 'src'.
"""

from src.realtime.gui_rt import run

if __name__ == "__main__":
    run()
