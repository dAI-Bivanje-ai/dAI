"""
Stabilizacija realtime napovedi modela.

Modul vsebuje razred PredictionStabilizer, ki z drsečim oknom zadnjih N
napovedi prepreči utripanje rezultata in potrdi nov razred šele, ko je
dovolj zanesljiv.
"""

from collections import Counter, deque


class PredictionStabilizer:
    """
    Stabilizira realtime napovedi modela.

    Hrani zadnjih N napovedi in potrdi razred šele,
    ko dovolj velik delež napovedi kaže na isti razred.
    """

    def __init__(self, window_size: int, min_ratio: float) -> None:
        """
        Inicializira stabilizator.

        Args:
            window_size: int — velikost drsečega okna zadnjih napovedi
            min_ratio: float — najmanjši delež enakih napovedi za potrditev
        """
        self.window: deque[str] = deque(maxlen=window_size)
        self.min_ratio = min_ratio
        self.last_confirmed: str | None = None

    def update(self, label: str | None) -> str | None:
        """
        Doda novo napoved in vrne stabilen razred, če je dovolj zanesljiv.

        Vrne:
        - None, če še ni dovolj napovedi ali stanje ni stabilno,
        - ime razreda, če je potrjen nov stabilen razred.
        """
        if label is None:
            return None

        self.window.append(label)

        if len(self.window) < self.window.maxlen:
            return None

        label_counts = Counter(self.window)
        most_common_label, count = label_counts.most_common(1)[0]

        if count / len(self.window) < self.min_ratio:
            return None

        self.last_confirmed = most_common_label
        return most_common_label
