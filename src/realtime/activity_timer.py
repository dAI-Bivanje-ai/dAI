import time
from collections import defaultdict


class ActivityTimer:
    """
    Beleži čas posamezne aktivnosti na podlagi stabiliziranih labelov.
    """

    def __init__(self) -> None:
        self.current_label: str | None = None
        self.last_change_time: float | None = None
        self.durations: dict[str, float] = defaultdict(float)

    def update(self, label: str | None) -> None:
        """
        Posodobi trenutno aktivnost.

        Če label ostane isti, ne naredi nič.
        Če se label spremeni, prišteje čas prejšnji aktivnosti.
        """
        if label is None:
            return

        now = time.time()

        if self.current_label is None:
            self.current_label = label
            self.last_change_time = now
            return

        if label == self.current_label:
            return

        elapsed = now - self.last_change_time

        self.durations[self.current_label] += elapsed

        self.current_label = label
        self.last_change_time = now

    def get_durations(self) -> dict[str, float]:
        """
        Vrne čase vseh aktivnosti

        Prišteje tudi trenutno aktivnost, ki še traja
        """

        result = dict(self.durations)

        if self.current_label is not None and self.last_change_time is not None:
            now = time.time()
            elapsed = now - self.last_change_time
            result[self.current_label] = result.get(self.current_label, 0.0) + elapsed

        return result

    def reset(self) -> None:
        """
        Ponastavi vse izmerjene čase
        """

        self.current_label = None
        self.last_change_time = None
        self.durations.clear()
