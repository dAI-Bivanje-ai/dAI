"""
Spremljanje trenutno aktivnega okna v realtime sistemu.

Modul s pomočjo knjižnice pywinctl ugotovi, katera aplikacija je trenutno
v ospredju, in jo uvrsti med produktivne ali neproduktivne. Razred
ActivityViewer ob vsaki menjavi aktivne aplikacije sproti vrne dogodek z
imenom aplikacije, naslovom okna in pripadajočo oznako.
"""

import pywinctl
import time


def get_active_window():
    """
    Vrne ime aplikacije in naslov trenutno aktivnega okna.

    Če aktivnega okna ni, vrne None za obe vrednosti. Morebitne napake pri
    branju lastnosti okna se izpišejo, vrednost pa ostane None.

    Returns:
        tuple:
            (ime aplikacije, naslov okna), oba lahko None.
    """
    app_name = None
    title = None

    window = pywinctl.getActiveWindow()

    if window is not None:
        try:
            app_name = window.getAppName()
        except Exception as e:
            print(str(e))

        try:
            title = window.title
        except Exception as e:
            print(str(e))

    return app_name, title


# Aplikacije, ki se štejejo za produktivno delo
PRODUKTIVNE = {
    "kitty",
    "ghostty",
    "pycharm",
    "PyCharm",
    "Code",
}

# Aplikacije, ki se štejejo za neproduktivno rabo
NEPRODUKTIVNE = {
    "Discord",
    "firefox",
    "Firefox",
    "netflix",
    "Spotify",
}


# Preslikava številčne kategorije v berljivo oznako
CATEGORY_LABELS = {0: "PRODUKTIVNE", 1: "NEPRODUKTIVNE"}


class ActivityViewer:
    """
    Spremlja aktivno okno in ga uvršča med produktivne ali neproduktivne.
    """

    def __init__(self):
        """
        Inicializira spremljanje brez zadnje znane aktivne aplikacije.
        """
        # Zadnja zaznana aktivna aplikacija in naslov okna
        self.last_active_app_name = None
        self.last_active_title = None
        # Razmik med dvema preverjanjema aktivnega okna (v sekundah)
        self.interval = 1

    def categorize(self, app_name):
        """
        Uvrsti aplikacijo v kategorijo.

        Parameters:
            app_name (str):
                Ime aplikacije.

        Returns:
            int:
                0 za produktivne, 1 za neproduktivne, None če ni znana.
        """
        if app_name in PRODUKTIVNE:
            return 0
        if app_name in NEPRODUKTIVNE:
            return 1
        return None

    def label_category(self, app_name):
        """
        Vrne berljivo oznako kategorije za podano aplikacijo.

        Parameters:
            app_name (str):
                Ime aplikacije.

        Returns:
            str:
                "PRODUKTIVNE", "NEPRODUKTIVNE" ali "NEZNANO".
        """
        category = self.categorize(app_name)
        if category is None:
            return "NEZNANO"
        return CATEGORY_LABELS[category]

    def run(self):
        """
        Sproti spremlja aktivno okno in vrača dogodke ob vsaki menjavi.

        V neskončni zanki preverja trenutno aktivno aplikacijo. Ko se ta
        spremeni glede na prejšnjo, vrne (yield) slovar z imenom aplikacije,
        naslovom okna in oznako kategorije.

        Returns:
            dict:
                Dogodek z ključi "app", "title" in "label".
        """
        while True:
            app_name, title = get_active_window()

            # Dogodek vrnemo samo ob dejanski menjavi aktivne aplikacije
            if app_name != self.last_active_app_name:
                self.last_active_app_name = app_name
                self.last_active_title = title

                yield {
                    "app": app_name,
                    "title": title,
                    "label": self.label_category(app_name),
                }

            time.sleep(self.interval)


if __name__ == "__main__":
    for event in ActivityViewer().run():
        print(f"{event['app']} -> {event['label']}")
