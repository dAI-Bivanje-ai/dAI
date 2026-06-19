import pywinctl
import time

def get_active_window():
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


PRODUKTIVNE = {
    "kitty",
    "ghostty",
    "pycharm",
    "PyCharm",
    "Code",
}

NEPRODUKTIVNE = {
    "Discord",
    "firefox",
    "Firefox",
    "netflix",
    "Spotify",
}


class ActivityViewer:

    def __init__(self):
        self.last_active_app_name = None
        self.last_active_title = None
        self.interval = 1

    def categorize(self, app_name):
        if app_name in PRODUKTIVNE:
            return 0
        if app_name in NEPRODUKTIVNE:
            return 1
        return None

    def run(self):

        while True:
            app_name, title = get_active_window()

            if app_name != self.last_active_app_name:
                category = self.categorize(app_name)
                label = {0: "PRODUKTIVNE", 1: "NEPRODUKTIVNE"}.get(category, "NEZNANO")
                print(f"{app_name} -> {label}")

                self.last_active_app_name = app_name
                self.last_active_title = title

            time.sleep(self.interval)




if __name__ == "__main__":
    ActivityViewer().run()
