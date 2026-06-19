import pywinctl


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


if __name__ == "__main__":
    app_name, title = get_active_window()
    print(app_name, title)
