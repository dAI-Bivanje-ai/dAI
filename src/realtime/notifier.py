"""
Sistemske obvestilne notifikacije za realtime sistem.

Modul prikaže namizno obvestilo glede na operacijski sistem: na macOS prek
osascript, na drugih platformah pa prek knjižnice plyer. Napake pri prikazu
se ne širijo naprej, da ne ustavijo glavne zanke.
"""

import shutil
import subprocess
import sys


def notify_macos(title: str, message: str) -> None:
    """
    Prikaže sistemsko notifikacijo na macOS prek osascript.

    Args:
        title: str — naslov obvestila
        message: str — vsebina obvestila

    Raises:
        RuntimeError — če osascript ni na voljo
    """
    osascript = shutil.which("osascript")
    if osascript is None:
        raise RuntimeError("osascript ni na voljo")

    script = f'display notification "{message}" with title "{title}"'
    subprocess.run([osascript, "-e", script], check=True)


def notify_regular(title: str, message: str, timeout: int) -> None:
    """
    Prikaže sistemsko notifikacijo na ne-macOS platformah prek plyer.

    Args:
        title: str — naslov obvestila
        message: str — vsebina obvestila
        timeout: int — čas prikaza v sekundah
    """
    from plyer import notification

    notification.notify(title=title, message=message, timeout=timeout)  # type: ignore


def notify(title: str, message: str, timeout: int = 5) -> None:
    """
    Prikaže sistemsko notifikacijo glede na operacijski sistem.

    Na macOS uporabi osascript, drugod plyer. Morebitne napake prestreže in
    izpiše, da ne ustavi delovanja programa.

    Args:
        title: str — naslov obvestila
        message: str — vsebina obvestila
        timeout: int — čas prikaza v sekundah (privzeto 5)
    """
    try:
        if sys.platform == "darwin":
            notify_macos(title, message)
        else:
            notify_regular(title, message, timeout)
    except Exception as e:
        print(str(e))


if __name__ == "__main__":
    notify("test", "test")
