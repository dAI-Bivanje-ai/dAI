import shutil
import subprocess
import sys


def notify_macos(title: str, message: str) -> None:

    osascript = shutil.which("osascript")
    if osascript is None:
        raise RuntimeError("osascript ni na voljo")

    script = f'display notification "{message}" with title "{title}"'
    subprocess.run([osascript, "-e", script], check=True)


def notify_regular(title: str, message: str, timeout: int) -> None:
    from plyer import notification

    notification.notify(title=title, message=message, timeout=timeout)  # type: ignore


if __name__ == "__main__":
    pass
