import shutil
import subprocess
import sys


def notify_macos(title: str, message: str):

    osascript = shutil.which("osascript")
    if osascript is None:
        raise RuntimeError("osascript ni na voljo")

    script = f'display notification "{message}" with title "{title}"'
    subprocess.run([osascript, "-e", script], check=True)


if __name__ == "__main__":
    pass
