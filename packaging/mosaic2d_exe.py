"""Entry point for the Windows executable (PyInstaller).

Double-click: opens the window. From a terminal: same commands as `mosaic2d`.

Any start-up error is shown in a message box, written to mosaic2d_error.txt
(next to the exe, or in the temp folder if that is not writable) and the console
stays open, so a failure is never a window that just flashes and disappears.
"""
import os
import sys
import tempfile
import traceback

GUI_MODE = len(sys.argv) == 1


def _report(tb: str) -> None:
    text = "mosaic2d could not start.\n\n" + tb
    path = None
    for folder in (os.path.dirname(sys.executable), tempfile.gettempdir()):
        try:
            path = os.path.join(folder, "mosaic2d_error.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            break
        except OSError:
            path = None
    print(text, flush=True)
    if path:
        print(f"\nDetails saved to {path}", flush=True)
    if sys.platform.startswith("win"):
        try:
            import ctypes
            tail = "\n".join(tb.strip().splitlines()[-12:])
            ctypes.windll.user32.MessageBoxW(
                0, f"mosaic2d could not start.\n\n{tail}\n\nDetails: {path}", "mosaic2d", 0x10)
        except Exception:
            pass
    if GUI_MODE:
        try:
            input("\nPress Enter to close this window.")
        except Exception:
            pass


if __name__ == "__main__":
    if GUI_MODE:
        print("mosaic2d is starting; the window opens in a few seconds.\n"
              "(The first start can take up to a minute while Windows checks the files.)\n"
              "Keep this console open: it shows the progress log too.", flush=True)
    try:
        from mosaic2d.cli import main
        code = main()
    except SystemExit:
        raise
    except BaseException:
        _report(traceback.format_exc())
        sys.exit(1)
    sys.exit(code)
