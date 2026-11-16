"""Entry point for the Windows executable (PyInstaller).

Double-click: opens the window. From a terminal: same commands as `mosaic2d`.
"""
import sys

from mosaic2d.cli import main

if __name__ == "__main__":
    sys.exit(main())
