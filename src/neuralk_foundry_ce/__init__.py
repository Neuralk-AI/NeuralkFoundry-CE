"""Neuralk Foundry Community Edition"""

from pathlib import Path

VERSION_PATH = Path(__file__).resolve().parent / "VERSION.txt"
__version__ = VERSION_PATH.read_text(encoding="utf-8").strip()
