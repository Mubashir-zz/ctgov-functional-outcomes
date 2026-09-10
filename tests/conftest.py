"""Put the analysis package on the import path.

The scripts live in analysis/ and are run as standalone programs, so there is no package
to install; the tests import them directly and need the directory on sys.path.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
