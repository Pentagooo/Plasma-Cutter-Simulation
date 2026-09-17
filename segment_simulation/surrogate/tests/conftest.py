"""Pytest-Konfiguration: Repo-Wurzel in den Importpfad legen.

So laufen die Tests unabhaengig vom Arbeitsverzeichnis (``import
plasma_cutter...`` funktioniert immer).
"""
import sys
from pathlib import Path

# .../plasma_cutter/segment_simulation/surrogate/tests -> parents[4] = IGMR
_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: Brute-Force-Lehrer auf mehreren Instanzen (Minuten)")
