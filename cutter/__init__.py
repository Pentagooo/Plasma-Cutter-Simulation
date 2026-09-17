"""Schneider-Paket (Cutter-Modell + physikalische Annahmen).

Re-exportiert die Symbole, die die Segment-Simulation nutzt: das
``Cutter``-Modell (Geschwindigkeiten, Zeiten, Klingenlänge) und die
``assumptions`` (Klingenlängen-Modell L(v), Pierce-Zeit).
"""

from .cutter import Cutter
from .assumptions import (
    CuttingAssumptions,
    BladeLengthModel,
    PierceTimeModel,
)

__all__ = [
    "Cutter",
    "CuttingAssumptions",
    "BladeLengthModel",
    "PierceTimeModel",
]
