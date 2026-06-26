"""Schneider-Paket (Cutter-Modell + physikalische Annahmen).

Was hier passiert
-----------------
Macht das ``cutter``-Paket importierbar und re-exportiert die Symbole,
die die Segment-Simulation tatsächlich nutzt: das ``Cutter``-Modell
(Geschwindigkeiten, Zeiten, Klingenlänge) und die ``assumptions``
(physikalische Annahmen wie Klingenlängen-Modell, Pierce-Zeit,
Brennerneigung, Presets).

from .cutter import Cutter
from .assumptions import (
    CuttingAssumptions,
    BladeLengthModel,
    PierceTimeModel,
    TorchTiltDistribution,
    preset_ideal,
    preset_realistic,
    preset_noisy,
)

__all__ = [
    "Cutter",
    "CuttingAssumptions",
    "BladeLengthModel",
    "PierceTimeModel",
    "TorchTiltDistribution",
    "preset_ideal",
    "preset_realistic",
    "preset_noisy",
]