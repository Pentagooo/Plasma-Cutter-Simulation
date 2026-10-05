"""Annahmen und Randbedingungen des Plasmaschneiders.

- Brenner steht orthogonal zum Profil
- Schnittbreite (Kerf) konstant
- Klingenlänge als Funktion der Schnittgeschwindigkeit: L(v)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Klingenlänge L(v)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BladeLengthModel:
    """Lineare Klingenlänge L(v) = clip(L0 - slope * v, 0, L0) [mm].

    Parameters
    ----------
    L0    : Klingenlänge bei v = 0 [mm]
    slope : Verkürzung pro Geschwindigkeit [mm / (mm/s)]
    """
    L0: float = 29.9
    slope: float = 0.327

    def __call__(self, v: float) -> float:
        # v > 0 erzwingen: so ist L0 im Parameterstempel (phys_hash) bestimmt.
        v = max(float(v), 1e-6)
        return float(np.clip(self.L0 - self.slope * v, 0.0, self.L0))


# ---------------------------------------------------------------------------
# Pierce-Zeit
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PierceTimeModel:
    """Wiedereintrittspauschale je Zündung [s].

    Parameters
    ----------
    t0     : Mindestpauschale je Zündung [s] (Initialisierungszeit)
    k      : Zeit pro mm Blechstärke [s/mm]
    fixed  : wenn True wird thickness ignoriert (Konstante = t0)
    """
    t0: float = 0.3     # Sekunden
    k:  float = 0.05    # Sekunden pro mm
    fixed: bool = False

    def __call__(self, thickness: float = 0.0) -> float:
        if self.fixed:
            return float(self.t0)
        return float(self.t0 + self.k * max(0.0, thickness))


# ---------------------------------------------------------------------------
# Kombi-Konfiguration
# ---------------------------------------------------------------------------

@dataclass
class CuttingAssumptions:
    """Bundle aller Annahmen + Randbedingungen.

    Wird einer ``Cutter``-Instanz übergeben. Änderungen an diesem Bundle
    wirken sich automatisch auf alle Berechnungen aus (Klingenlänge,
    Pierce-Zeit).
    """
    blade:  BladeLengthModel = field(default_factory=BladeLengthModel)
    pierce: PierceTimeModel  = field(default_factory=PierceTimeModel)

    # Materialbezogen (für Pierce + Vergleich mit L)
    sheet_thickness: float = 12.0  # [mm]

    # Globaler Schalter
    use_pierce_penalty: bool = True

    # ------------------------------------------------------------------
    # Komfort-Methoden
    # ------------------------------------------------------------------

    def effective_blade_length(self, v: float) -> float:
        return self.blade(v)

    def pierce_time(self) -> float:
        """Wiedereintrittspauschale [s]."""
        if not self.use_pierce_penalty:
            return 0.0
        return self.pierce(self.sheet_thickness)

    def summary(self) -> str:
        return (
            f"CuttingAssumptions(\n"
            f"  thickness={self.sheet_thickness:.1f} mm\n"
            f"  blade L(v)=clip({self.blade.L0} - {self.blade.slope}*v, 0, {self.blade.L0})\n"
            f"  pierce={'ON' if self.use_pierce_penalty else 'OFF'} "
            f"(t0={self.pierce.t0}s, k={self.pierce.k}s/mm)\n"
            f")"
        )
