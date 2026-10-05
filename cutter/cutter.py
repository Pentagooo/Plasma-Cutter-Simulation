"""Plasma-Schneidkopf: Geschwindigkeiten, Mindestabstand und Zeiten.

Klingenlänge L(v) und Pierce-Zeit kommen aus ``assumptions.py``.
"""
from __future__ import annotations

from .assumptions import CuttingAssumptions


class Cutter:
    """Plasma-Schneidkopf.

    Parameters
    ----------
    cutting_speed     : Schnittgeschwindigkeit v_cut [mm/s]
    max_cutting_speed : maximale Schnittgeschwindigkeit v_max [mm/s]
    rapid_speed       : Eilgang zwischen den Schnitten [mm/s]
    minimum_gap       : Mindestabstand TCP–Material [mm]
    t_switch          : Zeitaufschlag je Geschwindigkeitswechsel im
                        laufenden Schnitt [s]
    assumptions       : Klingenlänge L(v) und Pierce-Zeit
    """

    def __init__(
        self,
        cutting_speed: float = 5.0,
        max_cutting_speed: float | None = None,
        rapid_speed: float = 50.0,
        minimum_gap: float = 3.0,
        t_switch: float = 0.0,
        assumptions: CuttingAssumptions | None = None,
    ) -> None:
        if max_cutting_speed is not None and cutting_speed > max_cutting_speed:
            raise ValueError(
                f"cutting_speed = {cutting_speed} mm/s ueberschreitet "
                f"max_cutting_speed = {max_cutting_speed} mm/s."
            )
        self.cutting_speed = cutting_speed
        self.max_cutting_speed = max_cutting_speed
        self.rapid_speed = rapid_speed
        self.minimum_gap = minimum_gap
        self.t_switch = float(t_switch)
        self.assumptions = assumptions or CuttingAssumptions()

    def blade_length(self, v: float | None = None) -> float:
        """Klingenlänge L(v) bei Schneidgeschwindigkeit v
        (Standard: ``cutting_speed``)."""
        if v is None:
            v = self.cutting_speed
        return self.assumptions.effective_blade_length(v)

    def pierce_time(self) -> float:
        """Pauschalzeit pro Brennerzündung [s]."""
        return self.assumptions.pierce_time()

    def time_for_length(self, length: float, mode: str = "cut") -> float:
        """Zeit [s] für eine Strecke [mm]: 'cut' mit ``cutting_speed``,
        'rapid' mit ``rapid_speed``."""
        if mode == "cut":
            speed = self.cutting_speed
        elif mode == "rapid":
            speed = self.rapid_speed
        else:
            raise ValueError(f"Unknown mode '{mode}', use 'cut' or 'rapid'")
        return length / speed

    def __repr__(self) -> str:
        return (
            f"Cutter(L(v_cut)={self.blade_length():.1f} mm, "
            f"v_cut={self.cutting_speed} mm/s, "
            f"v_max={self.max_cutting_speed} mm/s, "
            f"v_rapid={self.rapid_speed} mm/s, "
            f"minimum_gap={self.minimum_gap} mm, "
            f"t_switch={self.t_switch} s, "
            f"pierce={self.pierce_time():.2f}s)"
        )
