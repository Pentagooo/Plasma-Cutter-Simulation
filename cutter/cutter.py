"""Plasma cutting head: speeds, minimum gap and times.

Blade length L(v) and pierce time come from ``assumptions.py``.
"""
from __future__ import annotations

from .assumptions import CuttingAssumptions


class Cutter:
    """Plasma cutting head.

    Parameters
    ----------
    cutting_speed     : cutting speed v_cut [mm/s]
    max_cutting_speed : maximum cutting speed v_max [mm/s]
    rapid_speed       : rapid traverse between cuts [mm/s]
    minimum_gap       : minimum gap between TCP and material [mm]
    t_switch          : time penalty per speed change during a
                        cut [s]
    assumptions       : blade length L(v) and pierce time
    """

    def __init__(
        self,
        cutting_speed: float = 19.4,
        max_cutting_speed: float | None = 34.7,
        rapid_speed: float = 100.0,
        minimum_gap: float = 3.6,
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
        """Blade length L(v) at cutting speed v
        (default: ``cutting_speed``)."""
        if v is None:
            v = self.cutting_speed
        return self.assumptions.effective_blade_length(v)

    def pierce_time(self) -> float:
        """Flat pierce time per torch ignition [s]."""
        return self.assumptions.pierce_time()

    def time_for_length(self, length: float, mode: str = "cut") -> float:
        """Time [s] for a distance [mm]: 'cut' with ``cutting_speed``,
        'rapid' with ``rapid_speed``."""
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
