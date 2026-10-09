"""Assumptions and constraints of the plasma cutter.

- Torch is orthogonal to the profile
- Constant cut width (kerf)
- Blade length as a function of cutting speed: L(v)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Blade length L(v)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BladeLengthModel:
    """Linear blade length L(v) = clip(L0 - slope * v, 0, L0) [mm].

    Parameters
    ----------
    L0    : blade length at v = 0 [mm]
    slope : shortening per unit speed [mm / (mm/s)]
    """
    L0: float = 29.9
    slope: float = 0.327

    def __call__(self, v: float) -> float:
        # Enforce v > 0: this fixes the L0 value in the parameter stamp (phys_hash).
        v = max(float(v), 1e-6)
        return float(np.clip(self.L0 - self.slope * v, 0.0, self.L0))


# ---------------------------------------------------------------------------
# Pierce time
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PierceTimeModel:
    """Flat pierce time (re-entry) per torch ignition [s].

    Parameters
    ----------
    t0     : minimum flat time per ignition [s] (initialization time)
    k      : time per mm of sheet thickness [s/mm]
    fixed  : if True, thickness is ignored (constant = t0)
    """
    t0: float = 1.0     # seconds
    k:  float = 0.0     # seconds per mm
    fixed: bool = False

    def __call__(self, thickness: float = 0.0) -> float:
        if self.fixed:
            return float(self.t0)
        return float(self.t0 + self.k * max(0.0, thickness))


# ---------------------------------------------------------------------------
# Combined configuration
# ---------------------------------------------------------------------------

@dataclass
class CuttingAssumptions:
    """Bundle of all assumptions + constraints.

    Passed to a ``Cutter`` instance. Changes to this bundle
    automatically apply to all computations (blade length,
    pierce time).
    """
    blade:  BladeLengthModel = field(default_factory=BladeLengthModel)
    pierce: PierceTimeModel  = field(default_factory=PierceTimeModel)

    # Material-related (for pierce + comparison with L)
    sheet_thickness: float = 15.0  # [mm]

    # Global switch
    use_pierce_penalty: bool = True

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def effective_blade_length(self, v: float) -> float:
        return self.blade(v)

    def pierce_time(self) -> float:
        """Flat pierce time (re-entry) [s]."""
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
