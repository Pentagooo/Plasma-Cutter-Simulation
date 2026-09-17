from __future__ import annotations

"""Zentrales Modul für Randbedingungen und Modellannahmen des Plasmaschneiders.

================================================================
A) Annahmen
================================================================

A1  Konstante Leistung über alle Schnitte
A2  Brenner steht orthogonal zum Profil
A3  Schnittbreite (Kerf) konstant
A4  Klingenlänge L(v) als Funktion der Schnittgeschwindigkeit
A5  Kein Verschleiß von Düse/Elektrode

================================================================
B) Randbedingungen
================================================================

B1  Eintritt nur vom Außenrand  der Geometrie
B2  Klingenlänge L(v) als Funktion der Geschwindigkeit
B3  Wiedereintrittspauschale (Pierce-Time) pro Zündung
B4  Einzeldurchgang -- Schnitt nur von einer Seite
"""

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# A4 / B2  -- Klingenlänge L(v)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BladeLengthModel:
    """Modelliert die effektiv erreichbare Klingenlänge in Abhängigkeit
    der Schneidgeschwindigkeit v.

    Näherung 1 (Standard, konservativ):  L(v) = L_ref * v_ref / v
    -- folgt aus der Annahme, dass Energie pro Längeneinheit
       E_l = P / v
    -- Begrenzt nach oben durch L_max (mechanische Reichweite des
       Brenners) und nach unten durch L_min (sonst kein Schnitt).

    Näherung 2 (linear, optional): L(v) = L_max - k*(v - v_ref)
    -- empirische lineare Approximation um den Arbeitspunkt.

    Parameters
    ----------
    L_ref   : Klingenlänge bei Referenzgeschwindigkeit [mm]
    v_ref   : Referenzgeschwindigkeit [mm/s]
    L_max   : maximale Klingenlänge (z.B. Brennerhub) [mm]
    L_min   : minimale Schnitt-Tiefe (sonst kein Durchschnitt) [mm]
    mode    : "inverse" (Standard, ~1/v) oder "linear"
    slope   : nur für mode="linear": dL/dv (negativ) [mm * s / mm]
    """
    L_ref: float = 20.0
    v_ref: float = 5.0
    L_max: float = 40.0
    L_min: float = 1.0
    mode:  str   = "inverse"
    slope: float = -1.5   # nur für "linear"

    def __call__(self, v: float) -> float:
        v = max(float(v), 1e-6)
        if self.mode == "inverse":
            L = self.L_ref * self.v_ref / v
        elif self.mode == "linear":
            L = self.L_ref + self.slope * (v - self.v_ref)
        else:
            raise ValueError(f"unknown BladeLengthModel.mode '{self.mode}'")
        return float(np.clip(L, self.L_min, self.L_max))


# ---------------------------------------------------------------------------
# B3  -- Wiedereintrittspauschale (Pierce-Time)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PierceTimeModel:
    """Wiedereintrittspauschale je Zündung [s].

    Der Brenner braucht nach jeder Zündung eine Pausen-Zeit, in der
    das Material durchstoßen wird.  Mit der Blechstärke und der
    Stromstörke wächst diese Zeit

    Modell:  t_pierce(thickness) = t0 + k * thickness
             (linear, über Hypertherm-Cut-Charts angepasst)

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

    # Globale Schalter
    use_velocity_dependent_blade: bool = True
    use_pierce_penalty:           bool = True

    # ------------------------------------------------------------------
    # Komfort-Methoden
    # ------------------------------------------------------------------

    def effective_blade_length(self, v: float, L_default: float) -> float:
        """Gibt die effektive Klingenlänge zurück (mit oder ohne
        Geschwindigkeitsabhängigkeit)."""
        if not self.use_velocity_dependent_blade:
            return L_default
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
            f"  blade-model={'L(v)' if self.use_velocity_dependent_blade else 'const'} "
            f"(L_ref={self.blade.L_ref}, v_ref={self.blade.v_ref}, "
            f"L_max={self.blade.L_max})\n"
            f"  pierce={'ON' if self.use_pierce_penalty else 'OFF'} "
            f"(t0={self.pierce.t0}s, k={self.pierce.k}s/mm)\n"
            f")"
        )
