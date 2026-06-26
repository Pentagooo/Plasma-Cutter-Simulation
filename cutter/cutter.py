"""Physikalisches Modell des Plasma-Schneidkopfes (Cutter).

Was hier passiert
-----------------
Der ``Cutter`` bündelt alle Eigenschaften des Brenners, die die
Simulation für  Zeit- und Reichweiten-Rechnungen braucht:
Geschwindigkeiten (Schneiden, Verfahren, Eilgang), die effektive
Klingen-/Eindringtiefe und die Zündpauschale (Pierce).

Wie es umgesetzt ist
--------------------
Die reinen Geschwindigkeits-/Geometrie-Werte stehen als Attribute hier;
die *physikalisch modellierten* Größen (geschwindigkeitsabhängige
Klingenlänge L(v), Pierce-Zeit, Brennerneigung) sind
in ein ``CuttingAssumptions``-Bundle (siehe ``assumptions.py``)
ausgelagert und werden von dort abgefragt. So bleibt der Cutter selbst
schlank und die Modell-Annahmen sind an einer Stelle austauschbar
(Presets: ideal / realistisch / verrauscht).

Die Segment-Simulation setzt zusätzlich ``max_cutting_speed``, weil
dort die Klingenlänge L(v) von der Schneidgeschwindigkeit abhängt.
"""
from __future__ import annotations

import numpy as np

from .assumptions import CuttingAssumptions


class Cutter:
    """Physikalisches Modell des Plasma-Schneidkopfes.

    Was es liefert: Geschwindigkeiten, effektive Klingenlänge L(v),
    erreichbare Tiefe und Schnitt-/Verfahr-/Pierce-Zeiten.

    Physical model of the plasma cutting head.

    Parameters
    ----------
    max_depth      : maximum penetration depth perpendicular to the cutter [mm].
    max_tilt       : maximum tilt angle from the surface normal [rad].
    cutting_speed  : speed while cutting through material (plasma on) [mm/s].
    max_cutting_speed : upper limit for cutting_speed [mm/s]. None = unbegrenzt
                     (rückwärtskompatibel). Wird von der Segment-Simulation
                     gesetzt, da dort L(v) von der Geschwindigkeit abhängt.
    moving_speed   : speed while traversing near geometry without cutting (plasma off) [mm/s].
    rapid_speed    : repositioning speed through free air  [mm/s].
    minimum_gap    : minimum distance between TCP and material surface [mm]. dmin
    assumptions    : Bundle modellbasierter Annahmen (L(v), Pierce, Tilt).
                     Siehe ``assumptions.py``. Wenn None wird das
                     Realitätsnähe-Preset verwendet.
    """

    def __init__(
        self,
        max_depth: float = 20.0,
        max_tilt: float = np.radians(20),
        cutting_speed: float = 5.0,
        max_cutting_speed: float | None = None,
        moving_speed: float = 20.0,
        rapid_speed: float = 50.0,
        minimum_gap: float = 3.0,
        assumptions: CuttingAssumptions | None = None,
    ) -> None:
        if max_cutting_speed is not None and cutting_speed > max_cutting_speed:
            raise ValueError(
                f"cutting_speed = {cutting_speed} mm/s ueberschreitet "
                f"max_cutting_speed = {max_cutting_speed} mm/s."
            )
        self.max_depth = max_depth
        self.max_tilt = max_tilt
        self.cutting_speed = cutting_speed
        self.max_cutting_speed = max_cutting_speed
        self.moving_speed = moving_speed
        self.rapid_speed = rapid_speed
        self.minimum_gap = minimum_gap
        self.assumptions = assumptions or CuttingAssumptions()

    # ------------------------------------------------------------------
    # B2: L(v) -- geschwindigkeitsabhängige Klingenlänge
    # ------------------------------------------------------------------

    def blade_length(self, v: float | None = None) -> float:
        """Effektive Klingenlänge bei Schneidgeschwindigkeit v.

        Wenn ``assumptions.use_velocity_dependent_blade`` aktiv ist,
        wird das L(v)-Modell (siehe assumptions.BladeLengthModel)
        ausgewertet. Sonst wird ``max_depth`` zurückgegeben.
        """
        if v is None:
            v = self.cutting_speed
        return self.assumptions.effective_blade_length(v, self.max_depth)

    # ------------------------------------------------------------------
    # B3: Wiedereintrittspauschale
    # ------------------------------------------------------------------

    def pierce_time(self) -> float:
        """Pauschalzeit pro Brennerzündung [s].

        Bezieht sich auf die Materialdicke (assumptions.sheet_thickness)
        und ist 0 wenn der Schalter use_pierce_penalty=False ist.
        """
        return self.assumptions.pierce_time()

    def effective_depth(self, tilt: float = 0.0) -> float:
        """Reachable depth when the cutter is tilted by *tilt* [rad].

        A tilted cutter reaches deeper into the material at the cost
        of a wider kerf:  d_eff = max_depth / cos(tilt).
        """
        if abs(tilt) > self.max_tilt:
            raise ValueError(
                f"|tilt| = {np.degrees(abs(tilt)):.1f}° exceeds "
                f"max_tilt = {np.degrees(self.max_tilt):.1f}°."
            )
        return self.max_depth / np.cos(tilt)

    def time_for_length(
        self,
        length: float,
        mode: str = "cut",
    ) -> float:
        """Time [s] to traverse *length* [mm].

        Parameters
        ----------
        length : distance [mm]
        mode   : 'cut'   -> cutting_speed  (Plasma an, im Material)
                 'move'  -> moving_speed   (Plasma aus, nahe Geometrie)
                 'rapid' -> rapid_speed    (Eilgang, freie Luft)
        """
        if mode == "cut":
            speed = self.cutting_speed
        elif mode == "move":
            speed = self.moving_speed
        elif mode == "rapid":
            speed = self.rapid_speed
        else:
            raise ValueError(f"Unknown mode '{mode}', use 'cut', 'move' or 'rapid'")
        return length / speed

    def __repr__(self) -> str:
        return (
            f"Cutter(max_depth={self.max_depth} mm, "
            f"L(v_cut)={self.blade_length():.1f} mm, "
            f"max_tilt={np.degrees(self.max_tilt):.1f} deg, "
            f"v_cut={self.cutting_speed} mm/s, "
            f"v_move={self.moving_speed} mm/s, "
            f"v_rapid={self.rapid_speed} mm/s, "
            f"minimum_gap={self.minimum_gap} mm, "
            f"pierce={self.pierce_time():.2f}s)"
        )
