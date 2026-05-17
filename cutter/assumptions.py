from __future__ import annotations

"""Zentrales Modul fuer Randbedingungen und Modellannahmen des Plasmaschneiders.

Dieses Modul fasst alle physikalisch/technologisch motivierten Annahmen
und Randbedingungen an einer Stelle zusammen.  Bisher waren diese Annahmen
implizit im Code verteilt; durch Konsolidierung lassen sie sich nun
zentral parametrisieren, dokumentieren und schrittweise verfeinern.

================================================================
A) Annahmen (idealisiertes Verhalten)
================================================================

A1  Konstante Leistung ueber alle Schnitte
A2  Brenner ist orthogonal zum Profil (mit optionaler Streuung A2*)
A3  Schnittbreite (Kerf) konstant
A4  Maximale Klingenlaenge konstant *bzw.* L(v) = L_ref * v_ref / v
A5  Kein Verschleiss von Duese/Elektrode

================================================================
B) Randbedingungen
================================================================

B1  Eintritt nur vom Aussenrand (oder Innenrand-Loch) der Geometrie
B2  Klingenlaenge L(v) als Funktion der Geschwindigkeit
B3  Wiedereintrittspauschale (Pierce-Time) pro Zuendung
B4  Einzeldurchgang -- Schnitt nur von einer Seite
B5  Innenrandpunkte: Brenner darf 45deg-Y-Fase schneiden (zusaetzlich zu 90deg)

================================================================
Literatur-Begruendung (Kurzform)
================================================================

[1] Hypertherm: "Cut angularity / perpendicularity tolerance ISO 9013".
    -> Standard-Brennertoleranz +/- 2deg gegenueber Senkrechte.
       Begruendung der Truncated-Normal-Verteilung A2*.
[2] ScienceDirect (Plasma Arc Cutting overview):
    "Cutting velocity is inversely proportional to depth of penetration."
    -> Begruendung der L(v) ~ 1/v Naeherung.
[3] DIN EN ISO 9013:2017 (Thermal cutting -- Geometrical product
    specifications and quality tolerances):
    Klassifiziert u = Senkrechtigkeits-/Neigungstoleranz fuer
    autogene/Plasma-/Laser-Schnitte. Liefert Wertebereiche.
[4] Liu, Tang, Tian (2024): "Model-driven path planning for robotic plasma
    cutting of branch pipe with single Y-groove". Adv. Manuf. 12:94-107.
    -> Begruendet 45deg-Y-Fase als Schweissnaht-Vorbereitung an
       Verschneidungskurven (Loch- bzw. Innenrandpunkte).
[5] Hypertherm Application Note "Bevel cutting":
    Bevelwinkel typisch 15-45deg, max. 45deg fuer Y-Faseanwendungen.
[6] Hypertherm Application Note "Pierce time delay":
    Wiedereintrittspauschale waechst monoton mit Blechstaerke;
    typische Werte 0.2-2.0 s je nach Blechstaerke und Stromstaerke.
"""

from dataclasses import dataclass, field
from typing import Callable
import math
import numpy as np


# ---------------------------------------------------------------------------
# A4 / B2  -- Klingenlaenge L(v)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BladeLengthModel:
    """Modelliert die effektiv erreichbare Klingenlaenge in Abhaengigkeit
    der Schneidgeschwindigkeit v.

    Naeherung 1 (Standard, konservativ):  L(v) = L_ref * v_ref / v
    -- folgt aus der Annahme, dass Energie pro Laengeneinheit
       E_l = P / v   konstant fuer einen vollstaendigen Durchschnitt
       benoetigt wird (siehe Lit. [2]).
    -- Begrenzt nach oben durch L_max (mechanische Reichweite des
       Brenners) und nach unten durch L_min (sonst kein Schnitt).

    Naeherung 2 (linear, optional): L(v) = L_max - k*(v - v_ref)
    -- empirische lineare Approximation um den Arbeitspunkt.

    Parameters
    ----------
    L_ref   : Klingenlaenge bei Referenzgeschwindigkeit [mm]
    v_ref   : Referenzgeschwindigkeit [mm/s]
    L_max   : maximale Klingenlaenge (z.B. Brennerhub) [mm]
    L_min   : minimale Schnitt-Tiefe (sonst kein Durchschnitt) [mm]
    mode    : "inverse" (Standard, ~1/v) oder "linear"
    slope   : nur fuer mode="linear": dL/dv (negativ) [mm * s / mm]
    """
    L_ref: float = 20.0
    v_ref: float = 5.0
    L_max: float = 40.0
    L_min: float = 1.0
    mode:  str   = "inverse"
    slope: float = -1.5   # nur fuer "linear"

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
    """Wiedereintrittspauschale je Zuendung [s].

    Der Brenner braucht nach jeder Zuendung eine Pausen-Zeit, in der
    das Material durchstossen wird.  Mit der Blechstaerke und der
    Stromstaerke waechst diese Zeit (Lit. [6]).

    Modell:  t_pierce(thickness) = t0 + k * thickness
             (linear, ueber Hypertherm-Cut-Charts angepasst)

    Parameters
    ----------
    t0     : Mindestpauschale je Zuendung [s] (Initialisierungszeit)
    k      : Zeit pro mm Blechstaerke [s/mm]
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
# A2*  -- Wahrscheinlichkeitsverteilung des Brennerwinkels
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TorchTiltDistribution:
    """Naehert die in der Realitaet auftretende Abweichung von der
    Orthogonalitaet (Senkrechtigkeit) durch eine **abgeschnittene
    Normalverteilung** an.

    Hintergrund
    -----------
    Industrielle Plasmaschneider erreichen i.d.R. eine Winkeltoleranz
    von +/- 2deg gegenueber der Senkrechten (Lit. [1], DIN EN ISO 9013
    [3]).  Im Mittel ist der Brenner orthogonal, kleine Abweichungen
    treten symmetrisch um Null herum auf -- ein Gauss-Modell ist die
    Standard-Annahme.  Da der Brenner mechanisch nicht beliebig
    schraegstellen kann, wird die Verteilung bei +/- max_tilt
    abgeschnitten.

    Verwendung
    ----------
    >>> rng = np.random.default_rng(seed=0)
    >>> dist = TorchTiltDistribution(sigma_deg=2.0, max_tilt_deg=5.0)
    >>> tilt = dist.sample(rng)                # einzelne Probe [rad]
    >>> tilts = dist.sample(rng, size=100)     # 100 Proben [rad]

    Fuer einen kontinuierlichen Pfad sollen die einzelnen Tilt-Proben
    *zeitlich korreliert* sein (sonst zittert die Klinge wild).  Die
    Methode :meth:`sample_path` zieht eine **glatte Trajektorie**
    aus einem Ornstein-Uhlenbeck-Prozess (Mittelwert 0, Standardabw.
    sigma, Korrelationslaenge correlation_length).

    Parameters
    ----------
    sigma_deg          : Standardabweichung der Verteilung [Grad]
                         (Standard: 2.0deg ~ ISO 9013 Range 3)
    max_tilt_deg       : Hard cut-off (truncation) [Grad]
                         (Standard: 5.0deg = mechanische Begrenzung)
    bias_deg           : optionaler Mittelwert (z.B. fuer "bad side"
                         der Plasmastrahl-Asymmetrie, Lit. [1])
    correlation_length : Korrelationsabstand entlang des Pfads [mm].
                         0 -> komplett unkorreliert (i.i.d. pro Waypoint).
    """
    sigma_deg:          float = 2.0
    max_tilt_deg:       float = 5.0
    bias_deg:           float = 0.0
    correlation_length: float = 20.0

    @property
    def sigma(self) -> float:
        return math.radians(self.sigma_deg)

    @property
    def max_tilt(self) -> float:
        return math.radians(self.max_tilt_deg)

    @property
    def bias(self) -> float:
        return math.radians(self.bias_deg)

    # ------------------------------------------------------------------
    # Einzelne Proben
    # ------------------------------------------------------------------

    def sample(
        self,
        rng: np.random.Generator | None = None,
        size: int | tuple[int, ...] | None = None,
    ) -> float | np.ndarray:
        """Zieht eine (oder mehrere) i.i.d. Proben aus der abgeschnittenen
        Normalverteilung. Rueckgabe in **Radiant**.
        """
        rng = rng or np.random.default_rng()
        if self.sigma <= 0:
            sample = np.zeros(size) if size is not None else 0.0
            return sample + self.bias  # noqa: RUF005

        if size is None:
            # Rejection sampling fuer eine einzelne Probe
            for _ in range(64):
                v = self.bias + self.sigma * rng.standard_normal()
                if abs(v) <= self.max_tilt:
                    return float(v)
            return float(np.clip(v, -self.max_tilt, self.max_tilt))

        out = np.full(size, self.bias, dtype=float)
        n_total = int(np.prod(size))
        accepted = 0
        flat = out.reshape(-1)
        while accepted < n_total:
            need = n_total - accepted
            cand = self.bias + self.sigma * rng.standard_normal(need * 2)
            cand = cand[np.abs(cand - self.bias) <= self.max_tilt]
            take = min(len(cand), need)
            flat[accepted:accepted + take] = cand[:take]
            accepted += take
        return out

    # ------------------------------------------------------------------
    # Glatte Trajektorie entlang eines Pfads
    # ------------------------------------------------------------------

    def sample_path(
        self,
        n_waypoints:    int,
        segment_length: float = 1.0,
        rng:            np.random.Generator | None = None,
    ) -> np.ndarray:
        """Erzeugt eine glatte (autokorrelierte) Tilt-Trajektorie der
        Laenge n_waypoints. Rueckgabe in **Radiant**.

        Mathematisch: diskrete Ornstein-Uhlenbeck-Trajektorie

            x[k+1] = bias + a*(x[k]-bias) + s * eps[k],   eps ~ N(0,1)

        mit a = exp(-segment_length / correlation_length)
        und s = sigma * sqrt(1 - a**2)

        Anschliessend werden Werte ausserhalb [-max_tilt, max_tilt]
        per Reflektion zurueckgespiegelt (sanfte Begrenzung).
        """
        rng = rng or np.random.default_rng()
        if n_waypoints <= 0:
            return np.zeros(0)
        if self.sigma <= 0:
            return np.full(n_waypoints, self.bias)
        if self.correlation_length <= 0:
            return self.sample(rng, size=n_waypoints)

        a = math.exp(-max(segment_length, 1e-9) / self.correlation_length)
        s = self.sigma * math.sqrt(max(0.0, 1.0 - a * a))

        out = np.empty(n_waypoints)
        x = self.bias + self.sigma * rng.standard_normal()
        for k in range(n_waypoints):
            out[k] = x
            x = self.bias + a * (x - self.bias) + s * rng.standard_normal()

        # Reflektion an den Schranken (sanfte Truncation, behaelt die
        # Korrelationsstruktur besser als hartes Clipping)
        lo, hi = -self.max_tilt, self.max_tilt
        out_b = out - self.bias
        span = hi - lo
        if span > 0:
            # 2*span periodische Reflektion
            m = np.mod(out_b - lo, 2.0 * span)
            m = np.where(m > span, 2.0 * span - m, m)
            out = lo + m + self.bias
        return out

    # ------------------------------------------------------------------
    # Diagnostik
    # ------------------------------------------------------------------

    def expected_lateral_deviation(self, depth: float) -> float:
        """Erwartete laterale Abweichung der Klingenspitze [mm] bei
        einer Schnitttiefe von *depth* [mm].

        E[|sin(theta)|] * depth  ~  E[|theta|] * depth   fuer kleine theta
        mit E[|theta|] = sigma * sqrt(2/pi)  fuer N(0,sigma^2).
        """
        if self.sigma <= 0:
            return abs(self.bias) * depth
        return float(self.sigma * math.sqrt(2.0 / math.pi)) * depth


# ---------------------------------------------------------------------------
# B5  -- 45deg-Y-Fase an Innenrandpunkten
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HoleBevelPolicy:
    """Steuert das Verhalten an Innenrandpunkten (Lochkontur).

    Liu et al. (2024) zeigen, dass an Verschneidungskurven (z.B. bei
    Bohrungen) eine Y-Fase als Schweissnahtvorbereitung gefordert ist.
    Die Y-Fase wird typischerweise mit 45deg geschnitten (Lit. [5]).

    Parameters
    ----------
    enabled         : globaler Schalter
    bevel_angle_deg : Fasenwinkel gegenueber der Senkrechten [Grad]
                      (Standard 45deg, Bereich 15deg..45deg laut Lit. [5])
    cut_perp_first  : Wenn True, wird zuerst der senkrechte Schnitt
                      (90deg) ausgefuehrt; danach folgt die Fase 45deg.
                      Entspricht Best-Practice "land first, then bevel"
                      (Lit. [5]).
    """
    enabled:         bool  = True
    bevel_angle_deg: float = 45.0
    cut_perp_first:  bool  = True

    @property
    def bevel_angle(self) -> float:
        return math.radians(self.bevel_angle_deg)


# ---------------------------------------------------------------------------
# Kombi-Konfiguration
# ---------------------------------------------------------------------------

@dataclass
class CuttingAssumptions:
    """Bundle aller Annahmen + Randbedingungen.

    Wird einer ``Cutter``- oder ``ContinuousPlanner``-Instanz uebergeben.
    Aenderungen an diesem Bundle wirken sich automatisch auf alle
    Berechnungen aus (Klingenlaenge, Schnittzeit, Schnittwinkel, etc.).

    Default-Werte entsprechen einem typischen 80 A Plasmaschneider
    fuer ~12 mm Baustahl.
    """
    blade:  BladeLengthModel      = field(default_factory=BladeLengthModel)
    pierce: PierceTimeModel       = field(default_factory=PierceTimeModel)
    tilt:   TorchTiltDistribution = field(default_factory=TorchTiltDistribution)
    hole:   HoleBevelPolicy       = field(default_factory=HoleBevelPolicy)

    # Materialbezogen (fuer Pierce + Vergleich mit L)
    sheet_thickness: float = 12.0  # [mm]

    # Globale Schalter
    use_velocity_dependent_blade: bool = True
    use_pierce_penalty:           bool = True
    use_tilt_noise:               bool = False
    use_hole_bevel:               bool = True

    # ------------------------------------------------------------------
    # Komfort-Methoden
    # ------------------------------------------------------------------

    def effective_blade_length(self, v: float, L_default: float) -> float:
        """Gibt die effektive Klingenlaenge zurueck (mit oder ohne
        Geschwindigkeitsabhaengigkeit)."""
        if not self.use_velocity_dependent_blade:
            return L_default
        return self.blade(v)

    def pierce_time(self) -> float:
        """Wiedereintrittspauschale [s]."""
        if not self.use_pierce_penalty:
            return 0.0
        return self.pierce(self.sheet_thickness)

    def sample_tilts(
        self,
        n_waypoints:    int,
        segment_length: float,
        rng:            np.random.Generator | None = None,
    ) -> np.ndarray:
        """Glatte Tilt-Trajektorie [rad] fuer einen Pfad. Wenn
        use_tilt_noise=False, wird ueberall 0 zurueckgegeben.
        """
        if not self.use_tilt_noise:
            return np.zeros(n_waypoints)
        return self.tilt.sample_path(n_waypoints, segment_length, rng)

    def summary(self) -> str:
        return (
            f"CuttingAssumptions(\n"
            f"  thickness={self.sheet_thickness:.1f} mm\n"
            f"  blade-model={'L(v)' if self.use_velocity_dependent_blade else 'const'} "
            f"(L_ref={self.blade.L_ref}, v_ref={self.blade.v_ref}, "
            f"L_max={self.blade.L_max})\n"
            f"  pierce={'ON' if self.use_pierce_penalty else 'OFF'} "
            f"(t0={self.pierce.t0}s, k={self.pierce.k}s/mm)\n"
            f"  tilt-noise={'ON' if self.use_tilt_noise else 'OFF'} "
            f"(sigma={self.tilt.sigma_deg}deg, max={self.tilt.max_tilt_deg}deg, "
            f"corr_len={self.tilt.correlation_length}mm)\n"
            f"  hole-bevel={'ON' if self.use_hole_bevel else 'OFF'} "
            f"(angle={self.hole.bevel_angle_deg}deg, "
            f"perp-first={self.hole.cut_perp_first})\n"
            f")"
        )


# ---------------------------------------------------------------------------
# Voreinstellungen (Presets) fuer haeufige Szenarien
# ---------------------------------------------------------------------------

def preset_ideal() -> CuttingAssumptions:
    """Ideales Modell: keine Streuung, keine Pauschale, konstante Klinge."""
    return CuttingAssumptions(
        use_velocity_dependent_blade=False,
        use_pierce_penalty=False,
        use_tilt_noise=False,
        use_hole_bevel=False,
    )


def preset_realistic() -> CuttingAssumptions:
    """Realitaetsnaehe-Modell (Default): L(v) + Pierce + 45deg-Fase."""
    return CuttingAssumptions()


def preset_noisy() -> CuttingAssumptions:
    """Wie 'realistic', aber mit Brennerwinkel-Streuung aktiv."""
    return CuttingAssumptions(use_tilt_noise=True)
