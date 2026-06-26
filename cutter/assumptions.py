from __future__ import annotations

"""Zentrales Modul für Randbedingungen und Modellannahmen des Plasmaschneiders.

================================================================
A) Annahmen 
================================================================

A1  Konstante Leistung über alle Schnitte
A2  Brenner ist orthogonal zum Profil (mit optionaler Streuung A2***)
A3  Schnittbreite (Kerf) konstant
A4  Maximale Tiefe konstant ***bzw.* L(v) = L_ref * v_ref / v
A5  Kein Verschleiß von Düsse/Elektrode

================================================================
B) Randbedingungen
================================================================

B1  Eintritt nur vom Außenrand  der Geometrie
B2  Klingenlänge L(v) als Funktion der Geschwindigkeit
B3  Wiedereintrittspauschale (Pierce-Time) pro Zündung
B4  Einzeldurchgang -- Schnitt nur von einer Seite
"""

from dataclasses import dataclass, field
from typing import Callable
import math
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
# A2***  -- Wahrscheinlichkeitsverteilung des Brennerwinkels
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TorchTiltDistribution:
    """Nähert die auftretende Abweichung von der
    Orthogonalität durch eine **abgeschnittene
    Normalverteilung** an

    >>> rng = np.random.default_rng(seed=0)
    >>> dist = TorchTiltDistribution(sigma_deg=2.0, max_tilt_deg=5.0)
    >>> tilt = dist.sample(rng)                # einzelne Probe [rad]
    >>> tilts = dist.sample(rng, size=100)     # 100 Proben [rad]

    Für einen kontinuierlichen Pfad sollen die einzelnen Tilt-Proben
    *zeitlich korreliert* sein (sonst zittert die Klinge wild).  Die
    Methode :meth:`sample_path` zieht eine **glatte Trajektorie**
    aus einem Ornstein-Uhlenbeck-Prozess (Mittelwert 0, Standardabw.
    sigma, Korrelationslänge correlation_length).

    Parameters
    ----------
    sigma_deg          : Standardabweichung der Verteilung [Grad]
                         (Standard: 2.0deg ~ ISO 9013 Range 3)
    max_tilt_deg       : Hard cut-off (truncation) [Grad]
                         (Standard: 5.0deg = mechanische Begrenzung)
    bias_deg           : optionaler Mittelwert (z.B. für "bad side"
                         der Plasmastrahl-Asymmetrie
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
        Normalverteilung. Rückgabe in **Radiant**.
        """
        rng = rng or np.random.default_rng()
        if self.sigma <= 0:
            sample = np.zeros(size) if size is not None else 0.0
            return sample + self.bias  # noqa: RUF005

        if size is None:
            # Rejection sampling für eine einzelne Probe
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
        """Erzeugt eine glatte Tilt-Trajektorie der
        Länge n_waypoints. Rückgabe in **Radiant**.

        Mathematisch: diskrete Ornstein-Uhlenbeck-Trajektorie

            x[k+1] = bias + a*(x[k]-bias) + s * eps[k],   eps ~ N(0,1)

        mit a = exp(-segment_length / correlation_length)
        und s = sigma * sqrt(1 - a**2)

        Anschließend werden Werte außerhalb [-max_tilt, max_tilt]
        zurückgespiegelt.
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

        # Reflektion an den Schranken (sanfte Truncation, behält die
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

        E[|sin(theta)|] * depth  ~  E[|theta|] * depth   für kleine theta
        mit E[|theta|] = sigma * sqrt(2/pi)  für N(0,sigma^2).
        """
        if self.sigma <= 0:
            return abs(self.bias) * depth
        return float(self.sigma * math.sqrt(2.0 / math.pi)) * depth


# ---------------------------------------------------------------------------
# Kombi-Konfiguration
# ---------------------------------------------------------------------------

@dataclass
class CuttingAssumptions:
    """Bundle aller Annahmen + Randbedingungen.

    Wird einer ``Cutter``- oder ``ContinuousPlanner``-Instanz übergeben.
    Änderungen an diesem Bundle wirken sich automatisch auf alle
    Berechnungen aus (Klingenlänge, Schnittzeit, Schnittwinkel, etc.).

    """
    blade:  BladeLengthModel      = field(default_factory=BladeLengthModel)
    pierce: PierceTimeModel       = field(default_factory=PierceTimeModel)
    tilt:   TorchTiltDistribution = field(default_factory=TorchTiltDistribution)

    # Materialbezogen (für Pierce + Vergleich mit L)
    sheet_thickness: float = 12.0  # [mm]

    # Globale Schalter
    use_velocity_dependent_blade: bool = True
    use_pierce_penalty:           bool = True
    use_tilt_noise:               bool = False

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

    def sample_tilts(
        self,
        n_waypoints:    int,
        segment_length: float,
        rng:            np.random.Generator | None = None,
    ) -> np.ndarray:
        """Glatte Tilt-Trajektorie [rad] für einen Pfad. Wenn
        use_tilt_noise=False, wird überall 0 zurückgegeben.
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
            f")"
        )


# ---------------------------------------------------------------------------
# Voreinstellungen (Presets) für häufige Szenarien
# ---------------------------------------------------------------------------

def preset_ideal() -> CuttingAssumptions:
    """Ideales Modell: keine Streuung, keine Pauschale, konstante Klinge."""
    return CuttingAssumptions(
        use_velocity_dependent_blade=False,
        use_pierce_penalty=False,
        use_tilt_noise=False,
    )


def preset_realistic() -> CuttingAssumptions:
    """Realitätsnähe-Modell (Default): L(v) + Pierce."""
    return CuttingAssumptions()


def preset_noisy() -> CuttingAssumptions:
    """Wie 'realistic', aber mit Brennerwinkel-Streuung aktiv."""
    return CuttingAssumptions(use_tilt_noise=True)
