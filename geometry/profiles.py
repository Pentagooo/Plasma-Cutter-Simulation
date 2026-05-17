from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

# CrossSection ist die abstrakte Basisklasse — alle Profile erben von ihr
from .base import CrossSection


def _fillet(p0: tuple, corner: tuple, p1: tuple, r: float,
            resolution: int = 8) -> list[tuple[float, float]]:
    """Berechnet einen Verrundungsbogen an einer Innenecke.

    An Steg-Flansch-Übergängen bei I-Profilen gibt es scharfe Innenecken,
    die bei realen Profilen abgerundet sind (Wurzelausrundungsradius r).
    Diese Hilfsfunktion ersetzt die scharfe Ecke durch einen Kreisbogen.

    Funktionsprinzip:
      - Die beiden Schenkel von 'corner' zu 'p0' und 'p1' werden normiert.
      - Der Kreismittelpunkt liegt auf der Winkelhalbierenden in Abstand d.
      - Entlang des Kreisbogens werden 'resolution+1' gleichmäßige Punkte
        berechnet, die den Bogen approximieren.

    Parameters
    ----------
    p0      : erster Nachbarpunkt der Ecke
    corner  : der eigentliche Eckpunkt, der abgerundet werden soll
    p1      : zweiter Nachbarpunkt der Ecke
    r       : Verrundungsradius in mm (0 = scharfe Ecke bleibt erhalten)
    resolution : Anzahl der Bogenpunkte (höher = glatter)

    Returns
    -------
    Liste von (x, y)-Punkten, die den Bogen beschreiben.
    Bei r=0 wird nur der Eckpunkt selbst zurückgegeben.
    """
    # Kein Radius → scharfe Ecke unverändert zurückgeben
    if r <= 0:
        return [corner]

    # Einheitsvektoren von der Ecke zu den Nachbarpunkten
    v0 = np.array(p0) - np.array(corner)
    v1 = np.array(p1) - np.array(corner)
    v0 /= np.linalg.norm(v0)   # normieren → Länge 1
    v1 /= np.linalg.norm(v1)

    # Winkelhalbierende zwischen den beiden Schenkeln
    bisector = v0 + v1
    bisector /= np.linalg.norm(bisector)

    # Cosinus des halben Öffnungswinkels zwischen den Schenkeln
    cos_half = np.dot(v0, bisector)
    # Abstand von der Ecke zum Kreismittelpunkt (Geometrie des einbeschriebenen Kreises)
    # +1e-12 verhindert Division durch Null bei 180°-Winkeln
    d = r / np.sqrt(1 - cos_half ** 2 + 1e-12)

    # Kreismittelpunkt auf der Winkelhalbierenden
    center = np.array(corner) + d * bisector

    # Winkel vom Kreismittelpunkt zu den beiden Tangentenpunkten
    a0 = np.arctan2((np.array(corner) + r * v0 - center)[1],
                    (np.array(corner) + r * v0 - center)[0])
    a1 = np.arctan2((np.array(corner) + r * v1 - center)[1],
                    (np.array(corner) + r * v1 - center)[0])

    # Sweeprichtung korrigieren: Bogen immer auf der kürzeren Seite
    if a1 - a0 > np.pi:
        a0 += 2 * np.pi
    elif a0 - a1 > np.pi:
        a1 += 2 * np.pi

    # Gleichmäßige Winkelunterteilung entlang des Bogens
    angles = np.linspace(a0, a1, resolution + 1)
    # Bogenpunkte aus Winkeln berechnen
    return [(center[0] + r * np.cos(a), center[1] + r * np.sin(a))
            for a in angles]


class IBeam(CrossSection):
    """Doppel-T / I-Träger Querschnitt (z.B. HEA, HEB, IPE).

    Das Profil ist symmetrisch und am Ursprung zentriert:
    Oberflansch oben, Unterflansch unten, Steg in der Mitte.

    Ohne Verrundungsradius (r=0) entstehen scharfe Innenecken an den
    vier Steg-Flansch-Übergängen. Mit r>0 werden diese durch Kreisbögen
    ersetzt, was realen Walzprofilen entspricht.

    Parameters
    ----------
    h  : Gesamthöhe des Profils in mm
    b  : Flanschbreite in mm
    t_f: Flanschdicke in mm
    t_w: Stegdicke in mm
    r  : Wurzelausrundungsradius in mm (0 = scharfe Ecken)
    """

    def __init__(self, h: float, b: float,
                 t_f: float, t_w: float, r: float = 0.0) -> None:
        self.h = h
        self.b = b
        self.t_f = t_f
        self.t_w = t_w
        self.r = r
        super().__init__()

    def _build_polygon(self) -> Polygon:
        h, b, tf, tw, r = self.h, self.b, self.t_f, self.t_w, self.r

        # y-Koordinaten: symmetrisch um y=0
        y_top = h / 2           # Oberkante Oberflansch
        y_bot = -h / 2          # Unterkante Unterflansch
        y_tf_top = y_top - tf   # Unterkante Oberflansch (= Steg-Oberkante)
        y_tf_bot = y_bot + tf   # Oberkante Unterflansch (= Steg-Unterkante)

        # x-Koordinaten
        x_fl = b / 2    # halbe Flanschbreite (äußere Flanschkante)
        x_tw = tw / 2   # halbe Stegdicke (Stegkante)

        if r <= 0:
            # Scharfkantige Variante: 12 Eckpunkte gegen den Uhrzeigersinn
            pts = [
                (-x_fl, y_bot),    # untere linke Flanschkante
                ( x_fl, y_bot),    # untere rechte Flanschkante
                ( x_fl, y_tf_bot), # Übergang Flansch/Steg unten rechts (außen)
                ( x_tw, y_tf_bot), # Übergang Flansch/Steg unten rechts (innen)
                ( x_tw, y_tf_top), # Übergang Flansch/Steg oben rechts (innen)
                ( x_fl, y_tf_top), # Übergang Flansch/Steg oben rechts (außen)
                ( x_fl, y_top),    # obere rechte Flanschkante
                (-x_fl, y_top),    # obere linke Flanschkante
                (-x_fl, y_tf_top), # Übergang Flansch/Steg oben links (außen)
                (-x_tw, y_tf_top), # Übergang Flansch/Steg oben links (innen)
                (-x_tw, y_tf_bot), # Übergang Flansch/Steg unten links (innen)
                (-x_fl, y_tf_bot), # Übergang Flansch/Steg unten links (außen)
            ]
        else:
            # Verrundete Variante: Kreisbögen an den vier Innenecken.
            # _fillet() gibt jeweils mehrere Bogenpunkte zurück, die
            # die scharfe Ecke ersetzen.
            br_arc = _fillet(( x_fl,  y_tf_bot), ( x_tw,  y_tf_bot), ( x_tw,  y_top - tf + r),  r)
            tr_arc = _fillet(( x_tw,  y_tf_top - r + tf), ( x_tw,  y_tf_top), ( x_fl,  y_tf_top), r)
            tl_arc = _fillet((-x_fl,  y_tf_top), (-x_tw,  y_tf_top), (-x_tw,  y_tf_bot + r - r),  r)
            bl_arc = _fillet((-x_tw,  y_tf_bot), (-x_fl,  y_tf_bot), (-x_fl,  y_bot + tf - r), r)

            # Kontur aus geraden Abschnitten und Bögen zusammensetzen
            pts = (
                [(-x_fl, y_bot), (x_fl, y_bot), (x_fl, y_tf_bot)]
                + br_arc
                + [(x_tw, y_tf_top)]
                + tr_arc
                + [(x_fl, y_top), (-x_fl, y_top), (-x_fl, y_tf_top)]
                + tl_arc
                + [(-x_tw, y_tf_bot)]
                + bl_arc
            )

        return Polygon(pts)


class UProfile(CrossSection):
    """U-Profil / C-Profil Querschnitt (nach rechts offen, am Ursprung zentriert).

    Besteht aus zwei horizontalen Flanschen (oben und unten) und
    einem vertikalen Steg auf der linken Seite. Die offene Seite
    zeigt in +x-Richtung.

    Parameters
    ----------
    h  : Gesamthöhe in mm
    b  : Gesamtbreite in mm
    t_f: Flanschdicke in mm (oben und unten gleich)
    t_w: Stegdicke in mm (linke Seite)
    """

    def __init__(self, h: float, b: float,
                 t_f: float, t_w: float) -> None:
        self.h = h
        self.b = b
        self.t_f = t_f
        self.t_w = t_w
        super().__init__()

    def _build_polygon(self) -> Polygon:
        h, b, tf, tw = self.h, self.b, self.t_f, self.t_w

        # Außengrenzen des Profils
        x_left  = -b / 2    # linke Außenkante (Steg)
        x_right =  b / 2    # rechte Außenkante (offene Seite)
        y_top   =  h / 2    # Oberkante
        y_bot   = -h / 2    # Unterkante

        # 8 Eckpunkte des U-Profils gegen den Uhrzeigersinn:
        pts = [
            (x_left,        y_bot),           # 1: untere linke Außenecke
            (x_right,       y_bot),           # 2: untere rechte Außenecke
            (x_right,       y_bot + tf),      # 3: Innenkante Unterflansch rechts
            (x_left + tw,   y_bot + tf),      # 4: Innenkante Unterflansch links
            (x_left + tw,   y_top - tf),      # 5: Innenkante Oberflansch links
            (x_right,       y_top - tf),      # 6: Innenkante Oberflansch rechts
            (x_right,       y_top),           # 7: obere rechte Außenecke
            (x_left,        y_top),           # 8: obere linke Außenecke
        ]
        return Polygon(pts)


class LProfile(CrossSection):
    """L-Profil / Winkelstahl Querschnitt.

    Die Innenecke liegt im Ursprung (0, 0). Der horizontale Schenkel
    verläuft in +x-Richtung, der vertikale Schenkel in +y-Richtung.

    Parameters
    ----------
    h  : Höhe des vertikalen Schenkels in mm
    b  : Breite des horizontalen Schenkels in mm
    t1 : Dicke des vertikalen Schenkels in mm
    t2 : Dicke des horizontalen Schenkels in mm
    """

    def __init__(self, h: float, b: float,
                 t1: float, t2: float) -> None:
        self.h = h
        self.b = b
        self.t1 = t1
        self.t2 = t2
        super().__init__()

    def _build_polygon(self) -> Polygon:
        h, b, t1, t2 = self.h, self.b, self.t1, self.t2

        # 6 Eckpunkte des L-Profils:
        # Ecke liegt im Ursprung, Schenkel gehen nach rechts (+x) und oben (+y)
        pts = [
            (0,  0),    # 1: Innenecke (Ursprung)
            (b,  0),    # 2: rechtes Ende des horizontalen Schenkels (unten)
            (b,  t2),   # 3: rechtes Ende des horizontalen Schenkels (oben)
            (t1, t2),   # 4: Übergang zum vertikalen Schenkel (innen)
            (t1, h),    # 5: oberes Ende des vertikalen Schenkels (rechts)
            (0,  h),    # 6: oberes Ende des vertikalen Schenkels (links)
        ]
        return Polygon(pts)


class RectHollow(CrossSection):
    """Rechteckiges Hohlprofil (RHS / SHS — Rectangular/Square Hollow Section).

    Außenrechteck minus Innenrechteck. Alle vier Wände haben die gleiche
    Dicke t. Das Profil ist am Ursprung zentriert.

    Parameters
    ----------
    h : Außenhöhe in mm
    b : Außenbreite in mm
    t : Wanddicke in mm (muss kleiner als h/2 und b/2 sein)
    """

    def __init__(self, h: float, b: float, t: float) -> None:
        # Prüfen ob die Wanddicke nicht zu groß ist (Innenmaß muss > 0 sein)
        if 2 * t >= min(h, b):
            raise ValueError("Wall thickness too large for given dimensions.")
        self.h = h
        self.b = b
        self.t = t
        super().__init__()

    def _build_polygon(self) -> Polygon:
        h, b, t = self.h, self.b, self.t

        # Außenrechteck: vier Eckpunkte, zentriert am Ursprung
        outer = Polygon([
            (-b / 2, -h / 2), ( b / 2, -h / 2),
            ( b / 2,  h / 2), (-b / 2,  h / 2),
        ])

        # Innenrechteck: um die Wanddicke t nach innen versetzt
        inner = Polygon([
            (-(b / 2 - t), -(h / 2 - t)),
            ( (b / 2 - t), -(h / 2 - t)),
            ( (b / 2 - t),  (h / 2 - t)),
            (-(b / 2 - t),  (h / 2 - t)),
        ])

        # Differenz ergibt das Hohlprofil: Shapely behandelt das Loch automatisch
        return outer.difference(inner)