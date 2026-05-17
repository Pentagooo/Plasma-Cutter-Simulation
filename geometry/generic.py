from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon, Point

# CrossSection ist die abstrakte Basisklasse — alle Profile erben von ihr
from .base import CrossSection


class PolygonSection(CrossSection):
    """Querschnitt aus einer beliebigen geschlossenen Punktliste.

    Damit lassen sich alle Profile abbilden, die sich nicht durch
    einfache Parameter (Höhe, Breite, Dicke) beschreiben lassen —
    zum Beispiel Konturen aus dem ContourEditor oder aus einer JSON-Datei.

    Parameters
    ----------
    points:
        Liste von (x, y)-Koordinaten in mm, die den Außenumriss bilden.
        Das Polygon wird automatisch geschlossen — der letzte Punkt muss
        nicht identisch mit dem ersten sein.
    holes:
        Optionale Liste von Innenkonturen (jede selbst eine Punktliste),
        um Hohlräume mit nicht-kreisförmiger Form zu modellieren.
        Beispiel: eine rechteckige Platte mit einem dreieckigen Loch.
    """

    def __init__(self,
                 points: list[tuple[float, float]],
                 holes: list[list[tuple[float, float]]] | None = None) -> None:
        # Außenkontur als einfache Python-Liste speichern
        self._points = list(points)
        # Hohlräume: leere Liste wenn keine angegeben (nie None intern)
        self._holes = holes or []
        # Eltern-Konstruktor aufrufen → ruft _build_polygon() auf
        super().__init__()

    def _build_polygon(self) -> Polygon:
        # Shapely erwartet: Polygon(außen, [loch1, loch2, ...])
        # Wenn keine Löcher vorhanden ist _holes eine leere Liste → kein Hohlraum
        return Polygon(self._points, self._holes)


class CircleSection(CrossSection):
    """Kreisförmiger Vollquerschnitt (Rundstab).

    Der Kreis wird intern als Vieleck mit vielen Seiten angenähert,
    da Shapely keine echten Kreise kennt. Mit resolution=128 ist die
    Abweichung für typische Stahlbau-Abmessungen vernachlässigbar klein.

    Parameters
    ----------
    radius:
        Außenradius in mm. Muss positiv sein.
    center:
        Mittelpunkt (cx, cy) in mm. Standard: Ursprung (0, 0).
    resolution:
        Anzahl der Polygonsegmente für die Kreisannäherung.
        Höhere Werte = glatterer Kreis, aber langsamer.
    """

    def __init__(self,
                 radius: float,
                 center: tuple[float, float] = (0.0, 0.0),
                 resolution: int = 128) -> None:
        # Negativer oder null Radius ist geometrisch sinnlos
        if radius <= 0:
            raise ValueError("radius must be positive.")
        self._radius = radius
        self._center = center
        self._resolution = resolution
        super().__init__()

    def _build_polygon(self) -> Polygon:
        # Point.buffer(r) erzeugt einen Kreis mit Radius r um den Mittelpunkt.
        # resolution steuert wie viele Ecken das Näherungspolygon hat.
        return Point(self._center).buffer(self._radius, resolution=self._resolution)

    @property
    def radius(self) -> float:
        """Außenradius des Kreisquerschnitts in mm."""
        return self._radius


class CircularHollow(CrossSection):
    """Kreisringquerschnitt (Rohr / Hohlzylinder).

    Wird als Differenz zweier konzentrischer Kreise aufgebaut:
    Außenkreis minus Innenkreis = Wandfläche.

    Parameters
    ----------
    outer_radius:
        Außenradius in mm.
    inner_radius:
        Innenradius in mm. Muss strikt kleiner als outer_radius sein.
    center:
        Mittelpunkt (cx, cy) in mm. Standard: Ursprung (0, 0).
    resolution:
        Anzahl der Polygonsegmente pro Kreis für die Annäherung.
    """

    def __init__(self,
                 outer_radius: float,
                 inner_radius: float,
                 center: tuple[float, float] = (0.0, 0.0),
                 resolution: int = 128) -> None:
        # Beide Radien müssen sinnvoll sein: innen > 0, außen > innen
        if inner_radius <= 0 or outer_radius <= inner_radius:
            raise ValueError("Require 0 < inner_radius < outer_radius.")
        self._outer_radius = outer_radius
        self._inner_radius = inner_radius
        self._center = center
        self._resolution = resolution
        super().__init__()

    def _build_polygon(self) -> Polygon:
        # Außenkreis erzeugen
        outer = Point(self._center).buffer(self._outer_radius,
                                           resolution=self._resolution)
        # Innenkreis erzeugen
        inner = Point(self._center).buffer(self._inner_radius,
                                           resolution=self._resolution)
        # Differenz: Außenkreis minus Innenkreis → Ringfläche
        # Shapely kümmert sich dabei automatisch um die Loch-Repräsentation
        return outer.difference(inner)

    @property
    def outer_radius(self) -> float:
        """Außenradius des Rohrs in mm."""
        return self._outer_radius

    @property
    def inner_radius(self) -> float:
        """Innenradius (Bohrung) des Rohrs in mm."""
        return self._inner_radius

    @property
    def wall_thickness(self) -> float:
        """Wanddicke des Rohrs in mm (outer_radius − inner_radius)."""
        return self._outer_radius - self._inner_radius