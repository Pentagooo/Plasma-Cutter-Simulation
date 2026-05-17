from __future__ import annotations

# Abstrakte Basisklassen aus Pythons Standard-Bibliothek:
# ABC = Abstract Base Class, abstractmethod = erzwingt Implementierung in Unterklassen
from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Shapely ist die Geometrie-Bibliothek, die alle 2D-Polygon-Berechnungen übernimmt
from shapely.geometry import Polygon


class CrossSection(ABC):
    """Abstrakte Basisklasse für alle 2D-Querschnitte.

    Jede Unterklasse muss _build_polygon() implementieren und ein
    gültiges Shapely-Polygon zurückgeben. Das Polygon wird beim
    Erzeugen des Objekts einmalig berechnet und dann gecacht,
    damit wiederholte Zugriffe (z.B. in der Optimierung) nicht
    jedes Mal neu rechnen müssen.
    """

    def __init__(self) -> None:
        # Beim Erzeugen des Objekts wird sofort das Polygon gebaut.
        # Die abstrakte Methode _build_polygon() wird dabei in der
        # jeweiligen Unterklasse (IBeam, CircleSection, ...) aufgerufen.
        self._polygon: Polygon = self._build_polygon()

        # Shapely kann ungültige Polygone erzeugen (z.B. sich selbst schneidende
        # Konturen). Wir prüfen das direkt beim Erstellen und werfen einen
        # verständlichen Fehler, bevor fehlerhafte Geometrie weiterverwendet wird.
        if not self._polygon.is_valid:
            raise ValueError(f"{self.__class__.__name__}: resulting polygon is not valid.")

    @abstractmethod
    def _build_polygon(self) -> Polygon:
        """Baut das Shapely-Polygon für diesen Querschnitt und gibt es zurück.

        Diese Methode muss in jeder Unterklasse überschrieben werden.
        Sie wird genau einmal im Konstruktor aufgerufen.
        """

    # ------------------------------------------------------------------
    # Geometrie-Schnittstelle
    # ------------------------------------------------------------------

    @property
    def polygon(self) -> Polygon:
        """Das gecachte Shapely-Polygon des Querschnitts.

        Wird von CutLine, CutExecution und dem Optimizer verwendet,
        um Schnitte zu berechnen und Flächen zu analysieren.
        """
        return self._polygon

    @property
    def area(self) -> float:
        """Querschnittsfläche in mm².

        Wird direkt von Shapely berechnet — für Hohlprofile (z.B. Rohr)
        zieht Shapely die innere Fläche automatisch ab.
        """
        return self._polygon.area

    @property
    def centroid(self) -> tuple[float, float]:
        """Schwerpunkt des Querschnitts als (cx, cy) in mm.

        Der Schwerpunkt wird geometrisch berechnet (Flächenschwerpunkt).
        Für symmetrische Profile liegt er im Ursprung.
        """
        c = self._polygon.centroid  # Shapely gibt ein Point-Objekt zurück
        return (c.x, c.y)          # wir geben ein einfaches Tupel zurück

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Achsenparallele Bounding Box: (x_min, y_min, x_max, y_max) in mm.

        Wird vom Optimizer genutzt, um den Suchbereich für Schnittlinien
        einzugrenzen, und von CutLine für die Visualisierung.
        """
        return self._polygon.bounds

    # ------------------------------------------------------------------
    # Visualisierung
    # ------------------------------------------------------------------

    def plot(self, ax: plt.Axes | None = None, **kwargs) -> plt.Axes:
        """Zeichnet den Querschnitt als gefüllte Fläche mit Umriss.

        Parameters
        ----------
        ax:
            Vorhandene Matplotlib-Achse. Wenn None, wird eine neue
            Figur mit einer Achse erstellt.
        **kwargs:
            Werden direkt an den Matplotlib-Polygon-Patch weitergegeben,
            z.B. color="red", alpha=0.5.
        """
        # Neue Figur nur anlegen wenn keine Achse übergeben wurde
        if ax is None:
            _, ax = plt.subplots()

        # Außenkontur des Polygons als NumPy-Array: Form (N, 2)
        coords = np.array(self._polygon.exterior.coords)

        # Standard-Darstellung: stahlblau, leicht transparent
        patch_kw = dict(facecolor="steelblue", edgecolor="black",
                        linewidth=1.2, alpha=0.4)
        # Überschreiben mit user-definierten Parametern
        patch_kw.update(kwargs)

        # Matplotlib-Patch aus den Koordinaten erzeugen und zur Achse hinzufügen
        patch = mpatches.Polygon(coords, closed=True, **patch_kw)
        ax.add_patch(patch)

        # Schwerpunkt als rotes Kreuz einzeichnen
        cx, cy = self.centroid
        ax.plot(cx, cy, "+", color="red", markersize=8, label="centroid")

        # Achsengrenzen automatisch auf Bounding Box + 10 % Rand setzen
        x_min, y_min, x_max, y_max = self.bounds
        margin = max(x_max - x_min, y_max - y_min) * 0.1
        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(y_min - margin, y_max + margin)

        # Gleiche Skalierung in x und y, damit das Profil nicht verzerrt wirkt
        ax.set_aspect("equal")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")

        # Klassenname als Titel (z.B. "IBeam", "CircleSection")
        ax.set_title(self.__class__.__name__)
        ax.grid(True, linestyle="--", alpha=0.4)

        return ax

    # ------------------------------------------------------------------
    # Dunder-Methoden
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Kompakte Textdarstellung mit Fläche und Schwerpunkt-Koordinaten."""
        cx, cy = self.centroid
        return (f"{self.__class__.__name__}("
                f"area={self.area:.2f} mm², "
                f"centroid=({cx:.2f}, {cy:.2f}))")
