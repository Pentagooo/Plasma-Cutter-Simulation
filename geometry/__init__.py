"""Geometrie-Paket der Segment-Simulation.

Was hier passiert
-----------------
Dieses ``__init__`` macht das ``geometry``-Paket importierbar und legt
fest, welche Symbole man direkt ``from plasma_cutter.geometry import X``
holen kann.
"""

from .point_grid import PointGrid, GridPoint, PointStatus

__all__ = ["PointGrid", "GridPoint", "PointStatus"]