"""Surrogat-Planer, Automatic Planner und Brute-Force-Lehrer.

Gemeinsame Pipeline: Segmentierung -> Auswahl -> DP-Split je Kette ->
Held-Karp + LinkPlanner -> exakte Coverage.

  teacher.exhaustive_plan  : Brute Force über alle Segment-Teilmengen
                             (Optimum, Trainingslabels, Benchmark)
  planner.greedy_plus_plan : Greedy Set Cover des AutoPlanner + gleiche
                             Geschwindigkeitsstufe (Vergleichsbasis)
  planner.surrogate_plan   : Modell ordnet die Segmente, Greedy Set Cover
                             auf exakten Masken, Fallback; Coverage hängt
                             nie am Modell

Weitere Module: features, instances, dataset, model, runutils (gemeinsame
Planungsstufen), benchmark, learning_curve.

Paketimport bleibt schlank (nur features), der Rest wird bei Bedarf
importiert.
"""

from .features import FEATURE_NAMES, segment_features

__all__ = ["FEATURE_NAMES", "segment_features"]
