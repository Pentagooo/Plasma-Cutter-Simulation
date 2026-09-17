"""Surrogat-Planer, Greedy+ und Brute-Force-Lehrer (BA Kap. 5).

Alle drei automatischen Auswahlverfahren sind Einstiege in dieselbe
Planungspipeline aus Kap. 4 (Segmentierung -> Auswahl -> DP-Split je
Kette -> Held-Karp + LinkPlanner -> exakte Coverage):

  * ``teacher.exhaustive_plan``  -- Brute Force: alle Segment-Teilmengen,
        jede vollstaendige Abdeckung exakt bewertet, Minimum = Optimum.
        Erzeugt die Trainingslabels und das Optimum im Benchmark.
  * ``planner.greedy_plus_plan`` -- Greedy Set Cover des ``AutoPlanner``
        mit derselben Geschwindigkeitsstufe (faire Vergleichsbasis).
  * ``planner.surrogate_plan``   -- gelerntes Modell ordnet die Segmente,
        Greedy Set Cover auf exakten Masken, ein Planbau, ein Verify,
        klassischer Fallback. Die Garantie haengt nie am Modell.

Weitere Module: ``features`` (Merkmale nur aus Kontur + Distanzen),
``instances`` (Katalog + Testgeometrien), ``dataset`` (Label-Pipeline),
``model`` (Training/Laden), ``runutils`` (gemeinsame Planungsstufen),
``benchmark`` und ``learning_curve``.

Der Paketimport bleibt schlank (nur ``features``); Planer, Lehrer und
Modell werden bei Bedarf importiert.
"""

from .features import FEATURE_NAMES, segment_features

__all__ = ["FEATURE_NAMES", "segment_features"]
