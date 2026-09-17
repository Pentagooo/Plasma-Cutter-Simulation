"""Surrogat-Modell: Segment-Auswahl-Klassifikator (BA Kap. 5).

Ein scikit-learn ``HistGradientBoostingClassifier`` (Fallback:
``GradientBoostingClassifier``) sagt je Segment die Wahrscheinlichkeit
p(s) vorher, in der zeitoptimalen Auswahl des Lehrers zu liegen. Der
Planer nutzt nur die RANGFOLGE von p(s); tau (Recall >= 0.95 auf den
Out-of-fold-Vorhersagen) wird der Vollstaendigkeit halber mitgespeichert.

Das Modell traegt KEINE Korrektheitsgarantie -- es entscheidet nur ueber
Geschwindigkeit. Die Coverage-Garantie stellt der Planer ueber exakte
Masken, Verify und Fallback sicher (siehe planner.py).

Das gespeicherte Modell traegt den Parameterstempel aus ``params``
(``label_version``, ``phys_hash``, ``params_hash``); ``load_model`` bricht
hart ab, wenn Version oder Physik nicht zum laufenden Code passen, und
warnt bei anderer Segmentierung/Katalog.

CLI (aus dem Elternordner von plasma_cutter):
    python -m plasma_cutter.segment_simulation.surrogate.model --train
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.inspection import permutation_importance

try:
    from sklearn.ensemble import HistGradientBoostingClassifier as _HGB
    _HAS_HGB = True
except ImportError:  # sehr alte sklearn-Versionen
    from sklearn.ensemble import GradientBoostingClassifier as _HGB
    _HAS_HGB = False
from sklearn.ensemble import GradientBoostingClassifier

try:
    from .features import FEATURE_NAMES
except ImportError:  # Direktstart ohne Paket-Kontext
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from plasma_cutter.segment_simulation.surrogate.features import FEATURE_NAMES


ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = ARTIFACTS / "surrogate_model.joblib"

# Ziel-Recall der positiven Klasse bei der tau-Kalibrierung.
TARGET_RECALL = 0.95
DEFAULT_TAU = 0.5


def _params():
    """``surrogate.params`` lazy importieren (zieht shapely/geometry)."""
    try:
        from . import params
    except ImportError:
        from plasma_cutter.segment_simulation.surrogate import params
    return params


def _make_estimator(random_state: int = 0):
    """Baut den Klassifikator (HGB, sonst GradientBoosting)."""
    if _HAS_HGB:
        return _HGB(
            max_iter=300, learning_rate=0.08, max_depth=None,
            max_leaf_nodes=31, l2_regularization=1.0,
            early_stopping=False, random_state=random_state,
        )
    return GradientBoostingClassifier(random_state=random_state)


def _sample_weight(y: np.ndarray) -> np.ndarray:
    """class_weight-Aequivalent ueber sample_weight (balanced)."""
    y = np.asarray(y)
    w = np.ones(len(y), dtype=float)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos > 0 and n_neg > 0:
        # balanced: n_samples / (2 * n_class)
        w[y == 1] = len(y) / (2.0 * n_pos)
        w[y == 0] = len(y) / (2.0 * n_neg)
    return w


def _calibrate_tau(y_true: np.ndarray, p: np.ndarray,
                   target_recall: float = TARGET_RECALL) -> float:
    """Groesstes tau, bei dem der Recall der positiven Klasse >=
    target_recall bleibt (maximiert damit die Praezision unter der
    Recall-Nebenbedingung)."""
    y_true = np.asarray(y_true).astype(int)
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return DEFAULT_TAU
    order = np.unique(np.concatenate([[0.0, 1.0], p]))
    best_tau = 0.0
    for tau in order:
        pred = p >= tau
        tp = int(np.sum(pred & (y_true == 1)))
        recall = tp / n_pos
        if recall >= target_recall:
            best_tau = float(tau)   # order aufsteigend -> letztes gueltiges
    return best_tau


@dataclass
class TrainReport:
    tau: float
    cv_recall_pos: float
    cv_precision_pos: float
    n_rows: int
    n_pos: int
    importances: list[tuple[str, float]]


class SurrogateModel:
    """Wrapper um den Segment-Auswahl-Klassifikator."""

    def __init__(self, estimator=None, tau: float = DEFAULT_TAU,
                 feature_names: list[str] | None = None):
        self.estimator = estimator
        self.tau = float(tau)
        self.feature_names = list(feature_names or FEATURE_NAMES)

    # ------------------------------------------------------------------

    def train(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
              n_splits: int = 5, random_state: int = 0,
              out_dir: Path | None = None,
              holdout_family: int | None = None,
              write_importances: bool = True) -> TrainReport:
        """Trainiert das Modell mit GroupKFold-Kalibrierung nach Formfamilie.

        Ablauf: Out-of-fold-Wahrscheinlichkeiten via GroupKFold ->
        tau-Kalibrierung (Recall >= 0.95) -> Refit auf allen Daten ->
        Permutations-Feature-Importances (als CSV gespeichert).

        ``holdout_family`` (Leave-one-family-out): Ist eine Familien-ID
        gesetzt, werden ALLE Zeilen dieser Familie vor dem Training entfernt.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        groups = np.asarray(groups, dtype=int)
        if out_dir is None:
            out_dir = ARTIFACTS
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if holdout_family is not None:
            keep = groups != int(holdout_family)
            X, y, groups = X[keep], y[keep], groups[keep]

        n_groups = len(np.unique(groups))
        splits = max(2, min(n_splits, n_groups))

        est = _make_estimator(random_state)

        # Out-of-fold-Wahrscheinlichkeiten (fuer die tau-Kalibrierung)
        gkf = GroupKFold(n_splits=splits)
        sw = _sample_weight(y)
        oof = cross_val_predict(
            est, X, y, groups=groups, cv=gkf, method="predict_proba",
            params={"sample_weight": sw},
        )[:, 1]
        self.tau = _calibrate_tau(y, oof)

        pred = oof >= self.tau
        n_pos = int(y.sum())
        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        cv_recall = tp / n_pos if n_pos else 0.0
        cv_prec = tp / (tp + fp) if (tp + fp) else 0.0

        # Refit auf allen Daten
        self.estimator = _make_estimator(random_state)
        self.estimator.fit(X, y, sample_weight=sw)

        # Permutations-Importances (modell-agnostisch); in der Lernkurve
        # ueberspringbar, damit die Haupt-CSV nicht ueberschrieben wird.
        importances: list[tuple[str, float]] = []
        if write_importances:
            try:
                pi = permutation_importance(
                    self.estimator, X, y, n_repeats=5,
                    random_state=random_state, scoring="average_precision")
                imp = pi.importances_mean
                order = np.argsort(imp)[::-1]
                importances = [(self.feature_names[i], float(imp[i]))
                               for i in order]
                with open(out_dir / "feature_importances.csv", "w",
                          newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["feature", "importance"])
                    for name, val in importances:
                        w.writerow([name, f"{val:.6f}"])
            except Exception as exc:  # pragma: no cover
                print(f"WARN: Feature-Importances nicht berechenbar: {exc}")

        return TrainReport(
            tau=self.tau, cv_recall_pos=cv_recall, cv_precision_pos=cv_prec,
            n_rows=len(y), n_pos=n_pos, importances=importances,
        )

    # ------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """p(s) je Segment (Wahrscheinlichkeit fuer 'im Optimum')."""
        X = np.asarray(X, dtype=float)
        if X.shape[0] == 0:
            return np.zeros(0)
        return self.estimator.predict_proba(X)[:, 1]

    def select(self, X: np.ndarray, tau: float | None = None) -> np.ndarray:
        """Boolesche Auswahlmaske S = {s : p(s) >= tau}."""
        t = self.tau if tau is None else float(tau)
        return self.predict_proba(X) >= t


# ---------------------------------------------------------------------------
# Training / Laden mit Herkunftsstempel
# ---------------------------------------------------------------------------

def train_model(out_dir: Path | None = None, random_state: int = 0) -> dict:
    """Trainiert auf ``<out_dir>/dataset.npz`` und speichert
    ``surrogate_model.joblib`` + ``model_meta.json`` mit Herkunftsstempel
    (label_version, k_max, seed, Instanz-/Zeilenzahl, CV-Kennzahlen)."""
    out_dir = Path(out_dir) if out_dir else ARTIFACTS
    meta_path = out_dir / "dataset_meta.json"
    if not meta_path.exists():
        raise SystemExit(
            f"FEHLER: {meta_path} fehlt -- zuerst 'dataset --n ...' laufen lassen.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    P = _params()
    found = meta.get("stamp") or {}
    # erwartet: Physik/Katalog aus dem CODE, Segmentierung aus dem Datensatz
    expected = P.label_params(
        seg_divisor=float(meta.get("seg_divisor", P.SEG_DIVISOR_DEFAULT)),
        seg_min_spacings=float(meta.get("seg_min_spacings",
                                        P.SEG_MIN_SPACINGS_DEFAULT)))
    P.check_stamp(found, expected, "dataset_meta.json", strict_seg=True)
    d = np.load(out_dir / "dataset.npz", allow_pickle=True)
    model = SurrogateModel()
    rep = model.train(d["X"], d["y"], d["groups"], out_dir=out_dir,
                      random_state=random_state)
    stamp = {**P.stamp(expected), "pricing": meta.get("pricing", "exact"),
             "k_max": meta.get("k_max"), "seed": meta.get("seed"),
             "n_instances": meta.get("n_instances_used"), "n_rows": rep.n_rows,
             "tau": rep.tau, "cv_recall_pos": rep.cv_recall_pos,
             "cv_precision_pos": rep.cv_precision_pos}
    joblib.dump({"estimator": model.estimator, "tau": model.tau,
                 "feature_names": model.feature_names, **stamp},
                out_dir / MODEL_PATH.name)
    with open(out_dir / "model_meta.json", "w", encoding="utf-8") as f:
        json.dump({**stamp, "importances": rep.importances}, f, indent=2)
    print(f"Modell: {rep.n_rows} Zeilen, {rep.n_pos} positiv, "
          f"tau={rep.tau:.3f}, CV Recall {rep.cv_recall_pos:.3f} / "
          f"Precision {rep.cv_precision_pos:.3f} -> {out_dir / MODEL_PATH.name}")
    print("Top-Features:", ", ".join(
        f"{n}={v:.3f}" for n, v in rep.importances[:6]))
    return stamp


def load_model(path: Path | None = None) -> SurrogateModel:
    """Laedt das Modell; bricht hart ab, wenn LABEL_VERSION oder Physik
    nicht zum laufenden Code passen (stale) oder der Stempel fehlt; warnt,
    wenn nur Segmentierung/Katalog abweichen."""
    path = Path(path) if path else MODEL_PATH
    data = joblib.load(path)
    P = _params()
    P.check_stamp(data, P.label_params(), f"Modell {path.name}",
                  strict_seg=False)
    return SurrogateModel(estimator=data["estimator"], tau=data["tau"],
                          feature_names=data["feature_names"])


def main() -> None:
    ap = argparse.ArgumentParser(description="Surrogat-Modell trainieren.")
    ap.add_argument("--train", action="store_true",
                    help="Training auf <out>/dataset.npz starten")
    ap.add_argument("--out", type=str, default=None,
                    help="Ordner mit dataset.npz (Default: artifacts/)")
    args = ap.parse_args()
    if args.train:
        train_model(Path(args.out) if args.out else None)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
