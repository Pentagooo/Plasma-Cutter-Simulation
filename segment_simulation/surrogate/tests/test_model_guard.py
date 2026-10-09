"""Model guard: ``load_model`` aborts on a different or missing
``LABEL_VERSION`` or on different physics (``phys_hash``); a different
segmentation (``params_hash``) only gives a warning.
"""

from __future__ import annotations

import joblib
import pytest

from plasma_cutter.segment_simulation.surrogate.model import load_model
from plasma_cutter.segment_simulation.surrogate.params import (
    LABEL_VERSION, label_params, stamp,
)


def _dump(path, **over):
    data = {"estimator": None, "tau": 0.42, "feature_names": [],
            **stamp(label_params()), **over}
    joblib.dump(data, path)
    return path


def test_load_accepts_current_stamp(tmp_path):
    m = load_model(_dump(tmp_path / "model.joblib"))
    assert m.tau == pytest.approx(0.42)


def test_load_warns_but_accepts_other_segmentation(tmp_path, capsys):
    fine = stamp(label_params(seg_divisor=24.0, seg_min_spacings=3.0))
    m = load_model(_dump(tmp_path / "fine.joblib", **fine))
    assert m.tau == pytest.approx(0.42)
    assert "WARNUNG" in capsys.readouterr().out


def test_load_rejects_stale_version(tmp_path):
    with pytest.raises(SystemExit):
        load_model(_dump(tmp_path / "stale.joblib",
                         label_version=LABEL_VERSION - 1))


def test_load_rejects_other_physics(tmp_path):
    with pytest.raises(SystemExit):
        load_model(_dump(tmp_path / "phys.joblib", phys_hash="deadbeef"))


def test_load_rejects_unversioned(tmp_path):
    path = tmp_path / "old.joblib"
    joblib.dump({"estimator": None, "tau": 0.5, "feature_names": []}, path)
    with pytest.raises(SystemExit):
        load_model(path)
