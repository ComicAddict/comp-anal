"""Pattern-class labels and the class figures."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.classes import class_colors, class_labels, class_summary, make_class_outputs
from src.config import DEFAULT_CONFIG
from src.envelope import CATEGORICAL


def metrics() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(12):
        rows.append({"test_id": f"plain_{i}", "param_family": "plain", "param_tpms_type": np.nan})
    for kind in ("gyroid", "schwarz_p"):
        for i in range(6):
            rows.append({"test_id": f"{kind}_{i}", "param_family": "tpms", "param_tpms_type": kind})
    rows.append({"test_id": "pilot", "param_family": np.nan, "param_tpms_type": np.nan})
    frame = pd.DataFrame(rows)
    frame["modulus_MPa"] = rng.uniform(450, 650, len(frame))
    frame["plateau_stress_MPa"] = frame["modulus_MPa"] / 22 + rng.normal(0, 0.5, len(frame))
    frame["stress_at_10pct_MPa"] = frame["plateau_stress_MPa"] * 0.8
    return frame


def test_first_non_blank_column_labels_each_test():
    labels, primary, reference = class_labels(metrics(), ["tpms_type", "family"])
    assert primary == ["gyroid", "schwarz_p"] and reference == ["plain"]
    assert labels.iloc[0] == "plain" and labels.iloc[12] == "gyroid"
    assert labels.iloc[-1] == "", "a test in no class stays unlabelled"


def test_compared_classes_get_colours_and_references_grey():
    colors = class_colors(["gyroid", "schwarz_p"], ["plain"])
    assert colors["gyroid"] == CATEGORICAL[0] and colors["schwarz_p"] == CATEGORICAL[1]
    assert colors["plain"] not in CATEGORICAL


def test_a_single_class_is_not_a_comparison():
    frame = metrics()
    frame["param_tpms_type"] = np.nan
    frame["param_family"] = "plain"
    assert class_labels(frame, ["tpms_type", "family"]) is None


def test_summary_counts_and_medians():
    frame = metrics()
    labels, _, _ = class_labels(frame, ["tpms_type", "family"])
    table = class_summary(frame, labels, ["modulus_MPa"]).set_index("class")
    assert table.loc["plain", "n_tests"] == 12
    expected = frame.loc[labels == "gyroid", "modulus_MPa"].median()
    assert table.loc["gyroid", "modulus_MPa_median"] == expected


def test_outputs_are_written_and_cleared(tmp_path):
    cfg = DEFAULT_CONFIG.replace(class_by=("tpms_type", "family"))
    written = make_class_outputs(metrics(), tmp_path / "fig", tmp_path / "res", cfg)
    names = {p.name for p in written}
    assert names == {"class_summary.csv", "class_space.png", "class_comparison.png"}
    # Turned off, a later run removes them rather than leaving them stale.
    assert make_class_outputs(metrics(), tmp_path / "fig", tmp_path / "res", DEFAULT_CONFIG) == []
    assert not (tmp_path / "fig" / "class_space.png").exists()
