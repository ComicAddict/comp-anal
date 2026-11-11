"""Searching the baselines' scale for the best match to the plain set."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import DEFAULT_CONFIG
from src.scale_match import ALL, analyse, robust_spread, summarise, write_outputs

TYPES = ("gyroid", "schwarz_p", "lidinoid", "split_p")
CFG = DEFAULT_CONFIG.replace(scale_match_column="scale", scale_match_window=(3.0, 6.0))
GRID = np.linspace(0, 0.35, 200)


def curve(strength: float, hardening: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Same shape at every strength, so only magnitude can tell tests apart."""
    return GRID, strength * (1 - np.exp(-GRID / 0.02) + hardening * GRID ** 2)


def project(match_at: float = 4.5, offset: float = 0.0, seed: int = 0):
    """Plain around strength 20; TPMS strength rises with scale, = 20 at `match_at`."""
    rng = np.random.default_rng(seed)
    rows, curves = [], {}
    for i in range(40):
        k = 20 * rng.normal(1, 0.08)
        rows.append({"test_id": f"plain_{i}", "param_family": "plain", "k": k})
    for t in TYPES:
        for scale in range(1, 9):
            k = 20 * (1 + 0.06 * (scale - match_at)) * rng.normal(1, 0.02) + offset
            rows.append({"test_id": f"{t}_{scale}", "param_family": "tpms",
                         "param_tpms_type": t, "param_scale": scale, "k": k,
                         "param_slicer_mass_g": 4 + 0.1 * scale if t == "lidinoid" else 4 + rng.normal(0, .05)})
    frame = pd.DataFrame(rows)
    frame["modulus_MPa"] = 30 * frame["k"]
    frame["plateau_stress_MPa"] = 1.1 * frame["k"]
    frame["stress_at_10pct_MPa"] = frame["k"]
    frame["stress_at_20pct_MPa"] = 1.2 * frame["k"]
    frame["energy_to_20pct_MJ_m3"] = 0.19 * frame["k"]
    for _, r in frame.iterrows():
        curves[r["test_id"]] = curve(r["k"])
    return frame, curves


@pytest.fixture(scope="module")
def matched():
    frame, curves = project()
    return frame, analyse(frame, curves, CFG, n_boot=150)


def test_the_planted_scale_is_found_with_an_interval(matched):
    frame, result = matched
    pooled = result.best.set_index("group").loc[ALL]
    # Where the TPMS trend truly meets the plain population (median 20); the
    # sample median of 40 noisy plain tests is off a little, and the interval
    # must allow for that as well as for the fit.
    assert pooled["best_scale"] == pytest.approx(4.5, abs=0.5)
    assert pooled["best_scale_lo"] <= 4.5 <= pooled["best_scale_hi"]
    sample = frame.loc[frame["param_family"] == "plain", "stress_at_10pct_MPa"].median()
    assert pooled["best_scale"] == pytest.approx(4.5 + (sample / 20 - 1) / 0.06, abs=0.15)
    assert pooled["distance_at_best"] < 1
    assert pooled["in_window"]


def test_each_type_is_searched_too(matched):
    _, result = matched
    assert set(result.best["group"]) == set(TYPES) | {ALL}


def test_crossings_of_the_plain_median(matched):
    _, result = matched
    row = result.crossings[(result.crossings["group"] == ALL)
                           & (result.crossings["property"] == "modulus_MPa")].iloc[0]
    assert float(row["crossing_scales"]) == pytest.approx(4.5, abs=0.5)


def test_identical_shapes_do_not_pick_a_scale(matched):
    _, result = matched
    assert "shape does not pick a scale" in summarise(result)


def test_mass_changing_with_scale_is_flagged(matched):
    _, result = matched
    assert any(note.startswith("lidinoid") for note in result.mass_notes)


def test_no_match_is_said_plainly():
    frame, curves = project(match_at=4.5, offset=25.0)   # every scale far stronger
    result = analyse(frame, curves, CFG, n_boot=50)
    text = summarise(result)
    assert "no scale matches" in text
    row = result.crossings[(result.crossings["group"] == ALL)].iloc[0]
    assert row["never"] == "above"


def test_works_without_curves():
    frame, _ = project()
    result = analyse(frame, None, CFG, n_boot=50)
    assert result.specimens["distance_shape"].isna().all()
    assert result.best.set_index("group").loc[ALL, "best_scale"] == pytest.approx(4.5, abs=0.5)


def test_robust_spread_matches_the_sd_for_normal_data():
    values = np.random.default_rng(1).normal(0, 2.0, 4000)
    assert robust_spread(values) == pytest.approx(2.0, rel=0.05)


def test_outputs(matched, tmp_path):
    frame, result = matched
    written = {p.name for p in write_outputs(result, frame, tmp_path / "t", tmp_path / "f", CFG)}
    assert {"best_scale.csv", "by_scale.csv", "summary.txt", "scale_trends.png",
            "scale_distance.png", "scale_curves.png"} <= written


def test_missing_reference_is_an_error():
    frame, _ = project()
    with pytest.raises(ValueError, match="reference"):
        analyse(frame, None, CFG.replace(scale_match_reference="family=woven"))
