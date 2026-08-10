"""Envelope construction and figure generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import DEFAULT_CONFIG
from src.envelope import (
    CATEGORICAL,
    assign_styles,
    build_curve_matrix,
    plot_metric_space,
    plot_metric_vs_param,
    plot_stress_strain_envelope,
)
from src.mechanics import StressStrainCurve


def make_curve(test_id: str, scale: float, max_strain: float = 0.6) -> StressStrainCurve:
    strain = np.linspace(0, max_strain, 500)
    stress = scale * strain
    return StressStrainCurve(
        test_id=test_id,
        strain=strain,
        stress_MPa=stress,
        force_N=stress * 400,
        disp_mm=strain * 20,
        time_s=np.linspace(0, 60, 500),
    )


# ----------------------------------------------------------------- styles ----
def test_styles_are_stable_when_patterns_are_added():
    """Colour follows the entity, not its rank in the current selection."""
    first = assign_styles(["alpha", "gamma"])
    second = assign_styles(["alpha", "beta", "gamma"])
    assert first["alpha"].color == second["alpha"].color


def test_styles_use_the_fixed_categorical_order():
    styles = assign_styles(["a", "b", "c"])
    assert [styles[k].color for k in ("a", "b", "c")] == list(CATEGORICAL[:3])


def test_many_patterns_stay_distinguishable_by_shape():
    """Past 8 patterns the colour repeats, so the marker must not."""
    names = [f"p{i:02d}" for i in range(12)]
    styles = assign_styles(names)
    pairs = {(s.color, s.marker) for s in styles.values()}
    assert len(pairs) == len(names), "two patterns share both colour and marker"


# ----------------------------------------------------------------- matrix ----
def test_grid_stops_at_the_shortest_test():
    curves = [
        ("a", "p1", make_curve("a", 10, max_strain=0.6)),
        ("b", "p1", make_curve("b", 12, max_strain=0.35)),
    ]
    matrix = build_curve_matrix(curves, n_points=100)
    assert matrix.grid[-1] == pytest.approx(0.35)
    assert matrix.limiting_test == "b"
    assert np.isfinite(matrix.stress).all(), "every curve must span the whole grid"


def test_grid_can_be_capped_explicitly():
    curves = [("a", "p1", make_curve("a", 10))]
    matrix = build_curve_matrix(curves, n_points=50, strain_max=0.25)
    assert matrix.grid[-1] == pytest.approx(0.25)


def test_resampling_preserves_values():
    curves = [("a", "p1", make_curve("a", 10.0, max_strain=0.5))]
    matrix = build_curve_matrix(curves, n_points=101)
    assert matrix.stress[0, 50] == pytest.approx(10.0 * matrix.grid[50], rel=1e-6)


def test_envelope_spans_min_to_max_across_patterns():
    curves = [
        ("a", "p1", make_curve("a", 5.0)),
        ("b", "p2", make_curve("b", 15.0)),
        ("c", "p3", make_curve("c", 10.0)),
    ]
    matrix = build_curve_matrix(curves, n_points=100)
    lo = np.nanmin(matrix.stress, axis=0)
    hi = np.nanmax(matrix.stress, axis=0)
    assert np.allclose(lo, 5.0 * matrix.grid)
    assert np.allclose(hi, 15.0 * matrix.grid)


def test_empty_input_is_rejected():
    with pytest.raises(ValueError, match="no curves"):
        build_curve_matrix([])


# ---------------------------------------------------------------- figures ----
def test_stress_strain_figure_is_written(tmp_path):
    curves = [
        ("a", "solid", make_curve("a", 10)),
        ("b", "solid", make_curve("b", 11)),
        ("c", "gyroid", make_curve("c", 5)),
    ]
    matrix = build_curve_matrix(curves, n_points=200)
    out = plot_stress_strain_envelope(matrix, tmp_path / "env.png", DEFAULT_CONFIG)
    assert out.exists() and out.stat().st_size > 5000


def test_metric_space_figure_is_written(tmp_path):
    metrics = pd.DataFrame(
        {
            "test_id": list("abcde"),
            "pattern_name": ["solid", "solid", "gyroid", "gyroid", "honeycomb"],
            "modulus_MPa": [60.0, 62.0, 25.0, 27.0, 40.0],
            "plateau_stress_MPa": [2.3, 2.4, 1.1, 1.2, 1.8],
        }
    )
    out = plot_metric_space(
        metrics, "modulus_MPa", "plateau_stress_MPa", tmp_path / "ms.png", DEFAULT_CONFIG
    )
    assert out is not None and out.exists()


def test_metric_space_skips_all_nan_metrics(tmp_path):
    metrics = pd.DataFrame(
        {
            "test_id": ["a", "b"],
            "pattern_name": ["solid", "solid"],
            "modulus_MPa": [60.0, 62.0],
            "sea_J_per_g": [np.nan, np.nan],
        }
    )
    out = plot_metric_space(
        metrics, "modulus_MPa", "sea_J_per_g", tmp_path / "skip.png", DEFAULT_CONFIG
    )
    assert out is None


def test_metric_space_survives_collinear_points(tmp_path):
    """A degenerate hull must not crash the figure."""
    metrics = pd.DataFrame(
        {
            "test_id": ["a", "b", "c"],
            "pattern_name": ["p", "p", "p"],
            "x": [1.0, 2.0, 3.0],
            "y": [1.0, 2.0, 3.0],
        }
    )
    out = plot_metric_space(metrics, "x", "y", tmp_path / "col.png", DEFAULT_CONFIG)
    assert out is not None and out.exists()


def test_unknown_metric_raises(tmp_path):
    metrics = pd.DataFrame({"test_id": ["a"], "pattern_name": ["p"], "x": [1.0]})
    with pytest.raises(KeyError, match="nope"):
        plot_metric_space(metrics, "x", "nope", tmp_path / "x.png", DEFAULT_CONFIG)


def test_metric_vs_param_figure(tmp_path):
    metrics = pd.DataFrame(
        {
            "test_id": list("abcd"),
            "pattern_name": ["gyroid"] * 4,
            "param_cell_mm": [4.0, 5.0, 6.0, 7.0],
            "modulus_MPa": [60.0, 45.0, 33.0, 25.0],
        }
    )
    out = plot_metric_vs_param(
        metrics, "cell_mm", "modulus_MPa", tmp_path / "p.png", DEFAULT_CONFIG
    )
    assert out is not None and out.exists()


def test_metric_vs_param_returns_none_when_absent(tmp_path):
    metrics = pd.DataFrame({"test_id": ["a"], "pattern_name": ["p"], "modulus_MPa": [1.0]})
    assert (
        plot_metric_vs_param(
            metrics, "missing", "modulus_MPa", tmp_path / "n.png", DEFAULT_CONFIG
        )
        is None
    )
