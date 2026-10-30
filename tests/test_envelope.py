"""Envelope construction and figure generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import DEFAULT_CONFIG
import src.envelope as envelope
from src.envelope import (
    CATEGORICAL,
    assign_colors,
    build_curve_matrix,
    color_spec,
    plot_metric_space,
    plot_stress_strain_envelope,
    resolve_column,
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


# ----------------------------------------------------------------- colour ----
def test_colors_are_stable_when_patterns_are_added():
    """Colour follows the entity, not its rank in the current selection."""
    first = assign_colors(["alpha", "gamma"])
    second = assign_colors(["alpha", "beta", "gamma"])
    assert first["alpha"] == second["alpha"]


def test_colors_use_the_fixed_categorical_order():
    colors = assign_colors(["a", "b", "c"])
    assert [colors[k] for k in ("a", "b", "c")] == list(CATEGORICAL[:3])


def test_few_patterns_are_coloured_by_pattern():
    spec = color_spec(["a", "b", "a"], DEFAULT_CONFIG)
    assert spec.mode == "levels" and spec.by_pattern
    assert [label for label, _ in spec.legend] == ["a", "b"]


def test_many_patterns_become_plain_data_points():
    """64 patterns cannot be told apart by colour: one colour, no legend."""
    names = [f"pattern_{i:02d}" for i in range(64)]
    spec = color_spec(names, DEFAULT_CONFIG)
    assert spec.mode == "single"
    assert spec.legend == []
    assert len(set(spec.colors)) == 1


def test_numeric_color_by_gives_a_continuous_scale():
    values = np.linspace(0.1, 0.4, 64)
    spec = color_spec([f"p{i}" for i in range(64)], DEFAULT_CONFIG, values, "relative_density")
    assert spec.mode == "continuous"
    assert spec.norm.vmin == pytest.approx(0.1) and spec.norm.vmax == pytest.approx(0.4)
    # Darker for larger values on the one-hue ramp.
    assert np.mean(spec.colors[-1][:3]) < np.mean(spec.colors[0][:3])


def test_categorical_color_by_uses_levels_not_tests():
    values = ["XY", "Z"] * 32
    spec = color_spec([f"p{i}" for i in range(64)], DEFAULT_CONFIG, values, "orientation")
    assert spec.mode == "levels" and not spec.by_pattern
    assert [label for label, _ in spec.legend] == ["XY", "Z"]


def test_missing_color_values_are_shown_as_missing():
    spec = color_spec(["a", "b", "c"], DEFAULT_CONFIG, [0.1, None, 0.3], "rd")
    assert spec.mode == "continuous"
    assert spec.n_missing == 1
    assert spec.colors[1] == envelope.MISSING_COLOR


def test_resolve_column_accepts_bare_parameter_names():
    frame = pd.DataFrame(columns=["relative_density", "param_cell_mm"])
    assert resolve_column(frame, "cell_mm") == "param_cell_mm"
    assert resolve_column(frame, "relative_density") == "relative_density"
    assert resolve_column(frame, "nope") is None


def capture_figure(monkeypatch):
    """Intercept save_figure so a test can inspect the figure it was given."""
    captured = {}

    def fake_save(fig, out_path, cfg):
        captured["fig"] = fig
        return out_path

    monkeypatch.setattr(envelope, "save_figure", fake_save)
    return captured


def legend_labels(fig) -> list[str]:
    legend = fig.axes[0].get_legend()
    return [] if legend is None else [t.get_text() for t in legend.get_texts()]


def test_overlay_of_64_patterns_has_a_short_legend(monkeypatch, tmp_path):
    captured = capture_figure(monkeypatch)
    curves = [(f"t{i}", f"pattern_{i}", make_curve(f"t{i}", 5 + i * 0.3)) for i in range(64)]
    plot_stress_strain_envelope(build_curve_matrix(curves, 100), tmp_path / "e.png")

    labels = legend_labels(captured["fig"])
    assert len(labels) <= 5, labels
    assert not any(label.startswith("pattern_") for label in labels)
    assert "Median across tests" in labels


def test_scatter_of_64_patterns_uses_one_marker_and_no_legend(monkeypatch, tmp_path):
    captured = capture_figure(monkeypatch)
    metrics = pd.DataFrame(
        {
            "test_id": [f"t{i}" for i in range(64)],
            "pattern_name": [f"pattern_{i}" for i in range(64)],
            "modulus_MPa": np.linspace(10, 90, 64),
            "plateau_stress_MPa": np.linspace(0.5, 3.0, 64),
        }
    )
    plot_metric_space(metrics, "modulus_MPa", "plateau_stress_MPa", tmp_path / "m.png")
    fig = captured["fig"]
    assert fig.axes[0].get_legend() is None
    collections = [c for c in fig.axes[0].collections if len(c.get_offsets()) == 64]
    assert len(collections) == 1, "all tests should be one set of identical markers"


def test_color_by_adds_a_colorbar_not_a_legend(monkeypatch, tmp_path):
    captured = capture_figure(monkeypatch)
    metrics = pd.DataFrame(
        {
            "test_id": [f"t{i}" for i in range(20)],
            "pattern_name": [f"p{i}" for i in range(20)],
            "modulus_MPa": np.linspace(10, 90, 20),
            "plateau_stress_MPa": np.linspace(0.5, 3.0, 20),
            "param_cell_mm": np.repeat([4.0, 5.0, 6.0, 8.0], 5),
        }
    )
    plot_metric_space(
        metrics, "modulus_MPa", "plateau_stress_MPa", tmp_path / "m.png",
        color_by="cell_mm",
    )
    fig = captured["fig"]
    assert len(fig.axes) == 2, "expected the plot plus a colorbar"
    assert fig.axes[0].get_legend() is None


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


# ------------------------------------------------------ densification check ----
from src.envelope import densification_trace, plot_densification_check  # noqa: E402
from src.synthetic import FoamCurveSpec, foam_stress  # noqa: E402


def foam_curve(test_id: str, max_strain: float, spec: FoamCurveSpec | None = None) -> StressStrainCurve:
    strain = np.linspace(0, max_strain, 3000)
    stress = foam_stress(strain, spec or FoamCurveSpec(peak_stress_MPa=None))
    return StressStrainCurve(test_id, strain, stress, stress * 256, strain * 16,
                             np.linspace(0, 100, strain.size))


def test_gauge_is_one_half_in_the_linear_elastic_region():
    """sigma = E*eps gives W = E*eps^2/2, so r = E * W / sigma^2 = 1/2 exactly."""
    trace = densification_trace(foam_curve("t", 0.6))
    # Knee at 2.15 / 60 = 0.036; keep clear of it by the derivative window
    # (0.01 strain), which otherwise blends the knee into the estimate.
    elastic = (trace.strain > 0.008) & (trace.strain < 0.024)
    assert np.nanmean(trace.ratio[elastic]) == pytest.approx(0.5, abs=0.01)


def test_gauge_crosses_one_after_densification_and_not_before():
    full = densification_trace(foam_curve("full", 0.6))     # densifies at 0.42
    cut = densification_trace(foam_curve("cut", 0.35))      # stops in the plateau
    assert full.reached
    crossing = full.strain[np.flatnonzero(full.ratio >= 1.0)[0]]
    assert 0.42 < crossing < 0.55
    assert not cut.reached and cut.final_ratio < 1.0


def test_densification_check_figure_states_the_verdict(monkeypatch, tmp_path):
    captured = capture_figure(monkeypatch)
    plot_densification_check([foam_curve("a", 0.35), foam_curve("b", 0.6)], tmp_path / "d.png",
                             load_limited=True)
    title = captured["fig"].axes[0].get_title(loc="left")
    assert "1 of 2 tests reached densification" in title
    assert "the others stop at r =" in title and "load limit" in title
