"""Design-space envelope: curve overlays and metric-space comparison.

Two views, per the project brief:
  (a) all stress-strain curves overlaid, with a shaded band showing the range
      of stress achievable at each strain across every pattern tested;
  (b) derived metrics plotted against each other, one point per test, with a
      convex hull marking the achievable property region.

Plot conventions
----------------
Colour identifies the pattern and is assigned from a fixed, CVD-validated
categorical order -- never a cycled or generated hue. Because pattern count is
open-ended and scatter plots put every pair of colours side by side, marker
shape and line style carry the same identity as a secondary encoding, so no
figure relies on colour alone. Figures render on a light surface; the
authoritative table view is `results/metrics_summary.csv`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")  # headless: figures are files, never windows

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG, AnalysisConfig
from .mechanics import StressStrainCurve, resample_curve

#: Validated categorical order (light surface). Assigned in this fixed order;
#: past 8 patterns the colour repeats but the marker/line style does not, so
#: the colour+shape pair stays unique.
CATEGORICAL: tuple[str, ...] = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
)
MARKERS: tuple[str, ...] = ("o", "s", "^", "D", "v", "P", "X", "*")
LINESTYLES: tuple[str, ...] = ("-", "--", "-.", ":")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e2e1dc"


@dataclass(frozen=True)
class PatternStyle:
    color: str
    marker: str
    linestyle: str


def assign_styles(patterns: Sequence[str]) -> dict[str, PatternStyle]:
    """Map pattern names to a stable colour/marker/linestyle triple.

    Sorted by name so a pattern keeps its appearance when other patterns are
    added or filtered out -- colour follows the entity, never its rank.
    """
    out: dict[str, PatternStyle] = {}
    for i, name in enumerate(sorted(dict.fromkeys(patterns))):
        out[name] = PatternStyle(
            color=CATEGORICAL[i % len(CATEGORICAL)],
            marker=MARKERS[(i // len(CATEGORICAL) + i) % len(MARKERS)],
            linestyle=LINESTYLES[(i // len(CATEGORICAL)) % len(LINESTYLES)],
        )
    return out


def _style_axes(
    ax: plt.Axes, title: str, xlabel: str, ylabel: str, subtitle: str = ""
) -> None:
    """Recessive grid and axes; ink for text, never the series colour.

    The title is padded enough to clear the subtitle line, which is drawn just
    above the axes -- otherwise the two overlap.
    """
    ax.set_title(title, color=INK, fontsize=12, pad=26 if subtitle else 12, loc="left")
    if subtitle:
        ax.text(
            0.0, 1.012, subtitle, transform=ax.transAxes,
            color=INK_MUTED, fontsize=9, va="bottom",
        )
    ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=10)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9, length=0)
    ax.set_facecolor(SURFACE)


def _new_figure(figsize: tuple[float, float]) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


# ----------------------------------------------------------------------
# (a) Stress-strain overlay + envelope
# ----------------------------------------------------------------------
@dataclass
class CurveMatrix:
    """All curves interpolated onto one common strain grid."""

    grid: np.ndarray                 # (n_points,)
    stress: np.ndarray               # (n_tests, n_points)
    test_ids: list[str]
    patterns: list[str]
    limiting_test: str = ""
    limiting_strain: float = float("nan")

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.stress.T, index=self.grid, columns=self.test_ids)


def build_curve_matrix(
    curves: Iterable[tuple[str, str, StressStrainCurve]],
    n_points: int = DEFAULT_CONFIG.envelope_n_points,
    strain_max: float | None = None,
) -> CurveMatrix:
    """Resample every curve onto a shared strain grid.

    The grid runs from 0 to the smallest maximum strain across all tests, so
    every curve is defined at every grid point and the envelope is a genuine
    min/max rather than an artefact of curves ending at different strains. The
    test that sets that limit is recorded, since one short test silently
    truncating the comparison is worth knowing about.
    """
    items = list(curves)
    if not items:
        raise ValueError("no curves to build an envelope from")

    max_strains = [float(c.max_strain) for _, _, c in items]
    if strain_max is None:
        limit_idx = int(np.argmin(max_strains))
        limit = max_strains[limit_idx]
        limiting_test = items[limit_idx][0]
    else:
        limit = float(strain_max)
        limiting_test = ""

    if not np.isfinite(limit) or limit <= 0:
        raise ValueError(f"common strain limit is not usable: {limit}")

    grid = np.linspace(0.0, limit, int(n_points))
    stress = np.vstack(
        [resample_curve(c.strain, c.stress_MPa, grid) for _, _, c in items]
    )
    return CurveMatrix(
        grid=grid,
        stress=stress,
        test_ids=[t for t, _, _ in items],
        patterns=[p for _, p, _ in items],
        limiting_test=limiting_test,
        limiting_strain=limit,
    )


def plot_stress_strain_envelope(
    matrix: CurveMatrix,
    out_path: str | Path,
    cfg: AnalysisConfig = DEFAULT_CONFIG,
    per_pattern_bands: bool | None = None,
    title: str = "Design-space envelope: engineering stress-strain",
) -> Path:
    """Overlay every curve, shade the achievable band across all patterns."""
    if per_pattern_bands is None:
        per_pattern_bands = cfg.envelope_per_pattern_bands

    styles = assign_styles(matrix.patterns)
    fig, ax = _new_figure((9.0, 6.0))

    # Global envelope: what is achievable at all, across the whole design space.
    with np.errstate(invalid="ignore"):
        lo = np.nanmin(matrix.stress, axis=0)
        hi = np.nanmax(matrix.stress, axis=0)
    ax.fill_between(
        matrix.grid,
        lo,
        hi,
        color=INK_MUTED,
        alpha=0.18,
        linewidth=0,
        label="Design-space envelope (all patterns)",
        zorder=1,
    )

    # Per-pattern envelope across replicates: separates pattern-to-pattern
    # spread from replicate-to-replicate scatter.
    if per_pattern_bands:
        for pattern in sorted(set(matrix.patterns)):
            rows = [i for i, p in enumerate(matrix.patterns) if p == pattern]
            if len(rows) < 2:
                continue
            block = matrix.stress[rows, :]
            with np.errstate(invalid="ignore"):
                ax.fill_between(
                    matrix.grid,
                    np.nanmin(block, axis=0),
                    np.nanmax(block, axis=0),
                    color=styles[pattern].color,
                    alpha=0.12,
                    linewidth=0,
                    zorder=2,
                )

    seen: set[str] = set()
    for row, (test_id, pattern) in enumerate(zip(matrix.test_ids, matrix.patterns)):
        style = styles[pattern]
        ax.plot(
            matrix.grid,
            matrix.stress[row],
            color=style.color,
            linestyle=style.linestyle,
            linewidth=1.4,
            alpha=0.85 if pattern not in seen else 0.55,
            label=pattern if pattern not in seen else None,
            zorder=3,
        )
        seen.add(pattern)

    subtitle = f"{len(matrix.test_ids)} tests · {len(seen)} patterns"
    if matrix.limiting_test:
        subtitle += f" · strain grid capped at {matrix.limiting_strain:.3f} (shortest test)"

    _style_axes(
        ax, title, "Engineering strain (mm/mm)", "Engineering stress (MPa)", subtitle
    )
    ax.set_xlim(0, matrix.grid[-1])
    ax.set_ylim(bottom=0)

    legend = ax.legend(
        frameon=False, fontsize=9, labelcolor=INK_SECONDARY,
        loc="upper left", ncol=1,
    )
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    return _save(fig, out_path, cfg)


# ----------------------------------------------------------------------
# (b) Metric-space scatter + achievable-property hull
# ----------------------------------------------------------------------
def _convex_hull_path(points: np.ndarray) -> np.ndarray | None:
    """Closed hull polygon, or None when a hull is not defined."""
    if len(points) < 3:
        return None
    try:
        from scipy.spatial import ConvexHull, QhullError
    except ImportError:  # pragma: no cover
        return None
    try:
        hull = ConvexHull(points)
    except Exception:  # collinear / degenerate input
        return None
    loop = np.append(hull.vertices, hull.vertices[0])
    return points[loop]


def plot_metric_space(
    metrics: pd.DataFrame,
    x: str,
    y: str,
    out_path: str | Path,
    cfg: AnalysisConfig = DEFAULT_CONFIG,
    hull: bool = True,
    labels: dict[str, str] | None = None,
) -> Path | None:
    """Scatter two metrics against each other, one point per test.

    The convex hull over all points is the achievable property envelope: any
    combination inside it has been realised by some pattern, and the boundary
    is the current limit of the design space.
    """
    labels = labels or {}
    for col in (x, y):
        if col not in metrics.columns:
            raise KeyError(f"metric {col!r} not in metrics table; have {list(metrics.columns)}")

    data = metrics[[c for c in ("test_id", "pattern_name", x, y) if c in metrics.columns]]
    data = data.dropna(subset=[x, y])
    if data.empty:
        return None

    styles = assign_styles(data["pattern_name"].astype(str).tolist())
    fig, ax = _new_figure((7.5, 6.0))

    if hull:
        path = _convex_hull_path(data[[x, y]].to_numpy(dtype=float))
        if path is not None:
            ax.fill(path[:, 0], path[:, 1], color=INK_MUTED, alpha=0.10, linewidth=0, zorder=1)
            ax.plot(
                path[:, 0], path[:, 1],
                color=INK_MUTED, linewidth=1.2, linestyle="--", alpha=0.8,
                label="Achievable envelope (convex hull)", zorder=2,
            )

    for pattern, block in data.groupby("pattern_name", sort=True):
        style = styles[str(pattern)]
        ax.scatter(
            block[x], block[y],
            s=70, color=style.color, marker=style.marker,
            edgecolors=SURFACE, linewidths=1.5,  # 2px surface ring on overlap
            label=str(pattern), zorder=3,
        )

    _style_axes(
        ax,
        f"Property space: {labels.get(y, y)} vs {labels.get(x, x)}",
        labels.get(x, x),
        labels.get(y, y),
        f"{len(data)} tests · one point per test · "
        f"{data['pattern_name'].nunique()} patterns",
    )
    legend = ax.legend(frameon=False, fontsize=9, loc="best")
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    return _save(fig, out_path, cfg)


def plot_metric_vs_param(
    metrics: pd.DataFrame,
    param: str,
    metric: str,
    out_path: str | Path,
    cfg: AnalysisConfig = DEFAULT_CONFIG,
    labels: dict[str, str] | None = None,
) -> Path | None:
    """Plot a metric against a numeric `pattern_params` value.

    Points are connected within a pattern family so a trend along the swept
    parameter is visible.
    """
    labels = labels or {}
    column = f"param_{param}"
    if column not in metrics.columns or metric not in metrics.columns:
        return None

    data = metrics[["test_id", "pattern_name", column, metric]].dropna(
        subset=[column, metric]
    )
    # A "trend" needs at least two values of the swept parameter; a single
    # value would produce a one-point plot showing nothing.
    if data.empty or data[column].nunique() < 2:
        return None

    styles = assign_styles(data["pattern_name"].astype(str).tolist())
    fig, ax = _new_figure((7.5, 5.5))

    for pattern, block in data.groupby("pattern_name", sort=True):
        style = styles[str(pattern)]
        block = block.sort_values(column)
        ax.plot(
            block[column], block[metric],
            color=style.color, marker=style.marker, linestyle=style.linestyle,
            linewidth=1.5, markersize=8, markeredgecolor=SURFACE,
            markeredgewidth=1.5, label=str(pattern), zorder=3,
        )

    _style_axes(
        ax,
        f"{labels.get(metric, metric)} vs {param}",
        param,
        labels.get(metric, metric),
    )
    legend = ax.legend(frameon=False, fontsize=9, loc="best")
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    return _save(fig, out_path, cfg)


def _save(fig: plt.Figure, out_path: str | Path, cfg: AnalysisConfig) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=cfg.figure_dpi, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out_path
