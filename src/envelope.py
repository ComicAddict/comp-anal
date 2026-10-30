"""Design-space envelope: curve overlays and metric-space comparison.

Two views, per the project brief:
  (a) all stress-strain curves overlaid, with a shaded band showing the range
      of stress achievable at each strain across every test;
  (b) derived metrics plotted against each other, one point per test, with a
      convex hull marking the achievable property region.

Plot conventions
----------------
Every test is a plain data point (or curve): one marker shape throughout,
because shape cannot usefully tell 64 patterns apart and only adds noise.

What colour encodes depends on what there is to encode:
  * colour-by a numeric parameter -> a continuous blue scale with a colorbar,
    which makes the figure itself a correlation view;
  * colour-by a categorical parameter, or by pattern, with at most
    `legend_max_patterns` levels -> a fixed, CVD-validated categorical order
    and one legend entry per level;
  * anything with more levels than that -> a single colour and no per-pattern
    legend, since a legend of dozens of near-identical swatches identifies
    nothing.

Text stays in ink colours, never series colours. Figures render on a light
surface; the authoritative table view is `results/metrics_summary.csv`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")  # headless: figures are files, never windows

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D

from .config import DEFAULT_CONFIG, AnalysisConfig
from .mechanics import StressStrainCurve, resample_curve

#: Validated categorical order (light surface), assigned in this fixed order.
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

#: One-hue blue ramp for continuous magnitude. Starts at step 250 rather than
#: the palest steps so the lowest values still stand off the light surface.
SEQUENTIAL: tuple[str, ...] = (
    "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
    "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
)

#: Blue (negative) <-> red (positive) around a neutral grey, for correlations.
#: The red arm is lightness-matched step for step to the blue arm (OKLab), so
#: +0.5 and -0.5 read as equally strong.
DIVERGING: tuple[str, ...] = (
    "#104281", "#2a78d6", "#6da7ec", "#b7d3f6",
    "#f0efec",
    "#f4c2be", "#e4857e", "#c74845", "#762221",
)

POINT_COLOR = CATEGORICAL[0]
MISSING_COLOR = "#c9c8c2"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e2e1dc"


def sequential_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL)


def diverging_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("div_blue_red", DIVERGING)


def assign_colors(levels: Sequence[str]) -> dict[str, str]:
    """Map level names to categorical colours in a stable order.

    Sorted by name so a level keeps its colour when others are added or
    filtered out -- colour follows the entity, never its rank.
    """
    ordered = sorted(dict.fromkeys(str(level) for level in levels))
    return {name: CATEGORICAL[i % len(CATEGORICAL)] for i, name in enumerate(ordered)}


def resolve_column(frame: pd.DataFrame, name: str | None) -> str | None:
    """Find the column a user means by `name`.

    Accepts a column as-is ("relative_density") or a parameter key without
    its prefix ("cell_mm" -> "param_cell_mm").
    """
    if not name:
        return None
    for candidate in (name, f"param_{name}"):
        if candidate in frame.columns:
            return candidate
    return None


def display_name(column: str) -> str:
    """Column name as shown on axes: parameters lose their `param_` prefix."""
    return column[len("param_"):] if column.startswith("param_") else column


# ----------------------------------------------------------------------
# Colour assignment
# ----------------------------------------------------------------------
@dataclass
class ColorSpec:
    """How the items in one figure are coloured, and how to explain it."""

    mode: str                                  # "continuous" | "levels" | "single"
    colors: list[Any]                          # one colour per item
    label: str = ""                            # colorbar / legend title
    legend: list[tuple[str, str]] = field(default_factory=list)
    cmap: Any = None
    norm: Any = None
    n_missing: int = 0
    by_pattern: bool = False
    note: str = ""


def color_spec(
    patterns: Sequence[str],
    cfg: AnalysisConfig = DEFAULT_CONFIG,
    values: Sequence[Any] | None = None,
    label: str = "",
) -> ColorSpec:
    """Decide how to colour `len(patterns)` items.

    `values`, when given, are one per item and take precedence over pattern.
    Numeric values get a continuous scale; labels get categorical colours
    when there are few enough levels to tell apart; otherwise everything is a
    single colour and `note` says why.
    """
    n = len(patterns)

    if values is not None:
        series = pd.Series(list(values), dtype=object)
        present = series.notna() & (series.astype(str).str.strip() != "")
        numeric = pd.to_numeric(series.where(present), errors="coerce")
        if present.any() and numeric[present].notna().all():
            finite = numeric[numeric.notna()].to_numpy(dtype=float)
            lo, hi = float(finite.min()), float(finite.max())
            if hi == lo:  # a constant still deserves a valid scale
                lo, hi = lo - 0.5, hi + 0.5
            cmap, norm = sequential_cmap(), Normalize(vmin=lo, vmax=hi)
            colors = [
                cmap(norm(v)) if np.isfinite(v) else MISSING_COLOR
                for v in numeric.to_numpy(dtype=float)
            ]
            return ColorSpec(
                "continuous", colors, label, cmap=cmap, norm=norm,
                n_missing=int((~numeric.notna()).sum()),
            )

        levels = sorted(series[present].astype(str).unique())
        if 0 < len(levels) <= cfg.legend_max_patterns:
            palette = assign_colors(levels)
            colors = [
                palette[str(v)] if keep else MISSING_COLOR
                for v, keep in zip(series, present)
            ]
            return ColorSpec(
                "levels", colors, label,
                legend=[(lvl, palette[lvl]) for lvl in levels],
                n_missing=int((~present).sum()),
            )
        return ColorSpec(
            "single", [POINT_COLOR] * n,
            note=(
                f"{label} has {len(levels)} levels -- too many to tell apart "
                f"by colour; drawn in one colour"
            ),
        )

    levels = sorted(dict.fromkeys(str(p) for p in patterns))
    if len(levels) <= cfg.legend_max_patterns:
        palette = assign_colors(levels)
        return ColorSpec(
            "levels", [palette[str(p)] for p in patterns], "Pattern",
            legend=[(lvl, palette[lvl]) for lvl in levels], by_pattern=True,
        )
    return ColorSpec("single", [POINT_COLOR] * n)


# ----------------------------------------------------------------------
# Shared figure styling
# ----------------------------------------------------------------------
def style_axes(
    ax: plt.Axes, title: str, xlabel: str, ylabel: str, subtitle: str = ""
) -> None:
    """Recessive grid and axes; ink for text, never the series colour.

    The title is padded enough to clear the subtitle line, which is drawn just
    above the axes -- otherwise the two overlap.
    """
    if title:
        ax.set_title(
            title, color=INK, fontsize=12, pad=26 if subtitle else 12, loc="left"
        )
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


def new_figure(figsize: tuple[float, float]) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def add_colorbar(fig: plt.Figure, ax: Any, spec: ColorSpec) -> None:
    """Colorbar for a continuous ColorSpec, styled to match the axes."""
    mappable = plt.cm.ScalarMappable(norm=spec.norm, cmap=spec.cmap)
    bar = fig.colorbar(mappable, ax=ax, pad=0.02, fraction=0.045)
    bar.set_label(spec.label, color=INK_SECONDARY, fontsize=9)
    bar.ax.tick_params(colors=INK_SECONDARY, labelsize=8, length=0)
    bar.outline.set_visible(False)


def style_legend(legend: Any) -> None:
    if legend is None:
        return
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)
    if legend.get_title() is not None:
        legend.get_title().set_color(INK_SECONDARY)
        legend.get_title().set_fontsize(8)


def level_handles(spec: ColorSpec, kind: str = "line") -> list[Line2D]:
    """Legend proxies for a categorical ColorSpec, plus "no value" if needed."""
    entries = list(spec.legend)
    if spec.n_missing:
        entries.append(("no value", MISSING_COLOR))
    if kind == "line":
        return [Line2D([], [], color=c, linewidth=2, label=t) for t, c in entries]
    return [
        Line2D([], [], color=c, marker="o", linestyle="", markersize=7,
               markeredgecolor=SURFACE, label=t)
        for t, c in entries
    ]


def save_figure(fig: plt.Figure, out_path: str | Path, cfg: AnalysisConfig) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=cfg.figure_dpi, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out_path


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
    color_values: Sequence[Any] | None = None,
    color_label: str = "",
) -> Path:
    """Overlay every curve and shade the stress range achieved at each strain.

    With many tests, individual curves become texture: what carries the
    information is the overall range, the middle-50% band and the median, so
    those are drawn on top. Colouring by a parameter (`color_values`) instead
    shows whether that parameter orders the curves -- a visual correlation.
    """
    if per_pattern_bands is None:
        per_pattern_bands = cfg.envelope_per_pattern_bands

    spec = color_spec(matrix.patterns, cfg, color_values, color_label)
    n_tests = len(matrix.test_ids)
    n_patterns = len(set(matrix.patterns))
    fig, ax = new_figure((9.0, 6.0))
    handles: list[Any] = []

    with np.errstate(invalid="ignore"), _quiet_nan_warnings():
        lo = np.nanmin(matrix.stress, axis=0)
        hi = np.nanmax(matrix.stress, axis=0)
    handles.append(
        ax.fill_between(
            matrix.grid, lo, hi, color=INK_MUTED, alpha=0.12, linewidth=0,
            label="Range across all tests", zorder=1,
        )
    )

    if spec.mode == "single":
        # Many tests: the distribution is the message, so draw it explicitly.
        if n_tests >= 8:
            with _quiet_nan_warnings():
                q25, q75 = np.nanpercentile(matrix.stress, [25, 75], axis=0)
            handles.append(
                ax.fill_between(
                    matrix.grid, q25, q75, color=INK_MUTED, alpha=0.38,
                    linewidth=0, label="Middle 50% of tests", zorder=2,
                )
            )
        alpha = float(np.clip(3.0 / np.sqrt(max(n_tests, 1)), 0.15, 0.85))
        for row in range(n_tests):
            ax.plot(
                matrix.grid, matrix.stress[row], color=POINT_COLOR,
                linewidth=0.9, alpha=alpha, zorder=3,
            )
        handles.append(
            Line2D([], [], color=POINT_COLOR, linewidth=1.2, alpha=max(alpha, 0.5),
                   label=f"Individual tests (n={n_tests})")
        )
        if n_tests >= 3:
            with _quiet_nan_warnings():
                median = np.nanmedian(matrix.stress, axis=0)
            (line,) = ax.plot(
                matrix.grid, median, color=INK, linewidth=2.0,
                label="Median across tests", zorder=4,
            )
            handles.append(line)
    else:
        if spec.by_pattern and per_pattern_bands:
            # Per-pattern replicate spread, separating pattern-to-pattern
            # differences from replicate-to-replicate noise.
            for pattern, color in spec.legend:
                rows = [i for i, p in enumerate(matrix.patterns) if str(p) == pattern]
                if len(rows) < 2:
                    continue
                block = matrix.stress[rows, :]
                with np.errstate(invalid="ignore"), _quiet_nan_warnings():
                    ax.fill_between(
                        matrix.grid, np.nanmin(block, axis=0), np.nanmax(block, axis=0),
                        color=color, alpha=0.12, linewidth=0, zorder=2,
                    )
        for row, color in enumerate(spec.colors):
            ax.plot(
                matrix.grid, matrix.stress[row], color=color,
                linewidth=1.3, alpha=0.85, zorder=3,
            )
        if spec.mode == "levels":
            handles += level_handles(spec, "line")
        elif spec.n_missing:
            handles.append(
                Line2D([], [], color=MISSING_COLOR, linewidth=2, label="no value")
            )

    subtitle = f"{n_tests} tests · {n_patterns} patterns"
    if spec.mode == "continuous":
        subtitle += f" · coloured by {spec.label}"
    if matrix.limiting_test:
        subtitle += f" · strain grid capped at {matrix.limiting_strain:.3f} (shortest test)"

    style_axes(
        ax, title, "Engineering strain (mm/mm)", "Engineering stress (MPa)", subtitle
    )
    ax.set_xlim(0, matrix.grid[-1])
    ax.set_ylim(bottom=0)

    legend = ax.legend(
        handles=handles, frameon=False, fontsize=8, loc="upper left",
        title=spec.label if spec.mode == "levels" else None,
        ncol=2 if len(handles) > 7 else 1,
    )
    style_legend(legend)
    if spec.mode == "continuous":
        add_colorbar(fig, ax, spec)

    return save_figure(fig, out_path, cfg)


# ----------------------------------------------------------------------
# (b) Metric-space scatter + achievable-property hull
# ----------------------------------------------------------------------
def _convex_hull_path(points: np.ndarray) -> np.ndarray | None:
    """Closed hull polygon, or None when a hull is not defined."""
    if len(points) < 3:
        return None
    try:
        from scipy.spatial import ConvexHull
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
    color_by: str | None = None,
) -> Path | None:
    """Scatter two metrics against each other, one point per test.

    The convex hull over all points is the achievable property envelope: any
    combination inside it has been realised by some test, and the boundary is
    the current limit of the design space.
    """
    labels = labels or {}
    for col in (x, y):
        if col not in metrics.columns:
            raise KeyError(f"metric {col!r} not in metrics table; have {list(metrics.columns)}")

    color_col = resolve_column(metrics, color_by)
    keep = [c for c in ("test_id", "pattern_name", x, y, color_col) if c and c in metrics.columns]
    data = metrics[list(dict.fromkeys(keep))].dropna(subset=[x, y])
    if data.empty:
        return None

    patterns = data["pattern_name"].astype(str).tolist() if "pattern_name" in data else ["-"] * len(data)
    spec = color_spec(
        patterns, cfg,
        data[color_col].tolist() if color_col else None,
        display_name(color_col) if color_col else "",
    )

    fig, ax = new_figure((7.5, 6.0))
    if hull:
        path = _convex_hull_path(data[[x, y]].to_numpy(dtype=float))
        if path is not None:
            ax.fill(path[:, 0], path[:, 1], color=INK_MUTED, alpha=0.10, linewidth=0, zorder=1)
            ax.plot(
                path[:, 0], path[:, 1], color=INK_MUTED, linewidth=1.2,
                linestyle="--", alpha=0.8, zorder=2,
            )

    ax.scatter(
        data[x], data[y], s=46, c=spec.colors, marker="o",
        edgecolors=SURFACE, linewidths=1.0, alpha=0.9, zorder=3,
    )

    n_patterns = len(set(patterns))
    subtitle = f"{len(data)} tests · {n_patterns} patterns"
    if hull and len(data) >= 3:
        subtitle += " · dashed outline = achievable envelope"
    if spec.mode == "continuous":
        subtitle += f" · coloured by {spec.label}"
    style_axes(
        ax,
        f"Property space: {labels.get(y, y)} vs {labels.get(x, x)}",
        labels.get(x, x),
        labels.get(y, y),
        subtitle,
    )

    if spec.mode == "levels":
        legend = ax.legend(
            handles=level_handles(spec, "marker"), frameon=False, fontsize=8,
            loc="best", title=spec.label,
            ncol=2 if len(spec.legend) > 5 else 1,
        )
        style_legend(legend)
    elif spec.mode == "continuous":
        add_colorbar(fig, ax, spec)

    return save_figure(fig, out_path, cfg)


class _quiet_nan_warnings:
    """Silence numpy's all-NaN-slice warnings for columns past a curve's end."""

    def __enter__(self):
        import warnings

        self._ctx = warnings.catch_warnings()
        self._ctx.__enter__()
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return self

    def __exit__(self, *exc):
        return self._ctx.__exit__(*exc)
