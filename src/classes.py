"""Property space by pattern class: which families of pattern land where.

Each test gets one class label from `cfg.class_by`, a list of columns tried in
order: with ``["tpms_type", "family"]`` a TPMS specimen is labelled by its
type (gyroid, schwarz_p, ...) and a plain one, having no type, by its family
("plain"). Levels of the first column get the categorical colours; levels
that only come from a later column are reference groups and are drawn in
neutral grey, behind, so the classes being compared stay in front.

Three outputs:
  * ``class_space.png`` -- small multiples of property pairs, every test a
    point coloured by class, each class's median marked;
  * ``class_comparison.png`` -- one strip per class for each property, sorted
    by median, so classes can be ranked and their spread compared directly;
  * ``class_summary.csv`` -- n, median and interquartile range per class and
    property, the table behind both.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .config import DEFAULT_CONFIG, AnalysisConfig
from .envelope import (
    CATEGORICAL,
    GRID,
    INK,
    INK_MUTED,
    INK_SECONDARY,
    MISSING_COLOR,
    SURFACE,
    display_name,
    resolve_column,
    save_figure,
    style_axes,
    style_legend,
)

#: Neutral fills for reference groups (levels from a fallback column).
REFERENCE_GREYS: tuple[str, ...] = ("#9d9b94", "#c9c8c2", "#6f6d67")

#: Property pairs for the class-space panels, as (x, y). Parameters may be
#: named without their param_ prefix. Pairs with too few values are skipped.
CLASS_PAIRS: tuple[tuple[str, str], ...] = (
    ("modulus_MPa", "plateau_stress_MPa"),
    ("stress_at_10pct_MPa", "stress_at_20pct_MPa"),
    ("plateau_stress_MPa", "energy_to_20pct_MJ_m3"),
    ("modulus_MPa", "energy_to_20pct_MJ_m3"),
    ("slicer_mass_g", "plateau_stress_MPa"),
    ("mass_g", "plateau_stress_MPa"),
    ("relative_density", "modulus_MPa"),
)

#: Properties compared class by class.
CLASS_PROPERTIES: tuple[str, ...] = (
    "modulus_MPa",
    "plateau_stress_MPa",
    "stress_at_10pct_MPa",
    "stress_at_20pct_MPa",
    "energy_to_20pct_MJ_m3",
    "sea_to_20pct_J_per_g",
)

_LABELS = {
    "modulus_MPa": "Modulus E (MPa)",
    "plateau_stress_MPa": "Plateau σ (MPa)",
    "first_peak_stress_MPa": "First-peak σ (MPa)",
    "stress_at_10pct_MPa": "σ at 10% (MPa)",
    "stress_at_20pct_MPa": "σ at 20% (MPa)",
    "energy_to_20pct_MJ_m3": "W to 20% (MJ/m³)",
    "sea_to_20pct_J_per_g": "SEA to 20% (J/g)",
    "relative_density": "Relative density (–)",
    "mass_g": "Mass (g)",
}


def _label(column: str) -> str:
    if column in _LABELS:
        return _LABELS[column]
    name = display_name(column)
    if name.endswith("_g"):
        return f"{name[:-2].replace('_', ' ')} (g)"
    return name.replace("_", " ")


def _text(series: pd.Series) -> pd.Series:
    """Cells as stripped text, missing as ''. (pandas 3 keeps NaN through astype(str).)"""
    return series.where(series.notna(), "").astype(str).str.strip().replace("nan", "")


def class_labels(
    metrics: pd.DataFrame, columns: Sequence[str]
) -> tuple[pd.Series, list[str], list[str]] | None:
    """Per-test class, the levels to compare, and the reference levels.

    Returns None when no column is found or fewer than two classes result.
    """
    resolved = [c for c in (resolve_column(metrics, name) for name in columns) if c]
    if not resolved:
        return None
    labels = pd.Series("", index=metrics.index, dtype=object)
    primary: list[str] = []
    reference: list[str] = []
    for i, column in enumerate(resolved):
        values = _text(metrics[column])
        fill = (labels == "") & (values != "")
        labels[fill] = values[fill]
        levels = sorted(set(values[fill]))
        (primary if i == 0 else reference).extend(l for l in levels if l not in primary + reference)
    if len(primary) + len(reference) < 2:
        return None
    return labels, primary, reference


def class_colors(primary: Sequence[str], reference: Sequence[str]) -> dict[str, str]:
    colors = {name: CATEGORICAL[i % len(CATEGORICAL)] for i, name in enumerate(primary)}
    colors.update({name: REFERENCE_GREYS[i % len(REFERENCE_GREYS)] for i, name in enumerate(reference)})
    return colors


def class_summary(
    metrics: pd.DataFrame, labels: pd.Series, properties: Sequence[str]
) -> pd.DataFrame:
    rows = []
    for name in sorted(set(labels) - {""}):
        block = metrics[labels == name]
        row: dict[str, object] = {"class": name, "n_tests": len(block)}
        for prop in properties:
            values = pd.to_numeric(block[prop], errors="coerce").dropna()
            row[f"{prop}_n"] = len(values)
            row[f"{prop}_median"] = values.median() if len(values) else np.nan
            row[f"{prop}_q1"] = values.quantile(0.25) if len(values) else np.nan
            row[f"{prop}_q3"] = values.quantile(0.75) if len(values) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _legend_handles(order: Sequence[str], colors: dict[str, str], counts: pd.Series) -> list[Line2D]:
    return [
        Line2D([], [], color=colors[name], marker="o", linestyle="", markersize=7,
               markeredgecolor=SURFACE, label=f"{name}  (n = {int(counts.get(name, 0))})")
        for name in order
    ]


def _titles(fig: plt.Figure, height_in: float, title: str, subtitle: str) -> None:
    """Title and subtitle a fixed distance from the top, whatever the height."""
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.75 / height_in))
    fig.text(0.01, 1 - 0.12 / height_in, title, color=INK, fontsize=13, ha="left", va="top")
    fig.text(0.01, 1 - 0.42 / height_in, subtitle, color=INK_MUTED, fontsize=9, ha="left", va="top")


def plot_class_space(
    metrics: pd.DataFrame,
    labels: pd.Series,
    primary: Sequence[str],
    reference: Sequence[str],
    out_path: str | Path,
    cfg: AnalysisConfig = DEFAULT_CONFIG,
    pairs: Sequence[tuple[str, str]] = CLASS_PAIRS,
) -> Path | None:
    usable = []
    for x_name, y_name in pairs:
        x, y = resolve_column(metrics, x_name), resolve_column(metrics, y_name)
        if not x or not y:
            continue
        xy = metrics[[x, y]].apply(pd.to_numeric, errors="coerce")
        ok = xy.notna().all(axis=1) & (labels != "")
        if ok.sum() >= 3 and xy.loc[ok, x].nunique() > 1 and xy.loc[ok, y].nunique() > 1:
            usable.append((x, y, xy))
    usable = usable[:6]
    if not usable:
        return None

    colors = class_colors(primary, reference)
    ncols = 2 if len(usable) > 1 else 1
    nrows = int(np.ceil(len(usable) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols + 2.6, 4.4 * nrows), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    # Reference groups first, so the compared classes sit on top.
    draw_order = list(reference) + list(primary)
    for ax, (x, y, xy) in zip(axes.flat, usable):
        for name in draw_order:
            mask = (labels == name) & xy.notna().all(axis=1)
            if not mask.any():
                continue
            is_ref = name in reference
            ax.scatter(xy.loc[mask, x], xy.loc[mask, y], s=22, color=colors[name],
                       alpha=0.55 if is_ref else 0.8, edgecolors="none", zorder=2 if is_ref else 3)
            ax.scatter([xy.loc[mask, x].median()], [xy.loc[mask, y].median()], s=95,
                       color=colors[name], edgecolors=INK, linewidths=1.1, zorder=4)
        n = int(((labels != "") & xy.notna().all(axis=1)).sum())
        style_axes(ax, "", _label(x), _label(y))
        ax.text(0.99, 0.02, f"n = {n}", transform=ax.transAxes, ha="right", va="bottom",
                color=INK_MUTED, fontsize=8)
    for ax in list(axes.flat)[len(usable):]:
        ax.set_visible(False)

    counts = labels[labels != ""].value_counts()
    handles = _legend_handles(list(primary) + list(reference), colors, counts)
    handles.append(Line2D([], [], color=MISSING_COLOR, marker="o", linestyle="", markersize=9,
                          markeredgecolor=INK, label="class median"))
    legend = fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5),
                        frameon=False, fontsize=9, title="class")
    style_legend(legend)
    _titles(fig, 4.4 * nrows, "Where each pattern class sits in property space",
            "Every test is a point; the outlined point is the class median. Grey: reference group.")
    return save_figure(fig, out_path, cfg)


def plot_class_comparison(
    metrics: pd.DataFrame,
    labels: pd.Series,
    primary: Sequence[str],
    reference: Sequence[str],
    out_path: str | Path,
    cfg: AnalysisConfig = DEFAULT_CONFIG,
    properties: Sequence[str] = CLASS_PROPERTIES,
) -> Path | None:
    props = []
    for name in properties:
        column = resolve_column(metrics, name)
        if column and pd.to_numeric(metrics[column], errors="coerce")[labels != ""].notna().sum() >= 3:
            props.append(column)
    if not props:
        return None

    colors = class_colors(primary, reference)
    classes = list(primary) + list(reference)
    # One row order for every panel, by the first property's median, so a
    # class is found on the same line throughout.
    first = pd.to_numeric(metrics[props[0]], errors="coerce")
    medians = {c: first[labels == c].median() for c in classes}
    order = sorted(classes, key=lambda c: (np.isnan(medians[c]), medians[c] if not np.isnan(medians[c]) else 0))
    ypos = {c: i for i, c in enumerate(order)}
    rng = np.random.default_rng(0)

    ncols = min(3, len(props))
    nrows = int(np.ceil(len(props) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 0.36 * len(order) * nrows + 1.3 * nrows),
                             squeeze=False, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, prop in zip(axes.flat, props):
        values = pd.to_numeric(metrics[prop], errors="coerce")
        for name in order:
            v = values[(labels == name) & values.notna()]
            if v.empty:
                continue
            y = ypos[name] + rng.uniform(-0.18, 0.18, len(v))
            ax.scatter(v, y, s=16, color=colors[name], alpha=0.75, edgecolors="none", zorder=3)
            q1, med, q3 = v.quantile([0.25, 0.5, 0.75])
            ax.plot([q1, q3], [ypos[name]] * 2, color=INK, linewidth=1.4, zorder=4)
            ax.plot([med], [ypos[name]], marker="|", markersize=14, markeredgewidth=2.2, color=INK, zorder=5)
        style_axes(ax, "", _label(prop), "")
        ax.grid(True, axis="x", color=GRID)
        ax.grid(False, axis="y")
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order)
        ax.set_ylim(-0.6, len(order) - 0.4)
    for ax in list(axes.flat)[len(props):]:
        ax.set_visible(False)
    _titles(fig, 0.36 * len(order) * nrows + 1.3 * nrows,
            "Pattern classes compared, property by property",
            f"Dots are tests; bar = median, line = middle 50%. Rows sorted by {_label(props[0])}.")
    return save_figure(fig, out_path, cfg)


def make_class_outputs(
    metrics: pd.DataFrame,
    figures_dir: Path,
    results_dir: Path,
    cfg: AnalysisConfig = DEFAULT_CONFIG,
    figures: bool = True,
) -> list[Path]:
    """All class outputs for `cfg.class_by`; nothing if it is unset or unusable."""
    found = class_labels(metrics, cfg.class_by) if cfg.class_by else None
    for stale in (figures_dir / "class_space.png", figures_dir / "class_comparison.png",
                  results_dir / "class_summary.csv"):
        stale.unlink(missing_ok=True)
    if found is None:
        return []
    labels, primary, reference = found
    written = []
    props = [c for c in (resolve_column(metrics, p) for p in CLASS_PROPERTIES) if c]
    table = class_summary(metrics, labels, props)
    results_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(results_dir / "class_summary.csv", index=False)
    written.append(results_dir / "class_summary.csv")
    if figures:
        for out in (
            plot_class_space(metrics, labels, primary, reference, figures_dir / "class_space.png", cfg),
            plot_class_comparison(metrics, labels, primary, reference,
                                  figures_dir / "class_comparison.png", cfg),
        ):
            if out is not None:
                written.append(out)
    return written
