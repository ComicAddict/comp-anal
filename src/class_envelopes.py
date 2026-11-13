"""Stress-strain envelopes per pattern class.

The overall envelope mixes every family together. These figures split it:

* ``class_envelopes.png`` -- one panel per class (each TPMS type, and the
  plain/woven set), every test drawn thin, with the class's range,
  middle 50% and median. The reference group's median and middle 50% (plain)
  are repeated in grey in every other panel, so each baseline is read
  directly against the woven foam.
* ``class_envelope_overlay.png`` -- every class's median and middle 50% on one
  axes, for the overall ranking.
* ``envelopes_by_<column>.png`` -- for each column in ``envelope_split_by``
  (the plain pattern-code digits), the tests that have a value split by its
  levels, against the whole subset in grey.

Curves run to their own end (the full strain range). Statistics are drawn
only where at least ``envelope_min_tests`` tests of that class still have
data, so a class band never narrows to one curve posing as a spread.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .classes import class_colors, class_labels
from .config import DEFAULT_CONFIG, AnalysisConfig
from .envelope import (
    CATEGORICAL,
    INK,
    INK_MUTED,
    INK_SECONDARY,
    SURFACE,
    CurveMatrix,
    display_name,
    draw_stat_band,
    draw_stat_line,
    envelope_stats,
    resolve_column,
    save_figure,
    style_axes,
    style_legend,
)

REFERENCE_BAND = "#dcdbd5"


def _titles(fig: plt.Figure, height_in: float, title: str, subtitle: str) -> None:
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.8 / height_in))
    fig.text(0.01, 1 - 0.12 / height_in, title, color=INK, fontsize=13, ha="left", va="top")
    fig.text(0.01, 1 - 0.44 / height_in, subtitle, color=INK_MUTED, fontsize=9, ha="left", va="top")


def _labels_for(matrix: CurveMatrix, metrics: pd.DataFrame, labels: pd.Series) -> np.ndarray:
    by_id = pd.Series(labels.to_numpy(), index=metrics["test_id"].astype(str))
    by_id = by_id[~by_id.index.duplicated()]
    return np.array([by_id.get(t, "") for t in matrix.test_ids], dtype=object)


def _draw_reference(ax: plt.Axes, grid: np.ndarray, ref) -> None:
    draw_stat_band(ax, grid, ref.q1, ref.q3, ref.complete, 1.0, zorder=1, color=REFERENCE_BAND)
    complete_median = np.where(ref.complete, ref.median, np.nan)
    ax.plot(grid, complete_median, color=INK_SECONDARY, linestyle="--", linewidth=1.3, zorder=2)


def plot_small_multiples(
    matrix: CurveMatrix,
    groups: np.ndarray,
    order: Sequence[str],
    colors: dict[str, str],
    reference_rows: np.ndarray | None,
    reference_label: str,
    out_path: Path,
    cfg: AnalysisConfig = DEFAULT_CONFIG,
    limit_stress: float | None = None,
    title: str = "Stress-strain envelope by class",
) -> Path | None:
    panels = [g for g in order if (groups == g).sum() >= 1]
    if len(panels) < 2:
        return None
    min_n = cfg.envelope_min_tests
    ref = envelope_stats(matrix.stress[reference_rows], min_n) if reference_rows is not None \
        and reference_rows.sum() >= min_n else None

    ncols = min(3, len(panels))
    nrows = int(np.ceil(len(panels) / ncols))
    height = 3.3 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, height), squeeze=False,
                             sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, name in zip(axes.flat, panels):
        rows = groups == name
        block = matrix.stress[rows]
        stats_ = envelope_stats(block, min(min_n, len(block)))
        is_reference = reference_rows is not None and np.array_equal(rows, reference_rows)
        if ref is not None and not is_reference:
            _draw_reference(ax, matrix.grid, ref)
        color = colors.get(name, CATEGORICAL[0])
        draw_stat_band(ax, matrix.grid, stats_.lo, stats_.hi, stats_.complete, 0.12, zorder=2, color=color)
        if len(block) >= 4:
            draw_stat_band(ax, matrix.grid, stats_.q1, stats_.q3, stats_.complete, 0.28, zorder=3, color=color)
        alpha = float(np.clip(2.5 / np.sqrt(len(block)), 0.12, 0.7))
        for curve in block:
            ax.plot(matrix.grid, curve, color=color, linewidth=0.7, alpha=alpha, zorder=4)
        draw_stat_line(ax, matrix.grid, stats_.median, stats_.complete, zorder=5,
                       color=color, linewidth=2.2)
        if limit_stress is not None and np.isfinite(limit_stress):
            ax.axhline(limit_stress, color=INK_MUTED, linestyle=":", linewidth=0.9, zorder=1)
        style_axes(ax, "", "", "")
        ax.set_title(f"{name}  (n = {len(block)})", color=INK, fontsize=10, loc="left")
        ax.set_ylim(bottom=0)
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Engineering strain (mm/mm)", color=INK_SECONDARY, fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("Engineering stress (MPa)", color=INK_SECONDARY, fontsize=9)
    ax0 = axes.flat[0]
    ax0.set_xlim(0, matrix.grid[-1])

    handles = [
        Line2D([], [], color=INK_SECONDARY, linewidth=2.2, label="class median"),
        Patch(color=INK_MUTED, alpha=0.4, label="class middle 50%"),
        Patch(color=INK_MUTED, alpha=0.15, label="class range"),
    ]
    if ref is not None:
        handles += [Line2D([], [], color=INK_SECONDARY, linestyle="--", label=f"{reference_label} median"),
                    Patch(color=REFERENCE_BAND, label=f"{reference_label} middle 50%")]
    if limit_stress is not None:
        handles.append(Line2D([], [], color=INK_MUTED, linestyle=":", label="load limit"))
    style_legend(fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
                            fontsize=8, bbox_to_anchor=(0.5, -0.01)))
    subtitle = ("Each curve runs to its own end. Statistics are solid while every test of the "
                "class is still running, faded once some have stopped.")
    if ref is not None:
        subtitle += f" Grey: {reference_label}, in every other panel."
    fig.tight_layout(rect=(0, 0.04, 1, 1 - 0.8 / height))
    fig.text(0.01, 1 - 0.12 / height, title, color=INK, fontsize=13, ha="left", va="top")
    fig.text(0.01, 1 - 0.44 / height, subtitle, color=INK_MUTED, fontsize=9, ha="left", va="top")
    return save_figure(fig, out_path, cfg)


def plot_overlay(
    matrix: CurveMatrix,
    groups: np.ndarray,
    order: Sequence[str],
    colors: dict[str, str],
    reference_label: str | None,
    out_path: Path,
    cfg: AnalysisConfig = DEFAULT_CONFIG,
    limit_stress: float | None = None,
) -> Path | None:
    panels = [g for g in order if (groups == g).sum() >= cfg.envelope_min_tests]
    if len(panels) < 2:
        return None
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    fig.patch.set_facecolor(SURFACE)
    handles = []
    for name in panels:
        stats_ = envelope_stats(matrix.stress[groups == name], cfg.envelope_min_tests)
        is_ref = name == reference_label
        color = colors.get(name, CATEGORICAL[0])
        draw_stat_band(ax, matrix.grid, stats_.q1, stats_.q3, stats_.complete,
                       0.3 if is_ref else 0.12, zorder=1 if is_ref else 2, color=color)
        if is_ref:
            (line,) = ax.plot(matrix.grid, np.where(stats_.complete, stats_.median, np.nan),
                              color=color, linewidth=2.6, linestyle="--", zorder=3,
                              label=f"{name}  (n = {(groups == name).sum()})")
        else:
            line = draw_stat_line(ax, matrix.grid, stats_.median, stats_.complete, zorder=3,
                                  color=color, linewidth=1.8,
                                  label=f"{name}  (n = {(groups == name).sum()})")
        handles.append(line)
    if limit_stress is not None and np.isfinite(limit_stress):
        handles.append(ax.axhline(limit_stress, color=INK_MUTED, linestyle=":", linewidth=0.9,
                                  label=f"load limit ({limit_stress:.1f} MPa)"))
    style_axes(ax, "", "Engineering strain (mm/mm)", "Engineering stress (MPa)")
    ax.set_xlim(0, matrix.grid[-1])
    ax.set_ylim(bottom=0)
    style_legend(ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
                           frameon=False, fontsize=9))
    _titles(fig, 6.2, "Class medians compared",
            "Line = median, band = middle 50%: solid while every test of the class runs, "
            "dotted once the strongest have stopped at the load limit."
            + (f" Dashed: {reference_label}." if reference_label else ""))
    return save_figure(fig, out_path, cfg)


def make_class_envelopes(
    metrics: pd.DataFrame,
    matrix: CurveMatrix,
    figures_dir: Path,
    cfg: AnalysisConfig = DEFAULT_CONFIG,
    limit_stress: float | None = None,
) -> list[Path]:
    """Every class envelope figure the config asks for."""
    for stale in list(figures_dir.glob("envelopes_by_*.png")) + [
        figures_dir / "class_envelopes.png", figures_dir / "class_envelope_overlay.png"
    ]:
        stale.unlink(missing_ok=True)
    written: list[Path] = []

    found = class_labels(metrics, cfg.class_by) if cfg.class_by else None
    if found is not None:
        labels, primary, reference = found
        groups = _labels_for(matrix, metrics, labels)
        colors = class_colors(primary, reference)
        order = list(primary) + list(reference)
        ref_name = reference[0] if reference else None
        ref_rows = (groups == ref_name) if ref_name else None
        for out in (
            plot_small_multiples(matrix, groups, order, colors, ref_rows, ref_name or "",
                                 figures_dir / "class_envelopes.png", cfg, limit_stress),
            plot_overlay(matrix, groups, order, colors, ref_name,
                         figures_dir / "class_envelope_overlay.png", cfg, limit_stress),
        ):
            if out is not None:
                written.append(out)

    for name in cfg.envelope_split_by:
        column = resolve_column(metrics, name)
        if column is None:
            continue
        values = metrics[column].where(metrics[column].notna(), "").astype(str).str.strip()
        values = values.replace("nan", "")
        groups = _labels_for(matrix, metrics, values)
        present = groups != ""
        levels = sorted(set(groups[present]), key=lambda v: (pd.to_numeric(v, errors="coerce"), v))
        if len(levels) < 2:
            continue
        label = display_name(column)
        panel_names = np.array([f"{label} = {g}" if g else "" for g in groups], dtype=object)
        order = [f"{label} = {lv}" for lv in levels]
        colors = {o: CATEGORICAL[i % len(CATEGORICAL)] for i, o in enumerate(order)}
        out = plot_small_multiples(
            matrix, panel_names, order, colors, present, f"all {int(present.sum())} with a {label}",
            figures_dir / f"envelopes_by_{label}.png", cfg, limit_stress,
            title=f"Stress-strain envelope by {label}",
        )
        if out is not None:
            written.append(out)
    return written
