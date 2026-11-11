"""Which scale of each TPMS baseline behaves most like the plain woven foam?

The TPMS baselines were printed at several *scales* of their defining
function: the features grow or shrink while the volume fraction stays the
same. This module searches along that scale for the value whose mechanical
response best matches the reference set (the plain woven specimens), so a
baseline can be compared with the woven foam at matching scale.

How "matching" is measured
--------------------------
The reference is a population -- 64 different plain patterns -- so a
candidate is compared with its typical range, not with one curve:

* **Magnitude.** Each property (modulus, plateau stress, stress at 10% and
  20%, energy to 20%) becomes a robust z-score against the reference:
  ``(value - plain median) / plain spread``, the spread being 1.4826 x the
  median absolute deviation. The distance is the RMS of those z-scores, so
  **below 1 means "inside plain's typical spread" on average**.
* **Shape.** Every stress-strain curve is divided by its own stress at 10%
  strain, and compared point by point with the reference's median normalised
  curve, again in units of the reference spread. This asks whether the curve
  *hardens like plain*, independent of how strong it is.

The search
----------
With one specimen per (type, scale), single points are noisy, so the
recommendation comes from a **trend**: each property is fitted as a
quadratic in scale (linear with fewer than four scales), the fitted
distance is minimised over a fine grid of scales, and a bootstrap over
specimens (and over the reference, whose median is itself an estimate) gives
an interval for the best scale. It is the 2.5-97.5 percentile interval;
checked by simulation on 64 baselines and 40 references it contained the
true scale in about 85% of runs, and the summary says so rather than
calling it a 95% interval. This is done per TPMS type
and for all types pooled; the pooled fit is unbiased for the scale trend
because every type was printed at every scale. Where a property's trend
crosses the plain median, the crossing scale is reported too -- matching
stiffness and matching strength need not happen at the same scale.

An optional *window* (the scales that look geometrically similar, 3-6 here)
is drawn on every figure and checked against the mechanical answer: a
geometric and a mechanical match that agree are much stronger evidence
than either alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from scipy import stats

from .config import DEFAULT_CONFIG, AnalysisConfig
from .envelope import (
    CATEGORICAL,
    GRID,
    INK,
    INK_MUTED,
    INK_SECONDARY,
    SURFACE,
    display_name,
    resolve_column,
    save_figure,
    sequential_cmap,
    style_axes,
    style_legend,
)
from .mechanics import resample_curve

#: Label for the pooled (all types together) rows.
ALL = "all types"
#: If the pooled shape distance varies less than this across scales, shape is
#: reported as not discriminating rather than naming a noise-level winner.
SHAPE_MIN_RANGE = 0.3

_LABELS = {
    "modulus_MPa": "Modulus E (MPa)",
    "plateau_stress_MPa": "Plateau σ (MPa)",
    "first_peak_stress_MPa": "First-peak σ (MPa)",
    "stress_at_10pct_MPa": "σ at 10% (MPa)",
    "stress_at_20pct_MPa": "σ at 20% (MPa)",
    "energy_to_20pct_MJ_m3": "W to 20% (MJ/m³)",
}
_SHORT = {
    "modulus_MPa": "Modulus E",
    "plateau_stress_MPa": "Plateau σ",
    "stress_at_10pct_MPa": "σ at 10%",
    "stress_at_20pct_MPa": "σ at 20%",
    "energy_to_20pct_MJ_m3": "W to 20%",
}
REFERENCE_FILL = "#e4e3de"
WINDOW_FILL = "#f1efe7"


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------
def _text(series: pd.Series) -> pd.Series:
    return series.where(series.notna(), "").astype(str).str.strip().replace("nan", "")


def select(metrics: pd.DataFrame, clause: str) -> pd.Series:
    """Boolean mask for a KEY=VALUE clause, e.g. ``family=plain``."""
    if "=" not in clause:
        raise ValueError(f"expected KEY=VALUE, got {clause!r}")
    key, value = (part.strip() for part in clause.split("=", 1))
    column = resolve_column(metrics, key)
    if column is None:
        raise ValueError(f"{key!r} matches no column")
    return _text(metrics[column]) == value


def robust_spread(values: np.ndarray) -> float:
    """1.4826 x MAD; IQR/1.349 or the SD if that is zero."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return float("nan")
    mad = 1.4826 * np.median(np.abs(values - np.median(values)))
    if mad > 0:
        return float(mad)
    iqr = np.subtract(*np.percentile(values, [75, 25])) / 1.349
    return float(iqr) if iqr > 0 else float(np.std(values, ddof=1) or np.nan)


def _fit(scale: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray | None:
    ok = np.isfinite(scale) & np.isfinite(values)
    distinct = len(np.unique(scale[ok]))
    if distinct < 2 or ok.sum() < 3:
        return None
    degree = 2 if distinct >= 4 and ok.sum() >= 5 else 1
    coeffs = np.polyfit(scale[ok], values[ok], degree)
    return np.polyval(coeffs, grid)


def _crossings(grid: np.ndarray, curve: np.ndarray, level: float) -> list[float]:
    diff = curve - level
    out = []
    for i in range(len(grid) - 1):
        a, b = diff[i], diff[i + 1]
        if a == 0:
            out.append(float(grid[i]))
        elif a * b < 0:
            out.append(float(grid[i] - a * (grid[i + 1] - grid[i]) / (b - a)))
    return out


# ----------------------------------------------------------------------
# Result
# ----------------------------------------------------------------------
@dataclass
class ScaleMatchResult:
    specimens: pd.DataFrame          # one row per candidate test
    by_scale: pd.DataFrame           # group x scale: n, distances, coverage, median z
    best: pd.DataFrame               # per group: best scale and its interval
    crossings: pd.DataFrame          # group x property: where the trend meets plain
    reference: pd.DataFrame          # property: median, q1, q3, spread
    properties: list[str]
    scale_column: str
    group_column: str | None
    window: tuple[float, float] | None
    n_reference: int
    n_candidates: int
    reference_label: str
    mass_notes: list[str] = field(default_factory=list)
    # Curve data for the figure: grid, reference band, candidate curves.
    grid: np.ndarray | None = None
    reference_curves: dict[str, np.ndarray] = field(default_factory=dict)
    candidate_curves: dict[str, np.ndarray] = field(default_factory=dict)


# ----------------------------------------------------------------------
# Curves
# ----------------------------------------------------------------------
def _curve_matrix(
    curves: Mapping[str, tuple[np.ndarray, np.ndarray]], ids: Sequence[str], grid: np.ndarray
) -> np.ndarray:
    rows = []
    for test_id in ids:
        if test_id in curves:
            strain, stress = curves[test_id]
            rows.append(resample_curve(np.asarray(strain, float), np.asarray(stress, float), grid))
        else:
            rows.append(np.full(len(grid), np.nan))
    return np.vstack(rows) if rows else np.empty((0, len(grid)))


def _normalise(matrix: np.ndarray, grid: np.ndarray, at: float = 0.10) -> np.ndarray:
    idx = int(np.argmin(np.abs(grid - at)))
    scale = matrix[:, idx:idx + 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(scale > 0, matrix / scale, np.nan)


def _band(matrix: np.ndarray, min_n: int = 5) -> dict[str, np.ndarray]:
    count = np.isfinite(matrix).sum(axis=0)
    with np.errstate(all="ignore"), _quiet():
        median = np.nanmedian(matrix, axis=0)
        q1, q3 = np.nanpercentile(matrix, [25, 75], axis=0)
        spread = 1.4826 * np.nanmedian(np.abs(matrix - median), axis=0)
    thin = count < min_n
    for arr in (median, q1, q3, spread):
        arr[thin] = np.nan
    spread = np.where(spread > 0, spread, np.nan)
    return {"median": median, "q1": q1, "q3": q3, "spread": spread}


class _quiet:
    def __enter__(self):
        import warnings
        self._ctx = warnings.catch_warnings()
        self._ctx.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *exc):
        return self._ctx.__exit__(*exc)


def _curve_distance(row: np.ndarray, band: dict[str, np.ndarray], min_points: int = 10) -> float:
    z = (row - band["median"]) / band["spread"]
    z = z[np.isfinite(z)]
    return float(np.sqrt(np.mean(z ** 2))) if len(z) >= min_points else float("nan")


# ----------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------
def analyse(
    metrics: pd.DataFrame,
    curves: Mapping[str, tuple[np.ndarray, np.ndarray]] | None = None,
    cfg: AnalysisConfig = DEFAULT_CONFIG,
    n_boot: int = 500,
    seed: int = 0,
) -> ScaleMatchResult:
    """Search the candidates' scale for the best match to the reference."""
    curves = curves or {}
    scale_col = resolve_column(metrics, cfg.scale_match_column)
    if scale_col is None:
        raise ValueError(f"no column {cfg.scale_match_column!r} for the scale")
    ref_mask = select(metrics, cfg.scale_match_reference)
    cand_mask = select(metrics, cfg.scale_match_candidates)
    scale = pd.to_numeric(metrics[scale_col], errors="coerce")
    cand_mask &= scale.notna()
    if ref_mask.sum() < 5:
        raise ValueError(f"only {int(ref_mask.sum())} reference tests ({cfg.scale_match_reference})")
    if cand_mask.sum() < 3:
        raise ValueError(f"only {int(cand_mask.sum())} candidate tests with a scale")
    group_col = resolve_column(metrics, cfg.scale_match_group_by) if cfg.scale_match_group_by else None
    window = tuple(float(v) for v in cfg.scale_match_window) if len(cfg.scale_match_window) == 2 else None

    # Reference statistics per property.
    props, ref_rows = [], []
    ref_values: dict[str, np.ndarray] = {}
    for name in cfg.scale_match_properties:
        column = resolve_column(metrics, name)
        if column is None:
            continue
        values = pd.to_numeric(metrics.loc[ref_mask, column], errors="coerce").dropna().to_numpy()
        spread = robust_spread(values)
        if len(values) < 5 or not np.isfinite(spread):
            continue
        props.append(column)
        ref_values[column] = values
        q1, med, q3 = np.percentile(values, [25, 50, 75])
        ref_rows.append({"property": column, "n": len(values), "median": med,
                         "q1": q1, "q3": q3, "spread": spread})
    if not props:
        raise ValueError("no property has enough reference values")
    reference = pd.DataFrame(ref_rows).set_index("property")

    # Curves: reference band, raw and normalised by stress at 10%.
    ids = metrics["test_id"].astype(str)
    have_curves = bool(curves)
    grid = np.linspace(0.01, 0.35, 69)
    ref_ids, cand_ids = list(ids[ref_mask]), list(ids[cand_mask])
    ref_raw = _curve_matrix(curves, ref_ids, grid) if have_curves else None
    cand_raw = _curve_matrix(curves, cand_ids, grid) if have_curves else None
    ref_band = _band(ref_raw) if have_curves else None
    ref_shape_band = _band(_normalise(ref_raw, grid)) if have_curves else None
    cand_shape = _normalise(cand_raw, grid) if have_curves else None

    # Per-specimen table.
    cand = metrics[cand_mask].reset_index(drop=True)
    table = pd.DataFrame({
        "test_id": cand["test_id"].astype(str),
        "group": _text(cand[group_col]) if group_col else ALL,
        "scale": pd.to_numeric(cand[scale_col], errors="coerce"),
    })
    z = pd.DataFrame({
        f"z_{p}": (pd.to_numeric(cand[p], errors="coerce") - reference.loc[p, "median"])
        / reference.loc[p, "spread"]
        for p in props
    })
    inside = pd.DataFrame({
        p: pd.to_numeric(cand[p], errors="coerce").between(reference.loc[p, "q1"], reference.loc[p, "q3"])
           .where(pd.to_numeric(cand[p], errors="coerce").notna())
        for p in props
    })
    table = pd.concat([table, z], axis=1)
    n_z = z.notna().sum(axis=1)
    table["distance_magnitude"] = np.sqrt((z ** 2).mean(axis=1)).where(n_z >= 2)
    table["coverage"] = inside.mean(axis=1)
    if have_curves:
        table["distance_shape"] = [_curve_distance(row, ref_shape_band) for row in cand_shape]
        table["distance_curve"] = [_curve_distance(row, ref_band) for row in cand_raw]
    else:
        table["distance_shape"] = np.nan
        table["distance_curve"] = np.nan

    groups = sorted(set(table["group"]) - {""})
    group_sets = [(g, table["group"] == g) for g in groups if g != ALL] + [(ALL, table["group"] != "")]
    if len(groups) <= 1:
        group_sets = [(ALL, table["group"] != "")]

    # Observed, per group and scale.
    by_rows = []
    for name, mask in group_sets:
        block = table[mask]
        for s, sub in block.groupby("scale"):
            row = {"group": name, "scale": s, "n": len(sub),
                   "distance_magnitude": sub["distance_magnitude"].median(),
                   "distance_shape": sub["distance_shape"].median(),
                   "distance_curve": sub["distance_curve"].median(),
                   "coverage": sub["coverage"].mean()}
            row.update({f"z_{p}": sub[f"z_{p}"].median() for p in props})
            by_rows.append(row)
    by_scale = pd.DataFrame(by_rows)

    # Trend search with bootstrap.
    rng = np.random.default_rng(seed)
    best_rows, cross_rows = [], []
    for name, mask in group_sets:
        block = table[mask].reset_index(drop=True)
        if len(block) < 3:
            continue
        values = cand.loc[mask.to_numpy(), props].apply(pd.to_numeric, errors="coerce")
        values = values.reset_index(drop=True)
        s_all = block["scale"].to_numpy(float)
        search = np.linspace(np.nanmin(s_all), np.nanmax(s_all), 141)

        def best_scale(
            idx: np.ndarray, ref_stats: Mapping[str, tuple[float, float]] | None = None
        ) -> tuple[float, float, dict[str, np.ndarray]]:
            fits = {}
            for p in props:
                f = _fit(s_all[idx], values[p].to_numpy(float)[idx], search)
                if f is not None:
                    centre, spread = ref_stats[p] if ref_stats else (
                        reference.loc[p, "median"], reference.loc[p, "spread"])
                    fits[p] = (f - centre) / spread
            if len(fits) < 2:
                return float("nan"), float("nan"), fits
            d = np.sqrt(np.mean(np.vstack(list(fits.values())) ** 2, axis=0))
            k = int(np.nanargmin(d))
            return float(search[k]), float(d[k]), fits

        def best_shape(idx: np.ndarray) -> float:
            f = _fit(s_all[idx], block["distance_shape"].to_numpy(float)[idx], search)
            return float(search[int(np.nanargmin(f))]) if f is not None and np.isfinite(f).any() else float("nan")

        full = np.arange(len(block))
        s_star, d_star, fits = best_scale(full)
        shape_star = best_shape(full)
        strata = [np.flatnonzero(block["group"].to_numpy() == g) for g in sorted(set(block["group"]))]
        boots, shape_boots = [], []
        for _ in range(n_boot):
            idx = np.concatenate([rng.choice(st, len(st)) for st in strata])
            # The reference is a sample too: resample it, so the interval
            # includes the uncertainty in plain's median, not just the fit's.
            ref_stats = {}
            for p in props:
                draw = rng.choice(ref_values[p], len(ref_values[p]))
                ref_stats[p] = (float(np.median(draw)), robust_spread(draw))
            boots.append(best_scale(idx, ref_stats)[0])
            shape_boots.append(best_shape(idx))
        boots, shape_boots = np.array(boots), np.array(shape_boots)
        lo, hi = (np.nanpercentile(boots, [2.5, 97.5]) if np.isfinite(boots).sum() > 10 else (np.nan, np.nan))
        slo, shi = (np.nanpercentile(shape_boots, [2.5, 97.5])
                    if np.isfinite(shape_boots).sum() > 10 else (np.nan, np.nan))

        observed = by_scale[(by_scale["group"] == name)].dropna(subset=["distance_magnitude"])
        obs_best = observed.loc[observed["distance_magnitude"].idxmin()] if len(observed) else None
        row = {
            "group": name, "n_tests": len(block),
            "best_scale": s_star, "best_scale_lo": lo, "best_scale_hi": hi,
            "distance_at_best": d_star,
            "observed_best_scale": obs_best["scale"] if obs_best is not None else np.nan,
            "observed_best_distance": obs_best["distance_magnitude"] if obs_best is not None else np.nan,
            "shape_best_scale": shape_star, "shape_best_scale_lo": slo, "shape_best_scale_hi": shi,
        }
        if window:
            row["in_window"] = bool(window[0] <= s_star <= window[1]) if np.isfinite(s_star) else None
            row["interval_overlaps_window"] = bool(hi >= window[0] and lo <= window[1]) \
                if np.isfinite(lo) else None
        best_rows.append(row)

        for p, zfit in fits.items():
            points = _crossings(search, zfit, 0.0)
            side = "above" if np.nanmin(zfit) > 0 else "below" if np.nanmax(zfit) < 0 else ""
            cross_rows.append({"group": name, "property": p,
                               "crossing_scales": "; ".join(f"{c:.2f}" for c in points),
                               "never": side})

    notes = []
    mass_col = resolve_column(metrics, cfg.scale_match_mass) if cfg.scale_match_mass else None
    if mass_col is not None:
        mass = pd.to_numeric(cand[mass_col], errors="coerce")
        table["mass"] = mass
        for name, mask in group_sets:
            if name == ALL:
                continue
            ok = mask & mass.notna()
            if ok.sum() >= 5:
                rho, p = stats.spearmanr(table.loc[ok, "scale"], mass[ok])
                if p < 0.05:
                    notes.append(f"{name} (ρ = {rho:+.2f})")

    result = ScaleMatchResult(
        specimens=table, by_scale=by_scale, best=pd.DataFrame(best_rows),
        crossings=pd.DataFrame(cross_rows), reference=reference.reset_index(),
        properties=props, scale_column=scale_col, group_column=group_col, window=window,
        n_reference=int(ref_mask.sum()), n_candidates=int(cand_mask.sum()),
        reference_label=cfg.scale_match_reference.split("=", 1)[-1], mass_notes=notes,
    )
    if have_curves:
        result.grid = grid
        result.reference_curves = {"raw": ref_raw, "shape": _normalise(ref_raw, grid)}
        result.candidate_curves = {"raw": cand_raw, "shape": cand_shape}
    return result


# ----------------------------------------------------------------------
# Words
# ----------------------------------------------------------------------
def _interval(row: pd.Series, key: str = "best_scale") -> str:
    lo, hi = row.get(f"{key}_lo"), row.get(f"{key}_hi")
    value = row[key]
    if not np.isfinite(value):
        return "–"
    if lo is None or not np.isfinite(lo):
        return f"{value:.1f}"
    return f"{value:.1f} [{lo:.1f}, {hi:.1f}]"


def _verdict(distance: float) -> str:
    if not np.isfinite(distance):
        return ""
    if distance < 1:
        return "a match: within plain's typical spread"
    if distance < 2:
        return "close, but outside plain's typical spread"
    return "no scale matches; this is only the nearest"


def summarise(result: ScaleMatchResult) -> str:
    ref = result.reference_label
    props = ", ".join(_SHORT.get(p, display_name(p)) for p in result.properties)
    lines = [
        f"Scale match: which {display_name(result.scale_column)} of the baselines behaves most like "
        f"{ref} ({result.n_reference} reference, {result.n_candidates} candidate tests)",
        f"  Distance = RMS robust z-score against {ref} over {props};",
        f"  below 1 = inside {ref}'s typical spread. Best scale from a quadratic trend;\n"
        f"  [bootstrap interval: in simulation it held the true scale ~85% of the time].",
    ]
    if result.window:
        lines.append(f"  Geometric window: {result.window[0]:g}–{result.window[1]:g}.")
    best = result.best.set_index("group") if not result.best.empty else pd.DataFrame()
    if ALL in best.index:
        row = best.loc[ALL]
        where = ""
        if result.window and row.get("in_window") is not None:
            where = "  — inside the geometric window" if row["in_window"] else (
                "  — outside the geometric window"
                + (", though the interval reaches it" if row.get("interval_overlaps_window") else ""))
        lines.append(
            f"\n  All types pooled: best scale {_interval(row)}, distance {row['distance_at_best']:.2f} "
            f"({_verdict(row['distance_at_best'])}){where}"
        )
        pooled = result.by_scale[result.by_scale["group"] == ALL].sort_values("scale")
        if len(pooled):
            obs = " · ".join(f"{s:g}: {d:.2f}" for s, d in zip(pooled["scale"], pooled["distance_magnitude"]))
            lines.append(f"    observed distance by scale: {obs}")
        shape = pooled["distance_shape"].dropna() if len(pooled) else pd.Series(dtype=float)
        if len(shape) >= 3 and shape.max() - shape.min() < SHAPE_MIN_RANGE:
            lines.append(
                f"    curve shape (normalised by σ at 10%): every scale is equally close "
                f"({shape.min():.2f}–{shape.max():.2f}), so shape does not pick a scale"
            )
        elif np.isfinite(row["shape_best_scale"]):
            lines.append(f"    curve shape (normalised by σ at 10%) matches best at {_interval(row, 'shape_best_scale')}")
        cross = result.crossings[result.crossings["group"] == ALL]
        parts = []
        for _, c in cross.iterrows():
            name = _SHORT.get(c["property"], display_name(c["property"]))
            if c["crossing_scales"]:
                parts.append(f"{name} at {c['crossing_scales']}")
            elif c["never"]:
                parts.append(f"{name} never ({c['never']} {ref} at every scale)")
        if parts:
            lines.append("    trend meets the plain median: " + "; ".join(parts))
    per_type = best.drop(index=ALL, errors="ignore")
    if len(per_type):
        lines.append("\n  Per type:")
        for name, row in per_type.sort_values("distance_at_best").iterrows():
            flag = ""
            if result.window and row.get("in_window") is not None:
                flag = "in window" if row["in_window"] else "outside window"
            lines.append(
                f"    {name:<16} best {_interval(row):<18} distance {row['distance_at_best']:.2f}  {flag}"
            )
        lines.append("    (one specimen per scale per type: per-type intervals are wide; "
                     "the pooled answer is the one to act on)")
    if result.mass_notes:
        lines.append(
            "\n  Caution: slicer mass changes with scale for " + ", ".join(result.mass_notes)
            + ".\n  Mass drives strength, so part of a scale trend there may be a mass trend."
        )
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
def _colors(groups: Sequence[str]) -> dict[str, str]:
    return {g: CATEGORICAL[i % len(CATEGORICAL)] for i, g in enumerate(sorted(groups))}


def _window(ax: plt.Axes, window: tuple[float, float] | None) -> None:
    if window:
        ax.axvspan(window[0], window[1], color=WINDOW_FILL, zorder=0, lw=0)


def plot_trends(result: ScaleMatchResult, metrics: pd.DataFrame, out_path: Path,
                cfg: AnalysisConfig = DEFAULT_CONFIG) -> Path | None:
    table = result.specimens
    props = result.properties
    if not props:
        return None
    cand = metrics.set_index(metrics["test_id"].astype(str)).loc[table["test_id"]]
    groups = sorted(g for g in set(table["group"]) if g and g != ALL)
    colors = _colors(groups)
    ncols = min(3, len(props))
    nrows = int(np.ceil(len(props) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols + 2.2, 3.9 * nrows), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    ref = result.reference.set_index("property")
    best = result.best.set_index("group") if not result.best.empty else pd.DataFrame()
    s = table["scale"].to_numpy(float)
    search = np.linspace(np.nanmin(s), np.nanmax(s), 141)
    for ax, p in zip(axes.flat, props):
        _window(ax, result.window)
        ax.axhspan(ref.loc[p, "q1"], ref.loc[p, "q3"], color=REFERENCE_FILL, zorder=1, lw=0)
        ax.axhline(ref.loc[p, "median"], color=INK_SECONDARY, linestyle="--", linewidth=1.2, zorder=2)
        values = pd.to_numeric(cand[p], errors="coerce").to_numpy(float)
        for g in groups:
            m = (table["group"] == g).to_numpy()
            order = np.argsort(s[m])
            ax.plot(s[m][order], values[m][order], color=colors[g], linewidth=0.9, alpha=0.45, zorder=3)
            ax.scatter(s[m], values[m], s=18, color=colors[g], edgecolors="none", zorder=4)
        fit = _fit(s, values, search)
        if fit is not None:
            ax.plot(search, fit, color=INK, linewidth=2.2, zorder=5)
        if ALL in best.index and np.isfinite(best.loc[ALL, "best_scale"]):
            ax.axvline(best.loc[ALL, "best_scale"], color=INK, linewidth=1, linestyle=":", zorder=5)
        style_axes(ax, "", display_name(result.scale_column), _LABELS.get(p, display_name(p)))
    for ax in list(axes.flat)[len(props):]:
        ax.set_visible(False)
    handles = [Line2D([], [], color=colors[g], marker="o", linestyle="-", linewidth=0.9,
                      markersize=5, label=g) for g in groups]
    handles += [
        Line2D([], [], color=INK, linewidth=2.2, label="trend, all types"),
        Line2D([], [], color=INK, linewidth=1, linestyle=":", label="best-matching scale"),
        Line2D([], [], color=INK_SECONDARY, linestyle="--", label=f"{result.reference_label} median"),
        Patch(color=REFERENCE_FILL, label=f"{result.reference_label} middle 50%"),
    ]
    if result.window:
        handles.append(Patch(color=WINDOW_FILL, label=f"geometric window {result.window[0]:g}–{result.window[1]:g}"))
    style_legend(fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5),
                            frameon=False, fontsize=9))
    _titles(fig, 3.9 * nrows, f"Baseline properties against {display_name(result.scale_column)}",
            f"Where a trend enters the grey band, that scale behaves like {result.reference_label}.")
    return save_figure(fig, out_path, cfg)


def _titles(fig: plt.Figure, height_in: float, title: str, subtitle: str) -> None:
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.75 / height_in))
    fig.text(0.01, 1 - 0.12 / height_in, title, color=INK, fontsize=13, ha="left", va="top")
    fig.text(0.01, 1 - 0.42 / height_in, subtitle, color=INK_MUTED, fontsize=9, ha="left", va="top")


def plot_distance(result: ScaleMatchResult, out_path: Path,
                  cfg: AnalysisConfig = DEFAULT_CONFIG) -> Path | None:
    by = result.by_scale
    if by.empty:
        return None
    measures = [("distance_magnitude", "Property distance")]
    if by["distance_shape"].notna().any():
        measures.append(("distance_shape", "Curve-shape distance"))
    groups = [g for g in sorted(set(by["group"])) if g != ALL] + ([ALL] if ALL in set(by["group"]) else [])
    scales = sorted(set(by["scale"]))
    height = 0.42 * len(groups) + 1.9
    fig, axes = plt.subplots(1, len(measures), figsize=(0.62 * len(scales) * len(measures) + 3.2, height),
                             squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    cmap = sequential_cmap()
    for ax, (key, title) in zip(axes[0], measures):
        grid = by.pivot(index="group", columns="scale", values=key).reindex(index=groups, columns=scales)
        values = grid.to_numpy(float)
        vmax = np.nanpercentile(values, 95) if np.isfinite(values).any() else 1
        norm = Normalize(0, max(vmax, 1.0))
        ax.imshow(np.ma.masked_invalid(values), cmap=cmap, norm=norm, aspect="auto")
        for i in range(len(groups)):
            for j in range(len(scales)):
                v = values[i, j]
                if np.isfinite(v):
                    dark = norm(v) > 0.55
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8,
                            color=SURFACE if dark else INK)
            row = values[i]
            if np.isfinite(row).any():
                j = int(np.nanargmin(row))
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=INK, linewidth=2))
        if len(groups) > 1 and groups[-1] == ALL:
            ax.axhline(len(groups) - 1.5, color=SURFACE, linewidth=3)
        if result.window:
            lo = next((j for j, s in enumerate(scales) if s >= result.window[0]), None)
            hi = max((j for j, s in enumerate(scales) if s <= result.window[1]), default=None)
            if lo is not None and hi is not None and hi >= lo:
                ax.annotate("", xy=(lo - 0.45, -0.75), xytext=(hi + 0.45, -0.75),
                            arrowprops=dict(arrowstyle="-", color=INK_SECONDARY, lw=2),
                            annotation_clip=False)
                ax.text((lo + hi) / 2, -0.95, "geometric window", ha="center", va="bottom",
                        fontsize=8, color=INK_SECONDARY)
        ax.set_xticks(range(len(scales)))
        ax.set_xticklabels([f"{s:g}" for s in scales])
        ax.set_yticks(range(len(groups)))
        ax.set_yticklabels(groups)
        ax.set_xlabel(display_name(result.scale_column), color=INK_SECONDARY, fontsize=10)
        ax.set_title(title, color=INK, fontsize=11, loc="left", pad=22)
        ax.tick_params(colors=INK_SECONDARY, labelsize=9, length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    _titles(fig, height, f"How far each baseline scale is from {result.reference_label}",
            "RMS robust z-score; below 1 = inside its typical spread. Box: closest scale in each row.")
    return save_figure(fig, out_path, cfg)


def plot_curves(result: ScaleMatchResult, out_path: Path,
                cfg: AnalysisConfig = DEFAULT_CONFIG) -> Path | None:
    if result.grid is None:
        return None
    grid = result.grid
    scales = sorted(set(result.specimens["scale"].dropna()))
    norm = Normalize(min(scales), max(scales))
    cmap = sequential_cmap()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    fig.patch.set_facecolor(SURFACE)
    panels = (("raw", "Stress (MPa)", "As measured"),
              ("shape", "Stress / stress at 10% (–)", "Shape: each curve divided by its σ at 10%"))
    for ax, (key, ylabel, title) in zip(axes, panels):
        band = _band(result.reference_curves[key])
        ok = np.isfinite(band["median"])
        ax.fill_between(grid[ok], band["q1"][ok], band["q3"][ok], color=REFERENCE_FILL, lw=0, zorder=1)
        ax.plot(grid[ok], band["median"][ok], color=INK_SECONDARY, linestyle="--", linewidth=1.6, zorder=3)
        matrix = result.candidate_curves[key]
        for s in scales:
            rows = matrix[(result.specimens["scale"] == s).to_numpy()]
            if not len(rows):
                continue
            with np.errstate(all="ignore"), _quiet():
                med = np.nanmedian(rows, axis=0)
            count = np.isfinite(rows).sum(axis=0)
            med[count < max(2, len(rows) // 2)] = np.nan
            in_window = result.window and result.window[0] <= s <= result.window[1]
            ax.plot(grid, med, color=cmap(norm(s)), linewidth=2.0 if in_window else 1.2,
                    alpha=1.0 if in_window else 0.75, zorder=4)
        style_axes(ax, title, "Engineering strain (mm/mm)", ylabel)
    fig.subplots_adjust(top=0.84, right=0.89, wspace=0.28)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    bar = fig.colorbar(mappable, cax=fig.add_axes((0.915, 0.14, 0.012, 0.66)))
    bar.set_label(f"{display_name(result.scale_column)} (median curve over all types)",
                  color=INK_SECONDARY, fontsize=9)
    bar.ax.tick_params(colors=INK_SECONDARY, labelsize=8, length=0)
    bar.outline.set_visible(False)
    handles = [Line2D([], [], color=INK_SECONDARY, linestyle="--", label=f"{result.reference_label} median"),
               Patch(color=REFERENCE_FILL, label=f"{result.reference_label} middle 50%")]
    if result.window:
        handles.append(Line2D([], [], color=INK_MUTED, linewidth=2.0,
                              label=f"scales {result.window[0]:g}–{result.window[1]:g} drawn heavier"))
    style_legend(axes[0].legend(handles=handles, loc="lower right", frameon=False, fontsize=8))
    fig.text(0.01, 0.995, f"Baseline curves by scale against {result.reference_label}",
             color=INK, fontsize=13, ha="left", va="top")
    return save_figure(fig, out_path, cfg)


# ----------------------------------------------------------------------
# Outputs
# ----------------------------------------------------------------------
def write_outputs(result: ScaleMatchResult, metrics: pd.DataFrame, table_dir: Path,
                  figure_dir: Path, cfg: AnalysisConfig = DEFAULT_CONFIG,
                  figures: bool = True) -> list[Path]:
    table_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, frame in (("specimens.csv", result.specimens), ("by_scale.csv", result.by_scale),
                        ("best_scale.csv", result.best), ("crossings.csv", result.crossings),
                        ("reference.csv", result.reference)):
        frame.to_csv(table_dir / name, index=False)
        written.append(table_dir / name)
    (table_dir / "summary.txt").write_text(summarise(result) + "\n", encoding="utf-8")
    written.append(table_dir / "summary.txt")
    if figures:
        for out in (plot_trends(result, metrics, figure_dir / "scale_trends.png", cfg),
                    plot_distance(result, figure_dir / "scale_distance.png", cfg),
                    plot_curves(result, figure_dir / "scale_curves.png", cfg)):
            if out is not None:
                written.append(out)
    return written


def curves_from_dir(directory: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out = {}
    for path in sorted(Path(directory).glob("*.csv")):
        frame = pd.read_csv(path, usecols=["strain", "stress_MPa"])
        out[path.stem] = (frame["strain"].to_numpy(float), frame["stress_MPa"].to_numpy(float))
    return out
