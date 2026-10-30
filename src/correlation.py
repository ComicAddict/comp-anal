"""Do design and manufacturing parameters correlate with measured properties?

Every recorded parameter (the `param_*` columns, plus relative density, size,
material and pattern) is tested against every property in
`correlation_properties`, one test per test-specimen row.

Method, and why
---------------
* **Spearman rank correlation (rho)** is the headline measure for numeric
  parameters. Cellular-solid properties scale as power laws (E ~ rho*^2), so
  relationships are monotonic but not straight lines; rank correlation sees
  those at full strength where Pearson r would understate them, and a single
  outlying test cannot drag it around. Pearson r is reported alongside.
* **False-discovery-rate control (Benjamini-Hochberg).** Screening 8
  parameters x 10 properties at p < 0.05 would produce about four "significant"
  associations from noise alone. Each association gets a q-value; only q below
  `correlation_fdr_alpha` is called significant.
* **Partial correlation.** In a multi-parameter design, parameters often move
  together -- relative density rises with wall thickness, for instance -- so a
  pairwise correlation can belong to a different parameter entirely. The
  partial rho holds the *other* numeric parameters fixed (rank-based, matching
  Spearman). Pairwise strong / partial weak means "probably borrowed from a
  correlated parameter"; pairwise weak / partial strong means "an effect masked
  by a larger one".
* **Collinearity check.** Rank-space variance inflation factors flag
  parameters so entangled with the others that their individual effects cannot
  be separated by this data; their partial correlations are marked unreliable.
* **Categorical factors** (material, print orientation, topology family...) use
  the Kruskal-Wallis test with epsilon-squared effect size. Levels with fewer
  than `correlation_min_group_size` tests are left out, and a factor that is
  unique per test (64 different pattern names) cannot be tested at all.
* **Censoring.** Values that are only lower bounds -- energy and densification
  strain on tests the load limit stopped before densifying -- are excluded by
  default. Correlating them would mostly measure where each test stopped.
* **Gibson-Ashby scaling.** With relative density recorded, modulus and
  strength are also fitted as power laws, E ~ (rho*)^n. The exponent separates
  bending-dominated (n ~ 2 for modulus, 1.5 for strength) from
  stretch-dominated (n ~ 1) cell architectures.

Correlation is not causation, and this is observational screening: a
significant association says where to look, and a designed follow-up
experiment says whether the parameter is responsible.
"""

from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, to_rgb
from scipy import stats

from .config import DEFAULT_CONFIG, AnalysisConfig
from .specimens import parse_datetimes
from .envelope import (
    GRID,
    INK,
    INK_MUTED,
    INK_SECONDARY,
    POINT_COLOR,
    SURFACE,
    diverging_cmap,
    display_name,
    resolve_column,
    save_figure,
    sequential_cmap,
)

#: Core specimens.csv columns that are parameters, alongside the param_* ones.
CORE_NUMERIC: tuple[str, ...] = ("relative_density", "side_length_mm", "initial_height_mm")
CORE_CATEGORICAL: tuple[str, ...] = ("material", "pattern_name")
#: Core date columns. Analysed as elapsed time, so a trend across the print
#: campaign (printer or filament drift) or the test campaign shows up.
CORE_DATES: tuple[str, ...] = ("print_date", "test_date")
#: Derived from the two dates by the pipeline: days from printing to testing.
AGE_COLUMN = "param_print_to_test_days"

#: Property -> column flagging rows where that property is only a lower bound.
CENSOR_FLAGS: dict[str, str] = {
    "densification_strain": "densification_censored",
    "energy_absorption_MJ_m3": "energy_censored",
    "energy_absorption_J": "energy_censored",
    "sea_J_per_g": "energy_censored",
    "crush_force_N": "crush_censored",
    "crush_stress_MPa": "crush_censored",
    "max_stress_MPa": "ended_at_force_limit",
    "max_force_N": "ended_at_force_limit",
}

#: Gibson-Ashby exponents for the two deformation modes of cellular solids.
SCALING_REFERENCES: dict[str, dict[str, float]] = {
    "modulus": {"bending-dominated": 2.0, "stretch-dominated": 1.0},
    "strength": {"bending-dominated": 1.5, "stretch-dominated": 1.0},
}

_Z95 = float(stats.norm.ppf(0.975))

_FIXED_PATTERNS = (
    (re.compile(r"^stress_at_(.+)pct_MPa$"), "σ at {}%", "MPa"),
    (re.compile(r"^energy_to_(.+)pct_MJ_m3$"), "W to {}%", "MJ/m³"),
    (re.compile(r"^sea_to_(.+)pct_J_per_g$"), "SEA to {}%", "J/g"),
)
_PROPERTY_LABELS: dict[str, tuple[str, str]] = {
    "modulus_MPa": ("Modulus E", "MPa"),
    "plateau_stress_MPa": ("Plateau σ", "MPa"),
    "first_peak_stress_MPa": ("First-peak σ", "MPa"),
    "crush_stress_MPa": ("Crush σ", "MPa"),
    "max_stress_MPa": ("Max σ", "MPa"),
    "energy_absorption_MJ_m3": ("W to ε_D", "MJ/m³"),
    "energy_absorption_J": ("W to ε_D", "J"),
    "sea_J_per_g": ("SEA to ε_D", "J/g"),
    "densification_strain": ("ε_D", "–"),
}


def property_label(column: str, with_unit: bool = False) -> str:
    """Short, readable name for a property column."""
    if column in _PROPERTY_LABELS:
        name, unit = _PROPERTY_LABELS[column]
    else:
        for pattern, template, unit in _FIXED_PATTERNS:
            match = pattern.match(column)
            if match:
                name = template.format(match.group(1))
                break
        else:
            name, unit = column, ""
    return f"{name} ({unit})" if with_unit and unit else name


def _scaling_kind(column: str) -> str | None:
    if column == "modulus_MPa":
        return "modulus"
    if column in ("plateau_stress_MPa", "first_peak_stress_MPa") or re.match(
        r"^stress_at_.+pct_MPa$", column
    ):
        return "strength"
    return None


def _as_bool(series: pd.Series) -> pd.Series:
    """Booleans as they come back from a CSV: True/False, "True"/"False", 1/0."""
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _fmt_q(q: float) -> str:
    if not np.isfinite(q):
        return "n/a"
    return "< 0.001" if q < 0.001 else f"= {q:.3f}"


# ----------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------
def bh_fdr(p_values: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg q-values; NaN p-values stay NaN and are not counted."""
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return q
    pv = p[ok]
    order = np.argsort(pv)
    scaled = pv[order] * m / np.arange(1, m + 1)
    scaled = np.minimum.accumulate(scaled[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.clip(scaled, 0.0, 1.0)
    q[ok] = out
    return q


def fisher_ci(r: float, n: int, variance_factor: float = 1.0) -> tuple[float, float]:
    """95% confidence interval for a correlation via Fisher's z.

    `variance_factor` 1.06 gives the Fieller-Hartley-Pearson interval for
    Spearman's rho; 1.0 is the textbook interval for Pearson's r.
    """
    if not np.isfinite(r) or n <= 3:
        return math.nan, math.nan
    z = math.atanh(float(np.clip(r, -0.999999, 0.999999)))
    half = _Z95 * math.sqrt(variance_factor / (n - 3))
    return math.tanh(z - half), math.tanh(z + half)


def spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """(rho, ci_lo, ci_hi, p)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = stats.spearmanr(x, y)
    rho, p = float(result.statistic), float(result.pvalue)
    lo, hi = fisher_ci(rho, len(x), 1.06)
    return rho, lo, hi, p


def pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """(r, ci_lo, ci_hi, p)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = stats.pearsonr(x, y)
    r, p = float(result.statistic), float(result.pvalue)
    lo, hi = fisher_ci(r, len(x))
    return r, lo, hi, p


def _residualise(values: np.ndarray, design: np.ndarray) -> np.ndarray:
    coef, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ coef


def _log_columns(values: np.ndarray) -> list[np.ndarray]:
    """Standardised logs of the strictly positive columns."""
    out = []
    for column in values.T:
        if np.all(column > 0):
            logged = np.log(column)
            if np.ptp(logged) > 0:
                out.append((logged - logged.mean()) / logged.std())
    return out


def determination(target: np.ndarray, controls: np.ndarray) -> float:
    """How far `target` is determined by `controls` (an R^2, 0 to 1).

    Judged on ranks *and* logs. Ranks catch any monotonic link; logs catch
    the multiplicative ones geometry is full of -- relative density is
    k * wall / cell, which no linear combination of ranks reproduces, but
    which is exactly linear in logs. Without them a topology that is fully
    determined by density, wall and cell scored R^2 = 0.77 and passed as
    separable. Logs are added only while the fit keeps at least half its
    observations as spare degrees of freedom, so a small sample cannot fake
    a perfect fit.
    """
    n = len(target)
    if controls.size == 0 or n < 4:
        return 0.0
    columns = [np.ones(n)] + [stats.rankdata(c) for c in controls.T]
    extra = _log_columns(controls)
    if len(columns) + len(extra) <= n // 2:
        columns += extra
    design = np.column_stack(columns)
    targets = [stats.rankdata(target)]
    if np.all(target > 0) and np.ptp(target) > 0:
        targets.append(np.log(target))
    best = 0.0
    for t in targets:
        spread = float(((t - t.mean()) ** 2).sum())
        if spread <= 0:
            continue
        resid = _residualise(t, design)
        best = max(best, 1.0 - float((resid**2).sum()) / spread)
    return float(min(best, 1.0))


def partial_spearman(
    x: np.ndarray, y: np.ndarray, controls: np.ndarray
) -> tuple[float, float, int, float]:
    """Rank partial correlation of x and y holding `controls` fixed.

    Ranks everything, regresses the controls out of both ranked variables,
    and correlates what is left. Returns (rho, p, df, r2_x), where r2_x is the
    share of x's rank variance the controls explain -- the collinearity that
    decides whether this particular estimate can be trusted. rho is NaN when
    the controls explain x or y entirely, or leave too few degrees of freedom.
    """
    n = len(x)
    k = 0 if controls.size == 0 else controls.shape[1]
    df = n - 2 - k
    if df < 3:
        return math.nan, math.nan, df, math.nan

    rx, ry = stats.rankdata(x), stats.rankdata(y)
    columns = [np.ones(n)] + [stats.rankdata(controls[:, j]) for j in range(k)]
    design = np.column_stack(columns)
    ex, ey = _residualise(rx, design), _residualise(ry, design)
    r2_x = determination(x, controls) if rx.var() > 0 else math.nan
    if ex.var() <= 1e-9 * rx.var() or ey.var() <= 1e-9 * ry.var():
        return math.nan, math.nan, df, r2_x

    r = float(np.clip(np.corrcoef(ex, ey)[0, 1], -0.999999, 0.999999))
    t = r * math.sqrt(df / (1.0 - r * r))
    p = float(2.0 * stats.t.sf(abs(t), df))
    return r, p, df, r2_x


def _vif_from_r2(r2: float) -> float:
    if not np.isfinite(r2):
        return math.nan
    return math.inf if r2 >= 1.0 - 1e-12 else 1.0 / (1.0 - r2)


def dummy_controls(categorical: pd.DataFrame, infos: Sequence["ParameterInfo"]) -> pd.DataFrame:
    """0/1 indicator columns for each testable categorical level.

    The first level of each factor is the reference and gets no column. Rows
    whose level is too rare to test are NaN, so they drop out wherever the
    factor is used as a control rather than being lumped in with a level.
    """
    columns: dict[str, pd.Series] = {}
    for info in infos:
        values = categorical[info.column]
        eligible = values.isin(info.levels)
        for level in info.levels[1:]:
            columns[f"{info.name}={level}"] = (values == level).astype(float).where(eligible)
    return pd.DataFrame(columns, index=categorical.index)


def control_matrix(
    numeric: pd.DataFrame,
    dummies: pd.DataFrame,
    exclude: str | Sequence[str],
    rows: pd.Series,
    exclude_factors: Sequence[str] = (),
) -> tuple[np.ndarray, list[str]]:
    """Every other parameter usable as a control on `rows`.

    A control must be recorded on all of those rows and vary across them; a
    parameter only defined for some patterns is used only where complete.
    `exclude` names numeric columns and `exclude_factors` categorical factors
    to leave out (the parameter under test, and its near-duplicates).
    """
    excluded = {exclude} if isinstance(exclude, str) else set(exclude)
    skip_prefixes = tuple(f"{factor}=" for factor in exclude_factors)
    blocks, names = [], []
    for column in numeric.columns:
        if column in excluded:
            continue
        values = numeric.loc[rows, column]
        if values.notna().all() and values.nunique() > 1:
            blocks.append(values.to_numpy(dtype=float))
            names.append(display_name(column))
    for column in dummies.columns:
        if skip_prefixes and column.startswith(skip_prefixes):
            continue
        values = dummies.loc[rows, column]
        if values.notna().all() and values.nunique() > 1:
            blocks.append(values.to_numpy(dtype=float))
            names.append(column)
    n = int(rows.sum())
    return (np.column_stack(blocks) if blocks else np.empty((n, 0))), names


def rank_vif(
    numeric: pd.DataFrame,
    dummies: pd.DataFrame | None = None,
    exclusions: dict[str, tuple[list[str], list[str]]] | None = None,
) -> dict[str, float]:
    """Variance inflation factor of each numeric parameter, on ranks.

    VIF_j = 1 / (1 - R^2_j), with R^2_j from regressing parameter j on every
    other parameter (categorical ones as indicators) over the tests where j is
    recorded. 1 means independent of the rest; above ~10, the parameter is so
    predictable from the others that its separate effect is not identifiable.
    `exclusions` leaves out a parameter's confounded twin, which is reported
    as such separately -- VIF then measures entanglement beyond it.
    """
    dummies = dummies if dummies is not None else pd.DataFrame(index=numeric.index)
    exclusions = exclusions or {}
    out: dict[str, float] = {}
    for column in numeric.columns:
        rows = numeric[column].notna()
        numeric_out, factors_out = exclusions.get(column, ([column], []))
        z, _ = control_matrix(numeric, dummies, numeric_out, rows, factors_out)
        n, k = int(rows.sum()), z.shape[1]
        if k == 0:
            out[column] = 1.0
            continue
        if n < k + 3:
            out[column] = math.nan
            continue
        r2 = determination(numeric.loc[rows, column].to_numpy(dtype=float), z)
        out[column] = _vif_from_r2(r2)
    return out


def correlation_ratio(values: np.ndarray, groups: np.ndarray) -> float:
    """eta: how well group membership predicts the (ranked) values, 0 to 1.

    The square root of the between-group share of rank variance, so it sits
    on the same 0-1 scale as |rho| when judging how entangled two parameters
    are.
    """
    ranks = stats.rankdata(values)
    total = float(((ranks - ranks.mean()) ** 2).sum())
    if total <= 0:
        return math.nan
    between = sum(
        (groups == g).sum() * (ranks[groups == g].mean() - ranks.mean()) ** 2
        for g in np.unique(groups)
    )
    return float(math.sqrt(between / total))


def cramers_v(a: np.ndarray, b: np.ndarray) -> float:
    """Association between two categorical factors, 0 (none) to 1 (identical)."""
    table = pd.crosstab(a, b).to_numpy(dtype=float)
    if min(table.shape) < 2:
        return math.nan
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    return float(math.sqrt(chi2 / (table.sum() * (min(table.shape) - 1))))


def quade_partial(
    y: np.ndarray, groups: np.ndarray, controls: np.ndarray
) -> tuple[float, float, float, float]:
    """Does a categorical factor matter once the controls are held fixed?

    Quade's rank analysis of covariance: rank y, regress out the ranked
    controls, then a one-way ANOVA of the residuals across the factor's
    levels. Returns (F, p, partial eta^2, r2_groups), where r2_groups is how
    well the controls predict level membership -- high means the factor's
    effect cannot be separated from theirs.
    """
    n = len(y)
    k = 0 if controls.size == 0 else controls.shape[1]
    levels = np.unique(groups)
    df1, df2 = len(levels) - 1, n - len(levels) - k
    if df1 < 1 or df2 < 3:
        return math.nan, math.nan, math.nan, math.nan

    design = np.column_stack([np.ones(n)] + [stats.rankdata(controls[:, j]) for j in range(k)])
    resid = _residualise(stats.rankdata(y), design)
    grand = resid.mean()
    ssb = float(sum((groups == g).sum() * (resid[groups == g].mean() - grand) ** 2 for g in levels))
    ssw = float(sum(((resid[groups == g] - resid[groups == g].mean()) ** 2).sum() for g in levels))
    if ssw <= 0:
        return math.nan, math.nan, math.nan, math.nan

    r2_groups = 0.0
    for g in levels[1:]:
        member = (groups == g).astype(float)
        if member.std() > 0:
            r2_groups = max(r2_groups, _linear_r2(member, controls))

    f_stat = (ssb / df1) / (ssw / df2)
    return f_stat, float(stats.f.sf(f_stat, df1, df2)), ssb / (ssb + ssw), r2_groups


def _linear_r2(target: np.ndarray, controls: np.ndarray) -> float:
    """R^2 of a 0/1 indicator on ranked and logged controls (see determination)."""
    n = len(target)
    columns = [np.ones(n)] + [stats.rankdata(c) for c in controls.T]
    extra = _log_columns(controls)
    if len(columns) + len(extra) <= n // 2:
        columns += extra
    resid = _residualise(target, np.column_stack(columns))
    spread = float(((target - target.mean()) ** 2).sum())
    return 1.0 - float((resid**2).sum()) / spread if spread > 0 else 0.0


def kruskal_effect(
    values: pd.Series, groups: pd.Series, min_group_size: int
) -> dict[str, Any] | None:
    """Kruskal-Wallis test of `values` across `groups`, with epsilon-squared.

    epsilon^2 = H / (n - 1) is the share of rank variance explained by group
    membership (0 to 1). None when fewer than two groups are big enough.
    """
    frame = pd.DataFrame({"v": values, "g": groups}).dropna()
    counts = frame["g"].value_counts()
    keep = sorted(counts[counts >= min_group_size].index, key=str)
    if len(keep) < 2:
        return None
    frame = frame[frame["g"].isin(keep)]
    samples = [frame.loc[frame["g"] == level, "v"].to_numpy(dtype=float) for level in keep]
    try:
        h, p = stats.kruskal(*samples)
    except ValueError:  # every value identical
        h, p = 0.0, 1.0
    n = len(frame)
    eps2 = float(np.clip(h / (n - 1), 0.0, 1.0)) if n > 1 else math.nan
    medians = {str(level): float(np.median(s)) for level, s in zip(keep, samples)}
    return {
        "H": float(h),
        "p": float(p),
        "epsilon_sq": eps2,
        "n": n,
        "k": len(keep),
        "medians": medians,
        "counts": {str(level): int(counts[level]) for level in keep},
    }


def power_law_fit(
    x: np.ndarray, y: np.ndarray, offsets: np.ndarray | None = None
) -> dict[str, Any] | None:
    """Fit y = C * x^n by least squares in log-log space.

    `offsets` (one 0/1 column per non-reference level) gives each level of a
    categorical factor its own prefactor C while sharing one exponent n. That
    is the Gibson-Ashby form: C depends on cell geometry, n on deformation
    mode. Without it, a factor that shifts both density and strength -- a
    denser *and* stronger topology -- tilts the pooled slope.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if offsets is not None and offsets.size:
        ok &= np.isfinite(offsets).all(axis=1)
    n = int(ok.sum())
    k = 0 if offsets is None or offsets.size == 0 else offsets.shape[1]
    if n < k + 4:
        return None
    lx, ly = np.log10(x[ok]), np.log10(y[ok])
    if np.ptp(lx) == 0:
        return None

    columns = [np.ones(n), lx] + ([offsets[ok, j] for j in range(k)] if k else [])
    design = np.column_stack(columns)
    coef, *_ = np.linalg.lstsq(design, ly, rcond=None)
    resid = ly - design @ coef
    dof = n - design.shape[1]
    sigma2 = float(resid @ resid) / dof
    try:
        cov = sigma2 * np.linalg.inv(design.T @ design)
    except np.linalg.LinAlgError:
        return None
    se = math.sqrt(max(float(cov[1, 1]), 0.0))
    t = float(stats.t.ppf(0.975, dof))
    total = float(((ly - ly.mean()) ** 2).sum())
    return {
        "exponent": float(coef[1]),
        "ci_lo": float(coef[1] - t * se),
        "ci_hi": float(coef[1] + t * se),
        "prefactor": float(10 ** coef[0]),
        # Prefactor of each non-reference level, in `offsets` column order.
        "level_prefactors": [float(10 ** (coef[0] + c)) for c in coef[2:]],
        "r2": 1.0 - float(resid @ resid) / total if total > 0 else math.nan,
        "p": float(2 * stats.t.sf(abs(coef[1] / se), dof)) if se > 0 else 0.0,
        "n": n,
    }


def interpret_exponent(kind: str, lo: float, hi: float) -> str:
    """Which deformation mode a fitted exponent's 95% CI is consistent with."""
    refs = SCALING_REFERENCES.get(kind, {})
    if not refs or not (np.isfinite(lo) and np.isfinite(hi)):
        return ""
    inside = [name for name, value in refs.items() if lo <= value <= hi]
    if len(inside) == 1:
        return f"consistent with {inside[0]} (n ≈ {refs[inside[0]]:g})"
    if len(inside) > 1:
        return "cannot distinguish bending from stretching (CI too wide)"
    if hi < min(refs.values()):
        return "shallower than either mode"
    if lo > max(refs.values()):
        return "steeper than either mode"
    return "between bending- and stretch-dominated"


def _strength(effect: float, kind: str) -> str:
    if not np.isfinite(effect):
        return ""
    bands = (0.01, 0.08, 0.26) if kind == "categorical" else (0.1, 0.3, 0.5)
    labels = ("negligible", "weak", "moderate", "strong")
    a = abs(effect)
    for band, label in zip(bands, labels):
        if a < band:
            return label
    return labels[-1]


# ----------------------------------------------------------------------
# Data preparation
# ----------------------------------------------------------------------
@dataclass
class ParameterInfo:
    """One candidate parameter and whether it could be analysed."""

    column: str
    kind: str                       # "numeric" | "datetime" | "categorical"
    n: int = 0
    n_distinct: int = 0
    included: bool = False
    reason: str = ""
    levels: list[str] = field(default_factory=list)
    vif: float = math.nan
    #: Datetime parameters are analysed as days since this moment.
    origin: pd.Timestamp | None = None
    #: Parameters so strongly associated with this one that the data cannot
    #: tell them apart. Never held fixed when testing it (that would erase
    #: both); reported as confounded instead.
    duplicates: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return display_name(self.column)

    @property
    def is_ordered(self) -> bool:
        """Numeric or datetime: has an order, so rank correlation applies."""
        return self.kind in ("numeric", "datetime")

    @property
    def axis_label(self) -> str:
        if self.kind == "datetime" and self.origin is not None:
            return f"{self.name} (days since {self.origin:%Y-%m-%d})"
        return self.name


def _parameter_order(metrics: pd.DataFrame) -> list[str]:
    params = [c for c in metrics.columns if c.startswith("param_")]
    ordered = [c for c in ("relative_density",) if c in metrics.columns]
    ordered += params
    ordered += [c for c in CORE_NUMERIC[1:] + CORE_DATES + CORE_CATEGORICAL if c in metrics.columns]
    return ordered


def discover_parameters(
    metrics: pd.DataFrame,
    cfg: AnalysisConfig = DEFAULT_CONFIG,
    include: Sequence[str] | None = None,
) -> list[ParameterInfo]:
    """Classify each candidate parameter and decide whether it can be tested."""
    candidates = _parameter_order(metrics)
    if include:
        resolved = []
        for name in include:
            column = resolve_column(metrics, name)
            if column is None:
                available = ", ".join(display_name(c) for c in candidates) or "none"
                raise ValueError(f"unknown parameter {name!r}; available: {available}")
            resolved.append(column)
        candidates = list(dict.fromkeys(resolved))

    infos: list[ParameterInfo] = []
    for column in candidates:
        raw = metrics[column]
        present = raw.notna() & (raw.astype(str).str.strip() != "")
        numeric = pd.to_numeric(raw.where(present), errors="coerce")
        is_numeric = bool(present.any()) and bool(numeric[present].notna().all())
        dates = None
        if column in CORE_CATEGORICAL:
            is_numeric = False
        elif not is_numeric:
            dates = parse_datetimes(raw)

        kind = "numeric" if is_numeric else "datetime" if dates is not None else "categorical"
        info = ParameterInfo(column=column, kind=kind)
        info.n = int(present.sum())
        if kind == "numeric":
            values = numeric[present]
        elif kind == "datetime":
            values = dates[present]
            info.origin = values.min().normalize()
        else:
            values = raw[present].astype(str).str.strip()
        info.n_distinct = int(values.nunique())

        if info.n == 0:
            info.reason = "no values recorded"
        elif info.n_distinct < 2:
            info.reason = "the same value for every test"
        elif info.is_ordered:
            if info.n < cfg.correlation_min_n:
                info.reason = f"only {info.n} tests have a value (need {cfg.correlation_min_n})"
            else:
                info.included = True
        else:
            counts = values.value_counts()
            levels = sorted(counts[counts >= cfg.correlation_min_group_size].index, key=str)
            if len(levels) < 2:
                info.reason = (
                    f"{info.n_distinct} levels, but fewer than two with "
                    f"≥{cfg.correlation_min_group_size} tests each"
                    + (" -- unique per test" if info.n_distinct == info.n else "")
                )
            elif int(counts[levels].sum()) < cfg.correlation_min_n:
                info.reason = f"only {int(counts[levels].sum())} tests in testable levels"
            else:
                info.included = True
                info.levels = [str(level) for level in levels]
        infos.append(info)

    # test_date = print_date + print_to_test_days exactly, so with all three
    # varying one is redundant. Keep print date (manufacturing drift) and age
    # (ageing/moisture); test date is available on request.
    by_column = {i.column: i for i in infos}
    trio = [by_column.get(c) for c in ("print_date", "test_date", AGE_COLUMN)]
    if not include and all(i is not None and i.included for i in trio):
        trio[1].included = False
        trio[1].reason = (
            "redundant -- it equals print_date + print_to_test_days; pass "
            "--parameters to analyse it in place of one of those"
        )
    return infos


@dataclass
class PropertyData:
    """Property values with lower bounds removed, and what was removed."""

    values: pd.DataFrame            # censored -> NaN
    raw: pd.DataFrame               # as reported
    censored: pd.DataFrame          # bool mask of removed lower bounds
    n_censored: dict[str, int]
    skipped: dict[str, str]         # property -> why it was not analysed


def property_data(
    metrics: pd.DataFrame,
    properties: Sequence[str],
    cfg: AnalysisConfig = DEFAULT_CONFIG,
) -> PropertyData:
    values, raw, censored = {}, {}, {}
    n_censored: dict[str, int] = {}
    skipped: dict[str, str] = {}
    for prop in properties:
        if prop not in metrics.columns:
            skipped[prop] = "not in the metrics table"
            continue
        series = pd.to_numeric(metrics[prop], errors="coerce")
        mask = pd.Series(False, index=metrics.index)
        flag = CENSOR_FLAGS.get(prop)
        if flag and flag in metrics.columns:
            mask = _as_bool(metrics[flag]) & series.notna()
        usable = series.where(~mask) if not cfg.correlation_include_censored else series
        finite = int(usable.notna().sum())
        if finite < cfg.correlation_min_n:
            if mask.any() and not cfg.correlation_include_censored:
                skipped[prop] = (
                    f"{int(mask.sum())} of {int(series.notna().sum())} values are "
                    f"lower bounds (test stopped at the load limit); {finite} left"
                )
            else:
                skipped[prop] = f"only {finite} tests have a value"
            continue
        values[prop] = usable
        raw[prop] = series
        censored[prop] = mask if not cfg.correlation_include_censored else mask & False
        n_censored[prop] = int(censored[prop].sum())
    index = metrics.index
    return PropertyData(
        values=pd.DataFrame(values, index=index),
        raw=pd.DataFrame(raw, index=index),
        censored=pd.DataFrame(censored, index=index),
        n_censored=n_censored,
        skipped=skipped,
    )


# ----------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------
@dataclass
class CorrelationResult:
    table: pd.DataFrame
    parameters: list[ParameterInfo]
    properties: list[str]
    collinearity: pd.DataFrame
    scaling: pd.DataFrame
    props: PropertyData
    numeric: pd.DataFrame             # numeric parameter values by column
    categorical: pd.DataFrame         # categorical parameter values by column
    n_tests: int
    alpha: float
    vif_threshold: float
    collinear_rho: float

    @property
    def included(self) -> list[ParameterInfo]:
        return [p for p in self.parameters if p.included]

    @property
    def excluded(self) -> list[ParameterInfo]:
        return [p for p in self.parameters if not p.included]

    def cell(self, parameter: str, prop: str) -> pd.Series | None:
        hit = self.table[(self.table["parameter_column"] == parameter) & (self.table["property"] == prop)]
        return None if hit.empty else hit.iloc[0]

    def parameters_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "parameter": p.name,
                    "column": p.column,
                    "kind": p.kind,
                    "n_tests": p.n,
                    "n_distinct": p.n_distinct,
                    "analysed": p.included,
                    "levels": ", ".join(p.levels),
                    "vif": p.vif,
                    "confounded_with": ", ".join(p.duplicates),
                    "days_since": "" if p.origin is None else f"{p.origin:%Y-%m-%d}",
                    "reason_not_analysed": p.reason,
                }
                for p in self.parameters
            ]
        )


def analyse(
    metrics: pd.DataFrame,
    cfg: AnalysisConfig = DEFAULT_CONFIG,
    properties: Sequence[str] | None = None,
    parameters: Sequence[str] | None = None,
) -> CorrelationResult:
    """Test every included parameter against every usable property."""
    metrics = metrics.reset_index(drop=True)
    infos = discover_parameters(metrics, cfg, parameters)
    included = [p for p in infos if p.included]
    ordered_infos = [p for p in included if p.is_ordered]
    categorical_infos = [p for p in included if p.kind == "categorical"]

    # Numeric and datetime parameters share one numeric frame; dates enter as
    # days since their earliest value (rank statistics ignore the offset).
    columns: dict[str, pd.Series] = {}
    for info in ordered_infos:
        if info.kind == "datetime":
            stamps = parse_datetimes(metrics[info.column])
            columns[info.column] = (stamps - info.origin).dt.total_seconds() / 86400.0
        else:
            columns[info.column] = pd.to_numeric(metrics[info.column], errors="coerce")
    numeric = pd.DataFrame(columns, index=metrics.index)
    categorical = pd.DataFrame(
        {
            p.column: metrics[p.column].where(
                metrics[p.column].notna() & (metrics[p.column].astype(str).str.strip() != "")
            ).astype(object).map(lambda v: str(v).strip() if pd.notna(v) else np.nan)
            for p in categorical_infos
        },
        index=metrics.index,
    )

    dummies = dummy_controls(categorical, categorical_infos)
    association = _association_matrix(included, numeric, categorical)
    for info in included:
        info.duplicates = [
            other.name for other in included
            if other is not info
            and np.isfinite(association.loc[info.name, other.name])
            and abs(association.loc[info.name, other.name]) >= cfg.correlation_collinear_rho
        ]
    by_name = {p.name: p for p in included}
    exclusions = {p.column: _exclusions(p, by_name) for p in ordered_infos}
    for column, vif in rank_vif(numeric, dummies, exclusions).items():
        by_name[display_name(column)].vif = vif

    requested = list(properties) if properties else list(cfg.correlation_properties)
    props = property_data(metrics, requested, cfg)
    prop_columns = list(props.values.columns)

    rows: list[dict[str, Any]] = []
    for prop in prop_columns:
        y_all = props.values[prop]
        for info in ordered_infos:
            rows.append(_numeric_row(info, prop, numeric, dummies, y_all, props, cfg, by_name))
        for info in categorical_infos:
            rows.append(
                _categorical_row(info, prop, categorical[info.column], y_all, props, cfg,
                                 numeric, dummies, by_name)
            )

    table = pd.DataFrame(rows, columns=_TABLE_COLUMNS)
    if not table.empty:
        table["q_value"] = bh_fdr(table["p_value"])
        table["significant"] = table["q_value"] < cfg.correlation_fdr_alpha
        table["partial_q"] = bh_fdr(table["partial_p"])
        table["partial_significant"] = table["partial_q"] < cfg.correlation_fdr_alpha
        table["strength"] = [
            _strength(e, k) for e, k in zip(table["effect"], table["kind"])
        ]

    scaling = _scaling_laws(
        metrics, numeric, props, prop_columns, cfg, categorical, categorical_infos, table
    )

    return CorrelationResult(
        table=table,
        parameters=infos,
        properties=prop_columns,
        collinearity=association,
        scaling=scaling,
        props=props,
        numeric=numeric,
        categorical=categorical,
        n_tests=len(metrics),
        alpha=cfg.correlation_fdr_alpha,
        vif_threshold=cfg.correlation_vif_threshold,
        collinear_rho=cfg.correlation_collinear_rho,
    )


def _association_matrix(
    infos: Sequence[ParameterInfo], numeric: pd.DataFrame, categorical: pd.DataFrame
) -> pd.DataFrame:
    """How entangled each pair of parameters is, whatever their kinds.

    numeric-numeric: Spearman rho (signed); numeric-categorical: correlation
    ratio eta; categorical-categorical: Cramer's V. All on a 0-1 magnitude
    scale, so one threshold decides "too entangled to separate".
    """
    names = [p.name for p in infos]
    matrix = pd.DataFrame(np.nan, index=names, columns=names, dtype=float)

    def values(info: ParameterInfo) -> pd.Series:
        if info.is_ordered:
            return numeric[info.column]
        return categorical[info.column].where(categorical[info.column].isin(info.levels))

    for i, a in enumerate(infos):
        matrix.loc[a.name, a.name] = 1.0
        for b in infos[i + 1:]:
            pair = pd.DataFrame({"a": values(a), "b": values(b)}).dropna()
            if len(pair) < 3 or pair["a"].nunique() < 2 or pair["b"].nunique() < 2:
                continue
            if a.is_ordered and b.is_ordered:
                value = spearman(pair["a"].to_numpy(float), pair["b"].to_numpy(float))[0]
            elif a.is_ordered:
                value = correlation_ratio(pair["a"].to_numpy(float), pair["b"].to_numpy())
            elif b.is_ordered:
                value = correlation_ratio(pair["b"].to_numpy(float), pair["a"].to_numpy())
            else:
                value = cramers_v(pair["a"].to_numpy(), pair["b"].to_numpy())
            matrix.loc[a.name, b.name] = matrix.loc[b.name, a.name] = value
    return matrix


def _exclusions(info: ParameterInfo, by_name: dict[str, ParameterInfo]) -> tuple[list[str], list[str]]:
    """(numeric columns, categorical factor names) not to hold fixed for `info`."""
    twins = [by_name[name] for name in info.duplicates if name in by_name]
    numeric_out = [t.column for t in twins if t.is_ordered]
    factors_out = [t.name for t in twins if t.kind == "categorical"]
    if info.is_ordered:
        numeric_out.append(info.column)
    else:
        factors_out.append(info.name)
    return numeric_out, factors_out


def _confounding_note(info: ParameterInfo) -> str:
    if not info.duplicates:
        return ""
    return (
        f"confounded with {', '.join(info.duplicates)} -- not held fixed, as the data "
        f"cannot separate them; read as one shared effect"
    )


_TABLE_COLUMNS = [
    "parameter", "parameter_column", "kind", "property", "n", "n_censored_excluded",
    "effect", "effect_measure", "effect_ci_lo", "effect_ci_hi", "p_value",
    "q_value", "significant", "strength", "direction",
    "pearson_r", "pearson_ci_lo", "pearson_ci_hi", "pearson_p",
    "partial_rho", "partial_eta_sq", "partial_p", "partial_q", "partial_significant",
    "partial_n", "partial_controls", "partial_reliable",
    "kruskal_H", "n_groups", "groups", "note",
]


def _base_row(info: ParameterInfo, prop: str, props: PropertyData) -> dict[str, Any]:
    row = {c: math.nan for c in _TABLE_COLUMNS}
    row.update(
        parameter=info.name, parameter_column=info.column, kind=info.kind,
        property=prop, n=0, n_censored_excluded=props.n_censored.get(prop, 0),
        effect_measure="epsilon_squared" if info.kind == "categorical" else "spearman_rho",
        direction="", groups="", note="", partial_controls="",
        significant=False, partial_significant=False, partial_reliable=False,
    )
    return row


def _numeric_row(
    info: ParameterInfo,
    prop: str,
    numeric: pd.DataFrame,
    dummies: pd.DataFrame,
    y_all: pd.Series,
    props: PropertyData,
    cfg: AnalysisConfig,
    by_name: dict[str, ParameterInfo] | None = None,
) -> dict[str, Any]:
    row = _base_row(info, prop, props)
    x_all = numeric[info.column]
    ok = x_all.notna() & y_all.notna()
    n = int(ok.sum())
    row["n"] = n
    if n < cfg.correlation_min_n:
        row["note"] = f"only {n} tests have both values"
        return row
    x = x_all[ok].to_numpy(dtype=float)
    y = y_all[ok].to_numpy(dtype=float)
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        row["note"] = "no variation among the tests that have both values"
        return row

    rho, lo, hi, p = spearman(x, y)
    r, rlo, rhi, rp = pearson(x, y)
    row.update(
        effect=rho, effect_ci_lo=lo, effect_ci_hi=hi, p_value=p,
        direction="+" if rho > 0 else "−",
        pearson_r=r, pearson_ci_lo=rlo, pearson_ci_hi=rhi, pearson_p=rp,
    )

    # Partial: hold every other parameter fixed -- categorical ones included.
    # Leaving a categorical confounder out is not neutral: with density held
    # fixed, a bigger cell can stand in for a stiffer topology and the partial
    # flips sign.
    numeric_out, factors_out = _exclusions(info, by_name or {})
    z, names = control_matrix(numeric, dummies, numeric_out, ok, factors_out)
    confounded = _confounding_note(info)
    if not names:
        row["note"] = confounded or "no other parameters to hold fixed"
        return row
    prho, pp, df, r2_x = partial_spearman(x, y, z)
    local_vif = _vif_from_r2(r2_x)
    reliable = bool(np.isfinite(prho)) and local_vif < cfg.correlation_vif_threshold
    row.update(
        partial_rho=prho, partial_p=pp, partial_n=n,
        partial_controls=", ".join(names), partial_reliable=reliable,
    )
    if not np.isfinite(prho):
        row["note"] = "fully determined by the other parameters; no partial estimate"
    elif not reliable:
        row["note"] = (
            f"largely determined by the other parameters (VIF {local_vif:.1f}); "
            f"partial estimate unreliable"
        )
    if confounded:
        row["note"] = "; ".join(x for x in (confounded, row["note"]) if x)
    return row


def _categorical_row(
    info: ParameterInfo,
    prop: str,
    groups: pd.Series,
    y_all: pd.Series,
    props: PropertyData,
    cfg: AnalysisConfig,
    numeric: pd.DataFrame | None = None,
    dummies: pd.DataFrame | None = None,
    by_name: dict[str, ParameterInfo] | None = None,
) -> dict[str, Any]:
    row = _base_row(info, prop, props)
    result = kruskal_effect(y_all, groups, cfg.correlation_min_group_size)
    if result is None:
        row["note"] = "fewer than two levels with enough tests for this property"
        row["n"] = int((groups.notna() & y_all.notna()).sum())
        return row
    row["n"] = result["n"]
    if result["n"] < cfg.correlation_min_n:
        row["note"] = f"only {result['n']} tests in testable levels"
        return row
    highest = max(result["medians"], key=result["medians"].get)
    row.update(
        effect=result["epsilon_sq"], p_value=result["p"], kruskal_H=result["H"],
        n_groups=result["k"], direction=f"highest: {highest}",
        groups=", ".join(f"{lvl} (n={result['counts'][lvl]}, median {med:.4g})"
                         for lvl, med in result["medians"].items()),
    )

    # Partial: the same question with every other parameter held fixed. A
    # batch or bed-position effect of a few percent is invisible next to
    # pattern-to-pattern differences until those are accounted for.
    if numeric is None or dummies is None:
        return row
    rows = y_all.notna() & groups.isin(list(result["counts"]))
    numeric_out, factors_out = _exclusions(info, by_name or {})
    z, names = control_matrix(numeric, dummies, numeric_out, rows, factors_out)
    confounded = _confounding_note(info)
    if not names:
        row["note"] = confounded or "no other parameters to hold fixed"
        return row
    f_stat, p, eta2, r2_groups = quade_partial(
        y_all[rows].to_numpy(float), groups[rows].to_numpy(), z
    )
    local_vif = _vif_from_r2(r2_groups)
    reliable = bool(np.isfinite(p)) and local_vif < cfg.correlation_vif_threshold
    row.update(
        partial_eta_sq=eta2, partial_p=p, partial_n=int(rows.sum()),
        partial_controls=", ".join(names), partial_reliable=reliable,
    )
    notes = [confounded]
    if np.isfinite(p) and not reliable:
        notes.append(
            f"largely determined by the other parameters (VIF {local_vif:.1f}); "
            f"partial estimate unreliable"
        )
    row["note"] = "; ".join(x for x in notes if x)
    return row


def _strongest_factor(
    table: pd.DataFrame | None, prop: str, infos: Sequence[ParameterInfo]
) -> list[ParameterInfo]:
    """The categorical factor that matters most for `prop` (as a 0/1-item list)."""
    if table is None or table.empty or not infos:
        return []
    rows = table[(table["property"] == prop) & (table["kind"] == "categorical") & table["significant"]]
    if rows.empty:
        return []
    best = rows.loc[rows["effect"].idxmax(), "parameter_column"]
    return [i for i in infos if i.column == best]


def _scaling_laws(
    metrics: pd.DataFrame,
    numeric: pd.DataFrame,
    props: PropertyData,
    prop_columns: list[str],
    cfg: AnalysisConfig,
    categorical: pd.DataFrame | None = None,
    categorical_infos: Sequence[ParameterInfo] = (),
    table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Gibson-Ashby power laws of each modulus/strength property vs rho*.

    One row per property with a shared exponent ("all"), the categorical
    factors absorbed as per-level prefactors; then one row per level of the
    factor with the strongest significant effect on that property, fitted on
    its own -- so a level that genuinely scales differently (stretch- vs
    bending-dominated) shows up rather than being averaged away.
    """
    columns = [
        "property", "kind", "group", "exponent", "ci_lo", "ci_hi", "prefactor",
        "r2", "n", "interpretation", "held_fixed",
    ]
    if "relative_density" not in numeric.columns:
        return pd.DataFrame(columns=columns)
    categorical = categorical if categorical is not None else pd.DataFrame(index=metrics.index)
    dummies = dummy_controls(categorical, categorical_infos)
    rho = numeric["relative_density"]

    rows = []
    for prop in prop_columns:
        kind = _scaling_kind(prop)
        if kind is None:
            continue
        y = props.values[prop]
        valid = rho.notna() & y.notna() & (rho > 0) & (y > 0)
        offsets_used = [
            c for c in dummies.columns
            if dummies.loc[valid, c].notna().all() and dummies.loc[valid, c].nunique() > 1
        ]
        offsets = dummies[offsets_used].to_numpy(dtype=float) if offsets_used else None
        fit = power_law_fit(rho.to_numpy(), y.to_numpy(), offsets)
        if fit is None or fit["n"] < cfg.correlation_min_n:
            continue
        held = sorted({c.split("=")[0] for c in offsets_used})
        rows.append(
            {
                "property": prop, "kind": kind, "group": "all",
                **{k: fit[k] for k in ("exponent", "ci_lo", "ci_hi", "prefactor", "r2", "n")},
                "interpretation": interpret_exponent(kind, fit["ci_lo"], fit["ci_hi"]),
                "held_fixed": ", ".join(held),
            }
        )
        for info in _strongest_factor(table, prop, categorical_infos):
            for level in info.levels:
                mask = valid & (categorical[info.column] == level)
                if int(mask.sum()) < cfg.correlation_min_n:
                    continue
                sub = power_law_fit(rho[mask].to_numpy(), y[mask].to_numpy())
                if sub is None:
                    continue
                rows.append(
                    {
                        "property": prop, "kind": kind, "group": f"{info.name}={level}",
                        **{k: sub[k] for k in ("exponent", "ci_lo", "ci_hi", "prefactor", "r2", "n")},
                        "interpretation": interpret_exponent(kind, sub["ci_lo"], sub["ci_hi"]),
                        "held_fixed": "",
                    }
                )
    return pd.DataFrame(rows, columns=columns)


# ----------------------------------------------------------------------
# Text summary
# ----------------------------------------------------------------------
def summarise(result: CorrelationResult, max_rows: int = 15) -> str:
    """Plain-language findings for the terminal and summary.txt."""
    table = result.table
    lines = [
        f"Parameter–property correlation: {result.n_tests} tests, "
        f"{len(result.included)} parameters × {len(result.properties)} properties"
    ]
    constant = [p.name for p in result.excluded if p.reason == "the same value for every test"]
    others = [p for p in result.excluded if p.reason != "the same value for every test"]
    if constant:
        lines.append(f"  Same for every test, so nothing to correlate: {', '.join(constant)}")
    if others:
        lines.append("  Not analysed:")
        for info in others:
            lines.append(f"    {info.name}: {info.reason}")
    for prop, why in result.props.skipped.items():
        if prop in table.get("property", pd.Series(dtype=str)).values:
            continue
        lines.append(f"  Property {prop} skipped: {why}")

    if table.empty or not table["p_value"].notna().any():
        lines.append("  Nothing could be tested -- see the reasons above.")
        return "\n".join(lines)

    tested = table[table["p_value"].notna()]
    sig = tested[tested["significant"]].copy()
    lines.append(
        f"\n  {len(sig)} of {len(tested)} pairs significant "
        f"(FDR q < {result.alpha:g}; Spearman ρ for numeric, ε² for categorical):"
    )
    if sig.empty:
        lines.append("    none")
    sig["abs_effect"] = sig["effect"].abs()
    for _, row in sig.sort_values(["q_value", "abs_effect"], ascending=[True, False]).head(max_rows).iterrows():
        lines.append("    " + _summary_line(row))
    if len(sig) > max_rows:
        lines.append(f"    … {len(sig) - max_rows} more in correlations.csv")

    reliable = tested[tested["partial_reliable"].astype(bool)]
    partial_sig = reliable[reliable["partial_significant"]]
    never = [
        p.name for p in result.included
        if not sig["parameter"].eq(p.name).any() and not partial_sig["parameter"].eq(p.name).any()
    ]
    if never:
        lines.append(f"\n  No association with any property, pairwise or partial: {', '.join(never)}")

    n_props = len(result.properties)
    masked = reliable[~reliable["significant"] & reliable["partial_significant"]]
    if not masked.empty:
        lines.append(
            "\n  Hidden until other parameters are held fixed (masked effects; pairwise n.s. for these):"
        )
        lines += _grouped_lines(masked, n_props, partial=True)
    borrowed = reliable[
        reliable["significant"]
        & ~reliable["partial_significant"]
        & ((reliable["kind"] == "categorical") | (reliable["partial_rho"].abs() < 0.5 * reliable["effect"].abs()))
    ]
    if not borrowed.empty:
        lines.append("\n  Fades when other parameters are held fixed (likely borrowed from a correlated one):")
        lines += _grouped_lines(borrowed, n_props, partial=False)
    if any("isolated" in line for line in lines):
        lines.append(
            f"\n  * isolated: significant for one property only. At FDR {result.alpha:g}, about "
            f"{result.alpha:.0%} of all discoveries are expected to be false, and a real\n"
            f"    effect usually shows across related properties (strength, energy, modulus) -- "
            f"treat these as leads to re-test, not findings."
        )

    pairs = _collinear_pairs(result)
    if pairs:
        lines.append(
            f"\n  Confounded parameters (association ≥ {result.collinear_rho:g}) — they move "
            f"together, so the data cannot separate their effects:"
        )
        for a, b, value, measure in pairs:
            lines.append(f"    {a} ~ {b}  ({measure} = {value:+.2f})" if measure == "ρ"
                         else f"    {a} ~ {b}  ({measure} = {value:.2f})")
        lines.append(
            "    Each is tested without holding the other fixed; read a shared result as one "
            "effect with two names."
        )
    unreliable = [
        p for p in result.included
        if p.is_ordered and (p.vif == math.inf or (np.isfinite(p.vif) and p.vif >= result.vif_threshold))
    ]
    if unreliable:
        if not pairs:
            lines.append("")
        lines.append(
            "  Largely predictable from the other parameters (VIF ≥ "
            f"{result.vif_threshold:g}): "
            + ", ".join(f"{p.name} ({p.vif:.0f})" if np.isfinite(p.vif) else f"{p.name} (∞)" for p in unreliable)
        )
        lines.append(
            "    Partials withheld where still entangled. If one is derived from the others "
            "(relative density from cell and wall size, say),\n"
            "    analyse one set at a time: scripts/correlate.py --parameters <names>"
        )

    cat_sig = sorted(
        set(sig[sig["kind"] == "categorical"]["parameter"])
        | set(partial_sig[partial_sig["kind"] == "categorical"]["parameter"])
    )
    if cat_sig:
        lines.append(
            f"\n  Categorical factors with an effect: {', '.join(cat_sig)}. To look inside one "
            f"level: scripts/correlate.py --where {cat_sig[0]}=<level>"
        )

    if not result.scaling.empty:
        overall = result.scaling[result.scaling["group"] == "all"]
        held = next((h for h in overall["held_fixed"] if h), "")
        lines.append(
            "\n  Scaling with relative density (Gibson–Ashby, property ∝ ρ*ⁿ"
            + (f"; one exponent, prefactor per {held}" if held else "")
            + "):"
        )
        for _, row in overall.iterrows():
            lines.append(
                f"    {property_label(row['property']):<14} n = {row['exponent']:.2f} "
                f"[{row['ci_lo']:.2f}, {row['ci_hi']:.2f}]  R² = {row['r2']:.2f}  — {row['interpretation']}"
            )
            levels = result.scaling[
                (result.scaling["property"] == row["property"]) & (result.scaling["group"] != "all")
            ]
            if not levels.empty:
                lines.append(
                    "      by level: "
                    + " · ".join(
                        f"{g.split('=', 1)[1]} {e:.2f} [{lo:.2f}, {hi:.2f}]"
                        for g, e, lo, hi in zip(levels["group"], levels["exponent"], levels["ci_lo"], levels["ci_hi"])
                    )
                )
    return "\n".join(lines)


def _grouped_lines(rows: pd.DataFrame, n_props: int, partial: bool) -> list[str]:
    """One line per parameter, listing the properties it affects.

    A parameter with a real effect on stiffness or strength normally shows it
    across every related property; one that turns up for a single property
    out of many is flagged as isolated.
    """
    out = []
    for parameter, block in rows.groupby("parameter", sort=False):
        kind = block["kind"].iloc[0]
        values = block["partial_eta_sq" if kind == "categorical" else "partial_rho"] if partial \
            else block["effect"]
        symbol = ("η²" if kind == "categorical" else "ρ") if partial else ("ε²" if kind == "categorical" else "ρ")
        lo, hi = float(values.min()), float(values.max())
        fmt = "{:.2f}" if kind == "categorical" else "{:+.2f}"
        span = fmt.format(lo) if math.isclose(lo, hi) else f"{fmt.format(lo)} to {fmt.format(hi)}"
        labels = [property_label(p) for p in block["property"]]
        shown = ", ".join(labels[:4]) + (f", +{len(labels) - 4} more" if len(labels) > 4 else "")
        flag = "  *isolated" if len(block) == 1 and n_props >= 4 else ""
        out.append(
            f"    {parameter:<20} {'partial ' if partial else 'pairwise '}{symbol} {span} on "
            f"{len(block)}/{n_props} properties ({shown}){flag}"
        )
    return out


def _effect_text(row: pd.Series) -> str:
    if row["kind"] == "categorical":
        return f"ε² = {row['effect']:.2f}"
    return f"ρ = {row['effect']:+.2f}"


def _partial_text(row: pd.Series) -> str:
    if row["kind"] == "categorical":
        return f"η² = {row['partial_eta_sq']:.2f}"
    return f"ρ = {row['partial_rho']:+.2f}"


def _summary_line(row: pd.Series) -> str:
    label = property_label(row["property"])
    head = f"{row['parameter']:<22} → {label:<14} {_effect_text(row)}  q {_fmt_q(row['q_value']):<8} {row['strength']}"
    if bool(row["partial_reliable"]) and np.isfinite(row["partial_p"]):
        head += f"  partial {_partial_text(row).split(' = ')[1]}"
    if row["kind"] == "categorical":
        head += f"  ({row['direction']})"
    elif row["kind"] == "datetime":
        head += "  (later → higher)" if row["effect"] > 0 else "  (later → lower)"
    return head


def _collinear_pairs(result: CorrelationResult) -> list[tuple[str, str, float, str]]:
    """(a, b, association, measure) for every pair too entangled to separate."""
    kinds = {p.name: p for p in result.included}
    pairs = []
    names = list(result.collinearity.index)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            value = result.collinearity.loc[a, b]
            if not (np.isfinite(value) and abs(value) >= result.collinear_rho):
                continue
            ka, kb = kinds[a].is_ordered, kinds[b].is_ordered
            measure = "ρ" if ka and kb else "η" if ka or kb else "V"
            pairs.append((a, b, float(value), measure))
    return pairs


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
def _text_color(rgba: Any) -> str:
    """Ink or white, whichever reads on a cell of this colour."""
    r, g, b = to_rgb(rgba)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#ffffff" if lum < 0.45 else INK


def _heatmap(
    ax: plt.Axes,
    values: np.ndarray,
    texts: list[list[str]],
    bold: np.ndarray,
    cmap: Any,
    norm: Normalize,
    row_labels: list[str],
    col_labels: list[str],
    show_rows: bool = True,
) -> Any:
    image = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")
    nrows, ncols = values.shape
    ax.set_xticks(range(ncols))
    ax.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=8.5, color=INK_SECONDARY)
    ax.set_yticks(range(nrows))
    ax.set_yticklabels(row_labels if show_rows else [], fontsize=8.5, color=INK_SECONDARY)
    ax.set_xticks(np.arange(-0.5, ncols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="both", length=0)
    ax.set_facecolor("#f4f3f0")
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(nrows):
        for j in range(ncols):
            value = values[i, j]
            if not np.isfinite(value):
                ax.text(j, i, "–", ha="center", va="center", fontsize=8, color=INK_MUTED)
                continue
            ax.text(
                j, i, texts[i][j], ha="center", va="center", fontsize=7.5,
                color=_text_color(cmap(norm(value))),
                fontweight="bold" if bold[i, j] else "normal",
            )
    return image


def plot_heatmap(result: CorrelationResult, out_path: str | Path, cfg: AnalysisConfig = DEFAULT_CONFIG) -> Path | None:
    """Numeric parameters × properties: pairwise ρ beside partial ρ."""
    numeric_infos = [p for p in result.included if p.is_ordered]
    if not numeric_infos or not result.properties:
        return None
    rows = [p.column for p in numeric_infos]
    cols = result.properties
    shape = (len(rows), len(cols))
    rho, sig = np.full(shape, np.nan), np.zeros(shape, bool)
    prho, psig = np.full(shape, np.nan), np.zeros(shape, bool)
    for i, column in enumerate(rows):
        for j, prop in enumerate(cols):
            cell = result.cell(column, prop)
            if cell is None:
                continue
            rho[i, j], sig[i, j] = cell["effect"], bool(cell["significant"])
            if bool(cell["partial_reliable"]):
                prho[i, j], psig[i, j] = cell["partial_rho"], bool(cell["partial_significant"])

    def texts(values, flags):
        return [[f"{v:+.2f}" + ("*" if f else "") if np.isfinite(v) else "" for v, f in zip(vr, fr)] for vr, fr in zip(values, flags)]

    show_partial = bool(np.isfinite(prho).any())
    ncols = 2 if show_partial else 1
    width = ncols * (0.62 * len(cols) + 0.4) + 2.6
    height = 0.42 * len(rows) + 2.6
    fig, axes = plt.subplots(1, ncols, figsize=(width, height), squeeze=False, layout="constrained")
    fig.patch.set_facecolor(SURFACE)
    cmap, norm = diverging_cmap(), Normalize(-1, 1)
    row_labels = [f"{p.name}" for p in numeric_infos]
    col_labels = [property_label(c) for c in cols]

    image = _heatmap(axes[0, 0], rho, texts(rho, sig), sig, cmap, norm, row_labels, col_labels)
    axes[0, 0].set_title("Pairwise ρ", loc="left", fontsize=10, color=INK)
    if show_partial:
        _heatmap(axes[0, 1], prho, texts(prho, psig), psig, cmap, norm, row_labels, col_labels, show_rows=False)
        axes[0, 1].set_title("Partial ρ (other parameters held fixed)", loc="left", fontsize=10, color=INK)

    bar = fig.colorbar(image, ax=axes[0, :].tolist(), fraction=0.04, pad=0.02)
    bar.set_label("Spearman ρ", color=INK_SECONDARY, fontsize=9)
    bar.ax.tick_params(colors=INK_SECONDARY, labelsize=8, length=0)
    bar.outline.set_visible(False)

    fig.suptitle(
        "Which parameters track which properties?", x=0.01, ha="left",
        fontsize=12, color=INK,
    )
    fig.text(
        0.01, -0.01,
        f"{result.n_tests} tests · bold* = significant after FDR correction (q < {result.alpha:g}) · "
        f"– = not estimable (too few tests, or parameter collinear with others)",
        fontsize=8, color=INK_MUTED, ha="left", va="top",
    )
    return save_figure(fig, out_path, cfg)


def plot_categorical(result: CorrelationResult, out_path: str | Path, cfg: AnalysisConfig = DEFAULT_CONFIG) -> Path | None:
    """Categorical factors × properties: effect size, pairwise beside partial.

    Manufacturing factors (batch, bed position) tend to be small next to
    pattern-to-pattern differences, so the partial panel is often where they
    show up.
    """
    infos = [p for p in result.included if p.kind == "categorical"]
    if not infos or not result.properties:
        return None
    cols = result.properties
    shape = (len(infos), len(cols))
    eps, eps_sig = np.full(shape, np.nan), np.zeros(shape, bool)
    eta, eta_sig = np.full(shape, np.nan), np.zeros(shape, bool)
    for i, info in enumerate(infos):
        for j, prop in enumerate(cols):
            cell = result.cell(info.column, prop)
            if cell is None:
                continue
            if np.isfinite(cell["effect"]):
                eps[i, j], eps_sig[i, j] = cell["effect"], bool(cell["significant"])
            if bool(cell["partial_reliable"]) and np.isfinite(cell["partial_eta_sq"]):
                eta[i, j], eta_sig[i, j] = cell["partial_eta_sq"], bool(cell["partial_significant"])

    def texts(values, flags):
        return [[f"{v:.2f}" + ("*" if f else "") if np.isfinite(v) else "" for v, f in zip(vr, fr)]
                for vr, fr in zip(values, flags)]

    show_partial = bool(np.isfinite(eta).any())
    ncols = 2 if show_partial else 1
    fig, axes = plt.subplots(
        1, ncols, figsize=(ncols * (0.62 * len(cols) + 0.4) + 3.0, 0.42 * len(infos) + 2.6),
        squeeze=False, layout="constrained",
    )
    fig.patch.set_facecolor(SURFACE)
    norm = Normalize(0, 0.6)
    labels = [f"{p.name} ({len(p.levels)} levels)" for p in infos]
    col_labels = [property_label(c) for c in cols]
    image = _heatmap(axes[0, 0], eps, texts(eps, eps_sig), eps_sig, sequential_cmap(), norm, labels, col_labels)
    axes[0, 0].set_title("Pairwise ε² (Kruskal–Wallis)", loc="left", fontsize=10, color=INK)
    if show_partial:
        _heatmap(axes[0, 1], eta, texts(eta, eta_sig), eta_sig, sequential_cmap(), norm,
                 labels, col_labels, show_rows=False)
        axes[0, 1].set_title("Partial η² (other parameters held fixed)", loc="left", fontsize=10, color=INK)
    bar = fig.colorbar(image, ax=axes[0, :].tolist(), fraction=0.04, pad=0.02, extend="max")
    bar.set_label("share of rank variance explained", color=INK_SECONDARY, fontsize=9)
    bar.ax.tick_params(colors=INK_SECONDARY, labelsize=8, length=0)
    bar.outline.set_visible(False)
    fig.suptitle("Categorical factors: how much does each one matter?", x=0.01, ha="left", fontsize=12, color=INK)
    fig.text(
        0.01, -0.01,
        f"bold* = significant (FDR q < {result.alpha:g}) · 0.01 small, 0.08 medium, 0.26 large · "
        f"partial: Quade rank analysis of covariance",
        fontsize=8, color=INK_MUTED, ha="left", va="top",
    )
    return save_figure(fig, out_path, cfg)


def plot_collinearity(result: CorrelationResult, out_path: str | Path, cfg: AnalysisConfig = DEFAULT_CONFIG) -> Path | None:
    """How entangled every pair of parameters is, numeric and categorical alike."""
    infos = result.included
    if len(infos) < 2:
        return None
    names = [p.name for p in infos]
    matrix = np.abs(result.collinearity.loc[names, names].to_numpy(dtype=float))
    lower = np.tril(np.ones_like(matrix, bool), k=-1)
    shown = np.where(lower, matrix, np.nan)
    flags = np.nan_to_num(shown) >= result.collinear_rho
    texts = [[f"{v:.2f}" if np.isfinite(v) else "" for v in row] for row in shown]

    size = 0.55 * len(infos) + 2.6
    fig, ax = plt.subplots(figsize=(size + 1.6, size), layout="constrained")
    fig.patch.set_facecolor(SURFACE)
    labels = []
    for p in infos:
        if p.is_ordered and np.isfinite(p.vif):
            labels.append(f"{p.name}  (VIF {p.vif:.1f})")
        elif p.is_ordered and p.vif == math.inf:
            labels.append(f"{p.name}  (VIF ∞)")
        else:
            labels.append(p.name)
    image = _heatmap(ax, shown, texts, flags, sequential_cmap(), Normalize(0, 1), labels, names)
    bar = fig.colorbar(image, ax=ax, fraction=0.05, pad=0.02)
    bar.set_label("association between parameters", color=INK_SECONDARY, fontsize=9)
    bar.ax.tick_params(colors=INK_SECONDARY, labelsize=8, length=0)
    bar.outline.set_visible(False)
    ax.set_title("Which parameters move together?", loc="left", fontsize=11, color=INK)
    fig.text(
        0.01, -0.01,
        f"numeric–numeric |Spearman ρ| · numeric–categorical correlation ratio η · "
        f"categorical–categorical Cramér's V\n"
        f"bold = ≥ {result.collinear_rho:g}: confounded, effects cannot be told apart · "
        f"VIF ≥ {result.vif_threshold:g}: predictable from the rest combined",
        fontsize=8, color=INK_MUTED, ha="left", va="top",
    )
    return save_figure(fig, out_path, cfg)


def plot_property(
    result: CorrelationResult,
    prop: str,
    out_path: str | Path,
    cfg: AnalysisConfig = DEFAULT_CONFIG,
) -> Path | None:
    """One property against every parameter, one panel per parameter."""
    infos = result.included
    if not infos or prop not in result.properties:
        return None
    ncols = min(4, len(infos))
    nrows = math.ceil(len(infos) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.1 * ncols, 2.7 * nrows + 0.9),
        squeeze=False, layout="constrained",
    )
    fig.patch.set_facecolor(SURFACE)
    values = result.props.values[prop]
    raw = result.props.raw[prop]
    censored = result.props.censored[prop]
    rng = np.random.default_rng(0)

    for ax, info in zip(axes.flat, infos):
        cell = result.cell(info.column, prop)
        _style_panel(ax, info.name)
        if info.kind == "datetime":
            ax.set_xlabel(f"days since {info.origin:%Y-%m-%d}", color=INK_MUTED, fontsize=7.5)
        if info.is_ordered:
            x = result.numeric[info.column]
            ok = x.notna() & values.notna()
            ax.scatter(x[ok], values[ok], s=24, color=POINT_COLOR, alpha=0.8,
                       edgecolors=SURFACE, linewidths=0.6, zorder=3)
            hollow = x.notna() & censored & raw.notna()
            if hollow.any():
                ax.scatter(x[hollow], raw[hollow], s=24, facecolors="none",
                           edgecolors=INK_MUTED, linewidths=0.9, zorder=2)
            if cell is not None and bool(cell["significant"]) and ok.sum() >= 3:
                slope, intercept = np.polyfit(x[ok].astype(float), values[ok].astype(float), 1)
                xs = np.linspace(float(x[ok].min()), float(x[ok].max()), 50)
                ax.plot(xs, slope * xs + intercept, color=INK, linewidth=1.3, zorder=4)
            _annotate_panel(ax, cell, "ρ")
        else:
            groups = result.categorical[info.column]
            for i, level in enumerate(info.levels):
                ys = values[(groups == level) & values.notna()].to_numpy(dtype=float)
                if ys.size == 0:
                    continue
                xs = i + rng.uniform(-0.18, 0.18, ys.size)
                ax.scatter(xs, ys, s=22, color=POINT_COLOR, alpha=0.8,
                           edgecolors=SURFACE, linewidths=0.6, zorder=3)
                ax.hlines(np.median(ys), i - 0.3, i + 0.3, color=INK, linewidth=2, zorder=4)
            ax.set_xticks(range(len(info.levels)))
            ax.set_xticklabels(info.levels, rotation=30 if max(map(len, info.levels)) > 5 else 0,
                               ha="right" if max(map(len, info.levels)) > 5 else "center", fontsize=8)
            ax.set_xlim(-0.6, len(info.levels) - 0.4)
            _annotate_panel(ax, cell, "ε²")
    for ax in list(axes.flat)[len(infos):]:
        ax.set_visible(False)
    for row in axes:
        row[0].set_ylabel(property_label(prop, with_unit=True), color=INK_SECONDARY, fontsize=9)

    fig.suptitle(f"{property_label(prop, with_unit=True)} vs each parameter", x=0.01, ha="left", fontsize=12, color=INK)
    note = (
        f"{int(values.notna().sum())} tests · * and line = significant after FDR "
        f"correction (q < {result.alpha:g}) · partial = other parameters held fixed · "
        f"bar = level median"
    )
    if censored.any():
        note += f" · hollow = lower bound (test stopped at the load limit), excluded"
    fig.text(0.01, -0.01, note, fontsize=8, color=INK_MUTED, ha="left", va="top")
    return save_figure(fig, out_path, cfg)


def _style_panel(ax: plt.Axes, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8, length=0)


def _annotate_panel(ax: plt.Axes, cell: pd.Series | None, symbol: str) -> None:
    """Pairwise line, then partial line, each styled by its own significance.

    Drawn as a legend so matplotlib's "best" placement keeps it off the data.
    A masked effect -- pairwise n.s., partial significant -- is often the most
    interesting thing in the figure, so its line does not inherit the muted
    style of the pairwise result above it.
    """
    lines: list[tuple[str, bool]] = []
    if cell is None or not np.isfinite(cell["effect"]):
        reason = "" if cell is None else str(cell["note"])
        lines.append(("not tested" + (f": {reason[:38]}" if reason else ""), False))
    else:
        value = f"{cell['effect']:+.2f}" if symbol == "ρ" else f"{cell['effect']:.2f}"
        significant = bool(cell["significant"])
        lines.append(
            (f"{symbol} = {value}{'*' if significant else ''}   q {_fmt_q(cell['q_value'])}", significant)
        )
        partial = cell["partial_rho"] if symbol == "ρ" else cell["partial_eta_sq"]
        if bool(cell["partial_reliable"]) and np.isfinite(partial):
            partial_sig = bool(cell["partial_significant"])
            shown = f"{partial:+.2f}" if symbol == "ρ" else f"η² {partial:.2f}"
            lines.append(
                (
                    f"partial {shown}{'*' if partial_sig else ''}"
                    + ("  ← masked effect" if partial_sig and not significant else ""),
                    partial_sig,
                )
            )

    from matplotlib.lines import Line2D

    blank = [Line2D([], [], linestyle="", marker="") for _ in lines[1:]]
    legend = ax.legend(
        blank, [text for text, _ in lines[1:]], title=lines[0][0], loc="best",
        handlelength=0, handletextpad=0, borderpad=0.35, labelspacing=0.2,
        fontsize=7.5, title_fontsize=7.5, alignment="left", frameon=True,
        framealpha=0.88,
    )
    legend.get_frame().set_facecolor(SURFACE)
    legend.get_frame().set_edgecolor("none")
    texts = [legend.get_title()] + list(legend.get_texts())
    for text, (_, emphasised) in zip(texts, lines):
        text.set_color(INK if emphasised else INK_MUTED)
        text.set_fontweight("bold" if emphasised else "normal")
    legend.set_zorder(5)


def _plain_log_axis(axis: Any) -> None:
    """Label a log axis with plain numbers (0.2, 0.3) instead of 2 x 10^-1."""
    from matplotlib.ticker import FuncFormatter, LogLocator

    def major(value: float, _pos: int) -> str:
        return f"{value:g}"

    def minor(value: float, _pos: int) -> str:
        if value <= 0:
            return ""
        lead = round(value / 10 ** math.floor(math.log10(value)), 6)
        return f"{value:g}" if lead in (2.0, 3.0, 5.0) else ""

    axis.set_major_locator(LogLocator(base=10))
    axis.set_minor_locator(LogLocator(base=10, subs=(2, 3, 4, 5, 6, 7, 8, 9)))
    axis.set_major_formatter(FuncFormatter(major))
    axis.set_minor_formatter(FuncFormatter(minor))


def plot_scaling(result: CorrelationResult, out_path: str | Path, cfg: AnalysisConfig = DEFAULT_CONFIG) -> Path | None:
    """Log-log property vs relative density, with Gibson-Ashby reference slopes.

    Points are coloured by the first categorical factor (topology, say), and
    each level gets its own line: shared exponent, own prefactor -- the model
    the exponent in the corner was fitted with.
    """
    from .envelope import CATEGORICAL, assign_colors

    if result.scaling.empty:
        return None
    overall = result.scaling[result.scaling["group"] == "all"].to_dict("records")
    ncols = min(3, len(overall))
    nrows = math.ceil(len(overall) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.7 * ncols, 3.4 * nrows + 0.9), squeeze=False, layout="constrained")
    fig.patch.set_facecolor(SURFACE)
    rho = result.numeric["relative_density"]

    per_level = result.scaling[result.scaling["group"] != "all"]["group"]
    factor_name = per_level.iloc[0].split("=", 1)[0] if not per_level.empty else None
    factor = next(
        (p for p in result.included
         if p.kind == "categorical" and p.name == factor_name and len(p.levels) <= len(CATEGORICAL)),
        None,
    )
    groups = result.categorical[factor.column] if factor is not None else None
    palette = assign_colors(factor.levels) if factor is not None else {}

    for ax, fit in zip(axes.flat, overall):
        prop = fit["property"]
        y = result.props.values[prop]
        ok = rho.notna() & y.notna() & (rho > 0) & (y > 0)
        _style_panel(ax, property_label(prop, with_unit=True))
        ax.set_xscale("log")
        ax.set_yscale("log")
        xs_all, ys_all = rho[ok].to_numpy(float), y[ok].to_numpy(float)
        line_x = np.geomspace(xs_all.min(), xs_all.max(), 50)

        levels = factor.levels if factor is not None else [None]
        for level in levels:
            mask = ok & ((groups == level) if level is not None else True)
            if not mask.any():
                continue
            color = palette.get(level, POINT_COLOR)
            xs, ys = rho[mask].to_numpy(float), y[mask].to_numpy(float)
            ax.scatter(xs, ys, s=24, color=color, alpha=0.85, edgecolors=SURFACE, linewidths=0.6, zorder=3)
            # Shared exponent, this level's own prefactor (geometric-mean fit).
            prefactor = float(np.exp(np.mean(np.log(ys) - fit["exponent"] * np.log(xs))))
            span = np.geomspace(xs.min(), xs.max(), 30)
            ax.plot(span, prefactor * span ** fit["exponent"], color=color if level is not None else INK,
                    linewidth=1.6, zorder=4)

        cx, cy = float(np.exp(np.log(xs_all).mean())), float(np.exp(np.log(ys_all).mean()))
        for name, slope in sorted(SCALING_REFERENCES[fit["kind"]].items(), key=lambda kv: kv[1]):
            ref = cy * (line_x / cx) ** slope
            ax.plot(line_x, ref, color=INK_MUTED, linewidth=0.9, linestyle="--", zorder=2)
            ax.text(line_x[-1], ref[-1], f" n={slope:g}", fontsize=7, color=INK_MUTED, va="center", ha="left")

        text = (
            f"n = {fit['exponent']:.2f} [{fit['ci_lo']:.2f}, {fit['ci_hi']:.2f}]\n"
            f"R² = {fit['r2']:.2f} · {fit['n']} tests\n{fit['interpretation']}"
        )
        per_level = result.scaling[(result.scaling["property"] == prop) & (result.scaling["group"] != "all")]
        for _, row in per_level.iterrows():
            text += f"\n  {row['group'].split('=', 1)[1]}: {row['exponent']:.2f} [{row['ci_lo']:.2f}, {row['ci_hi']:.2f}]"
        ax.text(
            0.03, 0.97, text, transform=ax.transAxes, ha="left", va="top", fontsize=7.2, color=INK,
            bbox=dict(boxstyle="round,pad=0.25", facecolor=SURFACE, edgecolor="none", alpha=0.85), zorder=5,
        )
        _plain_log_axis(ax.xaxis)
        _plain_log_axis(ax.yaxis)
        ax.tick_params(which="minor", labelsize=7, colors=INK_SECONDARY, length=0)
        ax.set_xlabel("Relative density ρ*", color=INK_SECONDARY, fontsize=9)

    for ax in list(axes.flat)[len(overall):]:
        ax.set_visible(False)
    if factor is not None:
        handles = [
            plt.Line2D([], [], color=palette[lvl], marker="o", linestyle="-", markersize=5, label=lvl)
            for lvl in factor.levels
        ]
        legend = fig.legend(handles=handles, title=factor.name, loc="outside upper right", frameon=False, fontsize=8)
        legend.get_title().set_fontsize(8)
        for t in legend.get_texts() + [legend.get_title()]:
            t.set_color(INK_SECONDARY)

    held = next((r["held_fixed"] for r in overall if r["held_fixed"]), "")
    fig.suptitle("Scaling with relative density: property ∝ (ρ*)ⁿ", x=0.01, ha="left", fontsize=12, color=INK)
    fig.text(
        0.01, -0.01,
        "Dashed: Gibson–Ashby reference slopes (modulus n=2 bending / 1 stretching · strength 1.5 / 1) · "
        "bracket = 95% CI"
        + (f" · one exponent shared across {held}, each level with its own prefactor" if held else ""),
        fontsize=8, color=INK_MUTED, ha="left", va="top",
    )
    return save_figure(fig, out_path, cfg)


# ----------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------
def write_outputs(
    result: CorrelationResult,
    table_dir: str | Path,
    figure_dir: str | Path,
    cfg: AnalysisConfig = DEFAULT_CONFIG,
    figures: bool = True,
) -> list[Path]:
    """Write the tables, the text summary and (optionally) every figure."""
    table_dir, figure_dir = Path(table_dir), Path(figure_dir)
    table_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    result.table.to_csv(table_dir / "correlations.csv", index=False)
    result.parameters_frame().to_csv(table_dir / "parameters.csv", index=False)
    written += [table_dir / "correlations.csv", table_dir / "parameters.csv"]
    if not result.collinearity.empty:
        result.collinearity.to_csv(table_dir / "collinearity.csv")
        written.append(table_dir / "collinearity.csv")
    if not result.scaling.empty:
        result.scaling.to_csv(table_dir / "scaling_laws.csv", index=False)
        written.append(table_dir / "scaling_laws.csv")
    (table_dir / "summary.txt").write_text(summarise(result) + "\n", encoding="utf-8")
    written.append(table_dir / "summary.txt")

    if not figures:
        return written

    # Stale per-property figures from an earlier run with other properties
    # would otherwise linger and mislead.
    if figure_dir.exists():
        for old in figure_dir.glob("property_*.png"):
            old.unlink()

    for maker, name in (
        (plot_heatmap, "heatmap.png"),
        (plot_categorical, "categorical_effects.png"),
        (plot_collinearity, "collinearity.png"),
        (plot_scaling, "scaling_laws.png"),
    ):
        out = maker(result, figure_dir / name, cfg)
        if out is not None:
            written.append(out)
    for prop in result.properties:
        out = plot_property(result, prop, figure_dir / f"property_{prop}.png", cfg)
        if out is not None:
            written.append(out)
    return written


def has_parameters(metrics: pd.DataFrame, cfg: AnalysisConfig = DEFAULT_CONFIG) -> bool:
    """True when at least one parameter varies enough to be analysed."""
    return any(p.included for p in discover_parameters(metrics, cfg))


#: Files `write_outputs` produces at the top of its two directories.
_OUTPUT_FILES = (
    "correlations.csv", "parameters.csv", "collinearity.csv", "scaling_laws.csv",
    "summary.txt", "heatmap.png", "categorical_effects.png", "collinearity.png",
    "scaling_laws.png",
)


def clear_outputs(table_dir: str | Path, figure_dir: str | Path) -> None:
    """Remove a previous run's top-level outputs (subset runs are kept)."""
    for directory in (Path(table_dir), Path(figure_dir)):
        if not directory.exists():
            continue
        for name in _OUTPUT_FILES:
            (directory / name).unlink(missing_ok=True)
        for old in directory.glob("property_*.png"):
            old.unlink()
