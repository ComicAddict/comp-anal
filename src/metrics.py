"""Extraction of cellular-solid metrics from a stress-strain curve.

Metrics implemented (see README for the conventions behind each):
  * Young's modulus      -- best-fit linear region found by sliding-window search
  * Plateau stress       -- mean stress over a strain window (ISO 13314 style)
  * Densification onset  -- energy-efficiency maximum (primary), tangent or ISO
  * Energy absorption W  -- integral of stress to densification
  * SEA                  -- W normalised by mass, else by volume
  * First peak / crush   -- initial load peak before post-peak softening
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .config import DEFAULT_CONFIG, AnalysisConfig
from .io import Diagnostic
from .mechanics import (
    StressStrainCurve,
    cumulative_energy,
    energy_to_strain,
    tangent_modulus,
)
from .specimens import Specimen

NAN = float("nan")


@dataclass
class ModulusFit:
    """Result of the linear-region search."""

    modulus_MPa: float = NAN
    intercept_MPa: float = NAN
    r2: float = NAN
    strain_lo: float = NAN
    strain_hi: float = NAN
    n_points: int = 0
    #: Strain at which the fitted line crosses zero stress. This is the
    #: seating/toe slack; large values mean the contact point was late.
    toe_strain: float = NAN


@dataclass
class DensificationResult:
    strain: float = NAN
    method: str = ""
    #: True when the criterion was only met at the very last sample, i.e. the
    #: test probably stopped before real densification.
    at_end_of_data: bool = False
    efficiency_max: float = NAN


@dataclass
class PeakResult:
    #: First distinct local maximum followed by real softening (may be absent).
    first_peak_stress_MPa: float = NAN
    first_peak_strain: float = NAN
    first_peak_force_N: float = NAN
    has_distinct_first_peak: bool = False
    #: Load the specimen crushes at: the first peak if there is one, otherwise
    #: the maximum before densification.
    crush_force_N: float = NAN
    crush_stress_MPa: float = NAN
    post_peak_drop_frac: float = NAN


@dataclass
class TestMetrics:
    """All metrics for one test, flattened to one CSV row by `as_row`."""

    test_id: str
    modulus: ModulusFit = field(default_factory=ModulusFit)
    plateau_stress_MPa: float = NAN
    plateau_stress_std_MPa: float = NAN
    plateau_strain_lo: float = NAN
    plateau_strain_hi: float = NAN
    densification: DensificationResult = field(default_factory=DensificationResult)
    energy_absorption_MJ_m3: float = NAN
    energy_absorption_J: float = NAN
    sea_J_per_g: float = NAN
    sea_basis: str = ""
    peaks: PeakResult = field(default_factory=PeakResult)
    max_stress_MPa: float = NAN
    max_force_N: float = NAN
    max_strain: float = NAN
    toe_compensation_applied: bool = False
    diagnostics: list[Diagnostic] = field(default_factory=list)

    def as_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {"test_id": self.test_id}
        row.update(
            {
                "modulus_MPa": self.modulus.modulus_MPa,
                "modulus_r2": self.modulus.r2,
                "modulus_fit_strain_lo": self.modulus.strain_lo,
                "modulus_fit_strain_hi": self.modulus.strain_hi,
                "modulus_fit_n_points": self.modulus.n_points,
                "toe_strain": self.modulus.toe_strain,
                "toe_compensation_applied": self.toe_compensation_applied,
                "plateau_stress_MPa": self.plateau_stress_MPa,
                "plateau_stress_std_MPa": self.plateau_stress_std_MPa,
                "plateau_strain_lo": self.plateau_strain_lo,
                "plateau_strain_hi": self.plateau_strain_hi,
                "densification_strain": self.densification.strain,
                "densification_method": self.densification.method,
                "densification_at_end_of_data": self.densification.at_end_of_data,
                "max_energy_efficiency": self.densification.efficiency_max,
                "energy_absorption_MJ_m3": self.energy_absorption_MJ_m3,
                "energy_absorption_J": self.energy_absorption_J,
                "sea_J_per_g": self.sea_J_per_g,
                "sea_basis": self.sea_basis,
            }
        )
        row.update({k: v for k, v in asdict(self.peaks).items()})
        row.update(
            {
                "max_stress_MPa": self.max_stress_MPa,
                "max_force_N": self.max_force_N,
                "max_strain": self.max_strain,
                "n_warnings": sum(1 for d in self.diagnostics if d.level == "warning"),
                "diagnostics": "; ".join(
                    f"{d.code}: {d.message}" for d in self.diagnostics
                ),
            }
        )
        return row


# ----------------------------------------------------------------------
# Young's modulus
# ----------------------------------------------------------------------
def youngs_modulus(
    strain: np.ndarray, stress: np.ndarray, cfg: AnalysisConfig = DEFAULT_CONFIG
) -> tuple[ModulusFit, list[Diagnostic]]:
    """Find the initial linear region by sliding-window search.

    Every window (start, end) inside the search region that is at least
    `modulus_min_window_strain` wide and `modulus_min_window_points` long is
    scored by R^2. The minimum-width constraint matters: without it the search
    degenerates to a two-point window, where R^2 is trivially 1.0 and the slope
    is noise.

    Which of the high-scoring windows wins is `modulus_selection`. R^2 alone is
    not enough to identify the *elastic* region: a foam plateau is often just as
    straight as the initial slope, so on a clean curve both score R^2 ~ 1.0 and
    the tie is settled by floating-point noise -- which can hand back the
    plateau slope as the modulus. The default therefore takes the steepest
    window among those statistically tied with the best, which is the initial
    linear region by construction. Set `modulus_selection="max_r2"` for the
    unmodified highest-R^2 window.

    Regression statistics come from prefix sums, so scoring a window is O(1)
    and the whole search is fast even on long files. The data is mean-centred
    first: the sums of squares are otherwise dominated by the offset and lose
    most of their significant digits to cancellation.
    """
    diags: list[Diagnostic] = []
    fit = ModulusFit()

    region = strain <= cfg.modulus_search_max_strain
    x = strain[region]
    y = stress[region]
    n_total = len(x)
    min_pts = max(3, int(cfg.modulus_min_window_points))

    if n_total < min_pts:
        diags.append(
            Diagnostic(
                "warning",
                "modulus_not_fitted",
                f"only {n_total} sample(s) below {cfg.modulus_search_max_strain:.0%} "
                f"strain, need {min_pts}",
            )
        )
        return fit, diags

    if np.ptp(y) <= 0 or not np.isfinite(y).any():
        diags.append(
            Diagnostic(
                "warning",
                "modulus_no_variation",
                f"stress does not vary below {cfg.modulus_search_max_strain:.0%} "
                f"strain (constant at {float(y[0]):.4g} MPa); no elastic region "
                f"to fit -- check that the file contains a real loading ramp",
            )
        )
        return fit, diags

    # Mean-centre before accumulating, for numerical conditioning.
    x_mean = float(x.mean())
    y_mean = float(y.mean())
    xc = x - x_mean
    yc = y - y_mean

    cx = np.concatenate(([0.0], np.cumsum(xc)))
    cy = np.concatenate(([0.0], np.cumsum(yc)))
    cxx = np.concatenate(([0.0], np.cumsum(xc * xc)))
    cyy = np.concatenate(([0.0], np.cumsum(yc * yc)))
    cxy = np.concatenate(([0.0], np.cumsum(xc * yc)))

    best_score = -np.inf
    best: tuple[int, int] | None = None
    candidates: list[tuple[float, float, int, int]] = []  # (r2, slope, i, j)
    stride = max(1, int(cfg.modulus_window_stride))

    for i in range(0, n_total - min_pts + 1, stride):
        # Window is x[i:j]; j ranges over all ends satisfying both minimums.
        j_min = i + min_pts
        j_lo = max(
            j_min,
            int(np.searchsorted(x, x[i] + cfg.modulus_min_window_strain, side="left")) + 1,
        )
        if j_lo > n_total:
            continue
        j = np.arange(j_lo, n_total + 1)
        n = (j - i).astype(float)
        sx = cx[j] - cx[i]
        sy = cy[j] - cy[i]
        sxx = cxx[j] - cxx[i]
        syy = cyy[j] - cyy[i]
        sxy = cxy[j] - cxy[i]

        den_x = n * sxx - sx * sx
        den_y = n * syy - sy * sy
        num = n * sxy - sx * sy
        with np.errstate(divide="ignore", invalid="ignore"):
            slope = np.where(den_x > 0, num / den_x, np.nan)
            r2 = np.where((den_x > 0) & (den_y > 0), (num * num) / (den_x * den_y), np.nan)

        valid = np.isfinite(r2)
        if not valid.any():
            continue
        k = int(np.nanargmax(np.where(valid, r2, -np.inf)))
        candidates.append((float(r2[k]), float(slope[k]), i, int(j[k])))
        if r2[k] > best_score:
            best_score = float(r2[k])
            best = (i, int(j[k]))

    if best is None:
        diags.append(
            Diagnostic(
                "warning",
                "modulus_not_fitted",
                "no window satisfied the minimum width and point count "
                f"({cfg.modulus_min_window_strain} strain, {min_pts} points)",
            )
        )
        return fit, diags

    if cfg.modulus_selection == "steepest_high_r2":
        # Among windows statistically as good as the best, take the steepest --
        # a long straight toe region can out-score the true elastic region.
        near = [c for c in candidates if c[0] >= best_score - cfg.modulus_r2_tolerance]
        if near:
            chosen = max(near, key=lambda c: c[1])
            best = (chosen[2], chosen[3])
    elif cfg.modulus_selection != "max_r2":
        raise ValueError(
            f"Unknown modulus_selection {cfg.modulus_selection!r}; "
            f"expected 'max_r2' or 'steepest_high_r2'"
        )

    i, j = best
    xw, yw = x[i:j], y[i:j]
    slope, intercept = np.polyfit(xw, yw, 1)
    resid = yw - (slope * xw + intercept)
    ss_tot = float(np.sum((yw - yw.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid**2)) / ss_tot if ss_tot > 0 else NAN

    fit = ModulusFit(
        modulus_MPa=float(slope),
        intercept_MPa=float(intercept),
        r2=float(r2),
        strain_lo=float(xw[0]),
        strain_hi=float(xw[-1]),
        n_points=int(j - i),
        toe_strain=float(-intercept / slope) if slope != 0 else NAN,
    )

    if fit.modulus_MPa <= 0:
        diags.append(
            Diagnostic(
                "warning",
                "modulus_non_positive",
                f"fitted modulus is {fit.modulus_MPa:.4g} MPa (<= 0); the "
                f"search region may contain no loading",
            )
        )
    if np.isfinite(fit.r2) and fit.r2 < 0.99:
        diags.append(
            Diagnostic(
                "warning",
                "modulus_poor_fit",
                f"best linear window has R^2={fit.r2:.4f} over strain "
                f"{fit.strain_lo:.4f}-{fit.strain_hi:.4f}; the elastic region "
                f"may be short or noisy",
            )
        )
    return fit, diags


# ----------------------------------------------------------------------
# Plateau
# ----------------------------------------------------------------------
def plateau_stress(
    strain: np.ndarray, stress: np.ndarray, cfg: AnalysisConfig = DEFAULT_CONFIG
) -> tuple[float, float, list[Diagnostic]]:
    """Mean (and std) stress between `plateau_strain_lo` and `..._hi`."""
    diags: list[Diagnostic] = []
    window = (strain >= cfg.plateau_strain_lo) & (strain <= cfg.plateau_strain_hi)
    count = int(window.sum())
    if count < cfg.plateau_min_points:
        diags.append(
            Diagnostic(
                "warning",
                "plateau_window_empty",
                f"only {count} sample(s) between {cfg.plateau_strain_lo:.0%} and "
                f"{cfg.plateau_strain_hi:.0%} strain (max strain reached: "
                f"{strain[-1]:.3f}); plateau stress not reported",
            )
        )
        return NAN, NAN, diags
    values = stress[window]
    return float(np.mean(values)), float(np.std(values, ddof=1) if count > 1 else 0.0), diags


# ----------------------------------------------------------------------
# Densification onset
# ----------------------------------------------------------------------
def densification_energy_efficiency(
    strain: np.ndarray, stress: np.ndarray, cfg: AnalysisConfig = DEFAULT_CONFIG
) -> tuple[DensificationResult, list[Diagnostic]]:
    """Densification strain as the maximum of the energy-efficiency function.

        eta(eps) = [integral_0^eps sigma d(eps)] / sigma(eps)

    Efficiency rises through the plateau (cheap extra energy per unit stress)
    and turns over once the cell walls contact and stress climbs steeply. The
    turnover point needs no threshold, which is why it is the primary method.
    """
    diags: list[Diagnostic] = []
    if len(strain) < 3:
        return DensificationResult(method="energy_efficiency"), [
            Diagnostic("warning", "densification_failed", "too few samples")
        ]

    work = cumulative_energy(strain, stress)
    peak_stress = float(np.nanmax(stress))
    # Dividing by a near-zero stress produces meaningless spikes; mask them and
    # the earliest strains, where eta is small and noisy by construction.
    valid = (
        (stress > max(1e-9, 1e-3 * peak_stress))
        & (strain >= cfg.densification_min_strain)
        & np.isfinite(work)
    )
    if not valid.any():
        diags.append(
            Diagnostic(
                "warning",
                "densification_failed",
                f"no samples above {cfg.densification_min_strain:.0%} strain with "
                f"usable stress; max strain reached {strain[-1]:.3f}",
            )
        )
        return DensificationResult(method="energy_efficiency"), diags

    eta = np.full(len(strain), np.nan)
    eta[valid] = work[valid] / stress[valid]
    idx = int(np.nanargmax(eta))
    at_end = idx >= len(strain) - 2

    if at_end:
        diags.append(
            Diagnostic(
                "warning",
                "densification_not_reached",
                f"energy efficiency is still rising at the last sample "
                f"(strain {strain[-1]:.3f}); the test likely stopped before "
                f"densification, so the reported strain is a lower bound",
            )
        )
    return (
        DensificationResult(
            strain=float(strain[idx]),
            method="energy_efficiency",
            at_end_of_data=at_end,
            efficiency_max=float(eta[idx]),
        ),
        diags,
    )


def densification_tangent(
    strain: np.ndarray,
    stress: np.ndarray,
    plateau_MPa: float,
    modulus_MPa: float,
    cfg: AnalysisConfig = DEFAULT_CONFIG,
) -> tuple[DensificationResult, list[Diagnostic]]:
    """Densification where the tangent modulus first exceeds a multiple of the
    plateau-region tangent modulus."""
    diags: list[Diagnostic] = []
    tangent = tangent_modulus(strain, stress, cfg.tangent_window_strain)
    window = (strain >= cfg.plateau_strain_lo) & (strain <= cfg.plateau_strain_hi)
    if window.sum() < 2 or not np.isfinite(tangent[window]).any():
        diags.append(
            Diagnostic(
                "warning", "densification_failed", "plateau tangent modulus unavailable"
            )
        )
        return DensificationResult(method="tangent"), diags

    plateau_tangent = float(np.nanmean(tangent[window]))
    # A flat or softening plateau makes a multiple of the plateau slope
    # meaningless; fall back to a small fraction of the elastic modulus.
    floor = 0.02 * modulus_MPa if np.isfinite(modulus_MPa) and modulus_MPa > 0 else 0.0
    reference = max(plateau_tangent, floor)
    if reference <= 0:
        diags.append(
            Diagnostic(
                "warning",
                "densification_failed",
                "no positive reference slope for the tangent criterion",
            )
        )
        return DensificationResult(method="tangent"), diags

    threshold = cfg.densification_tangent_multiple * reference
    after = strain > cfg.plateau_strain_hi
    hits = np.flatnonzero(after & np.isfinite(tangent) & (tangent > threshold))
    if not hits.size:
        diags.append(
            Diagnostic(
                "warning",
                "densification_not_reached",
                f"tangent modulus never exceeded {threshold:.4g} MPa after "
                f"{cfg.plateau_strain_hi:.0%} strain",
            )
        )
        return DensificationResult(method="tangent"), diags
    idx = int(hits[0])
    return (
        DensificationResult(
            strain=float(strain[idx]),
            method="tangent",
            at_end_of_data=idx >= len(strain) - 2,
        ),
        diags,
    )


def densification_iso(
    strain: np.ndarray,
    stress: np.ndarray,
    plateau_MPa: float,
    cfg: AnalysisConfig = DEFAULT_CONFIG,
) -> tuple[DensificationResult, list[Diagnostic]]:
    """ISO 13314 criterion: stress first reaching 1.3x the plateau stress."""
    diags: list[Diagnostic] = []
    if not np.isfinite(plateau_MPa) or plateau_MPa <= 0:
        diags.append(
            Diagnostic("warning", "densification_failed", "plateau stress unavailable")
        )
        return DensificationResult(method="iso"), diags

    threshold = cfg.densification_iso_multiple * plateau_MPa
    after = strain > cfg.plateau_strain_hi
    hits = np.flatnonzero(after & (stress > threshold))
    if not hits.size:
        diags.append(
            Diagnostic(
                "warning",
                "densification_not_reached",
                f"stress never exceeded {cfg.densification_iso_multiple}x the "
                f"plateau stress ({threshold:.4g} MPa)",
            )
        )
        return DensificationResult(method="iso"), diags
    idx = int(hits[0])
    return (
        DensificationResult(
            strain=float(strain[idx]), method="iso", at_end_of_data=idx >= len(strain) - 2
        ),
        diags,
    )


def densification_strain(
    strain: np.ndarray,
    stress: np.ndarray,
    plateau_MPa: float,
    modulus_MPa: float,
    cfg: AnalysisConfig = DEFAULT_CONFIG,
) -> tuple[DensificationResult, list[Diagnostic]]:
    """Dispatch to the configured densification method."""
    method = cfg.densification_method
    if method == "energy_efficiency":
        return densification_energy_efficiency(strain, stress, cfg)
    if method == "tangent":
        return densification_tangent(strain, stress, plateau_MPa, modulus_MPa, cfg)
    if method == "iso":
        return densification_iso(strain, stress, plateau_MPa, cfg)
    raise ValueError(
        f"Unknown densification_method {method!r}; expected 'energy_efficiency', "
        f"'tangent' or 'iso'"
    )


# ----------------------------------------------------------------------
# First peak / crush force
# ----------------------------------------------------------------------
def first_peak(
    strain: np.ndarray,
    stress: np.ndarray,
    force: np.ndarray,
    limit_strain: float,
    cfg: AnalysisConfig = DEFAULT_CONFIG,
) -> tuple[PeakResult, list[Diagnostic]]:
    """Locate the initial load peak, if the pattern shows one.

    A peak counts only when the stress afterwards drops by at least
    `first_peak_min_drop_frac` -- otherwise it is a shoulder on a monotonically
    rising curve, not the buckling signature we are looking for.
    """
    from scipy.signal import find_peaks

    diags: list[Diagnostic] = []
    result = PeakResult()
    if len(strain) < 5:
        return result, diags

    limit = limit_strain if np.isfinite(limit_strain) else float(strain[-1])
    region = strain <= limit
    if region.sum() < 5:
        region = np.ones(len(strain), dtype=bool)

    eps = strain[region]
    sig = stress[region]
    frc = force[region]

    # Fallback crush load: the largest force before densification.
    max_idx = int(np.nanargmax(sig))
    result.crush_force_N = float(frc[max_idx])
    result.crush_stress_MPa = float(sig[max_idx])

    span = float(np.nanmax(sig))
    if span <= 0:
        return result, diags

    peaks, _ = find_peaks(sig, prominence=cfg.first_peak_prominence_frac * span)
    for p in peaks:
        after = sig[p:]
        if after.size < 2:
            continue
        trough = float(np.nanmin(after))
        drop = (sig[p] - trough) / sig[p] if sig[p] > 0 else 0.0
        if drop >= cfg.first_peak_min_drop_frac:
            result.first_peak_stress_MPa = float(sig[p])
            result.first_peak_strain = float(eps[p])
            result.first_peak_force_N = float(frc[p])
            result.has_distinct_first_peak = True
            result.post_peak_drop_frac = float(drop)
            result.crush_force_N = float(frc[p])
            result.crush_stress_MPa = float(sig[p])
            diags.append(
                Diagnostic(
                    "info",
                    "first_peak",
                    f"distinct first peak at strain {eps[p]:.4f}, "
                    f"{sig[p]:.4g} MPa, followed by a {100 * drop:.1f}% drop",
                )
            )
            break

    if not result.has_distinct_first_peak:
        diags.append(
            Diagnostic(
                "info",
                "no_first_peak",
                "no local maximum with real post-peak softening was found; "
                "crush force reported as the maximum before densification",
            )
        )
    return result, diags


# ----------------------------------------------------------------------
# Top-level
# ----------------------------------------------------------------------
def compute_metrics(
    curve: StressStrainCurve, spec: Specimen, cfg: AnalysisConfig = DEFAULT_CONFIG
) -> tuple[TestMetrics, StressStrainCurve]:
    """Compute every metric for one test.

    Returns the metrics and the curve they were computed on -- which differs
    from the input when toe compensation is enabled.
    """
    diagnostics: list[Diagnostic] = []

    fit, d = youngs_modulus(curve.strain, curve.stress_MPa, cfg)
    diagnostics += d

    toe_applied = False
    if cfg.toe_compensation and np.isfinite(fit.toe_strain) and fit.toe_strain > 0:
        curve = curve.shifted(fit.toe_strain)
        diagnostics.append(
            Diagnostic(
                "info",
                "toe_compensated",
                f"strain axis shifted by {fit.toe_strain:.5f} so the fitted "
                f"elastic line passes through the origin",
            )
        )
        toe_applied = True
        fit, d = youngs_modulus(curve.strain, curve.stress_MPa, cfg)
        diagnostics += d

    strain, stress, force = curve.strain, curve.stress_MPa, curve.force_N

    plateau, plateau_std, d = plateau_stress(strain, stress, cfg)
    diagnostics += d

    dens, d = densification_strain(strain, stress, plateau, fit.modulus_MPa, cfg)
    diagnostics += d

    # Energy absorbed up to densification; fall back to a fixed strain when
    # densification could not be located, and say so.
    limit = dens.strain
    if not np.isfinite(limit):
        limit = min(cfg.energy_fallback_strain, float(strain[-1]) if len(strain) else NAN)
        diagnostics.append(
            Diagnostic(
                "warning",
                "energy_fallback_limit",
                f"densification strain unavailable; energy integrated to "
                f"{limit:.3f} strain instead",
            )
        )
    energy_MJ_m3 = energy_to_strain(strain, stress, limit)

    # 1 MPa = 1 mJ/mm^3, so W * volume gives mJ; /1000 -> J.
    energy_J = energy_MJ_m3 * spec.volume_mm3 / 1000.0 if np.isfinite(energy_MJ_m3) else NAN

    if spec.mass_g is not None and spec.mass_g > 0:
        sea = energy_J / spec.mass_g
        basis = "mass"
    else:
        sea = NAN
        basis = "volume"
        diagnostics.append(
            Diagnostic(
                "info",
                "sea_volumetric",
                "no mass_g in specimens.csv; SEA reported on a volumetric basis "
                "as energy_absorption_MJ_m3 (numerically equal to MJ/m^3)",
            )
        )

    peaks, d = first_peak(strain, stress, force, dens.strain, cfg)
    diagnostics += d

    metrics = TestMetrics(
        test_id=spec.test_id,
        modulus=fit,
        plateau_stress_MPa=plateau,
        plateau_stress_std_MPa=plateau_std,
        plateau_strain_lo=cfg.plateau_strain_lo,
        plateau_strain_hi=cfg.plateau_strain_hi,
        densification=dens,
        energy_absorption_MJ_m3=energy_MJ_m3,
        energy_absorption_J=energy_J,
        sea_J_per_g=sea,
        sea_basis=basis,
        peaks=peaks,
        max_stress_MPa=curve.max_stress_MPa,
        max_force_N=float(np.nanmax(force)) if len(force) else NAN,
        max_strain=curve.max_strain,
        toe_compensation_applied=toe_applied,
        diagnostics=diagnostics,
    )
    return metrics, curve
