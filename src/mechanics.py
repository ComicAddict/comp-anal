"""Stress/strain conversion and curve utilities.

Unit convention throughout: force in N, length in mm, area in mm^2, therefore
stress in N/mm^2 = MPa. Strain is dimensionless (mm/mm). A consequence worth
knowing: energy absorbed per unit volume, integral(sigma d(epsilon)), comes out
in MPa, which is numerically identical to MJ/m^3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .specimens import Specimen

# np.trapezoid replaced np.trapz in NumPy 2.0; support both.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


@dataclass
class StressStrainCurve:
    """An engineering stress-strain curve for one test."""

    test_id: str
    strain: np.ndarray      # dimensionless, compressive taken positive
    stress_MPa: np.ndarray
    force_N: np.ndarray
    disp_mm: np.ndarray
    time_s: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.strain)
        for name in ("stress_MPa", "force_N", "disp_mm", "time_s"):
            if len(getattr(self, name)) != n:
                raise ValueError(
                    f"[{self.test_id}] {name} has {len(getattr(self, name))} "
                    f"points but strain has {n}"
                )

    @property
    def max_strain(self) -> float:
        return float(self.strain[-1]) if len(self.strain) else float("nan")

    @property
    def max_stress_MPa(self) -> float:
        return float(np.nanmax(self.stress_MPa)) if len(self.stress_MPa) else float("nan")

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "time_s": self.time_s,
                "disp_mm": self.disp_mm,
                "force_N": self.force_N,
                "strain": self.strain,
                "stress_MPa": self.stress_MPa,
            }
        )

    def shifted(self, strain_offset: float) -> "StressStrainCurve":
        """Return a copy with the strain axis shifted and re-clipped at zero.

        Used for toe compensation: points that fall below zero strain after the
        shift are dropped rather than kept as negative strain.
        """
        strain = self.strain - strain_offset
        keep = strain >= 0.0
        if not keep.any():
            return self
        return StressStrainCurve(
            test_id=self.test_id,
            strain=strain[keep],
            stress_MPa=self.stress_MPa[keep],
            force_N=self.force_N[keep],
            disp_mm=self.disp_mm[keep],
            time_s=self.time_s[keep],
        )


def to_stress_strain(trimmed: pd.DataFrame, spec: Specimen) -> StressStrainCurve:
    """Convert a cleaned `[time_s, disp_mm, force_N]` frame to stress-strain.

        engineering_strain = disp_mm / initial_height_mm
        engineering_stress = force_N / cross_section_area_mm2   [MPa]
    """
    if spec.initial_height_mm <= 0:
        raise ValueError(f"[{spec.test_id}] initial_height_mm must be > 0")
    if spec.cross_section_area_mm2 <= 0:
        raise ValueError(f"[{spec.test_id}] cross_section_area_mm2 must be > 0")

    disp = trimmed["disp_mm"].to_numpy(dtype=float)
    force = trimmed["force_N"].to_numpy(dtype=float)
    return StressStrainCurve(
        test_id=spec.test_id,
        strain=disp / spec.initial_height_mm,
        stress_MPa=force / spec.cross_section_area_mm2,
        force_N=force,
        disp_mm=disp,
        time_s=trimmed["time_s"].to_numpy(dtype=float),
    )


def cumulative_energy(strain: np.ndarray, stress: np.ndarray) -> np.ndarray:
    """Cumulative integral(sigma d(epsilon)) at each point, in MPa (= MJ/m^3)."""
    if len(strain) == 0:
        return np.asarray([], dtype=float)
    out = np.zeros(len(strain), dtype=float)
    if len(strain) > 1:
        out[1:] = np.cumsum(np.diff(strain) * 0.5 * (stress[1:] + stress[:-1]))
    return out


def _first_crossing(strain: np.ndarray, level: float) -> int | None:
    """Index of the first sample at or beyond `level`, or None if never reached.

    "First" matters: measured strain can step back by a hair through noise, so
    a mask like `strain <= level` would also pick up stray samples from after
    the crossing and fold them into the integral out of order.
    """
    reached = np.flatnonzero(strain >= level)
    return int(reached[0]) if reached.size else None


def value_at_strain(strain: np.ndarray, values: np.ndarray, level: float) -> float:
    """`values` interpolated at the first point the curve reaches `level`.

    NaN when the test never reaches that strain -- an extrapolated stress is
    not a measurement.
    """
    if not np.isfinite(level) or len(strain) < 2:
        return float("nan")
    k = _first_crossing(strain, level)
    if k is None:
        return float("nan")
    if k == 0:
        return float(values[0])
    x0, x1 = float(strain[k - 1]), float(strain[k])
    y0, y1 = float(values[k - 1]), float(values[k])
    if x1 == x0:
        return y1
    return y0 + (y1 - y0) * (level - x0) / (x1 - x0)


def energy_to_strain(strain: np.ndarray, stress: np.ndarray, limit: float) -> float:
    """integral(sigma d(epsilon)) from 0 to `limit`, interpolating the endpoint.

    Integrates up to the first time the curve reaches `limit`. When it never
    does, the whole curve is integrated (the limit is clipped to the data);
    callers that need a strict "reached or NaN" check do it themselves.
    """
    if not np.isfinite(limit) or len(strain) < 2:
        return float("nan")
    k = _first_crossing(strain, limit)
    if k is None:
        return float(_trapezoid(stress, strain))
    if k == 0:
        return 0.0
    eps = np.append(strain[:k], limit)
    sig = np.append(stress[:k], value_at_strain(strain, stress, limit))
    return float(_trapezoid(sig, eps))


def tangent_modulus(
    strain: np.ndarray, stress: np.ndarray, window_strain: float
) -> np.ndarray:
    """Rolling instantaneous slope d(sigma)/d(epsilon).

    The window is defined in strain (not samples) so it means the same thing on
    curves recorded at different rates. Uses a centred least-squares slope.
    """
    n = len(strain)
    out = np.full(n, np.nan)
    if n < 3 or window_strain <= 0:
        return out

    lo_idx = np.searchsorted(strain, strain - window_strain, side="left")
    hi_idx = np.searchsorted(strain, strain + window_strain, side="right")
    for i in range(n):
        lo, hi = int(lo_idx[i]), int(hi_idx[i])
        if hi - lo < 3:
            continue
        x = strain[lo:hi]
        y = stress[lo:hi]
        x_span = x[-1] - x[0]
        if x_span <= 0:
            continue
        out[i] = float(np.polyfit(x, y, 1)[0])
    return out


def resample_curve(
    strain: np.ndarray, stress: np.ndarray, grid: np.ndarray
) -> np.ndarray:
    """Interpolate a curve onto `grid`; grid points beyond the data are NaN."""
    if len(strain) < 2:
        return np.full(len(grid), np.nan)
    order = np.argsort(strain, kind="stable")
    eps = strain[order]
    sig = stress[order]
    # np.interp needs strictly increasing x; average duplicate strains.
    unique_eps, inverse = np.unique(eps, return_inverse=True)
    if len(unique_eps) != len(eps):
        summed = np.bincount(inverse, weights=sig)
        counts = np.bincount(inverse)
        eps, sig = unique_eps, summed / counts
    out = np.interp(grid, eps, sig, left=np.nan, right=np.nan)
    return np.where((grid < eps[0]) | (grid > eps[-1]), np.nan, out)
