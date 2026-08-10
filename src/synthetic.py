"""Generation of instrument-format CSVs for tests and demos.

The real instrument export was not available when this pipeline was written, so
this module reproduces the format and the quirks documented in the project
brief: seven lines of export boilerplate, a header row, a units row, quoted
numeric values, load-cell noise around zero before contact, and a sampling
interval that is nominally 0.02 s but not exactly uniform.

Files produced here are synthetic. They exist so the parser can be tested
against the documented format and so the pipeline can be demonstrated end to
end before real data is registered. Drop the genuine export into
`tests/fixtures/` and the test suite will prefer it (see tests/conftest.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

BOILERPLATE = ["Results Table 1", "", "", "Results Table 2", "", "", ""]
HEADER = "Time,Displacement,Force"
UNITS = "(s),(mm),(kN)"


@dataclass
class FoamCurveSpec:
    """Parameters of a synthetic cellular-solid compression response."""

    modulus_MPa: float = 60.0
    #: Strain and stress at the first (buckling) peak; set peak_stress to None
    #: for a pattern with no distinct first peak.
    peak_strain: float = 0.045
    peak_stress_MPa: float | None = 2.70
    #: Stress the curve settles to after post-peak softening.
    plateau_stress_MPa: float = 2.15
    #: Gentle upward slope through the plateau.
    plateau_slope_MPa: float = 0.90
    softening_end_strain: float = 0.09
    densification_strain: float = 0.42
    densification_rise_MPa: float = 12.0
    densification_exponent: float = 2.2
    densification_span: float = 0.20


def foam_stress(strain: np.ndarray, spec: FoamCurveSpec) -> np.ndarray:
    """Piecewise stress-strain response: elastic, peak, plateau, densification.

    With `peak_stress_MPa` set, the elastic rise overshoots to a first peak and
    softens into the plateau. With it set to None the elastic rise meets the
    plateau directly, giving a strictly monotonic curve -- a pattern with no
    buckling signature at all, which the first-peak detector must not invent a
    peak for.
    """
    stress = np.zeros_like(strain, dtype=float)

    has_peak = spec.peak_stress_MPa is not None
    e_d = spec.densification_strain

    if has_peak:
        e_pk = spec.peak_strain
        plateau_start = spec.softening_end_strain
    else:
        # Elastic meets the plateau exactly: continuous and never decreasing.
        e_pk = spec.plateau_stress_MPa / spec.modulus_MPa
        plateau_start = e_pk

    elastic = strain < e_pk
    stress[elastic] = spec.modulus_MPa * strain[elastic]

    # Post-peak softening: half a cosine from the peak down to the plateau.
    if has_peak:
        soft = (strain >= e_pk) & (strain < plateau_start)
        if soft.any() and plateau_start > e_pk:
            t = (strain[soft] - e_pk) / (plateau_start - e_pk)
            stress[soft] = spec.plateau_stress_MPa + (
                float(spec.peak_stress_MPa) - spec.plateau_stress_MPa
            ) * 0.5 * (1 + np.cos(np.pi * t))

    plateau = (strain >= plateau_start) & (strain < e_d)
    stress[plateau] = spec.plateau_stress_MPa + spec.plateau_slope_MPa * (
        strain[plateau] - plateau_start
    )

    dense = strain >= e_d
    if dense.any():
        base = spec.plateau_stress_MPa + spec.plateau_slope_MPa * (e_d - plateau_start)
        t = np.clip((strain[dense] - e_d) / spec.densification_span, 0.0, None)
        stress[dense] = base + spec.densification_rise_MPa * t**spec.densification_exponent

    return stress


def make_time_axis(n: int, dt: float = 0.02, hiccup_at: float = 2.26) -> np.ndarray:
    """Nominally uniform time with one small irregularity.

    Reproduces the documented behaviour: exactly `dt` for the first couple of
    seconds, then a single slightly longer interval, after which the values are
    offset (…2.2600, 2.2820, 2.3020, 2.3220…). Downstream code must therefore
    derive rates from this column rather than assuming a constant rate.
    """
    time = np.arange(n, dtype=float) * dt
    shift_from = int(round(hiccup_at / dt)) + 1
    if 0 < shift_from < n:
        time[shift_from:] += 0.002
    return np.round(time, 4)


@dataclass
class SyntheticTest:
    """A synthetic test ready to be written in the instrument's format."""

    name: str
    curve: FoamCurveSpec = field(default_factory=FoamCurveSpec)
    side_length_mm: float = 20.0
    initial_height_mm: float = 20.0
    #: Crosshead travel before the platen actually contacts the specimen.
    slack_mm: float = 0.48
    crosshead_mm_per_s: float = 0.24
    max_strain: float = 0.60
    dt_s: float = 0.02
    #: Load-cell noise, one standard deviation in newtons.
    noise_N: float = 1.5
    seed: int = 0

    @property
    def area_mm2(self) -> float:
        return self.side_length_mm**2

    def generate(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return `(time_s, disp_mm, force_kN)` as the instrument would log them."""
        rng = np.random.default_rng(self.seed)
        travel = self.slack_mm + self.max_strain * self.initial_height_mm
        n = int(round(travel / (self.crosshead_mm_per_s * self.dt_s))) + 1

        time = make_time_axis(n, self.dt_s)
        disp = self.crosshead_mm_per_s * time
        # Crosshead position wobbles a little; encoder resolution is ~0.1 um.
        disp = np.round(disp + rng.normal(0, 2e-4, n), 4)

        engaged = np.clip(disp - self.slack_mm, 0.0, None)
        strain = engaged / self.initial_height_mm
        force = foam_stress(strain, self.curve) * self.area_mm2
        force = force + rng.normal(0, self.noise_N, n)
        # Before contact the cell reads noise around zero, not the ramp.
        force = np.where(disp < self.slack_mm, rng.normal(0, self.noise_N, n), force)

        return time, disp, np.round(force / 1000.0, 4)

    def write(self, path: str | Path) -> Path:
        """Write the test to `path` in the instrument's export format."""
        time, disp, force_kN = self.generate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        lines = list(BOILERPLATE) + [HEADER, UNITS]
        lines += [
            f'"{t:.4f}","{d:.4f}","{f:.4f}"' for t, d, f in zip(time, disp, force_kN)
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path


#: The example filename from the project brief, used as the parser fixture.
SAMPLE_FILENAME = "solid_compression_20260717_192209_1_1.csv"


def sample_test() -> SyntheticTest:
    """The reference synthetic test matching the brief's example filename."""
    return SyntheticTest(name=SAMPLE_FILENAME, seed=20260717)
