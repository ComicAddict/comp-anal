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
import pandas as pd

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
    #: Stop the test when force reaches this load, as a frame configured with
    #: a load limit does (9.5 kN on the 10 kN frame). None runs to max_strain.
    force_limit_N: float | None = None

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

        if self.force_limit_N is not None:
            hit = np.flatnonzero(force >= self.force_limit_N)
            if hit.size:
                # The frame stops on the first sample at the limit, which is
                # logged at the limit -- exactly how the real exports end.
                k = int(hit[0])
                time, disp, force = time[: k + 1], disp[: k + 1], force[: k + 1].copy()
                force[-1] = self.force_limit_N

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


# ----------------------------------------------------------------------
# A planted design space: known causes, for demos and for testing the
# correlation analysis against ground truth.
# ----------------------------------------------------------------------
#: Levels each factor is drawn from.
PLANTED_LEVELS: dict[str, tuple] = {
    "cell_mm": (4.0, 5.0, 6.0, 7.0, 8.0),
    "wall_mm": (0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2),
    "topology": ("gyroid", "honeycomb"),
    "layer_height_mm": (0.12, 0.20, 0.28),
    "print_speed_mm_s": (40.0, 60.0, 80.0),
    "nozzle_temp_C": (200.0, 210.0, 220.0),
}

#: Print batches, each printed on its own day -- so batch and print date are
#: confounded by construction, as they usually are in practice.
PLANTED_BATCH_DAYS: dict[str, str] = {
    "B1": "2026-07-01", "B2": "2026-07-03", "B3": "2026-07-06", "B4": "2026-07-08",
}
#: Effect sizes of the planted manufacturing factors (multipliers on modulus
#: and strength). Module-level so a power study can vary them.
PLANTED_BATCH_FACTOR = 0.88   # batch B3
PLANTED_EDGE_FACTOR = 0.85    # bed columns C1 and C4

#: Build-plate positions: 2 rows x 4 columns; columns 1 and 4 are the edges.
PLANTED_BED_POSITIONS: tuple[str, ...] = tuple(f"R{r}C{c}" for r in (1, 2) for c in (1, 2, 3, 4))

#: The ground truth `planted_design` builds in. Anything not listed here has
#: no effect at all -- print speed, nozzle temperature and the time between
#: printing and testing are pure nulls.
PLANTED_TRUTH = """\
relative density  rho* = k_topology * wall / cell * (1 + 3% noise)
                  k = 1.25 gyroid, 1.6 honeycomb
modulus           E = 2500 MPa * rho*^2 * f_topology * f_layer * f_batch * f_bed
plateau stress    sigma = 18 MPa * rho*^1.5 * (same factors)    (bending: n = 2, 1.5)
topology          f = 1.0 gyroid, 1.3 honeycomb; honeycomb also buckles (first
                  peak 1.25 x plateau), gyroid does not
layer height      f = 1 - 1.2 * (layer_height_mm - 0.20): about +/-10 percent --
                  small next to the ~30x spread from density, so it is MASKED
                  pairwise and only shows once density is held fixed
print batch       f = 0.88 for batch B3 (say, wet filament), 1.0 otherwise --
                  also masked pairwise. Each batch is printed on one day, so
                  print_date is CONFOUNDED with it: one effect, two names
bed position      f = 0.85 at the edge columns (C1, C4), 1.0 in the middle
densification     strain = 0.75 - rho*, clipped to 0.30-0.55
print speed       no effect
nozzle temp       no effect
print-to-test     no effect (tested 2-14 days after printing)
"""


def planted_design(n: int = 64, seed: int = 7) -> pd.DataFrame:
    """Draw `n` distinct patterns with known parameter -> property causes.

    Cell size, wall thickness and topology form the pattern (every one
    distinct, like a real design sweep); the manufacturing settings are drawn
    independently per test. Returns one row per test with the parameters and
    the *true* properties (before any measurement), per PLANTED_TRUTH.
    """
    rng = np.random.default_rng(seed)
    combos = [
        (c, w, t)
        for c in PLANTED_LEVELS["cell_mm"]
        for w in PLANTED_LEVELS["wall_mm"]
        for t in PLANTED_LEVELS["topology"]
    ]
    if n > len(combos):
        raise ValueError(f"at most {len(combos)} distinct patterns available")
    picks = rng.choice(len(combos), size=n, replace=False)

    batches = list(PLANTED_BATCH_DAYS)
    batch_of = rng.permutation(np.resize(batches, n))
    # Spread the bed positions evenly within each batch, as a real build plate is.
    bed_of = np.empty(n, dtype=object)
    for batch in batches:
        members = np.flatnonzero(batch_of == batch)
        bed_of[members] = rng.permutation(np.resize(PLANTED_BED_POSITIONS, len(members)))

    rows = []
    for i, pick in enumerate(picks):
        cell, wall, topology = combos[pick]
        layer = float(rng.choice(PLANTED_LEVELS["layer_height_mm"]))
        speed = float(rng.choice(PLANTED_LEVELS["print_speed_mm_s"]))
        temp = float(rng.choice(PLANTED_LEVELS["nozzle_temp_C"]))
        batch, bed = str(batch_of[i]), str(bed_of[i])
        printed = pd.Timestamp(PLANTED_BATCH_DAYS[batch]) + pd.Timedelta(hours=float(rng.uniform(8, 18)))
        tested = (printed.normalize() + pd.Timedelta(days=int(rng.integers(2, 15)))
                  + pd.Timedelta(hours=float(rng.uniform(9, 17))))

        k = 1.25 if topology == "gyroid" else 1.6
        rho = float(np.clip(k * wall / cell * (1 + rng.normal(0, 0.03)), 0.05, 0.6))
        f_top = 1.0 if topology == "gyroid" else 1.3
        f_layer = 1.0 - 1.2 * (layer - 0.20)
        f_batch = PLANTED_BATCH_FACTOR if batch == "B3" else 1.0
        f_bed = PLANTED_EDGE_FACTOR if bed.endswith(("C1", "C4")) else 1.0
        f_all = f_top * f_layer * f_batch * f_bed
        modulus = 2500.0 * rho**2 * f_all * float(rng.lognormal(0, 0.05))
        plateau = 18.0 * rho**1.5 * f_all * float(rng.lognormal(0, 0.05))

        rows.append(
            {
                "test_id": f"doe_{i + 1:02d}",
                "pattern_name": f"{topology}_c{cell:g}_w{wall:g}",
                "topology": topology,
                "cell_mm": cell,
                "wall_mm": wall,
                "layer_height_mm": layer,
                "print_speed_mm_s": speed,
                "nozzle_temp_C": temp,
                "print_batch": batch,
                "bed_position": bed,
                "print_date": f"{printed:%Y-%m-%d %H:%M}",
                "test_date": f"{tested:%Y-%m-%d %H:%M}",
                "relative_density": round(rho, 4),
                "material": "PLA",
                # Solid PLA ~1.24 g/cm^3 over a 20 mm cube (8 cm^3).
                "mass_g": round(1.24 * 8.0 * rho, 3),
                "true_modulus_MPa": modulus,
                "true_plateau_MPa": plateau,
                "true_peak_MPa": 1.25 * plateau if topology == "honeycomb" else np.nan,
                "true_densification_strain": float(np.clip(0.75 - rho, 0.30, 0.55)),
            }
        )
    return pd.DataFrame(rows)


def planted_curve_spec(row) -> FoamCurveSpec:
    """The synthetic compression curve for one `planted_design` row."""
    modulus = float(row["true_modulus_MPa"])
    plateau = float(row["true_plateau_MPa"])
    peak = row["true_peak_MPa"]
    has_peak = peak == peak  # not NaN
    peak_strain = float(peak) / modulus if has_peak else plateau / modulus
    return FoamCurveSpec(
        modulus_MPa=modulus,
        peak_strain=peak_strain,
        peak_stress_MPa=float(peak) if has_peak else None,
        plateau_stress_MPa=plateau,
        plateau_slope_MPa=0.4 * plateau,
        softening_end_strain=peak_strain + 0.04,
        densification_strain=float(row["true_densification_strain"]),
        densification_rise_MPa=6.0 * plateau,
        densification_exponent=2.2,
        densification_span=0.2,
    )
