"""Tunable analysis parameters.

Everything that could reasonably need per-project or per-pattern tuning lives
here rather than being hardcoded in the algorithms, so a change is one edit (or
one CLI flag) instead of a code hunt.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AnalysisConfig:
    """Parameters controlling cleaning, metric extraction and plotting."""

    # ------------------------------------------------------------------
    # Raw parsing / sampling diagnostics (src.io)
    # ------------------------------------------------------------------
    expected_units: tuple[str, str, str] = ("(s)", "(mm)", "(kN)")
    #: A dt larger than this multiple of the median dt is reported as a gap.
    sampling_gap_factor: float = 3.0
    #: A dt smaller than this fraction of the median dt is reported as a short
    #: interval. The observed exports carry one such sample per file.
    sampling_short_factor: float = 0.5
    #: Displacement decreases smaller than this (mm) are treated as noise
    #: rather than genuine non-monotonicity.
    displacement_monotonic_tol_mm: float = 1e-4

    # ------------------------------------------------------------------
    # Machine force limit (src.io)
    # ------------------------------------------------------------------
    #: The load limit the test was configured to stop at, in kN (the frame's
    #: capacity is usually higher). Set this and every test that terminates
    #: against it is flagged, because a force-limited test is truncated: the
    #: specimen never reached densification, so densification strain, energy
    #: absorption and SEA are lower bounds rather than measurements.
    #: None disables the explicit check; the automatic one below still runs.
    force_limit_kN: float | None = None
    #: How close to `force_limit_kN` counts as "at the limit".
    force_limit_tolerance_frac: float = 0.001
    #: Even without `force_limit_kN`, a test whose final force is this fraction
    #: of its own maximum was stopped by a limit rather than by the specimen.
    truncation_final_force_frac: float = 0.999
    #: Trailing samples held at the limit are material-free instrument output.
    #: They are trimmed only when there are at least this many of them, so a
    #: clean cut-off at the limit (what the observed files do) is left alone.
    saturation_min_samples: int = 5

    # ------------------------------------------------------------------
    # Load-cell tare (src.io)
    # ------------------------------------------------------------------
    #: Subtract the pre-contact force offset (a non-zero load-cell reading at
    #: the start of the file) from the trimmed series. The observed exports
    #: start at 0-2.3 N. Reported either way as `tare_offset_N`.
    tare_correction: bool = True
    #: Samples averaged to estimate that offset.
    tare_window_samples: int = 50

    # ------------------------------------------------------------------
    # Pre-load trimming (src.io)
    # ------------------------------------------------------------------
    #: Contact is declared where force first exceeds this fraction of peak
    #: force and stays above it for `preload_sustain_samples` samples.
    preload_threshold_frac: float = 0.01
    preload_sustain_samples: int = 5
    #: Absolute floor for the contact threshold, in newtons. Protects against
    #: tests where the peak force is so large that 1% still sits inside the
    #: load-cell noise band.
    preload_threshold_min_N: float = 1.0

    # ------------------------------------------------------------------
    # Young's modulus (src.metrics)
    # ------------------------------------------------------------------
    #: Sliding-window search is restricted to strains below this value.
    modulus_search_max_strain: float = 0.10
    #: Windows narrower than this (in strain, and in samples) are not
    #: considered. Without a minimum width, maximising R^2 degenerates to a
    #: 2-point window where R^2 is trivially 1.0.
    modulus_min_window_strain: float = 0.005
    modulus_min_window_points: int = 10
    #: Search stride in samples; 1 is exhaustive, larger is faster.
    modulus_window_stride: int = 1
    #: How to pick among high-scoring windows.
    #: "steepest_high_r2" -> among windows within `modulus_r2_tolerance` of the
    #:                       best R^2, take the steepest (default).
    #: "max_r2"           -> the single highest-R^2 window.
    #: R^2 alone does not identify the *elastic* region: a foam plateau is
    #: frequently as straight as the initial slope, so both score ~1.0 and the
    #: tie is broken by floating-point noise -- which can return the plateau
    #: slope as the modulus. Taking the steepest of the tied windows picks the
    #: initial linear region by construction. See metrics.youngs_modulus.
    modulus_selection: str = "steepest_high_r2"
    modulus_r2_tolerance: float = 0.002

    # ------------------------------------------------------------------
    # Toe compensation (src.metrics) -- OFF by default, see README §Toe
    # ------------------------------------------------------------------
    #: Shift the curve so the fitted elastic line passes through the origin,
    #: removing seating/compliance slack (ASTM-style toe compensation).
    toe_compensation: bool = False

    # ------------------------------------------------------------------
    # Plateau (src.metrics)
    # ------------------------------------------------------------------
    #: ISO 13314 plateau window. NOTE: the published standard averages between
    #: 20% and 40% strain; the project brief specified 20-30%. The brief's
    #: value is the default -- set 0.20/0.40 to follow ISO exactly.
    plateau_strain_lo: float = 0.20
    plateau_strain_hi: float = 0.30
    #: Minimum samples required inside the window before a plateau stress is
    #: reported (guards against tests that stop before 20% strain).
    plateau_min_points: int = 3
    #: Fraction of the plateau window the test must actually span. A test that
    #: stops partway through the window would average only its low end, which
    #: biases the plateau low on the usual upward-sloping plateau; below this
    #: coverage the plateau stress is not reported.
    plateau_min_coverage: float = 0.9

    # ------------------------------------------------------------------
    # Densification onset (src.metrics)
    # ------------------------------------------------------------------
    #: "energy_efficiency" (primary, parameter-light), "tangent" or "iso".
    densification_method: str = "energy_efficiency"
    #: Efficiency maxima below this strain are ignored -- they are numerical
    #: artefacts of dividing by a near-zero stress, not real densification.
    densification_min_strain: float = 0.02
    #: "tangent": densification where the rolling tangent modulus first
    #: exceeds this multiple of the plateau-region tangent modulus.
    densification_tangent_multiple: float = 2.0
    #: "iso": densification where stress first exceeds this multiple of the
    #: plateau stress (ISO 13314 uses 1.3).
    densification_iso_multiple: float = 1.3
    #: Half-width, in strain, of the rolling window used for tangent modulus.
    tangent_window_strain: float = 0.01
    #: A densification strain found within this much strain of the end of the
    #: data is treated as "not reached": the criterion was still climbing when
    #: the test stopped, so the value is a lower bound. Defined in strain
    #: rather than samples so sampling noise cannot move the verdict.
    densification_end_tolerance_strain: float = 0.01

    # ------------------------------------------------------------------
    # First peak / crush force (src.metrics)
    # ------------------------------------------------------------------
    #: Peak prominence required, as a fraction of peak stress.
    first_peak_prominence_frac: float = 0.02
    #: A peak only counts as a *first peak* if the stress subsequently drops
    #: by at least this fraction of the peak value (i.e. real softening).
    first_peak_min_drop_frac: float = 0.02
    #: Peaks and drops are judged on the curve smoothed over this much strain.
    #: Load-cell noise on a soft specimen can otherwise fake a 2% "drop"; a
    #: real buckling peak spans several percent strain and is unaffected.
    first_peak_smoothing_strain: float = 0.003

    # ------------------------------------------------------------------
    # Energy absorption (src.metrics)
    # ------------------------------------------------------------------
    #: Fallback integration limit when densification strain is unavailable.
    energy_fallback_strain: float = 0.50

    # ------------------------------------------------------------------
    # Fixed-strain metrics (src.metrics)
    # ------------------------------------------------------------------
    #: Strains at which stress, absorbed energy and (with a mass) SEA are also
    #: reported. These stay well-defined when a test is cut off by the load
    #: limit -- unlike energy to densification, which then becomes only a lower
    #: bound -- so they are the fair basis for comparing truncated tests.
    fixed_strain_levels: tuple[float, ...] = (0.10, 0.20)

    # ------------------------------------------------------------------
    # Envelope plotting (src.envelope)
    # ------------------------------------------------------------------
    envelope_n_points: int = 500
    envelope_per_pattern_bands: bool = True
    figure_dpi: int = 150
    #: Up to this many patterns, each gets its own colour and legend entry.
    #: Beyond it, colour can no longer identify a pattern, so tests are drawn
    #: as plain data points with no per-pattern legend.
    legend_max_patterns: int = 8
    #: Colour curves and points by this parameter (e.g. "relative_density" or
    #: a pattern_params key) on a continuous scale with a colorbar. None draws
    #: them in a single colour, or by pattern when there are few patterns.
    color_by: str | None = None

    # ------------------------------------------------------------------
    # Parameter-property correlation (src.correlation)
    # ------------------------------------------------------------------
    #: Properties tested against every parameter. Columns that are missing or
    #: empty are skipped.
    correlation_properties: tuple[str, ...] = (
        "modulus_MPa",
        "plateau_stress_MPa",
        "first_peak_stress_MPa",
        "stress_at_10pct_MPa",
        "stress_at_20pct_MPa",
        "energy_to_20pct_MJ_m3",
        "sea_to_20pct_J_per_g",
        "energy_absorption_MJ_m3",
        "sea_J_per_g",
        "densification_strain",
    )
    #: Fewest tests a parameter-property pair needs before it is tested.
    correlation_min_n: int = 8
    #: False-discovery-rate level for calling an association significant.
    correlation_fdr_alpha: float = 0.05
    #: A level of a categorical factor needs this many tests to take part.
    correlation_min_group_size: int = 3
    #: Parameters whose variance inflation factor exceeds this are too
    #: entangled with the others for their partial correlation to be trusted.
    correlation_vif_threshold: float = 10.0
    #: Parameter pairs correlated at least this strongly are reported as
    #: collinear: their separate effects cannot be told apart in the data.
    correlation_collinear_rho: float = 0.8
    #: Include values that are only lower bounds (test stopped at the load
    #: limit before densifying). Off by default: correlating a bound against a
    #: parameter mostly measures where each test happened to stop.
    correlation_include_censored: bool = False

    # ------------------------------------------------------------------

    @classmethod
    def from_json(cls, path: str | Path) -> "AnalysisConfig":
        """Load overrides from a JSON file; unspecified keys keep defaults."""
        data = json.loads(Path(path).read_text())
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnalysisConfig":
        # JSON has no comments, so keys starting with "_" are treated as notes
        # and ignored. Anything else unrecognised is a typo and is rejected.
        data = {k: v for k, v in data.items() if not k.startswith("_")}
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(
                f"Unknown config key(s): {sorted(unknown)}. "
                f"Valid keys: {sorted(known)}"
            )
        # JSON has lists, not tuples; restore tuples for tuple-valued fields so
        # a config read from disk compares equal to one built in code.
        tuple_fields = {
            f.name for f in fields(cls) if isinstance(f.default, tuple)
        }
        data = {
            k: tuple(v) if k in tuple_fields and isinstance(v, list) else v
            for k, v in data.items()
        }
        return cls(**data)

    def replace(self, **overrides: Any) -> "AnalysisConfig":
        return AnalysisConfig.from_dict({**asdict(self), **overrides})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_CONFIG = AnalysisConfig()

#: Project-level overrides, picked up automatically from the repository root.
#: This is where machine setup belongs (the load limit the frame stops at, for
#: instance) so every run agrees on it without repeating a flag.
PROJECT_CONFIG_FILENAME = "analysis_config.json"


def load_project_config(
    explicit: str | Path | None = None, root: str | Path | None = None
) -> AnalysisConfig:
    """Resolve the config: an explicit file, else the project file, else defaults."""
    if explicit is not None:
        return AnalysisConfig.from_json(explicit)

    base = Path(root) if root is not None else Path(__file__).resolve().parent.parent
    candidate = base / PROJECT_CONFIG_FILENAME
    if candidate.exists():
        return AnalysisConfig.from_json(candidate)
    return DEFAULT_CONFIG
