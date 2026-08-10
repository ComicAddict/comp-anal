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
    #: Displacement decreases smaller than this (mm) are treated as noise
    #: rather than genuine non-monotonicity.
    displacement_monotonic_tol_mm: float = 1e-4

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

    # ------------------------------------------------------------------
    # First peak / crush force (src.metrics)
    # ------------------------------------------------------------------
    #: Peak prominence required, as a fraction of peak stress.
    first_peak_prominence_frac: float = 0.02
    #: A peak only counts as a *first peak* if the stress subsequently drops
    #: by at least this fraction of the peak value (i.e. real softening).
    first_peak_min_drop_frac: float = 0.02

    # ------------------------------------------------------------------
    # Energy absorption (src.metrics)
    # ------------------------------------------------------------------
    #: Fallback integration limit when densification strain is unavailable.
    energy_fallback_strain: float = 0.50

    # ------------------------------------------------------------------
    # Envelope plotting (src.envelope)
    # ------------------------------------------------------------------
    envelope_n_points: int = 500
    envelope_per_pattern_bands: bool = True
    figure_dpi: int = 150

    # ------------------------------------------------------------------

    @classmethod
    def from_json(cls, path: str | Path) -> "AnalysisConfig":
        """Load overrides from a JSON file; unspecified keys keep defaults."""
        data = json.loads(Path(path).read_text())
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnalysisConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(
                f"Unknown config key(s): {sorted(unknown)}. "
                f"Valid keys: {sorted(known)}"
            )
        if "expected_units" in data:
            data = dict(data, expected_units=tuple(data["expected_units"]))
        return cls(**data)

    def replace(self, **overrides: Any) -> "AnalysisConfig":
        return AnalysisConfig.from_dict({**asdict(self), **overrides})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_CONFIG = AnalysisConfig()
