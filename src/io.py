"""Parsing and cleaning of raw instrument CSV exports.

The exporter writes a few lines of boilerplate, then a header row, then a units
row, then quoted numeric data:

    Results Table 1


    Results Table 2



    Time,Displacement,Force
    (s),(mm),(kN)
    "0.0000","0.0000","0.0000"

Two things this module refuses to guess at: the units row (a silent change from
kN to N would scale every result by 1000) and the location of the header (found
by scanning, not by a hardcoded line count, so an exporter that adds or drops a
blank line does not corrupt the data).

Quirks confirmed against real exports from the 10 kN frame:
  * CRLF line endings throughout.
  * The boilerplate block is *not* always present -- some exports begin with a
    single blank line and go straight to the header. Scanning handles both.
  * A non-zero load-cell reading at the start of the file (0 to 2.3 N observed),
    carried on every sample until contact. See `estimate_tare`.
  * One sampling interval per file that is *shorter* than the nominal 0.02 s
    (0.002 to 0.010 s observed), rather than the longer interval the project
    brief described. Both are reported; neither is assumed away.
  * Tests terminate against the configured load limit rather than by specimen
    behaviour, which truncates the curve. See `check_force_limit` -- this is the
    single most consequential thing to know about such a file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG, AnalysisConfig

HEADER_TOKENS = ("time", "displacement", "force")
BOILERPLATE_PREFIXES = ("results table",)


class RawParseError(ValueError):
    """Raised when a raw CSV does not match the expected export format."""


class UnitsError(RawParseError):
    """Raised when the units row is not the expected (s),(mm),(kN)."""


@dataclass
class Diagnostic:
    """A non-fatal observation about a test file.

    Flagged, never silently applied -- the caller decides what to do.
    """

    level: str  # "info" | "warning"
    code: str
    message: str

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"[{self.level.upper()}] {self.code}: {self.message}"


@dataclass
class LoadedTest:
    """A parsed raw file plus its cleaned, contact-zeroed series."""

    path: Path
    raw: pd.DataFrame           # untouched: time_s, disp_mm, force_kN, force_N
    trimmed: pd.DataFrame       # from contact onward, displacement re-zeroed
    contact_index: int
    contact_force_N: float
    diagnostics: list[Diagnostic] = field(default_factory=list)
    #: Pre-contact load-cell offset, in newtons (subtracted from `trimmed` when
    #: `tare_correction` is on).
    tare_offset_N: float = 0.0
    #: True when the test stopped against a load limit instead of running to
    #: densification -- every strain-referenced metric is then a lower bound.
    force_limited: bool = False
    #: Trailing samples removed because the frame was holding at its limit.
    saturated_samples: int = 0

    @property
    def warnings(self) -> list[Diagnostic]:
        return [d for d in self.diagnostics if d.level == "warning"]

    def diagnostics_text(self) -> str:
        return "; ".join(f"{d.code}: {d.message}" for d in self.diagnostics)


def _split_row(line: str) -> list[str]:
    """Split a CSV line and strip surrounding quotes/whitespace from cells."""
    return [cell.strip().strip('"').strip() for cell in line.split(",")]


def locate_header(lines: list[str]) -> int:
    """Index of the `Time,Displacement,Force` row.

    Scans rather than assuming a fixed offset, so an extra or missing blank
    line in the export is harmless.
    """
    for idx, line in enumerate(lines):
        cells = [c.lower() for c in _split_row(line)]
        if len(cells) >= 3 and tuple(cells[:3]) == HEADER_TOKENS:
            return idx
    raise RawParseError(
        "Could not find the 'Time,Displacement,Force' header row. "
        f"First 10 lines were: {lines[:10]!r}"
    )


def validate_units(cells: list[str], cfg: AnalysisConfig) -> None:
    """Fail loudly if the units row differs from the expected units."""
    actual = tuple(c.lower() for c in cells[:3])
    expected = tuple(u.lower() for u in cfg.expected_units)
    if actual != expected:
        raise UnitsError(
            f"Unexpected units row: got {list(cells[:3])}, expected "
            f"{list(cfg.expected_units)}. Refusing to parse -- a units change "
            f"(e.g. N instead of kN) would silently rescale every result. "
            f"If the instrument export really changed, set "
            f"AnalysisConfig.expected_units and add the conversion."
        )


def _check_boilerplate(lines: list[str], header_idx: int) -> list[Diagnostic]:
    """Confirm everything above the header is blank or 'Results Table N'."""
    unexpected = [
        f"line {i + 1}: {line.strip()!r}"
        for i, line in enumerate(lines[:header_idx])
        if line.strip()
        and not line.strip().lower().startswith(BOILERPLATE_PREFIXES)
    ]
    if unexpected:
        return [
            Diagnostic(
                "warning",
                "unexpected_preamble",
                "Non-boilerplate content above the header was skipped: "
                + ", ".join(unexpected),
            )
        ]
    return []


def check_sampling(time_s: np.ndarray, cfg: AnalysisConfig) -> list[Diagnostic]:
    """Report the sampling interval and any gaps, resets or duplicates.

    The nominal step is 0.02 s but is not exactly uniform, so every downstream
    calculation derives dt from this column instead of assuming a rate.
    """
    diags: list[Diagnostic] = []
    if time_s.size < 2:
        return [Diagnostic("warning", "too_few_samples", f"only {time_s.size} sample(s)")]

    dt = np.diff(time_s)
    median_dt = float(np.median(dt))
    diags.append(
        Diagnostic(
            "info",
            "sampling",
            f"n={time_s.size}, median dt={median_dt:.4f}s, "
            f"min dt={dt.min():.4f}s, max dt={dt.max():.4f}s, "
            f"duration={time_s[-1] - time_s[0]:.2f}s",
        )
    )

    backwards = np.flatnonzero(dt < 0)
    if backwards.size:
        diags.append(
            Diagnostic(
                "warning",
                "time_reset",
                f"time goes backwards at {backwards.size} sample(s), first at "
                f"index {int(backwards[0])} (t={time_s[backwards[0]]:.4f}s)",
            )
        )

    stalled = np.flatnonzero(dt == 0)
    if stalled.size:
        diags.append(
            Diagnostic(
                "warning",
                "duplicate_timestamps",
                f"{stalled.size} sample(s) share a timestamp with the previous "
                f"sample, first at index {int(stalled[0])}",
            )
        )

    if median_dt > 0:
        short = np.flatnonzero((dt > 0) & (dt < cfg.sampling_short_factor * median_dt))
        if short.size:
            worst = int(short[np.argmin(dt[short])])
            diags.append(
                Diagnostic(
                    "info",
                    "short_interval",
                    f"{short.size} interval(s) shorter than "
                    f"{cfg.sampling_short_factor:g}x the median dt; shortest is "
                    f"{dt[worst]:.4f}s at t={time_s[worst]:.4f}s",
                )
            )

        gaps = np.flatnonzero(dt > cfg.sampling_gap_factor * median_dt)
        if gaps.size:
            worst = int(gaps[np.argmax(dt[gaps])])
            diags.append(
                Diagnostic(
                    "warning",
                    "sampling_gap",
                    f"{gaps.size} interval(s) exceed {cfg.sampling_gap_factor}x "
                    f"the median dt; largest is {dt[worst]:.4f}s at t="
                    f"{time_s[worst]:.4f}s",
                )
            )
        # Non-uniform but not gap-sized: worth an info line, not a warning.
        jitter = float(np.max(np.abs(dt - median_dt)))
        if jitter > 0.05 * median_dt:
            diags.append(
                Diagnostic(
                    "info",
                    "nonuniform_sampling",
                    f"dt varies by up to {jitter:.4f}s around the median "
                    f"({100 * jitter / median_dt:.1f}%) -- rates derived from "
                    f"the Time column, not assumed",
                )
            )
    return diags


def check_monotonic(disp_mm: np.ndarray, cfg: AnalysisConfig) -> list[Diagnostic]:
    """Flag (never drop) displacement reversals beyond the noise tolerance."""
    if disp_mm.size < 2:
        return []
    d = np.diff(disp_mm)
    reversals = np.flatnonzero(d < -cfg.displacement_monotonic_tol_mm)
    if not reversals.size:
        return []
    return [
        Diagnostic(
            "warning",
            "non_monotonic_displacement",
            f"{reversals.size} displacement reversal(s) beyond "
            f"{cfg.displacement_monotonic_tol_mm} mm; largest is "
            f"{float(np.min(d)):.4f} mm at index {int(reversals[np.argmin(d[reversals])])}. "
            f"Data kept as-is.",
        )
    ]


def estimate_tare(
    force_N: np.ndarray, contact_index: int, cfg: AnalysisConfig
) -> tuple[float, list[Diagnostic]]:
    """Mean pre-contact load-cell reading, i.e. the zero offset.

    The observed exports hold a constant non-zero force (0 to 2.3 N) from the
    first sample until the specimen is actually touched -- a tare that was not
    cleared, not a real load. It is small next to a 9.5 kN test, but it sits
    directly on the toe region where the modulus is fitted.
    """
    if contact_index <= 0:
        return 0.0, []

    window = force_N[: min(contact_index, max(1, cfg.tare_window_samples))]
    if window.size == 0:
        return 0.0, []

    offset = float(np.mean(window))
    if abs(offset) < 1e-9:
        return 0.0, []

    return offset, [
        Diagnostic(
            "info",
            "tare_offset",
            f"pre-contact load-cell offset of {offset:.3f} N "
            f"({'subtracted' if cfg.tare_correction else 'left in place'}; "
            f"mean of {window.size} sample(s) before contact)",
        )
    ]


def _trailing_hold(
    force_N: np.ndarray, disp_mm: np.ndarray, threshold: float
) -> int:
    """Length of the trailing run where the frame is holding, not loading.

    Proximity to the limit is not enough to call a sample saturated -- a test
    ramping into its limit spends its last samples above any threshold while
    still genuinely loading, and trimming those would discard real data. The
    distinguishing feature of a hold is mechanical: the crosshead has stopped.
    So a sample only counts when it is at the limit *and* its displacement
    increment has collapsed relative to the test's own travel rate.
    """
    if force_N.size < 3 or disp_mm.size != force_N.size:
        return 0

    steps = np.diff(disp_mm, prepend=disp_mm[0])
    typical = float(np.median(np.abs(steps)))
    if typical <= 0:
        return 0

    stalled = np.abs(steps) < 0.1 * typical
    at_limit = force_N >= threshold

    count = 0
    for i in range(len(force_N) - 1, -1, -1):
        if at_limit[i] and stalled[i]:
            count += 1
        else:
            break
    return count


def check_force_limit(
    force_N: np.ndarray, disp_mm: np.ndarray, cfg: AnalysisConfig
) -> tuple[bool, int, list[Diagnostic]]:
    """Decide whether the test was stopped by a load limit, and how.

    Returns `(force_limited, saturated_samples, diagnostics)`. A force-limited
    test is a *truncated* test: it says where the specimen had got to when the
    frame gave up, not where the specimen failed. Densification strain, energy
    absorption and SEA all become lower bounds, so this is flagged loudly
    rather than left for the reader to infer from a suspiciously round peak.

    Two detectors, because they catch different things:
      * an explicit `force_limit_kN` from the test setup, and
      * the configuration-free signal that the file simply ends at its own
        maximum force, which a specimen-driven test does not do.
    """
    diags: list[Diagnostic] = []
    if force_N.size < 2:
        return False, 0, diags

    peak = float(np.nanmax(force_N))
    final = float(force_N[-1])
    limited = False
    saturated = 0

    if cfg.force_limit_kN is not None:
        limit_N = cfg.force_limit_kN * 1000.0
        threshold = limit_N * (1.0 - cfg.force_limit_tolerance_frac)
        # Overshoot is checked first: a peak well above the configured limit
        # means the configuration is wrong, and saying "reached the limit"
        # would be the less useful of the two messages.
        if peak > limit_N * (1.0 + cfg.force_limit_tolerance_frac):
            limited = True
            saturated = _trailing_hold(force_N, disp_mm, threshold)
            diags.append(
                Diagnostic(
                    "warning",
                    "force_limit_exceeded",
                    f"peak force {peak / 1000:.4f} kN exceeds the configured limit "
                    f"of {cfg.force_limit_kN:g} kN -- check that force_limit_kN "
                    f"matches the machine setup",
                )
            )
        elif (force_N >= threshold).any():
            limited = True
            saturated = _trailing_hold(force_N, disp_mm, threshold)
            diags.append(
                Diagnostic(
                    "warning",
                    "force_limit_reached",
                    f"test reached the configured {cfg.force_limit_kN:g} kN load "
                    f"limit (peak {peak / 1000:.4f} kN) and was stopped there. "
                    f"The specimen did not densify, so densification strain, "
                    f"energy absorption and SEA are LOWER BOUNDS, not "
                    f"measurements.",
                )
            )

    if not limited and peak > 0 and final >= cfg.truncation_final_force_frac * peak:
        limited = True
        diags.append(
            Diagnostic(
                "warning",
                "test_truncated_at_peak",
                f"recording ends at the maximum force ({peak / 1000:.4f} kN), so the "
                f"specimen was still taking increasing load when the test stopped. "
                f"Metrics that depend on the far end of the curve -- energy "
                f"absorption, and densification onset if it was not already passed "
                f"-- are lower bounds. Set force_limit_kN to confirm which limit "
                f"stopped it.",
            )
        )

    # Only a genuine hold at the limit is instrument output rather than
    # material response; a clean cut-off is left untouched.
    if saturated < cfg.saturation_min_samples:
        saturated = 0
    elif saturated:
        diags.append(
            Diagnostic(
                "warning",
                "force_saturated",
                f"{saturated} trailing sample(s) held at the load limit while "
                f"the frame stopped; trimmed, as they carry no material response",
            )
        )

    return limited, saturated, diags


def read_raw_csv(path: str | Path, cfg: AnalysisConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """Parse one raw export into `[time_s, disp_mm, force_kN, force_N]`.

    Force is converted to newtons here so every downstream calculation works in
    a single consistent unit system (N, mm, MPa).
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    lines = text.splitlines()
    if not lines:
        raise RawParseError(f"{path} is empty")

    header_idx = locate_header(lines)
    if header_idx + 1 >= len(lines):
        raise RawParseError(f"{path}: header row is not followed by a units row")

    units_cells = _split_row(lines[header_idx + 1])
    validate_units(units_cells, cfg)

    frame = pd.read_csv(
        path,
        skiprows=header_idx + 2,
        header=None,
        names=["time_s", "disp_mm", "force_kN"],
        usecols=[0, 1, 2],
        dtype=str,
        skip_blank_lines=True,
        keep_default_na=False,
    )
    frame = frame[~(frame.apply(lambda r: all(str(v).strip() == "" for v in r), axis=1))]
    if frame.empty:
        raise RawParseError(f"{path}: no data rows found after the units row")

    for col in ("time_s", "disp_mm", "force_kN"):
        cleaned = frame[col].astype(str).str.strip().str.strip('"').str.strip()
        numeric = pd.to_numeric(cleaned, errors="coerce")
        bad = numeric.isna()
        if bad.any():
            first = frame.index[bad][0]
            raise RawParseError(
                f"{path}: column '{col}' has {int(bad.sum())} non-numeric "
                f"value(s); first is {cleaned.loc[first]!r} on data row "
                f"{int(first) + 1}"
            )
        frame[col] = numeric.astype(float)

    frame = frame.reset_index(drop=True)
    frame["force_N"] = frame["force_kN"] * 1000.0
    return frame


def find_contact_index(force_N: np.ndarray, cfg: AnalysisConfig) -> tuple[int, float]:
    """First index where load is real, not load-cell noise.

    Contact is the first sample whose force exceeds the threshold *and* stays
    above it for `preload_sustain_samples` consecutive samples, so a single
    noise spike near zero cannot trigger it.
    """
    if force_N.size == 0:
        raise RawParseError("cannot find contact in an empty force series")

    peak = float(np.nanmax(force_N))
    threshold = max(cfg.preload_threshold_frac * peak, cfg.preload_threshold_min_N)
    n = max(1, int(cfg.preload_sustain_samples))

    above = force_N > threshold
    if above.size >= n:
        # Rolling all(): sustained[i] is True if above[i:i+n] are all True.
        window = np.lib.stride_tricks.sliding_window_view(above, n)
        sustained = np.flatnonzero(window.all(axis=1))
        if sustained.size:
            return int(sustained[0]), threshold
    if above.any():
        return int(np.flatnonzero(above)[0]), threshold
    return 0, threshold


def trim_preload(
    raw: pd.DataFrame, cfg: AnalysisConfig = DEFAULT_CONFIG
) -> tuple[pd.DataFrame, int, float, list[Diagnostic]]:
    """Derive the contact-zeroed series. `raw` is not modified.

    Kept for direct use and backwards compatibility; `load_test` calls the
    fuller `clean_series`, which also handles tare and force saturation.
    """
    trimmed, idx, contact_force, diags, _, _, _ = clean_series(raw, cfg)
    return trimmed, idx, contact_force, diags


def clean_series(
    raw: pd.DataFrame, cfg: AnalysisConfig = DEFAULT_CONFIG
) -> tuple[pd.DataFrame, int, float, list[Diagnostic], float, bool, int]:
    """Build the cleaned series: contact-zeroed, tared, saturation-trimmed.

    Returns `(trimmed, contact_index, contact_force_N, diagnostics,
    tare_offset_N, force_limited, saturated_samples)`. `raw` is not modified.
    """
    force = raw["force_N"].to_numpy(dtype=float)
    idx, threshold = find_contact_index(force, cfg)
    diags: list[Diagnostic] = []

    if idx == 0:
        diags.append(
            Diagnostic(
                "warning",
                "no_preload_detected",
                f"force never sustained above the contact threshold "
                f"({threshold:.2f} N); using the first sample as contact. "
                f"Peak force was {float(np.nanmax(force)):.2f} N.",
            )
        )
    else:
        diags.append(
            Diagnostic(
                "info",
                "preload_trimmed",
                f"contact at index {idx} (t={raw['time_s'].iloc[idx]:.4f}s, "
                f"d={raw['disp_mm'].iloc[idx]:.4f}mm, f={force[idx]:.2f}N); "
                f"threshold {threshold:.2f} N sustained over "
                f"{cfg.preload_sustain_samples} samples; {idx} pre-contact "
                f"sample(s) excluded from the trimmed series",
            )
        )

    tare, tare_diags = estimate_tare(force, idx, cfg)
    diags += tare_diags

    limited, saturated, limit_diags = check_force_limit(
        force, raw["disp_mm"].to_numpy(dtype=float), cfg
    )
    diags += limit_diags

    trimmed = raw.iloc[idx:].copy().reset_index(drop=True)
    if saturated:
        trimmed = trimmed.iloc[: len(trimmed) - saturated].reset_index(drop=True)
    if trimmed.empty:
        raise RawParseError(
            "no samples remain after trimming pre-contact and saturated data"
        )

    trimmed["time_s"] = trimmed["time_s"] - trimmed["time_s"].iloc[0]
    trimmed["disp_mm"] = trimmed["disp_mm"] - trimmed["disp_mm"].iloc[0]
    if cfg.tare_correction and tare != 0.0:
        trimmed["force_N"] = trimmed["force_N"] - tare
        trimmed["force_kN"] = trimmed["force_N"] / 1000.0

    return trimmed, idx, float(force[idx]), diags, tare, limited, saturated


def load_test(path: str | Path, cfg: AnalysisConfig = DEFAULT_CONFIG) -> LoadedTest:
    """Parse, diagnose and clean one raw file."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()

    raw = read_raw_csv(path, cfg)
    diagnostics = _check_boilerplate(lines, locate_header(lines))
    diagnostics += check_sampling(raw["time_s"].to_numpy(dtype=float), cfg)
    diagnostics += check_monotonic(raw["disp_mm"].to_numpy(dtype=float), cfg)

    trimmed, idx, contact_force, trim_diags, tare, limited, saturated = clean_series(
        raw, cfg
    )
    diagnostics += trim_diags

    return LoadedTest(
        path=path,
        raw=raw,
        trimmed=trimmed,
        contact_index=idx,
        contact_force_N=contact_force,
        diagnostics=diagnostics,
        tare_offset_N=tare,
        force_limited=limited,
        saturated_samples=saturated,
    )
