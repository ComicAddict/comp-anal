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
    """Derive the contact-zeroed series. `raw` is not modified."""
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

    trimmed = raw.iloc[idx:].copy().reset_index(drop=True)
    trimmed["time_s"] = trimmed["time_s"] - trimmed["time_s"].iloc[0]
    trimmed["disp_mm"] = trimmed["disp_mm"] - trimmed["disp_mm"].iloc[0]
    return trimmed, idx, float(force[idx]), diags


def load_test(path: str | Path, cfg: AnalysisConfig = DEFAULT_CONFIG) -> LoadedTest:
    """Parse, diagnose and clean one raw file."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()

    raw = read_raw_csv(path, cfg)
    diagnostics = _check_boilerplate(lines, locate_header(lines))
    diagnostics += check_sampling(raw["time_s"].to_numpy(dtype=float), cfg)
    diagnostics += check_monotonic(raw["disp_mm"].to_numpy(dtype=float), cfg)

    trimmed, idx, contact_force, trim_diags = trim_preload(raw, cfg)
    diagnostics += trim_diags

    return LoadedTest(
        path=path,
        raw=raw,
        trimmed=trimmed,
        contact_index=idx,
        contact_force_N=contact_force,
        diagnostics=diagnostics,
    )
