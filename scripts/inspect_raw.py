#!/usr/bin/env python3
"""Parse raw CSVs and report what the file actually contains.

Runs the full parse/clean path without needing specimens.csv, so a new export
can be checked the moment it lands -- before any geometry is known.

    python scripts/inspect_raw.py                       # every file in data/raw/
    python scripts/inspect_raw.py --force-limit-kN 9.5  # flag limit-stopped tests
    python scripts/inspect_raw.py --duplicates          # find re-exported tests
    python scripts/inspect_raw.py path/to/one.csv

Reports per file: sample count and duration, sampling regularity, contact point
and pre-load slack, tare offset, peak force and displacement, and whether the
test ran to completion or was stopped by a load limit.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (sys.path side effect)

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

from src.config import AnalysisConfig, load_project_config
from src.io import RawParseError, load_test
from src.pipeline import PipelinePaths


def describe(path: Path, cfg: AnalysisConfig) -> dict | None:
    try:
        loaded = load_test(path, cfg)
    except RawParseError as exc:
        print(f"\n{path.name}\n  PARSE FAILED: {exc}")
        return None

    raw, trimmed = loaded.raw, loaded.trimmed
    time = raw["time_s"].to_numpy(dtype=float)
    dt = np.diff(time)
    force = raw["force_N"].to_numpy(dtype=float)
    disp = raw["disp_mm"].to_numpy(dtype=float)
    rate = float(np.polyfit(time, disp, 1)[0]) * 60.0 if len(time) > 2 else float("nan")

    print(f"\n{path.name}")
    print(
        f"  samples    {len(raw)} raw -> {len(trimmed)} after cleaning"
        f"{f' (-{loaded.saturated_samples} saturated)' if loaded.saturated_samples else ''}"
    )
    print(
        f"  time       0 -> {time[-1]:.2f} s | dt median {np.median(dt):.4f}s "
        f"(min {dt.min():.4f}, max {dt.max():.4f})"
    )
    print(f"  crosshead  {rate:.3f} mm/min (fitted)")
    print(
        f"  contact    index {loaded.contact_index} at {disp[loaded.contact_index]:.4f} mm "
        f"-- {disp[loaded.contact_index] - disp[0]:.4f} mm of slack taken up"
    )
    print(f"  tare       {loaded.tare_offset_N:.3f} N pre-contact offset")
    print(
        f"  peak       {force.max() / 1000:.4f} kN at {disp[int(np.argmax(force))]:.4f} mm"
    )
    print(
        f"  travel     {disp.max():.4f} mm total, {trimmed['disp_mm'].max():.4f} mm after contact"
    )
    print(f"  ended      {'AT LOAD LIMIT (truncated)' if loaded.force_limited else 'normally'}")

    for diag in loaded.diagnostics:
        if diag.level == "warning":
            print(f"  ! {diag.code}: {diag.message}")

    return {
        "path": path,
        "digest": hashlib.sha256(
            raw[["time_s", "disp_mm", "force_kN"]].to_csv(index=False).encode()
        ).hexdigest(),
        "post_contact_travel": float(trimmed["disp_mm"].max()),
        "peak_kN": float(force.max() / 1000),
        "force_limited": loaded.force_limited,
    }


def report_duplicates(rows: list[dict]) -> bool:
    """Group files by the content of their data region."""
    groups: dict[str, list[Path]] = {}
    for row in rows:
        groups.setdefault(row["digest"], []).append(row["path"])

    dupes = {d: paths for d, paths in groups.items() if len(paths) > 1}
    if not dupes:
        print("\nNo duplicate exports: every file holds distinct data.")
        return False

    print(f"\n{'=' * 70}\nDUPLICATE DATA -- these files hold identical measurements:")
    for paths in dupes.values():
        print(f"  {' == '.join(p.name for p in sorted(paths))}")
    print(
        "\n  Register only ONE of each group. Registering both would count a\n"
        "  single physical test as two replicates and fabricate the spread\n"
        "  between them."
    )
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("files", nargs="*", type=Path, help="CSVs (default: data/raw/*.csv)")
    parser.add_argument("--root", type=Path, default=None, help="repository root")
    parser.add_argument(
        "--force-limit-kN",
        type=float,
        help="the load limit the tests were set to stop at, e.g. 9.5",
    )
    parser.add_argument(
        "--duplicates", action="store_true", help="also report files with identical data"
    )
    parser.add_argument(
        "--no-tare", action="store_true", help="do not subtract the pre-contact offset"
    )
    args = parser.parse_args(argv)

    cfg = load_project_config(None, args.root)
    if args.force_limit_kN is not None:
        cfg = cfg.replace(force_limit_kN=args.force_limit_kN)
    if args.no_tare:
        cfg = cfg.replace(tare_correction=False)

    if args.files:
        targets = sorted(args.files)
    else:
        raw_dir = PipelinePaths.from_root(args.root).raw_dir
        targets = sorted(raw_dir.glob("*.csv"))
        if not targets:
            print(f"No CSVs found in {raw_dir}", file=sys.stderr)
            return 1

    rows = [row for path in targets if (row := describe(path, cfg)) is not None]
    if not rows:
        return 1

    limited = [r for r in rows if r["force_limited"]]
    print(f"\n{'=' * 70}\n{len(rows)} file(s) parsed.")
    if limited:
        travel = [r["post_contact_travel"] for r in limited]
        print(
            f"{len(limited)} of {len(rows)} stopped at a load limit, reaching "
            f"{min(travel):.2f}-{max(travel):.2f} mm of post-contact travel.\n"
            f"Densification was not reached in these, so densification strain, "
            f"energy absorption\nand SEA will be lower bounds. Modulus, plateau "
            f"stress and first-peak metrics are\nunaffected where the curve "
            f"covers them."
        )

    if args.duplicates:
        report_duplicates(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
