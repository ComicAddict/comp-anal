#!/usr/bin/env python3
"""Regenerate data/specimens.csv from the metadata sheets.

    python scripts/import_metadata.py --check    # report only, write nothing
    python scripts/import_metadata.py            # write data/specimens.csv
    python scripts/run_analysis.py               # then analyse as usual

Which sheets, how their columns map, and where each series' raw exports live
are set once in metadata_sources.json. Rows already in specimens.csv whose
test_id does not come from a source (tests registered by hand) are kept.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (sys.path side effect)

import argparse
import sys
from pathlib import Path

from src.metadata_import import CONFIG_FILENAME, build_specimens
from src.specimens import SpecimenError, read_specimens_frame, validate_frame


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--config", type=Path, help=f"default: <root>/{CONFIG_FILENAME}")
    parser.add_argument("--check", action="store_true", help="report without writing")
    parser.add_argument(
        "--drop-manual", action="store_true",
        help="do not keep hand-registered rows that no source produces",
    )
    args = parser.parse_args(argv)

    root = args.root.resolve()
    specimens = root / "data" / "specimens.csv"
    existing = None
    if specimens.exists() and not args.drop_manual:
        existing = read_specimens_frame(specimens)

    try:
        frame, reports, kept = build_specimens(args.config or root / CONFIG_FILENAME, root, existing)
        if not frame.empty:
            validate_frame(frame.copy(), root / "data" / "raw")
    except SpecimenError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    for report in reports:
        print("\n".join(report.lines()) + "\n")
    if kept:
        print(f"Kept {len(kept)} hand-registered row(s): {', '.join(kept)}\n")
    if args.check:
        print(f"{len(frame)} specimen row(s) would be written (--check: nothing written).")
        return 0
    if frame.empty:
        print("Nothing to write: no sheet row has a raw file yet.")
        return 1
    frame.to_csv(specimens, index=False)
    print(f"Wrote {len(frame)} row(s) -> {specimens}\nNext: python scripts/run_analysis.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
