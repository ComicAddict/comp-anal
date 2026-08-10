#!/usr/bin/env python3
"""Register a raw CSV into data/specimens.csv.

Every raw file has to be registered once with the pattern it belongs to and the
geometry needed to convert force and displacement into stress and strain; from
then on the whole pipeline reads specimens.csv as the single source of truth.

    # what still needs registering?
    python scripts/register_test.py --list

    # walk through every unregistered file, prompting for each field
    python scripts/register_test.py --interactive

    # or register one file non-interactively (scriptable / batch)
    python scripts/register_test.py \
        --raw-file solid_compression_20260717_192209_1_1.csv \
        --pattern-name solid --replicate 1 \
        --side-length-mm 20 --initial-height-mm 20 \
        --material "PLA" --pattern-params '{"kind": "baseline"}'

Pattern identity is never inferred from the filename -- the trailing `_1_1` is
treated as an opaque part of the ID. The test date is the one exception: when
the filename carries a `_YYYYMMDD_` stamp it is offered as the default, and can
be overridden.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (sys.path side effect)

import argparse
import csv
import json
import re
import sys
from pathlib import Path

from src.pipeline import PipelinePaths
from src.specimens import COLUMNS, SpecimenError, unregistered_files

DATE_IN_NAME = re.compile(r"_(\d{4})(\d{2})(\d{2})_")


def date_from_filename(name: str) -> str | None:
    """Extract a `_YYYYMMDD_` stamp as an ISO date, if present."""
    match = DATE_IN_NAME.search(name)
    if not match:
        return None
    year, month, day = match.groups()
    return f"{year}-{month}-{day}"


def prompt(label: str, default: str | None = None, required: bool = False) -> str:
    """Ask for one field, offering a default."""
    suffix = f" [{default}]" if default not in (None, "") else ""
    while True:
        try:
            answer = input(f"  {label}{suffix}: ").strip()
        except EOFError:
            print("\nAborted.", file=sys.stderr)
            raise SystemExit(1)
        if not answer and default is not None:
            answer = str(default)
        if answer or not required:
            return answer
        print("    ^ required")


def collect_interactively(raw_file: str, args: argparse.Namespace) -> dict[str, str]:
    """Prompt for every field not already supplied on the command line."""
    stem = Path(raw_file).stem
    print(f"\nRegistering {raw_file}")

    test_id = args.test_id or prompt("test_id", stem, required=True)
    pattern_name = args.pattern_name or prompt(
        "pattern_name (e.g. gyroid_v2, solid)", required=True
    )
    pattern_params = args.pattern_params or prompt(
        'pattern_params (JSON, e.g. {"cell_mm": 5, "wall_mm": 0.8})', "{}"
    )
    replicate = args.replicate or prompt("replicate (int)", "1", required=True)
    side = args.side_length_mm or prompt("side_length_mm", required=True)
    area = args.cross_section_area_mm2 or prompt(
        "cross_section_area_mm2 (blank = side_length^2)", ""
    )
    height = args.initial_height_mm or prompt("initial_height_mm", side, required=True)
    rel_density = args.relative_density or prompt("relative_density (blank = unknown)", "")
    mass = args.mass_g or prompt("mass_g (blank = SEA reported by volume)", "")
    material = args.material or prompt("material", required=True)
    test_date = args.test_date or prompt("test_date", date_from_filename(raw_file) or "")
    notes = args.notes if args.notes is not None else prompt("notes", "")

    return {
        "test_id": test_id,
        "raw_file": raw_file,
        "pattern_name": pattern_name,
        "pattern_params": pattern_params,
        "replicate": replicate,
        "side_length_mm": side,
        "cross_section_area_mm2": area,
        "initial_height_mm": height,
        "relative_density": rel_density,
        "mass_g": mass,
        "material": material,
        "test_date": test_date,
        "notes": notes,
    }


def collect_from_args(raw_file: str, args: argparse.Namespace) -> dict[str, str]:
    """Build a row from CLI arguments alone, failing on anything missing."""
    missing = [
        name
        for name, value in (
            ("--pattern-name", args.pattern_name),
            ("--replicate", args.replicate),
            ("--side-length-mm", args.side_length_mm),
            ("--material", args.material),
        )
        if value in (None, "")
    ]
    if missing:
        raise SystemExit(
            f"error: missing required argument(s) for {raw_file}: {', '.join(missing)}\n"
            f"       (or re-run with --interactive to be prompted)"
        )

    return {
        "test_id": args.test_id or Path(raw_file).stem,
        "raw_file": raw_file,
        "pattern_name": args.pattern_name,
        "pattern_params": args.pattern_params or "{}",
        "replicate": args.replicate,
        "side_length_mm": args.side_length_mm,
        "cross_section_area_mm2": args.cross_section_area_mm2 or "",
        "initial_height_mm": args.initial_height_mm or args.side_length_mm,
        "relative_density": args.relative_density or "",
        "mass_g": args.mass_g or "",
        "material": args.material,
        "test_date": args.test_date or date_from_filename(raw_file) or "",
        "notes": args.notes or "",
    }


def validate_row(row: dict[str, str]) -> None:
    """Catch bad input at entry time rather than at analysis time."""
    if row["pattern_params"].strip():
        try:
            parsed = json.loads(row["pattern_params"])
        except json.JSONDecodeError as exc:
            raise SpecimenError(f"pattern_params is not valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise SpecimenError("pattern_params must be a JSON object")

    for field in ("replicate", "side_length_mm", "initial_height_mm"):
        try:
            float(row[field])
        except ValueError as exc:
            raise SpecimenError(f"{field} must be numeric, got {row[field]!r}") from exc

    for field in ("cross_section_area_mm2", "relative_density", "mass_g"):
        if row[field].strip():
            try:
                float(row[field])
            except ValueError as exc:
                raise SpecimenError(
                    f"{field} must be numeric or blank, got {row[field]!r}"
                ) from exc


def append_row(specimens_csv: Path, row: dict[str, str]) -> None:
    """Append one row, creating the file with a header when it does not exist."""
    specimens_csv.parent.mkdir(parents=True, exist_ok=True)
    is_new = not specimens_csv.exists() or specimens_csv.stat().st_size == 0
    with specimens_csv.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS))
        if is_new:
            writer.writeheader()
        writer.writerow({col: row.get(col, "") for col in COLUMNS})


def existing_ids(specimens_csv: Path) -> set[str]:
    if not specimens_csv.exists():
        return set()
    with specimens_csv.open(newline="", encoding="utf-8") as handle:
        return {r["test_id"].strip() for r in csv.DictReader(handle) if r.get("test_id")}


def duplicate_raw_files(raw_dir: Path, pending: list[str]) -> set[str]:
    """Files whose measurements duplicate an earlier file in the list.

    The first member of each group is left unmarked -- it is the one to keep.
    A file that cannot be parsed is simply not compared.
    """
    from src.io import RawParseError, read_raw_csv
    from src.pipeline import data_digest

    seen: dict[str, str] = {}
    duplicates: set[str] = set()
    for raw_file in pending:
        try:
            digest = data_digest(read_raw_csv(raw_dir / raw_file))
        except (RawParseError, OSError):
            continue
        if digest in seen:
            duplicates.add(raw_file)
        else:
            seen[digest] = raw_file
    return duplicates


def write_template(paths, pending: list[str], skip: set[str]) -> int:
    """Create a specimens.csv skeleton, one row per unregistered raw file.

    Fills in what can be known without touching the specimen -- test_id and
    test_date from the filename -- and leaves pattern and geometry blank for
    the user. Duplicate re-exports are listed but commented in `notes` rather
    than dropped, so nothing disappears silently.
    """
    if paths.specimens_csv.exists():
        print(
            f"error: {paths.specimens_csv} already exists; refusing to overwrite it.",
            file=sys.stderr,
        )
        return 1

    rows = []
    for raw_file in pending:
        duplicate = raw_file in skip
        rows.append(
            {
                "test_id": Path(raw_file).stem,
                "raw_file": raw_file,
                "pattern_name": "",
                "pattern_params": "{}",
                "replicate": "",
                "side_length_mm": "",
                "cross_section_area_mm2": "",
                "initial_height_mm": "",
                "relative_density": "",
                "mass_g": "",
                "material": "",
                "test_date": date_from_filename(raw_file) or "",
                "notes": (
                    "DUPLICATE of another export in this list -- delete this row "
                    "unless it really is a separate specimen"
                    if duplicate
                    else ""
                ),
            }
        )

    paths.specimens_csv.parent.mkdir(parents=True, exist_ok=True)
    with paths.specimens_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} row(s) to {paths.specimens_csv}\n")
    print("Fill in these columns for each row, then run scripts/run_analysis.py:")
    print("  pattern_name      the internal pattern, or 'solid' for the baseline")
    print("  replicate         repeat number within that pattern")
    print("  side_length_mm    nominal cube side")
    print("  initial_height_mm height along the load axis")
    print("  material          build material")
    print("\nOptional but worth having:")
    print("  mass_g            enables SEA per unit mass instead of per volume")
    print("  relative_density  specimen / bulk material density")
    print("  pattern_params    JSON, e.g. {\"cell_mm\": 5.0, \"wall_mm\": 0.8}")
    print("\ncross_section_area_mm2 can stay blank -- it is derived as side_length^2.")
    if skip:
        print(f"\n{len(skip)} row(s) are marked as duplicate re-exports; see 'notes'.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--root", type=Path, default=None, help="repository root")
    parser.add_argument(
        "--list", action="store_true", help="list raw CSVs not yet registered, then exit"
    )
    parser.add_argument(
        "--interactive", action="store_true", help="prompt for each field"
    )
    parser.add_argument(
        "--template",
        action="store_true",
        help="write a specimens.csv skeleton covering every unregistered file, "
        "with pattern and geometry left blank to fill in by hand",
    )
    parser.add_argument(
        "--raw-file",
        help="raw CSV to register, relative to data/raw/ (default: every unregistered file)",
    )
    parser.add_argument("--test-id", help="unique key (default: filename without extension)")
    parser.add_argument("--pattern-name", help="e.g. gyroid_v2, solid, honeycomb_neg_A")
    parser.add_argument("--pattern-params", help='JSON object, e.g. {"cell_mm": 5}')
    parser.add_argument("--replicate", help="repeat number for this pattern")
    parser.add_argument("--side-length-mm", help="nominal cube side length")
    parser.add_argument(
        "--cross-section-area-mm2", help="loaded area (default: side_length^2)"
    )
    parser.add_argument("--initial-height-mm", help="height along the load axis")
    parser.add_argument("--relative-density", help="specimen / bulk material density")
    parser.add_argument("--mass-g", help="specimen mass; enables SEA per unit mass")
    parser.add_argument("--material", help="build material")
    parser.add_argument("--test-date", help="YYYY-MM-DD (default: parsed from filename)")
    parser.add_argument("--notes", help="free text")
    parser.add_argument(
        "--dry-run", action="store_true", help="show the row without writing it"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = PipelinePaths.from_root(args.root)

    if not paths.raw_dir.exists():
        print(f"error: {paths.raw_dir} does not exist", file=sys.stderr)
        return 1

    pending = [
        str(p.relative_to(paths.raw_dir))
        for p in unregistered_files(paths.raw_dir, paths.specimens_csv)
    ]

    if args.list:
        if not pending:
            print(f"All raw CSVs in {paths.raw_dir} are registered.")
        else:
            print(f"{len(pending)} unregistered file(s) in {paths.raw_dir}:")
            for name in pending:
                print(f"  {name}")
        return 0

    if args.template:
        if not pending:
            print(f"Nothing to do -- every CSV in {paths.raw_dir} is registered.")
            return 0
        return write_template(paths, pending, duplicate_raw_files(paths.raw_dir, pending))

    if args.raw_file:
        targets = [args.raw_file]
        if not (paths.raw_dir / args.raw_file).exists():
            print(
                f"error: {paths.raw_dir / args.raw_file} not found", file=sys.stderr
            )
            return 1
    else:
        targets = pending
        if not targets:
            print(f"Nothing to register -- every CSV in {paths.raw_dir} is known.")
            return 0

    known = existing_ids(paths.specimens_csv)
    written = 0
    for raw_file in targets:
        if args.interactive:
            row = collect_interactively(raw_file, args)
        else:
            row = collect_from_args(raw_file, args)

        if row["test_id"] in known:
            print(
                f"error: test_id {row['test_id']!r} is already in specimens.csv",
                file=sys.stderr,
            )
            return 1

        try:
            validate_row(row)
        except SpecimenError as exc:
            print(f"error: {raw_file}: {exc}", file=sys.stderr)
            return 1

        if args.dry_run:
            print(f"[dry-run] would append: {row}")
        else:
            append_row(paths.specimens_csv, row)
            known.add(row["test_id"])
            written += 1
            print(f"Registered {row['test_id']} -> {paths.specimens_csv}")

    if written:
        print(f"\n{written} test(s) registered. Next: python scripts/run_analysis.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
