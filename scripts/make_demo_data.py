#!/usr/bin/env python3
"""Generate a synthetic 64-test design space, so the pipeline can be run today.

    python scripts/make_demo_data.py              # writes demo/
    python scripts/run_analysis.py --root demo    # analyse it (+ correlation)
    python scripts/run_analysis.py --root demo --force --color-by relative_density

Everything this writes is SYNTHETIC: 64 distinct patterns (cell size x wall
thickness x topology), each printed with its own layer height, print speed and
nozzle temperature, in one of four print batches at one of eight build-plate
positions, with print and test timestamps -- in the instrument's export format
and stopped at the same 9.5 kN load limit as the real frame. Nothing here is measured data; delete
`demo/` once your own tests are in `data/raw/`.

Because the causes are planted, the demo doubles as a check on the
correlation analysis -- this is what it should find:

{truth}
Parameters are written as extra specimens.csv columns, one per parameter --
the layout recommended for a real spreadsheet.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (sys.path side effect)

import argparse
import csv
import json
import shutil
from pathlib import Path

from src.specimens import COLUMNS, OPTIONAL_COLUMNS
from src.synthetic import PLANTED_TRUTH, SyntheticTest, planted_curve_spec, planted_design

__doc__ = __doc__.format(truth=PLANTED_TRUTH)

PARAMETER_COLUMNS = (
    "cell_mm",
    "wall_mm",
    "topology",
    "layer_height_mm",
    "print_speed_mm_s",
    "nozzle_temp_C",
    "print_batch",
    "bed_position",
)


def build(
    root: Path,
    n_tests: int = 64,
    seed: int = 7,
    force_limit_kN: float | None = 9.5,
    clean: bool = False,
) -> Path:
    if clean and root.exists():
        shutil.rmtree(root)
    raw_dir = root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    design = planted_design(n_tests, seed)
    rows: list[dict] = []
    for i, row in design.iterrows():
        name = f"{row['pattern_name']}_compression_20260801_{i:06d}_1_1.csv"
        SyntheticTest(
            name=name,
            curve=planted_curve_spec(row),
            seed=1000 + i,
            slack_mm=0.10 + 0.05 * (i % 5),
            max_strain=0.65,
            force_limit_N=force_limit_kN * 1000.0 if force_limit_kN else None,
        ).write(raw_dir / name)

        record = {
            "test_id": Path(name).stem,
            "raw_file": name,
            "pattern_name": row["pattern_name"],
            "pattern_params": "{}",
            "replicate": 1,
            "side_length_mm": 20.0,
            "cross_section_area_mm2": "",
            "initial_height_mm": 20.0,
            "relative_density": row["relative_density"],
            "mass_g": row["mass_g"],
            "material": row["material"],
            "test_date": row["test_date"],
            "print_date": row["print_date"],
            "notes": "SYNTHETIC demo data -- not a real measurement",
        }
        record.update({column: row[column] for column in PARAMETER_COLUMNS})
        rows.append(record)

    specimens = root / "data" / "specimens.csv"
    with specimens.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(COLUMNS) + list(OPTIONAL_COLUMNS) + list(PARAMETER_COLUMNS)
        )
        writer.writeheader()
        writer.writerows(rows)

    # Same machine setup as the real project, so truncation is flagged.
    config = {"force_limit_kN": force_limit_kN} if force_limit_kN else {}
    (root / "analysis_config.json").write_text(json.dumps(config, indent=2) + "\n")
    return specimens


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=Path("demo"), help="output directory")
    parser.add_argument("--tests", type=int, default=64, help="number of distinct patterns (max 70)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--force-limit-kN", type=float, default=9.5,
                        help="load limit the synthetic frame stops at; 0 for none")
    parser.add_argument("--clean", action="store_true", help="delete --root first")
    args = parser.parse_args(argv)

    specimens = build(
        args.root, args.tests, args.seed,
        args.force_limit_kN if args.force_limit_kN > 0 else None, args.clean,
    )
    count = len(list((args.root / "data" / "raw").glob("*.csv")))
    print(
        f"Wrote {count} synthetic test(s), one pattern each.\n"
        f"  raw       -> {args.root / 'data' / 'raw'}/\n"
        f"  specimens -> {specimens}  (parameters as extra columns)\n\n"
        f"Planted truth -- what the correlation analysis should recover:\n{PLANTED_TRUTH}\n"
        f"Next: python scripts/run_analysis.py --root {args.root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
