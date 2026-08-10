#!/usr/bin/env python3
"""Generate a synthetic demo design space, so the pipeline can be run today.

    python scripts/make_demo_data.py            # writes demo/
    python scripts/run_analysis.py --root demo  # analyse it
    open demo/results/figures/*.png

Everything this writes is SYNTHETIC -- five made-up patterns with three
replicates each, in the instrument's export format. It exists to show the
pipeline working end to end and to give the figures something to plot before
real tests are registered. Nothing here is measured data; delete `demo/` once
your own tests are in `data/raw/`.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (sys.path side effect)

import argparse
import csv
import json
import shutil
from pathlib import Path

from src.specimens import COLUMNS
from src.synthetic import FoamCurveSpec, SyntheticTest

#: A small design space: a solid baseline plus four cut-out patterns, two of
#: them a parameter sweep over cell size so the metric-vs-parameter view has
#: something to show.
PATTERNS: dict[str, dict] = {
    "solid": {
        "params": {"kind": "baseline"},
        "relative_density": 1.00,
        "mass_g": 9.9,
        "curve": {},
    },
    "gyroid_v2": {
        "params": {"cell_mm": 5.0, "wall_mm": 0.8},
        "relative_density": 0.32,
        "mass_g": 3.2,
        "curve": {
            "modulus_MPa": 24.0,
            "peak_stress_MPa": None,
            "plateau_stress_MPa": 1.05,
            "plateau_slope_MPa": 0.55,
            "densification_strain": 0.50,
            "densification_rise_MPa": 9.0,
        },
    },
    "gyroid_v2_coarse": {
        "params": {"cell_mm": 8.0, "wall_mm": 0.8},
        "relative_density": 0.24,
        "mass_g": 2.4,
        "curve": {
            "modulus_MPa": 14.0,
            "peak_stress_MPa": None,
            "plateau_stress_MPa": 0.62,
            "plateau_slope_MPa": 0.35,
            "densification_strain": 0.55,
            "densification_rise_MPa": 7.5,
        },
    },
    "honeycomb_neg_A": {
        "params": {"cell_mm": 6.0, "wall_mm": 1.0},
        "relative_density": 0.38,
        "mass_g": 3.8,
        "curve": {
            "modulus_MPa": 78.0,
            "peak_strain": 0.030,
            "peak_stress_MPa": 2.34,
            "plateau_stress_MPa": 1.55,
            "softening_end_strain": 0.075,
            "densification_strain": 0.44,
            "densification_rise_MPa": 11.0,
        },
    },
    "auxetic_reentrant": {
        "params": {"cell_mm": 6.0, "wall_mm": 0.9, "angle_deg": 70},
        "relative_density": 0.29,
        "mass_g": 2.9,
        "curve": {
            "modulus_MPa": 41.0,
            "peak_strain": 0.055,
            "peak_stress_MPa": 2.26,
            "plateau_stress_MPa": 1.35,
            "plateau_slope_MPa": 1.3,
            "densification_strain": 0.40,
            "densification_rise_MPa": 13.0,
        },
    },
}


def build(root: Path, replicates: int = 3, clean: bool = False) -> Path:
    raw_dir = root / "data" / "raw"
    if clean and root.exists():
        shutil.rmtree(root)
    raw_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for p_index, (pattern, config) in enumerate(PATTERNS.items()):
        for rep in range(1, replicates + 1):
            name = f"{pattern}_compression_2026071{p_index}_1{rep:02d}500_1_{rep}.csv"
            # Replicates differ slightly, as real repeats of one pattern do.
            jitter = 1.0 + 0.04 * (rep - (replicates + 1) / 2)
            curve = dict(config["curve"])
            for key in ("modulus_MPa", "plateau_stress_MPa", "peak_stress_MPa"):
                if curve.get(key):
                    curve[key] = round(curve[key] * jitter, 4)
            if "modulus_MPa" not in curve:
                curve["modulus_MPa"] = round(60.0 * jitter, 4)
                curve["peak_stress_MPa"] = round(2.70 * jitter, 4)
                curve["plateau_stress_MPa"] = round(2.15 * jitter, 4)

            test = SyntheticTest(
                name=name,
                curve=FoamCurveSpec(**curve),
                seed=100 * p_index + rep,
                slack_mm=0.40 + 0.05 * rep,
            )
            test.write(raw_dir / name)

            rows.append(
                {
                    "test_id": Path(name).stem,
                    "raw_file": name,
                    "pattern_name": pattern,
                    "pattern_params": json.dumps(config["params"], sort_keys=True),
                    "replicate": rep,
                    "side_length_mm": test.side_length_mm,
                    "cross_section_area_mm2": "",
                    "initial_height_mm": test.initial_height_mm,
                    "relative_density": config["relative_density"],
                    "mass_g": round(config["mass_g"] * jitter, 2),
                    "material": "PLA",
                    "test_date": f"2026-07-1{p_index}",
                    "notes": "SYNTHETIC demo data -- not a real measurement",
                }
            )

    specimens = root / "data" / "specimens.csv"
    with specimens.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS))
        writer.writeheader()
        writer.writerows(rows)

    return specimens


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=Path("demo"), help="output directory")
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--clean", action="store_true", help="delete --root first")
    args = parser.parse_args(argv)

    specimens = build(args.root, args.replicates, args.clean)
    count = len(list((args.root / "data" / "raw").glob("*.csv")))
    print(
        f"Wrote {count} synthetic test(s) across {len(PATTERNS)} pattern(s).\n"
        f"  raw       -> {args.root / 'data' / 'raw'}/\n"
        f"  specimens -> {specimens}\n\n"
        f"Next: python scripts/run_analysis.py --root {args.root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
