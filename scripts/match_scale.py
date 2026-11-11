#!/usr/bin/env python3
"""Which scale of each baseline behaves most like the reference (plain) set?

    python scripts/match_scale.py                         # settings from analysis_config.json
    python scripts/match_scale.py --window 3 6            # geometric window to check against
    python scripts/match_scale.py --reference family=plain --candidates family=tpms
    python scripts/match_scale.py --properties modulus_MPa plateau_stress_MPa

Reads results/metrics_summary.csv and results/curves/ from the last
run_analysis.py, so it is quick to re-run with other settings. Writes
results/scale_match/ (tables and summary.txt) and results/figures/scale_match/.
run_analysis.py already runs it when scale_match_column is set.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (sys.path side effect)

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.config import load_project_config
from src.pipeline import PipelinePaths
from src.scale_match import analyse, curves_from_dir, summarise, write_outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--scale", help="column holding the scale (default from config, else 'scale')")
    parser.add_argument("--reference", help="KEY=VALUE selecting the reference tests")
    parser.add_argument("--candidates", help="KEY=VALUE selecting the tests searched")
    parser.add_argument("--group-by", help="search each level of this column separately too")
    parser.add_argument("--window", nargs=2, type=float, metavar=("LOW", "HIGH"))
    parser.add_argument("--properties", nargs="+")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    paths = PipelinePaths.from_root(args.root)
    cfg = load_project_config(args.config, args.root)
    overrides = {"scale_match_column": args.scale or cfg.scale_match_column or "scale"}
    for key, value in (("scale_match_reference", args.reference),
                       ("scale_match_candidates", args.candidates),
                       ("scale_match_group_by", args.group_by),
                       ("scale_match_window", tuple(args.window) if args.window else None),
                       ("scale_match_properties", tuple(args.properties) if args.properties else None)):
        if value is not None:
            overrides[key] = value
    cfg = cfg.replace(**overrides)

    if not paths.metrics_csv.exists():
        print(f"error: {paths.metrics_csv} not found. Run: python scripts/run_analysis.py",
              file=sys.stderr)
        return 2
    metrics = pd.read_csv(paths.metrics_csv)
    try:
        result = analyse(metrics, curves_from_dir(paths.curves_dir), cfg)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    table_dir = paths.results_dir / "scale_match"
    figure_dir = paths.figures_dir / "scale_match"
    write_outputs(result, metrics, table_dir, figure_dir, cfg, figures=not args.no_figures)
    print(summarise(result))
    print(f"\n  tables  -> {table_dir}/")
    if not args.no_figures:
        print(f"  figures -> {figure_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
