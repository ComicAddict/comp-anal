#!/usr/bin/env python3
"""End-to-end analysis: raw CSVs -> results/metrics_summary.csv (+ figures).

    python scripts/run_analysis.py                 # analyse + regenerate figures
    python scripts/run_analysis.py --no-figures    # metrics only
    python scripts/run_analysis.py --force         # ignore the change detector
    python scripts/run_analysis.py --watch         # re-run whenever inputs change

Outputs land in results/:
    metrics_summary.csv   one row per test, joined with its pattern metadata
    diagnostics.csv       every warning raised while parsing and analysing
    curves/<test_id>.csv  the cleaned stress-strain curve per test
    figures/              envelope and property-space plots

Analysis parameters (plateau window, densification method, modulus search
range, ...) come from src/config.py and can be overridden with --config or the
individual flags below.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (sys.path side effect)

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from src.config import DEFAULT_CONFIG, AnalysisConfig
from src.pipeline import PipelinePaths, input_fingerprint, is_stale, run_pipeline
from src.specimens import SpecimenError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=None, help="repository root")
    parser.add_argument("--config", type=Path, help="JSON file of config overrides")
    parser.add_argument(
        "--force", action="store_true", help="re-run even when nothing changed"
    )
    parser.add_argument("--no-figures", action="store_true", help="skip plotting")
    parser.add_argument(
        "--watch",
        action="store_true",
        help="poll for changes to data/raw/ and specimens.csv, re-running on each change",
    )
    parser.add_argument(
        "--interval", type=float, default=5.0, help="--watch poll interval in seconds"
    )
    parser.add_argument(
        "--strain-max",
        type=float,
        help="cap the common strain grid instead of using the shortest test",
    )

    group = parser.add_argument_group("analysis parameter overrides")
    group.add_argument("--plateau-lo", type=float, help="plateau window start strain (0.20)")
    group.add_argument("--plateau-hi", type=float, help="plateau window end strain (0.30)")
    group.add_argument(
        "--densification-method",
        choices=("energy_efficiency", "tangent", "iso"),
        help="densification onset criterion (energy_efficiency)",
    )
    group.add_argument(
        "--modulus-max-strain", type=float, help="upper strain for the modulus search (0.10)"
    )
    group.add_argument(
        "--toe-compensation",
        action="store_true",
        help="shift the strain axis so the fitted elastic line passes through the origin",
    )
    return parser


def resolve_config(args: argparse.Namespace) -> AnalysisConfig:
    cfg = AnalysisConfig.from_json(args.config) if args.config else DEFAULT_CONFIG
    overrides = {}
    if args.plateau_lo is not None:
        overrides["plateau_strain_lo"] = args.plateau_lo
    if args.plateau_hi is not None:
        overrides["plateau_strain_hi"] = args.plateau_hi
    if args.densification_method:
        overrides["densification_method"] = args.densification_method
    if args.modulus_max_strain is not None:
        overrides["modulus_search_max_strain"] = args.modulus_max_strain
    if args.toe_compensation:
        overrides["toe_compensation"] = True
    return cfg.replace(**overrides) if overrides else cfg


def summarise(result, paths: PipelinePaths) -> None:
    metrics = result.metrics
    if result.skipped:
        print(f"Inputs unchanged -- reusing {paths.metrics_csv} (use --force to re-run).")
    else:
        print(f"Analysed {len(metrics)} test(s) across "
              f"{metrics['pattern_name'].nunique()} pattern(s).")
        print(f"  metrics     -> {paths.metrics_csv}")
        print(f"  diagnostics -> {paths.diagnostics_csv}")
        print(f"  curves      -> {paths.curves_dir}/")
        for figure in result.figures:
            print(f"  figure      -> {figure}")

    warnings = result.diagnostics
    if len(warnings) and "level" in warnings.columns:
        flagged = warnings[warnings["level"].isin(["warning", "error"])]
        if len(flagged):
            print(f"\n{len(flagged)} warning(s)/error(s) -- see {paths.diagnostics_csv}:")
            for code, block in flagged.groupby("code"):
                tests = ", ".join(sorted(set(block["test_id"].astype(str)))[:5])
                more = "" if block["test_id"].nunique() <= 5 else ", ..."
                print(f"  {code} ({len(block)}): {tests}{more}")

    if not result.skipped and len(metrics):
        columns = [
            c
            for c in (
                "test_id", "pattern_name", "modulus_MPa", "plateau_stress_MPa",
                "densification_strain", "energy_absorption_MJ_m3",
            )
            if c in metrics.columns
        ]
        print("\nSummary:")
        with pd.option_context("display.width", 160, "display.max_columns", None):
            print(metrics[columns].to_string(index=False, float_format=lambda v: f"{v:.4g}"))


def run_once(paths: PipelinePaths, args: argparse.Namespace, cfg: AnalysisConfig) -> int:
    try:
        result = run_pipeline(
            paths,
            cfg,
            force=args.force,
            figures=not args.no_figures,
            strain_max=args.strain_max,
        )
    except SpecimenError as exc:
        print(f"\nspecimens.csv problem:\n{exc}", file=sys.stderr)
        return 2
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1
    summarise(result, paths)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = PipelinePaths.from_root(args.root)
    cfg = resolve_config(args)

    if not paths.specimens_csv.exists():
        print(
            f"error: {paths.specimens_csv} not found.\n"
            f"       Register your tests first: python scripts/register_test.py --interactive",
            file=sys.stderr,
        )
        return 2

    status = run_once(paths, args, cfg)
    if not args.watch:
        return status

    print(f"\nWatching {paths.raw_dir} and {paths.specimens_csv} "
          f"(every {args.interval:g}s). Ctrl-C to stop.")
    args.force = False
    last = input_fingerprint(paths, cfg)
    try:
        while True:
            time.sleep(args.interval)
            current = input_fingerprint(paths, cfg)
            if current != last or is_stale(paths, cfg):
                print(f"\n--- change detected at {time.strftime('%H:%M:%S')} ---")
                run_once(paths, args, cfg)
                last = input_fingerprint(paths, cfg)
    except KeyboardInterrupt:
        print("\nStopped watching.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
