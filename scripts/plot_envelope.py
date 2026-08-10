#!/usr/bin/env python3
"""Regenerate the design-space figures from an existing metrics summary.

    python scripts/plot_envelope.py                       # both views, defaults
    python scripts/plot_envelope.py --pair modulus_MPa plateau_stress_MPa
    python scripts/plot_envelope.py --param cell_mm       # metric vs swept parameter
    python scripts/plot_envelope.py --watch               # redraw on any change

Reads results/metrics_summary.csv and results/curves/, so it does not re-parse
the raw files. If those are missing or stale, run scripts/run_analysis.py first
(or pass --run-analysis to do it here).
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (sys.path side effect)

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from src.config import AnalysisConfig, load_project_config
from src.pipeline import (
    DEFAULT_PAIRS,
    LABELS,
    PipelinePaths,
    input_fingerprint,
    is_stale,
    load_curves_from_disk,
    make_figures,
    run_pipeline,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=None, help="repository root")
    parser.add_argument("--config", type=Path, help="JSON file of config overrides")
    parser.add_argument(
        "--pair",
        nargs=2,
        action="append",
        metavar=("X_METRIC", "Y_METRIC"),
        help="metric pair to scatter; repeatable (default: a standard set)",
    )
    parser.add_argument(
        "--param",
        action="append",
        help="numeric pattern_params key to plot metrics against; repeatable",
    )
    parser.add_argument(
        "--strain-max",
        type=float,
        help="cap the common strain grid instead of using the shortest test",
    )
    parser.add_argument(
        "--no-per-pattern-bands",
        action="store_true",
        help="draw only the overall envelope, not per-pattern replicate bands",
    )
    parser.add_argument(
        "--run-analysis",
        action="store_true",
        help="run the analysis first if the metrics summary is missing or stale",
    )
    parser.add_argument(
        "--watch", action="store_true", help="poll for changes and redraw"
    )
    parser.add_argument(
        "--interval", type=float, default=5.0, help="--watch poll interval in seconds"
    )
    parser.add_argument("--list-metrics", action="store_true", help="list plottable columns")
    return parser


def resolve_config(args: argparse.Namespace) -> AnalysisConfig:
    cfg = load_project_config(args.config, args.root)
    if args.no_per_pattern_bands:
        cfg = cfg.replace(envelope_per_pattern_bands=False)
    return cfg


def draw(paths: PipelinePaths, args: argparse.Namespace, cfg: AnalysisConfig) -> int:
    if args.run_analysis and is_stale(paths, cfg):
        print("Metrics are stale -- running the analysis first.")
        run_pipeline(paths, cfg, force=False, figures=False)

    if not paths.metrics_csv.exists():
        print(
            f"error: {paths.metrics_csv} not found.\n"
            f"       Run: python scripts/run_analysis.py  (or pass --run-analysis)",
            file=sys.stderr,
        )
        return 2

    metrics = pd.read_csv(paths.metrics_csv)
    if metrics.empty:
        print(f"error: {paths.metrics_csv} has no rows.", file=sys.stderr)
        return 1

    if args.list_metrics:
        numeric = metrics.select_dtypes("number").columns.tolist()
        print("Plottable numeric columns:")
        for column in numeric:
            print(f"  {column}")
        params = [c for c in metrics.columns if c.startswith("param_")]
        if params:
            print("Numeric pattern_params:")
            for column in params:
                print(f"  {column[len('param_'):]}")
        return 0

    if is_stale(paths, cfg) and not args.run_analysis:
        print(
            "note: inputs have changed since the last analysis -- these figures may "
            "be out of date. Re-run scripts/run_analysis.py or pass --run-analysis."
        )

    curves = load_curves_from_disk(paths, metrics)
    if not curves:
        print(
            f"error: no cleaned curves in {paths.curves_dir}. Run scripts/run_analysis.py.",
            file=sys.stderr,
        )
        return 1

    pairs = tuple(tuple(p) for p in args.pair) if args.pair else DEFAULT_PAIRS
    for x, y in pairs:
        for column in (x, y):
            if column not in metrics.columns:
                print(
                    f"error: metric {column!r} is not a column in "
                    f"{paths.metrics_csv.name}. Use --list-metrics to see the options.",
                    file=sys.stderr,
                )
                return 1

    figures = make_figures(
        paths, metrics, curves, cfg, pairs=pairs, strain_max=args.strain_max
    )

    if args.param:
        available = {c[len("param_") :] for c in metrics.columns if c.startswith("param_")}
        for param in args.param:
            if param not in available:
                print(
                    f"warning: no numeric pattern_params key {param!r} "
                    f"(available: {sorted(available) or 'none'})",
                    file=sys.stderr,
                )

    print(f"Wrote {len(figures)} figure(s) to {paths.figures_dir}/")
    for figure in figures:
        print(f"  {figure.name}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = PipelinePaths.from_root(args.root)
    cfg = resolve_config(args)

    status = draw(paths, args, cfg)
    if not args.watch or status != 0:
        return status

    print(f"\nWatching for changes (every {args.interval:g}s). Ctrl-C to stop.")
    args.run_analysis = True
    last = input_fingerprint(paths, cfg)
    try:
        while True:
            time.sleep(args.interval)
            current = input_fingerprint(paths, cfg)
            if current != last:
                print(f"\n--- change detected at {time.strftime('%H:%M:%S')} ---")
                draw(paths, args, cfg)
                last = input_fingerprint(paths, cfg)
    except KeyboardInterrupt:
        print("\nStopped watching.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
