#!/usr/bin/env python3
"""Correlate design and manufacturing parameters with measured properties.

    python scripts/correlate.py                               # every parameter x property
    python scripts/correlate.py --list                        # what can be analysed, and why not
    python scripts/correlate.py --parameters cell_mm wall_mm layer_height_mm
    python scripts/correlate.py --properties modulus_MPa stress_at_20pct_MPa
    python scripts/correlate.py --where material=PLA          # within one level
    python scripts/correlate.py --where topology=gyroid --where layer_height_mm=0.2

Reads results/metrics_summary.csv, so run scripts/run_analysis.py first. (That
also runs this analysis with the defaults; use this script to slice it.)

Writes tables and summary.txt to results/correlation/ and figures to
results/figures/correlation/. A --where run goes to its own subfolder, e.g.
results/correlation/where_material-PLA/, so it never overwrites the full one.

Parameters are every param_* column (from pattern_params JSON and from any
extra columns in specimens.csv) plus relative density, specimen size,
material and pattern name. See src/correlation.py for the statistics.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (sys.path side effect)

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

from src.config import load_project_config
from src.correlation import (
    analyse,
    discover_parameters,
    property_data,
    summarise,
    write_outputs,
)
from src.envelope import display_name, resolve_column
from src.pipeline import PipelinePaths


def apply_where(metrics: pd.DataFrame, clauses: list[str]) -> tuple[pd.DataFrame, str]:
    """Filter rows by KEY=VALUE clauses; return the subset and a folder tag."""
    tags = []
    for clause in clauses:
        if "=" not in clause:
            raise SystemExit(f"error: --where expects KEY=VALUE, got {clause!r}")
        key, value = (part.strip() for part in clause.split("=", 1))
        column = resolve_column(metrics, key)
        if column is None:
            raise SystemExit(f"error: --where {key!r} matches no column")
        series = metrics[column]
        numeric = pd.to_numeric(series, errors="coerce")
        try:
            target = float(value)
            mask = numeric == target
        except ValueError:
            mask = series.astype(str).str.strip() == value
        if not mask.any():
            levels = sorted(series.dropna().astype(str).unique())[:12]
            raise SystemExit(
                f"error: no tests with {key} = {value!r}; values present: {levels}"
            )
        metrics = metrics[mask]
        tags.append(f"{display_name(column)}-{value}")
    tag = re.sub(r"[^0-9A-Za-z_.-]+", "_", "where_" + "_".join(tags)) if tags else ""
    return metrics.reset_index(drop=True), tag


def list_inputs(metrics: pd.DataFrame, cfg) -> None:
    print(f"{len(metrics)} tests in the metrics table.\n\nParameters:")
    for info in discover_parameters(metrics, cfg):
        status = "analysed" if info.included else f"skipped -- {info.reason}"
        levels = f" levels: {', '.join(info.levels)}" if info.levels else ""
        print(f"  {info.name:<24} {info.kind:<12} n={info.n:<4} {status}{levels}")

    print("\nProperties (default set; pass --properties to choose others):")
    props = property_data(metrics, cfg.correlation_properties, cfg)
    for prop in cfg.correlation_properties:
        if prop in props.values.columns:
            n = int(props.values[prop].notna().sum())
            censored = props.n_censored.get(prop, 0)
            extra = f", {censored} lower bounds excluded" if censored else ""
            print(f"  {prop:<26} n={n}{extra}")
        else:
            print(f"  {prop:<26} skipped -- {props.skipped.get(prop, 'unavailable')}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=None, help="repository root")
    parser.add_argument("--config", type=Path, help="JSON file of config overrides")
    parser.add_argument("--metrics", type=Path, help="metrics CSV (default: results/metrics_summary.csv)")
    parser.add_argument("--where", action="append", default=[], metavar="KEY=VALUE",
                        help="analyse only matching tests; repeatable")
    parser.add_argument("--parameters", nargs="+", metavar="NAME",
                        help="parameters to include (default: every one that varies)")
    parser.add_argument("--properties", nargs="+", metavar="COLUMN",
                        help="properties to test (default: correlation_properties in the config)")
    parser.add_argument("--alpha", type=float, help="FDR level (default 0.05)")
    parser.add_argument("--min-n", type=int, help="fewest tests per pair (default 8)")
    parser.add_argument("--include-censored", action="store_true",
                        help="keep lower-bound values from load-limited tests")
    parser.add_argument("--no-figures", action="store_true", help="tables and summary only")
    parser.add_argument("--list", action="store_true", help="list what can be analysed, then exit")
    args = parser.parse_args(argv)

    paths = PipelinePaths.from_root(args.root)
    cfg = load_project_config(args.config, args.root)
    overrides = {}
    if args.alpha is not None:
        overrides["correlation_fdr_alpha"] = args.alpha
    if args.min_n is not None:
        overrides["correlation_min_n"] = args.min_n
    if args.include_censored:
        overrides["correlation_include_censored"] = True
    if overrides:
        cfg = cfg.replace(**overrides)

    metrics_path = args.metrics or paths.metrics_csv
    if not metrics_path.exists():
        print(
            f"error: {metrics_path} not found. Run: python scripts/run_analysis.py",
            file=sys.stderr,
        )
        return 2
    metrics = pd.read_csv(metrics_path)
    metrics, tag = apply_where(metrics, args.where)

    if args.list:
        list_inputs(metrics, cfg)
        return 0

    try:
        result = analyse(metrics, cfg, properties=args.properties, parameters=args.parameters)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    table_dir = paths.correlation_dir / tag if tag else paths.correlation_dir
    figure_dir = paths.correlation_figures_dir / tag if tag else paths.correlation_figures_dir
    written = write_outputs(result, table_dir, figure_dir, cfg, figures=not args.no_figures)

    if tag:
        print(f"Subset: {', '.join(args.where)}  ({len(metrics)} tests)\n")
    print(summarise(result))
    print(f"\nWrote {len(written)} file(s):")
    print(f"  tables  -> {table_dir}/")
    if not args.no_figures:
        print(f"  figures -> {figure_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
