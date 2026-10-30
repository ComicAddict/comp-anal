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

from src.config import AnalysisConfig, load_project_config
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
    parser.add_argument(
        "--color-by",
        metavar="PARAMETER",
        help="colour curves and points by a parameter (e.g. relative_density, cell_mm)",
    )
    parser.add_argument(
        "--no-correlation",
        action="store_true",
        help="skip the parameter-property correlation analysis",
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
        "--force-limit-kN",
        type=float,
        help="load limit the tests were set to stop at (e.g. 9.5). Tests that "
        "terminate against it are flagged as truncated, since densification "
        "was never reached",
    )
    group.add_argument(
        "--no-tare",
        action="store_true",
        help="keep the pre-contact load-cell offset instead of subtracting it",
    )
    group.add_argument(
        "--toe-compensation",
        action="store_true",
        help="shift the strain axis so the fitted elastic line passes through the origin",
    )
    return parser


def resolve_config(args: argparse.Namespace) -> AnalysisConfig:
    cfg = load_project_config(args.config, args.root)
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
    if args.force_limit_kN is not None:
        overrides["force_limit_kN"] = args.force_limit_kN
    if args.no_tare:
        overrides["tare_correction"] = False
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

    # Truncation changes how the numbers may be read, so it is stated plainly
    # rather than left inside the diagnostics file.
    if "ended_at_force_limit" in metrics.columns:
        truncated = metrics[metrics["ended_at_force_limit"].astype(bool)]
        if len(truncated):
            print(
                f"\n{len(truncated)} of {len(metrics)} test(s) were still loading "
                f"when recording stopped (ended at peak force / load limit)."
            )
            # Ending at peak load and failing to densify are different problems:
            # a test can run past densification and still be cut off climbing.
            no_dens = truncated
            if "densification_at_end_of_data" in truncated.columns:
                no_dens = truncated[
                    truncated["densification_at_end_of_data"].astype(bool)
                    | truncated["densification_strain"].isna()
                ]
            densified = len(truncated) - len(no_dens)
            if len(no_dens):
                print(
                    f"  {len(no_dens)} of those never reached densification, so "
                    f"densification_strain,\n"
                    f"  energy_absorption_* and SEA are LOWER BOUNDS for them "
                    f"(flagged, and left out of the correlation)."
                )
            if densified:
                lead = "All of them" if not len(no_dens) else f"The other {densified}"
                print(
                    f"  {lead} reached densification before stopping; only energy "
                    f"beyond that point is cut off."
                )
            print(
                "  Modulus, plateau stress and first-peak metrics are unaffected "
                "where the curve reaches them."
            )

    if not result.skipped and len(metrics):
        if len(metrics) <= 20:
            print("\nSummary  (≥ = lower bound: the test stopped before densifying)")
            with pd.option_context("display.width", 200, "display.max_columns", None):
                print(summary_table(metrics).to_string(index=False))
        else:
            print(f"\n{len(metrics)} tests -- see {paths.metrics_csv} for the full table.")

    if result.correlation is not None:
        from src.correlation import summarise as correlation_summary

        print("\n" + correlation_summary(result.correlation))
        print(f"\n  tables  -> {paths.correlation_dir}/")
        print(f"  figures -> {paths.correlation_figures_dir}/")
    elif not result.skipped:
        explain_no_correlation(metrics, paths)


def summary_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """The headline numbers, truncation-safe ones first, bounds marked."""
    def fmt(value, bound=False):
        if pd.isna(value):
            return "–"
        return ("≥" if bound else "") + f"{value:.4g}"

    def flag(column):
        return metrics[column].astype(str).str.lower().eq("true") if column in metrics else False

    table = pd.DataFrame({"test_id": metrics["test_id"], "pattern": metrics["pattern_name"]})
    for column, label in (
        ("modulus_MPa", "E (MPa)"),
        ("plateau_stress_MPa", "plateau σ"),
        ("stress_at_10pct_MPa", "σ@10%"),
        ("stress_at_20pct_MPa", "σ@20%"),
        ("energy_to_20pct_MJ_m3", "W@20% (MJ/m³)"),
    ):
        if column in metrics:
            table[label] = [fmt(v) for v in metrics[column]]
    if "densification_strain" in metrics:
        bounds = flag("densification_censored")
        table["ε_D"] = [fmt(v, b) for v, b in zip(metrics["densification_strain"], bounds)]
    if "energy_absorption_MJ_m3" in metrics:
        bounds = flag("energy_censored")
        table["W@ε_D"] = [fmt(v, b) for v, b in zip(metrics["energy_absorption_MJ_m3"], bounds)]
    return table


def explain_no_correlation(metrics: pd.DataFrame, paths: PipelinePaths) -> None:
    """Say exactly why there is nothing to correlate, and what would fix it."""
    from src.correlation import discover_parameters
    from src.specimens import COLUMNS

    print("\nNo parameter varies across the tests yet, so no correlation analysis.")
    try:
        header = pd.read_csv(paths.specimens_csv, dtype=str, nrows=0).columns
        frame = pd.read_csv(paths.specimens_csv, dtype=str)
    except (OSError, ValueError):
        return
    # Optional core columns (print_date) and user-added parameter columns are
    # exactly the ones outside the required schema.
    blank = [
        c for c in header
        if c not in COLUMNS and frame[c].fillna("").str.strip().eq("").all()
    ]
    if blank:
        print(f"  Columns with no values yet: {', '.join(blank)}")
    constant = [p.name for p in discover_parameters(metrics) if p.reason == "the same value for every test"]
    if constant:
        print(f"  Same for every test: {', '.join(constant)}")
    print(
        "  Fill those in (or add a column per design/manufacturing parameter) and re-run;\n"
        "  see data/specimens.example.csv for the layout."
    )


def run_once(paths: PipelinePaths, args: argparse.Namespace, cfg: AnalysisConfig) -> int:
    try:
        result = run_pipeline(
            paths,
            cfg,
            force=args.force,
            figures=not args.no_figures,
            strain_max=args.strain_max,
            correlate=not args.no_correlation,
            color_by=args.color_by,
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
