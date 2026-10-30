"""End-to-end orchestration: raw CSVs -> metrics table -> figures.

Also handles change detection. A fingerprint of every input (specimens.csv plus
each raw file, by content hash) is stored beside the results; when nothing has
changed the pipeline can skip work, and when anything has changed the metrics
and both figure views regenerate together. That is what keeps the outputs from
drifting out of sync with the inputs without manual bookkeeping.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG, AnalysisConfig
from .envelope import (
    build_curve_matrix,
    display_name,
    plot_densification_check,
    plot_metric_space,
    plot_stress_strain_envelope,
    resolve_column,
)
from .io import Diagnostic, load_test
from .mechanics import StressStrainCurve, to_stress_strain
from .metrics import compute_metrics
from .specimens import Specimen, load_specimens, parse_datetimes

#: Axis labels for the generated figures.
LABELS: dict[str, str] = {
    "modulus_MPa": "Young's modulus (MPa)",
    "plateau_stress_MPa": "Plateau stress (MPa)",
    "energy_absorption_MJ_m3": "Energy absorption (MJ/m³)",
    "sea_J_per_g": "Specific energy absorption (J/g)",
    "densification_strain": "Densification strain (mm/mm)",
    "first_peak_stress_MPa": "First-peak stress (MPa)",
    "max_stress_MPa": "Peak stress (MPa)",
    "relative_density": "Relative density (–)",
}

#: Metric pairs plotted by default. Skipped automatically when a column is
#: empty (e.g. SEA with no mass recorded).
DEFAULT_PAIRS: tuple[tuple[str, str], ...] = (
    ("modulus_MPa", "plateau_stress_MPa"),
    ("plateau_stress_MPa", "energy_absorption_MJ_m3"),
    ("plateau_stress_MPa", "sea_J_per_g"),
    ("densification_strain", "plateau_stress_MPa"),
    ("relative_density", "modulus_MPa"),
)


@dataclass
class PipelinePaths:
    """Canonical locations, all derived from the repository root."""

    root: Path

    @property
    def raw_dir(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def specimens_csv(self) -> Path:
        return self.root / "data" / "specimens.csv"

    @property
    def results_dir(self) -> Path:
        return self.root / "results"

    @property
    def figures_dir(self) -> Path:
        return self.results_dir / "figures"

    @property
    def curves_dir(self) -> Path:
        return self.results_dir / "curves"

    @property
    def metrics_csv(self) -> Path:
        return self.results_dir / "metrics_summary.csv"

    @property
    def diagnostics_csv(self) -> Path:
        return self.results_dir / "diagnostics.csv"

    @property
    def manifest_json(self) -> Path:
        return self.results_dir / ".input_manifest.json"

    @property
    def correlation_dir(self) -> Path:
        return self.results_dir / "correlation"

    @property
    def correlation_figures_dir(self) -> Path:
        return self.figures_dir / "correlation"

    @classmethod
    def from_root(cls, root: str | Path | None = None) -> "PipelinePaths":
        if root is None:
            root = Path(__file__).resolve().parent.parent
        return cls(root=Path(root).resolve())


@dataclass
class PipelineResult:
    metrics: pd.DataFrame
    diagnostics: pd.DataFrame
    curves: dict[str, StressStrainCurve] = field(default_factory=dict)
    figures: list[Path] = field(default_factory=list)
    skipped: bool = False
    correlation: Any = None          # CorrelationResult, when parameters vary
    correlation_files: list[Path] = field(default_factory=list)


# ----------------------------------------------------------------------
# Change detection
# ----------------------------------------------------------------------
def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_fingerprint(paths: PipelinePaths, cfg: AnalysisConfig) -> dict[str, Any]:
    """Content hashes of every input, plus the config that shapes the output.

    Round-tripped through JSON so the value compares equal to what was written
    to disk -- otherwise types that JSON cannot represent exactly (a config
    tuple comes back as a list) would make every run look stale.
    """
    files: dict[str, str] = {}
    if paths.specimens_csv.exists():
        files["data/specimens.csv"] = _hash_file(paths.specimens_csv)
    if paths.raw_dir.exists():
        for csv in sorted(paths.raw_dir.rglob("*.csv")):
            files[str(csv.relative_to(paths.root))] = _hash_file(csv)
    return json.loads(_serialise({"files": files, "config": cfg.to_dict()}))


def _serialise(fingerprint: dict[str, Any]) -> str:
    return json.dumps(fingerprint, indent=2, sort_keys=True, default=str)


def is_stale(paths: PipelinePaths, cfg: AnalysisConfig) -> bool:
    """True when inputs, config or outputs have changed since the last run."""
    if not paths.metrics_csv.exists() or not paths.manifest_json.exists():
        return True
    try:
        stored = json.loads(paths.manifest_json.read_text())
    except (json.JSONDecodeError, OSError):
        return True
    return stored != input_fingerprint(paths, cfg)


def write_manifest(paths: PipelinePaths, cfg: AnalysisConfig) -> None:
    paths.results_dir.mkdir(parents=True, exist_ok=True)
    paths.manifest_json.write_text(_serialise(input_fingerprint(paths, cfg)))


# ----------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------
def analyse_specimen(
    spec: Specimen, paths: PipelinePaths, cfg: AnalysisConfig
) -> tuple[dict[str, Any], list[Diagnostic], StressStrainCurve]:
    """Run one test end to end: parse -> clean -> stress/strain -> metrics."""
    loaded = load_test(paths.raw_dir / spec.raw_file, cfg)
    curve = to_stress_strain(loaded.trimmed, spec)
    metrics, final_curve = compute_metrics(curve, spec, cfg)

    diagnostics = loaded.diagnostics + metrics.diagnostics
    metrics.diagnostics = diagnostics

    row = metrics.as_row()
    row.update(
        {
            "pattern_name": spec.pattern_name,
            "replicate": spec.replicate,
            "material": spec.material,
            "test_date": spec.test_date,
            "print_date": spec.print_date,
            "side_length_mm": spec.side_length_mm,
            "cross_section_area_mm2": spec.cross_section_area_mm2,
            "initial_height_mm": spec.initial_height_mm,
            "relative_density": spec.relative_density,
            "mass_g": spec.mass_g,
            "raw_file": spec.raw_file,
            "pattern_params": json.dumps(spec.pattern_params, sort_keys=True),
            "n_samples_raw": len(loaded.raw),
            "n_samples_trimmed": len(loaded.trimmed),
            "contact_index": loaded.contact_index,
            "preload_slack_mm": float(
                loaded.raw["disp_mm"].iloc[loaded.contact_index]
                - loaded.raw["disp_mm"].iloc[0]
            ),
            "tare_offset_N": loaded.tare_offset_N,
            "ended_at_force_limit": loaded.force_limited,
            "data_digest": data_digest(loaded.raw)[:16],
        }
    )
    row.update(censor_flags(metrics, loaded.force_limited))

    # Every design and manufacturing parameter -- numeric, categorical or a
    # date -- gets a param_<key> column; that is what the correlation reads.
    for key, value in spec.numeric_params().items():
        row[f"param_{key}"] = value
    for key, value in spec.categorical_params().items():
        row[f"param_{key}"] = value
    for key, value in spec.datetime_params().items():
        row[f"param_{key}"] = value.isoformat(sep=" ")
    age = spec.print_to_test_days()
    if age is not None and "print_to_test_days" not in spec.parameters():
        row["param_print_to_test_days"] = round(age, 4)
    return row, diagnostics, final_curve


def censor_flags(metrics: Any, force_limited: bool) -> dict[str, bool]:
    """Which reported values are only lower bounds, not measurements.

    A test stopped by the load limit before densifying still produces numbers
    for densification strain and energy absorption, but they record where the
    test stopped rather than what the specimen does. They stay in the table,
    flagged, so the correlation analysis can leave them out instead of
    correlating a parameter with the machine's cut-off.
    """
    dens = metrics.densification
    dens_found = bool(np.isfinite(dens.strain))
    dens_censored = dens_found and bool(dens.at_end_of_data)
    return {
        # Densification strain is a lower bound when it was still climbing at
        # the end of the data.
        "densification_censored": dens_censored,
        # Energy to densification is a bound for the same reason -- and is not
        # comparable at all when densification was never found and a fallback
        # strain was used instead.
        "energy_censored": dens_censored or not dens_found,
        # Without a distinct first peak, "crush" load is the maximum before
        # densification -- which, on a limit-stopped test, is the limit.
        "crush_censored": bool(force_limited)
        and not bool(metrics.peaks.has_distinct_first_peak),
    }


def _order_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Identity first, headline metrics next, provenance last."""
    lead = [
        "test_id", "pattern_name", "replicate", "material", "test_date",
        "modulus_MPa", "plateau_stress_MPa", "densification_strain",
        "energy_absorption_MJ_m3", "sea_J_per_g",
        "first_peak_stress_MPa", "crush_force_N", "max_stress_MPa",
        "ended_at_force_limit",
    ]
    tail = ["raw_file", "pattern_params", "data_digest", "diagnostics"]
    lead = [c for c in lead if c in frame.columns]
    tail = [c for c in tail if c in frame.columns]
    middle = [c for c in frame.columns if c not in lead and c not in tail]
    return frame[lead + middle + tail]


def data_digest(raw: pd.DataFrame) -> str:
    """Hash of a test's measurements, ignoring how the file was written.

    Hashing the *parsed columns* rather than the file bytes is deliberate: the
    instrument re-exports the same test with and without the boilerplate
    header, so two byte-different files can hold identical measurements.
    """
    payload = raw[["time_s", "disp_mm", "force_kN"]].to_csv(index=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def group_duplicates(digests: dict[str, str]) -> list[list[str]]:
    """Groups of test_ids whose measurements are identical.

    Registered as separate rows these masquerade as replicates and manufacture
    a spread that no second specimen ever produced.
    """
    by_digest: dict[str, list[str]] = {}
    for test_id, digest in digests.items():
        by_digest.setdefault(digest, []).append(test_id)
    return sorted(
        (sorted(ids) for ids in by_digest.values() if len(ids) > 1),
        key=lambda ids: ids[0],
    )


def run_analysis(
    paths: PipelinePaths, cfg: AnalysisConfig = DEFAULT_CONFIG
) -> PipelineResult:
    """Analyse every registered test and write the metrics summary."""
    specs = load_specimens(paths.specimens_csv, paths.raw_dir)

    rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    curves: dict[str, StressStrainCurve] = {}
    failures: list[str] = []

    for spec in specs:
        try:
            row, diagnostics, curve = analyse_specimen(spec, paths, cfg)
        except Exception as exc:
            # One unreadable file must not sink the whole batch; record it and
            # keep going, then report at the end.
            failures.append(f"{spec.test_id}: {type(exc).__name__}: {exc}")
            diag_rows.append(
                {
                    "test_id": spec.test_id,
                    "pattern_name": spec.pattern_name,
                    "level": "error",
                    "code": "analysis_failed",
                    "message": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        rows.append(row)
        curves[spec.test_id] = curve
        for diag in diagnostics:
            diag_rows.append(
                {
                    "test_id": spec.test_id,
                    "pattern_name": spec.pattern_name,
                    "level": diag.level,
                    "code": diag.code,
                    "message": diag.message,
                }
            )

    if not rows:
        raise RuntimeError(
            "No test could be analysed.\n  - " + "\n  - ".join(failures or ["no rows"])
        )

    metrics = _order_columns(pd.DataFrame(rows)).sort_values(
        ["pattern_name", "replicate", "test_id"]
    )

    # Identical measurements registered under two test_ids are not replicates.
    for group in group_duplicates(
        {str(r["test_id"]): str(r["data_digest"]) for r in rows}
    ):
        message = (
            f"identical measurements to {', '.join(group)} -- these are the same "
            f"test exported more than once, not separate specimens. Registering "
            f"them as replicates fabricates the spread between them; remove all "
            f"but one from specimens.csv."
        )
        for test_id in group:
            diag_rows.append(
                {
                    "test_id": test_id,
                    "pattern_name": next(
                        (r["pattern_name"] for r in rows if r["test_id"] == test_id), ""
                    ),
                    "level": "warning",
                    "code": "duplicate_test_data",
                    "message": message,
                }
            )

    diagnostics = pd.DataFrame(
        diag_rows, columns=["test_id", "pattern_name", "level", "code", "message"]
    )

    paths.results_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(paths.metrics_csv, index=False)
    diagnostics.to_csv(paths.diagnostics_csv, index=False)

    paths.curves_dir.mkdir(parents=True, exist_ok=True)
    for test_id, curve in curves.items():
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in test_id)
        curve.to_frame().to_csv(paths.curves_dir / f"{safe}.csv", index=False)

    if failures:
        print(f"WARNING: {len(failures)} test(s) failed to analyse:")
        for failure in failures:
            print(f"  - {failure}")

    return PipelineResult(metrics=metrics, diagnostics=diagnostics, curves=curves)


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
def load_curves_from_disk(
    paths: PipelinePaths, metrics: pd.DataFrame
) -> dict[str, StressStrainCurve]:
    """Rebuild curves from `results/curves/`, so plotting can run standalone."""
    curves: dict[str, StressStrainCurve] = {}
    for test_id in metrics["test_id"]:
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(test_id))
        path = paths.curves_dir / f"{safe}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        curves[str(test_id)] = StressStrainCurve(
            test_id=str(test_id),
            strain=frame["strain"].to_numpy(dtype=float),
            stress_MPa=frame["stress_MPa"].to_numpy(dtype=float),
            force_N=frame["force_N"].to_numpy(dtype=float),
            disp_mm=frame["disp_mm"].to_numpy(dtype=float),
            time_s=frame["time_s"].to_numpy(dtype=float),
        )
    return curves


def make_figures(
    paths: PipelinePaths,
    metrics: pd.DataFrame,
    curves: dict[str, StressStrainCurve],
    cfg: AnalysisConfig = DEFAULT_CONFIG,
    pairs: tuple[tuple[str, str], ...] = DEFAULT_PAIRS,
    strain_max: float | None = None,
    color_by: str | None = None,
) -> list[Path]:
    """Generate both envelope views. Returns the figures written.

    `color_by` (default: `cfg.color_by`) names a parameter to colour curves and
    points by. An unknown name is reported and ignored rather than failing the
    whole run.
    """
    paths.figures_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    color_by = color_by if color_by is not None else cfg.color_by
    color_col = resolve_column(metrics, color_by)
    if color_by and color_col is None:
        print(
            f"note: --color-by {color_by!r} matches no column; available "
            f"parameters: {available_parameters(metrics) or 'none'}"
        )

    # Dates colour as elapsed time, not as dozens of distinct labels.
    color_by_values, color_label = color_col, display_name(color_col) if color_col else ""
    if color_col:
        stamps = parse_datetimes(metrics[color_col])
        if stamps is not None:
            origin = stamps.min().normalize()
            metrics = metrics.assign(
                _color_days=(stamps - origin).dt.total_seconds() / 86400.0
            )
            color_by_values = "_color_days"
            color_label = f"{display_name(color_col)} (days since {origin:%Y-%m-%d})"

    # (a) Stress-strain overlay + shaded envelope.
    rows = [row for row in metrics.itertuples(index=False) if str(row.test_id) in curves]
    items = [(str(r.test_id), str(r.pattern_name), curves[str(r.test_id)]) for r in rows]
    if items:
        matrix = build_curve_matrix(items, cfg.envelope_n_points, strain_max)
        values = None
        if color_col:
            lookup = metrics.set_index(metrics["test_id"].astype(str))[color_by_values]
            values = [lookup.get(t) for t in matrix.test_ids]
        written.append(
            plot_stress_strain_envelope(
                matrix, paths.figures_dir / "stress_strain_envelope.png", cfg,
                color_values=values, color_label=color_label,
            )
        )
        matrix.frame().to_csv(paths.results_dir / "resampled_curves.csv")

        load_limited = (
            "ended_at_force_limit" in metrics.columns
            and metrics["ended_at_force_limit"].astype(str).str.lower().eq("true").any()
        )
        out = plot_densification_check(
            [c for _, _, c in items], paths.figures_dir / "densification_check.png", cfg,
            load_limited=bool(load_limited),
        )
        if out is not None:
            written.append(out)

    # (b) Metric-space scatter with achievable-property hull.
    for x, y in pairs:
        if x not in metrics.columns or y not in metrics.columns:
            continue
        if metrics[[x, y]].dropna().empty:
            continue
        out = plot_metric_space(
            metrics, x, y, paths.figures_dir / f"metric_space_{y}_vs_{x}.png",
            cfg, labels=LABELS, color_by=color_by_values, color_label=color_label,
        )
        if out is not None:
            written.append(out)

    return written


def available_parameters(metrics: pd.DataFrame) -> list[str]:
    """Parameter names a user can pass to --color-by."""
    names = [display_name(c) for c in metrics.columns if c.startswith("param_")]
    for core in ("relative_density", "material", "pattern_name"):
        if core in metrics.columns:
            names.append(core)
    return names


def run_pipeline(
    paths: PipelinePaths,
    cfg: AnalysisConfig = DEFAULT_CONFIG,
    force: bool = False,
    figures: bool = True,
    strain_max: float | None = None,
    correlate: bool = True,
    color_by: str | None = None,
) -> PipelineResult:
    """Run analysis, figures and correlation, skipping if nothing changed."""
    if not force and not is_stale(paths, cfg):
        metrics = pd.read_csv(paths.metrics_csv)
        diagnostics = (
            pd.read_csv(paths.diagnostics_csv)
            if paths.diagnostics_csv.exists()
            else pd.DataFrame()
        )
        return PipelineResult(metrics=metrics, diagnostics=diagnostics, skipped=True)

    result = run_analysis(paths, cfg)
    if figures:
        result.figures = make_figures(
            paths, result.metrics, result.curves, cfg,
            strain_max=strain_max, color_by=color_by,
        )
    if correlate:
        run_correlation(paths, result, cfg, figures=figures)
    write_manifest(paths, cfg)
    return result


def run_correlation(
    paths: PipelinePaths,
    result: PipelineResult,
    cfg: AnalysisConfig = DEFAULT_CONFIG,
    figures: bool = True,
) -> None:
    """Parameter-property correlation over the whole metrics table.

    Outputs from an earlier run are cleared first, so removing a parameter
    from specimens.csv cannot leave its old figures looking current.
    """
    from . import correlation

    correlation.clear_outputs(paths.correlation_dir, paths.correlation_figures_dir)
    if not correlation.has_parameters(result.metrics, cfg):
        return
    result.correlation = correlation.analyse(result.metrics, cfg)
    result.correlation_files = correlation.write_outputs(
        result.correlation, paths.correlation_dir, paths.correlation_figures_dir,
        cfg, figures=figures,
    )
