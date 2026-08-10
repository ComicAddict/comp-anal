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

import pandas as pd

from .config import DEFAULT_CONFIG, AnalysisConfig
from .envelope import (
    build_curve_matrix,
    plot_metric_space,
    plot_metric_vs_param,
    plot_stress_strain_envelope,
)
from .io import Diagnostic, load_test
from .mechanics import StressStrainCurve, to_stress_strain
from .metrics import compute_metrics
from .specimens import Specimen, load_specimens

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
        }
    )
    for key, value in spec.numeric_params().items():
        row[f"param_{key}"] = value
    return row, diagnostics, final_curve


def _order_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Identity first, headline metrics next, provenance last."""
    lead = [
        "test_id", "pattern_name", "replicate", "material", "test_date",
        "modulus_MPa", "plateau_stress_MPa", "densification_strain",
        "energy_absorption_MJ_m3", "sea_J_per_g",
        "first_peak_stress_MPa", "crush_force_N", "max_stress_MPa",
    ]
    tail = ["raw_file", "pattern_params", "diagnostics"]
    lead = [c for c in lead if c in frame.columns]
    tail = [c for c in tail if c in frame.columns]
    middle = [c for c in frame.columns if c not in lead and c not in tail]
    return frame[lead + middle + tail]


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
) -> list[Path]:
    """Generate both envelope views. Returns the figures written."""
    paths.figures_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # (a) Stress-strain overlay + shaded envelope.
    items = [
        (str(row.test_id), str(row.pattern_name), curves[str(row.test_id)])
        for row in metrics.itertuples()
        if str(row.test_id) in curves
    ]
    if items:
        matrix = build_curve_matrix(items, cfg.envelope_n_points, strain_max)
        written.append(
            plot_stress_strain_envelope(
                matrix, paths.figures_dir / "stress_strain_envelope.png", cfg
            )
        )
        matrix.frame().to_csv(paths.results_dir / "resampled_curves.csv")

    # (b) Metric-space scatter with achievable-property hull.
    for x, y in pairs:
        if x not in metrics.columns or y not in metrics.columns:
            continue
        if metrics[[x, y]].dropna().empty:
            continue
        out = plot_metric_space(
            metrics, x, y, paths.figures_dir / f"metric_space_{y}_vs_{x}.png",
            cfg, labels=LABELS,
        )
        if out is not None:
            written.append(out)

    # Metric vs swept numeric pattern parameter.
    params = [c[len("param_") :] for c in metrics.columns if c.startswith("param_")]
    for param in params:
        for metric in ("modulus_MPa", "plateau_stress_MPa", "energy_absorption_MJ_m3"):
            if metric not in metrics.columns:
                continue
            out = plot_metric_vs_param(
                metrics, param, metric,
                paths.figures_dir / f"param_{param}_{metric}.png",
                cfg, labels=LABELS,
            )
            if out is not None:
                written.append(out)

    return written


def run_pipeline(
    paths: PipelinePaths,
    cfg: AnalysisConfig = DEFAULT_CONFIG,
    force: bool = False,
    figures: bool = True,
    strain_max: float | None = None,
) -> PipelineResult:
    """Run analysis and figures, skipping if nothing changed (unless forced)."""
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
            paths, result.metrics, result.curves, cfg, strain_max=strain_max
        )
    write_manifest(paths, cfg)
    return result
