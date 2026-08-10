"""Specimen metadata: schema, validation and lookup.

`data/specimens.csv` is the single source of truth mapping each raw instrument
CSV to the pattern it belongs to and the geometry needed to turn force and
displacement into stress and strain. It is hand-maintained (via
`scripts/register_test.py`) and read-only for the analysis pipeline.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

# Column order used when creating or appending to specimens.csv.
COLUMNS: tuple[str, ...] = (
    "test_id",
    "raw_file",
    "pattern_name",
    "pattern_params",
    "replicate",
    "side_length_mm",
    "cross_section_area_mm2",
    "initial_height_mm",
    "relative_density",
    "mass_g",
    "material",
    "test_date",
    "notes",
)

#: Columns that must be present AND non-null for every row.
REQUIRED_NON_NULL: tuple[str, ...] = (
    "test_id",
    "raw_file",
    "pattern_name",
    "replicate",
    "side_length_mm",
    "initial_height_mm",
    "material",
)

#: Present in the file but allowed to be blank.
OPTIONAL: tuple[str, ...] = (
    "pattern_params",
    "cross_section_area_mm2",  # derived from side_length_mm when blank
    "relative_density",
    "mass_g",
    "test_date",
    "notes",
)

NUMERIC = (
    "replicate",
    "side_length_mm",
    "cross_section_area_mm2",
    "initial_height_mm",
    "relative_density",
    "mass_g",
)


class SpecimenError(ValueError):
    """Raised when specimens.csv is missing, malformed or incomplete."""


@dataclass(frozen=True)
class Specimen:
    """One registered test: metadata + geometry."""

    test_id: str
    raw_file: str
    pattern_name: str
    replicate: int
    side_length_mm: float
    cross_section_area_mm2: float
    initial_height_mm: float
    material: str
    pattern_params: dict[str, Any] = field(default_factory=dict)
    relative_density: float | None = None
    mass_g: float | None = None
    test_date: str | None = None
    notes: str | None = None

    @property
    def volume_mm3(self) -> float:
        """Bounding volume of the specimen along the load axis."""
        return self.cross_section_area_mm2 * self.initial_height_mm

    def numeric_params(self) -> dict[str, float]:
        """Subset of `pattern_params` whose values are numeric."""
        out: dict[str, float] = {}
        for key, value in self.pattern_params.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                out[key] = float(value)
        return out


def _parse_params(raw: Any, test_id: str) -> dict[str, Any]:
    """Parse the free-form JSON `pattern_params` cell."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return {}
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SpecimenError(
            f"[{test_id}] pattern_params is not valid JSON: {text!r} ({exc}). "
            'Use a JSON object, e.g. {"cell_mm": 5.0, "wall_mm": 0.8}'
        ) from exc
    if not isinstance(parsed, dict):
        raise SpecimenError(
            f"[{test_id}] pattern_params must be a JSON object, got "
            f"{type(parsed).__name__}: {text!r}"
        )
    return parsed


def _opt_float(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    out = float(value)
    return None if math.isnan(out) else out


def _opt_str(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text or None


def empty_frame() -> pd.DataFrame:
    """An empty specimens table with the canonical columns."""
    return pd.DataFrame(columns=list(COLUMNS))


def read_specimens_frame(path: str | Path) -> pd.DataFrame:
    """Read specimens.csv as raw strings, without validation."""
    path = Path(path)
    if not path.exists():
        raise SpecimenError(
            f"specimens.csv not found at {path}. Create it by registering a "
            f"test: python scripts/register_test.py --interactive"
        )
    frame = pd.read_csv(path, dtype=str, keep_default_na=True)
    frame.columns = [c.strip() for c in frame.columns]
    return frame


def validate_frame(frame: pd.DataFrame, raw_dir: Path | None = None) -> pd.DataFrame:
    """Validate the specimens table. Raises `SpecimenError` on any problem.

    Checks, in order: required columns exist, required cells are non-null,
    numeric columns parse, test_ids are unique, and (if `raw_dir` is given)
    every referenced raw file exists on disk.
    """
    problems: list[str] = []

    missing_cols = [c for c in COLUMNS if c not in frame.columns]
    if missing_cols:
        raise SpecimenError(
            f"specimens.csv is missing required column(s): {missing_cols}. "
            f"Expected columns: {list(COLUMNS)}"
        )

    if frame.empty:
        raise SpecimenError(
            "specimens.csv has no rows -- nothing to analyse. Register a test "
            "with: python scripts/register_test.py --interactive"
        )

    # Required cells non-null. Reported one line per column with the affected
    # rows collected, rather than one line per cell -- a freshly templated file
    # is missing the same few columns everywhere, and 25 near-identical lines
    # bury the point.
    for col in REQUIRED_NON_NULL:
        blank = frame[col].isna() | (frame[col].astype(str).str.strip() == "")
        rows = [int(idx) + 2 for idx in frame.index[blank]]
        if not rows:
            continue
        if len(rows) == len(frame):
            where = "every row"
        elif len(rows) > 4:
            where = f"rows {rows[0]}-{rows[-1]} ({len(rows)} rows)"
        else:
            where = "row " + ", ".join(str(r) for r in rows)
        problems.append(f"'{col}' is required but empty on {where}")

    # Numeric columns parse.
    for col in NUMERIC:
        for idx, value in frame[col].items():
            if value is None or (isinstance(value, float) and pd.isna(value)):
                continue
            text = str(value).strip()
            if not text:
                continue
            try:
                float(text)
            except ValueError:
                problems.append(
                    f"row {idx + 2}: '{col}' must be numeric, got {text!r}"
                )

    # Unique test_id.
    ids = frame["test_id"].astype(str).str.strip()
    duplicates = ids[ids.duplicated(keep=False) & (ids != "")]
    for dup in sorted(set(duplicates)):
        rows = [int(i) + 2 for i in frame.index[ids == dup]]
        problems.append(f"duplicate test_id {dup!r} on rows {rows}")

    if problems:
        raise SpecimenError(
            "specimens.csv failed validation:\n  - " + "\n  - ".join(problems)
        )

    # Referenced raw files exist (reported separately so the message is clear).
    if raw_dir is not None:
        missing_files = [
            f"row {idx + 2}: raw_file {value!r} not found under {raw_dir}"
            for idx, value in frame["raw_file"].items()
            if not (raw_dir / str(value).strip()).exists()
        ]
        if missing_files:
            raise SpecimenError(
                "specimens.csv references missing raw files:\n  - "
                + "\n  - ".join(missing_files)
            )

    return frame


def to_specimens(frame: pd.DataFrame) -> list[Specimen]:
    """Convert a validated frame into `Specimen` objects.

    `cross_section_area_mm2` is derived from `side_length_mm` (square section)
    when the cell is blank, matching the documented override behaviour.
    """
    out: list[Specimen] = []
    for _, row in frame.iterrows():
        test_id = str(row["test_id"]).strip()
        side = float(row["side_length_mm"])
        area = _opt_float(row["cross_section_area_mm2"])
        if area is None:
            area = side * side
        if area <= 0:
            raise SpecimenError(f"[{test_id}] cross_section_area_mm2 must be > 0")
        height = float(row["initial_height_mm"])
        if height <= 0:
            raise SpecimenError(f"[{test_id}] initial_height_mm must be > 0")

        out.append(
            Specimen(
                test_id=test_id,
                raw_file=str(row["raw_file"]).strip(),
                pattern_name=str(row["pattern_name"]).strip(),
                replicate=int(float(row["replicate"])),
                side_length_mm=side,
                cross_section_area_mm2=area,
                initial_height_mm=height,
                material=str(row["material"]).strip(),
                pattern_params=_parse_params(row.get("pattern_params"), test_id),
                relative_density=_opt_float(row.get("relative_density")),
                mass_g=_opt_float(row.get("mass_g")),
                test_date=_opt_str(row.get("test_date")),
                notes=_opt_str(row.get("notes")),
            )
        )
    return out


def load_specimens(path: str | Path, raw_dir: str | Path | None = None) -> list[Specimen]:
    """Read, validate and parse specimens.csv in one call."""
    frame = read_specimens_frame(path)
    validate_frame(frame, Path(raw_dir) if raw_dir is not None else None)
    return to_specimens(frame)


def unregistered_files(raw_dir: str | Path, specimens_csv: str | Path) -> list[Path]:
    """Raw CSVs present on disk but absent from specimens.csv."""
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        return []
    known: set[str] = set()
    path = Path(specimens_csv)
    if path.exists():
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        if "raw_file" in frame.columns:
            known = {str(v).strip() for v in frame["raw_file"]}
    found = sorted(p for p in raw_dir.rglob("*.csv"))
    return [p for p in found if str(p.relative_to(raw_dir)) not in known]
