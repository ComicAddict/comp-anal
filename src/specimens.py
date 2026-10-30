"""Specimen metadata: schema, validation and lookup.

`data/specimens.csv` is the single source of truth mapping each raw instrument
CSV to the pattern it belongs to and the geometry needed to turn force and
displacement into stress and strain. It is hand-maintained (via
`scripts/register_test.py`) and read-only for the analysis pipeline.

Design and manufacturing parameters can be given two ways, and both may be
used at once: as a JSON object in `pattern_params`, or -- easier for a large
spreadsheet -- as extra columns, one per parameter (`cell_mm`,
`layer_height_mm`, `print_orientation`, ...). Every column beyond the schema
below is treated as a parameter.
"""

from __future__ import annotations

import json
import math
import re
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

#: Core columns that may be absent from older files. When present they are
#: part of the schema (not parameters); when absent they read as blank.
OPTIONAL_COLUMNS: tuple[str, ...] = ("print_date",)

#: Core columns holding a date, or a date and time. ISO format
#: (2026-07-17 19:22) is the safe choice: 07/08/2026 reads as 8 July.
DATE_COLUMNS: tuple[str, ...] = ("print_date", "test_date")

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
    #: Values from any extra columns in specimens.csv, keyed by sanitised
    #: column name: floats for numeric columns, Timestamps for date columns,
    #: strings for the rest.
    extra_params: dict[str, Any] = field(default_factory=dict)
    #: When the specimen was printed (date, or date and time).
    print_date: str | None = None

    @property
    def volume_mm3(self) -> float:
        """Bounding volume of the specimen along the load axis."""
        return self.cross_section_area_mm2 * self.initial_height_mm

    def parameters(self) -> dict[str, Any]:
        """Every design and manufacturing parameter recorded for this test.

        The union of the `pattern_params` JSON and any extra columns. Conflicts
        between the two were already rejected in `to_specimens`.
        """
        merged = dict(self.pattern_params)
        merged.update(self.extra_params)
        return merged

    def numeric_params(self) -> dict[str, float]:
        """Parameters whose values are numeric."""
        out: dict[str, float] = {}
        for key, value in self.parameters().items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                out[key] = float(value)
        return out

    def datetime_params(self) -> dict[str, pd.Timestamp]:
        """Parameters whose values are dates or times."""
        return {k: v for k, v in self.parameters().items() if isinstance(v, pd.Timestamp)}

    def print_to_test_days(self) -> float | None:
        """Days between printing and testing, when both dates are known.

        PLA keeps changing after printing -- it ages physically and takes up
        moisture -- so the interval is a candidate cause in its own right.
        """
        printed, tested = parse_date(self.print_date), parse_date(self.test_date)
        if printed is None or tested is None:
            return None
        return (tested - printed).total_seconds() / 86400.0

    def categorical_params(self) -> dict[str, str]:
        """Parameters whose values are labels (text or true/false)."""
        out: dict[str, str] = {}
        for key, value in self.parameters().items():
            if isinstance(value, bool):
                out[key] = "true" if value else "false"
            elif isinstance(value, str) and value.strip():
                out[key] = value.strip()
        return out


def parse_date(value: Any) -> pd.Timestamp | None:
    """One cell as a Timestamp, or None when blank or not a date."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text or not re.search(r"\d", text):
        return None
    try:
        stamp = pd.to_datetime(text, format="mixed")
    except (ValueError, TypeError, OverflowError):
        return None
    if pd.isna(stamp):
        return None
    # Wall-clock time, as in parse_datetimes: a "Z" on one date must not make
    # it incomparable with a plain date in the other column.
    return stamp.tz_localize(None) if stamp.tzinfo is not None else stamp


def parse_datetimes(series: pd.Series) -> pd.Series | None:
    """A whole column as datetimes, or None if it is not a date column.

    Every filled cell must parse. Numbers are never dates (20260717 stays a
    number), and every cell needs a digit, since dateutil happily reads "May"
    as a date in the year 1.
    """
    present = series.notna() & (series.astype(str).str.strip() != "")
    if not present.any():
        return None
    cells = series[present].astype(str).str.strip()
    if pd.to_numeric(cells, errors="coerce").notna().all():
        return None
    if not cells.str.contains(r"\d").all():
        return None
    try:
        parsed = pd.to_datetime(cells, errors="coerce", format="mixed")
    except (ValueError, TypeError, OverflowError):
        return None
    if not pd.api.types.is_datetime64_any_dtype(parsed) or parsed.isna().any():
        return None
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_localize(None)
    out = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    out.loc[present] = parsed.to_numpy()
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
    numeric and date columns parse, no test predates its print, test_ids are
    unique, and (if `raw_dir` is given) every referenced raw file exists on
    disk.
    """
    problems: list[str] = []

    missing_cols = [c for c in COLUMNS if c not in frame.columns]  # optional ones may be absent
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

    # Date columns must hold dates.
    for col in DATE_COLUMNS:
        if col not in frame.columns:
            continue
        for idx, value in frame[col].items():
            if value is None or (isinstance(value, float) and pd.isna(value)):
                continue
            text = str(value).strip()
            if text and parse_date(text) is None:
                problems.append(
                    f"row {idx + 2}: '{col}' is not a date: {text!r} "
                    f"(use e.g. 2026-07-17 or 2026-07-17 19:22)"
                )

    # A specimen cannot be tested before it was printed. Usually two swapped
    # cells, and it would enter the analysis as a negative age.
    if "print_date" in frame.columns:
        for idx, row in frame.iterrows():
            printed, tested = parse_date(row["print_date"]), parse_date(row["test_date"])
            if printed is not None and tested is not None and tested < printed:
                problems.append(
                    f"row {idx + 2}: test_date ({str(row['test_date']).strip()}) is before "
                    f"print_date ({str(row['print_date']).strip()}); check the two dates"
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


def sanitise_name(name: str) -> str:
    """Turn a spreadsheet header into a safe parameter key.

    "Layer Height (mm)" -> "Layer_Height_mm". Case is kept, so the key still
    reads like the header it came from.
    """
    return re.sub(r"[^0-9A-Za-z_]+", "_", str(name)).strip("_")


def extra_parameter_columns(frame: pd.DataFrame) -> dict[str, str]:
    """Map each extra (non-schema) column to its parameter key.

    Any column beyond the documented schema is treated as a design or
    manufacturing parameter, so parameters can be typed straight into the
    spreadsheet one column each rather than as a JSON cell per row. Columns
    pandas names "Unnamed: N" are the artefact of a trailing delimiter in an
    Excel export, not data, and are skipped.
    """
    mapping: dict[str, str] = {}
    for column in frame.columns:
        if column in COLUMNS or column in OPTIONAL_COLUMNS or str(column).startswith("Unnamed:"):
            continue
        key = sanitise_name(column)
        if not key:
            raise SpecimenError(
                f"column {column!r} has no usable name once punctuation is "
                f"removed; rename it, e.g. to 'param_1'"
            )
        if key in mapping.values():
            other = next(c for c, k in mapping.items() if k == key)
            raise SpecimenError(
                f"columns {other!r} and {column!r} both become parameter "
                f"{key!r}; rename one of them"
            )
        mapping[column] = key
    return mapping


def _numeric_columns(frame: pd.DataFrame, columns: list[str]) -> set[str]:
    """Columns whose every non-blank cell parses as a number.

    Decided per column, not per cell, so one parameter never ends up half
    numeric and half text across tests.
    """
    numeric: set[str] = set()
    for column in columns:
        cells = frame[column].dropna().astype(str).str.strip()
        cells = cells[cells != ""]
        if cells.empty:
            continue
        if pd.to_numeric(cells, errors="coerce").notna().all():
            numeric.add(column)
    return numeric


def _datetime_columns(frame: pd.DataFrame, columns: list[str]) -> dict[str, pd.Series]:
    """Non-numeric columns whose every filled cell is a date or time."""
    found: dict[str, pd.Series] = {}
    for column in columns:
        parsed = parse_datetimes(frame[column])
        if parsed is not None:
            found[column] = parsed
    return found


def _extra_values(
    row: pd.Series,
    mapping: dict[str, str],
    numeric: set[str],
    dates: dict[str, pd.Series],
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for column, key in mapping.items():
        cell = row.get(column)
        if cell is None or (isinstance(cell, float) and pd.isna(cell)):
            continue
        text = str(cell).strip()
        if not text:
            continue
        if column in numeric:
            values[key] = float(text)
        elif column in dates:
            values[key] = pd.Timestamp(dates[column].loc[row.name])
        else:
            values[key] = text
    return values


def _merge_conflicts(
    json_params: dict[str, Any], extra: dict[str, Any]
) -> list[str]:
    """Keys set in both places to different values for the same test."""
    conflicts = []
    for key in set(json_params) & set(extra):
        a, b = json_params[key], extra[key]
        try:
            same = float(a) == float(b)
        except (TypeError, ValueError):
            same = str(a).strip() == str(b).strip()
        if not same:
            conflicts.append(f"{key!r} is {a!r} in pattern_params but {b!r} in its column")
    return conflicts


def to_specimens(frame: pd.DataFrame) -> list[Specimen]:
    """Convert a validated frame into `Specimen` objects.

    `cross_section_area_mm2` is derived from `side_length_mm` (square section)
    when the cell is blank, matching the documented override behaviour. Extra
    columns become parameters (see `extra_parameter_columns`).
    """
    mapping = extra_parameter_columns(frame)
    numeric = _numeric_columns(frame, list(mapping))
    dates = _datetime_columns(frame, [c for c in mapping if c not in numeric])

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

        json_params = _parse_params(row.get("pattern_params"), test_id)
        extra = _extra_values(row, mapping, numeric, dates)
        conflicts = _merge_conflicts(json_params, extra)
        if conflicts:
            raise SpecimenError(
                f"[{test_id}] parameter given twice with different values: "
                + "; ".join(conflicts)
                + ". Keep it in one place."
            )

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
                pattern_params=json_params,
                relative_density=_opt_float(row.get("relative_density")),
                mass_g=_opt_float(row.get("mass_g")),
                test_date=_opt_str(row.get("test_date")),
                notes=_opt_str(row.get("notes")),
                extra_params=extra,
                print_date=_opt_str(row.get("print_date")),
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
