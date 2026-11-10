"""Build specimens.csv from the lab's own metadata sheets.

The metadata is kept the way the lab keeps it -- one sheet per test series,
with its own column names and date style -- and `metadata_sources.json` says
how to read each sheet once. From then on `scripts/import_metadata.py`
regenerates the specimen rows, so registering 100+ tests is never typed by
hand and a column added to a sheet flows into the analysis by itself.

What the importer does, per source:

* reads the sheet (CSV or Excel) as text, so pattern codes like ``002`` keep
  their leading zeros;
* maps columns onto the specimens.csv schema: an explicit ``columns`` entry
  wins, otherwise common names are recognised (``testing date`` ->
  ``test_date``, ``weight`` -> ``mass_g``, ...), and every other column becomes
  a parameter under its own (sanitised) name. ``null`` drops a column;
* rewrites dates in ISO form, read in the sheet's stated day/month order, so
  ``8-6-26`` cannot be misread downstream;
* finds each test's raw export by its test number in the file name;
* fills geometry and material from ``defaults``; derives the distance from the
  bed centre from ``bed_x``/``bed_y``, splits coded pattern names into one
  numeric parameter per digit, and -- when a measured mass is present --
  relative density from mass, volume and the bulk density;
* leaves out rows whose notes say so (``exclude_where``), and reports them.

Nothing is guessed silently: unmatched rows, unmatched raw files, exclusions
and duplicate exports are all listed in the report.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .specimens import COLUMNS, OPTIONAL_COLUMNS, SpecimenError, sanitise_name

CONFIG_FILENAME = "metadata_sources.json"

#: Core fields recognised by column name when the config does not map them.
#: Matched after lower-casing and collapsing punctuation to single spaces.
SYNONYMS: dict[str, tuple[str, ...]] = {
    "test_date": ("test date", "testing date", "test time", "testing time", "tested", "date tested"),
    "print_date": (
        "print date", "print time", "printed", "manufacturing date", "manufacturing time",
        "manufactured date", "manufactured time", "manufactured",
    ),
    "mass_g": ("mass", "mass g", "weight", "weight g", "measured weight", "measured mass"),
    "print_batch": ("batch", "print batch", "manufacturing group", "manufacturing batch"),
    "notes": ("notes", "note", "comment", "comments", "discrepancy"),
    "pattern_name": ("pattern", "pattern name", "pattern_name"),
    "bed_x": ("bed x", "bed x mm", "x position"),
    "bed_y": ("bed y", "bed y mm", "y position"),
}

#: Targets that are core specimens.csv columns rather than parameters.
_CORE_TARGETS = set(COLUMNS) | set(OPTIONAL_COLUMNS)
_DATE_TARGETS = {"test_date", "print_date"}


@dataclass
class SourceReport:
    name: str
    rows_in_sheet: int = 0
    imported: int = 0
    excluded: list[str] = field(default_factory=list)      # "test 57: <reason>"
    no_raw_file: list[str] = field(default_factory=list)   # test numbers
    unclaimed_files: list[str] = field(default_factory=list)
    extra_exports: list[str] = field(default_factory=list)  # re-exports not used
    column_map: dict[str, str | None] = field(default_factory=dict)

    def lines(self) -> list[str]:
        out = [f"[{self.name}] {self.imported} of {self.rows_in_sheet} sheet rows imported"]
        mapped = ", ".join(
            f"{src} -> {dst}" for src, dst in self.column_map.items() if dst is not None
        )
        dropped = [src for src, dst in self.column_map.items() if dst is None]
        out.append(f"  columns: {mapped}")
        if dropped:
            out.append(f"  dropped: {', '.join(dropped)}")
        for label, items in (
            ("excluded", self.excluded),
            ("no raw file found for test no.", self.no_raw_file),
            ("raw files with no sheet row", self.unclaimed_files),
            ("re-exports not used (an earlier export of the same test was)", self.extra_exports),
        ):
            if items:
                shown = ", ".join(items[:12]) + (f", ... ({len(items)})" if len(items) > 12 else "")
                out.append(f"  {label}: {shown}")
        return out


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise SpecimenError(f"{path} not found -- see metadata_sources.json in the README")
    config = json.loads(path.read_text())
    if not config.get("sources"):
        raise SpecimenError(f"{path} lists no sources")
    names = [s.get("name") for s in config["sources"]]
    if len(set(names)) != len(names) or not all(names):
        raise SpecimenError("every source needs a unique 'name'")
    return config


def _key(text: str) -> str:
    return re.sub(r"[^0-9a-z]+", " ", str(text).lower()).strip()


def map_columns(columns: list[str], explicit: dict[str, str | None]) -> dict[str, str | None]:
    """Sheet column -> specimens.csv column (None = drop). Raises on clashes."""
    lookup = {_key(word): target for target, words in SYNONYMS.items() for word in words}
    explicit_keys = {_key(c): v for c, v in explicit.items() if not c.startswith("_")}
    mapping: dict[str, str | None] = {}
    for column in columns:
        k = _key(column)
        if k in explicit_keys:
            mapping[column] = explicit_keys[k]
        elif k in lookup:
            mapping[column] = lookup[k]
        else:
            mapping[column] = sanitise_name(column) or None
    targets: dict[str, list[str]] = {}
    for column, target in mapping.items():
        if target is not None:
            targets.setdefault(target, []).append(column)
    clashes = {t: cols for t, cols in targets.items() if len(cols) > 1}
    if clashes:
        detail = "; ".join(f"{', '.join(map(repr, cols))} -> {t}" for t, cols in clashes.items())
        raise SpecimenError(
            f"two sheet columns map to the same field ({detail}). In the source's "
            f"\"columns\", map one of them elsewhere, or to null to drop it."
        )
    return mapping


# ----------------------------------------------------------------------
# Values
# ----------------------------------------------------------------------
def read_sheet(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
        frame = pd.read_excel(path, dtype=str)
    else:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame.columns = [str(c).strip() for c in frame.columns]
    frame = frame.loc[:, [c for c in frame.columns if c and not c.startswith("Unnamed:")]]
    frame = frame.fillna("").apply(lambda col: col.str.strip())
    return frame[(frame != "").any(axis=1)].reset_index(drop=True)


def iso_date(text: str, day_first: bool) -> str:
    """One date cell in ISO form; '' stays ''. Raises with the cell on failure."""
    if not text:
        return ""
    try:
        stamp = pd.to_datetime(text, dayfirst=day_first, format="mixed")
    except (ValueError, TypeError, OverflowError) as exc:
        raise SpecimenError(f"{text!r} is not a date") from exc
    if stamp.tzinfo is not None:
        stamp = stamp.tz_localize(None)
    return stamp.strftime("%Y-%m-%d %H:%M") if (stamp.hour or stamp.minute) else stamp.strftime("%Y-%m-%d")


def _number(text: str) -> float | None:
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _fmt(value: float) -> str:
    return f"{value:.6g}"


# ----------------------------------------------------------------------
# Raw files
# ----------------------------------------------------------------------
DEFAULT_RAW_REGEX = r"_(\d+)_(\d+)\.csv$"


def index_raw_files(
    raw_root: Path, subdir: str, regex: str
) -> tuple[dict[str, Path], list[str], list[str]]:
    """Test number -> raw file, plus unparseable names and unused re-exports.

    The regex's first group is the test number; an optional second group is
    the export count, and the lowest is used when one test was exported twice.
    """
    folder = raw_root / subdir if subdir else raw_root
    pattern = re.compile(regex)
    found: dict[str, list[tuple[int, Path]]] = {}
    unparsed: list[str] = []
    if folder.exists():
        for path in sorted(folder.glob("*.csv")):
            match = pattern.search(path.name)
            if not match:
                unparsed.append(path.name)
                continue
            number = str(int(match.group(1)))
            export = int(match.group(2)) if match.re.groups >= 2 and match.group(2) else 1
            found.setdefault(number, []).append((export, path))
    chosen, extra = {}, []
    for number, files in found.items():
        files.sort()
        chosen[number] = files[0][1]
        extra.extend(p.name for _, p in files[1:])
    return chosen, unparsed, extra


# ----------------------------------------------------------------------
# One source
# ----------------------------------------------------------------------
def import_source(
    source: dict[str, Any], config: dict[str, Any], root: Path, raw_root: Path
) -> tuple[pd.DataFrame, SourceReport]:
    name = source["name"]
    report = SourceReport(name)
    sheet = read_sheet(root / source["metadata"])
    report.rows_in_sheet = len(sheet)

    number_col = source.get("test_number_column", "test no")
    matches = [c for c in sheet.columns if _key(c) == _key(number_col)]
    if not matches:
        raise SpecimenError(f"[{name}] no {number_col!r} column in {source['metadata']}")
    number_col = matches[0]

    mapping = map_columns([c for c in sheet.columns if c != number_col], source.get("columns", {}))
    report.column_map = mapping
    day_first = str(source.get("date_order", config.get("date_order", "MDY"))).upper().startswith("D")

    raw_dir = source.get("raw_dir", "")
    raw_files, unparsed, extra = index_raw_files(
        raw_root, raw_dir, source.get("raw_file_regex", DEFAULT_RAW_REGEX)
    )
    report.extra_exports = extra

    exclude = {
        _key(col): str(text).lower()
        for col, text in source.get("exclude_where", {}).items()
        if not col.startswith("_")
    }
    defaults = {**config.get("defaults", {}), **source.get("defaults", {})}
    constants = {k: v for k, v in source.get("constant", {}).items() if not k.startswith("_")}
    digits = {k: v for k, v in source.get("split_digits", {}).items() if not k.startswith("_")}
    bulk = config.get("bulk_density_g_cm3")

    rows: list[dict[str, str]] = []
    claimed: set[str] = set()
    for index, record in sheet.iterrows():
        number_text = record[number_col]
        number = _number(number_text)
        if number is None:
            raise SpecimenError(f"[{name}] sheet row {index + 2}: test number {number_text!r} is not a number")
        number_key = str(int(number))

        reason = next(
            (record[col] for col in sheet.columns
             if _key(col) in exclude and exclude[_key(col)] in record[col].lower()),
            None,
        )
        if reason is not None:
            report.excluded.append(f"test {number_key}: {reason}")
            claimed.add(number_key)
            continue
        raw = raw_files.get(number_key)
        if raw is None:
            report.no_raw_file.append(number_key)
            continue
        claimed.add(number_key)

        row: dict[str, str] = {k: str(v) for k, v in defaults.items()}
        row.update({k: str(v) for k, v in constants.items()})
        for column, target in mapping.items():
            if target is None:
                continue
            value = record[column]
            if target in _DATE_TARGETS and value:
                try:
                    value = iso_date(value, day_first)
                except SpecimenError as exc:
                    raise SpecimenError(f"[{name}] sheet row {index + 2}, {column!r}: {exc}") from exc
            row[target] = value
        for column, names in digits.items():
            code = record.get(column, "")
            if code and len(code) == len(names) and code.isdigit():
                row.update({n: d for n, d in zip(names, code)})
            elif code:
                raise SpecimenError(
                    f"[{name}] sheet row {index + 2}: {column!r} = {code!r} does not split into "
                    f"{len(names)} digits ({', '.join(names)})"
                )

        # A batch number is a name, not a quantity: batch 3 is not "more" than
        # batch 2. As text it is analysed as a categorical factor.
        batch = row.get("print_batch", "")
        if batch and _number(batch) is not None:
            row["print_batch"] = f"B{batch}"

        x, y = _number(row.get("bed_x", "")), _number(row.get("bed_y", ""))
        if x is not None and y is not None and source.get("derive_bed_r", True):
            row["bed_r"] = _fmt(math.hypot(x, y))
        mass = _number(row.get("mass_g", ""))
        side, height = _number(row.get("side_length_mm", "")), _number(row.get("initial_height_mm", ""))
        if mass and bulk and side and height:
            volume_cm3 = side * side * height / 1000.0
            row["relative_density"] = _fmt(mass / (volume_cm3 * float(bulk)))

        row["test_id"] = f"{name}_{int(number):03d}"
        row["raw_file"] = raw.relative_to(raw_root).as_posix()
        if not row.get("pattern_name"):
            row["pattern_name"] = row["test_id"]
        rows.append(row)

    report.unclaimed_files = sorted(
        unparsed + [p.name for n, p in raw_files.items() if n not in claimed]
    )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        # Replicate = order of the test within its pattern.
        frame["replicate"] = (frame.groupby("pattern_name").cumcount() + 1).astype(str)
        frame["pattern_params"] = "{}"
    report.imported = len(frame)
    return frame, report


# ----------------------------------------------------------------------
# All sources -> specimens.csv
# ----------------------------------------------------------------------
def build_specimens(
    config_path: str | Path, root: str | Path, existing: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, list[SourceReport], list[str]]:
    """Every source's rows, plus any hand-entered rows already in specimens.csv.

    A row in the existing file is replaced if its test_id belongs to a source
    (``<source name>_...``) and kept otherwise, so tests registered by hand
    survive a re-import while removed or excluded tests do not linger.
    """
    root = Path(root)
    config = load_config(config_path)
    raw_root = root / "data" / "raw"
    frames, reports = [], []
    for source in config["sources"]:
        frame, report = import_source(source, config, root, raw_root)
        frames.append(frame)
        reports.append(report)

    prefixes = tuple(f"{s['name']}_" for s in config["sources"])
    kept: list[str] = []
    if existing is not None and not existing.empty:
        manual = existing[~existing["test_id"].astype(str).str.startswith(prefixes)]
        if not manual.empty:
            frames.insert(0, manual)
            kept = list(manual["test_id"].astype(str))

    frames = [f for f in frames if not f.empty]
    combined = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if combined.empty:
        return combined, reports, kept
    ids = combined["test_id"]
    if ids.duplicated().any():
        raise SpecimenError(f"duplicate test_id(s): {sorted(set(ids[ids.duplicated()]))}")

    core = [c for c in list(COLUMNS) + list(OPTIONAL_COLUMNS)]
    for column in core:
        if column not in combined.columns:
            combined[column] = ""
    extras = [c for c in combined.columns if c not in core]
    return combined[core + extras].fillna(""), reports, kept
