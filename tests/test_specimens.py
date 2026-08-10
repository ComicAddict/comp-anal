"""Validation of the specimens.csv metadata table."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest

from src.specimens import (
    COLUMNS,
    SpecimenError,
    load_specimens,
    read_specimens_frame,
    to_specimens,
    unregistered_files,
    validate_frame,
)

GOOD_ROW = {
    "test_id": "t1",
    "raw_file": "t1.csv",
    "pattern_name": "gyroid_v2",
    "pattern_params": '{"cell_mm": 5.0, "wall_mm": 0.8}',
    "replicate": "1",
    "side_length_mm": "20",
    "cross_section_area_mm2": "",
    "initial_height_mm": "20",
    "relative_density": "0.31",
    "mass_g": "3.2",
    "material": "PLA",
    "test_date": "2026-07-17",
    "notes": "",
}


def write_csv(path: Path, rows: list[dict]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_valid_table_parses(tmp_path):
    path = write_csv(tmp_path / "specimens.csv", [GOOD_ROW])
    specs = load_specimens(path)
    assert len(specs) == 1
    spec = specs[0]
    assert spec.pattern_name == "gyroid_v2"
    assert spec.pattern_params == {"cell_mm": 5.0, "wall_mm": 0.8}
    assert spec.numeric_params() == {"cell_mm": 5.0, "wall_mm": 0.8}


def test_area_derived_from_side_length_when_blank(tmp_path):
    path = write_csv(tmp_path / "specimens.csv", [GOOD_ROW])
    spec = load_specimens(path)[0]
    assert spec.cross_section_area_mm2 == pytest.approx(400.0)
    assert spec.volume_mm3 == pytest.approx(8000.0)


def test_area_override_is_respected(tmp_path):
    row = dict(GOOD_ROW, cross_section_area_mm2="350")
    path = write_csv(tmp_path / "specimens.csv", [row])
    assert load_specimens(path)[0].cross_section_area_mm2 == pytest.approx(350.0)


@pytest.mark.parametrize(
    "field", ["test_id", "pattern_name", "replicate", "side_length_mm", "material"]
)
def test_missing_required_value_is_rejected(tmp_path, field):
    row = dict(GOOD_ROW, **{field: ""})
    path = write_csv(tmp_path / "specimens.csv", [row])
    with pytest.raises(SpecimenError, match=field):
        load_specimens(path)


def test_missing_column_is_rejected(tmp_path):
    path = tmp_path / "specimens.csv"
    path.write_text("test_id,raw_file\nt1,t1.csv\n")
    with pytest.raises(SpecimenError, match="missing required column"):
        validate_frame(read_specimens_frame(path))


def test_duplicate_test_id_is_rejected(tmp_path):
    path = write_csv(tmp_path / "specimens.csv", [GOOD_ROW, dict(GOOD_ROW)])
    with pytest.raises(SpecimenError, match="duplicate test_id"):
        load_specimens(path)


def test_non_numeric_geometry_is_rejected(tmp_path):
    row = dict(GOOD_ROW, side_length_mm="twenty")
    path = write_csv(tmp_path / "specimens.csv", [row])
    with pytest.raises(SpecimenError, match="must be numeric"):
        load_specimens(path)


def test_bad_json_params_are_rejected(tmp_path):
    row = dict(GOOD_ROW, pattern_params="{cell_mm: 5}")
    path = write_csv(tmp_path / "specimens.csv", [row])
    with pytest.raises(SpecimenError, match="not valid JSON"):
        load_specimens(path)


def test_blank_params_are_allowed(tmp_path):
    row = dict(GOOD_ROW, pattern_params="")
    path = write_csv(tmp_path / "specimens.csv", [row])
    assert load_specimens(path)[0].pattern_params == {}


def test_optional_fields_may_be_blank(tmp_path):
    row = dict(GOOD_ROW, relative_density="", mass_g="", notes="", test_date="")
    path = write_csv(tmp_path / "specimens.csv", [row])
    spec = load_specimens(path)[0]
    assert spec.relative_density is None
    assert spec.mass_g is None
    assert spec.test_date is None


def test_empty_table_is_rejected(tmp_path):
    path = write_csv(tmp_path / "specimens.csv", [])
    with pytest.raises(SpecimenError, match="no rows"):
        load_specimens(path)


def test_missing_file_gives_actionable_error(tmp_path):
    with pytest.raises(SpecimenError, match="register_test.py"):
        load_specimens(tmp_path / "nope.csv")


def test_missing_raw_file_is_reported(tmp_path):
    path = write_csv(tmp_path / "specimens.csv", [GOOD_ROW])
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    with pytest.raises(SpecimenError, match="missing raw files"):
        load_specimens(path, raw_dir)


def test_unregistered_files_lists_only_new_ones(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "t1.csv").write_text("x")
    (raw_dir / "t2.csv").write_text("x")
    path = write_csv(tmp_path / "specimens.csv", [GOOD_ROW])

    pending = unregistered_files(raw_dir, path)
    assert [p.name for p in pending] == ["t2.csv"]


def test_numeric_params_ignores_non_numeric_values(tmp_path):
    row = dict(GOOD_ROW, pattern_params='{"cell_mm": 5, "kind": "auxetic", "on": true}')
    path = write_csv(tmp_path / "specimens.csv", [row])
    assert load_specimens(path)[0].numeric_params() == {"cell_mm": 5.0}
