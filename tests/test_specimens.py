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


# ------------------------------------------------- extra parameter columns ----
def write_with_extra(path: Path, extra_header: list[str], extra_rows: list[list[str]]):
    """GOOD_ROW (one per extra row) followed by user-added parameter columns."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(COLUMNS) + extra_header)
        for i, extra in enumerate(extra_rows):
            row = dict(GOOD_ROW, test_id=f"t{i}", raw_file=f"t{i}.csv")
            writer.writerow([row[c] for c in COLUMNS] + extra)
    return path


def test_extra_columns_become_parameters(tmp_path):
    """A parameter per column is easier than a JSON cell per row."""
    path = write_with_extra(
        tmp_path / "specimens.csv",
        ["layer_height_mm", "orientation"],
        [["0.2", "XY"], ["0.28", "Z"]],
    )
    spec = load_specimens(path)[0]
    assert spec.parameters()["layer_height_mm"] == pytest.approx(0.2)
    assert spec.numeric_params()["layer_height_mm"] == pytest.approx(0.2)
    assert spec.categorical_params() == {"orientation": "XY"}
    # JSON parameters are still there alongside them.
    assert spec.numeric_params()["cell_mm"] == pytest.approx(5.0)


def test_column_type_is_decided_per_column(tmp_path):
    """One stray text cell makes the whole column categorical, not half of it."""
    path = write_with_extra(
        tmp_path / "specimens.csv", ["nozzle"], [["0.4"], ["0.6"], ["hardened"]]
    )
    specs = load_specimens(path)
    assert all("nozzle" in s.categorical_params() for s in specs)
    assert all("nozzle" not in s.numeric_params() for s in specs)


def test_blank_extra_cells_are_missing_not_zero(tmp_path):
    path = write_with_extra(tmp_path / "specimens.csv", ["infill"], [["15"], [""]])
    specs = load_specimens(path)
    assert specs[0].numeric_params()["infill"] == pytest.approx(15.0)
    assert "infill" not in specs[1].parameters()


def test_spreadsheet_headers_are_sanitised(tmp_path):
    path = write_with_extra(tmp_path / "specimens.csv", ["Layer Height (mm)"], [["0.2"]])
    assert "Layer_Height_mm" in load_specimens(path)[0].numeric_params()


def test_excel_trailing_delimiter_columns_are_ignored(tmp_path):
    path = write_with_extra(tmp_path / "specimens.csv", ["speed", ""], [["60", ""]])
    params = load_specimens(path)[0].parameters()
    assert "speed" in params
    assert not any(k.startswith("Unnamed") for k in params)


def test_parameter_given_twice_must_agree(tmp_path):
    clash = write_with_extra(tmp_path / "clash.csv", ["cell_mm"], [["6"]])
    with pytest.raises(SpecimenError, match="given twice"):
        load_specimens(clash)

    agree = write_with_extra(tmp_path / "agree.csv", ["cell_mm"], [["5.0"]])
    assert load_specimens(agree)[0].numeric_params()["cell_mm"] == pytest.approx(5.0)


def test_headers_that_collide_after_sanitising_are_rejected(tmp_path):
    path = write_with_extra(tmp_path / "s.csv", ["cell mm", "cell-mm"], [["5", "5"]])
    with pytest.raises(SpecimenError, match="both become parameter"):
        load_specimens(path)
