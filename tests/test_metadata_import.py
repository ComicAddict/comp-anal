"""Building specimens.csv from the lab's metadata sheets."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.metadata_import import build_specimens, iso_date, map_columns
from src.specimens import SpecimenError, load_specimens

FIXTURE = Path(__file__).parent / "fixtures" / "solid_compression_20260717_192209_1_1.csv"

PLAIN = """test no,pattern,date,manufacturing group,bed x,bed y,discrepancy
1,323,8-6-26,1,-4,4,
2,002,8-6-26,3,3,4,could be pattern 020
3,222,8-9-26,1,0,0,wrong-test-direction-needs-retest
4,222,8-9-26,4,0,0,
5,101,8-9-26,2,1,1,
"""

TPMS = """pattern name,scale,weight,manufactured time,test time,test no
gyroid,1,4.10,11/4/2026 22:40:00,11/6/2026 6:00:00,1
gyroid,2,4.20,11/4/2026 22:40:00,11/6/2026 0:53:00,2
"""


def make_project(tmp_path: Path, sources: list[dict] | None = None) -> Path:
    (tmp_path / "data" / "metadata").mkdir(parents=True)
    (tmp_path / "data" / "metadata" / "plain.csv").write_text(PLAIN)
    (tmp_path / "data" / "metadata" / "tpms.csv").write_text(TPMS)
    for folder, numbers in (("plain", [1, 2, 3, 4]), ("tpms", [1, 2])):
        raw = tmp_path / "data" / "raw" / folder
        raw.mkdir(parents=True)
        for n in numbers:
            (raw / f"solid_compression_20260806_101500_{n}_1.csv").write_bytes(FIXTURE.read_bytes())
    # A second export of plain test 1, and a file no sheet row claims.
    raw = tmp_path / "data" / "raw" / "plain"
    (raw / "solid_compression_20260806_101500_1_2.csv").write_bytes(FIXTURE.read_bytes())
    (raw / "solid_compression_20260806_101500_9_1.csv").write_bytes(FIXTURE.read_bytes())
    config = {
        "date_order": "MDY",
        "defaults": {"side_length_mm": 16, "initial_height_mm": 16, "material": "PLA",
                     "relative_density": 0.8},
        "bulk_density_g_cm3": 1.24,
        "sources": sources or [
            {"name": "plain", "metadata": "data/metadata/plain.csv", "raw_dir": "plain",
             "constant": {"family": "plain"},
             "columns": {"pattern": "pattern_name", "date": "test_date"},
             "split_digits": {"pattern": ["code_1", "code_2", "code_3"]},
             "exclude_where": {"discrepancy": "retest"}},
            {"name": "tpms", "metadata": "data/metadata/tpms.csv", "raw_dir": "tpms",
             "constant": {"family": "tpms"}, "columns": {"pattern name": "tpms_type"}},
        ],
    }
    (tmp_path / "metadata_sources.json").write_text(json.dumps(config))
    return tmp_path


@pytest.fixture
def built(tmp_path):
    root = make_project(tmp_path)
    frame, reports, kept = build_specimens(root / "metadata_sources.json", root)
    return root, frame.set_index("test_id"), {r.name: r for r in reports}


def test_rows_are_matched_to_raw_files_by_test_number(built):
    _, frame, reports = built
    assert frame.loc["plain_002", "raw_file"] == "plain/solid_compression_20260806_101500_2_1.csv"
    assert reports["plain"].no_raw_file == ["5"]
    assert reports["plain"].unclaimed_files == ["solid_compression_20260806_101500_9_1.csv"]
    assert reports["plain"].extra_exports == ["solid_compression_20260806_101500_1_2.csv"]


def test_pattern_codes_keep_leading_zeros_and_split_into_digits(built):
    _, frame, _ = built
    row = frame.loc["plain_002"]
    assert row["pattern_name"] == "002"
    assert (row["code_1"], row["code_2"], row["code_3"]) == ("0", "0", "2")


def test_flagged_rows_are_excluded_and_reported(built):
    _, frame, reports = built
    assert "plain_003" not in frame.index
    assert reports["plain"].excluded == ["test 3: wrong-test-direction-needs-retest"]
    # The retest of the same pattern is replicate 1 once the bad test is out.
    assert frame.loc["plain_004", "replicate"] == "1"


def test_dates_are_written_in_iso_form_month_first(built):
    _, frame, _ = built
    assert frame.loc["plain_001", "test_date"] == "2026-08-06"
    assert frame.loc["tpms_001", "print_date"] == "2026-11-04 22:40"
    assert frame.loc["tpms_001", "test_date"] == "2026-11-06 06:00"
    assert iso_date("8-6-26", day_first=True) == "2026-06-08"


def test_columns_are_recognised_by_name(built):
    _, frame, _ = built
    assert frame.loc["plain_001", "print_batch"] == "B1", "batch numbers become labels"
    assert frame.loc["plain_002", "notes"] == "could be pattern 020"
    assert frame.loc["tpms_002", "mass_g"] == "4.20"
    assert frame.loc["tpms_002", "scale"] == "2", "unknown columns become parameters"


def test_derived_columns(built):
    _, frame, _ = built
    assert float(frame.loc["plain_001", "bed_r"]) == pytest.approx(32 ** 0.5)
    # Measured mass -> relative density: 4.10 g / (4.096 cm3 * 1.24 g/cm3).
    assert float(frame.loc["tpms_001", "relative_density"]) == pytest.approx(4.10 / (4.096 * 1.24), rel=1e-4)
    assert frame.loc["plain_001", "relative_density"] == "0.8", "default without a mass"


def test_the_result_is_a_valid_specimens_file(built, tmp_path):
    root, frame, _ = built
    path = root / "data" / "specimens.csv"
    frame.reset_index().to_csv(path, index=False)
    specs = {s.test_id: s for s in load_specimens(path, root / "data" / "raw")}
    assert specs["plain_002"].categorical_params()["print_batch"] == "B3"
    assert specs["tpms_001"].print_to_test_days() == pytest.approx(1 + 7.333 / 24, abs=1e-3)


def test_hand_registered_rows_survive_a_reimport(tmp_path):
    root = make_project(tmp_path)
    existing = pd.DataFrame([
        {"test_id": "pilot_1", "raw_file": "x.csv"},
        {"test_id": "plain_099", "raw_file": "stale.csv"},
    ])
    frame, _, kept = build_specimens(root / "metadata_sources.json", root, existing)
    assert kept == ["pilot_1"]
    assert "pilot_1" in set(frame["test_id"]) and "plain_099" not in set(frame["test_id"])


def test_two_columns_for_one_field_is_an_error():
    with pytest.raises(SpecimenError, match="same field"):
        map_columns(["date", "testing date"], {"date": "test_date"})
    mapping = map_columns(["date", "testing date"], {"date": None})
    assert mapping == {"date": None, "testing date": "test_date"}


def test_a_code_that_does_not_split_is_an_error(tmp_path):
    root = make_project(tmp_path)
    sheet = root / "data" / "metadata" / "plain.csv"
    sheet.write_text(PLAIN.replace("1,323,", "1,32,"))
    with pytest.raises(SpecimenError, match="does not split"):
        build_specimens(root / "metadata_sources.json", root)
