"""Parser tests against the real instrument quirks documented in the brief."""

from __future__ import annotations

import numpy as np
import pytest

from src.config import DEFAULT_CONFIG
from src.io import (
    RawParseError,
    UnitsError,
    check_monotonic,
    check_sampling,
    find_contact_index,
    load_test,
    locate_header,
    read_raw_csv,
    trim_preload,
)
from src.synthetic import BOILERPLATE, HEADER, UNITS


# ---------------------------------------------------------------- format ----
def test_parses_sample_file(sample_csv):
    frame = read_raw_csv(sample_csv)
    assert list(frame.columns) == ["time_s", "disp_mm", "force_kN", "force_N"]
    assert len(frame) > 100
    assert frame.dtypes["time_s"] == float
    assert frame.dtypes["force_kN"] == float


def test_skips_boilerplate_and_units_rows(sample_csv):
    """No 'Results Table' or '(s)' string survives into the data."""
    frame = read_raw_csv(sample_csv)
    assert frame["time_s"].iloc[0] == pytest.approx(0.0, abs=1e-9)
    assert np.isfinite(frame.to_numpy(dtype=float)).all()


def test_header_located_by_scanning_not_by_fixed_offset(tmp_path):
    """An extra blank line in the export must not corrupt the parse."""
    path = tmp_path / "extra_blank.csv"
    lines = BOILERPLATE + ["", ""] + [HEADER, UNITS, '"0.0","0.0","0.0"', '"0.02","0.1","0.5"']
    path.write_text("\n".join(lines) + "\n")

    assert locate_header(lines) == len(BOILERPLATE) + 2
    frame = read_raw_csv(path)
    assert len(frame) == 2
    assert frame["force_kN"].iloc[1] == pytest.approx(0.5)


def test_quoted_values_are_cast_to_float(sample_csv):
    frame = read_raw_csv(sample_csv)
    assert isinstance(float(frame["disp_mm"].iloc[10]), float)
    assert frame["disp_mm"].to_numpy().dtype == np.float64


def test_force_converted_to_newtons(sample_csv):
    frame = read_raw_csv(sample_csv)
    assert np.allclose(frame["force_N"], frame["force_kN"] * 1000.0)


# ----------------------------------------------------------------- units ----
def test_unexpected_units_fail_loudly(tmp_path):
    """A silent kN -> N change would rescale every result by 1000."""
    path = tmp_path / "wrong_units.csv"
    path.write_text(
        "\n".join(BOILERPLATE + [HEADER, "(s),(mm),(N)", '"0.0","0.0","0.0"']) + "\n"
    )
    with pytest.raises(UnitsError, match="Unexpected units row"):
        read_raw_csv(path)


def test_missing_header_raises(tmp_path):
    path = tmp_path / "no_header.csv"
    path.write_text("Results Table 1\n\n\n1,2,3\n")
    with pytest.raises(RawParseError, match="Could not find"):
        read_raw_csv(path)


def test_non_numeric_data_raises_with_location(tmp_path):
    path = tmp_path / "corrupt.csv"
    path.write_text(
        "\n".join(
            BOILERPLATE + [HEADER, UNITS, '"0.0","0.0","0.0"', '"0.02","BAD","0.5"']
        )
        + "\n"
    )
    with pytest.raises(RawParseError, match="non-numeric"):
        read_raw_csv(path)


# -------------------------------------------------------------- sampling ----
def test_sampling_is_not_assumed_uniform(sample_csv):
    """The documented 2.2600 -> 2.2820 step must be measured, not assumed."""
    frame = read_raw_csv(sample_csv)
    dt = np.diff(frame["time_s"].to_numpy())
    assert np.median(dt) == pytest.approx(0.02, abs=1e-6)
    assert dt.max() > 0.02  # the irregular interval is present
    assert not np.allclose(dt, dt[0])

    diags = check_sampling(frame["time_s"].to_numpy(), DEFAULT_CONFIG)
    codes = {d.code for d in diags}
    assert "sampling" in codes
    assert "nonuniform_sampling" in codes


def test_sampling_gap_is_flagged():
    time = np.concatenate([np.arange(0, 1, 0.02), np.arange(5, 6, 0.02)])
    codes = {d.code for d in check_sampling(time, DEFAULT_CONFIG)}
    assert "sampling_gap" in codes


def test_time_reset_is_flagged():
    time = np.array([0.0, 0.02, 0.04, 0.01, 0.03])
    diags = check_sampling(time, DEFAULT_CONFIG)
    assert any(d.code == "time_reset" and d.level == "warning" for d in diags)


def test_non_monotonic_displacement_is_flagged_not_dropped():
    disp = np.array([0.0, 0.1, 0.2, 0.15, 0.3])
    diags = check_monotonic(disp, DEFAULT_CONFIG)
    assert len(diags) == 1
    assert diags[0].code == "non_monotonic_displacement"
    assert "kept as-is" in diags[0].message


# --------------------------------------------------------------- preload ----
def test_contact_detection_ignores_isolated_noise_spikes():
    """A single spike above threshold must not be mistaken for contact."""
    force = np.concatenate(
        [
            np.zeros(50),
            [500.0],          # one-sample spike
            np.zeros(49),
            np.linspace(0, 1000, 200),  # the real ramp
        ]
    )
    idx, threshold = find_contact_index(force, DEFAULT_CONFIG)
    assert idx > 100, "contact was triggered by the isolated spike"
    assert force[idx] > threshold


def test_trim_zeroes_displacement_at_contact(sample_csv):
    loaded = load_test(sample_csv)
    assert loaded.contact_index > 0
    assert loaded.trimmed["disp_mm"].iloc[0] == pytest.approx(0.0, abs=1e-12)
    assert loaded.trimmed["time_s"].iloc[0] == pytest.approx(0.0, abs=1e-12)
    # The raw series is left untouched.
    assert loaded.raw["disp_mm"].iloc[0] != pytest.approx(
        loaded.raw["disp_mm"].iloc[loaded.contact_index]
    )
    assert len(loaded.raw) > len(loaded.trimmed)


def test_trim_does_not_modify_raw(sample_csv):
    frame = read_raw_csv(sample_csv)
    before = frame.copy(deep=True)
    trim_preload(frame, DEFAULT_CONFIG)
    assert frame.equals(before)


def test_preload_noise_is_excluded(sample_csv):
    """Pre-contact samples oscillate around zero; they must not survive."""
    loaded = load_test(sample_csv)
    pre = loaded.raw["force_N"].iloc[: loaded.contact_index]
    assert abs(pre.mean()) < 10.0            # noise, centred near zero
    assert (loaded.trimmed["disp_mm"] >= -1e-9).all()


def test_no_loading_at_all_is_flagged(tmp_path):
    path = tmp_path / "flat.csv"
    rows = [f'"{i * 0.02:.4f}","{i * 0.001:.4f}","0.0000"' for i in range(100)]
    path.write_text("\n".join(BOILERPLATE + [HEADER, UNITS] + rows) + "\n")
    loaded = load_test(path)
    assert any(d.code == "no_preload_detected" for d in loaded.diagnostics)


def test_diagnostics_are_reported_not_raised(sample_csv):
    loaded = load_test(sample_csv)
    assert loaded.diagnostics, "expected at least the sampling info diagnostic"
    assert all(d.level in {"info", "warning"} for d in loaded.diagnostics)
