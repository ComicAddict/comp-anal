"""Shared test fixtures.

The parser fixture prefers a *real* instrument export if one is present in
`tests/fixtures/`, falling back to the synthetic file that reproduces the
documented format. Drop the genuine
`solid_compression_20260717_192209_1_1.csv` over the synthetic one and the
suite immediately validates the parser against real instrument output --
no test changes needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DEFAULT_CONFIG  # noqa: E402
from src.synthetic import (  # noqa: E402
    SAMPLE_FILENAME,
    FoamCurveSpec,
    SyntheticTest,
    foam_stress,
    sample_test,
)

FIXTURES = Path(__file__).parent / "fixtures"


#: A real export that omits the "Results Table" boilerplate entirely.
NO_BOILERPLATE_FILENAME = "no_boilerplate_solid_compression_20260717_192209_5_2.csv"


@pytest.fixture(scope="session")
def sample_csv() -> Path:
    """A real instrument export from the 10 kN frame (1 kN/min, 9.5 kN limit)."""
    path = FIXTURES / SAMPLE_FILENAME
    if not path.exists():  # fall back to the synthetic stand-in
        sample_test().write(path)
    return path


@pytest.fixture(scope="session")
def no_boilerplate_csv() -> Path:
    """A real export whose header is not preceded by the boilerplate block."""
    path = FIXTURES / NO_BOILERPLATE_FILENAME
    if not path.exists():
        pytest.skip(f"{NO_BOILERPLATE_FILENAME} not present")
    return path


@pytest.fixture(scope="session")
def duplicate_of_no_boilerplate_csv() -> Path:
    """The same measurements as `no_boilerplate_csv`, exported with boilerplate."""
    path = Path(__file__).parent.parent / "data" / "raw" / (
        "solid_compression_20260717_192209_5_1.csv"
    )
    if not path.exists():
        pytest.skip("paired export not present in data/raw/")
    return path


@pytest.fixture
def cfg():
    return DEFAULT_CONFIG


@pytest.fixture(scope="session")
def clean_curve() -> tuple[np.ndarray, np.ndarray, FoamCurveSpec]:
    """A noise-free stress-strain curve with exactly known properties."""
    spec = FoamCurveSpec()
    strain = np.linspace(0.0, 0.60, 4000)
    return strain, foam_stress(strain, spec), spec


def make_dataset(root: Path, patterns: dict[str, dict], replicates: int = 2) -> Path:
    """Build a throwaway repository: raw CSVs plus a matching specimens.csv.

    `patterns` maps a pattern name to the FoamCurveSpec keyword overrides for
    that pattern, so a test can create a small design space in one call.
    """
    import csv

    from src.specimens import COLUMNS

    raw_dir = root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for p_index, (pattern, overrides) in enumerate(patterns.items()):
        for rep in range(1, replicates + 1):
            name = f"{pattern}_compression_2026071{p_index}_1200{rep}0_1_{rep}.csv"
            test = SyntheticTest(
                name=name,
                curve=FoamCurveSpec(**overrides),
                seed=1000 * p_index + rep,
            )
            test.write(raw_dir / name)
            rows.append(
                {
                    "test_id": Path(name).stem,
                    "raw_file": name,
                    "pattern_name": pattern,
                    "pattern_params": '{"cell_mm": %.1f}' % (4.0 + p_index),
                    "replicate": rep,
                    "side_length_mm": test.side_length_mm,
                    "cross_section_area_mm2": "",
                    "initial_height_mm": test.initial_height_mm,
                    "relative_density": round(0.25 + 0.05 * p_index, 3),
                    "mass_g": round(2.7 + 0.4 * p_index, 2),
                    "material": "PLA",
                    "test_date": "2026-07-17",
                    "notes": "",
                }
            )

    specimens = root / "data" / "specimens.csv"
    with specimens.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    return specimens


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    """A two-pattern, two-replicate design space in a temp directory."""
    make_dataset(
        tmp_path,
        {
            "solid": {},
            "gyroid_v2": {
                "modulus_MPa": 25.0,
                "peak_stress_MPa": None,
                "plateau_stress_MPa": 1.1,
                "densification_strain": 0.48,
            },
        },
    )
    return tmp_path
