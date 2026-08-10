"""Stress/strain conversion and curve utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.mechanics import (
    StressStrainCurve,
    cumulative_energy,
    energy_to_strain,
    resample_curve,
    tangent_modulus,
    to_stress_strain,
)
from src.specimens import Specimen

SPEC = Specimen(
    test_id="t1",
    raw_file="t1.csv",
    pattern_name="solid",
    replicate=1,
    side_length_mm=20.0,
    cross_section_area_mm2=400.0,
    initial_height_mm=20.0,
    material="PLA",
)


def test_conversion_uses_height_and_area():
    frame = pd.DataFrame(
        {"time_s": [0.0, 1.0], "disp_mm": [0.0, 2.0], "force_N": [0.0, 800.0]}
    )
    curve = to_stress_strain(frame, SPEC)
    assert curve.strain[-1] == pytest.approx(0.1)          # 2 mm / 20 mm
    assert curve.stress_MPa[-1] == pytest.approx(2.0)      # 800 N / 400 mm^2


def test_stress_units_are_MPa():
    """N/mm^2 is MPa -- the identity the whole pipeline relies on."""
    frame = pd.DataFrame({"time_s": [0.0], "disp_mm": [0.0], "force_N": [400.0]})
    curve = to_stress_strain(frame, SPEC)
    assert curve.stress_MPa[0] == pytest.approx(1.0)


def test_zero_geometry_is_rejected():
    bad = Specimen(**{**SPEC.__dict__, "initial_height_mm": 0.0})
    frame = pd.DataFrame({"time_s": [0.0], "disp_mm": [0.0], "force_N": [0.0]})
    with pytest.raises(ValueError, match="initial_height_mm"):
        to_stress_strain(frame, bad)


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError, match="stress_MPa"):
        StressStrainCurve(
            test_id="t",
            strain=np.zeros(5),
            stress_MPa=np.zeros(4),
            force_N=np.zeros(5),
            disp_mm=np.zeros(5),
            time_s=np.zeros(5),
        )


def test_cumulative_energy_matches_analytic_integral():
    strain = np.linspace(0, 1, 1001)
    stress = 2.0 * strain            # integral = strain^2
    energy = cumulative_energy(strain, stress)
    assert energy[-1] == pytest.approx(1.0, rel=1e-6)
    assert energy[0] == 0.0


def test_energy_to_strain_interpolates_the_endpoint():
    strain = np.linspace(0, 1, 1001)
    stress = np.full_like(strain, 3.0)
    assert energy_to_strain(strain, stress, 0.5) == pytest.approx(1.5, rel=1e-6)


def test_energy_to_strain_clips_past_the_data():
    strain = np.linspace(0, 0.4, 401)
    stress = np.full_like(strain, 2.0)
    assert energy_to_strain(strain, stress, 0.9) == pytest.approx(0.8, rel=1e-6)


def test_tangent_modulus_recovers_a_known_slope():
    strain = np.linspace(0, 0.5, 2000)
    stress = 12.0 * strain + 1.0
    tangent = tangent_modulus(strain, stress, 0.01)
    middle = tangent[500:1500]
    assert np.nanmean(middle) == pytest.approx(12.0, rel=1e-6)


def test_tangent_modulus_tracks_a_changing_slope():
    strain = np.linspace(0, 1.0, 4000)
    stress = np.where(strain < 0.5, 2.0 * strain, 1.0 + 20.0 * (strain - 0.5))
    tangent = tangent_modulus(strain, stress, 0.01)
    assert np.nanmean(tangent[500:1500]) == pytest.approx(2.0, rel=0.05)
    assert np.nanmean(tangent[2500:3500]) == pytest.approx(20.0, rel=0.05)


def test_resample_is_nan_outside_the_measured_range():
    strain = np.linspace(0, 0.3, 100)
    stress = strain * 2
    grid = np.linspace(0, 0.6, 7)
    out = resample_curve(strain, stress, grid)
    assert np.isfinite(out[:4]).all()
    assert np.isnan(out[4:]).all()


def test_resample_handles_duplicate_strains():
    strain = np.array([0.0, 0.1, 0.1, 0.2])
    stress = np.array([0.0, 1.0, 3.0, 4.0])
    out = resample_curve(strain, stress, np.array([0.0, 0.1, 0.2]))
    assert out[1] == pytest.approx(2.0)   # duplicates averaged


def test_shifted_drops_negative_strain():
    curve = StressStrainCurve(
        test_id="t",
        strain=np.linspace(0, 0.5, 100),
        stress_MPa=np.linspace(0, 5, 100),
        force_N=np.linspace(0, 2000, 100),
        disp_mm=np.linspace(0, 10, 100),
        time_s=np.linspace(0, 50, 100),
    )
    shifted = curve.shifted(0.1)
    assert shifted.strain.min() >= 0
    assert len(shifted.strain) < len(curve.strain)
