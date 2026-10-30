"""Metric-extraction tests against curves with analytically known properties."""

from __future__ import annotations

import numpy as np
import pytest

from src.config import DEFAULT_CONFIG
from src.metrics import (
    densification_energy_efficiency,
    densification_iso,
    densification_tangent,
    first_peak,
    plateau_stress,
    youngs_modulus,
)
from src.synthetic import FoamCurveSpec, foam_stress


# --------------------------------------------------------------- modulus ----
def test_modulus_recovers_known_slope(clean_curve):
    strain, stress, spec = clean_curve
    fit, diags = youngs_modulus(strain, stress, DEFAULT_CONFIG)

    assert fit.modulus_MPa == pytest.approx(spec.modulus_MPa, rel=0.02)
    assert fit.r2 > 0.999
    assert fit.strain_hi <= spec.peak_strain + 1e-6, "fit leaked past the elastic region"
    assert not [d for d in diags if d.level == "warning"]


def test_modulus_window_has_a_minimum_width(clean_curve):
    """Maximising R^2 without a width floor collapses to a 2-point window."""
    strain, stress, _ = clean_curve
    fit, _ = youngs_modulus(strain, stress, DEFAULT_CONFIG)
    assert fit.n_points >= DEFAULT_CONFIG.modulus_min_window_points
    assert fit.strain_hi - fit.strain_lo >= DEFAULT_CONFIG.modulus_min_window_strain


def test_modulus_survives_noise():
    rng = np.random.default_rng(7)
    strain = np.linspace(0, 0.60, 3000)
    stress = foam_stress(strain, FoamCurveSpec()) + rng.normal(0, 0.004, strain.size)
    fit, _ = youngs_modulus(strain, stress, DEFAULT_CONFIG)
    assert fit.modulus_MPa == pytest.approx(60.0, rel=0.10)


def test_modulus_reports_toe_strain():
    """A curve shifted along strain must report that offset as toe slack."""
    offset = 0.01
    strain = np.linspace(0, 0.60, 3000)
    stress = foam_stress(np.clip(strain - offset, 0, None), FoamCurveSpec())
    fit, _ = youngs_modulus(strain, stress, DEFAULT_CONFIG)
    assert fit.toe_strain == pytest.approx(offset, abs=2e-3)


def test_modulus_flags_a_curve_with_no_loading():
    strain = np.linspace(0, 0.6, 500)
    fit, diags = youngs_modulus(strain, np.zeros_like(strain), DEFAULT_CONFIG)
    assert np.isnan(fit.modulus_MPa)
    assert any(d.code == "modulus_no_variation" and d.level == "warning" for d in diags)


def test_max_r2_alone_can_select_the_plateau(clean_curve):
    """Why 'steepest_high_r2' is the default, pinned as a regression test.

    On a clean curve the plateau is as straight as the elastic region, so the
    literal highest-R^2 window is not reliably the elastic one.
    """
    strain, stress, spec = clean_curve
    steep, _ = youngs_modulus(strain, stress, DEFAULT_CONFIG)
    assert steep.modulus_MPa == pytest.approx(spec.modulus_MPa, rel=0.02)

    plain, _ = youngs_modulus(
        strain, stress, DEFAULT_CONFIG.replace(modulus_selection="max_r2")
    )
    # Both windows are essentially perfect fits; only the slope tells them apart.
    assert plain.r2 == pytest.approx(steep.r2, abs=1e-6)


def test_unknown_modulus_selection_raises(clean_curve):
    strain, stress, _ = clean_curve
    cfg = DEFAULT_CONFIG.replace(modulus_selection="nonsense")
    with pytest.raises(ValueError, match="Unknown modulus_selection"):
        youngs_modulus(strain, stress, cfg)


# --------------------------------------------------------------- plateau ----
def test_plateau_stress_matches_analytic_mean(clean_curve):
    strain, stress, spec = clean_curve
    value, std, diags = plateau_stress(strain, stress, DEFAULT_CONFIG)

    # The synthetic plateau is linear, so its mean over [0.20, 0.30] is the
    # value at the midpoint.
    expected = spec.plateau_stress_MPa + spec.plateau_slope_MPa * (
        0.25 - spec.softening_end_strain
    )
    assert value == pytest.approx(expected, rel=0.005)
    assert std >= 0
    assert not diags


def test_plateau_window_is_configurable(clean_curve):
    """ISO 13314 proper averages 20-40%; the brief asked for 20-30%."""
    strain, stress, _ = clean_curve
    narrow, _, _ = plateau_stress(strain, stress, DEFAULT_CONFIG)
    wide, _, _ = plateau_stress(
        strain, stress, DEFAULT_CONFIG.replace(plateau_strain_hi=0.40)
    )
    assert wide > narrow  # the plateau slopes gently upward


def test_plateau_reports_when_test_stopped_too_early():
    strain = np.linspace(0, 0.10, 300)
    stress = foam_stress(strain, FoamCurveSpec())
    value, _, diags = plateau_stress(strain, stress, DEFAULT_CONFIG)
    assert np.isnan(value)
    assert any(d.code == "plateau_window_empty" for d in diags)


# ---------------------------------------------------- densification onset ----
def test_energy_efficiency_finds_the_densification_knee(clean_curve):
    strain, stress, spec = clean_curve
    result, diags = densification_energy_efficiency(strain, stress, DEFAULT_CONFIG)

    assert result.method == "energy_efficiency"
    assert not result.at_end_of_data
    # The knee sits at or just past the start of the densification rise.
    assert spec.densification_strain <= result.strain <= spec.densification_strain + 0.12
    assert not [d for d in diags if d.level == "warning"]


def test_energy_efficiency_ignores_the_near_zero_stress_spike(clean_curve):
    """eta = W/sigma blows up near zero stress; that is not densification."""
    strain, stress, _ = clean_curve
    result, _ = densification_energy_efficiency(strain, stress, DEFAULT_CONFIG)
    assert result.strain > DEFAULT_CONFIG.densification_min_strain


def test_densification_not_reached_is_flagged():
    """A test stopped inside the plateau must not report a confident value."""
    strain = np.linspace(0, 0.35, 1200)
    stress = foam_stress(strain, FoamCurveSpec())
    result, diags = densification_energy_efficiency(strain, stress, DEFAULT_CONFIG)
    assert result.at_end_of_data
    assert any(d.code == "densification_not_reached" for d in diags)


def test_iso_and_tangent_methods_agree_roughly(clean_curve):
    strain, stress, spec = clean_curve
    plateau, _, _ = plateau_stress(strain, stress, DEFAULT_CONFIG)

    iso, _ = densification_iso(strain, stress, plateau, DEFAULT_CONFIG)
    tangent, _ = densification_tangent(strain, stress, plateau, 60.0, DEFAULT_CONFIG)
    energy, _ = densification_energy_efficiency(strain, stress, DEFAULT_CONFIG)

    for result in (iso, tangent, energy):
        assert 0.35 < result.strain < 0.60, f"{result.method} gave {result.strain}"


def test_unknown_densification_method_raises(clean_curve):
    from src.metrics import densification_strain

    strain, stress, _ = clean_curve
    cfg = DEFAULT_CONFIG.replace(densification_method="nope")
    with pytest.raises(ValueError, match="Unknown densification_method"):
        densification_strain(strain, stress, 2.0, 60.0, cfg)


# ------------------------------------------------------------- first peak ----
def test_first_peak_detected_when_present(clean_curve):
    strain, stress, spec = clean_curve
    force = stress * 400.0
    result, _ = first_peak(strain, stress, force, 0.50, DEFAULT_CONFIG)

    assert result.has_distinct_first_peak
    assert result.first_peak_stress_MPa == pytest.approx(spec.peak_stress_MPa, rel=0.02)
    assert result.first_peak_strain == pytest.approx(spec.peak_strain, abs=0.005)
    assert result.crush_force_N == pytest.approx(spec.peak_stress_MPa * 400.0, rel=0.02)
    assert result.post_peak_drop_frac > 0.15


def test_no_first_peak_on_a_monotonic_curve():
    """A pattern without buckling softening must not invent a peak."""
    spec = FoamCurveSpec(peak_stress_MPa=None, modulus_MPa=25.0, plateau_stress_MPa=1.1)
    strain = np.linspace(0, 0.60, 3000)
    stress = foam_stress(strain, spec)
    result, diags = first_peak(strain, stress, stress * 400.0, 0.50, DEFAULT_CONFIG)

    assert not result.has_distinct_first_peak
    assert np.isnan(result.first_peak_stress_MPa)
    assert np.isfinite(result.crush_force_N)  # fallback still reported
    assert any(d.code == "no_first_peak" for d in diags)


def test_crush_force_excludes_the_densification_rise(clean_curve):
    """Crush load is the initial peak, not the huge densification stress."""
    strain, stress, spec = clean_curve
    result, _ = first_peak(strain, stress, stress * 400.0, 0.50, DEFAULT_CONFIG)
    assert result.crush_stress_MPa < stress.max() / 2


# ------------------------------------------------------ fixed-strain metrics ----
from src.metrics import (  # noqa: E402
    fixed_strain_metrics,
    plateau_window_coverage,
    strain_tag,
)
from src.specimens import Specimen  # noqa: E402

SPEC = Specimen(
    test_id="t",
    raw_file="t.csv",
    pattern_name="p",
    replicate=1,
    side_length_mm=20.0,
    cross_section_area_mm2=400.0,
    initial_height_mm=20.0,
    material="PLA",
    mass_g=4.0,
)


def test_strain_tag_is_a_clean_column_fragment():
    assert strain_tag(0.1) == "10pct"
    assert strain_tag(0.2) == "20pct"
    assert strain_tag(0.05) == "5pct"
    assert strain_tag(0.125) == "12.5pct"


def test_fixed_strain_metrics_match_the_curve(clean_curve):
    strain, stress, spec = clean_curve
    out = fixed_strain_metrics(strain, stress, SPEC, DEFAULT_CONFIG)

    # On the linear plateau, stress at 20% is analytic.
    expected = spec.plateau_stress_MPa + spec.plateau_slope_MPa * (
        0.20 - spec.softening_end_strain
    )
    assert out["stress_at_20pct_MPa"] == pytest.approx(expected, rel=1e-3)

    mask = strain <= 0.20
    energy = np.trapezoid(stress[mask], strain[mask])
    assert out["energy_to_20pct_MJ_m3"] == pytest.approx(energy, rel=2e-3)

    # SEA: W [MJ/m^3 = mJ/mm^3] * 8000 mm^3 / 1000 -> J, / 4 g.
    assert out["sea_to_20pct_J_per_g"] == pytest.approx(
        out["energy_to_20pct_MJ_m3"] * 8000 / 1000 / 4.0
    )


def test_fixed_strain_levels_not_reached_are_nan_not_extrapolated():
    """A test cut off at 15% has no stress at 20% -- it was never measured."""
    strain = np.linspace(0, 0.15, 500)
    stress = foam_stress(strain, FoamCurveSpec())
    out = fixed_strain_metrics(strain, stress, SPEC, DEFAULT_CONFIG)
    assert np.isfinite(out["stress_at_10pct_MPa"])
    assert np.isnan(out["stress_at_20pct_MPa"])
    assert np.isnan(out["energy_to_20pct_MJ_m3"])


def test_fixed_strain_sea_needs_a_mass(clean_curve):
    strain, stress, _ = clean_curve
    massless = Specimen(**{**SPEC.__dict__, "mass_g": None})
    out = fixed_strain_metrics(strain, stress, massless, DEFAULT_CONFIG)
    assert np.isnan(out["sea_to_20pct_J_per_g"])
    assert np.isfinite(out["energy_to_20pct_MJ_m3"])


def test_fixed_strain_levels_are_configurable(clean_curve):
    strain, stress, _ = clean_curve
    cfg = DEFAULT_CONFIG.replace(fixed_strain_levels=(0.05, 0.40))
    out = fixed_strain_metrics(strain, stress, SPEC, cfg)
    assert {"stress_at_5pct_MPa", "stress_at_40pct_MPa"} <= set(out)
    assert "stress_at_20pct_MPa" not in out


def test_energy_ignores_noise_reversals_after_the_crossing():
    """A strain dip after crossing the limit must not be folded back in."""
    from src.mechanics import energy_to_strain

    strain = np.array([0.0, 0.1, 0.2, 0.3, 0.19, 0.4])
    stress = np.array([0.0, 1.0, 1.0, 1.0, 50.0, 1.0])  # huge value on the dip
    # Integral of the clean part up to 0.2: 0.5*0.1*1 + 0.1*1 = 0.15
    assert energy_to_strain(strain, stress, 0.2) == pytest.approx(0.15)


# --------------------------------------------------------- plateau coverage ----
def test_plateau_window_coverage():
    cfg = DEFAULT_CONFIG  # window 0.20-0.30
    assert plateau_window_coverage(np.linspace(0, 0.60, 10), cfg) == pytest.approx(1.0)
    assert plateau_window_coverage(np.linspace(0, 0.28, 10), cfg) == pytest.approx(0.8)
    assert plateau_window_coverage(np.linspace(0, 0.15, 10), cfg) == pytest.approx(0.0)


def test_plateau_not_reported_when_window_mostly_uncovered():
    """Stopping at 28% would average only the low end of a 20-30% window."""
    strain = np.linspace(0, 0.28, 2000)
    stress = foam_stress(strain, FoamCurveSpec())
    value, _, diags = plateau_stress(strain, stress, DEFAULT_CONFIG)
    assert np.isnan(value)
    assert any(d.code == "plateau_window_partial" and d.level == "warning" for d in diags)


def test_plateau_reported_with_note_when_nearly_covered():
    strain = np.linspace(0, 0.295, 2000)  # 95% coverage
    stress = foam_stress(strain, FoamCurveSpec())
    value, _, diags = plateau_stress(strain, stress, DEFAULT_CONFIG)
    assert np.isfinite(value)
    assert any(d.code == "plateau_window_partial" and d.level == "info" for d in diags)


# ------------------------------------------------- densification end check ----
def test_densification_near_the_end_counts_as_not_reached():
    """A criterion still climbing at the cut-off gives a lower bound only."""
    strain = np.linspace(0, 0.35, 1200)
    stress = foam_stress(strain, FoamCurveSpec())
    noisy = stress + np.random.default_rng(3).normal(0, 0.003, stress.size)
    result, _ = densification_energy_efficiency(strain, noisy, DEFAULT_CONFIG)
    assert result.at_end_of_data


def test_noise_on_a_soft_specimen_does_not_fake_a_first_peak():
    """A monotonic curve at ~240 N with 1.5 N noise must not report a peak.

    Before smoothing, the noise produced 2.4-2.8% "drops" on 7 of 32 such
    synthetic specimens, just over the 2% threshold.
    """
    spec = FoamCurveSpec(peak_stress_MPa=None, modulus_MPa=25.0, plateau_stress_MPa=0.6)
    strain = np.linspace(0, 0.5, 2100)
    clean = foam_stress(strain, spec)
    hits = 0
    for seed in range(20):
        noisy = clean + np.random.default_rng(seed).normal(0, 1.5 / 400.0, strain.size)
        result, _ = first_peak(strain, noisy, noisy * 400.0, 0.45, DEFAULT_CONFIG)
        hits += result.has_distinct_first_peak
    assert hits == 0, f"{hits} of 20 noisy monotonic curves reported a first peak"


def test_real_buckling_peak_survives_smoothing_on_a_noisy_curve():
    spec = FoamCurveSpec()  # 2.7 MPa peak softening to 2.15 MPa
    strain = np.linspace(0, 0.5, 2100)
    noisy = foam_stress(strain, spec) + np.random.default_rng(0).normal(0, 1.5 / 400.0, strain.size)
    result, _ = first_peak(strain, noisy, noisy * 400.0, 0.45, DEFAULT_CONFIG)
    assert result.has_distinct_first_peak
    assert result.first_peak_stress_MPa == pytest.approx(2.70, rel=0.02)
    assert result.first_peak_strain == pytest.approx(0.045, abs=0.004)
