"""Correlation analysis, tested against a design space with planted causes.

`planted_design` builds 64 distinct patterns where the truth is known:
modulus and strength scale with relative density as bending-dominated
power laws, topology multiplies them, layer height has a small effect that
density masks, and print speed and nozzle temperature do nothing at all.
The analysis has to find each of those -- and nothing else.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.config import DEFAULT_CONFIG
from src.correlation import (
    analyse,
    bh_fdr,
    discover_parameters,
    fisher_ci,
    interpret_exponent,
    kruskal_effect,
    partial_spearman,
    power_law_fit,
    property_data,
    property_label,
    rank_vif,
    summarise,
    write_outputs,
)
from src.synthetic import planted_design

GEOMETRY = ["cell_mm", "wall_mm", "topology", "layer_height_mm", "print_speed_mm_s", "nozzle_temp_C"]
MANUFACTURING = ["print_batch", "bed_position"]


def planted_metrics(n: int = 64, seed: int = 7, noise: float = 0.02) -> pd.DataFrame:
    """A metrics table as the pipeline would write it for the planted design."""
    d = planted_design(n, seed)
    rng = np.random.default_rng(seed + 1)
    m = pd.DataFrame(
        {
            "test_id": d["test_id"],
            "pattern_name": d["pattern_name"],
            "material": d["material"],
            "relative_density": d["relative_density"],
            "modulus_MPa": d["true_modulus_MPa"] * rng.lognormal(0, noise, n),
            "plateau_stress_MPa": d["true_plateau_MPa"] * rng.lognormal(0, noise, n),
            "first_peak_stress_MPa": d["true_peak_MPa"],
            "densification_strain": d["true_densification_strain"] + rng.normal(0, 0.005, n),
            "print_date": d["print_date"],
            "test_date": d["test_date"],
        }
    )
    for column in GEOMETRY + MANUFACTURING:
        m[f"param_{column}"] = d[column]
    m["param_print_to_test_days"] = (
        pd.to_datetime(d["test_date"]) - pd.to_datetime(d["print_date"])
    ).dt.total_seconds() / 86400.0
    return m


@pytest.fixture(scope="module")
def planted():
    return planted_metrics()


@pytest.fixture(scope="module")
def result(planted):
    return analyse(planted, DEFAULT_CONFIG)


@pytest.fixture(scope="module")
def geometry_result(planted):
    return analyse(planted, DEFAULT_CONFIG, parameters=GEOMETRY)


def cell(result, parameter, prop):
    column = parameter if parameter == "relative_density" else f"param_{parameter}"
    found = result.cell(column, prop)
    assert found is not None, f"no row for {parameter} x {prop}"
    return found


# ------------------------------------------------------------ statistics ----
def test_bh_fdr_matches_scipy():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    assert np.allclose(bh_fdr(p), stats.false_discovery_control(p))


def test_bh_fdr_leaves_untested_pairs_out_of_the_family():
    q = bh_fdr([0.01, np.nan, 0.02])
    assert np.isnan(q[1])
    assert np.allclose(q[[0, 2]], stats.false_discovery_control([0.01, 0.02]))


def test_fisher_interval_brackets_the_estimate_and_shrinks_with_n():
    lo, hi = fisher_ci(0.5, 20)
    assert lo < 0.5 < hi
    lo_big, hi_big = fisher_ci(0.5, 200)
    assert (hi_big - lo_big) < (hi - lo)
    assert all(math.isnan(v) for v in fisher_ci(0.5, 3))


def test_partial_correlation_removes_a_shared_driver():
    """x and y both follow z; holding z fixed leaves nothing between them."""
    rng = np.random.default_rng(0)
    z = rng.normal(size=300)
    x = z + 0.3 * rng.normal(size=300)
    y = z + 0.3 * rng.normal(size=300)
    assert stats.spearmanr(x, y).statistic > 0.8
    rho, p, df, r2_x = partial_spearman(x, y, z[:, None])
    assert abs(rho) < 0.15
    assert df == 300 - 3
    assert r2_x > 0.8


def test_partial_correlation_reveals_a_masked_effect():
    """y is dominated by z; x's small effect only shows with z held fixed."""
    rng = np.random.default_rng(1)
    z = rng.uniform(0, 10, 200)
    x = rng.uniform(0, 1, 200)
    y = 5 * z + 1.0 * x + 0.2 * rng.normal(size=200)
    assert abs(stats.spearmanr(x, y).statistic) < 0.2
    rho, p, _, _ = partial_spearman(x, y, z[:, None])
    # Rank residuals are attenuated relative to raw ones (about 0.6 here across
    # seeds, where a raw-value partial would be ~0.98), but the effect is
    # unmistakable either way: p ~ 1e-20 against a pairwise rho near zero.
    assert rho > 0.5 and p < 1e-10


def test_partial_is_undefined_when_x_is_determined_by_the_controls():
    z = np.arange(30, dtype=float)
    rho, p, _, r2_x = partial_spearman(z * 2, np.sin(z), z[:, None])
    assert math.isnan(rho) and r2_x == pytest.approx(1.0)


def test_rank_vif_flags_a_derived_parameter():
    rng = np.random.default_rng(2)
    a, b = rng.uniform(1, 2, 80), rng.uniform(1, 2, 80)
    frame = pd.DataFrame({"a": a, "b": b, "ratio": b / a, "free": rng.uniform(size=80)})
    vif = rank_vif(frame)
    assert vif["free"] < 1.5
    assert vif["ratio"] > 10


def test_kruskal_effect_size_and_small_groups():
    values = pd.Series([1, 2, 3, 10, 11, 12, 5])
    groups = pd.Series(["a", "a", "a", "b", "b", "b", "lonely"])
    out = kruskal_effect(values, groups, min_group_size=3)
    assert out["k"] == 2, "the one-test level must be left out"
    assert out["n"] == 6
    assert 0 < out["epsilon_sq"] <= 1
    assert out["medians"] == {"a": 2.0, "b": 11.0}
    assert kruskal_effect(values, groups, min_group_size=4) is None


def test_power_law_fit_recovers_the_exponent():
    rng = np.random.default_rng(3)
    x = rng.uniform(0.1, 0.4, 60)
    y = 2500 * x**2 * rng.lognormal(0, 0.03, 60)
    fit = power_law_fit(x, y)
    assert fit["ci_lo"] < 2.0 < fit["ci_hi"]
    assert fit["exponent"] == pytest.approx(2.0, abs=0.1)
    assert fit["prefactor"] == pytest.approx(2500, rel=0.2)


@pytest.mark.parametrize(
    "kind, lo, hi, expected",
    [
        ("modulus", 1.9, 2.1, "bending-dominated"),
        ("modulus", 0.9, 1.1, "stretch-dominated"),
        ("modulus", 0.8, 2.3, "cannot distinguish"),
        ("modulus", 2.5, 2.9, "steeper"),
        ("strength", 1.4, 1.6, "bending-dominated"),
        ("strength", 1.15, 1.35, "between"),
    ],
)
def test_exponent_interpretation(kind, lo, hi, expected):
    assert expected in interpret_exponent(kind, lo, hi)


# ------------------------------------------------------ parameter handling ----
def test_unique_pattern_names_cannot_be_a_factor(planted):
    infos = {p.column: p for p in discover_parameters(planted, DEFAULT_CONFIG)}
    assert not infos["pattern_name"].included
    assert "unique per test" in infos["pattern_name"].reason
    assert not infos["material"].included  # all PLA
    assert infos["param_topology"].included and infos["param_topology"].kind == "categorical"
    assert infos["param_cell_mm"].kind == "numeric"


def test_constant_and_sparse_parameters_are_excluded_with_a_reason(planted):
    frame = planted.copy()
    frame["param_constant"] = 1.0
    frame["param_sparse"] = np.nan
    frame.loc[:3, "param_sparse"] = [1.0, 2.0, 3.0, 4.0]
    infos = {p.column: p for p in discover_parameters(frame, DEFAULT_CONFIG)}
    assert not infos["param_constant"].included
    assert "same value" in infos["param_constant"].reason
    assert not infos["param_sparse"].included
    assert "only 4 tests" in infos["param_sparse"].reason


def test_unknown_parameter_name_is_an_error(planted):
    with pytest.raises(ValueError, match="unknown parameter 'nope'"):
        discover_parameters(planted, DEFAULT_CONFIG, include=["nope"])


def test_censored_values_are_excluded_and_counted():
    frame = planted_metrics()
    frame["energy_absorption_MJ_m3"] = np.linspace(1, 2, len(frame))
    frame["energy_censored"] = [i % 4 == 0 for i in range(len(frame))]
    props = property_data(frame, ["energy_absorption_MJ_m3"], DEFAULT_CONFIG)
    assert props.n_censored["energy_absorption_MJ_m3"] == 16
    assert props.values["energy_absorption_MJ_m3"].notna().sum() == 48

    # Read back from CSV, booleans are strings -- must still work.
    frame["energy_censored"] = frame["energy_censored"].map(str)
    again = property_data(frame, ["energy_absorption_MJ_m3"], DEFAULT_CONFIG)
    assert again.n_censored["energy_absorption_MJ_m3"] == 16

    kept = property_data(
        frame, ["energy_absorption_MJ_m3"], DEFAULT_CONFIG.replace(correlation_include_censored=True)
    )
    assert kept.values["energy_absorption_MJ_m3"].notna().sum() == 64


def test_a_property_that_is_all_lower_bounds_is_skipped_with_the_reason():
    frame = planted_metrics()
    frame["energy_absorption_MJ_m3"] = 1.0
    frame["energy_censored"] = True
    props = property_data(frame, ["energy_absorption_MJ_m3"], DEFAULT_CONFIG)
    assert "energy_absorption_MJ_m3" in props.skipped
    assert "lower bounds" in props.skipped["energy_absorption_MJ_m3"]


# ------------------------------------------- recovering the planted truth ----
def test_density_drives_modulus_and_strength(result):
    for prop in ("modulus_MPa", "plateau_stress_MPa"):
        row = cell(result, "relative_density", prop)
        assert row["effect"] > 0.9 and row["significant"]


def test_null_parameters_are_not_significant(result):
    """Print speed and nozzle temperature have no effect -- FDR must hold."""
    for parameter in ("print_speed_mm_s", "nozzle_temp_C"):
        rows = result.table[result.table["parameter"] == parameter]
        assert not rows["significant"].any(), rows[["property", "effect", "q_value"]]
        assert not rows["partial_significant"].any()


def test_masked_layer_height_effect_is_found_by_the_partial(result):
    """+/-10% from layer height is invisible next to density's 30x spread."""
    row = cell(result, "layer_height_mm", "modulus_MPa")
    assert not row["significant"], "pairwise should NOT see it"
    assert row["partial_significant"] and row["partial_rho"] < -0.3
    assert row["partial_reliable"]


def test_topology_is_detected_as_a_categorical_effect(result):
    row = cell(result, "topology", "modulus_MPa")
    assert row["kind"] == "categorical" and row["significant"]
    assert row["direction"] == "highest: honeycomb"


def test_derived_parameters_are_flagged_rather_than_given_wrong_partials(result):
    """rho* is computed from wall/cell, so the three cannot be separated.

    Before categorical factors were held fixed, cell size's partial came out
    with the wrong sign (topology leaking through); now it must be withheld.
    """
    vif = {p.name: p.vif for p in result.included if p.kind == "numeric"}
    assert vif["relative_density"] > 10 and vif["cell_mm"] > 10
    row = cell(result, "cell_mm", "modulus_MPa")
    assert not row["partial_reliable"]
    assert "largely determined" in row["note"]


def test_geometry_only_partials_have_the_physical_sign(geometry_result):
    """Without the derived density, cell and wall separate cleanly."""
    assert cell(geometry_result, "cell_mm", "modulus_MPa")["partial_rho"] < -0.8
    assert cell(geometry_result, "wall_mm", "modulus_MPa")["partial_rho"] > 0.8


def test_gibson_ashby_exponents_are_recovered(result):
    shared = result.scaling[result.scaling["group"] == "all"].set_index("property")
    modulus = shared.loc["modulus_MPa"]
    assert modulus["ci_lo"] <= 2.0 <= modulus["ci_hi"]
    assert "bending" in modulus["interpretation"]
    assert "topology" in modulus["held_fixed"].split(", ")
    plateau = shared.loc["plateau_stress_MPa"]
    assert plateau["ci_lo"] <= 1.5 <= plateau["ci_hi"]


def test_pooled_exponent_would_be_confounded_by_topology(planted):
    """Why the fit gives each topology its own prefactor.

    Honeycomb is both denser and stronger, so pooling the two tilts the
    plateau slope to ~1.6 and puts the planted 1.5 outside its CI.
    """
    rho = planted["relative_density"].to_numpy()
    plateau = planted["plateau_stress_MPa"].to_numpy()
    pooled = power_law_fit(rho, plateau)

    offsets = (planted["param_topology"] == "honeycomb").astype(float).to_numpy()[:, None]
    controlled = power_law_fit(rho, plateau, offsets)
    # Pooled: ~1.61, tilted steep by about +0.1. Controlled: ~1.49.
    assert pooled["exponent"] - controlled["exponent"] > 0.08
    assert controlled["ci_lo"] <= 1.5 <= controlled["ci_hi"]
    assert controlled["r2"] > pooled["r2"]


def test_each_level_is_also_fitted_on_its_own(result):
    levels = result.scaling[
        (result.scaling["property"] == "modulus_MPa") & (result.scaling["group"] != "all")
    ]
    assert set(levels["group"]) == {"topology=gyroid", "topology=honeycomb"}
    # 32 tests per level: each 95% CI misses the truth 1 time in 20, so check
    # the estimate rather than demanding both intervals cover it.
    for _, row in levels.iterrows():
        assert row["exponent"] == pytest.approx(2.0, abs=0.15), row["group"]


def test_summary_tells_the_story(result):
    text = summarise(result)
    assert "Hidden until other parameters are held fixed" in text
    assert "layer_height_mm" in text
    assert "No association with any property, pairwise or partial" in text
    assert "print_speed_mm_s" in text and "nozzle_temp_C" in text
    # A parameter whose only effect is masked must not ALSO be listed as
    # having no association -- the old wording contradicted itself.
    none_line = next(l for l in text.splitlines() if "No association with any property" in l)
    assert "layer_height_mm" not in none_line
    assert "relative_density" in text and "Largely predictable" in text
    assert "bending-dominated" in text


def test_property_labels():
    assert property_label("stress_at_20pct_MPa") == "σ at 20%"
    assert property_label("energy_to_10pct_MJ_m3", with_unit=True) == "W to 10% (MJ/m³)"
    assert property_label("modulus_MPa", with_unit=True) == "Modulus E (MPa)"
    assert property_label("something_else") == "something_else"


# ---------------------------------------------------------------- outputs ----
def test_outputs_are_written(result, tmp_path):
    written = write_outputs(result, tmp_path / "tables", tmp_path / "figs", DEFAULT_CONFIG)
    names = {p.name for p in written}
    for expected in (
        "correlations.csv", "parameters.csv", "collinearity.csv", "scaling_laws.csv",
        "summary.txt", "heatmap.png", "categorical_effects.png", "collinearity.png",
        "scaling_laws.png", "property_modulus_MPa.png",
    ):
        assert expected in names, expected
    table = pd.read_csv(tmp_path / "tables" / "correlations.csv")
    assert {"q_value", "partial_rho", "effect_ci_lo", "n_censored_excluded"} <= set(table.columns)


def test_stale_property_figures_are_removed(result, tmp_path):
    figs = tmp_path / "figs"
    figs.mkdir()
    (figs / "property_old_metric.png").write_bytes(b"old")
    write_outputs(result, tmp_path / "tables", figs, DEFAULT_CONFIG)
    assert not (figs / "property_old_metric.png").exists()


def test_nothing_to_analyse_is_reported_plainly():
    frame = pd.DataFrame(
        {"test_id": list("abcdefghij"), "pattern_name": list("abcdefghij"),
         "material": ["PLA"] * 10, "modulus_MPa": np.arange(10.0)}
    )
    result = analyse(frame, DEFAULT_CONFIG)
    assert result.table.empty or not result.table["p_value"].notna().any()
    assert "Nothing could be tested" in summarise(result)


# --------------------------------------------- manufacturing factors & dates ----
from src.correlation import (  # noqa: E402
    correlation_ratio,
    cramers_v,
    determination,
    quade_partial,
)


def test_batch_effect_is_masked_pairwise_and_found_by_the_partial(result):
    """Batch B3 is 12% weaker -- invisible next to 30x density spread."""
    row = cell(result, "print_batch", "plateau_stress_MPa")
    assert row["kind"] == "categorical"
    assert not row["significant"], "pairwise should NOT see it"
    assert row["partial_significant"] and row["partial_reliable"]


def test_bed_edge_effect_is_found_by_the_partial(result):
    row = cell(result, "bed_position", "modulus_MPa")
    assert not row["significant"]
    assert row["partial_significant"]


def test_batch_and_print_date_are_reported_as_confounded(result):
    """Each batch is printed on one day: one effect, two names."""
    infos = {p.name: p for p in result.included}
    assert "print_date" in infos["print_batch"].duplicates
    assert "print_batch" in infos["print_date"].duplicates
    # Neither is held fixed when testing the other -- that would erase both.
    row = cell(result, "print_batch", "modulus_MPa")
    assert "print_date" not in row["partial_controls"]
    assert "confounded with print_date" in row["note"]
    assert "print_batch ~ print_date" in summarise(result)


def test_print_to_test_age_has_no_planted_effect(result):
    rows = result.table[result.table["parameter"] == "print_to_test_days"]
    assert not rows["significant"].any() and not rows["partial_significant"].any()


def test_test_date_is_dropped_as_redundant_with_print_date_and_age(planted):
    infos = {p.column: p for p in discover_parameters(planted, DEFAULT_CONFIG)}
    assert not infos["test_date"].included
    assert "redundant" in infos["test_date"].reason
    # ...but can still be analysed on request.
    asked = {p.column: p for p in discover_parameters(planted, DEFAULT_CONFIG, include=["test_date"])}
    assert asked["test_date"].included


def test_dates_are_analysed_as_elapsed_time(planted):
    infos = {p.column: p for p in discover_parameters(planted, DEFAULT_CONFIG)}
    printed = infos["print_date"]
    assert printed.kind == "datetime" and printed.is_ordered
    assert printed.origin == pd.Timestamp("2026-07-01")
    assert "days since 2026-07-01" in printed.axis_label


def test_topology_determined_by_a_ratio_is_not_trusted_as_separable(result):
    """rho* = k_topology * wall / cell: linear in logs, not in ranks.

    Ranks alone gave R^2 = 0.77 and passed topology's partial as reliable,
    which then reported its effect as "borrowed" -- an artefact.
    """
    row = cell(result, "topology", "modulus_MPa")
    assert not row["partial_reliable"]


# ------------------------------------------------------- the new statistics ----
def test_determination_sees_multiplicative_links():
    rng = np.random.default_rng(4)
    wall, cell_size = rng.uniform(0.6, 1.2, 80), rng.uniform(4, 8, 80)
    ratio = wall / cell_size * (1 + rng.normal(0, 0.01, 80))
    controls = np.column_stack([wall, cell_size])
    assert determination(ratio, controls) > 0.95
    assert determination(rng.uniform(size=80), controls) < 0.2


def test_quade_finds_a_group_effect_hidden_behind_a_covariate():
    rng = np.random.default_rng(5)
    x = rng.uniform(1, 10, 90)
    groups = np.repeat(["a", "b", "c"], 30)
    y = 10 * x + np.where(groups == "b", 3.0, 0.0) + rng.normal(0, 1, 90)
    assert stats.kruskal(*(y[groups == g] for g in "abc")).pvalue > 0.05
    f_stat, p, eta2, r2_groups = quade_partial(y, groups, x[:, None])
    assert p < 1e-4 and eta2 > 0.2 and r2_groups < 0.1


def test_quade_is_calibrated_under_the_null():
    rng = np.random.default_rng(6)
    hits = 0
    for _ in range(200):
        x = rng.uniform(size=40)
        y = x + rng.normal(0, 0.5, 40)
        groups = rng.choice(["a", "b", "c", "d"], 40)
        hits += quade_partial(y, groups, x[:, None])[1] < 0.05
    assert 0.02 <= hits / 200 <= 0.09, hits


def test_association_measures():
    groups = np.repeat(["a", "b"], 20)
    assert cramers_v(groups, groups) == pytest.approx(1.0)
    assert correlation_ratio(np.r_[np.arange(20), np.arange(100, 120)], groups) > 0.85
    rng = np.random.default_rng(7)
    assert cramers_v(groups, rng.choice(["x", "y"], 40)) < 0.4
