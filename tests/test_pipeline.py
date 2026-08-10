"""End-to-end pipeline tests, including change detection and the CLIs."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import DEFAULT_CONFIG
from src.pipeline import (
    PipelinePaths,
    is_stale,
    load_curves_from_disk,
    run_analysis,
    run_pipeline,
)
from src.specimens import SpecimenError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


# ------------------------------------------------------------ end to end ----
def test_full_pipeline_produces_metrics_and_figures(dataset):
    paths = PipelinePaths.from_root(dataset)
    result = run_pipeline(paths, DEFAULT_CONFIG)

    assert paths.metrics_csv.exists()
    assert len(result.metrics) == 4                     # 2 patterns x 2 replicates
    assert set(result.metrics["pattern_name"]) == {"solid", "gyroid_v2"}
    assert result.figures, "no figures were generated"
    assert (paths.figures_dir / "stress_strain_envelope.png").exists()
    assert paths.diagnostics_csv.exists()
    assert len(list(paths.curves_dir.glob("*.csv"))) == 4


def test_metrics_are_physically_sensible(dataset):
    """The two synthetic patterns must come out ordered as they were built."""
    paths = PipelinePaths.from_root(dataset)
    metrics = run_pipeline(paths, DEFAULT_CONFIG).metrics.set_index("test_id")

    solid = metrics[metrics["pattern_name"] == "solid"]
    gyroid = metrics[metrics["pattern_name"] == "gyroid_v2"]

    assert solid["modulus_MPa"].mean() == pytest.approx(60.0, rel=0.15)
    assert gyroid["modulus_MPa"].mean() == pytest.approx(25.0, rel=0.15)
    assert solid["plateau_stress_MPa"].mean() > gyroid["plateau_stress_MPa"].mean()
    assert (metrics["densification_strain"] > 0.3).all()
    assert (metrics["energy_absorption_MJ_m3"] > 0).all()


def test_first_peak_reported_only_for_the_pattern_that_has_one(dataset):
    paths = PipelinePaths.from_root(dataset)
    metrics = run_pipeline(paths, DEFAULT_CONFIG).metrics

    solid = metrics[metrics["pattern_name"] == "solid"]
    gyroid = metrics[metrics["pattern_name"] == "gyroid_v2"]
    assert solid["has_distinct_first_peak"].all()
    assert not gyroid["has_distinct_first_peak"].any()


def test_sea_uses_mass_when_available(dataset):
    paths = PipelinePaths.from_root(dataset)
    metrics = run_pipeline(paths, DEFAULT_CONFIG).metrics
    assert (metrics["sea_basis"] == "mass").all()
    assert metrics["sea_J_per_g"].notna().all()
    assert (metrics["sea_J_per_g"] > 0).all()


def test_sea_falls_back_to_volume_without_mass(dataset):
    specimens = pd.read_csv(dataset / "data" / "specimens.csv")
    specimens["mass_g"] = ""
    specimens.to_csv(dataset / "data" / "specimens.csv", index=False)

    paths = PipelinePaths.from_root(dataset)
    metrics = run_pipeline(paths, DEFAULT_CONFIG, force=True).metrics
    assert (metrics["sea_basis"] == "volume").all()
    assert metrics["energy_absorption_MJ_m3"].notna().all()


def test_metadata_is_joined_onto_metrics(dataset):
    paths = PipelinePaths.from_root(dataset)
    metrics = run_pipeline(paths, DEFAULT_CONFIG).metrics
    for column in ("pattern_name", "pattern_params", "replicate", "material", "raw_file"):
        assert column in metrics.columns
    assert "param_cell_mm" in metrics.columns  # numeric params expanded


def test_energy_absorption_matches_manual_integration(dataset):
    """Cross-check the reported energy against the saved curve."""
    paths = PipelinePaths.from_root(dataset)
    result = run_pipeline(paths, DEFAULT_CONFIG)
    row = result.metrics.iloc[0]
    curve = result.curves[row["test_id"]]

    limit = row["densification_strain"]
    mask = curve.strain <= limit
    expected = np.trapezoid(curve.stress_MPa[mask], curve.strain[mask])
    assert row["energy_absorption_MJ_m3"] == pytest.approx(expected, rel=0.01)


# ------------------------------------------------------ change detection ----
def test_second_run_is_skipped_when_nothing_changed(dataset):
    paths = PipelinePaths.from_root(dataset)
    run_pipeline(paths, DEFAULT_CONFIG)
    assert not is_stale(paths, DEFAULT_CONFIG)
    assert run_pipeline(paths, DEFAULT_CONFIG).skipped


def test_editing_specimens_marks_the_results_stale(dataset):
    paths = PipelinePaths.from_root(dataset)
    run_pipeline(paths, DEFAULT_CONFIG)

    specimens = pd.read_csv(paths.specimens_csv)
    specimens["notes"] = specimens["notes"].astype("object")
    specimens.loc[0, "notes"] = "re-measured"
    specimens.to_csv(paths.specimens_csv, index=False)

    assert is_stale(paths, DEFAULT_CONFIG)
    assert not run_pipeline(paths, DEFAULT_CONFIG).skipped


def test_adding_a_raw_file_marks_the_results_stale(dataset):
    from src.synthetic import SyntheticTest

    paths = PipelinePaths.from_root(dataset)
    run_pipeline(paths, DEFAULT_CONFIG)
    SyntheticTest(name="new.csv", seed=99).write(paths.raw_dir / "new.csv")
    assert is_stale(paths, DEFAULT_CONFIG)


def test_changing_config_marks_the_results_stale(dataset):
    paths = PipelinePaths.from_root(dataset)
    run_pipeline(paths, DEFAULT_CONFIG)
    assert is_stale(paths, DEFAULT_CONFIG.replace(plateau_strain_hi=0.40))


def test_curves_round_trip_through_disk(dataset):
    paths = PipelinePaths.from_root(dataset)
    result = run_pipeline(paths, DEFAULT_CONFIG)
    reloaded = load_curves_from_disk(paths, result.metrics)

    assert set(reloaded) == set(result.curves)
    for test_id, curve in result.curves.items():
        assert np.allclose(reloaded[test_id].strain, curve.strain)
        assert np.allclose(reloaded[test_id].stress_MPa, curve.stress_MPa)


# ------------------------------------------------------------- resilience ----
def test_one_bad_file_does_not_sink_the_batch(dataset, capsys):
    """A corrupt export is reported; the remaining tests still analyse."""
    paths = PipelinePaths.from_root(dataset)
    target = sorted(paths.raw_dir.glob("*.csv"))[0]
    target.write_text("Results Table 1\n\ngarbage\n")

    result = run_analysis(paths, DEFAULT_CONFIG)
    assert len(result.metrics) == 3
    assert "analysis_failed" in set(result.diagnostics["code"])
    assert "failed to analyse" in capsys.readouterr().out


def test_all_files_bad_raises(dataset):
    paths = PipelinePaths.from_root(dataset)
    for csv in paths.raw_dir.glob("*.csv"):
        csv.write_text("nonsense\n")
    with pytest.raises(RuntimeError, match="No test could be analysed"):
        run_analysis(paths, DEFAULT_CONFIG)


def test_missing_raw_file_is_reported_before_analysis(dataset):
    paths = PipelinePaths.from_root(dataset)
    sorted(paths.raw_dir.glob("*.csv"))[0].unlink()
    with pytest.raises(SpecimenError, match="missing raw files"):
        run_analysis(paths, DEFAULT_CONFIG)


def test_duplicate_exports_are_flagged_not_counted_as_replicates(dataset):
    """The same test exported twice must not masquerade as two specimens."""
    paths = PipelinePaths.from_root(dataset)
    specimens = pd.read_csv(paths.specimens_csv)
    source = str(specimens.loc[0, "raw_file"])
    # Re-export one test under a new name, as the instrument software does.
    shutil.copy(paths.raw_dir / source, paths.raw_dir / "reexport.csv")

    clone = specimens.iloc[[0]].copy()
    clone["test_id"] = "reexport"
    clone["raw_file"] = "reexport.csv"
    clone["replicate"] = 99
    pd.concat([specimens, clone], ignore_index=True).to_csv(
        paths.specimens_csv, index=False
    )

    result = run_analysis(paths, DEFAULT_CONFIG)
    dupes = result.diagnostics[result.diagnostics["code"] == "duplicate_test_data"]
    assert len(dupes) == 2, "both members of the duplicate pair should be flagged"
    assert set(dupes["test_id"]) == {"reexport", str(specimens.loc[0, "test_id"])}
    assert (dupes["level"] == "warning").all()


def test_duplicate_detection_ignores_how_the_file_was_written(dataset):
    """Identical measurements count as duplicates even with different headers."""
    from src.io import read_raw_csv
    from src.pipeline import data_digest, group_duplicates

    paths = PipelinePaths.from_root(dataset)
    original = sorted(paths.raw_dir.glob("*.csv"))[0]

    # Same data, boilerplate stripped -- byte-different, measurement-identical.
    lines = original.read_text().splitlines()
    header = next(i for i, l in enumerate(lines) if l.startswith("Time,"))
    bare = paths.raw_dir / "bare.csv"
    bare.write_text("\n".join([""] + lines[header:]) + "\n")

    assert original.read_bytes() != bare.read_bytes()
    digests = {
        "original": data_digest(read_raw_csv(original)),
        "bare": data_digest(read_raw_csv(bare)),
    }
    assert digests["original"] == digests["bare"]
    assert group_duplicates(digests) == [["bare", "original"]]


def test_distinct_tests_are_not_flagged_as_duplicates(dataset):
    paths = PipelinePaths.from_root(dataset)
    result = run_analysis(paths, DEFAULT_CONFIG)
    assert "duplicate_test_data" not in set(result.diagnostics["code"])


def test_force_limit_surfaces_in_the_metrics_table(dataset):
    paths = PipelinePaths.from_root(dataset)
    cfg = DEFAULT_CONFIG.replace(force_limit_kN=0.5)  # below the synthetic peak
    metrics = run_analysis(paths, cfg).metrics
    assert "ended_at_force_limit" in metrics.columns
    assert metrics["ended_at_force_limit"].astype(bool).all()
    assert "tare_offset_N" in metrics.columns
    assert "preload_slack_mm" in metrics.columns


def test_diagnostics_capture_parser_and_metric_warnings(dataset):
    paths = PipelinePaths.from_root(dataset)
    result = run_pipeline(paths, DEFAULT_CONFIG)
    codes = set(result.diagnostics["code"])
    assert "sampling" in codes            # from the parser
    assert "preload_trimmed" in codes     # from cleaning
    assert codes & {"first_peak", "no_first_peak"}  # from metric extraction


# -------------------------------------------------------------------- CLI ----
def test_run_analysis_cli(dataset, capsys):
    import run_analysis as cli

    assert cli.main(["--root", str(dataset)]) == 0
    out = capsys.readouterr().out
    assert "Analysed 4 test(s)" in out
    assert (dataset / "results" / "metrics_summary.csv").exists()


def test_run_analysis_cli_rejects_missing_specimens(tmp_path, capsys):
    import run_analysis as cli

    assert cli.main(["--root", str(tmp_path)]) == 2
    assert "register_test.py" in capsys.readouterr().err


def test_run_analysis_cli_config_overrides(dataset):
    import run_analysis as cli

    assert cli.main(["--root", str(dataset), "--plateau-hi", "0.40"]) == 0
    metrics = pd.read_csv(dataset / "results" / "metrics_summary.csv")
    assert (metrics["plateau_strain_hi"] == 0.40).all()


def test_plot_envelope_cli(dataset, capsys):
    import plot_envelope as cli
    import run_analysis as analysis_cli

    analysis_cli.main(["--root", str(dataset), "--no-figures"])
    assert cli.main(["--root", str(dataset)]) == 0
    assert "figure(s)" in capsys.readouterr().out
    assert (dataset / "results" / "figures" / "stress_strain_envelope.png").exists()


def test_plot_envelope_cli_without_metrics(tmp_path, capsys):
    import plot_envelope as cli

    assert cli.main(["--root", str(tmp_path)]) == 2
    assert "run_analysis.py" in capsys.readouterr().err


def test_register_test_cli_appends_a_row(tmp_path, capsys):
    import register_test as cli
    from src.synthetic import SyntheticTest

    raw_dir = tmp_path / "data" / "raw"
    SyntheticTest(name="x.csv", seed=1).write(raw_dir / "x.csv")

    assert (
        cli.main(
            [
                "--root", str(tmp_path),
                "--raw-file", "x.csv",
                "--pattern-name", "gyroid_v2",
                "--replicate", "1",
                "--side-length-mm", "20",
                "--material", "PLA",
                "--pattern-params", '{"cell_mm": 5}',
            ]
        )
        == 0
    )

    specimens = pd.read_csv(tmp_path / "data" / "specimens.csv")
    assert len(specimens) == 1
    assert specimens.loc[0, "pattern_name"] == "gyroid_v2"
    assert specimens.loc[0, "test_id"] == "x"
    assert specimens.loc[0, "initial_height_mm"] == 20.0   # defaulted from side


def test_register_test_cli_template_covers_every_file(tmp_path, capsys):
    """--template pre-fills what the filename knows and leaves geometry blank."""
    import register_test as cli
    from src.synthetic import SyntheticTest

    raw_dir = tmp_path / "data" / "raw"
    for name in ("solid_compression_20260717_192209_1_1.csv", "b.csv"):
        SyntheticTest(name=name, seed=hash(name) % 1000).write(raw_dir / name)

    assert cli.main(["--root", str(tmp_path), "--template"]) == 0
    specimens = pd.read_csv(tmp_path / "data" / "specimens.csv", dtype=str)

    assert len(specimens) == 2
    assert set(specimens["raw_file"]) == {
        "solid_compression_20260717_192209_1_1.csv",
        "b.csv",
    }
    dated = specimens[specimens["raw_file"].str.startswith("solid")]
    assert dated["test_date"].iloc[0] == "2026-07-17"
    assert specimens["pattern_name"].isna().all()  # left for the user
    assert "run_analysis" in capsys.readouterr().out


def test_register_test_cli_template_marks_duplicate_exports(tmp_path, capsys):
    """A re-export of the same test is flagged in the template, not dropped."""
    import register_test as cli
    from src.synthetic import SyntheticTest

    raw_dir = tmp_path / "data" / "raw"
    SyntheticTest(name="a.csv", seed=5).write(raw_dir / "a.csv")
    # Same measurements, written without the boilerplate block.
    lines = (raw_dir / "a.csv").read_text().splitlines()
    header = next(i for i, l in enumerate(lines) if l.startswith("Time,"))
    (raw_dir / "a_reexport.csv").write_text("\n".join([""] + lines[header:]) + "\n")

    assert cli.main(["--root", str(tmp_path), "--template"]) == 0
    specimens = pd.read_csv(tmp_path / "data" / "specimens.csv", dtype=str).fillna("")

    notes = dict(zip(specimens["raw_file"], specimens["notes"]))
    assert len(specimens) == 2, "the duplicate must still be listed"
    assert notes["a.csv"] == "", "the first of the pair is the one to keep"
    assert "DUPLICATE" in notes["a_reexport.csv"]


def test_register_test_cli_template_refuses_to_overwrite(tmp_path, capsys):
    """A filled-in specimens.csv must never be clobbered by a new export."""
    import register_test as cli
    from src.synthetic import SyntheticTest

    raw_dir = tmp_path / "data" / "raw"
    SyntheticTest(name="a.csv", seed=1).write(raw_dir / "a.csv")
    assert cli.main(["--root", str(tmp_path), "--template"]) == 0

    # A new test arrives after the table has been started.
    SyntheticTest(name="b.csv", seed=2).write(raw_dir / "b.csv")
    assert cli.main(["--root", str(tmp_path), "--template"]) == 1
    assert "refusing to overwrite" in capsys.readouterr().err


def test_register_test_cli_template_is_a_no_op_when_all_registered(tmp_path, capsys):
    import register_test as cli
    from src.synthetic import SyntheticTest

    SyntheticTest(name="a.csv", seed=1).write(tmp_path / "data" / "raw" / "a.csv")
    assert cli.main(["--root", str(tmp_path), "--template"]) == 0
    assert cli.main(["--root", str(tmp_path), "--template"]) == 0
    assert "Nothing to do" in capsys.readouterr().out


def test_register_test_cli_reads_date_from_filename(tmp_path):
    import register_test as cli
    from src.synthetic import SyntheticTest

    name = "solid_compression_20260717_192209_1_1.csv"
    SyntheticTest(name=name, seed=1).write(tmp_path / "data" / "raw" / name)
    cli.main(
        ["--root", str(tmp_path), "--raw-file", name, "--pattern-name", "solid",
         "--replicate", "1", "--side-length-mm", "20", "--material", "PLA"]
    )
    specimens = pd.read_csv(tmp_path / "data" / "specimens.csv")
    assert specimens.loc[0, "test_date"] == "2026-07-17"


def test_register_test_cli_lists_unregistered(tmp_path, capsys):
    import register_test as cli
    from src.synthetic import SyntheticTest

    SyntheticTest(name="a.csv", seed=1).write(tmp_path / "data" / "raw" / "a.csv")
    assert cli.main(["--root", str(tmp_path), "--list"]) == 0
    assert "a.csv" in capsys.readouterr().out


def test_register_test_cli_rejects_duplicate_id(tmp_path, capsys):
    import register_test as cli
    from src.synthetic import SyntheticTest

    SyntheticTest(name="a.csv", seed=1).write(tmp_path / "data" / "raw" / "a.csv")
    args = ["--root", str(tmp_path), "--raw-file", "a.csv", "--pattern-name", "p",
            "--replicate", "1", "--side-length-mm", "20", "--material", "PLA"]
    assert cli.main(args) == 0
    assert cli.main(args) == 1
    assert "already in specimens.csv" in capsys.readouterr().err


def test_register_test_cli_rejects_bad_json(tmp_path, capsys):
    import register_test as cli
    from src.synthetic import SyntheticTest

    SyntheticTest(name="a.csv", seed=1).write(tmp_path / "data" / "raw" / "a.csv")
    status = cli.main(
        ["--root", str(tmp_path), "--raw-file", "a.csv", "--pattern-name", "p",
         "--replicate", "1", "--side-length-mm", "20", "--material", "PLA",
         "--pattern-params", "{bad json}"]
    )
    assert status == 1
    assert "not valid JSON" in capsys.readouterr().err


def test_register_then_analyse_round_trip(tmp_path):
    """The documented workflow: register a file, then analyse it."""
    import register_test as cli
    import run_analysis as analysis_cli
    from src.synthetic import SyntheticTest

    for i in (1, 2):
        name = f"t{i}.csv"
        SyntheticTest(name=name, seed=i).write(tmp_path / "data" / "raw" / name)
        cli.main(
            ["--root", str(tmp_path), "--raw-file", name, "--pattern-name", "solid",
             "--replicate", str(i), "--side-length-mm", "20", "--material", "PLA"]
        )

    assert analysis_cli.main(["--root", str(tmp_path)]) == 0
    metrics = pd.read_csv(tmp_path / "results" / "metrics_summary.csv")
    assert len(metrics) == 2
    assert metrics["modulus_MPa"].notna().all()
