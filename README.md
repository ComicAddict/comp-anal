# Compression Test Analysis

Tools for analysing uniaxial compression tests on cube specimens containing
negative-volume internal patterns. Raw instrument CSVs go in; engineering
stress–strain curves, standard cellular-solid metrics, and design-space
envelope figures come out.

Two questions this is built to answer:

1. **Per pattern** — how stiff is it, where does it plateau, when does it
   densify, how much energy does it absorb?
2. **Across the design space** — given every pattern tested, what range of
   mechanical response is achievable, and how much of that spread is
   pattern-to-pattern rather than replicate-to-replicate noise?

---

## Quickstart

```bash
pip install -r requirements.txt

# 0. Check what a new export actually contains (no metadata needed).
python scripts/inspect_raw.py --duplicates

# 1. Drop instrument CSVs into data/raw/, then register each one once.
python scripts/register_test.py --list          # what still needs registering
python scripts/register_test.py --template      # skeleton specimens.csv to fill in
python scripts/register_test.py --interactive   # or be prompted field by field

# 2. Analyse everything and draw the figures.
python scripts/run_analysis.py

# 3. Redraw figures without re-parsing (or explore other metric pairs).
python scripts/plot_envelope.py --pair modulus_MPa sea_J_per_g
```

Machine setup lives in `analysis_config.json` at the repo root and is picked up
automatically. It currently records the 9.5 kN load limit:

```json
{ "force_limit_kN": 9.5 }
```

**Want to see it work before you have data registered?** Generate a synthetic
five-pattern design space and run the whole pipeline on it:

```bash
python scripts/make_demo_data.py            # writes demo/ (synthetic, disposable)
python scripts/run_analysis.py --root demo
```

---

## How the pieces fit together

```
data/raw/*.csv ──┐
                 ├─→ src/io.py ──→ src/mechanics.py ──→ src/metrics.py ──→ results/metrics_summary.csv
data/specimens.csv ┘   parse &        force,disp →         per-test              │
                       clean          stress,strain        metrics               ↓
                                                                          src/envelope.py
                                                                                 ↓
                                                                          results/figures/
```

`data/specimens.csv` is the single source of truth for what pattern each test
belongs to. Nothing downstream infers anything from filenames.

---

## Raw data format

The instrument export looks like this:

```
Results Table 1


Results Table 2



Time,Displacement,Force
(s),(mm),(kN)
"0.0000","0.0000","0.0000"
```

What the parser does about it (`src/io.py`). Everything marked ✔ is confirmed
against the real exports in `data/raw/`:

| Quirk | Handling |
|---|---|
| Boilerplate preamble ✔ | The header row is **found by scanning**, not by skipping a fixed 7 lines. This is not hypothetical: `..._5_2.csv` has no boilerplate at all — one blank line, then straight to the header. A fixed skip would have silently eaten its first rows. |
| CRLF line endings ✔ | Handled; a stray `\r` never reaches the units check. |
| Units row | **Validated, and a mismatch is fatal.** A silent kN→N change would rescale every result by 1000. |
| Quoted values ✔ | Stripped and cast to float; a non-numeric cell raises with its row number. |
| Non-uniform sample rate ✔ | Never assumed. Each real file carries one interval *shorter* than the nominal 0.02 s (0.002–0.010 s observed) — the opposite of the longer gap originally described. Both directions are reported. |
| Load-cell tare ✔ | The cell reads a constant non-zero force before contact (0–2.3 N observed). Measured as `tare_offset_N` and subtracted by default (`--no-tare` to keep it). Small next to 9.5 kN, but it sits exactly where the modulus is fitted. |
| Pre-contact slack ✔ | Contact is the first sample above a threshold that *stays* above it, so a noise spike cannot trigger it. Slack varies a lot between specimens — 0.10 mm to 0.89 mm across these five — so this is doing real work, not trimming a fixed offset. |
| Load-limit termination ✔ | See below — the most consequential property of this dataset. |
| Non-monotonic displacement | **Flagged, never dropped.** |

Force is converted to newtons at parse time so everything downstream is in one
unit system. The raw frame is never modified — cleaning produces a separate
trimmed series.

Every diagnostic lands in `results/diagnostics.csv` and is summarised at the end
of a run.

### Load-limited (truncated) tests

All five current tests **stop against the 9.5 kN limit rather than densifying.**
The frame runs out of load before the specimen runs out of behaviour, so the
curve is cut off partway.

This is detected two ways: explicitly against `force_limit_kN`, and — with no
configuration at all — from the fact that the file *ends at its own maximum
force*, which a specimen-driven test does not do. Either way the test is
flagged, `ended_at_force_limit` is set in `metrics_summary.csv`, and the run
prints a plain-language warning.

What it means for the metrics:

| Metric | Status on a truncated test |
|---|---|
| Young's modulus | **Valid** — the elastic region is early in the curve |
| Plateau stress | **Valid if** the curve reaches the plateau window |
| First peak / crush force | **Valid if** the peak occurs before cut-off |
| Densification strain | **Lower bound** — densification never happened |
| Energy absorption, SEA | **Lower bound** — integrated over a truncated curve |

A genuine *hold* at the limit (frame parked, force flat, crosshead stopped) is
trimmed, since it carries no material response. Samples that are merely near the
limit while still loading are kept — the distinguishing test is mechanical
(has the crosshead stopped?), not a proximity threshold, because a test ramping
into its limit spends its final samples above any threshold while still
measuring real material.

### Duplicate exports

The instrument re-exports the same test under different filenames —
`..._5_1.csv` and `..._5_2.csv` here hold **byte-identical measurements**, one
with the boilerplate header and one without. Registered as two rows they would
pose as replicates and manufacture a spread no second specimen ever produced.

Detection hashes the *parsed columns*, not the file bytes, so a re-export with a
different header is still caught. `scripts/inspect_raw.py --duplicates` reports
groups before you register anything; `run_analysis.py` warns if duplicates are
registered anyway.

---

## `data/specimens.csv`

One row per test. See `data/specimens.example.csv` for a filled-in example.

| column | required | description |
|---|---|---|
| `test_id` | ✔ | unique key; the raw filename without extension works well |
| `raw_file` | ✔ | path relative to `data/raw/` |
| `pattern_name` | ✔ | human label, e.g. `gyroid_v2`, `solid`, `honeycomb_neg_A` |
| `pattern_params` | | free-form **JSON object** of geometric parameters, e.g. `{"cell_mm": 5.0, "wall_mm": 0.8}`. Numeric entries become `param_*` columns and can be plotted against any metric. |
| `replicate` | ✔ | repeat number for that pattern |
| `side_length_mm` | ✔ | nominal cube side |
| `cross_section_area_mm2` | | **leave blank to derive as `side_length²`**; fill in to override |
| `initial_height_mm` | ✔ | height along the load axis |
| `relative_density` | | specimen / bulk material density |
| `mass_g` | | enables SEA per unit mass; without it SEA is reported per unit volume |
| `material` | ✔ | build material |
| `test_date` | | defaulted from a `_YYYYMMDD_` stamp in the filename |
| `notes` | | free text |

All columns must **exist**; the ticked ones must also be **non-null**. Validation
runs before any analysis, and reports every problem at once with row numbers —
it does not fail on the first one.

`scripts/register_test.py` appends rows for you, interactively or from CLI
flags, and refuses duplicate `test_id`s and malformed JSON at entry time.
`--template` writes a skeleton covering every unregistered file, pre-filling
`test_id` and `test_date` from the filename and marking duplicate re-exports in
`notes`, leaving pattern and geometry to enter by hand.

### Current state of the real data

`data/specimens.csv` is a template covering the five exports in `data/raw/`,
with geometry left blank — **the analysis will not run until it is filled in**,
and says exactly which columns are missing if you try. What is still needed:

- **`side_length_mm` and `initial_height_mm`.** Nothing in the raw files records
  specimen size, and without it displacement cannot become strain. Worth a
  sanity check when you fill it: at 9.5 kN a 20 mm cube sees only ~24 MPa, and
  the yield knee in these curves sits near ~14 MPa, which is low for solid PLA.
- **`pattern_name` and `replicate`.** The filenames suffix `_1_1`, `_2_1`,
  `_4_1`, `_5_1`, `_5_2` (no `_3_1`), but per the brief nothing is inferred from
  them. Note `_5_1` and `_5_2` are the same test exported twice — delete one row
  unless it really is a second specimen.
- **`material`**, and optionally `mass_g` to get SEA per unit mass.

---

## Metrics

All in `results/metrics_summary.csv`, one row per `test_id`, joined with the
pattern metadata.

**Units: N, mm, mm², MPa.** Since N/mm² = MPa, the energy integral `∫σ dε` comes
out in MPa, which is numerically identical to MJ/m³. Absolute energy is also
reported in joules.

### Young's modulus
Sliding-window search over strains below 10%: every window at least 0.5% strain
and 10 samples wide is scored by R².

> **A note on the selection rule.** The brief suggested taking the
> highest-R² window. In practice that is not reliable: a foam plateau is often
> just as straight as the initial elastic slope, so on a clean curve both score
> R² ≈ 1.0 and the tie gets settled by floating-point noise — which can hand
> back the *plateau* slope (~1 MPa) as the modulus. This is a real failure, not
> a hypothetical: it is pinned as a regression test in
> `tests/test_metrics.py::test_max_r2_alone_can_select_the_plateau`.
>
> The default is therefore `steepest_high_r2`: among windows statistically tied
> with the best R², take the steepest — which is the initial linear region by
> construction. Set `modulus_selection="max_r2"` for the literal
> highest-R² window.

Also reported: `modulus_r2`, the fitted strain window, and `toe_strain` (where
the fitted line crosses zero stress — a measure of seating slack).

### Plateau stress
Mean stress between `plateau_strain_lo` and `plateau_strain_hi`, with the
standard deviation alongside it.

> **Default is 20–30% strain, per the brief.** For the record, published
> ISO 13314 averages between 20% and **40%** strain. The brief's window is the
> default because it was specified; `--plateau-hi 0.40` follows the standard
> exactly. This is one of the open items worth settling once real densification
> behaviour is visible — for a pattern that densifies before 40% strain, a
> 20–40% window would average the densification rise into the "plateau" and
> overstate it.

### Onset of densification
Three methods; `energy_efficiency` is the default because it needs no threshold.

- **`energy_efficiency`** (default) — the strain maximising
  `η(ε) = [∫₀^ε σ dε] / σ(ε)`. Efficiency climbs through the plateau and turns
  over once cell walls contact. Guarded against the divide-by-near-zero spike at
  very low strain.
- **`tangent`** — first strain where the rolling tangent modulus exceeds a
  multiple of the plateau-region tangent modulus.
- **`iso`** — ISO 13314's companion criterion: stress first reaching 1.3× the
  plateau stress.

If the criterion is only met at the last sample, the test probably stopped
before real densification: the value is flagged `densification_at_end_of_data`
and should be read as a lower bound.

### Energy absorption and SEA
`W = ∫₀^εd σ dε` by trapezoidal integration to the densification strain (falling
back to 50% strain, flagged, if densification could not be located).
`SEA = W·V / mass` in J/g when `mass_g` is recorded; otherwise the volumetric
figure (`energy_absorption_MJ_m3`) is the answer and `sea_basis` says `volume`.

### First peak and crush force
A local maximum only counts as a first peak if the stress afterwards drops by a
real margin — otherwise it is a shoulder on a rising curve, not the cell-wall
buckling signature. `crush_force_N` is the first-peak load when there is one,
and the maximum before densification when there is not;
`has_distinct_first_peak` tells you which.

---

## Figures

### (a) `stress_strain_envelope.png`
Every curve resampled onto a common strain grid, overlaid and coloured by
pattern. The grey band spans min→max stress at each strain **across all
patterns** — the "what is achievable at all" envelope. A lighter band per
pattern shows the spread across *its replicates*, so pattern-to-pattern
differences are visually separable from measurement noise.

The grid runs to the **shortest** test's maximum strain, so every curve is
defined at every grid point and the envelope is real rather than an artefact of
curves ending at different strains. The subtitle names that cap; use
`--strain-max` to override it.

### (b) `metric_space_*.png`
Each derived metric against another, one point per test, coloured by pattern,
with a convex hull marking the achievable property region. Also
`param_<key>_<metric>.png` for any numeric `pattern_params` key with at least
two distinct values across the dataset.

**Chart conventions.** Colour is assigned from a fixed, colourblind-validated
categorical order — never cycled or generated. Because pattern count is
open-ended and scatter plots put every pair of colours side by side, marker
shape and line style carry the same identity, so no figure depends on colour
alone; past eight patterns the colour repeats but the marker does not.
`metrics_summary.csv` is the authoritative table view.

---

## Regenerating when data changes

`results/.input_manifest.json` records a content hash of `specimens.csv`, every
raw CSV, and the analysis config. A run is skipped when nothing changed and
redone when anything did — editing a specimen row, adding a raw file, or
changing a parameter all invalidate it.

```bash
python scripts/run_analysis.py --watch     # re-run on every change
python scripts/plot_envelope.py --watch    # redraw on every change
python scripts/run_analysis.py --force     # ignore the manifest
```

A file that fails to parse does not sink the batch: it is recorded in
`diagnostics.csv` with the reason, the other tests still analyse, and the
failure count is printed at the end.

---

## Configuration

Defaults live in `src/config.py`, all documented inline. Override per run:

```bash
python scripts/run_analysis.py --plateau-lo 0.20 --plateau-hi 0.40
python scripts/run_analysis.py --densification-method iso
python scripts/run_analysis.py --toe-compensation
python scripts/run_analysis.py --force-limit-kN 9.5
python scripts/run_analysis.py --config my_params.json    # any field
```

Project-wide settings belong in `analysis_config.json` at the repo root, which
every script loads unless `--config` points elsewhere. Keys starting with `_`
are treated as comments, since JSON has none.

**Toe compensation** (off by default) shifts the strain axis so the fitted
elastic line passes through the origin, removing seating slack that threshold-
based contact detection leaves behind. It changes every strain-referenced metric,
so it is opt-in; `toe_strain` is always reported so you can see what it would do.

Worth trying on this dataset: pre-contact slack ranges from 0.10 mm to 0.89 mm
across the five tests, and the toe is curved rather than a clean offset, so
threshold-based contact detection leaves a different amount of seating
compliance in each specimen. That inflates the apparent spread in modulus
between nominally identical specimens.

---

## Tests

```bash
python -m pytest tests/ -q       # 125 tests
```

Coverage is aimed at the two things most likely to go quietly wrong: parsing
real instrument output, and metric extraction. Metrics are verified against
synthetic curves with analytically known modulus, plateau stress, densification
strain and first peak, so a wrong answer fails rather than merely looking
plausible.

The parser is tested against **real instrument exports**, not a description of
them: `tests/fixtures/` holds `solid_compression_20260717_192209_1_1.csv` (the
full format) and the no-boilerplate variant of `..._5_2.csv`. Both are genuine
files from the 10 kN frame, so the format tests fail if the parser stops
handling what the machine actually writes. `src/synthetic.py` still generates
instrument-format files, now used for the demo and for tests that need a curve
with analytically known properties.

---

## Repository layout

```
data/
  raw/                     untouched instrument CSVs
  specimens.csv            metadata, single source of truth (you maintain this)
  specimens.example.csv    filled-in example of the schema
results/                   all generated
  metrics_summary.csv      one row per test — the main deliverable
  diagnostics.csv          every warning raised, per test
  resampled_curves.csv     the common-grid curve matrix behind the envelope
  curves/<test_id>.csv     cleaned stress-strain curve per test
  figures/                 envelope and property-space plots
src/
  config.py                every tunable parameter, documented
  specimens.py             metadata schema and validation
  io.py                    raw CSV parsing and cleaning
  mechanics.py             stress/strain conversion, integration, resampling
  metrics.py               metric extraction
  envelope.py              plotting
  pipeline.py              orchestration and change detection
  synthetic.py             instrument-format generation for tests and the demo
analysis_config.json       machine setup (load limit); loaded automatically
scripts/
  inspect_raw.py           report what a raw file contains, before registering
  register_test.py         onboard a raw CSV into specimens.csv
  run_analysis.py          raw -> metrics_summary.csv (+ figures)
  plot_envelope.py         figures from an existing metrics summary
  make_demo_data.py        synthetic design space for trying the pipeline
tests/
```

---

## Open items

Resolutions to the questions raised in the brief, all reversible:

1. **Filename parsing.** `_1_1` is treated as an opaque part of the ID —
   `pattern_name` and `replicate` are entered manually, as assumed in the brief.
   The one exception is `test_date`, which is offered as a *default* from a
   `_YYYYMMDD_` stamp and can be overridden.
2. **Plateau window.** Defaults to the brief's 20–30%; ISO 13314 proper is
   20–40%. Revisit per pattern once real densification strains are known — see
   the note under *Plateau stress*.
3. **Units.** MPa / N / mm throughout, as proposed. Energy is reported both
   volumetrically (MJ/m³) and absolutely (J).

Now that real data has arrived, one more worth settling:

6. **The plateau window may not be reachable.** These tests stop at 5.5–5.9 mm
   of travel. On a 20 mm specimen that is ~28–30% strain, so the 20–30% plateau
   window is only just covered and the ISO 20–40% window would not be reachable
   at all. If the plateau stress matters, either raise the load limit or expect
   the window to need moving down for this geometry.

Two additions worth knowing about:

4. **`mass_g` added to the schema.** SEA was requested per unit mass, and no
   column carried mass. It is optional — without it, SEA falls back to the
   volumetric figure and says so in `sea_basis`.
5. **Modulus selection rule changed** from the brief's suggestion, for the
   reason documented under *Young's modulus*.
