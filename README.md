# Compression Test Analysis

Tools for analysing uniaxial compression tests on cube specimens containing
negative-volume internal patterns. Raw instrument CSVs go in; engineering
stress–strain curves, standard cellular-solid metrics, and design-space
envelope figures come out.

Three questions this is built to answer:

1. **Per pattern** — how stiff is it, where does it plateau, when does it
   densify, how much energy does it absorb?
2. **Across the design space** — given every pattern tested, what range of
   mechanical response is achievable, and how much of that spread is
   pattern-to-pattern rather than replicate-to-replicate noise?
3. **What drives it** — do the design and manufacturing parameters (cell size,
   wall thickness, layer height, print batch, position on the build plate,
   print and test dates, …) correlate with the measured properties, and which
   ones genuinely matter once the others are accounted for?

---

## Quickstart

```bash
python -m pip install -r requirements.txt

# 0. Check what a new export actually contains (no metadata needed).
python scripts/inspect_raw.py --duplicates

# 1. Drop instrument CSVs into data/raw/, then register each one once.
python scripts/register_test.py --list          # what still needs registering
python scripts/register_test.py --template      # skeleton specimens.csv to fill in
python scripts/register_test.py --interactive   # or be prompted field by field

# 2. Analyse everything: metrics, figures, and parameter-property correlation.
python scripts/run_analysis.py
python scripts/run_analysis.py --color-by relative_density   # colour by a parameter

# 3. Slice the correlation analysis without re-parsing.
python scripts/correlate.py --list                    # what can be analysed, and why not
python scripts/correlate.py --where material=PLA      # within one level
python scripts/correlate.py --parameters cell_mm wall_mm layer_height_mm

# 4. Redraw the envelope figures (or explore other metric pairs).
python scripts/plot_envelope.py --pair modulus_MPa sea_J_per_g
```

Machine setup lives in `analysis_config.json` at the repo root and is picked up
automatically. It currently records the 9.5 kN load limit:

```json
{ "force_limit_kN": 9.5 }
```

**Want to see it work before you have data registered?** Generate a synthetic
64-pattern design space — cell size × wall thickness × topology, each printed
with its own layer height, print speed and nozzle temperature, in one of four
print batches at one of eight build-plate positions, with print and test dates,
and stopped at the same 9.5 kN limit — and run the whole pipeline on it:

```bash
python scripts/make_demo_data.py            # writes demo/ (synthetic, disposable)
python scripts/run_analysis.py --root demo
```

The causes in the demo are planted (see `PLANTED_TRUTH` in `src/synthetic.py`),
so it doubles as a check that the correlation analysis finds what is there and
nothing that is not — see *Parameter–property correlation* below.

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

All four current tests **stop against the 9.5 kN limit before densifying** —
37.1 MPa on a 16 mm cube, reached at 29–36% strain. The frame runs out of load
before the specimen runs out of behaviour, so the curve is cut off partway.
`figures/densification_check.png` shows how far short each test stopped (see
*Onset of densification*).

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
| `pattern_params` | | optional **JSON object** of parameters, e.g. `{"cell_mm": 5.0, "wall_mm": 0.8}` — though extra columns (below) are easier |
| `replicate` | ✔ | repeat number for that pattern |
| `side_length_mm` | ✔ | nominal cube side |
| `cross_section_area_mm2` | | **leave blank to derive as `side_length²`**; fill in to override |
| `initial_height_mm` | ✔ | height along the load axis |
| `relative_density` | | specimen / bulk material density |
| `mass_g` | | enables SEA per unit mass; without it SEA is reported per unit volume |
| `material` | ✔ | build material |
| `test_date` | | when the test ran — a date or date-time (`2026-07-17`, `2026-07-17 19:22`). Defaulted from a `_YYYYMMDD_` stamp in the filename, which is the *export* time: replace it with the real one |
| `notes` | | free text |
| `print_date` | | when the specimen was printed, same formats. With both dates, the specimen's age at test is derived as `print_to_test_days` |

All columns must **exist** — except `print_date`, which can be left out until
you have it (`register_test.py` adds the column the first time it is given one)
— and the ticked ones must also be **non-null**. Validation runs before any
analysis, and reports every problem at once with row numbers — it does not fail
on the first one. A test dated before its print is one of those problems.

### Design and manufacturing parameters: just add columns

**Every column beyond the table above is a parameter.** For 64 tests, typing
`layer_height_mm` once as a header beats typing a JSON object 64 times:

```
test_id, raw_file, …, notes, cell_mm, wall_mm, topology, layer_height_mm, print_orientation, nozzle_temp_C
gyroid_c5_w0.8_r1, …,        5.0,     0.8,     gyroid,   0.20,            XY,                210
```

- A column whose every filled cell is a number is **numeric**; anything else
  (`XY`/`Z`, `gyroid`) is **categorical**. Decided per column, so one stray text
  cell makes the whole column categorical rather than half of it.
- Blank cells are *missing*, not zero — a parameter that only applies to some
  patterns (a re-entrant angle, say) is simply left empty elsewhere.
- Headers are sanitised (`Layer Height (mm)` → `Layer_Height_mm`), and the
  empty `Unnamed: N` columns Excel leaves behind are ignored.
- `pattern_params` JSON still works and can be mixed with columns; a parameter
  given in both places must agree, or validation stops.

Each parameter appears in `metrics_summary.csv` as `param_<name>`, and is picked
up by the correlation analysis automatically. See `data/specimens.example.csv`.

`scripts/register_test.py` appends rows for you, interactively or from CLI
flags, and refuses duplicate `test_id`s and malformed JSON at entry time.
`--template` writes a skeleton covering every unregistered file, pre-filling
`test_id` and `test_date` from the filename and marking duplicate re-exports in
`notes`, leaving pattern and geometry to enter by hand.

### Print time, test time, batch and bed position

These are recorded like any other parameter, and get some special handling:

| column | record as | catches |
|---|---|---|
| `print_date` | date or date-time (core column) | drift across the print campaign: filament moisture, nozzle wear, recalibration |
| `test_date` | date or date-time (core column) | drift across the test campaign |
| `print_batch` | a label: `B1`, `plate_0715_a`, … | a one-off bad batch, which no time trend can see |
| `bed_position` | a label: `R1C3`, `front_left`, … | particular spots on the plate behaving differently |

- **Dates are analysed as elapsed time** — days since the earliest — so the
  question is "does it matter how late in the campaign?". Figures label the axis
  that way. The gap between the two dates becomes `print_to_test_days`: PLA
  keeps taking up moisture and ageing physically after printing, so a
  specimen's age at test is a parameter in its own right.
- **The three cannot all be analysed together**, because test date = print
  date + age, exactly. With all three present `test_date` is left out and the
  summary says so; `correlate.py --parameters` analyses it in place of one of
  the others.
- **Batch and print date will usually be confounded**, since a batch is printed
  on one day. Parameters this entangled (association ≥ 0.8) are reported as
  confounded, and neither is held fixed when testing the other — that would
  cancel both. A shared result is one effect with two names. They are not
  interchangeable, though: in the demo, batch B3 is weak, and the batch label
  finds it while print date shows nothing, because B3 being the third of four
  print days is not a trend. The same check covers design parameters: if every
  large-cell pattern went into batch 1, cell size and batch are reported as
  confounded. Spreading patterns across batches, and batches across days, is
  what lets the data tell them apart.
- **Bed position: a label, numbers, or both.** A label treats each spot as a
  level of its own, and a spot needs at least 3 tests to take part. If you
  suspect a *gradient* (warping, one edge of the bed running cooler), add
  numeric `bed_x_mm` and `bed_y_mm` columns as well. If you suspect centre
  versus edge, a `bed_r_mm` column — distance from the plate centre — turns it
  into a single trend, which is the most sensitive test of the three.

### Current state of the real data

`data/specimens.csv` registers the four distinct tests in `data/raw/` as 16 mm
PLA cubes at relative density 0.8. `_5_2` is a byte-identical re-export of
`_5_1`, so it is left out; its raw file stays in place. At the 9.5 kN limit a
16 × 16 mm face reaches 37.1 MPa.

| test | E (MPa) | plateau σ, 20–30% (MPa) | σ at 10% (MPa) | σ at 20% (MPa) | W to 20% (MJ/m³) | strain at cut-off |
|---|---|---|---|---|---|---|
| `_1_1` | 659 | 33.5 | 26.1 | 30.4 | 4.72 | 29.0% |
| `_2_1` | 625 | 29.1 | 22.7 | 26.3 | 4.14 | 35.5% |
| `_4_1` | 603 | 28.9 | 21.9 | 25.7 | 4.00 | 34.9% |
| `_5_1` | 594 | 28.5 | 21.9 | 25.4 | 3.93 | 35.7% |

- **None of the four reached densification.** The gauge r (see *Onset of
  densification*) ends at 0.50–0.62 where onset is r = 1: at the cut-off the
  curves stiffen at about 90 MPa, and onset needs 154–178 MPa. Extrapolating r
  linearly puts onset near 50–60% strain at 55 MPa or more, which is at least
  14 kN on this face; extrapolating the stress curve's current shape, r never
  reaches 1. The curves look like a dense solid yielding and hardening, as
  expected with so little void space, not a foam collapsing and locking up.
  Seeing densification
  would take a smaller cross-section (a 12 mm cube reaches 66 MPa at 9.5 kN, at
  the cost of fewer cells across it) or a bigger frame.
- So **ε_D and energy to densification are lower bounds**, and are left out of
  the correlation. Modulus, plateau stress, σ at 10% and 20%, and W to 20% are
  complete for all four tests.
- **`_1_1` is the odd one out.** It is about 8% stiffer and 16–18% stronger
  than the other three, which agree within 5%. It is not seating: toe
  compensation leaves E unchanged and moves the stresses by about 1%. Whether
  it is a different pattern or a stronger print is for the metadata to say.
- **Relative density 0.8 is a reading of "0.8 density".** If it meant
  0.8 g/cm³, then against ~1.24 g/cm³ for solid PLA, ρ\* ≈ 0.65. It is the same
  for every test either way, so it does not move the correlations. Weighing the
  cubes into `mass_g` settles it, and gives SEA per gram.

Still to fill in before the correlation analysis has anything to work with:

- **`pattern_name`** — currently placeholders taken from the filename
  (`pattern_1`, …).
- **`print_date`**, and the real **`test_date`**: the filename stamp is a single
  export time shared by all four files.
- **`print_batch`** and **`bed_position`**.
- **One column per design parameter** of the patterns (cell size, wall
  thickness, …).

When there is nothing to correlate, `run_analysis.py` lists which columns are
still blank and which are the same for every test.

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

**Absolute values include the frame.** The export has no extensometer channel,
so `Displacement` is crosshead travel, and the stiffness measured is the
specimen's in series with the frame, load cell and platens. E therefore comes
out below the specimen's own, by an amount set by the frame. Within one
specimen size the correction is the same monotonic function for every test, so
the *order* of the tests is preserved and rank correlations are unaffected. For
an absolute modulus, measure the frame's compliance (platens together, or a
steel block) and subtract it.

### Plateau stress
Mean stress between `plateau_strain_lo` and `plateau_strain_hi`, with the
standard deviation alongside it.

Not reported when the test spans less than 90% of the window
(`plateau_min_coverage`): a test that stops at 23% strain would average only the
low end of a 20–30% window, which reads low on the usual rising plateau. The
coverage is in `plateau_window_coverage`. On truncated tests, compare the
fixed-strain stresses below instead.

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

If the criterion is only met within 1% strain of the end of the data, the test
probably stopped before real densification: the value is flagged
`densification_at_end_of_data` (and `densification_censored`) and should be
read as a lower bound.

**How close did a test get?** A flag says that a test stopped short;
`figures/densification_check.png` says by how much. It plots stress, η, and the
gauge

    r = (dσ/dε) · W / σ²

for every test. Since dη/dε = 1 − r, η rises while r < 1 and peaks, which is
densification onset, exactly where r crosses 1. r is ½ through linear-elastic
loading, drops after yield, and climbs as the specimen hardens. The subtitle
gives the verdict, and a dot marks where each test ends.

### Energy absorption and SEA
`W = ∫₀^εd σ dε` by trapezoidal integration to the densification strain (falling
back to 50% strain, flagged, if densification could not be located).
`SEA = W·V / mass` in J/g when `mass_g` is recorded; otherwise the volumetric
figure (`energy_absorption_MJ_m3`) is the answer and `sea_basis` says `volume`.

### First peak and crush force
A local maximum only counts as a first peak if the stress afterwards drops by a
real margin — otherwise it is a shoulder on a rising curve, not the cell-wall
buckling signature. Peaks and drops are judged on the curve smoothed over 0.3%
strain: on a soft specimen, load-cell noise alone produced 2–3% "drops" that
passed the test (7 false peaks in 32 peak-free synthetic specimens before the
smoothing, none after). `crush_force_N` is the first-peak load when there is
one, and the maximum before densification when there is not;
`has_distinct_first_peak` tells you which.

### Fixed-strain metrics — the fair comparison for truncated tests
At each strain in `fixed_strain_levels` (default 10% and 20%):
`stress_at_10pct_MPa`, `energy_to_10pct_MJ_m3`, and with a mass
`sea_to_10pct_J_per_g` — likewise for 20%. A test that stops short of a level
gets NaN there, never an extrapolation.

These matter because of the load limit. On a test the frame stopped before
densification, energy to densification is only a lower bound — it records where
the test stopped. Energy *to 20% strain* is a complete measurement for every test
that got that far. On the real exports, these are the properties that stay
defined for every test.

### Lower bounds are flagged, not silently mixed in
`densification_censored`, `energy_censored` and `crush_censored` mark values
that are only lower bounds. They stay in the table, and the correlation
analysis leaves them out (`--include-censored` to override).

---

## Figures

### (a) `stress_strain_envelope.png`
Every curve resampled onto a common strain grid and overlaid. The light band
spans min→max stress at each strain **across all tests** — the "what is
achievable at all" envelope.

How the curves are coloured depends on how many patterns there are:

- **Up to 8 patterns** — one colour and legend entry per pattern, plus a
  per-pattern band showing the spread across *its replicates*, so
  pattern-to-pattern differences are separable from measurement noise.
- **More than 8** — colour can no longer identify a pattern, so every test is
  drawn as a plain curve in one colour, with the **middle-50% band** and the
  **median curve** on top. The legend has four entries, not 64.
- **`--color-by <parameter>`** — curves coloured on a continuous scale by a
  numeric parameter (with a colorbar), or by the levels of a categorical one. A
  date colours by days since the earliest. If the colour gradient lines up with
  curve height, that parameter orders the response: a visual correlation.

The grid runs to the **shortest** test's maximum strain, so every curve is
defined at every grid point and the envelope is real rather than an artefact of
curves ending at different strains. The subtitle names that cap; use
`--strain-max` to override it.

### (b) `metric_space_*.png`
Each derived metric against another, one point per test, with a convex hull
marking the achievable property region. Colouring follows the same rule as the
overlay; every test is the same plain circle.

### (c) `densification_check.png`
Stress, energy efficiency and the densification gauge r for every test, with
the verdict in the subtitle — see *Onset of densification*. Above 8 tests the
curves go to one colour, as in the overlay.

**Chart conventions.** One marker shape throughout. Categorical colour comes
from a fixed, colourblind-validated order and is used only when there are few
enough levels to tell apart; continuous colour is a one-hue blue ramp; text is
always ink, never series colour. `metrics_summary.csv` is the authoritative
table view.

---

## Parameter–property correlation

Runs automatically at the end of `run_analysis.py` whenever at least one
parameter varies; `scripts/correlate.py` re-runs or slices it. Every parameter
(the `param_*` columns, plus relative density, specimen size, material and
pattern) is tested against every property in `correlation_properties`.

### What it does, and why

| Step | Why |
|---|---|
| **Spearman ρ** (rank correlation) is the headline measure, Pearson r alongside | Cellular-solid properties follow power laws (E ∝ ρ\*²): monotonic but not straight. Rank correlation sees those at full strength, and no single outlying test can drag it. |
| **False-discovery-rate control** (Benjamini–Hochberg) → a q-value per pair | Screening 7 parameters × 10 properties at p < 0.05 would produce ~4 "significant" associations from noise alone. Only q < 0.05 is called significant. |
| **Partial ρ**: the same correlation with every *other* parameter held fixed | In a real design, parameters move together. Strong pairwise / weak partial means *borrowed from a correlated parameter*. Weak pairwise / strong partial means *a real effect masked by a bigger one*. |
| Categorical factors are held fixed in the partials too | Leaving one out is not neutral. In the demo, before this, cell size's partial with modulus came out at **+0.69** — "bigger cells are stiffer at equal density" — because at fixed density a bigger cell implied the stiffer topology. |
| **Collinearity**: a VIF per parameter, from ranks *and* logs | A parameter predictable from the others (VIF ≥ 10) has no separable effect in this data; its partial is withheld rather than reported wrong. Logs matter because derived parameters are usually ratios: ρ\* = k·wall/cell is exact in logs but not in ranks. Ranks alone (R² = 0.77) passed topology as separable when the other three fixed it. |
| **Confounding** between every two parameters: \|Spearman ρ\| for two numeric, correlation ratio η for numeric–categorical, Cramér's V for two categorical | Pairs at ≥ 0.8 are reported as confounded, and are not held fixed for each other, since that would erase both effects. |
| **Kruskal–Wallis + ε²** for categorical factors | Levels with fewer than 3 tests are left out; a factor that is unique per test (64 different pattern names) cannot be tested and is reported as such. |
| **Quade's rank ANCOVA + partial η²**, the categorical partial | Ranks the property, takes out what the other parameters explain, and compares the levels on what is left. In the demo, a 12% batch defect is invisible next to the 30× density spread (ε² = 0.02, not significant) and plain once density is held fixed (partial η² = 0.33, q = 0.002). |
| **Gibson–Ashby scaling** vs relative density: one exponent, a prefactor per categorical level, plus each level fitted alone | The exponent separates bending-dominated (E ∝ ρ\*², σ ∝ ρ\*^1.5) from stretch-dominated (∝ ρ\*) architectures. The per-level prefactor matters: pooling a denser-*and*-stronger topology with a weaker one tilts the slope — in the demo, pooled strength gave 1.62, "steeper than either mode"; with per-topology prefactors, 1.51 against a planted 1.5. |
| **Censored values excluded** | A lower bound from a load-limited test mostly measures where the test stopped. |

### Outputs

`results/correlation/` — `correlations.csv` (one row per parameter × property:
ρ with 95% CI, p, q, partial ρ or partial η² and its q, Pearson r, ε², n, how
many lower bounds were excluded, and a plain-language note), `parameters.csv`
(what was analysed, VIF, what each is confounded with, the origin of each date
axis, and why anything was not), `collinearity.csv`, `scaling_laws.csv`, and
`summary.txt` — the findings in words, also printed at the end of a run.

`results/figures/correlation/`:

- `heatmap.png` — parameters × properties, **pairwise ρ beside partial ρ**.
  The fastest way to see what matters, and what is borrowed or masked.
- `property_<name>.png` — that property against every parameter, one panel
  each. Trend line only where significant; a masked effect is labelled as such;
  hollow points are excluded lower bounds; dates run as days since the earliest.
- `collinearity.png` — how strongly every two parameters move together (|ρ|, η
  or Cramér's V, by kind), with each one's VIF; confounded pairs in bold.
- `categorical_effects.png` — pairwise ε² beside partial η² for each
  categorical factor. A manufacturing effect hidden behind the pattern
  differences shows up only in the partial panel.
- `scaling_laws.png` — log-log against relative density, coloured by the factor
  whose levels are fitted separately, with the Gibson–Ashby reference slopes.

### Checked against planted truth

The demo's causes are known, and the analysis recovers them:

| Planted | Found |
|---|---|
| Modulus ∝ ρ\*² | n = 2.00 [1.92, 2.07], bending-dominated |
| Strength ∝ ρ\*^1.5 | n = 1.50 [1.43, 1.57], bending-dominated |
| Layer height: ±10% — tiny next to density's 30× spread | modulus: pairwise ρ = −0.01 (n.s.), **partial ρ = −0.69, q < 0.001** — flagged as masked |
| Batch B3 12% weaker | plateau σ: ε² = 0.02 (n.s.), **partial η² = 0.33, q = 0.002** — masked |
| Edge columns of the bed 15% weaker | modulus: ε² = 0.09 (n.s.), **partial η² = 0.46, q < 0.001** — masked |
| Each batch printed on one day | batch ~ print date reported as confounded (η = 0.97) |
| Honeycomb 1.3× stiffer than gyroid | modulus: ε² = 0.16, q = 0.004; partial withheld, since ρ\*, wall and cell fix the topology |
| Print speed, nozzle temperature, print-to-test age: **no effect** | not significant, apart from one partial on a single property (nozzle temperature → ε_D, q = 0.02). The summary marks it *isolated*: FDR control at 0.05 expects about 1 in 20 discoveries to be false |
| Relative density computed from wall ÷ cell | VIF 346 / 172 / 137 — reported as inseparable, partials withheld |

The findings — masked, found, confounded, null — are pinned in
`tests/test_correlation.py`.

### How small an effect it can see

Simulated on the demo design (64 tests, 4 batches of 16, 8 bed positions of 8),
re-run with 20 seeds:

| planted effect | found by the partial (q < 0.05) |
|---|---|
| bed edge 6% weaker | 4 of 20 runs |
| bed edge 10% weaker | 10 of 20 |
| bed edge 15% weaker | 18 of 20 |
| one batch 8% weaker | about half |

So at 64 tests, a ~15% manufacturing effect is caught reliably and a ~10% one
about half the time. Smaller effects need more tests per level: fewer bed
positions, or more specimens at each. With no effect present, p < 0.05 comes up
in 4–7% of simulated tests, close to the nominal 5%, and FDR control acts on
top of that.

### Reading it

- **Correlation is not causation.** This is screening: a significant
  association says where to look, and a designed follow-up says whether the
  parameter is responsible.
- If relative density sits beside the geometry it is computed from, the three
  cannot be separated (the summary says so). Analyse one set at a time:
  `correlate.py --parameters cell_mm wall_mm topology layer_height_mm` — with the
  derived density gone, cell and wall come apart cleanly.
- To look inside one level of a significant categorical factor:
  `correlate.py --where topology=gyroid`. Subset runs go to their own folder.
- A real effect usually shows across related properties: strength, energy and
  modulus together. One that shows on a single property (*isolated* in the
  summary) is a lead to re-test, not a finding.
- With few tests per level, a 95% interval will occasionally miss — one in
  twenty by construction.

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

Pre-contact slack ranges from 0.10 mm to 0.89 mm across the real tests, and the
toe is curved rather than a clean offset, so it was worth checking. On these
tests toe compensation leaves E unchanged and moves the stresses by about 1%
(W to 20% by up to 3%), so seating is not what separates them.

---

## Tests

```bash
python -m pytest tests/ -q       # 226 tests
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
  correlation/             correlation tables and summary.txt
  figures/                 envelope, property-space and correlation/ plots
src/
  config.py                every tunable parameter, documented
  specimens.py             metadata schema and validation
  io.py                    raw CSV parsing and cleaning
  mechanics.py             stress/strain conversion, integration, resampling
  metrics.py               metric extraction
  envelope.py              plotting
  correlation.py           parameter-property statistics and figures
  pipeline.py              orchestration and change detection
  synthetic.py             instrument-format generation for tests and the demo
analysis_config.json       machine setup (load limit); loaded automatically
scripts/
  inspect_raw.py           report what a raw file contains, before registering
  register_test.py         onboard a raw CSV into specimens.csv
  correlate.py             parameter-property correlation, sliced as needed
  run_analysis.py          raw -> metrics_summary.csv (+ figures)
  plot_envelope.py         figures from an existing metrics summary
  make_demo_data.py        64-test synthetic design space with planted causes
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

6. **The plateau window is only just reachable.** On the 16 mm cubes these
   tests stop at 29–36% strain. The 20–30% window is covered (`_1_1` by 90%,
   the minimum allowed), but the ISO 20–40% window is not. Moving to ISO would
   need a higher load limit or smaller specimens.

Two additions worth knowing about:

4. **`mass_g` added to the schema.** SEA was requested per unit mass, and no
   column carried mass. It is optional — without it, SEA falls back to the
   volumetric figure and says so in `sea_basis`.
5. **Modulus selection rule changed** from the brief's suggestion, for the
   reason documented under *Young's modulus*.
