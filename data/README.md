# Data provenance

Two Parquet files, both derived from one published dataset. Nothing else in this project
reads a database file, a network endpoint, or a local path outside this directory.

| File | Rows | Grain | What it is |
|---|---|---|---|
| `honey_daily_source.parquet` | 41,710 | one row per hive per day | daily-grain slice of the published sensor archive |
| `honey_model.parquet` | 26,215 | one row per hive-day with a valid neighbour on both sides | the modelling table |

---

## Source

> Senger, D., Gruber, C., Kluss, T., & Johannsen, C. (2024). Weight, temperature and
> humidity sensor data of honey bee colonies in Germany, 2019–2022.
> *Data in Brief, 52*, 110015. https://doi.org/10.1016/j.dib.2023.110015

Dataset: https://doi.org/10.5281/zenodo.10407693 (Zenodo record 10407693, v2,
20 December 2023, open access), by the same authors at the University of Bremen and
Hiveeyes.

78 honey bee colonies across Germany, June 2019 – December 2022, instrumented under a
citizen-science project. Each hive carries five internal temperature sensors, an external
temperature sensor, a combined temperature/humidity/pressure sensor, and a scale.
Beekeepers logged their own inspections and interventions through a web app.

**Attribution correction.** Milestone 3, Section 1, described the primary dataset as
"HOBOS" and credited the observation logs to "USDA Tucson". Both are wrong. HOBOS is a
separate Kaggle dataset listed in [`../data-guide.md`](../data-guide.md) and is not used
here; the observation logs are the German project's own web-app records. Cite the paper
and the Zenodo DOI above.

### Licensing — unresolved, read before redistributing

The Zenodo record is marked **open access**, but it declares **no license**: the DataCite
`rightsList` for `10.5281/zenodo.10407693` is empty and the record shows no license
statement. Open access is not itself a grant of redistribution rights.

The files here are a *derived* table: a daily-grain subset of the publisher's own daily
aggregates, joined to features this project engineered, at roughly 0.05% of the 5.6 GB
source archive. They are committed so the milestone can be reproduced without credentials,
which is what the grader asked for.

If the authors' intended license turns out to prohibit redistribution, the fallback is to
drop both Parquet files, keep `scripts/build_honey_model.py`, and ship a synthetic
schema-only fixture for the smoke test. **Open item: email the corresponding author to
confirm the license.**

## What the publishers did to the data before we saw it

This section constrains every downstream interpretation, so it belongs first.

From the paper's methods:

- **Range filters.** Weight values above 150 kg or below −50 kg, temperatures above 85 °C
  or below −40 °C, and relative humidity outside 0–100% were excluded.
- **Rate filter.** Weight changes were differentiated and values above 0.3 kg/minute
  excluded, on the stated grounds that such "sudden drastic changes in the weight […] are
  usually induced by activities by the beekeeper."
- **Aggregation.** Raw readings arrive every 5–10 seconds. The one-minute series is the
  median of each minute; the hourly and daily series are *averages of the one-minute
  series*. Files are suffixed `_m`, `_h`, `_d`.

Three consequences that Milestone 3 did not account for:

1. **The rate filter produced a second weight column, and we used the other one.**
   `weight_kg` retains every jump. `weight_delta_noOutlier` is the same delta series with
   flagged jumps zeroed, and `weight_kg_noOutlier` is its cumulative sum. The pipeline
   built its target from `weight_kg`. Rebuilt on `weight_kg_noOutlier`, the same modelling
   grain yields **101 extremes instead of 445** — 94% of them are exactly the jumps the
   publishers attribute to beekeeper handling. See
   [`../results/extremes_weight_series.csv`](../results/extremes_weight_series.csv).
   Neither column is "correct": one keeps handling artifacts, the other erases genuine
   abrupt mass loss along with them.
2. **`end_of_day_weight_kg` is a misnomer.** The daily value is the *mean* of the day's
   minute readings, not the last one. The target is a change in daily mean weight.
3. **`honey_daily_summary` does not aggregate.** The source is already daily, so the
   `avg_*` prefixes are accurate descriptions of upstream work, not of anything our SQL
   does. Raw, clean and summary row counts are all 29,172.

### A defect in the published event-distance columns

The `*_last_dif` / `*_next_dif` columns give the time distance to the nearest logged
beekeeper event — but **not in a consistent unit**. Measured against consecutive-day
observations, they advance by exactly 1.0 per day for some hives and by exactly 86400.0
for others; the unit follows the hive's source file, not the event type.

`honeymodel.data.normalise_event_distance_units` infers the unit per hive and converts
everything to days. Any threshold comparison made on the raw columns is wrong for the
seconds-unit hives — which is part of why Milestone 3's extremes notebook reported that
0% of extreme events fall within one day of a logged event. Corrected, it is 7.8% for
honey harvest and 13.2% for feeding. See
[`../results/event_distance_units.csv`](../results/event_distance_units.csv).

## Which daily files are in scope

The publisher's daily files come in two arrangements:

| dataset | files | rows | hives | what it is |
|---|---|---|---|---|
| `years` | 78 | 29,172 | 78 | the full 2019–2022 archive, one file per hive-year |
| `events` | 191 | 12,538 | 55 | 3-month windows either side of a swarming or colony-death event |

`events` re-slices hive-days that `years` already contains: only 948 of its hive-days
(3.2%) are new, and it holds 527 internal duplicates. The pipeline therefore uses `years`
only — `01_raw_anchor.sql` filters on it. Unioning `events` in would inject duplicate
hive-days rather than new coverage.

This also settles an open risk from the Milestone 3 plan, which recorded that only 151 of
453 daily CSVs had been processed and that all figures were subset estimates. The 29,172
rows ingested are 100% of the `years` daily grain, so the hive count, coverage and 1.7%
extreme rate are full-archive figures, not estimates.

## Regenerating these files

```bash
# Rebuild the modelling table from the committed source snapshot. No network needed.
python scripts/build_honey_model.py

# Refresh the source snapshot itself from the team's remote DuckDB.
# The ONLY step that needs Tailscale and .env credentials.
python scripts/pull_source_snapshot.py
```

`build_honey_model.py` asserts that the result reproduces the published Milestone 3
dataset exactly — 26,215 rows, 78 hives, 2019-06-25 to 2022-12-30, target mean 0.002042,
sd 1.840522, min −65.322940, max 39.098176, 445 extremes — and fails the build if any of
those move. It also regenerates [`../sql/06_schema.sql`](../sql/06_schema.sql) so the DDL
cannot drift from the table.

## Columns worth knowing about

`honey_model` carries 71 columns. Three groups need care:

- **Never use as features.** `next_day_weight_kg`, `next_observation_date`,
  `days_to_next_observation`, every `*_next_dif`, every `nearest_*_event_days`, and every
  `extreme_*_flag` (those are `ABS(target) > k`, i.e. the label). The list is enforced as
  `honeymodel.features.FORBIDDEN_COLUMNS`; `build_feature_matrix` raises `LeakageError`
  rather than returning a matrix containing one. They are present because the
  extreme-event investigation needs them.
- **Quality flags, not filters.** `implausible_weight_flag` (any of previous/current/next
  weight below a 5 kg floor), `robust_outlier_flag` (>5 MAD from the hive's own centred
  7-day median, scaled by that hive's day-to-day change MAD), `sensor_dropout_flag`
  (a large drop followed by a comparable recovery). Nothing is dropped; the project's
  convention is to retain and flag so each rule's effect can be measured.
- **Publisher flags.** `source_outlier_flag`, `source_no_jump_flag`,
  `source_weight_no_outlier_kg`. Note that `source_outlier_flag` is never TRUE at daily
  grain — the 0.3 kg/minute rule fires on minute-grain rows — so it documents upstream
  processing rather than marking anything in this table.
