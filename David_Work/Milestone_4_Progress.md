# Milestone 4 — Working Progress Log

Live status of the Milestone 4 build. Plan: [Milestone_4_Plan.md](Milestone_4_Plan.md).
Update this file as work lands so a fresh session can resume without re-deriving anything.

**Last updated:** 2026-08-04

---

## Status at a glance

| Part | Item | Status |
|---|---|---|
| A1 | `data/honey_daily_source.parquet` snapshot pulled from Quack (3.0 MB) | done — David approved committing (2026-08-04) |
| A1 | `data/honey_model.parquet` built, parity-asserted (5.0 MB) | done |
| A2 | `sql/01..06_*.sql` extracted + extended | done |
| A3 | `scripts/build_honey_model.py`, `scripts/pull_source_snapshot.py` | done |
| A4 | Compiled mode removed / `Milestone_4.ipynb` | **done** — 61 cells, executes clean end to end |
| A5 | `requirements.txt`, README recipe, `data/README.md` | done |
| B | Data-prep write-up | **done** — notebook §2, incl. row-count waterfall (Stephanie to review) |
| C1 | Plausibility gates | done as flags in `05_honey_model.sql` |
| C2 | Extremes rework at modelling grain | **done** — notebook §3, unit defect found and corrected |
| C3 | Threshold sensitivity | **done** — `results/threshold_sweep.csv`, notebook §3.6 |
| D | `src/honeymodel/evaluation.py` | **done** — D1–D6 all implemented and exercised |
| E | Models | baseline board, ablation, two-stage, classifier all run (Joshua to extend: LightGBM, tuning) |
| F | NOAA weather | not started (Joshua) — hive lat/lon in the table, 34 distinct sites |
| G/H | Interpretation, conclusions | **done** — notebook §6 and §7 |

**Deliverable state:** `jupyter nbconvert --to html --execute Milestone_4.ipynb` runs clean
with no network, no Tailscale and no DuckDB file. `Milestone_4.html` is 645 KB, 4 figures.

---

## Key findings so far (these change the milestone's content)

### 1. The data layer is fully reproducible from Quack, and the snapshot is exact

`bob_sensor_processed` on the team's Quack server holds the complete published archive:
52.1M rows, 78 hives, 2019-06-24 → 2022-12-31, with `lat`/`lon` and the beekeeper-event
distance columns. Filtering to `interval = 'd' AND dataset = 'years'` gives **29,172 rows**
— exactly Stephanie's raw anchor count — and rebuilding her SQL on top reproduces the
published modelling table to the digit:

```
26,215 rows · 78 hives · 2019-06-25 → 2022-12-30
target mean 0.002042 · sd 1.840522 · min −65.322940 · max 39.098176 · 445 extremes
```

`scripts/build_honey_model.py` asserts all nine of these on every run.

### 2. W10 dissolves — the "151 of 453 CSVs" concern was wrong

Daily files split into two publisher datasets:

| dataset | files | rows | hives | what it is |
|---|---|---|---|---|
| `years` | 78 | 29,172 | 78 | the full 2019–2022 daily archive, one file per hive-year |
| `events` | 191 | 12,538 | 55 | 3-month windows around swarming / colony-death events |

The `events` daily rows are **re-slices of the same hive-days**: only 948 hive-days
(3.2%) are not already in `years`, and they contain 527 internal duplicates. So the
pipeline already ingested 100% of the daily grain; unioning `events` in would inject
duplicate hive-days, not new coverage. Nothing downstream needs re-running for W10.

### 3. W2 is real but points the opposite way — and it is the milestone's biggest correction

From the publishers' own methods (Senger et al. 2024, *Data in Brief* 52:110015):

- `outlier_lim` is TRUE when `weight_delta` exceeds **0.3 kg per minute**;
- `weight_delta_noOutlier` = weight_delta with those flagged jumps set to 0;
- `weight_kg_noOutlier` = the cumulative sum of that cleaned delta;
- the stated purpose is to exclude "sudden drastic changes in the weight, which are
  usually induced by activities by the beekeeper."

**Milestone 3 built its target on `weight_kg` — the series where those jumps are still
present.** The publishers ship both columns; the pipeline picked the uncleaned one.

Rebuilding the identical modelling grain on `weight_kg_noOutlier` instead:

| target series | extremes (\|Δ\| > 5 kg) |
|---|---|
| `weight_kg` (Milestone 3) | **445** |
| `weight_kg_noOutlier` (publisher-cleaned) | **101** |

419 of the 445 extremes (94%) disappear under the cleaned series, and the two targets
correlate at only **r = 0.124**. So the extremes are overwhelmingly the >0.3 kg/min jumps
the publishers attribute to beekeeper handling. Milestone 3's Finding 1 — "extremes are
not explained by beekeeper actions" — is not merely an artifact of upstream removal as
the plan assumed; it is **contradicted by the publishers' own outlier flag**, which our
pipeline never consulted.

Caveat to state in the write-up: neither series is "correct". `weight_kg` retains
handling artifacts; `weight_kg_noOutlier` zeroes *every* abrupt change, including real
mass loss from a genuine harvest.

### 4. Also confirmed at daily grain

- `outlier_lim` is **never TRUE** in the daily rows (values are False or NULL). The
  98,304 flagged records in `David_Work/extremes.ipynb` are minute-grain. This confirms
  W3: that investigation and the model's 445 extremes are disjoint populations.
- Raw daily weight runs from **−36.08 kg** to 111.73 kg; 1,022 daily rows are below 1 kg.
  The Milestone 3 `> 0` guard admits these. W5 confirmed, and worse than the plan stated.
- 34 distinct lat/lon sites for 78 hives, spanning 47.7–53.8 N, 6.7–13.4 E — exactly the
  Part F bounding box. Weather work needs no geocoding.
- Daily files are publisher-computed averages of the minute series, so the `avg_*` names
  in `honey_daily_summary` are accurate (correcting W12's premise) — but the aggregation
  happens upstream, not in our SQL, and the write-up must say so.
- **New (W18):** `end_of_day_weight_kg` is a misnomer. The daily value is the *mean* of
  the day's minute readings, not the last one. The target is therefore a change in daily
  mean weight. Rename or document in the schema.

### 5. A1 blocker — the Zenodo record declares no license

Record 10.5281/zenodo.10407693 is open access, but the DataCite `rightsList` is empty and
no license is shown in the record metadata. Redistributing a derived table is defensible;
committing `honey_daily_source.parquet` is closer to redistributing publisher data.
**Awaiting David's decision before either file is committed.** Both are staged in
`data/` and gitignored until then.

---

## Quality-flag counts (current build)

Of 26,215 modelling rows: 597 implausible-weight (any of prev/current/next below the
5 kg floor), 968 hive-local robust outliers (>5 MAD from the hive's own centred 7-day
median, scaled by that hive's day-to-day change MAD), 26 dropout/re-tare signatures.
Of the 445 extremes, 19 are implausible-weight and 26 carry a dropout signature.

Nothing is dropped — all are flags, per the project's retain-and-flag convention.

---

## Files created so far

```
sql/01_raw_anchor.sql          sql/02_clean.sql        sql/03_daily_summary.sql
sql/04_feature_candidates.sql  sql/05_honey_model.sql  sql/06_schema.sql (generated)
scripts/build_honey_model.py   scripts/pull_source_snapshot.py
src/honeymodel/__init__.py     src/honeymodel/data.py  src/honeymodel/features.py
data/honey_daily_source.parquet  data/honey_model.parquet   (staged, uncommitted)
```

## Headline results (all in `results/*.csv`, rendered by the notebook)

Four rolling-origin folds, 6-month horizons, expanding window. Best naive = predict-zero,
MAE **0.5128**.

| Framing | MAE | R² | Skill vs naive |
|---|---|---|---|
| Routine only, oracle-gated (Milestone 3's headline) | 0.315 | 0.154 | +38.0% — **upper bound only** |
| Two-stage, unfiltered test (hard gate) | **0.479** | 0.044 | **+5.9% — the number to report** |
| Two-stage, unfiltered test (blended) | 0.516 | 0.078 | −1.2% |
| Single regressor, all data | 0.567 | 0.031 | −11.1% |

- **No single-stage model beats predict-zero on MAE.** Best is history/HistGB at 0.542
  (−6% skill), though it wins on RMSE (1.698 vs 1.743) and R² (+0.036).
- **Milestone 3's own configuration scores MAE 0.78–0.98, R² −1.13 to −1.16** under proper
  CV. The published 0.216 came from the leaky split plus oracle gating.
- **Classifier is the strongest result:** PR-AUC 0.17–0.31 vs no-skill 0.010–0.018
  (10–17× lift), ROC-AUC 0.80–0.85, consistent across all four folds.
- **Season split is stark:** skill +51% autumn, +34% winter, **−95% summer**. Good where
  it is easy, bad where it is valuable.
- **Generalises to unseen hives:** leave-hives-out MAE 0.42–0.66, R² 0.05–0.13 — matches
  temporal folds, so no hive-identity memorisation.
- **Seed stability:** HistGB 0.434–0.457 vs RF 0.503–0.507, non-overlapping. The
  boosting-beats-forest claim survives.
- **Permutation importance:** `previous_day_weight_change_kg` dominates at ~3× the next
  feature. Yesterday's *change*, not absolute weight level.

## Extremes rework (C2/C3)

- Event-distance columns **mix days and seconds** across hives (median day-over-day
  increment is exactly 1.0 or exactly 86400.0). Fixed by
  `honeymodel.data.normalise_event_distance_units`. Milestone 3's "0% within 1 day" was
  largely this bug; corrected figures are 7.8% (harvest) and 13.2% (feeding) — against
  8.7% and 8.7% on routine days, so still no strong association either way.
- **Seasonality is the real signal:** July extreme rate 5.11% (89 losses vs 36 gains),
  May 3.49%, January 0.06%. Late-July loss clustering across hives and years is a harvest
  signature. Milestone 3's "localized, not apiary-level" conclusion does not hold.
- The M3 date-clustering correlation is **1.0 by construction** at daily grain — a hive
  contributes at most one extreme per date, so the two columns are identical. Use
  `share_of_reporting_hives` instead (max 28% on 2021-07-23).
- Data-quality flags explain **31%** of the 445 extremes.
- Threshold sweep: 3 kg → PR-AUC 0.315; 5 kg → 0.228; 10 kg → 0.144; **hive-relative
  5 MAD → 0.523** at 10.96% positives. The relative definition is the better operational
  choice; ±5 kg retained for comparability.

## Open items / handoff

1. **Email the corresponding author** about the Zenodo licence (record declares none).
   David approved committing meanwhile; `data/README.md` documents the fallback.
2. **Tailscale MagicDNS hostname appears in 4 tracked notebooks** —
   `Milestone_3.ipynb`, `Joshua_Work/Joshua_Amaya_Pipeline.ipynb`,
   `Stephanie_Work/Archive/Honey_EDA_Scratch.ipynb`,
   `Stephanie_Work/Project_Log_Stephanie/Project_Log_Steph.ipynb`. The **token is not**
   in any tracked file (verified). Scrubbing the working files does not remove it from
   git history, so a rewrite is a team decision. None of the Milestone 4 files contain it.
3. **Part F weather join** (lat/lon and 34 site coordinates are in the table;
   `WEATHER_FEATURES` is already defined so the ablation ladder picks it up automatically),
   LightGBM rung, PR curve with a chosen operating point.
4. **Review notebook §2 against the pipeline notebook**; `Modeling_schema.md`
   needs the new flag columns and the `end_of_day_weight_kg` naming correction.
5. **Consider modelling the minute/hourly grain** — daily averaging is the most
   likely reason the target looks noise-dominated.
