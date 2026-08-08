# Milestone 4 Development Plan — Honey Yield Predictive Model

**Team:** Stephanie Nord, David Jorgensen, Joshua Amaya
**Date:** 2026-08-04
**Predecessor:** [Milestone_3.ipynb](../Milestone_3.ipynb) · [David_Work/Milestone_3_Plan.md](Milestone_3_Plan.md)

---

## Context

Milestone 3 established a working DuckDB pipeline, an EDA pass, an extreme-event
investigation, and three baseline models. Its central finding — that hive-day weight
change splits into a learnable routine core (98.3%) and an unexplained extreme tail
(1.7%) — was accepted by the grader as a strong modeling decision.

Milestone 4 must now deliver four things the rubric asks for (documented data-prep
process, at least one built and evaluated model, interpretation of results, and the
beginning of a conclusion/recommendation), while closing five pieces of grader feedback.

Two of those feedback items are about reproducibility and are currently **hard blockers**:
the modeling table `honey_model` exists only in Stephanie's local, off-repo
`Data/honey.duckdb`, built from a `Data/Daily_Only/` CSV tree that is also off-repo, and
the fallback data source (the Quack server) needs Tailscale membership. Nobody outside the
team can run [Milestone_3.ipynb](../Milestone_3.ipynb) in live mode. Milestone 3 papered over
this with a "compiled mode" that renders hard-coded `PUBLISHED` constants and archived
PNGs. That is exactly what the grader flagged, so Milestone 4 retires it.

Beyond the stated feedback, this plan documents **eleven additional weak points** found
while auditing the Milestone 3 notebooks. Three of them are severe enough that leaving
them unaddressed would make the final model — and the conclusions drawn from it —
wrong rather than merely weak. They are listed in the next section because they should
drive the schedule, not be appended to it.

**Decisions already made** (from planning discussion): commit a data snapshot rather than
rely on a rebuild script alone; NOAA weather integration is in scope for this milestone;
and the routine model gets a real two-stage pipeline with honest end-to-end metrics.

---

## Part 0 — Weak points found in the audit (read this first)

These are *in addition to* the grader's five items. Severity is about impact on model
correctness and on the validity of the write-up, not effort.

### Critical — these make current conclusions wrong

**W1. Routine-model metrics are oracle-gated and not achievable in deployment.**
The routine models are trained *and tested* on rows filtered by
`extreme_weight_change_flag == False`. That flag is derived from
`ABS(target_next_day_weight_change_kg) > 5` — i.e. from the label itself. Reporting
"routine GB, R² = 0.216" silently assumes something tells you tomorrow isn't an extreme
day *before* you predict. In production nothing does. The headline number is optimistically
biased and the milestone never says so.
*Fix:* Part E builds the two-stage classifier→regressor pipeline and reports end-to-end
metrics on the **unfiltered** chronological test set, alongside the oracle-gated numbers
clearly labelled as an upper bound.

**W2. "Extremes are not explained by beekeeper actions" is an artifact of upstream
preprocessing, not a finding.**
The source dataset (Senger, Gruber, Kluss & Johannsen, *Data in Brief* 52 (2024) 110015;
Zenodo) states in its own methods that preprocessing removed threshold-based outliers and
**"excluded changes in weight that were induced by beekeeping activities."** The
beekeeper-induced weight changes were stripped by the data publishers before we ever saw
the data. So Milestone 3's Finding 1 ("0% of extreme events occur within 1 day of a logged
observation") is close to guaranteed by construction. It is currently written up as a
discovery about hive behavior. It is not.
*Fix:* Read the Data in Brief methods section, document exactly what the publishers
removed and how (Part B), and rewrite the Section 4.3 interpretation. The corrected
reading actually *strengthens* the modeling decision: whatever survives as a ±5 kg jump is
most likely a sensor fault, re-tare, or unlogged handling — which is a much better
argument for anomaly detection than the current one.

**W3. The extreme-event investigation was run on a different table, grain, and
definition than the extremes in the model.**
[David_Work/extremes.ipynb](extremes.ipynb) queries the remote
`bob_sensor_processed` table where `outlier_lim = TRUE` — 98,304 raw sensor records, with
per-category counts like 21,865 (honey) and 10,631 (feeding). The modeling extremes are
**445 daily rows** in `honey_model` where `|Δ| > 5 kg`. Two different populations, two
different outlier definitions, two different grains. The date-clustering correlation
(0.213) and the beekeeper-proximity percentages therefore do not transfer to the 445 events
the model actually has to handle.
*Fix:* Re-run both analyses directly against the 445 `honey_model` extremes (Part C).
Keep the raw-grain analysis as a separate, clearly-labelled appendix.

### Major — these distort the numbers

**W4. No naive baselines, so R² ≈ 0.2 is uninterpretable.**
The target has mean ≈ 0.002 kg. Nobody has checked whether the models beat *predicting
zero*. Standard time-series baselines are missing entirely: persistence (repeat yesterday's
change), per-hive mean, per-hive × month climatology, seasonal naive.
*Fix:* Part D adds a baseline board and reports a **skill score** (% error reduction vs.
the best naive predictor) next to every model. If GB's advantage over predict-zero is
small, that is itself a legitimate and reportable finding — but we need to know.

**W5. Physically impossible weights survive into the modeling table.**
`honey_model` reports `end_of_day_weight_kg` min = **0.0005 kg** (half a gram) against a
mean of 34.3 kg. The pipeline's only guard is `end_of_day_weight_kg > 0`
([Stephanie_Work/Honey_Project_Pipeline.ipynb](../Stephanie_Work/Honey_Project_Pipeline.ipynb)
cell 39). A hive dropping 40 kg → 0.0005 kg produces a fabricated −40 kg "extreme event."
This is a prime suspect for the −65.3 kg minimum and for a meaningful share of the 445
extremes.
*Fix:* Part C adds physical-plausibility gates and quantifies exactly how many extremes
they explain. Expect this to change the extreme count and the narrative.

**W6. The chronological split leaks across its own boundary and covers only one season.**
`frame.sort_values("measurement_date").iloc[:k]` splits by **row index**, so 2022-04-25
rows land on both sides — `train_end` and `test_start` are the same date. Separately, the
single test window (2022-04-25 → 2022-12-30) is one spring-through-winter stretch of one
year, so every reported metric is a one-season, one-year estimate.
*Fix:* Split on a date cutoff, not a row index; add rolling-origin CV (Part D — this is
also the grader's request).

**W7. R² is unstable on this target and shouldn't carry the headline alone.**
R² is normalized by test-window variance, and this target's variance is strongly seasonal
(summer nectar flow vs. winter dormancy). A winter-heavy fold will show a poor R² even
with excellent MAE, and vice versa — so R² will swing across folds for reasons that have
nothing to do with model quality.
*Fix:* Lead with MAE and skill-vs-naive; report R² per fold and per season rather than as
a single pooled number.

### Moderate — these cost accuracy or credibility

**W8. `.ffill()` bleeds values across hive boundaries.** Both
[Milestone_3.ipynb](../Milestone_3.ipynb) (feature-importance cell) and the pipeline notebook
call `model_df[features].ffill()` on a frame ordered by `hive_id, measurement_date`. The
last sensor reading of hive *A* is carried into the first rows of hive *B*. With 23–35%
sensor missingness this is not a rare edge case.
*Fix:* `groupby("hive_id").ffill()` plus explicit `*_was_missing` indicator columns, or
switch to `HistGradientBoostingRegressor` / LightGBM, which consume NaN natively and avoid
both the bleed and the row loss.

**W9. Leakage landmines sit in the table.** `honey_model` ships `next_day_weight_kg` and
`next_observation_date`. Any `X = df.drop(columns=[TARGET])` yields a perfect-score model.
Nobody has done it yet; the columns are still loaded.
*Fix:* Define a module-level `FORBIDDEN_COLUMNS` set and assert against it inside the
feature-builder so it cannot happen silently.

**W10. Only 151 of 453 daily CSVs have ever been processed.** The pipeline notebook says
so explicitly ("developed and validated using a representative subset... designed to scale
to the complete archive"). The full archive has never been run, so hive count, coverage,
and the 1.7% extreme rate are all subset estimates. The team's own Milestone 3 plan listed
this as a risk; it is still open.
*Fix:* Run the full archive in Part B and diff the distributions against the subset.

**W11. No per-hive generalization test.** All 78 hives appear in both train and test, and
the features are absolute weights, so the models can lean on hive identity. "Will this work
on a hive we've never instrumented?" is unanswered — and it is the question a beekeeper
would actually ask.
*Fix:* Add a grouped leave-hives-out evaluation alongside the temporal one (Part D).

**W12. `honey_daily_summary` doesn't aggregate anything.** Cell 27 of the pipeline is a
plain `SELECT` with no `GROUP BY`, yet renames columns to `avg_internal_temp_1` etc. Row
counts confirm no aggregation happens (raw 29,172 = clean 29,172 = summary 29,172) — the
source daily files are evidently already daily means. Harmless as code, but the data-prep
write-up would misdescribe the process, and the naming implies work that isn't done here.
*Fix:* Verify against the source data dictionary and document accurately in Part B.

**W13. The ±5 kg extreme threshold is arbitrary and never tested.** It is also absolute:
5 kg on a 10 kg nucleus colony is a very different event from 5 kg on a 100 kg production
hive, and hive weights here span 0 → 111.7 kg.
*Fix:* Sensitivity sweep at 3/5/7/10 kg plus one hive-relative variant (robust MAD
multiple), reporting how the split, class balance, and metrics move (Part C).

**W14. The GB-beats-RF claim is probably inside the noise.** ΔR² = 0.018 from a single
split with a single seed. Cross-validation (Part D) plus 3–5 seeds is needed before the
claim survives.

**W15. Impurity-based feature importance is unreliable here.** `previous_day_weight_kg`,
`rolling_3_day_weight_kg`, and `rolling_7_day_weight_kg` are near-collinear, and MDI is
biased toward high-cardinality continuous splits. "Recent weight history dominates" may be
partly artifact.
*Fix:* Report permutation importance on the test set (and optionally SHAP) instead.

**W16. Level features are being used to predict a delta target.** The features are
absolute weights (mean 34.3 kg, sd 14.7) predicting a ~0-mean, 1.84-sd delta. Trees will
spend their capacity splitting on hive mass. Differenced and hive-normalized features
(weight minus its own rolling mean, rolling slope, weight relative to the hive's seasonal
baseline) are a cheap and likely material accuracy gain. Folded into Part C.

**W17. Source citations are wrong in the report.** Section 1 of
[Milestone_3.ipynb](../Milestone_3.ipynb) labels the primary dataset "HOBOS" and attributes
the beekeeper observation logs to "USDA Tucson." HOBOS is a *separate* Kaggle dataset
listed in [data-guide.md](../data-guide.md), and the beekeeper logs come from the German
project's own web app. Cite the Data in Brief paper and Zenodo DOI properly.

**Minor housekeeping.** `.env` is correctly gitignored and untracked (verified — no
credentials are in git). However, the Tailscale MagicDNS hostname is embedded in a cell
output inside the committed `Milestone_3.ipynb`; scrub notebook outputs before export.
There is no `requirements.txt` / `environment.yml` anywhere in the repo.
`Milestone_3.html` and `Milestone_3.pdf` are untracked and should be committed with the
milestone.

---

## Part A — Reproducibility (grader feedback #1 and #2)

Target repository layout:

```
honey-yield-predictive-model/
  Milestone_4.ipynb              <- the deliverable; ALWAYS runs live
  data/
    honey_model.parquet          <- committed snapshot (~26k rows x 27 cols, ~1-2 MB)
    honey_weather.parquet        <- committed NOAA join result (Part F)
    README.md                    <- provenance, DOI, license, regeneration command
  sql/
    01_raw_anchor.sql            <- extracted from the pipeline notebook
    02_clean.sql
    03_daily_summary.sql
    04_feature_candidates.sql
    05_honey_model.sql
    06_schema.sql                <- CREATE TABLE DDL for honey_model
  scripts/
    build_honey_model.py         <- Zenodo download -> DuckDB -> parquet, end to end
    fetch_noaa_weather.py        <- GHCN-Daily pull for the German bounding box
  src/honeymodel/
    data.py  features.py  evaluation.py  models.py
  requirements.txt
  README.md                      <- 3-command reproduction recipe
```

**A1. Commit the modeling snapshot.** 26,215 × 27 is tiny; ZSTD Parquet lands around
1–2 MB. This single step resolves the feedback outright — the notebook loads
`data/honey_model.parquet` with no DuckDB file, no Tailscale, no local paths. Confirm the
Zenodo license permits redistribution of a derived table (the dataset is a public research
release, so this is expected to be fine, but record the license in `data/README.md` and
cite the DOI). If redistribution turns out to be restricted, fall back to committing the
build script plus a synthetic-schema smoke-test fixture and say so explicitly.

**A2. Extract the pipeline DDL into `sql/`.** Copy the SQL verbatim out of
[Stephanie_Work/Honey_Project_Pipeline.ipynb](../Stephanie_Work/Honey_Project_Pipeline.ipynb)
cells 17, 21, 27, 35, and 39. This is the "required local database/table structure" the
grader asked for. `06_schema.sql` should carry the explicit `CREATE TABLE honey_model`
DDL with column types and comments, generated from `DESCRIBE honey_model`.

**A3. Write `scripts/build_honey_model.py`.** Downloads the Zenodo archive, extracts the
daily CSVs, runs `sql/01..05` in order against a fresh local DuckDB, and writes
`data/honey_model.parquet`. Fully parameterized by a `--data-dir` flag — **no
`Path.cwd().parent` path guessing** (the current pipeline's `project_root = working_dir.parent`
breaks the moment anyone runs it from a different directory).

**A4. Delete compiled mode.** Remove `PUBLISHED`, `LIVE`, `show_archived()`,
`runtime_only()`, and the `load_model_df()` probe chain from the Milestone 4 notebook. Every
figure and number regenerates from the committed Parquet. Keep
[Milestone_3_Figures/](../Milestone_3_Figures/) in the repo as an archive only.

**A5. Pin the environment.** `requirements.txt` with pinned `duckdb`, `pandas`, `numpy`,
`scikit-learn`, `matplotlib`, `lightgbm`, `pyarrow`. Note the Python version in the README.
`root README.md` gets a literal three-command recipe:
`pip install -r requirements.txt` → `jupyter nbconvert --execute Milestone_4.ipynb` → done.

**A6. Keep Quack strictly optional.** The remote path stays available for the team's raw
`bob_sensor_processed` work but is never on the critical path for the deliverable. Guard it
behind an explicit `USE_REMOTE = False` flag.

---

## Part B — Data preparation write-up (rubric item 1)

New notebook section: **"2. Data Preparation Process."** This is a graded narrative
deliverable, not just code — it should read as a description of decisions and their
rationale, with each claim backed by a number the notebook computes.

Cover, in order:

1. **Source and provenance.** Senger et al., *Data in Brief* 52 (2024) 110015, Zenodo;
   78 German colonies, 2019–2022, citizen-science collection with a beekeeper web app.
   Correct the HOBOS / USDA Tucson misattribution (**W17**).
2. **What the publishers already did to the data.** Threshold-based outlier removal and
   removal of beekeeping-induced weight changes (**W2**). This belongs early and
   prominently — it constrains every downstream interpretation.
3. **Ingestion.** 453 daily CSVs (full archive, up from 151 — **W10**) via
   `read_csv_auto(union_by_name=TRUE, all_varchar=TRUE)` into a text-preserving raw anchor
   that is never overwritten.
4. **Typing and null handling.** `TRY_CAST` + `NULLIF(col, 'NA')`; report cast-failure
   counts per column rather than assuming zero.
5. **Grain.** One row per hive per day; `hive_id` parsed from the source filename.
   Document that the source files are already daily aggregates and that
   `honey_daily_summary` renames rather than aggregates (**W12**). Add an explicit
   duplicate `(hive_id, measurement_date)` assertion *before* the window functions run.
6. **Feature engineering.** `LAG`/`LEAD` for previous/next observations, 3- and 7-day
   rolling means, calendar features — all `PARTITION BY hive_id`.
7. **Filtering, with a row-count waterfall.** A table showing rows dropped at each gate:
   non-consecutive days, null weights, non-positive weights, and the new plausibility
   gates from Part C. Milestone 3 asserted "29,172 → 26,215" without itemizing where the
   2,957 rows went. Itemize them.
8. **Missingness.** The 23–35% sensor gaps, plus the chosen handling strategy and why
   (**W8**).
9. **Leakage controls.** State the `FORBIDDEN_COLUMNS` rule and why `next_day_weight_kg`
   must never enter a feature matrix (**W9**).

---

## Part C — Data-quality remediation and extreme-event rework

**C1. Physical plausibility gates (W5).** Add to `05_honey_model.sql`: an absolute weight
floor (proposed 5 kg — justify against the observed distribution), and a per-hive robust
screen flagging readings more than *k* MAD from that hive's local rolling median. Produce
a table: how many of the 445 extremes are explained by implausible weights, sensor
dropouts, or re-tare signatures. Flag rather than silently drop, consistent with the
project's existing "retain and flag" convention.

**C2. Re-run the extremes investigation at the modeling grain (W3).** Redo the
beekeeper-proximity and date-clustering analyses on the 445 `honey_model` extremes.
Reuse the methodology in [David_Work/extremes.ipynb](extremes.ipynb) — the
`*_last_dif` / `*_next_dif` minimum-distance logic is sound — but point it at the modeling
table. Report the corrected numbers and, per **W2**, interpret them against what the
publishers already removed.

**C3. Threshold sensitivity (W13).** Sweep 3/5/7/10 kg plus a hive-relative MAD variant.
One table: threshold → routine/extreme counts → routine model MAE → classifier AUC/PR-AUC.
Justify the final choice with evidence instead of convention.

**C4. Feature upgrades (W16).** Add to `src/honeymodel/features.py`:
- Differenced/normalized weight features: `weight − rolling_7`, rolling slope,
  weight relative to the hive's seasonal baseline.
- Cyclical day encoding `sin_day` / `cos_day` — reuse the pattern from
  [Joshua_Work/Joshua_Model_Experiments.ipynb](../Joshua_Work/Joshua_Model_Experiments.ipynb)
  (methodology only; his metrics are on simulated data and must stay out of every results
  table).
- Group-aware sensor imputation with `*_was_missing` indicators (**W8**).
- Hive-level context features, motivated by the hive-local extremes finding.

---

## Part D — Validation framework (grader feedback #4, plus W4/W6/W7/W11/W14)

New module `src/honeymodel/evaluation.py`. This is the technical heart of the milestone.

**D1. `date_chronological_split(df, cutoff_date)`** — splits on a date, not a row index,
so no observation date straddles the boundary (**W6**).

**D2. `rolling_origin_cv(df, n_splits, horizon)`** — expanding-window folds
(e.g. train ≤2020-12 → test 2021-H1; train ≤2021-06 → test 2021-H2; …), which is the
grader's explicit request. Report mean ± sd across folds, never a single number.

**D3. `naive_baselines(df)`** (**W4**) — predict-zero, global mean, persistence
(yesterday's Δ), per-hive mean, per-hive × month climatology. Compute a **skill score**
(% MAE reduction vs. the best naive) for every model.

**D4. `grouped_hive_cv(df, n_folds)`** (**W11**) — `GroupKFold` on `hive_id` to answer
"does this generalize to an uninstrumented hive?"

**D5. `segmented_report(y_true, y_pred, meta)`** (grader feedback #5) — one function
returning MAE / RMSE / R² / n **by hive, by season, by month, and by weight-change decile**,
in a tidy DataFrame ready to plot. This is the reusable evaluator Milestone 3 promised and
never built.

**D6. Seed stability** (**W14**) — every headline model runs at 3–5 seeds; report the
spread. No algorithm ranking claim without it.

---

## Part E — Models (rubric item 2; grader feedback #3)

Everything trains on the fixed rolling-origin protocol from Part D, and every result
carries a skill score.

**E1. Baseline board.** All five naive predictors from D3, evaluated first. This is the bar
every learned model must clear.

**E2. Routine regressor (Track A).** Random Forest and Gradient Boosting carried forward,
plus **LightGBM** (native NaN handling, so the 23–35% sensor-missing rows stay in). Feature
sets run as an ablation ladder so feature gains are separable from algorithm gains:
history-only → +sensors → +weather → +hive-context.

**E3. Extreme-event classifier (Track B).** Binary `|Δ| > 5 kg` classification — the
formalization of the routine/extreme separation the grader praised. Class balance is
~1.7%, so: PR-AUC as the primary metric (not accuracy, not ROC-AUC), class weighting or
threshold tuning, and a precision-recall curve with an explicit operating point chosen for
a stated beekeeper use case (early warning favors recall).

**E4. Two-stage pipeline (W1) — the milestone's key modeling result.** Classifier gates
the regressor; report **end-to-end MAE/RMSE/R² on the full unfiltered test set**. Present
three rows side by side:

| Framing | What it measures | Honest label |
|---|---|---|
| Single regressor, all data | Milestone 3 baseline | Realistic but poor (R² = −0.30) |
| Routine-only, oracle-gated | Milestone 3 routine result | **Upper bound only** — assumes the label is known in advance |
| Two-stage, unfiltered test | Deployable performance | The number to report |

**E5. Permutation importance (W15)** on the test set for the winning model, replacing MDI.

---

## Part F — NOAA weather integration

Templates are already verified against the live bucket in
[the-beehive/weather_duckdb_queries.sql](../the-beehive/weather_duckdb_queries.sql) (GHCN-Daily
Parquet on `s3://noaa-ghcn-pds`, anonymous reads, no credentials) — adapt query #2's
bounding-box pattern.

1. Station search over the German box (lat 47.7–53.8, lon 6.7–13.4) from
   `ghcnd-stations.txt`.
2. Pull TMAX / TMIN / PRCP for 2019–2022; **values are tenths of a unit — divide by 10**.
3. Assign each hive its nearest station (haversine); record and report the
   hive→station distance distribution, since a 60 km nearest station is a real caveat.
4. Join on (station, date); engineer same-day, lagged, and rolling weather features
   mirroring the existing weight features.
5. Materialize to `data/honey_weather.parquet` and **commit it** — the S3 pull is slow, and
   Part A requires the notebook to run without network access.
6. Enter the ablation ladder (E2) as its own rung so the weather contribution is measured,
   not assumed.

*Fallback:* if GHCN station density over the box is inadequate, pull Open-Meteo via its API
and materialize to Parquet. Do **not** attempt to read `s3://openmeteo` with DuckDB — as
documented in the SQL file, it stores a custom binary `.om` format DuckDB cannot read.

---

## Part G — Interpretation (rubric item 3)

Notebook section **"6. Results and Interpretation."** Required content:

- **Corrections to Milestone 3**, stated plainly rather than buried: the oracle-gating
  issue (**W1**), the upstream-preprocessing artifact (**W2**), and the grain mismatch
  (**W3**). A milestone that catches and corrects its own predecessor reads as rigor.
- **Skill vs. naive baselines** — the honest answer to "is this model actually learning
  anything?"
- **Segmented performance** (grader feedback #5): where the model works and where it
  fails, by hive, by season, and by weight-change range. Expected shape, to be confirmed:
  best in winter dormancy (low variance, easy), worst during peak nectar flow (high
  variance, high value) — which is the finding that matters, because the hard regime is the
  useful one.
- **Fold-to-fold variance** from rolling-origin CV, and what it says about stability.
- **Per-hive vs. temporal generalization** (**W11**) — the gap between them is the answer
  to "can we deploy this on a new hive?"
- **Feature importance done properly** (**W15**), with the collinearity caveat stated.
- **Extreme-event track**: what the classifier can and cannot detect, and the honest
  ceiling given that the publishers already removed the management-induced events.

---

## Part H — Conclusions and recommendations (rubric item 4)

Notebook section **"7. Conclusions and Recommendations (Preliminary)."** Framed as
*beginning to formulate*, per the rubric — commit to what the evidence supports and flag
what is still open.

Structure:
1. **What we can now claim**, with the metric and its uncertainty attached.
2. **Practical recommendations for beekeepers** — this is the project's actual point.
   Where does a next-day forecast change a decision (harvest timing, feeding, swarm
   early-warning)? What does the two-stage model's precision/recall mean for someone
   deciding whether to drive out to the apiary?
3. **What the data cannot support** — extremes as a prediction target, given **W2**; and
   the limits of a German-only, 78-hive, citizen-science sample.
4. **Open questions for Milestone 5** — hyperparameter tuning, multi-day horizons,
   per-hive model personalization, external validation on a second dataset.
5. **Honest limitations section**, including everything from Part 0 that remains unfixed.

---

## Files to create / modify

| Path | Action |
|---|---|
| `Milestone_4.ipynb` | **New** — the deliverable; no compiled mode |
| `sql/01..06_*.sql` | **New** — DDL extracted from the pipeline notebook |
| `scripts/build_honey_model.py` | **New** — Zenodo → DuckDB → Parquet |
| `scripts/fetch_noaa_weather.py` | **New** — adapts `the-beehive/weather_duckdb_queries.sql` |
| `src/honeymodel/{data,features,evaluation,models}.py` | **New** — importable, testable logic |
| `data/honey_model.parquet`, `data/honey_weather.parquet`, `data/README.md` | **New** — committed snapshots + provenance |
| `requirements.txt` | **New** — pinned |
| `README.md` | **Modify** — 3-command reproduction recipe; Tailscale demoted to optional |
| [Stephanie_Work/Modeling_schema.md](../Stephanie_Work/Modeling_schema.md) | **Modify** — plausibility gates, new features, corrected `honey_daily_summary` description |
| [data-guide.md](../data-guide.md) | **Modify** — correct source attribution (**W17**) |
| `Milestone_3.html`, `Milestone_3.pdf` | **Commit** — currently untracked |

---

## Team assignments

Roughly following the Milestone 3 split, weighted toward each person's existing context.

**Stephanie** — pipeline owner. Parts A2/A3 (SQL extraction, build script), B (data-prep
write-up), C1 (plausibility gates), C4 (feature module), W10 (full 453-CSV run), W12
(grain verification).

**David** — validation and extremes. Parts A1/A4/A5 (snapshot, compiled-mode removal,
environment), D (entire evaluation module — the largest single piece), C2/C3 (extremes
rework, threshold sensitivity), G (interpretation write-up).

**Joshua** — models and weather. Parts F (NOAA integration), E (baseline board, LightGBM,
classifier, two-stage pipeline), E5 (permutation importance). Reuses his algorithm-comparison
harness and `sin_day`/`cos_day` encoding — **re-run against the real `honey_model`**; his
existing simulated-data metrics must not appear in any results table.

**Shared** — Part H, and a final end-to-end reproduction test on a clean checkout by
someone who did not write the pipeline.

---

## Timeline

| Window | Work |
|---|---|
| Week 1 | **Reproducibility first.** Part A end to end + full-archive run (W10) + plausibility gates (C1). Nothing downstream is trustworthy until the data layer is fixed and complete. |
| Week 2 | Part D evaluation module + baselines (D3). Part C2/C3 extremes rework. Part F weather pull started in parallel (it is I/O-bound and can run unattended). |
| Week 3 | Part E models: ablation ladder, classifier, two-stage pipeline. First segmented reports. |
| Week 4 | Parts B, G, H write-ups. Clean-checkout reproduction test. Export HTML/PDF with scrubbed outputs. Submit. |

Sequencing note: Part D must land before Part E. Building models against a leaky split and
no baselines would mean re-running everything.

---

## Verification

The milestone is done when all of the following pass:

1. **Clean-checkout reproduction.** `git clone` → `pip install -r requirements.txt` →
   `jupyter nbconvert --to html --execute Milestone_4.ipynb` completes with **no Tailscale,
   no network, and no pre-existing local DuckDB file**. This is the direct test of grader
   feedback #1 and #2.
2. **Rebuild parity.** `python scripts/build_honey_model.py --data-dir <zenodo>` reproduces
   `data/honey_model.parquet` — assert identical row count, column set, and target
   mean/std against the committed snapshot.
3. **No leakage.** An assertion in `features.py` fails loudly if any `FORBIDDEN_COLUMNS`
   member reaches a feature matrix. Add a deliberate negative test.
4. **Split integrity.** Assert `max(train.measurement_date) < min(test.measurement_date)`
   for every fold, strictly.
5. **Baselines beaten.** Every reported model has a computed skill score vs. the best naive
   predictor. If a model doesn't beat naive, that gets reported, not hidden.
6. **Two-stage honesty.** End-to-end metrics on the unfiltered test set are present, and
   every oracle-gated number is labelled as an upper bound.
7. **Segmented tables render** for hive, season, and weight-change range (feedback #5).
8. **CV present** — rolling-origin results with mean ± sd across folds (feedback #4).
9. **Secrets clean.** `git grep` for the Quack token, Tailscale auth key, and MagicDNS
   hostname returns nothing across all tracked files including notebook outputs.
10. **No absolute paths.** `grep -rE "C:/|/home/|Path.cwd\(\).parent"` finds nothing in
    shipped code.

---

## Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Zenodo license restricts redistributing a derived table | Part A1 blocked | Check license Week 1, day 1. Fallback: build script + synthetic fixture, documented explicitly. |
| Plausibility gates (C1) materially change the extreme count | Milestone 3 headline numbers shift | Expected and fine — report as a corrected finding with the before/after diff. Budget rewrite time in Week 3. |
| Models fail to beat naive baselines | Undercuts the project's premise | Also a legitimate, publishable finding. Report honestly; it redirects effort toward richer features rather than more algorithms. |
| Sparse GHCN coverage over the German box | Weak weather features | Assess station density before building features; Open-Meteo → Parquet fallback (Part F). |
| Full 453-CSV archive shifts distributions vs. the 151-file subset | Prior results don't hold | Run it in Week 1 and diff distributions before anything else is built on top. |
| Part D slips and blocks Part E | Schedule compression | D3 (baselines) and D5 (segmented report) are the minimum viable subset; D4 (grouped CV) can slip to Milestone 5 if needed. |
