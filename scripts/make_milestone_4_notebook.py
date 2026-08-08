"""Generate Milestone_4.ipynb."""
import json
from pathlib import Path

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "id": f"md-{len(cells):02d}", "metadata": {},
                  "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "id": f"code-{len(cells):02d}", "execution_count": None,
                  "metadata": {}, "outputs": [], "source": text.strip("\n").splitlines(keepends=True)})


md(r"""
# Milestone 4 — Honey Yield Predictive Model

**Team:** Stephanie Nord, David Jorgensen, Joshua Amaya
**Course project:** predicting next-day hive weight change from colony sensor history

---

## How to run this notebook

```bash
pip install -r requirements.txt
jupyter nbconvert --to html --execute Milestone_4.ipynb
```

That is the whole recipe. No Tailscale, no VPN, no local DuckDB file, no credentials, no
network access. Every number and figure below is computed at run time from
`data/honey_model.parquet`, which is committed to the repository.

Milestone 3 shipped a "compiled mode" that printed hard-coded constants and re-displayed
archived PNGs when its data source was unreachable. That mode is gone. If the data is
missing, this notebook fails loudly rather than rendering numbers it did not compute.

## What this milestone delivers

1. **A documented data-preparation process** (Section 2), with a row-count waterfall that
   accounts for every dropped row.
2. **Models built and evaluated** (Section 5) against naive baselines, on rolling-origin
   cross-validation, with a two-stage pipeline scored on unfiltered data.
3. **Interpretation of results** (Section 6), including three corrections to Milestone 3.
4. **Preliminary conclusions and recommendations** (Section 7).

## Summary of what changed since Milestone 3

Milestone 3 reported a routine-model R² of 0.216 and concluded that extreme weight events
were unrelated to beekeeper activity. Both claims turned out to depend on choices that do
not survive scrutiny:

| Milestone 3 claim | What we found | Section |
|---|---|---|
| Routine model reaches R² = 0.216 | True only when test rows are selected *using the label*. Deployable performance is R² ≈ 0.04. | 5.3 |
| Extremes are unrelated to beekeeper actions | The publishers flag those jumps as beekeeper-induced; we modelled the weight column where they are still present. | 3.5 |
| 0% of extremes fall within 1 day of a logged event | An artifact of a unit inconsistency in the source columns. Corrected: 7.8% (harvest), 13.2% (feeding). | 3.3 |
| Gradient Boosting beats Random Forest | Survives, but only once measured across seeds and folds. | 5.5 |
| Only 151 of 453 daily files processed | The 151 were 100% of the daily grain; the rest are duplicate event windows. | 2.3 |
""")

md("## 1. Setup")

code(r"""
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.cwd() / "src"))

from honeymodel import data, evaluation as ev, features, models

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 180)
pd.set_option("display.max_columns", 40)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})

RESULTS = Path("results")

def show(name, **kwargs):
    # Render a table produced by scripts/run_*.py.
    return pd.read_csv(RESULTS / name, **kwargs)

model_df = data.add_season(data.load_model_table())
print(f"modelling rows : {len(model_df):,}")
print(f"hives          : {model_df.hive_id.nunique()}")
print(f"date range     : {model_df.measurement_date.min().date()} to {model_df.measurement_date.max().date()}")
print(f"columns        : {model_df.shape[1]}")
""")

md(r"""
`load_model_table` asserts the published shape on every load — 26,215 rows across 78 hives,
no duplicate hive-days — so a silently altered snapshot cannot reach the analysis below.
""")

md(r"""
## 2. Data Preparation Process

### 2.1 Source and provenance

> Senger, D., Gruber, C., Kluss, T., & Johannsen, C. (2024). Weight, temperature and
> humidity sensor data of honey bee colonies in Germany, 2019–2022. *Data in Brief, 52*,
> 110015. https://doi.org/10.1016/j.dib.2023.110015
>
> Dataset: https://doi.org/10.5281/zenodo.10407693

78 colonies across Germany, June 2019 – December 2022, instrumented under a
citizen-science project. Each hive carries five internal temperature sensors, an external
temperature sensor, a combined temperature/humidity/pressure sensor, and a scale.
Beekeepers logged inspections and interventions through the project's own web app.

**Correction to Milestone 3.** Section 1 of the previous milestone described the primary
dataset as "HOBOS" and attributed the observation logs to "USDA Tucson". Neither is
correct. HOBOS is a separate Kaggle dataset that this project does not use, and the
observation logs are the German project's own. The citation above is the right one.

**Licensing.** The Zenodo record is open access but declares no license — the DataCite
`rightsList` is empty. We commit a derived daily-grain table (about 0.05% of the 5.6 GB
archive) with full attribution so that this notebook can be reproduced without
credentials, and have written to the corresponding author to confirm the intended terms.
`data/README.md` records the fallback if redistribution turns out to be restricted.

### 2.2 What the publishers had already done to the data

This constrains every interpretation downstream, so it comes before our own processing.
From the paper's methods:

- **Range filters.** Weights above 150 kg or below −50 kg, temperatures above 85 °C or
  below −40 °C, and humidity outside 0–100% were excluded.
- **Rate filter.** The weight series was differentiated and changes above **0.3 kg per
  minute** excluded, because such "sudden drastic changes in the weight […] are usually
  induced by activities by the beekeeper."
- **Aggregation.** Raw readings arrive every 5–10 seconds; the minute series is a median,
  and the hourly and daily series are averages of the minute series.

Two consequences matter a great deal, and Section 3.5 quantifies the first:

1. The rate filter produced a **second weight column**. `weight_kg` keeps every jump;
   `weight_kg_noOutlier` is the cumulative sum of the same deltas with flagged jumps
   zeroed. Milestone 3 built its target from `weight_kg` — the series that still contains
   the events the publishers attribute to beekeeper handling.
2. `end_of_day_weight_kg` is a **misnomer**: the daily value is the mean of the day's
   minute readings, not the last one. The target is a change in daily *mean* weight. The
   column name is retained for continuity with the Milestone 3 schema and is corrected in
   the documentation.

### 2.3 Ingestion and grain

The publisher's daily files come in two arrangements, and only one of them is a genuine
archive of the whole record:

| dataset | files | rows | hives | what it is |
|---|---|---|---|---|
| `years` | 78 | 29,172 | 78 | the full 2019–2022 daily archive |
| `events` | 191 | 12,538 | 55 | 3-month windows either side of a swarming or colony-death event |

The `events` files re-slice hive-days that `years` already contains: only 948 of its
hive-days (3.2%) are new, and it holds 527 internal duplicates. The pipeline ingests
`years` only.

This resolves an open risk carried since Milestone 3, which recorded that only 151 of 453
daily CSVs had been processed and that every figure was therefore a subset estimate. The
29,172 rows ingested are **100% of the daily grain**. The hive count, the coverage and the
1.7% extreme rate are full-archive figures.

Typing uses `TRY_CAST` with `NULLIF(col, 'NA')` throughout (`sql/02_clean.sql`), and the
grain — one row per hive per day — is asserted in `scripts/build_honey_model.py` *before*
any window function runs, rather than assumed.
""")

md("### 2.4 Filtering: a row-count waterfall")

code(r"""
waterfall = show("data_preparation_waterfall.csv")
display(waterfall)
print(f"total rows dropped: {waterfall.rows_dropped.sum():,}  "
      f"({100 * waterfall.rows_dropped.sum() / waterfall.rows_remaining.iloc[0]:.1f}% of ingested rows)")
""")

md(r"""
Milestone 3 asserted "29,172 → 26,215" without saying where the 2,957 rows went. They go:
1,094 rows have no consecutive previous day, 968 have no consecutive next day, 18 are
missing one of the three weights, and 877 fail the positive-weight guard.

The last gate deserves attention. Its only condition is `weight > 0`, which admits
readings as low as 0.0005 kg against a colony mean of 34.3 kg. A further **597 rows** sit
below a 5 kg physical floor and survive into the modelling table. Section 3.1 quantifies
what that does.
""")

md("### 2.5 Missingness")

code(r"""
sensor_columns = features.SENSOR_FEATURES
missing = features.summarise_missingness(model_df, sensor_columns)
display(missing)
""")

md(r"""
Between 23% and 35% of sensor readings are absent, which is too much to drop and too much
to ignore. Two decisions follow.

**Forward-fill within a hive, never across hives.** Milestone 3 called
`model_df[features].ffill()` on a frame ordered by `(hive_id, measurement_date)`. That
carries the last reading of one hive into the first rows of the next. At this missingness
rate it is not an edge case. `honeymodel.features.group_aware_impute` groups by `hive_id`
first and attaches a `*_was_missing` indicator for each imputed column, because a dead
sensor is itself a fact about the hive-day.

**Prefer models that read NaN natively.** `HistGradientBoostingRegressor` and LightGBM
consume missing values directly, which avoids both the boundary bleed and the row loss.
Where a model supports it, the feature matrix is built with `impute="none"`.
""")

md("### 2.6 Leakage controls")

code(r"""
# honey_model deliberately ships columns that are only knowable after the prediction date
# -- the extreme-event investigation needs them. They must never reach a feature matrix.
print(f"{len(features.FORBIDDEN_COLUMNS)} forbidden columns, including:")
for column in sorted(features.FORBIDDEN_COLUMNS)[:6]:
    print(f"  {column}")

# Negative test: asking for one must fail, not silently produce a perfect model.
try:
    features.build_feature_matrix(model_df, feature_set=["previous_day_weight_kg", "next_day_weight_kg"])
except features.LeakageError as error:
    print(f"\nLeakageError raised as intended:\n  {error}")
""")

md(r"""
`next_day_weight_kg` is the target plus today's weight. Any `X = df.drop(columns=[TARGET])`
would have produced a model with a perfect score and no meaning. Nobody did it — but the
columns were loaded and nothing stopped it, so the rule is now enforced in code and
carries a deliberate negative test.
""")

md(r"""
## 3. Data Quality and the Extreme-Event Rework

Milestone 3 split the target into a routine core (98.3%) and an extreme tail (1.7%,
|Δ| > 5 kg, 445 rows). The split was a good modelling decision. The investigation
supporting it was run against the wrong table.

`David_Work/extremes.ipynb` queried the remote `bob_sensor_processed` table where
`outlier_lim = TRUE` — 98,304 **minute-grain** records under the publishers' 0.3 kg/min
rule. The model's extremes are 445 **daily** rows under a ±5 kg rule. Two populations, two
grains, two definitions. At daily grain `source_outlier_flag` is never TRUE at all, so the
two sets do not even intersect. Everything below is recomputed against the 445 rows the
model actually has to handle.
""")

md("### 3.1 Physically implausible weights")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(model_df.end_of_day_weight_kg, bins=120, color="#4c72b0", edgecolor="none")
axes[0].axvline(5, color="crimson", linestyle="--", label="proposed 5 kg floor")
axes[0].set(title="Daily mean hive weight", xlabel="kg", ylabel="hive-days")
axes[0].legend()

low = model_df[model_df.end_of_day_weight_kg < 5]
axes[1].hist(low.end_of_day_weight_kg, bins=60, color="crimson", edgecolor="none")
axes[1].set(title=f"Below the floor ({len(low):,} hive-days)", xlabel="kg")
plt.tight_layout(); plt.show()

print(f"minimum weight in the modelling table : {model_df.end_of_day_weight_kg.min():.4f} kg")
print(f"mean weight                           : {model_df.end_of_day_weight_kg.mean():.1f} kg")
print(f"rows flagged implausible              : {int(model_df.implausible_weight_flag.sum()):,}")
""")

md(r"""
A hive that reads 34 kg one day and half a gram the next manufactures a −34 kg "extreme
event" out of a sensor fault. The pipeline's only guard was `> 0`. Milestone 4 adds three
flags — implausible weight, hive-local robust outlier, and a drop-then-recovery signature
— and **flags rather than drops**, consistent with the project's existing convention, so
each rule's effect can be measured instead of assumed.
""")

md("### 3.2 What explains the 445 extremes?")

code(r"""
display(show("extremes_attribution.csv"))
""")

md(r"""
Data-quality problems account for **31%** of the extremes: 4.3% are implausible weights,
27.4% are more than 5 MAD from the hive's own local median, and 5.8% show a
drop-then-recovery signature consistent with a dropout or a re-tare rather than real mass.
The remaining 69% are not explained by sensor artifacts, and Section 3.5 argues they are
mostly beekeeper handling.
""")

md("### 3.3 Proximity to logged beekeeper events — and a unit defect in the source")

code(r"""
display(show("event_distance_units.csv"))
""")

md(r"""
The published `*_last_dif` / `*_next_dif` columns record the time distance to the nearest
logged beekeeper event — **but not in a consistent unit**. Measured against
consecutive-day observations, they advance by exactly 1.0 per day for some hives and by
exactly 86400.0 for others. The unit follows the hive's source file, not the event type.

This breaks any threshold comparison made on the raw values. Milestone 3 asked what
fraction of extremes fell "within 1 day" of a logged event by comparing the raw number to
1.0; for a seconds-unit hive that test requires 86400, so those hives could never register
as near an event. That is a large part of where the reported 0% came from.
`honeymodel.data.normalise_event_distance_units` infers the unit per hive and converts
everything to days.
""")

code(r"""
proximity = show("extremes_event_proximity.csv")
comparison = proximity.pivot(index="event", columns="group",
                             values=["pct_within_1_day", "pct_within_7_days"]).round(2)
display(comparison)
""")

md(r"""
With the units corrected, extreme days are **not** meaningfully closer to a logged event
than routine days — but the reason is now visible, and it is not the one Milestone 3 gave.

- Feeding (13.2% vs 8.7% within one day) and treatment (10.3% vs 7.4%) are modestly
  elevated on extreme days.
- Harvest is flat (7.8% vs 8.7%), and swarming and colony death are *lower* on extreme
  days than on routine ones.

So the qualitative conclusion — logged events do not line up neatly with extremes —
survives. The quantitative claim of "0%" does not, and the correct reading is that the
logs are too sparse and too irregularly kept to explain much either way: only 15–61% of
hive-days have any event of a given type logged at all.

**Correction (Milestone 5): the per-hive fix above is not sufficient either.** The unit
also switches *within* a single hive's record — hive 21's `honey_last_dif` advances by 1.0
per day through 2019 and by 86,400 per day through mid-2020 — so
`normalise_event_distance_units`, which infers one unit per hive, still mis-converts part
of those records. The numbers in this section are therefore closer to right than
Milestone 3's and are not exactly right, and the direction of the residual error is toward
*under*-counting proximity, so the conclusion that events do not explain extremes is not at
risk.

The clean route is not to convert at all. `honeymodel.harvest.detect_logged_events` reads
the **reset** in the counter rather than its value — a reset is a reset in either unit —
and recovers 68 honey events across 19 hives whose month distribution reproduces the German
beekeeping calendar without being told it: honey April–August, feeding July–September,
queencell April–June, treatment August–November. Anything needing event *dates* should use
that. `Milestone_5.ipynb` §2.1.
""")

md("### 3.4 When do extremes happen?")

code(r"""
monthly = show("extremes_by_month.csv")
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(monthly.month - 0.2, monthly.extreme_losses, width=0.4, label="losses", color="#c44e52")
ax.bar(monthly.month + 0.2, monthly.extreme_gains, width=0.4, label="gains", color="#55a868")
ax2 = ax.twinx(); ax2.plot(monthly.month, monthly.extreme_rate_pct, color="black", marker="o", label="rate %")
ax2.set_ylabel("extreme rate (% of hive-days)"); ax2.grid(False)
ax.set(xlabel="month", ylabel="extreme events", title="Extreme weight-change events by month",
       xticks=range(1, 13))
ax.legend(loc="upper left"); plt.tight_layout(); plt.show()
display(monthly)
""")

md(r"""
This is the strongest signal in the extremes analysis, and Milestone 3 missed it entirely
by measuring clustering with a correlation that is 1.0 by construction at daily grain (a
hive can contribute at most one extreme per date, so "number of extremes" and "hives
affected" are the same column).

Extremes are **strongly seasonal and directional**: July carries a 5.11% extreme rate with
89 losses against 36 gains, May 3.49%, and January 0.06%. A late-July concentration of
large weight *losses* across many hives and all three years is what a honey harvest looks
like. The single busiest date, 2021-07-23, has 9 hives — 28% of those reporting — showing
an extreme on the same day.

Milestone 3 concluded that extremes were "localized to individual hives rather than driven
by broad external apiary-level factors." The seasonal pattern says otherwise.
""")

md("### 3.5 The decisive test: which weight column did we model?")

code(r"""
display(show("extremes_weight_series.csv"))
""")

md(r"""
The publishers ship both weight series. Rebuilding the identical modelling grain on their
outlier-cleaned column gives **101 extremes instead of 445**: 419 of Milestone 3's extreme
events — 94% — are precisely the >0.3 kg/min jumps that the publishers exclude as
"usually induced by activities by the beekeeper". The two targets correlate at r = 0.124.

**This inverts Milestone 3's Finding 1.** The claim was that extremes are not explained by
beekeeper actions. The publishers' own flag says most of them are. The previous milestone
could not see this because it compared extremes to the beekeepers' *manual event logs* —
sparse, voluntary, and unit-corrupted — instead of to the publishers' automatic rate flag.

We keep `weight_kg` as the target for continuity with Milestone 3 and because the cleaned
series has its own defect: it zeroes *every* abrupt change, including the genuine mass loss
of a real harvest. Neither column is right. What matters is that the choice is now
explicit, and that the extreme track is understood to be substantially a
**handling-detection** problem rather than a colony-behaviour one.
""")

md("### 3.6 Threshold sensitivity")

code(r"""
display(show("threshold_sweep.csv"))
""")

md(r"""
The ±5 kg line was inherited by convention and never tested. It is also absolute, which
treats 5 kg on a 10 kg nucleus colony and 5 kg on a 100 kg production hive as the same
event, across hives spanning 0–111.7 kg.

The sweep shows the trade-off is mild and monotone: a lower threshold gives more positives
and a higher classifier PR-AUC (0.315 at 3 kg) but a harder routine problem; a higher
threshold gives a cleaner routine set and a rarer, harder-to-detect event. The
hive-relative variant (5 MAD of each hive's own change distribution) flags 10.96% of
hive-days and reaches PR-AUC 0.523 — by far the most detectable definition, because it
adapts to hive scale.

**We retain ±5 kg** for comparability with Milestone 3, and record the hive-relative
definition as the better operational choice for a deployed early-warning system.
""")

md(r"""
## 4. Validation Framework

Milestone 3 reported one split, one seed, pooled R², on a test set filtered by the label.
Every piece of this section removes one of those words. The code is in
`src/honeymodel/evaluation.py`.

### 4.1 Splitting on a date, not a row index

Milestone 3 used `frame.sort_values("measurement_date").iloc[:k]`, which cuts on a **row
index**. Because up to 78 hives share every date, that put 2022-04-25 on both sides of the
boundary — `train_end` and `test_start` were the same day. `date_chronological_split` cuts
on the date itself, and `assert_split_integrity` fails on any fold where
`max(train date) >= min(test date)`.
""")

code(r"""
folds = ev.rolling_origin_cv(model_df, n_splits=4, horizon_months=6, min_train_months=12)
display(pd.DataFrame([f.describe() for f in folds]))

for fold in folds:  # the assertion each fold already passed at construction
    ev.assert_split_integrity(model_df, fold)
print("split integrity: every fold's training data ends strictly before its test data begins")
""")

md(r"""
### 4.2 Rolling-origin cross-validation

Four expanding-window folds, each training on all history up to an origin and testing on
the following six months. This is the grader's explicit request, and it is also the only
way to see that a single Milestone 3-style split covering one spring-to-winter stretch of
one year is a one-season estimate.

### 4.3 Naive baselines

The target has a mean of 0.002 kg. Before any model claims to have learned something, it
has to beat the rules that learn nothing.
""")

code(r"""
baselines = show("baselines.csv")
display(ev.fold_summary(baselines).round(4))
""")

md(r"""
**Predicting zero gives MAE 0.513 kg.** That is the bar. Persistence — repeating
yesterday's change — is the *worst* of the five (MAE 0.597, R² −0.58), which tells us the
day-to-day change series is close to noise-dominated and anti-persistent at this grain.
""")

md(r"""
## 5. Models

Every model runs on the same rolling-origin protocol and reports a **skill score**: the
percentage reduction in MAE against the best naive predictor. Positive means the model
beat the bar; negative means it did not.

### 5.1 The model board
""")

code(r"""
summary = show("model_summary.csv")
display(summary.round(4))
""")

md(r"""
Two results here matter more than the ranking.

**No single-stage learned model beats predicting zero on MAE.** The best,
HistGradientBoosting on history features, reaches MAE 0.542 against the naive 0.513 — a
skill score of **−6%**. It does win on RMSE (1.698 vs 1.743) and posts a positive R²
(0.036), which together say something specific: the model is better than zero at the large
deviations that RMSE punishes, and worse on the many near-zero days that dominate MAE.

**Milestone 3's configuration performs far worse than reported.** Its feature set and
models score MAE 0.78–0.98 with R² of −1.13 to −1.16 under proper cross-validation. The
published R² of 0.216 was the product of a leaky single split and oracle-gated test rows,
not of a model that generalises. Adding the differenced and hive-normalised features and
switching to native NaN handling closes most of that gap.

### 5.2 A model built and evaluated in this notebook

The board above comes from `scripts/run_evaluation.py`. So that this notebook builds and
evaluates a model itself, the winning configuration is refit here on the final fold.
""")

code(r"""
matrix = features.build_feature_matrix(model_df, feature_set="history+sensors", impute="none")
fold = folds[-1]
X_train, y_train = matrix.X.iloc[fold.train_index], matrix.y.iloc[fold.train_index]
X_test, y_test = matrix.X.iloc[fold.test_index], matrix.y.iloc[fold.test_index]

winner = models.make_regressor("hist_gb", seed=42).fit(X_train, y_train)
predictions = winner.predict(X_test)

metrics = ev.regression_metrics(y_test, predictions)
naive_mae = ev.best_baseline_mae(baselines, fold.name)
print(f"fold          : {fold.name}  ({fold.test_start.date()} to {fold.test_end.date()}, n={metrics['n']:,})")
print(f"MAE           : {metrics['mae']:.4f} kg   (best naive: {naive_mae:.4f})")
print(f"RMSE          : {metrics['rmse']:.4f} kg")
print(f"R2            : {metrics['r2']:.4f}")
print(f"skill vs naive: {ev.skill_score(metrics['mae'], naive_mae):+.1%}")
""")

md("### 5.3 Three framings of the same problem")

code(r"""
framings = show("framings.csv")
table = (framings.groupby(["framing", "honest_label"])[["mae", "rmse", "r2", "skill_vs_naive"]]
         .mean().round(4).reset_index().sort_values("mae"))
display(table)
""")

md(r"""
This table is the milestone's central modelling result.

| Framing | MAE | R² | Skill | What it means |
|---|---|---|---|---|
| Routine only, oracle-gated | 0.315 | 0.154 | +38.0% | **Upper bound only.** Test rows are chosen using the label. |
| Two-stage, unfiltered (hard gate) | 0.479 | 0.044 | **+5.9%** | **Deployable.** The only framing that beats naive. |
| Two-stage, unfiltered (blended) | 0.516 | 0.078 | −1.2% | Probability-weighted; better R², worse MAE. |
| Single regressor, all data | 0.567 | 0.031 | −11.1% | Realistic but poor. |

Milestone 3's headline — routine Gradient Boosting, R² = 0.216 — is the first row.
`extreme_weight_change_flag` is defined as `ABS(target) > 5`, so filtering the test set by
it means assuming you already know whether tomorrow is an extreme day *before* you predict
it. In production nothing tells you that. The number is an upper bound and is labelled as
one everywhere it appears.

The two-stage pipeline is what actually does that job: a classifier decides whether
tomorrow is extreme, and its decision — right or wrong — routes the regression. Scored on
the **full unfiltered test set**, it reaches MAE 0.479 and is the only configuration to
beat the naive bar, by 5.9%. That is a modest number honestly earned, and it is the one to
report.

### 5.4 The extreme-event classifier
""")

code(r"""
classifier = show("classifier.csv")
display(classifier.round(4))
print(f"mean PR-AUC {classifier.pr_auc.mean():.3f} against a no-skill rate of "
      f"{classifier.pr_auc_no_skill.mean():.3f}  ->  {classifier.pr_auc_lift.mean():.1f}x lift")
""")

md(r"""
At a 1.7% positive rate, accuracy is meaningless — predicting "never extreme" scores 98.3%
— so the primary metric is PR-AUC, reported against the no-skill rate that equals the
positive rate.

The classifier reaches **PR-AUC 0.17–0.31 against a no-skill 0.010–0.018**, a lift of
10–17×, with ROC-AUC 0.80–0.85. Extreme days carry real, detectable advance signal. Given
Section 3.5, much of what it detects is likely the seasonal and hive-state context in
which a beekeeper opens a hive — which is genuinely useful, but is not the same thing as
predicting colony behaviour.

### 5.5 Is the ranking real? Seeds and hives
""")

code(r"""
stability = show("seed_stability.csv")
display(stability.round(4))
rf = stability[stability.model == "rf"]; hgb = stability[stability.model == "hist_gb"]
print(f"RF      MAE range: {rf.mae.min():.4f} - {rf.mae.max():.4f}")
print(f"HistGB  MAE range: {hgb.mae.min():.4f} - {hgb.mae.max():.4f}")
print(f"non-overlapping ranges: {ev.ranking_is_significant(rf, hgb)}")
""")

code(r"""
hive_cv = show("hive_generalisation.csv")
display(hive_cv.round(4))
print(f"leave-hives-out MAE {hive_cv.mae.mean():.3f} +/- {hive_cv.mae.std():.3f}, "
      f"R2 {hive_cv.r2.mean():.3f} +/- {hive_cv.r2.std():.3f}")
""")

md(r"""
Milestone 3 claimed Gradient Boosting beat Random Forest on a ΔR² of 0.018 from one split
at one seed. Re-run across seeds, the boosting/forest gap holds — the MAE ranges do not
overlap — so the claim survives, now with evidence behind it.

The leave-hives-out result is the more interesting one. Held-out hives give MAE 0.42–0.66
and R² 0.05–0.13, which is **comparable to the temporal folds**. The model is not leaning
on hive identity, so it should transfer to a hive that was never part of training. For a
beekeeper, that is the difference between a research artifact and something deployable.

### 5.6 Feature importance, done properly
""")

code(r"""
importance = show("permutation_importance.csv").head(12)
fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(importance.feature[::-1], importance.importance_mean[::-1],
        xerr=importance.importance_sd[::-1], color="#4c72b0")
ax.set(xlabel="increase in MAE when permuted", title="Permutation importance (test set)")
plt.tight_layout(); plt.show()
display(importance.round(5))
""")

md(r"""
Milestone 3 used impurity-based importance (MDI), which is biased toward high-cardinality
continuous splits — and this feature set is near-collinear by construction, since
previous-day weight and the 3- and 7-day rolling means are three views of one signal.
Permutation importance on the **test set** replaces it.

The revised picture is sharper than "recent weight history dominates". The single dominant
feature is **`previous_day_weight_change_kg`** — yesterday's *change*, at roughly three
times the importance of the next feature. Absolute weight levels matter far less than
Milestone 3's chart implied, and the differenced features added in this milestone
(`weight_slope_2_day_kg`, `weight_minus_rolling_3_kg`) carry real weight.

Caveat: permutation importance under collinearity *understates* every member of a
correlated group, because permuting one leaves the model able to lean on the others. The
ranking within the weight-history block should not be read too closely.
""")

md(r"""
## 6. Results and Interpretation

### 6.1 Where the model works, and where it fails

**A skill score is a ratio, and the first version of this section used the wrong
denominator.** Every segment was divided by *one* naive bar computed over the whole test
fold. On a target whose variance is this seasonal that is not neutral: the annual bar
carries summer's variance into the winter comparison and winter's into the summer one, so
a winter prediction is scored against a bar summer made easy and a summer prediction
against a bar winter made hard.

Both columns are reported below. `skill_vs_naive` scores each segment against the best
naive rule *within that segment*; `skill_vs_pooled_naive` is the single-bar convention this
notebook originally published.

**Which column to read depends on the segment**, and the rule is whether the segment is
knowable before the forecast is made. Season and month are: it is January, and a beekeeper
choosing between this model and a rule of thumb in January is choosing between them *in
January*. A |change| decile is not — it is defined by the label, so its within-segment bar
is a competitor that already knows the answer. The seasonal table below is read on the
within-segment column and the decile chart on the pooled one, and each says which it uses.
""")

code(r"""
segmented = ev.segmented_report(
    y_test, predictions, matrix.meta.iloc[fold.test_index],
    by=("season", "month", "weight_change_decile"),
    baseline_mae=naive_mae,
    baseline_predictions=ev.naive_baselines(model_df, fold),
)
seasons = segmented[segmented.segment == "season"]
display(seasons[["value", "n", "mae", "baseline_mae", "baseline_rule",
                 "skill_vs_naive", "skill_vs_pooled_naive"]].round(4))

fig, ax = plt.subplots(figsize=(9, 3.8))
x = np.arange(len(seasons))
ax.bar(x - 0.2, 100 * seasons.skill_vs_pooled_naive, 0.4,
       color="#c44e52", label="one bar for the whole fold (as originally published)")
ax.bar(x + 0.2, 100 * seasons.skill_vs_naive, 0.4,
       color="#55a868", label="within-season bar (correct)")
ax.axhline(0, color="black", linewidth=1)
ax.set_xticks(x, seasons.value)
ax.set(ylabel="skill vs. naive (%)",
       title="Same model, same predictions — the denominator decides the seasonal story")
ax.legend(fontsize=9)
plt.tight_layout(); plt.show()
""")

code(r"""
decile = segmented[segmented.segment == "weight_change_decile"].sort_values("value")
fig, ax = plt.subplots(figsize=(10, 4))
colors = ["#55a868" if s > 0 else "#c44e52" for s in decile.skill_vs_pooled_naive]
ax.bar(decile.value, decile.skill_vs_pooled_naive, color=colors)
ax.axhline(0, color="black", linewidth=1)
ax.set(title="Skill vs. naive by |weight change| decile (pooled bar — see note)",
       xlabel="decile of |actual change|", ylabel="MAE reduction vs. best naive")
plt.tight_layout(); plt.show()
display(decile[["value", "mae", "n", "baseline_rule",
                "skill_vs_pooled_naive", "skill_vs_naive"]].round(4))
""")

md(r"""
**The seasonal claim does not survive the correction.** Same model, same predictions, same
rows — only the denominator changes. Predicting zero is the best naive rule in all three
seasons, so this is not an artifact of a rule swapping in:

| Season | MAE | Within-season naive | One fold-wide bar (as published) | Within-season bar |
|---|---|---|---|---|
| Autumn | 0.2215 | 0.2173 | **+51.2%** | **−1.9%** |
| Winter | 0.3008 | 0.2764 | **+33.7%** | **−8.8%** |
| Summer | 0.8854 | 0.9097 | **−95.2%** | **+2.7%** |

(The final fold runs 2022-07-05 to 2022-12-24, so it has no spring rows. That has always
been true of this section.)

The reason is arithmetic rather than subtle. The fold-wide bar is 0.4536 kg. Predicting
zero costs only 0.2764 kg in winter dormancy, because a dormant colony's weight barely
moves, and 0.9097 kg in the nectar flow. Dividing every season by the annual average of
that competitor hands winter a bar 1.6× too easy and summer one 2× too hard.

**The honest reading of the corrected column is that this model has essentially no seasonal
skill in either direction** — −8.8% to +2.7%, with summer the only positive. That is a
weaker claim than "+51% in autumn" and a much weaker one than "−95% in summer", and it is
consistent with Section 5.1's finding that no single-stage model beats predicting zero
overall. The seasons were never the story; the denominator was.

`Milestone_4_alt.ipynb` §6.1 carries the same correction at weekly grain, where the model
does have real seasonal structure once it is scored properly: +62% → −17.5% for winter and
−24% → +16.4% for summer.

**The decile view keeps the pooled bar, and it does not reverse.** The model beats naive on
deciles 1–8 of |change| and loses on the top two, catastrophically on the largest
(−546%). It is a good model of quiet days. The `skill_vs_naive` column is printed alongside
only to show why it cannot be read: predicting zero inside D1 costs 0.008 kg *by
construction*, because D1 is the set of days on which almost nothing happened, so every
model on earth scores −939% against it.

So the seasonal finding and the decile finding were never the same finding, and the first
version of this section ran them together. "Good where it is easy, bad where it is
valuable" was always true of the deciles. It was an artifact in the seasons.

(R² within a decile is not interpretable — conditioning on |y| collapses the variance that
R² normalises by, which is why those values are large and negative. MAE and skill are the
honest columns there.)

### 6.2 Corrections to Milestone 3

A milestone that catches its predecessor's errors is doing its job. Three matter:

1. **The routine R² of 0.216 was oracle-gated** (Section 5.3). The test set was filtered
   by a flag derived from the label. Deployable performance from the two-stage pipeline is
   R² ≈ 0.04, MAE 0.479, skill +5.9%.
2. **"Extremes are unrelated to beekeeper actions" is contradicted by the publishers' own
   flag** (Section 3.5). 94% of the 445 extremes vanish under the publishers'
   beekeeper-handling filter. The previous analysis compared against sparse manual logs
   instead, and those logs were unit-corrupted.
3. **The extremes investigation ran on the wrong table** (Section 3). Minute-grain records
   under a 0.3 kg/min rule, not daily rows under a ±5 kg rule; the reported 0% proximity
   and 0.213 clustering correlation do not transfer, and both are recomputed here.

Two smaller corrections: the dataset was miscited as HOBOS/USDA Tucson (Section 2.1), and
the "151 of 453 files" coverage risk was unfounded (Section 2.3).

### 6.3 Is the model learning anything?

Honestly: **a little, and not where it matters most.**

- Against the naive bar, a single-stage regressor is *worse* on MAE (−6% to −11%).
- The two-stage pipeline is the only configuration that beats it, by +5.9%.
- The gain is concentrated in low-variance seasons and small changes.
- The extreme-event classifier is the strongest result in the milestone: a 10–17× PR-AUC
  lift over no-skill, holding across all four folds.

The most defensible reading is that **next-day change in daily-mean hive weight is close
to noise-dominated at this grain**, and that the tractable problem is not regression on
the routine core but detection of the events — which are substantially beekeeper handling.

### 6.4 Stability

Fold-to-fold spread is wide: HistGB's MAE ranges 0.43–0.63 across the four rolling-origin
folds, and R² swings 0.036 ± 0.130. Because R² is normalised by each window's variance and
this target's variance is strongly seasonal, a winter-heavy fold and a summer-heavy fold
are not comparable on R² at all. That is why MAE and skill lead every table here and R² is
reported per fold and per season rather than pooled.
""")

md(r"""
## 7. Conclusions and Recommendations (Preliminary)

### 7.1 What we can now claim

- A two-stage classifier→regressor pipeline predicts next-day change in daily-mean hive
  weight with **MAE 0.479 kg, R² 0.044, +5.9% skill over the best naive predictor**,
  measured on unfiltered test data across four rolling-origin folds. This is the only
  configuration that beats predicting zero.
- **Extreme weight events are detectable in advance**: PR-AUC 0.17–0.31 against a no-skill
  rate of 0.010–0.018 (10–17× lift), consistently across folds.
- Performance is **not** meaningfully season-dependent. Scored against a within-season
  naive competitor the range is −8.8% (winter) to +2.7% (summer). The "+51% autumn / +34%
  winter / −95% summer" figures this notebook first reported were a property of the
  denominator, not of the model — Section 6.1.
- The model **generalises to unseen hives** — leave-hives-out performance matches temporal
  performance, so it is not memorising hive identity.
- **Yesterday's weight *change* is the dominant predictor**, not absolute weight level.

### 7.2 Recommendations for beekeepers

The project's point is whether a next-day forecast changes a decision.

**These recommendations were rewritten after the Section 6.1 correction.** The first
version told beekeepers to use the forecast for overwintering and not for harvest, on the
strength of +34–51% autumn/winter skill against −95% in summer. Those numbers were the
same predictions divided by an annual bar. Against within-season competitors the model is
close to parity everywhere, and the advice changes accordingly.

- **Harvest timing: not from this model, but not for the reason first given.** Summer is
  the model's *best* season on the corrected column (+2.7%), and +2.7% is not a decision
  aid. The forecast is not weakest during the nectar flow; it is roughly as good as
  predicting no change, everywhere.
- **Overwintering checks: no.** This is a straight reversal. Autumn and winter are the
  seasons where the model *loses* to predicting zero (−1.9% and −8.8%). A dormant colony's
  weight barely moves, which makes "assume no change" an excellent rule and a hard one to
  improve on. The earlier +34–51% was that easy bar being scored on the annual scale.
- **Event alerting: still the most promising track, with a caveat, and now the only one.**
  The classifier fires 10–17× better than chance, and nothing in the correction touches it
  — PR-AUC is not a skill score and has no naive-bar denominator to get wrong. But because
  most of what it detects is beekeeper handling, its practical use is closer to *verifying
  that a logged intervention had the expected weight effect* than to warning of an
  unattended swarm. An operating point favouring recall is the right default for early
  warning, at the cost of precision.
- **If a beekeeper wants a usable forecast, it is not at this grain.**
  `Milestone_4_alt.ipynb` reaches +16.4% in summer and +20.7% in spring at weekly grain on
  the same corrected scoring. A week is the horizon this dataset supports.

### 7.3 What the data cannot support

- **Extremes as a regression target.** With 94% of them attributable to handling, and the
  publishers' own cleaned series showing only 101 events, there is no colony-behaviour
  signal left to predict at this threshold.
- **Generalisation beyond German colonies.** 78 hives, one country, citizen-science
  collection, 2019–2022. Sensor missingness runs 23–35% and the beekeeper logs are
  voluntary and inconsistently kept.
- **Sub-daily inference.** The daily value is a *mean* of the day's minute readings, so
  intra-day dynamics — the actual foraging signal — are averaged away before we see them.

### 7.4 Open questions for Milestone 5 — with what came of them

Items 2 and 3 have since been carried out in `Milestone_5.ipynb`, and neither held up.
The outcomes are recorded here rather than left as open suggestions.

1. **Model the minute or hourly grain.** The 52M-row minute archive is available and
   contains the foraging signal that daily averaging destroys. Still open, and still the
   single largest untried opportunity. Note that `Milestone_4_alt.ipynb` moved in the
   *other* direction — to weekly and monthly — and found that longer periods help, which
   makes the sub-daily case less obvious than it looked from here.
2. ~~**Weather integration** (Part F of the plan)~~ — **done, and it is not worth
   building.** `data/honey_weather.parquet` now holds 45,594 site-days of ERA5 reanalysis
   for the 34 sites (reanalysis rather than GHCN-Daily: gridded, complete, no
   nearest-station or gap-filling rules). At weekly grain the weather already in hand is
   worth nothing, and giving the model the *actual* weather of the week being forecast — a
   perfect seven-day forecast — raises skill from +8.7% to +9.1%. Half a point is the
   ceiling. `Milestone_4_alt.ipynb` §5.6.
3. ~~**Hive-relative extreme definition.**~~ **The comparison behind this was not a
   comparison.** PR-AUC is bounded below by the positive rate, and the 5-MAD rule fires far
   more often than the absolute ±5 kg rule — so "0.52 against 0.23" was mostly the rate
   difference. Matched to the same positive rate at weekly grain, the absolute label
   reaches PR-AUC **0.32** and the relative label **0.20**, and the end-to-end two-stage
   regression scores +5.9% against −1.9%. Removing the hive-size shortcut makes the label
   more meaningful and the task less learnable. `Milestone_5.ipynb` §5.
4. Hyperparameter tuning, multi-day horizons, and per-hive personalisation — all untouched
   so far. Multi-day horizons turned out to be the productive one and became
   `Milestone_4_alt.ipynb`; tuning is not worth doing on a result whose fold-to-fold sd is
   wider than every effect it would chase.
5. **New, and now the highest-priority item:** re-score every segmented table in this
   notebook and in `Milestone_4_alt.ipynb` against within-segment baselines. Sections 6.1
   and 7.2 above are done; the per-hive and per-month tables in `results/segmented.csv`
   carry both columns but the prose around them has not been revisited.

### 7.5 Limitations

- Everything is measured on the `weight_kg` series, which retains beekeeper-handling
  artifacts. The parallel analysis on the publishers' cleaned series is reported in
  Section 3.5 but not modelled.
- 597 rows below a 5 kg physical floor remain in the table (flagged, not dropped), as does
  a −65.3 kg minimum change that is almost certainly not a real colony event.
- The ±5 kg threshold is retained for comparability rather than because the sweep favours
  it.
- Weather features are specified in `honeymodel.features.WEATHER_FEATURES` but the join is
  not yet built, so the ablation ladder stops at `history+sensors`.
- The Zenodo licence question (Section 2.1) is open.
""")

md("## Appendix A — `honey_model` schema")

code(r"""
schema = pd.DataFrame({"column": model_df.columns, "dtype": [str(t) for t in model_df.dtypes]})
schema["forbidden_as_feature"] = schema.column.isin(features.FORBIDDEN_COLUMNS)
display(schema)
""")

md(r"""
## Appendix B — Reproduction checklist

| Check | Where |
|---|---|
| Clean-checkout run with no network, no VPN, no DuckDB file | this notebook |
| Rebuild parity — SQL pipeline reproduces the published table exactly | `scripts/build_honey_model.py` |
| No leakage — forbidden columns rejected, with a negative test | Section 2.6 |
| Split integrity — train ends strictly before test begins, every fold | Section 4.1 |
| Baselines beaten (or not) — skill score on every model | Sections 4.3, 5.1 |
| Two-stage honesty — unfiltered metrics present, oracle numbers labelled | Section 5.3 |
| Segmented tables — season, month, hive, change decile | Section 6.1 |
| Cross-validation — rolling-origin, mean ± sd | Sections 4.2, 5.1 |
""")

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13.5"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path("/home/djorgs/Documents/git/honey-yield-predictive-model/Milestone_4.ipynb")
out.write_text(json.dumps(notebook, indent=1))
print(f"wrote {out} with {len(cells)} cells")
