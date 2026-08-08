"""Generate Milestone_5.ipynb -- the five recommended next steps from Milestone 4 alt 7.3."""
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
# Milestone 5 — Acting on the Recommendations

**Team:** Stephanie Nord, David Jorgensen, Joshua Amaya
**Course project:** predicting hive weight change from colony sensor history

---

## What this notebook is

`Milestone_4_alt.ipynb` closed with five recommended next steps, Section 7.3. This
notebook does them, in order, and reports what each one turned out to be worth.

| # | Recommendation | Section | Verdict |
|---|---|---|---|
| 1 | Model gross gain, not net weight change | 2 | **Done. It makes the problem harder, not easier** — and the beekeeper log will not support it |
| 2 | Give the summer regime its own model | 3 | **Done. Gating does not help** — and the seasonal story it was meant to fix was mostly a scoring artifact |
| 3 | Add weather | 4 | **Done. Nothing deployable.** A *perfect* one-week forecast is worth about +1.5 skill points, and only with a compact feature set |
| 4 | Make the extreme threshold per-hive relative | 5 | **Done. It is worse** once the two rules are compared at the same positive rate |
| 5 | More years before trusting the monthly result | 6 | **Cannot be done** — the archive ends 2022-12-30. What is run instead is the sample-size curve that says how much more would be needed |

Four of the five recommendations do not survive being carried out. That is the finding,
and it is worth more than a fifth ablation table would have been: each one was a
plausible-sounding idea written into a conclusions section, and each one is now measured.

## What is new in the repository

| Added | What it is |
|---|---|
| `data/honey_weather.parquet` | 45,594 site-days of ERA5 reanalysis weather, one series per distinct hive site, 2019-05-01 to 2022-12-31 |
| `scripts/pull_weather_snapshot.py` | the one network step that builds it |
| `src/honeymodel/harvest.py` | harvest reconstruction and the gross-gain target |
| `src/honeymodel/weather.py` | weather features at daily and period grain, with a forbidden look-ahead rung |
| `src/honeymodel/regimes.py` | season-gated regressor and per-regime scoring |
| `src/honeymodel/power.py` | the sample-size curve |
| `scripts/run_next_steps.py` | computes every table this notebook renders |

## How to run this notebook

```bash
pip install -r requirements.txt
PYTHONPATH=src python scripts/run_next_steps.py   # ~4 minutes, writes results/ns*.csv
jupyter nbconvert --to html --execute Milestone_5.ipynb
```

No network, no VPN, no credentials. `scripts/pull_weather_snapshot.py` is the only thing
in the project that touches a network endpoint, it has already been run, and its output is
committed. Reproducing this notebook does not re-run it.

## Headline

Carrying out four measurable recommendations produced **one** improvement, and it is an
upper bound rather than a deliverable: giving the model a compact set of weather features
*for the week being forecast* raises weekly skill from +7.6% to +9.2%. Every other change
either cost skill or moved it inside the fold-to-fold noise. The most useful output of the
exercise is Section 3.2, which shows that Milestone 4's seasonal narrative — "+62% in
winter, −24% in summer" — is largely an artifact of scoring every season against one
annual naive bar.

**All four corrections this notebook found have since been applied** to the notebooks they
concern, so the three deliverables no longer contradict each other. Section 7.2 lists what
changed where. The weather layer itself now lives in `Milestone_4_alt.ipynb` §5.6, where
it belongs — this notebook keeps the ablation that decided it was not worth building.
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

from honeymodel import data, harvest, periods as pr, power, regimes, weather as wx

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})

RESULTS = Path("results")

def show(name, **kwargs):
    # Render a table produced by scripts/run_next_steps.py. Every number below comes from
    # that run, so nothing in the write-up can drift from the code that produced it.
    return pd.read_csv(RESULTS / name, **kwargs)

daily_df = data.add_season(data.load_model_table())
week_df = pr.add_period_features(pr.aggregate_to_period(daily_df, "week"))

print(f"daily rows  : {len(daily_df):,}")
print(f"hive-weeks  : {len(week_df):,}")
print(f"weather rows: {len(wx.load_weather()):,} across {wx.load_weather().site_id.nunique()} sites")
""")

# ---------------------------------------------------------------------------
md(r"""
## 2. Item 1 — Model gross gain, not net weight change

> **7.3.1** The `honey` event log records when a harvest happened but not how much was
> removed. Reconstructing removed mass from the weight series — a step change coincident
> with a logged harvest — would turn the target into something closer to honey production.
> This is the change with the largest expected payoff and it is not a modelling change, it
> is a target change.

The premise is right. Net weekly weight change is honey production *minus* whatever the
beekeeper carried off, and in the weeks a beekeeper cares about the second term is much
the larger. A model that forecasts −14 kg for the week of a harvest is not wrong about the
colony; it is right about the beekeeper.

The recommendation assumed the log would tell us when. Section 2.1 shows that recovering
*when* is possible and the recovery validates beautifully. Section 2.3 shows that
recovering *how much* from it does not work at all.
""")

md("### 2.1 Recovering the beekeeper's log from a counter that resets")

code(r"""
calendar = show("ns1_logged_event_calendar.csv")
calendar[["event_type", "resets_found", "accepted", "hives", "peak_month"]]
""")

md(r"""
The publishers ship no event table at daily grain — `honey_last` and `honey_next` are
entirely null in the `years` files. What survives is `{event}_last_dif`, the time since the
previous event of that type, which climbs monotonically and **resets when a new event
happens**. `harvest.detect_logged_events` reads the reset, not the value.

Reading the reset also sidesteps a defect one layer deeper than the one Milestone 4 found.
`data.normalise_event_distance_units` infers one unit per hive, days or seconds. That is
not enough: hive 21's counter increments by 1.0 per day through 2019 and by 86,400 per day
through mid-2020. **The unit switches inside a single hive's record.** A reset is a reset
in either unit, so nothing has to be converted to detect one.

Two filters remove 61 of 129 candidate honey resets, and both removals are defensible:

- a residual counter value that, divided by the locally estimated per-day increment, dates
  the event *before* the counter was last seen higher — a contradiction, and the signature
  of a mixed-unit stretch;
- 12 of 16 January events landing on 1 January exactly, across 12 different hives, at
  counter values two orders of magnitude above the local scale. That is the publisher's
  per-year processing restarting the counter, not a midwinter harvest.
""")

md("**The reconstruction was never told the beekeeping calendar, and reproduces it:**")

code(r"""
months = calendar.set_index("event_type")[[f"m{m:02d}" for m in range(1, 13)]]
months.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

fig, ax = plt.subplots(figsize=(11, 3.4))
for event in ["honey", "feeding", "queencell", "treatment"]:
    ax.plot(months.columns, months.loc[event], marker="o", label=event)
ax.set_ylabel("accepted events")
ax.set_title("Recovered beekeeper events by month — no calendar was supplied to the detector")
ax.legend(ncol=4, fontsize=9)
plt.tight_layout()
plt.show()
""")

md(r"""
Honey peaks April–August; feeding peaks July–September, which is when winter stores go on
after the last harvest; queencell peaks April–June, the swarm season; treatment peaks
August–November, which is when varroa is treated. Nothing in `detect_logged_events`
encodes any of that. Getting four independent event types to land on their correct months
is about as strong a validation as a reconstruction of this kind can get.
""")

md("### 2.2 Detecting removals from the weight series instead")

code(r"""
show("ns1_removal_threshold_sweep.csv")
""")

md(r"""
The second route ignores the log. A removal is a day-over-day drop of at least
`threshold_kg` that is **still there a week later** — at least 60% of it — on a row not
already flagged `sensor_dropout_flag` or `implausible_weight_flag`, measured across a
single day rather than across a gap.

The sustain test is what does the work. Milestone 4's `sensor_dropout_flag` exists because
a large drop with a matching recovery is a sensor artifact; a hot dry afternoon produces
the same shape at smaller amplitude. Requiring the loss to persist removes both.

`share_may_to_aug` is the column that justifies 3 kg. A threshold picking up harvests
concentrates in the harvest months and one picking up weather does not: the share climbs
from 0.62 at 2 kg to 0.83 at 10 kg, so higher thresholds are cleaner but progressively
throw away real harvests. 3 kg — roughly one full shallow frame of capped honey, and 1.6
daily sd — keeps 230 removals across 51 hives at a 0.68 in-season share.
""")

md("### 2.3 The two routes barely agree, and that ends the recommendation as written")

code(r"""
show("ns1_corroboration.csv")
""")

md(r"""
**Between 5% and 10%, in both directions, and widening the matching window from 1 day to
14 barely moves it.** Restricted to the 19 hives that keep a honey log at all — so this is
not measuring record-keeping — only 7 of 68 logged honey events have a sustained weight
drop anywhere near them, and only 7 of 93 sustained drops on those hives sit near a logged
honey event.

Three readings, and they are not mutually exclusive:

1. **The `honey` log is not a removal log.** Adding a super is also a honey operation, and
   several logged events coincide with weight *increases* of 30–39 kg. The event type
   records that the beekeeper did something about honey, not that honey left.
2. **Voluntary logs are kept loosely.** 19 of 78 hives log anything under `honey` at all.
3. **Daily means smear the step.** A harvest at 14:00 lands as a partial change on both
   days either side.

Whichever dominates, the consequence is the same: the recommendation's mechanism —
*a step change coincident with a logged harvest* — is not available in this dataset. The
step changes are there. The coincidence is not. Everything below therefore uses the
weight-series route alone, and the resulting target is "gross of detected removals", not
"gross of harvest".
""")

md("### 2.4 The target change, and what it costs")

code(r"""
show("ns1_target_comparison.csv")
""")

md(r"""
The two targets are meaningfully different where it matters and nowhere else. Pooled, a
removal touches 4.3% of hive-weeks. In May–August it touches 8.1%, and on those weeks the
target moves by a mean of **13.7 kg** — which is the whole point: the pooled May–August net
change is −0.03 kg and the gross gain is +1.08 kg. Net weight change says a German colony
does not gain mass over the nectar flow. It gains 1 kg a week and the beekeeper takes it.

The two targets correlate at r = 0.73 pooled and r = 0.70 across May–August, so this is a
different target rather than a relabelling of the same one.
""")

code(r"""
board = show("ns1_target_board.csv")
summary = (board.groupby(["target", "model"])[["mae", "skill_vs_naive"]]
                .agg(["mean", "std"]).round(4))
summary.columns = ["mae_mean", "mae_sd", "skill_mean", "skill_sd"]
summary.reset_index()
""")

md(r"""
**The physically meaningful target is the harder one.** Random Forest reaches +7.6% skill
on net change and +1.9% on gross gain, against baselines re-derived separately for each.
The gross target adds 13 kg spikes that the model has no feature capable of anticipating —
nothing in the feature set knows the beekeeper's calendar — so the extra variance lands
entirely in the error term.

This is worth stating plainly because it inverts the recommendation. 7.3.1 called this
"the change with the largest expected payoff". Measured, it is the change with the largest
*cost*, and the payoff it was reaching for — a target that means honey production — is
real but is not collectable without a removal log the publishers did not ship.
""")

# ---------------------------------------------------------------------------
md(r"""
## 3. Item 2 — Give the summer regime its own model

> **7.3.2** Skill is +62% in winter and −24% in summer with one pooled model. A
> season-conditional model, or a mixture with a seasonal gate, directly attacks the failure
> mode instead of averaging over it.

`regimes.SeasonGatedRegressor` fits one regressor per regime and routes on the `month`
feature. **The gate reads a feature, not the label** — the calendar month of the week being
forecast is known on Sunday evening — so this is not oracle gating in the sense
`models.oracle_gated_predict` warns about. Contrast the two-stage extreme pipeline, whose
gate has to be learned because whether next week is extreme is not knowable.

Three schemes are tried, all with the same regressor so that a gain is attributable to
splitting rather than to model selection: `flow_vs_rest` (April–August against the rest),
`colony_cycle` (build-up / flow / dormancy) and `season` (the four meteorological seasons).
""")

md("### 3.1 Gating does not help")

code(r"""
folds_board = show("ns2_regime_fold_board.csv")
overall = (folds_board.groupby("model")[["mae", "skill_vs_naive"]]
                      .agg(["mean", "std"]).round(4))
overall.columns = ["mae_mean", "mae_sd", "skill_mean", "skill_sd"]
overall.sort_values("skill_mean", ascending=False).reset_index()
""")

md(r"""
Pooled +7.6%, and every gated variant below it: `colony_cycle` +6.7%, `season` +5.2%,
`flow_vs_rest` +5.1%. The differences are inside a fold-to-fold sd of 0.13–0.16 and no
ranking should be claimed from them — but the direction is consistent across all three
schemes, and nothing here supports gating.

The reason is a sample-size one rather than a conceptual one. 3,012 weekly rows split four
ways leaves the Summer partition around 100 test rows per fold and a correspondingly thin
training slice, and each regime model gives up the cross-regime rows that taught the
pooled model what a weight series does. `min_rows=150` catches the worst cases by falling
back to the pooled fit, which is why the gated numbers are close rather than catastrophic.
""")

md("### 3.2 The seasonal story was mostly a scoring artifact")

code(r"""
regime_summary = show("ns2_regime_summary.csv")
pooled_only = regime_summary[regime_summary.model == "pooled"]
pooled_only[["regime", "n", "mae", "skill_vs_naive", "skill_vs_fold_naive"]].round(4)
""")

md(r"""
This is the table that matters most in this notebook, and it corrects both Milestone 4
notebooks rather than extending them.

`skill_vs_fold_naive` is the Milestone 4 convention: every segment scored against **one
naive bar computed over the whole fold**. `skill_vs_naive` scores each season against the
best naive rule *within that season*. Same predictions, same rows, different denominator:

| Season | Milestone 4 convention | Within-season bar |
|---|---|---|
| Winter | **+51.7%** | **−38.9%** |
| Summer | **−46.5%** | **+9.1%** |
| Spring | +18.0% | +13.5% |
| Autumn | +4.3% | +0.6% |

**The two headline seasonal claims reverse.** The annual bar is inflated by summer's
variance and deflated by winter's, so a winter prediction is compared against a bar that
summer made hard, and a summer prediction against a bar that winter made easy. Once each
season is scored against its own naive competitor:

- **Winter is the model's worst season, not its best.** In dormancy the weight series is a
  slow monotone drain and predicting zero is very hard to beat; the model loses to it by
  39%.
- **Summer is the model's best season, not a disaster.** Against a summer-specific bar it
  is +9.1%, not −46.5%.

The practical recommendation in `Milestone_4.ipynb` §7.2 — *harvest timing: not yet;
overwintering checks: usable* — was derived from the reversed version of this table and
should be read the other way round.

None of this rescues gating, and it partly explains why gating failed: the failure mode it
was designed to attack was not there.
""")

code(r"""
seasons = ["Winter", "Spring", "Summer", "Autumn"]
subset = pooled_only.set_index("regime").loc[seasons]

fig, ax = plt.subplots(figsize=(9, 3.8))
x = np.arange(len(seasons))
ax.bar(x - 0.2, 100 * subset.skill_vs_fold_naive, 0.4, label="scored vs the fold-wide bar (Milestone 4)")
ax.bar(x + 0.2, 100 * subset.skill_vs_naive, 0.4, label="scored vs the within-season bar")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x, seasons)
ax.set_ylabel("skill vs naive (%)")
ax.set_title("Same model, same predictions — the bar decides the seasonal story")
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()
""")

# ---------------------------------------------------------------------------
md(r"""
## 4. Item 3 — Add weather

> **7.3.3** `honey_weather.parquet` is referenced by the daily feature set but is not in
> the repository. Weekly rainfall and degree-day totals are far more plausibly predictive
> of a week's nectar flow than a single day's reading is of tomorrow's, and `foraging_days`
> — the crudest possible proxy — already carries measurable weight.

### 4.1 The data layer

`scripts/pull_weather_snapshot.py` pulls ERA5 reanalysis from the Open-Meteo Historical
Weather API for each of the **34 distinct hive sites**, 2019-05-01 to 2022-12-31, and
writes `data/honey_weather.parquet` — 45,594 site-days, no missing values.

Reanalysis rather than GHCN-Daily, which the Milestone 4 plan named first and listed
sparse German station coverage as the matching risk. 34 sites over six degrees of
longitude would each need a nearest-station search, a distance threshold and a gap-filling
rule; a gridded product removes three judgement calls from the feature layer. The cost is
that these are modelled values on a ~9 km grid, not measurements — fine against hive
coordinates published to 0.1°, and worth remembering before reading much into one
site-day.

Ten daily variables were requested, each with a mechanism rather than for coverage:
temperature and sunshine gate whether bees fly at all, precipitation stops foraging
outright, and reference evapotranspiration is the closest free proxy for nectar secretion.
""")

code(r"""
coverage = show("ns3_weather_coverage.csv")
print(f"period weather features: {len(coverage)}   "
      f"minimum non-null: {coverage.non_null_pct.min():.1f}%")
show("ns3_weather_correlation.csv").head(8)
""")

md(r"""
Spearman rather than Pearson — rainfall is zero-inflated and heavy-tailed, and one
thunderstorm week would otherwise set the correlation.

`rho_next_period` is the column with a mechanism: next week's radiation should drive next
week's gain. It is positive and ordered exactly as the mechanism predicts — radiation
(0.11), evapotranspiration (0.09), foraging days (0.08), sunshine (0.07) — and it is
**smaller** than the current-period correlation for every one of them. That ordering says
these features work partly as seasonality proxies, which `sin_year`/`cos_year` already
supply, and only partly as weather.
""")

md("### 4.2 The ablation ladder")

code(r"""
ladder = show("ns3_weather_board.csv")
rf = ladder[ladder.model.str.endswith("/ rf")]
per_fold = rf.pivot(index="fold", columns="rung", values="skill_vs_naive")
order = ["history+sensors", "history+sensors+weather_core", "history+sensors+weather",
         "history+sensors+weather_core+lookahead", "history+sensors+weather+lookahead",
         "weather-only"]
per_fold = per_fold[order]
(per_fold * 100).round(1)
""")

code(r"""
tidy = pd.DataFrame({
    "all four folds": 100 * per_fold.mean(),
    "folds 2-4 only": 100 * per_fold.iloc[1:].mean(),
    "n_features": rf.groupby("rung").n_features.first().reindex(order),
    "honest_label": rf.groupby("rung").honest_label.first().reindex(order),
}).round(2)
tidy
""")

md(r"""
Three things fall out, and only the third is a gain.

**Sixteen weather features do not fit in this dataset.** The full weather rung drops mean
skill from +7.6% to −2.1%, and the whole loss is fold 1: −11.5% → −48.5%, while folds 2–4
are unchanged. Fold 1 is the first expanding-window origin and has barely a year of
training rows. The compact five-feature rung — one variable per mechanism, chosen before
the boards were run — costs nothing on fold 1 and is the honest way to ask the question.
The full set's failure is a dimensionality result, not a weather result.

**Weather you already have is worth nothing.** `history+sensors+weather_core` scores
+7.7% against +7.6% for no weather at all. The current week's weather adds nothing the
hive's own weight trajectory has not already recorded, which is unsurprising: the weight
series *is* an instrument for the weather that just happened.

**A perfect forecast is worth about 1.5 skill points.** The look-ahead rung — the model
given the forecast week's actual radiation, rain, degree-days, foraging days and wind —
reaches **+9.2%**, and on folds 2–4 alone +16.0% against +14.6%. That column is labelled
`UPPER BOUND` in the results table and is enforced in code: `weather.WEATHER_LOOKAHEAD_FEATURES`
is registered with `periods.assert_no_period_leakage`, so building a matrix from them
raises `PeriodLeakageError` unless the caller passes `allow=` and takes responsibility for
the label.
""")

code(r"""
try:
    pr.build_period_matrix(
        wx.add_period_weather(week_df, daily_df, lookahead=True),
        wx.weather_feature_sets()["history+sensors+weather+lookahead"],
    )
    print("NO GUARD — this should not print")
except pr.PeriodLeakageError as error:
    print("PeriodLeakageError raised as designed:\n ", str(error)[:220], "...")
""")

md(r"""
+1.5 points is the ceiling for a *perfect* seven-day forecast of five variables. A real
forecast is not perfect, and skill degrades over a week, so the deployable value is some
fraction of 1.5 points on a base of 7.6. **The weather integration Milestone 4 called
"Part F" is not worth building.**

`weather-only` — the weather plus the seasonality terms and nothing about the hive —
scores −36.6%. Whatever the model knows, it does not learn it from the sky.
""")

code(r"""
importance = show("ns3_weather_importance.csv")
weather_share = (importance[importance.is_weather].importance_mean.clip(lower=0).sum()
                 / importance.importance_mean.clip(lower=0).sum())
print(f"weather share of total positive permutation importance: {weather_share:.1%}")
print(f"weather features in the top 20: {int(importance.head(20).is_weather.sum())} of 20\n")
importance.head(10)[["feature", "importance_mean", "importance_sd", "is_weather"]]
""")

md(r"""
The model does *use* the weather — `wx_shortwave_radiation_sum_sum` is the fourth most
important feature and the highest-ranked non-weight column — it just does not get anything
out of it. 5% of total permutation importance, and `within_period_slope_kg_per_day` alone
carries 25 times more than any weather feature. Consistent with Section 4.1's correlation
ordering: the weather features are largely re-deriving the season.
""")

# ---------------------------------------------------------------------------
md(r"""
## 5. Item 4 — Make the extreme threshold per-hive relative

> **7.3.4** 11.1 kg on a 10 kg nucleus and on a 100 kg production hive are not the same
> event. `hive_target_mad_kg` already exists in the daily table for exactly this.

The argument is sound and the implementation is one line. What was missing from the
recommendation, and from the daily notebook's supporting number, is that **the two rules
have to fire at the same rate before they can be compared.**

`Milestone_4.ipynb` §7.4 records "the 5-MAD variant reaches PR-AUC 0.52 against 0.23 for
the absolute ±5 kg rule". PR-AUC is bounded below by the positive rate. At weekly grain a
5-MAD rule fires on 12.95% of weeks and the sd-derived absolute rule on 3.05%; a label
that fires four times as often will post a much higher PR-AUC while being a *worse*
detector. `periods.calibrate_relative_multiplier` bisects for the multiplier that matches
the absolute rule's rate, which here is 13.47 × the hive's own period-change MAD.
""")

code(r"""
show("ns4_relative_sweep.csv")
""")

md("### 5.1 What the two rules actually select, at a matched rate")

code(r"""
show("ns4_label_comparison.csv")
""")

md(r"""
At 92 positives each, the relative rule spreads them over 38 hives against the absolute
rule's 33, and puts 32.6% of them on the top five hives against 35.9%. **The concentration
argument is real but small.** It is not the 3-to-1 difference the recommendation's framing
implies, because `period_hive_change_mad_kg` is itself correlated with hive size — a big
hive has a big change MAD — so dividing by it removes less of the size effect than
expected.
""")

md("### 5.2 Detectability, and the two-stage pipeline")

code(r"""
classifier = show("ns4_classifier_board.csv")
classifier.groupby("rule")[["pr_auc", "pr_auc_no_skill", "pr_auc_lift", "roc_auc"]].mean().round(4)
""")

code(r"""
two_stage = show("ns4_two_stage_board.csv")
two_stage.groupby("rule")[["mae", "baseline_mae", "skill_vs_naive"]].agg(["mean", "std"]).round(4)
""")

md(r"""
**The relative label is the harder one to detect, not the easier one.** Mean PR-AUC 0.20
against 0.32 for the absolute rule at an identical positive rate, and the end-to-end
two-stage regression built on it scores −1.9% skill against +5.9%.

The two-stage comparison needs one implementation note stated rather than buried. Fitting
a per-row threshold means fitting on the target divided by the hive's own change MAD and
multiplying back afterwards; that division is floored at the 5th percentile of the MAD,
because a hive whose change MAD is 0.03 kg would otherwise have its errors multiplied by
thirty on the way back to kilograms and the comparison would be measuring the division.
Even with the floor the relative variant loses.

The mechanism is visible in the label comparison. The absolute rule's positives cluster on
a few large hives, which is exactly what a tree can learn from `hive_median_weight_kg` and
`period_weight_kg`; the relative rule deliberately removes that shortcut and leaves a
harder problem with no more signal in it. **Item 4 makes the label more meaningful and the
task less learnable.** Whether that trade is worth making is a judgement about what the
alert is for, and this notebook does not settle it — but the recommendation as written,
that the relative rule "should probably replace" the absolute one, is not supported.
""")

# ---------------------------------------------------------------------------
md(r"""
## 6. Item 5 — More years before trusting the monthly result

> **7.3.5** The monthly board's +12.3% is the best skill score in the notebook and the one
> least supported by its sample. Two more seasons would roughly double the monthly row
> count and make the fold-to-fold spread interpretable rather than merely small.

Two more seasons are not available. The published archive ends 2022-12-30 and there is no
2023 to pull, so this item cannot be done as written. What can be done is the shape of the
curve: refit the identical protocol on deliberately smaller samples and watch how fast the
uncertainty falls.

**The resampling unit is the hive, not the row.** Rows within a hive are one autocorrelated
series, and resampling them independently would make 622 monthly rows look like 622
independent observations — the exact error the exercise exists to measure. Each subsample
rebuilds folds, baselines and all from scratch; reusing the full-sample folds would hold
the test windows fixed while the training set shrank, which measures something else.
""")

code(r"""
curve = show("ns5_power_curve.csv")
curve.round(4)
""")

code(r"""
fig, ax = plt.subplots(figsize=(9, 3.8))
ax.errorbar(curve.rows_mean, 100 * curve.skill_mean, yerr=100 * curve.skill_sd.fillna(0),
            marker="o", capsize=4, label="mean skill +/- 1 sd across hive subsamples")
ax.fill_between(curve.rows_mean, 100 * curve.skill_min, 100 * curve.skill_max,
                alpha=0.15, label="observed range")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("monthly rows in the sample")
ax.set_ylabel("skill vs naive (%)")
ax.set_title("Monthly skill against sample size — 8 hive subsamples per point")
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()
""")

code(r"""
show("ns5_power_projection.csv").T.rename(columns={0: "value"})
""")

md(r"""
Two readings, and the second is the honest one.

**The narrow reading says the sample is already adequate.** Fitting `sd = c/sqrt(n)` and
solving for the sample size at which `skill − 1.96·sd > 0` gives about 480 monthly rows
against the 622 available, and no subsample at 85% or 100% of the fleet produced a negative
skill score. On sampling variability alone, the monthly result clears zero.

**The wider reading says the curve does not behave, and that is the result.** Mean skill
runs +1.2%, −0.9%, −2.0%, +5.9%, +7.2%, +7.9% as the sample grows — not monotone, and
negative in the middle. `share_negative` says that a study drawing 40% of this fleet would
have concluded "no skill" **half the time**. The fitted sd exponent is −0.77 rather than
−0.5, meaning the spread collapses faster than independent sampling would predict, which is
what happens when a handful of well-instrumented hives carry the result.

So the substantive answer to item 5 is that **more hives are not the binding constraint.**
Four folds drawn from 3.5 years of one country's weather is the constraint, and hive
subsampling cannot measure it: every subsample sees the same three summers. Two more
seasons would double the rows *and* add two independent weather years, and only the second
half of that is worth having. The number this section produces — "roughly 480 rows" — is a
lower bound on what is needed, computed under an assumption (that the effect size stays put
as the sample grows) that a small sample is exactly what cannot verify.
""")

# ---------------------------------------------------------------------------
md(r"""
## 7. Conclusions

### 7.1 The scoreboard

| # | Recommendation | Expected | Measured | Verdict |
|---|---|---|---|---|
| 1 | Gross gain, not net change | "largest expected payoff" | +7.6% skill on net, **+1.9% on gross** | Target is more meaningful and **harder**; the log cannot supply removed mass |
| 2 | Season-conditional model | fixes the −24% summer | pooled +7.6%, **best gated +6.7%** | No gain — and the failure mode it targeted was a scoring artifact |
| 3 | Weather | "far more plausibly predictive" | +7.6% → **+7.7%** deployable, **+9.2%** with a perfect forecast | Only gain in the notebook, and it is an upper bound |
| 4 | Per-hive relative extremes | "should probably replace" the absolute rule | PR-AUC **0.20 vs 0.32** at a matched rate | Worse. More meaningful label, less learnable task |
| 5 | More years | "would make the spread interpretable" | ~480 rows needed, 622 available | Not the binding constraint — independent *seasons* are |

### 7.2 What this changed about the earlier milestones — and what has since been fixed

All four corrections below have been applied to the notebooks they describe. They are
recorded here because the finding belongs to this notebook even though the fix does not.

- **The seasonal narrative in both Milestone 4 notebooks was wrong in its two headline
  claims.** Scored against a within-season naive bar, winter is the model's worst season
  and summer its best — the opposite of what both notebooks published. Section 3.2.
  **Fixed:** `Milestone_4_alt.ipynb` §6.1 now reports Winter −17.5% / Summer +16.4% against
  the +62% / −24% it first published, and `Milestone_4.ipynb` §6.1 reports Autumn −1.9% /
  Winter −8.8% / Summer +2.7% against +51% / +34% / −95%. Both carry the pooled column
  alongside so the size of the correction is visible.
- **Milestone 4's beekeeper recommendations followed from the reversed table.**
  **Fixed:** `Milestone_4.ipynb` §7.2 is rewritten. "Overwintering checks: usable" is now
  "overwintering checks: no" — autumn and winter are where the model loses to predicting
  zero — and the honest daily-grain conclusion is that it is near parity everywhere, with
  the usable horizon being a week rather than a day.
- **The 5-MAD PR-AUC comparison in `Milestone_4.ipynb` §7.4 was not a comparison** — the
  two labels fired at different rates. Matched, the ordering reverses. Section 5.2.
  **Fixed:** that item is struck through and carries the matched-rate result.
- **`data.normalise_event_distance_units` is not sufficient.** The published counters switch
  unit *within* a hive's record, not just between hives. Reset detection avoids the problem
  entirely. Section 2.1. **Fixed:** noted in `Milestone_4.ipynb` §3.3,
  `Milestone_4_alt.ipynb` §1 and `data/README.md`, each pointing at
  `harvest.detect_logged_events`.

One methodological point came out of applying the fix, and it is not a detail:
**within-segment baselines are right for segments that are knowable in advance and wrong
for segments defined by the label.** Season and month qualify; a |change| decile does not,
because predicting zero inside the lowest decile is nearly exact by construction. Both
notebooks now read the within-segment column for seasons and the pooled column for
deciles, and say which in both places.

### 7.3 What we would do next, having done this

Not a wish list this time — two items, each with the specific thing that would make it
worth doing:

1. **Ask the publishers for the raw event log.** Item 1's target change is the right idea
   and it failed on data availability, not on principle. A log with removed mass in it —
   which beekeepers do record, in their own notebooks — turns net weight change into honey
   production. Nothing in the modelling layer would need to change.
2. **Stop adding features.** Weather was the last untried external source with a plausible
   mechanism, and given a *perfect* forecast it bought 1.5 skill points at this grain and
   0.45 at weekly. The ceiling on this target is low, and the honest next step is to say so
   rather than to try hyperparameter tuning on a +7.6% result whose fold-to-fold sd is 14
   points.

The item that stood at the top of this list when the notebook was first written —
re-scoring the Milestone 4 segmented tables — has been done. `Milestone_4.ipynb` §7.4 item
5 carries what remains of it: the per-hive and per-month tables in `results/segmented.csv`
now carry both columns, but only the seasonal prose has been revisited.

### 7.4 Limitations

- Every removal in Section 2 is *detected*, not *logged*. The step-drop rule cannot
  distinguish a harvest from a swarm, a colony death, or a beekeeper moving frames between
  hives, and the corroboration rate says the beekeeper log will not disambiguate it.
- The weather is reanalysis on a ~9 km grid joined to coordinates rounded to 0.1°. It is
  complete and internally consistent, and it is not what the sensor at the hive measured.
- The look-ahead weather rung uses the forecast period's *actual* weather. It is labelled
  an upper bound everywhere it appears and enforced by the leakage guard, and it is not a
  deployable result.
- All of Section 6 measures variability across hives. It cannot measure variability across
  years, which is the constraint that actually binds, because every subsample sees the same
  three summers.
- Sections 3, 4 and 5 run at weekly grain on 3,012 rows with four folds. Fold-to-fold sd on
  the skill score is 13–16 percentage points throughout, which is wider than every effect
  measured in this notebook. No ranking here is claimed to be significant; the claims are
  about direction and consistency across folds, and where a result rests on one fold —
  Section 4.2 — that is said.

## Appendix A — Reproduction checklist

| Check | Where |
|---|---|
| Clean-checkout run, no network, no credentials | this notebook |
| The one network step is isolated and its output committed | `scripts/pull_weather_snapshot.py`, `data/honey_weather.parquet` |
| Event reconstruction validated against an external calendar | Section 2.1 |
| Removal threshold swept, not assumed | Section 2.2 |
| Both target definitions scored against separately re-derived baselines | Section 2.4 |
| Every segment scored against a within-segment bar, with the pooled convention shown alongside | Section 3.2 |
| Look-ahead weather forbidden in code, whitelisted only with an explicit `allow=` | Section 4.2 |
| Extreme labels compared at a matched positive rate | Section 5 |
| Resampling unit is the hive, folds rebuilt inside every subsample | Section 6 |
| Every table rendered from `results/ns*.csv`, produced by one deterministic script | `scripts/run_next_steps.py` |
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

out = Path(__file__).resolve().parent.parent / "Milestone_5.ipynb"
out.write_text(json.dumps(notebook, indent=1))
print(f"wrote {out} with {len(cells)} cells")
