# honey-yield-predictive-model

Predicting next-day honey bee hive weight change from sensor history, using the German
citizen-science colony archive (Senger et al. 2024). Tree-based models, a two-stage
routine/extreme pipeline, and an evaluation framework built to keep the numbers honest.

## The notebooks

| Notebook | Question | Headline |
|---|---|---|
| `Milestone_4.ipynb` | can we predict *tomorrow's* weight change? | Only a two-stage pipeline beats predicting zero, at +5.9% skill |
| `Milestone_4_alt.ipynb` | does a *weekly* or monthly horizon do better? | Yes — +15.2% two-stage, and the first single-stage model to beat naive |
| `Milestone_5.ipynb` | do that notebook's five recommended next steps hold up? | Four of five do not. And the seasonal story in both Milestone 4 notebooks is a scoring artifact |

## Reproducing the milestones

Three commands. No VPN, no database file, no credentials, no network access.

```bash
pip install -r requirements.txt
python scripts/build_honey_model.py          # optional: rebuilds data/honey_model.parquet
jupyter nbconvert --to html --execute Milestone_5.ipynb
```

Python 3.13.5. The notebooks read `data/honey_model.parquet` and `data/honey_weather.parquet`,
both committed — step 2 only verifies that the SQL pipeline still reproduces the first.

To re-run the analyses outside the notebooks:

```bash
PYTHONPATH=src python scripts/run_evaluation.py         # results/*.csv: baselines, models, framings
PYTHONPATH=src python scripts/run_extremes_analysis.py  # results/*.csv: extremes, threshold sweep
PYTHONPATH=src python scripts/run_next_steps.py         # results/ns*.csv: the Milestone 5 tables
```

## Layout

```
Milestone_4.ipynb          daily-horizon deliverable; every figure and number regenerates on run
Milestone_4_alt.ipynb      the weekly/monthly re-grain
Milestone_5.ipynb          carries out Milestone 4 alt's Section 7.3 recommendations
data/                      committed Parquet snapshots + provenance (data/README.md)
sql/01..06_*.sql           the pipeline, in order, extracted from notebooks
scripts/                   build the tables, refresh the snapshots, run the analyses
src/honeymodel/            data / features / evaluation / models / periods / harvest /
                           weather / regimes / power -- importable and testable
results/                   every table the notebooks render, written by the scripts
David_Work/  Stephanie_Work/  Joshua_Work/     per-member working notebooks
Milestone_3_Figures/       archive of the previous milestone's exported figures
```

Read [`data/README.md`](data/README.md) before using the data: it documents what the
publishers filtered out before release, a unit inconsistency in their event-distance
columns, and which columns must never enter a feature matrix.

## The two network steps, and why nothing else needs one

Both write a committed Parquet file and are never part of reproducing a result.

```bash
python scripts/pull_source_snapshot.py     # data/honey_daily_source.parquet -- needs Tailscale
python scripts/pull_weather_snapshot.py    # data/honey_weather.parquet -- needs the open internet
```

`pull_weather_snapshot.py` fetches ERA5 reanalysis from the Open-Meteo Historical Weather
API for each of the 34 distinct hive sites. It caches per site under `data/_weather_cache/`
so a rate-limited run resumes rather than starting over.

## Optional: the team's remote DuckDB

Only `scripts/pull_source_snapshot.py` uses this, and only to refresh the committed
snapshot. Nothing needed to reproduce a milestone touches it.

1. Contact @CmdrJorgs for a Tailscale invite, then
   [install Tailscale](https://tailscale.com/download/) and join the team network using
   the URL @CmdrJorgs provides.
2. Copy the `.env` file provided by @CmdrJorgs into the repository root. It is gitignored
   and must stay that way.
3. `pip install duckdb -U` if your environment predates DuckDB 1.5 (default Anaconda
   installs usually do).

The remote server also holds the minute- and hourly-grain sensor tables (52M rows), which
the daily modelling pipeline does not use.
