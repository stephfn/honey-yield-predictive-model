# honey-yield-predictive-model

Predicting next-day honey bee hive weight change from sensor history, using the German
citizen-science colony archive (Senger et al. 2024). Tree-based models, a two-stage
routine/extreme pipeline, and an evaluation framework built to keep the numbers honest.

## Reproducing the milestone

Three commands. No VPN, no database file, no credentials, no network access.

```bash
pip install -r requirements.txt
python scripts/build_honey_model.py          # optional: rebuilds data/honey_model.parquet
jupyter nbconvert --to html --execute Milestone_4.ipynb
```

Python 3.13.5. The notebook reads `data/honey_model.parquet`, which is committed to the
repository — step 2 only verifies that the SQL pipeline still reproduces it.

To re-run the analysis outside the notebook:

```bash
PYTHONPATH=src python scripts/run_evaluation.py         # results/*.csv: baselines, models, framings
PYTHONPATH=src python scripts/run_extremes_analysis.py  # results/*.csv: extremes, threshold sweep
```

## Layout

```
Milestone_4.ipynb          the deliverable; every figure and number regenerates on run
data/                      committed Parquet snapshots + provenance (data/README.md)
sql/01..06_*.sql           the pipeline, in order, extracted from notebooks
scripts/                   build the table, refresh the snapshot, run the analyses
src/honeymodel/            data / features / evaluation / models -- importable and testable
results/                   every table the notebook renders, written by the scripts
David_Work/  Stephanie_Work/  Joshua_Work/     per-member working notebooks
Milestone_3_Figures/       archive of the previous milestone's exported figures
```

Read [`data/README.md`](data/README.md) before using the data: it documents what the
publishers filtered out before release, a unit inconsistency in their event-distance
columns, and which columns must never enter a feature matrix.

## Optional: the team's remote DuckDB

Only `scripts/pull_source_snapshot.py` uses this, and only to refresh the committed
snapshot. Nothing needed to reproduce the milestone touches it.

1. Contact @CmdrJorgs for a Tailscale invite, then
   [install Tailscale](https://tailscale.com/download/) and join the team network using
   the URL @CmdrJorgs provides.
2. Copy the `.env` file provided by @CmdrJorgs into the repository root. It is gitignored
   and must stay that way.
3. `pip install duckdb -U` if your environment predates DuckDB 1.5 (default Anaconda
   installs usually do).

The remote server also holds the minute- and hourly-grain sensor tables (52M rows), which
the daily modelling pipeline does not use.
