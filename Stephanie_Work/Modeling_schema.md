# Honey Yield Modeling Schema

## Purpose

This document describes the final modeling dataset created by the DuckDB pipeline for predicting next-day hive weight change. It documents the data lineage, engineered features, modeling eligibility rules, evaluation strategy, and reproducibility considerations so the workflow can be reproduced and integrated with downstream project notebooks.

---

## Source Data

The modeling pipeline reads from the shared DuckDB table:

`bob_sensor_processed`

The source data are filtered to retain only daily hive observations from the multi-year dataset:

- `interval = 'd'`
- `dataset = 'years'`

The filtered dataset is materialized locally as:

`bob_sensor_source`

---

## Final Modeling Table

The final modeling dataset is stored as:

`honey_model`

The table is created by `Honey_Project_Pipeline.ipynb` through a sequence of standardized transformation tables.

---

## Dataset Scope

| Measure | Value |
|---------|------:|
| Daily observations after cleaning | 29,155 |
| Unique hives | 78 |
| Final modeling observations | 26,215 |
| Modeling start date | 2019-06-25 |
| Modeling end date | 2022-12-30 |
| Extreme next-day events (>5 kg) | 445 |
| Extreme-event prevalence | 1.70% |

---

## Observation Level

Each row represents a single hive on a single calendar day.

The expected row identifier is:

- `hive_id`
- `measurement_date`

---

## Target Variable

The prediction target is:

`target_next_day_weight_change_kg`

which is calculated as:

`next_day_weight_kg - end_of_day_weight_kg`

This is treated as a continuous regression target.

---

## Feature Engineering

Historical predictor variables are generated using SQL window functions partitioned by `hive_id` and ordered by `measurement_date`. These engineered features preserve chronological ordering while capturing recent hive behavior.

### Historical Weight Features

- `previous_day_weight_kg`
- `next_day_weight_kg`
- `previous_day_weight_change_kg`
- `rolling_3_day_weight_kg`
- `rolling_7_day_weight_kg`

These variables are created using SQL `LAG()`, `LEAD()`, and rolling `AVG()` window functions.

### Environmental Features

- Internal hive temperatures (Sensors 1–5)
- Outside temperature
- Humidity
- Atmospheric pressure
- Latitude
- Longitude

### Temporal Features

- `year`
- `month`
- `day_of_year`

### Extreme-Event Feature

- `extreme_weight_change_flag`

This Boolean indicator identifies observations where:

`ABS(target_next_day_weight_change_kg) > 5`

---

## Modeling Eligibility Rules

The final modeling dataset retains only observations that satisfy the following requirements:

- Previous observation occurred exactly one day earlier.
- Next observation occurred exactly one day later.
- Current hive weight is available.
- Previous hive weight is available.
- Next hive weight is available.
- Target next-day weight change is available.
- Hive weights are positive.

These filters ensure that lag variables, rolling features, and prediction targets are computed from consecutive daily observations.

---

## Pipeline Logic

The modeling dataset is constructed through the following sequence:

1. Connect to the shared Quack data source.
2. Query the `bob_sensor_processed` table.
3. Filter to daily observations from the multi-year dataset.
4. Materialize the filtered data as `bob_sensor_source`.
5. Create the raw anchor table (`honey_daily_raw`).
6. Standardize data types and clean measurements (`honey_daily_clean`).
7. Aggregate to one observation per hive per day (`honey_daily_summary`).
8. Generate lag, lead, rolling-window, temporal, and target variables (`honey_feature_candidates`).
9. Apply modeling eligibility filters and create the final modeling table (`honey_model`).

---

## Baseline Models Evaluated

The baseline modeling workflow includes:

- Random Forest trained using all eligible observations.
- Random Forest trained using routine observations only (excluding extreme target events).
- Gradient Boosting Regressor trained using routine observations only.

The prediction target for all baseline regression models is:

`target_next_day_weight_change_kg`

The daily pipeline is retained as the stable, reproducible baseline while alternative weekly-grain modeling approaches are evaluated separately.

---

## Evaluation Design

Baseline models are evaluated using a chronological train/test split.

Current evaluation metrics include:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R²)

Future work includes rolling-origin validation and expanded weather feature evaluation.

---

## Reproducibility Notes

The modeling pipeline was refactored to support reproducible execution across development environments.

The notebook:

- Uses repository-relative paths.
- Loads credentials from an untracked `.env` file.
- Connects to the shared Quack source.
- Rebuilds intermediate DuckDB tables using `CREATE OR REPLACE TABLE`.
- Validates row counts, hive counts, duplicate hive-date combinations, and date ranges after each major transformation.
- Recreates `honey_model` directly from the shared source without relying on archived local datasets.

Downstream notebooks should query `honey_model` rather than reproduce intermediate transformations.