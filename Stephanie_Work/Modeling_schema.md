# Honey Yield Modeling Schema

## Purpose

This document describes the modeling dataset created by the DuckDB pipeline for predicting next-day hive weight change. It summarizes the table structure, target variable, engineered features, filtering rules, and modeling workflow so that the analysis can be reproduced or integrated with other project notebooks.

## Table Structure

The full schema for `honey_model` is generated automatically in the modeling notebook (Appendix A).

This document summarizes the purpose of the table and the major engineered features used by the predictive models.

## Source Table

The modeling notebook reads from the DuckDB table:

`honey_model`

The table is produced upstream by `Honey_Project_Pipeline.ipynb`.

## Observation Level

Each row represents one daily observation for one hive.

The expected row identifier is the combination of:

- `hive_id`
- `measurement_date`

## Target Variable

| Column | Description |
|---|---|
| `target_next_day_weight_change_kg` | Next-day hive weight minus the current end-of-day hive weight, measured in kilograms. This is the continuous regression target. |

## Identifier and Date Columns

| Column | Description |
|---|---|
| `hive_id` | Unique hive identifier. |
| `measurement_date` | Date of the current daily hive observation. |
| `previous_observation_date` | Date of the preceding observation for the same hive. |
| `next_observation_date` | Date of the following observation for the same hive. |
| `days_since_previous_observation` | Number of days between the current and previous observations. |
| `days_to_next_observation` | Number of days between the current and next observations. |

## Historical Weight Features

| Column | Description |
|---|---|
| `end_of_day_weight_kg` | Current hive weight at the end of the observation day. |
| `previous_day_weight_kg` | Previous observed end-of-day hive weight, created with `LAG()`. |
| `next_day_weight_kg` | Following observed end-of-day hive weight, created with `LEAD()`. |
| `previous_day_weight_change_kg` | Current weight minus previous-day weight. |
| `rolling_3_day_weight_kg` | Rolling mean of current and two preceding daily weights. |
| `rolling_7_day_weight_kg` | Rolling mean of current and six preceding daily weights. |

## Environmental Features

| Column | Description |
|---|---|
| `avg_internal_temp_1` | Daily average from internal hive temperature sensor 1. |
| `avg_internal_temp_2` | Daily average from internal hive temperature sensor 2. |
| `avg_internal_temp_3` | Daily average from internal hive temperature sensor 3. |
| `avg_internal_temp_4` | Daily average from internal hive temperature sensor 4. |
| `avg_outside_temp` | Daily average outside temperature available in the source data. |
| `avg_humidity` | Daily average humidity. |
| `avg_pressure` | Daily average atmospheric pressure. |

## Temporal Features

| Column | Description |
|---|---|
| `year` | Calendar year of the observation. |
| `month` | Calendar month of the observation. |
| `day_of_year` | Day number within the calendar year. |

## Extreme Event Indicator

| Column | Description |
|---|---|
| `extreme_weight_change_flag` | Boolean indicator identifying observations where the absolute next-day weight change exceeds 5 kg. |

Routine observations are defined as:

`ABS(target_next_day_weight_change_kg) <= 5`

## Pipeline Logic

1. Read and standardize daily hive measurement files.
2. Aggregate measurements to one row per hive and date.
3. Order observations by `hive_id` and `measurement_date`.
4. Generate previous and next observations with SQL `LAG()` and `LEAD()`.
5. Calculate the previous-day weight change and create the target_next_day_weight_change_kg prediction target..
6. Generate 3-day and 7-day rolling weight averages.
7. Create calendar features.
8. Retain observations with consecutive daily coverage where required.
9. Flag next-day changes greater than 5 kg in absolute value.
10. Store the final feature table as `honey_model`.

## Preliminary Modeling Feature Set

The history-only models currently use:

- `previous_day_weight_kg`
- `previous_day_weight_change_kg`
- `rolling_3_day_weight_kg`
- `rolling_7_day_weight_kg`
- `day_of_year`

The prediction target is:

- `target_next_day_weight_change_kg`

## Evaluation Design

Models use an 80/20 chronological train/test split. Earlier observations are used for training and later observations are reserved for testing.

Current evaluation metrics include:

- Mean Absolute Error
- Root Mean Squared Error
- R²

## Current Models

- Baseline Random Forest using all eligible observations
- Routine-only Random Forest excluding extreme target events
- Routine-only Gradient Boosting model

## Important Integration Note

The local modeling database was refactored around the next-day weight-change target and contains derived lag, rolling-window, temporal, and extreme-event fields. Its table structure may therefore differ from the remote DuckDB server or earlier project tables. Downstream notebooks should query `honey_model` or reproduce the transformations described above rather than assume identical schemas.


The shared source is accessed through the Quack connection and filtered to the project’s daily, multi-year observations:

WHERE interval = 'd'
  AND dataset = 'years'

The filtered source is materialized into the local DuckDB database so downstream transformations can run against stable project tables.

## Final Modeling Table

The final modeling table is:

honey_model

The table is created upstream by Honey_Project_Pipeline.ipynb and consumed by the modeling and evaluation notebooks.

## Dataset Scope

The current reproducible pipeline produces:

Measure	Value
Source observations after required-field cleaning	29,155
Hives	78
Final modeling observations	26,215
Modeling start date	2019-06-25
Modeling end date	2022-12-30
Extreme next-day events	445
Extreme-event prevalence	1.70%

The reduction from 29,172 filtered source records to 29,155 cleaned records results from excluding observations with missing measurement timestamps or hive weights.

## Observation Level

Each row in honey_model represents one eligible daily observation for one hive.

The logical row identifier is:

hive_id
measurement_date

The final table contains no duplicate hive-date combinations.

Only records with valid consecutive previous-day and next-day observations are retained for modeling.

## Target Variable
Column	Type	Description
target_next_day_weight_change_kg	DOUBLE	Next-day end-of-day hive weight minus the current end-of-day hive weight, measured in kilograms. This is the continuous regression target.

The target is calculated as:

next_day_weight_kg - end_of_day_weight_kg

A positive value indicates a net increase in hive weight on the following day. A negative value indicates a net decrease.

## Identifier and Observation-Date Columns
Column	Type	Description
hive_id	INTEGER	Numeric hive identifier derived from the source filename.
measurement_date	DATE	Date of the current daily hive observation.
previous_observation_date	DATE	Date of the preceding observation for the same hive.
next_observation_date	DATE	Date of the following observation for the same hive.
days_since_previous_observation	BIGINT	Number of calendar days between the current and previous observations.
days_to_next_observation	BIGINT	Number of calendar days between the current and following observations.

## Historical Weight Features
Column	Type	Description
end_of_day_weight_kg	DOUBLE	Current end-of-day hive weight.
previous_day_weight_kg	DOUBLE	Previous observed end-of-day hive weight, created with LAG().
next_day_weight_kg	DOUBLE	Following observed end-of-day hive weight, created with LEAD().
previous_day_weight_change_kg	DOUBLE	Current end-of-day weight minus the previous-day weight.
rolling_3_day_weight_kg	DOUBLE	Rolling mean of the current and two preceding daily hive weights.
rolling_7_day_weight_kg	DOUBLE	Rolling mean of the current and six preceding daily hive weights.

All lag, lead, and rolling-window calculations are partitioned by hive_id and ordered by measurement_date.

## Environmental Features
Column	Type	Description
avg_internal_temp_1	DOUBLE	Daily internal hive temperature from sensor 1.
avg_internal_temp_2	DOUBLE	Daily internal hive temperature from sensor 2.
avg_internal_temp_3	DOUBLE	Daily internal hive temperature from sensor 3.
avg_internal_temp_4	DOUBLE	Daily internal hive temperature from sensor 4.
avg_internal_temp_5	DOUBLE	Daily internal hive temperature from sensor 5.
avg_outside_temp	DOUBLE	Daily outside temperature included in the source dataset.
avg_humidity	DOUBLE	Daily relative humidity measurement.
avg_pressure	DOUBLE	Daily atmospheric pressure measurement.
latitude	DOUBLE	Latitude associated with the hive or monitoring site.
longitude	DOUBLE	Longitude associated with the hive or monitoring site.

The source data are already recorded at daily grain, so the avg_ prefixes preserve compatibility with the original modeling schema rather than indicating that multiple intraday rows were averaged in the revised pipeline.

## Source Metadata
Column	Type	Description
source_filename	VARCHAR	Original source filename associated with the hive observation.

The hive identifier is extracted from the numeric portion of filenames such as 11.csv.

## Temporal Features
Column	Type	Description
year	BIGINT	Calendar year of the observation.
month	BIGINT	Calendar month of the observation.
day_of_year	BIGINT	Sequential day number within the calendar year.

## Extreme-Event Indicator
Column	Type	Description
extreme_weight_change_flag	BOOLEAN	Indicates whether the absolute next-day hive-weight change exceeds 5 kilograms.

The flag is defined as:

ABS(target_next_day_weight_change_kg) > 5

Routine observations are defined as:

ABS(target_next_day_weight_change_kg) <= 5

Extreme observations remain in honey_model. They are flagged rather than deleted so the project can compare:

models trained on all eligible observations
routine-only regression models
separate extreme-event classification or anomaly-detection approaches

## Modeling-Table Eligibility Rules

Rows are retained in honey_model only when all of the following conditions are met:

days_since_previous_observation = 1
AND days_to_next_observation = 1
AND end_of_day_weight_kg IS NOT NULL
AND previous_day_weight_kg IS NOT NULL
AND next_day_weight_kg IS NOT NULL
AND end_of_day_weight_kg > 0
AND previous_day_weight_kg > 0
AND next_day_weight_kg > 0
AND target_next_day_weight_change_kg IS NOT NULL

These rules ensure that the regression target represents a true next-calendar-day change and that all required weight values are valid.

## Pipeline Logic
Connect to the shared Quack data source.
Query bob_sensor_processed.
Retain records where interval = 'd' and dataset = 'years'.
Materialize the filtered source as bob_sensor_source.
Create honey_daily_raw as the minimally modified raw anchor.
Standardize column names and retain valid timestamp and weight observations in honey_daily_clean.
Derive the numeric hive identifier from source_filename.
Create one validated observation per hive and calendar date in honey_daily_summary.
Partition records by hive and order them by measurement date.
Generate previous and next observations with SQL LAG() and LEAD().
Calculate previous-day weight change and the next-day regression target.
Generate 3-day and 7-day rolling weight features.
Add calendar features.
Retain observations with consecutive previous-day and next-day coverage.
Flag next-day changes exceeding 5 kilograms in absolute value.
Store the final modeling dataset as honey_model.

## Preliminary Modeling Feature Set

The history-only regression models use:

previous_day_weight_kg
previous_day_weight_change_kg
rolling_3_day_weight_kg
rolling_7_day_weight_kg
day_of_year

The prediction target is:

target_next_day_weight_change_kg

Expanded model configurations may incorporate:

internal hive temperatures
outside temperature
humidity
pressure
geographic location
weather-derived features
hive-management or beekeeper-event variables

## Evaluation Design

The daily regression models use an 80/20 chronological train-test split.

Earlier observations are used for model training, while later observations are reserved for testing. This preserves temporal order and prevents future observations from influencing model development.

Current evaluation metrics include:

Mean Absolute Error
Root Mean Squared Error
R²

Chronological cross-validation or rolling-origin validation may be added to strengthen the final evaluation.

## Current Daily Models

The current daily modeling workflow includes:

Baseline Random Forest using all eligible observations
Routine-only Random Forest excluding extreme target events
Routine-only Gradient Boosting regression

The daily pipeline is retained as the stable, reproducible baseline while alternative weekly-grain modeling approaches are evaluated separately.

# Reproducibility Notes

The pipeline no longer depends on the original local Data/Daily_Only CSV directory.

All project paths are resolved relative to the repository root. The notebook:

loads credentials from an untracked .env file
connects to the shared Quack source
rebuilds intermediate DuckDB tables with CREATE OR REPLACE TABLE
validates row counts, hive counts, date ranges, and duplicate hive-date records
recreates honey_model from the defined shared source

The .env file and authentication token must never be committed to GitHub.

Downstream notebooks should query honey_model rather than depend on archived DataFrames, manually copied outputs, or older local database tables.