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