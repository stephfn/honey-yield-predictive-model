-- 03_daily_summary.sql
-- One standardised observation per hive per calendar day, with the hive identifier
-- derived from the source filename.
--
-- NOTE ON GRAIN (corrects the Milestone 3 description): this step does NOT aggregate.
-- The source daily files are already one row per hive per day -- raw, clean and summary
-- row counts are all 29,172 -- so the `avg_*` prefixes are inherited column names, not
-- the result of a GROUP BY performed here. The names are kept for continuity with the
-- Milestone 3 schema and downstream code; the aggregation that produced them happened
-- upstream, in the published dataset. A uniqueness assertion on (hive_id,
-- measurement_date) runs in scripts/build_honey_model.py before the window functions,
-- so the one-row-per-hive-day assumption is checked rather than assumed.

CREATE OR REPLACE TABLE honey_daily_summary AS
SELECT
    TRY_CAST(
        regexp_extract(source_filename, '([^/\\]+)\.csv$', 1) AS INTEGER
    )                                   AS hive_id,

    CAST(measurement_time AS DATE)      AS measurement_date,

    weight_kg                           AS end_of_day_weight_kg,

    internal_temp_1                     AS avg_internal_temp_1,
    internal_temp_2                     AS avg_internal_temp_2,
    internal_temp_3                     AS avg_internal_temp_3,
    internal_temp_4                     AS avg_internal_temp_4,
    outside_temp                        AS avg_outside_temp,
    humidity                            AS avg_humidity,
    pressure                            AS avg_pressure,

    latitude,
    longitude,

    source_outlier_flag,
    source_no_jump_flag,
    source_weight_no_outlier_kg,

    honey_last_dif,     honey_next_dif,
    feeding_last_dif,   feeding_next_dif,
    swarming_last_dif,  swarming_next_dif,
    treatment_last_dif, treatment_next_dif,
    died_last_dif,      died_next_dif,
    queencell_last_dif, queencell_next_dif,

    source_filename

FROM honey_daily_clean;
