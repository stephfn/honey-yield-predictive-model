-- 02_clean.sql
-- Typing and null normalisation. Converts source placeholders such as 'NA' to SQL NULL
-- and gives every column a project name.
--
-- Carried forward beyond the Milestone 3 version:
--   * outlier_lim / no_jump / weight_kg_noOutlier -- the data publishers' own quality
--     flags. Needed to document what was removed upstream (see data/README.md).
--   * the beekeeper-event distance columns (*_last_dif / *_next_dif, in days), so the
--     extreme-event investigation can run at the modelling grain instead of against a
--     separate raw-grain table.
--   * lat / lon, for the weather station assignment.

CREATE OR REPLACE TABLE honey_daily_clean AS
SELECT
    TRY_CAST(time AS TIMESTAMP)                             AS measurement_time,

    TRY_CAST(NULLIF(CAST(weight_kg AS VARCHAR), 'NA') AS DOUBLE)    AS weight_kg,
    TRY_CAST(NULLIF(CAST(weight_delta AS VARCHAR), 'NA') AS DOUBLE) AS source_weight_delta_kg,

    TRY_CAST(NULLIF(CAST(t_i_1 AS VARCHAR), 'NA') AS DOUBLE) AS internal_temp_1,
    TRY_CAST(NULLIF(CAST(t_i_2 AS VARCHAR), 'NA') AS DOUBLE) AS internal_temp_2,
    TRY_CAST(NULLIF(CAST(t_i_3 AS VARCHAR), 'NA') AS DOUBLE) AS internal_temp_3,
    TRY_CAST(NULLIF(CAST(t_i_4 AS VARCHAR), 'NA') AS DOUBLE) AS internal_temp_4,
    TRY_CAST(NULLIF(CAST(t_o   AS VARCHAR), 'NA') AS DOUBLE) AS outside_temp,

    TRY_CAST(NULLIF(CAST(h AS VARCHAR), 'NA') AS DOUBLE)     AS humidity,
    TRY_CAST(NULLIF(CAST(p AS VARCHAR), 'NA') AS DOUBLE)     AS pressure,

    TRY_CAST(NULLIF(CAST(lat AS VARCHAR), 'NA') AS DOUBLE)   AS latitude,
    TRY_CAST(NULLIF(CAST(lon AS VARCHAR), 'NA') AS DOUBLE)   AS longitude,

    -- Publisher quality flags (documentation of upstream preprocessing, not features).
    outlier_lim                                              AS source_outlier_flag,
    no_jump                                                  AS source_no_jump_flag,
    TRY_CAST(NULLIF(CAST(weight_kg_noOutlier AS VARCHAR), 'NA') AS DOUBLE)
                                                             AS source_weight_no_outlier_kg,

    -- Beekeeper-event distances, in days, relative to this observation.
    TRY_CAST(NULLIF(CAST(honey_last_dif      AS VARCHAR), 'NA') AS DOUBLE) AS honey_last_dif,
    TRY_CAST(NULLIF(CAST(honey_next_dif      AS VARCHAR), 'NA') AS DOUBLE) AS honey_next_dif,
    TRY_CAST(NULLIF(CAST(feeding_last_dif    AS VARCHAR), 'NA') AS DOUBLE) AS feeding_last_dif,
    TRY_CAST(NULLIF(CAST(feeding_next_dif    AS VARCHAR), 'NA') AS DOUBLE) AS feeding_next_dif,
    TRY_CAST(NULLIF(CAST(swarming_last_dif   AS VARCHAR), 'NA') AS DOUBLE) AS swarming_last_dif,
    TRY_CAST(NULLIF(CAST(swarming_next_dif   AS VARCHAR), 'NA') AS DOUBLE) AS swarming_next_dif,
    TRY_CAST(NULLIF(CAST(treatment_last_dif  AS VARCHAR), 'NA') AS DOUBLE) AS treatment_last_dif,
    TRY_CAST(NULLIF(CAST(treatment_next_dif  AS VARCHAR), 'NA') AS DOUBLE) AS treatment_next_dif,
    TRY_CAST(NULLIF(CAST(died_last_dif       AS VARCHAR), 'NA') AS DOUBLE) AS died_last_dif,
    TRY_CAST(NULLIF(CAST(died_next_dif       AS VARCHAR), 'NA') AS DOUBLE) AS died_next_dif,
    TRY_CAST(NULLIF(CAST(queencell_last_dif  AS VARCHAR), 'NA') AS DOUBLE) AS queencell_last_dif,
    TRY_CAST(NULLIF(CAST(queencell_next_dif  AS VARCHAR), 'NA') AS DOUBLE) AS queencell_next_dif,

    key                                                      AS source_hive_key,
    source_file                                              AS source_filename

FROM honey_daily_raw;
