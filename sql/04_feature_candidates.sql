-- 04_feature_candidates.sql
-- Per-hive history features. Every window is PARTITION BY hive_id, so no value ever
-- crosses a hive boundary.
--
-- Added beyond Milestone 3:
--   * hive-local robust statistics (median and MAD of end-of-day weight, plus a
--     centred 7-day rolling median) used by 05_honey_model.sql for the physical
--     plausibility flags;
--   * differenced / normalised weight features, because the target is a delta while
--     every Milestone 3 feature was an absolute level;
--   * cyclical calendar encodings.
--
-- Columns whose value is only known after the prediction date -- next_day_weight_kg,
-- next_observation_date, and every *_next_dif -- are carried here for analysis and
-- labelling only. They are listed in honeymodel.features.FORBIDDEN_COLUMNS and are
-- rejected if they reach a feature matrix.

CREATE OR REPLACE TABLE honey_feature_candidates AS

WITH weight_windows AS (
    SELECT
        hive_id,
        measurement_date,
        end_of_day_weight_kg,

        LAG(measurement_date) OVER hive_time  AS previous_observation_date,
        LEAD(measurement_date) OVER hive_time AS next_observation_date,

        LAG(end_of_day_weight_kg) OVER hive_time  AS previous_day_weight_kg,
        LEAD(end_of_day_weight_kg) OVER hive_time AS next_day_weight_kg,

        LAG(end_of_day_weight_kg, 2) OVER hive_time AS weight_lag_2_kg,
        LAG(end_of_day_weight_kg, 7) OVER hive_time AS weight_lag_7_kg,

        AVG(end_of_day_weight_kg) OVER (
            PARTITION BY hive_id ORDER BY measurement_date
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ) AS rolling_3_day_weight_kg,

        AVG(end_of_day_weight_kg) OVER (
            PARTITION BY hive_id ORDER BY measurement_date
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) AS rolling_7_day_weight_kg,

        STDDEV_SAMP(end_of_day_weight_kg) OVER (
            PARTITION BY hive_id ORDER BY measurement_date
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) AS rolling_7_day_weight_sd_kg,

        -- Hive-local robust statistics for the plausibility screen. The centred window
        -- is used only to flag implausible readings, never as a model feature.
        MEDIAN(end_of_day_weight_kg) OVER (
            PARTITION BY hive_id ORDER BY measurement_date
            ROWS BETWEEN 3 PRECEDING AND 3 FOLLOWING
        ) AS local_median_weight_kg,

        MEDIAN(end_of_day_weight_kg) OVER (PARTITION BY hive_id) AS hive_median_weight_kg

    FROM honey_daily_summary
    WINDOW hive_time AS (PARTITION BY hive_id ORDER BY measurement_date)
),

hive_change_median AS (
    SELECT
        hive_id,
        MEDIAN(end_of_day_weight_kg - previous_day_weight_kg) AS hive_change_median_kg
    FROM weight_windows
    GROUP BY hive_id
),

hive_scale AS (
    -- Two robust scales per hive:
    --   hive_mad_weight_kg  spread of the weight LEVEL. Large by construction (a colony
    --                       gains and loses tens of kg over a season), so it is reported
    --                       for context but is too loose to screen single readings.
    --   hive_change_mad_kg  spread of the DAY-TO-DAY CHANGE. This is the hive's own
    --                       sensor-and-behaviour noise scale, and it is what the
    --                       plausibility screen in 05_honey_model.sql compares against.
    SELECT
        w.hive_id,
        MEDIAN(ABS(w.end_of_day_weight_kg - w.hive_median_weight_kg)) AS hive_mad_weight_kg,
        MEDIAN(ABS((w.end_of_day_weight_kg - w.previous_day_weight_kg) - c.hive_change_median_kg))
            AS hive_change_mad_kg
    FROM weight_windows AS w
    JOIN hive_change_median AS c USING (hive_id)
    GROUP BY w.hive_id
),

engineered_weights AS (
    SELECT
        w.*,
        s.hive_mad_weight_kg,
        s.hive_change_mad_kg,

        DATE_DIFF('day', w.previous_observation_date, w.measurement_date) AS days_since_previous_observation,
        DATE_DIFF('day', w.measurement_date, w.next_observation_date)     AS days_to_next_observation,

        w.end_of_day_weight_kg - w.previous_day_weight_kg  AS previous_day_weight_change_kg,
        w.next_day_weight_kg  - w.end_of_day_weight_kg     AS target_next_day_weight_change_kg,

        -- Differenced / normalised features (the target is a delta; these are too).
        w.end_of_day_weight_kg - w.rolling_7_day_weight_kg AS weight_minus_rolling_7_kg,
        w.end_of_day_weight_kg - w.rolling_3_day_weight_kg AS weight_minus_rolling_3_kg,
        (w.end_of_day_weight_kg - w.weight_lag_2_kg) / 2.0 AS weight_slope_2_day_kg,
        (w.end_of_day_weight_kg - w.weight_lag_7_kg) / 7.0 AS weight_slope_7_day_kg,
        w.end_of_day_weight_kg - w.hive_median_weight_kg   AS weight_minus_hive_median_kg

    FROM weight_windows AS w
    JOIN hive_scale AS s USING (hive_id)
)

SELECT
    e.hive_id,
    e.measurement_date,
    e.end_of_day_weight_kg,
    e.previous_observation_date,
    e.next_observation_date,
    e.previous_day_weight_kg,
    e.next_day_weight_kg,
    e.weight_lag_2_kg,
    e.weight_lag_7_kg,
    e.rolling_3_day_weight_kg,
    e.rolling_7_day_weight_kg,
    e.rolling_7_day_weight_sd_kg,
    e.days_since_previous_observation,
    e.days_to_next_observation,
    e.previous_day_weight_change_kg,
    e.target_next_day_weight_change_kg,
    e.weight_minus_rolling_7_kg,
    e.weight_minus_rolling_3_kg,
    e.weight_slope_2_day_kg,
    e.weight_slope_7_day_kg,
    e.weight_minus_hive_median_kg,
    e.local_median_weight_kg,
    e.hive_median_weight_kg,
    e.hive_mad_weight_kg,
    e.hive_change_mad_kg,

    s.avg_internal_temp_1,
    s.avg_internal_temp_2,
    s.avg_internal_temp_3,
    s.avg_internal_temp_4,
    s.avg_outside_temp,
    s.avg_humidity,
    s.avg_pressure,
    s.latitude,
    s.longitude,

    s.source_outlier_flag,
    s.source_no_jump_flag,
    s.source_weight_no_outlier_kg,

    s.honey_last_dif,     s.honey_next_dif,
    s.feeding_last_dif,   s.feeding_next_dif,
    s.swarming_last_dif,  s.swarming_next_dif,
    s.treatment_last_dif, s.treatment_next_dif,
    s.died_last_dif,      s.died_next_dif,
    s.queencell_last_dif, s.queencell_next_dif,

    s.source_filename,

    EXTRACT(YEAR  FROM e.measurement_date) AS year,
    EXTRACT(MONTH FROM e.measurement_date) AS month,
    EXTRACT(DOY   FROM e.measurement_date) AS day_of_year,
    SIN(2 * PI() * EXTRACT(DOY FROM e.measurement_date) / 365.25) AS sin_day_of_year,
    COS(2 * PI() * EXTRACT(DOY FROM e.measurement_date) / 365.25) AS cos_day_of_year

FROM engineered_weights AS e
LEFT JOIN honey_daily_summary AS s
    ON e.hive_id = s.hive_id
   AND e.measurement_date = s.measurement_date;
