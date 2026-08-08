-- 05_honey_model.sql
-- Final modelling table: one row per hive-day that has a true consecutive-day
-- predecessor and successor.
--
-- The row filter is byte-for-byte the Milestone 3 filter, so this table reproduces the
-- published 26,215 x 78-hive dataset exactly (asserted in scripts/build_honey_model.py).
-- Everything added here is a FLAG, never a filter -- consistent with the project's
-- "retain and flag" convention, and so that the effect of each quality rule can be
-- measured rather than assumed.
--
-- Flags added in Milestone 4:
--   implausible_weight_flag      any of previous/current/next weight below the physical
--                                floor. The Milestone 3 guard was only `> 0`, which
--                                admits readings such as 0.0005 kg against a 34.3 kg mean
--                                and manufactures 40 kg "extreme events".
--   robust_outlier_flag          reading more than k MAD from the hive's own centred
--                                7-day rolling median (hive-relative, so a nucleus colony
--                                and a production hive are judged on their own scale).
--   sensor_dropout_flag          a large drop followed by a comparable recovery, i.e. the
--                                signature of a dropout or re-tare rather than real mass.
--   extreme_weight_change_flag_{3,5,7,10}   threshold sensitivity sweep.
--   extreme_relative_flag        |target| beyond k_rel MAD of that hive's own target
--                                distribution (absolute-threshold-free variant).
--   nearest_*_event_days         days to the closest logged beekeeper event of each type,
--                                so the extreme-event investigation runs at THIS grain.
--
-- Substituted by the build script: ${WEIGHT_FLOOR_KG}, ${MAD_K}, ${RELATIVE_MAD_K}.

CREATE OR REPLACE TABLE honey_model AS

WITH filtered AS (
    SELECT *
    FROM honey_feature_candidates
    WHERE days_since_previous_observation = 1
      AND days_to_next_observation = 1
      AND end_of_day_weight_kg IS NOT NULL
      AND previous_day_weight_kg IS NOT NULL
      AND next_day_weight_kg IS NOT NULL
      AND end_of_day_weight_kg > 0
      AND previous_day_weight_kg > 0
      AND next_day_weight_kg > 0
      AND target_next_day_weight_change_kg IS NOT NULL
),

hive_target_median AS (
    SELECT hive_id, MEDIAN(target_next_day_weight_change_kg) AS hive_target_median_kg
    FROM filtered
    GROUP BY hive_id
),

target_scale AS (
    -- Hive-relative scale of the target itself, for the threshold-free extreme variant.
    SELECT
        f.hive_id,
        MEDIAN(ABS(f.target_next_day_weight_change_kg - m.hive_target_median_kg))
            AS hive_target_mad_kg
    FROM filtered AS f
    JOIN hive_target_median AS m USING (hive_id)
    GROUP BY f.hive_id
)

SELECT
    f.*,

    -- ---- Milestone 3 label, unchanged -------------------------------------------
    ABS(f.target_next_day_weight_change_kg) > 5 AS extreme_weight_change_flag,

    -- ---- Threshold sensitivity sweep (C3) ---------------------------------------
    ABS(f.target_next_day_weight_change_kg) > 3  AS extreme_weight_change_flag_3,
    ABS(f.target_next_day_weight_change_kg) > 5  AS extreme_weight_change_flag_5,
    ABS(f.target_next_day_weight_change_kg) > 7  AS extreme_weight_change_flag_7,
    ABS(f.target_next_day_weight_change_kg) > 10 AS extreme_weight_change_flag_10,

    t.hive_target_mad_kg,
    CASE
        WHEN t.hive_target_mad_kg IS NULL OR t.hive_target_mad_kg = 0 THEN FALSE
        ELSE ABS(f.target_next_day_weight_change_kg)
             > ${RELATIVE_MAD_K} * 1.4826 * t.hive_target_mad_kg
    END AS extreme_relative_flag,

    -- ---- Physical plausibility (C1) ---------------------------------------------
    (f.end_of_day_weight_kg   < ${WEIGHT_FLOOR_KG}
     OR f.previous_day_weight_kg < ${WEIGHT_FLOOR_KG}
     OR f.next_day_weight_kg     < ${WEIGHT_FLOOR_KG})        AS implausible_weight_flag,

    CASE
        WHEN f.hive_change_mad_kg IS NULL OR f.hive_change_mad_kg = 0 THEN FALSE
        ELSE ABS(f.end_of_day_weight_kg - f.local_median_weight_kg)
             > ${MAD_K} * 1.4826 * f.hive_change_mad_kg
    END                                                        AS robust_outlier_flag,

    -- Drop then recovery of comparable size within one day: dropout / re-tare, not mass.
    (f.previous_day_weight_change_kg < -${WEIGHT_FLOOR_KG}
     AND f.target_next_day_weight_change_kg > ${WEIGHT_FLOOR_KG})
    OR (f.previous_day_weight_change_kg > ${WEIGHT_FLOOR_KG}
        AND f.target_next_day_weight_change_kg < -${WEIGHT_FLOOR_KG})
                                                               AS sensor_dropout_flag,

    -- ---- Beekeeper-event proximity at the modelling grain (C2) -------------------
    LEAST(ABS(f.honey_last_dif),     ABS(f.honey_next_dif))     AS nearest_honey_event_days,
    LEAST(ABS(f.feeding_last_dif),   ABS(f.feeding_next_dif))   AS nearest_feeding_event_days,
    LEAST(ABS(f.swarming_last_dif),  ABS(f.swarming_next_dif))  AS nearest_swarming_event_days,
    LEAST(ABS(f.treatment_last_dif), ABS(f.treatment_next_dif)) AS nearest_treatment_event_days,
    LEAST(ABS(f.died_last_dif),      ABS(f.died_next_dif))      AS nearest_died_event_days,
    LEAST(ABS(f.queencell_last_dif), ABS(f.queencell_next_dif)) AS nearest_queencell_event_days

FROM filtered AS f
LEFT JOIN target_scale AS t USING (hive_id);
