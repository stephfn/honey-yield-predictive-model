-- 01_raw_anchor.sql
-- Raw anchor table. Preserves the source daily measurements with no modification
-- beyond column selection. Never overwritten by downstream steps.
--
-- Source: data/honey_daily_source.parquet -- a committed snapshot of the daily-grain
-- (interval = 'd') rows of the Bee Observer sensor archive. See data/README.md for
-- provenance and for the script that regenerates it.
--
-- The original Milestone 3 pipeline read this same data from a local `Data/Daily_Only/`
-- CSV tree with read_csv_auto(all_varchar = TRUE). The Parquet snapshot carries the
-- already-typed columns, so the TRY_CAST/NULLIF step in 02_clean.sql is now defensive
-- rather than load-bearing. Row-for-row parity with the CSV path is asserted by
-- scripts/build_honey_model.py.
--
-- ${SOURCE} is substituted by the build script.

CREATE OR REPLACE TABLE honey_daily_raw AS
SELECT
    dataset,
    category,
    source_file,
    key,
    time,
    weight_kg,
    weight_delta,
    weight_kg_noOutlier,
    weight_delta_noOutlier,
    outlier_lim,
    no_jump,
    t_i_1, t_i_2, t_i_3, t_i_4, t_i_5,
    t_o, h, t, p,
    lat, lon,
    honey_last_dif,      honey_next_dif,
    feeding_last_dif,    feeding_next_dif,
    swarming_last_dif,   swarming_next_dif,
    treatment_last_dif,  treatment_next_dif,
    died_last_dif,       died_next_dif,
    queencell_last_dif,  queencell_next_dif
FROM read_parquet('${SOURCE}')
WHERE dataset = 'years';   -- see data/README.md, "Which daily files are in scope"
