-- ============================================================================
-- Regional weather data in DuckDB
-- Source: NOAA GHCN-Daily, published as Parquet on AWS Open Data (no account
--         needed, anonymous reads). Bucket: s3://noaa-ghcn-pds  (region us-east-1)
-- All queries below were run and verified against the live bucket with
-- DuckDB v1.5.3 on 2026-06-20.
--
-- Why this source and not Open-Meteo:
--   Open-Meteo's AWS bucket (s3://openmeteo) does NOT store Parquet. It stores a
--   custom binary ".om" format (multi-dimensional arrays, like NetCDF/HDF5).
--   DuckDB cannot read .om over httpfs, so there is no "point DuckDB at it"
--   path for Open-Meteo. GHCN-Daily IS real Parquet and is the clean fit.
--   If you specifically need Open-Meteo's gridded MODEL data, pull it via their
--   API/Docker first and materialize to Parquet, then query locally (see bottom).
--
-- Data notes for GHCN-Daily:
--   DATE       is a VARCHAR like '20230601' (YYYYMMDD), NOT an ISO date.
--   DATA_VALUE is an integer in tenths of the unit: TMAX/TMIN in 0.1 deg C,
--              PRCP in 0.1 mm. Divide by 10.0 for real units.
--   ELEMENT    TMAX, TMIN, PRCP, SNOW, SNWD, etc. (used as a Hive partition).
-- ============================================================================

-- One-time setup. httpfs auto-loads on remote reads, so this is belt-and-braces.
INSTALL httpfs;
LOAD httpfs;
SET s3_region = 'us-east-1';

-- ----------------------------------------------------------------------------
-- 1) Single station, single variable, date range.
--    Partitioned by STATION + ELEMENT, so this touches just one tiny file.
--    USW00094728 = New York Central Park.
-- ----------------------------------------------------------------------------
SELECT
    ID,
    strptime(DATE, '%Y%m%d')::DATE AS day,
    DATA_VALUE / 10.0              AS tmax_c
FROM read_parquet(
    's3://noaa-ghcn-pds/parquet/by_station/STATION=USW00094728/ELEMENT=TMAX/*.parquet'
)
WHERE DATE BETWEEN '20230601' AND '20230607'
ORDER BY day;

-- ----------------------------------------------------------------------------
-- 2) Regional query: every station inside a lat/lon bounding box.
--    Station coordinates live in a fixed-width text file (ghcnd-stations.txt);
--    parse lat/lon, filter the box, then join to the year-partitioned Parquet.
--    Box below ~= the New York City metro area. Swap the four numbers for your
--    region and YEAR / ELEMENT / DATE range as needed.
-- ----------------------------------------------------------------------------
WITH stations AS (
    SELECT
        trim(substr(line, 1, 11))                       AS id,
        TRY_CAST(trim(substr(line, 13, 8)) AS DOUBLE)   AS lat,
        TRY_CAST(trim(substr(line, 22, 9)) AS DOUBLE)   AS lon,
        trim(substr(line, 42, 30))                      AS name
    FROM read_csv(
        's3://noaa-ghcn-pds/ghcnd-stations.txt',
        columns = {'line': 'VARCHAR'}, delim = '\x07', header = false
    )
    WHERE TRY_CAST(trim(substr(line, 13, 8)) AS DOUBLE) BETWEEN 40.4 AND 41.0
      AND TRY_CAST(trim(substr(line, 22, 9)) AS DOUBLE) BETWEEN -74.3 AND -73.6
)
SELECT
    s.name,
    strptime(o.DATE, '%Y%m%d')::DATE AS day,
    o.DATA_VALUE / 10.0              AS tmax_c
FROM read_parquet(
        's3://noaa-ghcn-pds/parquet/by_year/YEAR=2023/ELEMENT=TMAX/*.parquet'
     ) o
JOIN stations s ON o.ID = s.id
WHERE o.DATE BETWEEN '20230601' AND '20230603'
ORDER BY day, tmax_c DESC;

-- ----------------------------------------------------------------------------
-- 3) Quick aggregate: regional daily mean high across the box.
-- ----------------------------------------------------------------------------
WITH stations AS (
    SELECT trim(substr(line, 1, 11)) AS id
    FROM read_csv('s3://noaa-ghcn-pds/ghcnd-stations.txt',
                  columns = {'line': 'VARCHAR'}, delim = '\x07', header = false)
    WHERE TRY_CAST(trim(substr(line, 13, 8)) AS DOUBLE) BETWEEN 40.4 AND 41.0
      AND TRY_CAST(trim(substr(line, 22, 9)) AS DOUBLE) BETWEEN -74.3 AND -73.6
)
SELECT
    strptime(o.DATE, '%Y%m%d')::DATE   AS day,
    round(avg(o.DATA_VALUE / 10.0), 1) AS mean_tmax_c,
    count(*)                           AS n_stations
FROM read_parquet('s3://noaa-ghcn-pds/parquet/by_year/YEAR=2023/ELEMENT=TMAX/*.parquet') o
JOIN stations s ON o.ID = s.id
WHERE o.DATE BETWEEN '20230601' AND '20230630'
GROUP BY day
ORDER BY day;

-- ============================================================================
-- HOURLY data, same direct-from-S3 pattern.
-- Source: NOAA Global Hourly (Integrated Surface Database / ISD).
--   Bucket: s3://noaa-global-hourly-pds  (region us-east-1, anonymous reads)
--   Layout: one CSV per station per year -> s3://.../<YEAR>/<STATION_ID>.csv
--   <STATION_ID> is the 11-digit USAF+WBAN id (e.g. JFK = 74486094789).
--   Coverage is sub-hourly in practice (synoptic + special obs), 1901->present.
-- Verified live with DuckDB v1.5.3 on 2026-06-20.
--
-- Field encoding (the catch): scientific columns are comma-packed varchars with
-- a value and a quality flag. TMP = '+0072,1' means 7.2 deg C, flag 1.
-- Missing = +9999. Split on ',', cast, divide by 10. DATE is a real TIMESTAMP.
-- ----------------------------------------------------------------------------

-- 4) Hourly air temperature, one station, date range.
SELECT
    NAME,
    DATE                                              AS ts_utc,
    TRY_CAST(split_part(TMP, ',', 1) AS INTEGER)/10.0 AS temp_c,
    split_part(TMP, ',', 2)                           AS qc
FROM read_csv_auto('s3://noaa-global-hourly-pds/2023/74486094789.csv')  -- JFK Intl
WHERE DATE BETWEEN '2023-06-01' AND '2023-06-02'
  AND split_part(TMP, ',', 1) <> '+9999'
ORDER BY DATE;

-- 5) Hourly across several stations at once: pass an explicit LIST of paths.
--    (A {a,b,c} brace string in the path does NOT expand over S3 -- it 404s.
--    Use a list, or a '*' glob, instead.)
SELECT
    NAME,
    DATE AS ts_utc,
    TRY_CAST(split_part(TMP, ',', 1) AS INTEGER)/10.0 AS temp_c
FROM read_csv([
        's3://noaa-global-hourly-pds/2023/74486094789.csv',  -- JFK
        's3://noaa-global-hourly-pds/2023/72503014732.csv',  -- LaGuardia
        's3://noaa-global-hourly-pds/2023/72505394728.csv'   -- NY Central Park
     ], union_by_name = true)
WHERE DATE BETWEEN '2023-06-01' AND '2023-06-02'
  AND split_part(TMP, ',', 1) <> '+9999'
ORDER BY ts_utc, NAME;
-- To discover ids for a region,
-- list a year's files: SELECT * FROM glob('s3://noaa-global-hourly-pds/2023/*');
-- or join the ISD station history (lat/lon) the same way as the GHCN box query.

-- ----------------------------------------------------------------------------
-- Tips
--   * For repeated work, copy a slice local once, then query at memory speed:
--       CREATE TABLE nyc_tmax AS <query 2 without LIMIT>;
--     or  COPY (<query>) TO 'nyc_tmax.parquet' (FORMAT parquet);
--   * Filter on the partition columns (STATION/ELEMENT or YEAR/ELEMENT) whenever
--     you can -- DuckDB prunes whole files and only the needed pieces transfer.
--   * Other Parquet weather sources that work the same way: NOAA GSOD,
--     and the NOAA GHCN 'by_year' tree above for whole-planet scans.
--
-- Open-Meteo path (when you truly need their model output, e.g. ERA5 reanalysis):
--   Easiest is their HTTP API -> request CSV/JSON for your coords + range, write
--   to Parquet, then query in DuckDB. Sketch:
--     import openmeteo_requests, pandas as pd, duckdb
--     # ... call api, build df ...
--     df.to_parquet('era5_local.parquet')
--     duckdb.sql("SELECT * FROM 'era5_local.parquet' WHERE ...")
-- ============================================================================
