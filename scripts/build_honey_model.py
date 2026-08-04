#!/usr/bin/env python
"""Build the modelling table from the committed daily source snapshot.

    python scripts/build_honey_model.py

Runs sql/01..05 in order against a fresh DuckDB database and writes
data/honey_model.parquet. Needs no network, no Tailscale, and no pre-existing
DuckDB file -- everything it reads is committed to the repository.

Every path is explicit or derived from this file's location; nothing is guessed from
the current working directory, so the script gives the same result from anywhere.

Data-quality assertions run inline and fail the build loudly:
  * one row per (hive_id, measurement_date) before any window function;
  * the modelling table reproduces the published Milestone 3 dataset exactly
    (row count, hive count, date range, target mean/sd/min/max, extreme count).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = REPO_ROOT / "data" / "honey_daily_source.parquet"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "honey_model.parquet"
SQL_DIR = REPO_ROOT / "sql"

# Quality-rule parameters. Substituted into 05_honey_model.sql; recorded in data/README.md.
WEIGHT_FLOOR_KG = 5.0   # below this a hive-day reading is not a physically credible colony
MAD_K = 5.0             # hive-local robust outlier screen, in MAD units
RELATIVE_MAD_K = 5.0    # hive-relative extreme-target screen, in MAD units

# The Milestone 3 published dataset. The build must reproduce these exactly.
EXPECTED = {
    "observations": 26_215,
    "hives": 78,
    "first_date": "2019-06-25",
    "last_date": "2022-12-30",
    "target_mean": 0.002042,
    "target_sd": 1.840522,
    "target_min": -65.322940,
    "target_max": 39.098176,
    "extremes": 445,
}
TOLERANCE = 1e-5


def run_sql_file(con: duckdb.DuckDBPyConnection, path: Path, **substitutions: object) -> None:
    sql = path.read_text()
    for key, value in substitutions.items():
        sql = sql.replace("${" + key + "}", str(value))
    remaining = [tok for tok in ("${SOURCE}", "${WEIGHT_FLOOR_KG}", "${MAD_K}", "${RELATIVE_MAD_K}") if tok in sql]
    if remaining:
        raise RuntimeError(f"{path.name}: unsubstituted placeholders {remaining}")
    con.execute(sql)
    print(f"  ran {path.name}")


def assert_unique_hive_days(con: duckdb.DuckDBPyConnection) -> None:
    duplicates = con.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT hive_id, measurement_date
            FROM honey_daily_summary
            GROUP BY 1, 2
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]
    if duplicates:
        raise AssertionError(
            f"{duplicates} duplicate (hive_id, measurement_date) pairs in honey_daily_summary. "
            "The one-row-per-hive-day grain assumed by every window function does not hold."
        )
    print("  grain check: one row per (hive_id, measurement_date)")


def assert_milestone_3_parity(con: duckdb.DuckDBPyConnection) -> None:
    actual = con.execute(
        """
        SELECT
            COUNT(*)                                   AS observations,
            COUNT(DISTINCT hive_id)                    AS hives,
            CAST(MIN(measurement_date) AS VARCHAR)     AS first_date,
            CAST(MAX(measurement_date) AS VARCHAR)     AS last_date,
            AVG(target_next_day_weight_change_kg)      AS target_mean,
            STDDEV(target_next_day_weight_change_kg)   AS target_sd,
            MIN(target_next_day_weight_change_kg)      AS target_min,
            MAX(target_next_day_weight_change_kg)      AS target_max,
            SUM(CASE WHEN extreme_weight_change_flag THEN 1 ELSE 0 END) AS extremes
        FROM honey_model
        """
    ).fetchdf().iloc[0].to_dict()

    failures = []
    for key, expected in EXPECTED.items():
        got = actual[key]
        if isinstance(expected, str):
            ok = str(got) == expected
        elif isinstance(expected, int):
            ok = int(got) == expected
        else:
            ok = abs(float(got) - expected) < TOLERANCE
        if not ok:
            failures.append(f"    {key}: expected {expected}, got {got}")
    if failures:
        raise AssertionError(
            "honey_model does not reproduce the Milestone 3 published dataset:\n"
            + "\n".join(failures)
        )
    print("  parity check: reproduces the Milestone 3 dataset exactly")


def report_quality_flags(con: duckdb.DuckDBPyConnection) -> None:
    flags = con.execute(
        """
        SELECT
            COUNT(*)                                                       AS rows,
            SUM(implausible_weight_flag::INT)                              AS implausible_weight,
            SUM(robust_outlier_flag::INT)                                  AS robust_outlier,
            SUM(sensor_dropout_flag::INT)                                  AS sensor_dropout,
            SUM(extreme_weight_change_flag::INT)                           AS extremes_5kg,
            SUM((extreme_weight_change_flag AND implausible_weight_flag)::INT)
                                                                           AS extremes_implausible,
            SUM((extreme_weight_change_flag AND sensor_dropout_flag)::INT) AS extremes_dropout
        FROM honey_model
        """
    ).fetchdf().iloc[0]
    print(
        "  quality flags: "
        f"{flags['implausible_weight']:,} implausible-weight, "
        f"{flags['robust_outlier']:,} robust-outlier, "
        f"{flags['sensor_dropout']:,} dropout-signature rows; "
        f"of {flags['extremes_5kg']:,} extremes, {flags['extremes_implausible']:,} are "
        f"implausible-weight and {flags['extremes_dropout']:,} carry a dropout signature"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                        help="daily-grain source Parquet (default: data/honey_daily_source.parquet)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="destination Parquet (default: data/honey_model.parquet)")
    parser.add_argument("--database", type=Path, default=None,
                        help="persist the intermediate DuckDB tables to this file (default: in-memory)")
    parser.add_argument("--skip-parity", action="store_true",
                        help="skip the Milestone 3 parity assertion (use when the source snapshot changes on purpose)")
    args = parser.parse_args()

    source = args.source.resolve()
    if not source.exists():
        print(f"source snapshot not found: {source}", file=sys.stderr)
        print("regenerate it with scripts/pull_source_snapshot.py (needs Tailscale)", file=sys.stderr)
        return 1

    con = duckdb.connect(str(args.database) if args.database else ":memory:")
    print(f"building from {source}")

    run_sql_file(con, SQL_DIR / "01_raw_anchor.sql", SOURCE=source.as_posix())
    run_sql_file(con, SQL_DIR / "02_clean.sql")
    run_sql_file(con, SQL_DIR / "03_daily_summary.sql")
    assert_unique_hive_days(con)
    run_sql_file(con, SQL_DIR / "04_feature_candidates.sql")
    run_sql_file(
        con,
        SQL_DIR / "05_honey_model.sql",
        WEIGHT_FLOOR_KG=WEIGHT_FLOOR_KG,
        MAD_K=MAD_K,
        RELATIVE_MAD_K=RELATIVE_MAD_K,
    )

    if not args.skip_parity:
        assert_milestone_3_parity(con)
    report_quality_flags(con)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    con.execute(
        f"COPY (SELECT * FROM honey_model ORDER BY hive_id, measurement_date) "
        f"TO '{args.output.resolve().as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    size_mb = args.output.stat().st_size / 1e6
    print(f"wrote {args.output} ({size_mb:.2f} MB)")

    # Regenerate the schema DDL so sql/06_schema.sql never drifts from the table.
    schema = con.execute("DESCRIBE honey_model").fetchdf()
    ddl_columns = ",\n".join(
        f"    {row.column_name:<40} {row.column_type}" for row in schema.itertuples()
    )
    (SQL_DIR / "06_schema.sql").write_text(
        "-- 06_schema.sql\n"
        "-- Generated by scripts/build_honey_model.py from DESCRIBE honey_model.\n"
        "-- Do not edit by hand; re-run the build script instead.\n\n"
        f"CREATE TABLE honey_model (\n{ddl_columns}\n);\n"
    )
    print(f"wrote {SQL_DIR / '06_schema.sql'} ({len(schema)} columns)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
