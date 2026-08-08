#!/usr/bin/env python
"""Regenerate data/honey_daily_source.parquet from the team's remote DuckDB server.

    python scripts/pull_source_snapshot.py

This is the ONLY script in the project that needs Tailscale and the team's `.env`
credentials. It is not on the path to reproducing the milestone: the snapshot it writes
is committed to the repository, and scripts/build_honey_model.py reads that committed
file. Run this only to refresh the snapshot from source.

Source table: `bob_sensor_processed` on the team's Quack server -- the published Bee
Observer sensor archive. Only the daily-grain rows (interval = 'd') are pulled; the
minute and hourly grains are 52M rows and are not used by the daily modelling pipeline.
Both the `years` and `events` daily datasets are pulled so the overlap between them can
be documented; 01_raw_anchor.sql then selects `years` only. See data/README.md.

Timestamps and the event-distance columns are cast to VARCHAR in the remote query: the
DuckDB `quack` extension crashes (Vector::Reference) on some of those vector types.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import duckdb
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "data" / "honey_daily_source.parquet"

REMOTE_SQL = """
SELECT
    dataset,
    category,
    source_file,
    key,
    CAST(time AS VARCHAR)                AS time,
    weight_kg,
    weight_delta,
    weight_kg_noOutlier,
    weight_delta_noOutlier,
    outlier_lim,
    no_jump,
    t_i_1, t_i_2, t_i_3, t_i_4, t_i_5,
    t_o, h, t, p,
    lat, lon,
    CAST(honey_last_dif      AS VARCHAR) AS honey_last_dif,
    CAST(honey_next_dif      AS VARCHAR) AS honey_next_dif,
    CAST(feeding_last_dif    AS VARCHAR) AS feeding_last_dif,
    CAST(feeding_next_dif    AS VARCHAR) AS feeding_next_dif,
    CAST(swarming_last_dif   AS VARCHAR) AS swarming_last_dif,
    CAST(swarming_next_dif   AS VARCHAR) AS swarming_next_dif,
    CAST(treatment_last_dif  AS VARCHAR) AS treatment_last_dif,
    CAST(treatment_next_dif  AS VARCHAR) AS treatment_next_dif,
    CAST(died_last_dif       AS VARCHAR) AS died_last_dif,
    CAST(died_next_dif       AS VARCHAR) AS died_next_dif,
    CAST(queencell_last_dif  AS VARCHAR) AS queencell_last_dif,
    CAST(queencell_next_dif  AS VARCHAR) AS queencell_next_dif
FROM bob_sensor_processed
WHERE interval = 'd'
"""


def main() -> int:
    output = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_OUTPUT

    load_dotenv(dotenv_path=REPO_ROOT / ".env")
    host = os.getenv("TAILNET_HOST")
    token = os.getenv("QUACK_TOKEN")
    if not host or not token:
        print("TAILNET_HOST and QUACK_TOKEN must be set in .env (see README).", file=sys.stderr)
        return 1

    con = duckdb.connect()
    con.execute("FORCE INSTALL quack")
    con.execute("LOAD quack")
    con.execute(f"ATTACH 'quack:{host}' AS bees (TOKEN '{token}', DISABLE_SSL true)")

    escaped = REMOTE_SQL.replace("'", "''")
    output.parent.mkdir(parents=True, exist_ok=True)
    con.execute(
        f"COPY (SELECT * FROM bees.query('{escaped}')) "
        f"TO '{output.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )

    summary = con.execute(
        f"""
        SELECT dataset,
               COUNT(*)                    AS row_count,
               COUNT(DISTINCT key)         AS hives,
               COUNT(DISTINCT source_file) AS files
        FROM read_parquet('{output.as_posix()}')
        GROUP BY dataset ORDER BY dataset
        """
    ).fetchdf()
    print(summary.to_string(index=False))
    print(f"wrote {output} ({output.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
