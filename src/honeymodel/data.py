"""Loading the modelling dataset.

One entry point, one source of truth: the committed Parquet snapshot. There is no
remote path, no local-DuckDB probe chain, and no archived-output fallback -- if the
file is missing the load fails and says how to rebuild it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
MODEL_TABLE = DATA_DIR / "honey_model.parquet"
DAILY_SOURCE = DATA_DIR / "honey_daily_source.parquet"
WEATHER_TABLE = DATA_DIR / "honey_weather.parquet"

TARGET = "target_next_day_weight_change_kg"
GROUP = "hive_id"
DATE = "measurement_date"

#: Published Milestone 3 dataset shape, asserted on load.
EXPECTED_ROWS = 26_215
EXPECTED_HIVES = 78


def load_model_table(path: Path | str | None = None, validate: bool = True) -> pd.DataFrame:
    """Load `honey_model` from the committed Parquet snapshot.

    Parameters
    ----------
    path : optional override for the snapshot location.
    validate : assert the published row/hive counts and the absence of duplicate
        hive-days. Turn off only when deliberately loading a modified table.
    """
    source = Path(path) if path is not None else MODEL_TABLE
    if not source.exists():
        raise FileNotFoundError(
            f"{source} not found. Rebuild it with:\n"
            f"    python scripts/build_honey_model.py"
        )

    frame = pd.read_parquet(source)
    frame[DATE] = pd.to_datetime(frame[DATE])
    for column in ("previous_observation_date", "next_observation_date"):
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column])

    frame = frame.sort_values([GROUP, DATE], kind="mergesort").reset_index(drop=True)

    if validate:
        duplicates = frame.duplicated(subset=[GROUP, DATE]).sum()
        if duplicates:
            raise AssertionError(f"{duplicates} duplicate (hive_id, measurement_date) rows")
        if len(frame) != EXPECTED_ROWS or frame[GROUP].nunique() != EXPECTED_HIVES:
            raise AssertionError(
                f"expected {EXPECTED_ROWS:,} rows across {EXPECTED_HIVES} hives, "
                f"got {len(frame):,} rows across {frame[GROUP].nunique()} hives. "
                "If the snapshot changed on purpose, pass validate=False and update "
                "EXPECTED_ROWS/EXPECTED_HIVES."
            )
    return frame


def load_daily_source(path: Path | str | None = None) -> pd.DataFrame:
    """Load the daily-grain source snapshot (pre-filter, both `years` and `events`).

    Used for the data-preparation waterfall and for comparing the publishers' raw and
    outlier-cleaned weight series. Not a modelling input.
    """
    source = Path(path) if path is not None else DAILY_SOURCE
    if not source.exists():
        raise FileNotFoundError(
            f"{source} not found. Regenerate it with scripts/pull_source_snapshot.py "
            "(this is the only step that needs Tailscale)."
        )
    frame = pd.read_parquet(source)
    frame["time"] = pd.to_datetime(frame["time"])
    frame["measurement_date"] = frame["time"].dt.normalize()
    frame["hive_id"] = frame["key"].astype(int)
    return frame


EVENT_TYPES = ("honey", "feeding", "swarming", "treatment", "died", "queencell")
SECONDS_PER_DAY = 86_400.0


def normalise_event_distance_units(
    frame: pd.DataFrame, return_report: bool = False
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Put every beekeeper-event distance column into days.

    The published `*_last_dif` / `*_next_dif` columns are not in a consistent unit. For
    some hives they advance by exactly 1.0 per calendar day; for others by exactly
    86400.0. The unit is a property of the hive's source file, not of the event type:
    33 hives are in days and 24 are in seconds.

    That inconsistency silently breaks any threshold comparison. Milestone 3's extremes
    notebook asked what fraction of events fall "within 1 day" by comparing the raw
    value to 1.0 -- for a seconds-unit hive that test needs 86400, so those hives could
    never register as near an event and pushed the reported figure toward 0%.

    The unit is inferred per (hive, column) from the median increment between
    consecutive-day observations, which is 1 or 86400 and nothing in between.
    """
    frame = frame.copy()
    columns = [f"{event}_{side}_dif" for event in EVENT_TYPES for side in ("last", "next")]
    columns = [c for c in columns if c in frame.columns]

    ordered = frame.sort_values([GROUP, DATE], kind="mergesort")
    day_gap = ordered.groupby(GROUP)[DATE].diff().dt.days == 1

    report_rows = []
    for column in columns:
        increment = ordered.groupby(GROUP)[column].diff().where(day_gap)
        median_increment = increment.abs().groupby(ordered[GROUP]).median()
        in_seconds = median_increment > 1_000
        seconds_hives = set(median_increment.index[in_seconds.fillna(False)])
        if seconds_hives:
            mask = frame[GROUP].isin(seconds_hives)
            frame.loc[mask, column] = frame.loc[mask, column] / SECONDS_PER_DAY
        report_rows.append({
            "column": column,
            "hives_in_days": int((~in_seconds.fillna(True)).sum()),
            "hives_in_seconds": int(in_seconds.fillna(False).sum()),
            "hives_undetermined": int(median_increment.isna().sum()),
        })

    # Recompute the nearest-event distances from the corrected columns.
    for event in EVENT_TYPES:
        last, nxt = f"{event}_last_dif", f"{event}_next_dif"
        target_column = f"nearest_{event}_event_days"
        if last in frame.columns and nxt in frame.columns:
            frame[target_column] = frame[[last, nxt]].abs().min(axis=1)

    if return_report:
        return frame, pd.DataFrame(report_rows)
    return frame


def add_season(frame: pd.DataFrame, date_column: str = DATE) -> pd.DataFrame:
    """Attach a meteorological-season label, used by the segmented report.

    Seasons are the interpretive unit for this target: a colony gains mass during the
    nectar flow and loses it slowly through dormancy, so pooled metrics average over
    two regimes that behave nothing alike.
    """
    month = pd.to_datetime(frame[date_column]).dt.month
    season = pd.Series("Winter", index=frame.index, dtype=object)
    season[month.isin([3, 4, 5])] = "Spring"
    season[month.isin([6, 7, 8])] = "Summer"
    season[month.isin([9, 10, 11])] = "Autumn"
    return frame.assign(season=season)
