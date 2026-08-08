"""Weather features, at daily and at period grain.

Section 7.3 item 3 of `Milestone_4_alt.ipynb`:

    `honey_weather.parquet` is referenced by the daily feature set but is not in the
    repository. Weekly rainfall and degree-day totals are far more plausibly predictive of
    a week's nectar flow than a single day's reading is of tomorrow's, and `foraging_days`
    -- the crudest possible proxy -- already carries measurable weight.

The mechanism is not subtle. A colony gains mass when bees fly, and bees fly when it is
warm, dry and bright. Every feature the two Milestone 4 notebooks have is a function of the
hive's own past weight, so the models can only ever extrapolate a trend; nothing tells them
that next week is forecast to rain for five days. The sensor columns come closest --
`avg_outside_temp` is measured at the hive -- but they are the *current* period's weather,
23-35% missing, and say nothing about the period being forecast.

Which is the honest caveat this module has to carry, and it is stated wherever these
features are used: **the weather joined here is the weather that happened.** Using next
week's actual weather to predict next week's weight gain measures the ceiling a perfect
forecast would reach, not what a beekeeper could do on Sunday evening. Both are computed:

    add_period_weather(..., lookahead=False)   only weather up to the end of the current
                                               period. Deployable today.
    add_period_weather(..., lookahead=True)    the forecast period's own weather.
                                               An upper bound, labelled as one, in the same
                                               way `models.oracle_gated_predict` is.

Data provenance is in `scripts/pull_weather_snapshot.py`: ERA5 reanalysis via Open-Meteo,
one series per distinct hive site, complete with no missing days. Reanalysis is modelled,
not measured, at roughly 9 km resolution -- fine against 0.1-degree hive coordinates, and
worth remembering before reading much into a single site-day.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from honeymodel.data import DATE, GROUP, WEATHER_TABLE

#: Raw daily variables as pulled. Kept in one list so the aggregation, the feature names
#: and the assertion in the loader cannot drift apart.
DAILY_WEATHER_COLUMNS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "rain_sum",
    "precipitation_hours",
    "sunshine_duration",
    "shortwave_radiation_sum",
    "wind_speed_10m_max",
    "et0_fao_evapotranspiration",
    "growing_degree_days_10c",
    "foraging_hours_proxy",
    "is_foraging_day",
]

#: How each variable collapses over a week or a month. Sums for the things that accumulate
#: (rain, degree-days, foraging opportunity), means and extremes for the things that
#: describe the period's character.
PERIOD_AGGREGATIONS: dict[str, tuple[str, ...]] = {
    "temperature_2m_max": ("mean", "max"),
    "temperature_2m_min": ("mean", "min"),
    "temperature_2m_mean": ("mean", "std"),
    "precipitation_sum": ("sum", "max"),
    "precipitation_hours": ("sum",),
    "sunshine_duration": ("sum",),
    "shortwave_radiation_sum": ("sum",),
    "wind_speed_10m_max": ("mean", "max"),
    "et0_fao_evapotranspiration": ("sum",),
    "growing_degree_days_10c": ("sum",),
    "is_foraging_day": ("sum",),
}

WEATHER_PERIOD_FEATURES = [
    f"wx_{column}_{statistic}"
    for column, statistics in PERIOD_AGGREGATIONS.items()
    for statistic in statistics
]

#: Same features measured over the period being forecast rather than the current one.
WEATHER_LOOKAHEAD_FEATURES = [f"next_{name}" for name in WEATHER_PERIOD_FEATURES]

#: Five features, one per mechanism: energy in, water, accumulated warmth, flight
#: opportunity, and the thing that keeps bees at home on a warm day. The full 16 are
#: heavily redundant -- four temperature summaries and three ways of counting rain -- and
#: the first fold has barely a year of training rows, so this rung separates "weather does
#: not help" from "sixteen extra columns do not fit in 3,000 rows".
WEATHER_CORE_FEATURES = [
    "wx_shortwave_radiation_sum_sum",
    "wx_precipitation_sum_sum",
    "wx_growing_degree_days_10c_sum",
    "wx_is_foraging_day_sum",
    "wx_wind_speed_10m_max_mean",
]
WEATHER_CORE_LOOKAHEAD_FEATURES = [f"next_{name}" for name in WEATHER_CORE_FEATURES]

# Registered with the period leakage guard on import, so a feature matrix built from these
# fails unless the caller passes `allow=` and takes responsibility for labelling the
# result an upper bound. Registered here rather than declared in `periods` to keep the
# weather layer optional: `periods` must import with no weather snapshot present.
def _register_lookahead_guard() -> None:
    from honeymodel import periods

    periods.PERIOD_LOOKAHEAD_WEATHER = frozenset(WEATHER_LOOKAHEAD_FEATURES)


_register_lookahead_guard()


class WeatherUnavailable(FileNotFoundError):
    """Raised when the weather snapshot is missing, with the command that builds it."""


def load_weather(path: Path | str | None = None) -> pd.DataFrame:
    """Load the committed weather snapshot, keyed on (latitude, longitude, date)."""
    source = Path(path) if path is not None else WEATHER_TABLE
    if not source.exists():
        raise WeatherUnavailable(
            f"{source} not found. Build it with:\n"
            f"    python scripts/pull_weather_snapshot.py\n"
            "This is the only step in the project that needs a network connection."
        )
    frame = pd.read_parquet(source)
    frame[DATE] = pd.to_datetime(frame[DATE])
    missing = [column for column in DAILY_WEATHER_COLUMNS if column not in frame.columns]
    if missing:
        raise AssertionError(f"weather snapshot is missing columns: {missing}")
    return frame


def attach_daily_weather(daily: pd.DataFrame, weather: pd.DataFrame | None = None) -> pd.DataFrame:
    """Join daily weather onto the modelling table on rounded coordinates and date.

    The published hive coordinates are rounded to 0.1 degrees and the snapshot was pulled
    on exactly those values, so this is an equality join rather than a nearest-neighbour
    search. Rounding both sides again guards against a float representation difference
    turning a match into a miss silently.
    """
    weather = load_weather() if weather is None else weather
    left = daily.copy()
    left["_lat"] = left.latitude.round(1)
    left["_lon"] = left.longitude.round(1)

    right = weather.copy()
    right["_lat"] = right.latitude.round(1)
    right["_lon"] = right.longitude.round(1)
    right = right.drop(columns=["latitude", "longitude"])

    merged = left.merge(right, on=["_lat", "_lon", DATE], how="left")
    merged = merged.drop(columns=["_lat", "_lon"])

    matched = merged[DAILY_WEATHER_COLUMNS[0]].notna().mean()
    if matched < 0.99:
        raise AssertionError(
            f"only {matched:.1%} of hive-days matched a weather site-day; expected ~100%. "
            "Rebuild the snapshot -- its date range or site list is out of step with the "
            "modelling table."
        )
    return merged


def _aggregate(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    grouped = frame.groupby(keys, sort=True)
    pieces = []
    for column, statistics in PERIOD_AGGREGATIONS.items():
        piece = grouped[column].agg(list(statistics))
        piece.columns = [f"wx_{column}_{statistic}" for statistic in statistics]
        pieces.append(piece)
    return pd.concat(pieces, axis=1).reset_index()


def add_period_weather(
    period_frame: pd.DataFrame,
    daily_frame: pd.DataFrame,
    weather: pd.DataFrame | None = None,
    period: str = "week",
    lookahead: bool = False,
) -> pd.DataFrame:
    """Attach period-aggregated weather to a period table.

    With `lookahead=False` the row carries the weather of the period it describes, all of
    which is known when the forecast is made. With `lookahead=True` it additionally carries
    the *next* period's weather under `next_wx_*` names -- the period being forecast. Those
    columns are an upper bound on what a perfect weather forecast would provide and must
    never be reported as a deployable result.

    The next-period join uses the same contiguity rule as the target: no adjacent period
    index, no next weather, so a hive that goes dark cannot borrow a stranger's week.
    """
    from honeymodel.periods import PERIOD_ALIAS, PERIOD_KEY

    weather = load_weather() if weather is None else weather
    joined = attach_daily_weather(daily_frame, weather)
    joined[PERIOD_KEY] = pd.PeriodIndex(joined[DATE], freq=PERIOD_ALIAS[period]).astype("int64")

    aggregated = _aggregate(joined, [GROUP, PERIOD_KEY])
    out = period_frame.merge(aggregated, on=[GROUP, PERIOD_KEY], how="left")
    out = out.sort_values([GROUP, PERIOD_KEY], kind="mergesort").reset_index(drop=True)

    if lookahead:
        grouped = out.groupby(GROUP, sort=False)
        next_contiguous = (grouped[PERIOD_KEY].shift(-1) - out[PERIOD_KEY]) == 1
        for name in WEATHER_PERIOD_FEATURES:
            out[f"next_{name}"] = grouped[name].shift(-1).where(next_contiguous)
    return out


def weather_coverage(period_frame: pd.DataFrame) -> pd.DataFrame:
    """Non-null share of every weather feature, so a silent join failure cannot hide.

    Reanalysis has no gaps, so anything below 100% here is a join problem or a period with
    no observed days, not a missing measurement -- which makes this a cheap integrity
    check rather than a missingness report.
    """
    rows = []
    for name in WEATHER_PERIOD_FEATURES:
        if name not in period_frame.columns:
            continue
        series = period_frame[name]
        rows.append(
            {
                "feature": name,
                "non_null_pct": round(100 * float(series.notna().mean()), 2),
                "mean": round(float(series.mean()), 3),
                "sd": round(float(series.std()), 3),
            }
        )
    return pd.DataFrame(rows)


def weather_target_correlation(
    period_frame: pd.DataFrame, target: str, lookahead: bool = True
) -> pd.DataFrame:
    """Rank correlation of each weather feature with the target, current and next period.

    Spearman rather than Pearson: rainfall is zero-inflated and heavy-tailed, and a single
    thunderstorm week would otherwise set the correlation. The `next_*` column is the one
    with the mechanism -- next week's rain suppresses next week's gain -- and if it is not
    larger in magnitude than the current-period column, the weather features are working
    as seasonality proxies rather than as weather.
    """
    usable = period_frame.dropna(subset=[target])
    rows = []
    for name in WEATHER_PERIOD_FEATURES:
        if name not in usable.columns:
            continue
        row = {
            "feature": name,
            "rho_current_period": round(float(usable[name].corr(usable[target], method="spearman")), 4),
        }
        next_name = f"next_{name}"
        if lookahead and next_name in usable.columns:
            paired = usable.dropna(subset=[next_name])
            row["rho_next_period"] = round(
                float(paired[next_name].corr(paired[target], method="spearman")), 4
            )
            row["n_next"] = len(paired)
        rows.append(row)
    table = pd.DataFrame(rows)
    sort_column = "rho_next_period" if "rho_next_period" in table.columns else "rho_current_period"
    return table.reindex(table[sort_column].abs().sort_values(ascending=False).index).reset_index(
        drop=True
    )


def weather_feature_sets(base: str = "history+sensors") -> dict[str, list[str]]:
    """The ablation ladder for the weather question, built on a period feature set.

    Four rungs, and the gap between the third and the fourth is the answer to "would a
    weather forecast help?" as distinct from "does weather matter?".
    """
    from honeymodel.periods import PERIOD_FEATURE_SETS

    baseline = list(PERIOD_FEATURE_SETS[base])
    return {
        base: baseline,
        f"{base}+weather_core": baseline + WEATHER_CORE_FEATURES,
        f"{base}+weather": baseline + WEATHER_PERIOD_FEATURES,
        f"{base}+weather_core+lookahead": baseline
        + WEATHER_CORE_FEATURES
        + WEATHER_CORE_LOOKAHEAD_FEATURES,
        f"{base}+weather+lookahead": baseline + WEATHER_PERIOD_FEATURES + WEATHER_LOOKAHEAD_FEATURES,
        "weather-only": WEATHER_PERIOD_FEATURES + ["month", "sin_year", "cos_year"],
    }
