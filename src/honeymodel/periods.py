"""Weekly and monthly re-grain of the daily modelling table.

The Milestone 4 pipeline predicts *tomorrow's* change in daily-mean hive weight. That
target is noise-dominated: predicting zero gives MAE 0.525 kg and no single-stage learned
model beats it. Aggregating to a longer period is not a cosmetic change -- it moves every
load-bearing piece of the protocol -- so each move lives here rather than being improvised
in a notebook cell.

What changes, and why each function exists:

    aggregate_to_period      26,215 hive-days -> ~3,000 hive-weeks / ~620 hive-months.
                             Needs an explicit coverage gate, because a "week" built from
                             one observed day is not a week.
    add_period_features      every daily-lag feature (previous_day_weight_kg,
                             rolling_3/7_day, weight_slope_2_day) is a daily construct and
                             has to be rebuilt at the new grain. Also chooses the *anchor*
                             -- what "the hive's weight this week" means -- which turns out
                             to decide whether the model beats its baseline at all.
    rolling_origin_period_cv the daily splitter is safe because a row's target lands the
                             next day. At week grain the target lands up to 7 days later,
                             so a training row adjacent to the boundary carries a label
                             measured inside the test window. That needs an embargo.
    period_baselines         predicting zero is a much weaker competitor once the target
                             has sd 5 kg instead of 1.8, while calendar climatology gets
                             much stronger. The bar has to be re-set or the skill score
                             flatters the model.
    relative_extreme_threshold
                             +/-5 kg is 2.7 daily sd but only 1.0 weekly sd. The absolute
                             threshold stops meaning "extreme" and has to be re-derived.

Nothing here re-fits the daily table; `honeymodel.data` remains the single loader.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from honeymodel.data import DATE, EVENT_TYPES, GROUP, add_season
from honeymodel.evaluation import Fold, regression_metrics, skill_score

#: Period target and the chain columns it is built from.
PERIOD_TARGET = "target_next_period_weight_change_kg"
PERIOD_START = "period_start_date"
PERIOD_END = "period_end_date"
PERIOD_KEY = "period_index"

#: Minimum observed days for a period to be admitted. A week reconstructed from two
#: readings is a different measurement from one reconstructed from seven, and the
#: difference lands entirely in the target.
DEFAULT_MIN_DAYS = {"week": 5, "month": 20}

PERIOD_ALIAS = {"week": "W-SUN", "month": "M"}

SENSOR_COLUMNS = [
    "avg_internal_temp_1",
    "avg_internal_temp_2",
    "avg_internal_temp_3",
    "avg_internal_temp_4",
    "avg_outside_temp",
    "avg_humidity",
    "avg_pressure",
]

#: Columns that must never reach a period feature matrix. Same three kinds of poison as
#: the daily FORBIDDEN_COLUMNS -- the target, anything measured after the period ends,
#: and anything derived from the label -- restated at this grain because the column names
#: are different and a shared frozenset would silently pass everything through.
PERIOD_FORBIDDEN = frozenset(
    {
        PERIOD_TARGET,
        "next_period_weight_kg",
        "next_period_index",
        "next_period_start_date",
        "next_period_days_observed",
        "period_extreme_flag",
        "period_extreme_relative_flag",
        # Milestone 5 additions. The gross-gain target and the removal that produced it
        # are both measured inside the period being forecast; `period_removed_kg` (the
        # current period's removal) is deliberately *not* here, because it is knowable.
        "target_next_period_gross_gain_kg",
        "next_period_removed_kg",
    }
    | {f"{event}_next_dif_days" for event in EVENT_TYPES}
)


#: Weather measured over the period being forecast. Forbidden by default and whitelisted
#: only for the explicitly-labelled upper-bound rung in `weather.weather_feature_sets`:
#: next week's actual rainfall is not available on Sunday evening, and a model given it is
#: answering "what could a perfect forecast buy?" rather than "what can we ship?".
#:
#: Populated by `honeymodel.weather` on import rather than declared here, so that this
#: module keeps working with no weather snapshot present. That is not a hole: the only way
#: a `next_wx_*` column can exist on a frame is `weather.add_period_weather`, so the guard
#: is armed whenever there is anything for it to catch.
PERIOD_LOOKAHEAD_WEATHER = frozenset()


class PeriodLeakageError(AssertionError):
    """Raised when a forbidden period column reaches a feature matrix."""


def _forbidden(allow: frozenset[str] | set[str] | None = None) -> frozenset[str]:
    blocked = PERIOD_FORBIDDEN | PERIOD_LOOKAHEAD_WEATHER
    return blocked - frozenset(allow or ())


def assert_no_period_leakage(
    columns: object, allow: frozenset[str] | set[str] | None = None
) -> None:
    """Fail if any forbidden column reached a feature matrix.

    `allow` exists for exactly one caller -- the upper-bound weather rung -- and every
    result computed with it set is labelled as an upper bound where it is reported. It is
    a deliberate, named exception rather than a hole: without it the honest comparison
    "what does a perfect weather forecast buy?" could not be measured at all, and with it
    unlabelled the number would be indistinguishable from a deployable one.
    """
    offenders = sorted(_forbidden(allow).intersection(set(columns)))
    if offenders:
        raise PeriodLeakageError(
            "these columns are only knowable after the period ends, or are derived from "
            f"the label, and must not be used as features: {offenders}"
        )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _within_period_slope(frame: pd.DataFrame, keys: list[str], value: str) -> pd.Series:
    """OLS slope of daily weight against day-of-period, per period.

    Closed form rather than a groupby-apply: 4,275 groups through a Python callable is
    slow enough to notice, and the algebra is three sums.
    """
    x = frame.groupby(keys, sort=False).cumcount().astype(float)
    work = pd.DataFrame({"x": x, "y": frame[value].astype(float)})
    work[keys] = frame[keys].to_numpy()
    work["xy"] = work.x * work.y
    work["xx"] = work.x * work.x

    grouped = work.groupby(keys, sort=False).agg(
        n=("y", "size"), sx=("x", "sum"), sy=("y", "sum"), sxy=("xy", "sum"), sxx=("xx", "sum")
    )
    denominator = grouped.n * grouped.sxx - grouped.sx**2
    slope = (grouped.n * grouped.sxy - grouped.sx * grouped.sy) / denominator
    return slope.where(denominator != 0)


def aggregate_to_period(
    frame: pd.DataFrame,
    period: str = "week",
    min_days: int | None = None,
    foraging_temp_c: float = 12.0,
) -> pd.DataFrame:
    """Collapse the daily table to one row per (hive, period).

    Both candidate weight anchors are produced here -- `period_last_weight_kg` and
    `period_mean_weight_kg` -- and `add_period_features` chooses between them. Neither is
    obviously right, and the choice is not cosmetic: see that function.

    Sensor columns are summarised by more than a mean: at this grain the useful signal is
    the shape of the week, not its average. `days_above_{foraging_temp_c}c` is a crude
    foraging-opportunity count and is the one aggregate with a mechanism behind it.

    Every period carries `days_observed`. Periods below `min_days` are dropped here, and
    the count of what that removed is reported by `coverage_waterfall`.
    """
    if period not in PERIOD_ALIAS:
        raise ValueError(f"period must be one of {sorted(PERIOD_ALIAS)}, got {period!r}")
    min_days = DEFAULT_MIN_DAYS[period] if min_days is None else min_days

    daily = frame.sort_values([GROUP, DATE], kind="mergesort").copy()
    stamps = pd.PeriodIndex(daily[DATE], freq=PERIOD_ALIAS[period])
    daily["_period"] = stamps
    daily[PERIOD_KEY] = stamps.astype("int64")
    daily["_hot_day"] = (daily["avg_outside_temp"] > foraging_temp_c).astype(float)
    daily.loc[daily["avg_outside_temp"].isna(), "_hot_day"] = np.nan

    keys = [GROUP, PERIOD_KEY]
    grouped = daily.groupby(keys, sort=True)

    out = grouped.agg(
        days_observed=(DATE, "size"),
        **{PERIOD_START: (DATE, "first")},
        **{PERIOD_END: (DATE, "last")},
        period_last_weight_kg=("end_of_day_weight_kg", "last"),
        period_first_weight_kg=("end_of_day_weight_kg", "first"),
        period_mean_weight_kg=("end_of_day_weight_kg", "mean"),
        period_max_weight_kg=("end_of_day_weight_kg", "max"),
        period_min_weight_kg=("end_of_day_weight_kg", "min"),
        within_period_weight_sd_kg=("end_of_day_weight_kg", "std"),
        daily_change_mean_kg=("previous_day_weight_change_kg", "mean"),
        daily_change_sd_kg=("previous_day_weight_change_kg", "std"),
        daily_change_max_abs_kg=("previous_day_weight_change_kg", lambda s: s.abs().max()),
        foraging_days=("_hot_day", "sum"),
        hive_median_weight_kg=("hive_median_weight_kg", "first"),
        hive_mad_weight_kg=("hive_mad_weight_kg", "first"),
        latitude=("latitude", "first"),
        longitude=("longitude", "first"),
    ).reset_index()

    sensor_summary = grouped[SENSOR_COLUMNS].agg(["mean", "std", "max"])
    sensor_summary.columns = [f"{column}_{statistic}" for column, statistic in sensor_summary.columns]
    out = out.merge(sensor_summary.reset_index(), on=keys, how="left")

    out["within_period_slope_kg_per_day"] = (
        _within_period_slope(daily, keys, "end_of_day_weight_kg").reindex(
            pd.MultiIndex.from_frame(out[keys])
        ).to_numpy()
    )

    # Past-facing beekeeper-event distances only. The *_next_dif columns are forbidden,
    # and the published units are inconsistent per hive, so they arrive already
    # normalised to days by data.normalise_event_distance_units.
    for event in EVENT_TYPES:
        column = f"{event}_last_dif"
        if column in daily.columns:
            out[f"days_since_{event}_event"] = (
                grouped[column].last().abs().reindex(pd.MultiIndex.from_frame(out[keys])).to_numpy()
            )

    out["period"] = period
    out["coverage"] = out.days_observed / (7.0 if period == "week" else 30.4375)
    out = out[out.days_observed >= min_days].reset_index(drop=True)
    return out.sort_values([GROUP, PERIOD_KEY], kind="mergesort").reset_index(drop=True)


def coverage_waterfall(frame: pd.DataFrame, period: str = "week") -> pd.DataFrame:
    """Row-count accounting for the re-grain, in the style of the daily waterfall.

    Aggregation is not free and the notebook should not pretend it is: the coverage gate
    and the two contiguity requirements each remove rows, and each removal is a decision
    somebody could disagree with.
    """
    raw = aggregate_to_period(frame, period=period, min_days=1)
    gated = aggregate_to_period(frame, period=period)
    chained = add_period_features(gated).dropna(subset=[PERIOD_TARGET])
    with_lag = chained.dropna(subset=["period_weight_change_kg"])

    steps = [
        ("daily rows ingested", len(frame)),
        (f"aggregated to hive-{period}s", len(raw)),
        (f"coverage gate (>= {DEFAULT_MIN_DAYS[period]} observed days)", len(gated)),
        ("next period contiguous (target exists)", len(chained)),
        ("previous period contiguous (lag exists)", len(with_lag)),
    ]
    rows = []
    previous = None
    for label, remaining in steps:
        rows.append(
            {
                "step": label,
                "rows_remaining": remaining,
                "rows_dropped": 0 if previous is None else previous - remaining,
            }
        )
        previous = remaining
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Features at the new grain
# ---------------------------------------------------------------------------


ANCHOR_COLUMNS = {"mean": "period_mean_weight_kg", "last": "period_last_weight_kg"}


def add_period_features(
    frame: pd.DataFrame, anchor: str = "mean", rolling: int = 4
) -> pd.DataFrame:
    """Rebuild the weight-history chain, seasonality and the target at period grain.

    **The anchor is the decision this function exists to make.** "How much weight did the
    hive gain this week" has two defensible readings:

    `anchor="last"`  the change between the last observed day of each period. Consecutive
                     changes then partition the record exactly and sum to the total change
                     over it, which is the property a physical mass balance should have.
                     It also means the entire target rests on two single-day readings, so
                     every bit of day-level sensor noise -- and every beekeeper handling
                     jump, which Section 3.5 of Milestone 4 shows is 94% of daily extremes
                     -- survives aggregation at full amplitude.

    `anchor="mean"`  the change between period mean weights. Averaging over 5-7 days
                     cancels independent day-level noise and damps single-day handling
                     spikes to about a seventh of their amplitude. It gives up exact
                     additivity: consecutive mean-to-mean changes overlap in what they
                     measure, so they no longer sum to the record's total change.

    The daily table already anchors on a mean -- `end_of_day_weight_kg` is the mean of the
    day's minute readings, not the last one -- so `anchor="mean"` is also the choice that
    keeps the grain change consistent with what the level below it already does. It is the
    default because it wins the head-to-head in Section 5, not because it is tidier.

    Contiguity is required in both directions and is checked on `period_index`, not on
    dates. A hive that goes dark for a month has no valid change across the gap, and
    filling one in would fabricate the single most important feature in the daily model.

    Everything else here is knowable at the close of the current period. `rolling` windows
    include the current period deliberately -- that is information the beekeeper has --
    and the target is the change over the period that follows.
    """
    if anchor not in ANCHOR_COLUMNS:
        raise ValueError(f"anchor must be one of {sorted(ANCHOR_COLUMNS)}, got {anchor!r}")

    out = frame.sort_values([GROUP, PERIOD_KEY], kind="mergesort").copy()
    out["period_weight_kg"] = out[ANCHOR_COLUMNS[anchor]]
    out["anchor"] = anchor
    grouped = out.groupby(GROUP, sort=False)

    previous_index = grouped[PERIOD_KEY].shift(1)
    next_index = grouped[PERIOD_KEY].shift(-1)
    previous_contiguous = (out[PERIOD_KEY] - previous_index) == 1
    next_contiguous = (next_index - out[PERIOD_KEY]) == 1

    previous_weight = grouped["period_weight_kg"].shift(1).where(previous_contiguous)
    next_weight = grouped["period_weight_kg"].shift(-1).where(next_contiguous)

    out["previous_period_weight_kg"] = previous_weight
    out["period_weight_change_kg"] = out.period_weight_kg - previous_weight
    out[PERIOD_TARGET] = next_weight - out.period_weight_kg
    out["target_period_end_date"] = grouped[PERIOD_END].shift(-1).where(next_contiguous)
    out["previous_period_days_observed"] = grouped["days_observed"].shift(1).where(previous_contiguous)

    change = out["period_weight_change_kg"]
    out["previous_period_change_kg"] = grouped["period_weight_change_kg"].shift(1).where(
        previous_contiguous
    )
    out[f"rolling_{rolling}_period_change_kg"] = grouped["period_weight_change_kg"].transform(
        lambda s: s.rolling(rolling, min_periods=2).mean()
    )
    out[f"rolling_{rolling}_period_change_sd_kg"] = grouped["period_weight_change_kg"].transform(
        lambda s: s.rolling(rolling, min_periods=2).std()
    )
    out[f"rolling_{rolling}_period_weight_kg"] = grouped["period_weight_kg"].transform(
        lambda s: s.rolling(rolling, min_periods=2).mean()
    )
    out[f"weight_minus_rolling_{rolling}_kg"] = (
        out.period_weight_kg - out[f"rolling_{rolling}_period_weight_kg"]
    )
    out["change_minus_rolling_kg"] = change - out[f"rolling_{rolling}_period_change_kg"]
    out["weight_minus_hive_median_kg"] = out.period_weight_kg - out.hive_median_weight_kg

    dates = pd.to_datetime(out[PERIOD_END])
    out["year"] = dates.dt.year
    out["month"] = dates.dt.month
    out["week_of_year"] = dates.dt.isocalendar().week.astype(int)
    fraction = dates.dt.dayofyear / 365.25
    out["sin_year"] = np.sin(2 * np.pi * fraction)
    out["cos_year"] = np.cos(2 * np.pi * fraction)
    out = add_season(out, date_column=PERIOD_END)

    out["period_hive_change_mad_kg"] = grouped["period_weight_kg"].transform(
        lambda s: s.diff().abs().median()
    )
    return out.reset_index(drop=True)


PERIOD_HISTORY_FEATURES = [
    "period_weight_kg",
    "previous_period_weight_kg",
    "period_weight_change_kg",
    "previous_period_change_kg",
    "rolling_4_period_change_kg",
    "rolling_4_period_change_sd_kg",
    "rolling_4_period_weight_kg",
    "weight_minus_rolling_4_kg",
    "change_minus_rolling_kg",
    "within_period_weight_sd_kg",
    "within_period_slope_kg_per_day",
    "daily_change_sd_kg",
    "daily_change_max_abs_kg",
    "days_observed",
    "previous_period_days_observed",
    "month",
    "week_of_year",
    "sin_year",
    "cos_year",
]

PERIOD_SENSOR_FEATURES = (
    [f"{column}_mean" for column in SENSOR_COLUMNS]
    + [f"{column}_std" for column in SENSOR_COLUMNS]
    + ["avg_outside_temp_max", "foraging_days"]
)

PERIOD_EVENT_FEATURES = [f"days_since_{event}_event" for event in EVENT_TYPES]

PERIOD_HIVE_FEATURES = [
    "hive_median_weight_kg",
    "hive_mad_weight_kg",
    "weight_minus_hive_median_kg",
    "latitude",
    "longitude",
]

PERIOD_FEATURE_SETS: dict[str, list[str]] = {
    "history": PERIOD_HISTORY_FEATURES,
    "history+sensors": PERIOD_HISTORY_FEATURES + PERIOD_SENSOR_FEATURES,
    "history+sensors+events": (
        PERIOD_HISTORY_FEATURES + PERIOD_SENSOR_FEATURES + PERIOD_EVENT_FEATURES
    ),
    "history+sensors+events+hive": (
        PERIOD_HISTORY_FEATURES
        + PERIOD_SENSOR_FEATURES
        + PERIOD_EVENT_FEATURES
        + PERIOD_HIVE_FEATURES
    ),
}


@dataclass(frozen=True)
class PeriodMatrix:
    X: pd.DataFrame
    y: pd.Series
    meta: pd.DataFrame
    frame: pd.DataFrame
    feature_names: list[str]

    def __len__(self) -> int:
        return len(self.X)


def build_period_matrix(
    frame: pd.DataFrame,
    feature_set: str | list[str] = "history+sensors",
    target: str = PERIOD_TARGET,
    allow: frozenset[str] | set[str] | None = None,
) -> PeriodMatrix:
    """X, y, and the metadata the segmented report needs, at period grain.

    Rows without a target are dropped -- at this grain that is a contiguity failure, not
    a missing measurement. NaN features are left in place for the native-NaN models; the
    daily `group_ffill` path is deliberately not offered, because forward-filling a week
    carries seven days of one hive's state into the next observed week.
    """
    names = (
        PERIOD_FEATURE_SETS[feature_set] if isinstance(feature_set, str) else list(feature_set)
    )
    missing = [column for column in names if column not in frame.columns]
    if missing:
        raise KeyError(f"period feature set is missing columns: {missing}")
    assert_no_period_leakage(names, allow=allow)

    usable = frame.dropna(subset=[target]).reset_index(drop=True)
    X = usable[names].astype(float)
    assert_no_period_leakage(X.columns, allow=allow)

    meta_columns = [
        GROUP,
        PERIOD_KEY,
        PERIOD_START,
        PERIOD_END,
        "target_period_end_date",
        "year",
        "month",
        "season",
        "days_observed",
        "coverage",
    ]
    meta = usable[[c for c in meta_columns if c in usable.columns]].copy()
    return PeriodMatrix(
        X=X,
        y=usable[target].astype(float),
        meta=meta,
        frame=usable,
        feature_names=list(X.columns),
    )


# ---------------------------------------------------------------------------
# Splitting, with the embargo the daily splitter does not need
# ---------------------------------------------------------------------------


def rolling_origin_period_cv(
    frame: pd.DataFrame,
    n_splits: int = 4,
    horizon_months: int = 6,
    min_train_months: int = 12,
    embargo: bool = True,
) -> list[Fold]:
    """Expanding-window folds with a one-period embargo at every boundary.

    The daily splitter can cut on the measurement date because tomorrow's label lands one
    day later. Here a period row observed just before the origin carries a label measured
    over the period *after* it, which straddles or falls inside the test window. Training
    on it is training on the test set.

    The rule is stated on the label, not on the feature date: a training row is admitted
    only when its `target_period_end_date` falls strictly before the first test
    observation. Test rows are those whose period starts on or after the origin, so no
    period is scored twice or scored on partly-seen data.

    `embargo=False` reproduces the daily rule -- admit any row whose *features* predate
    the origin -- so the notebook can measure what the embargo is worth rather than assert
    that it matters. It is not a supported configuration for reporting a result, and the
    split-integrity assertion is skipped for it because it is precisely what it catches.
    """
    starts = pd.to_datetime(frame[PERIOD_START])
    target_ends = pd.to_datetime(frame["target_period_end_date"])
    first, last = starts.min(), starts.max()

    origins: list[pd.Timestamp] = []
    origin = (first + pd.DateOffset(months=min_train_months)).normalize()
    while origin + pd.DateOffset(months=horizon_months) <= last + pd.Timedelta(days=1):
        origins.append(origin)
        origin = origin + pd.DateOffset(months=horizon_months)
    if not origins:
        raise ValueError(
            f"date range {first.date()}..{last.date()} is too short for "
            f"{min_train_months}-month training plus a {horizon_months}-month horizon"
        )
    origins = origins[-n_splits:] if n_splits and n_splits < len(origins) else origins

    folds: list[Fold] = []
    for number, origin in enumerate(origins, start=1):
        horizon_end = origin + pd.DateOffset(months=horizon_months)
        test_mask = ((starts >= origin) & (starts < horizon_end)).to_numpy()
        if test_mask.sum() == 0:
            continue
        test_start = starts[test_mask].min()
        naive_train = (starts < origin).to_numpy()
        train_mask = (target_ends < test_start).to_numpy() if embargo else naive_train
        if train_mask.sum() == 0:
            continue

        embargoed = int((naive_train & ~train_mask).sum())
        fold = Fold(
            name=f"fold_{number}_{origin.date()}",
            train_index=np.flatnonzero(train_mask),
            test_index=np.flatnonzero(test_mask),
            train_end=target_ends[train_mask].max(),
            test_start=test_start,
            test_end=starts[test_mask].max(),
            extra={"embargoed_rows": embargoed},
        )
        if embargo:
            assert_period_split_integrity(frame, fold)
        folds.append(fold)
    return folds


def assert_period_split_integrity(frame: pd.DataFrame, fold: Fold) -> None:
    """Fail if any training label was measured on or after the first test observation.

    Deliberately stricter than the daily check, which compares feature dates. Comparing
    feature dates at this grain passes a split whose training labels were measured inside
    the test window.
    """
    target_ends = pd.to_datetime(frame["target_period_end_date"]).to_numpy()
    starts = pd.to_datetime(frame[PERIOD_START]).to_numpy()
    train_label_end = target_ends[fold.train_index].max()
    test_first = starts[fold.test_index].min()
    if not train_label_end < test_first:
        raise AssertionError(
            f"{fold.name}: a training label ends {pd.Timestamp(train_label_end).date()} but "
            f"the test window opens {pd.Timestamp(test_first).date()} -- the label was "
            "measured inside the test period"
        )


# ---------------------------------------------------------------------------
# Baselines -- re-set, because the daily bar does not transfer
# ---------------------------------------------------------------------------


def period_baselines(
    frame: pd.DataFrame, fold: Fold, target: str = PERIOD_TARGET
) -> dict[str, np.ndarray]:
    """Six naive rules at period grain.

    `month_climatology` is the addition that matters. At daily grain seasonality is a
    weak per-row signal and predicting zero dominates; over a week or a month the seasonal
    mean *is* most of the answer, so a model can post a large skill score against
    predict-zero while having learned nothing except that July gains and November loses.
    Scoring against the best of these makes that impossible.
    """
    train = frame.iloc[fold.train_index]
    test = frame.iloc[fold.test_index]
    n_test = len(test)
    global_mean = float(train[target].mean())

    predictions = {
        "predict_zero": np.zeros(n_test),
        "train_mean": np.full(n_test, global_mean),
    }

    persistence = test["period_weight_change_kg"].to_numpy(dtype=float)
    predictions["persistence"] = np.nan_to_num(persistence, nan=0.0)

    hive_mean = train.groupby(GROUP)[target].mean()
    predictions["hive_mean"] = (
        test[GROUP].map(hive_mean).fillna(global_mean).to_numpy(dtype=float)
    )

    month_mean = train.groupby("month")[target].mean()
    predictions["month_climatology"] = (
        test["month"].map(month_mean).fillna(global_mean).to_numpy(dtype=float)
    )

    hive_month_mean = train.groupby([GROUP, "month"])[target].mean()
    keys = list(zip(test[GROUP], test["month"]))
    predictions["hive_month_climatology"] = np.array(
        [hive_month_mean.get(key, month_mean.get(key[1], global_mean)) for key in keys],
        dtype=float,
    )
    return predictions


def period_baseline_board(
    frame: pd.DataFrame, folds: list[Fold], target: str = PERIOD_TARGET
) -> pd.DataFrame:
    rows = []
    for fold in folds:
        y_true = frame.iloc[fold.test_index][target].to_numpy(dtype=float)
        for name, y_pred in period_baselines(frame, fold, target=target).items():
            rows.append({"fold": fold.name, "model": name, **regression_metrics(y_true, y_pred)})
    return pd.DataFrame(rows)


def evaluate_period_folds(
    matrix: PeriodMatrix,
    folds: list[Fold],
    make_model,
    model_name: str,
    baseline_board: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Fit `make_model()` per fold and return a tidy per-fold result table with skill."""
    rows = []
    for fold in folds:
        model = make_model()
        model.fit(matrix.X.iloc[fold.train_index], matrix.y.iloc[fold.train_index])
        y_pred = model.predict(matrix.X.iloc[fold.test_index])
        y_true = matrix.y.iloc[fold.test_index].to_numpy(dtype=float)
        metrics = regression_metrics(y_true, y_pred)
        row = {"fold": fold.name, "model": model_name, **metrics}
        if baseline_board is not None:
            bar = float(baseline_board[baseline_board.fold == fold.name].mae.min())
            row["skill_vs_naive"] = skill_score(metrics["mae"], bar)
            row["baseline_mae"] = bar
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Extremes at the new grain
# ---------------------------------------------------------------------------


def relative_extreme_threshold(y: pd.Series, sigma: float = 2.7) -> float:
    """The absolute threshold that reproduces the daily +/-5 kg rule in sd units.

    5 kg is 2.7 sd of the daily target. Carrying the same 5 kg to weekly grain, where the
    sd is 5.0, redefines "extreme" as anything past 1.0 sd -- roughly a third of rows
    instead of 1.7%. Either the threshold moves or the two-stage pipeline is modelling a
    different thing under the same name.
    """
    return float(sigma * pd.Series(y).std())


def period_threshold_sweep(
    frame: pd.DataFrame, thresholds: list[float], target: str = PERIOD_TARGET
) -> pd.DataFrame:
    """Positive rate and class balance across candidate thresholds at this grain."""
    y = frame[target].dropna()
    rows = []
    for threshold in thresholds:
        positive = (y.abs() > threshold).sum()
        rows.append(
            {
                "threshold_kg": threshold,
                "n_extreme": int(positive),
                "positive_rate_pct": round(100 * positive / len(y), 2),
                "threshold_in_sd": round(threshold / y.std(), 2),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-hive relative extremes -- Milestone 4 alt, Section 7.3 item 4
# ---------------------------------------------------------------------------

#: The hive's own period-to-period change MAD, computed in `add_period_features`. The
#: daily table's analogue is `hive_target_mad_kg`, which item 4 named; at period grain the
#: daily column measures the wrong quantity, so the period version is used instead.
HIVE_SCALE_COLUMN = "period_hive_change_mad_kg"


def hive_relative_threshold_kg(
    frame: pd.DataFrame, multiplier: float = 5.0, scale_column: str = HIVE_SCALE_COLUMN
) -> pd.Series:
    """Per-row extreme threshold in kg: `multiplier` x the hive's own change MAD.

    Section 7.3 item 4: 11.1 kg on a 10 kg nucleus and on a 100 kg production hive are not
    the same event. The absolute rule does not define "extreme", it defines "large", and
    large is a property of the hive as much as of the week.
    """
    return multiplier * frame[scale_column].astype(float)


def relative_extreme_flag(
    frame: pd.DataFrame,
    multiplier: float = 5.0,
    target: str = PERIOD_TARGET,
    scale_column: str = HIVE_SCALE_COLUMN,
) -> pd.Series:
    """Boolean label under the per-hive rule. NaN scale yields False, not NaN.

    A hive with no estimable change MAD -- too few contiguous periods -- cannot have a
    relative extreme defined for it, and treating that as "not extreme" is the
    conservative reading: it keeps the row in the denominator and out of the positives,
    so nothing is scored on a label that was never computed.
    """
    threshold = hive_relative_threshold_kg(frame, multiplier, scale_column)
    return (frame[target].abs() > threshold).fillna(False)


def calibrate_relative_multiplier(
    frame: pd.DataFrame,
    target_rate: float,
    target: str = PERIOD_TARGET,
    scale_column: str = HIVE_SCALE_COLUMN,
    bounds: tuple[float, float] = (0.5, 60.0),
    tolerance: float = 1e-4,
) -> float:
    """The multiplier whose positive rate matches `target_rate`.

    Without this the comparison in Section 5 is not a comparison. PR-AUC is bounded below
    by the positive rate, so a relative rule that fires on 13% of weeks will post a much
    higher PR-AUC than an absolute rule firing on 3% while being a *worse* detector. Fixing
    the positive rate makes the two labels differ only in *which* weeks they select, which
    is the question. Bisection, because the positive rate is a monotone step function of
    the multiplier and anything cleverer would be pretending it is smooth.
    """
    low, high = bounds
    for _ in range(80):
        middle = 0.5 * (low + high)
        rate = float(relative_extreme_flag(frame, middle, target, scale_column).mean())
        if abs(rate - target_rate) < tolerance:
            return middle
        if rate > target_rate:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def extreme_label_comparison(
    frame: pd.DataFrame,
    absolute_kg: float,
    multiplier: float,
    target: str = PERIOD_TARGET,
    scale_column: str = HIVE_SCALE_COLUMN,
) -> pd.DataFrame:
    """How the absolute and relative labels differ, at whatever rates they are set to.

    `hives_with_no_positive` is the column that carries the argument. An absolute rule
    concentrates its positives on the heaviest, most productive hives and never fires at
    all on the small ones, so a classifier trained on it learns "which hive is this"
    rather than "is something unusual happening". The relative rule spreads the same
    number of positives across the fleet.
    """
    usable = frame.dropna(subset=[target])
    absolute = usable[target].abs() > absolute_kg
    relative = relative_extreme_flag(usable, multiplier, target, scale_column)
    n_hives = usable[GROUP].nunique()

    rows = []
    for name, label in (("absolute", absolute), ("relative", relative)):
        by_hive = label.groupby(usable[GROUP]).sum()
        rows.append(
            {
                "rule": name,
                "definition": (
                    f"|change| > {absolute_kg:.2f} kg"
                    if name == "absolute"
                    else f"|change| > {multiplier:.2f} x hive change MAD"
                ),
                "n_positive": int(label.sum()),
                "positive_rate_pct": round(100 * float(label.mean()), 2),
                "hives_with_a_positive": int((by_hive > 0).sum()),
                "hives_with_no_positive": int(n_hives - (by_hive > 0).sum()),
                "max_positives_on_one_hive": int(by_hive.max()),
                "share_on_top_5_hives": round(
                    float(by_hive.sort_values(ascending=False).head(5).sum() / max(label.sum(), 1)), 3
                ),
                "median_hive_weight_of_positives_kg": round(
                    float(usable.loc[label.to_numpy(), "hive_median_weight_kg"].median()), 1
                )
                if label.any()
                else np.nan,
            }
        )
    table = pd.DataFrame(rows)
    table.attrs["agreement"] = float((absolute & relative).sum())
    return table


def relative_threshold_sweep(
    frame: pd.DataFrame,
    multipliers: tuple[float, ...] = (2, 3, 4, 5, 6, 8),
    target: str = PERIOD_TARGET,
    scale_column: str = HIVE_SCALE_COLUMN,
) -> pd.DataFrame:
    """Positive rate and fleet coverage across candidate multipliers."""
    usable = frame.dropna(subset=[target])
    n_hives = usable[GROUP].nunique()
    rows = []
    for multiplier in multipliers:
        label = relative_extreme_flag(usable, float(multiplier), target, scale_column)
        by_hive = label.groupby(usable[GROUP]).sum()
        rows.append(
            {
                "multiplier": float(multiplier),
                "n_extreme": int(label.sum()),
                "positive_rate_pct": round(100 * float(label.mean()), 2),
                "hives_covered": int((by_hive > 0).sum()),
                "hives_total": int(n_hives),
                "median_threshold_kg": round(
                    float(hive_relative_threshold_kg(usable, float(multiplier), scale_column).median()), 2
                ),
            }
        )
    return pd.DataFrame(rows)
