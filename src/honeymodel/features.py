"""Feature matrices, with leakage made impossible rather than merely avoided.

`honey_model` deliberately ships columns that are only knowable after the prediction
date -- `next_day_weight_kg`, `next_observation_date`, every `*_next_dif`, and every
label-derived flag. They are there so the extreme-event investigation can run at the
modelling grain. A single `X = df.drop(columns=[TARGET])` would turn any of them into a
perfect-score model, so `build_feature_matrix` refuses to return a matrix containing one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from honeymodel.data import DATE, GROUP, TARGET

#: Columns that must never reach a feature matrix.
#:
#: Three kinds of poison:
#:   1. the target and its direct components (next_day_weight_kg is target + today);
#:   2. anything measured after the prediction date (*_next_dif, next_observation_date);
#:   3. anything derived from the label (the extreme flags, which are ABS(target) > k).
FORBIDDEN_COLUMNS = frozenset(
    {
        TARGET,
        "next_day_weight_kg",
        "next_observation_date",
        "days_to_next_observation",
        "honey_next_dif",
        "feeding_next_dif",
        "swarming_next_dif",
        "treatment_next_dif",
        "died_next_dif",
        "queencell_next_dif",
        "nearest_honey_event_days",
        "nearest_feeding_event_days",
        "nearest_swarming_event_days",
        "nearest_treatment_event_days",
        "nearest_died_event_days",
        "nearest_queencell_event_days",
        "extreme_weight_change_flag",
        "extreme_weight_change_flag_3",
        "extreme_weight_change_flag_5",
        "extreme_weight_change_flag_7",
        "extreme_weight_change_flag_10",
        "extreme_relative_flag",
        "sensor_dropout_flag",  # defined partly from the next-day change
        "source_weight_no_outlier_kg",  # a cumulative series that encodes future deltas
    }
)

# ---------------------------------------------------------------------------
# Feature sets -- the ablation ladder. Each rung adds one source of information,
# so a gain can be attributed to the data rather than to the algorithm.
# ---------------------------------------------------------------------------

HISTORY_FEATURES = [
    "previous_day_weight_kg",
    "rolling_3_day_weight_kg",
    "rolling_7_day_weight_kg",
    "previous_day_weight_change_kg",
    "weight_minus_rolling_3_kg",
    "weight_minus_rolling_7_kg",
    "weight_slope_2_day_kg",
    "weight_slope_7_day_kg",
    "rolling_7_day_weight_sd_kg",
    "day_of_year",
    "month",
    "sin_day_of_year",
    "cos_day_of_year",
]

SENSOR_FEATURES = [
    "avg_internal_temp_1",
    "avg_internal_temp_2",
    "avg_internal_temp_3",
    "avg_internal_temp_4",
    "avg_outside_temp",
    "avg_humidity",
    "avg_pressure",
]

HIVE_CONTEXT_FEATURES = [
    "end_of_day_weight_kg",
    "weight_minus_hive_median_kg",
    "hive_median_weight_kg",
    "hive_mad_weight_kg",
    "hive_change_mad_kg",
    "latitude",
    "longitude",
]

WEATHER_FEATURES = [
    "tmax_c",
    "tmin_c",
    "prcp_mm",
    "tmax_c_lag_1",
    "tmin_c_lag_1",
    "prcp_mm_lag_1",
    "tmax_c_rolling_3",
    "prcp_mm_rolling_3",
    "prcp_mm_rolling_7",
]

FEATURE_SETS: dict[str, list[str]] = {
    "history": HISTORY_FEATURES,
    "history+sensors": HISTORY_FEATURES + SENSOR_FEATURES,
    "history+sensors+weather": HISTORY_FEATURES + SENSOR_FEATURES + WEATHER_FEATURES,
    "history+sensors+weather+hive": (
        HISTORY_FEATURES + SENSOR_FEATURES + WEATHER_FEATURES + HIVE_CONTEXT_FEATURES
    ),
    # Milestone 3's feature list, kept so the new results are comparable to the old ones.
    "milestone_3": [
        "previous_day_weight_kg",
        "rolling_3_day_weight_kg",
        "rolling_7_day_weight_kg",
        "avg_internal_temp_1",
        "avg_internal_temp_2",
        "avg_internal_temp_3",
        "avg_internal_temp_4",
        "avg_outside_temp",
        "avg_humidity",
        "avg_pressure",
        "day_of_year",
        "month",
    ],
}


class LeakageError(AssertionError):
    """Raised when a forbidden column reaches a feature matrix."""


def assert_no_leakage(columns: object) -> None:
    """Fail loudly if any forbidden column is present.

    Called by `build_feature_matrix`, and worth calling directly on anything hand-rolled.
    """
    offenders = sorted(FORBIDDEN_COLUMNS.intersection(set(columns)))
    if offenders:
        raise LeakageError(
            "these columns are only knowable after the prediction date, or are derived "
            f"from the label, and must not be used as features: {offenders}"
        )


@dataclass(frozen=True)
class FeatureMatrix:
    """A feature matrix plus everything needed to evaluate and segment it."""

    X: pd.DataFrame
    y: pd.Series
    meta: pd.DataFrame  # hive_id, measurement_date, season, and the label-derived flags
    feature_names: list[str]

    def __len__(self) -> int:
        return len(self.X)


def group_aware_impute(
    frame: pd.DataFrame,
    columns: list[str],
    group: str = GROUP,
    add_indicators: bool = True,
) -> pd.DataFrame:
    """Forward-fill within each hive, never across hives.

    Milestone 3 called `model_df[features].ffill()` on a frame ordered by
    (hive_id, measurement_date), which carries the last reading of one hive into the
    first rows of the next. With 23-35% sensor missingness that is not an edge case.

    Missingness is informative here -- a dead sensor is a fact about the hive-day -- so
    every imputed column gets a `*_was_missing` indicator alongside it.
    """
    filled = frame[columns].copy()
    indicators = filled.isna().astype("int8").add_suffix("_was_missing")
    filled = filled.groupby(frame[group], sort=False).ffill()
    if add_indicators:
        keep = [c for c in indicators.columns if indicators[c].any()]
        filled = pd.concat([filled, indicators[keep]], axis=1)
    return filled


def build_feature_matrix(
    frame: pd.DataFrame,
    feature_set: str | list[str] = "history+sensors",
    impute: str = "group_ffill",
    target: str = TARGET,
) -> FeatureMatrix:
    """Assemble X, y and the metadata needed for segmented evaluation.

    Parameters
    ----------
    feature_set : a key of FEATURE_SETS, or an explicit column list.
    impute : "group_ffill" -- per-hive forward fill plus `*_was_missing` indicators;
             "none" -- leave NaN in place, for models that consume it natively
             (HistGradientBoosting, LightGBM). "none" is preferred where possible: it
             avoids both the boundary bleed and the row loss.
    """
    names = FEATURE_SETS[feature_set] if isinstance(feature_set, str) else list(feature_set)
    missing = [c for c in names if c not in frame.columns]
    if missing:
        raise KeyError(
            f"feature set is missing columns not present in the table: {missing}. "
            "Weather features require the honey_weather join (see scripts/fetch_noaa_weather.py)."
        )
    assert_no_leakage(names)

    if impute == "group_ffill":
        X = group_aware_impute(frame, names)
    elif impute == "none":
        X = frame[names].copy()
    else:
        raise ValueError(f"unknown impute strategy: {impute!r}")

    # A leading NaN survives a forward fill (nothing precedes it). Fill those with the
    # training-agnostic per-hive median so no row is silently dropped.
    if impute == "group_ffill":
        base = [c for c in X.columns if not c.endswith("_was_missing")]
        hive_median = X[base].groupby(frame[GROUP], sort=False).transform("median")
        X[base] = X[base].fillna(hive_median).fillna(X[base].median())

    assert_no_leakage(X.columns)

    meta_columns = [GROUP, DATE, "year", "month", "season"]
    meta_columns += [c for c in frame.columns if c.startswith("extreme_") or c.endswith("_flag")]
    meta = frame[[c for c in dict.fromkeys(meta_columns) if c in frame.columns]].copy()

    return FeatureMatrix(
        X=X.reset_index(drop=True),
        y=frame[target].reset_index(drop=True),
        meta=meta.reset_index(drop=True),
        feature_names=list(X.columns),
    )


def weight_change_decile(y: pd.Series) -> pd.Series:
    """Decile of |weight change|, for the segmented report.

    Pooled error hides the shape that matters: a model can look good simply by being
    right about the many near-zero days.
    """
    magnitude = y.abs()
    return pd.qcut(magnitude.rank(method="first"), 10, labels=[f"D{i}" for i in range(1, 11)])


def summarise_missingness(frame: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Per-column missing counts and rates, for the data-preparation write-up."""
    subset = frame[columns] if columns else frame
    missing = subset.isna().sum()
    return (
        pd.DataFrame(
            {
                "column": missing.index,
                "missing": missing.to_numpy(),
                "missing_pct": np.round(100 * missing.to_numpy() / len(subset), 2),
            }
        )
        .sort_values("missing", ascending=False)
        .reset_index(drop=True)
    )
