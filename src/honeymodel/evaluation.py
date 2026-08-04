"""Validation framework.

Milestone 3 reported single-split, single-seed, pooled R2 on a test set filtered by the
label. Every function here exists to remove one of those words.

    date_chronological_split   split on a date, not a row index
    rolling_origin_cv          expanding-window folds, mean +/- sd
    grouped_hive_cv            leave-hives-out: does this work on a new hive?
    naive_baselines            the bar a learned model has to clear
    skill_score                % error reduction against that bar
    segmented_report           where the model works and where it fails
    seed_stability             is the algorithm ranking inside the noise?
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterator, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from honeymodel.data import DATE, GROUP, TARGET, add_season

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float]:
    """MAE, RMSE, R2, bias and n.

    MAE leads. R2 is normalised by the variance of whatever window it is computed on,
    and this target's variance is strongly seasonal -- a winter fold can show a poor R2
    with excellent absolute error, and a summer fold the reverse. R2 is reported per
    fold and per segment, never as a single pooled headline.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) == 0:
        return {"mae": np.nan, "rmse": np.nan, "r2": np.nan, "bias": np.nan, "n": 0}
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
        "bias": float(np.mean(y_pred - y_true)),
        "n": int(len(y_true)),
    }


def skill_score(model_mae: float, baseline_mae: float) -> float:
    """Fractional MAE reduction against a baseline. 0 means no better than the baseline.

    Positive is an improvement, negative means the baseline wins. This is the number
    that answers "is the model learning anything?", which a raw R2 near 0.2 does not.
    """
    if baseline_mae in (0, np.nan) or np.isnan(baseline_mae):
        return np.nan
    return float(1.0 - (model_mae / baseline_mae))


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Fold:
    """One train/test split, with the metadata needed to describe it in a table."""

    name: str
    train_index: np.ndarray
    test_index: np.ndarray
    train_end: pd.Timestamp | None = None
    test_start: pd.Timestamp | None = None
    test_end: pd.Timestamp | None = None
    extra: dict = field(default_factory=dict)

    def describe(self) -> dict:
        return {
            "fold": self.name,
            "n_train": len(self.train_index),
            "n_test": len(self.test_index),
            "train_end": self.train_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
            **self.extra,
        }


def date_chronological_split(
    frame: pd.DataFrame,
    cutoff_date: str | pd.Timestamp,
    date_column: str = DATE,
) -> Fold:
    """Split on a calendar date: train strictly before the cutoff, test from it onward.

    Milestone 3 used `sort_values(date).iloc[:k]`, which cuts on a row index. Because
    78 hives share every date, that puts the same date on both sides of the boundary --
    train_end and test_start were literally the same day. Splitting on the date itself
    makes that impossible.
    """
    cutoff = pd.Timestamp(cutoff_date)
    dates = pd.to_datetime(frame[date_column])
    train_mask = dates < cutoff
    test_mask = ~train_mask

    train_index = np.flatnonzero(train_mask.to_numpy())
    test_index = np.flatnonzero(test_mask.to_numpy())
    if len(train_index) == 0 or len(test_index) == 0:
        raise ValueError(f"cutoff {cutoff.date()} leaves one side of the split empty")

    fold = Fold(
        name=f"cutoff_{cutoff.date()}",
        train_index=train_index,
        test_index=test_index,
        train_end=dates[train_mask].max(),
        test_start=dates[test_mask].min(),
        test_end=dates[test_mask].max(),
    )
    assert_split_integrity(frame, fold, date_column)
    return fold


def rolling_origin_cv(
    frame: pd.DataFrame,
    n_splits: int = 4,
    horizon_months: int = 6,
    min_train_months: int = 12,
    date_column: str = DATE,
) -> list[Fold]:
    """Expanding-window folds: train on everything up to T, test on the next horizon.

    Each fold trains on strictly more history than the last, which is how the model
    would actually be used. Report mean +/- sd across folds; a single split on this
    dataset is a one-season estimate.
    """
    dates = pd.to_datetime(frame[date_column])
    first, last = dates.min(), dates.max()

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
        train_mask = (dates < origin).to_numpy()
        test_mask = ((dates >= origin) & (dates < horizon_end)).to_numpy()
        if test_mask.sum() == 0 or train_mask.sum() == 0:
            continue
        fold = Fold(
            name=f"fold_{number}_{origin.date()}",
            train_index=np.flatnonzero(train_mask),
            test_index=np.flatnonzero(test_mask),
            train_end=dates[train_mask].max(),
            test_start=dates[test_mask].min(),
            test_end=dates[test_mask].max(),
        )
        assert_split_integrity(frame, fold, date_column)
        folds.append(fold)
    return folds


def grouped_hive_cv(
    frame: pd.DataFrame,
    n_folds: int = 5,
    group_column: str = GROUP,
    seed: int = 42,
) -> list[Fold]:
    """Leave-hives-out folds. Answers "does this generalise to a hive we never instrumented?"

    The temporal split has all 78 hives on both sides, and the features include absolute
    weights, so a model can lean on hive identity. The gap between this and the temporal
    result is the honest answer to the deployment question.
    """
    hives = np.sort(frame[group_column].unique())
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(hives)
    blocks = np.array_split(shuffled, n_folds)

    folds: list[Fold] = []
    values = frame[group_column].to_numpy()
    for number, held_out in enumerate(blocks, start=1):
        test_mask = np.isin(values, held_out)
        if test_mask.sum() == 0 or (~test_mask).sum() == 0:
            continue
        folds.append(
            Fold(
                name=f"hives_{number}",
                train_index=np.flatnonzero(~test_mask),
                test_index=np.flatnonzero(test_mask),
                extra={"held_out_hives": len(held_out)},
            )
        )
    return folds


def assert_split_integrity(frame: pd.DataFrame, fold: Fold, date_column: str = DATE) -> None:
    """Fail if any training date is on or after the earliest test date.

    Strict inequality -- a shared boundary date is exactly the Milestone 3 bug.
    """
    dates = pd.to_datetime(frame[date_column]).to_numpy()
    train_max = dates[fold.train_index].max()
    test_min = dates[fold.test_index].min()
    if not train_max < test_min:
        raise AssertionError(
            f"{fold.name}: train ends {pd.Timestamp(train_max).date()} but test starts "
            f"{pd.Timestamp(test_min).date()} -- the boundary date is on both sides"
        )


# ---------------------------------------------------------------------------
# Naive baselines
# ---------------------------------------------------------------------------


def naive_baselines(
    frame: pd.DataFrame,
    fold: Fold,
    target: str = TARGET,
    group_column: str = GROUP,
    date_column: str = DATE,
) -> dict[str, np.ndarray]:
    """Predictions from five naive rules, fitted on train and applied to test.

    The target has mean ~0.002 kg, so `predict_zero` is a genuinely strong competitor.
    Any learned model that cannot beat these is not learning the target, and that is a
    reportable result rather than a hidden one.

    predict_zero          tomorrow's change is zero
    train_mean            the training-set mean change
    persistence           tomorrow repeats today's observed change
    hive_mean             each hive's own mean change in training
    hive_month_mean       each hive's mean change for that calendar month (climatology)
    """
    train = frame.iloc[fold.train_index]
    test = frame.iloc[fold.test_index]
    n_test = len(test)

    predictions = {
        "predict_zero": np.zeros(n_test),
        "train_mean": np.full(n_test, float(train[target].mean())),
    }

    if "previous_day_weight_change_kg" in test.columns:
        persistence = test["previous_day_weight_change_kg"].to_numpy(dtype=float)
        predictions["persistence"] = np.nan_to_num(persistence, nan=0.0)

    global_mean = float(train[target].mean())

    hive_mean = train.groupby(group_column)[target].mean()
    predictions["hive_mean"] = (
        test[group_column].map(hive_mean).fillna(global_mean).to_numpy(dtype=float)
    )

    months = pd.to_datetime(train[date_column]).dt.month
    hive_month_mean = train.groupby([train[group_column], months])[target].mean()
    test_keys = list(zip(test[group_column], pd.to_datetime(test[date_column]).dt.month))
    predictions["hive_month_mean"] = np.array(
        [hive_month_mean.get(key, global_mean) for key in test_keys], dtype=float
    )

    return predictions


def baseline_board(
    frame: pd.DataFrame,
    folds: Sequence[Fold],
    target: str = TARGET,
) -> pd.DataFrame:
    """Every naive rule, every fold. The bar the model board is scored against."""
    rows = []
    for fold in folds:
        y_true = frame.iloc[fold.test_index][target].to_numpy(dtype=float)
        for name, y_pred in naive_baselines(frame, fold, target=target).items():
            rows.append({"fold": fold.name, "model": name, **regression_metrics(y_true, y_pred)})
    return pd.DataFrame(rows)


def best_baseline_mae(board: pd.DataFrame, fold_name: str | None = None) -> float:
    """Lowest naive MAE -- the denominator for every skill score."""
    subset = board if fold_name is None else board[board["fold"] == fold_name]
    return float(subset["mae"].min())


# ---------------------------------------------------------------------------
# Segmented reporting
# ---------------------------------------------------------------------------


def segmented_report(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    meta: pd.DataFrame,
    by: Sequence[str] = ("season", "month", GROUP),
    baseline_mae: float | None = None,
) -> pd.DataFrame:
    """Metrics broken out by one or more segments, in a tidy frame ready to plot.

    Pooled numbers average over regimes that behave nothing alike: winter dormancy is
    low-variance and easy, peak nectar flow is high-variance and useful. Reporting only
    the pooled value hides the fact that the model is worst exactly where it matters.

    `by` accepts column names present in `meta`, plus the derived segment
    "weight_change_decile" (decile of |actual change|).
    """
    frame = meta.reset_index(drop=True).copy()
    frame["y_true"] = np.asarray(y_true, dtype=float)
    frame["y_pred"] = np.asarray(y_pred, dtype=float)

    if "season" in by and "season" not in frame.columns and DATE in frame.columns:
        frame = add_season(frame)
    if "weight_change_decile" in by:
        from honeymodel.features import weight_change_decile

        frame["weight_change_decile"] = weight_change_decile(frame["y_true"])

    rows = []
    for segment in by:
        if segment not in frame.columns:
            continue
        for value, group in frame.groupby(segment, observed=True):
            metrics = regression_metrics(group["y_true"], group["y_pred"])
            row = {"segment": segment, "value": value, **metrics}
            if baseline_mae is not None:
                row["skill_vs_naive"] = skill_score(metrics["mae"], baseline_mae)
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["segment", "value"]).reset_index(drop=True)


def fold_summary(results: pd.DataFrame, by: str = "model") -> pd.DataFrame:
    """Mean +/- sd across folds. A single fold's number is never the headline."""
    grouped = results.groupby(by)
    summary = grouped.agg(
        folds=("fold", "nunique"),
        mae_mean=("mae", "mean"),
        mae_sd=("mae", "std"),
        rmse_mean=("rmse", "mean"),
        rmse_sd=("rmse", "std"),
        r2_mean=("r2", "mean"),
        r2_sd=("r2", "std"),
        n=("n", "sum"),
    ).reset_index()
    if "skill_vs_naive" in results.columns:
        skill = grouped["skill_vs_naive"].agg(["mean", "std"])
        summary["skill_mean"] = skill["mean"].to_numpy()
        summary["skill_sd"] = skill["std"].to_numpy()
    return summary.sort_values("mae_mean").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Seed stability
# ---------------------------------------------------------------------------


def seed_stability(
    fit_predict: Callable[[int], tuple[np.ndarray, np.ndarray]],
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
) -> pd.DataFrame:
    """Run one model at several seeds and return per-seed metrics.

    Milestone 3 claimed Gradient Boosting beat Random Forest on dR2 = 0.018 from one
    split at one seed. If the seed-to-seed spread is that wide, there is no ranking.

    `fit_predict(seed)` must return (y_true, y_pred) for that seed.
    """
    rows = []
    for seed in seeds:
        y_true, y_pred = fit_predict(seed)
        rows.append({"seed": seed, **regression_metrics(y_true, y_pred)})
    return pd.DataFrame(rows)


def ranking_is_significant(metrics_a: pd.DataFrame, metrics_b: pd.DataFrame, metric: str = "mae") -> bool:
    """True when two models' seed distributions do not overlap on `metric`.

    Deliberately crude -- with 3-5 seeds a formal test is overkill. Non-overlapping
    ranges is the minimum bar for claiming one algorithm beats another.
    """
    a_lo, a_hi = metrics_a[metric].min(), metrics_a[metric].max()
    b_lo, b_hi = metrics_b[metric].min(), metrics_b[metric].max()
    return bool(a_hi < b_lo or b_hi < a_lo)


# ---------------------------------------------------------------------------
# Driving an experiment
# ---------------------------------------------------------------------------


def evaluate_folds(
    frame: pd.DataFrame,
    folds: Sequence[Fold],
    fit_predict: Callable[[pd.DataFrame, pd.DataFrame], np.ndarray],
    model_name: str,
    target: str = TARGET,
    baseline_board_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run one model across folds and return a tidy per-fold result table.

    `fit_predict(train_frame, test_frame)` returns predictions for the test rows.
    Passing the baseline board attaches a per-fold skill score, so no result is ever
    reported without the bar it cleared (or failed to).
    """
    rows = []
    for fold in folds:
        train = frame.iloc[fold.train_index]
        test = frame.iloc[fold.test_index]
        y_pred = fit_predict(train, test)
        metrics = regression_metrics(test[target].to_numpy(dtype=float), y_pred)
        row = {"fold": fold.name, "model": model_name, **metrics}
        if baseline_board_frame is not None:
            row["skill_vs_naive"] = skill_score(
                metrics["mae"], best_baseline_mae(baseline_board_frame, fold.name)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def iter_folds(frame: pd.DataFrame, folds: Sequence[Fold]) -> Iterator[tuple[Fold, pd.DataFrame, pd.DataFrame]]:
    """Yield (fold, train_frame, test_frame) for hand-rolled experiments."""
    for fold in folds:
        yield fold, frame.iloc[fold.train_index], frame.iloc[fold.test_index]
