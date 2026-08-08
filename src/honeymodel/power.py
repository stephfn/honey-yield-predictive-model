"""How much data the monthly result would need before it means anything.

Section 7.3 item 5 of `Milestone_4_alt.ipynb`:

    The monthly board's +12.3% is the best skill score in the notebook and the one least
    supported by its sample. Two more seasons would roughly double the monthly row count
    and make the fold-to-fold spread interpretable rather than merely small.

Two more seasons is not available -- the published archive ends 2022-12-30 and there is no
2023 to pull. What *is* available is the shape of the curve: refit the same protocol on
deliberately smaller samples, watch how fast the uncertainty in the skill estimate falls,
and read off the sample size at which it would stop overlapping zero. That turns "not yet
evidence" from an opinion into a number, and it is the only part of item 5 this dataset
can answer.

The resampling unit is the **hive**, not the row. Rows within a hive are a single
autocorrelated series and resampling them independently would make 622 monthly rows look
like 622 independent observations, which is the exact error the exercise is meant to
measure. Subsampling hives keeps each retained series intact and shrinks the sample in the
way adding or removing colonies would.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from honeymodel.data import GROUP
from honeymodel.evaluation import regression_metrics, skill_score

DEFAULT_FRACTIONS = (0.25, 0.4, 0.55, 0.7, 0.85, 1.0)


def sample_size_curve(
    frame: pd.DataFrame,
    target: str,
    feature_set: str,
    make_model,
    period: str = "month",
    fractions: tuple[float, ...] = DEFAULT_FRACTIONS,
    repeats: int = 8,
    seed: int = 42,
    n_splits: int = 4,
    horizon_months: int = 6,
    min_train_months: int = 12,
) -> pd.DataFrame:
    """Skill and its spread as a function of how many hives are in the sample.

    For each fraction, `repeats` independent hive subsamples are drawn, the full
    rolling-origin protocol is rebuilt *inside* each subsample -- folds, baselines and all
    -- and the mean skill across folds is recorded. The spread reported is the spread
    across those repeats: it is the sampling variability of the headline number, which is
    the quantity item 5 is asking about, and it is not the same as the fold-to-fold sd the
    model board already reports.

    Rebuilding the folds per subsample matters. Reusing the full-sample folds would hold
    the test windows fixed while the training set shrank, which measures something else.
    """
    from honeymodel.periods import (
        build_period_matrix,
        period_baselines,
        rolling_origin_period_cv,
    )

    rng = np.random.default_rng(seed)
    hives = np.sort(frame[GROUP].unique())
    rows = []

    for fraction in fractions:
        n_hives = max(3, int(round(fraction * len(hives))))
        draws = 1 if n_hives == len(hives) else repeats
        for repeat in range(draws):
            chosen = rng.choice(hives, size=n_hives, replace=False)
            subset = frame[frame[GROUP].isin(chosen)].reset_index(drop=True)
            try:
                matrix = build_period_matrix(subset, feature_set=feature_set, target=target)
                folds = rolling_origin_period_cv(
                    matrix.frame,
                    n_splits=n_splits,
                    horizon_months=horizon_months,
                    min_train_months=min_train_months,
                )
            except (ValueError, KeyError):
                continue
            if not folds:
                continue

            skills = []
            for fold in folds:
                if len(fold.train_index) < 30 or len(fold.test_index) < 10:
                    continue
                model = make_model()
                model.fit(matrix.X.iloc[fold.train_index], matrix.y.iloc[fold.train_index])
                y_pred = model.predict(matrix.X.iloc[fold.test_index])
                y_true = matrix.y.iloc[fold.test_index].to_numpy(dtype=float)
                bar = min(
                    regression_metrics(y_true, prediction)["mae"]
                    for prediction in period_baselines(matrix.frame, fold, target=target).values()
                )
                skills.append(skill_score(regression_metrics(y_true, y_pred)["mae"], bar))
            if not skills:
                continue
            rows.append(
                {
                    "fraction": fraction,
                    "repeat": repeat,
                    "hives": n_hives,
                    "rows": len(matrix),
                    "folds": len(skills),
                    "skill": float(np.mean(skills)),
                }
            )

    return pd.DataFrame(rows)


def summarise_curve(curve: pd.DataFrame) -> pd.DataFrame:
    """Mean skill and its across-repeat spread at each sample size."""
    summary = (
        curve.groupby("fraction")
        .agg(
            hives=("hives", "first"),
            rows_mean=("rows", "mean"),
            repeats=("skill", "size"),
            skill_mean=("skill", "mean"),
            skill_sd=("skill", "std"),
            skill_min=("skill", "min"),
            skill_max=("skill", "max"),
            # The legible power statement: how often a study of this size would have
            # concluded the model does not beat naive at all. Far easier to act on than a
            # standard error, and it needs no distributional assumption.
            share_negative=("skill", lambda s: float((s <= 0).mean())),
        )
        .reset_index()
    )
    summary["rows_mean"] = summary.rows_mean.round(0).astype(int)
    return summary


def required_sample_size(
    curve: pd.DataFrame, confidence_sd: float = 1.96
) -> dict[str, float]:
    """Project the sample size at which the skill estimate would clear zero.

    Fits `sd = c / sqrt(n)` by least squares on log-log, which is the standard-error
    scaling any sane estimator obeys asymptotically, then solves for the `n` where
    `skill_mean - confidence_sd * sd > 0`.

    This is an extrapolation and is reported as one. It assumes the effect size stays put
    as the sample grows -- which is exactly what a small sample cannot tell you, since the
    +12.3% is itself an estimate with the spread being measured here. The number is a
    lower bound on what would be needed, useful for saying "two more seasons is not
    obviously enough", and not a promise about what those seasons would show.
    """
    summary = summarise_curve(curve).dropna(subset=["skill_sd"])
    summary = summary[summary.skill_sd > 0]
    if len(summary) < 2:
        return {"fitted_c": np.nan, "required_rows": np.nan, "required_multiple": np.nan}

    slope_fit = np.polyfit(np.log(summary.rows_mean), np.log(summary.skill_sd), 1)
    observed_slope = float(slope_fit[0])
    # Refit with the exponent pinned to -1/2 so the projection is a standard-error
    # extrapolation rather than a two-parameter curve through six noisy points.
    c = float(np.exp(np.mean(np.log(summary.skill_sd) + 0.5 * np.log(summary.rows_mean))))

    full = summarise_curve(curve).iloc[-1]
    skill = float(full.skill_mean)
    current_rows = float(full.rows_mean)
    if skill <= 0:
        return {
            "fitted_c": c,
            "observed_sd_exponent": observed_slope,
            "current_rows": current_rows,
            "current_skill": skill,
            "required_rows": np.inf,
            "required_multiple": np.inf,
        }

    required = (confidence_sd * c / skill) ** 2
    return {
        "fitted_c": c,
        "observed_sd_exponent": observed_slope,
        "current_rows": current_rows,
        "current_skill": skill,
        "required_rows": float(required),
        "required_multiple": float(required / current_rows),
    }
