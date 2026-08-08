"""Season-conditional models.

Section 7.3 item 2 of `Milestone_4_alt.ipynb`:

    Skill is +62% in winter and -24% in summer with one pooled model. A season-conditional
    model, or a mixture with a seasonal gate, directly attacks the failure mode instead of
    averaging over it.

The premise is that a hive in July and a hive in January are two different processes
sharing a sensor. In July the weight series is dominated by nectar income and its variance
is several times the winter figure; from October the series is a slow monotone drain. One
model fitted across both spends its capacity on the average of two regimes that share no
mechanism, and the pooled skill score hides that it is winning on the easy half.

    SeasonGatedRegressor    one regressor per regime, hard-routed at predict time
    regime_of               the regime label, read off the `month` feature
    regime_report           per-regime metrics for any model, with the pooled row alongside

**The gate reads `month`, which is a feature, not the label.** That is what makes this
legitimate: the calendar month of the period being forecast is known before the forecast
is made, so routing on it is not oracle gating in the sense
`models.oracle_gated_predict` warns about. Contrast the two-stage extreme pipeline, whose
gate has to be *learned* because whether tomorrow is extreme is not knowable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from honeymodel.evaluation import Fold, regression_metrics, skill_score
from honeymodel.models import make_regressor

#: Regime definitions, as month -> label maps. Which one to use is an empirical question
#: and `regime_report` is how it gets answered rather than argued.
REGIME_SCHEMES: dict[str, dict[int, str]] = {
    # The failure mode as stated: the nectar flow against everything else. Two regimes
    # keeps ~2,000 and ~1,000 weekly rows on either side, which a tree can still fit.
    "flow_vs_rest": {
        **{month: "flow" for month in (4, 5, 6, 7, 8)},
        **{month: "rest" for month in (9, 10, 11, 12, 1, 2, 3)},
    },
    # Meteorological seasons, matching the segmented reports in both Milestone 4
    # notebooks so the per-regime numbers are directly comparable to them.
    "season": {
        **{month: "Spring" for month in (3, 4, 5)},
        **{month: "Summer" for month in (6, 7, 8)},
        **{month: "Autumn" for month in (9, 10, 11)},
        **{month: "Winter" for month in (12, 1, 2)},
    },
    # Three regimes on the colony's own calendar: build-up, harvest and dormancy. The
    # split that a beekeeper would draw, and the one with the strongest mechanism.
    "colony_cycle": {
        **{month: "buildup" for month in (3, 4, 5)},
        **{month: "flow" for month in (6, 7, 8)},
        **{month: "dormancy" for month in (9, 10, 11, 12, 1, 2)},
    },
}


def regime_of(months: pd.Series, scheme: str = "flow_vs_rest") -> pd.Series:
    """Map a month column to regime labels."""
    if scheme not in REGIME_SCHEMES:
        raise ValueError(f"scheme must be one of {sorted(REGIME_SCHEMES)}, got {scheme!r}")
    mapping = REGIME_SCHEMES[scheme]
    return pd.Series(months, copy=False).astype(int).map(mapping)


@dataclass
class SeasonGatedRegressor:
    """One regressor per season regime, routed on the `month` feature.

    Parameters
    ----------
    regressor : short name passed to `models.make_regressor`, same one for every regime.
        Deliberately not per-regime tuned -- the claim under test is that *splitting*
        helps, and letting each regime pick its own algorithm would confound that with
        model selection.
    scheme : key into `REGIME_SCHEMES`.
    min_rows : below this, a regime does not get its own model and falls back to the
        pooled fit. With four folds and an expanding window the first fold's Winter
        partition can be genuinely small, and fitting a forest to 40 rows would make the
        comparison a statement about overfitting rather than about regimes.
    month_column : the feature the gate reads. Present in every period feature set.

    Exposes the sklearn `fit`/`predict` pair so it drops into
    `periods.evaluate_period_folds` with no special-casing.
    """

    regressor: str = "rf"
    scheme: str = "flow_vs_rest"
    min_rows: int = 150
    seed: int = 42
    month_column: str = "month"
    models_: dict[str, object] = field(default_factory=dict, init=False)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SeasonGatedRegressor":
        if self.month_column not in X.columns:
            raise KeyError(
                f"the gate reads {self.month_column!r}, which is not in the feature matrix"
            )
        y = pd.Series(np.asarray(y, dtype=float), index=X.index)
        labels = regime_of(X[self.month_column], self.scheme)

        self.pooled_ = make_regressor(self.regressor, seed=self.seed)
        self.pooled_.fit(X, y)

        self.models_ = {}
        self.regime_rows_ = {}
        for label, index in labels.groupby(labels).groups.items():
            self.regime_rows_[label] = len(index)
            if len(index) < self.min_rows:
                continue
            model = make_regressor(self.regressor, seed=self.seed)
            model.fit(X.loc[index], y.loc[index])
            self.models_[label] = model
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        labels = regime_of(X[self.month_column], self.scheme)
        out = self.pooled_.predict(X)
        for label, model in self.models_.items():
            mask = (labels == label).to_numpy()
            if mask.any():
                out[mask] = model.predict(X[mask])
        return out


def regime_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    months: pd.Series,
    scheme: str = "season",
    baseline_mae: dict[str, float] | None = None,
    fold_baseline_mae: float | None = None,
) -> pd.DataFrame:
    """Per-regime metrics with the pooled row alongside.

    `baseline_mae` is a per-regime dict, not a scalar. Both Milestone 4 notebooks score
    every segment against one fold-wide bar, and at this grain that is not a neutral
    choice: the annual bar carries summer's variance into the winter comparison and vice
    versa. Summer is scored against a bar the winter rows helped make easy, so it looks
    worse than it is, and winter is scored against a bar summer made hard, so it looks
    better.

    Both are reported. `skill_vs_naive` uses the within-regime bar and is the honest
    number; `skill_vs_fold_naive` reproduces the Milestone 4 convention so the size of
    the artifact is visible rather than asserted.
    """
    frame = pd.DataFrame(
        {
            "y_true": np.asarray(y_true, dtype=float),
            "y_pred": np.asarray(y_pred, dtype=float),
            "regime": regime_of(pd.Series(months).reset_index(drop=True), scheme).to_numpy(),
        }
    )
    rows = []
    for label, group in frame.groupby("regime"):
        metrics = regression_metrics(group.y_true, group.y_pred)
        row = {"regime": label, **metrics}
        if baseline_mae and label in baseline_mae:
            row["baseline_mae"] = baseline_mae[label]
            row["skill_vs_naive"] = skill_score(metrics["mae"], baseline_mae[label])
        if fold_baseline_mae is not None:
            row["skill_vs_fold_naive"] = skill_score(metrics["mae"], fold_baseline_mae)
        rows.append(row)
    pooled = regression_metrics(frame.y_true, frame.y_pred)
    pooled_row = {"regime": "pooled", **pooled}
    if fold_baseline_mae is not None:
        pooled_row["baseline_mae"] = fold_baseline_mae
        pooled_row["skill_vs_naive"] = skill_score(pooled["mae"], fold_baseline_mae)
        pooled_row["skill_vs_fold_naive"] = pooled_row["skill_vs_naive"]
    rows.append(pooled_row)
    return pd.DataFrame(rows).sort_values("regime").reset_index(drop=True)


def regime_baseline_mae(
    frame: pd.DataFrame,
    fold: Fold,
    baselines: dict[str, np.ndarray],
    target: str,
    scheme: str = "season",
    month_column: str = "month",
) -> dict[str, float]:
    """Best naive MAE *within each regime*, which is the bar `regime_report` needs.

    The winner can differ by regime and usually does -- predicting zero is very hard to
    beat in dormancy and climatology is the one to beat in the flow -- so the minimum is
    taken per regime rather than picking one rule globally and reusing it.
    """
    test = frame.iloc[fold.test_index]
    y_true = test[target].to_numpy(dtype=float)
    labels = regime_of(test[month_column].reset_index(drop=True), scheme).to_numpy()

    out: dict[str, float] = {}
    for label in pd.unique(labels):
        mask = labels == label
        out[label] = min(
            regression_metrics(y_true[mask], prediction[mask])["mae"]
            for prediction in baselines.values()
        )
    return out


def evaluate_regime_models(
    matrix,
    frame: pd.DataFrame,
    folds: list[Fold],
    target: str,
    schemes: tuple[str, ...] = ("flow_vs_rest", "colony_cycle", "season"),
    regressor: str = "rf",
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pooled against gated, per fold and per regime, on one shared set of folds.

    Returns (fold_table, regime_table). The fold table answers "does gating help overall?"
    and will mostly say no, because the flow months are a minority of rows and a pooled
    average is dominated by the dormancy months the pooled model already handles. The
    regime table answers the question item 2 actually asked, which is whether summer got
    better.
    """
    from honeymodel.periods import period_baselines

    fold_rows: list[dict] = []
    regime_rows: list[dict] = []

    specs: list[tuple[str, object]] = [("pooled", None)]
    specs += [(f"gated:{scheme}", scheme) for scheme in schemes]

    for fold in folds:
        baselines = period_baselines(frame, fold, target=target)
        y_true = matrix.y.iloc[fold.test_index].to_numpy(dtype=float)
        fold_bar = min(regression_metrics(y_true, p)["mae"] for p in baselines.values())
        months = frame.iloc[fold.test_index]["month"]

        for name, scheme in specs:
            if scheme is None:
                model = make_regressor(regressor, seed=seed)
            else:
                model = SeasonGatedRegressor(regressor=regressor, scheme=scheme, seed=seed)
            model.fit(matrix.X.iloc[fold.train_index], matrix.y.iloc[fold.train_index])
            y_pred = model.predict(matrix.X.iloc[fold.test_index])

            metrics = regression_metrics(y_true, y_pred)
            fold_rows.append(
                {
                    "fold": fold.name,
                    "model": name,
                    **metrics,
                    "baseline_mae": fold_bar,
                    "skill_vs_naive": skill_score(metrics["mae"], fold_bar),
                }
            )

            bars = regime_baseline_mae(frame, fold, baselines, target, scheme="season")
            report = regime_report(
                y_true, y_pred, months, scheme="season",
                baseline_mae=bars, fold_baseline_mae=fold_bar,
            )
            regime_rows.append(report.assign(fold=fold.name, model=name))

    return pd.DataFrame(fold_rows), pd.concat(regime_rows, ignore_index=True)
