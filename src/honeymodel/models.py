"""Model constructors and the two-stage routine/extreme pipeline.

Milestone 3's headline -- "routine Gradient Boosting, R2 = 0.216" -- was measured on rows
selected by `extreme_weight_change_flag == False`, a flag derived from
`ABS(target) > 5`, i.e. from the label. That assumes something tells you tomorrow is not
an extreme day *before* you predict it. In deployment nothing does.

`TwoStageModel` is what actually does that job: a classifier decides whether tomorrow is
extreme, and its decision -- right or wrong -- routes the regression. It is scored on the
full unfiltered test set, so the number it produces is one you could actually ship.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression

try:  # optional -- native NaN handling and faster, but not required to reproduce
    from lightgbm import LGBMClassifier, LGBMRegressor

    HAS_LIGHTGBM = True
except ImportError:  # pragma: no cover
    HAS_LIGHTGBM = False

#: Which regressors tolerate NaN natively. For these, build the feature matrix with
#: impute="none": no cross-hive bleed, no dropped rows, and missingness stays informative.
NATIVE_NAN_MODELS = frozenset({"hist_gb", "lightgbm"})


def make_regressor(name: str = "gb", seed: int = 42, **kwargs):
    """Regressor by short name. Keep hyperparameters here so every notebook agrees."""
    if name == "rf":
        return RandomForestRegressor(
            n_estimators=kwargs.pop("n_estimators", 200),
            n_jobs=kwargs.pop("n_jobs", -1),
            random_state=seed,
            **kwargs,
        )
    if name == "gb":
        return GradientBoostingRegressor(random_state=seed, **kwargs)
    if name == "hist_gb":
        return HistGradientBoostingRegressor(random_state=seed, **kwargs)
    if name == "linear":
        return LinearRegression(**kwargs)
    if name == "lightgbm":
        if not HAS_LIGHTGBM:
            raise ImportError("lightgbm is not installed; `pip install -r requirements.txt`")
        return LGBMRegressor(random_state=seed, verbose=-1, **kwargs)
    raise ValueError(f"unknown regressor: {name!r}")


def make_classifier(name: str = "rf", seed: int = 42, balanced: bool = True, **kwargs):
    """Classifier by short name.

    `balanced` matters: the extreme class is ~1.7% of rows, so an unweighted classifier
    scores 98.3% accuracy by predicting "never extreme" and is useless. Accuracy is not
    reported for this task at all -- PR-AUC is.
    """
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=kwargs.pop("n_estimators", 300),
            class_weight="balanced" if balanced else None,
            n_jobs=kwargs.pop("n_jobs", -1),
            random_state=seed,
            **kwargs,
        )
    if name == "lightgbm":
        if not HAS_LIGHTGBM:
            raise ImportError("lightgbm is not installed; `pip install -r requirements.txt`")
        return LGBMClassifier(
            random_state=seed,
            class_weight="balanced" if balanced else None,
            verbose=-1,
            **kwargs,
        )
    raise ValueError(f"unknown classifier: {name!r}")


@dataclass
class TwoStageModel:
    """Classifier gates two regressors; scored end to end on unfiltered data.

    Parameters
    ----------
    threshold_kg : the |change| that defines an extreme day.
    operating_threshold : classifier probability above which a row is routed to the
        extreme regressor. Tune it against a stated beekeeper use case -- early warning
        favours recall, so a low value is defensible even at poor precision.
    blend : if True, predict the probability-weighted mixture of the two regressors
        instead of hard routing. Usually lower MAE, because a 30%-confident extreme
        contributes proportionally rather than all-or-nothing.
    """

    regressor: str = "hist_gb"
    classifier: str = "rf"
    threshold_kg: float = 5.0
    operating_threshold: float = 0.5
    blend: bool = False
    seed: int = 42

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TwoStageModel":
        y = pd.Series(np.asarray(y, dtype=float))
        is_extreme = y.abs() > self.threshold_kg

        self.classifier_ = make_classifier(self.classifier, seed=self.seed)
        self.classifier_.fit(X, is_extreme.to_numpy())

        self.routine_regressor_ = make_regressor(self.regressor, seed=self.seed)
        self.routine_regressor_.fit(X[~is_extreme.to_numpy()], y[~is_extreme.to_numpy()])

        # With ~1.7% positives a fold can hold very few extremes. Below a workable
        # minimum, fall back to the routine model rather than fitting noise.
        extreme_rows = int(is_extreme.sum())
        if extreme_rows >= 30:
            self.extreme_regressor_ = make_regressor(self.regressor, seed=self.seed)
            self.extreme_regressor_.fit(X[is_extreme.to_numpy()], y[is_extreme.to_numpy()])
        else:
            self.extreme_regressor_ = None
        self.n_extreme_train_ = extreme_rows
        return self

    def predict_proba_extreme(self, X: pd.DataFrame) -> np.ndarray:
        return self.classifier_.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        routine = self.routine_regressor_.predict(X)
        if self.extreme_regressor_ is None:
            return routine
        extreme = self.extreme_regressor_.predict(X)
        probability = self.predict_proba_extreme(X)
        if self.blend:
            return (1 - probability) * routine + probability * extreme
        return np.where(probability >= self.operating_threshold, extreme, routine)


def oracle_gated_predict(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    threshold_kg: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Milestone 3's routine-only framing, reproduced so it can be labelled honestly.

    Returns (y_true, y_pred) for routine rows only. The row selection uses the label,
    which is why any metric computed from this is an **upper bound**, not a deployable
    result. Kept in the codebase precisely so the comparison table can show all three
    framings side by side.
    """
    y = pd.Series(np.asarray(y, dtype=float))
    routine = (y.abs() <= threshold_kg).to_numpy()
    return y[routine].to_numpy(), model.predict(X[routine])


def permutation_feature_importance(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 10,
    seed: int = 42,
    scoring: str = "neg_mean_absolute_error",
) -> pd.DataFrame:
    """Permutation importance on held-out data, replacing impurity-based importance.

    MDI is biased toward high-cardinality continuous splits, and this feature set is
    near-collinear by construction (previous-day weight, 3-day and 7-day rolling means
    are three views of the same signal). "Recent weight history dominates" needed a
    measure that is not partly an artifact of how trees split.

    Collinearity still inflates nothing and deflates everything: permuting one of three
    correlated features leaves the model able to lean on the other two, so all three can
    look unimportant. State that caveat wherever this table appears.
    """
    result = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=seed, scoring=scoring, n_jobs=-1
    )
    return (
        pd.DataFrame(
            {
                "feature": X.columns,
                "importance_mean": result.importances_mean,
                "importance_sd": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def classifier_report(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    """PR-AUC first, ROC-AUC for context, plus the positive rate.

    At a 1.7% positive rate ROC-AUC looks flattering while the model is still mostly
    wrong when it fires; PR-AUC is the metric that tracks whether a beekeeper alert
    would be worth acting on. The no-skill PR-AUC equals the positive rate, so it is
    reported alongside as the bar.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    y_true = np.asarray(y_true).astype(bool)
    positive_rate = float(y_true.mean())
    if y_true.sum() == 0 or y_true.all():
        return {"pr_auc": np.nan, "roc_auc": np.nan, "positive_rate": positive_rate,
                "pr_auc_no_skill": positive_rate, "pr_auc_lift": np.nan}
    pr_auc = float(average_precision_score(y_true, probability))
    return {
        "pr_auc": pr_auc,
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "positive_rate": positive_rate,
        "pr_auc_no_skill": positive_rate,
        "pr_auc_lift": pr_auc / positive_rate if positive_rate else np.nan,
    }
