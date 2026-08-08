#!/usr/bin/env python
"""Run the Milestone 4 validation protocol and write every result table to results/.

    PYTHONPATH=src python scripts/run_evaluation.py

Deterministic and self-contained: reads only data/honey_model.parquet, writes only CSVs.
The notebook renders these tables rather than recomputing them, so a notebook run is
fast and a result can never silently differ from the one in the write-up.

Tables produced
    baselines.csv              five naive rules x every rolling-origin fold
    model_board.csv            per-fold model results with skill vs. the best naive
    model_summary.csv          mean +/- sd across folds
    framings.csv               single regressor / oracle-gated / two-stage, side by side
    segmented_<...>.csv        error by season, month, hive and |change| decile
    hive_generalisation.csv    leave-hives-out vs. temporal
    seed_stability.csv         3-5 seeds per headline model
    permutation_importance.csv test-set permutation importance for the winning model
    classifier.csv             extreme-event classifier, PR-AUC against no-skill
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from honeymodel import data, evaluation as ev, features, models  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
SEEDS = (0, 1, 2, 3, 4)
THRESHOLD_KG = 5.0

# (label, feature set, model, imputation). Ordered as an ablation ladder so a gain can
# be attributed to the features or to the algorithm, not to both at once.
EXPERIMENTS = [
    ("M3 features / RF",          "milestone_3",      "rf",      "group_ffill"),
    ("M3 features / GB",          "milestone_3",      "gb",      "group_ffill"),
    ("history / HistGB",          "history",          "hist_gb", "none"),
    ("history+sensors / HistGB",  "history+sensors",  "hist_gb", "none"),
    ("history+sensors / RF",      "history+sensors",  "rf",      "group_ffill"),
]


def build(frame: pd.DataFrame, feature_set: str, impute: str) -> features.FeatureMatrix:
    return features.build_feature_matrix(frame, feature_set=feature_set, impute=impute)


def fit_predict_factory(matrix: features.FeatureMatrix, model_name: str, seed: int):
    def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        model = models.make_regressor(model_name, seed=seed)
        model.fit(matrix.X.iloc[train.index], matrix.y.iloc[train.index])
        return model.predict(matrix.X.iloc[test.index])

    return fit_predict


def main() -> int:
    RESULTS_DIR.mkdir(exist_ok=True)
    frame = data.add_season(data.load_model_table())
    print(f"loaded {len(frame):,} rows x {frame.shape[1]} columns")

    folds = ev.rolling_origin_cv(frame, n_splits=4, horizon_months=6, min_train_months=12)
    pd.DataFrame([f.describe() for f in folds]).to_csv(RESULTS_DIR / "folds.csv", index=False)
    print(f"rolling-origin folds: {len(folds)}")

    # ---- baselines ---------------------------------------------------------
    board = ev.baseline_board(frame, folds)
    board.to_csv(RESULTS_DIR / "baselines.csv", index=False)
    print("\nbaselines\n", ev.fold_summary(board).round(4).to_string(index=False))

    # ---- model board -------------------------------------------------------
    results = []
    matrices: dict[str, features.FeatureMatrix] = {}
    for label, feature_set, model_name, impute in EXPERIMENTS:
        key = f"{feature_set}|{impute}"
        matrices.setdefault(key, build(frame, feature_set, impute))
        matrix = matrices[key]
        results.append(
            ev.evaluate_folds(
                frame, folds, fit_predict_factory(matrix, model_name, seed=42),
                model_name=label, baseline_board_frame=board,
            )
        )
        print(f"  ran {label}")
    model_board = pd.concat(results, ignore_index=True)
    model_board.to_csv(RESULTS_DIR / "model_board.csv", index=False)

    summary = ev.fold_summary(pd.concat([model_board, board.assign(skill_vs_naive=np.nan)]))
    summary.to_csv(RESULTS_DIR / "model_summary.csv", index=False)
    print("\nmodel board\n", summary.round(4).to_string(index=False))

    # ---- three framings ----------------------------------------------------
    matrix = matrices["history+sensors|none"]
    framing_rows = []
    for fold in folds:
        train_index, test_index = fold.train_index, fold.test_index
        X_train, y_train = matrix.X.iloc[train_index], matrix.y.iloc[train_index]
        X_test, y_test = matrix.X.iloc[test_index], matrix.y.iloc[test_index]

        single = models.make_regressor("hist_gb", seed=42).fit(X_train, y_train)
        framing_rows.append({
            "fold": fold.name, "framing": "single regressor, all data",
            "honest_label": "realistic",
            **ev.regression_metrics(y_test, single.predict(X_test)),
        })

        routine_train = y_train.abs() <= THRESHOLD_KG
        routine = models.make_regressor("hist_gb", seed=42).fit(
            X_train[routine_train.to_numpy()], y_train[routine_train.to_numpy()]
        )
        y_oracle, pred_oracle = models.oracle_gated_predict(routine, X_test, y_test, THRESHOLD_KG)
        framing_rows.append({
            "fold": fold.name, "framing": "routine only, oracle-gated",
            "honest_label": "UPPER BOUND - test rows selected using the label",
            **ev.regression_metrics(y_oracle, pred_oracle),
        })

        for blend in (False, True):
            two_stage = models.TwoStageModel(
                regressor="hist_gb", classifier="rf", threshold_kg=THRESHOLD_KG,
                blend=blend, seed=42,
            ).fit(X_train, y_train)
            framing_rows.append({
                "fold": fold.name,
                "framing": f"two-stage, unfiltered test ({'blended' if blend else 'hard gate'})",
                "honest_label": "deployable",
                **ev.regression_metrics(y_test, two_stage.predict(X_test)),
            })
        print(f"  framings done for {fold.name}")

    framings = pd.DataFrame(framing_rows)
    baseline_by_fold = {f.name: ev.best_baseline_mae(board, f.name) for f in folds}
    framings["skill_vs_naive"] = [
        ev.skill_score(row.mae, baseline_by_fold[row.fold]) for row in framings.itertuples()
    ]
    framings.to_csv(RESULTS_DIR / "framings.csv", index=False)
    print("\nframings\n", framings.groupby(["framing", "honest_label"])[["mae", "rmse", "r2", "skill_vs_naive"]]
          .mean().round(4).to_string())

    # ---- classifier --------------------------------------------------------
    classifier_rows = []
    for fold in folds:
        X_train, y_train = matrix.X.iloc[fold.train_index], matrix.y.iloc[fold.train_index]
        X_test, y_test = matrix.X.iloc[fold.test_index], matrix.y.iloc[fold.test_index]
        classifier = models.make_classifier("rf", seed=42)
        classifier.fit(X_train, (y_train.abs() > THRESHOLD_KG).to_numpy())
        probability = classifier.predict_proba(X_test)[:, 1]
        classifier_rows.append({
            "fold": fold.name,
            **models.classifier_report((y_test.abs() > THRESHOLD_KG).to_numpy(), probability),
        })
    pd.DataFrame(classifier_rows).to_csv(RESULTS_DIR / "classifier.csv", index=False)
    print("\nclassifier\n", pd.DataFrame(classifier_rows).round(4).to_string(index=False))

    # ---- segmented report on the last fold --------------------------------
    #
    # Two skill columns, and the difference between them is a correction rather than a
    # refinement. `skill_vs_pooled_naive` divides every segment by one bar computed over
    # the whole fold, which is what this project reported first. On a target this
    # seasonal that denominator carries summer's variance into the winter comparison and
    # winter's into the summer one. `skill_vs_naive` scores each segment against the best
    # naive rule inside it.
    #
    # Which to read depends on the segment. Season, month and hive are knowable before
    # the forecast is made, so the within-segment bar is the honest competitor. A
    # |change| decile is defined by the label, so its within-segment bar is a rule that
    # already knows the answer -- for that one row group the pooled column is correct.
    last = folds[-1]
    X_train, y_train = matrix.X.iloc[last.train_index], matrix.y.iloc[last.train_index]
    X_test, y_test = matrix.X.iloc[last.test_index], matrix.y.iloc[last.test_index]
    winner = models.make_regressor("hist_gb", seed=42).fit(X_train, y_train)
    predictions = winner.predict(X_test)
    segmented = ev.segmented_report(
        y_test, predictions, matrix.meta.iloc[last.test_index],
        by=("season", "month", "hive_id", "weight_change_decile"),
        baseline_mae=ev.best_baseline_mae(board, last.name),
        baseline_predictions=ev.naive_baselines(frame, last),
    )
    segmented.to_csv(RESULTS_DIR / "segmented.csv", index=False)
    print("\nsegmented (season)\n",
          segmented[segmented.segment == "season"].round(4).to_string(index=False))
    print("\nsegmented (|change| decile)\n",
          segmented[segmented.segment == "weight_change_decile"].round(4).to_string(index=False))

    # ---- permutation importance -------------------------------------------
    importance = models.permutation_feature_importance(winner, X_test, y_test, n_repeats=5)
    importance.to_csv(RESULTS_DIR / "permutation_importance.csv", index=False)
    print("\npermutation importance (top 10)\n", importance.head(10).round(5).to_string(index=False))

    # ---- per-hive generalisation ------------------------------------------
    hive_folds = ev.grouped_hive_cv(frame, n_folds=5)
    hive_results = ev.evaluate_folds(
        frame, hive_folds, fit_predict_factory(matrix, "hist_gb", seed=42),
        model_name="hist_gb (leave-hives-out)",
    )
    hive_results.to_csv(RESULTS_DIR / "hive_generalisation.csv", index=False)
    print("\nleave-hives-out\n", ev.fold_summary(hive_results).round(4).to_string(index=False))

    # ---- seed stability ----------------------------------------------------
    seed_rows = []
    for model_name in ("rf", "hist_gb"):
        def fit_predict(seed: int, model_name: str = model_name):
            model = models.make_regressor(model_name, seed=seed)
            model.fit(X_train, y_train)
            return y_test.to_numpy(), model.predict(X_test)

        stability = ev.seed_stability(fit_predict, seeds=SEEDS[:3])
        stability["model"] = model_name
        seed_rows.append(stability)
    stability = pd.concat(seed_rows, ignore_index=True)
    stability.to_csv(RESULTS_DIR / "seed_stability.csv", index=False)
    print("\nseed stability\n", stability.round(4).to_string(index=False))

    print(f"\nwrote {len(list(RESULTS_DIR.glob('*.csv')))} tables to {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
