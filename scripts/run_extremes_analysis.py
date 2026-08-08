#!/usr/bin/env python
"""Re-run the extreme-event investigation at the modelling grain, and sweep the threshold.

    PYTHONPATH=src python scripts/run_extremes_analysis.py

Milestone 3's extremes work (David_Work/extremes.ipynb) queried `bob_sensor_processed`
where `outlier_lim = TRUE` -- 98,304 MINUTE-grain records. The model's extremes are 445
DAILY rows where |next-day change| > 5 kg. Different table, different grain, different
outlier definition, so none of those percentages transferred. This script redoes both
analyses against the 445 rows the model actually has to handle.

It also answers the question Milestone 3 never asked: what do the publishers' own
outlier flags say about these events?

Tables produced
    extremes_event_proximity.csv    distance to each beekeeper event type, extremes vs routine
    extremes_date_clustering.csv    events per date and hives affected
    extremes_attribution.csv        how many extremes each quality flag accounts for
    extremes_weight_series.csv      Milestone 3's weight series vs the publisher-cleaned one
    threshold_sweep.csv             3 / 5 / 7 / 10 kg and a hive-relative variant
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from honeymodel import data, evaluation as ev, features, models  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
EVENTS = ("honey", "feeding", "swarming", "treatment", "died", "queencell")
THRESHOLDS = (3.0, 5.0, 7.0, 10.0)


def event_proximity(frame: pd.DataFrame) -> pd.DataFrame:
    """Distance to the nearest logged beekeeper event, extremes vs. routine days.

    Milestone 3 reported the extreme-day percentages with no comparison group, so
    "0% within 1 day" had nothing to be surprising against. Routine days are the
    control: if extremes and routine days sit the same distance from a logged event,
    the events explain nothing either way.
    """
    rows = []
    for group_name, subset in (
        ("extreme", frame[frame.extreme_weight_change_flag]),
        ("routine", frame[~frame.extreme_weight_change_flag]),
    ):
        for event in EVENTS:
            distances = subset[f"nearest_{event}_event_days"].dropna()
            rows.append({
                "group": group_name,
                "event": event,
                "n_rows": len(subset),
                "n_with_event_logged": len(distances),
                "pct_with_event_logged": round(100 * len(distances) / max(len(subset), 1), 2),
                "median_days": round(float(distances.median()), 2) if len(distances) else np.nan,
                "pct_within_1_day": round(100 * float((distances <= 1).mean()), 2) if len(distances) else np.nan,
                "pct_within_7_days": round(100 * float((distances <= 7).mean()), 2) if len(distances) else np.nan,
            })
    return pd.DataFrame(rows)


def date_clustering(frame: pd.DataFrame) -> pd.DataFrame:
    """Do extremes land on shared dates (external cause) or scatter per hive (local cause)?"""
    extremes = frame[frame.extreme_weight_change_flag]
    per_date = (
        extremes.groupby("measurement_date")
        .agg(n_extremes=("hive_id", "size"), affected_hives=("hive_id", "nunique"))
        .reset_index()
        .sort_values("n_extremes", ascending=False)
    )
    active = (
        frame.groupby("measurement_date")["hive_id"].nunique().rename("hives_reporting").reset_index()
    )
    per_date = per_date.merge(active, on="measurement_date", how="left")
    per_date["share_of_reporting_hives"] = (
        per_date["affected_hives"] / per_date["hives_reporting"]
    ).round(4)
    return per_date


def extreme_seasonality(frame: pd.DataFrame) -> pd.DataFrame:
    """Extreme rate by calendar month, split by direction.

    The clustering question is better answered here than by a per-date correlation: if
    extremes concentrate in the harvest months and skew negative, that is an apiary-level
    seasonal cause, not a per-hive accident.
    """
    frame = frame.copy()
    frame["direction"] = np.where(
        frame["target_next_day_weight_change_kg"] < 0, "loss", "gain"
    )
    extremes = frame[frame.extreme_weight_change_flag]
    per_month = (
        frame.groupby("month")
        .agg(hive_days=("hive_id", "size"))
        .join(extremes.groupby("month").size().rename("extremes"))
        .join(extremes[extremes.direction == "loss"].groupby("month").size().rename("extreme_losses"))
        .join(extremes[extremes.direction == "gain"].groupby("month").size().rename("extreme_gains"))
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    per_month["extreme_rate_pct"] = (
        100 * per_month["extremes"] / per_month["hive_days"]
    ).round(2)
    return per_month


def attribution(frame: pd.DataFrame) -> pd.DataFrame:
    """How many of the 445 extremes does each data-quality explanation account for?"""
    extremes = frame[frame.extreme_weight_change_flag]
    total = len(extremes)
    rows = []
    for label, mask in (
        ("implausible weight (below 5 kg floor)", extremes.implausible_weight_flag),
        ("hive-local robust outlier (>5 MAD)", extremes.robust_outlier_flag),
        ("drop-then-recovery (dropout / re-tare)", extremes.sensor_dropout_flag),
    ):
        rows.append({
            "explanation": label,
            "extremes_flagged": int(mask.sum()),
            "pct_of_extremes": round(100 * float(mask.mean()), 2),
            "total_extremes": total,
        })
    any_flag = (
        extremes.implausible_weight_flag | extremes.robust_outlier_flag | extremes.sensor_dropout_flag
    )
    rows.append({
        "explanation": "any data-quality flag",
        "extremes_flagged": int(any_flag.sum()),
        "pct_of_extremes": round(100 * float(any_flag.mean()), 2),
        "total_extremes": total,
    })
    return pd.DataFrame(rows)


def weight_series_comparison() -> pd.DataFrame:
    """Milestone 3's `weight_kg` target against the publishers' outlier-cleaned series.

    The publishers ship two weight columns. `weight_kg` keeps every jump.
    `weight_kg_noOutlier` is the cumulative sum of the same minute-level deltas with
    every change above 0.3 kg/min zeroed -- their stated aim being to remove "sudden
    drastic changes in the weight, which are usually induced by activities by the
    beekeeper". Milestone 3 used the first column and then concluded that extremes were
    unrelated to beekeeper activity.
    """
    con = duckdb.connect()
    source = (REPO_ROOT / "data" / "honey_daily_source.parquet").as_posix()
    con.execute(f"""
        CREATE VIEW s AS
        SELECT CAST(key AS INTEGER) AS hive_id,
               CAST(CAST(time AS TIMESTAMP) AS DATE) AS d,
               weight_kg AS w, weight_kg_noOutlier AS wn
        FROM read_parquet('{source}') WHERE dataset = 'years'
    """)
    con.execute("""
        CREATE VIEW f AS
        SELECT *,
               LAG(d) OVER h AS pd, LEAD(d) OVER h AS nd,
               LAG(w) OVER h AS pw, LEAD(w) OVER h AS nw,
               LEAD(wn) OVER h AS nwn
        FROM s WINDOW h AS (PARTITION BY hive_id ORDER BY d)
    """)
    return con.execute("""
        SELECT
            COUNT(*)                                                   AS modelling_rows,
            SUM(CASE WHEN ABS(nw - w)  > 5 THEN 1 ELSE 0 END)          AS extremes_weight_kg,
            SUM(CASE WHEN ABS(nwn - wn) > 5 THEN 1 ELSE 0 END)         AS extremes_weight_kg_noOutlier,
            SUM(CASE WHEN ABS(nw - w) > 5 AND ABS(nwn - wn) <= 5 THEN 1 ELSE 0 END)
                                                                       AS extremes_removed_by_cleaning,
            ROUND(CORR(nw - w, nwn - wn), 4)                           AS target_correlation
        FROM f
        WHERE DATE_DIFF('day', pd, d) = 1 AND DATE_DIFF('day', d, nd) = 1
          AND w IS NOT NULL AND pw IS NOT NULL AND nw IS NOT NULL
          AND w > 0 AND pw > 0 AND nw > 0
    """).fetchdf()


def threshold_sweep(frame: pd.DataFrame) -> pd.DataFrame:
    """Split, class balance, routine-model error and classifier PR-AUC at each threshold.

    The +/-5 kg line was inherited from Milestone 3 by convention and never tested. It is
    also absolute, which treats 5 kg on a nucleus colony and 5 kg on a production hive as
    the same event -- so a hive-relative variant is swept alongside it.
    """
    matrix = features.build_feature_matrix(frame, feature_set="history+sensors", impute="none")
    fold = ev.date_chronological_split(frame, "2022-01-01")
    X_train, X_test = matrix.X.iloc[fold.train_index], matrix.X.iloc[fold.test_index]
    y_train, y_test = matrix.y.iloc[fold.train_index], matrix.y.iloc[fold.test_index]

    definitions: list[tuple[str, pd.Series]] = [
        (f"absolute {t:g} kg", frame["target_next_day_weight_change_kg"].abs() > t) for t in THRESHOLDS
    ]
    definitions.append(("hive-relative 5 MAD", frame["extreme_relative_flag"]))

    rows = []
    for label, is_extreme in definitions:
        is_extreme = is_extreme.reset_index(drop=True)
        train_labels = is_extreme.iloc[fold.train_index].to_numpy()
        test_labels = is_extreme.iloc[fold.test_index].to_numpy()

        routine = models.make_regressor("hist_gb", seed=42)
        routine.fit(X_train[~train_labels], y_train[~train_labels])
        routine_metrics = ev.regression_metrics(
            y_test[~test_labels], routine.predict(X_test[~test_labels])
        )

        report = {"pr_auc": np.nan, "roc_auc": np.nan, "pr_auc_lift": np.nan}
        if train_labels.sum() >= 30 and test_labels.sum() >= 5:
            classifier = models.make_classifier("rf", seed=42)
            classifier.fit(X_train, train_labels)
            report = models.classifier_report(test_labels, classifier.predict_proba(X_test)[:, 1])

        rows.append({
            "definition": label,
            "extremes": int(is_extreme.sum()),
            "extreme_pct": round(100 * float(is_extreme.mean()), 3),
            "routine_rows": int((~is_extreme).sum()),
            "routine_mae_oracle_gated": round(routine_metrics["mae"], 4),
            "routine_r2_oracle_gated": round(routine_metrics["r2"], 4),
            "classifier_pr_auc": round(report["pr_auc"], 4) if report["pr_auc"] == report["pr_auc"] else np.nan,
            "classifier_roc_auc": round(report["roc_auc"], 4) if report["roc_auc"] == report["roc_auc"] else np.nan,
            "pr_auc_lift_over_no_skill": round(report["pr_auc_lift"], 2) if report["pr_auc_lift"] == report["pr_auc_lift"] else np.nan,
        })
    return pd.DataFrame(rows)


def main() -> int:
    RESULTS_DIR.mkdir(exist_ok=True)
    frame = data.add_season(data.load_model_table())
    frame, unit_report = data.normalise_event_distance_units(frame, return_report=True)
    unit_report.to_csv(RESULTS_DIR / "event_distance_units.csv", index=False)
    print("=== event-distance unit inference (published columns mix days and seconds) ===")
    print(unit_report.to_string(index=False))
    print()

    proximity = event_proximity(frame)
    proximity.to_csv(RESULTS_DIR / "extremes_event_proximity.csv", index=False)
    print("=== beekeeper-event proximity, extremes vs routine ===")
    print(proximity.to_string(index=False))

    clustering = date_clustering(frame)
    clustering.to_csv(RESULTS_DIR / "extremes_date_clustering.csv", index=False)
    correlation = clustering[["n_extremes", "affected_hives"]].corr().iloc[0, 1]
    print(f"\n=== date clustering: {len(clustering)} dates carry at least one extreme; "
          f"corr(n_extremes, affected_hives) = {correlation:.3f} ===")
    print("(that correlation is 1.0 by construction at this grain -- a hive contributes at "
          "most one extreme per date, so the two columns are the same number. The "
          "Milestone 3 figure of 0.213 was measured on minute-grain rows, where a hive "
          "can contribute many. Use share_of_reporting_hives instead.)")
    print(clustering.head(10).to_string(index=False))

    monthly = extreme_seasonality(frame)
    monthly.to_csv(RESULTS_DIR / "extremes_by_month.csv", index=False)
    print("\n=== extreme rate by month ===")
    print(monthly.to_string(index=False))

    attributed = attribution(frame)
    attributed.to_csv(RESULTS_DIR / "extremes_attribution.csv", index=False)
    print("\n=== data-quality attribution of the 445 extremes ===")
    print(attributed.to_string(index=False))

    series = weight_series_comparison()
    series.to_csv(RESULTS_DIR / "extremes_weight_series.csv", index=False)
    print("\n=== Milestone 3 weight series vs publisher outlier-cleaned series ===")
    print(series.to_string(index=False))

    sweep = threshold_sweep(frame)
    sweep.to_csv(RESULTS_DIR / "threshold_sweep.csv", index=False)
    print("\n=== threshold sensitivity ===")
    print(sweep.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
