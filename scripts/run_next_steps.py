#!/usr/bin/env python
"""Run the five recommended next steps from `Milestone_4_alt.ipynb` Section 7.3.

    PYTHONPATH=src python scripts/run_next_steps.py

Deterministic and self-contained: reads only `data/honey_model.parquet` and
`data/honey_weather.parquet`, writes only CSVs under `results/`. `Milestone_5.ipynb`
renders these tables rather than recomputing them, so a notebook run is fast and no number
in the write-up can drift from the run that produced it.

The five items, and where each one lands:

    1. gross gain, not net change      ns1_*   harvest reconstruction and the target change
    2. a season-conditional model      ns2_*   pooled vs gated, per regime
    3. weather                         ns3_*   the ablation ladder, with a forecast ceiling
    4. per-hive relative extremes      ns4_*   matched-rate comparison of the two labels
    5. more years                      ns5_*   what "more" would have to mean

Item 5 cannot be done as written -- the archive ends 2022-12-30 -- so what is run is the
sample-size curve that says how much more would be needed. That substitution is stated in
the notebook, not buried here.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from honeymodel import (  # noqa: E402
    data,
    evaluation as ev,
    harvest,
    models,
    periods as pr,
    power,
    regimes,
    weather as wx,
)

RESULTS_DIR = REPO_ROOT / "results"
SEED = 42
GRAIN = "week"


def _write(frame: pd.DataFrame, name: str, note: str = "") -> pd.DataFrame:
    RESULTS_DIR.mkdir(exist_ok=True)
    frame.to_csv(RESULTS_DIR / name, index=False)
    print(f"  wrote {name:44s} {len(frame):5d} rows  {note}")
    return frame


# ---------------------------------------------------------------------------
# Item 1 -- gross gain rather than net weight change
# ---------------------------------------------------------------------------


def item_1_gross_gain(daily: pd.DataFrame, week: pd.DataFrame) -> pd.DataFrame:
    print("\n[1] gross gain, not net weight change")

    _write(harvest.logged_event_calendar(daily), "ns1_logged_event_calendar.csv",
           "beekeeper log recovered from the reset counters")
    _write(harvest.removal_threshold_sweep(daily), "ns1_removal_threshold_sweep.csv",
           "step-drop detector, threshold sensitivity")

    removals = harvest.detect_weight_removals(daily)
    _write(removals, "ns1_removals.csv", "every detected removal")
    _write(harvest.corroboration_sweep(daily, removals), "ns1_corroboration.csv",
           "do the two routes agree?")

    gross = harvest.add_gross_gain_target(week, daily, removals, period=GRAIN)
    _write(harvest.target_comparison(gross), "ns1_target_comparison.csv",
           "net against gross")

    # Both targets, same folds, same features, same model. The folds are built on the net
    # frame and reused, so the two boards are scored on identical rows.
    rows = []
    for target, label in ((pr.PERIOD_TARGET, "net change"), (harvest.GROSS_TARGET, "gross gain")):
        frame = gross.dropna(subset=[target]).reset_index(drop=True)
        matrix = pr.build_period_matrix(frame, "history+sensors", target=target)
        folds = pr.rolling_origin_period_cv(matrix.frame)
        board = pr.period_baseline_board(matrix.frame, folds, target=target)
        for model_name in ("rf", "hist_gb"):
            result = pr.evaluate_period_folds(
                matrix, folds,
                lambda name=model_name: models.make_regressor(name, seed=SEED),
                model_name=f"{label} / {model_name}",
                baseline_board=board,
            )
            rows.append(result.assign(target=label, n_rows=len(matrix)))
        best = board.groupby("model").mae.mean().idxmin()
        rows.append(
            board[board.model == best]
            .assign(model=f"{label} / best naive ({best})", target=label, n_rows=len(matrix),
                    skill_vs_naive=0.0)
        )
    return _write(pd.concat(rows, ignore_index=True), "ns1_target_board.csv",
                  "can either target be predicted?")


# ---------------------------------------------------------------------------
# Item 2 -- a season-conditional model
# ---------------------------------------------------------------------------


def item_2_regimes(week: pd.DataFrame) -> None:
    print("\n[2] season-conditional models")
    matrix = pr.build_period_matrix(week, "history+sensors")
    folds = pr.rolling_origin_period_cv(matrix.frame)

    fold_table, regime_table = regimes.evaluate_regime_models(
        matrix, matrix.frame, folds, pr.PERIOD_TARGET, seed=SEED
    )
    _write(fold_table, "ns2_regime_fold_board.csv", "pooled vs gated, overall")
    _write(regime_table, "ns2_regime_breakdown.csv", "per season, both bars")

    summary = (
        regime_table.groupby(["model", "regime"])[
            ["mae", "skill_vs_naive", "skill_vs_fold_naive", "n"]
        ]
        .mean()
        .reset_index()
    )
    _write(summary, "ns2_regime_summary.csv", "means across folds")


# ---------------------------------------------------------------------------
# Item 3 -- weather
# ---------------------------------------------------------------------------


def item_3_weather(daily: pd.DataFrame, week: pd.DataFrame) -> pd.DataFrame:
    print("\n[3] weather")
    with_weather = wx.add_period_weather(week, daily, period=GRAIN, lookahead=True)
    _write(wx.weather_coverage(with_weather), "ns3_weather_coverage.csv", "join integrity")
    _write(wx.weather_target_correlation(with_weather, pr.PERIOD_TARGET),
           "ns3_weather_correlation.csv", "rank correlation with the target")

    ladder = wx.weather_feature_sets("history+sensors")
    allow = set(wx.WEATHER_LOOKAHEAD_FEATURES)

    rows = []
    reference_folds = None
    for rung, names in ladder.items():
        is_upper_bound = any(name.startswith("next_wx_") for name in names)
        matrix = pr.build_period_matrix(
            with_weather, names, allow=allow if is_upper_bound else None
        )
        folds = pr.rolling_origin_period_cv(matrix.frame)
        reference_folds = reference_folds or folds
        board = pr.period_baseline_board(matrix.frame, folds, target=pr.PERIOD_TARGET)
        for model_name in ("rf", "hist_gb"):
            result = pr.evaluate_period_folds(
                matrix, folds,
                lambda name=model_name: models.make_regressor(name, seed=SEED),
                model_name=f"{rung} / {model_name}",
                baseline_board=board,
            )
            rows.append(
                result.assign(
                    rung=rung,
                    n_features=len(names),
                    honest_label="UPPER BOUND - uses the forecast period's actual weather"
                    if is_upper_bound
                    else "deployable",
                )
            )
        print(f"    ran {rung} ({len(names)} features)")

    board_frame = _write(pd.concat(rows, ignore_index=True), "ns3_weather_board.csv",
                         "the ablation ladder")

    # Permutation importance on the deployable weather rung, to see whether the model
    # leans on weather at all once it already has weight history.
    names = ladder["history+sensors+weather"]
    matrix = pr.build_period_matrix(with_weather, names)
    folds = pr.rolling_origin_period_cv(matrix.frame)
    last = folds[-1]
    model = models.make_regressor("rf", seed=SEED)
    model.fit(matrix.X.iloc[last.train_index], matrix.y.iloc[last.train_index])
    importance = models.permutation_feature_importance(
        model, matrix.X.iloc[last.test_index], matrix.y.iloc[last.test_index], n_repeats=5
    )
    importance["is_weather"] = importance.feature.str.startswith("wx_")
    _write(importance, "ns3_weather_importance.csv", "does the model use it?")
    return board_frame


# ---------------------------------------------------------------------------
# Item 4 -- per-hive relative extremes
# ---------------------------------------------------------------------------


def item_4_relative_threshold(week: pd.DataFrame) -> None:
    print("\n[4] per-hive relative extreme threshold")
    matrix = pr.build_period_matrix(week, "history+sensors")
    frame = matrix.frame
    folds = pr.rolling_origin_period_cv(frame)

    absolute_kg = pr.relative_extreme_threshold(frame[pr.PERIOD_TARGET])
    absolute_rate = float((frame[pr.PERIOD_TARGET].abs() > absolute_kg).mean())
    multiplier = pr.calibrate_relative_multiplier(frame, absolute_rate)
    print(f"    absolute {absolute_kg:.2f} kg fires on {100 * absolute_rate:.2f}% of weeks; "
          f"matched multiplier {multiplier:.2f} x hive change MAD")

    _write(pr.relative_threshold_sweep(frame), "ns4_relative_sweep.csv", "multiplier sensitivity")
    comparison = pr.extreme_label_comparison(frame, absolute_kg, multiplier)
    _write(comparison, "ns4_label_comparison.csv",
           f"matched at {100 * absolute_rate:.2f}%, {int(comparison.attrs['agreement'])} weeks in both")

    labels = {
        "absolute": (frame[pr.PERIOD_TARGET].abs() > absolute_kg).to_numpy(),
        "relative": pr.relative_extreme_flag(frame, multiplier).to_numpy(),
    }

    rows = []
    for name, label in labels.items():
        for fold in folds:
            y_train = label[fold.train_index]
            y_test = label[fold.test_index]
            if y_train.sum() < 5 or y_test.sum() < 2:
                rows.append({"fold": fold.name, "rule": name, "pr_auc": np.nan,
                             "note": "too few positives to score"})
                continue
            classifier = models.make_classifier("rf", seed=SEED)
            classifier.fit(matrix.X.iloc[fold.train_index], y_train)
            probability = classifier.predict_proba(matrix.X.iloc[fold.test_index])[:, 1]
            rows.append({
                "fold": fold.name, "rule": name, "note": "",
                **models.classifier_report(y_test, probability),
            })
    _write(pd.DataFrame(rows), "ns4_classifier_board.csv", "detectability under each label")

    # A two-stage pipeline is only worth re-pointing at the relative label if the label is
    # the thing limiting it, so run the end-to-end regression under both.
    rows = []
    board = pr.period_baseline_board(frame, folds, target=pr.PERIOD_TARGET)
    for name, threshold in (("absolute", absolute_kg), ("relative", None)):
        for fold in folds:
            X_train = matrix.X.iloc[fold.train_index]
            y_train = matrix.y.iloc[fold.train_index]
            X_test = matrix.X.iloc[fold.test_index]
            y_test = matrix.y.iloc[fold.test_index].to_numpy(dtype=float)

            if threshold is None:
                # Per-row threshold: scale the target by the hive's own MAD, so a single
                # global cut on the scaled series *is* the per-hive rule. The scale is
                # floored at its 5th percentile -- a hive whose change MAD is 0.03 kg
                # would otherwise have its errors multiplied by 30 on the way back to
                # kilograms, and the comparison would be measuring that division rather
                # than the labelling rule.
                scale = frame[pr.HIVE_SCALE_COLUMN].to_numpy(dtype=float)
                floor = float(np.nanquantile(scale, 0.05))
                scale = np.where(np.isfinite(scale), np.maximum(scale, floor), np.nanmedian(scale))
                model = models.TwoStageModel(
                    regressor="hist_gb", classifier="rf",
                    threshold_kg=multiplier, blend=True, seed=SEED,
                )
                model.fit(X_train, y_train.to_numpy() / scale[fold.train_index])
                y_pred = model.predict(X_test) * scale[fold.test_index]
            else:
                model = models.TwoStageModel(
                    regressor="hist_gb", classifier="rf",
                    threshold_kg=threshold, blend=True, seed=SEED,
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            metrics = ev.regression_metrics(y_test, y_pred)
            bar = float(board[board.fold == fold.name].mae.min())
            rows.append({"fold": fold.name, "rule": name, **metrics,
                         "baseline_mae": bar,
                         "skill_vs_naive": ev.skill_score(metrics["mae"], bar)})
    _write(pd.DataFrame(rows), "ns4_two_stage_board.csv", "two-stage under each label")


# ---------------------------------------------------------------------------
# Item 5 -- how much more data the monthly result needs
# ---------------------------------------------------------------------------


def item_5_power(daily: pd.DataFrame) -> None:
    print("\n[5] how much more data the monthly result would need")
    month = pr.add_period_features(pr.aggregate_to_period(daily, "month"))

    curve = power.sample_size_curve(
        month, target=pr.PERIOD_TARGET, feature_set="history+sensors",
        make_model=lambda: models.make_regressor("rf", seed=SEED),
        period="month", repeats=8, seed=SEED,
    )
    _write(curve, "ns5_power_curve_raw.csv", "every subsample")
    summary = _write(power.summarise_curve(curve), "ns5_power_curve.csv",
                     "skill and spread by sample size")
    projection = power.required_sample_size(curve)
    _write(pd.DataFrame([projection]), "ns5_power_projection.csv",
           "extrapolated sample size for a non-zero result")
    print("\n", summary.round(4).to_string(index=False))
    print("\n  projection:", {k: (round(v, 3) if isinstance(v, float) else v)
                              for k, v in projection.items()})


def main() -> int:
    warnings.filterwarnings("ignore", category=FutureWarning)
    RESULTS_DIR.mkdir(exist_ok=True)

    daily = data.load_model_table()
    week = pr.add_period_features(pr.aggregate_to_period(daily, GRAIN))
    print(f"loaded {len(daily):,} hive-days -> {len(week):,} hive-{GRAIN}s")

    item_1_gross_gain(daily, week)
    item_2_regimes(week)
    item_3_weather(daily, week)
    item_4_relative_threshold(week)
    item_5_power(daily)

    written = sorted(RESULTS_DIR.glob("ns*.csv"))
    print(f"\nwrote {len(written)} tables to {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
