"""Harvest reconstruction, and the gross-gain target it makes possible.

Section 7.3 item 1 of `Milestone_4_alt.ipynb`, which is the one recommendation there that
is not a modelling change:

    At weekly and monthly grain a harvest lands *inside* the forecast window and subtracts
    10-20 kg of exactly the quantity being predicted.

Net weight change is therefore not honey production. It is honey production minus whatever
the beekeeper carried away, and the second term is the larger of the two in the weeks that
matter. `gross gain = net change + mass removed` is the quantity a beekeeper actually wants
forecast, and this module estimates the subtrahend.

Two independent routes to a removal, because neither is trustworthy alone:

    detect_logged_events        the beekeeper's own `honey` log, recovered from the
                                published `*_last_dif` counters. Authoritative about
                                *when*, silent about *how much*, and only 19 hives keep it.
    detect_weight_removals      a sustained step down in the weight series. Quantitative
                                and available for every hive, but blind to why the step
                                happened.

`corroboration` crosses the two. That cross is the honest measure of how well either route
works, and Section 2 of `Milestone_5.ipynb` reports it before anything is built on top.

Nothing here re-fits the daily table; `honeymodel.data` remains the single loader.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from honeymodel.data import DATE, EVENT_TYPES, GROUP

#: Weight column the step detector reads. The uncleaned publisher series, deliberately:
#: `weight_kg_noOutlier` zeroes every abrupt change, which is precisely the signal here.
WEIGHT = "end_of_day_weight_kg"

#: A step this large, sustained, is treated as mass leaving the hive. 3 kg is roughly
#: one full shallow frame of capped honey and 1.6 daily sd, and the sweep in
#: `removal_threshold_sweep` is what justifies it rather than convention.
DEFAULT_REMOVAL_THRESHOLD_KG = 3.0

#: How long the drop has to hold, and how much of it. A colony that loses 6 kg and has
#: recovered 5 of them a week later did not have honey taken out of it.
SUSTAIN_DAYS = 7
SUSTAIN_FRACTION = 0.6

GROSS_TARGET = "target_next_period_gross_gain_kg"
PERIOD_REMOVED = "period_removed_kg"
NEXT_PERIOD_REMOVED = "next_period_removed_kg"


# ---------------------------------------------------------------------------
# Route 1 -- the beekeeper's own log, recovered from the distance counters
# ---------------------------------------------------------------------------


def detect_logged_events(
    frame: pd.DataFrame,
    event: str = "honey",
    scale_window: int = 5,
) -> pd.DataFrame:
    """Recover the dates of logged beekeeper events from `{event}_last_dif`.

    The publishers ship no event table at daily grain -- `honey_last`/`honey_next` are
    entirely null in the `years` files. What survives is `{event}_last_dif`, the time
    since the previous event of that type, which climbs monotonically and **resets when a
    new event happens**. The reset, not the value, is the observable.

    Reading the reset rather than the value sidesteps the unit defect documented in
    `data.normalise_event_distance_units`. That function infers one unit per hive; these
    columns turn out to switch unit *within* a hive (hive 21 increments by 1.0 per day
    through 2019 and by 86,400 through mid-2020), so a per-hive conversion is not enough.
    A reset is a reset in either unit.

    Two filters, and both remove real-looking events:

    `days_since <= gap + 1`   the residual counter value, divided by the locally estimated
                              per-day increment, must place the event inside the
                              observation gap that the reset was seen across. This is what
                              makes the estimate unit-free: the scale is read off the
                              increments immediately following the reset, so a hive that
                              switches unit mid-record is handled at each reset
                              separately. Resets failing it are dated to a moment before
                              the counter was last seen higher, which is a contradiction --
                              usually a mixed-unit stretch. They are dropped, not repaired.

    `not jan_boundary`        12 of the 16 January "honey events" land on 1 January
                              exactly, across 12 different hives, at counter values two
                              orders of magnitude above the local scale. That is the
                              publisher's per-year processing restarting the counter, not
                              a midwinter harvest.

    The result validates against the German beekeeping calendar without being told it:
    honey peaks April-August, feeding peaks July-September (winter stores go on after the
    last harvest), queencell peaks April-June (swarm season), treatment peaks
    August-November (varroa). None of that is imposed anywhere in this function.
    """
    column = f"{event}_last_dif"
    if column not in frame.columns:
        raise KeyError(f"{column} is not in the frame; expected one of {EVENT_TYPES}")

    records = []
    for hive, group in frame.sort_values([GROUP, DATE], kind="mergesort").groupby(GROUP, sort=True):
        group = group[[column, DATE]].dropna(subset=[column]).reset_index(drop=True)
        if len(group) < 3:
            continue
        values = group[column].to_numpy(dtype=float)
        dates = group[DATE].to_numpy()
        gaps = np.diff(group[DATE].values).astype("timedelta64[D]").astype(int)
        per_day = np.where(gaps > 0, np.diff(values) / gaps, np.nan)

        for i in range(1, len(values)):
            if not values[i] < values[i - 1] * 0.95:
                continue
            local = per_day[i : i + scale_window]
            local = local[(local > 0) & np.isfinite(local)]
            scale = float(np.median(local)) if len(local) else np.nan
            days_since = values[i] / scale if scale and np.isfinite(scale) else np.nan
            records.append(
                {
                    GROUP: hive,
                    "event_type": event,
                    "observed_on": pd.Timestamp(dates[i]),
                    "previous_observation": pd.Timestamp(dates[i - 1]),
                    "observation_gap_days": int(gaps[i - 1]),
                    "counter_value": values[i],
                    "counter_scale_per_day": scale,
                    "days_since_estimate": days_since,
                }
            )

    events = pd.DataFrame(records)
    if events.empty:
        return events.assign(event_date=pd.NaT, accepted=False)

    events["event_date"] = events.observed_on - pd.to_timedelta(
        events.days_since_estimate.clip(upper=3650).fillna(0), unit="D"
    )
    events["jan_boundary"] = (events.observed_on.dt.month == 1) & (events.observed_on.dt.day <= 2)
    events["accepted"] = (
        events.days_since_estimate.notna()
        & (events.days_since_estimate <= events.observation_gap_days + 1)
        & ~events.jan_boundary
    )
    return events.sort_values([GROUP, "observed_on"]).reset_index(drop=True)


def logged_event_calendar(frame: pd.DataFrame) -> pd.DataFrame:
    """Accepted events per type per month -- the table that validates the reconstruction."""
    rows = []
    for event in EVENT_TYPES:
        events = detect_logged_events(frame, event=event)
        if events.empty:
            continue
        accepted = events[events.accepted]
        counts = accepted.observed_on.dt.month.value_counts()
        rows.append(
            {
                "event_type": event,
                "resets_found": len(events),
                "accepted": len(accepted),
                "hives": accepted[GROUP].nunique(),
                "peak_month": int(counts.idxmax()) if len(counts) else None,
                **{f"m{month:02d}": int(counts.get(month, 0)) for month in range(1, 13)},
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Route 2 -- a sustained step down in the weight series
# ---------------------------------------------------------------------------


def detect_weight_removals(
    frame: pd.DataFrame,
    threshold_kg: float = DEFAULT_REMOVAL_THRESHOLD_KG,
    sustain_days: int = SUSTAIN_DAYS,
    sustain_fraction: float = SUSTAIN_FRACTION,
) -> pd.DataFrame:
    """Days on which mass appears to have left the hive and stayed out.

    Three conditions, and the second is the one that does the work:

    1. the day-over-day change is at most `-threshold_kg`;
    2. the weight `sustain_days` later is still at least `sustain_fraction` of the drop
       below the pre-drop level -- a colony that loses 6 kg to a hot dry afternoon and has
       it back within the week did not have honey taken out of it;
    3. the row is not already flagged `sensor_dropout_flag` (a large drop with a matching
       recovery) or `implausible_weight_flag` (a reading below the 5 kg physical floor);
    4. the drop is measured across a *single* day. Without this, a hive that goes dark for
       six weeks and comes back 4 kg lighter registers as a harvest, when what it actually
       did was overwinter.

    Condition 3 is why this reuses the Milestone 4 quality flags rather than re-deriving
    them: those two failure modes look exactly like a harvest for one day and are the
    reason a naive "big negative change" rule cannot be used here.

    Returns one row per detected removal with the estimated mass, not a mask -- the
    removals are an object the notebook inspects, cross-checks against the beekeeper log,
    and only then sums into a target.
    """
    ordered = frame.sort_values([GROUP, DATE], kind="mergesort").copy()
    grouped = ordered.groupby(GROUP, sort=False)

    change = grouped[WEIGHT].diff()
    previous = grouped[WEIGHT].shift(1)
    forward = grouped[WEIGHT].shift(-sustain_days)
    recovered = (previous - forward) >= sustain_fraction * (-change)

    dropout = ordered.get("sensor_dropout_flag", pd.Series(False, index=ordered.index)).fillna(False)
    implausible = ordered.get("implausible_weight_flag", pd.Series(False, index=ordered.index)).fillna(False)
    one_day = grouped[DATE].diff().dt.days == 1

    mask = (
        (change <= -threshold_kg)
        & one_day.fillna(False)
        & recovered.fillna(False)
        & ~dropout.astype(bool)
        & ~implausible.astype(bool)
    )

    removals = ordered.loc[mask, [GROUP, DATE, WEIGHT]].copy()
    removals["removed_kg"] = -change[mask].to_numpy()
    removals["weight_before_kg"] = previous[mask].to_numpy()
    removals[f"weight_after_{sustain_days}d_kg"] = forward[mask].to_numpy()
    removals["month"] = removals[DATE].dt.month
    return removals.reset_index(drop=True)


def removal_threshold_sweep(
    frame: pd.DataFrame, thresholds: list[float] | tuple[float, ...] = (2, 3, 4, 5, 7, 10)
) -> pd.DataFrame:
    """What the removal threshold buys and costs, so 3 kg is a measured choice.

    The share landing in May-August is the discriminating column: a threshold that is
    picking up harvests concentrates there, and one that is picking up weather does not.
    """
    rows = []
    for threshold in thresholds:
        removals = detect_weight_removals(frame, threshold_kg=float(threshold))
        in_season = removals.month.isin([5, 6, 7, 8]).mean() if len(removals) else np.nan
        rows.append(
            {
                "threshold_kg": float(threshold),
                "n_removals": len(removals),
                "hives": removals[GROUP].nunique() if len(removals) else 0,
                "median_kg": round(float(removals.removed_kg.median()), 2) if len(removals) else np.nan,
                "total_kg": round(float(removals.removed_kg.sum()), 1) if len(removals) else 0.0,
                "share_may_to_aug": round(float(in_season), 3),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Crossing the two routes
# ---------------------------------------------------------------------------


def corroboration_sweep(
    frame: pd.DataFrame,
    removals: pd.DataFrame | None = None,
    tolerances: tuple[int, ...] = (1, 3, 7, 14),
) -> pd.DataFrame:
    """`corroboration` across matching windows, because one window is a choice.

    A logged event carries a timestamp the beekeeper entered, sometimes days after the
    fact, and the weight step is smeared by daily averaging. If the two routes agreed at
    all, widening the window would show it. Whether the share climbs with the window or
    stays flat is the whole result.
    """
    removals = detect_weight_removals(frame) if removals is None else removals
    frames = []
    for tolerance in tolerances:
        table = corroboration(frame, removals, tolerance_days=tolerance)
        if table.empty:
            continue
        frames.append(table.assign(tolerance_days=tolerance))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)[
        ["tolerance_days", "direction", "n", "matched", "share", "population"]
    ]


def corroboration(
    frame: pd.DataFrame,
    removals: pd.DataFrame | None = None,
    tolerance_days: int = 3,
) -> pd.DataFrame:
    """How often the two routes agree, in both directions.

    Both directions matter and they answer different questions. "What share of logged
    harvests show a weight drop?" tests whether the log means what its name says. "What
    share of weight drops sit near a logged harvest?" tests whether the step detector is
    finding harvests or finding something else.

    The comparison is restricted to the 19 hives that keep a `honey` log at all --
    scoring a detected drop as uncorroborated because its beekeeper never logged anything
    would measure record-keeping, not detection.
    """
    removals = detect_weight_removals(frame) if removals is None else removals
    events = detect_logged_events(frame, event="honey")
    logged = events[events.accepted] if len(events) else events

    if logged.empty:
        return pd.DataFrame()

    logging_hives = set(logged[GROUP].unique())
    tolerance = pd.Timedelta(days=tolerance_days)

    by_hive = {hive: group[DATE].to_numpy() for hive, group in removals.groupby(GROUP)}
    logged_near_drop = []
    for row in logged.itertuples():
        dates = by_hive.get(getattr(row, GROUP), np.array([], dtype="datetime64[ns]"))
        near = np.abs(dates - np.datetime64(row.observed_on)) <= tolerance
        logged_near_drop.append(bool(near.any()))

    logged_hive_removals = removals[removals[GROUP].isin(logging_hives)]
    by_event = {hive: group.observed_on.to_numpy() for hive, group in logged.groupby(GROUP)}
    drop_near_logged = []
    for row in logged_hive_removals.itertuples():
        dates = by_event.get(getattr(row, GROUP), np.array([], dtype="datetime64[ns]"))
        near = np.abs(dates - np.datetime64(getattr(row, DATE))) <= tolerance
        drop_near_logged.append(bool(near.any()))

    return pd.DataFrame(
        [
            {
                "direction": "logged honey event -> weight drop nearby",
                "n": len(logged),
                "matched": int(np.sum(logged_near_drop)),
                "share": round(float(np.mean(logged_near_drop)), 3) if logged_near_drop else np.nan,
                "population": f"{len(logging_hives)} hives that keep a honey log",
            },
            {
                "direction": "weight drop -> logged honey event nearby",
                "n": len(logged_hive_removals),
                "matched": int(np.sum(drop_near_logged)),
                "share": round(float(np.mean(drop_near_logged)), 3) if drop_near_logged else np.nan,
                "population": f"{len(logging_hives)} hives that keep a honey log",
            },
        ]
    )


# ---------------------------------------------------------------------------
# The target change
# ---------------------------------------------------------------------------


def add_gross_gain_target(
    period_frame: pd.DataFrame,
    daily_frame: pd.DataFrame,
    removals: pd.DataFrame | None = None,
    period: str = "week",
) -> pd.DataFrame:
    """Attach `target_next_period_gross_gain_kg` = net change + mass removed in that period.

    Removals are summed into the period they fall in, then shifted onto the row whose
    target that period is. The shift is guarded by the same contiguity rule
    `periods.add_period_features` uses for the net target: if the next period index is not
    exactly one greater, there is no next period and no gross target either.

    `period_removed_kg` -- the removals inside the *current* period -- is added too, and
    is a legitimate feature: it is knowable when the forecast is made. The next period's
    removal is not, and is dropped into `PERIOD_FORBIDDEN` by `next_period_removed_kg`'s
    name so a feature matrix cannot pick it up.
    """
    from honeymodel.periods import PERIOD_ALIAS, PERIOD_KEY, PERIOD_TARGET

    removals = detect_weight_removals(daily_frame) if removals is None else removals
    out = period_frame.sort_values([GROUP, PERIOD_KEY], kind="mergesort").copy()

    if removals.empty:
        out[PERIOD_REMOVED] = 0.0
        out[NEXT_PERIOD_REMOVED] = np.nan
        out[GROSS_TARGET] = out[PERIOD_TARGET]
        return out.reset_index(drop=True)

    stamped = removals.copy()
    stamped[PERIOD_KEY] = pd.PeriodIndex(stamped[DATE], freq=PERIOD_ALIAS[period]).astype("int64")
    per_period = (
        stamped.groupby([GROUP, PERIOD_KEY]).removed_kg.sum().rename(PERIOD_REMOVED).reset_index()
    )

    out = out.merge(per_period, on=[GROUP, PERIOD_KEY], how="left")
    out[PERIOD_REMOVED] = out[PERIOD_REMOVED].fillna(0.0)

    grouped = out.groupby(GROUP, sort=False)
    next_contiguous = (grouped[PERIOD_KEY].shift(-1) - out[PERIOD_KEY]) == 1
    out[NEXT_PERIOD_REMOVED] = grouped[PERIOD_REMOVED].shift(-1).where(next_contiguous)
    out[GROSS_TARGET] = out[PERIOD_TARGET] + out[NEXT_PERIOD_REMOVED]
    return out.sort_values([GROUP, PERIOD_KEY], kind="mergesort").reset_index(drop=True)


def target_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    """Net against gross, side by side. The row that matters is the summer one.

    A target change is only worth making if the two targets are different where the
    forecast is used. Pooled they barely differ -- removals touch a few percent of
    hive-weeks -- and in the harvest months they differ a lot.
    """
    from honeymodel.periods import PERIOD_TARGET

    usable = frame.dropna(subset=[PERIOD_TARGET, GROSS_TARGET])
    rows = []
    for label, subset in [("all periods", usable), ("May-August", usable[usable.month.isin([5, 6, 7, 8])])]:
        touched = subset[subset[NEXT_PERIOD_REMOVED].fillna(0) > 0]
        rows.append(
            {
                "segment": label,
                "n": len(subset),
                "n_with_removal": len(touched),
                "share_with_removal": round(len(touched) / len(subset), 4) if len(subset) else np.nan,
                "net_mean_kg": round(float(subset[PERIOD_TARGET].mean()), 3),
                "gross_mean_kg": round(float(subset[GROSS_TARGET].mean()), 3),
                "net_sd_kg": round(float(subset[PERIOD_TARGET].std()), 3),
                "gross_sd_kg": round(float(subset[GROSS_TARGET].std()), 3),
                "mean_shift_on_touched_kg": round(float(touched[NEXT_PERIOD_REMOVED].mean()), 2)
                if len(touched)
                else np.nan,
                "correlation": round(float(subset[PERIOD_TARGET].corr(subset[GROSS_TARGET])), 4),
            }
        )
    return pd.DataFrame(rows)
