# Milestone 3 Completion Plan

**Project:** Honey Yield Predictive Model  
**Team:** Stephanie Nord, David Jorgensen, Joshua Amaya  
**Date:** July 14, 2026

---

## Current State of the Project

The team has made significant progress toward Milestone 3. Here is a summary of what has been accomplished and what remains.

### What's Done

| Area | Status | Key Outcomes |
|------|--------|-------------|
| DuckDB pipeline (Stephanie) | ✅ End-to-end | Raw → Clean → Features → Model table; 29,172 raw rows from 151 daily CSVs across 78 hives (2019-06-24 to 2022-12-31) |
| Feature engineering | ✅ Initial pass | Lag features, rolling 3-day and 7-day averages, next-day target, temporal features (year, month, day_of_year) via SQL window functions |
| Baseline Random Forest model | ✅ Trained | Chronological train/test split; performance poor on full dataset |
| Target distribution analysis | ✅ Investigated | 1.7% extreme events (>5 kg next-day change); routine-only model performs substantially better |
| Extreme event flagging | ✅ Implemented | `extreme_weight_change_flag` column in `honey_model` table |
| NOAA weather query templates | ✅ Verified | GHCN-Daily (Parquet on S3) and Global Hourly (ISD) queries working in DuckDB; templates in `weather_duckdb_queries.sql` |
| Remote DuckDB server (Quack) | ✅ Running | 20 tables available including HOBOS, USDA Tucson, bob sensor data, USDA NASS |

### What's Remaining

The sections below lay out a structured plan to close the remaining gaps and produce a Milestone 3 deliverable.

---

## Milestone 3 Questions & How We'll Answer Them

### 1. Will I be able to answer the questions I want to answer with the data I have?

**Status: Partially answered — needs formal write-up and supporting evidence.**

**Key findings so far:**
- The German citizen-science dataset (78 hives, ~29K daily records) is sufficient to train a baseline model for predicting next-day hive weight change under *routine* conditions.
- The model performs well on routine changes (98.3% of observations) but fails on extreme events (1.7%), which likely require additional data.
- The dataset includes beekeeper observation columns (`queencell`, `feeding`, `honey`, `treatment`, `died`, `swarming`) that have *not yet been incorporated* into modeling — these may help explain some extreme events.

**Remaining work:**
- [ ] Quantify missingness rates across all columns — especially the beekeeper observation columns (`swarming.last`, `feeding.last`, `honey.last`, `treatment.last`, `died.last`, etc.) — to determine whether they contain enough non-null data to be useful as features.
- [ ] Run a formal summary: total observations, hive count, date range, observations per hive (min/median/max), and temporal coverage gaps per hive.
- [ ] Write a brief narrative answering this question with specific numbers (include in the milestone deliverable).

### 2. What visualizations are especially useful for explaining my data?

**Status: Planned but not yet produced.**

Create the following core visualizations, prioritized by explanatory value:

#### Priority 1 — Must have for Milestone 3
- [ ] **Target distribution histogram** — Distribution of `target_next_day_weight_change_kg` with the ±5 kg extreme threshold marked. Include both the full distribution and a zoomed-in view of the routine range (|change| ≤ 5 kg).
- [ ] **Hive weight time series (multi-panel)** — Weight over time for a representative sample of hives (e.g., 6-8 hives), colored or annotated to show extreme events. This is the single most important plot for explaining the data structure to a non-technical audience.
- [ ] **Correlation heatmap** — Show relationships among `weight_kg`, `internal_temp` (average), `outside_temp`, `humidity`, `pressure`, lag features, and the target.
- [ ] **Feature importance bar chart** — From the baseline Random Forest. Compare importance rankings between the full model and the routine-only model side-by-side.

#### Priority 2 — Strongly recommended
- [ ] **Seasonal weight trend** — Monthly average hive weight change across all hives (box plot or violin plot by month) to reveal seasonal foraging and dormancy patterns.
- [ ] **Extreme event scatter/timeline** — Plot extreme events on a calendar heatmap or timeline, colored by hive, to show whether they cluster by date (weather/seasonal) or by hive (management/colony).
- [ ] **Actual vs. Predicted scatter** — For the routine-only model, showing residual patterns.
- [ ] **Missing data heatmap** — Rows = hives, columns = date range, color = data presence/absence. This immediately shows coverage gaps and sensor outages.

#### Priority 3 — Nice to have
- [ ] Internal vs. external temperature over time.
- [ ] Pressure and humidity trends aligned with weight changes.
- [ ] Geographic plot of hive locations (latitude/longitude from the dataset).

### 3. Do I need to adjust the data and/or driving questions?

**Status: Partially answered — needs formalization.**

**Emerging conclusions:**
- **Data adjustment needed:** The 1.7% extreme events require special treatment. Rather than simply excluding them, we should investigate their causes (weather? swarming? beekeeper interventions?) and decide whether to:
  - Model them separately (anomaly detection or classification approach).
  - Include explanatory features (NOAA weather, beekeeper notes) that may make them predictable.
  - Filter them with an explicit, documented rationale.
- **Driving question refinement:** The original question ("Can we predict next-day hive weight change?") should be refined to distinguish between:
  - *Routine* weight changes (small daily fluctuations driven by foraging, weather, and colony metabolism).
  - *Extreme* weight changes (driven by interventions, swarming, equipment issues, or severe weather).

**Remaining work:**
- [ ] Investigate extreme events in detail:
  - Cross-reference extreme event dates with beekeeper observation columns (`swarming.last`, `feeding.last`, `honey.last`, `treatment.last`).
  - Check whether extreme events cluster on specific calendar dates (suggesting weather/regional cause) vs. specific hives (suggesting management cause).
  - Attempt to pull NOAA weather data for the German hive locations and dates to see if extreme events correlate with severe weather.
- [ ] Document the recommendation on whether to adjust the driving question (likely: keep the main question, add a sub-question about extreme event prediction/explanation).

### 4. Do I need to adjust my model/evaluation choices?

**Status: Partially answered — the dual-model finding is the key insight.**

**Emerging conclusions:**
- Random Forest is a reasonable starting point but should be compared with Gradient Boosting (XGBoost or LightGBM).
- A single model evaluated on all data gives misleading results because extreme events inflate error metrics. The evaluation strategy should report:
  - Overall MAE, RMSE, R² on the full test set.
  - MAE, RMSE, R² on routine-only observations.
  - MAE, RMSE, R² on extreme-event observations (even though sample size is small).
- The chronological train/test split is correct and should be maintained (no random splitting for time-series data).
- Consider whether the layered model approach from the Dataset Notes (Model 1: weight-only → Model 2: + internal sensors → Model 3: + environment → Model 4: + beekeeper observations) is still the right progression.

**Remaining work:**
- [ ] Implement the dual evaluation framework (overall + routine + extreme) as reusable functions.
- [ ] Run a side-by-side comparison of at least two models (Random Forest vs. Gradient Boosting) using the dual evaluation.
- [ ] Document model/evaluation adjustment recommendations.

### 5. Are my original expectations still reasonable?

**Status: Needs explicit reflection and documentation.**

**Remaining work:**
- [ ] Write a brief assessment comparing original project expectations with current findings. Key points to address:
  - Can we predict daily weight changes? → *Yes, for routine events; unclear for extreme events.*
  - Is the German citizen-science dataset sufficient? → *Sufficient for initial modeling; NOAA weather data will add value.*
  - Are tree-based models appropriate? → *Yes, Random Forest baseline is reasonable; Gradient Boosting should be tested.*
  - Is the feature set adequate? → *Lag features and rolling averages work; environmental and beekeeper features are the logical next additions.*

---

## Task Assignments

Below is a suggested task breakdown. Assignments can be adjusted based on individual availability.

### Joshua
Focus: **Visualizations and NOAA weather data integration**
1. Build the Priority 1 and Priority 2 visualizations listed above. Start with the target distribution histogram and hive weight time series — these are the most impactful for the milestone deliverable.
2. Adapt the NOAA weather query templates (already in `the-beehive/weather_duckdb_queries.sql`) to pull GHCN-Daily data (TMAX, TMIN, PRCP) for the German hive locations (lat/lon from the dataset) and date range (2019–2022).
3. Cross-reference extreme event dates with weather data to begin testing whether weather explains the large weight fluctuations.

### Stephanie
Focus: **Pipeline completion, feature importance analysis, and model comparison**
1. Incorporate beekeeper observation columns into the feature engineering pipeline and assess their coverage/usefulness.
2. Run and document feature importance comparisons (full model vs. routine-only model).
3. Implement the dual evaluation framework and run the Random Forest vs. Gradient Boosting comparison.
4. Scale the pipeline from the representative subset to the full daily dataset (453 CSVs) and verify results hold.

### David (You)
Focus: **Extreme event investigation and milestone write-up**
1. Investigate whether extreme events correlate with beekeeper observations (swarming, feeding, honey harvest, treatment, colony death) by querying the flagged events against those columns.
2. Check date clustering of extreme events — do they concentrate on certain dates (suggesting external cause) or are they spread across the timeline per-hive (suggesting hive-specific cause)?
3. Help draft the Milestone 3 narrative document that answers all five milestone questions with supporting evidence and visualizations.

---

## Milestone 3 Deliverable Outline

The final deliverable should be a Jupyter Notebook (or structured markdown/report) that walks through:

1. **Introduction** — Restate the project objective and data sources.
2. **Data Overview** — Summary statistics, column inventory, missingness, and coverage.
3. **Exploratory Visualizations** — All Priority 1 visualizations with interpretive captions.
4. **Target Analysis** — The routine vs. extreme event finding, with supporting visualizations.
5. **Preliminary Modeling** — Baseline Random Forest results (full and routine-only), feature importance.
6. **Milestone 3 Q&A** — Explicit, numbered answers to all five milestone questions.
7. **Adjusted Plan** — Any refinements to the driving questions, model choices, evaluation strategy, or data sources for subsequent milestones.
8. **Next Steps** — Concrete plan for NOAA weather integration, expanded feature engineering, and model iteration.

---

## Timeline

| Target Date | Milestone |
|-------------|-----------|
| Week 1 (Jul 14–20) | Complete data audit, missingness assessment, and extreme event investigation. Produce Priority 1 visualizations. |
| Week 2 (Jul 21–27) | Run model comparison (RF vs. GB). Pull NOAA weather for German locations. Produce Priority 2 visualizations. Feature importance comparison. |
| Week 3 (Jul 28–Aug 3) | Draft Milestone 3 narrative. Integrate NOAA weather as features (preliminary). Answer all five milestone questions. |
| Submission | Finalize notebook/report, review as team, submit. |

---

## Key Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Beekeeper observation columns are mostly null | Lose a potential feature family | Quantify early (Week 1); if sparse, flag as a limitation and focus on NOAA weather instead |
| German hive locations don't have nearby GHCN stations | Weather data has poor spatial coverage | Use the station-search bounding-box query to assess coverage; fall back to Open-Meteo API if needed |
| Full dataset (453 CSVs) reveals different patterns than subset (151 CSVs) | Model results change | Run the full pipeline early (Week 1) and compare distributions |
| Extreme events remain unexplained even with weather data | Model can't capture these events | Document as a finding; propose beekeeper-intervention labels as future work |
