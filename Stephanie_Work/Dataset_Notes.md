# German Citizen Science Dataset

## Source:
Zenodo

## Years:
2019-2022

## Colonies:
78

## Status

Potential Primary Dataset

**Advantages**
- Publicly available
- Longitudinal data (2019–2022)
- Multiple sensor streams
- Multiple temporal resolutions
- Includes beekeeper observations

**Limitations**
- German colonies only
- Missing data assessment pending
- Event label availability not yet confirmed

## Granularity
- 1-minute measurements
- Hourly aggregates
- Daily aggregates

## Variables
- Weight
- Internal Temperature
- External Temperature
- Humidity
- Pressure
- Beekeeper Observations

---

## Feature Inventory

| Feature | Type | Potential Use |
|----------|----------|----------|
| Hive Weight | Numeric | Target variable, lag features |
| Internal Temperature | Numeric | Colony health |
| External Temperature | Numeric | Weather |
| Humidity | Numeric | Environmental conditions |
| Air Pressure | Numeric | Weather-related predictor |
| Beekeeper Notes | Categorical | Management interventions |
| Swarming Events | Binary | Colony disruption |

---

## Potential Target

- Next-day hive weight change
- Daily net hive weight gain/loss

### Rationale

Hive weight is a proxy for colony productivity and may reflect
foraging success, environmental conditions, and colony health.

---

## Potential Models

### Model 1: Hive History
Features:
- Weight
- Previous Weight
- Moving Average

Question:
Can hive history predict future weight changes?

### Model 2: Hive + Internal Sensors
Features:
- Model 1
- Internal Temperature

Question:
Does colony health improve prediction?

### Model 3: Hive + Environment
Features:
- Model 2
- External Temperature
- Humidity
- Pressure

Question:
Do environmental conditions improve prediction?

### Model 4: Full Model
Features:
- Model 3
- Beekeeper Observations

Question:
Do beekeeper observations add predictive value beyond sensor data?

---

## Questions To Investigate

- Can weight change be calculated directly?
- Are swarming events explicitly labeled?
- How many total observations exist?
- How much missing data exists?
- Which variables are continuous versus event-based?
- What percentage of observations contain beekeeper notes?