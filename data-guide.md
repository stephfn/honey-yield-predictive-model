# Summary
| Dataset | Granularity	| Key Variables | Accessibility/Status | Potential Role | URL |
| --- | --- | --- | --- | --- | --- |
2023 Tree-based Paper | 431 Hives (Daily) | Temp, Humidity, Wind Speed, Rain Level, Weight | Confirmed | Primary Training Set | https://arxiv.org/pdf/2304.01215 |
| HOBOS (Kaggle) | German Hives (hourly) | Weight, Climate | Confirmed | External Validation/Test Set | https://www.kaggle.com/datasets/se18m502/bee-hive-metrics |
USDA (Tucson) | 10 Colonies (every 5 min) | 5-min intervals | Confirmed | Time-Series Baseline | https://pmc.ncbi.nlm.nih.gov/articles/PMC11479372/ |
USDA NASS | State-level (annual) | Annual Yield | Confirmed | Regional Yield Covariates | https://www.nass.usda.gov/Surveys/Guide_to_NASS_Surveys/Bee_and_Honey/ |

# DuckDB Datasets
| Table Name | Database Name | Columns | Notes |
| --- | --- | --- | --- |
| HOBOS_flow_2017 | HOBOS | timestamp, flow | |