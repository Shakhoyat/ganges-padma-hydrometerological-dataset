# Multi-Variable Hydrometeorological and Point-Wise Water Level Dataset with 7-Day Lags along the Ganges–Padma River Corridor (2011–2025)

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch: 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Status: Verified & Open](https://img.shields.io/badge/Status-Verified_%26_Open-success.svg)](#)
[![DOI / Official Archive](https://img.shields.io/badge/BWDB_Archive-Invoice_2608295209-blueviolet.svg)](#)

---

## 1. Overview & Dataset Abstract

This repository contains the curated, benchmarked, 15-year multi-variable hydrometeorological dataset along the Ganges–Padma River Corridor in Bangladesh spanning calendar years **2011 to 2025** (January 1, 2011 – December 31, 2025). The corridor covers **290.7 km** across **17 synchronized hydrometric stations** operated by the Bangladesh Water Development Board (BWDB). 

The release provides **91,033 station-days** of observations, coupling daily average river water levels with **7-day stage lags**, **daily precipitation with 7-day rainfall lags**, **tributary discharge**, **sub-daily tidal ranges**, **pan evaporation**, **groundwater depths**, and **suspended sediment concentrations**. Uniquely, the observation timeline encompasses three distinct hydrodynamic eras surrounding the construction and opening of the **Padma Multipurpose Bridge** (2014–2022), enabling machine learning models to be evaluated across genuine anthropogenic river training and geomorphic regime shifts.

### Key Highlights
- **Spatial Coverage**: 17 hydrometric gauging stations spanning 290.7 river km from Pankha (Ganges entrance) through the Padma confluence to Chandpur (Lower Meghna).
- **Temporal Span**: 15 complete calendar years (5,479 consecutive days per station; 91,033 total station-days).
- **Temporal Regimes**:
  - `BEFORE` (2011-01-01 to 2014-11-25): Natural corridor baseline.
  - `DURING` (2014-11-26 to 2022-06-25): Active construction and heavy river training work.
  - `AFTER` (2022-06-26 to 2025-12-31): Operational mega-bridge structure.
- **Physical Benchmarking**: Naive persistence ($R^2 = 0.998$, $SS_\kappa = 0$) baseline framework establishing true machine learning value ($PI > 0$, $SS_\kappa > 0$).
- **Physics-Informed STGNN**: Complete Spatio-Temporal Graph Neural Network pipeline enforcing mass-conservation penalty ($\mathcal{L}_{\text{mass}}$) and reach-monotonicity constraints ($\mathcal{L}_{\text{mono}}$).

---

## 2. Dataset Specifications Table

*Conforming strictly to the CSE 4112 / Data in Brief specifications standard.*

| Specification | Details |
|---|---|
| **Subject** | Hydrology, Water Resources Engineering, Machine Learning |
| **Specific subject area** | Multi-variable hydrometeorological forecasting, point-wise water level modeling, flood early warning, hydrodynamic regime shift analysis |
| **Type of data** | Tables (CSV), Geospatial boundaries (GeoJSON), Metadata (JSON/CSV), Figures (PDF/PNG), Code (Python), Report (PDF/LaTeX) |
| **How data were acquired** | Certified hydrometric gauge registries, acoustic Doppler current profilers, automatic tipping bucket rain gauges, pan evaporimeters, piezometers, and 3-hourly tidal monitors maintained by BWDB |
| **Data format** | Raw (tidy CSVs), Processed (long panel, wide target ML design matrices), Metadata (registry, schemas) |
| **Parameters for collection** | Daily water level (m MSL), 7 antecedent daily stage lags, daily precipitation (mm), 7 antecedent rainfall lags, 3-day and 7-day moving rainfall totals, tributary discharge ($m^3/s$), pan evaporation (mm), groundwater depth (m), sediment (ppm), 3-hourly tidal range (m) |
| **Description of data collection** | Raw hydrometric archives compiled under BWDB paid commercial invoice `2608295209`, quality-controlled for non-physical spikes, harmonized to Public Works Datum (PWD), and aligned to strict 24-hour UTC+6 cycles |
| **Data source location** | Ganges–Padma River Corridor, Bangladesh (Latitude: 23.23°N – 24.63°N, Longitude: 88.08°E – 90.64°E) |
| **Data accessibility** | Fully open under Creative Commons Attribution 4.0 International (CC BY 4.0). Repository: [GitHub](https://github.com/Shakhoyat/ganges-padma-hydrometerological-dataset) |

---

## 3. Repository Directory Structure

```text
ganges-padma-hydrometerological-dataset/
├── raw/                              # Original collected records/files (BWDB raw CSVs & GIS boundaries)
│   ├── bgd_adm0.geojson              # Bangladesh national administrative boundary
│   ├── bgd_adm1.geojson              # Divisional boundaries
│   ├── cross_section_raw.csv         # River cross-section bathymetry
│   ├── discharge_daily_raw.csv       # Daily tributary discharge observations
│   ├── evaporation_daily_raw.csv     # Daily pan evaporation
│   ├── groundwater_weekly_raw.csv    # Weekly groundwater monitoring levels
│   ├── rainfall_daily_raw.csv        # Daily rain gauge observations
│   ├── sediment_raw.csv              # Suspended sediment concentration samples
│   ├── wl_3hourly_raw.csv            # 3-hourly tidal water levels
│   └── wl_daily_raw.csv              # Daily average river stage
├── processed/                        # Cleaned/derived data
│   ├── pointwise_hydromet_7day_long.csv # Primary 31-col multi-variable panel (91,033 × 31)
│   ├── pointwise_wl_7day_long.csv    # Stage-only compatibility long panel (91,033 × 16)
│   └── wide/                         # Target-specific wide ML design matrices
│       ├── wide_SW91_9L.csv          # Baruria Transit (56 km upstream of Padma Bridge)
│       ├── wide_SW91_9R.csv          # Goalundo Transit (60 km upstream)
│       ├── wide_SW93_4L.csv          # Bhagyakul (5 km upstream)
│       ├── wide_SW93_5L.csv          # Mawa (Padma Bridge north abutment)
│       └── wide_SW95.csv             # Sureswar (31 km downstream)
├── metadata/                         # Data dictionary, labels, provenance
│   ├── feature_dictionary.csv        # Complete variable definitions, types, and units
│   ├── feature_sets.json             # 4-tier feature groups per target station
│   ├── provenance.csv                # Data transformation lineage audit
│   ├── source_metadata.csv           # Gauge instrumentation and sampling metadata
│   ├── station_registry.csv          # 17 stations: IDs, coordinates, chainage, danger levels
│   └── LICENSE.txt                   # Creative Commons Attribution 4.0 International (CC BY 4.0)
└── README.md                         # Access and usage instructions
```

---

## 4. Primary Data Dictionary

| Variable / Field | Meaning | Type | Unit / Range | Allowed Values / Coding | Missing-Value Rule |
|---|---|---|---|---|---|
| `Station_Id` | BWDB hydrometric station code | Categorical | String | 17 codes (e.g., `SW93.5L`) | Mandatory key |
| `Date` | Observation calendar date | Date | YYYY-MM-DD | 2011-01-01 to 2025-12-31 | Complete sequence |
| `WL` | Daily average water level | Float | m MSL [0.00, 25.00] | Continuous numeric | Retained as NaN; flagged |
| `WLD-1` .. `WLD-7`| 1 to 7 antecedent stage lags | Float | m MSL [0.00, 25.00] | Continuous numeric | Complete-case window rule |
| `Rain` | Daily accumulated precipitation | Float | mm [0.0, 450.0] | Non-negative numeric | 0.0 if gauge confirmed dry |
| `Rain_D1` .. `D7` | 1 to 7 antecedent rainfall lags | Float | mm [0.0, 450.0] | Non-negative numeric | Complete-case window rule |
| `Rain_sum_3d` | 3-day cumulative rainfall total | Float | mm [0.0, 1000.0] | Non-negative numeric | Rolling sum |
| `Rain_sum_7d` | 7-day cumulative rainfall total | Float | mm [0.0, 2000.0] | Non-negative numeric | Rolling sum |
| `Rainfall_Level` | 4-tier precipitation class | Ordinal | Integer [0–3] | 0: None, 1: Light, 2: Mod, 3: Heavy | 0 if missing |
| `Discharge_m3s` | Mean daily tributary discharge | Float | $m^3/s$ [500, 140,000] | Continuous numeric | Available at 5 nodes |
| `Tidal_Range_m` | Sub-daily tidal fluctuation | Float | m [0.00, 4.50] | Non-negative numeric | Monitored downstream |
| `Period` | Bridge construction regime | Categorical | String | `BEFORE`, `DURING`, `AFTER` | Exact calendar mapping |
| `Risk_Class` | Flood danger classification | Categorical | String | `Normal`, `Warning`, `Danger`, `Severe`| BWDB threshold mapping |
| `Window_Complete`| 7-day lag validity indicator | Boolean | True / False | Binary flag | Excludes boundary seams |

---

## 5. Quickstart: How to Load and Use the Data

### 5.1 Installation

```bash
git clone https://github.com/Shakhoyat/ganges-padma-hydrometerological-dataset.git
cd ganges-padma-hydrometerological-dataset
pip install pandas numpy scikit-learn
```

### 5.2 Loading the Primary Multi-Variable Panel

```python
import pandas as pd

# Load the primary multi-variable long panel
df_long = pd.read_csv("processed/pointwise_hydromet_7day_long.csv", parse_dates=["Date"])

# Filter to complete lag windows (90,760 verified station-days)
df_clean = df_long[df_long["Window_Complete"] == True]

print(f"Total valid observations: {len(df_clean):,}")
print(df_clean[["Station_Id", "Date", "WL", "WLD-1", "Rain", "Period", "Risk_Class"]].head())
```

### 5.3 Loading Wide Matrices & Target Modeling (Mawa / Padma Bridge)

```python
import pandas as pd
import json

# 1. Load wide feature matrix for Mawa (SW93.5L) at Padma Bridge
df_mawa = pd.read_csv("processed/wide/wide_SW93_5L.csv", parse_dates=["Date"])

# 2. Load 4-tier feature set configuration
with open("metadata/feature_sets.json", "r") as f:
    feature_sets = json.load(f)["SW93.5L"]

# Available tiers: 'stage_only', 'hydromet', 'full_multivariate', 'forecast_safe'
features = feature_sets["hydromet"]

X = df_mawa[features]
y_stage = df_mawa["WL"]                    # Direct water level target
y_delta = df_mawa["Delta_WL_1d"]           # 1-day stage change target (Recommended)
y_risk  = df_mawa["Risk_Class"]            # 4-tier flood risk category

# 3. Chronological Train / Validation / Test Split (Strict leakage control)
train_mask = (df_mawa["Date"] < "2023-01-01")
test_mask  = (df_mawa["Date"] >= "2023-01-01")

X_train, y_train = X[train_mask], y_delta[train_mask]
X_test,  y_test  = X[test_mask],  y_delta[test_mask]

print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)} | Features: {len(features)}")
```

---

## 6. Machine Learning Benchmarks & Baseline Results

### The Critical Persistence Rule
Daily river stage displays an astronomical auto-correlation ($\rho = 0.999$). A trivial naive persistence model ($\hat{y}_t = y_{t-1}$) yields $R^2 = 0.998$ on raw stage. Reporting raw stage metrics without comparing to persistence constitutes scientific negligence. All benchmarks in this repository evaluate:
1. **Persistence Index ($PI$)**: Normalized improvement over persistence. Positive values ($PI > 0$) prove true machine learning skill.
2. **Cohen's Kappa Skill Score ($SS_\kappa$)**: Evaluates flood risk classification skill above persistence.
3. **Change Modeling ($\Delta WL$)**: Predicting daily stage delta $\Delta = y_t - y_{t-1}$ rather than raw level, completely eliminating artificial auto-regressive ceilings.

### Test Set Benchmark Evaluation (Mawa Target Node: `SW93.5L`)

| Model | Target Mode | $R^2$ | RMSE (m) | MAE (m) | Persistence Index ($PI$) | Risk $F_1$ Macro | $SS_\kappa$ | Conservation Audit |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Naive Persistence** | Stage ($y_{t-1}$) | 0.9980 | 0.0984 | 0.0652 | 0.0000 | 0.4721 | 0.0000 | Passive (Unchecked) |
| **Ridge Regression** | Stage ($y_t$) | 0.9984 | 0.0881 | 0.0592 | +0.1047 | 0.5412 | +0.1840 | Violates bounds |
| **Random Forest** | Stage ($y_t$) | 0.9989 | 0.0743 | 0.0487 | +0.2449 | 0.6234 | +0.3120 | Bounded by training |
| **XGBoost** | Stage ($y_t$) | 0.9991 | 0.0672 | 0.0435 | +0.3171 | 0.6582 | +0.3754 | Prone to overshoot |
| **CatBoost** | Stage ($y_t$) | 0.9992 | 0.0641 | 0.0418 | +0.3486 | 0.6720 | +0.3980 | Smooth |
| **LightGBM** | Stage ($y_t$) | 0.9991 | 0.0664 | 0.0429 | +0.3252 | 0.6651 | +0.3845 | Fast |
| **XGBoost** | Delta ($\Delta WL$) | 0.8120 | 0.0668 | 0.0431 | +0.3211 | 0.6610 | +0.3790 | Unbiased residual |
| **PI-STGNN (Proposed)** | Spatio-Temporal | **0.9995** | **0.0498** | **0.0315** | **+0.4939** | **0.7845** | **+0.5821** | **99.98% Mass Conserved** |

### Benchmark Baseline Summary

All benchmark results, training partitions, and performance metrics are documented in the companion course research report (`CSE 4112`), with model metrics verified across 5,479 daily records and 90,760 verified station-days using fixed random seed `20260913`.


---

## 7. CRediT Contributor Roles & Team Responsibilities

This dataset, pipeline, benchmark evaluation, and documentation were created by the following five-member team for **CSE 4112**:

| Student Name | Roll | Contribution (%) | Primary Responsibilities (CRediT) |
|---|:---:|:---:|---|
| **Md. Shakhoyat Rahman Shujon** *(Lead)* | 2107104 | **24%** | Conceptualization, Lead Data Curation, Spatio-Temporal Graph Architecture, Pipeline Engineering, LaTeX & Documentation |
| **Mohammad Moin Uddin Moin** | 2107111 | **23%** | Machine Learning Benchmarking, XGBoost/CatBoost Optimization, Cross-Validation Design, Persistence Index Framework |
| **Md. Tariful Islam Jony** | 2107119 | **23%** | Hydrometeorological Feature Engineering, 7-Day Lag Formulation, Quality Dimension Auditing, Data Cleaning |
| **Kamrul Islam** | 2107102 | **15%** | Data Verification, Missing Value Imputation Auditing, Station Registry Geocoding, Descriptive Statistics |
| **Tanha Sayed Ahona** | 2107093 | **15%** | Visualizations & Cartography, Figure Generation, Ethical & FAIR Compliance Review, Data Dictionary Auditing |

---

## 8. Ethics, FAIR Readiness & License

### Ethical Compliance & Privacy
The dataset contains solely abiotic, physical hydrometeorological observations collected in public river corridors. No human subjects, personal data, or private spatial properties are included. Public infrastructure nodes (Padma Bridge) are documented using publicly accessible geographic and hydrometric station coordinates.

### FAIR Principles Compliance
- **Findable**: Documented with rich schemas, persistent file identifiers, standard column naming, and explicit metadata files (`station_registry.csv`, `feature_dictionary.csv`, `feature_sets.json`).
- **Accessible**: Openly hosted on GitHub with no authentication barriers or paywalls.
- **Interoperable**: Formatted in RFC 4180 compliant CSV, GeoJSON standard EPSG:4326, and open JSON.
- **Reusable**: Released under the unrestricted **Creative Commons Attribution 4.0 International (CC BY 4.0)** license with full provenance logs and build recipes.

---

## 9. Citation

If you use this dataset, benchmark framework, or code in your research, please cite:

```bibtex
@dataset{shujon2026gangespadma,
  author    = {Md. Shakhoyat Rahman Shujon and Mohammad Moin Uddin Moin and Md. Tariful Islam Jony and Kamrul Islam and Tanha Sayed Ahona},
  title     = {Multi-Variable Hydrometeorological and Point-Wise Water Level Dataset with 7-Day Lags along the Ganges--Padma River Corridor (2011--2025)},
  year      = {2026},
  publisher = {GitHub / Bangladesh Water Development Board Archive},
  url       = {https://github.com/Shakhoyat/ganges-padma-hydrometerological-dataset},
  note      = {Course CSE 4112 Dataset Article, Department of Computer Science and Engineering, Khulna University of Engineering \& Technology (KUET)}
}
```

---

## 10. Contact & Support

For questions, issues, or collaborative research involving the Ganges–Padma hydrometeorological corridor:
- **Lead Author**: Md. Shakhoyat Rahman Shujon (`shakoyatsujon@gmail.com` / GitHub: [@Shakhoyat](https://github.com/Shakhoyat))
- **Affiliation**: Department of Computer Science and Engineering, Khulna University of Engineering & Technology (KUET), Khulna-9203, Bangladesh.
