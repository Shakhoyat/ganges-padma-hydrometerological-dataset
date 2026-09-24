# A 15-Year Point-wise Water Level and Rainfall Dataset with 7-Day Lags along the Ganges–Padma River Corridor, Bangladesh (2011–2025)

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/version-v3.1-green.svg)](CHANGELOG.md)

Dataset release for the **CSE 4112: Machine Learning Laboratory Dataset Report**, Department of Computer Science and Engineering, Khulna University of Engineering & Technology (KUET). The README follows the structure of the course's dataset-report template.

---

## 1. Dataset article information

| Item | Details |
|---|---|
| Dataset title | A 15-Year Point-wise Water Level and Rainfall Dataset with 7-Day Lags along the Ganges–Padma River Corridor, Bangladesh (2011–2025) |
| Group and students | Group B2(03): Md. Shakhoyat Rahman Shujon (2107104), Mohammad Moin Uddin Moin (2107111), Md. Tariful Islam Jony (2107119), Kamrul Islam (2107102), Tanha Sayed Ahona (2107093) |
| Affiliation | Department of CSE, KUET, Khulna-9203, Bangladesh |
| Corresponding student | Md. Shakhoyat Rahman Shujon (2107104), shujon2107104@stud.kuet.ac.bd |
| Supervisor | Prof. Dr. Muhammad Aminul Haque Akhand, Department of CSE, KUET |
| Keywords | river stage; time-lagged features; upstream gauges; Padma Bridge; flood early warning; chronological split |
| Dataset version | v3.1 (September 2026); history in [`CHANGELOG.md`](CHANGELOG.md) |
| Repository | https://github.com/Shakhoyat/ganges-padma-hydrometerological-dataset |
| License | CC BY 4.0 for processed files and code; original records © BWDB ([`LICENSE`](LICENSE)) |
| Source | Bangladesh Water Development Board (BWDB), Processing & Flood Forecasting Circle, Dhaka; invoice 2608295209; Memo No. KUET/CSE/26/262 |

## 2. Abstract

Flood warnings in Bangladesh are issued as daily water levels (stage) at gauging stations, yet historical stage records are distributed as separate spreadsheets that are not ready for machine learning (ML). This dataset compiles 50 workbooks purchased from BWDB into an ML-ready benchmark for 17 gauging stations on the Ganges, Padma, Jamuna and Meghna rivers, covering every day from 1 January 2011 to 31 December 2025. The long panel (91,033 station-days) contains the daily average stage, its seven daily lags, a four-class rainfall level, eight leakage-free covariates and a three-class rise/fall label. Three wide matrices (4,715–4,737 rows, 103–127 predictors) pair each target gauge around the Padma Multipurpose Bridge with the lagged stage of every upstream gauge and of two Meghna boundary gauges. No value was interpolated or synthesised; all lags were verified automatically, and the daily averages agree with independent 3-hourly readings to within 0.006–0.012 m. The unmodified raw workbooks, including discharge, suspended-sediment and river cross-section surveys, are released with the processed files, and a single script regenerates the release.

## 3. Dataset specifications

| Item | Information |
|---|---|
| Subject / domain | Environment; water resources engineering; machine learning for hydrology |
| Specific subject area | Daily river-stage prediction and bridge-reach change along the Ganges–Padma corridor |
| Type of data | Tabular time series. Raw: 50 BWDB workbooks. Processed: one long panel and three wide ML matrices |
| File formats | Raw: XLSX (as delivered). Processed and metadata: CSV (UTF-8). Code: Python |
| Unit of observation | One station-day (long panel); one target-station day with all upstream predictors (wide matrices) |
| Data source location | Ganges–Padma, Jamuna and Meghna rivers, Bangladesh; 23.23–25.13°N, 88.11–90.99°E |
| Collection period | 2011-01-01 to 2025-12-31 (5,479 days) |
| Dataset size | Long panel 91,033 × 20 (10.3 MB); wide matrices 4,737 × 106, 4,737 × 114, 4,715 × 130 |
| Targets / labels | Regression: daily average water level `WL` (mMSL). Classification: `WL_Trend` (falling / steady / rising), a fixed rule on `WL`; no manual labels |
| Accessibility | Public repository; plain CSV; no login |

## 4. Value of the data

* **ML-ready:** the long panel and wide matrices follow the course brief's Table 1 and Table 2 layouts; every column has a documented role, and chronological splits with a persistence benchmark are provided.
* **Network-ordered:** station Ids follow the river (for every connection u → v, Id(u) < Id(v)), so upstream predictors and flood-wave travel times over 291 km are unambiguous.
* **Spans a major structure:** gauges and bed surveys on both sides of the Padma Bridge cover the periods before, during and after construction.
* **Reusers:** BWDB's Flood Forecasting and Warning Centre, the Bridges Division, hydrology and ML researchers, and instructors.

## 5. Repository / folder structure

```text
ganges-padma-hydrometerological-dataset/
├── raw/                    50 original BWDB .xlsx workbooks, unmodified                  [raw]
├── processed/                                                                            [processed]
│   ├── ganges_padma_hydromet_dataset_2011_2025.csv   long panel, 91,033 × 20 (brief's Table 1 + features)
│   └── wide/                                         one ML design matrix per target (brief's Table 2)
│       ├── wide_sw91_9l_baruria_transit.csv          Baruria,   56 km upstream of the bridge (4,737 × 106)
│       ├── wide_sw93_4l_bhagyakul.csv                Bhagyakul,  5 km upstream of the bridge (4,737 × 114)
│       └── wide_sw95_sureswar.csv                    Sureswar,  31 km downstream of the bridge (4,715 × 130)
├── metadata/                                                                             [metadata]
│   ├── feature_dictionary.csv          data dictionary: role, meaning, type, unit, missing-value rule
│   ├── station_registry.csv            Id → BWDB station, coordinates, chainage, rain gauge, coverage
│   ├── danger_levels.csv               BWDB/FFWC danger levels (external reference; not a model input)
│   ├── provenance.csv                  every raw workbook: rows read, rows used, purpose
│   ├── source_metadata.csv             header block of each workbook as delivered
│   ├── validation_daily_vs_3hourly.csv daily average checked against 3-hourly readings
│   ├── feature_screening*.csv          candidate-feature screens (why each column is or is not released)
│   ├── trend_classification_wide.csv   WL_Trend classification on the wide matrices
│   ├── checksums_sha256.csv            SHA-256 of every processed file
│   └── LICENSE.txt                     CC BY 4.0
├── code/
│   ├── build_dataset.py                raw/ → processed/ + metadata/ (deterministic, ~40 s)
│   ├── screen_features.py              feature screening → metadata/feature_screening*.csv
│   ├── train_baselines.py              baseline models (persistence, Ridge, LightGBM, LSTM, RS-GNN)
│   ├── validation_r3.py                report benchmark (Problem 2) on Kaggle 2× T4 or CPU
│   └── analysis/                       report Section 10, Problems 1 and 3–18 (run_all.py)
├── results/
│   ├── r3/                             benchmark metrics, tables, predictions, run manifest
│   └── analysis/                       Section 10 outputs: csv/, tables/, figures/
├── README.md  CHANGELOG.md  CITATION.cff  LICENSE  requirements.txt
```

## 6. Data acquisition, materials and methods

* **Source and sampling:** daily water levels were requested for all 13 main-stem gauges from Panka to Sureswar, two Jamuna and two Meghna boundary gauges, and nearby rain gauges, 2011–2025. No station or day was sampled out.
* **Collection setting:** records compiled by BWDB's Processing & Flood Forecasting Circle (72 Green Road, Dhaka-1205), which approved a 90% fee waiver on the group's academic request (Memo No. KUET/CSE/26/262, 18 August 2026); delivered against invoice 2608295209 (31 August 2026). Stations lie in 11 districts from Chapainawabganj to Chandpur; Ids 12–17 are tidal.
* **Instruments and software:** BWDB reports stage in mMSL as daily maximum, minimum and average (3-hourly at Mawa and Sureswar) and rainfall as daily totals; instrument models are not stated in the source. Python 3.12, pandas 2.3.3, openpyxl ([`requirements.txt`](requirements.txt)).
* **Annotation:** none manual. `WL_Trend` is a fixed rule on `WL − WLD-1` (falling < −0.03 m, rising > +0.03 m); the ±0.03 m band exceeds the 95th-percentile disagreement between daily and 3-hourly means (0.014–0.028 m). Classes: 34.9 / 37.6 / 27.5%.
* **Provenance and versioning:** raw files are never modified; `provenance.csv` and `source_metadata.csv` document every workbook; versions are listed in [`CHANGELOG.md`](CHANGELOG.md).

## 7. Data records and data dictionary

### 7.1 Long panel (`processed/ganges_padma_hydromet_dataset_2011_2025.csv`)

One row per station-day with a reported daily average. The first 11 columns are the brief's Table 1; the next 8 are leakage-free features; the last is a classification label. The full dictionary is [`metadata/feature_dictionary.csv`](metadata/feature_dictionary.csv).

| Column | Role | Meaning | Unit / coding | Missing-value rule |
|---|---|---|---|---|
| `Id` | key | Corridor point 1–17 (`station_registry.csv`) | 1–17 | never missing |
| `Date` | key | Observation date | YYYY-MM-DD | never missing |
| `WL` | **regression target** | Daily average water level | mMSL | row omitted if not observed |
| `WLD-1` … `WLD-7` | feature | WL 1…7 **calendar days** earlier, same station | mMSL | blank if that day was not observed; never interpolated |
| `Rainfall_Level` | feature | Rain class at the nearest rain gauge (BMD scale) | 0 < 1 mm; 1: 1–10; 2: >10–22; 3: > 22 mm | blank if no gauge record (498 rows) |
| `Rainfall_mm` | feature | Daily rain total at the same gauge | mm | as above |
| `Rain_3dSum` | feature | Rain summed over `Date−2 … Date` | mm | blank unless all 3 days recorded (588 rows) |
| `WL_Range_D-1` | feature | Previous day's maximum − minimum WL (tidal range) | m | blank if `Date−1` not observed (39 rows) |
| `DOY_sin`, `DOY_cos` | feature | Seasonal phase, sin/cos(2π·doy/365.25) | −1…1 | never missing |
| `Latitude`, `Longitude` | feature (static) | Gauge coordinates | decimal degrees | never missing |
| `Tidal` | feature (static) | BWDB station type | 1 tidal (Ids 12–17), 0 non-tidal | never missing |
| `WL_Trend` | **classification target** | Day-on-day change as a class; **never an input for WL** | 0 falling, 1 steady, 2 rising | blank if `WLD-1` blank (39 rows) |

### 7.2 Wide ML matrices (`processed/wide/`)

```
Id, Date, WL (target), WLD-1..WLD-7,
P01_WL, P01_WLD-1..P01_WLD-7, ..., P<k>_WL, P<k>_WLD-1..P<k>_WLD-7     upstream points (brief's layout)
B16_WL, B16_WLD-1..B16_WLD-7, B17_WL, B17_WLD-1..B17_WLD-7              Meghna boundary gauges
```

| Target | Id | Position | Upstream points | Boundary | Rows | Predictors |
|---|---|---|---|---|---|---|
| Baruria Transit (SW91.9L) | 11 | 56 km upstream of the bridge | 1–10 | 16, 17 | 4,737 | 103 |
| Bhagyakul (SW93.4L) | 12 | 5 km upstream of the bridge | 1–11 | 16, 17 | 4,737 | 111 |
| Sureswar (SW95) | 15 | 31 km downstream of the bridge | 1–13 (not 14) | 16, 17 | 4,715 | 127 |

Complete rows only. Tarpasa (Id 14) is not a predictor (1,369-day gap). The Meghna boundary gauges raise Ridge's test persistence index from 0.78 to 0.85 at Bhagyakul.

### 7.3 Composition

91,033 of 93,143 possible station-days (97.7%); 99.70% have all seven lags. Coverage exceeds 99.4% at 14 stations; exceptions: Tarpasa 74.5%, Bahadurabad 95.0%, Sardah 95.6%. Derived four-class flood-risk label (relative to `danger_levels.csv`): 88.6% Normal, 9.5% Warning, 1.9% Danger or Severe.

## 8. Preprocessing, curation and leakage control

* **Cleaning:** header rows located, footers dropped, dates parsed. `build_dataset.py` stops on any failure of: unique `(Id, Date)`; WL within −2…30 mMSL; `MIN ≤ AVERAGE ≤ MAX`; valid rain classes; `Rain_3dSum ≥` that day's rain; `WL_Trend` consistent with `WL − WLD-1`; every `WLD-k` equal to the WL observed k days earlier; Ids increasing downstream; wide files without blanks.
* **Missing data:** stations are reindexed to the full calendar; gaps stay blank and are never interpolated.
* **Outliers:** values kept as reported; 15 station-days (0.02%) change by more than 1 m in a day.
* **Feature selection:** a candidate is released only if measured, free of same-day target stage, nearly complete, and not harmful to test skill (`metadata/feature_screening*.csv`). Rejected: same-day max/min (leak `WL`), discharge (rated from same-day stage; ends 2024), evaporation (single station, negative values), groundwater and sediment (weekly/monthly).
* **Augmentation / de-identification:** none; no personal data.
* **Leakage control:** split by date, never at random (lag-1 autocorrelation ≈ 0.99): train 2011–2021, validation 2022, test 2023–2025. Lags look backward only. `WL_Trend` is never an input for `WL`. Same-day upstream columns (`P<id>_WL`, `B<id>_WL`) make the wide layout a nowcast; drop them for 1-day-ahead forecasting.

## 9. Technical validation and data quality

| Dimension | Evidence |
|---|---|
| Completeness | 97.7% of station-days; 99.70% with all 7 lags |
| Consistency | Automated schema, range and lag-identity checks (above) |
| Measurement quality | Daily average vs mean of 3-hourly readings: Mawa 0.006 m (r = 0.99999, 4,018 days), Sureswar 0.012 m (r = 0.99992, 4,017 days) |
| Integrity | Raw files unmodified; SHA-256 of every processed file in `metadata/checksums_sha256.csv` |
| Reproducibility | `python code/build_dataset.py` rebuilds the release byte-identically |

## 10. ML use demonstration (summary of the report's Section 10)

Stage-nowcast benchmark, test 2023–2025 (`code/validation_r3.py`; RMSE in cm, persistence index PI in brackets; PI = 1 − SSE_model / SSE_persistence):

| Target | Persistence | Ridge | T-GCN | LSTM | RS-GNN (custom) |
|---|:---:|:---:|:---:|:---:|:---:|
| Baruria | 9.9 | 2.8 (0.92) | 7.1 (0.48) | 3.5 (0.87) | 2.8 (0.92) |
| Bhagyakul | 9.8 | 3.8 (0.85) | 5.8 (0.65) | 4.3 (0.81) | 3.5 (0.87) |
| Sureswar | 10.5 | 5.7 (0.70) | 8.6 (0.33) | 6.6 (0.60) | 5.7 (0.71) |

The report demonstrates 18 problems (regression, classification, clustering); all are reproducible with `python code/analysis/run_all.py` (outputs in `results/analysis/`). Selected findings:

* **Surge travel time:** a change at Panka reaches Baruria after ~1 day, Bhagyakul after 1–2 days and Sureswar after ~3 days (celerity 133.5 km/day); the 35 largest surges arrive after a median 2, 4 and 5 days.
* **Flood-risk warning:** 7 days ahead, a random forest detects 93% of Baruria's warning days (CSI 0.81 vs 0.74 for persistence).
* **Padma Bridge:** no backwater rise at flood flow; the scour pool on the bridge line filled (thalweg −30.9 → −9.7 mMSL) and the dry-season channel area shrank by 31–49% around the bridge, but not at the Ganges control; 7 km downstream the deepest channel now runs along the Lohajang bank; the dry-season tidal range at Mawa grew from 33 to 54 cm.
* **Model transfer:** a Sureswar model trained before construction scores PI −0.10 after opening; forecast models must be recalibrated after in-channel works.

## 11. Usage notes

```python
import pandas as pd

long = pd.read_csv("processed/ganges_padma_hydromet_dataset_2011_2025.csv", parse_dates=["Date"],
                   dtype={"Rainfall_Level": "Int64", "WL_Trend": "Int64"})
features = long.columns.drop(["Id", "Date", "WL", "WL_Trend"])      # 16 features
y_reg, y_clf = long["WL"], long["WL_Trend"]

wide = pd.read_csv("processed/wide/wide_sw93_4l_bhagyakul.csv", parse_dates=["Date"])
X, y = wide.drop(columns=["Id", "Date", "WL"]), wide["WL"]
train = wide.Date < "2022-01-01"                                     # chronological, never random
valid = (wide.Date >= "2022-01-01") & (wide.Date < "2023-01-01")
test = wide.Date >= "2023-01-01"
```

* Report the persistence index alongside RMSE; for `WL_Trend`, compare with "same class as yesterday".
* Bridge periods: before 2011-01-01 to 2014-11-25; during 2014-11-26 to 2022-06-25; after 2022-06-26 to 2025-12-31.
* **Inappropriate uses:** R²/NSE without a persistence comparison; random k-fold cross-validation; treating `Rainfall_Level` at Bahadurabad or Bhairab Bazar as local rain.

## 12. Limitations and known biases

17 gauges on one river system; Tarpasa has a 1,369-day gap. Rainfall comes from the nearest of eight gauges (up to 130 km away) and explains little of the stage, which is driven by upstream inflow. Daily averages smooth the tidal cycle at Ids 12–17. Danger levels are reference values (FFWC roster with fitted offsets), and Danger/Severe days are rare (1.9%). Bridge analyses rest on 3.5 post-opening years with mild floods, annual cross-section surveys and a sediment series with a probable sampling change in 2015; they are descriptive, not causal. The discharge workbooks' own stage column changes datum by 0.71 m at Mawa in 2022; use the released `WL` instead.

## 13. Ethics, privacy and responsible use

No human participants or personal data. The original records are © BWDB, obtained for academic use under invoice 2608295209 with a 90% fee waiver; the processed data are shared for non-commercial research and teaching with attribution to BWDB and will be withdrawn at BWDB's request. No generative-AI or synthetic data were used in collection, labelling or augmentation.

## 14. Data and code availability

| Item | Details |
|---|---|
| Repository | GitHub, https://github.com/Shakhoyat/ganges-padma-hydrometerological-dataset |
| DOI | not yet minted (a Zenodo DOI can be minted from a GitHub release) |
| Access | public; clone or download the CSV files |
| License | CC BY 4.0 ([`LICENSE`](LICENSE)); cite BWDB as the original source |
| Code | `code/` (dataset build, screening, baselines) and `code/analysis/` (report Section 10) |
| Version | v3.1, see [`CHANGELOG.md`](CHANGELOG.md) |
| Environment | [`requirements.txt`](requirements.txt); Python ≥ 3.10; seed 20260913 |

## 15. Student contributions (CRediT)

| Roll | Student | % | Main contributions |
|---|---|:---:|---|
| 2107104 | Md. Shakhoyat Rahman Shujon (lead) | 24 | Conceptualization, data acquisition from BWDB, methodology, project administration, writing |
| 2107111 | Mohammad Moin Uddin Moin | 23 | Software, ML baselines (Ridge, T-GCN, LSTM, RS-GNN), Kaggle training |
| 2107119 | Md. Tariful Islam Jony | 23 | Data curation, extraction, 7-day lag structure, provenance |
| 2107102 | Kamrul Islam | 15 | Validation: 3-hourly check, quality checks, target-station analysis |
| 2107093 | Tanha Sayed Ahona | 15 | Visualization, documentation, FAIR review |

## 16. Acknowledgements

We thank our supervisor, Prof. Dr. Muhammad Aminul Haque Akhand (Department of CSE, KUET), for guidance and for writing the fee-waiver application; Prof. Dr. K. A. Mahmud (Head, Department of CSE, KUET) for endorsing it; the course teachers of CSE 4112, including Nabil Faiyaz Sadi; and the Processing & Flood Forecasting Circle, BWDB, for approving the waiver and supplying the data. No external funding was received; the authors declare no competing interests.

## 17. Citation

See [`CITATION.cff`](CITATION.cff), or:

```bibtex
@dataset{shujon2026gangespadma,
  author    = {Md. Shakhoyat Rahman Shujon and Mohammad Moin Uddin Moin and Md. Tariful Islam Jony and Kamrul Islam and Tanha Sayed Ahona},
  title     = {A 15-Year Point-wise Water Level and Rainfall Dataset with 7-Day Lags along the Ganges--Padma River Corridor, Bangladesh (2011--2025)},
  year      = {2026},
  version   = {3.1},
  publisher = {GitHub},
  url       = {https://github.com/Shakhoyat/ganges-padma-hydrometerological-dataset},
  note      = {CSE 4112 Dataset Report, Department of CSE, Khulna University of Engineering \& Technology (KUET). Source data: Bangladesh Water Development Board}
}
```
