# Changelog

All notable changes to this dataset. Versions follow the dataset report.

## v3.1 (September 2026)

* Long panel: added eight leakage-free features (`Rainfall_mm`, `Rain_3dSum`, `WL_Range_D-1`, `DOY_sin`, `DOY_cos`, `Latitude`, `Longitude`, `Tidal`) and the classification label `WL_Trend`.
* Wide matrices: added the two Meghna boundary gauges (`B16_*` Bhairab Bazar, `B17_*` Chandpur); targets reduced to Baruria, Bhagyakul and Sureswar (Goalundo duplicates Baruria; Mawa sits on the bridge).
* Removed `processed/ganges_padma_stage_only_subset_2011_2025.csv`, the Goalundo and Mawa wide matrices, and `metadata/feature_sets.json`.
* Metadata: added `checksums_sha256.csv`, `feature_screening*.csv`, `trend_classification_wide.csv`, `validation_daily_vs_3hourly.csv` and `danger_levels.csv` (external reference).
* Code: added `code/` (build, feature screening, baselines, benchmark) and `code/analysis/` (report Section 10); results in `results/`.
* Documentation: README restructured to the CSE 4112 dataset-report template; added `CITATION.cff`, `requirements.txt` and a root `LICENSE`.

## v3.0 (September 2026)

* Long panel reduced to the course brief's 11 columns (`Id`, `Date`, `WL`, `WLD-1..7`, `Rainfall_Level`); `Station_ID` dropped (Id is the key).
* Raw directory replaced with the 50 unmodified BWDB workbooks of invoice 2608295209.

## v2.0 (September 2026)

* First public release: 15-year panel with 30 columns, including derived covariates later found to leak same-day stage (superseded by v3.0).
