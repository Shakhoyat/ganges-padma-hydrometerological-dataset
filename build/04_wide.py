"""Step 04 - one multi-variable wide design matrix per target station.

Fulfills Table 2 from the task brief and extends it with rainfall lags, discharge nodes,
and environmental covariates across all 3 temporal bridge eras:
  - Target own stage lags: WL (to be predicted), WLD-1 .. WLD-7, Delta_WL_1d
  - Target own rainfall lags: Rain, Rain_D1 .. Rain_D7, Rain_sum_3d, Rain_sum_7d, Rainfall_Level
  - Upstream prior points (P<id>): P<id>_WL, P<id>_WLD-1..7, P<id>_Rain, P<id>_Rain_D1..7
  - Boundary and corridor discharge nodes (m3/s): Jamuna (SW46.9L), Ganges (SW90), Padma (SW91.9L), Bridge (SW93.5L), Meghna (SW273)
  - Covariates: Evaporation (mm), Groundwater depth (m), Sediment concentration (ppm), Sub-daily tidal range (m)
  - Temporal regimes: Period (BEFORE / DURING / AFTER)
  - Flood risk classification: Risk_Class (Normal / Warning / Danger / Severe)

Outputs
    data/wide/wide_<station>.csv
    data/wide/feature_sets.json
    data/wide/feature_dictionary.csv
    results/wide_row_budget.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
import corridor as C  # noqa: E402


def predictors_for(target: str) -> list[str]:
    """Every station with a lower corridor id, minus the held-out gauges."""
    return [s for s in C.MAIN_STEM[: C.STATION_ID[target] - 1]
            if s not in C.EXCLUDED_PREDICTORS]


def main() -> None:
    long = pd.read_csv(C.DATA / "pointwise_hydromet_7day_long.csv", parse_dates=["Date"])
    reg = pd.read_csv(C.DATA / "station_registry.csv").set_index("Station_ID")

    wl_lags = C.lag_columns()
    rain_lags = C.rain_lag_columns()

    per_station = {
        sid: g.drop_duplicates("Date").set_index("Date")
        for sid, g in long.groupby("Station_ID")
    }

    # Extract discharge series for key nodes
    q_nodes = {}
    for qsid in C.DISCHARGE_STATIONS:
        if qsid in per_station:
            q_nodes[qsid] = per_station[qsid]["Discharge_m3s"].ffill().fillna(0.0)

    budget, sets, dictionary = [], {}, []

    for tgt in C.TARGETS:
        preds = predictors_for(tgt)
        t = per_station[tgt]

        block = {}
        # Target own stage & lags
        block["WL"] = t["WL"]
        for k in range(1, C.N_LAGS + 1):
            block[f"WLD-{k}"] = t[f"WLD-{k}"]
        block["Delta_WL_1d"] = t["WL"] - t["WLD-1"]

        # Target own rainfall & lags
        block["Rain"] = t["Rain"]
        for k in range(1, C.N_LAGS + 1):
            block[f"Rain_D{k}"] = t[f"Rain_D{k}"]
        block["Rain_sum_3d"] = t["Rain_sum_3d"]
        block["Rain_sum_7d"] = t["Rain_sum_7d"]
        block["Rainfall_Level"] = t["Rainfall_Level"]

        # Target own discharge (if gauged)
        if pd.notna(t["Discharge_m3s"].notna().sum()) and t["Discharge_m3s"].notna().sum() > 0:
            block["Discharge_m3s"] = t["Discharge_m3s"]

        # Environmental covariates
        block["Evap_mm"] = t["Evap_mm"]
        block["GW_Depth_m"] = t["GW_Depth_m"]
        block["Sed_Conc_ppm"] = t["Sed_Conc_ppm"]
        block["Tidal_Range_m"] = t["Tidal_Range_m"]

        complete = t["Window_Complete"].eq(True)

        stage_same_day, stage_lagged = [], []
        rain_same_day, rain_lagged = [], []
        q_cols = []

        # <<snip:wide>>
        # Upstream prior points
        for p in preds:
            pid = C.STATION_ID[p]
            src = per_station[p].reindex(t.index)

            # Upstream water level lags
            for col in wl_lags:
                name = f"P{pid:02d}_{col}"
                block[name] = src[col]  # <<endsnip>>
                (stage_same_day if col == "WL" else stage_lagged).append(name)
                if tgt == C.TARGETS[-1]:
                    dictionary.append({
                        "Column": name, "Prior_Point_Id": pid, "Station_ID": p,
                        "Station_Name": reg.at[p, "Station_Name"],
                        "Chainage_km": reg.at[p, "Chainage_km"],
                        "Quantity": ("daily stage, same day" if col == "WL"
                                     else f"daily stage, t-{col.split('-')[1]} d"),
                        "Units": "mMSL",
                    })

            # Upstream rainfall lags
            for col in rain_lags:
                name = f"P{pid:02d}_{col}"
                block[name] = src[col]
                (rain_same_day if col == "Rain" else rain_lagged).append(name)
                if tgt == C.TARGETS[-1]:
                    dictionary.append({
                        "Column": name, "Prior_Point_Id": pid, "Station_ID": p,
                        "Station_Name": reg.at[p, "Station_Name"],
                        "Chainage_km": reg.at[p, "Chainage_km"],
                        "Quantity": ("daily rainfall, same day" if col == "Rain"
                                     else f"daily rainfall, t-{col.split('_D')[1]} d"),
                        "Units": "mm",
                    })

            complete &= src["Window_Complete"].eq(True)

        # Discharge features from major upstream/boundary nodes
        for qsid, qseries in q_nodes.items():
            if C.STATION_ID[qsid] < C.STATION_ID[tgt] or qsid in ["SW46.9L", "SW90", "SW91.9L"]:
                qname = f"Q_{qsid.replace('.', '_')}"
                block[qname] = qseries.reindex(t.index)
                q_cols.append(qname)
                for qk in (1, 3, 7):
                    qlag_name = f"{qname}_D{qk}"
                    block[qlag_name] = qseries.shift(qk).reindex(t.index)
                    q_cols.append(qlag_name)

        d = pd.DataFrame(block, index=t.index).reset_index(names="Date")
        d.insert(0, "Id", C.STATION_ID[tgt])
        d.insert(1, "Station_ID", tgt)
        d["Period"] = C.period_of(d.Date).to_numpy()
        d["Risk_Class"] = C.risk_class(d.WL, float(reg.at[tgt, "Danger_Level_mMSL"]))
        d["Window_Complete"] = complete.to_numpy()

        rows = d[d.Window_Complete].drop(columns=["Window_Complete"])
        path = C.WIDE / f"wide_{tgt.replace('.', '_')}.csv"
        rows.to_csv(path, index=False, float_format="%.3f", date_format="%Y-%m-%d")

        own_stage_lags = [f"WLD-{k}" for k in range(1, C.N_LAGS + 1)]
        own_rain_lags = [f"Rain_D{k}" for k in range(1, C.N_LAGS + 1)] + ["Rain_sum_3d", "Rain_sum_7d"]
        covar_cols = ["Evap_mm", "GW_Depth_m", "Sed_Conc_ppm", "Tidal_Range_m"]

        sets[tgt] = {
            "target": "WL",
            "target_delta": "Delta_WL_1d",
            "target_class": "Risk_Class",
            "n_prior_points": len(preds),
            "prior_point_ids": [C.STATION_ID[p] for p in preds],
            "excluded_prior_points": [
                {"id": C.STATION_ID[s], "station": s,
                 "reason": "1369-day gap inside the construction period"}
                for s in C.EXCLUDED_PREDICTORS if C.STATION_ID[s] < C.STATION_ID[tgt]],
            # Feature Sets
            "spec": own_stage_lags + stage_same_day + stage_lagged,
            "stage_only": own_stage_lags + stage_same_day + stage_lagged,
            "hydromet": own_stage_lags + stage_same_day + stage_lagged + ["Rain"] + own_rain_lags + rain_same_day + rain_lagged + q_cols,
            "full_multivariate": own_stage_lags + stage_same_day + stage_lagged + ["Rain"] + own_rain_lags + rain_same_day + rain_lagged + q_cols + covar_cols,
            "forecast_safe": own_stage_lags + stage_lagged + own_rain_lags + rain_lagged + [c for c in q_cols if "_D" in c],
        }

        for name, a, b in C.PERIODS:
            w = rows[(rows.Date >= a) & (rows.Date <= b)]
            budget.append({
                "Station_ID": tgt, "Id": C.STATION_ID[tgt], "Period": name,
                "N_Rows": len(w), "N_Prior_Points": len(preds),
                "N_Features_StageOnly": len(sets[tgt]["stage_only"]),
                "N_Features_Hydromet": len(sets[tgt]["hydromet"]),
                "N_Features_Full": len(sets[tgt]["full_multivariate"]),
                "N_Features_ForecastSafe": len(sets[tgt]["forecast_safe"]),
                "First": w.Date.min().date() if len(w) else None,
                "Last": w.Date.max().date() if len(w) else None,
            })
        print(f"  {tgt:<9} id={C.STATION_ID[tgt]:>2}  {len(preds):>2} prior points  "
              f"{len(rows):>5} rows  "
              f"{len(sets[tgt]['stage_only']):>3} stage / "
              f"{len(sets[tgt]['hydromet']):>3} hydromet / "
              f"{len(sets[tgt]['full_multivariate']):>3} full / "
              f"{len(sets[tgt]['forecast_safe']):>3} safe features  -> {path.name}")

    (C.WIDE / "feature_sets.json").write_text(json.dumps(sets, indent=2), encoding="utf-8")
    pd.DataFrame(dictionary).to_csv(C.WIDE / "feature_dictionary.csv", index=False)
    bud = pd.DataFrame(budget)
    bud.to_csv(C.RESULTS / "wide_row_budget.csv", index=False)
    print()
    print(bud.pivot(index="Station_ID", columns="Period",
                    values="N_Rows").reindex(C.TARGETS)[C.PERIOD_NAMES].to_string())


if __name__ == "__main__":
    main()

