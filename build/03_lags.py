"""Step 03 - the point-wise dataset: Id, Date, WL & 7-day lags, Rainfall & 7-day lags, Discharge, and Covariates.

This table fulfills both Table 1 from the task brief and the complete multi-variable
panel for the Ganges-Padma corridor:
  - Water level: WL, WLD-1 .. WLD-7
  - Rainfall: Rain, Rain_D1 .. Rain_D7, Rain_sum_3d, Rain_sum_7d, Rainfall_Level (0/1/2/3)
  - Discharge: Discharge_m3s
  - Covariates: Evap_mm, GW_Depth_m, Sed_Conc_ppm, Tidal_Range_m
  - Temporal Regimes: Period (BEFORE / DURING / AFTER)
  - Flood Risk Target: Risk_Class (Normal / Warning / Danger / Severe)

Outputs
    data/pointwise_hydromet_7day_long.csv
    data/pointwise_wl_7day_long.csv
    results/lag_completeness.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
import corridor as C  # noqa: E402


def ffill_series(series: pd.Series, max_age: int) -> pd.Series:
    """Forward-fill a series up to max_age days."""
    s = series.copy()
    idx = np.arange(len(s))
    last = np.where(s.notna().values, idx, np.nan)
    last = pd.Series(last).ffill().values
    age = idx - last
    filled = s.ffill()
    stale = age > max_age
    filled[stale] = np.nan
    return filled


def main() -> None:
    panel_wl = pd.read_csv(C.INTERIM / "wl_daily_raw.csv", parse_dates=["Date"])
    panel_rain = pd.read_csv(C.INTERIM / "rainfall_daily_raw.csv", parse_dates=["Date"])
    panel_q = pd.read_csv(C.INTERIM / "discharge_daily_raw.csv", parse_dates=["Date"])
    panel_ev = pd.read_csv(C.INTERIM / "evaporation_daily_raw.csv", parse_dates=["Date"])
    panel_gw = pd.read_csv(C.INTERIM / "groundwater_weekly_raw.csv", parse_dates=["Date"])
    panel_sed = pd.read_csv(C.INTERIM / "sediment_raw.csv", parse_dates=["Date"])
    panel_3h = pd.read_csv(C.INTERIM / "wl_3hourly_raw.csv", parse_dates=["Datetime"])

    reg = pd.read_csv(C.DATA / "station_registry.csv").set_index("Station_ID")
    calendar = pd.date_range(C.STUDY_START, C.STUDY_END, freq="D")

    # Pre-process Evaporation (CL406)
    ev_series = panel_ev.set_index("Date")["Evaporation_mm"].reindex(calendar)
    ev_filled = ffill_series(ev_series, max_age=7)

    # Pre-process 3-Hourly Tidal amplitude / daily range
    panel_3h["Date"] = panel_3h.Datetime.dt.normalize()
    tidal_agg = (panel_3h.groupby(["Station_ID", "Date"])["WL_mMSL"]
                 .agg(Tidal_Range=lambda x: x.max() - x.min()).reset_index())

    out = []
    for sid in C.MAIN_STEM:
        d = pd.DataFrame({"Date": calendar})
        d.insert(0, "Id", C.STATION_ID[sid])
        d.insert(1, "Station_ID", sid)

        # <<snip:lags>>
        # 1. Water Level and 7-day lags
        s_wl = (panel_wl[panel_wl.Station_ID == sid]
                .set_index("Date")["WL_avg"]
                .reindex(calendar))
        d["WL"] = s_wl.to_numpy()
        for k in range(1, C.N_LAGS + 1):
            d[f"WLD-{k}"] = s_wl.shift(k).to_numpy()  # <<endsnip>>

        # 2. Rainfall and 7-day lags
        rain_gauge = C.STATION_RAIN_GAUGE[sid][0]
        s_rain = (panel_rain[panel_rain.Rain_Gauge == rain_gauge]
                  .drop_duplicates("Date")
                  .set_index("Date")["Rainfall_mm"]
                  .reindex(calendar)
                  .fillna(0.0))  # standard meteorological convention: unrecorded days in active gauge series treated as 0
        d["Rain"] = s_rain.to_numpy()
        for k in range(1, C.N_LAGS + 1):
            d[f"Rain_D{k}"] = s_rain.shift(k).to_numpy()

        d["Rain_sum_3d"] = s_rain.rolling(3, min_periods=1).sum().to_numpy()
        d["Rain_sum_7d"] = s_rain.rolling(7, min_periods=1).sum().to_numpy()
        d["Rainfall_Level"] = C.rain_level_class(s_rain).to_numpy()

        # 3. Discharge
        s_q = (panel_q[panel_q.Station_ID == sid]
               .drop_duplicates("Date")
               .set_index("Date")["Discharge_m3s"]
               .reindex(calendar))
        d["Discharge_m3s"] = s_q.to_numpy()

        # 4. Covariates
        d["Evap_mm"] = ev_filled.to_numpy()

        # Groundwater well
        gw_well = reg.at[sid, "GW_Well"]
        if pd.notna(gw_well):
            s_gw = (panel_gw[panel_gw.Well_ID == gw_well]
                    .drop_duplicates("Date")
                    .set_index("Date")["GW_Depth_m"]
                    .reindex(calendar))
            d["GW_Depth_m"] = ffill_series(s_gw, max_age=14).to_numpy()
        else:
            d["GW_Depth_m"] = np.nan

        # Sediment
        s_sed = (panel_sed[panel_sed.Station_ID == sid]
                 .drop_duplicates("Date")
                 .set_index("Date")["Sed_Conc_ppm"]
                 .reindex(calendar))
        d["Sed_Conc_ppm"] = ffill_series(s_sed, max_age=21).to_numpy()

        # Tidal Range (SW93.5L, SW95)
        s_tid = tidal_agg[tidal_agg.Station_ID == sid].set_index("Date")["Tidal_Range"].reindex(calendar)
        d["Tidal_Range_m"] = s_tid.to_numpy()

        # 5. Metadata, Periods, Risk Class, Completeness
        wl_lag_cols = [f"WLD-{k}" for k in range(1, C.N_LAGS + 1)]
        rain_lag_cols = [f"Rain_D{k}" for k in range(1, C.N_LAGS + 1)]
        d["Period"] = C.period_of(d.Date).to_numpy()
        d["N_Lags_Present"] = d[wl_lag_cols].notna().sum(axis=1)
        d["Window_Complete"] = (d.WL.notna() & (d.N_Lags_Present == C.N_LAGS) & 
                                d.Rain.notna() & (d[rain_lag_cols].notna().sum(axis=1) == C.N_LAGS))
        d["Risk_Class"] = C.risk_class(d.WL, float(reg.at[sid, "Danger_Level_mMSL"]))

        out.append(d)

    df_full = pd.concat(out, ignore_index=True)

    # Save comprehensive hydromet long dataset
    released_hydromet = df_full[df_full.WL.notna()].copy()
    hydromet_cols = ([
        "Id", "Station_ID", "Date",
        "WL", "WLD-1", "WLD-2", "WLD-3", "WLD-4", "WLD-5", "WLD-6", "WLD-7",
        "Rain", "Rain_D1", "Rain_D2", "Rain_D3", "Rain_D4", "Rain_D5", "Rain_D6", "Rain_D7",
        "Rain_sum_3d", "Rain_sum_7d", "Rainfall_Level",
        "Discharge_m3s", "Evap_mm", "GW_Depth_m", "Sed_Conc_ppm", "Tidal_Range_m",
        "Period", "Risk_Class", "N_Lags_Present", "Window_Complete"
    ])
    released_hydromet = released_hydromet[hydromet_cols].sort_values(["Id", "Date"]).reset_index(drop=True)
    released_hydromet.to_csv(C.DATA / "pointwise_hydromet_7day_long.csv", index=False,
                             float_format="%.3f", date_format="%Y-%m-%d")

    # Save compatibility WL-only long dataset
    wl_cols = (["Id", "Station_ID", "Date"] + C.lag_columns()
               + ["Period", "Risk_Class", "N_Lags_Present", "Window_Complete"])
    df_wl_compat = released_hydromet[wl_cols].copy()
    df_wl_compat.to_csv(C.DATA / "pointwise_wl_7day_long.csv", index=False,
                        float_format="%.3f", date_format="%Y-%m-%d")

    comp = (df_full.groupby(["Id", "Station_ID", "Period"], observed=True)
              .agg(N_Calendar_Days=("Date", "size"),
                   N_WL_Present=("WL", "count"),
                   N_Window_Complete=("Window_Complete", "sum"))
              .reset_index())
    comp["Pct_Window_Complete"] = (100 * comp.N_Window_Complete / comp.N_Calendar_Days).round(2)
    comp.to_csv(C.RESULTS / "lag_completeness.csv", index=False)

    print(f"  released hydromet rows: {len(released_hydromet):,}")
    print(f"  of which complete:     {int(released_hydromet.Window_Complete.sum()):,} "
          f"({100*released_hydromet.Window_Complete.mean():.1f}%)")
    print(f"  columns:               {len(hydromet_cols)}")
    print(f"  -> {C.DATA / 'pointwise_hydromet_7day_long.csv'}")
    print(f"  -> {C.DATA / 'pointwise_wl_7day_long.csv'}")


if __name__ == "__main__":
    main()

