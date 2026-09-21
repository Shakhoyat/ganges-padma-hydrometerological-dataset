"""Step 02 - one row per station: id, name, position, datum reference, covariates, role.

Outputs
    data/station_registry.csv
    results/coverage_by_period.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
import corridor as C  # noqa: E402


def main() -> None:
    panel = pd.read_csv(C.INTERIM / "wl_daily_raw.csv", parse_dates=["Date"])
    meta = pd.read_csv(C.RESULTS / "source_metadata.csv")
    meta_wl = meta[meta["Station ID"].isin(C.MAIN_STEM)].drop_duplicates("Station ID").set_index("Station ID")

    if not C.REGISTRY_SRC.exists():
        raise SystemExit(f"corridor register not found at {C.REGISTRY_SRC}")
    geo = pd.read_csv(C.REGISTRY_SRC).set_index("station_id")

    full = pd.date_range(C.STUDY_START, C.STUDY_END, freq="D")
    wide = panel.pivot(index="Date", columns="Station_ID", values="WL_avg").reindex(full)

    rows, cov_rows = [], []
    for sid in C.MAIN_STEM:
        s = wide[sid]
        anc = C.ANCESTORS[sid]
        preds = [p for p in C.MAIN_STEM[: C.STATION_ID[sid] - 1]
                 if p not in C.EXCLUDED_PREDICTORS]
        
        rain_gauge, rain_dist = C.STATION_RAIN_GAUGE.get(sid, (pd.NA, pd.NA))
        gw_well = geo.at[sid, "gw_well"] if sid in geo.index and "gw_well" in geo.columns else pd.NA
        gw_km = geo.at[sid, "gw_well_km"] if sid in geo.index and "gw_well_km" in geo.columns else pd.NA

        rows.append({
            "Id": C.STATION_ID[sid],
            "Station_ID": sid,
            "Station_Name": meta_wl.at[sid, "Station Name"] if sid in meta_wl.index and pd.notna(meta_wl.at[sid, "Station Name"]) else geo.at[sid, "name"],
            "River": meta_wl.at[sid, "River Name"] if sid in meta_wl.index and pd.notna(meta_wl.at[sid, "River Name"]) else geo.at[sid, "river"],
            "District": meta_wl.at[sid, "District"] if sid in meta_wl.index and pd.notna(meta_wl.at[sid, "District"]) else geo.at[sid, "district"],
            "Upazila": meta_wl.at[sid, "Upazila"] if sid in meta_wl.index and pd.notna(meta_wl.at[sid, "Upazila"]) else geo.at[sid, "upazila"],
            "Station_Type": meta_wl.at[sid, "Station Type"] if sid in meta_wl.index and pd.notna(meta_wl.at[sid, "Station Type"]) else geo.at[sid, "station_type"],
            "Latitude": float(meta_wl.at[sid, "Latitude"]) if sid in meta_wl.index and pd.notna(meta_wl.at[sid, "Latitude"]) else geo.at[sid, "lat"],
            "Longitude": float(meta_wl.at[sid, "Longitude"]) if sid in meta_wl.index and pd.notna(meta_wl.at[sid, "Longitude"]) else geo.at[sid, "lon"],
            "Chainage_km": geo.at[sid, "chainage_km"] if sid in geo.index else pd.NA,
            "Reach": geo.at[sid, "reach"] if sid in geo.index else pd.NA,
            "Danger_Level_mMSL": geo.at[sid, "danger_level"] if sid in geo.index else pd.NA,
            "Danger_Level_Source": geo.at[sid, "danger_level_source"] if sid in geo.index else pd.NA,
            "Rain_Gauge": rain_gauge,
            "Rain_Gauge_Distance_km": rain_dist,
            "GW_Well": gw_well,
            "GW_Well_Distance_km": gw_km,
            "Has_Discharge": sid in C.DISCHARGE_STATIONS,
            "Has_Sediment": sid in ["SW90", "SW91.9L", "SW93.5L"],
            "Has_3Hourly_Tidal": sid in ["SW93.5L", "SW95"],
            "N_Days_Observed": int(s.notna().sum()),
            "N_Days_Missing": int(s.isna().sum()),
            "Coverage_pct": round(100.0 * s.notna().mean(), 2),
            "N_True_Ancestors": len(anc),
            "N_Prior_Ids": C.STATION_ID[sid] - 1,
            "Ids_Are_All_Upstream": len(anc) == C.STATION_ID[sid] - 1,
            "N_Predictor_Stations": len(preds),
            "Is_Target": sid in C.TARGETS,
            "Role": C.TARGET_ROLE.get(sid, ""),
        })

        for name, a, b in C.PERIODS:
            w = s.loc[a:b]
            cov_rows.append({
                "Id": C.STATION_ID[sid], "Station_ID": sid, "Period": name,
                "N_Days": len(w), "N_Observed": int(w.notna().sum()),
                "Coverage_pct": round(100.0 * w.notna().mean(), 2),
            })

    reg = pd.DataFrame(rows).sort_values("Id")
    reg.to_csv(C.DATA / "station_registry.csv", index=False)
    pd.DataFrame(cov_rows).to_csv(C.RESULTS / "coverage_by_period.csv", index=False)

    print(f"{'id':>3} {'station':<9} {'name':<20} {'river':<18} {'km':>7} {'DL':>6} "
          f"{'rain':<6} {'disch':<5} {'cov%':>6} {'pred':>5}")
    for _, r in reg.iterrows():
        km = f"{r.Chainage_km:.1f}" if pd.notna(r.Chainage_km) else "-"
        dl = f"{r.Danger_Level_mMSL:.2f}" if pd.notna(r.Danger_Level_mMSL) else "-"
        rain = str(r.Rain_Gauge) if pd.notna(r.Rain_Gauge) else "-"
        disch = "Yes" if r.Has_Discharge else "No"
        flag = " <- TARGET" if r.Is_Target else ""
        print(f"{r.Id:>3} {r.Station_ID:<9} {r.Station_Name:<20} {r.River:<18} "
              f"{km:>7} {dl:>6} {rain:<6} {disch:<5} {r.Coverage_pct:>6.1f} "
              f"{r.N_Predictor_Stations:>5}{flag}")

    odd = reg[~reg.Ids_Are_All_Upstream]
    print(f"\n  stations whose lower ids are NOT all true ancestors: "
          f"{', '.join(odd.Station_ID)} (branch heads)")
    print(f"  -> {C.DATA / 'station_registry.csv'}")


if __name__ == "__main__":
    main()
