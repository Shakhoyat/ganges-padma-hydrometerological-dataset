"""Step 01 - read all 50 BWDB workbooks into tidy interim tables.

The delivery ships 50 .xlsx files across 7 data categories:
  1. Water Level (Daily, 3-Hourly, Annual) - 17 stations
  2. Rainfall (Daily) - 11 stations (CL15, CL19, CL20, CL30, CL205, CL215, CL354, CL402, CL406, CL413, CL414)
  3. Discharge (Mean Daily Discharge, m3/s) - 5 stations (SW46.9L, SW90, SW91.9L, SW93.5L, SW273)
  4. Evaporation (Daily pan evaporation, mm) - CL406 (Faridpur)
  5. Groundwater (Weekly depth, m) - 3 wells (GT2947007, GT8276015, GT8669006)
  6. Sediment (Suspended concentration & transport) - 3 stations (SW90, SW91.9L, SW93.5L)
  7. Cross Section (River morphology bed levels) - 4 cross sections (RMG12, RMP2, RMP3, RMP4)

Outputs
    data/interim/wl_daily_raw.csv
    data/interim/rainfall_daily_raw.csv
    data/interim/discharge_daily_raw.csv
    data/interim/evaporation_daily_raw.csv
    data/interim/groundwater_weekly_raw.csv
    data/interim/sediment_raw.csv
    data/interim/wl_3hourly_raw.csv
    data/interim/cross_section_raw.csv
    results/provenance.csv
    results/source_metadata.csv
"""
from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path

import openpyxl
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
import corridor as C  # noqa: E402

HEADER_FIRST_CELL = {"sl", "sl.", "date", "year"}
DATE_FORMATS = ["%d/%m/%Y", "%d-%m-%Y", "%d-%b-%Y", "%d-%b-%y", "%d/%m/%y"]
FOOTER = "system generated"


def parse_date(raw) -> pd.Timestamp | None:
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return pd.Timestamp(raw).normalize()
    s = str(raw).strip()
    if not s:
        return None
    s = s.split()[0]
    for fmt in DATE_FORMATS:
        try:
            return pd.Timestamp(datetime.strptime(s, fmt)).normalize()
        except ValueError:
            continue
    return None


def parse_datetime(raw) -> pd.Timestamp | None:
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return pd.Timestamp(raw)
    s = str(raw).strip()
    for fmt in ("%d-%b-%Y %I:%M:%S %p", "%d-%b-%Y %H:%M:%S",
                "%d-%m-%Y %I:%M:%S %p", "%d/%m/%Y %I:%M:%S %p"):
        try:
            return pd.Timestamp(datetime.strptime(s, fmt))
        except ValueError:
            continue
    return parse_date(s)


def num(raw) -> float | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.upper() in {"NA", "N/A", "-", "NULL", "NONE"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


# <<snip:read>>
def read_sheet(path: Path) -> list[list]:
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.active
    rows = [list(r) for r in ws.iter_rows(values_only=True)]
    wb.close()
    return rows  # <<endsnip>>


def extract_meta(rows: list[list], header_idx: int) -> dict[str, str]:
    meta: dict[str, str] = {}
    for r in rows[:header_idx]:
        for cell in r:
            if cell is None:
                continue
            s = str(cell).replace("\n", " ").strip()
            if ":" not in s or len(s) > 200:
                continue
            key, _, val = s.partition(":")
            key, val = key.strip(), val.strip()
            if not key or not val or len(key) > 40:
                continue
            m = re.match(r"^(\d{4})\s+To\s+(\d{4})$", val, re.I)
            if key.lower() == "from" and m:
                meta["year_from"], meta["year_to"] = m.group(1), m.group(2)
                continue
            meta[key] = val
    return meta


def find_header(rows: list[list]) -> int | None:
    for i, r in enumerate(rows[:20]):
        if not r:
            continue
        first = str(r[0]).strip().lower() if r[0] is not None else ""
        if first in HEADER_FIRST_CELL:
            return i
    return None


def station_from_filename(name: str) -> str:
    stem = name[: -len(".xlsx")]
    if stem.endswith("2608295209"):
        stem = stem[: -len("2608295209") - 1]
    return stem.rsplit("_", 1)[-1]


def main() -> None:
    files = sorted(C.RAW.glob("*.xlsx"))
    if not files:
        raise SystemExit(f"No .xlsx files found under {C.RAW}")

    print(f"Reading {len(files)} BWDB workbooks from {C.RAW}...")

    wl_daily_recs = []
    rain_recs = []
    disch_recs = []
    evap_recs = []
    gw_recs = []
    sed_recs = []
    wl_3h_recs = []
    xs_recs = []
    ledger = []
    metas = []

    for path in files:
        name = path.name
        rows = read_sheet(path)
        hidx = find_header(rows)
        if hidx is None:
            print(f"  Warning: No header found for {name}")
            continue

        meta = extract_meta(rows, hidx)
        meta["file"] = name
        sid_file = station_from_filename(name)
        sid = meta.get("Station ID", sid_file)
        body = [r for r in rows[hidx + 1:] if r and not (len(r) > 0 and r[0] is not None and FOOTER in str(r[0]))]

        n_read = len(body)
        n_kept = 0

        # 1. Daily Water Level
        if name.startswith("Water_Level_Daily"):
            for r in body:
                d = parse_date(r[1]) if len(r) > 1 else None
                if d is None or d < C.STUDY_START or d > C.STUDY_END:
                    continue
                max_v = num(r[2]) if len(r) > 2 else None
                min_v = num(r[3]) if len(r) > 3 else None
                avg_v = num(r[4]) if len(r) > 4 else None
                if avg_v is not None and sid in C.STATION_ID:
                    wl_daily_recs.append({
                        "Station_ID": sid,
                        "Id": C.STATION_ID[sid],
                        "Date": d,
                        "WL_avg": avg_v,
                        "WL_max": max_v,
                        "WL_min": min_v,
                    })
                    n_kept += 1

        # 2. Rainfall
        elif name.startswith("Rainfall_Daily"):
            gauge_id = sid
            for r in body:
                d = parse_date(r[1]) if len(r) > 1 else None
                if d is None or d < C.STUDY_START or d > C.STUDY_END:
                    continue
                rain_v = num(r[2]) if len(r) > 2 else None
                if rain_v is not None:
                    rain_recs.append({
                        "Rain_Gauge": gauge_id,
                        "Date": d,
                        "Rainfall_mm": rain_v,
                    })
                    n_kept += 1

        # 3. Discharge
        elif name.startswith("Discharge_"):
            for r in body:
                if len(r) < 9:
                    continue
                d = parse_date(r[6])
                if d is None or d < C.STUDY_START or d > C.STUDY_END:
                    continue
                q = num(r[8])
                dsid = str(r[4]).strip() if r[4] is not None else sid_file
                if q is not None:
                    disch_recs.append({
                        "Station_ID": dsid,
                        "Date": d,
                        "Discharge_m3s": q,
                    })
                    n_kept += 1

        # 4. Evaporation
        elif name.startswith("Evaporation_"):
            for r in body:
                if len(r) < 7:
                    continue
                d = parse_date(r[5])
                if d is None or d < C.STUDY_START or d > C.STUDY_END:
                    continue
                ev_v = num(r[6])
                esid = str(r[3]).strip() if r[3] is not None else sid_file
                if ev_v is not None:
                    evap_recs.append({
                        "Evap_Station": esid,
                        "Date": d,
                        "Evaporation_mm": ev_v,
                    })
                    n_kept += 1

        # 5. Groundwater
        elif name.startswith("Ground_Water_"):
            well_id = meta.get("Well ID", sid_file)
            for r in body:
                d = parse_date(r[1]) if len(r) > 1 else None
                if d is None or d < C.STUDY_START or d > C.STUDY_END:
                    continue
                depth_v = num(r[2]) if len(r) > 2 else None
                if depth_v is not None:
                    gw_recs.append({
                        "Well_ID": well_id,
                        "Date": d,
                        "GW_Depth_m": depth_v,
                    })
                    n_kept += 1

        # 6. Sediment
        elif name.startswith("Sediment_"):
            for r in body:
                d = parse_date(r[1]) if len(r) > 1 else None
                if d is None or d < C.STUDY_START or d > C.STUDY_END:
                    continue
                coarse = num(r[2]) if len(r) > 2 else None
                bulk = num(r[3]) if len(r) > 3 else None
                ppm = num(r[4]) if len(r) > 4 else None
                total = num(r[5]) if len(r) > 5 else None
                if any(v is not None for v in (coarse, bulk, ppm, total)):
                    sed_recs.append({
                        "Station_ID": sid,
                        "Date": d,
                        "Sed_Coarse_kgs": coarse,
                        "Sed_Bulk_kgs": bulk,
                        "Sed_Conc_ppm": ppm,
                        "Sed_Total_kgs": total,
                    })
                    n_kept += 1

        # 7. 3-Hourly Water Level
        elif name.startswith("Water_Level_3_Hourly"):
            for r in body:
                dt = parse_datetime(r[2]) if len(r) > 2 else None
                if dt is None:
                    continue
                dtype = str(r[1]).strip().upper() if len(r) > 1 and r[1] is not None else ""
                v = num(r[3]) if len(r) > 3 else None
                if v is not None:
                    wl_3h_recs.append({
                        "Station_ID": sid,
                        "Datetime": dt,
                        "Reading_Type": dtype,
                        "WL_mMSL": v,
                    })
                    n_kept += 1

        # 8. Cross Section
        elif name.startswith("Cross_Section_"):
            for r in body:
                d = parse_date(r[1]) if len(r) > 1 else None
                dist = num(r[2]) if len(r) > 2 else None
                rl = num(r[3]) if len(r) > 3 else None
                if d is not None and dist is not None and rl is not None:
                    xs_recs.append({
                        "Section_ID": sid,
                        "Date": d,
                        "Distance_m": dist,
                        "Reduced_Level_m": rl,
                    })
                    n_kept += 1

        ledger.append({
            "File": name,
            "Station_ID": sid,
            "Category": name.split("_")[0],
            "Rows_Read": n_read,
            "Rows_Kept": n_kept,
        })
        metas.append(meta)

    # Save all datasets
    if wl_daily_recs:
        df_wl = pd.DataFrame(wl_daily_recs).drop_duplicates(["Station_ID", "Date"]).sort_values(["Id", "Date"])
        df_wl.to_csv(C.INTERIM / "wl_daily_raw.csv", index=False)
        print(f"  Saved wl_daily_raw.csv: {len(df_wl):,} rows across {df_wl.Station_ID.nunique()} stations")

    if rain_recs:
        df_rain = pd.DataFrame(rain_recs).drop_duplicates(["Rain_Gauge", "Date"]).sort_values(["Rain_Gauge", "Date"])
        df_rain.to_csv(C.INTERIM / "rainfall_daily_raw.csv", index=False)
        print(f"  Saved rainfall_daily_raw.csv: {len(df_rain):,} rows across {df_rain.Rain_Gauge.nunique()} gauges")

    if disch_recs:
        df_q = pd.DataFrame(disch_recs).drop_duplicates(["Station_ID", "Date"]).sort_values(["Station_ID", "Date"])
        df_q.to_csv(C.INTERIM / "discharge_daily_raw.csv", index=False)
        print(f"  Saved discharge_daily_raw.csv: {len(df_q):,} rows across {df_q.Station_ID.nunique()} stations")

    if evap_recs:
        df_ev = pd.DataFrame(evap_recs).drop_duplicates(["Evap_Station", "Date"]).sort_values(["Evap_Station", "Date"])
        df_ev.to_csv(C.INTERIM / "evaporation_daily_raw.csv", index=False)
        print(f"  Saved evaporation_daily_raw.csv: {len(df_ev):,} rows")

    if gw_recs:
        df_gw = pd.DataFrame(gw_recs).drop_duplicates(["Well_ID", "Date"]).sort_values(["Well_ID", "Date"])
        df_gw.to_csv(C.INTERIM / "groundwater_weekly_raw.csv", index=False)
        print(f"  Saved groundwater_weekly_raw.csv: {len(df_gw):,} rows across {df_gw.Well_ID.nunique()} wells")

    if sed_recs:
        df_sed = pd.DataFrame(sed_recs).drop_duplicates(["Station_ID", "Date"]).sort_values(["Station_ID", "Date"])
        df_sed.to_csv(C.INTERIM / "sediment_raw.csv", index=False)
        print(f"  Saved sediment_raw.csv: {len(df_sed):,} rows across {df_sed.Station_ID.nunique()} stations")

    if wl_3h_recs:
        df_3h = pd.DataFrame(wl_3h_recs).sort_values(["Station_ID", "Datetime"])
        df_3h.to_csv(C.INTERIM / "wl_3hourly_raw.csv", index=False)
        print(f"  Saved wl_3hourly_raw.csv: {len(df_3h):,} sub-daily readings")

    if xs_recs:
        df_xs = pd.DataFrame(xs_recs).sort_values(["Section_ID", "Date", "Distance_m"])
        df_xs.to_csv(C.INTERIM / "cross_section_raw.csv", index=False)
        print(f"  Saved cross_section_raw.csv: {len(df_xs):,} bathymetry points")

    pd.DataFrame(ledger).to_csv(C.RESULTS / "provenance.csv", index=False)
    pd.DataFrame(metas).to_csv(C.RESULTS / "source_metadata.csv", index=False)
    print("Done Step 01.")



if __name__ == "__main__":
    main()
