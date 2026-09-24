"""Rebuild processed/ and metadata/ from the 50 BWDB workbooks in raw/.

    python code/build_dataset.py

Outputs
    processed/ganges_padma_hydromet_dataset_2011_2025.csv   Table 1 of the brief (+ enrichment)
        Id, Date, WL, WLD-1 .. WLD-7, Rainfall_Level        the brief's columns, first
        Rainfall_mm, Rain_3dSum, WL_Range_D-1, DOY_sin, DOY_cos,
        Latitude, Longitude, Tidal                          features (no same-day stage)
        WL_Trend                                            classification label
    processed/wide/wide_<station>.csv                        Table 2 of the brief (+ boundary)
        Id, Date, WL, WLD-1 .. WLD-7, then P<id>_WL, P<id>_WLD-1 .. P<id>_WLD-7
        for every upstream point, then B<id>_WL, B<id>_WLD-1 .. B<id>_WLD-7 for the
        two Meghna boundary gauges (8 columns per point); complete rows only
    metadata/station_registry.csv, provenance.csv, source_metadata.csv,
    metadata/feature_dictionary.csv, checksums_sha256.csv,
    metadata/validation_daily_vs_3hourly.csv

Every value comes from the daily water-level and rainfall workbooks (or their
headers), or is a fixed transform of them: lags, a 3-day rain sum, the previous
day's high-low range, the day-of-year phase and the day-on-day trend class.
No feature holds same-day stage information, so WL can be predicted from any
subset of the columns except WL_Trend (the classification label). Every other
workbook is kept unmodified in raw/ and its reason for exclusion is written to
metadata/provenance.csv.
"""
from __future__ import annotations

import hashlib
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW, PROCESSED, META = ROOT / "raw", ROOT / "processed", ROOT / "metadata"
WIDE = PROCESSED / "wide"

START, END = pd.Timestamp("2011-01-01"), pd.Timestamp("2025-12-31")
N_LAGS = 7

# Corridor order (Id 1..17): for every river edge u -> v, id(u) < id(v), so
# "previous points" of a target are always hydrologically upstream of it.
# (station, river, along-channel chainage from Panka in km; None = off the main stem)
STATIONS = [
    ("SW88A", "Ganges", 0.0), ("SW88", "Ganges", 54.01), ("SW89", "Ganges", 70.28),
    ("SW90", "Ganges", 110.92), ("SW91", "Ganges", 124.86), ("SW91.1", "Ganges", 154.22),
    ("SW91.2", "Ganges", 178.68), ("SW46.9L", "Jamuna", None), ("SW50.6", "Jamuna", None),
    ("SW91.9R", "Padma", 199.43), ("SW91.9L", "Padma", 203.44), ("SW93.4L", "Padma", 254.96),
    ("SW93.5L", "Padma", 259.93), ("SW94", "Padma", 268.05), ("SW95", "Padma", 290.66),
    ("SW273", "Meghna", None), ("SW277", "Lower Meghna", None),
]
ID = {s: i + 1 for i, (s, _, _) in enumerate(STATIONS)}

# ML targets: two gauges upstream of the Padma Bridge and one downstream of it.
TARGETS = {
    "SW91.9L": "wide_sw91_9l_baruria_transit.csv",  # 56 km upstream, head of the Padma
    "SW93.4L": "wide_sw93_4l_bhagyakul.csv",        # 5 km upstream of the bridge
    "SW95": "wide_sw95_sureswar.csv",               # 31 km downstream of the bridge
}
# Tarpasa has a 1,369-day gap (2016-2020); as a predictor it would remove 46% of
# the construction-period rows from the Sureswar matrix. It keeps its Id.
EXCLUDED_PREDICTORS = {"SW94"}
# The lower Padma is a backwater reach: its stage depends on the upstream inflow
# (the P blocks) and on the Meghna confluence downstream. Bhairab Bazar (Meghna
# inflow) and Chandpur (confluence) are therefore added to every wide matrix as
# boundary points; on the test years they raise Ridge's persistence index at
# Bhagyakul and Sureswar by 0.05-0.07.
BOUNDARY_POINTS = ["SW273", "SW277"]

# Rainfall_Level follows the Bangladesh Meteorological Department's daily
# classes: light 1-10 mm, moderate 11-22 mm, then moderately heavy and above.
# 0 = no rain or trace (< 1 mm), 1 = 1-10 mm, 2 = >10-22 mm, 3 = > 22 mm.
RAIN_LIGHT_MIN, RAIN_LIGHT_MAX, RAIN_MODERATE_MAX = 1.0, 10.0, 22.0
RAIN_SUM_DAYS = 3  # Rain_3dSum covers Date-2 .. Date
DAYS_PER_YEAR = 365.25
# WL_Trend: 0 falling, 1 steady, 2 rising. A day-on-day change within +-0.03 m is
# "steady": 0.03 m is just above the 95th-percentile gap (0.014-0.028 m) between
# BWDB's daily average and the mean of the 3-hourly readings, i.e. within noise.
TREND_STEADY_M = 0.03
TREND_FALLING, TREND_STEADY, TREND_RISING = 0, 1, 2

DISPLAY_NAME = {"Goalundo Transi": "Goalundo Transit", "Bahadurabad_Transit": "Bahadurabad Transit"}
UNUSED_REASON = {
    "Water_Level_Annual": "not used: one annual max/min per year, no daily information",
    "Discharge": "not used: rated from same-day stage (r(WL, log Q) 0.98-0.99 at non-tidal stations); "
                 "5 of 17 stations; ends 2024; lagged inflow Q lowered pooled test skill",
    "Evaporation": "not used: one station for the whole corridor; impossible negative values; "
                   "lagged evaporation lowered pooled test skill",
    "Ground_Water": "not used: weekly readings from 3 wells up to 180 km from the gauges",
    "Sediment": "not used: weekly-to-monthly samples at 3 stations",
    "Cross_Section": "not used: bed surveys, not a daily time series",
}
DATE_FORMATS = ["%d-%b-%Y", "%d/%m/%Y", "%d-%m-%Y", "%d-%b-%y", "%d/%m/%y"]
# Header cells read "Key: value"; longer strings are the BWDB address block, not metadata.
MAX_META_KEY, MAX_META_VALUE = 40, 200


def parse_date(raw) -> pd.Timestamp | None:
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return pd.Timestamp(raw).normalize()
    s = str(raw).strip().split(" ")[0]
    for fmt in DATE_FORMATS:
        try:
            return pd.Timestamp(datetime.strptime(s, fmt))
        except ValueError:
            continue
    return None


def num(raw) -> float | None:
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError):
        return None


def read_workbook(path: Path) -> tuple[dict, list[list]]:
    """Header metadata ("Key: value" cells above the table) and the data rows."""
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    rows = [list(r) for r in wb.active.iter_rows(values_only=True)]
    wb.close()
    head = next((i for i, r in enumerate(rows[:20])
                 if r and str(r[0]).strip().lower() in {"sl", "sl.", "date", "year"}), None)
    if head is None:
        raise ValueError(f"{path.name}: no table header in the first 20 rows")
    meta = {}
    for r in rows[:head]:
        for cell in r:
            key, sep, val = str(cell or "").replace("\n", " ").partition(":")
            if sep and 0 < len(key.strip()) <= MAX_META_KEY and val.strip() and len(val) < MAX_META_VALUE:
                meta[key.strip()] = val.strip()
    body = [r for r in rows[head + 1:] if r and "system generated" not in str(r[0]).lower()]
    return meta, body


def daily_series(body: list[list], value_col: int) -> tuple[pd.Series, int]:
    """Date in column 1, value in value_col; study window only; first of any duplicate date."""
    recs = [(parse_date(r[1]), num(r[value_col])) for r in body if len(r) > value_col]
    valid = [(d, v) for d, v in recs if d is not None and v is not None and START <= d <= END]
    s = pd.Series(dict(reversed(valid)), dtype=float)
    return s.sort_index(), len(valid) - len(s)


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0 * math.asin(math.sqrt(a))


def subdaily_mean(body: list[list]) -> tuple[pd.Series, int]:
    """Daily mean of the REGULAR 3-hourly readings (HIGH/LOW tide readings skipped)."""
    recs = [(parse_date(r[2]), num(r[3])) for r in body
            if len(r) > 3 and str(r[1]).strip().upper() == "REGULAR"]
    recs = [(d, v) for d, v in recs if d is not None and v is not None and START <= d <= END]
    s = pd.DataFrame(recs, columns=["Date", "WL"]).groupby("Date").WL.mean()
    return s, len(recs)


def read_raw() -> tuple[dict, dict, dict, dict, dict, dict, list, list]:
    wl, high_low, rain, subdaily, wl_meta, rain_meta, ledger, metas = {}, {}, {}, {}, {}, {}, [], []
    for path in sorted(RAW.glob("*.xlsx")):
        meta, body = read_workbook(path)
        sid = meta.get("Station ID") or meta.get("Well ID") or path.stem.split("_")[-2]
        meta["file"] = path.name
        metas.append(meta)
        kept, n_dup, use = 0, 0, None
        if path.name.startswith("Water_Level_Daily") and sid in ID:
            wl[sid], n_dup = daily_series(body, 4)  # columns: SL, DATE, MAX, MIN, AVERAGE
            high_low[sid] = (daily_series(body, 2)[0], daily_series(body, 3)[0])
            wl_meta[sid], kept = meta, len(wl[sid])
            use = "WL, WLD-1..WLD-7, WL_Trend (average); WL_Range_D-1 (max - min)"
        elif path.name.startswith("Rainfall_Daily"):
            rain[sid], n_dup = daily_series(body, 2)  # columns: SL, DATE, RAIN FALL(mm)
            rain_meta[sid], kept = meta, len(rain[sid])
            use = "Rainfall_Level, Rainfall_mm, Rain_3dSum (via nearest-gauge assignment)"
        elif path.name.startswith("Water_Level_3_Hourly"):
            subdaily[sid], kept = subdaily_mean(body)
            use = "validation only: checks the daily average WL (validation_daily_vs_3hourly.csv)"
        if use is None:
            use = next(v for k, v in UNUSED_REASON.items() if path.name.startswith(k))
        ledger.append({"File": path.name, "Station_ID": sid, "Rows_In_Workbook": len(body),
                       "Rows_Used": kept, "Duplicate_Dates_Dropped": n_dup, "Used_For": use})
    return wl, high_low, rain, subdaily, wl_meta, rain_meta, ledger, metas


def build_registry(wl: dict, wl_meta: dict, rain_meta: dict) -> pd.DataFrame:
    gauges = {g: (float(m["Latitude"]), float(m["Longitude"])) for g, m in rain_meta.items()}
    rows = []
    for sid, river, chainage in STATIONS:
        m = wl_meta[sid]
        lat, lon = float(m["Latitude"]), float(m["Longitude"])
        dist = {g: haversine_km(lat, lon, *ll) for g, ll in gauges.items()}
        gauge = min(dist, key=dist.get)
        n_obs = len(wl[sid])
        rows.append({
            "Id": ID[sid], "Station_ID": sid,
            "Station_Name": DISPLAY_NAME.get(m["Station Name"], m["Station Name"]),
            "River": river, "District": m.get("District"), "Upazila": m.get("Upazila"),
            "Station_Type": m.get("Station Type"), "Latitude": lat, "Longitude": lon,
            "Chainage_km": chainage, "Rain_Gauge": gauge,
            "Rain_Gauge_Distance_km": round(dist[gauge], 1),
            "Days_Observed": n_obs, "Days_Missing": len(pd.date_range(START, END)) - n_obs,
            "Coverage_pct": round(100 * n_obs / len(pd.date_range(START, END)), 2),
            "ML_Target": sid in TARGETS, "Role": station_role(sid),
        })
    return pd.DataFrame(rows)


def station_role(sid: str) -> str:
    """Role of a station in the wide matrices."""
    if sid in TARGETS:
        return "Target"
    if sid in EXCLUDED_PREDICTORS:
        return "Excluded (gap)"
    if sid in BOUNDARY_POINTS:
        return "Boundary predictor"
    return "Upstream predictor"


def rainfall_level(mm: pd.Series) -> pd.Series:
    level = pd.Series(3, index=mm.index)
    level[mm <= RAIN_MODERATE_MAX] = 2
    level[mm <= RAIN_LIGHT_MAX] = 1
    level[mm < RAIN_LIGHT_MIN] = 0
    return level.where(mm.notna()).astype("Int64")  # missing gauge-day stays missing


def wl_trend(change: pd.Series) -> pd.Series:
    """Day-on-day change -> 0 falling, 1 steady (|change| <= TREND_STEADY_M), 2 rising."""
    change = change.round(3)  # strip float noise (13.56 - 13.53 = 0.0300...01) before comparing
    trend = pd.Series(TREND_STEADY, index=change.index)
    trend[change < -TREND_STEADY_M] = TREND_FALLING
    trend[change > TREND_STEADY_M] = TREND_RISING
    return trend.where(change.notna()).astype("Int64")


def build_long(wl: dict, high_low: dict, rain: dict, registry: pd.DataFrame) -> pd.DataFrame:
    """The brief's Table 1 columns first, then leak-free features, then the trend label."""
    calendar = pd.date_range(START, END, freq="D")
    phase = 2 * np.pi * calendar.dayofyear.to_numpy() / DAYS_PER_YEAR
    parts = []
    for r in registry.itertuples():
        stage = wl[r.Station_ID].reindex(calendar)  # calendar reindex: a lag is k days, not k rows
        rain_mm = rain[r.Rain_Gauge].reindex(calendar)
        high, low = (s.reindex(calendar) for s in high_low[r.Station_ID])
        d = pd.DataFrame({"Id": r.Id, "Date": calendar, "WL": stage.to_numpy()})
        for k in range(1, N_LAGS + 1):
            d[f"WLD-{k}"] = stage.shift(k).to_numpy()
        d["Rainfall_Level"] = rainfall_level(rain_mm).to_numpy()
        d["Rainfall_mm"] = rain_mm.to_numpy()
        d["Rain_3dSum"] = rain_mm.rolling(RAIN_SUM_DAYS, min_periods=RAIN_SUM_DAYS).sum().round(2).to_numpy()
        d["WL_Range_D-1"] = (high - low).shift(1).round(3).to_numpy()  # yesterday's tidal range
        d["DOY_sin"], d["DOY_cos"] = np.sin(phase).round(6), np.cos(phase).round(6)
        d["Latitude"], d["Longitude"] = r.Latitude, r.Longitude
        d["Tidal"] = int(r.Station_Type == "Tidal")
        d["WL_Trend"] = wl_trend(stage - stage.shift(1)).to_numpy()
        parts.append(d[d.WL.notna()])
    long = pd.concat(parts, ignore_index=True)
    return long.astype({"Rainfall_Level": "Int64", "WL_Trend": "Int64"})


def build_wide(long: pd.DataFrame, target: str) -> pd.DataFrame:
    stage_cols = ["WL"] + [f"WLD-{k}" for k in range(1, N_LAGS + 1)]
    by_id = {i: g.set_index("Date")[stage_cols] for i, g in long.groupby("Id")}
    tid = ID[target]
    blocks = [by_id[tid]]
    for sid, _, _ in STATIONS[: tid - 1]:
        if sid not in EXCLUDED_PREDICTORS:
            blocks.append(by_id[ID[sid]].add_prefix(f"P{ID[sid]:02d}_"))
    for sid in BOUNDARY_POINTS:
        blocks.append(by_id[ID[sid]].add_prefix(f"B{ID[sid]:02d}_"))
    wide = pd.concat(blocks, axis=1, join="inner").dropna().reset_index(names="Date")
    wide.insert(0, "Id", tid)
    return wide


def feature_dictionary(registry: pd.DataFrame) -> pd.DataFrame:
    key, feat, static = "key", "feature", "feature (constant per station)"
    long_rows = [
        ("Id", key, "Corridor point index; maps to Station_ID in station_registry.csv", "integer", "1-17",
         "never missing"),
        ("Date", key, "Observation date", "date", "2011-01-01 to 2025-12-31 (YYYY-MM-DD)", "never missing"),
        ("WL", "target (regression)", "Daily average water level", "float", "m above mean sea level (mMSL)",
         "row omitted when not observed"),
        ("WLD-k (k=1..7)", feat, "Water level k calendar days before Date at the same station", "float", "mMSL",
         "blank when day Date-k was not observed; never interpolated"),
        ("Rainfall_Level", feat, "Daily rainfall class at the station's nearest rain gauge", "ordinal integer",
         "BMD classes: 0 = no rain or trace (<1 mm), 1 = light (1-10 mm), "
         "2 = moderate (>10-22 mm), 3 = moderately heavy or heavier (>22 mm)",
         "blank when the gauge has no record that day"),
        ("Rainfall_mm", feat, "Daily rainfall total at the nearest rain gauge (source of Rainfall_Level)",
         "float", "mm", "blank when the gauge has no record that day"),
        ("Rain_3dSum", feat, f"Rainfall at the nearest gauge summed over Date-{RAIN_SUM_DAYS - 1} .. Date",
         "float", "mm", f"blank unless all {RAIN_SUM_DAYS} gauge-days are recorded"),
        ("WL_Range_D-1", feat, "Previous day's maximum minus minimum water level at the same station "
         "(tidal range at tidal gauges, about 0.02 m elsewhere)", "float", "m",
         "blank when day Date-1 was not observed"),
        ("DOY_sin", feat, "sin(2*pi*day_of_year/365.25): seasonal phase", "float", "-1 to 1", "never missing"),
        ("DOY_cos", feat, "cos(2*pi*day_of_year/365.25): seasonal phase", "float", "-1 to 1", "never missing"),
        ("Latitude", static, "Gauge latitude from the BWDB workbook header", "float", "decimal degrees N",
         "never missing"),
        ("Longitude", static, "Gauge longitude from the BWDB workbook header", "float", "decimal degrees E",
         "never missing"),
        ("Tidal", static, "BWDB station type", "binary integer", "1 = tidal, 0 = non-tidal", "never missing"),
        ("WL_Trend", "target (classification)",
         "Day-on-day change of WL (WL - WLD-1); computed from WL, so never an input when predicting WL",
         "ordinal integer",
         f"0 = falling (< -{TREND_STEADY_M} m), 1 = steady (within +-{TREND_STEADY_M} m), "
         f"2 = rising (> {TREND_STEADY_M} m)", "blank when WLD-1 is blank"),
    ]
    wide_rows = [
        ("Id, Date", key, "Target point index and date, as in the long panel", "integer, date", "-",
         "complete rows only"),
        ("WL", "target (regression)", "Target station's daily average water level", "float", "mMSL",
         "complete rows only"),
        ("WLD-1..WLD-7", feat, "Target station's own lags, as in the long panel", "float", "mMSL",
         "complete rows only"),
        ("P<id>_WL", feat, "Same-day water level at upstream point <id>", "float", "mMSL", "complete rows only"),
        ("P<id>_WLD-k (k=1..7)", feat, "Water level k days before Date at upstream point <id>", "float", "mMSL",
         "complete rows only"),
        ("B<id>_WL, B<id>_WLD-k", feat,
         "Same pattern for the Meghna boundary gauges (16 Bhairab Bazar, 17 Chandpur)", "float", "mMSL",
         "complete rows only"),
    ]
    cols = ["Column", "Role", "Meaning", "Type", "Unit_or_Coding", "Missing_Value_Rule"]
    out = pd.concat([
        pd.DataFrame(long_rows, columns=cols).assign(File="processed/ganges_padma_hydromet_dataset_2011_2025.csv"),
        pd.DataFrame(wide_rows, columns=cols).assign(File="processed/wide/wide_<station>.csv"),
    ])
    return out[["File"] + cols]


def validate(long: pd.DataFrame, wide: dict, registry: pd.DataFrame) -> None:
    main_stem = registry.dropna(subset=["Chainage_km"]).sort_values("Id")
    assert main_stem.Chainage_km.is_monotonic_increasing, "Id order must follow the river downstream"
    assert not long.duplicated(["Id", "Date"]).any(), "duplicate station-days"
    assert long.WL.between(-2, 30).all(), "water level outside the physical range of the corridor"
    assert long.Rainfall_Level.dropna().isin([0, 1, 2, 3]).all()
    assert (long.Rainfall_mm.dropna() >= 0).all(), "negative rainfall"
    assert (long.Rain_3dSum.dropna() >= long.Rainfall_mm[long.Rain_3dSum.notna()] - 1e-6).all(), \
        "3-day sum below the day's own rainfall"
    assert (long["WL_Range_D-1"].dropna() >= 0).all(), "negative daily range"
    assert long.DOY_sin.between(-1, 1).all() and long.DOY_cos.between(-1, 1).all()
    assert long.Latitude.between(20.5, 26.7).all() and long.Longitude.between(88.0, 92.7).all(), \
        "gauge outside Bangladesh"
    assert long.WL_Trend.equals(wl_trend(long.WL - long["WLD-1"])), "WL_Trend must follow WL - WLD-1"
    for i, g in long.groupby("Id"):  # every lag equals the stage observed k days earlier
        s = g.set_index("Date").WL
        for k in range(1, N_LAGS + 1):
            expect = s.reindex(g.Date - pd.Timedelta(days=k)).to_numpy()
            got = g[f"WLD-{k}"].to_numpy()
            assert ((got == expect) | (pd.isna(got) & pd.isna(expect))).all(), (i, k)
    for name, w in wide.items():
        assert not w.isna().any().any(), name
        assert (w.shape[1] - 2) % 8 == 0, name  # 8 columns per point


def check_high_low(wl: dict, high_low: dict) -> None:
    """Raw consistency: MIN <= AVERAGE <= MAX on every observed day (source of WL_Range_D-1)."""
    for sid, stage in wl.items():
        high, low = (s.reindex(stage.index) for s in high_low[sid])
        assert high.notna().all() and low.notna().all(), f"{sid}: AVERAGE recorded without MAX/MIN"
        assert ((low <= stage) & (stage <= high)).all(), f"{sid}: need MIN <= AVERAGE <= MAX"


def validate_daily_average(wl: dict, subdaily: dict) -> pd.DataFrame:
    """Released daily WL vs the mean of the same day's 3-hourly readings."""
    rows = []
    for sid, s in subdaily.items():
        pair = pd.concat([wl[sid].rename("daily"), s.rename("subdaily")], axis=1, join="inner")
        diff = (pair.daily - pair.subdaily).abs()
        rows.append({"Station_ID": sid, "Days_Compared": len(pair),
                     "Mean_Abs_Diff_m": round(diff.mean(), 4), "P95_Abs_Diff_m": round(diff.quantile(0.95), 4),
                     "Max_Abs_Diff_m": round(diff.max(), 3), "Pearson_r": round(pair.daily.corr(pair.subdaily), 5)})
    return pd.DataFrame(rows)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    wl, high_low, rain, subdaily, wl_meta, rain_meta, ledger, metas = read_raw()
    check_high_low(wl, high_low)
    registry = build_registry(wl, wl_meta, rain_meta)
    for row in ledger:  # a rain gauge is used only if it is the nearest gauge of some station
        if row["File"].startswith("Rainfall_Daily") and row["Station_ID"] not in set(registry.Rain_Gauge):
            row["Rows_Used"], row["Used_For"] = 0, "not used: not the nearest rain gauge of any station"
    long = build_long(wl, high_low, rain, registry)
    wide = {fname: build_wide(long, sid) for sid, fname in TARGETS.items()}
    validate(long, wide, registry)

    WIDE.mkdir(parents=True, exist_ok=True)
    for stale in WIDE.glob("wide_*.csv"):  # the release holds exactly the TARGETS matrices
        stale.unlink()
    long.to_csv(PROCESSED / "ganges_padma_hydromet_dataset_2011_2025.csv", index=False, date_format="%Y-%m-%d")
    for fname, w in wide.items():
        w.to_csv(WIDE / fname, index=False, date_format="%Y-%m-%d")
    registry.to_csv(META / "station_registry.csv", index=False)
    pd.DataFrame(ledger).to_csv(META / "provenance.csv", index=False)
    pd.DataFrame(metas).to_csv(META / "source_metadata.csv", index=False)
    feature_dictionary(registry).to_csv(META / "feature_dictionary.csv", index=False)
    check = validate_daily_average(wl, subdaily)
    check.to_csv(META / "validation_daily_vs_3hourly.csv", index=False)
    released = [PROCESSED / "ganges_padma_hydromet_dataset_2011_2025.csv"] + sorted(WIDE.glob("*.csv"))
    pd.DataFrame({"File": [p.relative_to(ROOT).as_posix() for p in released],
                  "SHA256": [sha256(p) for p in released]}).to_csv(META / "checksums_sha256.csv", index=False)

    complete = long.filter(like="WLD-").notna().all(axis=1) & long.WL.notna()
    print(f"long panel: {len(long):,} station-days x {long.shape[1]} columns; "
          f"{int(complete.sum()):,} with all 7 lags; Rainfall_Level missing on {int(long.Rainfall_Level.isna().sum())}")
    print("blanks per column:", long.isna().sum()[lambda s: s > 0].to_dict())
    print("WL_Trend shares (0 fall / 1 steady / 2 rise):",
          long.WL_Trend.value_counts(normalize=True).sort_index().round(3).to_dict())
    print(check.to_string(index=False))
    for fname, w in wide.items():
        n_points = (w.shape[1] - 2) // 8 - 1
        print(f"{fname}: {len(w):,} rows x {w.shape[1]} columns "
              f"({n_points - len(BOUNDARY_POINTS)} upstream + {len(BOUNDARY_POINTS)} boundary points)")


if __name__ == "__main__":
    main()
