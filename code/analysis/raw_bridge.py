"""Parsers for the raw BWDB workbooks that the daily release does not tabulate but
that bear directly on the Padma Bridge reach: discharge, suspended sediment and
river cross-section surveys. All files are read unmodified from raw/.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

import common as cm

RAW = cm.DATASET / "raw"


def _header_meta(x: pd.DataFrame) -> dict[str, str]:
    meta = {}
    for v in x.iloc[:12].to_numpy().ravel():
        if isinstance(v, str) and ":" in v:
            k, _, val = v.partition(":")
            meta[k.strip()] = val.strip()
    return meta


def _table(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    x = pd.read_excel(path, header=None)
    hdr = next(i for i in range(len(x)) if str(x.iat[i, 0]).strip() == "SL")
    cols = [str(c).strip() for c in x.iloc[hdr]]
    body = x.iloc[hdr + 1:].copy()
    body.columns = cols
    body = body[pd.to_numeric(body["SL"], errors="coerce").notna()]
    return body.loc[:, [c for c in cols if c != "nan"]], _header_meta(x)


def discharge(station: str) -> pd.DataFrame:
    path = next(RAW.glob(f"Discharge_*_{station}_*.xlsx"))
    t, _ = _table(path)
    out = pd.DataFrame({"date": pd.to_datetime(t["DATETIME"], format="%d-%m-%Y", errors="coerce"),
                        "wl": pd.to_numeric(t["WATER LEVEL(m)"], errors="coerce"),
                        "q": pd.to_numeric(t["MDD(m)3/s"], errors="coerce")})
    return out.dropna(subset=["date"]).drop_duplicates("date").sort_values("date").reset_index(drop=True)


def sediment(station: str) -> pd.DataFrame:
    path = next(RAW.glob(f"Sediment_*_{station}_*.xlsx"))
    t, _ = _table(path)
    cols = {c: c for c in t.columns}
    get = lambda key: next(c for c in cols if key.lower() in c.lower())  # noqa: E731
    out = pd.DataFrame({"date": pd.to_datetime(t["DATE"], format="%d-%b-%y", errors="coerce"),
                        "coarse": pd.to_numeric(t[get("Coarse")], errors="coerce"),
                        "fine": pd.to_numeric(t[get("Bulk Suspendent Sedi")], errors="coerce"),
                        "conc_ppm": pd.to_numeric(t[get("Cons")], errors="coerce"),
                        "total_kg_s": pd.to_numeric(t[get("Total Sediment")], errors="coerce")})
    return out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def cross_sections(station: str) -> tuple[pd.DataFrame, dict[str, str]]:
    path = next(RAW.glob(f"Cross_Section_*_{station}_*.xlsx"))
    t, meta = _table(path)
    out = pd.DataFrame({"date": pd.to_datetime(t["DATE"], format="%d-%b-%y", errors="coerce"),
                        "dist_m": pd.to_numeric(t["DISTANCE(m)"], errors="coerce"),
                        "rl_m": pd.to_numeric(t["RL(m)"], errors="coerce")})
    return out.dropna().reset_index(drop=True), meta


if __name__ == "__main__":
    for s in ("SW90", "SW91.9L", "SW93.5L"):
        d = sediment(s)
        print(s, "sediment", len(d), d.date.min().date(), d.date.max().date(),
              d.groupby(d.date.dt.year).size().to_dict())
        print(d.describe().round(1).to_string())
    for s in ("SW91.9L", "SW93.5L", "SW90", "SW46.9L", "SW273"):
        q = discharge(s)
        print(s, "Q", len(q), q.date.min().date(), q.date.max().date(), q.groupby(q.date.dt.year).size().to_dict())
    for s in ("RMG12", "RMP2", "RMP3", "RMP4"):
        c, m = cross_sections(s)
        print(s, {k: v for k, v in m.items() if re.search("Bank|River", k)})
        print("  surveys:", c.groupby("date").size().to_dict())
