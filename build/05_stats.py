"""Step 05 - statistical characteristics of the series, before and after
preprocessing, plus two validation checks on the delivered daily average.

"Before" is the series exactly as the workbooks deliver it. "After" is the
analysis-ready view: inside the study window, on a complete daily calendar,
restricted to rows whose seven-day history is whole. Reporting both answers the
question that matters for a complete-case design - whether dropping incomplete
rows moved the distribution.

Outputs
    results/stats_before_after.csv    per station, both stages
    results/stats_by_period.csv       per target station x period
    results/gap_inventory.csv         every gap, with length and dates
    results/stationarity.csv          ADF / KPSS per station
    results/autocorrelation.csv       ACF of stage and of its daily change
    results/cross_correlation.csv     station x station correlation
    results/dailyavg_validation.csv   delivered average vs 3-hourly readings
    results/completecase_bias.csv     kept vs dropped rows compared
"""
from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
from statsmodels.tsa.stattools import acf, adfuller, kpss

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
import corridor as C  # noqa: E402

warnings.filterwarnings("ignore")
MAX_LAG = 14


def describe(x: pd.Series) -> dict[str, float]:
    v = x.dropna().to_numpy()
    if v.size < 3:
        return {}
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return {
        "N": int(v.size), "Mean": v.mean(), "SD": v.std(ddof=1),
        "Min": v.min(), "Q1": q1, "Median": med, "Q3": q3, "Max": v.max(),
        "IQR": q3 - q1, "Range": v.max() - v.min(),
        "Skewness": sps.skew(v), "Kurtosis": sps.kurtosis(v),
        "CV": v.std(ddof=1) / v.mean() if v.mean() else np.nan,
    }


def gaps(s: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    na = s.isna()
    if not na.any():
        return []
    grp = (na != na.shift()).cumsum()[na]
    idx = s.index.to_series()[na]
    return [(g.min(), g.max(), len(g)) for _, g in idx.groupby(grp)]


def main() -> None:
    raw = pd.read_csv(C.INTERIM / "wl_daily_raw.csv", parse_dates=["Date"])
    long = pd.read_csv(C.DATA / "pointwise_wl_7day_long.csv", parse_dates=["Date"])
    reg = pd.read_csv(C.DATA / "station_registry.csv").set_index("Station_ID")
    calendar = pd.date_range(C.STUDY_START, C.STUDY_END, freq="D")

    before_after, gap_rows, stat_rows, acf_rows, bias_rows = [], [], [], [], []
    series = {}

    for sid in C.MAIN_STEM:
        r = raw[raw.Station_ID == sid].set_index("Date")["WL_avg"]
        sub = long[long.Station_ID == sid]
        after = sub[sub.Window_Complete].set_index("Date")["WL"]
        s_cal = r.reindex(calendar)
        series[sid] = s_cal

        for stage, x in (("before (as delivered)", r), ("after (analysis-ready)", after)):
            d = describe(x)
            if d:
                before_after.append({"Id": C.STATION_ID[sid], "Station_ID": sid,
                                     "Stage": stage, **d})

        # did complete-case filtering move the distribution?
        dropped = r[~r.index.isin(set(after.index))].dropna()
        if len(dropped) >= 3:
            ks = sps.ks_2samp(after.to_numpy(), dropped.to_numpy())
            bias_rows.append({
                "Id": C.STATION_ID[sid], "Station_ID": sid,
                "N_Kept": len(after), "N_Dropped": len(dropped),
                "Mean_Kept": after.mean(), "Mean_Dropped": dropped.mean(),
                "Mean_Diff": after.mean() - dropped.mean(),
                "SD_Kept": after.std(ddof=1), "SD_Dropped": dropped.std(ddof=1),
                "KS_Statistic": ks.statistic, "KS_p": ks.pvalue,
            })

        for a, b, n in gaps(s_cal):
            # A gap is a calendar-month block when it starts on the first of a
            # month and ends on a month end - whether it spans one month or
            # forty-four. Testing only against a single month's end would miss
            # every multi-month block, which is most of the missing volume.
            gap_rows.append({"Id": C.STATION_ID[sid], "Station_ID": sid,
                             "Start": a.date(), "End": b.date(), "Length_Days": n,
                             "Whole_Calendar_Month": bool(
                                 a.day == 1 and b == b + pd.offsets.MonthEnd(0)),
                             "Period": C.period_of(pd.Series([a]))[0]})

        v = s_cal.interpolate(limit_area="inside").dropna()
        adf = adfuller(v, autolag="AIC")
        kp = kpss(v, regression="c", nlags="auto")
        stat_rows.append({
            "Id": C.STATION_ID[sid], "Station_ID": sid,
            "ADF_Statistic": adf[0], "ADF_p": adf[1],
            "KPSS_Statistic": kp[0], "KPSS_p": kp[1],
            "ADF_Rejects_Unit_Root": adf[1] < 0.05,
            "KPSS_Rejects_Stationary": kp[1] < 0.05,
        })

        a_lvl = acf(v, nlags=MAX_LAG, fft=True)
        a_dif = acf(v.diff().dropna(), nlags=MAX_LAG, fft=True)
        for k in range(1, MAX_LAG + 1):
            acf_rows.append({"Id": C.STATION_ID[sid], "Station_ID": sid, "Lag": k,
                             "ACF_Level": a_lvl[k], "ACF_Daily_Change": a_dif[k]})

    pd.DataFrame(before_after).to_csv(C.RESULTS / "stats_before_after.csv", index=False)
    pd.DataFrame(gap_rows).to_csv(C.RESULTS / "gap_inventory.csv", index=False)
    pd.DataFrame(stat_rows).to_csv(C.RESULTS / "stationarity.csv", index=False)
    pd.DataFrame(acf_rows).to_csv(C.RESULTS / "autocorrelation.csv", index=False)
    pd.DataFrame(bias_rows).to_csv(C.RESULTS / "completecase_bias.csv", index=False)

    # per target x period, on the analysis-ready view
    per = []
    for sid in C.TARGETS:
        sub = long[(long.Station_ID == sid) & long.Window_Complete]
        for name, a, b in C.PERIODS:
            w = sub[(sub.Date >= a) & (sub.Date <= b)]
            d = describe(w["WL"])
            if d:
                dl = float(reg.at[sid, "Danger_Level_mMSL"])
                per.append({"Id": C.STATION_ID[sid], "Station_ID": sid, "Period": name,
                            **d, "Danger_Level": dl,
                            "N_Days_Above_DL": int((w["WL"] >= dl).sum()),
                            "Pct_Days_Above_DL": round(100 * (w["WL"] >= dl).mean(), 2),
                            "Lag1_ACF": w["WL"].autocorr(1),
                            "Mean_Abs_Daily_Change": w["WL"].diff().abs().mean()})
    pd.DataFrame(per).to_csv(C.RESULTS / "stats_by_period.csv", index=False)

    M = pd.DataFrame(series)[C.MAIN_STEM]
    M.corr().to_csv(C.RESULTS / "cross_correlation.csv")

    # ---- delivered Daily Avg vs the 3-hourly readings, where both exist
    val, comparison = [], []
    for p in sorted(C.RAW.glob("Water_Level_3_Hourly_*.xlsx")):
        sid = re.search(r"_(SW[0-9.]+[A-Z]?)_2608", p.name).group(1)
        h = pd.read_excel(p, header=10, engine="openpyxl")
        h = h.rename(columns=lambda c: str(c).strip())
        cols = {c.lower(): c for c in h.columns}
        dt = pd.to_datetime(h[cols["date time"]], format="%d-%b-%Y %I:%M:%S %p",
                            errors="coerce")
        wl = pd.to_numeric(h[next(v for k, v in cols.items() if k.startswith("wl"))],
                           errors="coerce")
        typ = h[cols["data type"]].astype(str).str.upper()
        # REGULAR rows are the scheduled 3-hourly readings; HIGH and LOW repeat
        # a regular timestamp whenever a turning point falls on one, so counting
        # them would weight turning points twice.
        f = pd.DataFrame({"Date": dt.dt.normalize(), "WL": wl, "Type": typ})
        f = f[f.Type.eq("REGULAR") & f.Date.notna() & f.WL.notna()]
        recomputed = f.groupby("Date").agg(Mean_3h=("WL", "mean"),
                                           N_Readings=("WL", "size"))
        src = raw[raw.Station_ID == sid].set_index("Date")
        j = recomputed.join(src["WL_avg"].rename("Delivered_Avg"), how="inner")
        j = j.join(((src.WL_max + src.WL_min) / 2).rename("Midpoint"), how="left")
        j = j.dropna(subset=["Delivered_Avg", "Mean_3h"])
        if len(j):
            keep = j.reset_index(names="Date")
            keep.insert(0, "Station_ID", sid)
            comparison.append(keep)
            diff = (j.Delivered_Avg - j.Mean_3h).abs()
            val.append({
                "Station_ID": sid, "Id": C.STATION_ID[sid], "N_Days_Compared": len(j),
                "First": j.index.min().date(), "Last": j.index.max().date(),
                "Median_Readings_Per_Day": float(j.N_Readings.median()),
                "Mean_Abs_Diff_vs_3h": float(diff.mean()),
                "Max_Abs_Diff_vs_3h": float(diff.max()),
                "Corr_vs_3h": float(j.Delivered_Avg.corr(j.Mean_3h)),
                "Pct_Within_5cm_of_3h": round(100 * (diff < 0.05).mean(), 2),
                "Pct_Equal_To_Midpoint": round(
                    100 * ((j.Delivered_Avg - j.Midpoint).abs() < 5e-4).mean(), 2),
            })
    pd.DataFrame(val).to_csv(C.RESULTS / "dailyavg_validation.csv", index=False)
    if comparison:
        pd.concat(comparison, ignore_index=True).to_csv(
            C.RESULTS / "dailyavg_comparison.csv", index=False)

    ba = pd.DataFrame(before_after)
    print("  stage comparison, target stations (mean / SD, m):")
    for sid in C.TARGETS:
        rows = ba[ba.Station_ID == sid]
        b = rows[rows.Stage.str.startswith("before")].iloc[0]
        a = rows[rows.Stage.str.startswith("after")].iloc[0]
        print(f"    {sid:<9} before n={b.N:>5} {b.Mean:>6.3f}/{b.SD:>5.3f}   "
              f"after n={a.N:>5} {a.Mean:>6.3f}/{a.SD:>5.3f}   "
              f"dmean={a.Mean - b.Mean:+.4f}")
    n_month = sum(g["Whole_Calendar_Month"] for g in gap_rows)
    print(f"\n  gaps inventoried     {len(gap_rows)}  (whole calendar months: {n_month})")
    lag1 = [r["ACF_Level"] for r in acf_rows if r["Lag"] == 1]
    print(f"  lag-1 ACF range      {min(lag1):.4f} .. {max(lag1):.4f}")
    if val:
        print("  daily-average validation vs 3-hourly:")
        for v in val:
            print(f"    {v['Station_ID']:<9} n={v['N_Days_Compared']:>5}  "
                  f"MAD={v['Mean_Abs_Diff_vs_3h']:.4f} m  r={v['Corr_vs_3h']:.5f}  "
                  f"within 5 cm: {v['Pct_Within_5cm_of_3h']:.1f}%  "
                  f"= midpoint: {v['Pct_Equal_To_Midpoint']:.1f}%")


if __name__ == "__main__":
    main()
