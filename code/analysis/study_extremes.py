"""Study A: flood hazard facts for riverbank communities (descriptive + regression).

A1  days above the danger level, per station and year
A2  danger spells: count, longest spell, days from peak back below danger
A3  Gumbel return levels (2/10/25/50/100-yr) of the annual maximum stage,
    and the return period of the danger level itself
A4  speed of rise: 99th-percentile 1-day rise, largest 3-day rise
A5  flood calendar: first/last day above the warning level, season length
A6  compound floods: timing of the Ganges and Jamuna annual peaks, and the
    Padma peak it produces
A7  trends: Sen slope and Mann-Kendall test of annual max and dry-season min
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import common as cm

MIN_MONSOON_COVER = 0.9  # a year counts for extremes only if Jul-Oct >= 90% observed
RETURN_PERIODS = [2, 10, 25, 50, 100]
MAIN_STATIONS = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17]


def flood_days(stage: pd.DataFrame, danger: pd.Series, offset: float = 0.0) -> pd.DataFrame:
    """Year x station count of days with WL >= DL + offset."""
    above = stage.ge(danger.reindex(stage.columns) + offset) & stage.notna()
    return above.groupby(stage.index.year).sum().astype(int)


def spells(margin: pd.Series) -> list[tuple[pd.Timestamp, int]]:
    """(start, length) of each run of consecutive days at/above danger."""
    above = (margin >= 0).astype(int)
    starts = above.diff().fillna(above.iloc[0]) == 1
    out = []
    for s in margin.index[starts]:
        run = above.loc[s:]
        n = int(run.cumprod().sum())
        out.append((s, n))
    return out


def spell_table(stage: pd.DataFrame, danger: pd.Series) -> pd.DataFrame:
    rows = []
    for sid in MAIN_STATIONS:
        sp = spells(stage[sid] - danger[sid])
        lengths = np.array([n for _, n in sp]) if sp else np.array([0])
        longest = max(sp, key=lambda t: t[1]) if sp else (pd.NaT, 0)
        rows.append({"Id": sid, "n_spells": len(sp), "mean_len": float(lengths.mean()) if sp else 0,
                     "longest": int(longest[1]), "longest_start": longest[0]})
    return pd.DataFrame(rows).set_index("Id")


def annual_maxima(stage: pd.DataFrame) -> pd.DataFrame:
    mon = stage[stage.index.month.isin([7, 8, 9, 10])]
    cover = mon.notna().groupby(mon.index.year).mean()
    amax = stage.groupby(stage.index.year).max()
    return amax.where(cover >= MIN_MONSOON_COVER)


def gumbel_return_levels(x: np.ndarray, rng: np.random.Generator, n_boot: int = 2000) -> dict:
    loc, scale = stats.gumbel_r.fit(x)
    y = {T: loc - scale * np.log(-np.log(1 - 1 / T)) for T in RETURN_PERIODS}
    boots = []
    for _ in range(n_boot):
        b = rng.choice(x, size=len(x), replace=True)
        bl, bs = stats.gumbel_r.fit(b)
        boots.append([bl - bs * np.log(-np.log(1 - 1 / T)) for T in RETURN_PERIODS])
    lo, hi = np.percentile(np.array(boots), [5, 95], axis=0)
    return {"loc": loc, "scale": scale, "levels": y,
            "lo": dict(zip(RETURN_PERIODS, lo)), "hi": dict(zip(RETURN_PERIODS, hi))}


def return_level_table(stage: pd.DataFrame, danger: pd.Series) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(cm.SEED)
    amax = annual_maxima(stage)
    rows, fits = [], {}
    for sid in MAIN_STATIONS:
        x = amax[sid].dropna().to_numpy()
        fit = gumbel_return_levels(x, rng)
        fits[sid] = {"fit": fit, "amax": amax[sid].dropna()}
        p_exceed = float(stats.gumbel_r.sf(danger[sid], fit["loc"], fit["scale"]))
        row = {"Id": sid, "n_years": len(x), "DL": danger[sid], "record": x.max(),
               "record_year": int(amax[sid].idxmax()), "p_exceed_DL": p_exceed,
               "T_DL": 1 / p_exceed if p_exceed > 0 else np.inf,
               "years_above_DL": int((x >= danger[sid]).sum())}
        for T in RETURN_PERIODS:
            row[f"RL{T}"] = fit["levels"][T]
            row[f"RL{T}_lo"], row[f"RL{T}_hi"] = fit["lo"][T], fit["hi"][T]
        rows.append(row)
    return pd.DataFrame(rows).set_index("Id"), fits


def rise_table(stage: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sid in MAIN_STATIONS:
        s = stage[sid]
        d1 = (s - s.shift(1))[s.index.month.isin(range(5, 11))]
        d3 = s - s.shift(3)
        rows.append({"Id": sid, "rise_p99_cm": 100 * d1.quantile(0.99),
                     "rise3_max_cm": 100 * d3.max(), "rise3_max_date": d3.idxmax()})
    return pd.DataFrame(rows).set_index("Id")


def flood_calendar(stage: pd.DataFrame, danger: pd.Series) -> pd.DataFrame:
    """First/last warning-level day and peak day per target and year."""
    rows = []
    for sid in [10, 11, 12, 13, 15]:
        s = stage[sid]
        for year, g in s.groupby(s.index.year):
            warn = g[g >= danger[sid] - 1.0]
            if g.notna().mean() < 0.9:
                continue
            rows.append({"Id": sid, "year": year,
                         "onset_doy": warn.index[0].dayofyear if len(warn) else np.nan,
                         "end_doy": warn.index[-1].dayofyear if len(warn) else np.nan,
                         "peak_doy": g.idxmax().dayofyear, "peak_margin": g.max() - danger[sid],
                         "warn_days": len(warn)})
    return pd.DataFrame(rows)


def compound_peaks(stage: pd.DataFrame, danger: pd.Series) -> pd.DataFrame:
    """Ganges (Hardinge Br., 4) vs Jamuna (Aricha, 9) peak timing and the Padma response."""
    rows = []
    for year in range(2011, 2026):
        g = stage.loc[str(year)]
        if g[[4, 9, 10, 12]].notna().mean().min() < 0.9:
            continue
        pg, pj = g[4].idxmax(), g[9].idxmax()
        rows.append({"year": year, "ganges_peak": pg, "jamuna_peak": pj,
                     "gap_days": abs((pg - pj).days),
                     "goalundo_margin": g[10].max() - danger[10],
                     "bhagyakul_margin": g[12].max() - danger[12],
                     "bhagyakul_danger_days": int((g[12] >= danger[12]).sum())})
    return pd.DataFrame(rows)


def sen_mk(y: pd.Series) -> tuple[float, float, float, float]:
    """Sen slope (per year) with 90% CI, and Mann-Kendall two-sided p."""
    y = y.dropna()
    res = stats.theilslopes(y.to_numpy(), y.index.to_numpy(), alpha=0.90)
    tau = stats.kendalltau(y.index.to_numpy(), y.to_numpy())
    return res.slope, res.low_slope, res.high_slope, tau.pvalue


def trend_table(stage: pd.DataFrame) -> pd.DataFrame:
    amax = annual_maxima(stage)
    dry = stage[stage.index.month.isin([1, 2, 3, 4, 5])]
    dcover = dry.notna().groupby(dry.index.year).mean()
    amin = dry.groupby(dry.index.year).min().where(dcover >= 0.9)
    rows = []
    for sid in MAIN_STATIONS:
        smax = sen_mk(amax[sid])
        smin = sen_mk(amin[sid])
        rows.append({"Id": sid, "max_slope_cm": 100 * smax[0], "max_lo": 100 * smax[1],
                     "max_hi": 100 * smax[2], "max_p": smax[3],
                     "min_slope_cm": 100 * smin[0], "min_lo": 100 * smin[1],
                     "min_hi": 100 * smin[2], "min_p": smin[3]})
    return pd.DataFrame(rows).set_index("Id")


def main() -> None:
    reg = cm.load_registry()
    danger = reg["Danger_Level_mMSL"]
    stage = cm.stage_matrix(cm.load_long())
    flood_days(stage, danger).to_csv(cm.RES_DIR / "flood_days.csv")
    flood_days(stage, danger, -1.0).to_csv(cm.RES_DIR / "warning_days.csv")
    spell_table(stage, danger).to_csv(cm.RES_DIR / "danger_spells.csv")
    rl, fits = return_level_table(stage, danger)
    rl.to_csv(cm.RES_DIR / "return_levels.csv")
    pd.to_pickle(fits, cm.RES_DIR / "gumbel_fits.pkl")
    rise_table(stage).to_csv(cm.RES_DIR / "rise_rates.csv")
    flood_calendar(stage, danger).to_csv(cm.RES_DIR / "flood_calendar.csv", index=False)
    compound_peaks(stage, danger).to_csv(cm.RES_DIR / "compound_peaks.csv", index=False)
    trend_table(stage).to_csv(cm.RES_DIR / "trends.csv")
    print("extremes done")


if __name__ == "__main__":
    main()
