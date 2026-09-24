"""Padma Bridge impact study (temporal x spatial), from the raw BWDB workbooks.

Spatial design (upstream -> downstream):
  RMG12  Ganges at Hardinge Bridge, ~200 km upstream (control, unaffected reach)
  RMP4   Dohar-Faridpur, ~18 km upstream of the bridge
  RMP3   Lohajang-Shibchar, on the bridge alignment (90.255 E)
  RMP2   Lohajang-Zanjira, ~7 km downstream
Temporal design: Before (surveys/samples up to 25 Nov 2014), During (to 25 Jun
2022), After (from 26 Jun 2022). Dry-season surveys of early 2022 are "During".

Analyses
  1. Specific-gauge analysis: stage at a fixed discharge per year (bed level / afflux proxy).
  2. Sediment rating: suspended-sediment concentration vs discharge, era offsets,
     bridge site (Mawa) against the upstream control (Baruria) - difference in differences.
  3. Cross-section morphology: channel area, width, mean bed level and thalweg per survey,
     bed change against the 2011-2014 baseline (erosion / deposition map).
  4. Tidal range along the reach by era.
  5. Clustering of cross-section shapes (hypsometry) - do post-bridge channels differ?
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import common as cm
import raw_bridge as rb
from study_bridge import ERAS

OUT = cm.RES_DIR
RNG = np.random.default_rng(cm.SEED)
ERA_ORDER = ["Before", "During", "After"]
TRANSECTS = {"RMG12": ("Hardinge Br. (control)", 4, -200.0),
             "RMP4": ("Dohar-Faridpur", None, -18.0),
             "RMP3": ("Bridge line (Lohajang-Shibchar)", 13, 0.0),
             "RMP2": ("Lohajang-Zanjira", 13, 7.0)}
Q_STATIONS = {"SW90": ("Hardinge Br.", 4, (10000, 40000)), "SW91.9L": ("Baruria", 11, (30000, 70000)),
              "SW93.5L": ("Mawa (bridge)", 13, (30000, 70000))}


def era(dates: pd.Series) -> pd.Series:
    d = pd.to_datetime(dates)
    out = pd.Series("After", index=d.index)
    out[d <= ERAS["During"][1]] = "During"
    out[d <= ERAS["Before"][1]] = "Before"
    return out


# ---------------------------------------------------------------- 1. specific gauge
def clean_q(q: pd.DataFrame, station: str) -> pd.DataFrame:
    q = q[(q.q > 500) & q.wl.between(-2, 25)].copy()
    # drop stage outliers against a robust annual rating (typos such as Mawa WL 17.2 m)
    fit = np.polyfit(np.log(q.q), q.wl, 2)
    res = q.wl - np.polyval(fit, np.log(q.q))
    mad = 1.4826 * np.median(np.abs(res - np.median(res)))
    return q[np.abs(res) < 6 * mad]


def specific_gauge() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, curves = [], []
    stage = cm.stage_matrix(cm.load_long())
    for st, (name, gid, qstars) in Q_STATIONS.items():
        # The workbook's own stage column changes datum (Mawa: +0.713 m until 2021, 0 after;
        # Hardinge: 0 before 2020), so stage is taken from the released daily WL instead.
        q = rb.discharge(st)
        q["wl"] = q.date.map(stage[gid])
        q = clean_q(q.dropna(), st)
        q["era"] = era(q.date).to_numpy()
        for yr, g in q.groupby(q.date.dt.year):
            for qs in qstars:
                w = g[g.q.between(0.75 * qs, 1.33 * qs)]
                if len(w) < 8:
                    continue
                b = np.polyfit(np.log(w.q), w.wl, 1)
                est = np.polyval(b, np.log(qs))
                boots = []
                for _ in range(300):
                    s = w.sample(len(w), replace=True, random_state=int(RNG.integers(1e9)))
                    if s.q.nunique() > 2:
                        boots.append(np.polyval(np.polyfit(np.log(s.q), s.wl, 1), np.log(qs)))
                rows.append({"station": st, "name": name, "year": yr, "q_star": qs, "wl": est,
                             "lo": np.percentile(boots, 5), "hi": np.percentile(boots, 95), "n": len(w)})
        for e, g in q.groupby("era"):
            b = np.polyfit(np.log(g.q), g.wl, 2)
            grid = np.geomspace(g.q.quantile(0.02), g.q.quantile(0.98), 60)
            curves.append(pd.DataFrame({"station": st, "era": e, "q": grid, "wl": np.polyval(b, np.log(grid))}))
    return pd.DataFrame(rows), pd.concat(curves)


def era_change(t: pd.DataFrame, keys: list[str], value: str) -> pd.DataFrame:
    """Era means plus a Sen slope per group (for the tables)."""
    t = t.assign(era=era(pd.to_datetime(t.year.astype(str) + "-07-01")).to_numpy())
    rows = []
    for k, g in t.groupby(keys):
        rec = dict(zip(keys, k if isinstance(k, tuple) else (k,)))
        for e in ERA_ORDER:
            rec[e] = g.loc[g.era == e, value].mean()
        s = stats.theilslopes(g[value], g.year)
        mk = stats.kendalltau(g.year, g[value])
        rec.update({"after_minus_before": rec["After"] - rec["Before"], "sen_per_yr": s[0], "p": mk.pvalue})
        rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 2. sediment
def sediment_rating() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    samples = []
    for st in ("SW91.9L", "SW93.5L"):
        s = rb.sediment(st)
        q = clean_q(rb.discharge(st), st).set_index("date").q
        s = s[s.conc_ppm > 0].copy()
        s["q"] = s.date.map(q)
        s["station"] = st
        s["name"] = Q_STATIONS[st][0]
        samples.append(s.dropna(subset=["q"]))
    s = pd.concat(samples, ignore_index=True)
    s["era"] = era(s.date).to_numpy()
    s["logc"], s["logq"] = np.log10(s.conc_ppm), np.log10(s.q)
    s["monsoon"] = s.date.dt.month.isin(range(6, 11))
    fits, eras = [], []
    for st, g in s.groupby("station"):
        base = g[g.era == "Before"]
        b = np.polyfit(base.logq, base.logc, 1)
        g = g.assign(resid=g.logc - np.polyval(b, g.logq))
        s.loc[g.index, "resid"] = g.resid
        fits.append({"station": st, "slope": b[0], "intercept": b[1], "n_before": len(base),
                     "r2_before": np.corrcoef(base.logq, base.logc)[0, 1] ** 2})
        for e in ERA_ORDER:
            r = g.loc[g.era == e, "resid"].to_numpy()
            boots = [RNG.choice(r, len(r)).mean() for _ in range(2000)]
            eras.append({"station": st, "era": e, "n": len(r), "mean_resid_log10": r.mean(),
                         "factor": 10 ** r.mean(), "lo": 10 ** np.percentile(boots, 5),
                         "hi": 10 ** np.percentile(boots, 95),
                         "median_conc_monsoon": g[(g.era == e) & g.monsoon].conc_ppm.median()})
    return s, pd.DataFrame(fits), pd.DataFrame(eras)


# ---------------------------------------------------------------- 3. cross-sections
def reference_levels() -> dict[str, tuple[float, float]]:
    """Dry-season median stage (low-water channel) and BWDB danger level (flood channel)."""
    long = cm.load_long()
    stage = cm.stage_matrix(long)
    danger = cm.load_registry()["Danger_Level_mMSL"]
    dry = stage[stage.index.month.isin([1, 2, 3, 4])].median()
    out = {}
    for t, (_, gid, _) in TRANSECTS.items():
        if gid is None:  # between Baruria (11) and Bhagyakul (12): chainage-weighted mean
            out[t] = (float((dry[11] + dry[12]) / 2), float((danger[11] + danger[12]) / 2))
        else:
            out[t] = (float(dry[gid]), float(danger[gid]))
    return out


def profiles() -> pd.DataFrame:
    rows = []
    for t in TRANSECTS:
        c, _ = rb.cross_sections(t)
        c["year"] = c.date.dt.year
        xmax = c.groupby("year").dist_m.max().min()
        grid = np.arange(0, xmax + 1, 10.0)
        for yr, g in c.groupby("year"):
            g = g.groupby("dist_m").rl_m.mean().sort_index()
            rows.append(pd.DataFrame({"transect": t, "year": yr, "x": grid,
                                      "z": np.interp(grid, g.index, g.to_numpy())}))
    return pd.concat(rows, ignore_index=True)


def xs_metrics(prof: pd.DataFrame, refs: dict) -> pd.DataFrame:
    rows = []
    for (t, yr), g in prof.groupby(["transect", "year"]):
        low, flood = refs[t]
        z, dx = g.z.to_numpy(), 10.0
        wet_low, wet_flood = z < low, z < flood
        rows.append({"transect": t, "year": yr,
                     "area_low_m2": float(((low - z) * wet_low).sum() * dx),
                     "width_low_m": float(wet_low.sum() * dx),
                     "area_flood_m2": float(((flood - z) * wet_flood).sum() * dx),
                     "width_flood_m": float(wet_flood.sum() * dx),
                     "mean_bed_m": float(z[wet_flood].mean()),
                     "thalweg_m": float(np.percentile(z, 1)),
                     "thalweg_x_m": float(g.x.to_numpy()[np.argmin(z)]),
                     "char_width_m": float(((z >= low) & (z < flood)).sum() * dx)})
    m = pd.DataFrame(rows)
    m["era"] = era(pd.to_datetime(m.year.astype(str) + "-03-01")).to_numpy()
    return m


def bed_change(prof: pd.DataFrame) -> pd.DataFrame:
    base = prof[prof.year <= 2014].groupby(["transect", "x"]).z.mean().rename("z0")
    p = prof.join(base, on=["transect", "x"])
    p["dz"] = p.z - p.z0
    return p


# ---------------------------------------------------------------- 4. tide
def tidal_range() -> pd.DataFrame:
    long = cm.load_long()
    rows = []
    for gid in (12, 13, 15, 17):
        g = long[(long.Id == gid) & long.Date.dt.month.isin([1, 2, 3, 4])]
        for yr, y in g.groupby(g.Date.dt.year):
            rows.append({"gauge": gid, "year": yr, "range_m": y["WL_Range_D-1"].median()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 5. clustering
def xs_clusters(prof: pd.DataFrame, refs: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hypsometry: share of the flood-channel width whose bed lies >= d metres below the
    danger level, d = 0..20 m. Shape only - comparable across transects."""
    rows, feats = [], []
    depths = np.arange(0, 21, 1.0)
    for (t, yr), g in prof.groupby(["transect", "year"]):
        z = g.z.to_numpy()
        below = refs[t][1] - z
        wet = below > 0
        feats.append([(below[wet] >= d).mean() for d in depths])
        rows.append({"transect": t, "year": yr})
    x = StandardScaler().fit_transform(np.array(feats))
    ktab = []
    for k in range(2, 7):
        lab = KMeans(k, n_init=50, random_state=cm.SEED).fit_predict(x)
        ktab.append({"k": k, "silhouette": silhouette_score(x, lab), "min_size": np.bincount(lab).min()})
    ktab = pd.DataFrame(ktab)
    k = int(ktab[ktab.min_size > 1].sort_values("silhouette").k.iloc[-1])
    km = KMeans(k, n_init=50, random_state=cm.SEED).fit(x)
    meta = pd.DataFrame(rows)
    hyp = np.array(feats)
    order = np.argsort([hyp[km.labels_ == c, 5].mean() for c in range(k)])  # shallow -> deep
    meta["cluster"] = pd.Series(km.labels_).map({o: n for n, o in enumerate(order)})
    meta["era"] = era(pd.to_datetime(meta.year.astype(str) + "-03-01")).to_numpy()
    hdf = pd.DataFrame(hyp, columns=[f"d{int(d)}" for d in depths])
    meta = pd.concat([meta, hdf], axis=1)
    return meta, ktab


def main() -> None:
    sg, curves = specific_gauge()
    sg.to_csv(OUT / "bi_specific_gauge.csv", index=False)
    curves.to_csv(OUT / "bi_rating_curves.csv", index=False)
    era_change(sg, ["station", "name", "q_star"], "wl").to_csv(OUT / "bi_specific_gauge_eras.csv", index=False)
    s, fits, eras = sediment_rating()
    s.to_csv(OUT / "bi_sediment_samples.csv", index=False)
    fits.to_csv(OUT / "bi_sediment_fits.csv", index=False)
    eras.to_csv(OUT / "bi_sediment_eras.csv", index=False)
    refs = reference_levels()
    pd.DataFrame(refs, index=["low_ref", "flood_ref"]).T.to_csv(OUT / "bi_xs_refs.csv")
    prof = profiles()
    bed_change(prof).to_csv(OUT / "bi_xs_profiles.csv", index=False)
    m = xs_metrics(prof, refs)
    m.to_csv(OUT / "bi_xs_metrics.csv", index=False)
    rows = []
    for col in ("area_low_m2", "area_flood_m2", "width_flood_m", "mean_bed_m", "thalweg_m", "char_width_m"):
        e = era_change(m.rename(columns={col: "v"})[["transect", "year", "v"]], ["transect"], "v")
        rows.append(e.assign(metric=col))
    pd.concat(rows).to_csv(OUT / "bi_xs_eras.csv", index=False)
    tidal_range().to_csv(OUT / "bi_tide.csv", index=False)
    meta, ktab = xs_clusters(prof, refs)
    meta.to_csv(OUT / "bi_xs_clusters.csv", index=False)
    ktab.to_csv(OUT / "bi_xs_clusters_k.csv", index=False)
    print("done")


if __name__ == "__main__":
    main()
