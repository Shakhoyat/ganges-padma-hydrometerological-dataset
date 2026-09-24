"""Study C: unsupervised structure in the corridor (clustering).

C1  Gauges grouped by behaviour only (no coordinates, no Id): k-means and
    Ward, k chosen by silhouette with single-member partitions rejected.
C2  Flood years grouped by how the monsoon unfolded along the corridor.
C3  Daily hydraulic regimes of the bridge reach: k-means on the state of seven
    gauges (margin to danger level and 3-day change). Regime shares per
    bridge era, the regime transition matrix, and each regime's chance of a
    warning-level day (DL - 1 m) at Bhagyakul/Mawa within 3 days (a clustering-based warning).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import common as cm
from study_bridge import era_of

REGIME_GAUGES = [4, 9, 10, 11, 12, 13, 15]


def choose_k(x: np.ndarray, ks: range) -> tuple[int, pd.DataFrame]:
    rows = []
    for k in ks:
        lab = KMeans(k, n_init=50, random_state=cm.SEED).fit_predict(x)
        sizes = np.bincount(lab)
        rows.append({"k": k, "silhouette": silhouette_score(x, lab), "min_size": sizes.min(),
                     "sizes": "/".join(map(str, sorted(sizes)))})
    tab = pd.DataFrame(rows)
    ok = tab[tab.min_size > 1]
    return int(ok.loc[ok.silhouette.idxmax(), "k"]), tab


def station_features(long: pd.DataFrame, stage: pd.DataFrame, danger: pd.Series) -> pd.DataFrame:
    rows = []
    for sid in range(1, 18):
        s = stage[sid]
        d = s.diff()
        amp = (s.groupby(s.index.year).max() - s.groupby(s.index.year).min()).median()
        peak_doy = s.groupby(s.index.year).idxmax().dropna().dt.dayofyear
        ang = 2 * np.pi * peak_doy / 365.25
        rng = long.loc[long.Id == sid, "WL_Range_D-1"]
        rows.append({"Id": sid, "annual_amplitude_m": amp, "daily_change_sd_cm": 100 * d.std(),
                     "change_memory_acf1": d.autocorr(1), "tidal_range_m": rng.median(),
                     "peak_timing_sin": np.sin(ang).mean(), "peak_timing_cos": np.cos(ang).mean(),
                     "warning_share": float((s >= danger[sid] - 1).mean()),
                     "rise_p99_cm": 100 * d.quantile(0.99)})
    return pd.DataFrame(rows).set_index("Id")


def cluster_stations(feat: pd.DataFrame) -> dict:
    x = StandardScaler().fit_transform(feat)
    k, tab = choose_k(x, range(2, 7))
    lab = KMeans(k, n_init=50, random_state=cm.SEED).fit_predict(x)
    pca = PCA(2).fit(x)
    return {"k": k, "k_table": tab, "labels": pd.Series(lab, index=feat.index),
            "pcs": pd.DataFrame(pca.transform(x), index=feat.index, columns=["PC1", "PC2"]),
            "loadings": pd.DataFrame(pca.components_.T, index=feat.columns, columns=["PC1", "PC2"]),
            "evr": pca.explained_variance_ratio_, "ward": linkage(x, "ward")}


def year_features() -> pd.DataFrame:
    comp = pd.read_csv(cm.RES_DIR / "compound_peaks.csv").set_index("year")
    days = pd.read_csv(cm.RES_DIR / "flood_days.csv", index_col=0)
    cal = pd.read_csv(cm.RES_DIR / "flood_calendar.csv")
    goal = cal[cal.Id == 10].set_index("year")
    feat = pd.DataFrame({
        "goalundo_peak_margin_m": comp.goalundo_margin,
        "bhagyakul_peak_margin_m": comp.bhagyakul_margin,
        "padma_danger_days": days[["10", "11", "12", "13", "15"]].sum(axis=1),
        "ganges_jamuna_peak_gap_d": comp.gap_days,
        "goalundo_warning_onset_doy": goal.onset_doy,
    })
    return feat.dropna()


def cluster_years(feat: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    x = StandardScaler().fit_transform(feat)
    lab = KMeans(k, n_init=100, random_state=cm.SEED).fit_predict(x)
    out = feat.copy()
    out["cluster"] = lab
    severity = out.groupby("cluster").padma_danger_days.mean().rank(ascending=False).astype(int)
    names = {1: "Severe compound flood", 2: "Moderate flood", 3: "Mild / non-coincident"}
    out["type"] = out.cluster.map(lambda c: names[severity[c]])
    out["silhouette"] = silhouette_score(x, lab)
    return out


def regime_table(stage: pd.DataFrame, danger: pd.Series) -> pd.DataFrame:
    cols = {}
    for g in REGIME_GAUGES:
        cols[f"m{g}"] = stage[g] - danger[g]
        cols[f"d3_{g}"] = stage[g] - stage[g].shift(3)
    return pd.DataFrame(cols).dropna()


def name_regimes(centres: pd.DataFrame) -> dict[int, str]:
    m = centres[[c for c in centres if c.startswith("m")]].mean(axis=1)
    d = centres[[c for c in centres if c.startswith("d3")]].mean(axis=1)
    names = {}
    for c in centres.index:
        if m[c] > -1.3:
            names[c] = "Flood high water"
        elif d[c] > 0.3:
            names[c] = "Monsoon rising limb"
        elif d[c] < -0.15:
            names[c] = "Post-monsoon recession"
        elif d[c] > 0.05:
            names[c] = "Pre-monsoon rise"
        else:
            names[c] = "Dry-season low"
    return names


def cluster_regimes(stage: pd.DataFrame, danger: pd.Series) -> dict:
    tab = regime_table(stage, danger)
    scaler = StandardScaler().fit(tab)
    x = scaler.transform(tab)
    sample = np.random.default_rng(cm.SEED).choice(len(x), size=min(3000, len(x)), replace=False)
    k_rows = []
    for k in range(3, 8):
        km = KMeans(k, n_init=20, random_state=cm.SEED).fit(x)
        k_rows.append({"k": k, "silhouette": silhouette_score(x[sample], km.labels_[sample])})
    k = 5
    km = KMeans(k, n_init=50, random_state=cm.SEED).fit(x)
    centres = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=tab.columns)
    order = centres[[c for c in tab if c.startswith("m")]].mean(axis=1).sort_values().index
    remap = {old: new for new, old in enumerate(order)}
    lab = pd.Series([remap[v] for v in km.labels_], index=tab.index, name="regime")
    centres = centres.loc[order].reset_index(drop=True)
    names = name_regimes(centres)
    return {"labels": lab, "centres": centres, "names": names, "k_table": pd.DataFrame(k_rows),
            "table": tab}


def regime_outcomes(lab: pd.Series, stage: pd.DataFrame, danger: pd.Series) -> pd.DataFrame:
    era = era_of(lab.index)
    danger_soon = pd.Series(False, index=stage.index)
    for g in (12, 13):
        above = (stage[g] >= danger[g] - 1.0).astype(float)
        danger_soon |= above[::-1].rolling(3, min_periods=1).max()[::-1].shift(-1).fillna(0).astype(bool)
    rows = []
    for r in sorted(lab.unique()):
        sel = lab == r
        run_id = (lab != lab.shift()).cumsum()
        runs = run_id[sel].value_counts()
        rows.append({"regime": r, "days": int(sel.sum()),
                     "mean_run_days": runs.mean(),
                     "p_warning_3d": float(danger_soon.reindex(lab.index)[sel].mean()),
                     **{f"share_{e}": float((lab[era == e] == r).mean()) for e in ["Before", "During", "After"]},
                     "peak_month": int(pd.Series(lab.index[sel].month).mode()[0])})
    trans = pd.crosstab(lab.shift().dropna().astype(int), lab.iloc[1:], normalize="index")
    return pd.DataFrame(rows).set_index("regime"), trans


def main() -> None:
    long = cm.load_long()
    stage = cm.stage_matrix(long)
    danger = cm.load_registry()["Danger_Level_mMSL"]
    feat = station_features(long, stage, danger)
    st = cluster_stations(feat)
    feat.assign(cluster=st["labels"]).join(st["pcs"]).to_csv(cm.RES_DIR / "cluster_stations.csv")
    st["k_table"].to_csv(cm.RES_DIR / "cluster_stations_k.csv", index=False)
    st["loadings"].to_csv(cm.RES_DIR / "cluster_stations_loadings.csv")
    pd.to_pickle(st, cm.RES_DIR / "cluster_stations.pkl")
    cluster_years(year_features()).to_csv(cm.RES_DIR / "cluster_years.csv")
    reg = cluster_regimes(stage, danger)
    out, trans = regime_outcomes(reg["labels"], stage, danger)
    out["name"] = out.index.map(reg["names"])
    out.to_csv(cm.RES_DIR / "regimes.csv")
    trans.to_csv(cm.RES_DIR / "regime_transitions.csv")
    reg["centres"].to_csv(cm.RES_DIR / "regime_centres.csv")
    reg["k_table"].to_csv(cm.RES_DIR / "regime_k.csv", index=False)
    reg["labels"].to_csv(cm.RES_DIR / "regime_labels.csv")
    print("clustering done")


if __name__ == "__main__":
    main()
