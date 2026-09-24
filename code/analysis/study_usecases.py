"""Ten further classification (C3-C7) and clustering (U4-U8) use cases.

Every use case reads only the released files (long panel, station registry)
plus the external BWDB danger levels already used in Section 10. Splits are
chronological (or grouped by gauge / year) exactly as stated in the report;
nothing is tuned on test data; CPU only, seed 20260913.

C3  Rise/fall nowcast, incremental feature groups (own -> rain -> rivers).
C4  Rapid-rise alert one day ahead (top-5% monsoon daily rise).
C5  Warning-in-3-days classifier transferred to an unseen gauge (leave-one-gauge-out).
C6  Gauge-type identification from 30 days of daily changes (leave-one-gauge-out).
C7  Bridge-era recognition from the reach water-surface profile (leave-one-year-out).
U4  Annual hydrograph shapes of every gauge-year (k-means).
U5  Unsupervised anomaly screening of daily readings (Isolation Forest).
U6  Gauge co-movement network, monsoon vs dry season (hierarchical clustering).
U7  Rainfall weather types over the 8 used rain gauges and the river's response.
U8  Flood-event (rising-limb) typology and the chance of reaching danger.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import kurtosis, skew
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (adjusted_rand_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, silhouette_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import common as cm
from study_bridge import era_of

GANGES, JAMUNA, MEGHNA = list(range(1, 8)), [8, 9], [16, 17]
PADMA_UP = {11: [10], 12: [10, 11], 15: [10, 11, 12, 13]}
# Nearest upstream gauge along the network (Tarpasa, Id 14, excluded everywhere).
UP = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 9: 8, 10: 7, 11: 10, 12: 11, 13: 12, 15: 13, 17: 16}
GAUGES = [g for g in range(1, 18) if g != 14]


# ------------------------------------------------------------------ helpers
def load() -> dict:
    long = cm.load_long()
    stage = cm.stage_matrix(long)
    full = stage.index
    danger = cm.load_registry()["Danger_Level_mMSL"]
    return {"long": long, "stage": stage, "dwl": stage.diff(), "margin": stage - danger,
            "danger": danger, "reg": cm.load_registry(),
            "rain": long.pivot(index="Date", columns="Id", values="Rainfall_mm").reindex(full),
            "trend": long.pivot(index="Date", columns="Id", values="WL_Trend").reindex(full).astype(float)}


def season(idx: pd.DatetimeIndex) -> pd.DataFrame:
    a = 2 * np.pi * idx.dayofyear / 365.25
    return pd.DataFrame({"doy_sin": np.sin(a), "doy_cos": np.cos(a)}, index=idx)


def logit() -> object:
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(class_weight="balanced", max_iter=3000))


def hgb(**kw) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                          class_weight="balanced", random_state=cm.SEED, **kw)


def event_scores(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    a = int(((p == 1) & (y == 1)).sum())
    b = int(((p == 1) & (y == 0)).sum())
    c = int(((p == 0) & (y == 1)).sum())
    return {"POD": a / (a + c) if a + c else np.nan, "FAR": b / (a + b) if a + b else np.nan,
            "CSI": a / (a + b + c) if a + b + c else np.nan, "events": int(y.sum())}


def choose_k(x: np.ndarray, ks: range, metric: str = "euclidean") -> tuple[int, pd.DataFrame]:
    rows = []
    for k in ks:
        lab = KMeans(k, n_init=30, random_state=cm.SEED).fit_predict(x)
        rows.append({"k": k, "silhouette": silhouette_score(x, lab, metric=metric),
                     "min_size": int(np.bincount(lab).min())})
    tab = pd.DataFrame(rows)
    ok = tab[tab.min_size > 1]
    return int(ok.loc[ok.silhouette.idxmax(), "k"]), tab


# ---------------------------------------------------------------- C3
def c3_incremental_trend(d: dict) -> pd.DataFrame:
    rows = []
    for tgt in cm.TARGETS:
        idx = d["stage"].index
        own = pd.DataFrame({f"own_d{k}": d["dwl"][tgt].shift(k) for k in range(1, 7)}, index=idx)
        own["own_margin1"] = d["margin"][tgt].shift(1)
        rain = season(idx).assign(rain=d["rain"][tgt], rain3=d["rain"][tgt].rolling(3).sum())

        def river(ids: list[int]) -> pd.DataFrame:
            cols = {}
            for g in ids:
                cols[f"g{g}_d0"] = d["dwl"][g]
                cols[f"g{g}_d1"] = d["dwl"][g].shift(1)
            return pd.DataFrame(cols, index=idx)

        steps = [("Own lags", own), ("+ rain, season", rain), ("+ Ganges (1-7)", river(GANGES)),
                 ("+ Jamuna (8-9)", river(JAMUNA)), ("+ Padma upstream", river(PADMA_UP[tgt])),
                 ("+ Meghna (16-17)", river(MEGHNA))]
        y = d["trend"][tgt]
        ok = y.notna() & own.notna().all(axis=1)
        tr = ok & (idx <= cm.TRAIN_END)
        te = ok & (idx >= cm.TEST_START)
        persist = f1_score(y[te], d["trend"][tgt].shift(1)[te].fillna(1), average="macro")
        x = pd.DataFrame(index=idx)
        for i, (name, block) in enumerate(steps):
            x = pd.concat([x, block], axis=1)
            for mname, model in (("Logistic regression", logit()), ("Gradient boosting", hgb())):
                model.fit(x[tr], y[tr].astype(int))
                f1 = f1_score(y[te], model.predict(x[te]), average="macro")
                rows.append({"target": tgt, "step": i, "step_name": name, "model": mname,
                             "n_features": x.shape[1], "macro_f1": f1, "persistence_f1": persist})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- C4
RAPID_GAUGES = {4: [1, 2, 3], 10: list(range(1, 10)), 12: [g for g in range(1, 12)]}


def c4_rapid_rise(d: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, curves = [], []
    idx = d["stage"].index
    monsoon = idx.month.isin(range(5, 11))
    for g, ups in RAPID_GAUGES.items():
        dw = d["dwl"][g]
        thr = dw[monsoon & (idx <= cm.TRAIN_END)].quantile(0.95)
        y = (dw.shift(-1) >= thr).astype(float).where(dw.shift(-1).notna())
        own = pd.DataFrame({"d0": dw, "d1": dw.shift(1), "d2": dw.shift(2),
                            "d3sum": dw.rolling(3).sum(), "margin": d["margin"][g]}, index=idx)
        up = pd.DataFrame({**{f"u{u}_d0": d["dwl"][u] for u in ups},
                           **{f"u{u}_d3": d["dwl"][u].rolling(3).sum() for u in ups}}, index=idx)
        rain = season(idx).assign(rain3=d["rain"][g].rolling(3).sum())
        steps = [("Own gauge", own), ("+ upstream gauges", pd.concat([own, up], axis=1)),
                 ("+ upstream, rain, season", pd.concat([own, up, rain], axis=1))]
        ok = y.notna() & monsoon & own.d0.notna()
        tr, te = ok & (idx <= cm.TRAIN_END), ok & (idx.year >= 2022)
        pers = (dw >= thr).astype(int)
        rows.append({"gauge": g, "threshold_cm": 100 * thr, "step": "Persistence", "model": "Persistence",
                     "PR_AUC": np.nan, **event_scores(y[te].to_numpy(), pers[te].to_numpy())})
        for name, x in steps:
            for mname, model in (("Logistic regression", logit()), ("Gradient boosting", hgb())):
                model.fit(x[tr], y[tr].astype(int))
                prob = model.predict_proba(x[te])[:, 1]
                rows.append({"gauge": g, "threshold_cm": 100 * thr, "step": name, "model": mname,
                             "PR_AUC": average_precision_score(y[te], prob),
                             **event_scores(y[te].to_numpy(), (prob >= 0.5).astype(int))})
                if mname == "Gradient boosting" and name == steps[-1][0]:
                    curves.append(pd.DataFrame({"gauge": g, "date": idx[te], "prob": prob,
                                                "y": y[te].to_numpy(), "dwl_next": dw.shift(-1)[te].to_numpy()}))
    return pd.DataFrame(rows), pd.concat(curves)


# ---------------------------------------------------------------- C5
def gauge_frame(d: dict, g: int, h: int = 3) -> pd.DataFrame:
    s, m = d["stage"][g], d["margin"][g]
    f = pd.DataFrame({"margin": m, "d1": s.diff(), "d3": s - s.shift(3), "d7": s - s.shift(7)})
    u = UP.get(g)
    su = d["stage"][u] if u else pd.Series(np.nan, index=s.index)
    mu = d["margin"][u] if u else pd.Series(np.nan, index=s.index)
    f["up_margin"], f["up_d1"], f["up_d3"] = mu, su.diff(), su - su.shift(3)
    f = f.join(season(s.index))
    f["y"] = (m.shift(-h) >= -1.0).astype(float).where(m.shift(-h).notna())
    f["persist"] = (m >= -1.0).astype(int)
    f["gauge"] = g
    return f[f.margin.notna() & f.y.notna()]


def c5_logo(d: dict) -> pd.DataFrame:
    frames = {g: gauge_frame(d, g) for g in GAUGES}
    feats = ["margin", "d1", "d3", "d7", "up_margin", "up_d1", "up_d3", "doy_sin", "doy_cos"]
    rows = []
    for g in GAUGES:
        test = frames[g][frames[g].index.year >= 2022]
        if test.y.sum() < 20:
            continue
        pool = pd.concat([f[f.index <= cm.TRAIN_END] for k, f in frames.items() if k != g])
        own = frames[g][frames[g].index <= cm.TRAIN_END]
        for name, train in (("Unseen gauge (trained on the other 15)", pool), ("Own-gauge model", own)):
            model = hgb().fit(train[feats], train.y.astype(int))
            rows.append({"gauge": g, "model": name,
                         **event_scores(test.y.to_numpy(), model.predict(test[feats]))})
        rows.append({"gauge": g, "model": "Persistence",
                     **event_scores(test.y.to_numpy(), test.persist.to_numpy())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- C6
def window_features(dw: pd.Series) -> dict[str, float] | None:
    x = dw.dropna().to_numpy()
    if len(x) < 25:
        return None
    xc = x - x.mean()
    power = np.abs(np.fft.rfft(xc)) ** 2
    ac = lambda k: np.corrcoef(x[:-k], x[k:])[0, 1] if x[k:].std() > 0 and x[:-k].std() > 0 else 0.0  # noqa: E731
    return {"sd_cm": 100 * x.std(), "mean_abs_cm": 100 * np.abs(x).mean(), "acf1": ac(1), "acf2": ac(2),
            "acf7": ac(7), "share_rising": float((x > 0.005).mean()), "skew": float(skew(x)),
            "kurt": float(kurtosis(x)),
            "fortnight_power": float(power[2] / power[1:].sum()) if power[1:].sum() > 0 else 0.0}


def c6_gauge_type(d: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    reg = d["reg"]
    tidal = (reg.Station_Type.str.strip().str.lower() == "tidal").astype(int)
    river = reg.River.replace("Lower Meghna", "Meghna")
    rows = []
    for g in range(1, 18):
        dw = d["dwl"][g]
        for start in pd.date_range("2011-01-02", "2025-12-01", freq="30D"):
            f = window_features(dw[start:start + pd.Timedelta(days=29)])
            if f:
                rows.append({"gauge": g, "start": start, **f})
    w = pd.DataFrame(rows)
    feats = [c for c in w.columns if c not in ("gauge", "start")]
    out, per_gauge = [], []
    for label, lab in (("Tidal vs non-tidal", w.gauge.map(tidal)), ("River (4 classes)", w.gauge.map(river))):
        pred = pd.Series(index=w.index, dtype=object)
        for g in range(1, 18):
            te = w.gauge == g
            rf = RandomForestClassifier(300, min_samples_leaf=3, class_weight="balanced",
                                        random_state=cm.SEED, n_jobs=-1).fit(w.loc[~te, feats], lab[~te])
            pred[te] = rf.predict(w.loc[te, feats])
        out.append({"label": label, "windows": len(w),
                    "balanced_accuracy": balanced_accuracy_score(lab.astype(str), pred.astype(str)),
                    "chance": 1 / lab.nunique()})
        for g in range(1, 18):
            te = w.gauge == g
            vote = pred[te].astype(str).mode().iloc[0]
            per_gauge.append({"label": label, "gauge": g, "true": str(lab[te].iloc[0]),
                              "window_accuracy": float((pred[te].astype(str) == str(lab[te].iloc[0])).mean()),
                              "gauge_vote": vote})
    return pd.DataFrame(out), pd.DataFrame(per_gauge)


# ---------------------------------------------------------------- C7
def c7_bridge_era(d: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    s = d["stage"]
    idx = s.index
    ctrl = season(idx).assign(goalundo_margin=d["margin"][10], chandpur_margin=d["margin"][17])
    drops = pd.DataFrame({f"drop_{a}_{b}": s[a] - s[b] for a, b in ((10, 11), (11, 12), (12, 13), (13, 15))},
                         index=idx)
    era = era_of(idx)
    ok = drops.notna().all(axis=1) & ctrl.notna().all(axis=1) & era.notna()
    years = idx.year
    rows, cms = [], []
    for name, x in (("Flow and season only", ctrl), ("+ reach water-surface drops", pd.concat([ctrl, drops], axis=1)),
                    ("Reach drops only", drops)):
        pred = pd.Series(index=idx[ok], dtype=object)
        for yr in range(2011, 2026):
            te = ok & (years == yr)
            tr = ok & (years != yr)
            model = hgb().fit(x[tr], era[tr])
            pred[idx[te]] = model.predict(x[te])
        truth = era[ok]
        rows.append({"features": name, "balanced_accuracy": balanced_accuracy_score(truth, pred),
                     "macro_f1": f1_score(truth, pred, average="macro"), "days": int(ok.sum())})
        m = confusion_matrix(truth, pred, labels=["Before", "During", "After"])
        cms.append(pd.DataFrame(m, index=["Before", "During", "After"],
                                columns=["Before", "During", "After"]).assign(features=name))
    return pd.DataFrame(rows), pd.concat(cms)


# ---------------------------------------------------------------- U4
def u4_hydrograph_shapes(d: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, curves = [], []
    for g in GAUGES:
        s = d["stage"][g]
        for yr in range(2011, 2026):
            y = s[str(yr)]
            if y.notna().mean() < 0.9:
                continue
            wk = y.interpolate(limit=7).groupby((y.index.dayofyear - 1) // 7).mean().iloc[:52]
            if wk.isna().any():
                continue
            curves.append(((wk - wk.min()) / (wk.max() - wk.min())).to_numpy())
            rows.append({"gauge": g, "year": yr})
    meta, x = pd.DataFrame(rows), np.vstack(curves)
    k, ktab = choose_k(x, range(2, 9))
    km = KMeans(k, n_init=50, random_state=cm.SEED).fit(x)
    meta["cluster"] = km.labels_
    # order clusters by the week of the centroid peak so labels read early -> late
    order = np.argsort(km.cluster_centers_.argmax(axis=1))
    remap = {old: new for new, old in enumerate(order)}
    meta["cluster"] = meta.cluster.map(remap)
    cent = pd.DataFrame(km.cluster_centers_[order]).T
    cent.index.name = "week"
    return meta, cent, ktab


# ---------------------------------------------------------------- U5
U5_SETS = {"Own change": ["z"], "+ next-day change": ["z", "z_next"],
           "+ neighbour residual": ["z", "z_next", "z_resid"], "Neighbour residual only": ["z_resid"]}


def _mad(s: pd.Series) -> float:
    return float(1.4826 * (s - s.median()).abs().median())


def u5_anomalies(d: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Isolation Forest (0.1% contamination) on robust (MAD-scaled) daily-change features.

    Reference set: the 15 station-days with |dWL| > 1 m; 'isolated' ones are those the
    network neighbours did not echo (neighbour mean change < 30% of the gauge's own).
    """
    dw = d["dwl"]
    down = {v: k for k, v in UP.items()}
    rows = []
    for g in range(1, 18):
        x = dw[g]
        nbs = [n for n in (UP.get(g), down.get(g)) if n]
        nb = dw[nbs].mean(axis=1) if nbs else pd.Series(np.nan, index=x.index)
        r = x - nb
        rows.append(pd.DataFrame({"gauge": g, "dwl": x, "dwl_next": x.shift(-1), "nb": nb,
                                  "z": x / _mad(x), "z_next": x.shift(-1) / _mad(x),
                                  "z_resid": r / _mad(r) if nbs else 0.0}).dropna(subset=["z", "z_next"]))
    f = pd.concat(rows)
    f["z_resid"] = f.z_resid.fillna(0.0)
    known = f.dwl.abs() > 1.0
    isolated = known & ((f.nb.abs() < 0.3 * f.dwl.abs()) | f.nb.isna())
    summary, flags = [], {}
    for name, feats in U5_SETS.items():
        iso = IsolationForest(n_estimators=400, contamination=0.001, random_state=cm.SEED).fit(f[feats])
        flag = iso.predict(f[feats]) == -1
        flags[name] = flag
        summary.append({"features": name, "flagged": int(flag.sum()), "known": int(known.sum()),
                        "known_caught": int((known & flag).sum()), "isolated": int(isolated.sum()),
                        "isolated_caught": int((isolated & flag).sum())})
    flagged = f[flags["Own change"]].copy()
    echoed = flagged.nb.abs() >= 0.3 * flagged.dwl.abs()
    spike = (flagged.dwl * flagged.dwl_next < 0) & (flagged.dwl_next.abs() > 0.5 * flagged.dwl.abs())
    flagged["kind"] = np.where(echoed, "echoed by neighbours",
                               np.where(spike, "spike and return", "isolated step"))
    flagged = flagged.reset_index().rename(columns={"index": "date"})
    return flagged, pd.DataFrame(summary)


# ---------------------------------------------------------------- U6
def comovement(dw: pd.DataFrame, months: list[int], max_lag: int = 3) -> pd.DataFrame:
    x = dw[dw.index.month.isin(months)]
    ids = [g for g in range(1, 18)]
    r = pd.DataFrame(np.eye(len(ids)), index=ids, columns=ids)
    for i in ids:
        for j in ids:
            if i < j:
                best = max(x[i].corr(x[j].shift(k)) for k in range(-max_lag, max_lag + 1))
                r.loc[i, j] = r.loc[j, i] = best
    return r


def u6_comovement(d: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    reg = d["reg"]
    u1 = pd.read_csv(cm.RES_DIR / "cluster_stations.csv").set_index("Id")["cluster"]
    rows, labels = [], []
    for season_name, months in (("Monsoon (Jun-Oct)", [6, 7, 8, 9, 10]), ("Dry (Jan-Apr)", [1, 2, 3, 4])):
        r = comovement(d["dwl"], months)
        r.to_csv(cm.RES_DIR / f"uc_u6_corr_{season_name.split()[0].lower()}.csv")
        dist = squareform((1 - r).clip(lower=0).to_numpy(), checks=False)
        z = linkage(dist, "average")
        best = None
        for k in range(2, 7):
            lab = fcluster(z, k, "maxclust")
            if np.bincount(lab)[1:].min() < 2:
                continue
            sil = silhouette_score((1 - r).clip(lower=0).to_numpy(), lab, metric="precomputed")
            if best is None or sil > best[1]:
                best = (k, sil, lab)
        k, sil, lab = best
        rows.append({"season": season_name, "k": k, "silhouette": sil,
                     "ARI_vs_river": adjusted_rand_score(reg.loc[r.index, "River"].replace("Lower Meghna", "Meghna"), lab),
                     "ARI_vs_U1": adjusted_rand_score(u1.loc[r.index], lab),
                     "median_r": float(np.median(r.to_numpy()[np.triu_indices(17, 1)]))})
        labels.append(pd.DataFrame({"gauge": r.index, "season": season_name, "cluster": lab}))
    return pd.DataFrame(rows), pd.concat(labels)


# ---------------------------------------------------------------- U7
def u7_rain_types(d: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reg = d["reg"]
    first = reg.reset_index().groupby("Rain_Gauge").Id.first().sort_values()
    rain = d["rain"][first.to_numpy()]
    rain.columns = [f"{cm.SHORT_NAME[i]}" for i in first]
    idx = rain.index
    m = idx.month.isin(range(5, 11)) & rain.notna().all(axis=1) & (rain.max(axis=1) > 1.0)
    x = np.log1p(rain[m])
    xs = StandardScaler().fit_transform(x)
    k, ktab = choose_k(xs, range(2, 9))
    lab = KMeans(k, n_init=50, random_state=cm.SEED).fit_predict(xs)
    # order types by mean corridor rain
    means = pd.Series(rain[m].mean(axis=1).to_numpy()).groupby(lab).mean().sort_values()
    remap = {old: new for new, old in enumerate(means.index)}
    lab = pd.Series(lab, index=rain[m].index).map(remap)
    cent = rain[m].groupby(lab).mean()
    s = d["stage"]
    resp = []
    dry = idx.month.isin(range(5, 11)) & rain.notna().all(axis=1) & (rain.max(axis=1) <= 1.0)
    groups = [("No rain (<= 1 mm everywhere)", dry)] + [(f"T{t + 1}", idx.isin(lab[lab == t].index))
                                                      for t in range(k)]
    for name, sel in groups:
        rec = {"type": name, "days": int(sel.sum()), "mean_rain_mm": float(rain[sel].mean(axis=1).mean())}
        for g in (10, 12):
            rise3 = (s[g].shift(-3) - s[g])[sel]
            rec[f"g{g}_rise3_cm"] = 100 * rise3.mean()
            rec[f"g{g}_p_rise30"] = float((rise3 > 0.3).mean())
        resp.append(rec)
    return cent, pd.DataFrame(resp), ktab


# ---------------------------------------------------------------- U8
EVENT_GAUGES = list(range(1, 14))


def rising_limbs(s: pd.Series, dl: float, g: int) -> list[dict]:
    out = []
    for yr in range(2011, 2026):
        y = s[f"{yr}-05-01":f"{yr}-10-31"]
        sm = y.rolling(3, center=True).mean()
        up = (sm.diff() > 0).to_numpy()
        i, n = 0, len(up)
        while i < n:
            if not up[i]:
                i += 1
                continue
            j = i
            while j + 1 < n and up[j + 1]:
                j += 1
            a, b = y.index[i - 1], y.index[j]
            seg = y[a:b]
            rise = seg.iloc[-1] - seg.iloc[0]
            if j - i + 1 >= 5 and rise >= 0.5 and seg.notna().all():
                out.append({"gauge": g, "year": yr, "start": a, "duration_d": j - i + 2, "total_rise_m": rise,
                            "max_daily_rise_cm": 100 * seg.diff().max(), "start_margin_m": seg.iloc[0] - dl,
                            "start_doy": a.dayofyear, "peak_margin_m": seg.max() - dl})
            i = j + 1
    return out


def u8_events(d: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ev = pd.DataFrame([e for g in EVENT_GAUGES
                       for e in rising_limbs(d["stage"][g], d["danger"][g], g)])
    feats = ["duration_d", "total_rise_m", "max_daily_rise_cm", "start_margin_m", "start_doy"]
    x = StandardScaler().fit_transform(ev[feats])
    k, ktab = choose_k(x, range(2, 7))
    lab = KMeans(k, n_init=50, random_state=cm.SEED).fit_predict(x)
    ev["reached_danger"] = (ev.peak_margin_m >= 0).astype(int)
    order = ev.groupby(lab).reached_danger.mean().sort_values().index
    ev["cluster"] = pd.Series(lab).map({old: new for new, old in enumerate(order)}).to_numpy()
    summ = ev.groupby("cluster").agg(events=("gauge", "size"), duration_d=("duration_d", "median"),
                                     total_rise_m=("total_rise_m", "median"),
                                     max_daily_rise_cm=("max_daily_rise_cm", "median"),
                                     start_margin_m=("start_margin_m", "median"),
                                     start_doy=("start_doy", "median"),
                                     share_danger=("reached_danger", "mean")).reset_index()
    return ev, summ, ktab


# ---------------------------------------------------------------- main
def main() -> None:
    d = load()
    out = cm.RES_DIR
    print("C3"); c3_incremental_trend(d).to_csv(out / "uc_c3_incremental.csv", index=False)
    print("C4"); a, b = c4_rapid_rise(d); a.to_csv(out / "uc_c4_rapid.csv", index=False)
    b.to_csv(out / "uc_c4_probs.csv", index=False)
    print("C5"); c5_logo(d).to_csv(out / "uc_c5_logo.csv", index=False)
    print("C6"); a, b = c6_gauge_type(d); a.to_csv(out / "uc_c6_summary.csv", index=False)
    b.to_csv(out / "uc_c6_gauges.csv", index=False)
    print("C7"); a, b = c7_bridge_era(d); a.to_csv(out / "uc_c7_summary.csv", index=False)
    b.to_csv(out / "uc_c7_confusion.csv")
    print("U4"); a, b, c = u4_hydrograph_shapes(d); a.to_csv(out / "uc_u4_labels.csv", index=False)
    b.to_csv(out / "uc_u4_centroids.csv"); c.to_csv(out / "uc_u4_k.csv", index=False)
    print("U5"); a, b = u5_anomalies(d); a.to_csv(out / "uc_u5_flagged.csv", index=False)
    b.to_csv(out / "uc_u5_summary.csv", index=False)
    print("U6"); a, b = u6_comovement(d); a.to_csv(out / "uc_u6_summary.csv", index=False)
    b.to_csv(out / "uc_u6_labels.csv", index=False)
    print("U7"); a, b, c = u7_rain_types(d); a.to_csv(out / "uc_u7_centroids.csv")
    b.to_csv(out / "uc_u7_response.csv", index=False); c.to_csv(out / "uc_u7_k.csv", index=False)
    print("U8"); a, b, c = u8_events(d); a.to_csv(out / "uc_u8_events.csv", index=False)
    b.to_csv(out / "uc_u8_summary.csv", index=False); c.to_csv(out / "uc_u8_k.csv", index=False)


if __name__ == "__main__":
    main()
