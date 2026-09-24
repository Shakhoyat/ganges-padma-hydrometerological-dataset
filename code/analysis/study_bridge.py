"""Study B: what changed around the Padma Bridge (regression + classification).

B1  Water-surface drop across the bridge reach, by construction era and
    season, at matched upstream flow (Goalundo stage tercile):
    Baruria-Bhagyakul (56 -> 5 km above), Bhagyakul-Mawa (bridge site) and
    Mawa-Sureswar (31 km below). A larger drop just upstream at the same
    flow is what backwater (afflux) from piers and river training would show.
B2  Probability that the bridge-site gauges reach the warning level
    (DL - 1 m) as a logistic function of the upstream Goalundo margin, fitted
    separately per era: the Goalundo margin at which the bridge reach has a
    50% chance of warning, before vs after.
B3  Cross-era transfer: ridge nowcast of the daily change trained on one era
    and tested on another (released wide matrices), persistence index.

Gauge data alone cannot separate the bridge from natural channel change or
from year-to-year flood differences; every result is phrased as a change in
the record, not as a causal bridge effect.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import common as cm

ERAS = {"Before": (pd.Timestamp("2011-01-01"), pd.Timestamp("2014-11-25")),
        "During": (pd.Timestamp("2014-11-26"), pd.Timestamp("2022-06-25")),
        "After": (pd.Timestamp("2022-06-26"), pd.Timestamp("2025-12-31"))}
REACHES = [(11, 12), (12, 13), (13, 15)]
SEASONS = {"Monsoon (Jul-Oct)": [7, 8, 9, 10], "Dry (Jan-Apr)": [1, 2, 3, 4]}
BLOCK = 30


def era_of(index: pd.DatetimeIndex) -> pd.Series:
    out = pd.Series(index=index, dtype=object)
    for name, (a, b) in ERAS.items():
        out[(index >= a) & (index <= b)] = name
    return out


def block_boot_mean(x: np.ndarray, rng: np.random.Generator, n: int = 2000) -> tuple[float, float]:
    x = x[~np.isnan(x)]
    if len(x) < BLOCK * 2:
        return np.nan, np.nan
    nb = int(np.ceil(len(x) / BLOCK))
    starts = rng.integers(0, len(x) - BLOCK, size=(n, nb))
    means = np.array([np.concatenate([x[s:s + BLOCK] for s in row])[:len(x)].mean() for row in starts])
    return tuple(np.percentile(means, [2.5, 97.5]))


def reach_drop_table(stage: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(cm.SEED)
    era = era_of(stage.index)
    rows = []
    for season, months in SEASONS.items():
        in_season = stage.index.month.isin(months)
        gq = stage.loc[in_season, 10]
        terciles = gq.quantile([1 / 3, 2 / 3]).to_numpy()
        flow = pd.cut(stage[10], [-np.inf, *terciles, np.inf], labels=["Low", "Mid", "High"])
        for up, dn in REACHES:
            drop = stage[up] - stage[dn]
            for e in ERAS:
                for fl in ["All", "Low", "Mid", "High"]:
                    sel = in_season & (era == e).to_numpy()
                    if fl != "All":
                        sel &= (flow == fl).to_numpy()
                    x = drop[sel].to_numpy()
                    lo, hi = block_boot_mean(x, rng)
                    rows.append({"season": season, "reach": f"{up}-{dn}", "era": e, "flow": fl,
                                 "n": int(np.sum(~np.isnan(x))), "mean_m": np.nanmean(x),
                                 "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def warning_probability(stage: pd.DataFrame, danger: pd.Series) -> pd.DataFrame:
    era = era_of(stage.index)
    x_all = (stage[10] - danger[10]).to_frame("goalundo")
    rows, curves = [], []
    grid = np.linspace(-3, 2, 101)
    for tgt in (12, 13):
        y_all = (stage[tgt] >= danger[tgt] - 1.0).astype(float).where(stage[tgt].notna())
        for e in ERAS:
            ok = (era == e).to_numpy() & y_all.notna().to_numpy() & x_all["goalundo"].notna().to_numpy()
            x, y = x_all[ok], y_all[ok].astype(int)
            model = LogisticRegression(C=100.0).fit(x, y)
            b0, b1 = model.intercept_[0], model.coef_[0, 0]
            p = model.predict_proba(pd.DataFrame({"goalundo": grid}))[:, 1]
            curves.append(pd.DataFrame({"target": tgt, "era": e, "goalundo_margin": grid, "p": p}))
            rows.append({"target": tgt, "era": e, "n": len(y), "x50_m": -b0 / b1,
                         "warn_share": y.mean()})
    return pd.DataFrame(rows), pd.concat(curves)


def transfer_matrix() -> pd.DataFrame:
    rows = []
    for tgt in cm.TARGETS:
        w = cm.load_wide(tgt)
        feats = w.drop(columns=["Id", "Date", "WL"])
        dy = w["WL"] - w["WLD-1"]
        era = era_of(pd.DatetimeIndex(w["Date"])).to_numpy()
        for a in ERAS:
            for b in ERAS:
                if a == b:  # chronological 70/30 inside the era
                    idx = np.where(era == a)[0]
                    cut = int(len(idx) * 0.7)
                    tr, te = idx[:cut], idx[cut:]
                else:
                    tr, te = np.where(era == a)[0], np.where(era == b)[0]
                model = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(feats.iloc[tr], dy.iloc[tr])
                err = dy.iloc[te].to_numpy() - model.predict(feats.iloc[te])
                pi = 1 - np.sum(err ** 2) / np.sum(dy.iloc[te].to_numpy() ** 2)
                rows.append({"target": tgt, "train": a, "test": b, "n_train": len(tr),
                             "n_test": len(te), "PI": pi, "rmse_cm": 100 * np.sqrt(np.mean(err ** 2))})
    return pd.DataFrame(rows)


def main() -> None:
    danger = cm.load_registry()["Danger_Level_mMSL"]
    stage = cm.stage_matrix(cm.load_long())
    reach_drop_table(stage).to_csv(cm.RES_DIR / "bridge_reach_drop.csv", index=False)
    table, curves = warning_probability(stage, danger)
    table.to_csv(cm.RES_DIR / "bridge_warning_x50.csv", index=False)
    curves.to_csv(cm.RES_DIR / "bridge_warning_curves.csv", index=False)
    transfer_matrix().to_csv(cm.RES_DIR / "bridge_transfer.csv", index=False)
    print("bridge done")


if __name__ == "__main__":
    main()
