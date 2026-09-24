"""Study I1-I3: flood-risk early warning from the released data (CPU only).

I1  4-class flood-risk classification (Normal / Warning / Danger / Severe
    relative to each gauge's BWDB danger level; the backup_v2 labels) h days
    ahead, h = 1..7, at the three release targets.
I2  Lead-time skill: how many days ahead a Danger-or-worse day can be warned
    (POD / FAR / CSI), and how the stage-forecast error grows with lead time.
I3  Upstream rule of thumb: when an upstream gauge crosses its danger level,
    how often and how many days later does the target follow?

Evaluation is rolling-origin by calendar year: for each test year Y in
2016..2025 every model is refitted on the days whose *label* date falls before
1 January Y. No test-year information reaches any fit; hyperparameters are
fixed a priori (no tuning), so no validation split is needed.

Features are built from the long panel (not the wide matrices) because the
wide matrices keep complete rows only and so lose 51% of 2020, the record
flood year. Tree models handle the remaining gaps natively; logistic
regression and ridge impute with training-period medians.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import common as cm

HORIZONS = [1, 2, 3, 5, 7]
TEST_YEARS = list(range(2016, 2026))
# Predictor gauges per target: target first, then upstream, then Meghna
# boundary. Tarpasa (14) is excluded as in the release (1,369-day gap).
PREDICTORS = {
    11: [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 17, 16],
    12: [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 17, 16],
    15: [15, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 17, 16],
}
UPSTREAM_ALARMS = [4, 8, 9, 10]  # Hardinge Br., Bahadurabad, Aricha, Goalundo


def feature_table(stage: pd.DataFrame, danger: pd.Series, gauges: list[int]) -> pd.DataFrame:
    """Issue-day features: margin to danger level and 1/3/7-day changes per gauge."""
    cols = {}
    for g in gauges:
        s = stage[g]
        cols[f"m{g}"] = s - danger[g]
        cols[f"d1_{g}"] = s - s.shift(1)
        cols[f"d3_{g}"] = s - s.shift(3)
        cols[f"d7_{g}"] = s - s.shift(7)
    doy = stage.index.dayofyear
    cols["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    cols["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return pd.DataFrame(cols, index=stage.index)


def classifiers() -> dict[str, object]:
    return {
        "Logistic regression": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000)),
        "Random forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample",
            n_jobs=-1, random_state=cm.SEED),
        "Gradient boosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
            class_weight="balanced", random_state=cm.SEED),
    }


def _fit_predict(model, x_tr, y_tr, x_te) -> np.ndarray:
    classes = np.unique(y_tr)
    if len(classes) == 1:
        return np.full(len(x_te), classes[0])
    model.fit(x_tr, y_tr)
    return model.predict(x_te)


def run_classification(stage: pd.DataFrame, danger: pd.Series) -> pd.DataFrame:
    rows = []
    for tgt, gauges in PREDICTORS.items():
        feats = feature_table(stage, danger, gauges)
        margin = stage[tgt] - danger[tgt]
        for h in HORIZONS:
            y = pd.Series(cm.risk_class(margin.shift(-h)), index=stage.index)
            label_date = stage.index + pd.Timedelta(days=h)
            ok = y.notna() & margin.notna()
            for year in TEST_YEARS:
                tr = ok & (label_date.year < year)
                te = ok & (label_date.year == year)
                persist = cm.risk_class(margin[te])
                base = pd.DataFrame({"target": tgt, "h": h, "year": year,
                                     "date": stage.index[te], "y": y[te].to_numpy(),
                                     "Persistence": persist})
                for name, model in classifiers().items():
                    base[name] = _fit_predict(model, feats[tr], y[tr].astype(int), feats[te])
                rows.append(base)
            print(f"    classification target {tgt} h={h} done")
    return pd.concat(rows, ignore_index=True)


def binary_scores(y: np.ndarray, p: np.ndarray, level: int) -> dict[str, float]:
    obs, fc = y >= level, p >= level
    hits = np.sum(obs & fc)
    miss = np.sum(obs & ~fc)
    false = np.sum(~obs & fc)
    denom = hits + miss + false
    return {"POD": hits / max(hits + miss, 1), "FAR": false / max(hits + false, 1),
            "CSI": hits / denom if denom else np.nan, "n_event": int(obs.sum())}


def score_classification(pred: pd.DataFrame) -> pd.DataFrame:
    models = ["Persistence", "Logistic regression", "Random forest", "Gradient boosting"]
    out = []
    for (tgt, h), g in pred.groupby(["target", "h"]):
        y = g["y"].to_numpy().astype(int)
        for m in models:
            p = g[m].to_numpy().astype(int)
            rec = {"target": tgt, "h": h, "model": m,
                   "macroF1": f1_score(y, p, average="macro", labels=[0, 1, 2, 3], zero_division=0),
                   "kappa": cohen_kappa_score(y, p, weights="linear")}
            for lvl, tag in ((1, "W"), (2, "D")):
                rec.update({f"{k}_{tag}": v for k, v in binary_scores(y, p, lvl).items()})
            out.append(rec)
    return pd.DataFrame(out)


def run_regression(stage: pd.DataFrame, danger: pd.Series) -> pd.DataFrame:
    """Stage forecast h days ahead; ridge and persistence, rolling origin."""
    out = []
    for tgt, gauges in PREDICTORS.items():
        feats = feature_table(stage, danger, gauges)
        s = stage[tgt]
        for h in HORIZONS:
            dy = s.shift(-h) - s
            label_date = stage.index + pd.Timedelta(days=h)
            ok = dy.notna()
            err_m, err_p, err_h = [], [], []
            for year in TEST_YEARS:
                tr, te = ok & (label_date.year < year), ok & (label_date.year == year)
                model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                      Ridge(alpha=10.0))
                model.fit(feats[tr], dy[tr])
                err_m.append(dy[te].to_numpy() - model.predict(feats[te]))
                err_p.append(dy[te].to_numpy())
                hgb = HistGradientBoostingRegressorLite()
                err_h.append(dy[te].to_numpy() - hgb.fit(feats[tr], dy[tr]).predict(feats[te]))
            em, ep, eh = (np.concatenate(e) for e in (err_m, err_p, err_h))
            out.append({"target": tgt, "h": h,
                        "rmse_persist_cm": 100 * np.sqrt(np.mean(ep ** 2)),
                        "rmse_ridge_cm": 100 * np.sqrt(np.mean(em ** 2)),
                        "rmse_hgb_cm": 100 * np.sqrt(np.mean(eh ** 2)),
                        "PI_ridge": 1 - np.sum(em ** 2) / np.sum(ep ** 2),
                        "PI_hgb": 1 - np.sum(eh ** 2) / np.sum(ep ** 2)})
        print(f"    regression target {tgt} done")
    return pd.DataFrame(out)


def HistGradientBoostingRegressorLite():  # noqa: N802 - factory mirrors class name
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                         max_leaf_nodes=15, random_state=cm.SEED)


def _onsets(margin: pd.Series, quiet_days: int = 7) -> pd.DatetimeIndex:
    """Days a gauge crosses its danger level after >= quiet_days below it."""
    above = margin >= 0
    observed_below = (margin < 0).astype(int)  # NaN (gap) is NOT counted as quiet
    below_before = observed_below.rolling(quiet_days).min().shift(1) == 1
    return margin.index[above & below_before]


def upstream_alarm_table(stage: pd.DataFrame, danger: pd.Series, window: int = 15) -> pd.DataFrame:
    """For each upstream alarm gauge and target: hit rate and lead time."""
    out = []
    for tgt in cm.TARGETS:
        m_t = stage[tgt] - danger[tgt]
        for up in UPSTREAM_ALARMS:
            m_u = stage[up] - danger[up]
            leads = []
            n = 0
            for d in _onsets(m_u):
                seg = m_t.loc[d: d + pd.Timedelta(days=window)]
                if m_t.get(d, np.nan) >= 0 or seg.isna().all():
                    continue  # target already in danger, or unobserved
                n += 1
                hit = seg[seg >= 0]
                if len(hit):
                    leads.append((hit.index[0] - d).days)
            leads = np.array(leads)
            out.append({"target": tgt, "upstream": up, "n_alarms": n, "n_followed": len(leads),
                        "hit_rate": len(leads) / n if n else np.nan,
                        "lead_median": float(np.median(leads)) if len(leads) else np.nan,
                        "lead_q25": float(np.percentile(leads, 25)) if len(leads) else np.nan,
                        "lead_q75": float(np.percentile(leads, 75)) if len(leads) else np.nan})
    return pd.DataFrame(out)


def main() -> None:
    reg = cm.load_registry()
    danger = reg["Danger_Level_mMSL"]
    stage = cm.stage_matrix(cm.load_long())
    pred = run_classification(stage, danger)
    pred.to_csv(cm.RES_DIR / "risk_predictions.csv", index=False)
    score_classification(pred).to_csv(cm.RES_DIR / "risk_scores.csv", index=False)
    run_regression(stage, danger).to_csv(cm.RES_DIR / "leadtime_regression.csv", index=False)
    upstream_alarm_table(stage, danger).to_csv(cm.RES_DIR / "upstream_alarms.csv", index=False)


if __name__ == "__main__":
    main()
