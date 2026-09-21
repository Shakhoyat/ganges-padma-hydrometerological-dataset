"""Step 06 - Random Forest regression and classification, per target and period.

For each target station and each of BEFORE / DURING / AFTER / ALL, a forest is
fitted on the earlier 80% of that stratum and scored once on the later 20%. The
split is chronological because the lag-1 autocorrelation of daily stage on this
corridor is above 0.99: a random split would put a row's near-duplicate
neighbours on both sides of the partition and the test score would be
meaningless.

Every score is reported beside the persistence forecast for the same rows.
Stage tomorrow is very nearly stage today, so R-squared and NSE above 0.99 are
properties of the river, not evidence about a model. The Persistence Index and
the kappa skill score are the numbers that carry information: both are zero for
a model that only matches persistence.

Hyper-parameters are chosen on the training block by out-of-bag score, so the
test block is touched exactly once.

Outputs
    results/model_regression.csv        test metrics, both feature sets
    results/model_classification.csv    test metrics, both feature sets
    results/model_bootstrap.csv         bootstrap CIs for the test metrics
    results/model_hyperparams.csv       the OOB search and the winner
    results/predictions.csv             per-row test predictions
    results/importance_permutation.csv  permutation importance, test block
    results/importance_grouped.csv      importance aggregated by gauge and lag
    results/residual_diagnostics.csv    normality, ACF and heteroscedasticity
    results/transfer_matrix.csv         train on one period, test on another
    results/class_distribution.csv      risk-class balance per stratum
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support)
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.tsa.stattools import acf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
import corridor as C  # noqa: E402

warnings.filterwarnings("ignore")

SEED = 20260913
TEST_FRACTION = 0.20
N_TREES = 200
N_TREES_SEARCH = 60
N_BOOT = 200
N_PERM = 3
GRID = [{"max_features": mf, "min_samples_leaf": leaf}
        for mf in ("sqrt", 0.33) for leaf in (1, 5)]
STRATA = C.PERIOD_NAMES + ["ALL"]
FEATURE_SETS = ("stage_only", "hydromet", "full_multivariate", "forecast_safe")


# ------------------------------------------------------------------- metrics
def regression_metrics(y: np.ndarray, p: np.ndarray, naive: np.ndarray) -> dict:
    err = y - p
    sse = float((err ** 2).sum())
    sst = float(((y - y.mean()) ** 2).sum())
    ssn = float(((y - naive) ** 2).sum())
    r = float(np.corrcoef(y, p)[0, 1]) if y.std() and p.std() else np.nan
    alpha = float(p.std(ddof=1) / y.std(ddof=1)) if y.std(ddof=1) else np.nan
    beta = float(p.mean() / y.mean()) if y.mean() else np.nan
    return {
        "N_Test": int(y.size),
        "RMSE": float(np.sqrt(sse / y.size)),
        "MAE": float(np.abs(err).mean()),
        "MaxAE": float(np.abs(err).max()),
        "R2": 1 - sse / sst if sst else np.nan,
        "NSE": 1 - sse / sst if sst else np.nan,
        "KGE": 1 - float(np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)),
        "PBIAS_pct": 100.0 * float(err.sum() / y.sum()) if y.sum() else np.nan,
        "Pearson_r": r,
        # Persistence Index (Kitanidis & Bras 1980): 0 means exactly as good as
        # repeating yesterday's stage, 1 means perfect.
        "Persistence_Index": 1 - sse / ssn if ssn else np.nan,
        "RMSE_Persistence": float(np.sqrt(ssn / y.size)),
        "NSE_Persistence": 1 - ssn / sst if sst else np.nan,
    }


def classification_metrics(y, p, naive, labels) -> dict:
    k_model = cohen_kappa_score(y, p, labels=labels)
    k_naive = cohen_kappa_score(y, naive, labels=labels)
    pr, rc, f1, sup = precision_recall_fscore_support(
        y, p, labels=labels, zero_division=0)
    out = {
        "N_Test": int(len(y)),
        "Accuracy": accuracy_score(y, p),
        "Macro_F1": f1_score(y, p, labels=labels, average="macro", zero_division=0),
        "Weighted_F1": f1_score(y, p, labels=labels, average="weighted", zero_division=0),
        "Cohen_Kappa": k_model,
        "Accuracy_Persistence": accuracy_score(y, naive),
        "Kappa_Persistence": k_naive,
        # Skill relative to persistence: 0 means no better than the naive rule.
        "Kappa_Skill_Score": ((k_model - k_naive) / (1 - k_naive)
                              if k_naive < 1 else np.nan),
    }
    for lab, a, b, c, n in zip(labels, pr, rc, f1, sup):
        out[f"Precision_{lab}"] = a
        out[f"Recall_{lab}"] = b
        out[f"F1_{lab}"] = c
        out[f"Support_{lab}"] = int(n)
    return out


def bootstrap_ci(y, p, naive, n=N_BOOT, seed=SEED) -> dict:
    """Percentile CIs for the headline test metrics, resampling test rows."""
    rng = np.random.default_rng(seed)
    keep = {"RMSE": [], "MAE": [], "NSE": [], "KGE": [], "Persistence_Index": []}
    for _ in range(n):
        i = rng.integers(0, y.size, y.size)
        m = regression_metrics(y[i], p[i], naive[i])
        for k in keep:
            keep[k].append(m[k])
    out = {}
    for k, v in keep.items():
        lo, hi = np.percentile(v, [2.5, 97.5])
        out[f"{k}_CI_low"], out[f"{k}_CI_high"] = float(lo), float(hi)
    return out


# --------------------------------------------------------------------- model
# <<snip:split>>
def chronological_split(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = d.sort_values("Date")
    cut = int(round(len(d) * (1 - TEST_FRACTION)))
    return d.iloc[:cut], d.iloc[cut:]  # <<endsnip>>


def pick_hyperparams(X, y, is_clf: bool, seed=SEED):
    """Choose by out-of-bag score on the training block only."""
    rows, best, best_score = [], GRID[0], -np.inf
    for g in GRID:
        Model = RandomForestClassifier if is_clf else RandomForestRegressor
        m = Model(n_estimators=N_TREES_SEARCH, oob_score=True, bootstrap=True,
                  random_state=seed, n_jobs=-1, **g)
        m.fit(X, y)
        s = float(m.oob_score_)
        rows.append({**g, "OOB_Score": s})
        if s > best_score:
            best, best_score = g, s
    return best, best_score, rows


def dump(reg_rows, clf_rows, boot_rows, hp_rows, pred_rows, imp_rows,
         resid_rows, transfer_rows, classdist_rows, cm_rows) -> None:
    """Write every result table accumulated so far."""
    for rows, name in ((reg_rows, "model_regression"),
                       (clf_rows, "model_classification"),
                       (boot_rows, "model_bootstrap"),
                       (hp_rows, "model_hyperparams"),
                       (pred_rows, "predictions"),
                       (imp_rows, "importance_permutation"),
                       (resid_rows, "residual_diagnostics"),
                       (transfer_rows, "transfer_matrix"),
                       (classdist_rows, "class_distribution"),
                       (cm_rows, "confusion_matrices")):
        pd.DataFrame(rows).to_csv(C.RESULTS / f"{name}.csv", index=False)

    # Grouped importance: with predictors correlated above 0.99, a single
    # feature's importance is diluted across its near-duplicates. Summing by
    # gauge and by lag is the readable view.
    imp = pd.DataFrame(imp_rows)
    if imp.empty:
        return
    out = []
    for key, name in (("Source_Point_Id", "by_point"), ("Lag_Days", "by_lag")):
        g = (imp.groupby(["Station_ID", "Stratum", "Feature_Set", key])
                .Permutation_Importance.sum().reset_index())
        g["Grouping"] = name
        out.append(g.rename(columns={key: "Group_Value"}))
    pd.concat(out, ignore_index=True).to_csv(
        C.RESULTS / "importance_grouped.csv", index=False)


def main() -> None:
    sets = json.loads((C.WIDE / "feature_sets.json").read_text(encoding="utf-8"))
    reg_tbl = pd.read_csv(C.DATA / "station_registry.csv").set_index("Station_ID")

    reg_rows, clf_rows, boot_rows, hp_rows = [], [], [], []
    pred_rows, imp_rows, resid_rows = [], [], []
    transfer_rows, classdist_rows, cm_rows = [], [], []

    for tgt in C.TARGETS:
        wide = pd.read_csv(C.WIDE / f"wide_{tgt.replace('.', '_')}.csv",
                           parse_dates=["Date"])
        dl = float(reg_tbl.at[tgt, "Danger_Level_mMSL"])
        tid = C.STATION_ID[tgt]

        for stratum in STRATA:
            d = wide if stratum == "ALL" else wide[wide.Period == stratum]
            if len(d) < 200:
                continue
            tr, te = chronological_split(d)

            for lab in C.RISK_LABELS:
                classdist_rows.append({
                    "Station_ID": tgt, "Id": tid, "Stratum": stratum, "Class": lab,
                    "N_Train": int((tr.Risk_Class == lab).sum()),
                    "N_Test": int((te.Risk_Class == lab).sum()),
                    "Pct_All": round(100 * (d.Risk_Class == lab).mean(), 2)})

            for fs in FEATURE_SETS:
                print(f"    {tgt} {stratum} {fs} ...", flush=True)
                feats = sets[tgt][fs]
                Xtr, Xte = tr[feats].to_numpy(), te[feats].to_numpy()
                ytr, yte = tr.WL.to_numpy(), te.WL.to_numpy()
                naive = te["WLD-1"].to_numpy()

                # ---------------------------------------------- regression
                g, oob, search = pick_hyperparams(Xtr, ytr, is_clf=False)
                for s in search:
                    hp_rows.append({"Station_ID": tgt, "Stratum": stratum,
                                    "Feature_Set": fs, "Task": "regression",
                                    **s, "Selected": s["max_features"] == g["max_features"]
                                    and s["min_samples_leaf"] == g["min_samples_leaf"]})
                rf = RandomForestRegressor(n_estimators=N_TREES, oob_score=True,
                                           random_state=SEED, n_jobs=-1, **g)
                rf.fit(Xtr, ytr)
                pred = rf.predict(Xte)

                m = regression_metrics(yte, pred, naive)
                reg_rows.append({"Station_ID": tgt, "Id": tid, "Stratum": stratum,
                                 "Feature_Set": fs, "N_Features": len(feats),
                                 "N_Train": len(tr), **m, "OOB_R2": float(rf.oob_score_),
                                 "Train_Start": tr.Date.min().date(),
                                 "Train_End": tr.Date.max().date(),
                                 "Test_Start": te.Date.min().date(),
                                 "Test_End": te.Date.max().date(),
                                 **{f"HP_{k}": v for k, v in g.items()}})
                boot_rows.append({"Station_ID": tgt, "Stratum": stratum,
                                  "Feature_Set": fs, **bootstrap_ci(yte, pred, naive)})

                # --------------------------------------------- residuals
                res = yte - pred
                bp_p = np.nan
                try:
                    Z = np.column_stack([np.ones_like(pred), pred])
                    bp_p = float(het_breuschpagan(res, Z)[1])
                except Exception:
                    pass
                ra = acf(res, nlags=min(7, len(res) - 2), fft=True)
                resid_rows.append({
                    "Station_ID": tgt, "Id": tid, "Stratum": stratum, "Feature_Set": fs,
                    "Mean_Residual": float(res.mean()), "SD_Residual": float(res.std(ddof=1)),
                    "Skew_Residual": float(sps.skew(res)),
                    "Kurtosis_Residual": float(sps.kurtosis(res)),
                    "Shapiro_p": float(sps.shapiro(res[:5000]).pvalue),
                    "JarqueBera_p": float(sps.jarque_bera(res).pvalue),
                    "BreuschPagan_p": bp_p,
                    **{f"Residual_ACF_lag{k}": float(ra[k])
                       for k in range(1, min(8, len(ra)))}})

                for dte, a, b, c in zip(te.Date, yte, pred, naive):
                    pred_rows.append({"Station_ID": tgt, "Stratum": stratum,
                                      "Feature_Set": fs, "Date": dte.date(),
                                      "Observed": a, "Predicted": b, "Persistence": c})

                # -------------------------------------------- importance
                if stratum == "ALL" and fs in ("stage_only", "hydromet"):
                    try:
                        pi = permutation_importance(
                            rf, Xte, yte, n_repeats=N_PERM, random_state=SEED,
                            n_jobs=-1, scoring="neg_root_mean_squared_error")
                        perm_mean, perm_sd = pi.importances_mean, pi.importances_std
                    except Exception as e:
                        print(f"      permutation importance failed: {e}", flush=True)
                        perm_mean = perm_sd = np.full(len(feats), np.nan)
                else:
                    perm_mean = perm_sd = np.full(len(feats), np.nan)

                for name, mean, sd, imp in zip(feats, perm_mean, perm_sd,
                                               rf.feature_importances_):
                    if name.startswith("P"):
                        pid = int(name[1:3])
                        lagpart = name.split("_", 1)[1]
                        if "Rain_D" in lagpart:
                            lag = int(lagpart.split("Rain_D")[1])
                        elif "WLD-" in lagpart:
                            lag = int(lagpart.split("WLD-")[1])
                        else:
                            lag = 0
                    elif name.startswith("Q_"):
                        pid = tid
                        lag = int(name.split("_D")[1]) if "_D" in name else 0
                    else:
                        pid = tid
                        if "WLD-" in name:
                            lag = int(name.split("WLD-")[1])
                        elif "Rain_D" in name:
                            lag = int(name.split("Rain_D")[1])
                        else:
                            lag = 0

                    imp_rows.append({
                        "Station_ID": tgt, "Stratum": stratum, "Feature_Set": fs,
                        "Feature": name, "Source_Point_Id": pid, "Lag_Days": lag,
                        "Permutation_Importance": float(mean),
                        "Permutation_SD": float(sd), "Impurity_Importance": float(imp)})

                # ------------------------------------- cross-period transfer
                if stratum in C.PERIOD_NAMES and fs == "hydromet":
                    for other in C.PERIOD_NAMES:
                        o = wide[wide.Period == other]
                        if other == stratum:
                            o = te
                        if len(o) < 50:
                            continue
                        pm = regression_metrics(o.WL.to_numpy(),
                                                rf.predict(o[feats].to_numpy()),
                                                o["WLD-1"].to_numpy())
                        transfer_rows.append({
                            "Station_ID": tgt, "Id": tid, "Train_Period": stratum,
                            "Test_Period": other, "Same_Period": other == stratum,
                            **{k: pm[k] for k in ("N_Test", "RMSE", "MAE", "NSE",
                                                  "KGE", "Persistence_Index")}})

                # ------------------------------------------ classification
                ctr = tr.Risk_Class.astype(str).to_numpy()
                cte = te.Risk_Class.astype(str).to_numpy()
                labels = [l for l in C.RISK_LABELS
                          if l in set(ctr) | set(cte)]
                if len(set(ctr)) < 2:
                    continue
                gc, oobc, searchc = pick_hyperparams(Xtr, ctr, is_clf=True)
                for s in searchc:
                    hp_rows.append({"Station_ID": tgt, "Stratum": stratum,
                                    "Feature_Set": fs, "Task": "classification",
                                    **s,
                                    "Selected": s["max_features"] == gc["max_features"]
                                    and s["min_samples_leaf"] == gc["min_samples_leaf"]})
                rc = RandomForestClassifier(n_estimators=N_TREES, oob_score=True,
                                            class_weight="balanced_subsample",
                                            random_state=SEED, n_jobs=-1, **gc)
                rc.fit(Xtr, ctr)
                cpred = rc.predict(Xte)
                cnaive = C.risk_class(te["WLD-1"], dl).astype(str).to_numpy()

                cm = classification_metrics(cte, cpred, cnaive, labels)
                clf_rows.append({"Station_ID": tgt, "Id": tid, "Stratum": stratum,
                                 "Feature_Set": fs, "N_Features": len(feats),
                                 "N_Train": len(tr), "N_Classes": len(labels),
                                 **cm, "OOB_Accuracy": float(rc.oob_score_),
                                 **{f"HP_{k}": v for k, v in gc.items()}})
                M = confusion_matrix(cte, cpred, labels=labels)
                for i, t in enumerate(labels):
                    for j, pl in enumerate(labels):
                        cm_rows.append({"Station_ID": tgt, "Stratum": stratum,
                                        "Feature_Set": fs, "True": t, "Predicted": pl,
                                        "Count": int(M[i, j])})
            print(f"  {tgt:<9} {stratum:<7} done  "
                  f"({datetime.now():%H:%M:%S})", flush=True)

        dump(reg_rows, clf_rows, boot_rows, hp_rows, pred_rows, imp_rows,
             resid_rows, transfer_rows, classdist_rows, cm_rows)

    reg_out = pd.DataFrame(reg_rows)
    print(f"\n  regression runs     {len(reg_rows)}")
    print(f"  classification runs {len(clf_rows)}")
    hm = reg_out[(reg_out.Feature_Set == "hydromet")]
    print("\n  regression, hydromet feature set (+Rainfall & Discharge):")
    print(f"  {'station':<9}{'stratum':<8}{'RMSE':>8}{'RMSE_p':>9}{'NSE':>8}"
          f"{'PI':>8}{'KGE':>8}")
    for _, r in hm.iterrows():
        print(f"  {r.Station_ID:<9}{r.Stratum:<8}{r.RMSE:>8.4f}{r.RMSE_Persistence:>9.4f}"
              f"{r.NSE:>8.4f}{r.Persistence_Index:>8.3f}{r.KGE:>8.4f}")


if __name__ == "__main__":
    main()

