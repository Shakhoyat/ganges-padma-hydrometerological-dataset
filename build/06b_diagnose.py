"""Step 06b - why the forest loses to persistence, and the target that fixes it.

The headline regression predicts stage itself, and on most strata it scores
below the naive rule. Two things are checked here.

1. Extrapolation. A forest predicts the mean of a training leaf, so it cannot
   return a value outside the range it was trained on. With a chronological
   split on a series that wanders, part of the test block routinely lies
   outside the training range. Persistence has no such ceiling. This step
   measures how much of each test block is out of range, and how much of the
   error is concentrated there.

2. The change target. Predicting the daily change and adding it back to
   yesterday's stage removes the level from the problem, so the forest no
   longer has to reproduce a number it has never seen. Same features, same
   split, same forest - only the target differs.

Outputs
    results/extrapolation.csv    train/test range overlap and where error sits
    results/model_delta.csv      the change-target regression, beside the level
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
import corridor as C  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module  # noqa: E402

M = import_module("06_models")
warnings.filterwarnings("ignore")


def main() -> None:
    sets = json.loads((C.WIDE / "feature_sets.json").read_text(encoding="utf-8"))
    extrap, delta = [], []

    for tgt in C.TARGETS:
        wide = pd.read_csv(C.WIDE / f"wide_{tgt.replace('.', '_')}.csv",
                           parse_dates=["Date"])
        tid = C.STATION_ID[tgt]
        for stratum in C.PERIOD_NAMES + ["ALL"]:
            d = wide if stratum == "ALL" else wide[wide.Period == stratum]
            if len(d) < 200:
                continue
            tr, te = M.chronological_split(d)
            feats = sets[tgt].get("hydromet", sets[tgt].get("spec"))
            ytr, yte = tr.WL.to_numpy(), te.WL.to_numpy()
            naive = te["WLD-1"].to_numpy()

            # ---- 1. the level target, tuned exactly as the change target is,
            #         so the two differ only in what is being predicted
            Xtr, Xte = tr[feats].to_numpy(), te[feats].to_numpy()
            gl, _, _ = M.pick_hyperparams(Xtr, ytr, is_clf=False)
            rf = RandomForestRegressor(n_estimators=M.N_TREES, random_state=M.SEED,
                                       n_jobs=-1, **gl)
            rf.fit(Xtr, ytr)
            pred = rf.predict(Xte)
            err = (yte - pred) ** 2

            # how much of the test block lies outside the training range at all
            lo, hi = ytr.min(), ytr.max()
            out = (yte < lo) | (yte > hi)
            step = np.abs(np.diff(yte)).mean()      # typical day-to-day move
            extrap.append({
                "Station_ID": tgt, "Id": tid, "Stratum": stratum,
                "Train_Min": lo, "Train_Max": hi,
                "Test_Min": yte.min(), "Test_Max": yte.max(),
                "N_Test": len(yte), "N_Outside_Train_Range": int(out.sum()),
                "Pct_Outside": round(100 * out.mean(), 2),
                "Pct_SSE_From_Outside": round(100 * err[out].sum() / err.sum(), 2)
                if out.any() else 0.0,
                "Max_Pred": float(pred.max()), "Max_Observed": float(yte.max()),
                "Ceiling_Gap": float(yte.max() - pred.max()),
                # the comparison that actually explains the failure: the error a
                # level model makes against the size of the move persistence
                # already gets for free
                "Mean_Abs_Daily_Change": float(step),
                "RMSE_Level": float(np.sqrt(err.mean())),
                "Error_vs_Daily_Move": float(np.sqrt(err.mean()) / step),
            })

            # ---- 2. the change target, identical in every other respect
            dtr = ytr - tr["WLD-1"].to_numpy()
            g, _, _ = M.pick_hyperparams(Xtr, dtr, is_clf=False)
            rfd = RandomForestRegressor(n_estimators=M.N_TREES, oob_score=True,
                                        random_state=M.SEED, n_jobs=-1, **g)
            rfd.fit(Xtr, dtr)
            pred_d = naive + rfd.predict(Xte)

            level = M.regression_metrics(yte, pred, naive)
            chg = M.regression_metrics(yte, pred_d, naive)
            delta.append({
                "Station_ID": tgt, "Id": tid, "Stratum": stratum,
                "N_Train": len(tr), "N_Test": len(te),
                "RMSE_Level": level["RMSE"], "RMSE_Change": chg["RMSE"],
                "RMSE_Persistence": level["RMSE_Persistence"],
                "PI_Level": level["Persistence_Index"],
                "PI_Change": chg["Persistence_Index"],
                "NSE_Level": level["NSE"], "NSE_Change": chg["NSE"],
                "KGE_Change": chg["KGE"], "MAE_Change": chg["MAE"],
                "PBIAS_Change": chg["PBIAS_pct"],
                **{f"HP_{k}": v for k, v in g.items()},
            })
            print(f"  {tgt:<9} {stratum:<7} outside={out.mean()*100:5.1f}%  "
                  f"PI level={level['Persistence_Index']:+.3f}  "
                  f"PI change={chg['Persistence_Index']:+.3f}", flush=True)

    pd.DataFrame(extrap).to_csv(C.RESULTS / "extrapolation.csv", index=False)
    dl = pd.DataFrame(delta)
    dl.to_csv(C.RESULTS / "model_delta.csv", index=False)
    print(f"\n  mean PI, level target  {dl.PI_Level.mean():+.3f}")
    print(f"  mean PI, change target {dl.PI_Change.mean():+.3f}")
    print(f"  strata where change beats persistence: "
          f"{int((dl.PI_Change > 0).sum())} / {len(dl)}")


if __name__ == "__main__":
    main()
