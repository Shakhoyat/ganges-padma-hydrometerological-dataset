"""Study D: how much data is enough? (regression, CPU only)

D1  Points x history grid: test persistence index of the daily-change nowcast
    when the model sees only the n nearest upstream gauges (n = 0 .. all,
    then + the two Meghna boundary gauges) and only the last L years of
    training history (L = 0.5 .. 11 y, all ending 31 Dec 2021). Validation
    2022 picks the ridge penalty; test 2023-2025 is never used for a choice.
D2  The same grid without any same-day level (true 1-day-ahead forecast).
D3  Test-window length: how widely the persistence index of one fixed model
    scatters when it is measured on short windows (1 week .. 2 years) cut
    from the test period, i.e. how long a test must be before it can be trusted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import common as cm

NEAREST_FIRST = {
    11: [10, 9, 7, 6, 8, 5, 4, 3, 2, 1],
    12: [11, 10, 9, 7, 6, 8, 5, 4, 3, 2, 1],
    15: [13, 12, 11, 10, 9, 7, 6, 8, 5, 4, 3, 2, 1],
}
N_POINTS = [0, 1, 2, 3, 4, 6, 8, "all", "all+B"]
HISTORY_YEARS = [0.5, 1, 2, 3, 5, 8, 11]
ALPHAS = [0.1, 1, 10, 100, 1000]
WINDOWS = [7, 14, 30, 60, 90, 180, 365, 730]


def columns_for(w: pd.DataFrame, target: int, n: int | str, same_day: bool) -> list[str]:
    own = [f"WLD-{k}" for k in range(1, 8)]
    ups = NEAREST_FIRST[target]
    chosen = ups if n in ("all", "all+B") else ups[:n]
    prefixes = [f"P{p:02d}_" for p in chosen] + (["B16_", "B17_"] if n == "all+B" else [])
    cols = own + [c for c in w.columns if any(c.startswith(p) for p in prefixes)]
    if not same_day:
        cols = [c for c in cols if not c.endswith("_WL")]
    return cols


def fit_ridge(x_tr, y_tr, x_va, y_va):
    best = None
    for a in ALPHAS:
        m = make_pipeline(StandardScaler(), Ridge(alpha=a)).fit(x_tr, y_tr)
        rmse = np.sqrt(np.mean((y_va - m.predict(x_va)) ** 2))
        if best is None or rmse < best[0]:
            best = (rmse, m)
    return best[1]


def grid(target: int, same_day: bool) -> list[dict]:
    w = cm.load_wide(target)
    dy = (w["WL"] - w["WLD-1"]).to_numpy()
    date = w["Date"]
    va = (date.dt.year == cm.VAL_YEAR).to_numpy()
    te = (date >= cm.TEST_START).to_numpy()
    rows = []
    for n in N_POINTS:
        cols = columns_for(w, target, n, same_day)
        x = w[cols].to_numpy()
        for years in HISTORY_YEARS:
            start = cm.TRAIN_END - pd.DateOffset(months=int(years * 12)) + pd.Timedelta(days=1)
            tr = ((date >= start) & (date <= cm.TRAIN_END)).to_numpy()
            ridge = fit_ridge(x[tr], dy[tr], x[va], dy[va])
            hgb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                                min_samples_leaf=10, random_state=cm.SEED)
            hgb.fit(x[tr], dy[tr])
            for name, model in (("Ridge", ridge), ("Gradient boosting", hgb)):
                err = dy[te] - model.predict(x[te])
                rows.append({"target": target, "same_day": same_day, "n_points": str(n),
                             "n_features": len(cols), "history_years": years,
                             "n_train": int(tr.sum()), "model": name,
                             "PI": 1 - np.sum(err ** 2) / np.sum(dy[te] ** 2),
                             "rmse_cm": 100 * np.sqrt(np.mean(err ** 2))})
    print(f"    sufficiency grid target {target} same_day={same_day} done")
    return rows


def window_scatter(target: int) -> list[dict]:
    """PI of the full-data ridge on every short window cut from the test years."""
    w = cm.load_wide(target)
    cols = columns_for(w, target, "all+B", True)
    dy = (w["WL"] - w["WLD-1"]).to_numpy()
    date = w["Date"]
    tr = (date <= cm.TRAIN_END).to_numpy()
    va = (date.dt.year == cm.VAL_YEAR).to_numpy()
    te = (date >= cm.TEST_START).to_numpy()
    model = fit_ridge(w.loc[tr, cols], dy[tr], w.loc[va, cols], dy[va])
    err = dy[te] - model.predict(w.loc[te, cols])
    base = dy[te]
    tdate = date[te].reset_index(drop=True)
    full_pi = 1 - np.sum(err ** 2) / np.sum(base ** 2)
    rows = []
    for length in WINDOWS:
        for s in range(0, len(err) - length + 1, 3):
            e, b = err[s:s + length], base[s:s + length]
            if np.sum(b ** 2) == 0:
                continue
            month = tdate.iloc[s + length // 2].month
            rows.append({"target": target, "window_days": length, "start": tdate.iloc[s],
                         "monsoon": month in (6, 7, 8, 9, 10),
                         "PI": 1 - np.sum(e ** 2) / np.sum(b ** 2), "full_PI": full_pi})
    return rows


def main() -> None:
    rows = []
    for tgt in cm.TARGETS:
        for same_day in (True, False):
            rows += grid(tgt, same_day)
    pd.DataFrame(rows).to_csv(cm.RES_DIR / "sufficiency_grid.csv", index=False)
    win = []
    for tgt in cm.TARGETS:
        win += window_scatter(tgt)
    pd.DataFrame(win).to_csv(cm.RES_DIR / "sufficiency_windows.csv", index=False)
    print("sufficiency done")


if __name__ == "__main__":
    main()
