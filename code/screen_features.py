"""Screen candidate variables from the BWDB delivery before adding them to the released tables.

    python code/screen_features.py              (after build_dataset.py; about 9 minutes)
    python code/screen_features.py --long-only  (long-panel screen only; about 6 minutes)

Both screens predict the daily change WL - WLD-1 with Ridge (standardised, alpha
chosen on validation) and LightGBM (early stopping on validation); split train
< 2022, validation 2022, test >= 2023. Skill is the test persistence index
PI = 1 - SSE_model / SSE_persistence. Every candidate is leakage-safe:
target-station values enter only at t-1..t-7.

1. Wide screen (per ML target) -> metadata/feature_screening.csv
Start from the brief's layout (own WLD-1..7 + upstream WL and lags) and add one
candidate group at a time; then test the chosen combination.

  own_range  target's daily range WL_Max - WL_Min at t-1..t-7 (tide)
  inflow_q   log discharge at Hardinge Bridge (SW90) and Bahadurabad (SW46.9L), t..t-7
  chandpur   water level at Chandpur (Id 17, Meghna confluence), t..t-7
  bhairab    water level at Bhairab Bazar (Id 16, Meghna inflow), t..t-7
  rain_mm    target's nearest-gauge rainfall in mm, t..t-7
  season     sin/cos of the day of year

Screen 1 uses the rows where every candidate exists; screen 2 (the released
combination) uses the rows where the brief layout, own range and both Meghna
gauges exist.

2. Long screen (pooled over the 17 stations) -> metadata/feature_screening_long.csv
Rows of the long panel with all seven lags and a Rainfall_Level. BASE is
WLD-1..7, Rainfall_Level, Latitude, Longitude, Tidal; each set adds one group:

  Rainfall_mm, Rain_3dSum, WL_Range_D-1   released columns, one at a time
  season         DOY_sin, DOY_cos
  inflow_q_D-1   discharge at Hardinge Bridge and Bahadurabad on Date-1 (ends 2024)
  evap_D-1       Faridpur (CL406) evaporation on Date-1, negative readings blanked
  released       BASE + every released feature (the five above)

Ridge gets blanks filled with the training median and a one-hot of Id; LightGBM
takes blanks as they are. WL_Trend is screened with a class-balanced LightGBM
classifier on base and released against trend persistence (the class of
WLD-1 - WLD-2); these rows report test macro-F1 (lightgbm) and accuracy.
WL_Trend is derived from WL and is never a feature.

3. Trend classification on the wide matrices -> metadata/trend_classification_wide.csv
Per ML target, the WL_Trend rule applied to WL - WLD-1 is predicted from the
target's own lags or from the whole release layout, by a class-balanced logistic
regression (standardised) and a class-balanced LightGBM classifier, and compared
with "same class as yesterday". Test macro-F1 and accuracy.
"""
from __future__ import annotations

import argparse
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

import build_dataset as b

warnings.filterwarnings("ignore", category=UserWarning)
SEED = 20260913
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
GBM_PARAMS = {"n_estimators": 3000, "learning_rate": 0.03, "num_leaves": 31, "subsample": 0.8,
              "subsample_freq": 1, "colsample_bytree": 0.8, "random_state": SEED,
              "deterministic": True, "n_jobs": 4, "verbose": -1}
EARLY_STOPPING_ROUNDS = 100
VALID_START, TEST_START = "2022-01-01", "2023-01-01"
CALENDAR = pd.date_range(b.START, b.END)
LONG_FILE = "ganges_padma_hydromet_dataset_2011_2025.csv"

LAGS = [f"WLD-{k}" for k in range(1, b.N_LAGS + 1)]
BASE = LAGS + ["Rainfall_Level", "Latitude", "Longitude", "Tidal"]
INFLOW_GAUGES = ("SW90", "SW46.9L")  # Hardinge Bridge (Ganges), Bahadurabad (Jamuna)
EVAP_GAUGE = "CL406"                 # Faridpur, the delivery's only evaporation station
RELEASED = ["Rainfall_mm", "Rain_3dSum", "WL_Range_D-1", "DOY_sin", "DOY_cos"]
LONG_SETS = {
    "base": [],
    "+Rainfall_mm": ["Rainfall_mm"],
    "+Rain_3dSum": ["Rain_3dSum"],
    "+WL_Range_D-1": ["WL_Range_D-1"],
    "+season": ["DOY_sin", "DOY_cos"],
    "+inflow_q_D-1": [f"Q_{sid}_D-1" for sid in INFLOW_GAUGES],
    "+evap_D-1": ["Evap_D-1"],
    "released": RELEASED,
}
CLASS_SETS = ("base", "released")
LABEL = "WL_Trend"


def dated_series(body: list[list], date_col: int, value_col: int) -> pd.Series:
    """Value column by date column on the study calendar; first of any duplicate date."""
    recs = [(b.parse_date(r[date_col]), b.num(r[value_col])) for r in body if len(r) > value_col]
    valid = [(d, v) for d, v in recs if d is not None and v is not None and b.START <= d <= b.END]
    return pd.Series(dict(reversed(valid)), dtype=float).reindex(CALENDAR)


def discharge_series(body: list[list]) -> pd.Series:
    """Mean daily discharge (MDD, column 8; date in column 6) of a Discharge workbook."""
    return dated_series(body, 6, 8)


def read_candidates() -> tuple[dict, dict, dict]:
    """Daily range per station, discharge per station and rainfall per gauge from raw/."""
    rng, q, rain = {}, {}, {}
    for path in sorted(b.RAW.glob("*.xlsx")):
        meta, body = b.read_workbook(path)
        sid = meta.get("Station ID")
        if path.name.startswith("Water_Level_Daily") and sid in b.ID:
            rng[sid] = (b.daily_series(body, 2)[0] - b.daily_series(body, 3)[0]).reindex(CALENDAR)
        elif path.name.startswith("Discharge_"):  # station id sits in the file name
            q[path.stem.split("_")[-2]] = discharge_series(body)
        elif path.name.startswith("Rainfall_Daily"):
            rain[sid] = b.daily_series(body, 2)[0].reindex(CALENDAR)
    return rng, q, rain


def lagged(s: pd.Series, name: str, same_day: bool) -> pd.DataFrame:
    first = 0 if same_day else 1
    return pd.concat({f"{name}_{k}": s.shift(k) for k in range(first, b.N_LAGS + 1)}, axis=1)


def candidate_groups(tid: int, sid: str, wl: pd.DataFrame, rng: dict, q: dict,
                     rain: dict, gauge: str) -> dict[str, pd.DataFrame]:
    doy = 2 * np.pi * CALENDAR.dayofyear / 365.25
    return {
        "own_range": lagged(rng[sid], "RNG", same_day=False),
        "inflow_q": pd.concat([lagged(np.log(q["SW90"]), "LQ90", True),
                               lagged(np.log(q["SW46.9L"]), "LQ469", True)], axis=1),
        "chandpur": lagged(wl[17], "W17", True),
        "bhairab": lagged(wl[16], "W16", True),
        "rain_mm": lagged(rain[gauge], "RAIN", True),
        "season": pd.DataFrame({"doy_sin": np.sin(doy), "doy_cos": np.cos(doy)}, index=CALENDAR),
    }


def split(dates: pd.Index | pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chronological train / validation / test masks."""
    d = pd.DatetimeIndex(dates)
    return d < VALID_START, (d >= VALID_START) & (d < TEST_START), d >= TEST_START


def fit_ridge(x: pd.DataFrame, y: np.ndarray, tr: np.ndarray, va: np.ndarray) -> Pipeline:
    """Standardised Ridge with the alpha of lowest validation MSE."""
    return min((make_pipeline(StandardScaler(), Ridge(alpha=a)).fit(x[tr], y[tr]) for a in ALPHAS),
               key=lambda m: float(np.mean((y[va] - m.predict(x[va])) ** 2)))


def fit_gbm(model: lgb.LGBMModel, x: pd.DataFrame, y: np.ndarray, tr: np.ndarray,
            va: np.ndarray) -> lgb.LGBMModel:
    """Fit a LightGBM model with early stopping on the validation year."""
    return model.fit(x[tr], y[tr], eval_set=[(x[va], y[va])],
                     callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])


def persistence_index(change: np.ndarray, pred: np.ndarray) -> float:
    """1 - SSE_model / SSE_persistence; persistence predicts a zero daily change."""
    return 1 - float(np.sum((change - pred) ** 2)) / float(np.sum(change ** 2))


def test_pi(frame: pd.DataFrame, cols: list[str]) -> dict[str, float]:
    """Test-period persistence index of Ridge and LightGBM on the daily change."""
    y = (frame.WL - frame["WLD-1"]).to_numpy()
    tr, va, te = split(frame.index)
    x = frame[cols]
    ridge = fit_ridge(x, y, tr, va)
    gbm = fit_gbm(lgb.LGBMRegressor(**GBM_PARAMS), x, y, tr, va)
    return {model: persistence_index(y[te], m.predict(x[te])) for model, m in (("ridge", ridge), ("lightgbm", gbm))}


def screen_target(sid: str, fname: str, wl: pd.DataFrame, cand: tuple, gauge: str) -> list[dict]:
    tid = b.ID[sid]
    wide = pd.read_csv(b.WIDE / fname, parse_dates=["Date"]).set_index("Date")
    brief = [c for c in wide.columns if c.startswith(("WLD-", "P"))]  # own lags + upstream blocks
    groups = candidate_groups(tid, sid, wl, *cand, gauge)
    rows = []
    screen1 = wide[["WL"] + brief].join(pd.concat(groups.values(), axis=1)).dropna()
    rows.append({"screen": 1, "set": "brief layout", **test_pi(screen1, brief)})
    for name, g in groups.items():
        rows.append({"screen": 1, "set": f"brief + {name}", **test_pi(screen1, brief + list(g.columns))})
    chosen = {"meghna": ["chandpur", "bhairab"], "meghna + own_range": ["chandpur", "bhairab", "own_range"]}
    screen2 = wide[["WL"] + brief].join(pd.concat([groups[g] for g in chosen["meghna + own_range"]],
                                                  axis=1)).dropna()
    rows.append({"screen": 2, "set": "brief layout", **test_pi(screen2, brief)})
    for label, names in chosen.items():
        cols = brief + [c for g in names for c in groups[g].columns]
        rows.append({"screen": 2, "set": f"brief + {label}", **test_pi(screen2, cols)})
    return [{"target_id": tid, "rows": len(screen1 if r["screen"] == 1 else screen2), **r} for r in rows]


def screen_wide(long: pd.DataFrame) -> None:
    wl = long.pivot(index="Date", columns="Id", values="WL").reindex(CALENDAR)
    registry = pd.read_csv(b.META / "station_registry.csv").set_index("Station_ID")
    cand = read_candidates()
    rows = []
    for sid, fname in b.TARGETS.items():
        rows += screen_target(sid, fname, wl, cand, registry.at[sid, "Rain_Gauge"])
    out = pd.DataFrame(rows)
    out.to_csv(b.META / "feature_screening.csv", index=False, float_format="%.4f")
    print(out.pivot_table(index=["screen", "set"], columns="target_id",
                          values=["ridge", "lightgbm"]).round(3).to_string())


def read_inflow_evap() -> pd.DataFrame:
    """Date-1 discharge at the inflow gauges and Date-1 evaporation, on the study calendar."""
    cols = {}
    for sid in INFLOW_GAUGES:
        _, body = b.read_workbook(next(b.RAW.glob(f"Discharge_*_{sid}_*.xlsx")))
        cols[f"Q_{sid}_D-1"] = discharge_series(body).shift(1)
    _, body = b.read_workbook(next(b.RAW.glob(f"Evaporation_*{EVAP_GAUGE}*.xlsx")))
    evap = dated_series(body, 5, 6)  # date in column 5, evaporation in column 6
    cols["Evap_D-1"] = evap.where(evap >= 0).shift(1)  # negative evaporation is impossible
    return pd.DataFrame(cols, index=CALENDAR)


def long_frame(long: pd.DataFrame) -> pd.DataFrame:
    """Long-panel rows with all 7 lags and a Rainfall_Level, joined to the Date-1 externals."""
    rows = long.dropna(subset=LAGS + ["Rainfall_Level"]).reset_index(drop=True)
    return rows.join(read_inflow_evap(), on="Date")


def ridge_matrix(frame: pd.DataFrame, cols: list[str], tr: np.ndarray) -> pd.DataFrame:
    """Blanks filled with the training median, plus a one-hot of Id (station intercepts)."""
    x = frame[cols].astype(float)
    return pd.concat([x.fillna(x[tr].median()), pd.get_dummies(frame.Id, prefix="Id", dtype=float)], axis=1)


def long_regression(frame: pd.DataFrame, cols: list[str]) -> dict[str, float]:
    """Pooled test persistence index of Ridge and LightGBM on the daily change."""
    y = (frame.WL - frame["WLD-1"]).to_numpy()
    tr, va, te = split(frame.Date)
    x, xr = frame[cols].astype(float), ridge_matrix(frame, cols, tr)
    ridge = fit_ridge(xr, y, tr, va)
    gbm = fit_gbm(lgb.LGBMRegressor(**GBM_PARAMS), x, y, tr, va)
    return {"ridge": persistence_index(y[te], ridge.predict(xr[te])),
            "lightgbm": persistence_index(y[te], gbm.predict(x[te]))}


def class_scores(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {"lightgbm": f1_score(true, pred, average="macro"), "accuracy": accuracy_score(true, pred)}


def long_classification(frame: pd.DataFrame, cols: list[str]) -> dict[str, float]:
    """Test macro-F1 and accuracy of a class-balanced LightGBM classifier for WL_Trend."""
    label = frame[LABEL].to_numpy(dtype=int)
    tr, va, te = split(frame.Date)
    x = frame[cols].astype(float)
    clf = fit_gbm(lgb.LGBMClassifier(class_weight="balanced", **GBM_PARAMS), x, label, tr, va)
    return class_scores(label[te], clf.predict(x[te]))


def trend_persistence(frame: pd.DataFrame) -> dict[str, float]:
    """Baseline: today's WL_Trend is yesterday's (class of WLD-1 - WLD-2, same +-0.03 m rule)."""
    te = split(frame.Date)[2]
    previous = b.wl_trend(frame["WLD-1"] - frame["WLD-2"]).to_numpy(dtype=int)
    return class_scores(frame[LABEL].to_numpy(dtype=int)[te], previous[te])


def screen_long(long: pd.DataFrame) -> None:
    frame = long_frame(long)
    test_rows = int(split(frame.Date)[2].sum())
    rows = []
    for name, extra in LONG_SETS.items():
        cols = BASE + extra
        assert LABEL not in cols, "WL_Trend is derived from WL; it must never be a feature"
        rows.append({"task": "regression", "set": name, "n_features": len(cols), "test_rows": test_rows,
                     **long_regression(frame, cols)})
    for name in CLASS_SETS:
        cols = BASE + LONG_SETS[name]
        rows.append({"task": "classification", "set": name, "n_features": len(cols), "test_rows": test_rows,
                     **long_classification(frame, cols)})
    rows.append({"task": "classification", "set": "trend persistence", "n_features": 2, "test_rows": test_rows,
                 **trend_persistence(frame)})
    out = pd.DataFrame(rows, columns=["task", "set", "n_features", "test_rows", "ridge", "lightgbm", "accuracy"])
    out.to_csv(b.META / "feature_screening_long.csv", index=False, float_format="%.4f")
    print(f"long panel: {len(frame):,} rows, {test_rows:,} in test")
    print(out.round(3).to_string(index=False))


def fit_logistic(x: pd.DataFrame, label: np.ndarray, tr: np.ndarray) -> Pipeline:
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")).fit(
        x[tr], label[tr])


def trend_target(sid: str, fname: str) -> list[dict]:
    """WL_Trend on one wide matrix: own lags vs release layout, two models, and the rule."""
    wide = pd.read_csv(b.WIDE / fname, parse_dates=["Date"])
    label = b.wl_trend(wide["WL"] - wide["WLD-1"]).to_numpy(dtype=int)
    tr, va, te = split(wide.Date)
    previous = b.wl_trend(wide["WLD-1"] - wide["WLD-2"]).to_numpy(dtype=int)
    rows = [{"set": "same class as yesterday", "model": "rule", **class_scores(label[te], previous[te])}]
    sets = {"own lags": LAGS, "release layout": [c for c in wide.columns if c not in ("Id", "Date", "WL")]}
    for name, cols in sets.items():
        x = wide[cols].astype(float)
        logit = fit_logistic(x, label, tr)
        gbm = fit_gbm(lgb.LGBMClassifier(class_weight="balanced", **GBM_PARAMS), x, label, tr, va)
        rows += [{"set": name, "model": "logistic", **class_scores(label[te], logit.predict(x[te]))},
                 {"set": name, "model": "lightgbm", **class_scores(label[te], gbm.predict(x[te]))}]
    return [{"target_id": b.ID[sid], "test_rows": int(te.sum()), **r} for r in rows]


def screen_trend_wide() -> None:
    rows = [r for sid, fname in b.TARGETS.items() for r in trend_target(sid, fname)]
    out = pd.DataFrame(rows).rename(columns={"lightgbm": "macro_f1"})
    out = out[["target_id", "test_rows", "set", "model", "macro_f1", "accuracy"]]
    out.to_csv(b.META / "trend_classification_wide.csv", index=False, float_format="%.4f")
    print(out.round(3).to_string(index=False))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Screen candidate features for the released tables")
    parser.add_argument("--long-only", action="store_true",
                        help="run only the pooled long-panel screen (skip the per-target wide screen)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    long = pd.read_csv(b.PROCESSED / LONG_FILE, parse_dates=["Date"])
    if not args.long_only:
        screen_wide(long)
    screen_long(long)
    screen_trend_wide()


if __name__ == "__main__":
    main()
