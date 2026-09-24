"""Baseline models and PI-STGNN for the Ganges-Padma point-wise water-level dataset.

How to run
----------
Locally, from the repository root (Python 3.10+, numpy, pandas, scikit-learn,
lightgbm, torch)::

    python code/train_baselines.py --out results/local          # full run
    python code/train_baselines.py --quick --out results/smoke  # smoke test
    python code/train_baselines.py --skip-lstm --out results/x  # no LSTM
    python code/train_baselines.py --skip-gnn --out results/x   # no PI-STGNN

The data files are found automatically: a recursive search under
``/kaggle/input`` (Kaggle) and then under ``../processed`` relative to this
script. ``--data-dir`` overrides the search root. On Kaggle, attach a dataset
holding the long panel and the three wide matrices and run the script as it
is; outputs go to ``/kaggle/working`` unless ``--out`` is given.

Task
----
For each target gauge (Baruria Transit Id 11, Bhagyakul Id 12, Sureswar Id 15)
predict the daily water level WL on day t from the full released wide design
matrix: the target's own WLD-1..WLD-7; for every upstream point its same-day
P<id>_WL and P<id>_WLD-1..7; and for every boundary point (parsed from the
file: B16 Bhairab Bazar, Meghna inflow, and B17 Chandpur, the Meghna
confluence that acts as the downstream backwater control) its same-day
B<id>_WL and B<id>_WLD-1..7. The target's own day-t WL is never an input.
All learned models are trained on the daily change delta = WL - WLD-1, and
the level is reconstructed as WL_hat = WLD-1 + delta_hat.

Split by date: train < 2022-01-01, validation = 2022 (used only for Ridge
alpha choice and early stopping), test >= 2023-01-01. Scalers are fitted on
the training rows only.

Models: persistence (WL_hat = WLD-1), Ridge regression, LightGBM, an LSTM
(PyTorch, three seeds) and PI-STGNN (three seeds), plus the same graph network
trained without its physics term ("stgnn_nophys", three seeds) as the physics
ablation. The LSTM reads t-7..t-1 levels of [target, every upstream, every
boundary] point as sequence channels and every upstream + boundary WL at t as a
same-day vector. A feature ablation (Ridge and LightGBM, feature sets A-G, on
the rows where the target's own Rainfall_Level exists) and LightGBM gain
importance per source (own lags, P<id>, B<id>) are also produced:
A own lags; B A + Rainfall_Level; C A + upstream P blocks (brief layout);
D C + Rainfall_Level; E C without same-day P<id>_WL; F C + boundary B blocks
(full release layout); G F without every same-day column (P<id>_WL, B<id>_WL),
i.e. a true 1-day-ahead forecast with the full layout.

PI-STGNN (physics-informed spatio-temporal graph network, own section below).
Nodes are the target, its upstream gauges and the boundary gauges 16 and 17
(N = 13 / 14 / 16). Directed edges follow the river downstream (Id 14
excluded) plus the Meghna edge 16 -> 17, and one "backwater link" 17 -> target
(the confluence stage propagates upstream as backwater). Edges carry the
along-channel distance (chainage difference) or, where a chainage is missing
(Jamuna reach, 16 -> 17, 17 -> target), the great-circle distance, / 100 km.
Each node gets an 8-step window t-7..t of [standardised stage, mask]; the
target's step-t slot is masked (value 0, mask 0), so WL_t is never an input.
A GRU shared by all nodes encodes each window, one message-passing sweep in
topological order (16 and 17 before the target) carries the states to the
target, and an MLP on the target's state predicts the standardised delta.
Training minimises MSE + lambda * mean(r^2), where r is a continuity residual
built from learnable rating curves of the target and its nearest upstream
gauge (10, 11, 13; lambda = 0.1 for "pistgnn", 0 for "stgnn_nophys"). Same
split, early stopping on validation delta RMSE, seeds and metrics as the LSTM.

Outputs (CSV): metrics.csv, ablation.csv, predictions_test.csv,
importance_by_point.csv, lstm_history.csv, gnn_history.csv, and
environment.json.
"""
import argparse
import glob
import hashlib
import json
import math
import os
import platform
import random
import re
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # before torch

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
import torch  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from torch import nn  # noqa: E402

SEED = 20260913
LSTM_SEEDS = (20260913, 20260914, 20260915)
GNN_SEEDS = (20260913, 20260914, 20260915)
TRAIN_END = pd.Timestamp("2022-01-01")   # train: Date < TRAIN_END
TEST_START = pd.Timestamp("2023-01-01")  # validation: TRAIN_END <= Date < TEST_START
MONSOON_MONTHS = (7, 8, 9, 10)
N_LAGS = 7
LAG_COLS = [f"WLD-{k}" for k in range(1, N_LAGS + 1)]
RAIN_COL = "Rainfall_Level"
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
LONG_FILE = "ganges_padma_hydromet_dataset_2011_2025.csv"
TARGETS = (  # (target Id, short name, wide design matrix)
    (11, "Baruria", "wide_sw91_9l_baruria_transit.csv"),
    (12, "Bhagyakul", "wide_sw93_4l_bhagyakul.csv"),
    (15, "Sureswar", "wide_sw95_sureswar.csv"),
)
STATION_NAMES = {  # metadata/station_registry.csv
    1: "Panka", 2: "Rajshahi", 3: "Sardah", 4: "Hardinge Bridge",
    5: "Talbaria", 6: "Sengram", 7: "Mohendrapur", 8: "Bahadurabad Transit",
    9: "Aricha", 10: "Goalundo Transit", 11: "Baruria Transit",
    12: "Bhagyakul", 13: "Mawa", 14: "Tarpasa", 15: "Sureswar",
    16: "Bhairab Bazar", 17: "Chandpur",
}
ABLATION_SETS = {
    "A": "own WLD-1..7",
    "B": "A + own Rainfall_Level",
    "C": "A + upstream WL and WLD-1..7 (brief layout)",
    "D": "C + own Rainfall_Level",
    "E": "C without same-day upstream WL (1-day-ahead)",
    "F": "C + boundary WL and WLD-1..7 (full release layout)",
    "G": "F without same-day upstream and boundary WL (1-day-ahead)",
}
LGBM_PARAMS = {
    "learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 20,
    "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
    "deterministic": True, "force_col_wise": True, "n_jobs": 4,
    "random_state": SEED, "verbosity": -1,
}
LSTM_HIDDEN = 64
HEAD_HIDDEN = 32
HEAD_DROPOUT = 0.1
LSTM_LR = 1e-3
BATCH_SIZE = 64


@dataclass(frozen=True)
class RunConfig:
    """Training budget and switches."""

    lgbm_max_rounds: int = 3000
    lgbm_patience: int = 100
    lstm_max_epochs: int = 300
    lstm_patience: int = 30
    lstm_seeds: tuple[int, ...] = LSTM_SEEDS
    gnn_max_epochs: int = 300
    gnn_patience: int = 30
    gnn_seeds: tuple[int, ...] = GNN_SEEDS
    skip_lstm: bool = False
    skip_gnn: bool = False
    skip_ablation: bool = False
    device: str = "cpu"


QUICK_OVERRIDES = {"lgbm_max_rounds": 200, "lgbm_patience": 20,
                   "lstm_max_epochs": 4, "lstm_patience": 2,
                   "lstm_seeds": LSTM_SEEDS[:1],
                   "gnn_max_epochs": 4, "gnn_patience": 2,
                   "gnn_seeds": GNN_SEEDS[:1]}


@dataclass(frozen=True)
class TargetData:
    """One target's wide matrix (sorted by date) with its own rainfall level;
    upstream (P<id>_) and boundary (B<id>_) point Ids are parsed from the file."""

    target_id: int
    name: str
    frame: pd.DataFrame
    upstream_ids: tuple[int, ...]
    boundary_ids: tuple[int, ...]


# --------------------------------------------------------------------------- setup
def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (CPU and CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_torch_determinism() -> None:
    """Ask torch for deterministic kernels (warn instead of failing)."""
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def script_dir() -> Path:
    """Directory of this script, or the working directory inside a notebook."""
    return Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()


def on_kaggle() -> bool:
    """True when running inside a Kaggle kernel."""
    return Path("/kaggle/input").is_dir()


def search_file(name: str, roots: list[Path]) -> Path | None:
    """First file called ``name`` found recursively under the roots, else None."""
    for root in roots:
        if root.is_dir():
            hits = sorted(glob.glob(str(root / "**" / name), recursive=True))
            if hits:
                return Path(hits[0])
    return None


def find_data_file(name: str, data_dir: Path | None) -> Path:
    """Locate a data file by name under --data-dir, /kaggle/input or ../processed."""
    roots = [data_dir] if data_dir else []
    roots += [Path("/kaggle/input"), script_dir().parent / "processed"]
    hit = search_file(name, roots)
    if hit is None:
        raise FileNotFoundError(f"{name} not found under {[str(r) for r in roots]}")
    return hit


def sha256(path: Path) -> str:
    """SHA-256 of a file (to show that runs used identical data)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- data
UPSTREAM, BOUNDARY = "P", "B"  # wide-matrix block prefixes
BLOCK_RE = re.compile(r"([PB])(\d{2})_(WL|WLD-[1-7])")


def point_ids(columns: pd.Index, prefix: str) -> tuple[int, ...]:
    """Sorted point Ids of the <prefix><id>_WL blocks of a wide matrix."""
    return tuple(sorted(int(m.group(1)) for c in columns
                        if (m := re.fullmatch(rf"{prefix}(\d{{2}})_WL", c))))


def load_target(wide_path: Path, long: pd.DataFrame, target_id: int,
                name: str) -> TargetData:
    """Read a wide matrix and join the target's own Rainfall_Level on Date."""
    wide = pd.read_csv(wide_path, parse_dates=["Date"])
    assert set(wide["Id"].unique()) == {target_id}, "wide file has wrong target Id"
    assert not wide.isna().any().any(), "wide matrix must be complete"
    assert wide["Date"].is_unique, "duplicate dates in wide matrix"
    up_ids, bnd_ids = point_ids(wide.columns, UPSTREAM), point_ids(wide.columns, BOUNDARY)
    assert target_id not in up_ids + bnd_ids, "the target must not be its own predictor"
    assert not set(up_ids) & set(bnd_ids), "a point is either upstream or boundary"
    predictors = [c for c in wide.columns if c not in ("Id", "Date", "WL")]
    assert predictors == full_columns(up_ids, bnd_ids), "unexpected wide-matrix layout"
    rain = long.loc[long["Id"] == target_id, ["Date", RAIN_COL]]
    frame = (wide.merge(rain, on="Date", how="left", validate="one_to_one")
             .sort_values("Date").reset_index(drop=True))
    return TargetData(target_id, name, frame, up_ids, bnd_ids)


def block_columns(prefix: str, point_id: int) -> list[str]:
    """One point's block: <prefix><id>_WL, then <prefix><id>_WLD-1..7."""
    return [f"{prefix}{point_id:02d}_WL"] + [f"{prefix}{point_id:02d}_{c}" for c in LAG_COLS]


def same_day_columns(prefix: str, ids: tuple[int, ...]) -> list[str]:
    """Same-day water-level columns <prefix><id>_WL."""
    return [f"{prefix}{i:02d}_WL" for i in ids]


def brief_columns(up_ids: tuple[int, ...]) -> list[str]:
    """Brief layout: own lags, then P<id>_WL and P<id>_WLD-1..7 per point."""
    return list(LAG_COLS) + [c for i in up_ids for c in block_columns(UPSTREAM, i)]


def full_columns(up_ids: tuple[int, ...], bnd_ids: tuple[int, ...]) -> list[str]:
    """Full released layout: the brief layout plus B<id>_WL, B<id>_WLD-1..7 per
    boundary point (every predictor column of the wide matrix, in file order)."""
    return brief_columns(up_ids) + [c for i in bnd_ids for c in block_columns(BOUNDARY, i)]


def td_full_columns(td: TargetData) -> list[str]:
    """Full released layout of one target."""
    return full_columns(td.upstream_ids, td.boundary_ids)


def ablation_columns(set_id: str, up_ids: tuple[int, ...],
                     bnd_ids: tuple[int, ...]) -> list[str]:
    """Feature columns for ablation sets A-G."""
    same_up = set(same_day_columns(UPSTREAM, up_ids))
    same_all = same_up | set(same_day_columns(BOUNDARY, bnd_ids))
    brief, full = brief_columns(up_ids), full_columns(up_ids, bnd_ids)
    return {
        "A": list(LAG_COLS),
        "B": list(LAG_COLS) + [RAIN_COL],
        "C": brief,
        "D": brief + [RAIN_COL],
        "E": [c for c in brief if c not in same_up],
        "F": full,
        "G": [c for c in full if c not in same_all],
    }[set_id]


def split_masks(dates: pd.Series) -> dict[str, np.ndarray]:
    """Chronological train / validation / test masks, checked for order."""
    masks = {
        "train": (dates < TRAIN_END).to_numpy(),
        "val": ((dates >= TRAIN_END) & (dates < TEST_START)).to_numpy(),
        "test": (dates >= TEST_START).to_numpy(),
    }
    assert sum(m.sum() for m in masks.values()) == len(dates), "splits must cover all rows"
    assert all(m.any() for m in masks.values()), "every split needs rows"
    assert dates[masks["train"]].max() < dates[masks["val"]].min(), "train must precede val"
    assert dates[masks["val"]].max() < dates[masks["test"]].min(), "val must precede test"
    return masks


def check_features(frame: pd.DataFrame, cols: list[str]) -> None:
    """Leakage guard: the target's day-t WL (or a copy of it) is never a
    feature. Allowed are only the target's own lags, its Rainfall_Level and
    upstream/boundary blocks of OTHER points."""
    assert not {"WL", "Id", "Date"} & set(cols), "WL/Id/Date must not be features"
    assert len(set(cols)) == len(cols), "duplicate feature columns"
    target_id = int(frame["Id"].iloc[0])
    for col in cols:
        if col in LAG_COLS or col == RAIN_COL:
            continue
        m = BLOCK_RE.fullmatch(col)
        assert m is not None, f"{col} is not an allowed feature column"
        assert int(m.group(2)) != target_id, f"{col} is a column of the target itself"
    y = frame["WL"].to_numpy()
    for col in cols:
        assert not np.array_equal(frame[col].to_numpy(), y), f"{col} equals WL"


# --------------------------------------------------------------------------- metrics
def compute_metrics(obs: np.ndarray, pred: np.ndarray, pers: np.ndarray,
                    dates: pd.Series) -> dict[str, float]:
    """RMSE, MAE, NSE, persistence index and monsoon (Jul-Oct) RMSE on WL."""
    err = pred - obs
    sse = float(np.sum(err ** 2))
    monsoon = dates.dt.month.isin(MONSOON_MONTHS).to_numpy()
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "nse": 1.0 - sse / float(np.sum((obs - obs.mean()) ** 2)),
        "pi": 1.0 - sse / float(np.sum((pers - obs) ** 2)),
        "rmse_monsoon": float(np.sqrt(np.mean(err[monsoon] ** 2))),
        "n": int(len(obs)),
        "n_monsoon": int(monsoon.sum()),
    }


def split_metrics(frame: pd.DataFrame, masks: dict[str, np.ndarray],
                  wl_hat: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    """Metrics for the validation and test splits of one prediction set."""
    out = {}
    for split in ("val", "test"):
        rows = frame[masks[split]]
        out[split] = compute_metrics(rows["WL"].to_numpy(), wl_hat[split],
                                     rows["WLD-1"].to_numpy(), rows["Date"])
    return out


# --------------------------------------------------------------------------- tabular
def fit_ridge(x_tr: np.ndarray, d_tr: np.ndarray, x_va: np.ndarray,
              d_va: np.ndarray) -> tuple[Pipeline, float, dict[str, float]]:
    """Ridge with train-only StandardScaler; alpha chosen by validation RMSE."""
    best: tuple[float, float, Pipeline] | None = None
    scores = {}
    for alpha in RIDGE_ALPHAS:
        model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha))])
        model.fit(x_tr, d_tr)
        rmse = float(np.sqrt(np.mean((model.predict(x_va) - d_va) ** 2)))
        scores[str(alpha)] = rmse
        if best is None or rmse < best[0]:
            best = (rmse, alpha, model)
    assert best is not None
    scaler = best[2].named_steps["scale"]
    assert scaler.n_samples_seen_ == len(x_tr), "scaler must see train rows only"
    assert np.allclose(scaler.mean_, x_tr.mean(axis=0)), "scaler mean must be train mean"
    return best[2], best[1], scores


def fit_lgbm(x_tr: pd.DataFrame, d_tr: np.ndarray, x_va: pd.DataFrame,
             d_va: np.ndarray, cfg: RunConfig) -> lgb.LGBMRegressor:
    """LightGBM on delta with early stopping on the validation year."""
    model = lgb.LGBMRegressor(n_estimators=cfg.lgbm_max_rounds, **LGBM_PARAMS)
    model.fit(x_tr, d_tr, eval_set=[(x_va, d_va)], eval_metric="l2",
              callbacks=[lgb.early_stopping(cfg.lgbm_patience, verbose=False)])
    return model


def run_tabular(frame: pd.DataFrame, cols: list[str], masks: dict[str, np.ndarray],
                cfg: RunConfig) -> dict[str, dict]:
    """Fit Ridge and LightGBM on delta; return WL predictions and fit info."""
    check_features(frame, cols)
    x = {s: frame.loc[m, cols] for s, m in masks.items()}
    d = {s: (frame.loc[m, "WL"] - frame.loc[m, "WLD-1"]).to_numpy() for s, m in masks.items()}
    base = {s: frame.loc[m, "WLD-1"].to_numpy() for s, m in masks.items()}
    ridge, alpha, scores = fit_ridge(x["train"].to_numpy(), d["train"],
                                     x["val"].to_numpy(), d["val"])
    gbm = fit_lgbm(x["train"], d["train"], x["val"], d["val"], cfg)
    best_iter = int(gbm.best_iteration_)
    out = {
        "ridge": {"wl_hat": {s: base[s] + ridge.predict(x[s].to_numpy()) for s in ("val", "test")},
                  "info": {"alpha": alpha, "val_rmse_by_alpha": scores}},
        "lightgbm": {"wl_hat": {s: base[s] + gbm.predict(x[s], num_iteration=best_iter)
                                for s in ("val", "test")},
                     "info": {"best_iteration": best_iter}, "model": gbm},
    }
    return out


def importance_by_point(gbm: lgb.LGBMRegressor, td: TargetData) -> pd.DataFrame:
    """Total LightGBM gain per source (own lags, upstream point P<id> or
    boundary point B<id>), in %."""
    booster = gbm.booster_
    gain = pd.Series(booster.feature_importance("gain", iteration=gbm.best_iteration_),
                     index=booster.feature_name())
    rows = [{"source": "own", "source_id": td.target_id,
             "gain": gain[LAG_COLS].sum(), "gain_same_day": 0.0}]
    blocks = ([(UPSTREAM, i) for i in td.upstream_ids]
              + [(BOUNDARY, i) for i in td.boundary_ids])
    for prefix, i in blocks:
        cols = block_columns(prefix, i)
        rows.append({"source": f"{prefix}{i:02d}", "source_id": i,
                     "gain": gain[cols].sum(), "gain_same_day": gain[cols[0]]})
    imp = pd.DataFrame(rows)
    total = imp["gain"].sum()
    assert np.isclose(total, gain.sum()), "every feature must belong to one source"
    imp["gain_pct"] = 100.0 * imp["gain"] / total
    imp["gain_pct_same_day"] = 100.0 * imp["gain_same_day"] / total
    imp["gain_pct_lags"] = imp["gain_pct"] - imp["gain_pct_same_day"]
    imp.insert(0, "target", td.name)
    imp.insert(3, "station", imp["source_id"].map(STATION_NAMES))
    imp["rank"] = imp["gain_pct"].rank(ascending=False, method="first").astype(int)
    return imp.drop(columns="gain_same_day")


# --------------------------------------------------------------------------- LSTM
class LstmNowcaster(nn.Module):
    """LSTM over t-7..t-1 levels; last hidden state + same-day upstream and
    boundary WL -> delta."""

    def __init__(self, n_seq: int, n_now: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(n_seq, LSTM_HIDDEN, num_layers=1, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(LSTM_HIDDEN + n_now, HEAD_HIDDEN), nn.ReLU(),
            nn.Dropout(HEAD_DROPOUT), nn.Linear(HEAD_HIDDEN, 1))

    def forward(self, seq: torch.Tensor, now: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(seq)
        return self.head(torch.cat([h_n[-1], now], dim=1)).squeeze(-1)


def build_sequences(td: TargetData) -> tuple[np.ndarray, np.ndarray]:
    """Sequence (n, 7, 1+P+B): step j = day t-7+j, channels [target, every
    upstream, every boundary]; plus the same-day vector (n, P+B) of every
    upstream and boundary WL at t (the target's own WL_t is never read)."""
    points = ([f"{UPSTREAM}{i:02d}" for i in td.upstream_ids]
              + [f"{BOUNDARY}{i:02d}" for i in td.boundary_ids])
    steps = []
    for k in range(N_LAGS, 0, -1):
        cols = [f"WLD-{k}"] + [f"{p}_WLD-{k}" for p in points]
        check_features(td.frame, cols)
        steps.append(td.frame[cols].to_numpy(dtype=np.float64))
    seq = np.stack(steps, axis=1)
    now_cols = [f"{p}_WL" for p in points]
    check_features(td.frame, now_cols)
    now = td.frame[now_cols].to_numpy(dtype=np.float64)
    assert seq.shape[1:] == (N_LAGS, 1 + len(points)) and now.shape[1] == len(points)
    return seq, now


@dataclass(frozen=True)
class Standardizer:
    """Train-only statistics: per sequence channel (pooled over the 7 steps),
    per same-day column, and for delta."""

    seq_mean: np.ndarray
    seq_std: np.ndarray
    now_mean: np.ndarray
    now_std: np.ndarray
    y_mean: float
    y_std: float
    n_fit: int

    @classmethod
    def fit(cls, seq: np.ndarray, now: np.ndarray, y: np.ndarray) -> "Standardizer":
        return cls(seq.mean(axis=(0, 1)), seq.std(axis=(0, 1)), now.mean(axis=0),
                   now.std(axis=0), float(y.mean()), float(y.std()), len(y))

    def transform(self, seq: np.ndarray, now: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (seq - self.seq_mean) / self.seq_std, (now - self.now_mean) / self.now_std


def to_tensor(arr: np.ndarray, device: str) -> torch.Tensor:
    """float32 tensor on the requested device."""
    return torch.as_tensor(np.ascontiguousarray(arr), dtype=torch.float32, device=device)


def predict_delta(model: nn.Module, seq: torch.Tensor, now: torch.Tensor,
                  scaler: Standardizer) -> np.ndarray:
    """Model output mapped back to delta in metres."""
    model.eval()
    with torch.no_grad():
        out = model(seq, now).double().cpu().numpy()
    return out * scaler.y_std + scaler.y_mean


def train_epoch(model: nn.Module, opt: torch.optim.Optimizer, seq: torch.Tensor,
                now: torch.Tensor, y: torch.Tensor, gen: torch.Generator) -> float:
    """One shuffled pass over the training rows; returns mean MSE (scaled delta)."""
    model.train()
    perm = torch.randperm(len(y), generator=gen).to(y.device)
    total = 0.0
    for start in range(0, len(y), BATCH_SIZE):
        idx = perm[start:start + BATCH_SIZE]
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(seq[idx], now[idx]), y[idx])
        loss.backward()
        opt.step()
        total += float(loss.item()) * len(idx)
    return total / len(y)


def fit_lstm(data: dict[str, dict[str, torch.Tensor]], d_val: np.ndarray,
             scaler: Standardizer, seed: int, cfg: RunConfig) -> tuple[nn.Module, list[dict]]:
    """Adam training with early stopping on validation RMSE; best weights restored."""
    set_seed(seed)
    tr, va = data["train"], data["val"]
    model = LstmNowcaster(tr["seq"].shape[2], tr["now"].shape[1]).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=LSTM_LR)
    gen = torch.Generator().manual_seed(seed)
    best_rmse, best_state, wait, history = np.inf, None, 0, []
    for epoch in range(1, cfg.lstm_max_epochs + 1):
        train_loss = train_epoch(model, opt, tr["seq"], tr["now"], tr["y"], gen)
        val_err = predict_delta(model, va["seq"], va["now"], scaler) - d_val
        val_rmse = float(np.sqrt(np.mean(val_err ** 2)))
        improved = val_rmse < best_rmse
        history.append({"epoch": epoch, "train_mse_scaled": train_loss,
                        "val_mse_scaled": float(np.mean((val_err / scaler.y_std) ** 2)),
                        "val_rmse_m": val_rmse, "improved": improved})
        if improved:
            best_rmse, wait = val_rmse, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg.lstm_patience:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, history


def run_lstm(td: TargetData, masks: dict[str, np.ndarray],
             cfg: RunConfig) -> tuple[dict[int, dict[str, np.ndarray]], pd.DataFrame, dict]:
    """Train the LSTM once per seed; return WL predictions per seed and history."""
    seq, now = build_sequences(td)
    delta = (td.frame["WL"] - td.frame["WLD-1"]).to_numpy()
    base = td.frame["WLD-1"].to_numpy()
    tr = masks["train"]
    scaler = Standardizer.fit(seq[tr], now[tr], delta[tr])
    assert scaler.n_fit == int(tr.sum()), "LSTM scaler must see train rows only"
    seq_s, now_s = scaler.transform(seq, now)
    y_s = (delta - scaler.y_mean) / scaler.y_std
    data = {s: {"seq": to_tensor(seq_s[m], cfg.device), "now": to_tensor(now_s[m], cfg.device),
                "y": to_tensor(y_s[m], cfg.device)} for s, m in masks.items()}
    preds, hist, info = {}, [], {}
    for seed in cfg.lstm_seeds:
        model, history = fit_lstm(data, delta[masks["val"]], scaler, seed, cfg)
        preds[seed] = {s: base[masks[s]] + predict_delta(model, data[s]["seq"], data[s]["now"],
                                                         scaler) for s in ("val", "test")}
        best = min(history, key=lambda h: h["val_rmse_m"])
        info[str(seed)] = {"best_epoch": best["epoch"], "epochs_run": len(history)}
        hist += [{"target": td.name, "seed": seed, **h} for h in history]
    return preds, pd.DataFrame(hist), info


# =========================================================================== PI-STGNN
# Physics-informed spatio-temporal graph neural network (one model per target).
#   graph    nodes = target + upstream gauges + boundary gauges 16, 17; edges =
#            river edges (downstream) + 16 -> 17 (Meghna, downstream) + the
#            "backwater link" 17 -> target (confluence stage acts upstream)
#   input    X (n, N, 8, 2): node windows t-7..t of [standardised stage, mask];
#            the target's step-t slot is masked (0, 0), so WL_t is never an input
#   encoder  one GRU(2 -> 32) shared by all nodes -> h_i (last hidden state)
#   spatial  ONE sweep in topological order (16, 17 before the target):
#            h_j <- h_j + sum_{i -> j} ReLU(W [h_i, h_j, d_ij] + b), W shared
#   readout  h_target -> Linear(32, 16) -> ReLU -> Linear(16, 1) = delta_hat (std)
#   loss     MSE(delta_hat, delta) + lambda * mean(r^2) with the continuity
#            residual r = delta_hat - (Q_u(h_u,t) - Q_s(h_s,t-1)) and learnable
#            rating curves Q_k(h) = a_k softplus(h - h0_k)^b_k (u = nearest
#            upstream gauge, s = target; stages in standardised node units)
GNN_HIDDEN = 32
GNN_HEAD_HIDDEN = 16
GNN_STEPS = N_LAGS + 1  # t-7 ... t
GNN_CHANNELS = 2        # standardised stage, observation mask
GNN_LR = 1e-3
PHYSICS_LAMBDA = 0.1
GNN_VARIANTS = (("pistgnn", PHYSICS_LAMBDA), ("stgnn_nophys", 0.0))
EDGE_KM_SCALE = 100.0
EARTH_RADIUS_KM = 6371.0
RIVER_EDGES = (  # directed downstream; Id 14 (Tarpasa) is excluded, so 13 -> 15
    (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 10), (8, 9), (9, 10),
    (10, 11), (11, 12), (12, 13), (13, 15))
BOUNDARY_EDGES = ((16, 17),)  # Meghna, downstream: Bhairab Bazar -> Chandpur
BACKWATER_SOURCE = 17  # Chandpur (confluence, downstream control) -> target
EDGE_DOWNSTREAM, EDGE_BACKWATER = "downstream", "backwater"
STATION_GEO = {  # Id: (Latitude, Longitude, Chainage_km); metadata/station_registry.csv
    1: (24.59997, 88.10575, 0.0), 2: (24.36103, 88.57042, 54.01),
    3: (24.30611, 88.71928, 70.28), 4: (24.07944, 89.03361, 110.92),
    5: (23.96306, 89.08444, 124.86), 6: (23.90306, 89.36583, 154.22),
    7: (23.79911, 89.5778, 178.68), 8: (25.13028, 89.73464, None),
    9: (23.83663, 89.77978, None), 10: (23.76785, 89.77881, 199.43),
    11: (23.78434, 89.81391, 203.44), 12: (23.50587, 90.21816, 254.96),
    13: (23.47478, 90.25306, 259.93), 14: (23.46621, 90.33216, 268.05),
    15: (23.31147, 90.47589, 290.66), 16: (24.04548, 90.99107, None),
    17: (23.22905, 90.6448, None),
}
PHYSICS_PARENT = {11: 10, 12: 11, 15: 13}  # nearest upstream gauge of each target


def verify_station_geo(data_dir: Path | None) -> str:
    """Check STATION_GEO against metadata/station_registry.csv when it is found."""
    roots = ([data_dir] if data_dir else []) + [Path("/kaggle/input"),
                                                 script_dir().parent / "metadata"]
    path = search_file("station_registry.csv", roots)
    if path is None:
        return "station_registry.csv not found; embedded STATION_GEO used"
    reg = pd.read_csv(path).set_index("Id")
    for sid, (lat, lon, chainage) in STATION_GEO.items():
        row = reg.loc[sid]
        assert np.isclose(row["Latitude"], lat) and np.isclose(row["Longitude"], lon)
        if chainage is None:
            assert pd.isna(row["Chainage_km"]), f"Id {sid}: registry has a chainage"
        else:
            assert np.isclose(row["Chainage_km"], chainage), f"Id {sid}: chainage differs"
    if "Role" in reg.columns:
        graph_boundary = {n for e in BOUNDARY_EDGES for n in e} | {BACKWATER_SOURCE}
        boundary = set(reg.index[reg["Role"] == "Boundary predictor"])
        assert boundary == graph_boundary, "registry boundary points != graph's"
    return f"STATION_GEO matches {path}"


def haversine_km(a: int, b: int) -> float:
    """Great-circle distance between two gauges (km)."""
    lat1, lon1, lat2, lon2 = map(math.radians, (*STATION_GEO[a][:2], *STATION_GEO[b][:2]))
    hav = (math.sin((lat2 - lat1) / 2) ** 2
           + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 2.0 * EARTH_RADIUS_KM * math.asin(math.sqrt(hav))


def edge_distance_km(i: int, j: int) -> tuple[float, str]:
    """Along-channel distance |chainage_j - chainage_i| if both exist, else haversine."""
    ci, cj = STATION_GEO[i][2], STATION_GEO[j][2]
    if ci is not None and cj is not None:
        return abs(cj - ci), "chainage"
    return haversine_km(i, j), "haversine"


def is_downstream(i: int, j: int) -> bool:
    """j lies downstream of i: larger chainage, or (Jamuna reach, no chainage)
    further south."""
    ci, cj = STATION_GEO[i][2], STATION_GEO[j][2]
    if ci is not None and cj is not None:
        return cj > ci
    return STATION_GEO[j][0] < STATION_GEO[i][0]


@dataclass(frozen=True)
class RiverGraph:
    """Directed graph over one target's gauges (downstream river edges plus the
    backwater link 17 -> target); nodes in topological (sweep) order, edges
    grouped by destination node."""

    target_id: int
    node_ids: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]
    edge_km: tuple[float, ...]
    edge_basis: tuple[str, ...]
    edge_kind: tuple[str, ...]
    boundary_ids: tuple[int, ...]

    def pos(self, node_id: int) -> int:
        """Position of a gauge Id in the node order."""
        return self.node_ids.index(node_id)

    @property
    def physics_parent(self) -> int:
        """The target's single upstream neighbour along a downstream edge (used
        by the physics loss; the backwater link is not a flow edge)."""
        parents = [i for (i, j), kind in zip(self.edges, self.edge_kind)
                   if j == self.target_id and kind == EDGE_DOWNSTREAM]
        assert len(parents) == 1, "target must have exactly one upstream parent"
        return parents[0]


def topological_order(nodes: set[int], edges: list[tuple[int, int]]) -> tuple[int, ...]:
    """Kahn's algorithm, smaller Id first among ready nodes."""
    indegree = {n: sum(1 for _, j in edges if j == n) for n in nodes}
    ready, order = sorted(n for n in nodes if indegree[n] == 0), []
    while ready:
        node = ready.pop(0)
        order.append(node)
        for i, j in edges:
            if i == node:
                indegree[j] -= 1
                if indegree[j] == 0:
                    ready = sorted(ready + [j])
    assert len(order) == len(nodes), "river graph must be acyclic"
    return tuple(order)


def check_edge_direction(graph: RiverGraph, edge: tuple[int, int], kind: str,
                         basis: str) -> None:
    """Downstream edges point downstream; the backwater link runs from the
    confluence gauge UP to the target. Edges touching a boundary gauge (no
    chainage) use the great-circle distance."""
    i, j = edge
    if kind == EDGE_DOWNSTREAM:
        assert is_downstream(i, j), f"edge {i}->{j} is not downstream"
    else:
        assert kind == EDGE_BACKWATER and edge == (BACKWATER_SOURCE, graph.target_id)
        assert is_downstream(j, i), f"backwater edge {i}->{j} must point upstream"
    if {i, j} & set(graph.boundary_ids):
        assert basis == "haversine", f"edge {i}->{j} must use the haversine distance"


def check_river_graph(graph: RiverGraph) -> None:
    """Edge directions, forward edges in the node order, 16 and 17 swept before
    the target, the target last and the only sink, every gauge drains to it."""
    order = {n: k for k, n in enumerate(graph.node_ids)}
    assert len(graph.edges) == len(set(graph.edges)), "duplicate edges"
    for edge, kind, basis in zip(graph.edges, graph.edge_kind, graph.edge_basis):
        check_edge_direction(graph, edge, kind, basis)
        assert order[edge[0]] < order[edge[1]], f"edge {edge} violates the sweep order"
    assert graph.node_ids[-1] == graph.target_id, "target must be visited last"
    assert all(order[b] < order[graph.target_id] for b in graph.boundary_ids)
    assert graph.edge_kind.count(EDGE_BACKWATER) == 1, "exactly one backwater link"
    assert not [e for e in graph.edges if e[0] == graph.target_id], "target is the sink"
    reached, frontier = {graph.target_id}, [graph.target_id]
    while frontier:
        node = frontier.pop()
        new = {i for i, j in graph.edges if j == node} - reached
        reached |= new
        frontier += sorted(new)
    assert reached == set(graph.node_ids), "every gauge must drain to the target"
    assert graph.physics_parent == PHYSICS_PARENT[graph.target_id]


def build_river_graph(target_id: int, upstream_ids: tuple[int, ...],
                      boundary_ids: tuple[int, ...]) -> RiverGraph:
    """RIVER_EDGES restricted to the target's gauges, the boundary edge 16 -> 17
    and the backwater link 17 -> target, with distance features."""
    nodes = set(upstream_ids) | set(boundary_ids) | {target_id}
    assert len(nodes) == len(upstream_ids) + len(boundary_ids) + 1
    graph_boundary = {n for e in BOUNDARY_EDGES for n in e} | {BACKWATER_SOURCE}
    assert set(boundary_ids) == graph_boundary, "file boundary points != graph's"
    down = [(i, j) for i, j in RIVER_EDGES + BOUNDARY_EDGES if i in nodes and j in nodes]
    kinds = {e: EDGE_DOWNSTREAM for e in down}
    kinds[(BACKWATER_SOURCE, target_id)] = EDGE_BACKWATER
    order = topological_order(nodes, list(kinds))
    edges = sorted(kinds, key=lambda e: (order.index(e[1]), order.index(e[0])))
    dist = [edge_distance_km(i, j) for i, j in edges]
    graph = RiverGraph(target_id, order, tuple(edges), tuple(d for d, _ in dist),
                       tuple(b for _, b in dist), tuple(kinds[e] for e in edges),
                       tuple(sorted(boundary_ids)))
    check_river_graph(graph)
    return graph


def graph_info(graph: RiverGraph) -> dict:
    """JSON-friendly description of a river graph (nodes in sweep order)."""
    return {"nodes": list(graph.node_ids), "n_nodes": len(graph.node_ids),
            "target": graph.target_id, "boundary_nodes": list(graph.boundary_ids),
            "physics_parent": graph.physics_parent,
            "edges": [{"from": i, "to": j, "kind": kind, "km": round(km, 3),
                       "basis": basis, "d_feature": round(km / EDGE_KM_SCALE, 5)}
                      for (i, j), km, basis, kind in zip(graph.edges, graph.edge_km,
                                                         graph.edge_basis,
                                                         graph.edge_kind)]}


def node_window_columns(node_id: int, graph: RiverGraph) -> list[str]:
    """Wide-matrix columns for steps t-7..t of one node (P<id>_ for upstream,
    B<id>_ for boundary gauges). The target has only t-7..t-1 (its step-t slot
    is masked), so its WL column is never read."""
    lags = range(N_LAGS, 0, -1)
    if node_id == graph.target_id:
        return [f"WLD-{k}" for k in lags]
    prefix = BOUNDARY if node_id in graph.boundary_ids else UPSTREAM
    return [f"{prefix}{node_id:02d}_WLD-{k}" for k in lags] + [f"{prefix}{node_id:02d}_WL"]


@dataclass(frozen=True)
class NodeScaler:
    """Train-only stage mean/std per node (pooled over its observed steps) and
    delta mean/std."""

    stage_mean: np.ndarray
    stage_std: np.ndarray
    y_mean: float
    y_std: float
    n_fit: int


def build_node_inputs(frame: pd.DataFrame, graph: RiverGraph,
                      train: np.ndarray) -> tuple[np.ndarray, NodeScaler]:
    """X (n, N, 8, 2); channel 0 standardised per node, channel 1 the mask.
    Unobserved slots (the target at step t) stay 0 in both channels."""
    x = np.zeros((len(frame), len(graph.node_ids), GNN_STEPS, GNN_CHANNELS))
    means, stds = [], []
    for k, node in enumerate(graph.node_ids):
        cols = node_window_columns(node, graph)
        check_features(frame, cols)
        vals = frame[cols].to_numpy(dtype=np.float64)
        means.append(float(vals[train].mean()))
        stds.append(float(vals[train].std()))
        x[:, k, :len(cols), 0] = (vals - means[-1]) / stds[-1]
        x[:, k, :len(cols), 1] = 1.0
    delta = (frame["WL"] - frame["WLD-1"]).to_numpy()
    scaler = NodeScaler(np.array(means), np.array(stds), float(delta[train].mean()),
                        float(delta[train].std()), int(train.sum()))
    return x, scaler


def check_node_inputs(x: np.ndarray, graph: RiverGraph) -> None:
    """Shape, finiteness and masking: the target's step-t slot is 0 for every row."""
    n_nodes, t = len(graph.node_ids), graph.pos(graph.target_id)
    assert x.shape[1:] == (n_nodes, GNN_STEPS, GNN_CHANNELS), "X must be N x 8 x 2"
    assert np.isfinite(x).all(), "node inputs must be finite"
    assert np.all(x[:, t, -1, :] == 0.0), "target WL_t slot must be masked (0, 0)"
    mask = np.ones((n_nodes, GNN_STEPS))
    mask[t, -1] = 0.0
    assert np.array_equal(x[..., 1], np.broadcast_to(mask, x.shape[:3])), "bad mask"


class RatingCurves(nn.Module):
    """Learnable Q_k(h) = a_k softplus(h - h0_k)^b_k with a_k = softplus(alpha_k)
    and b_k = 1 + softplus(beta_k), for k = [nearest upstream gauge u, target s]."""

    def __init__(self) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(2))
        self.beta = nn.Parameter(torch.zeros(2))
        self.h0 = nn.Parameter(torch.zeros(2))

    def forward(self, stage: torch.Tensor) -> torch.Tensor:
        """stage (B, 2) standardised [h_u, h_s] -> discharge-like Q (B, 2)."""
        a = nn.functional.softplus(self.alpha)
        b = 1.0 + nn.functional.softplus(self.beta)
        return a * nn.functional.softplus(stage - self.h0) ** b

    def summary(self) -> dict[str, list[float]]:
        """Learned a, b, h0 for [u, s]."""
        with torch.no_grad():
            return {"a": nn.functional.softplus(self.alpha).tolist(),
                    "b": (1.0 + nn.functional.softplus(self.beta)).tolist(),
                    "h0": self.h0.tolist()}


def incoming_edges(graph: RiverGraph) -> tuple[tuple[tuple[int, ...], int, int], ...]:
    """Per node position: (source positions, first edge index, end edge index)."""
    out = []
    for node in graph.node_ids:
        idx = [e for e, (_, j) in enumerate(graph.edges) if j == node]
        start, stop = (idx[0], idx[-1] + 1) if idx else (0, 0)
        assert idx == list(range(start, stop)), "edges must be grouped by destination"
        out.append((tuple(graph.pos(graph.edges[e][0]) for e in idx), start, stop))
    return tuple(out)


class PiStgnn(nn.Module):
    """Shared GRU encoder + one downstream message-passing sweep + target head."""

    def __init__(self, graph: RiverGraph) -> None:
        super().__init__()
        self.gru = nn.GRU(GNN_CHANNELS, GNN_HIDDEN, num_layers=1, batch_first=True)
        self.message = nn.Linear(2 * GNN_HIDDEN + 1, GNN_HIDDEN)
        self.head = nn.Sequential(nn.Linear(GNN_HIDDEN, GNN_HEAD_HIDDEN), nn.ReLU(),
                                  nn.Linear(GNN_HEAD_HIDDEN, 1))
        self.rating = RatingCurves()
        self.register_buffer("edge_d", torch.tensor(graph.edge_km) / EDGE_KM_SCALE)
        self.incoming = incoming_edges(graph)
        self.target_pos = graph.pos(graph.target_id)
        self.parent_pos = graph.pos(graph.physics_parent)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, 8, 2) -> last GRU hidden state per node (B, N, 32)."""
        b, n = x.shape[:2]
        _, h_n = self.gru(x.reshape(b * n, GNN_STEPS, GNN_CHANNELS))
        return h_n[-1].reshape(b, n, GNN_HIDDEN)

    def sweep(self, h: torch.Tensor) -> torch.Tensor:
        """One pass in river order; parents are already updated when j is visited."""
        states = list(h.unbind(dim=1))
        for j, (srcs, start, stop) in enumerate(self.incoming):
            if not srcs:
                continue
            h_i = torch.stack([states[i] for i in srcs], dim=1)          # (B, P, H)
            h_j = states[j].unsqueeze(1).expand_as(h_i)                   # (B, P, H)
            d_ij = self.edge_d[start:stop].to(h_i.dtype).view(1, -1, 1)
            d_ij = d_ij.expand(h_i.shape[0], -1, -1)                      # (B, P, 1)
            msg = torch.relu(self.message(torch.cat([h_i, h_j, d_ij], dim=-1)))
            states[j] = states[j] + msg.sum(dim=1)
        return torch.stack(states, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standardised delta_hat (B,)."""
        h = self.sweep(self.encode(x))
        return self.head(h[:, self.target_pos]).squeeze(-1)

    def physics_residual(self, x: torch.Tensor, delta_hat: torch.Tensor) -> torch.Tensor:
        """r = delta_hat - (Q_u(h_u,t) - Q_s(h_s,t-1)), stages from the inputs."""
        stage = torch.stack([x[:, self.parent_pos, -1, 0],
                             x[:, self.target_pos, -2, 0]], dim=1)
        q = self.rating(stage)
        return delta_hat - (q[:, 0] - q[:, 1])


def parameter_counts(model: PiStgnn) -> dict[str, int]:
    """Total, predictive (GRU + message + head) and rating-curve parameters."""
    total = sum(p.numel() for p in model.parameters())
    rating = sum(p.numel() for p in model.rating.parameters())
    return {"total": total, "predictive": total - rating, "rating_curves": rating}


def gnn_losses(model: PiStgnn, x: torch.Tensor, y: torch.Tensor,
               lam: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(total, mse, physics) losses; with lam = 0 the physics term is not used."""
    pred = model(x)
    mse = nn.functional.mse_loss(pred, y)
    phys = model.physics_residual(x, pred).pow(2).mean()
    total = mse + lam * phys if lam > 0 else mse
    return total, mse, phys


def gnn_train_epoch(model: PiStgnn, opt: torch.optim.Optimizer, x: torch.Tensor,
                    y: torch.Tensor, lam: float, gen: torch.Generator) -> np.ndarray:
    """One shuffled pass; mean [total, mse, physics] loss (NaN if it diverged)."""
    model.train()
    perm = torch.randperm(len(y), generator=gen).to(y.device)
    sums = np.zeros(3)
    for start in range(0, len(y), BATCH_SIZE):
        idx = perm[start:start + BATCH_SIZE]
        opt.zero_grad()
        total, mse, phys = gnn_losses(model, x[idx], y[idx], lam)
        vals = torch.stack([total, mse, phys]).detach().double().cpu().numpy()
        if not np.isfinite(vals[:2]).all():
            return np.full(3, np.nan)
        total.backward()
        opt.step()
        sums += vals * len(idx)
    return sums / len(y)


def gnn_predict(model: PiStgnn, x: torch.Tensor,
                scaler: NodeScaler) -> tuple[np.ndarray, float]:
    """Delta in metres and the mean squared physics residual (full batch)."""
    model.eval()
    with torch.no_grad():
        out = model(x)
        phys = float(model.physics_residual(x, out).pow(2).mean())
    return out.double().cpu().numpy() * scaler.y_std + scaler.y_mean, phys


def fit_gnn(graph: RiverGraph, data: dict[str, dict[str, torch.Tensor]],
            d_val: np.ndarray, scaler: NodeScaler, seed: int, lam: float,
            cfg: RunConfig) -> tuple[PiStgnn, list[dict], bool]:
    """Adam with early stopping on validation delta RMSE (m); best weights
    restored. Returns the model, per-epoch history and a divergence flag."""
    set_seed(seed)
    tr, va = data["train"], data["val"]
    model = PiStgnn(graph).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=GNN_LR)
    gen = torch.Generator().manual_seed(seed)
    best_rmse, best_state, wait, history, diverged = np.inf, None, 0, [], False
    for epoch in range(1, cfg.gnn_max_epochs + 1):
        loss = gnn_train_epoch(model, opt, tr["x"], tr["y"], lam, gen)
        diverged = not np.isfinite(loss[:2]).all()
        if diverged:
            break  # keep the best weights seen so far
        d_hat, val_phys = gnn_predict(model, va["x"], scaler)
        val_rmse = float(np.sqrt(np.mean((d_hat - d_val) ** 2)))
        improved = val_rmse < best_rmse
        history.append({"lambda": lam, "epoch": epoch, "train_loss": loss[0],
                        "train_mse_scaled": loss[1], "train_phys": loss[2],
                        "val_mse_scaled": float(np.mean(((d_hat - d_val) / scaler.y_std) ** 2)),
                        "val_phys": val_phys, "val_rmse_m": val_rmse, "improved": improved})
        if improved:
            best_rmse, wait = val_rmse, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg.gnn_patience:
                break
    assert best_state is not None, "GNN diverged in its first epoch"
    model.load_state_dict(best_state)
    return model, history, diverged


def prepare_gnn_data(td: TargetData, masks: dict[str, np.ndarray], graph: RiverGraph,
                     cfg: RunConfig) -> tuple[dict[str, dict[str, torch.Tensor]], NodeScaler]:
    """Node tensors and standardised delta per split, with leakage checks."""
    x, scaler = build_node_inputs(td.frame, graph, masks["train"])
    assert scaler.n_fit == int(masks["train"].sum()), "GNN scaler must see train rows only"
    check_node_inputs(x, graph)
    delta = (td.frame["WL"] - td.frame["WLD-1"]).to_numpy()
    y_s = (delta - scaler.y_mean) / scaler.y_std
    data = {s: {"x": to_tensor(x[m], cfg.device), "y": to_tensor(y_s[m], cfg.device)}
            for s, m in masks.items()}
    t = graph.pos(graph.target_id)
    for split in data.values():  # the model's input tensors never hold WL_t
        assert bool((split["x"][:, t, -1, :] == 0).all()), "WL_t leaked into X"
    return data, scaler


def upstream_reach(model: PiStgnn, xb: torch.Tensor, graph: RiverGraph) -> dict[str, list]:
    """Perturb each node's inputs: every other node (upstream gauges, and the
    boundary gauges via the backwater link) must change the target's state;
    the target must change no other node's state (it is the sink)."""
    t = graph.pos(graph.target_id)
    with torch.no_grad():
        base = model.sweep(model.encode(xb))
        reaches, leaks = [], []
        for k, node in enumerate(graph.node_ids):
            xp = xb.clone()
            xp[:, k, :, 0] += 1.0
            diff = (model.sweep(model.encode(xp)) - base).abs().amax(dim=(0, 2))
            if k == t:
                leaks = [n for n, d in zip(graph.node_ids, diff.tolist()) if n != node and d > 0]
            elif float(diff[t]) > 0:
                reaches.append(node)
    assert set(reaches) == set(graph.node_ids) - {graph.target_id}, "unreached gauge"
    assert not leaks, "target state leaked upstream"
    return {"upstream_nodes_reaching_target": reaches, "nodes_changed_by_target": leaks}


def gnn_sanity_checks(graph: RiverGraph, train: dict[str, torch.Tensor],
                      device: str) -> dict:
    """Shapes, masking, reachability, finite loss and gradient flow to the rating
    curves on one batch with a throw-away model (every fit re-seeds, so the
    reported models are unaffected)."""
    n_nodes, t = len(graph.node_ids), graph.pos(graph.target_id)
    xb, yb = train["x"][:BATCH_SIZE], train["y"][:BATCH_SIZE]
    model = PiStgnn(graph).to(device)
    assert tuple(xb.shape[1:]) == (n_nodes, GNN_STEPS, GNN_CHANNELS)
    assert bool((xb[:, t, -1, :] == 0).all()), "target WL_t slot must be masked"
    h_gru = model.encode(xb)
    h_sweep = model.sweep(h_gru)
    assert tuple(h_gru.shape[1:]) == tuple(h_sweep.shape[1:]) == (n_nodes, GNN_HIDDEN)
    reach = upstream_reach(model, xb, graph)
    total, mse, phys = gnn_losses(model, xb, yb, PHYSICS_LAMBDA)
    assert bool(torch.isfinite(total)), "loss must be finite"
    total.backward()
    grads = {name: float(p.grad.abs().sum()) for name, p in model.rating.named_parameters()}
    assert all(np.isfinite(g) and g > 0 for g in grads.values()), "no gradient to rating"
    return {"n_nodes": n_nodes, "sweep_order": list(graph.node_ids),
            "input_shape_per_sample": list(xb.shape[1:]),
            "gru_state_shape_per_sample": list(h_gru.shape[1:]),
            "sweep_state_shape_per_sample": list(h_sweep.shape[1:]),
            "target_t_slot_zero_all_rows": True, **reach,
            "batch_loss": {"total": total.item(), "mse": mse.item(), "phys": phys.item()},
            "rating_grad_abs_sum": grads}


def run_gnn(td: TargetData, masks: dict[str, np.ndarray], cfg: RunConfig
            ) -> tuple[dict[str, dict[int, dict[str, np.ndarray]]], pd.DataFrame, dict]:
    """Train every variant (with / without physics) once per seed."""
    graph = build_river_graph(td.target_id, td.upstream_ids, td.boundary_ids)
    data, scaler = prepare_gnn_data(td, masks, graph, cfg)
    info = {"graph": graph_info(graph), "checks": gnn_sanity_checks(graph, data["train"],
                                                                      cfg.device),
            "stage_mean": dict(zip(map(str, graph.node_ids), scaler.stage_mean.tolist())),
            "stage_std": dict(zip(map(str, graph.node_ids), scaler.stage_std.tolist())),
            "delta_mean": scaler.y_mean, "delta_std": scaler.y_std, "fits": {}}
    delta = (td.frame["WL"] - td.frame["WLD-1"]).to_numpy()
    base = td.frame["WLD-1"].to_numpy()
    preds, hist = {name: {} for name, _ in GNN_VARIANTS}, []
    for name, lam in GNN_VARIANTS:
        for seed in cfg.gnn_seeds:
            t0 = time.time()
            model, history, diverged = fit_gnn(graph, data, delta[masks["val"]], scaler,
                                               seed, lam, cfg)
            preds[name][seed] = {s: base[masks[s]] + gnn_predict(model, data[s]["x"], scaler)[0]
                                 for s in ("val", "test")}
            best = min(history, key=lambda h: h["val_rmse_m"])
            info["n_parameters"] = parameter_counts(model)
            info["fits"][f"{name}/{seed}"] = {
                "lambda": lam, "best_epoch": best["epoch"], "epochs_run": len(history),
                "best_val_rmse_m": best["val_rmse_m"], "diverged": diverged,
                "rating_curves_u_s": model.rating.summary(),
                "runtime_s": round(time.time() - t0, 1)}
            hist += [{"target": td.name, "model": name, "seed": seed, **h} for h in history]
            print(f"    {td.name} {name} seed {seed}: best epoch {best['epoch']}/"
                  f"{len(history)}, val RMSE {best['val_rmse_m']:.4f} m", flush=True)
    return preds, pd.DataFrame(hist), info


def gnn_results(td: TargetData, masks: dict[str, np.ndarray], cfg: RunConfig
                ) -> tuple[list[dict], dict[str, np.ndarray], pd.DataFrame, dict]:
    """PI-STGNN and its no-physics ablation: metric rows, seed-averaged test
    predictions per variant, history and fit info."""
    t0 = time.time()
    preds, hist, info = run_gnn(td, masks, cfg)
    rows, test_pred = [], {}
    for name, _ in GNN_VARIANTS:
        model_rows, avg = seeded_model_rows(td, masks, name, preds[name])
        rows += model_rows
        test_pred[name] = avg["test"]
    info["runtime_s"] = round(time.time() - t0, 1)
    return rows, test_pred, hist, info


# --------------------------------------------------------------------------- runs
def metric_rows(target: str, model: str, seed: str,
                mets: dict[str, dict[str, float]]) -> list[dict]:
    """Tidy rows (one per split) for metrics.csv."""
    return [{"target": target, "model": model, "split": s, "seed": seed, **m}
            for s, m in mets.items()]


def seed_summary_rows(target: str, model: str, per_seed: dict[int, dict]) -> list[dict]:
    """Mean and sample SD across seeds for every metric of a seeded model."""
    rows = []
    for split in ("val", "test"):
        tab = pd.DataFrame([m[split] for m in per_seed.values()])
        mean = tab.mean()
        sd = tab.std(ddof=1) if len(tab) > 1 else mean * np.nan
        for label, stats in (("mean", mean), ("sd", sd)):
            row = {k: float(stats[k]) for k in ("rmse", "mae", "nse", "pi", "rmse_monsoon")}
            row.update(n=int(tab["n"].iloc[0]), n_monsoon=int(tab["n_monsoon"].iloc[0]))
            rows.append({"target": target, "model": model, "split": split,
                         "seed": label, **row})
    return rows


def seeded_model_rows(td: TargetData, masks: dict[str, np.ndarray], model: str,
                      preds: dict[int, dict[str, np.ndarray]]
                      ) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Metric rows of a seeded model (per seed, mean, sd, seed-averaged
    prediction) and its seed-averaged WL predictions per split."""
    per_seed = {s: split_metrics(td.frame, masks, p) for s, p in preds.items()}
    rows = [r for seed, m in per_seed.items()
            for r in metric_rows(td.name, model, str(seed), m)]
    rows += seed_summary_rows(td.name, model, per_seed)
    avg = {s: np.mean([p[s] for p in preds.values()], axis=0) for s in ("val", "test")}
    rows += metric_rows(td.name, model, "seed_avg_prediction",
                        split_metrics(td.frame, masks, avg))
    return rows, avg


def persistence_check(frame: pd.DataFrame, masks: dict[str, np.ndarray],
                      mets: dict[str, dict[str, float]]) -> None:
    """Persistence must have PI = 0 and RMSE = sqrt(mean((WL - WLD-1)^2))."""
    for split, m in mets.items():
        rows = frame[masks[split]]
        direct = float(np.sqrt(np.mean((rows["WL"] - rows["WLD-1"]) ** 2)))
        assert abs(m["pi"]) < 1e-12 and abs(m["rmse"] - direct) < 1e-12


def lstm_results(td: TargetData, masks: dict[str, np.ndarray],
                 cfg: RunConfig) -> tuple[list[dict], np.ndarray, pd.DataFrame, dict]:
    """LSTM over all seeds: metric rows (per seed, mean, sd, seed-averaged
    prediction), the seed-averaged test prediction, history and fit info."""
    preds, hist, info = run_lstm(td, masks, cfg)
    rows, avg = seeded_model_rows(td, masks, "lstm", preds)
    return rows, avg["test"], hist, info


def run_target(td: TargetData, cfg: RunConfig) -> dict:
    """Main experiment for one target: all models, metrics, predictions."""
    frame, masks = td.frame, split_masks(td.frame["Date"])
    pers = {s: frame.loc[masks[s], "WLD-1"].to_numpy() for s in ("val", "test")}
    pers_metrics = split_metrics(frame, masks, pers)
    persistence_check(frame, masks, pers_metrics)
    tab = run_tabular(frame, td_full_columns(td), masks, cfg)
    rows = metric_rows(td.name, "persistence", "none", pers_metrics)
    test_pred = {"persistence": pers["test"]}
    for name in ("ridge", "lightgbm"):
        mets = split_metrics(frame, masks, tab[name]["wl_hat"])
        rows += metric_rows(td.name, name, str(SEED), mets)
        test_pred[name] = tab[name]["wl_hat"]["test"]
    info = {"ridge": tab["ridge"]["info"], "lightgbm": tab["lightgbm"]["info"]}
    hist = pd.DataFrame()
    if not cfg.skip_lstm:
        lstm_rows, test_pred["lstm"], hist, info["lstm"] = lstm_results(td, masks, cfg)
        rows += lstm_rows
    gnn_hist = pd.DataFrame()
    if not cfg.skip_gnn:
        gnn_rows, gnn_pred, gnn_hist, info["gnn"] = gnn_results(td, masks, cfg)
        rows += gnn_rows
        test_pred.update(gnn_pred)
    test = frame.loc[masks["test"], ["Date", "WL"]].rename(columns={"WL": "WL_obs"})
    predictions = test.assign(target=td.name, **test_pred)
    predictions = predictions[["Date", "target", "WL_obs", *test_pred]]
    assert len(predictions) == int(masks["test"].sum())
    return {"metrics": rows, "predictions": predictions, "history": hist, "info": info,
            "gnn_history": gnn_hist,
            "importance": importance_by_point(tab["lightgbm"]["model"], td),
            "counts": {s: int(m.sum()) for s, m in masks.items()}}


def ablation_rows(td: TargetData, set_id: str, n_features: int, frame: pd.DataFrame,
                  masks: dict[str, np.ndarray], tab: dict[str, dict]) -> list[dict]:
    """Tidy ablation rows (model x split) for one feature set."""
    rows = []
    for name in ("ridge", "lightgbm"):
        extra = {"ridge_alpha": tab["ridge"]["info"]["alpha"] if name == "ridge" else np.nan,
                 "lgbm_best_iteration": (tab["lightgbm"]["info"]["best_iteration"]
                                         if name == "lightgbm" else np.nan)}
        for split, m in split_metrics(frame, masks, tab[name]["wl_hat"]).items():
            rows.append({"target": td.name, "feature_set": set_id,
                         "description": ABLATION_SETS[set_id], "model": name,
                         "n_features": n_features, "split": split, **m, **extra})
    return rows


def run_ablation(td: TargetData, cfg: RunConfig) -> tuple[list[dict], dict]:
    """Feature sets A-G on the rows where the target's own Rainfall_Level exists."""
    frame = td.frame[td.frame[RAIN_COL].notna()].reset_index(drop=True)
    masks = split_masks(frame["Date"])
    pers = {s: frame.loc[masks[s], "WLD-1"].to_numpy() for s in ("val", "test")}
    rows = [{"target": td.name, "feature_set": "-", "description": "persistence reference",
             "model": "persistence", "n_features": 0, "split": s, **m}
            for s, m in split_metrics(frame, masks, pers).items()]
    info = {"n_rows": len(frame), "counts": {s: int(m.sum()) for s, m in masks.items()}}
    for set_id in ABLATION_SETS:
        cols = ablation_columns(set_id, td.upstream_ids, td.boundary_ids)
        tab = run_tabular(frame, cols, masks, cfg)
        rows += ablation_rows(td, set_id, len(cols), frame, masks, tab)
        info[set_id] = {"ridge_alpha": tab["ridge"]["info"]["alpha"],
                        "lgbm_best_iteration": tab["lightgbm"]["info"]["best_iteration"]}
    return rows, info


# --------------------------------------------------------------------------- main
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line options (unknown arguments, e.g. from a notebook, are ignored)."""
    parser = argparse.ArgumentParser(description="Ganges-Padma WL baseline models")
    parser.add_argument("--quick", action="store_true", help="small budgets (smoke test)")
    parser.add_argument("--skip-lstm", action="store_true", help="skip the LSTM")
    parser.add_argument("--skip-gnn", action="store_true",
                        help="skip PI-STGNN and its no-physics ablation")
    parser.add_argument("--skip-ablation", action="store_true", help="skip the ablation")
    parser.add_argument("--data-dir", type=Path, default=None, help="search root for CSVs")
    parser.add_argument("--out", type=Path, default=None, help="output directory")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args, _unknown = parser.parse_known_args(argv)
    return args


def make_config(args: argparse.Namespace) -> RunConfig:
    """Build the run configuration from the command line."""
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = RunConfig(skip_lstm=args.skip_lstm, skip_gnn=args.skip_gnn,
                    skip_ablation=args.skip_ablation, device=device)
    return replace(cfg, **QUICK_OVERRIDES) if args.quick else cfg


def environment(cfg: RunConfig, args: argparse.Namespace, files: dict[str, Path]) -> dict:
    """Library versions, hardware and run settings for environment.json."""
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__,
        "lightgbm": lgb.__version__, "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(), "gpu_name": gpu,
        "gpu_count": torch.cuda.device_count(), "lstm_device": cfg.device,
        "on_kaggle": on_kaggle(), "quick": bool(args.quick), "config": cfg.__dict__,
        "seed": SEED, "lstm_seeds": list(cfg.lstm_seeds),
        "split": {"train": f"Date < {TRAIN_END.date()}",
                  "val": f"{TRAIN_END.date()} <= Date < {TEST_START.date()}",
                  "test": f"Date >= {TEST_START.date()}"},
        "monsoon_months": list(MONSOON_MONTHS), "ridge_alphas": list(RIDGE_ALPHAS),
        "lgbm_params": LGBM_PARAMS,
        "lstm": {"hidden": LSTM_HIDDEN, "head_hidden": HEAD_HIDDEN, "dropout": HEAD_DROPOUT,
                 "lr": LSTM_LR, "batch_size": BATCH_SIZE},
        "gnn": gnn_environment(cfg, args),
        "data_files": {k: {"path": str(p), "sha256": sha256(p)} for k, p in files.items()},
    }


def gnn_environment(cfg: RunConfig, args: argparse.Namespace) -> dict:
    """PI-STGNN settings for environment.json (per-target graphs, best epochs and
    parameter counts are under models.<target>.gnn)."""
    return {"seeds": list(cfg.gnn_seeds), "hidden": GNN_HIDDEN,
            "head_hidden": GNN_HEAD_HIDDEN, "steps": GNN_STEPS, "channels": GNN_CHANNELS,
            "lr": GNN_LR, "batch_size": BATCH_SIZE, "max_epochs": cfg.gnn_max_epochs,
            "patience": cfg.gnn_patience, "physics_lambda": PHYSICS_LAMBDA,
            "variants": {name: lam for name, lam in GNN_VARIANTS},
            "river_edges": [list(e) for e in RIVER_EDGES],
            "boundary_edges": [list(e) for e in BOUNDARY_EDGES],
            "backwater_link": f"{BACKWATER_SOURCE} -> target",
            "physics_parent": {str(k): v for k, v in PHYSICS_PARENT.items()},
            "edge_km_scale": EDGE_KM_SCALE,
            "station_geo_check": verify_station_geo(args.data_dir)}


def write_outputs(out: Path, results: dict[str, dict], ablation: list[dict]) -> None:
    """Write the tidy CSV outputs."""
    out.mkdir(parents=True, exist_ok=True)
    res = list(results.values())
    pd.DataFrame([r for x in res for r in x["metrics"]]).to_csv(out / "metrics.csv", index=False)
    pd.concat([x["predictions"] for x in res]).to_csv(out / "predictions_test.csv", index=False)
    pd.concat([x["importance"] for x in res]).to_csv(out / "importance_by_point.csv", index=False)
    pd.concat([x["history"] for x in res]).to_csv(out / "lstm_history.csv", index=False)
    gnn_hist = [x["gnn_history"] for x in res if not x["gnn_history"].empty]
    if gnn_hist:
        pd.concat(gnn_hist).to_csv(out / "gnn_history.csv", index=False)
    if ablation:
        pd.DataFrame(ablation).to_csv(out / "ablation.csv", index=False)


def main(argv: list[str] | None = None) -> None:
    """Run every target, the ablation, and write all outputs."""
    t0 = time.time()
    args = parse_args(argv)
    cfg = make_config(args)
    set_seed(SEED)
    configure_torch_determinism()
    default_out = Path("/kaggle/working") if on_kaggle() else Path.cwd() / "results"
    out = args.out or default_out
    files = {"long": find_data_file(LONG_FILE, args.data_dir)}
    files.update({name: find_data_file(f, args.data_dir) for _, name, f in TARGETS})
    long = pd.read_csv(files["long"], parse_dates=["Date"])
    env = {**environment(cfg, args, files), "row_counts": {}, "features": {},
           "models": {}, "ablation": {}}
    env["gnn"]["graphs"] = {}
    results, ablation, timings = {}, [], {}
    for target_id, name, _ in TARGETS:
        td = load_target(files[name], long, target_id, name)
        env["features"][name] = {"upstream_ids": list(td.upstream_ids),
                                 "boundary_ids": list(td.boundary_ids),
                                 "n_predictors_main": len(td_full_columns(td))}
        env["gnn"]["graphs"][name] = graph_info(
            build_river_graph(target_id, td.upstream_ids, td.boundary_ids))
        t1 = time.time()
        results[name] = run_target(td, cfg)
        timings[f"{name}_main_s"] = round(time.time() - t1, 1)
        if "gnn" in results[name]["info"]:
            timings[f"{name}_gnn_s"] = results[name]["info"]["gnn"]["runtime_s"]
        env["row_counts"][name] = {"total": len(td.frame), **results[name]["counts"]}
        env["models"][name] = results[name]["info"]
        print(f"[{time.time() - t0:7.1f}s] {name}: main models done", flush=True)
        if not cfg.skip_ablation:
            t1 = time.time()
            rows, env["ablation"][name] = run_ablation(td, cfg)
            ablation += rows
            timings[f"{name}_ablation_s"] = round(time.time() - t1, 1)
            print(f"[{time.time() - t0:7.1f}s] {name}: ablation done", flush=True)
    write_outputs(out, results, ablation)
    env["timings"] = {**timings, "total_runtime_s": round(time.time() - t0, 1)}
    env_text = json.dumps(env, indent=2, default=str)
    (out / "environment.json").write_text(env_text, encoding="utf-8")
    test_metrics = pd.read_csv(out / "metrics.csv").query("split == 'test'")
    print(test_metrics.round(4).to_string(index=False))
    print(f"Done in {time.time() - t0:.1f}s; outputs in {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
