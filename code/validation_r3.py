"""R3 validation run: one entry point for every experiment behind the CSE 4112
report's Section 10 (teacher revision R3).

    python code/validation_r3.py --exp all             # everything, Kaggle GPU
    python code/validation_r3.py --exp all --smoke      # 1 epoch, 1 seed, 2 targets
    python code/validation_r3.py --exp e2               # one experiment only

Reuses ``train_baselines.py`` (data loading, feature-column helpers, Ridge/
LightGBM/LSTM/PI-STGNN fitting, the river graph, metrics, seeded-model rows)
without copying it; adds only what R3 needs on top:

  M2 T-GCN     a second, independent graph baseline (Zhao et al. 2019: a graph
               convolution per timestep, normalised river adjacency, then a
               shared GRU over time), implemented in plain PyTorch on the same
               node tensors PI-STGNN already uses -- no torch_geometric_temporal.
  Bootstrap CI 30-day moving-block bootstrap (2000 resamples) for PI and for
               paired Delta-PI between two settings, on test days only.
  ARFF export  every Ridge (M1) split, every experiment, for an independent
               Weka cross-check (``python-weka-wrapper3`` is attempted on
               Kaggle; skipped with a logged reason if the JVM install fails).
  E2 lag depth Extends the released 7-day lag layout to k in {3, 7, 10} for
               Bhagyakul by computing WLD-8..10 from the long panel with the
               same calendar-reindex, no-interpolation rule as build_dataset.py.
  E3 upstream  Pooled monsoon cross-correlation (Panka -> every main-stem
               travel time  gauge) plus event-based arrival lags from detected Panka
               surges, on validation-only choices; a Farakka case study is
               folded in from an externally verified, cited date (see
               FARAKKA_CANDIDATES), kept only if Panka shows a matching surge.
  E4/E5 spans  Peak-anchored, nested test windows (test_end = peak + 15 d);
               predictor stations with >5% missing days inside the window
               union are dropped from every span and every model alike.

Every experiment predicts delta_t = WL_t - WLD-1 and reconstructs
WL_hat_t = WLD-1 + delta_hat_t; the target's own day-t level is never an input
(train_baselines.check_features is reused everywhere). Scaling is fit on
training rows only. Neural models use 3 seeds (20260913-15) everywhere,
Adam lr 1e-3, batch 64, <= 300 epochs, patience 30, minimum 30 epochs.

Outputs, all under ``results/r3/`` (or ``--out``):
    metrics/e*.csv, preds/e*.csv, formats/e*.csv, arff/*.arff, tables/*.tex,
    macros.tex, run_manifest.json
"""
from __future__ import annotations

import argparse
import contextlib
import json
import re
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

# E4/E5's short validation carve-outs can legitimately contain zero monsoon-month
# (Jul-Oct) days, so train_baselines.compute_metrics' rmse_monsoon divides over an
# empty slice and reports NaN -- correct ("no monsoon days here"), not a bug.
warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide",
                        category=RuntimeWarning)

def _train_baselines_dir() -> Path:
    """This script's own directory (local runs), else wherever a Kaggle
    dataset attached train_baselines.py under /kaggle/input."""
    local = Path(__file__).resolve().parent
    if (local / "train_baselines.py").exists():
        return local
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        hits = sorted(kaggle_input.glob("**/train_baselines.py"))
        if hits:
            return hits[0].parent
    raise FileNotFoundError("train_baselines.py not found next to this script or under /kaggle/input")


sys.path.insert(0, str(_train_baselines_dir()))
import train_baselines as tb  # noqa: E402

# --------------------------------------------------------------------------- constants
OUT_DEFAULT = "results/r3"
SEEDS = (20260913, 20260914, 20260915)          # every neural model, every experiment
MIN_EPOCHS = 30                                  # floor under tb's early stopping
MIN_VAL_ROWS = 21                                # floor for E4/E5's validation carve-out
VAL_FRACTION = 0.15                              # of each train window, for E4/E5
BOOT_RESAMPLES = 2000
BOOT_BLOCK_DAYS = 30
GAP_MISSING_FRAC = 0.05                          # E4/E5: drop a predictor above this
MODEL_ORDER = ("ridge", "tgcn", "lstm", "custom")  # M1..M4, report order everywhere
MODEL_LABEL = {"ridge": "Ridge (M1, classical)", "tgcn": "T-GCN (M2)",
               "lstm": "LSTM (M3)", "custom": None}  # custom label set at runtime
BHAGYAKUL, BARURIA, SURESWAR = 12, 11, 15
PEAKS = {  # target Id -> (date, stage_m); verified from the long panel, not assumed
    BHAGYAKUL: ("2020-07-27", 6.35), BARURIA: ("2017-08-18", 8.93),
    SURESWAR: ("2021-09-07", 4.01)}
SPANS = (  # name, test_days, train_days
    ("S6", 31, 151), ("S12", 92, 274), ("S36", 366, 730))
MAIN_STEM_GAUGES = (2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 15)  # Id 1 (Panka) is the source
# Farakka Barrage gate-opening dates found by web search (BWDB itself ships no gate
# records); kept only if Panka shows a matching surge within +-2 d (checked at runtime).
# Source: The Daily Star, "Farakka: gates opened as water level rises",
# https://www.thedailystar.net (archived reporting on monsoon gate operations); dates
# are the ones with an unambiguous calendar date in the reporting.
FARAKKA_CANDIDATES: tuple[tuple[str, str], ...] = (
    # India opened Farakka's gates 24 Aug 2024 as Bihar/Jharkhand flooded, with all
    # 109 gates confirmed open by 26 Aug 2024 (multiply corroborated: The Business
    # Standard, "India opens 109 gates of Farakka Barrage", 26 Aug 2024,
    # https://www.tbsnews.net/bangladesh/india-opens-109-gates-farakka-barrage-926486;
    # Prothom Alo, same date, https://en.prothomalo.com/bangladesh/c01d95ncvw).
    ("2024-08-26", "The Business Standard, 26 Aug 2024"),
)


@contextlib.contextmanager
def lag_depth_leakage_check(k: int):
    """Widen train_baselines' leakage-check regex/own-lag list from WLD-1..7 to
    WLD-1..k for the duration of an E2 call, then restore it exactly. Every
    other leakage rule in check_features (no WL/Id/Date, no column of the
    target itself, no column literally equal to WL) is unchanged and still
    enforced at depth k."""
    old_lag_cols, old_block_re = tb.LAG_COLS, tb.BLOCK_RE
    tb.LAG_COLS = [f"WLD-{j}" for j in range(1, k + 1)]
    tb.BLOCK_RE = re.compile(rf"([PB])(\d{{2}})_(WL|WLD-(?:[1-9]|10))")
    try:
        yield
    finally:
        tb.LAG_COLS, tb.BLOCK_RE = old_lag_cols, old_block_re


# --------------------------------------------------------------------------- generalised split
def custom_split_masks(dates: pd.Series, train_start: pd.Timestamp, train_end: pd.Timestamp,
                       test_start: pd.Timestamp, test_end: pd.Timestamp,
                       val_fraction: float = VAL_FRACTION,
                       min_val_days: int = MIN_VAL_ROWS) -> dict[str, np.ndarray]:
    """Chronological train/val/test masks over an arbitrary window: val is the
    last ``val_fraction`` of [train_start, train_end), at least ``min_val_days``
    calendar days; never touches test. Rows outside [train_start, test_end] are
    excluded from every mask (as if they did not exist for this experiment)."""
    train_len = (train_end - train_start).days
    val_days = max(min_val_days, round(train_len * val_fraction))
    val_start = train_end - pd.Timedelta(days=val_days)
    assert val_start > train_start, "train window too short for the minimum validation slice"
    masks = {
        "train": ((dates >= train_start) & (dates < val_start)).to_numpy(),
        "val": ((dates >= val_start) & (dates < train_end)).to_numpy(),
        "test": ((dates >= test_start) & (dates < test_end)).to_numpy(),
    }
    for split in ("train", "val", "test"):
        assert masks[split].any(), f"{split} window has no rows"
    assert dates[masks["train"]].max() < dates[masks["val"]].min()
    assert dates[masks["val"]].max() < test_start
    return masks


# --------------------------------------------------------------------------- bootstrap CIs
def moving_block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """One moving-block bootstrap resample of row indices 0..n-1 (length n)."""
    n_blocks = -(-n // block)
    starts = rng.integers(0, max(1, n - block + 1), size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
    return np.clip(idx, 0, n - 1)


def bootstrap_pi_ci(obs: np.ndarray, pred: np.ndarray, pers: np.ndarray, *,
                    block: int = BOOT_BLOCK_DAYS, resamples: int = BOOT_RESAMPLES,
                    seed: int = 20260913) -> tuple[float, float, float]:
    """(point PI, 2.5th pct, 97.5th pct) from a moving-block bootstrap of test days."""
    rng = np.random.default_rng(seed)
    n = len(obs)
    point = 1.0 - np.sum((pred - obs) ** 2) / np.sum((pers - obs) ** 2)
    boots = np.empty(resamples)
    for b in range(resamples):
        idx = moving_block_indices(n, block, rng)
        sse_m, sse_p = np.sum((pred[idx] - obs[idx]) ** 2), np.sum((pers[idx] - obs[idx]) ** 2)
        boots[b] = 1.0 - sse_m / sse_p
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def bootstrap_delta_pi_ci(obs: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
                          pers: np.ndarray, *, block: int = BOOT_BLOCK_DAYS,
                          resamples: int = BOOT_RESAMPLES,
                          seed: int = 20260913) -> tuple[float, float, float]:
    """Paired bootstrap CI of PI(a) - PI(b), same resample used for both models."""
    rng = np.random.default_rng(seed)
    n = len(obs)
    sse_p_full = np.sum((pers - obs) ** 2)
    point = (1.0 - np.sum((pred_a - obs) ** 2) / sse_p_full
             - (1.0 - np.sum((pred_b - obs) ** 2) / sse_p_full))
    boots = np.empty(resamples)
    for b in range(resamples):
        idx = moving_block_indices(n, block, rng)
        sse_p = np.sum((pers[idx] - obs[idx]) ** 2)
        pi_a = 1.0 - np.sum((pred_a[idx] - obs[idx]) ** 2) / sse_p
        pi_b = 1.0 - np.sum((pred_b[idx] - obs[idx]) ** 2) / sse_p
        boots[b] = pi_a - pi_b
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(point), float(lo), float(hi)


# --------------------------------------------------------------------------- M2: T-GCN
class TGCN(nn.Module):
    """Zhao et al. (2019): a graph convolution at every timestep (shared,
    symmetric-normalised river adjacency), then a GRU over time (shared across
    nodes), then a linear head on the target node's final state. Same input
    tensor (B, N, T, C) and masking convention as PiStgnn; no external graph
    library, only a dense N x N matmul per timestep."""

    def __init__(self, n_nodes: int, target_pos: int, a_hat: torch.Tensor,
                gc_hidden: int = 32, gru_hidden: int = 32) -> None:
        super().__init__()
        self.register_buffer("a_hat", a_hat)  # (N, N), symmetric-normalised, self-loops
        self.target_pos = target_pos
        self.gc = nn.Linear(tb.GNN_CHANNELS, gc_hidden)
        self.gru = nn.GRU(gc_hidden, gru_hidden, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Linear(gru_hidden, tb.GNN_HEAD_HIDDEN), nn.ReLU(),
                                  nn.Linear(tb.GNN_HEAD_HIDDEN, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, T, C) -> standardised delta_hat (B,)."""
        b, n, t, c = x.shape
        xg = torch.einsum("nm,bmtc->bntc", self.a_hat, x)          # graph conv (spatial)
        xg = torch.relu(self.gc(xg))                                # (B, N, T, gc_hidden)
        xg = xg.reshape(b * n, t, -1)
        _, h_n = self.gru(xg)                                       # temporal, per node
        h = h_n[-1].reshape(b, n, -1)
        return self.head(h[:, self.target_pos]).squeeze(-1)


def normalised_adjacency(graph: "tb.RiverGraph") -> torch.Tensor:
    """Symmetric-normalised adjacency D^-1/2 (A + I) D^-1/2 over the graph's
    nodes (undirected: every river/backwater edge counted both ways)."""
    n = len(graph.node_ids)
    a = np.eye(n)
    for i, j in graph.edges:
        pi, pj = graph.pos(i), graph.pos(j)
        a[pi, pj] = a[pj, pi] = 1.0
    d = a.sum(axis=1)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(d))
    a_hat = d_inv_sqrt @ a @ d_inv_sqrt
    return torch.as_tensor(a_hat, dtype=torch.float32)


def fit_tgcn(graph: "tb.RiverGraph", data: dict[str, dict[str, torch.Tensor]],
            d_val: np.ndarray, scaler: "tb.NodeScaler", seed: int,
            cfg: "tb.RunConfig") -> tuple[TGCN, list[dict]]:
    """Same Adam/early-stopping recipe as fit_gnn, with a floor of MIN_EPOCHS."""
    tb.set_seed(seed)
    tr, va = data["train"], data["val"]
    a_hat = normalised_adjacency(graph).to(cfg.device)
    model = TGCN(len(graph.node_ids), graph.pos(graph.target_id), a_hat).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=tb.GNN_LR)
    gen = torch.Generator().manual_seed(seed)
    best_rmse, best_state, wait, history = np.inf, None, 0, []
    for epoch in range(1, cfg.gnn_max_epochs + 1):
        model.train()
        perm = torch.randperm(len(tr["y"]), generator=gen).to(tr["y"].device)
        train_loss = 0.0
        for start in range(0, len(perm), tb.BATCH_SIZE):
            idx = perm[start:start + tb.BATCH_SIZE]
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(tr["x"][idx]), tr["y"][idx])
            loss.backward()
            opt.step()
            train_loss += float(loss.item()) * len(idx)
        train_loss /= len(perm)
        model.eval()
        with torch.no_grad():
            d_hat = (model(va["x"]).double().cpu().numpy() * scaler.y_std + scaler.y_mean)
        val_rmse = float(np.sqrt(np.mean((d_hat - d_val) ** 2)))
        improved = val_rmse < best_rmse
        history.append({"epoch": epoch, "train_mse_scaled": train_loss,
                        "val_rmse_m": val_rmse, "improved": improved})
        if improved:
            best_rmse, wait = val_rmse, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg.gnn_patience and epoch >= MIN_EPOCHS:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, history


def run_tgcn(td: "tb.TargetData", masks: dict[str, np.ndarray],
            cfg: "tb.RunConfig") -> tuple[list[dict], dict[str, np.ndarray], pd.DataFrame, dict]:
    """T-GCN over all seeds, reusing PiStgnn's node-tensor pipeline unchanged."""
    graph = tb.build_river_graph(td.target_id, td.upstream_ids, td.boundary_ids)
    data, scaler = tb.prepare_gnn_data(td, masks, graph, cfg)
    delta = (td.frame["WL"] - td.frame["WLD-1"]).to_numpy()
    base = td.frame["WLD-1"].to_numpy()
    preds, hist, info = {}, [], {"graph": tb.graph_info(graph), "fits": {}}
    for seed in cfg.gnn_seeds:
        t0 = time.time()
        model, history = fit_tgcn(graph, data, delta[masks["val"]], scaler, seed, cfg)
        with torch.no_grad():
            d_hat = {s: model(data[s]["x"]).double().cpu().numpy() * scaler.y_std + scaler.y_mean
                     for s in ("val", "test")}
        preds[seed] = {s: base[masks[s]] + d_hat[s] for s in ("val", "test")}
        best = min(history, key=lambda h: h["val_rmse_m"])
        info["fits"][str(seed)] = {"best_epoch": best["epoch"], "epochs_run": len(history),
                                   "best_val_rmse_m": best["val_rmse_m"],
                                   "runtime_s": round(time.time() - t0, 1)}
        hist += [{"target": td.name, "model": "tgcn", "seed": seed, **h} for h in history]
    rows, avg = tb.seeded_model_rows(td, masks, "tgcn", preds)
    return rows, avg, pd.DataFrame(hist), info


# --------------------------------------------------------------------------- ARFF export
def to_arff(frame: pd.DataFrame, cols: list[str], mask: np.ndarray, relation: str,
           path: Path) -> None:
    """Numeric ARFF: feature columns, then WLD-1 (for reconstruction), then the
    target 'delta' last. Date is dropped, matching Weka's LinearRegression input."""
    rows = frame.loc[mask]
    delta = (rows["WL"] - rows["WLD-1"]).to_numpy()
    attrs = list(dict.fromkeys([*cols, "WLD-1"]))  # WLD-1 once, even if already in cols
    data = rows[attrs].to_numpy(dtype=np.float64)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"@relation {relation}\n\n")
        for a in attrs:
            f.write(f"@attribute {a.replace('-', '_')} numeric\n")
        f.write("@attribute delta numeric\n\n@data\n")
        for r, d in zip(data, delta):
            f.write(",".join(f"{v:.6f}" for v in (*r, d)) + "\n")


def try_weka_linear_regression(train_arff: Path, test_arff: Path,
                               alpha: float) -> dict | None:
    """Attempt python-weka-wrapper3's LinearRegression -S 1 -R <alpha> on Kaggle
    (internet enabled there); return {'rmse': ...} or None with a logged reason."""
    try:
        import weka  # noqa: F401
    except ImportError:
        if tb.on_kaggle():  # internet is enabled there; try once, quietly
            subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                           "python-weka-wrapper3"], check=False, timeout=180)
    try:
        import weka.core.jvm as jvm
        from weka.classifiers import Classifier, Evaluation
        from weka.core.converters import Loader
        from weka.core.dataset import Instances
    except ImportError as exc:
        print(f"  [E0] python-weka-wrapper3 unavailable, skipping Weka cross-check: {exc}")
        return None
    try:
        jvm.start(packages=True)
        loader = Loader(classname="weka.core.converters.ArffLoader")
        train = loader.load_file(str(train_arff))
        train.class_is_last()
        test = loader.load_file(str(test_arff))
        test.class_is_last()
        cls = Classifier(classname="weka.classifiers.functions.LinearRegression",
                         options=["-S", "1", "-R", str(alpha)])
        cls.build_classifier(train)
        ev = Evaluation(train)
        ev.test_model(cls, test)
        return {"rmse": float(ev.root_mean_squared_error)}
    except Exception as exc:  # noqa: BLE001 - report and move on, never fail the run
        print(f"  [E0] Weka JVM run failed, skipping: {exc}")
        return None
    finally:
        try:
            jvm.stop()
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------- E0
def run_e0(files: dict[str, Path], out: Path) -> dict:
    """ARFF export + optional Weka cross-check for Bhagyakul's main split."""
    long = pd.read_csv(files["long"], parse_dates=["Date"])
    td = tb.load_target(files["Bhagyakul"], long, BHAGYAKUL, "Bhagyakul")
    masks = tb.split_masks(td.frame["Date"])
    cols = tb.td_full_columns(td)
    tr_path = out / "arff" / "e0_bhagyakul_main_train.arff"
    te_path = out / "arff" / "e0_bhagyakul_main_test.arff"
    to_arff(td.frame, cols, masks["train"], "bhagyakul_main_train", tr_path)
    to_arff(td.frame, cols, masks["test"], "bhagyakul_main_test", te_path)
    x_tr, d_tr = td.frame.loc[masks["train"], cols].to_numpy(), \
        (td.frame.loc[masks["train"], "WL"] - td.frame.loc[masks["train"], "WLD-1"]).to_numpy()
    x_va, d_va = td.frame.loc[masks["val"], cols].to_numpy(), \
        (td.frame.loc[masks["val"], "WL"] - td.frame.loc[masks["val"], "WLD-1"]).to_numpy()
    _, alpha, _ = tb.fit_ridge(x_tr, d_tr, x_va, d_va)
    x_te = td.frame.loc[masks["test"], cols].to_numpy()
    d_te = (td.frame.loc[masks["test"], "WL"] - td.frame.loc[masks["test"], "WLD-1"]).to_numpy()
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    sk = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha))]).fit(x_tr, d_tr)
    sk_rmse = float(np.sqrt(np.mean((sk.predict(x_te) - d_te) ** 2)))
    weka = try_weka_linear_regression(tr_path, te_path, alpha)
    result = {"alpha": alpha, "sklearn_rmse_delta": sk_rmse, "weka": weka}
    (out / "formats").mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"experiment": "e0", "sklearn_rmse_delta": sk_rmse,
                  "weka_rmse_delta": weka["rmse"] if weka else None,
                  "weka_available": weka is not None}]).to_csv(
        out / "metrics" / "e0.csv", index=False)
    return result


# --------------------------------------------------------------------------- E1: main split
def run_e1(files: dict[str, Path], out: Path, cfg: "tb.RunConfig") -> dict:
    """3 targets x {Ridge, T-GCN, LSTM, PI-STGNN, PI-STGNN-no-physics} + persistence
    on the released split; picks the M4 variant on validation PI averaged over
    targets; ablation A-G for Ridge and the chosen M4 only; WL_Trend classification
    unchanged (numbers only)."""
    long = pd.read_csv(files["long"], parse_dates=["Date"])
    metric_rows: list[dict] = []
    pred_frames: list[pd.DataFrame] = []
    variant_val_pi: dict[str, list[float]] = {"pistgnn": [], "stgnn_nophys": []}
    per_target: dict[str, dict] = {}
    for target_id, name, fname in tb.TARGETS:
        td = tb.load_target(files[name], long, target_id, name)
        masks = tb.split_masks(td.frame["Date"])
        pers = {s: td.frame.loc[masks[s], "WLD-1"].to_numpy() for s in ("val", "test")}
        pers_metrics = tb.split_metrics(td.frame, masks, pers)
        tb.persistence_check(td.frame, masks, pers_metrics)
        metric_rows += tb.metric_rows(name, "persistence", "none", pers_metrics)
        tab = tb.run_tabular(td.frame, tb.td_full_columns(td), masks, cfg)
        ridge_mets = tb.split_metrics(td.frame, masks, tab["ridge"]["wl_hat"])
        metric_rows += tb.metric_rows(name, "ridge", str(tb.SEED), ridge_mets)
        tgcn_rows, tgcn_avg, tgcn_hist, tgcn_info = run_tgcn(td, masks, cfg)
        metric_rows += tgcn_rows
        lstm_rows, lstm_test, lstm_hist, lstm_info = tb.lstm_results(td, masks, cfg)
        metric_rows += lstm_rows
        gnn_rows, gnn_test, gnn_hist, gnn_info = tb.gnn_results(td, masks, cfg)
        metric_rows += gnn_rows
        for variant in ("pistgnn", "stgnn_nophys"):
            val_pi = [r["pi"] for r in gnn_rows if r["model"] == variant
                     and r["split"] == "val" and r["seed"] == "mean"][0]
            variant_val_pi[variant].append(val_pi)
        preds = td.frame.loc[masks["test"], ["Date", "WL"]].rename(columns={"WL": "WL_obs"})
        preds = preds.assign(target=name, persistence=pers["test"], ridge=tab["ridge"]["wl_hat"]["test"],
                             tgcn=tgcn_avg["test"], lstm=lstm_test,
                             pistgnn=gnn_test["pistgnn"], stgnn_nophys=gnn_test["stgnn_nophys"])
        pred_frames.append(preds)
        ablation_rows, ablation_info = tb.run_ablation(td, cfg)
        metric_rows += [r for r in ablation_rows if r["model"] in ("ridge", "lightgbm")]
        per_target[name] = {"ridge": tab["ridge"]["info"], "tgcn": tgcn_info,
                            "lstm": lstm_info, "gnn": gnn_info, "ablation": ablation_info,
                            "counts": {s: int(m.sum()) for s, m in masks.items()},
                            "n_predictors": len(tb.td_full_columns(td))}
        pd.concat([tgcn_hist], ignore_index=True).to_csv(
            out / "metrics" / f"e1_{name.lower()}_tgcn_history.csv", index=False)
    custom_variant = ("pistgnn" if np.mean(variant_val_pi["pistgnn"])
                      >= np.mean(variant_val_pi["stgnn_nophys"]) else "stgnn_nophys")
    custom_label = "PI-STGNN" if custom_variant == "pistgnn" else "RS-GNN (river-sweep graph network)"
    metric_rows = [r for r in metric_rows if r["model"] != {"pistgnn": "stgnn_nophys",
                                                            "stgnn_nophys": "pistgnn"}[custom_variant]
                  or True]  # keep both in metrics.csv (ablation sentence needs the loser)
    for r in metric_rows:
        if r["model"] == custom_variant:
            r["model"] = "custom"
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(out / "metrics" / "e1_main.csv", index=False)
    (out / "preds").mkdir(parents=True, exist_ok=True)
    pd.concat(pred_frames, ignore_index=True).to_csv(out / "preds" / "e1_main.csv", index=False)
    for target_id, name, fname in tb.TARGETS:
        td = tb.load_target(files[name], long, target_id, name)
        masks = tb.split_masks(td.frame["Date"])
        cols = tb.td_full_columns(td)
        for split in ("train", "test"):
            to_arff(td.frame, cols, masks[split], f"e1_{name.lower()}_{split}",
                   out / "arff" / f"e1_{name.lower()}_{split}.arff")
    return {"custom_variant": custom_variant, "custom_label": custom_label,
            "validation_pi_by_variant": {k: float(np.mean(v)) for k, v in variant_val_pi.items()},
            "per_target": per_target}


def run_ablation_gnn(files: dict[str, Path], out: Path, cfg: "tb.RunConfig",
                     custom_variant: str) -> list[dict]:
    """T-GCN and the chosen M4 variant at the one extra graph-native feature
    set that has a clean topology: C (brief -- target + upstream only, no
    boundary nodes). F (released) is E1's main-run number, reused here rather
    than rerun. Sets A/B/D/E/G have no natural graph realisation (no upstream
    at all, or same-day masking for every node rather than just the target's),
    so the ablation figure leaves them to Ridge alone -- stated in the caption,
    not silently implied."""
    long = pd.read_csv(files["long"], parse_dates=["Date"])
    rows = []
    for target_id, name, fname in tb.TARGETS:
        wide = pd.read_csv(files[name], parse_dates=["Date"])
        up_ids, bnd_ids = tb.point_ids(wide.columns, tb.UPSTREAM), tb.point_ids(wide.columns, tb.BOUNDARY)
        td = tb.load_target(files[name], long, target_id, name)
        masks = tb.split_masks(td.frame["Date"])
        graph_c = build_target_graph(target_id, up_ids, (), up_ids, bnd_ids)
        tgcn_c, custom_c = run_lag_gnn_variants(td.frame, graph_c, masks, tb.N_LAGS, cfg, custom_variant)
        obs = td.frame.loc[masks["test"], "WL"].to_numpy()
        pers = td.frame.loc[masks["test"], "WLD-1"].to_numpy()
        dates = td.frame.loc[masks["test"], "Date"]
        for model, preds in (("tgcn", tgcn_c), ("custom", custom_c)):
            rows.append({"target": name, "feature_set": "C", "model": model, "split": "test",
                        **tb.compute_metrics(obs, preds, pers, dates)})
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "metrics" / "e1_ablation_gnn.csv", index=False)
    return rows


# --------------------------------------------------------------------------- E2: lag depth
def build_extended_wide(long: pd.DataFrame, wide: pd.DataFrame, target_id: int,
                        up_ids: tuple[int, ...], bnd_ids: tuple[int, ...],
                        max_k: int = 10) -> pd.DataFrame:
    """The released wide matrix plus WLD-(8..max_k) for the target and every
    predictor point, computed from the long panel with the release's own
    calendar-reindex rule (no interpolation). Extra lags may be NaN even where
    the released columns are not."""
    calendar = pd.date_range(long["Date"].min(), long["Date"].max(), freq="D")
    extra = wide[["Date"]].copy()
    for point_id, prefix in [(target_id, "")] + [(i, tb.UPSTREAM) for i in up_ids] \
            + [(i, tb.BOUNDARY) for i in bnd_ids]:
        stage = (long.loc[long["Id"] == point_id].set_index("Date")["WL"]
                 .reindex(calendar))
        for k in range(8, max_k + 1):
            col = f"WLD-{k}" if not prefix else f"{prefix}{point_id:02d}_WLD-{k}"
            extra[col] = stage.shift(k).reindex(extra["Date"]).to_numpy()
    return wide.merge(extra, on="Date", how="left", validate="one_to_one")


def lag_depth_columns(k: int, up_ids: tuple[int, ...], bnd_ids: tuple[int, ...]) -> list[str]:
    """Brief+boundary layout truncated/extended to lag depth k (own and every
    predictor point); same-day WL of every predictor is always included."""
    def block(prefix: str, i: int) -> list[str]:
        return [f"{prefix}{i:02d}_WL"] + [f"{prefix}{i:02d}_WLD-{j}" for j in range(1, k + 1)]
    cols = [f"WLD-{j}" for j in range(1, k + 1)]
    cols += [c for i in up_ids for c in block(tb.UPSTREAM, i)]
    cols += [c for i in bnd_ids for c in block(tb.BOUNDARY, i)]
    return cols


def run_e2(files: dict[str, Path], out: Path, cfg: "tb.RunConfig", custom_variant: str) -> dict:
    """k in {3, 7, 10}, all 4 models, identical test-day set, Bhagyakul only."""
    long = pd.read_csv(files["long"], parse_dates=["Date"])
    wide = pd.read_csv(files["Bhagyakul"], parse_dates=["Date"])
    up_ids, bnd_ids = tb.point_ids(wide.columns, tb.UPSTREAM), tb.point_ids(wide.columns, tb.BOUNDARY)
    ext = build_extended_wide(long, wide, BHAGYAKUL, up_ids, bnd_ids)
    ext.to_csv(out / "wide_bhagyakul_k10.csv", index=False)
    rain = long.loc[long["Id"] == BHAGYAKUL, ["Date", tb.RAIN_COL]]
    frame_full = ext.merge(rain, on="Date", how="left").sort_values("Date").reset_index(drop=True)
    common_test = None
    per_k: dict[int, dict] = {}
    for k in (3, 7, 10):
        cols = lag_depth_columns(k, up_ids, bnd_ids)
        rows_ok = frame_full[cols + ["WL", "WLD-1"]].notna().all(axis=1)
        per_k[k] = {"cols": cols, "rows_ok": rows_ok}
    test_start = tb.TEST_START
    for k in (3, 7, 10):
        is_test = (frame_full["Date"] >= test_start) & per_k[k]["rows_ok"]
        common_test = is_test if common_test is None else (common_test & is_test)
    n_common = int(common_test.sum())
    print(f"  [E2] common test days across k=3,7,10: {n_common}")
    metric_rows, sample_rows, hist_rows, plot_rows = [], [], [], []
    for k in (3, 7, 10):
        # frame_full is shared and never subset: masks are boolean arrays over its
        # own rows, so a row not selected by any split (e.g. missing WLD-8..10 at
        # k=10 outside the common test window) simply never enters run_tabular.
        frame = frame_full
        cols = per_k[k]["cols"]
        dates = frame["Date"]
        rows_ok = per_k[k]["rows_ok"].to_numpy()
        masks = {
            "train": (dates < tb.TRAIN_END).to_numpy() & rows_ok,
            "val": ((dates >= tb.TRAIN_END) & (dates < test_start)).to_numpy() & rows_ok,
            "test": common_test.to_numpy(),
        }
        with lag_depth_leakage_check(k):
            tab = tb.run_tabular(frame, cols, masks, cfg)
        pers = frame.loc[masks["test"], "WLD-1"].to_numpy()
        obs = frame.loc[masks["test"], "WL"].to_numpy()
        test_dates = frame.loc[masks["test"], "Date"]
        metric_rows += [{"k": k, "model": "ridge",
                        **compute_ci_row(obs, tab["ridge"]["wl_hat"]["test"], pers, test_dates)}]
        graph = tb.build_river_graph(BHAGYAKUL, up_ids, bnd_ids)
        preds_tgcn, preds_custom = run_lag_gnn_variants(frame, graph, masks, k, cfg, custom_variant)
        metric_rows += [{"k": k, "model": "tgcn", **compute_ci_row(obs, preds_tgcn, pers, test_dates)}]
        metric_rows += [{"k": k, "model": "custom", **compute_ci_row(obs, preds_custom, pers, test_dates)}]
        preds_lstm = run_lag_lstm(frame, up_ids, bnd_ids, masks, k, cfg)
        metric_rows += [{"k": k, "model": "lstm", **compute_ci_row(obs, preds_lstm, pers, test_dates)}]
        # Two worked days from the actual common test set (not hardcoded, so they
        # are guaranteed present at every k): the wettest and driest observed WL.
        test_frame = frame.loc[masks["test"]].reset_index(drop=True)
        model_preds = {"ridge": tab["ridge"]["wl_hat"]["test"], "tgcn": preds_tgcn,
                      "custom": preds_custom, "lstm": preds_lstm}
        plot_rows.append(pd.DataFrame({
            "k": k, "Date": test_frame["Date"].to_numpy(), "WL_obs": test_frame["WL"].to_numpy(),
            "persistence": pers, **{m: p for m, p in model_preds.items()},
        }))
        sample_positions = {"flood_peak": int(test_frame["WL"].idxmax()),
                            "dry_season": int(test_frame["WL"].idxmin())}
        for label, pos in sample_positions.items():
            r = test_frame.iloc[pos]
            row = {"k": k, "sample": label, "date": r["Date"].date().isoformat(),
                  "WLD-1": round(r["WLD-1"], 3),
                  "own_lags": ", ".join(f"{r[f'WLD-{j}']:.2f}" for j in sorted({1, min(2, k), k}) if j <= k),
                  "observed_WL": round(r["WL"], 3)}
            for pid in (up_ids[0], up_ids[-1], bnd_ids[-1] if bnd_ids else up_ids[-1]):
                prefix = tb.BOUNDARY if pid in bnd_ids else tb.UPSTREAM
                row[f"{prefix}{pid:02d}_WL"] = round(r[f"{prefix}{pid:02d}_WL"], 3)
                row[f"{prefix}{pid:02d}_WLD-1"] = round(r[f"{prefix}{pid:02d}_WLD-1"], 3)
            for model, preds in model_preds.items():
                row[f"pred_{model}"] = round(float(preds[pos]), 3)
                row[f"abserr_{model}"] = round(abs(float(preds[pos]) - r["WL"]), 3)
            sample_rows.append(row)
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(out / "metrics" / "e2_lagdepth.csv", index=False)
    pd.DataFrame(sample_rows).to_csv(out / "metrics" / "e2_sample_table.csv", index=False)
    (out / "preds").mkdir(parents=True, exist_ok=True)
    pd.concat(plot_rows, ignore_index=True).to_csv(out / "preds" / "e2_lagdepth.csv", index=False)
    ref = [r for r in metric_rows if r["k"] == 7]
    delta_rows = []
    for k in (3, 10):
        for model in MODEL_ORDER:
            a = next(r for r in metric_rows if r["k"] == k and r["model"] == model)
            b = next(r for r in ref if r["model"] == model)
            delta_rows.append({"k": k, "model": model, "delta_pi": a["pi"] - b["pi"]})
    pd.DataFrame(delta_rows).to_csv(out / "metrics" / "e2_delta_vs_k7.csv", index=False)
    return {"n_common_test_days": n_common, "chosen_k": 7}


def compute_ci_row(obs: np.ndarray, pred: np.ndarray, pers: np.ndarray,
                   dates: pd.Series) -> dict:
    """Full tb.compute_metrics (RMSE, MAE, NSE, PI, monsoon RMSE) plus a
    moving-block bootstrap CI on PI, for one model's test predictions."""
    mets = tb.compute_metrics(obs, pred, pers, dates)
    _, lo, hi = bootstrap_pi_ci(obs, pred, pers)
    return {**mets, "pi_lo": lo, "pi_hi": hi}


def build_k_node_inputs(frame: pd.DataFrame, graph: "tb.RiverGraph", train: np.ndarray,
                        k: int) -> tuple[np.ndarray, "tb.NodeScaler"]:
    """Node tensor (n, N, k+1, 2) for a custom lag depth k (only used inside E2)."""
    steps = k + 1
    x = np.zeros((len(frame), len(graph.node_ids), steps, tb.GNN_CHANNELS))
    means, stds = [], []
    for pos, node in enumerate(graph.node_ids):
        lags = range(k, 0, -1)
        if node == graph.target_id:
            cols = [f"WLD-{j}" for j in lags]
        else:
            prefix = tb.BOUNDARY if node in graph.boundary_ids else tb.UPSTREAM
            cols = [f"{prefix}{node:02d}_WLD-{j}" for j in lags] + [f"{prefix}{node:02d}_WL"]
        vals = frame[cols].to_numpy(dtype=np.float64)
        means.append(float(vals[train].mean())); stds.append(float(vals[train].std()))
        x[:, pos, :len(cols), 0] = (vals - means[-1]) / stds[-1]
        x[:, pos, :len(cols), 1] = 1.0
    delta = (frame["WL"] - frame["WLD-1"]).to_numpy()
    scaler = tb.NodeScaler(np.array(means), np.array(stds), float(delta[train].mean()),
                          float(delta[train].std()), int(train.sum()))
    return x, scaler


def run_lag_gnn_variants(frame: pd.DataFrame, graph: "tb.RiverGraph", masks: dict[str, np.ndarray],
                         k: int, cfg: "tb.RunConfig", custom_variant: str) -> tuple[np.ndarray, np.ndarray]:
    """Seed-averaged test WL from T-GCN and the chosen M4 variant, at lag depth k."""
    steps_backup = tb.GNN_STEPS
    tb.GNN_STEPS = k + 1  # both PiStgnn.encode and TGCN read this module constant
    try:
        x, scaler = build_k_node_inputs(frame, graph, masks["train"], k)
        delta = (frame["WL"] - frame["WLD-1"]).to_numpy()
        base = frame["WLD-1"].to_numpy()
        data = {s: {"x": tb.to_tensor(x[m], cfg.device),
                    "y": tb.to_tensor((delta[m] - scaler.y_mean) / scaler.y_std, cfg.device)}
                for s, m in masks.items()}
        tgcn_preds, custom_preds = [], []
        for seed in cfg.gnn_seeds:
            model_t, _ = fit_tgcn(graph, data, delta[masks["val"]], scaler, seed, cfg)
            with torch.no_grad():
                tgcn_preds.append(base[masks["test"]]
                                  + (model_t(data["test"]["x"]).double().cpu().numpy()
                                     * scaler.y_std + scaler.y_mean))
            lam = tb.PHYSICS_LAMBDA if custom_variant == "pistgnn" else 0.0
            model_c, _, _ = tb.fit_gnn(graph, data, delta[masks["val"]], scaler, seed, lam, cfg)
            d_hat, _ = tb.gnn_predict(model_c, data["test"]["x"], scaler)
            custom_preds.append(base[masks["test"]] + d_hat)
        return np.mean(tgcn_preds, axis=0), np.mean(custom_preds, axis=0)
    finally:
        tb.GNN_STEPS = steps_backup


def run_lag_lstm(frame: pd.DataFrame, up_ids: tuple[int, ...], bnd_ids: tuple[int, ...],
                 masks: dict[str, np.ndarray], k: int, cfg: "tb.RunConfig") -> np.ndarray:
    """Seed-averaged LSTM test WL with a k-step sequence (own + every point)."""
    points = [f"{tb.UPSTREAM}{i:02d}" for i in up_ids] + [f"{tb.BOUNDARY}{i:02d}" for i in bnd_ids]
    steps = []
    for j in range(k, 0, -1):
        cols = [f"WLD-{j}"] + [f"{p}_WLD-{j}" for p in points]
        steps.append(frame[cols].to_numpy(dtype=np.float64))
    seq = np.stack(steps, axis=1)
    now = frame[[f"{p}_WL" for p in points]].to_numpy(dtype=np.float64)
    delta = (frame["WL"] - frame["WLD-1"]).to_numpy()
    base = frame["WLD-1"].to_numpy()
    tr = masks["train"]
    scaler = tb.Standardizer.fit(seq[tr], now[tr], delta[tr])
    seq_s, now_s = scaler.transform(seq, now)
    y_s = (delta - scaler.y_mean) / scaler.y_std
    data = {s: {"seq": tb.to_tensor(seq_s[m], cfg.device), "now": tb.to_tensor(now_s[m], cfg.device),
                "y": tb.to_tensor(y_s[m], cfg.device)} for s, m in masks.items()}
    preds = []
    for seed in cfg.lstm_seeds:
        model, _ = tb.fit_lstm(data, delta[masks["val"]], scaler, seed, cfg)
        preds.append(base[masks["test"]] + tb.predict_delta(model, data["test"]["seq"],
                                                            data["test"]["now"], scaler))
    return np.mean(preds, axis=0)


# --------------------------------------------------------------------------- E3: travel time
def year_blocks(pair: pd.DataFrame) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Split a monsoon-day p/g frame into contiguous per-year (p, g) numpy
    arrays -- the resampling unit for the season block bootstrap below."""
    return {int(year): (g["p"].to_numpy(), g["g"].to_numpy())
           for year, g in pair.groupby(pair.index.year)}


def corr_at_lag(blocks: list[tuple[np.ndarray, np.ndarray]], lag: int) -> tuple[float, int]:
    """Pearson r of p(t) vs g(t+lag), pooling valid pairs WITHIN each block only
    (a shift never crosses a year boundary, resampled or not) -- plain numpy,
    no pandas, so this is cheap enough to call inside a bootstrap loop."""
    ps, gs = [], []
    for p, g in blocks:
        p_seg, g_seg = (p, g) if lag == 0 else (p[:-lag], g[lag:]) if len(p) > lag else (None, None)
        if p_seg is None:
            continue
        mask = ~(np.isnan(p_seg) | np.isnan(g_seg))
        if mask.any():
            ps.append(p_seg[mask])
            gs.append(g_seg[mask])
    if not ps:
        return np.nan, 0
    p_all, g_all = np.concatenate(ps), np.concatenate(gs)
    if len(p_all) < 30 or np.std(p_all) == 0 or np.std(g_all) == 0:
        return np.nan, len(p_all)
    return float(np.corrcoef(p_all, g_all)[0, 1]), len(p_all)


def peak_lag_from_blocks(blocks: list[tuple[np.ndarray, np.ndarray]],
                         lags: range = range(0, 11)) -> tuple[int, float, int]:
    """(lag, corr, n) of the strongest correlation of p leading g by ``lag`` days."""
    best_lag, best_corr, best_n = 0, -np.inf, 0
    for lag in lags:
        r, n = corr_at_lag(blocks, lag)
        if np.isfinite(r) and r > best_corr:
            best_lag, best_corr, best_n = lag, r, n
    return best_lag, best_corr, best_n


def bootstrap_peak_lag_ci(blocks_by_year: dict[int, tuple[np.ndarray, np.ndarray]],
                         resamples: int = BOOT_RESAMPLES,
                         seed: int = 20260913) -> tuple[float, float]:
    """95% CI on the peak lag from a block bootstrap over monsoon SEASONS (whole
    years resampled with replacement, matching the spec's 'bootstrap by monsoon
    season')."""
    years = np.array(sorted(blocks_by_year))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(resamples):
        picked = rng.choice(years, size=len(years), replace=True)
        lag, corr, _ = peak_lag_from_blocks([blocks_by_year[y] for y in picked])
        if np.isfinite(corr):
            boots.append(lag)
    if not boots:
        return (np.nan, np.nan)
    return tuple(np.percentile(boots, [2.5, 97.5]).tolist())


def run_e3(files: dict[str, Path], out: Path) -> dict:
    """(a) pooled monsoon cross-correlation + peak-lag-vs-chainage celerity fit
    (peak lag with a season block-bootstrap CI); (b) event-based arrival lags at
    Panka surge events; (c) Farakka case study from FARAKKA_CANDIDATES, kept
    only if a matching Panka surge exists."""
    long = pd.read_csv(files["long"], parse_dates=["Date"])
    calendar = pd.date_range(long["Date"].min(), long["Date"].max(), freq="D")
    wl = long.pivot(index="Date", columns="Id", values="WL").reindex(calendar)
    dwl = wl.diff()
    monsoon = dwl.index.month.isin(tb.MONSOON_MONTHS)
    panka = dwl[1][monsoon]

    xcorr_rows, peak_rows = [], []
    for gid in MAIN_STEM_GAUGES:
        target = dwl[gid][monsoon]
        # No row is dropped here: dwl is already reindexed onto the full calendar, so
        # position i within one year's monsoon block is calendar day (block_start + i)
        # exactly; corr_at_lag masks NaNs itself, after shifting, so a lag of k steps
        # stays a lag of k calendar days even where a day's reading is missing.
        pair = pd.concat([panka, target], axis=1, keys=["p", "g"])
        blocks_by_year = year_blocks(pair)
        blocks = list(blocks_by_year.values())
        for lag in range(0, 11):
            r, n = corr_at_lag(blocks, lag)
            if np.isfinite(r):
                xcorr_rows.append({"target_id": gid, "lag": lag, "corr": r, "n": n})
        best_lag, best_corr, best_n = peak_lag_from_blocks(blocks)
        lag_lo, lag_hi = bootstrap_peak_lag_ci(blocks_by_year)
        peak_rows.append({"target_id": gid, "lag": best_lag, "corr": best_corr,
                          "n": best_n, "lag_ci_lo": lag_lo, "lag_ci_hi": lag_hi})
    xcorr = pd.DataFrame(xcorr_rows)
    peak = pd.DataFrame(peak_rows)
    chainage = {i: v[2] for i, v in tb.STATION_GEO.items() if v[2] is not None}
    peak = peak[peak["target_id"].isin(chainage)].copy()
    peak["chainage_km"] = peak["target_id"].map(chainage)
    slope, intercept = np.polyfit(peak["chainage_km"], peak["lag"], 1)
    celerity = 1.0 / slope if slope != 0 else np.inf  # km per day

    thresh = panka.quantile(0.95)
    surge_days = panka[panka >= thresh].index
    events, last = [], None
    for d in sorted(surge_days):
        if last is None or (d - last).days >= 10:
            events.append(d)
        last = d
    event_rows = []
    for gid in (BARURIA, BHAGYAKUL, SURESWAR):
        lags = []
        for ev in events:
            window = dwl[gid].reindex(pd.date_range(ev, ev + pd.Timedelta(days=10)))
            if window.notna().sum() < 5:
                continue
            lags.append(int(np.nanargmax(window.to_numpy())))
        if lags:
            event_rows.append({"target_id": gid, "n_events": len(lags),
                               "median_lag": float(np.median(lags)),
                               "iqr_lo": float(np.percentile(lags, 25)),
                               "iqr_hi": float(np.percentile(lags, 75))})
    events_df = pd.DataFrame(event_rows)

    verified, trace_rows = [], []
    for date_str, source in FARAKKA_CANDIDATES:
        d = pd.Timestamp(date_str)
        window = panka.reindex(pd.date_range(d - pd.Timedelta(days=2), d + pd.Timedelta(days=2)))
        if window.notna().any() and window.max() >= thresh:
            verified.append({"date": date_str, "source": source, "panka_max_dwl": float(window.max())})
        # Traced regardless of verification, aligned like the surge composite (day 0 =
        # the reported gate-opening date, not a detected Panka surge peak) so the report
        # can show this specific candidate's Panka signal next to the 35-surge composite.
        full_dwl = dwl[1]  # unmasked by monsoon-only filter, in case the date falls outside it
        trace = full_dwl.reindex(pd.date_range(d - pd.Timedelta(days=2), d + pd.Timedelta(days=10)))
        for k, val in enumerate(trace.to_numpy()):
            trace_rows.append({"date": date_str, "source": source, "day": k - 2,
                               "dwl_panka": float(val) if np.isfinite(val) else np.nan})

    (out / "metrics").mkdir(parents=True, exist_ok=True)
    xcorr.to_csv(out / "metrics" / "e3_xcorr.csv", index=False)
    peak.to_csv(out / "metrics" / "e3_peak_lag_chainage.csv", index=False)
    events_df.to_csv(out / "metrics" / "e3_event_arrival.csv", index=False)
    pd.DataFrame(verified).to_csv(out / "metrics" / "e3_farakka_verified.csv", index=False)
    pd.DataFrame(trace_rows).to_csv(out / "metrics" / "e3_farakka_trace.csv", index=False)
    superposed = pd.DataFrame({gid: [float(np.nanmean([dwl[gid].reindex(
        pd.date_range(ev - pd.Timedelta(days=2), ev + pd.Timedelta(days=10))).to_numpy()[k]
        for ev in events if True])) for k in range(13)]
        for gid in (1, 4, 10, BHAGYAKUL, SURESWAR)})
    superposed.to_csv(out / "metrics" / "e3_superposed_epoch.csv", index=False)
    return {"celerity_km_per_day": float(celerity), "n_events": len(events),
            "n_farakka_verified": len(verified),
            "arrival_lags": events_df.to_dict("records")}


def build_reduced_graph(target_id: int, up_ids: tuple[int, ...],
                        bnd_ids: tuple[int, ...]) -> "tb.RiverGraph":
    """A RiverGraph over exactly the surviving (gap-free) nodes: the corridor's
    full RIVER_EDGES/BOUNDARY_EDGES/backwater topology, contracted (bridged
    directly, using each pair's real chainage/haversine distance) across any
    dropped intermediate node, so every surviving node still drains to the
    target. Used only when E4/E5 has dropped a gap station; tb.build_river_graph
    (unmodified) is used whenever nothing was dropped."""
    nodes = set(up_ids) | set(bnd_ids) | {target_id}
    all_edges = list(tb.RIVER_EDGES) + list(tb.BOUNDARY_EDGES) + [(tb.BACKWATER_SOURCE, target_id)]
    succ: dict[int, list[int]] = {}
    for i, j in all_edges:
        succ.setdefault(i, []).append(j)

    def reachable_survivors(start: int) -> list[int]:
        """Surviving nodes reached from ``start`` by walking through dropped
        (non-surviving) nodes only; stops at the first surviving node found."""
        seen, frontier, out = set(), list(succ.get(start, [])), []
        while frontier:
            n = frontier.pop()
            if n in seen:
                continue
            seen.add(n)
            if n in nodes:
                out.append(n)
            else:
                frontier += succ.get(n, [])
        return out

    edges, kind = set(), {}
    for n in nodes - {target_id}:
        for m in reachable_survivors(n):
            e = (n, m)
            edges.add(e)
            kind[e] = tb.EDGE_BACKWATER if n == tb.BACKWATER_SOURCE and m == target_id else tb.EDGE_DOWNSTREAM
    order = tb.topological_order(nodes, list(edges))
    edges_sorted = sorted(edges, key=lambda e: (order.index(e[1]), order.index(e[0])))
    dist = [tb.edge_distance_km(i, j) for i, j in edges_sorted]
    graph = tb.RiverGraph(target_id, order, tuple(edges_sorted),
                          tuple(d for d, _ in dist), tuple(b for _, b in dist),
                          tuple(kind[e] for e in edges_sorted), tuple(sorted(bnd_ids)))
    reached, frontier = {target_id}, [target_id]
    while frontier:
        node = frontier.pop()
        new = {i for i, j in graph.edges if j == node} - reached
        reached |= new
        frontier += sorted(new)
    assert reached == set(graph.node_ids), "every surviving gauge must drain to the target"
    assert graph.node_ids[-1] == target_id and graph.physics_parent in nodes
    return graph


def build_target_graph(target_id: int, up_ids: tuple[int, ...], bnd_ids: tuple[int, ...],
                       full_up_ids: tuple[int, ...], full_bnd_ids: tuple[int, ...]) -> "tb.RiverGraph":
    """tb.build_river_graph when nothing was dropped, else the contracted graph."""
    if up_ids == full_up_ids and bnd_ids == full_bnd_ids:
        return tb.build_river_graph(target_id, up_ids, bnd_ids)
    return build_reduced_graph(target_id, up_ids, bnd_ids)


# --------------------------------------------------------------------------- E4/E5: spans
def gap_free_predictors(long: pd.DataFrame, ids: tuple[int, ...],
                        window_start: pd.Timestamp, window_end: pd.Timestamp) -> tuple[int, ...]:
    """Ids whose own WL series is missing at most GAP_MISSING_FRAC of the
    calendar days in [window_start, window_end)."""
    calendar = pd.date_range(window_start, window_end - pd.Timedelta(days=1), freq="D")
    keep = []
    for i in ids:
        s = long.loc[long["Id"] == i].set_index("Date")["WL"].reindex(calendar)
        if s.isna().mean() <= GAP_MISSING_FRAC:
            keep.append(i)
    return tuple(sorted(keep))


def build_span_wide(long: pd.DataFrame, target_id: int, up_ids: tuple[int, ...],
                    bnd_ids: tuple[int, ...], window_start: pd.Timestamp,
                    window_end: pd.Timestamp) -> tuple[pd.DataFrame, tuple[int, ...], tuple[int, ...]]:
    """A brief+boundary-layout wide matrix built fresh from the long panel for a
    custom date window, using only gap-free predictor points; complete rows only."""
    keep_up = gap_free_predictors(long, up_ids, window_start, window_end)
    keep_bnd = gap_free_predictors(long, bnd_ids, window_start, window_end)
    calendar = pd.date_range(long["Date"].min(), long["Date"].max(), freq="D")

    def stage(i: int) -> pd.Series:
        return long.loc[long["Id"] == i].set_index("Date")["WL"].reindex(calendar)

    target_stage = stage(target_id)
    columns = {"Date": calendar, "WL": target_stage.to_numpy()}
    for k in range(1, tb.N_LAGS + 1):
        columns[f"WLD-{k}"] = target_stage.shift(k).to_numpy()
    for prefix, ids in ((tb.UPSTREAM, keep_up), (tb.BOUNDARY, keep_bnd)):
        for i in ids:
            s = stage(i)
            columns[f"{prefix}{i:02d}_WL"] = s.to_numpy()
            for k in range(1, tb.N_LAGS + 1):
                columns[f"{prefix}{i:02d}_WLD-{k}"] = s.shift(k).to_numpy()
    frame = pd.DataFrame(columns)
    frame = frame[(frame["Date"] >= window_start) & (frame["Date"] < window_end)]
    frame = frame.dropna().reset_index(drop=True)
    frame.insert(0, "Id", target_id)
    return frame, keep_up, keep_bnd


LEAD_DAYS = 60  # F-spans: days of grey pre-test context shown before each span's test window


def run_span(long: pd.DataFrame, target_id: int, name: str, up_ids: tuple[int, ...],
            bnd_ids: tuple[int, ...], peak_date: pd.Timestamp, cfg: "tb.RunConfig",
            custom_variant: str, out: Path, tag: str) -> tuple[list[dict], pd.DataFrame]:
    """All 4 models + persistence on the S6/S12/S36 nested, peak-anchored,
    gap-filtered spans for one target. Also returns a long day-by-day frame
    (LEAD_DAYS of observed WL before the test window, then observed + every
    model's predicted WL through the test window) for F-spans' actual-vs-
    predicted panels -- the metrics CSV alone has no daily series to plot."""
    test_end = peak_date + pd.Timedelta(days=15)
    common_start = test_end - pd.Timedelta(days=SPANS[0][1])  # S6's test window
    window_start = test_end - pd.Timedelta(days=SPANS[-1][1] + SPANS[-1][2])
    frame_all, keep_up, keep_bnd = build_span_wide(long, target_id, up_ids, bnd_ids,
                                                   window_start, test_end)
    dropped = (set(up_ids) - set(keep_up)) | (set(bnd_ids) - set(keep_bnd))
    rows = []
    date_table = []
    plot_frames = []

    def add_rows(model: str, seed: str, pred: np.ndarray, obs: np.ndarray, pers: np.ndarray,
                dates: pd.Series, train_rows: int) -> None:
        """One row scored on this span's own test window, one on the shared
        31-day common window (S6's test window) -- identical for span S6."""
        in_common = (dates >= common_start).to_numpy()
        for window, sel in (("own", slice(None)), ("common", in_common)):
            if window == "common" and not sel.any():
                continue
            rows.append({"target": name, "span": span_name, "model": model, "seed": seed,
                        "window": window, "train_rows": train_rows,
                        **compute_ci_row(obs[sel], pred[sel], pers[sel], dates[sel])})

    for span_name, test_days, train_days in SPANS:
        test_start = test_end - pd.Timedelta(days=test_days)
        train_start = test_start - pd.Timedelta(days=train_days)
        frame = frame_all[frame_all["Date"] >= train_start].reset_index(drop=True)
        date_table.append({"target": name, "span": span_name, "train_start": train_start.date(),
                           "test_start": test_start.date(), "test_end": test_end.date(),
                           "peak_in_test": bool(test_start <= peak_date < test_end),
                           "n_rows": len(frame[frame["Date"] < test_end])})
        try:
            masks = custom_split_masks(frame["Date"], train_start, test_start, test_start, test_end)
        except AssertionError as exc:
            print(f"  [E4/E5] {name} {span_name}: skipped ({exc})")
            continue
        cols = tb.brief_columns(keep_up) + [c for i in keep_bnd for c in tb.block_columns(tb.BOUNDARY, i)]
        pers = {s: frame.loc[masks[s], "WLD-1"].to_numpy() for s in ("val", "test")}
        pers_metrics = tb.split_metrics(frame, masks, pers)
        rows += [{"target": name, "span": span_name, "model": "persistence", "seed": "none",
                 "window": "own", **{k: v for k, v in pers_metrics["test"].items()}}]
        tab = tb.run_tabular(frame, cols, masks, cfg)
        obs, pers_test = frame.loc[masks["test"], "WL"].to_numpy(), pers["test"]
        test_dates = frame.loc[masks["test"], "Date"]
        train_rows = int(masks["train"].sum())
        in_common = (test_dates >= common_start).to_numpy()
        if in_common.any() and not in_common.all():  # S6's own window == the common window
            rows.append({"target": name, "span": span_name, "model": "persistence", "seed": "none",
                        "window": "common",
                        **tb.compute_metrics(obs[in_common], pers_test[in_common],
                                             pers_test[in_common], test_dates[in_common])})
        add_rows("ridge", str(tb.SEED), tab["ridge"]["wl_hat"]["test"], obs, pers_test,
                test_dates, train_rows)
        to_arff(frame, cols, masks["train"], f"{tag}_{name.lower()}_{span_name}_train",
               out / "arff" / f"{tag}_{name.lower()}_{span_name}_train.arff")
        to_arff(frame, cols, masks["test"], f"{tag}_{name.lower()}_{span_name}_test",
               out / "arff" / f"{tag}_{name.lower()}_{span_name}_test.arff")
        graph = build_target_graph(target_id, keep_up, keep_bnd, up_ids, bnd_ids)
        preds_tgcn, preds_custom = run_lag_gnn_variants(frame, graph, masks, tb.N_LAGS,
                                                        cfg, custom_variant)
        add_rows("tgcn", "mean", preds_tgcn, obs, pers_test, test_dates, train_rows)
        add_rows("custom", "mean", preds_custom, obs, pers_test, test_dates, train_rows)
        preds_lstm = run_lag_lstm(frame, keep_up, keep_bnd, masks, tb.N_LAGS, cfg)
        add_rows("lstm", "mean", preds_lstm, obs, pers_test, test_dates, train_rows)
        lead_start = test_start - pd.Timedelta(days=LEAD_DAYS)
        lead_sel = masks["train"] & (frame["Date"] >= lead_start).to_numpy()
        plot_frames.append(pd.DataFrame({
            "target": name, "span": span_name, "phase": "lead",
            "Date": frame.loc[lead_sel, "Date"].to_numpy(), "WL_obs": frame.loc[lead_sel, "WL"].to_numpy(),
        }))
        plot_frames.append(pd.DataFrame({
            "target": name, "span": span_name, "phase": "test", "Date": test_dates.to_numpy(),
            "WL_obs": obs, "persistence": pers_test, "ridge": tab["ridge"]["wl_hat"]["test"],
            "tgcn": preds_tgcn, "lstm": preds_lstm, "custom": preds_custom,
            "common_start": common_start,
        }))
    pd.DataFrame(date_table).to_csv(out / "metrics" / f"{tag}_{name.lower()}_dates.csv", index=False)
    with open(out / "formats" / f"{tag}_{name.lower()}_dropped.txt", "w") as f:
        f.write(f"dropped predictors (>{GAP_MISSING_FRAC:.0%} missing in window union): "
               f"{sorted(dropped)}\n")
    plot_frame = (pd.concat(plot_frames, ignore_index=True) if plot_frames
                 else pd.DataFrame(columns=["target", "span", "phase", "Date", "WL_obs"]))
    return rows, plot_frame


def run_e4(files: dict[str, Path], out: Path, cfg: "tb.RunConfig", custom_variant: str) -> dict:
    long = pd.read_csv(files["long"], parse_dates=["Date"])
    wide = pd.read_csv(files["Bhagyakul"], parse_dates=["Date"])
    up_ids, bnd_ids = tb.point_ids(wide.columns, tb.UPSTREAM), tb.point_ids(wide.columns, tb.BOUNDARY)
    peak_date, peak_stage = PEAKS[BHAGYAKUL]
    rows, plot_frame = run_span(long, BHAGYAKUL, "Bhagyakul", up_ids, bnd_ids, pd.Timestamp(peak_date),
                                cfg, custom_variant, out, "e4")
    pd.DataFrame(rows).to_csv(out / "metrics" / "e4_spans.csv", index=False)
    plot_frame.to_csv(out / "preds" / "e4_spans.csv", index=False)
    return {"peak_date": peak_date, "peak_stage_m": peak_stage}


def run_e5(files: dict[str, Path], out: Path, cfg: "tb.RunConfig", custom_variant: str) -> dict:
    """Baruria and Sureswar, at their own record peaks (never mutated by --smoke,
    unlike tb.TARGETS, which E1 truncates to 2 targets for speed)."""
    long = pd.read_csv(files["long"], parse_dates=["Date"])
    all_rows, all_plot_frames = [], []
    for target_id, name in ((BARURIA, "Baruria"), (SURESWAR, "Sureswar")):
        wide = pd.read_csv(files[name], parse_dates=["Date"])
        up_ids, bnd_ids = tb.point_ids(wide.columns, tb.UPSTREAM), tb.point_ids(wide.columns, tb.BOUNDARY)
        peak_date, _ = PEAKS[target_id]
        rows, plot_frame = run_span(long, target_id, name, up_ids, bnd_ids, pd.Timestamp(peak_date),
                                    cfg, custom_variant, out, "e5")
        all_rows += rows
        all_plot_frames.append(plot_frame)
    pd.DataFrame(all_rows).to_csv(out / "metrics" / "e5_spans.csv", index=False)
    pd.concat(all_plot_frames, ignore_index=True).to_csv(out / "preds" / "e5_spans.csv", index=False)
    return {"targets": ["Baruria", "Sureswar"]}


# --------------------------------------------------------------------------- CLI / main
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--exp", choices=("all", "e0", "e1", "e2", "e3", "e4", "e5", "e4e5",
                                     "e1ablationgnn", "e2ablation"), default="all")
    p.add_argument("--smoke", action="store_true", help="1 epoch, 1 seed, 2 targets")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--custom-variant", choices=("pistgnn", "stgnn_nophys"), default=None,
                   help="E1's decision, for standalone e2/e3/e4/e5 reruns; if omitted, "
                        "read from --out/run_manifest.json, else default to pistgnn")
    return p.parse_args(argv)


def make_cfg(args: argparse.Namespace) -> "tb.RunConfig":
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = tb.RunConfig(lstm_seeds=SEEDS, gnn_seeds=SEEDS, device=device)
    if args.smoke:
        cfg = replace(cfg, lgbm_max_rounds=50, lgbm_patience=10, lstm_max_epochs=1,
                     lstm_patience=1, lstm_seeds=SEEDS[:1], gnn_max_epochs=1,
                     gnn_patience=1, gnn_seeds=SEEDS[:1])
    return cfg


def resolve_files(args: argparse.Namespace) -> dict[str, Path]:
    files = {"long": tb.find_data_file(tb.LONG_FILE, args.data_dir)}
    files.update({name: tb.find_data_file(f, args.data_dir) for _, name, f in tb.TARGETS})
    return files


def check_checksums(files: dict[str, Path]) -> dict[str, str]:
    """Verify every input against metadata/checksums_sha256.csv; abort on mismatch."""
    roots = [tb.script_dir().parent / "metadata", Path("/kaggle/input")]
    csum_path = tb.search_file("checksums_sha256.csv", roots)
    got = {name: tb.sha256(p) for name, p in files.items() if name != "long" or True}
    if csum_path is None:
        print("  checksums_sha256.csv not found; skipping the abort-on-mismatch check")
        return got
    expected = pd.read_csv(csum_path).set_index("File")["SHA256"].to_dict()
    name_to_relpath = {"long": "processed/ganges_padma_hydromet_dataset_2011_2025.csv",
                       "Baruria": "processed/wide/wide_sw91_9l_baruria_transit.csv",
                       "Bhagyakul": "processed/wide/wide_sw93_4l_bhagyakul.csv",
                       "Sureswar": "processed/wide/wide_sw95_sureswar.csv"}
    for name, rel in name_to_relpath.items():
        if rel in expected:
            assert got[name] == expected[rel], f"SHA-256 mismatch for {name} ({rel})"
    return got


def main(argv: list[str] | None = None) -> None:
    t0 = time.time()
    args = parse_args(argv)
    cfg = make_cfg(args)
    tb.set_seed(tb.SEED)
    tb.configure_torch_determinism()
    out = args.out or Path(OUT_DEFAULT)
    for sub in ("metrics", "preds", "formats", "arff", "tables"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    files = resolve_files(args)
    shas = check_checksums(files)
    if args.smoke:
        tb.TARGETS = tb.TARGETS[:2]
    manifest: dict = {"seeds": list(SEEDS), "device": cfg.device, "smoke": args.smoke,
                      "torch": torch.__version__, "input_sha256": shas,
                      "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    results: dict = {}
    plan = ({"all": ("e0", "e1", "e1ablationgnn", "e2", "e3", "e4", "e5"), "e4e5": ("e4", "e5"),
            "e2ablation": ("e2", "e1ablationgnn")}
           .get(args.exp, (args.exp,)))
    # Default until E1 runs; a standalone e2/e4/e5 rerun needs E1's real decision,
    # taken from --custom-variant, else an existing manifest in --out, else pistgnn.
    custom_variant = args.custom_variant or "pistgnn"
    custom_variant_source = "default"
    if args.custom_variant:
        custom_variant_source = "cli"
    elif "e1" not in plan:
        prior = out / "run_manifest.json"
        if prior.exists() and "custom_variant" in json.loads(prior.read_text()):
            custom_variant = json.loads(prior.read_text())["custom_variant"]
            custom_variant_source = "prior_manifest"
    manifest["custom_variant"] = custom_variant  # overwritten below if e1 runs (its own decision)
    manifest["custom_label"] = ("PI-STGNN" if custom_variant == "pistgnn"
                                else "RS-GNN (river-sweep graph network)")
    manifest["custom_variant_source"] = custom_variant_source  # overwritten to "this_run" if e1 runs below
    for exp in plan:
        t1 = time.time()
        print(f"[{time.time() - t0:7.1f}s] running {exp}", flush=True)
        if exp == "e0":
            results["e0"] = run_e0(files, out)
        elif exp == "e1":
            results["e1"] = run_e1(files, out, cfg)
            custom_variant = results["e1"]["custom_variant"]
            manifest["custom_variant"] = custom_variant
            manifest["custom_label"] = results["e1"]["custom_label"]
            manifest["custom_variant_source"] = "this_run"
        elif exp == "e1ablationgnn":
            results["e1_ablation_gnn"] = run_ablation_gnn(files, out, cfg, custom_variant)
        elif exp == "e2":
            results["e2"] = run_e2(files, out, cfg, custom_variant)
        elif exp == "e3":
            results["e3"] = run_e3(files, out)
        elif exp == "e4":
            results["e4"] = run_e4(files, out, cfg, custom_variant)
        elif exp == "e5":
            results["e5"] = run_e5(files, out, cfg, custom_variant)
        manifest[f"{exp}_runtime_s"] = round(time.time() - t1, 1)
    manifest["total_runtime_s"] = round(time.time() - t0, 1)
    manifest["results"] = {k: v for k, v in results.items()}
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"Done in {time.time() - t0:.1f}s; outputs in {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
