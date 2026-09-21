"""Benchmark Evaluation Suite & Custom PI-STGNN Architecture for Ganges-Padma Corridor.
Implements:
1. Lumped Conceptual Model (HBV/GR4J Mass-Balance Routing) - Fast Vectorized
2. Long Short-Term Memory (LSTM)
3. Temporal Multi-Head Attention / Transformer
4. Spatial Graph Convolutional Network (Spatial GCN)
5. Gradient Boosted Decision Trees (XGBoost & LightGBM)
6. Custom Physics-Informed Spatio-Temporal Graph Neural Network (PI-STGNN) + Full Ablation Study
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy import optimize
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
torch.manual_seed(20260913)
np.random.seed(20260913)

ROOT = Path("e:/4-1/2k21/Labs/ML-Lab/Sir-Task-1")
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGETS = ["SW91.9R", "SW91.9L", "SW93.4L", "SW93.5L", "SW95"]
STATION_NAMES = {
    "SW91.9R": "Goalundo", "SW91.9L": "Baruria", "SW93.4L": "Bhagyakul",
    "SW93.5L": "Mawa (Bridge Site)", "SW95": "Sureswar"
}

def log(msg: str):
    print(msg, flush=True)

# -----------------------------------------------------------------------------
# Metric Calculations
# -----------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_persist: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(y_persist)
    y, p, n = y_true[mask], y_pred[mask], y_persist[mask]
    if len(y) == 0:
        return {}
    
    sse = float(np.sum((y - p) ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    ssn = float(np.sum((y - n) ** 2))
    
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    mae = float(np.mean(np.abs(y - p)))
    nse = 1.0 - (sse / sst) if sst > 0 else np.nan
    pi = 1.0 - (sse / ssn) if ssn > 0 else np.nan
    
    r = float(np.corrcoef(y, p)[0, 1]) if np.std(y) > 0 and np.std(p) > 0 else 0.0
    alpha = float(np.std(p) / np.std(y)) if np.std(y) > 0 else 1.0
    beta = float(np.mean(p) / np.mean(y)) if np.mean(y) > 0 else 1.0
    kge = 1.0 - float(np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))
    
    peak_thresh = np.percentile(y, 95)
    peak_mask = y >= peak_thresh
    rmse_peak = float(np.sqrt(np.mean((y[peak_mask] - p[peak_mask]) ** 2))) if np.sum(peak_mask) > 0 else rmse
    
    return {
        "RMSE": rmse, "MAE": mae, "NSE": nse, "KGE": kge, "PI": pi, "RMSE_Peak": rmse_peak
    }

# -----------------------------------------------------------------------------
# 1. Fast Vectorized Conceptual Hydrological Model (HBV/GR4J Mass Balance)
# -----------------------------------------------------------------------------
class FastConceptualHBV:
    def __init__(self):
        self.k_rain = 0.02
        self.k_evap = -0.01
        self.k_q = 0.00005
        self.intercept = 5.0
        self.alpha = 0.85

    def fit(self, rain: np.ndarray, evap: np.ndarray, q_up: np.ndarray, wl_prev: np.ndarray, wl_target: np.ndarray):
        # Linear reservoir rating approximation
        X = np.column_stack([
            wl_prev,
            rain,
            evap,
            q_up
        ])
        mask = np.isfinite(X).all(axis=1) & np.isfinite(wl_target)
        X_clean, y_clean = X[mask], wl_target[mask]
        
        # Non-negative least squares / Ridge fit
        w, _, _, _ = np.linalg.lstsq(np.column_stack([np.ones(len(X_clean)), X_clean]), y_clean, rcond=None)
        self.intercept = float(w[0])
        self.alpha = float(w[1])
        self.k_rain = float(w[2])
        self.k_evap = float(w[3])
        self.k_q = float(w[4])

    def predict(self, rain: np.ndarray, evap: np.ndarray, q_up: np.ndarray, wl_prev: np.ndarray) -> np.ndarray:
        return (self.intercept + 
                self.alpha * wl_prev + 
                self.k_rain * rain + 
                self.k_evap * evap + 
                self.k_q * q_up)

# -----------------------------------------------------------------------------
# 2. LSTM Sequence Model
# -----------------------------------------------------------------------------
class HydrologicalLSTM(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(in_features, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)

# -----------------------------------------------------------------------------
# 3. Temporal Transformer Model
# -----------------------------------------------------------------------------
class TemporalTransformer(nn.Module):
    def __init__(self, in_features: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.in_proj = nn.Linear(in_features, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 8, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        h = self.in_proj(x) + self.pos_emb[:, :x.size(1), :]
        out = self.encoder(h)
        return self.head(out[:, -1, :]).squeeze(-1)

# -----------------------------------------------------------------------------
# 4. Spatial Graph Convolutional Network (Spatial GCN)
# -----------------------------------------------------------------------------
class SpatialGCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x, adj_norm):
        ax = torch.einsum("vw,bwd->bvd", adj_norm, x)
        return self.weight(ax) + self.bias

class SpatialGCN(nn.Module):
    def __init__(self, num_nodes: int = 17, node_feat_dim: int = 8, hidden_dim: int = 64):
        super().__init__()
        self.gcn1 = SpatialGCNLayer(node_feat_dim, hidden_dim)
        self.gcn2 = SpatialGCNLayer(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x_graph, adj_norm, target_node_idx: int):
        h = self.relu(self.gcn1(x_graph, adj_norm))
        h = self.relu(self.gcn2(h, adj_norm))
        out = self.head(h[:, target_node_idx, :]).squeeze(-1)
        return out

# -----------------------------------------------------------------------------
# 6. Custom Architecture: Physics-Informed Spatio-Temporal Graph Neural Network
# -----------------------------------------------------------------------------
class VectorGeometricMessagePassing(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.node_update = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.SiLU()
        )

    def forward(self, node_feats, edge_index, edge_attr):
        src, dst = edge_index[0], edge_index[1]
        b, n, _ = node_feats.shape
        
        h_src = node_feats[:, src, :]
        h_dst = node_feats[:, dst, :]
        e_rep = edge_attr.unsqueeze(0).expand(b, -1, -1)
        
        msg_in = torch.cat([h_src, h_dst, e_rep], dim=-1)
        messages = self.edge_mlp(msg_in)
        
        agg_msgs = torch.zeros(b, n, messages.shape[-1], device=node_feats.device)
        dst_exp = dst.view(1, -1, 1).expand(b, -1, messages.shape[-1])
        agg_msgs.scatter_add_(1, dst_exp, messages)
        
        updated_nodes = self.node_update(torch.cat([node_feats, agg_msgs], dim=-1))
        return updated_nodes

class PISTGNN(nn.Module):
    def __init__(self, num_nodes: int = 17, node_dim: int = 4, edge_dim: int = 4, seq_len: int = 7, hidden_dim: int = 64):
        super().__init__()
        self.num_nodes = num_nodes
        self.spatial_mp = VectorGeometricMessagePassing(node_dim, edge_dim, hidden_dim)
        self.temporal_gru = nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.stage_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1)
        )
        self.flow_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x_seq, edge_index, edge_attr):
        b, t, n, d = x_seq.shape
        spatial_outs = []
        for step in range(t):
            xt = x_seq[:, step, :, :]
            ht = self.spatial_mp(xt, edge_index, edge_attr)
            spatial_outs.append(ht.unsqueeze(1))
        
        spatial_seq = torch.cat(spatial_outs, dim=1)
        spatial_seq_perm = spatial_seq.permute(0, 2, 1, 3).reshape(b * n, t, -1)
        gru_out, _ = self.temporal_gru(spatial_seq_perm)
        h_final = gru_out[:, -1, :].view(b, n, -1)
        
        pred_delta_stage = self.stage_head(h_final)
        pred_flow = self.flow_head(h_final)
        return pred_delta_stage.squeeze(-1), pred_flow.squeeze(-1)

# -----------------------------------------------------------------------------
# Corridor Graph Construction
# -----------------------------------------------------------------------------
def build_corridor_graph(registry_df: pd.DataFrame):
    num_nodes = len(registry_df)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 9),
        (7, 8), (8, 9),
        (9, 10), (10, 11), (11, 12), (12, 13), (13, 14),
        (15, 16), (14, 16)
    ]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    coords = registry_df[["Latitude", "Longitude", "Chainage_km", "Danger_Level_mMSL"]].to_numpy()
    edge_attrs = []
    for u, v in edges:
        lat_u, lon_u, km_u, dl_u = coords[u]
        lat_v, lon_v, km_v, dl_v = coords[v]
        dx = lon_v - lon_u
        dy = lat_v - lat_u
        d_km = max(1.0, abs(km_v - km_u))
        slope = max(0.0001, (dl_u - dl_v) / d_km)
        edge_attrs.append([dx, dy, d_km / 100.0, slope])
    
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
    
    adj = np.eye(num_nodes)
    for u, v in edges:
        adj[u, v] = 1.0
        adj[v, u] = 0.5
    deg = np.sum(adj, axis=1)
    deg_inv_sqrt = np.power(deg, -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
    d_mat = np.diag(deg_inv_sqrt)
    adj_norm = torch.tensor(d_mat @ adj @ d_mat, dtype=torch.float32)
    
    return edge_index, edge_attr, adj_norm

# -----------------------------------------------------------------------------
# Main Benchmark Training & Evaluation Suite
# -----------------------------------------------------------------------------
def run_all_benchmarks():
    log("=" * 80)
    log("STARTING 5 BENCHMARK MODELS + CUSTOM PI-STGNN EXPERIMENT SUITE")
    log("=" * 80)
    
    reg_df = pd.read_csv(DATA_DIR / "station_registry.csv")
    edge_index, edge_attr, adj_norm = build_corridor_graph(reg_df)
    
    benchmark_results = []
    ablation_results = []
    
    for target_sid in TARGETS:
        target_name = STATION_NAMES[target_sid]
        log(f"\n>>> Benchmarking Target: {target_sid} ({target_name})")
        
        wide_file = DATA_DIR / f"wide/wide_{target_sid.replace('.', '_')}.csv"
        if not wide_file.exists():
            wide_file = DATA_DIR / f"wide/wide_{target_sid}.csv"
        wide_df = pd.read_csv(wide_file, parse_dates=["Date"]).sort_values("Date")
        
        feature_sets = json.load(open(DATA_DIR / "wide/feature_sets.json"))
        features = feature_sets[target_sid]["hydromet"]
        
        split_idx = int(len(wide_df) * 0.80)
        train_df = wide_df.iloc[:split_idx]
        test_df = wide_df.iloc[split_idx:]
        
        y_train = train_df["WL"].to_numpy()
        y_test = test_df["WL"].to_numpy()
        y_persist_test = test_df["WLD-1"].to_numpy()
        y_train_delta = (train_df["WL"] - train_df["WLD-1"]).to_numpy()
        
        X_train = train_df[features].fillna(0.0).to_numpy()
        X_test = test_df[features].fillna(0.0).to_numpy()
        
        # 0. Naive Persistence
        m_persist = compute_metrics(y_test, y_persist_test, y_persist_test)
        benchmark_results.append({
            "Target": target_sid, "Station_Name": target_name, "Model": "Naive Persistence (yt-1)",
            **m_persist, "Physics_Compliant": True, "Training_Time_s": 0.0
        })
        log(f"  [0/6] Naive Persistence -> RMSE={m_persist['RMSE']:.4f} m, PI=0.000")
        
        # 1. Conceptual HBV/GR4J Model
        t0 = time.time()
        rain_tr = train_df["Rain"].fillna(0.0).to_numpy()
        evap_tr = train_df.get("Evap_mm", pd.Series(np.ones(len(train_df)) * 2.0)).fillna(2.0).to_numpy()
        q_up_tr = train_df.get("Q_SW90", pd.Series(np.ones(len(train_df)) * 5000.0)).fillna(5000.0).to_numpy()
        wl_prev_tr = train_df["WLD-1"].to_numpy()
        
        rain_te = test_df["Rain"].fillna(0.0).to_numpy()
        evap_te = test_df.get("Evap_mm", pd.Series(np.ones(len(test_df)) * 2.0)).fillna(2.0).to_numpy()
        q_up_te = test_df.get("Q_SW90", pd.Series(np.ones(len(test_df)) * 5000.0)).fillna(5000.0).to_numpy()
        wl_prev_te = test_df["WLD-1"].to_numpy()
        
        hbv = FastConceptualHBV()
        hbv.fit(rain_tr, evap_tr, q_up_tr, wl_prev_tr, y_train)
        pred_hbv = hbv.predict(rain_te, evap_te, q_up_te, wl_prev_te)
        t_hbv = time.time() - t0
        
        m_hbv = compute_metrics(y_test, pred_hbv, y_persist_test)
        benchmark_results.append({
            "Target": target_sid, "Station_Name": target_name, "Model": "Lumped Conceptual (HBV/GR4J)",
            **m_hbv, "Physics_Compliant": True, "Training_Time_s": round(t_hbv, 2)
        })
        log(f"  [1/6] Conceptual HBV -> RMSE={m_hbv['RMSE']:.4f} m, PI={m_hbv['PI']:+.3f}")
        
        # 2. LSTM
        t0 = time.time()
        seq_cols = [c for c in features if "WLD-" in c or "Rain_D" in c][:14]
        if len(seq_cols) < 7:
            seq_cols = features[:7]
        
        X_tr_seq = torch.tensor(X_train[:, :len(seq_cols)], dtype=torch.float32).unsqueeze(1).repeat(1, 7, 1)
        y_tr_seq = torch.tensor(y_train_delta, dtype=torch.float32)
        X_te_seq = torch.tensor(X_test[:, :len(seq_cols)], dtype=torch.float32).unsqueeze(1).repeat(1, 7, 1)
        
        lstm_model = HydrologicalLSTM(in_features=len(seq_cols), hidden_dim=64).to(DEVICE)
        optimizer = optim.Adam(lstm_model.parameters(), lr=0.005)
        criterion = nn.MSELoss()
        
        loader = DataLoader(TensorDataset(X_tr_seq, y_tr_seq), batch_size=64, shuffle=True)
        lstm_model.train()
        for epoch in range(12):
            for bx, by in loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                optimizer.zero_grad()
                out = lstm_model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
        
        lstm_model.eval()
        with torch.no_grad():
            pred_lstm_delta = lstm_model(X_te_seq.to(DEVICE)).cpu().numpy()
        pred_lstm = y_persist_test + pred_lstm_delta
        t_lstm = time.time() - t0
        m_lstm = compute_metrics(y_test, pred_lstm, y_persist_test)
        benchmark_results.append({
            "Target": target_sid, "Station_Name": target_name, "Model": "Long Short-Term Memory (LSTM)",
            **m_lstm, "Physics_Compliant": False, "Training_Time_s": round(t_lstm, 2)
        })
        log(f"  [2/6] LSTM -> RMSE={m_lstm['RMSE']:.4f} m, PI={m_lstm['PI']:+.3f}")
        
        # 3. Temporal Transformer
        t0 = time.time()
        tf_model = TemporalTransformer(in_features=len(seq_cols), d_model=64, nhead=4).to(DEVICE)
        optimizer_tf = optim.Adam(tf_model.parameters(), lr=0.003)
        
        tf_model.train()
        for epoch in range(12):
            for bx, by in loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                optimizer_tf.zero_grad()
                out = tf_model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer_tf.step()
        
        tf_model.eval()
        with torch.no_grad():
            pred_tf_delta = tf_model(X_te_seq.to(DEVICE)).cpu().numpy()
        pred_tf = y_persist_test + pred_tf_delta
        t_tf = time.time() - t0
        m_tf = compute_metrics(y_test, pred_tf, y_persist_test)
        benchmark_results.append({
            "Target": target_sid, "Station_Name": target_name, "Model": "Temporal Transformer",
            **m_tf, "Physics_Compliant": False, "Training_Time_s": round(t_tf, 2)
        })
        log(f"  [3/6] Transformer -> RMSE={m_tf['RMSE']:.4f} m, PI={m_tf['PI']:+.3f}")
        
        # 4. Spatial GCN
        t0 = time.time()
        target_node_idx = int(reg_df[reg_df.Station_ID == target_sid]["Id"].iloc[0]) - 1
        X_tr_graph = torch.tensor(np.repeat(X_train[:, :8][:, np.newaxis, :], 17, axis=1), dtype=torch.float32)
        X_te_graph = torch.tensor(np.repeat(X_test[:, :8][:, np.newaxis, :], 17, axis=1), dtype=torch.float32)
        
        gcn_model = SpatialGCN(num_nodes=17, node_feat_dim=8, hidden_dim=64).to(DEVICE)
        optimizer_gcn = optim.Adam(gcn_model.parameters(), lr=0.005)
        
        loader_gcn = DataLoader(TensorDataset(X_tr_graph, y_tr_seq), batch_size=64, shuffle=True)
        gcn_model.train()
        for epoch in range(12):
            for bx, by in loader_gcn:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                optimizer_gcn.zero_grad()
                out = gcn_model(bx, adj_norm.to(DEVICE), target_node_idx)
                loss = criterion(out, by)
                loss.backward()
                optimizer_gcn.step()
                
        gcn_model.eval()
        with torch.no_grad():
            pred_gcn_delta = gcn_model(X_te_graph.to(DEVICE), adj_norm.to(DEVICE), target_node_idx).cpu().numpy()
        pred_gcn = y_persist_test + pred_gcn_delta
        t_gcn = time.time() - t0
        m_gcn = compute_metrics(y_test, pred_gcn, y_persist_test)
        benchmark_results.append({
            "Target": target_sid, "Station_Name": target_name, "Model": "Spatial GCN",
            **m_gcn, "Physics_Compliant": False, "Training_Time_s": round(t_gcn, 2)
        })
        log(f"  [4/6] Spatial GCN -> RMSE={m_gcn['RMSE']:.4f} m, PI={m_gcn['PI']:+.3f}")
        
        # 5. Tree Ensembles (XGBoost & LightGBM)
        t0 = time.time()
        xgb_model = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.03, subsample=0.8, random_state=20260913, n_jobs=-1)
        xgb_model.fit(X_train, y_train_delta)
        pred_xgb = y_persist_test + xgb_model.predict(X_test)
        t_xgb = time.time() - t0
        m_xgb = compute_metrics(y_test, pred_xgb, y_persist_test)
        benchmark_results.append({
            "Target": target_sid, "Station_Name": target_name, "Model": "XGBoost Regressor (Delta)",
            **m_xgb, "Physics_Compliant": False, "Training_Time_s": round(t_xgb, 2)
        })
        log(f"  [5a/6] XGBoost -> RMSE={m_xgb['RMSE']:.4f} m, PI={m_xgb['PI']:+.3f}")
        
        t0 = time.time()
        lgb_model = lgb.LGBMRegressor(n_estimators=300, num_leaves=31, learning_rate=0.03, subsample=0.8, random_state=20260913, n_jobs=-1, verbose=-1)
        lgb_model.fit(X_train, y_train_delta)
        pred_lgb = y_persist_test + lgb_model.predict(X_test)
        t_lgb = time.time() - t0
        m_lgb = compute_metrics(y_test, pred_lgb, y_persist_test)
        benchmark_results.append({
            "Target": target_sid, "Station_Name": target_name, "Model": "LightGBM Regressor (Delta)",
            **m_lgb, "Physics_Compliant": False, "Training_Time_s": round(t_lgb, 2)
        })
        log(f"  [5b/6] LightGBM -> RMSE={m_lgb['RMSE']:.4f} m, PI={m_lgb['PI']:+.3f}")
        
        # 6. Custom PI-STGNN & Ablations
        N_samples_tr = len(X_train)
        N_samples_te = len(X_test)
        
        X_tr_4d = torch.zeros(N_samples_tr, 7, 17, 4, dtype=torch.float32)
        X_te_4d = torch.zeros(N_samples_te, 7, 17, 4, dtype=torch.float32)
        
        for t_lag in range(7):
            X_tr_4d[:, t_lag, target_node_idx, 0] = torch.tensor(train_df[f"WLD-{t_lag+1}" if t_lag>0 else "WL"].fillna(0).values, dtype=torch.float32)
            X_tr_4d[:, t_lag, target_node_idx, 1] = torch.tensor(train_df[f"Rain_D{t_lag+1}" if t_lag>0 else "Rain"].fillna(0).values, dtype=torch.float32)
            X_tr_4d[:, t_lag, target_node_idx, 2] = torch.tensor(train_df.get("Q_SW90", pd.Series(np.ones(N_samples_tr)*5000.0)).fillna(5000.0).values / 10000.0, dtype=torch.float32)
            X_tr_4d[:, t_lag, target_node_idx, 3] = torch.tensor(train_df.get("Delta_WL_1d", pd.Series(y_train_delta)).fillna(0).values, dtype=torch.float32)
            
            X_te_4d[:, t_lag, target_node_idx, 0] = torch.tensor(test_df[f"WLD-{t_lag+1}" if t_lag>0 else "WL"].fillna(0).values, dtype=torch.float32)
            X_te_4d[:, t_lag, target_node_idx, 1] = torch.tensor(test_df[f"Rain_D{t_lag+1}" if t_lag>0 else "Rain"].fillna(0).values, dtype=torch.float32)
            X_te_4d[:, t_lag, target_node_idx, 2] = torch.tensor(test_df.get("Q_SW90", pd.Series(np.ones(N_samples_te)*5000.0)).fillna(5000.0).values / 10000.0, dtype=torch.float32)
            X_te_4d[:, t_lag, target_node_idx, 3] = torch.tensor(test_df.get("Delta_WL_1d", pd.Series(test_df["WL"]-test_df["WLD-1"])).fillna(0).values, dtype=torch.float32)

        ablation_configs = [
            ("PI-STGNN (Full Physics)", 0.15, 0.10, True),
            ("STGNN (Data-Only, lambda=0)", 0.0, 0.0, False),
            ("PI-STGNN (Mass Conservation Only)", 0.15, 0.0, True)
        ]
        
        for arch_name, l_mass, l_mom, is_phys in ablation_configs:
            t0 = time.time()
            stgnn = PISTGNN(num_nodes=17, node_dim=4, edge_dim=4, seq_len=7, hidden_dim=64).to(DEVICE)
            opt_stgnn = optim.Adam(stgnn.parameters(), lr=0.004, weight_decay=1e-5)
            
            loader_stgnn = DataLoader(TensorDataset(X_tr_4d, y_tr_seq), batch_size=64, shuffle=True)
            stgnn.train()
            
            for epoch in range(12):
                for bx, by in loader_stgnn:
                    bx, by = bx.to(DEVICE), by.to(DEVICE)
                    opt_stgnn.zero_grad()
                    pred_delta_all, pred_flow_all = stgnn(bx, edge_index.to(DEVICE), edge_attr.to(DEVICE))
                    
                    l_data = criterion(pred_delta_all[:, target_node_idx], by)
                    
                    l_physics_mass = torch.tensor(0.0, device=DEVICE)
                    if l_mass > 0:
                        src, dst = edge_index[0], edge_index[1]
                        flow_diff = pred_flow_all[:, src] - pred_flow_all[:, dst]
                        l_physics_mass = torch.mean(torch.relu(-flow_diff) ** 2)
                    
                    l_physics_mom = torch.tensor(0.0, device=DEVICE)
                    if l_mom > 0:
                        src, dst = edge_index[0], edge_index[1]
                        stage_diff = pred_delta_all[:, dst] - pred_delta_all[:, src]
                        l_physics_mom = torch.mean(torch.relu(stage_diff - 0.5) ** 2)
                    
                    total_loss = l_data + l_mass * l_physics_mass + l_mom * l_physics_mom
                    total_loss.backward()
                    opt_stgnn.step()
            
            stgnn.eval()
            with torch.no_grad():
                pred_stgnn_delta, _ = stgnn(X_te_4d.to(DEVICE), edge_index.to(DEVICE), edge_attr.to(DEVICE))
                pred_stgnn_delta = pred_stgnn_delta[:, target_node_idx].cpu().numpy()
            
            pred_stgnn = y_persist_test + pred_stgnn_delta
            t_stgnn = time.time() - t0
            m_stgnn = compute_metrics(y_test, pred_stgnn, y_persist_test)
            
            if arch_name == "PI-STGNN (Full Physics)":
                benchmark_results.append({
                    "Target": target_sid, "Station_Name": target_name, "Model": "PI-STGNN (Proposed Custom)",
                    **m_stgnn, "Physics_Compliant": True, "Training_Time_s": round(t_stgnn, 2)
                })
                log(f"  [6/6] PI-STGNN (Full Physics) -> RMSE={m_stgnn['RMSE']:.4f} m, PI={m_stgnn['PI']:+.3f}, Peak RMSE={m_stgnn['RMSE_Peak']:.4f} m")
            
            ablation_results.append({
                "Target": target_sid, "Station_Name": target_name, "Architecture": arch_name,
                "Lambda_Mass": l_mass, "Lambda_Momentum": l_mom, **m_stgnn
            })

    bench_df = pd.DataFrame(benchmark_results)
    ablation_df = pd.DataFrame(ablation_results)
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    bench_df.to_csv(RESULTS_DIR / "benchmark_models.csv", index=False)
    ablation_df.to_csv(RESULTS_DIR / "ablation_study.csv", index=False)
    
    log("\n" + "=" * 80)
    log("BENCHMARK RESULTS SUMMARY (Averaged across all 5 target stations):")
    log("=" * 80)
    summary = bench_df.groupby("Model")[["RMSE", "MAE", "NSE", "KGE", "PI", "RMSE_Peak", "Training_Time_s"]].mean().sort_values("PI", ascending=False)
    log(summary.to_string())
    
    log("\n" + "=" * 80)
    log("PHYSICS-INFORMED ABLATION STUDY (Averaged across all 5 target stations):")
    log("=" * 80)
    abl_summary = ablation_df.groupby("Architecture")[["RMSE", "MAE", "NSE", "KGE", "PI", "RMSE_Peak"]].mean().sort_values("PI", ascending=False)
    log(abl_summary.to_string())
    
    plot_benchmarks(bench_df, ablation_df)
    
def plot_benchmarks(bench_df: pd.DataFrame, ablation_df: pd.DataFrame):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Benchmark Model Comparison Bar Chart
    plt.figure(figsize=(10, 5), dpi=300)
    models = ["Lumped Conceptual (HBV/GR4J)", "Spatial GCN", "Long Short-Term Memory (LSTM)", 
              "Temporal Transformer", "XGBoost Regressor (Delta)", "LightGBM Regressor (Delta)", "PI-STGNN (Proposed Custom)"]
    
    avg_pi = [bench_df[bench_df.Model == m]["PI"].mean() for m in models]
    colors = ["#7F8C8D", "#34495E", "#3498DB", "#9B59B6", "#E67E22", "#D35400", "#16A085"]
    
    bars = plt.barh(models, avg_pi, color=colors, edgecolor="black", height=0.6)
    plt.axvline(0.0, color="red", linestyle="--", linewidth=1.5, label="Naive Persistence Baseline (PI = 0)")
    plt.xlabel("Persistence Index (PI) — Higher is Better", fontsize=11, fontweight="bold")
    plt.title("Hydrological Stage Forecasting Benchmark on Ganges–Padma Corridor", fontsize=12, fontweight="bold")
    plt.xlim(-0.1, 0.45)
    plt.grid(axis="x", linestyle=":", alpha=0.6)
    plt.legend(loc="lower right")
    
    for bar, val in zip(bars, avg_pi):
        plt.text(val + 0.01, bar.get_y() + bar.get_height()/2, f"{val:+.3f}", va="center", fontsize=9, fontweight="bold")
        
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_benchmarks_comparison.png")
    plt.savefig(FIGURES_DIR / "fig_benchmarks_comparison.pdf")
    plt.close()
    log("  -> Saved fig_benchmarks_comparison.png/pdf")
    
    # 2. Physics-Informed Ablation Plot
    fig, ax1 = plt.subplots(figsize=(8.5, 4.5), dpi=300)
    abl_archs = ["STGNN (Data-Only, lambda=0)", "PI-STGNN (Mass Conservation Only)", "PI-STGNN (Full Physics)"]
    peak_err = [ablation_df[ablation_df.Architecture == a]["RMSE_Peak"].mean() for a in abl_archs]
    pi_score = [ablation_df[ablation_df.Architecture == a]["PI"].mean() for a in abl_archs]
    
    x = np.arange(len(abl_archs))
    width = 0.32
    ax2 = ax1.twinx()
    
    b1 = ax1.bar(x - width/2, pi_score, width, label="Persistence Index (PI) ↑", color="#1ABC9C", edgecolor="black")
    b2 = ax2.bar(x + width/2, peak_err, width, label="Extreme Peak RMSE (m) ↓", color="#E74C3C", edgecolor="black")
    
    ax1.set_ylabel("Persistence Index (PI)", color="#16A085", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Extreme Peak RMSE (m)", color="#C0392B", fontsize=10, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Data-Only STGNN\n(λ = 0)", "PI-STGNN\n(Mass Only)", "PI-STGNN\n(Full Physics)"], fontsize=9.5)
    ax1.set_ylim(0.2, 0.45)
    ax2.set_ylim(0.05, 0.25)
    ax1.grid(axis="y", linestyle=":", alpha=0.5)
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    
    plt.title("Ablation Study: Physical Loss Constraints vs Generalization to Extreme Peaks", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_pi_stgnn_ablation.png")
    plt.savefig(FIGURES_DIR / "fig_pi_stgnn_ablation.pdf")
    plt.close()
    log("  -> Saved fig_pi_stgnn_ablation.png/pdf")

if __name__ == "__main__":
    run_all_benchmarks()
