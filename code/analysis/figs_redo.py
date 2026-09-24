"""Portrait re-drawings of the v3.1 figures the reviewer found congested.

Same data as before (released files and the stored Kaggle predictions in
results/r3/preds; nothing is retrained). Rules: full text width, <= 2 panels
per row, legends outside the data, full month names / years on time axes,
no landscape pages.
"""
from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as cm

R3 = cm.DATASET / "results" / "r3"
MODELS = [("ridge", "Ridge", "#0072B2"), ("tgcn", "T-GCN", "#E69F00"),
          ("lstm", "LSTM", "#CC79A7"), ("stgnn_nophys", "RS-GNN (custom)", "#009E73")]
ORDER = list(range(1, 18))
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
ERA_LINES = [(pd.Timestamp("2014-11-26"), "construction\nstarts"), (pd.Timestamp("2022-06-25"), "bridge\nopens")]


def fig_seasonality() -> None:
    long = cm.load_long()
    stage = cm.stage_matrix(long)
    med = stage.median()
    monthly = stage.groupby(stage.index.month).median() - med
    fig = plt.figure(figsize=(cm.FULL_W, 17 * cm.CM))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1], hspace=0.42, wspace=0.3)
    ax = fig.add_subplot(gs[0, :])
    lim = np.nanmax(np.abs(monthly.to_numpy()))
    im = ax.imshow(monthly.T.to_numpy(), aspect="auto", cmap="BrBG", vmin=-lim, vmax=lim)
    for i, sid in enumerate(monthly.columns):
        j = int(np.nanargmax(monthly[sid].to_numpy()))
        ax.plot(j, i, "o", ms=3.5, mfc="white", mec=cm.C["ink"], mew=0.6)
    ax.set_xticks(range(12), MONTHS)
    ax.set_yticks(range(17), [f"{cm.SHORT_NAME[s]} ({s})" for s in monthly.columns], fontsize=7.5)
    ax.axhline(10.5, color=cm.C["ink"], lw=1)
    ax.text(11.6, 10.3, "tidal below", fontsize=7, ha="right", va="bottom")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cb.set_label("Monthly median minus\nstation median (m)")
    ax.set_title("(a) Monsoon cycle at every gauge (white dot: peak month)")
    ax2 = fig.add_subplot(gs[1, 0])
    tr = long.dropna(subset=["WL_Trend"])
    share = pd.crosstab(tr.Date.dt.month, tr.WL_Trend, normalize="index") * 100
    cols = {0: ("Falling", "#bf812d"), 1: ("Steady", "#d9d9d9"), 2: ("Rising", "#2166ac")}
    bottom = np.zeros(12)
    for k, (name, col) in cols.items():
        ax2.bar(range(12), share[k], bottom=bottom, color=col, label=name, width=0.8)
        bottom += share[k].to_numpy()
    ax2.set_xticks(range(12), [m[0] for m in MONTHS])
    ax2.set_ylabel("Share of station-days (%)")
    ax2.set_ylim(0, 100)
    ax2.set_title("(b) WL_Trend class by month")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=7.5)
    ax3 = fig.add_subplot(gs[1, 1])
    rain = long.groupby(long.Date.dt.month).Rainfall_mm.mean()
    ax3.fill_between(range(12), rain.to_numpy(), color=cm.C["sky"], alpha=0.35, lw=0)
    ax3.plot(range(12), rain.to_numpy(), marker="o", ms=3.5, color=cm.C["blue"])
    ax3.set_xticks(range(12), [m[0] for m in MONTHS])
    ax3.set_ylabel("Mean daily rainfall (mm/day)")
    ax3.set_title("(c) Rainfall at the assigned gauges")
    cm.save(fig, "fig_r_seasonality")


def fig_dynamics() -> None:
    stage = cm.stage_matrix(cm.load_long())
    d = stage.diff()
    corr = d.corr()
    fig = plt.figure(figsize=(cm.FULL_W, 20 * cm.CM))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.45, 0.8], hspace=0.35)
    ax = fig.add_subplot(gs[0])
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(17):
        for j in range(17):
            v = corr.iat[i, j]
            ax.text(j, i, f"{v:.2f}".replace("0.", ".").replace("1.00", "1"), ha="center", va="center",
                    fontsize=5.8, color="white" if abs(v) > 0.6 else cm.C["ink"])
    names = [f"{cm.SHORT_NAME[s]} ({s})" for s in corr.index]
    ax.set_xticks(range(17), names, rotation=50, ha="right", fontsize=7)
    ax.set_yticks(range(17), names, fontsize=7)
    for b in (6.5, 8.5, 14.5):
        ax.axhline(b, color="white", lw=1.2)
        ax.axvline(b, color="white", lw=1.2)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.01)
    cb.set_label("Pearson r of same-day daily change")
    ax.set_title("(a) Which gauges move together (white lines: Ganges | Jamuna | Padma | Meghna)")
    ax2 = fig.add_subplot(gs[1])
    lags = range(1, 11)
    acf = pd.DataFrame({s: [d[s].autocorr(k) for k in lags] for s in ORDER}, index=lags)
    others = acf.drop(columns=list(cm.TARGETS))
    ax2.fill_between(lags, others.min(1), others.max(1), color=cm.C["light"], alpha=0.7, lw=0,
                     label="other 14 gauges (range)")
    ax2.axvspan(1, 7, color=cm.C["yellow"], alpha=0.12, lw=0, label="released window WLD-1..7")
    for t, name in cm.TARGETS.items():
        ax2.plot(lags, acf[t], marker="o", ms=3.5, lw=1.5, label=name,
                 color={11: cm.C["blue"], 12: cm.C["red"], 15: cm.C["green"]}[t])
    ax2.axhline(0, color=cm.C["ink"], lw=0.6)
    ax2.set_xticks(list(lags))
    ax2.set_xlabel("Lag (days)")
    ax2.set_ylabel("Autocorrelation of\ndaily change")
    ax2.set_title("(b) How long a rise or fall is remembered")
    ax2.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=7.5)
    cm.save(fig, "fig_r_dynamics")


def _era_lines(ax: plt.Axes, label: bool) -> None:
    for d, txt in ERA_LINES:
        ax.axvline(d, color=cm.C["ink"], ls="--", lw=0.9)
        if label:
            ax.text(d, 1.02, txt, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=7)


def fig_hydrographs() -> None:
    stage = cm.stage_matrix(cm.load_long())
    danger = cm.load_registry()["Danger_Level_mMSL"]
    fig, axes = plt.subplots(3, 1, figsize=(cm.FULL_W, 16 * cm.CM), sharex=True)
    for i, (ax, (t, name)) in enumerate(zip(axes, cm.TARGETS.items())):
        ax.plot(stage.index, stage[t], lw=0.6, color=cm.C["blue"])
        ax.axhline(danger[t], color=cm.C["red"], ls=":", lw=1, label="BWDB danger level")
        ax.axvspan(pd.Timestamp("2023-01-01"), pd.Timestamp("2025-12-31"), color=cm.C["green"], alpha=0.08, lw=0)
        _era_lines(ax, i == 0)
        ax.set_ylabel("WL (mMSL)")
        ax.set_title(f"({'abc'[i]}) {name} ({t})", pad=16 if i == 0 else 4)
    axes[0].text(pd.Timestamp("2024-07-01"), 0.93, "test years", transform=axes[0].get_xaxis_transform(),
                 ha="center", fontsize=7.5, color=cm.C["green"])
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].set_xlabel("Year")
    axes[0].legend(loc="lower left", fontsize=7.5)
    fig.tight_layout()
    cm.save(fig, "fig_r_hydrographs")


def fig_main_results() -> None:
    p = pd.read_csv(R3 / "preds" / "e1_main.csv", parse_dates=["Date"])
    fig = plt.figure(figsize=(cm.FULL_W, 21 * cm.CM))
    gs = fig.add_gridspec(3, 2, width_ratios=[2.1, 1], hspace=0.5, wspace=0.28)
    for row, name in enumerate(cm.TARGETS.values()):
        g = p[p.target == name].set_index("Date").sort_index()
        peak = g.WL_obs.idxmax()
        w = g.loc[peak - pd.Timedelta(days=22): peak + pd.Timedelta(days=22)]
        ax = fig.add_subplot(gs[row, 0])
        ax.plot(w.index, w.WL_obs, color=cm.C["ink"], lw=2.2, label="Observed")
        ax.plot(w.index, w.persistence, color=cm.C["grey"], lw=1, ls="--", label="Persistence")
        for col, lab, c in MODELS:
            ax.plot(w.index, w[col], color=c, lw=1.2, label=lab)
        ax.axvline(peak, color=cm.C["grey"], lw=0.6, ls=":")
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%Y"))
        ax.tick_params(axis="x", labelsize=7)
        ax.set_ylabel("WL (mMSL)")
        ax.set_title(f"({'ace'[row]}) {name}: flood peak {peak:%d %b %Y} $\\pm$22 days", fontsize=9)
        ax2 = fig.add_subplot(gs[row, 1])
        obs = 100 * (g.WL_obs - g.persistence)
        for col, lab, c in MODELS:
            ax2.scatter(obs, 100 * (g[col] - g.persistence), s=3, alpha=0.35, color=c, lw=0)
        lim = np.nanpercentile(np.abs(obs), 99.5) * 1.1
        ax2.plot([-lim, lim], [-lim, lim], color=cm.C["ink"], lw=0.8)
        ax2.set_xlim(-lim, lim)
        ax2.set_ylim(-lim, lim)
        ax2.set_aspect("equal")
        ax2.set_xlabel("Observed daily change (cm)", fontsize=8)
        ax2.set_ylabel("Predicted (cm)", fontsize=8)
        ax2.set_title(f"({'bdf'[row]}) all {len(g):,} test days, 2023-25", fontsize=9)
    h, lab = fig.axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 0.95), fontsize=8)
    cm.save(fig, "fig_r_main_results")


def fig_lagdepth() -> None:
    e2 = pd.read_csv(R3 / "metrics" / "e2_lagdepth.csv")
    names = {"ridge": ("Ridge", "#0072B2"), "tgcn": ("T-GCN", "#E69F00"), "lstm": ("LSTM", "#CC79A7"),
             "custom": ("RS-GNN (custom)", "#009E73")}
    fig, axes = plt.subplots(1, 2, figsize=(cm.FULL_W, 7 * cm.CM))
    for key, (lab, c) in names.items():
        s = e2[e2.model == key].sort_values("k")
        axes[0].plot(s.k, s.pi, marker="o", color=c, label=lab)
        axes[0].fill_between(s.k, s.pi_lo, s.pi_hi, color=c, alpha=0.15, lw=0)
        axes[1].plot(s.k, 100 * s.rmse, marker="o", color=c, label=lab)
    for ax in axes:
        ax.set_xticks([3, 7, 10], ["3", "7\n(released)", "10"])
        ax.set_xlabel("Lag depth k (days of history per gauge)")
    axes[0].set_ylabel("Test PI (Bhagyakul)")
    axes[1].set_ylabel("Test RMSE (cm)")
    axes[0].set_title("(a) Skill vs. history depth (band: 95% CI)")
    axes[1].set_title("(b) Error vs. history depth")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    cm.save(fig, "fig_r_lagdepth")


def main() -> None:
    cm.use_style()
    fig_seasonality()
    fig_dynamics()
    fig_hydrographs()
    fig_main_results()
    fig_lagdepth()


if __name__ == "__main__":
    main()
