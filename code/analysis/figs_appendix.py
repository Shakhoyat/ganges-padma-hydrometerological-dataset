"""Travel-time figures (Section 10) drawn for a landscape page (17 cm wide, 8-9 pt text).

Same data as report-v3/make_figures_r3.py::fig_travel and make_figures.py::fig_upstream_signal,
which were drawn on a 25 cm canvas and shrank to about 6 pt text in the PDF.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as cm

R3 = cm.DATASET / "results" / "r3" / "metrics"
TARGET_IDS = {11: "Baruria", 12: "Bhagyakul", 15: "Sureswar"}


def fig_travel() -> None:
    peak = pd.read_csv(R3 / "e3_peak_lag_chainage.csv").sort_values("chainage_km")
    sup = pd.read_csv(R3 / "e3_superposed_epoch.csv")
    fk = pd.read_csv(R3 / "e3_farakka_trace.csv")
    fig, axes = plt.subplots(1, 2, figsize=(cm.LAND_W, 7.0 * cm.CM), gridspec_kw={"wspace": 0.2,
                                                                                "width_ratios": [1.1, 1]})
    ax = axes[0]
    ax.errorbar(peak.chainage_km, peak.lag, yerr=[peak.lag - peak.lag_ci_lo, peak.lag_ci_hi - peak.lag],
                fmt="o", color=cm.C["blue"], ms=4, ecolor=cm.C["light"], capsize=2, lw=1)
    slope, icpt = np.polyfit(peak.chainage_km, peak.lag, 1)
    xs = np.linspace(0, peak.chainage_km.max(), 20)
    ax.plot(xs, icpt + slope * xs, color=cm.C["ink"], ls="--", lw=1)
    ax.text(10, 2.6, f"fitted celerity\n{1 / slope:.1f} km/day", fontsize=8)
    label_pos = {11: (150, 1.7), 12: (215, 2.3), 15: (235, 3.6)}
    for tid, name in TARGET_IDS.items():
        r = peak[peak.target_id == tid].iloc[0]
        ax.annotate(name, (r.chainage_km, r.lag), xytext=label_pos[tid], textcoords="data",
                    fontsize=7.8, ha="center", fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=cm.C["grey"], lw=0.6))
    ax.set_xlabel("Chainage from Panka (km)")
    ax.set_ylabel("Lag of peak correlation (days)")
    ax.set_ylim(-0.6, 4.5)
    ax.set_title("(a) Lag grows downstream")
    ax = axes[1]
    days = np.arange(-2, -2 + len(sup))
    shades = {"1": "#9ecae1", "4": "#6baed6", "10": "#2171b5", "12": "#08519c", "15": "#08306b"}
    for col in sup.columns:
        ax.plot(days, sup[col], color=shades.get(col, cm.C["blue"]), lw=1.6,
                label=f"{cm.SHORT_NAME[int(col)]} ({col})")
    f = fk[fk.day.between(days[0], days[-1])]
    ax.plot(f.day, f.dwl_panka, ":", color=cm.C["red"], marker="o", ms=3, lw=1.2,
            label="Panka, 26 Aug 2024 (not counted)")
    ax.axvline(0, color=cm.C["grey"], lw=0.7, ls="--")
    ax.set_xticks(range(-2, 11, 2))
    ax.set_xlabel("Days from a Panka surge (day 0)")
    ax.set_ylabel("Mean daily change (m)")
    ax.set_title("(b) A surge traced downstream")
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.08),
               fontsize=7.5)
    cm.save(fig, "fig_ap_travel")


def fig_signal() -> None:
    stage = cm.stage_matrix(cm.load_long())
    dw = stage.diff()
    rows = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    lags = range(8)
    fig, axes = plt.subplots(1, 3, figsize=(cm.LAND_W, 7.4 * cm.CM), sharey=True, gridspec_kw={"wspace": 0.05})
    for ax, (tid, name) in zip(axes, TARGET_IDS.items()):
        m = np.full((len(rows), 8), np.nan)
        for i, g in enumerate(rows):
            if g == tid:
                continue
            for k in lags:
                m[i, k] = dw[tid].corr(dw[g].shift(k))
        im = ax.imshow(m, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        for i in range(len(rows)):
            if np.all(np.isnan(m[i])):
                ax.text(3.5, i, "target", ha="center", va="center", fontsize=7, color=cm.C["grey"])
                continue
            k = int(np.nanargmax(m[i]))
            ax.text(k, i, f"{m[i, k]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if m[i, k] > 0.6 else cm.C["ink"])
        ax.set_xticks(list(lags))
        ax.tick_params(axis="both", length=0)
        ax.set_xlabel("Lag $k$ (days)")
        ax.set_title(f"({'abc'[list(TARGET_IDS).index(tid)]}) {name} ({tid})")
        ax.grid(False)
    axes[0].set_yticks(range(len(rows)), [f"{cm.SHORT_NAME[g]} ({g})" for g in rows], fontsize=7.5)
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cb.set_label(r"corr($\Delta WL_{target}(t)$, $\Delta WL_{gauge}(t-k)$)")
    cm.save(fig, "fig_ap_signal")


def main() -> None:
    cm.use_style()
    fig_travel()
    fig_signal()


if __name__ == "__main__":
    main()
