"""Figures for studies B (bridge), I (early warning), C (clustering), D (sufficiency).

Layout rule for every figure: full text width, at most two panels per row,
legends outside the data area, named months / full years on time axes.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from sklearn.metrics import confusion_matrix

import common as cm

ERA_COLORS = {"Before": "#999999", "During": cm.C["orange"], "After": cm.C["blue"]}
TGT_COLORS = {11: cm.C["blue"], 12: cm.C["red"], 15: cm.C["green"]}
MODEL_ORDER = ["Persistence", "Logistic regression", "Random forest", "Gradient boosting"]
MARKERS = {"Persistence": "s", "Logistic regression": "o", "Random forest": "^", "Gradient boosting": "D"}


# ----------------------------------------------------------------- bridge
def fig_bridge() -> None:
    drop = pd.read_csv(cm.RES_DIR / "bridge_reach_drop.csv")
    curves = pd.read_csv(cm.RES_DIR / "bridge_warning_curves.csv")
    x50 = pd.read_csv(cm.RES_DIR / "bridge_warning_x50.csv")
    danger = cm.load_registry()["Danger_Level_mMSL"]
    fig = plt.figure(figsize=(cm.FULL_W, 15.5 * cm.CM))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.55, wspace=0.28)
    reach_names = {"11-12": "Baruria to\nBhagyakul (51 km)", "12-13": "Bhagyakul to\nMawa (5 km)",
                   "13-15": "Mawa to\nSureswar (31 km)"}
    for col, season in enumerate(["Monsoon (Jul-Oct)", "Dry (Jan-Apr)"]):
        ax = fig.add_subplot(gs[0, col])
        d = drop[(drop.season == season) & (drop.flow == "High" if season.startswith("M") else drop.flow == "All")]
        for i, era in enumerate(["Before", "During", "After"]):
            e = d[d.era == era].set_index("reach").loc[list(reach_names)]
            xs = np.arange(3) + (i - 1) * 0.26
            ax.bar(xs, e.mean_m, width=0.25, color=ERA_COLORS[era], label=era)
            ax.errorbar(xs, e.mean_m, yerr=[e.mean_m - e.lo, e.hi - e.mean_m], fmt="none",
                        ecolor=cm.C["ink"], lw=0.7, capsize=2)
        ax.set_xticks(range(3), list(reach_names.values()), fontsize=7.5)
        ax.set_ylabel("Mean water-surface drop (m)")
        flow = "high upstream flow" if season.startswith("M") else "all days"
        ax.set_title(f"({'ab'[col]}) {season.split(' ')[0]}, {flow}")
    h, lab = fig.axes[0].get_legend_handles_labels()
    fig.legend(h, [f"{x} bridge" if x != "During" else "During construction" for x in lab],
               loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.01))
    for col, tgt in enumerate((12, 13)):
        ax = fig.add_subplot(gs[1, col])
        for era in ["Before", "During", "After"]:
            c = curves[(curves.target == tgt) & (curves.era == era)]
            ax.plot(c.goalundo_margin, 100 * c.p, color=ERA_COLORS[era], lw=1.8)
            r = x50[(x50.target == tgt) & (x50.era == era)].iloc[0]
            ax.plot(r.x50_m, 50, "o", color=ERA_COLORS[era], ms=5)
            off = {"Before": (-38, 6), "During": (-40, -14), "After": (6, -16)}[era]
            ax.annotate(f"{r.x50_m:+.2f} m", (r.x50_m, 50), xytext=off,
                        textcoords="offset points", fontsize=7.5, color=ERA_COLORS[era])
        ax.axhline(50, color=cm.C["grey"], lw=0.6, ls=":")
        ax.axvline(0, color=cm.C["red"], lw=0.8, ls="--")
        ax.set_xlim(-2.5, 1.5)
        ax.set_xlabel("Goalundo stage minus its danger level (m)")
        ax.set_ylabel(f"P({cm.SHORT_NAME[tgt]} in warning) (%)")
        ax.set_title(f"({'cd'[col]}) {cm.SHORT_NAME[tgt]} ({tgt}) in warning, by era")
    cm.save(fig, "fig_i_bridge")


# ----------------------------------------------------------------- early warning
def fig_risk_skill() -> None:
    sc = pd.read_csv(cm.RES_DIR / "risk_scores.csv")
    reg = pd.read_csv(cm.RES_DIR / "leadtime_regression.csv")
    fig, axes = plt.subplots(3, 2, figsize=(cm.FULL_W, 17 * cm.CM), sharex=True)
    for row, tgt in enumerate(cm.TARGETS):
        ax = axes[row, 0]
        for m in MODEL_ORDER:
            s = sc[(sc.target == tgt) & (sc.model == m)].sort_values("h")
            ax.plot(s.h, s.CSI_W, marker=MARKERS[m], ms=4, lw=1.4,
                    color=cm.MODEL_COLORS[m], label=m)
        ax.set_ylim(0, 1)
        ax.set_ylabel("CSI, warning-or-worse")
        ax.set_title(f"({'ace'[row]}) {cm.TARGETS[tgt]} ({tgt}): will it be in warning?")
        ax2 = axes[row, 1]
        r = reg[reg.target == tgt].sort_values("h")
        ax2.plot(r.h, r.rmse_persist_cm, marker="s", ms=4, color=cm.C["grey"], label="Persistence")
        ax2.plot(r.h, r.rmse_ridge_cm, marker="o", ms=4, color=cm.C["blue"], label="Ridge")
        ax2.plot(r.h, r.rmse_hgb_cm, marker="D", ms=4, color=cm.C["orange"], label="Gradient boosting")
        ax2.set_ylabel("Stage error, RMSE (cm)")
        ax2.set_title(f"({'bdf'[row]}) {cm.TARGETS[tgt]}: how wrong is the stage forecast?")
    for ax in axes[-1]:
        ax.set_xlabel("Lead time (days ahead)")
        ax.set_xticks([1, 2, 3, 5, 7])
    h1, l1 = axes[0, 0].get_legend_handles_labels()
    fig.legend(h1, l1, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout()
    cm.save(fig, "fig_i_risk_skill")


def fig_confusion(model: str = "Random forest", h: int = 3) -> None:
    pred = pd.read_csv(cm.RES_DIR / "risk_predictions.csv")
    fig, axes = plt.subplots(1, 3, figsize=(cm.FULL_W, 6.6 * cm.CM))
    for ax, tgt, tag in zip(axes, cm.TARGETS, "abc"):
        g = pred[(pred.target == tgt) & (pred.h == h)]
        mat = confusion_matrix(g.y.astype(int), g[model].astype(int), labels=[0, 1, 2, 3])
        rown = mat / np.maximum(mat.sum(1, keepdims=True), 1)
        ax.imshow(rown, cmap="Blues", vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                if mat.sum(1)[i] == 0:
                    continue
                ax.text(j, i, f"{100 * rown[i, j]:.0f}%\n({mat[i, j]})", ha="center", va="center",
                        fontsize=6.5, color="white" if rown[i, j] > 0.55 else cm.C["ink"])
        short = ["Nor.", "Warn.", "Dang.", "Sev."]
        ax.set_xticks(range(4), short, fontsize=7.5)
        ax.set_yticks(range(4), short if tag == "a" else [""] * 4, fontsize=7.5)
        ax.set_xlabel("Predicted")
        if tag == "a":
            ax.set_ylabel("Observed")
        ax.set_title(f"({tag}) {cm.TARGETS[tgt]}", fontsize=9)
        ax.grid(False)
    fig.tight_layout()
    cm.save(fig, "fig_i_confusion")


def fig_importance() -> None:
    imp = pd.read_csv(cm.RES_DIR / "importance_groups.csv")
    order = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "15", "16", "17", "season"]
    fig, ax = plt.subplots(figsize=(cm.FULL_W, 7.5 * cm.CM))
    x = np.arange(len(order))
    for i, tgt in enumerate(cm.TARGETS):
        s = imp[imp.target == tgt].set_index("group").reindex(order)
        ax.bar(x + (i - 1) * 0.27, 100 * s.drop_mean, width=0.26, color=TGT_COLORS[tgt],
               yerr=100 * s.drop_sd, error_kw=dict(lw=0.6, capsize=1.2),
               label=f"{cm.TARGETS[tgt]} (F1 with all gauges {s.base_f1.dropna().iloc[0]:.2f})")
    names = [f"{cm.SHORT_NAME[int(g)]} ({g})" if g != "season" else "season" for g in order]
    ax.set_xticks(x, names, rotation=40, ha="right", fontsize=7.5)
    ax.axhline(0, color=cm.C["ink"], lw=0.6)
    ax.set_ylabel("F1 lost when the gauge\nfeed is scrambled (points)")
    ax.legend(loc="upper left", fontsize=7.5)
    ax.set_title("Warning-or-worse 3 days ahead, random forest, test 2022-2025 (bars below 0: the gauge adds noise)",
                 fontsize=8.5)
    fig.tight_layout()
    cm.save(fig, "fig_i_importance")


# ----------------------------------------------------------------- clustering
def fig_cluster_stations() -> None:
    st = pd.read_pickle(cm.RES_DIR / "cluster_stations.pkl")
    lab, pcs = st["labels"], st["pcs"]
    palette = [cm.C["red"], cm.C["orange"], cm.C["blue"], cm.C["green"], cm.C["pink"]]
    names = {0: "Upper Ganges", 1: "Tidal Padma / Meghna", 2: "Confluence & Jamuna",
             3: "Strongly tidal (estuary)"}
    fig = plt.figure(figsize=(cm.FULL_W, 16 * cm.CM))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.8, 1], hspace=0.45)
    ax = fig.add_subplot(gs[0])
    labels = [f"{cm.SHORT_NAME[i]} ({i})" for i in lab.index]
    dendrogram(st["ward"], labels=labels, ax=ax, color_threshold=0, above_threshold_color=cm.C["ink"],
               leaf_rotation=40, leaf_font_size=7.5)
    for t in ax.get_xticklabels():
        sid = int(t.get_text().split("(")[-1].rstrip(")"))
        t.set_color(palette[lab[sid]])
        t.set_ha("right")
    ax.set_ylabel("Ward distance")
    ax.grid(False)
    ax.set_title("(a) Ward hierarchy of the 17 gauges (label colour = k-means cluster)")
    ax2 = fig.add_subplot(gs[1])
    for c in sorted(lab.unique()):
        p = pcs[lab == c]
        ax2.scatter(p.PC1, p.PC2, s=45, color=palette[c], ec="white", lw=0.6, label=names.get(c, c), zorder=3)
    offs = {11: (-38, 4), 10: (5, -9), 13: (5, -9), 14: (-22, 7), 16: (5, -10), 3: (-32, -9), 5: (4, 5), 4: (5, 4)}
    for sid, r in pcs.iterrows():
        ax2.annotate(cm.SHORT_NAME[sid], (r.PC1, r.PC2), xytext=offs.get(sid, (5, 3)),
                     textcoords="offset points", fontsize=7)
    ev = st["evr"]
    ax2.set_xlabel(f"PC1 ({100 * ev[0]:.0f}% of variance): large, fast swings (right) vs. tidal (left)")
    ax2.set_ylabel(f"PC2 ({100 * ev[1]:.0f}%): slow, flood-prone (up)")
    ax2.set_title(f"(b) k-means, k = {st['k']} (silhouette {st['k_table'].set_index('k').silhouette[st['k']]:.2f}) "
                  "on 8 behaviour features, shown in PCA space")
    ax2.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=7.5)
    cm.save(fig, "fig_i_cluster_stations")


def fig_flood_years() -> None:
    yrs = pd.read_csv(cm.RES_DIR / "cluster_years.csv", index_col=0)
    stage = cm.stage_matrix(cm.load_long())
    danger = cm.load_registry()["Danger_Level_mMSL"]
    colors = {"Severe compound flood": cm.C["red"], "Moderate flood": cm.C["orange"],
              "Mild / non-coincident": cm.C["sky"]}
    fig, axes = plt.subplots(2, 1, figsize=(cm.FULL_W, 14 * cm.CM), sharex=True)
    for ax, sid, tag in zip(axes, (10, 12), "ab"):
        for year, r in yrs.iterrows():
            s = stage.loc[f"{year}-05-15":f"{year}-11-15", sid] - danger[sid]
            doy = s.index.dayofyear
            lw = 2.0 if year in (2020, 2024) else 0.9
            ax.plot(doy, s.to_numpy(), color=colors[r["type"]], lw=lw, alpha=0.9)
            if year in (2020, 2024, 2017):
                i = int(np.nanargmax(s.to_numpy()))
                ax.annotate(str(year), (doy[i], s.iloc[i]), xytext=(4, 3), textcoords="offset points",
                            fontsize=7.5, color=colors[r["type"]])
        ax.axhline(0, color=cm.C["red"], ls="--", lw=1)
        ax.axhline(-1, color=cm.C["grey"], ls=":", lw=0.8)
        ax.set_ylabel("Stage minus danger level (m)")
        ax.set_title(f"({tag}) {cm.SHORT_NAME[sid]} ({sid}), 15 monsoons coloured by flood-year cluster")
    ticks = pd.to_datetime([f"2001-{m:02d}-01" for m in range(6, 12)]).dayofyear
    axes[1].set_xticks(ticks, ["1 Jun", "1 Jul", "1 Aug", "1 Sep", "1 Oct", "1 Nov"])
    axes[1].set_xlabel("Date within the year")
    handles = [plt.Line2D([], [], color=c, lw=2,
                          label=f"{t}: {', '.join(str(y) for y in yrs.index[yrs['type'] == t])}")
               for t, c in colors.items()]
    handles += [plt.Line2D([], [], color=cm.C["red"], ls="--", label="danger level"),
                plt.Line2D([], [], color=cm.C["grey"], ls=":", label="warning (danger - 1 m)")]
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.legend(handles=handles, loc="upper center", ncol=1, bbox_to_anchor=(0.5, 1.0), fontsize=8)
    cm.save(fig, "fig_i_flood_years")


def fig_regimes() -> None:
    lab = pd.read_csv(cm.RES_DIR / "regime_labels.csv", index_col=0, parse_dates=True)["regime"]
    reg = pd.read_csv(cm.RES_DIR / "regimes.csv", index_col=0)
    palette = ["#c6dbef", "#9ecae1", cm.C["orange"], cm.C["sky"], cm.C["red"]]
    fig = plt.figure(figsize=(cm.FULL_W, 14.5 * cm.CM))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.9], hspace=0.55, wspace=0.35)
    ax = fig.add_subplot(gs[0, :])
    share = pd.crosstab(lab.index.month, lab, normalize="index")
    bottom = np.zeros(12)
    for r in share.columns:
        ax.bar(range(1, 13), 100 * share[r], bottom=bottom, color=palette[r], ec="white", lw=0.5,
               label=f"R{r + 1}: {reg.at[r, 'name']}")
        bottom += 100 * share[r].to_numpy()
    ax.set_xticks(range(1, 13), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_ylabel("Share of days (%)")
    ax.set_xlim(0.4, 12.6)
    ax.set_title("(a) Hydraulic regime calendar of the bridge reach, 2011-2025 (k-means, 7 gauges)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=7.5)
    ax2 = fig.add_subplot(gs[1, 0])
    y = np.arange(len(reg))
    ax2.barh(y, 100 * reg.p_warning_3d, color=[palette[r] for r in reg.index], ec=cm.C["ink"], lw=0.4)
    for yi, v in zip(y, reg.p_warning_3d):
        ax2.text(100 * v + 1, yi, f"{100 * v:.0f}%", va="center", fontsize=7.5)
    ax2.set_yticks(y, [f"R{r + 1}" for r in reg.index])
    ax2.invert_yaxis()
    ax2.set_xlim(0, 55)
    ax2.set_xlabel("P(Bhagyakul or Mawa in warning\nwithin the next 3 days) (%)")
    ax2.set_title("(b) Warning risk by regime")
    ax3 = fig.add_subplot(gs[1, 1])
    eras = ["Before", "During", "After"]
    for i, r in enumerate(reg.index):
        ax3.plot(range(3), [100 * reg.at[r, f"share_{e}"] for e in eras], marker="o", ms=4,
                 color=palette[r] if r > 1 else cm.C["grey"], lw=1.4, ls="-" if r > 1 else ":")
        nudge = {1: 1.6, 3: 0.0, 4: -1.6}.get(r, 0.0)
        ax3.text(2.08, 100 * reg.at[r, "share_After"] + nudge, f"R{r + 1}", va="center", fontsize=7.5)
    ax3.set_xticks(range(3), ["Before", "During", "After"])
    ax3.set_xlim(-0.2, 2.4)
    ax3.set_ylabel("Share of days (%)")
    ax3.set_xlabel("Bridge era")
    ax3.set_title("(c) Regime share by bridge era")
    cm.save(fig, "fig_i_regimes")


# ----------------------------------------------------------------- sufficiency
def fig_sufficiency() -> None:
    g = pd.read_csv(cm.RES_DIR / "sufficiency_grid.csv")
    w = pd.read_csv(cm.RES_DIR / "sufficiency_windows.csv")
    fig, axes = plt.subplots(2, 2, figsize=(cm.FULL_W, 14 * cm.CM))
    show = {"0": ("own lags only", cm.C["grey"], ":"), "1": ("+ nearest 1 gauge", cm.C["sky"], "-"),
            "3": ("+ nearest 3 gauges", cm.C["blue"], "-"),
            "all+B": ("+ all upstream + Meghna", cm.C["red"], "-")}
    for ax, tgt, tag in zip(axes.flat[:3], cm.TARGETS, "abc"):
        s = g[(g.target == tgt) & (g.same_day) & (g.model == "Ridge")]
        for n, (name, col, ls) in show.items():
            r = s[s.n_points == n].sort_values("history_years")
            ax.plot(r.history_years, r.PI, marker="o", ms=3.5, color=col, ls=ls, lw=1.5, label=name)
        hb = g[(g.target == tgt) & (g.same_day) & (g.model == "Gradient boosting") & (g.n_points == "all+B")]
        hb = hb.sort_values("history_years")
        ax.plot(hb.history_years, hb.PI, marker="D", ms=3, color=cm.C["orange"], lw=1.1, ls="--",
                label="all + Meghna, gradient boosting")
        ax.axhline(0, color=cm.C["ink"], lw=0.6)
        ax.set_xscale("log")
        ax.set_xticks([0.5, 1, 2, 3, 5, 8, 11], ["0.5", "1", "2", "3", "5", "8", "11"])
        ax.set_ylim(-0.3, 1.0)
        ax.set_xlabel("Training history (years before 2022)")
        ax.set_ylabel("Test PI (2023-2025)")
        ax.set_title(f"({tag}) {cm.TARGETS[tgt]}")
    ax = axes[1, 1]
    for tgt in cm.TARGETS:
        q = w[w.target == tgt].groupby("window_days").PI.quantile([0.05, 0.5, 0.95]).unstack()
        ax.fill_between(q.index, q[0.05], q[0.95], color=TGT_COLORS[tgt], alpha=0.15, lw=0)
        ax.plot(q.index, q[0.5], color=TGT_COLORS[tgt], marker="o", ms=3, lw=1.4, label=cm.TARGETS[tgt])
    ax.set_xscale("log")
    ax.set_xticks([7, 14, 30, 90, 180, 365, 730], ["1 wk", "2 wk", "1 mo", "3 mo", "6 mo", "1 yr", "2 yr"])
    ax.set_ylim(-0.3, 1.0)
    ax.axhline(0, color=cm.C["ink"], lw=0.6)
    ax.set_xlabel("Length of the test window")
    ax.set_ylabel("PI of one fixed model")
    ax.set_title("(d) Short tests mislead: median and 5-95% range")
    ax.legend(loc="lower right", fontsize=7.5)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06), fontsize=8)
    fig.tight_layout()
    cm.save(fig, "fig_i_sufficiency")


def main() -> None:
    cm.use_style()
    fig_bridge()
    fig_cluster_stations()
    fig_flood_years()
    fig_regimes()
    fig_sufficiency()
    if (cm.RES_DIR / "risk_scores.csv").exists():
        fig_risk_skill()
        fig_confusion()
        fig_importance()


if __name__ == "__main__":
    main()
