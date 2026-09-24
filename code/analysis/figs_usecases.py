"""Figures for the ten further use cases (C3-C7, U4-U8), from study_usecases.py CSVs.

Same rules as figs_ml.py: full text width, at most two panels per row, legends
outside the data area, no text drawn on top of data.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as cm

R = cm.RES_DIR
TGT_COLORS = {11: cm.C["blue"], 12: cm.C["red"], 15: cm.C["green"]}
STEP_COLORS = [cm.C["light"], cm.C["sky"], cm.C["blue"]]
CLUSTER_COLORS = [cm.C["sky"], cm.C["orange"], cm.C["green"], cm.C["pink"], cm.C["red"]]
MONTH_WEEKS = [0, 4.4, 8.4, 12.9, 17.1, 21.6, 25.9, 30.3, 34.7, 39.0, 43.4, 47.7]
MONTHS = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]


def names(ids: list[int]) -> list[str]:
    return [f"{cm.SHORT_NAME[i]} ({i})" for i in ids]


# ---------------------------------------------------------------- C3
def fig_c3() -> None:
    t = pd.read_csv(R / "uc_c3_incremental.csv")
    fig, axes = plt.subplots(1, 2, figsize=(cm.FULL_W, 7.4 * cm.CM), sharey=True,
                             gridspec_kw={"wspace": 0.08})
    steps = t.drop_duplicates("step").sort_values("step").step_name.tolist()
    for ax, model in zip(axes, ["Logistic regression", "Gradient boosting"]):
        for tgt, col in TGT_COLORS.items():
            s = t[(t.target == tgt) & (t.model == model)].sort_values("step")
            ax.plot(s.step, s.macro_f1, "o-", color=col, lw=1.6, ms=4, label=cm.TARGETS[tgt])
            ax.axhline(s.persistence_f1.iloc[0], color=col, lw=0.9, ls=":")
        ax.set_xticks(range(len(steps)), steps, rotation=35, ha="right", fontsize=7.5)
        ax.set_title(f"({'ab'[axes.tolist().index(ax)]}) {model}")
        ax.set_ylim(0.55, 0.87)
    axes[0].set_ylabel("Macro-F1, test 2023-2025")
    h, lab = axes[0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], color=cm.C["grey"], ls=":", lw=0.9))
    lab.append("same class as yesterday (per target)")
    fig.legend(h, lab, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.06))
    cm.save(fig, "fig_uc_c3_incremental")


# ---------------------------------------------------------------- C4
def fig_c4() -> None:
    t = pd.read_csv(R / "uc_c4_rapid.csv")
    g_hgb = t[t.model == "Gradient boosting"]
    steps = ["Own gauge", "+ upstream gauges", "+ upstream, rain, season"]
    gauges = [4, 10, 12]
    fig, axes = plt.subplots(1, 2, figsize=(cm.FULL_W, 6.8 * cm.CM), gridspec_kw={"wspace": 0.3})
    for i, step in enumerate(steps):
        s = g_hgb[g_hgb.step == step].set_index("gauge").loc[gauges]
        xs = np.arange(3) + (i - 1) * 0.26
        axes[0].bar(xs, s.PR_AUC, 0.25, color=STEP_COLORS[i], edgecolor=cm.C["ink"], lw=0.4, label=step)
        axes[1].bar(xs, s.CSI, 0.25, color=STEP_COLORS[i], edgecolor=cm.C["ink"], lw=0.4)
    pers = t[t.model == "Persistence"].set_index("gauge").loc[gauges]
    for j in range(3):
        axes[1].plot([j - 0.42, j + 0.42], [pers.CSI.iloc[j]] * 2, color=cm.C["red"], lw=1.4)
    for ax, lab in zip(axes, ["PR-AUC (area under precision-recall)", "CSI of the alert (p >= 0.5)"]):
        ax.set_xticks(range(3), [f"{cm.SHORT_NAME[g]}\n>= {t[t.gauge == g].threshold_cm.iloc[0]:.0f} cm/d"
                                 for g in gauges], fontsize=7.8)
        ax.set_ylabel(lab)
        ax.set_ylim(0, 1)
    axes[0].set_title("(a) Ranking skill")
    axes[1].set_title("(b) Alert skill (red: persistence)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, [f"Gradient boosting, {x.lower()}" for x in lab], loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.08), fontsize=7.8)
    cm.save(fig, "fig_uc_c4_rapid")


# ---------------------------------------------------------------- C5
def fig_c5() -> None:
    t = pd.read_csv(R / "uc_c5_logo.csv")
    gauges = sorted(t.gauge.unique())
    models = ["Persistence", "Own-gauge model", "Unseen gauge (trained on the other 15)"]
    cols = [cm.C["grey"], cm.C["sky"], cm.C["blue"]]
    fig, ax = plt.subplots(figsize=(cm.FULL_W, 6.6 * cm.CM))
    for i, (m, c) in enumerate(zip(models, cols)):
        s = t[t.model == m].set_index("gauge").loc[gauges]
        ax.bar(np.arange(len(gauges)) + (i - 1) * 0.27, s.CSI, 0.26, color=c, label=m)
    ev = t[t.model == "Persistence"].set_index("gauge").loc[gauges].events
    ax.set_xticks(range(len(gauges)), [f"{cm.SHORT_NAME[g]}\n({int(ev[g])})" for g in gauges],
                  rotation=45, ha="right", fontsize=7.3)
    ax.set_ylabel("CSI, warning within 3 days\n(test 2022-2025)")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.16))
    cm.save(fig, "fig_uc_c5_logo")


# ---------------------------------------------------------------- C6
def fig_c6() -> None:
    g = pd.read_csv(R / "uc_c6_gauges.csv")
    fig, axes = plt.subplots(1, 2, figsize=(cm.FULL_W, 8.2 * cm.CM), sharey=True,
                             gridspec_kw={"wspace": 0.06})
    ids = list(range(1, 18))
    y = np.arange(17)
    for ax, label, title in zip(axes, ["Tidal vs non-tidal", "River (4 classes)"],
                                ["(a) Tidal or non-tidal?", "(b) Which river?"]):
        s = g[g.label == label].set_index("gauge").loc[ids]
        ok = s.gauge_vote.astype(str) == s["true"].astype(str)
        ax.barh(y, 100 * s.window_accuracy, color=np.where(ok, cm.C["blue"], cm.C["red"]), height=0.7)
        ax.axvline(50 if label.startswith("T") else 25, color=cm.C["grey"], ls=":", lw=0.9)
        ax.set_xlim(0, 100)
        ax.set_xlabel("30-day windows classified correctly (%)")
        ax.set_title(title)
    axes[0].set_yticks(y, names(ids), fontsize=7.5)
    axes[0].invert_yaxis()
    h = [plt.Rectangle((0, 0), 1, 1, color=cm.C["blue"]), plt.Rectangle((0, 0), 1, 1, color=cm.C["red"]),
         plt.Line2D([], [], color=cm.C["grey"], ls=":")]
    fig.legend(h, ["majority vote correct", "majority vote wrong", "chance"], loc="upper center",
               ncol=3, bbox_to_anchor=(0.55, 1.05))
    cm.save(fig, "fig_uc_c6_type")


# ---------------------------------------------------------------- C7
def fig_c7() -> None:
    s = pd.read_csv(R / "uc_c7_summary.csv")
    c = pd.read_csv(R / "uc_c7_confusion.csv", index_col=0)
    fig, axes = plt.subplots(1, 2, figsize=(cm.FULL_W, 6.4 * cm.CM), gridspec_kw={"wspace": 0.45,
                                                                                "width_ratios": [1.1, 1]})
    ax = axes[0]
    ax.barh(range(3), 100 * s.balanced_accuracy, color=[cm.C["grey"], cm.C["blue"], cm.C["sky"]], height=0.6)
    for i, v in enumerate(s.balanced_accuracy):
        ax.text(100 * v + 1.5, i, f"{100 * v:.0f}%", va="center", fontsize=8)
    ax.axvline(100 / 3, color=cm.C["red"], ls="--", lw=0.9)
    ax.text(100 / 3 + 1.5, -0.42, "chance (33%)", color=cm.C["red"], fontsize=7.5, va="center")
    ax.set_yticks(range(3), ["Flow and season\nonly", "+ reach water-\nsurface drops", "Reach drops\nonly"],
                  fontsize=7.8)
    ax.set_ylim(2.5, -0.7)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Balanced accuracy, leave-one-year-out (%)")
    ax.set_title("(a) Can a day's era be recognised?")
    m = c[c.features == "+ reach water-surface drops"].drop(columns="features").astype(float)
    pct = 100 * m.div(m.sum(axis=1), axis=0)
    ax = axes[1]
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{pct.iloc[i, j]:.0f}%\n({int(m.iloc[i, j])})", ha="center", va="center",
                    fontsize=7.5, color="white" if pct.iloc[i, j] > 55 else cm.C["ink"])
    ax.set_xticks(range(3), m.columns)
    ax.set_yticks(range(3), m.index)
    ax.set_xlabel("Predicted era")
    ax.set_ylabel("True era")
    ax.grid(False)
    ax.set_title("(b) Confusion, flow + drops")
    cm.save(fig, "fig_uc_c7_era")


# ---------------------------------------------------------------- U4
def fig_u4() -> None:
    lab = pd.read_csv(R / "uc_u4_labels.csv")
    cen = pd.read_csv(R / "uc_u4_centroids.csv", index_col=0)
    k = pd.read_csv(R / "uc_u4_k.csv")
    fig = plt.figure(figsize=(cm.FULL_W, 13.5 * cm.CM))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.25], width_ratios=[0.7, 1.3], hspace=0.5, wspace=0.3)
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(k.k, k.silhouette, "o-", color=cm.C["ink"], ms=4)
    best = k.loc[k.silhouette.idxmax()]
    ax.plot(best.k, best.silhouette, "o", color=cm.C["red"], ms=7)
    ax.set_xlabel("Number of clusters k")
    ax.set_ylabel("Silhouette")
    ax.set_title("(a) Choice of k")
    ax = fig.add_subplot(gs[0, 1])
    shape_names = ["S1: early rise (Jamuna-fed)", "S2: late rise (upper Ganges)"]
    for c in cen.columns:
        ax.plot(cen.index, cen[c], color=CLUSTER_COLORS[int(c)], lw=2, label=shape_names[int(c)])
    ax.set_xticks(MONTH_WEEKS, MONTHS)
    ax.set_ylabel("Stage, scaled 0-1\nwithin the year")
    ax.set_title("(b) Cluster centroids (weekly mean)")
    ax.legend(loc="upper left", fontsize=7.5)
    ax = fig.add_subplot(gs[1, :])
    piv = lab.pivot(index="gauge", columns="year", values="cluster")
    from matplotlib.colors import ListedColormap
    ax.imshow(piv, cmap=ListedColormap(CLUSTER_COLORS[:2]), aspect="auto", vmin=0, vmax=1)
    for (i, j), v in np.ndenumerate(piv.to_numpy()):
        if np.isnan(v):
            ax.text(j, i, "gap", ha="center", va="center", fontsize=6, color=cm.C["grey"])
    ax.set_xticks(range(piv.shape[1]), piv.columns, fontsize=7.5)
    ax.set_yticks(range(piv.shape[0]), names(piv.index.tolist()), fontsize=7.3)
    ax.grid(False)
    ax.set_title("(c) Shape of every gauge-year (blue S1, orange S2; gap: < 90% of days observed)")
    cm.save(fig, "fig_uc_u4_shapes")


# ---------------------------------------------------------------- U5
def fig_u5() -> None:
    f = pd.read_csv(R / "uc_u5_flagged.csv")
    s = pd.read_csv(R / "uc_u5_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(cm.FULL_W, 7.0 * cm.CM), gridspec_kw={"wspace": 0.55,
                                                                                "width_ratios": [1, 1.1]})
    ax = axes[0]
    y = np.arange(len(s))
    ax.barh(y - 0.18, s.known_caught, 0.34, color=cm.C["blue"], label=f"all |dWL| > 1 m ({s.known.iloc[0]})")
    ax.barh(y + 0.18, s.isolated_caught, 0.34, color=cm.C["orange"],
            label=f"not echoed by neighbours ({s.isolated.iloc[0]})")
    ax.set_yticks(y, ["Own change", "+ next-day\nchange", "+ neighbour\nresidual", "Neighbour\nresidual only"],
                  fontsize=7.8)
    ax.invert_yaxis()
    ax.set_xlabel("Reference jumps caught (of 0.1% flagged)")
    ax.set_title("(a) Feature sets, added in turn")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.3), fontsize=7.3)
    ax = axes[1]
    kinds = {"echoed by neighbours": cm.C["grey"], "isolated step": cm.C["red"], "spike and return": cm.C["pink"]}
    for kind, col in kinds.items():
        k = f[f.kind == kind]
        if len(k):
            ax.scatter(k.nb, k.dwl, s=16, color=col, label=f"{kind} ({len(k)})", edgecolor="white", lw=0.3)
    lim = [-0.3, 1.5]
    ax.plot(lim, lim, color=cm.C["ink"], lw=0.6, ls="--")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Neighbours' mean change (m/day)")
    ax.set_ylabel("Gauge's own change (m/day)")
    ax.set_title("(b) The 87 flagged gauge-days")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), fontsize=7.3, ncol=1)
    cm.save(fig, "fig_uc_u5_anomaly")


# ---------------------------------------------------------------- U6
def fig_u6() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(cm.FULL_W, 8.4 * cm.CM), gridspec_kw={"wspace": 0.28})
    lab = pd.read_csv(R / "uc_u6_labels.csv")
    for ax, (key, title) in zip(axes, [("monsoon", "(a) Monsoon, Jun-Oct"), ("dry", "(b) Dry, Jan-Apr")]):
        r = pd.read_csv(R / f"uc_u6_corr_{key}.csv", index_col=0)
        im = ax.imshow(r, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(17), range(1, 18), fontsize=7)
        ax.set_yticks(range(17), range(1, 18), fontsize=7)
        ax.grid(False)
        cl = lab[lab.season.str.lower().str.startswith(key[:3])].set_index("gauge").cluster.loc[range(1, 18)]
        cuts = [i + 0.5 for i in range(16) if cl.iloc[i] != cl.iloc[i + 1]]
        for c in cuts:
            ax.axhline(c, color=cm.C["ink"], lw=1.4)
            ax.axvline(c, color=cm.C["ink"], lw=1.4)
        med = np.median(r.to_numpy()[np.triu_indices(17, 1)])
        ax.set_title(f"{title} (median r {med:.2f})", fontsize=8.5)
        ax.set_xlabel("Gauge Id (Figure 1)")
    axes[0].set_ylabel("Gauge Id")
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cb.set_label("Best-lag correlation of dWL (|lag| <= 3 d)")
    cm.save(fig, "fig_uc_u6_network")


# ---------------------------------------------------------------- U7
def fig_u7() -> None:
    cen = pd.read_csv(R / "uc_u7_centroids.csv", index_col=0)
    resp = pd.read_csv(R / "uc_u7_response.csv")
    fig, axes = plt.subplots(1, 2, figsize=(cm.FULL_W, 7.0 * cm.CM), gridspec_kw={"wspace": 0.35,
                                                                                "width_ratios": [1.3, 1]})
    ax = axes[0]
    x = np.arange(cen.shape[1])
    for i, (t, row) in enumerate(cen.iterrows()):
        ax.bar(x + (i - 0.5) * 0.38, row, 0.36, color=[cm.C["sky"], cm.C["blue"]][i],
               label=f"T{int(t) + 1} ({int(resp.days.iloc[i + 1])} days)")
    ax.set_xticks(x, cen.columns, rotation=40, ha="right", fontsize=7.5)
    ax.set_ylabel("Mean daily rain (mm)")
    ax.set_title("(a) Rain types, monsoon wet days")
    ax.legend(loc="upper left", fontsize=7.5)
    ax = axes[1]
    xs = np.arange(len(resp))
    for j, (g, col) in enumerate([(10, cm.C["blue"]), (12, cm.C["red"])]):
        ax.bar(xs + (j - 0.5) * 0.38, resp[f"g{g}_rise3_cm"], 0.36, color=col, label=cm.SHORT_NAME[g])
    ax.axhline(0, color=cm.C["ink"], lw=0.6)
    ax.set_xticks(xs, ["No rain", "T1 light", "T2 heavy"])
    ax.set_ylabel("Mean stage change over\nthe next 3 days (cm)")
    ax.set_title("(b) River response")
    ax.legend(loc="upper left", fontsize=7.5)
    cm.save(fig, "fig_uc_u7_rain")


# ---------------------------------------------------------------- U8
def fig_u8() -> None:
    ev = pd.read_csv(R / "uc_u8_events.csv")
    summ = pd.read_csv(R / "uc_u8_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(cm.FULL_W, 7.2 * cm.CM), gridspec_kw={"wspace": 0.32,
                                                                                "width_ratios": [1.25, 1]})
    names_ = ["E1: early, from low water", "E2: mid-monsoon, near danger"]
    ax = axes[0]
    for c in sorted(ev.cluster.unique()):
        e = ev[ev.cluster == c]
        ax.scatter(e.start_margin_m, e.total_rise_m, s=9, color=CLUSTER_COLORS[c], alpha=0.6,
                   label=f"{names_[c]} ({len(e)})", edgecolor="none")
    d = ev[ev.reached_danger == 1]
    ax.scatter(d.start_margin_m, d.total_rise_m, s=16, facecolor="none", edgecolor=cm.C["ink"], lw=0.6,
               label=f"reached danger level ({len(d)})")
    ax.set_xlabel("Stage at start of rise minus danger level (m)")
    ax.set_ylabel("Total rise (m)")
    ax.set_title("(a) 1,075 rising limbs, 13 gauges")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, fontsize=7.2)
    ax = axes[1]
    order = list(range(1, 14))
    share = ev.groupby(["gauge", "cluster"]).reached_danger.mean().unstack().reindex(order)
    y = np.arange(len(order))
    for c in share.columns:
        ax.barh(y + (c - 0.5) * 0.4, 100 * share[c], 0.38, color=CLUSTER_COLORS[c], label=names_[c][:2])
    ax.set_yticks(y, names(order), fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Rises that reached danger (%)")
    ax.set_title("(b) By gauge and event type")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, fontsize=7.5)
    cm.save(fig, "fig_uc_u8_events")


def main() -> None:
    cm.use_style()
    for f in (fig_c3, fig_c4, fig_c5, fig_c6, fig_c7, fig_u4, fig_u5, fig_u6, fig_u7, fig_u8):
        f()


if __name__ == "__main__":
    main()
