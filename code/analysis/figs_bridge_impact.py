"""Figures for the Padma Bridge impact problems (study_bridge_impact.py outputs).

Colour code everywhere: grey = before construction, orange = during, blue = after opening.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

import common as cm

R = cm.RES_DIR
ERA_C = {"Before": "#8c8c8c", "During": cm.C["orange"], "After": cm.C["blue"]}
ERA_LABEL = {"Before": "Before (2011-Nov 2014)", "During": "During construction", "After": "After opening (Jun 2022-)"}
TRANSECT_NAME = {"RMG12": "Control: Ganges at Hardinge Br.",
                 "RMP4": "18 km upstream (Dohar)",
                 "RMP3": "Bridge line (Mawa-Shibchar)",
                 "RMP2": "7 km downstream (Zanjira)"}
ORDER = ["RMG12", "RMP4", "RMP3", "RMP2"]


def shade_eras(ax: plt.Axes, y0: float = 0, y1: float = 1) -> None:
    ax.axvspan(2014.9, 2022.48, color=cm.C["orange"], alpha=0.08, lw=0)
    ax.axvspan(2022.48, 2026, color=cm.C["blue"], alpha=0.08, lw=0)


def era_legend(fig: plt.Figure, y: float = 1.03) -> None:
    h = [plt.Line2D([], [], color=c, lw=2.2) for c in ERA_C.values()]
    fig.legend(h, list(ERA_LABEL.values()), loc="upper center", ncol=3, bbox_to_anchor=(0.5, y))


# ---------------------------------------------------------------- P3 specific gauge
def fig_rating() -> None:
    curves = pd.read_csv(R / "bi_rating_curves.csv")
    sg = pd.read_csv(R / "bi_specific_gauge.csv")
    fig, axes = plt.subplots(1, 2, figsize=(cm.FULL_W, 7.6 * cm.CM), gridspec_kw={"wspace": 0.3,
                                                                                "width_ratios": [1, 1.25]})
    ax = axes[0]
    for e in ERA_C:
        c = curves[(curves.station == "SW93.5L") & (curves.era == e)]
        ax.plot(c.q / 1000, c.wl, color=ERA_C[e], lw=2)
    ax.set_xscale("log")
    ax.set_xticks([10, 20, 30, 50, 70, 100], ["10", "20", "30", "50", "70", "100"])
    ax.set_xlabel("Discharge (1000 m$^3$/s)")
    ax.set_ylabel("Stage at Mawa (mMSL)")
    ax.set_title("(a) Mawa rating curve by era")
    ax = axes[1]
    styles = {"Hardinge Br.": (cm.C["grey"], "s"), "Baruria": (cm.C["green"], "^"),
              "Mawa (bridge)": (cm.C["red"], "o")}
    for name, (col, mk) in styles.items():
        s = sg[sg.name == name]
        q = s.q_star.min()
        s = s[s.q_star == q]
        base = s[s.year <= 2014].wl.mean()
        ax.errorbar(s.year, s.wl - base, yerr=[s.wl - s.lo, s.hi - s.wl], fmt=mk + "-", color=col, ms=4,
                    lw=1.2, capsize=1.5, elinewidth=0.6, label=f"{name}, Q = {q / 1000:.0f}k m$^3$/s")
    shade_eras(ax)
    ax.axhline(0, color=cm.C["ink"], lw=0.6)
    ax.set_xlim(2010.5, 2024.5)
    ax.set_xticks(range(2011, 2025, 2))
    ax.set_ylabel("Stage at fixed discharge\nminus 2011-14 mean (m)")
    ax.set_title("(b) Specific-gauge change (90% CI)")
    ax.legend(loc="lower left", fontsize=7.2)
    era_legend(fig, 1.08)
    cm.save(fig, "fig_bi_rating")


# ---------------------------------------------------------------- P4 sediment
def fig_sediment() -> None:
    s = pd.read_csv(R / "bi_sediment_samples.csv", parse_dates=["date"])
    fits = pd.read_csv(R / "bi_sediment_fits.csv").set_index("station")
    fig, axes = plt.subplots(1, 2, figsize=(cm.FULL_W, 7.6 * cm.CM), gridspec_kw={"wspace": 0.32})
    ax = axes[0]
    for e in ERA_C:
        g = s[(s.station == "SW93.5L") & (s.era == e)]
        ax.scatter(g.q / 1000, g.conc_ppm, s=10, color=ERA_C[e], alpha=0.75, edgecolor="none")
    f = fits.loc["SW93.5L"]
    qq = np.geomspace(3, 110, 50)
    ax.plot(qq, 10 ** (f.intercept + f.slope * np.log10(qq * 1000)), color=ERA_C["Before"], lw=1.4, ls="--",
            label="rating fitted before construction")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Discharge on sampling day (1000 m$^3$/s)")
    ax.set_ylabel("Suspended sediment (ppm)")
    ax.set_title("(a) Mawa (bridge site): every sample")
    ax.legend(loc="lower right", fontsize=7.2)
    ax = axes[1]
    m = s[s.date.dt.month.isin([7, 8, 9])]
    yr = m.groupby([m.date.dt.year, "station"]).conc_ppm.median().unstack()
    ax.plot(yr.index, yr["SW91.9L"], "^-", color=cm.C["green"], ms=4, lw=1.3, label="Baruria (51 km upstream)")
    ax.plot(yr.index, yr["SW93.5L"], "o-", color=cm.C["red"], ms=4, lw=1.3, label="Mawa (bridge site)")
    shade_eras(ax)
    ax.set_yscale("log")
    ax.set_xlim(2010.5, 2025.5)
    ax.set_xticks(range(2011, 2026, 2))
    ax.set_ylabel("Jul-Sep median concentration (ppm)")
    ax.set_title("(b) Monsoon sediment, upstream vs bridge")
    ax2 = ax.twinx()
    ratio = yr["SW93.5L"] / yr["SW91.9L"]
    ax2.bar(yr.index, ratio, width=0.5, color=cm.C["red"], alpha=0.18)
    ax2.set_ylabel("Mawa / Baruria (bars)", color=cm.C["red"])
    ax2.set_ylim(0, 40)
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.legend(loc="upper right", fontsize=7.2)
    era_legend(fig, 1.08)
    cm.save(fig, "fig_bi_sediment")


# ---------------------------------------------------------------- P5 cross-sections
def fig_profiles() -> None:
    p = pd.read_csv(R / "bi_xs_profiles.csv")
    refs = pd.read_csv(R / "bi_xs_refs.csv", index_col=0)
    fig, axes = plt.subplots(2, 2, figsize=(cm.LAND_W, 13.0 * cm.CM), gridspec_kw={"hspace": 0.42, "wspace": 0.14})
    for ax, t in zip(axes.flat, ORDER):
        g = p[p.transect == t]
        for e, yrs in (("Before", range(2011, 2015)), ("During", range(2015, 2023)), ("After", range(2023, 2025))):
            prof = g[g.year.isin(yrs)].groupby("x").z.mean()
            ax.plot(prof.index / 1000, prof, color=ERA_C[e], lw=1.5)
        ax.axhline(refs.at[t, "flood_ref"], color=cm.C["red"], lw=0.8, ls="--")
        ax.axhline(refs.at[t, "low_ref"], color=cm.C["sky"], lw=0.8, ls="--")
        ax.set_title(f"({'abcd'[ORDER.index(t)]}) {TRANSECT_NAME[t]}")
        ax.set_xlabel("Distance from left bank (km)")
        ax.set_ylabel("Bed level (mMSL)")
    h = [plt.Line2D([], [], color=c, lw=2) for c in ERA_C.values()]
    h += [plt.Line2D([], [], color=cm.C["red"], ls="--"), plt.Line2D([], [], color=cm.C["sky"], ls="--")]
    fig.legend(h, ["mean profile before (2011-14)", "during (2015-22)", "after (2023-24)",
                   "danger level", "dry-season water level"], loc="upper center", ncol=5,
               bbox_to_anchor=(0.5, 1.02))
    cm.save(fig, "fig_bi_profiles")


def fig_bedchange() -> None:
    p = pd.read_csv(R / "bi_xs_profiles.csv")
    fig, axes = plt.subplots(1, 3, figsize=(cm.LAND_W, 10.0 * cm.CM), sharey=True, gridspec_kw={"wspace": 0.06})
    norm = TwoSlopeNorm(0, -8, 8)
    for ax, t in zip(axes, ["RMP4", "RMP3", "RMP2"]):
        g = p[p.transect == t].pivot(index="year", columns="x", values="dz")
        im = ax.pcolormesh(g.columns / 1000, g.index, g.to_numpy(), cmap="BrBG_r", norm=norm, shading="nearest")
        ax.axhline(2014.5, color=cm.C["ink"], lw=0.8, ls="--")
        ax.axhline(2022.5, color=cm.C["ink"], lw=0.8)
        ax.set_title({"RMP4": "(a) 18 km upstream", "RMP3": "(b) Bridge line",
                      "RMP2": "(c) 7 km downstream"}[t])
        ax.grid(False)
    axes[1].set_xlabel("Distance from left bank (km)")
    axes[0].set_ylabel("Survey year")
    axes[0].invert_yaxis()
    axes[0].set_yticks(range(2011, 2025, 2))
    cb = fig.colorbar(im, ax=axes, pad=0.02, shrink=0.9)
    cb.set_label("Bed change vs 2011-14 (m)\nbrown = deposition, green = erosion")
    cm.save(fig, "fig_bi_bedchange")


# ---------------------------------------------------------------- P6 clustering
def fig_xs_clusters() -> None:
    c = pd.read_csv(R / "bi_xs_clusters.csv")
    k = pd.read_csv(R / "bi_xs_clusters_k.csv")
    names = ["C1 deep scour channel", "C2 wide and shallow (sandbars)", "C3 deep, uniform"]
    cols = [cm.C["red"], cm.C["orange"], cm.C["blue"]]
    fig = plt.figure(figsize=(cm.FULL_W, 11.5 * cm.CM))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.95], width_ratios=[1.35, 0.65], hspace=0.6, wspace=0.3)
    ax = fig.add_subplot(gs[0, 0])
    depths = np.arange(0, 21)
    for cl in range(3):
        h = c[c.cluster == cl][[f"d{d}" for d in depths]]
        ax.plot(depths, 100 * h.mean(), color=cols[cl], lw=2, label=f"{names[cl]} ({len(h)})")
        ax.fill_between(depths, 100 * h.min(), 100 * h.max(), color=cols[cl], alpha=0.12, lw=0)
    ax.set_xlabel("Depth below danger level (m)")
    ax.set_ylabel("Share of channel width\nat least this deep (%)")
    ax.set_title("(a) Channel-shape clusters (hypsometry)")
    ax.legend(loc="upper right", fontsize=7.2)
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(k.k, k.silhouette, "o-", color=cm.C["ink"], ms=4)
    ax.plot(3, k.set_index("k").silhouette[3], "o", color=cm.C["red"], ms=7)
    ax.set_xlabel("k")
    ax.set_ylabel("Silhouette")
    ax.set_title("(b) Choice of k")
    ax = fig.add_subplot(gs[1, :])
    piv = c.pivot(index="transect", columns="year", values="cluster").reindex(ORDER)
    from matplotlib.colors import ListedColormap
    ax.imshow(piv, cmap=ListedColormap(cols), vmin=0, vmax=2, aspect="auto")
    for (i, j), v in np.ndenumerate(piv.to_numpy()):
        ax.text(j, i, "-" if np.isnan(v) else f"C{int(v) + 1}", ha="center", va="center", fontsize=7,
                color="white" if not np.isnan(v) else cm.C["grey"])
    ax.set_xticks(range(piv.shape[1]), piv.columns, fontsize=7.5)
    ax.set_yticks(range(4), ["Hardinge Br. (control)", "18 km upstream", "Bridge line", "7 km downstream"],
                  fontsize=7.5)
    ax.axvline(3.5, color=cm.C["ink"], lw=1, ls="--")
    ax.axvline(11.5, color=cm.C["ink"], lw=1.2)
    ax.grid(False)
    ax.set_title("(c) Cluster of every survey (dashed: construction starts; solid: bridge opens)")
    cm.save(fig, "fig_bi_xs_clusters")


# ---------------------------------------------------------------- P7 slope and tide
def fig_slope_tide() -> None:
    drop = pd.read_csv(R / "bridge_reach_drop.csv")
    tide = pd.read_csv(R / "bi_tide.csv")
    fig = plt.figure(figsize=(cm.LAND_W, 8.8 * cm.CM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.7], wspace=0.3)
    reach = {"11-12": "Baruria-\nBhagyakul\n(51 km)", "12-13": "Bhagyakul-\nMawa\n(5 km)",
             "13-15": "Mawa-\nSureswar\n(31 km)"}
    for col, (season, flow, title) in enumerate([("Monsoon (Jul-Oct)", "High", "(a) Monsoon, high flow"),
                                                 ("Dry (Jan-Apr)", "All", "(b) Dry season")]):
        ax = fig.add_subplot(gs[0, col])
        d = drop[(drop.season == season) & (drop.flow == flow)]
        for i, e in enumerate(ERA_C):
            r = d[d.era == e].set_index("reach").loc[list(reach)]
            xs = np.arange(3) + (i - 1) * 0.26
            ax.bar(xs, r.mean_m, 0.25, color=ERA_C[e])
            ax.errorbar(xs, r.mean_m, yerr=[r.mean_m - r.lo, r.hi - r.mean_m], fmt="none", ecolor=cm.C["ink"],
                        lw=0.7, capsize=2)
        ax.set_xticks(range(3), list(reach.values()), fontsize=7.3)
        ax.set_ylabel("Water-surface drop (m)")
        ax.set_title(title)
    ax = fig.add_subplot(gs[0, 2])
    cols = {12: cm.C["orange"], 13: cm.C["red"], 15: cm.C["green"], 17: cm.C["blue"]}
    for gid, col in cols.items():
        t = tide[tide.gauge == gid]
        ax.plot(t.year, 100 * t.range_m, "o-", color=col, ms=4, lw=1.4, label=f"{cm.SHORT_NAME[gid]} ({gid})")
    shade_eras(ax)
    ax.set_xlim(2010.5, 2025.5)
    ax.set_xticks(range(2011, 2026, 2))
    ax.set_ylabel("Median daily range, Jan-Apr (cm)")
    ax.set_title("(c) Dry-season daily (tidal) range")
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    ax.set_ylim(20, 135)
    era_legend(fig, 1.08)
    cm.save(fig, "fig_bi_slope_tide")


# ---------------------------------------------------------------- P9 warning curves
def fig_warning() -> None:
    curves = pd.read_csv(R / "bridge_warning_curves.csv")
    x50 = pd.read_csv(R / "bridge_warning_x50.csv")
    fig, axes = plt.subplots(1, 2, figsize=(cm.FULL_W, 6.8 * cm.CM), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, tgt in zip(axes, (12, 13)):
        for e in ERA_C:
            c = curves[(curves.target == tgt) & (curves.era == e)]
            ax.plot(c.goalundo_margin, 100 * c.p, color=ERA_C[e], lw=1.8)
            r = x50[(x50.target == tgt) & (x50.era == e)].iloc[0]
            ax.plot(r.x50_m, 50, "o", color=ERA_C[e], ms=5, zorder=4)
        ax.axhline(50, color=cm.C["grey"], lw=0.6, ls=":")
        ax.axvline(0, color=cm.C["red"], lw=0.8, ls="--")
        ax.set_xlim(-2.0, 1.2)
        ax.set_xlabel("Goalundo stage minus its danger level (m)")
        ax.set_title(f"({'ab'[(12, 13).index(tgt)]}) {cm.SHORT_NAME[tgt]} ({tgt})")
        tab = x50[x50.target == tgt].set_index("era").loc[list(ERA_C)]
        txt = "50% point:\n" + "\n".join(f"{e.lower()} {v:+.2f} m" for e, v in tab.x50_m.items())
        ax.text(-1.95, 96, txt, fontsize=7.3, va="top", ha="left",
                bbox=dict(facecolor="white", edgecolor=cm.C["light"], pad=2.5))
    axes[0].set_ylabel("P(gauge within 1 m of\nits danger level) (%)")
    era_legend(fig, 1.1)
    cm.save(fig, "fig_bi_warning")


def main() -> None:
    cm.use_style()
    for f in (fig_rating, fig_sediment, fig_profiles, fig_bedchange, fig_xs_clusters, fig_slope_tide, fig_warning, fig_transfer):
        f()



# ---------------------------------------------------------------- P12 transfer
def fig_transfer() -> None:
    t = pd.read_csv(R / "bridge_transfer.csv")
    eras = ["Before", "During", "After"]
    fig, axes = plt.subplots(1, 3, figsize=(cm.FULL_W, 5.6 * cm.CM), gridspec_kw={"wspace": 0.12})
    for ax, tgt in zip(axes, cm.TARGETS):
        m = t[t.target == tgt].pivot(index="train", columns="test", values="PI").loc[eras, eras]
        im = ax.imshow(m, cmap="RdYlBu", vmin=-0.2, vmax=1)
        for (i, j), v in np.ndenumerate(m.to_numpy()):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    fontweight="bold" if i == j else "normal", color="white" if v < 0.1 or v > 0.85 else cm.C["ink"])
        ax.set_xticks(range(3), eras)
        ax.set_yticks(range(3), eras if tgt == 11 else [""] * 3)
        ax.set_xlabel("Tested on")
        ax.grid(False)
        ax.set_title(f"({'abc'[list(cm.TARGETS).index(tgt)]}) {cm.TARGETS[tgt]}")
    axes[0].set_ylabel("Trained on")
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cb.set_label("Test PI")
    cm.save(fig, "fig_bi_transfer")


if __name__ == "__main__":
    main()
