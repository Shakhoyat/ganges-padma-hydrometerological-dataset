"""Figures and tables for study A (flood hazard) -- full-width, <= 2 panels per row."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy import stats

import common as cm

ORDER = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17]
MONTH_TICKS = pd.to_datetime([f"2001-{m:02d}-01" for m in range(1, 13)]).dayofyear
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _label(sid: int) -> str:
    return f"{cm.SHORT_NAME[sid]} ({sid})"


def fig_flood_days() -> None:
    days = pd.read_csv(cm.RES_DIR / "flood_days.csv", index_col=0)
    days.columns = days.columns.astype(int)
    mat = days[ORDER].T
    bounds = [0, 1, 5, 10, 20, 40, 100]
    cmap = ListedColormap(["#f7f7f7", "#fee8c8", "#fdbb84", "#fc8d59", "#e34a33", "#990000"])
    fig, ax = plt.subplots(figsize=(cm.LAND_W, 12.5 * cm.CM))
    im = ax.imshow(mat.to_numpy(), aspect="auto", cmap=cmap, norm=BoundaryNorm(bounds, cmap.N))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.iat[i, j]
            if v > 0:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=8.5,
                        color="white" if v >= 20 else cm.C["ink"])
    ax.set_xticks(range(mat.shape[1]), mat.columns, rotation=0)
    ax.set_yticks(range(mat.shape[0]), [_label(s) for s in mat.index])
    ax.set_xlabel("Year")
    ax.grid(False)
    # construction began in the last month of 2014: mark the 2014|2015 boundary; the opening
    # (mid-2022) is marked above its column so no line crosses the cell numbers
    ax.axvline(3.5, color=cm.C["ink"], lw=1.0, ls="--")
    ax.text(3.5, -0.8, "construction starts\n26 Nov 2014", ha="center", va="bottom", fontsize=8)
    ax.plot(11, -0.75, marker="v", color=cm.C["blue"], ms=7, clip_on=False)
    ax.text(11, -0.95, "bridge opens\n25 Jun 2022", ha="center", va="bottom", fontsize=8, color=cm.C["blue"])
    bi = mat.index.get_loc(12)
    ax.add_patch(plt.Rectangle((-0.5, bi - 0.5), mat.shape[1], 2, fill=False, ec=cm.C["blue"], lw=1.4))
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, ticks=bounds)
    cb.set_label("Days at or above danger level")
    cm.save(fig, "fig_i_flood_days")


def _gringorten(x: np.ndarray) -> np.ndarray:
    n = len(x)
    rank = stats.rankdata(-x)
    p = (rank - 0.44) / (n + 0.12)
    return 1 / p


def fig_return_levels() -> None:
    fits = pd.read_pickle(cm.RES_DIR / "gumbel_fits.pkl")
    danger = cm.load_registry()["Danger_Level_mMSL"]
    sites = [10, 11, 12, 13]
    fig, axes = plt.subplots(2, 2, figsize=(cm.FULL_W, 12.5 * cm.CM), sharex=True)
    t_grid = np.logspace(np.log10(1.05), 2, 80)
    for ax, sid, tag in zip(axes.flat, sites, "abcd"):
        f, amax = fits[sid]["fit"], fits[sid]["amax"]
        yline = f["loc"] - f["scale"] * np.log(-np.log(1 - 1 / t_grid))
        ax.plot(t_grid, yline, color=cm.C["blue"], lw=1.6, label="Gumbel fit")
        tp = np.array(list(f["lo"].keys()))
        ax.fill_between(tp, list(f["lo"].values()), list(f["hi"].values()),
                        color=cm.C["blue"], alpha=0.15, lw=0, label="90% bootstrap band")
        rp = _gringorten(amax.to_numpy())
        ax.scatter(rp, amax.to_numpy(), s=16, color=cm.C["ink"], zorder=3, label="Annual maximum")
        for yr in amax.sort_values().index[-2:]:
            ax.annotate(str(yr), (rp[list(amax.index).index(yr)], amax[yr]), xytext=(4, -9),
                        textcoords="offset points", fontsize=7.5)
        ax.axhline(danger[sid], color=cm.C["red"], ls="--", lw=1.1, label="BWDB danger level")
        ax.set_xscale("log")
        ax.set_xticks([2, 5, 10, 25, 50, 100], ["2", "5", "10", "25", "50", "100"])
        ax.set_title(f"({tag}) {cm.SHORT_NAME[sid]} ({sid})")
        ax.set_ylabel("Annual max. stage (mMSL)")
    for ax in axes[1]:
        ax.set_xlabel("Return period (years)")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    cm.save(fig, "fig_i_return_levels")


def fig_flood_calendar() -> None:
    cal = pd.read_csv(cm.RES_DIR / "flood_calendar.csv")
    sites = [(10, "Goalundo (10): confluence"), (12, "Bhagyakul (12): 5 km above the bridge"),
             (15, "Sureswar (15): 31 km below the bridge")]
    fig, axes = plt.subplots(3, 1, figsize=(cm.FULL_W, 15 * cm.CM), sharex=True)
    for ax, (sid, title), tag in zip(axes, sites, "abc"):
        c = cal[cal.Id == sid].set_index("year")
        for y, r in c.iterrows():
            if np.isnan(r.onset_doy):
                continue
            ax.plot([r.onset_doy, r.end_doy], [y, y], color=cm.C["orange"], lw=4.5,
                    solid_capstyle="butt")
            ax.plot(r.peak_doy, y, marker="D", ms=4.5, color=cm.C["ink"])
        ax.axhspan(2022.5, 2025.5, color=cm.C["green"], alpha=0.08, lw=0)
        ax.set_yticks(range(2011, 2026), [str(y) for y in range(2011, 2026)], fontsize=7)
        ax.set_ylim(2025.7, 2010.3)
        ax.set_title(f"({tag}) {title}")
        ax.set_ylabel("Year")
        med = c[["onset_doy", "end_doy"]].median()
        ax.axvline(med.onset_doy, color=cm.C["grey"], ls=":", lw=1)
        ax.axvline(med.end_doy, color=cm.C["grey"], ls=":", lw=1)
    axes[-1].set_xticks(MONTH_TICKS[4:11], ["1 " + m for m in MONTH_NAMES[4:11]])
    axes[-1].set_xlim(MONTH_TICKS[4] + 14, MONTH_TICKS[10] + 20)
    axes[-1].set_xlabel("Date within the year")
    handles = [plt.Line2D([], [], color=cm.C["orange"], lw=4.5, label="Stage within 1 m of danger (warning period)"),
               plt.Line2D([], [], color=cm.C["ink"], marker="D", ls="", label="Annual peak"),
               plt.Line2D([], [], color=cm.C["grey"], ls=":", label="15-year median onset / end"),
               plt.Rectangle((0, 0), 1, 1, color=cm.C["green"], alpha=0.15, label="After bridge opening")]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout()
    cm.save(fig, "fig_i_flood_calendar")


def fig_compound() -> None:
    comp = pd.read_csv(cm.RES_DIR / "compound_peaks.csv")
    years = pd.read_csv(cm.RES_DIR / "cluster_years.csv", index_col=0)
    comp["type"] = comp.year.map(years["type"])
    colors = {"Severe compound flood": cm.C["red"], "Moderate flood": cm.C["orange"],
              "Mild / non-coincident": cm.C["sky"]}
    fig, ax = plt.subplots(figsize=(cm.FULL_W, 9.5 * cm.CM))
    for t in colors:
        g = comp[comp.type == t]
        ax.scatter(g.gap_days, g.bhagyakul_margin, s=25 + 8 * g.bhagyakul_danger_days,
                   color=colors[t], alpha=0.9, ec="white", lw=0.6, label=t, zorder=3)
    # crowded years inside the one-week band get a label in free space with a leader line
    leader = {2017: (10.5, 0.30), 2012: (10.5, 0.17), 2013: (10.5, 0.06), 2014: (10.5, -0.05),
              2015: (10.5, -0.16)}
    offs = {2018: (7, -2), 2020: (14, -3), 2021: (7, -2), 2022: (7, -8), 2024: (7, 2),
            2025: (7, -6), 2019: (-14, -13), 2011: (7, 4), 2016: (10, 2), 2023: (7, 2)}
    for _, r in comp.iterrows():
        xy = (r.gap_days, r.bhagyakul_margin)
        if r.year in leader:
            ax.annotate(str(r.year), xy, xytext=leader[r.year], textcoords="data", fontsize=7.5,
                        va="center", arrowprops=dict(arrowstyle="-", color=cm.C["grey"], lw=0.5))
        else:
            ax.annotate(str(r.year), xy, xytext=offs.get(r.year, (7, 2)), textcoords="offset points",
                        fontsize=7.5)
    ax.axhline(0, color=cm.C["red"], ls="--", lw=1)
    ax.text(47, -0.03, "Bhagyakul danger level", color=cm.C["red"], fontsize=7.5, va="top")
    ax.axvspan(-2, 7, color=cm.C["grey"], alpha=0.10, lw=0)
    ax.text(2.5, -0.92, "peaks within\n1 week", ha="center", va="bottom", fontsize=7.5, color=cm.C["grey"])
    ax.set_xlim(-2, 82)
    ax.set_ylim(-0.95, 0.65)
    ax.set_xlabel("Days between the Ganges annual peak (Hardinge Br.) and the Jamuna annual peak (Aricha)")
    ax.set_ylabel("Bhagyakul annual peak\nminus danger level (m)")
    ax.legend(title="Flood-year type (Problem 13); marker area = days above danger at Bhagyakul",
              loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, title_fontsize=8)
    fig.tight_layout()
    cm.save(fig, "fig_i_compound")


def fig_trends() -> None:
    stage = cm.stage_matrix(cm.load_long())
    tr = pd.read_csv(cm.RES_DIR / "trends.csv", index_col=0)
    dry = stage[stage.index.month.isin([1, 2, 3, 4, 5])]
    amin = dry.groupby(dry.index.year).min()
    fig = plt.figure(figsize=(cm.FULL_W, 15 * cm.CM))
    ax1 = fig.add_subplot(2, 1, 1)
    pick = {1: cm.C["red"], 4: cm.C["orange"], 10: cm.C["blue"], 15: cm.C["green"]}
    for sid, col in pick.items():
        y = amin[sid] - amin[sid].iloc[:3].mean()
        ax1.plot(y.index, y, marker="o", ms=3.5, color=col, lw=1.3,
                 label=f"{cm.SHORT_NAME[sid]}: {tr.at[sid, 'min_slope_cm']:+.1f} cm/yr (p={tr.at[sid, 'min_p']:.3f})")
        res = stats.theilslopes(y.dropna().to_numpy(), y.dropna().index.to_numpy())
        ax1.plot(y.index, res.intercept + res.slope * y.index, color=col, lw=0.9, ls="--")
    ax1.set_xticks(range(2011, 2026))
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Dry-season (Jan-May) minimum\nrelative to 2011-13 mean (m)")
    ax1.set_title("(a) Dry-season low water, four gauges (dashed: Sen slope)")
    ax1.legend(loc="lower left", ncol=2, fontsize=8)
    ax1.set_ylim(-5.2, 1.1)
    ax2 = fig.add_subplot(2, 1, 2)
    ids = ORDER
    x = np.arange(len(ids))
    for off, col, key, lab in ((-0.2, cm.C["sky"], "min", "Dry-season minimum"),
                               (0.2, cm.C["red"], "max", "Annual maximum")):
        s = tr.loc[ids, f"{key}_slope_cm"]
        err = np.vstack([s - tr.loc[ids, f"{key}_lo"], tr.loc[ids, f"{key}_hi"] - s])
        sig = tr.loc[ids, f"{key}_p"] < 0.05
        ax2.bar(x + off, s, width=0.38, color=[col if q else "white" for q in sig], ec=col, lw=1,
                label=lab)
        ax2.errorbar(x + off, s, yerr=err, fmt="none", ecolor=cm.C["ink"], lw=0.7, capsize=1.5)
    ax2.axhline(0, color=cm.C["ink"], lw=0.7)
    ax2.set_xticks(x, [_label(s) for s in ids], rotation=40, ha="right")
    ax2.set_ylabel("Trend 2011-2025 (cm/year)")
    ax2.set_title("(b) Sen slope with 90% interval, every gauge (upstream to downstream)")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8, title_fontsize=7.5,
               title="filled: Mann-Kendall p < 0.05; hollow: not significant", frameon=True,
               facecolor="white", edgecolor="none")
    ax2.set_ylim(-29, 19)
    fig.tight_layout()
    cm.save(fig, "fig_i_trends")


def hazard_card_table() -> None:
    """One row per gauge: the facts a riverbank resident or planner needs."""
    rl = pd.read_csv(cm.RES_DIR / "return_levels.csv", index_col=0)
    sp = pd.read_csv(cm.RES_DIR / "danger_spells.csv", index_col=0)
    rise = pd.read_csv(cm.RES_DIR / "rise_rates.csv", index_col=0)
    tr = pd.read_csv(cm.RES_DIR / "trends.csv", index_col=0)
    days = pd.read_csv(cm.RES_DIR / "flood_days.csv", index_col=0)
    days.columns = days.columns.astype(int)
    lines = []
    for sid in ORDER:
        r = rl.loc[sid]
        t_dl = "$<$1.1" if r.T_DL < 1.1 else (f"{r.T_DL:.0f}" if r.T_DL < 99 else "$>$99")
        lines.append(
            f"{cm.SHORT_NAME[sid]} ({sid}) & {r.DL:.2f} & {r.record:.2f} ({int(r.record_year)}) & "
            f"{r.RL10:.2f} & {r.RL100:.2f} & {t_dl} & {days[sid].mean():.1f} & "
            f"{int(sp.at[sid, 'longest'])} & {rise.at[sid, 'rise_p99_cm']:.0f} & "
            f"{tr.at[sid, 'min_slope_cm']:+.1f}{'$^*$' if tr.at[sid, 'min_p'] < 0.05 else ''} \\\\ \\hline")
    (cm.GEN_DIR / "tab_hazard_card.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("  wrote tab_hazard_card.tex")


def main() -> None:
    cm.use_style()
    fig_flood_days()
    fig_return_levels()
    fig_flood_calendar()
    fig_compound()
    fig_trends()
    hazard_card_table()


if __name__ == "__main__":
    main()
