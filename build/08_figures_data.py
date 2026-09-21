"""Step 08 - figures describing the source data and the preprocessing.

    fig_coverage      availability of every gauge across the study window
    fig_gaps          gap lengths, and the evidence they are calendar months
    fig_hydrographs   the four target gauges, with the three bridge periods
    fig_profile       stage and danger level along the corridor
    fig_distributions before and after preprocessing, per target
    fig_acf           autocorrelation of stage and of its daily change
    fig_crosscorr     station-by-station correlation
    fig_dailyavg      the delivered daily average against the 3-hourly readings
    fig_classbalance  risk-class composition per target and period
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
import corridor as C  # noqa: E402
import figstyle as S  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

S.use()


def shade_periods(ax, alpha: float = 1.0, label: bool = False) -> None:
    for name, a, b in C.PERIODS:
        ax.axvspan(a, b, color=S.PERIOD_FILL[name], alpha=alpha, zorder=0, lw=0)
    for _, a, _ in C.PERIODS[1:]:
        ax.axvline(a, color=S.MUTED, lw=0.6, ls=(0, (4, 2)), zorder=1)
    if label:
        yl = ax.get_ylim()
        for name, a, b in C.PERIODS:
            ax.text(a + (b - a) / 2, yl[1], name, ha="center", va="bottom",
                    fontsize=6, color=S.PERIOD_COLOR[name], fontweight="bold")


def fig_coverage(raw, reg) -> None:
    cal = pd.date_range(C.STUDY_START, C.STUDY_END, freq="D")
    M = raw.pivot(index="Date", columns="Station_ID", values="WL_avg").reindex(cal)
    M = M[C.MAIN_STEM]

    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    ax.set_facecolor("white")
    ax.grid(False)
    for i, sid in enumerate(C.MAIN_STEM):
        present = M[sid].notna().to_numpy()
        y = len(C.MAIN_STEM) - 1 - i
        ax.fill_between(cal, y - 0.38, y + 0.38, where=present,
                        color=S.BLUE, alpha=0.75, lw=0, step="mid")
        ax.fill_between(cal, y - 0.38, y + 0.38, where=~present,
                        color=S.RUST, alpha=0.85, lw=0, step="mid")
    for _, a, _ in C.PERIODS[1:]:
        ax.axvline(a, color=S.INK, lw=0.8, ls=(0, (4, 2)), zorder=3)
    ax.set_yticks(range(len(C.MAIN_STEM)))
    ax.set_yticklabels([f"{C.STATION_ID[s]:>2}  {s}"
                        for s in reversed(C.MAIN_STEM)], fontsize=6)
    ax.set_ylim(-3.0, len(C.MAIN_STEM) + 0.55)
    ax.set_xlim(cal[0], cal[-1])
    S.despine(ax, keep=())
    S.title(ax, "Gauge availability, 2011–2025",
            "blue = daily average present · red = absent · dashed = bridge period boundary")
    for name, a, b in C.PERIODS:
        ax.text(a + (b - a) / 2, len(C.MAIN_STEM) - 0.05, name, ha="center",
                va="bottom", fontsize=6.4, color=S.PERIOD_COLOR[name],
                fontweight="bold")
    ax.annotate("id 14 (Tarpasa) is absent for 1 369 consecutive days,\n"
                "2016-11-01 to 2020-07-31 — inside the construction period",
                xy=(pd.Timestamp("2018-09-01"), 2.55),
                xytext=(pd.Timestamp("2011-04-01"), -2.85), fontsize=6,
                color=S.RUST, va="bottom", linespacing=1.4,
                arrowprops=dict(arrowstyle="->", color=S.RUST, lw=0.7,
                                connectionstyle="arc3,rad=0.16"))
    S.save(fig, C.FIGURES / "fig_coverage.pdf")


def fig_gaps(gaps) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6),
                             gridspec_kw={"width_ratios": [1.1, 1.0], "wspace": 0.3})
    ax = axes[0]
    order = gaps.sort_values("Length_Days")
    colors = [S.RUST if w else S.GREY for w in order.Whole_Calendar_Month]
    ax.barh(range(len(order)), order.Length_Days, color=colors, height=0.75)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{r.Station_ID} {r.Start:%Y-%m}" for _, r in order.iterrows()],
                       fontsize=5.2)
    ax.set_xscale("log")
    ax.set_xlabel("gap length (days, log scale)")
    S.despine(ax)
    S.title(ax, "Every gap in the delivery", "red = spans whole calendar months")

    ax = axes[1]
    bins = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    ax.hist(gaps.Length_Days, bins=bins, color=S.BLUE, alpha=0.85, edgecolor="white")
    ax.set_xscale("log")
    ax.set_xlabel("gap length (days)")
    ax.set_ylabel("number of gaps")
    S.despine(ax)
    n_short = int((gaps.Length_Days <= 3).sum())
    S.title(ax, "Gap length distribution",
            f"{n_short} single days, {int(gaps.Whole_Calendar_Month.sum())} whole-month blocks")
    ax.set_ylim(0, 12.2)   # headroom so the note clears the tallest bar
    ax.text(0.97, 0.97, "Interpolation is not admissible\nacross a monsoon month.",
            transform=ax.transAxes, ha="right", va="top", fontsize=6,
            color=S.MUTED, style="italic")
    S.save(fig, C.FIGURES / "fig_gaps.pdf")


def fig_hydrographs(long, reg) -> None:
    fig, axes = plt.subplots(len(C.TARGETS), 1, figsize=(7.2, 6.8), sharex=True,
                             gridspec_kw={"hspace": 0.30})
    for ax, sid in zip(axes, C.TARGETS):
        d = long[long.Station_ID == sid]
        dl = float(reg.at[sid, "Danger_Level_mMSL"])
        shade_periods(ax)
        ax.plot(d.Date, d.WL, color=S.TARGET_COLOR[sid], lw=0.35)
        ax.axhline(dl, color=S.RUST, lw=0.7, ls=(0, (3, 2)))
        ax.text(C.STUDY_END, dl, " danger level", color=S.RUST, fontsize=5.8,
                va="center", ha="left")
        ax.set_ylabel("stage (mMSL)")
        S.despine(ax)
        ax.set_title(f"{C.STATION_ID[sid]}  "
                     f"{C.display_name(reg.at[sid, 'Station_Name'])}  ({sid}) — "
                     f"{C.TARGET_ROLE[sid]}",
                     loc="left", fontsize=6.8, color=S.INK, pad=3)
    for _, a, _ in C.PERIODS[1:]:
        for ax in axes:
            ax.axvline(a, color=S.MUTED, lw=0.6, ls=(0, (4, 2)))
    S.figtitle(fig, "Daily average stage at the four target gauges",
               "shading marks the Padma Bridge periods: "
               "BEFORE · DURING · AFTER")
    axes[-1].set_xlabel("date")
    axes[-1].set_xlim(C.STUDY_START, C.STUDY_END)
    S.save(fig, C.FIGURES / "fig_hydrographs.pdf")


def fig_profile(reg, stats) -> None:
    m = reg[reg.Chainage_km.notna()].sort_values("Chainage_km")
    st = stats[stats.Stage.str.startswith("after")].set_index("Station_ID")

    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    x = m.Chainage_km.to_numpy()
    lo = np.array([st.at[s, "Q1"] for s in m.Station_ID])
    hi = np.array([st.at[s, "Q3"] for s in m.Station_ID])
    mx = np.array([st.at[s, "Max"] for s in m.Station_ID])
    md = np.array([st.at[s, "Median"] for s in m.Station_ID])

    ax.fill_between(x, lo, hi, color=S.BLUE, alpha=0.20, lw=0,
                    label="interquartile range")
    ax.plot(x, mx, color=S.GREY, lw=0.9, ls=(0, (3, 2)), label="maximum observed")
    ax.plot(x, md, color=S.BLUE, lw=1.4, marker="o", ms=3.4, label="median stage")
    ax.plot(x, m.Danger_Level_mMSL, color=S.RUST, lw=1.2, marker="s", ms=3.0,
            label="danger level")
    bridge_km = float(m[m.Station_ID == "SW93.5L"].Chainage_km.iloc[0])
    ax.axvline(bridge_km, color=S.GREEN, lw=1.0)

    # Headroom so the legend clears the tallest curve, and the bridge label
    # rotated along its own line: both were colliding when placed inline.
    ax.set_xlim(-12, 303)
    ax.set_ylim(0, float(mx.max()) * 1.30)
    ax.text(bridge_km - 4, ax.get_ylim()[1] * 0.42, "Padma Bridge", color=S.GREEN,
            fontsize=6.4, fontweight="bold", rotation=90, va="center", ha="right")

    # Corridor ids go on their own axis above the plot rather than beside the
    # markers, where ids 10/11 and 12/13 sat on top of one another.
    top = ax.secondary_xaxis("top")
    top.set_xticks(x)
    # Rotated, because ids 10/11 and 12/13 are only 4-5 km apart and their
    # labels run together when set horizontally.
    top.set_xticklabels([str(int(i)) for i in m.Id], fontsize=5.4, rotation=90)
    top.tick_params(length=2, width=0.5, pad=1.5, colors=S.MUTED)
    top.spines["top"].set_color(S.RULE)

    ax.set_xlabel("chainage along the Ganges–Padma main stem (km from Panka)")
    ax.set_ylabel("stage (mMSL)")
    ax.legend(loc="upper right", ncols=2, framealpha=0.95, frameon=True,
              edgecolor="none", facecolor="white")
    S.despine(ax)
    # Only a title here: the id axis already occupies the band a subtitle would
    # use, so the descriptive line lives in the LaTeX caption instead.
    ax.set_title("Longitudinal profile of the corridor", loc="left",
                 color=S.INK, pad=26)
    ax.text(1.0, 1.105, "corridor id, at true chainage", transform=ax.transAxes,
            fontsize=6.0, color=S.MUTED, va="bottom", ha="right")
    S.save(fig, C.FIGURES / "fig_profile.pdf")


def fig_distributions(stats, bias) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8),
                             gridspec_kw={"width_ratios": [1.35, 1.0], "wspace": 0.28})
    ax = axes[0]
    w, ids = 0.38, np.arange(len(C.MAIN_STEM))
    for k, (stage, col) in enumerate((("before", S.GREY), ("after", S.BLUE))):
        sub = stats[stats.Stage.str.startswith(stage)].set_index("Station_ID")
        med = [sub.at[s, "Median"] for s in C.MAIN_STEM]
        q1 = [sub.at[s, "Q1"] for s in C.MAIN_STEM]
        q3 = [sub.at[s, "Q3"] for s in C.MAIN_STEM]
        off = (k - 0.5) * w
        ax.bar(ids + off, np.array(q3) - np.array(q1), bottom=q1, width=w,
               color=col, alpha=0.75, label=f"{stage} preprocessing", lw=0)
        ax.scatter(ids + off, med, s=6, color="white", zorder=4, lw=0)
    ax.set_xticks(ids)
    ax.set_xticklabels([str(C.STATION_ID[s]) for s in C.MAIN_STEM], fontsize=6)
    ax.set_xlabel("corridor id")
    ax.set_ylabel("stage (mMSL)")
    ax.set_ylim(0, 23.5)          # headroom so the legend clears the id-8 bar
    ax.legend(loc="upper right", framealpha=0.95, frameon=True,
              edgecolor="none", facecolor="white")
    S.despine(ax)
    S.title(ax, "Interquartile range, before and after preprocessing",
            "bars overlap almost exactly — the filtering does not move the distribution")

    # The selection effect that matters is how far the analysis sample's mean
    # moves from the delivered sample's mean. mean(kept) - mean(dropped) is the
    # wrong panel to draw: only a few dozen rows are ever dropped, so it
    # measures which days those were, not a bias in what remains.
    ax = axes[1]
    bef = stats[stats.Stage.str.startswith("before")].set_index("Station_ID")
    aft = stats[stats.Stage.str.startswith("after")].set_index("Station_ID")
    nd = bias.set_index("Station_ID").N_Dropped
    shift = [aft.at[s, "Mean"] - bef.at[s, "Mean"] for s in C.MAIN_STEM]
    y = np.arange(len(C.MAIN_STEM))
    ax.barh(y, shift, color=S.BLUE, height=0.7)
    ax.axvline(0, color=S.INK, lw=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{C.STATION_ID[s]} {s}" for s in C.MAIN_STEM],
                       fontsize=5.4)
    # The dropped-row count belongs outside the panel: printed against the bars
    # it landed on top of them.
    right = ax.secondary_yaxis("right")
    right.set_yticks(y)
    right.set_yticklabels([f"{int(nd.get(s, 0))}" for s in C.MAIN_STEM],
                          fontsize=5.4)
    right.tick_params(length=0, pad=1.5, colors=S.MUTED)
    right.spines["right"].set_visible(False)
    right.set_ylabel("rows dropped", fontsize=6.2, color=S.MUTED, rotation=270,
                     labelpad=9)
    ax.set_xlabel("mean(after) − mean(before), m")
    ax.set_xlim(-0.012, 0.012)
    S.despine(ax)
    S.title(ax, "Complete-case selection effect",
            "every shift is under 1 cm — inside gauge precision")
    S.save(fig, C.FIGURES / "fig_distributions.pdf")


def fig_acf(acf_df) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharex=True,
                            gridspec_kw={"wspace": 0.22})
    for ax, col, ttl, sub in (
        (axes[0], "ACF_Level", "Stage",
         "lag-1 above 0.99 at every gauge — persistence is a strong baseline"),
        (axes[1], "ACF_Daily_Change", "Daily change in stage",
         "still correlated at 0.43–0.81, and negative around lag 7"),
    ):
        for sid in C.MAIN_STEM:
            d = acf_df[acf_df.Station_ID == sid]
            is_t = sid in C.TARGETS
            ax.plot(d.Lag, d[col], color=S.TARGET_COLOR.get(sid, S.GREY),
                    lw=1.2 if is_t else 0.6, alpha=1.0 if is_t else 0.45,
                    zorder=3 if is_t else 2)
        ax.axhline(0, color=S.INK, lw=0.6)
        ax.set_xlabel("lag (days)")
        ax.set_ylabel("autocorrelation")
        S.despine(ax)
        S.title(ax, ttl, sub)
    axes[0].set_ylim(0, 1.02)
    axes[0].legend(handles=[Line2D([], [], color=S.TARGET_COLOR[s], lw=1.2,
                                   label=f"{C.STATION_ID[s]} {s}")
                            for s in C.TARGETS]
                   + [Line2D([], [], color=S.GREY, lw=0.6, label="other gauges")],
                   loc="lower left", ncols=2)

    # The sign change around lag 7 is the flood-wave timescale, and it is the
    # reason a seven-day history is the right window for this river.
    ax = axes[1]
    # Read the trough off the data so the figure and the text cannot disagree.
    trough = int(acf_df.groupby("Lag").ACF_Daily_Change.mean().idxmin())
    lo_lag, hi_lag = trough - 1, trough + 1
    ax.axvspan(lo_lag, hi_lag, color=S.AMBER, alpha=0.13, lw=0, zorder=0)
    ax.annotate(f"most negative at lags {lo_lag}–{hi_lag}:\none rise-and-fall cycle",
                xy=(trough, -0.30), xytext=(hi_lag + 1.3, -0.30), fontsize=5.8,
                color=S.MUTED, va="center", linespacing=1.4,
                arrowprops=dict(arrowstyle="->", color=S.MUTED, lw=0.6))
    S.save(fig, C.FIGURES / "fig_acf.pdf")


def fig_crosscorr(cc) -> None:
    M = cc.set_index(cc.columns[0]).loc[C.MAIN_STEM, C.MAIN_STEM]
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ax.grid(False)
    im = ax.imshow(M.to_numpy(), cmap="RdYlBu_r", vmin=0.75, vmax=1.0)
    lbl = [f"{C.STATION_ID[s]} {s}" for s in C.MAIN_STEM]
    ax.set_xticks(range(17), lbl, rotation=90, fontsize=5.2)
    ax.set_yticks(range(17), lbl, fontsize=5.2)
    for i in range(17):
        for j in range(17):
            v = M.iat[i, j]
            ax.text(j, i, f"{v:.2f}".lstrip("0"), ha="center", va="center",
                    fontsize=3.7, color="white" if v > 0.965 or v < 0.80 else S.INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_linewidth(0.5)
    S.despine(ax, keep=())
    S.title(ax, "Correlation between gauges",
            "the Ganges block and the Padma block are each near-collinear")
    S.save(fig, C.FIGURES / "fig_crosscorr.pdf")


def fig_dailyavg(val, cmp_df) -> None:
    """Two panels that settle what the delivered Daily Avg column is.

    Left  the delivered value against the mean recomputed from that day's
          3-hourly readings, on a 1:1 line.
    Right the distribution of the delivered value minus each candidate
          definition. Against the all-readings mean the error is a spike at
          zero; against the high-low midpoint it is wide.
    """
    if val.empty or cmp_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9),
                             gridspec_kw={"width_ratios": [1.0, 1.15],
                                          "wspace": 0.26})

    ax = axes[0]
    lo = min(cmp_df.Mean_3h.min(), cmp_df.Delivered_Avg.min())
    hi = max(cmp_df.Mean_3h.max(), cmp_df.Delivered_Avg.max())
    ax.plot([lo, hi], [lo, hi], color=S.MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1,
            label="1:1")
    for sid in val.Station_ID:
        d = cmp_df[cmp_df.Station_ID == sid]
        ax.scatter(d.Mean_3h, d.Delivered_Avg, s=2.2,
                   color=S.TARGET_COLOR.get(sid, S.BLUE), alpha=0.35, lw=0,
                   zorder=2, label=f"{C.STATION_ID[sid]} {sid}")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("mean of that day's 3-hourly readings (mMSL)")
    ax.set_ylabel("delivered Daily Avg (mMSL)")
    leg = ax.legend(loc="upper left", markerscale=3.2, handletextpad=0.4)
    for h in leg.legend_handles:
        try:
            h.set_alpha(1.0)
        except Exception:
            pass
    S.despine(ax)
    S.title(ax, "Delivered average vs recomputed average",
            f"{len(cmp_df):,} station-days at two gauges, 2015 onward")

    ax = axes[1]
    d_mean = (cmp_df.Delivered_Avg - cmp_df.Mean_3h).to_numpy()
    d_mid = (cmp_df.Delivered_Avg - cmp_df.Midpoint).dropna().to_numpy()
    bins = np.linspace(-0.25, 0.25, 101)
    ax.hist(d_mid, bins=bins, color=S.AMBER, alpha=0.80, lw=0,
            label="minus (daily max + daily min) / 2")
    ax.hist(d_mean, bins=bins, color=S.TEAL, alpha=0.90, lw=0,
            label="minus mean of the 3-hourly readings")
    ax.axvline(0, color=S.INK, lw=0.7)
    ax.set_xlim(-0.25, 0.25)
    ax.set_xlabel("delivered Daily Avg − candidate definition (m)")
    ax.set_ylabel("station-days")
    ax.legend(loc="upper right")
    S.despine(ax)
    S.title(ax, "Which definition does it match?",
            f"sd {d_mean.std(ddof=1):.3f} m against the all-readings mean, "
            f"{d_mid.std(ddof=1):.3f} m against the midpoint")
    S.save(fig, C.FIGURES / "fig_dailyavg.pdf")


def fig_classbalance(long, reg) -> None:
    fig, axes = plt.subplots(1, len(C.TARGETS), figsize=(7.2, 2.4), sharey=True,
                             gridspec_kw={"wspace": 0.16})
    for ax, sid in zip(axes, C.TARGETS):
        d = long[(long.Station_ID == sid) & long.Window_Complete]
        bottoms = np.zeros(len(C.PERIOD_NAMES))
        for lab in C.RISK_LABELS:
            share = [100 * (d[d.Period == p].Risk_Class == lab).mean()
                     for p in C.PERIOD_NAMES]
            ax.bar(C.PERIOD_NAMES, share, bottom=bottoms, color=S.RISK_COLOR[lab],
                   width=0.66, label=lab, lw=0)
            for i, (s, b) in enumerate(zip(share, bottoms)):
                if s > 4:
                    ax.text(i, b + s / 2, f"{s:.0f}", ha="center", va="center",
                            fontsize=5.6, color="white")
            bottoms += np.array(share)
        ax.set_ylim(0, 100)
        ax.tick_params(axis="x", labelsize=6, rotation=0)
        S.despine(ax)
        ax.grid(False)
        ax.set_title(f"{C.STATION_ID[sid]}  {C.display_name(reg.at[sid, 'Station_Name'])}",
                     loc="left", fontsize=7)
    axes[0].set_ylabel("percent of station-days")
    # Figure-level legend below the panels: inside the axes it sat on top of
    # the Normal bar in the first panel.
    handles = [Patch(facecolor=S.RISK_COLOR[l], label=l) for l in C.RISK_LABELS]
    fig.legend(handles=handles, loc="lower center", ncols=4, fontsize=6.4,
               bbox_to_anchor=(0.5, -0.10), frameon=False)
    fig.suptitle("Flood-risk class composition by bridge period", x=0.125,
                 ha="left", fontsize=8.5, color=S.INK, y=1.04)
    S.save(fig, C.FIGURES / "fig_classbalance.pdf")


def main() -> None:
    raw = pd.read_csv(C.INTERIM / "wl_daily_raw.csv", parse_dates=["Date"])
    long = pd.read_csv(C.DATA / "pointwise_wl_7day_long.csv", parse_dates=["Date"])
    reg = pd.read_csv(C.DATA / "station_registry.csv").set_index("Station_ID")
    stats = pd.read_csv(C.RESULTS / "stats_before_after.csv")
    bias = pd.read_csv(C.RESULTS / "completecase_bias.csv")
    gaps = pd.read_csv(C.RESULTS / "gap_inventory.csv", parse_dates=["Start", "End"])
    acf_df = pd.read_csv(C.RESULTS / "autocorrelation.csv")
    cc = pd.read_csv(C.RESULTS / "cross_correlation.csv")
    val = pd.read_csv(C.RESULTS / "dailyavg_validation.csv")
    cmp_path = C.RESULTS / "dailyavg_comparison.csv"
    cmp_df = (pd.read_csv(cmp_path, parse_dates=["Date"])
              if cmp_path.exists() else pd.DataFrame())

    fig_coverage(raw, reg)
    fig_gaps(gaps)
    fig_hydrographs(long, reg)
    fig_profile(reg.reset_index(), stats)
    fig_distributions(stats, bias)
    fig_acf(acf_df)
    fig_crosscorr(cc)
    fig_dailyavg(val, cmp_df)
    fig_classbalance(long, reg)


if __name__ == "__main__":
    main()
