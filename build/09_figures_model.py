"""Step 09 - figures for the results and analysis section.

    fig_skill       skill above persistence, per target and period (the headline)
    fig_scatter     observed against predicted on the test block
    fig_testseries  the test block as a hydrograph, with persistence alongside
    fig_residuals   residual structure: against fitted, normal quantiles, and ACF
    fig_importance  permutation importance grouped by gauge and by lag
    fig_transfer    train on one bridge period, test on another
    fig_confusion   risk-class confusion matrices
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
import corridor as C  # noqa: E402
import figstyle as S  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

S.use()
PRIMARY = "hydromet"


def label(sid: str, reg) -> str:
    return f"{C.STATION_ID[sid]} {C.display_name(reg.at[sid, 'Station_Name'])}"


def fig_skill(reg_m, clf_m, reg) -> None:
    """One bar per target-and-period, on a clipped axis.

    Grouping by target and colouring by period ran the four station names
    together under the axis and left the value labels sitting on the bars; a
    single compact category per bar avoids both. A few strata sit far enough
    below the rest to flatten everything else, so the axis is clipped and those
    bars are labelled with their true value.
    """
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"wspace": 0.26})
    strata = C.PERIOD_NAMES + ["ALL"]
    cats = [(s, st) for s in C.TARGETS for st in strata]
    x = np.arange(len(cats))
    ticks = [f"{C.STATION_ID[s]}·{st[:3]}" for s, st in cats]

    for ax, frame, col, ylab, ttl, sub, floor in (
        (axes[0], reg_m, "Persistence_Index", "Persistence Index",
         "Regression skill above persistence",
         "0 = no better than repeating yesterday's stage; 1 = perfect", -2.2),
        (axes[1], clf_m, "Kappa_Skill_Score", "κ skill score",
         "Classification skill above persistence",
         "0 = the naive rule's κ; negative means worse than naive", -2.2),
    ):
        d = frame[frame.Feature_Set == PRIMARY].set_index(["Station_ID", "Stratum"])
        v = [float(d.at[(s, st), col]) if (s, st) in d.index else np.nan
             for s, st in cats]
        colors = [S.PERIOD_COLOR.get(st, S.BLUE) for _, st in cats]
        ax.bar(x, v, width=0.68, color=colors, lw=0)
        ax.axhline(0, color=S.INK, lw=0.9)
        top = max(0.15, np.nanmax(v) * 2.6)
        ax.set_ylim(floor, top)
        for xi, vi in zip(x, v):
            if np.isfinite(vi) and vi < floor:
                ax.annotate("", (xi, floor), xytext=(xi, floor + 0.30),
                            arrowprops=dict(arrowstyle="-|>", color=S.RUST,
                                            lw=0.7))
                ax.text(xi, floor + 0.34, f"{vi:.1f}", ha="center", va="bottom",
                        fontsize=5.2, color=S.RUST, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.12", fc="white",
                                  ec="none", alpha=0.92))
        ax.set_xticks(x, ticks, rotation=90, fontsize=5.2)
        ax.set_ylabel(ylab)
        S.despine(ax)
        S.title(ax, ttl, sub)

    axes[0].legend(handles=[Patch(facecolor=S.PERIOD_COLOR.get(st, S.BLUE),
                                  label=st) for st in strata],
                   loc="upper left", ncols=2, fontsize=5.8)
    S.save(fig, C.FIGURES / "fig_skill.pdf")


def fig_scatter(pred, reg_m, reg) -> None:
    fig, axes = plt.subplots(1, len(C.TARGETS), figsize=(7.2, 2.2),
                             gridspec_kw={"wspace": 0.26})
    m = reg_m[(reg_m.Feature_Set == PRIMARY) & (reg_m.Stratum == "ALL")] \
        .set_index("Station_ID")
    for ax, sid in zip(axes, C.TARGETS):
        d = pred[(pred.Station_ID == sid) & (pred.Stratum == "ALL")
                 & (pred.Feature_Set == PRIMARY)]
        lo = min(d.Observed.min(), d.Predicted.min())
        hi = max(d.Observed.max(), d.Predicted.max())
        ax.plot([lo, hi], [lo, hi], color=S.MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
        ax.scatter(d.Observed, d.Predicted, s=3.0, color=S.TARGET_COLOR[sid],
                   alpha=0.45, lw=0, zorder=2)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel("observed (mMSL)")
        ax.set_title(label(sid, reg), loc="left", fontsize=6.8)
        r = m.loc[sid]
        ax.text(0.04, 0.96, f"RMSE {r.RMSE:.3f} m\nNSE {r.NSE:.4f}\nPI {r.Persistence_Index:.3f}",
                transform=ax.transAxes, fontsize=5.8, va="top", color=S.MUTED,
                linespacing=1.5)
        S.despine(ax)
    axes[0].set_ylabel("predicted (mMSL)")
    S.figtitle(fig, "Observed against predicted, test block, all years pooled",
               top=1.06)
    S.save(fig, C.FIGURES / "fig_scatter.pdf")


def fig_testseries(pred, reg) -> None:
    fig, axes = plt.subplots(len(C.TARGETS), 1, figsize=(7.2, 6.0), sharex=False,
                             gridspec_kw={"hspace": 0.42})
    for ax, sid in zip(axes, C.TARGETS):
        d = pred[(pred.Station_ID == sid) & (pred.Stratum == "ALL")
                 & (pred.Feature_Set == PRIMARY)].sort_values("Date")
        dt = pd.to_datetime(d.Date)
        ax.plot(dt, d.Persistence, color=S.GREY, lw=0.8, label="persistence (WLD-1)")
        ax.plot(dt, d.Observed, color=S.INK, lw=0.9, label="observed")
        ax.plot(dt, d.Predicted, color=S.TARGET_COLOR[sid], lw=0.9,
                ls=(0, (3, 1.4)), label="random forest")
        ax.set_ylabel("stage (mMSL)")
        ax.set_title(label(sid, reg), loc="left", fontsize=6.8)
        S.despine(ax)
    axes[-1].set_xlabel("date")
    # Legend below the panels rather than inside them: the three traces lie on
    # top of one another and fill the plotting area.
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncols=3, fontsize=6.6,
               bbox_to_anchor=(0.5, -0.015), frameon=False)
    S.figtitle(fig, "Test block: the forecast against the naive rule",
               "the model and persistence are visually indistinguishable at "
               "this scale — which is the point")
    S.save(fig, C.FIGURES / "fig_testseries.pdf")


def fig_residuals(pred, resid, reg) -> None:
    fig, axes = plt.subplots(3, len(C.TARGETS), figsize=(7.2, 5.4),
                             gridspec_kw={"hspace": 0.55, "wspace": 0.30})
    rd = resid[(resid.Feature_Set == PRIMARY) & (resid.Stratum == "ALL")] \
        .set_index("Station_ID")
    for j, sid in enumerate(C.TARGETS):
        d = pred[(pred.Station_ID == sid) & (pred.Stratum == "ALL")
                 & (pred.Feature_Set == PRIMARY)].sort_values("Date")
        r = (d.Observed - d.Predicted).to_numpy()
        col = S.TARGET_COLOR[sid]

        ax = axes[0, j]
        ax.axhline(0, color=S.MUTED, lw=0.7)
        ax.scatter(d.Predicted, r, s=2.6, color=col, alpha=0.4, lw=0)
        ax.set_xlabel("fitted (mMSL)")
        ax.set_title(label(sid, reg), loc="left", fontsize=6.6)
        if j == 0:
            ax.set_ylabel("residual (m)")
        S.despine(ax)
        lo_, hi_ = ax.get_ylim()          # headroom for the test annotation
        ax.set_ylim(lo_, hi_ + 0.30 * (hi_ - lo_))
        p = rd.at[sid, "BreuschPagan_p"]
        ax.text(0.04, 0.95, f"Breusch–Pagan p = {p:.3g}", transform=ax.transAxes,
                fontsize=5.5, va="top", color=S.MUTED)

        ax = axes[1, j]
        sps.probplot(r, dist="norm", plot=None)
        osm, osr = sps.probplot(r, dist="norm", fit=False)
        ax.scatter(osm, osr, s=2.6, color=col, alpha=0.45, lw=0)
        lim = [osm.min(), osm.max()]
        slope, inter = np.polyfit(osm, osr, 1)
        ax.plot(lim, [slope * v + inter for v in lim], color=S.MUTED, lw=0.7,
                ls=(0, (3, 2)))
        ax.set_xlabel("normal quantile")
        if j == 0:
            ax.set_ylabel("residual (m)")
        S.despine(ax)
        lo_, hi_ = ax.get_ylim()
        ax.set_ylim(lo_, hi_ + 0.30 * (hi_ - lo_))
        ax.text(0.04, 0.95, f"Shapiro p = {rd.at[sid, 'Shapiro_p']:.3g}",
                transform=ax.transAxes, fontsize=5.5, va="top", color=S.MUTED)

        ax = axes[2, j]
        lags = [c for c in resid.columns if c.startswith("Residual_ACF_lag")]
        vals = [rd.at[sid, c] for c in lags]
        ax.bar(range(1, len(vals) + 1), vals, color=col, width=0.62, lw=0)
        ax.axhline(0, color=S.INK, lw=0.7)
        n = len(r)
        for sgn in (1, -1):
            ax.axhline(sgn * 1.96 / np.sqrt(n), color=S.RUST, lw=0.6,
                       ls=(0, (2, 2)))
        ax.set_xlabel("lag (days)")
        if j == 0:
            ax.set_ylabel("residual ACF")
        S.despine(ax)
    S.figtitle(fig, "Residual diagnostics, pooled model, test block",
               "rows: residual against fitted · normal quantile plot · "
               "autocorrelation of residuals (red = 95% band)")
    S.save(fig, C.FIGURES / "fig_residuals.pdf")


def fig_importance(grp, reg) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9),
                             gridspec_kw={"width_ratios": [1.45, 1.0], "wspace": 0.24})
    ax = axes[0]
    g = grp[(grp.Grouping == "by_point") & (grp.Stratum == "ALL")
            & (grp.Feature_Set == PRIMARY)]
    for sid in C.TARGETS:
        d = g[g.Station_ID == sid].sort_values("Group_Value")
        tot = d.Permutation_Importance.sum()
        if tot <= 0:
            continue
        ax.plot(d.Group_Value, 100 * d.Permutation_Importance / tot, marker="o",
                ms=3.2, color=S.TARGET_COLOR[sid], label=label(sid, reg))
    ax.axvline(13, color=S.GREEN, lw=0.9)
    # Headroom: the target's own columns are grouped under its own id, so the
    # curve peaks there and was running off the top of the axes.
    ax.set_ylim(-3, 118)
    ax.text(13.2, 112, "bridge (id 13)", color=S.GREEN, fontsize=6, va="top")
    ax.set_xlabel("source gauge (corridor id)")
    ax.set_ylabel("share of permutation importance (%)")
    ax.set_xticks(range(1, 16))
    ax.legend(loc="upper left", ncols=2, framealpha=0.95, frameon=True,
              edgecolor="none", facecolor="white")
    S.despine(ax)
    S.title(ax, "Where the information comes from",
            "importance summed over the eight columns contributed by each gauge")

    ax = axes[1]
    g = grp[(grp.Grouping == "by_lag") & (grp.Stratum == "ALL")
            & (grp.Feature_Set == PRIMARY)]
    for sid in C.TARGETS:
        d = g[g.Station_ID == sid].sort_values("Group_Value")
        tot = d.Permutation_Importance.sum()
        if tot <= 0:
            continue
        ax.plot(d.Group_Value, 100 * d.Permutation_Importance / tot, marker="o",
                ms=3.2, color=S.TARGET_COLOR[sid])
    ax.set_xlabel("lag (days; 0 = same day)")
    ax.set_ylabel("share of permutation importance (%)")
    ax.set_xticks(range(0, 8))
    S.despine(ax)
    S.title(ax, "How far back it comes from",
            "the same-day and one-day columns carry nearly all of it")
    S.save(fig, C.FIGURES / "fig_importance.pdf")


def fig_transfer(tr, reg) -> None:
    fig, axes = plt.subplots(1, len(C.TARGETS), figsize=(7.2, 2.4),
                             gridspec_kw={"wspace": 0.40})
    vmax = float(tr.RMSE.max())
    for ax, sid in zip(axes, C.TARGETS):
        d = tr[tr.Station_ID == sid]
        M = d.pivot(index="Train_Period", columns="Test_Period", values="RMSE")
        M = M.reindex(index=C.PERIOD_NAMES, columns=C.PERIOD_NAMES)
        ax.grid(False)
        im = ax.imshow(M.to_numpy(), cmap="YlOrRd", vmin=0, vmax=vmax)
        for i in range(3):
            for j in range(3):
                v = M.iat[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                            fontsize=5.6,
                            color="white" if v > 0.62 * vmax else S.INK,
                            fontweight="bold" if i == j else "normal")
        ax.set_xticks(range(3), C.PERIOD_NAMES, fontsize=5.6, rotation=45,
                      ha="right")
        # Row labels on the first panel only: repeated on every panel they
        # printed over the neighbouring panel's cells.
        ax.set_yticks(range(3))
        ax.set_yticklabels(C.PERIOD_NAMES if ax is axes[0] else [], fontsize=5.6)
        ax.set_title(label(sid, reg), loc="left", fontsize=6.6)
        ax.set_xlabel("tested on", fontsize=6.2)
        S.despine(ax, keep=())
    axes[0].set_ylabel("trained on", fontsize=6.2)
    S.figtitle(fig, "Does a model fitted in one bridge period hold in another?",
               "test-block RMSE in metres; the diagonal is the within-period "
               "score (bold)", top=1.16)
    S.save(fig, C.FIGURES / "fig_transfer.pdf")


def fig_confusion(cm, reg) -> None:
    fig, axes = plt.subplots(1, len(C.TARGETS), figsize=(7.2, 2.3),
                             gridspec_kw={"wspace": 0.30})
    for ax, sid in zip(axes, C.TARGETS):
        d = cm[(cm.Station_ID == sid) & (cm.Stratum == "ALL")
               & (cm.Feature_Set == PRIMARY)]
        if d.empty:
            ax.axis("off")
            continue
        labs = [l for l in C.RISK_LABELS if l in set(d["True"])]
        M = d.pivot(index="True", columns="Predicted", values="Count") \
             .reindex(index=labs, columns=labs).fillna(0).to_numpy()
        row = M.sum(axis=1, keepdims=True)
        pct = np.divide(M, row, out=np.zeros_like(M, dtype=float), where=row > 0)
        ax.grid(False)
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(labs)):
            for j in range(len(labs)):
                ax.text(j, i, f"{int(M[i, j])}", ha="center", va="center",
                        fontsize=5.6,
                        color="white" if pct[i, j] > 0.55 else S.INK)
        ax.set_xticks(range(len(labs)), labs, fontsize=5.4, rotation=45, ha="right")
        ax.set_yticks(range(len(labs)), labs, fontsize=5.4)
        ax.set_xlabel("predicted", fontsize=6.2)
        ax.set_title(label(sid, reg), loc="left", fontsize=6.6)
        S.despine(ax, keep=())
    axes[0].set_ylabel("observed", fontsize=6.2)
    S.figtitle(fig, "Flood-risk classification, test block, all years pooled",
               "counts, shaded by row share", top=1.16)
    S.save(fig, C.FIGURES / "fig_confusion.pdf")


def main() -> None:
    reg = pd.read_csv(C.DATA / "station_registry.csv").set_index("Station_ID")
    reg_m = pd.read_csv(C.RESULTS / "model_regression.csv")
    clf_m = pd.read_csv(C.RESULTS / "model_classification.csv")
    pred = pd.read_csv(C.RESULTS / "predictions.csv", parse_dates=["Date"])
    resid = pd.read_csv(C.RESULTS / "residual_diagnostics.csv")
    grp = pd.read_csv(C.RESULTS / "importance_grouped.csv")
    tr = pd.read_csv(C.RESULTS / "transfer_matrix.csv")
    cm = pd.read_csv(C.RESULTS / "confusion_matrices.csv")

    fig_skill(reg_m, clf_m, reg)
    fig_scatter(pred, reg_m, reg)
    fig_testseries(pred, reg)
    fig_residuals(pred, resid, reg)
    fig_importance(grp, reg)
    fig_transfer(tr, reg)
    fig_confusion(cm, reg)


if __name__ == "__main__":
    main()
