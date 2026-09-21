"""Step 09b - the two figures that carry the diagnosis.

Left panel   skill above persistence for the level target and the change
             target, side by side, so the sign flip is visible at a glance.
Right panel  the extrapolation ceiling: predicted against observed stage on the
             pooled stratum, with the training range marked. Points to the right
             of the marked range are levels the forest was never able to reach.
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

S.use()


def main() -> None:
    dl = pd.read_csv(C.RESULTS / "model_delta.csv")
    ex = pd.read_csv(C.RESULTS / "extrapolation.csv")
    pred = pd.read_csv(C.RESULTS / "predictions.csv", parse_dates=["Date"])
    reg = pd.read_csv(C.DATA / "station_registry.csv").set_index("Station_ID")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1),
                             gridspec_kw={"width_ratios": [1.3, 1.0],
                                          "wspace": 0.26})

    # ---------------------------------------------- skill, both targets
    ax = axes[0]
    strata = C.PERIOD_NAMES + ["ALL"]
    labels, pil, pic = [], [], []
    for tgt in C.TARGETS:
        for st in strata:
            r = dl[(dl.Station_ID == tgt) & (dl.Stratum == st)]
            if r.empty:
                continue
            labels.append(f"{C.STATION_ID[tgt]}·{st[:3]}")
            pil.append(float(r.PI_Level.iloc[0]))
            pic.append(float(r.PI_Change.iloc[0]))
    x = np.arange(len(labels))
    ax.bar(x - 0.21, pil, width=0.42, color=S.GREY, lw=0,
           label="target = stage (level)")
    ax.bar(x + 0.21, pic, width=0.42, color=S.TEAL, lw=0,
           label="target = daily change")
    ax.axhline(0, color=S.INK, lw=0.9)
    ax.set_xticks(x, labels, rotation=90, fontsize=5.2)
    ax.set_ylabel("Persistence Index")
    # One stratum is far enough below the rest to flatten everything else, so
    # the axis is clipped and that bar is labelled instead.
    floor = -2.2
    ax.set_ylim(floor, max(pil + pic) + 0.55)
    for xi, v in zip(x, pil):
        if v < floor:
            ax.annotate(f"{v:.2f}", (xi - 0.21, floor), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=5.4, color=S.RUST, fontweight="bold")
            ax.annotate("", (xi - 0.21, floor), xytext=(xi - 0.21, floor + 0.30),
                        arrowprops=dict(arrowstyle="-|>", color=S.RUST, lw=0.7))
    ax.legend(loc="upper center", ncols=2)
    S.despine(ax)
    S.title(ax, "Skill above persistence, by target variable",
            "above the line beats the naive rule; below it, does not")

    # ------------------------------------ error on the scale that decides it
    # Everything is divided by the average daily movement, because that is the
    # error persistence makes by construction. Above 1 the model is doing worse
    # than standing still; below it, better.
    ax = axes[1]
    m = dl.merge(ex[["Station_ID", "Stratum", "Mean_Abs_Daily_Change"]],
                 on=["Station_ID", "Stratum"])
    m = m[m.Stratum.isin(strata)]
    y = np.arange(len(m))
    move = m.Mean_Abs_Daily_Change.to_numpy()
    ax.barh(y + 0.22, m.RMSE_Level / move, height=0.30, color=S.GREY, lw=0,
            label="level target")
    ax.barh(y - 0.10, m.RMSE_Persistence / move, height=0.30, color=S.AMBER,
            lw=0, label="persistence")
    ax.barh(y - 0.42, m.RMSE_Change / move, height=0.30, color=S.TEAL, lw=0,
            label="change target")
    ax.axvline(1.0, color=S.INK, lw=0.8)
    ax.set_yticks(y, [f"{C.STATION_ID[s]}·{st[:3]}"
                      for s, st in zip(m.Station_ID, m.Stratum)], fontsize=5.2)
    ax.invert_yaxis()
    ax.set_xlabel("RMSE ÷ average daily movement")
    ax.legend(loc="lower right")
    S.despine(ax)
    S.title(ax, "Error on the scale that decides it",
            "one unit is the size of a single day's move in the river")
    S.save(fig, C.FIGURES / "fig_delta.pdf")


if __name__ == "__main__":
    main()
