"""LaTeX table rows for the Section 10 problems added in v4.1 (bridge impact + use cases).

Every row is generated from a results CSV so that text, tables and figures cannot drift.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import common as cm

R, G = cm.RES_DIR, cm.GEN_DIR
END = r" \\ \hline"
ERAS = ["Before", "During", "After"]


def write(name: str, rows: list[str]) -> None:
    (G / name).write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"  wrote {name}")


def pval(p: float) -> str:
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def rating() -> None:
    e = pd.read_csv(R / "bi_specific_gauge_eras.csv")
    order = ["Hardinge Br.", "Baruria", "Mawa (bridge)"]
    rows = []
    for name in order:
        for _, r in e[e.name == name].sort_values("q_star").iterrows():
            rows.append(f"{name} & {r.q_star / 1000:.0f} & {r.Before:.2f} & {r.During:.2f} & {r.After:.2f} & "
                        f"{r.after_minus_before:+.2f} & {pval(r.p)}{END}")
    write("tab_bi_rating.tex", rows)


def sediment() -> None:
    e = pd.read_csv(R / "bi_sediment_eras.csv")
    names = {"SW91.9L": "Baruria (51 km upstream)", "SW93.5L": "Mawa (bridge site)"}
    rows = []
    for st, name in names.items():
        g = e[e.station == st].set_index("era").loc[ERAS]
        cells = " & ".join(f"{int(r.n)} / {r.median_conc_monsoon:.0f}" for _, r in g.iterrows())
        fac = " & ".join(f"{r.factor:.2f} ({r.lo:.2f}--{r.hi:.2f})" for _, r in g.iloc[1:].iterrows())
        rows.append(f"{name} & {cells} & {fac}{END}")
    s = pd.read_csv(R / "bi_sediment_samples.csv", parse_dates=["date"])
    m = s[s.date.dt.month.isin([7, 8, 9])]
    yr = m.groupby([m.date.dt.year, "station"]).conc_ppm.median().unstack()
    ratio = (yr["SW93.5L"] / yr["SW91.9L"]).dropna()
    per = {e_: ratio[[y for y in ratio.index if (y <= 2014 if e_ == "Before" else
                                                (y >= 2023 if e_ == "After" else 2015 <= y <= 2022))]].median()
           for e_ in ERAS}
    rows.append(r"\multicolumn{6}{|l|}{\textit{Mawa / Baruria, median of yearly Jul--Sep ratios:} "
                f"before {per['Before']:.1f}, during {per['During']:.1f}, after {per['After']:.1f}}}{END}")
    write("tab_bi_sediment.tex", rows)


def cross_sections() -> None:
    e = pd.read_csv(R / "bi_xs_eras.csv").set_index(["transect", "metric"])
    m = pd.read_csv(R / "bi_xs_metrics.csv")
    names = {"RMG12": "Control, Hardinge Br.", "RMP4": "18 km upstream", "RMP3": "Bridge line",
             "RMP2": "7 km downstream"}
    rows = []
    for t, name in names.items():
        th = e.loc[(t, "thalweg_m")]
        mb = e.loc[(t, "mean_bed_m")]
        al = e.loc[(t, "area_low_m2")]
        xs = m[m.transect == t].set_index("year").thalweg_x_m
        x_b = xs[xs.index <= 2014].median() / 1000
        x_a = xs[xs.index >= 2023].median() / 1000
        rows.append(f"{name} & {th.Before:.1f} & {th.After:.1f} & {pval(th.p)} & "
                    f"{mb.Before:.2f} & {mb.After:.2f} & {al.Before / 1000:.1f} & {al.After / 1000:.1f} & "
                    f"{x_b:.1f} $\\rightarrow$ {x_a:.1f}{END}")
    write("tab_bi_xs.tex", rows)


def xs_clusters() -> None:
    c = pd.read_csv(R / "bi_xs_clusters.csv")
    names = ["C1 deep scour channel", "C2 wide, shallow (sandbars)", "C3 deep, uniform"]
    rows = []
    for k, name in enumerate(names):
        g = c[c.cluster == k]
        cnt = " & ".join(str(int((g.era == e_).sum())) for e_ in ERAS)
        rows.append(f"{name} & {len(g)} & {cnt} & {100 * g.d5.mean():.0f}\\% & {100 * g.d15.mean():.0f}\\%{END}")
    write("tab_bi_xs_clusters.tex", rows)


def slope_tide() -> None:
    d = pd.read_csv(R / "bridge_reach_drop.csv")
    reach = {"11-12": "Baruria--Bhagyakul (51 km)", "12-13": "Bhagyakul--Mawa (5 km)",
             "13-15": "Mawa--Sureswar (31 km)"}
    rows = []
    for season, flow, label in (("Monsoon (Jul-Oct)", "High", "Monsoon, high flow"), ("Dry (Jan-Apr)", "All", "Dry season")):
        for r_, name in reach.items():
            g = d[(d.season == season) & (d.flow == flow) & (d.reach == r_)].set_index("era").loc[ERAS]
            rows.append(f"Water-surface drop (m) & {label}: {name} & {g.mean_m.Before:.2f} & {g.mean_m.During:.2f} & "
                        f"{g.mean_m.After:.2f}{END}")
    t = pd.read_csv(R / "bi_tide.csv")
    t["era"] = np.where(t.year <= 2014, "Before", np.where(t.year <= 2022, "During", "After"))
    for gid in (12, 13, 15, 17):
        g = t[t.gauge == gid].groupby("era").range_m.mean() * 100
        rows.append(f"Dry-season daily range (cm) & {cm.SHORT_NAME[gid]} ({gid}) & {g.Before:.0f} & {g.During:.0f} & "
                    f"{g.After:.0f}{END}")
    write("tab_bi_slope_tide.tex", rows)


def warning() -> None:
    x = pd.read_csv(R / "bridge_warning_x50.csv")
    rows = []
    for tgt in (12, 13):
        g = x[x.target == tgt].set_index("era").loc[ERAS]
        rows.append(f"{cm.SHORT_NAME[tgt]} ({tgt}) & " + " & ".join(f"{v:+.2f}" for v in g.x50_m) + " & "
                    + " & ".join(f"{100 * v:.1f}\\%" for v in g.warn_share) + END)
    write("tab_bi_warning.tex", rows)


def c3() -> None:
    t = pd.read_csv(R / "uc_c3_incremental.csv")
    lr = t[t.model == "Logistic regression"].pivot(index="step_name", columns="target", values="macro_f1")
    gb = t[t.model == "Gradient boosting"].pivot(index="step_name", columns="target", values="macro_f1")
    order = t.drop_duplicates("step").sort_values("step").step_name
    rows = []
    for s in order:
        cells = " & ".join(f"{lr.at[s, k]:.2f} / {gb.at[s, k]:.2f}" for k in (11, 12, 15))
        rows.append(f"{s.replace('-', '--')} & {cells}{END}")
    p = t.groupby("target").persistence_f1.first()
    rows.append("Same class as yesterday & " + " & ".join(f"{p[k]:.2f}" for k in (11, 12, 15)) + END)
    write("tab_uc_c3.tex", rows)


def c4() -> None:
    t = pd.read_csv(R / "uc_c4_rapid.csv")
    rows = []
    for g in (4, 10, 12):
        s = t[t.gauge == g]
        per = s[s.model == "Persistence"].iloc[0]
        own = s[(s.model == "Gradient boosting") & (s.step == "Own gauge")].iloc[0]
        up = s[(s.model == "Gradient boosting") & (s.step == "+ upstream, rain, season")].iloc[0]
        rows.append(f"{cm.SHORT_NAME[g]} ({g}) & {per.threshold_cm:.0f} & {int(per.events)} & "
                    f"{per.CSI:.2f} & {own.PR_AUC:.2f} / {own.CSI:.2f} & {up.PR_AUC:.2f} / {up.CSI:.2f} & "
                    f"{up.POD:.2f} / {up.FAR:.2f}{END}")
    write("tab_uc_c4.tex", rows)


def c7() -> None:
    s = pd.read_csv(R / "uc_c7_summary.csv")
    rows = [f"{r.features} & {100 * r.balanced_accuracy:.0f}\\% & {r.macro_f1:.2f}{END}" for _, r in s.iterrows()]
    write("tab_uc_c7.tex", rows)


def regimes() -> None:
    r = pd.read_csv(R / "regimes.csv")
    rows = [f"R{int(x.regime) + 1} {x['name']} & {x.mean_run_days:.1f} & {100 * x.p_warning_3d:.0f}\\% & "
            + " & ".join(f"{100 * x[f'share_{e_}']:.1f}\\%" for e_ in ERAS) + END for _, x in r.iterrows()]
    write("tab_regimes.tex", rows)


def flood_years() -> None:
    y = pd.read_csv(R / "cluster_years.csv", index_col=0)
    rows = []
    for t in ["Severe compound flood", "Moderate flood", "Mild / non-coincident"]:
        g = y[y.type == t]
        yrs = ", ".join(str(i) for i in g.index)
        rows.append(f"{t} & {yrs} & {g.bhagyakul_peak_margin_m.mean():+.2f} & {g.padma_danger_days.mean():.0f} & "
                    f"{g.ganges_jamuna_peak_gap_d.median():.0f}{END}")
    write("tab_flood_years.tex", rows)


def u8() -> None:
    s = pd.read_csv(R / "uc_u8_summary.csv")
    names = ["E1 early rise from low water", "E2 mid-monsoon rise near danger"]
    rows = [f"{names[int(r.cluster)]} & {int(r.events)} & {r.start_doy:.0f} & {r.start_margin_m:+.1f} & "
            f"{r.duration_d:.0f} & {r.total_rise_m:.2f} & {r.max_daily_rise_cm:.0f} & {100 * r.share_danger:.0f}\\%{END}"
            for _, r in s.iterrows()]
    write("tab_uc_u8.tex", rows)


def extra_use_cases() -> None:
    """Appendix: the further use cases computed but not bridge-specific."""
    c5 = pd.read_csv(R / "uc_c5_logo.csv")
    uns = c5[c5.model.str.startswith("Unseen")].CSI
    own = c5[c5.model == "Own-gauge model"].CSI
    per = c5[c5.model == "Persistence"].CSI
    c6 = pd.read_csv(R / "uc_c6_summary.csv").set_index("label")
    u4k = pd.read_csv(R / "uc_u4_k.csv")
    u5 = pd.read_csv(R / "uc_u5_summary.csv").set_index("features")
    u6 = pd.read_csv(R / "uc_u6_summary.csv").set_index("season")
    u7 = pd.read_csv(R / "uc_u7_response.csv").set_index("type")
    rows = [
        f"C5 Warning model for an unseen gauge & Classification, leave-one-gauge-out & median CSI {uns.median():.2f} "
        f"(own-gauge model {own.median():.2f}, persistence {per.median():.2f}) over {len(uns)} gauges{END}",
        f"C6 Gauge type from 30 days of changes & Classification, leave-one-gauge-out & balanced accuracy "
        f"{c6.balanced_accuracy['Tidal vs non-tidal']:.2f} tidal/non-tidal, {c6.balanced_accuracy['River (4 classes)']:.2f} river "
        f"(chance 0.50 / 0.25){END}",
        f"U4 Annual hydrograph shapes & $k$-means on 228 gauge-years & $k$ = 2 (silhouette {u4k.silhouette.max():.2f}): "
        f"early Jamuna-fed rise vs late upper-Ganges rise{END}",
        f"U5 Anomalous daily readings & Isolation Forest, 0.1\\% flagged & own change alone catches "
        f"{int(u5.known_caught['Own change'])}/{int(u5.known['Own change'])} jumps $>$1~m; adding features drops it to "
        f"{int(u5.known_caught['+ neighbour residual'])}{END}",
        f"U6 Gauge co-movement network & Average-linkage on 1$-r$ & monsoon median $r$ = {u6.median_r.iloc[0]:.2f}, "
        f"dry {u6.median_r.iloc[1]:.2f}: the corridor decouples in the dry season{END}",
        f"U7 Monsoon rain types (8 gauges) & $k$-means & heavy-rain days: Goalundo +{u7.g10_rise3_cm['T2']:.0f}~cm in 3 days "
        f"vs {u7.g10_rise3_cm['No rain (<= 1 mm everywhere)']:.0f}~cm on dry days{END}",
    ]
    write("tab_uc_extra.tex", rows)


def main() -> None:
    for f in (rating, sediment, cross_sections, xs_clusters, slope_tide, warning, c3, c4, c7, regimes,
              flood_years, u8, extra_use_cases, trends, travel):
        f()



def travel() -> None:
    """Travel time of a Panka surge: peak-correlation lag, celerity estimate, large-surge arrival."""
    r3 = cm.DATASET / "results" / "r3" / "metrics"
    peak = pd.read_csv(r3 / "e3_peak_lag_chainage.csv").set_index("target_id")
    ev = pd.read_csv(r3 / "e3_event_arrival.csv").set_index("target_id")
    slope = np.polyfit(peak.chainage_km, peak.lag, 1)[0]
    rows = []
    for tid, r in peak.iterrows():
        ci = f"{r.lag_ci_lo:.0f}--{r.lag_ci_hi:.0f}" if r.lag_ci_hi > r.lag_ci_lo else f"{r.lag_ci_lo:.0f}"
        arr = (f"{ev.at[tid, 'median_lag']:.0f} ({ev.at[tid, 'iqr_lo']:.0f}--{ev.at[tid, 'iqr_hi']:.1f})"
               .replace(".0)", ")") if tid in ev.index else "--")
        name = f"\\textbf{{{cm.SHORT_NAME[tid]} ({tid})}}" if tid in cm.TARGETS else f"{cm.SHORT_NAME[tid]} ({tid})"
        rows.append(f"{name} & {r.chainage_km:.0f} & {r.lag:.0f} ({ci}) & {r['corr']:.2f} & "
                    f"{r.chainage_km * slope:.1f} & {arr}{END}")
    write("tab_travel.tex", rows)


def trends() -> None:
    t = pd.read_csv(R / "trends.csv", index_col=0)
    rows = []
    for sid in (1, 4, 10, 11, 12, 13, 15):
        r = t.loc[sid]
        star = lambda p: "$^*$" if p < 0.05 else ""  # noqa: E731
        rows.append(f"{cm.SHORT_NAME[sid]} ({sid}) & {r.min_slope_cm:+.1f}{star(r.min_p)} & {pval(r.min_p)} & "
                    f"{r.max_slope_cm:+.1f}{star(r.max_p)} & {pval(r.max_p)}{END}")
    write("tab_trends.tex", rows)


if __name__ == "__main__":
    main()
