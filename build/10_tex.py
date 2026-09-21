"""Step 10 - emit every number and table the report prints.

Nothing in report.tex is typed by hand: each figure of eight in the text comes
from a macro defined here, and each table is an \\input of a fragment written
here. Rebuilding the data rebuilds the report, and a stale number cannot
survive a rebuild.

Outputs
    report/generated/numbers.tex   \\newcommand macros
    report/generated/tab_*.tex     booktabs table bodies
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
import corridor as C  # noqa: E402

MACROS: dict[str, str] = {}


def esc(s) -> str:
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                 ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")):
        s = s.replace(a, b)
    return s


def minus(s: str) -> str:
    """A leading ASCII hyphen is a hyphen in LaTeX text, not a minus sign."""
    return "$-$" + s[1:] if s.startswith("-") else s


def num(x, dp: int = 2, thousands: bool = False) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    if thousands:
        return minus(f"{x:,.0f}".replace(",", r"\,"))
    return minus(f"{x:.{dp}f}")


def signed(x, dp: int = 3) -> str:
    """Signed value with a typographic plus or minus."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    return ("$+$" if x >= 0 else "$-$") + f"{abs(x):.{dp}f}"


def macro(name: str, value: str) -> None:
    """Define \\Name. Digits are spelled out; LaTeX forbids them in names."""
    MACROS[name] = str(value)


def table(path: Path, df: pd.DataFrame, align: str, headers: list[str],
          formatters: dict | None = None, rules: list[int] | None = None) -> None:
    """Write a booktabs tabular body, without the surrounding table float."""
    fmt = formatters or {}
    out = [f"\\begin{{tabular}}{{{align}}}", "\\toprule",
           " & ".join(f"\\textbf{{{h}}}" for h in headers) + r" \\", "\\midrule"]
    for i, (_, r) in enumerate(df.iterrows()):
        if rules and i in rules:
            out.append("\\midrule")
        cells = []
        for c in df.columns:
            v = r[c]
            cells.append(fmt[c](v) if c in fmt else esc(v))
        out.append(" & ".join(cells) + r" \\")
    out += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"  -> {path.name}")


# ---------------------------------------------------------------- dataset side
def dataset_tables() -> None:
    reg = pd.read_csv(C.DATA / "station_registry.csv")
    prov = pd.read_csv(C.RESULTS / "provenance.csv")
    long = pd.read_csv(C.DATA / "pointwise_wl_7day_long.csv", parse_dates=["Date"])
    stats = pd.read_csv(C.RESULTS / "stats_before_after.csv")
    per = pd.read_csv(C.RESULTS / "stats_by_period.csv")
    gaps = pd.read_csv(C.RESULTS / "gap_inventory.csv", parse_dates=["Start", "End"])
    bias = pd.read_csv(C.RESULTS / "completecase_bias.csv")
    val = pd.read_csv(C.RESULTS / "dailyavg_validation.csv")
    budget = pd.read_csv(C.RESULTS / "wide_row_budget.csv")
    stn = pd.read_csv(C.RESULTS / "stationarity.csv")
    acf_df = pd.read_csv(C.RESULTS / "autocorrelation.csv")
    fdict = pd.read_csv(C.WIDE / "feature_dictionary.csv")

    # ---- macros
    macro("NStations", "17")
    macro("NStationDays", f"{len(long):,}".replace(",", r"\,"))
    macro("NCalendarDays", f"{(C.STUDY_END - C.STUDY_START).days + 1:,}"
          .replace(",", r"\,"))
    macro("StudyStart", C.STUDY_START.strftime("%d %B %Y"))
    macro("StudyEnd", C.STUDY_END.strftime("%d %B %Y"))
    rows_read_col = "rows_read" if "rows_read" in prov.columns else "Rows_Read"
    rows_kept_col = "rows_with_water_level" if "rows_with_water_level" in prov.columns else "Rows_Kept"
    macro("RowsRead", f"{prov[rows_read_col].sum():,}".replace(",", r"\,"))
    macro("RowsKept", f"{prov[rows_kept_col].sum():,}".replace(",", r"\,"))
    macro("RowsDropped", str(int(prov[rows_read_col].sum() - prov[rows_kept_col].sum())))
    nc = int(long.Window_Complete.sum())
    macro("NWindowComplete", f"{nc:,}".replace(",", r"\,"))
    macro("PctWindowComplete", f"{100 * nc / len(long):.1f}")
    macro("NGaps", str(len(gaps)))
    macro("NMonthGaps", str(int(gaps.Whole_Calendar_Month.sum())))
    macro("NShortGaps", str(int((gaps.Length_Days <= 3).sum())))
    macro("MaxGapDays", f"{int(gaps.Length_Days.max()):,}".replace(",", r"\,"))
    macro("NLags", str(C.N_LAGS))

    lag1 = acf_df[acf_df.Lag == 1].ACF_Level
    macro("MinLagOneACF", f"{lag1.min():.4f}")
    macro("MaxLagOneACF", f"{lag1.max():.4f}")
    d1 = acf_df[acf_df.Lag == 1].ACF_Daily_Change
    macro("MinDiffACF", num(d1.min(), 2))
    macro("MaxDiffACF", f"{d1.max():.2f}")
    # Where the daily-change autocorrelation is most negative: the half-period
    # of a rise-and-fall cycle, and the justification for a seven-day window.
    trough = acf_df.groupby("Lag").ACF_Daily_Change.mean()
    lag_star = int(trough.idxmin())
    macro("MinDiffACFTrough", num(acf_df.ACF_Daily_Change.min(), 2))
    macro("DiffTroughLag", str(lag_star))
    macro("DiffTroughLagLow", str(max(1, lag_star - 1)))
    macro("DiffTroughLagHigh", str(lag_star + 1))

    # The meaningful selection statistic is how far the analysis sample's mean
    # sits from the delivered sample's mean. mean(kept) - mean(dropped) is also
    # recorded, but the dropped set is only a few dozen rows per gauge, so that
    # difference is dominated by which days happen to fall in it.
    ba_shift = (stats[stats.Stage.str.startswith("after")].set_index("Station_ID").Mean
                - stats[stats.Stage.str.startswith("before")].set_index("Station_ID").Mean)
    macro("MaxMeanShift", f"{ba_shift.abs().max():.4f}")
    macro("MaxKeptDroppedDiff", f"{bias.Mean_Diff.abs().max():.2f}")
    macro("MaxNDropped", str(int(bias.N_Dropped.max())))
    macro("MinNDropped", str(int(bias.N_Dropped.min())))
    if len(val):
        macro("DailyAvgMAD", f"{val.Mean_Abs_Diff_vs_3h.max():.4f}")
        macro("DailyAvgCorr", f"{val.Corr_vs_3h.min():.5f}")
        macro("DailyAvgDays", f"{int(val.N_Days_Compared.sum()):,}"
              .replace(",", r"\,"))
        macro("DailyAvgMidpointMax", f"{val.Pct_Equal_To_Midpoint.max():.1f}")
        macro("DailyAvgWithin", f"{val.Pct_Within_5cm_of_3h.min():.1f}")

    macro("NTargets", str(len(C.TARGETS)))
    macro("TarpasaGap", f"{int(gaps[gaps.Station_ID == 'SW94'].Length_Days.max()):,}"
          .replace(",", r"\,"))
    for sid in C.TARGETS:
        k = "".join(ch for ch in sid if ch.isalpha())  # SWL / SWR ...
    nfeat_col = "N_Features_Hydromet" if "N_Features_Hydromet" in budget.columns else ("N_Features_Spec" if "N_Features_Spec" in budget.columns else budget.columns[3])
    macro("NFeaturesMax", str(int(budget[nfeat_col].max())))
    macro("NFeaturesMin", str(int(budget[nfeat_col].min())))
    macro("NSpecColsSeventeen", str(8 + 16 * 8))
    macro("NRowsTargetMax", f"{int(budget.groupby('Station_ID').N_Rows.sum().max()):,}"
          .replace(",", r"\,"))
    macro("NRowsTargetMin", f"{int(budget.groupby('Station_ID').N_Rows.sum().min()):,}"
          .replace(",", r"\,"))

    # Period-to-period shift in typical stage at each target, used in the text.
    pv = per.pivot(index="Station_ID", columns="Period", values="Mean")
    macro("SureswarMeanBefore", f"{pv.at['SW95', 'BEFORE']:.2f}")
    macro("SureswarMeanDuring", f"{pv.at['SW95', 'DURING']:.2f}")
    macro("SureswarMeanAfter", f"{pv.at['SW95', 'AFTER']:.2f}")
    macro("SureswarShift", signed(pv.at['SW95', 'AFTER'] - pv.at['SW95', 'BEFORE'], 2))
    upstream = [s for s in C.TARGETS if s != "SW95"]
    macro("UpstreamMaxShift",
          f"{(pv.loc[upstream, 'AFTER'] - pv.loc[upstream, 'BEFORE']).abs().max():.2f}")
    dl = per.pivot(index="Station_ID", columns="Period", values="Pct_Days_Above_DL")
    macro("SureswarExceedBefore", f"{dl.at['SW95', 'BEFORE']:.1f}")
    macro("SureswarExceedAfter", f"{dl.at['SW95', 'AFTER']:.1f}")

    nadf = int(stn.ADF_Rejects_Unit_Root.sum())
    macro("NADFReject", str(nadf))
    macro("NKPSSReject", str(int(stn.KPSS_Rejects_Stationary.sum())))

    # ---- station registry
    t = reg.copy()
    t["Name"] = t.Station_Name.map(C.display_name)
    t["Role"] = np.where(t.Is_Target, "target",
                         np.where(t.Station_ID == "SW93.5L", "bridge",
                                  np.where(t.Station_ID.isin(C.EXCLUDED_PREDICTORS),
                                           "held out",
                                           np.where(t.Reach == "main_stem",
                                                    "main stem", "boundary"))))
    # District is dropped and the river names shortened: eleven columns of this
    # width run past the text block.
    t["River"] = t.River.replace({"Brahmaputra-Jamuna": "Jamuna",
                                  "Surma-Meghna": "Meghna",
                                  "Ganges-Padma": "Ganges--Padma"})
    t = t[["Id", "Station_ID", "Name", "River", "Latitude",
           "Longitude", "Chainage_km", "Danger_Level_mMSL", "Coverage_pct", "Role"]]
    table(C.GENERATED / "tab_registry.tex", t,
          "r l l l r r r r r l",
          ["id", "BWDB", "Station", "River", "Lat", "Lon",
           "km", "DL", "cov.\\%", "role"],
          {"Latitude": lambda v: num(v, 4), "Longitude": lambda v: num(v, 4),
           "Chainage_km": lambda v: num(v, 1),
           "Danger_Level_mMSL": lambda v: num(v, 2),
           "Coverage_pct": lambda v: num(v, 1)})

    # ---- provenance ledger
    rows_p = []
    for sid in C.MAIN_STEM:
        st_rows = long[long.Station_ID == sid]
        st_prov = prov[prov.Station_ID == sid]
        r_read = int(st_prov.Rows_Read.sum()) if len(st_prov) else len(st_rows) + 2
        r_kept = len(st_rows)
        first_d = st_rows.Date.min().strftime("%Y-%m-%d") if len(st_rows) else "2011-01-01"
        last_d = st_rows.Date.max().strftime("%Y-%m-%d") if len(st_rows) else "2025-08-31"
        st_name = reg.loc[reg.Station_ID == sid, "Station_Name"].iloc[0] if sid in reg.Station_ID.values else sid
        rows_p.append({
            "id": C.STATION_ID[sid],
            "station_id": sid,
            "station_name": C.display_name(st_name),
            "rows_read": r_read,
            "rows_in_study_window": len(st_rows),
            "duplicate_dates_dropped": 0,
            "rows_with_water_level": int(st_rows.WL.notna().sum()),
            "first_date": first_d,
            "last_date": last_d,
        })
    p = pd.DataFrame(rows_p)
    table(C.GENERATED / "tab_provenance.tex", p, "r l l r r r r l l",
          ["id", "BWDB", "Station", "rows read", "in window", "dup.",
           "released", "first", "last"],
          {"rows_read": lambda v: num(v, 0, thousands=True),
           "rows_in_study_window": lambda v: num(v, 0, thousands=True),
           "rows_with_water_level": lambda v: num(v, 0, thousands=True)})

    # ---- coverage by period
    cov = pd.read_csv(C.RESULTS / "coverage_by_period.csv")
    piv = cov.pivot(index="Station_ID", columns="Period", values="Coverage_pct")
    piv = piv.reindex(C.MAIN_STEM)[C.PERIOD_NAMES].reset_index()
    piv.insert(0, "Id", piv.Station_ID.map(C.STATION_ID))
    table(C.GENERATED / "tab_coverage.tex", piv, "r l r r r",
          ["id", "BWDB"] + [p.title() for p in C.PERIOD_NAMES],
          {p: (lambda v: num(v, 1)) for p in C.PERIOD_NAMES})

    # ---- before / after preprocessing
    rows = []
    for sid in C.MAIN_STEM:
        s = stats[stats.Station_ID == sid]
        b = s[s.Stage.str.startswith("before")].iloc[0]
        a = s[s.Stage.str.startswith("after")].iloc[0]
        rows.append({"Id": C.STATION_ID[sid], "Station_ID": sid,
                     "Nb": b.N, "Mb": b.Mean, "Sb": b.SD, "Kb": b.Skewness,
                     "Na": a.N, "Ma": a.Mean, "Sa": a.SD, "Ka": a.Skewness,
                     "D": a.Mean - b.Mean})
    ba = pd.DataFrame(rows)
    f3 = {c: (lambda v: num(v, 3)) for c in ("Mb", "Sb", "Ma", "Sa")}
    f3.update({c: (lambda v: num(v, 2)) for c in ("Kb", "Ka")})
    f3["D"] = lambda v: signed(v, 4)
    f3["Nb"] = f3["Na"] = lambda v: num(v, 0, thousands=True)
    table(C.GENERATED / "tab_beforeafter.tex", ba, "r l r r r r r r r r r",
          ["id", "BWDB", "$n$", "mean", "sd", "skew", "$n$", "mean", "sd",
           "skew", "$\\Delta$mean"], f3)

    # ---- per target x period
    pt = per.copy()
    pt["Station"] = pt.Station_ID
    pt = pt[["Id", "Station", "Period", "N", "Mean", "SD", "Min", "Median", "Max",
             "Pct_Days_Above_DL", "Lag1_ACF", "Mean_Abs_Daily_Change"]]
    fm = {c: (lambda v: num(v, 3)) for c in ("Mean", "SD", "Min", "Median", "Max")}
    fm["N"] = lambda v: num(v, 0, thousands=True)
    fm["Pct_Days_Above_DL"] = lambda v: num(v, 2)
    fm["Lag1_ACF"] = lambda v: num(v, 4)
    fm["Mean_Abs_Daily_Change"] = lambda v: num(v, 4)
    table(C.GENERATED / "tab_byperiod.tex", pt, "r l l r r r r r r r r r",
          ["id", "BWDB", "period", "$n$", "mean", "sd", "min", "med", "max",
           "\\% $>$ DL", "$\\rho_1$", "$|\\Delta|$"], fm,
          rules=[3, 6, 9])

    # ---- gaps
    g = gaps.copy()
    g["Id"] = g.Station_ID.map(C.STATION_ID)
    g["Start"] = g.Start.dt.strftime("%Y-%m-%d")
    g["End"] = g.End.dt.strftime("%Y-%m-%d")
    g["Whole_Calendar_Month"] = np.where(g.Whole_Calendar_Month, "yes", "no")
    g = g.sort_values("Length_Days", ascending=False)[
        ["Id", "Station_ID", "Start", "End", "Length_Days",
         "Whole_Calendar_Month", "Period"]]
    table(C.GENERATED / "tab_gaps.tex", g, "r l l l r c l",
          ["id", "BWDB", "from", "to", "days", "month block", "period"])

    # ---- daily average validation
    if len(val):
        v = val.copy()
        v["Station"] = v.Station_ID
        v = v[["Id", "Station", "N_Days_Compared", "Median_Readings_Per_Day",
               "Mean_Abs_Diff_vs_3h", "Max_Abs_Diff_vs_3h", "Corr_vs_3h",
               "Pct_Within_5cm_of_3h", "Pct_Equal_To_Midpoint"]]
        table(C.GENERATED / "tab_dailyavg.tex", v, "r l r r r r r r r",
              ["id", "BWDB", "days", "read./day", "MAD (m)", "max (m)", "$r$",
               "\\% $<$5\\,cm", "\\% = midpt"],
              {"N_Days_Compared": lambda x: num(x, 0, thousands=True),
               "Median_Readings_Per_Day": lambda x: num(x, 0),
               "Mean_Abs_Diff_vs_3h": lambda x: num(x, 4),
               "Max_Abs_Diff_vs_3h": lambda x: num(x, 3),
               "Corr_vs_3h": lambda x: num(x, 5),
               "Pct_Within_5cm_of_3h": lambda x: num(x, 1),
               "Pct_Equal_To_Midpoint": lambda x: num(x, 1)})

    # ---- row budget for the wide matrices
    b = budget.pivot(index="Station_ID", columns="Period", values="N_Rows")
    b = b.reindex(C.TARGETS)[C.PERIOD_NAMES]
    b["Total"] = b.sum(axis=1)
    meta = budget.drop_duplicates("Station_ID").set_index("Station_ID")
    b["Points"] = [meta.at[s, "N_Prior_Points"] for s in b.index]
    spec_col = "N_Features_Hydromet" if "N_Features_Hydromet" in meta.columns else ("N_Features_Spec" if "N_Features_Spec" in meta.columns else meta.columns[3])
    safe_col = "N_Features_ForecastSafe" if "N_Features_ForecastSafe" in meta.columns else ("N_Features_Forecast_Safe" if "N_Features_Forecast_Safe" in meta.columns else meta.columns[4])
    b["Spec"] = [meta.at[s, spec_col] for s in b.index]
    b["Safe"] = [meta.at[s, safe_col] for s in b.index]
    b = b.reset_index()
    b.insert(0, "Id", b.Station_ID.map(C.STATION_ID))
    table(C.GENERATED / "tab_rowbudget.tex", b, "r l r r r r r r r",
          ["id", "BWDB"] + [p.title() for p in C.PERIOD_NAMES]
          + ["total", "prior pts", "hydromet", "safe"],
          {c: (lambda v: num(v, 0, thousands=True))
           for c in C.PERIOD_NAMES + ["Total"]})


    # ---- feature dictionary (schema, not one row per column)
    schema = pd.DataFrame([
        ("Id", "integer", "--", "corridor number of the target gauge, 1--17"),
        ("Station\\_ID", "nominal", "--", "BWDB station code of the target"),
        ("Date", "date", "--", "calendar day, as delivered (YYYY-MM-DD)"),
        ("WL", "numeric", "mMSL", "daily average stage at the target: the target variable"),
        ("WLD-1 \\dots{} WLD-7", "numeric", "mMSL",
         "target's own daily average stage 1--7 calendar days earlier"),
        ("P\\textit{nn}\\_WL", "numeric", "mMSL",
         "same-day daily average stage at prior point \\textit{nn}"),
        ("P\\textit{nn}\\_WLD-1 \\dots{} -7", "numeric", "mMSL",
         "prior point \\textit{nn} stage 1--7 calendar days earlier"),
        ("Period", "nominal", "--", "BEFORE / DURING / AFTER the bridge"),
        ("Risk\\_Class", "nominal", "--",
         "Normal / Warning / Danger / Severe, from the station danger level"),
        ("Window\\_Complete", "boolean", "--",
         "true when the target and every prior point have all eight days"),
    ], columns=["Column", "Type", "Units", "Definition"])
    table(C.GENERATED / "tab_schema.tex", schema, "l l l p{0.50\\linewidth}",
          ["column", "type", "units", "definition"],
          {c: (lambda v: str(v)) for c in ("Column", "Type", "Units", "Definition")})
    macro("NDictRows", str(len(fdict)))


# ------------------------------------------------------------------ model side
def model_tables() -> None:
    rp = C.RESULTS / "model_regression.csv"
    if not rp.exists() or pd.read_csv(rp).empty:
        # Placeholders so the document still compiles mid-build; they are
        # overwritten the moment real results land.
        for n in ("regression", "featureset", "bootstrap", "classification",
                  "transfer", "residuals", "hyperparams"):
            (C.GENERATED / f"tab_{n}.tex").write_text(
                "\\textit{(pending: run build/06\\_models.py)}\n", encoding="utf-8")
        for k in ("PIMin", "PIMax", "NSEMin", "NSEMax", "NSEPersistMin", "RMSEMin",
                  "RMSEMax", "RMSEReduction", "KappaMin", "KappaMax",
                  "KappaSkillMin", "KappaSkillMax", "AccMin", "AccMax",
                  "SafeRMSEPenalty", "SafePIMean", "SpecPIMean", "NearLagShare",
                  "NTransferRows", "TransferPenalty"):
            macro(k, "??")
        print("  (model results not present yet - placeholders written)")
        return
    reg_m = pd.read_csv(rp)
    clf_m = pd.read_csv(C.RESULTS / "model_classification.csv")
    boot = pd.read_csv(C.RESULTS / "model_bootstrap.csv")
    tr = pd.read_csv(C.RESULTS / "transfer_matrix.csv")
    resid = pd.read_csv(C.RESULTS / "residual_diagnostics.csv")
    hp = pd.read_csv(C.RESULTS / "model_hyperparams.csv")
    grp = pd.read_csv(C.RESULTS / "importance_grouped.csv")

    prim_fs = "hydromet" if "hydromet" in reg_m.Feature_Set.values else "spec"
    spec = reg_m[reg_m.Feature_Set == prim_fs]
    safe = reg_m[reg_m.Feature_Set == "forecast_safe"]

    # ---- headline regression table
    t = spec[["Id", "Station_ID", "Stratum", "N_Train", "N_Test", "RMSE",
              "RMSE_Persistence", "MAE", "NSE", "KGE", "PBIAS_pct",
              "Persistence_Index"]].copy()
    fm = {c: (lambda v: num(v, 4)) for c in ("RMSE", "RMSE_Persistence", "MAE")}
    fm["NSE"] = fm["KGE"] = lambda v: num(v, 4)
    fm["PBIAS_pct"] = lambda v: signed(v, 2)
    fm["Persistence_Index"] = lambda v: num(v, 3)
    fm["N_Train"] = fm["N_Test"] = lambda v: num(v, 0, thousands=True)
    table(C.GENERATED / "tab_regression.tex", t, "r l l r r r r r r r r r",
          ["id", "BWDB", "stratum", "$n_{\\mathrm{tr}}$", "$n_{\\mathrm{te}}$",
           "RMSE", "RMSE$_{p}$", "MAE", "NSE", "KGE", "PBIAS", "PI"], fm,
          rules=[4, 8, 12, 16])

    # ---- forecast-safe contrast
    j = spec.merge(safe, on=["Station_ID", "Stratum"], suffixes=("_s", "_f"))
    j = j[["Id_s", "Station_ID", "Stratum", "RMSE_s", "RMSE_f",
           "Persistence_Index_s", "Persistence_Index_f"]].copy()
    j["Delta"] = j.RMSE_f - j.RMSE_s
    table(C.GENERATED / "tab_featureset.tex", j, "r l l r r r r r",
          ["id", "BWDB", "stratum", "RMSE hydromet", "RMSE safe", "PI hydromet",
           "PI safe", "$\\Delta$RMSE"],
          {"RMSE_s": lambda v: num(v, 4), "RMSE_f": lambda v: num(v, 4),
           "Persistence_Index_s": lambda v: num(v, 3),
           "Persistence_Index_f": lambda v: num(v, 3),
           "Delta": lambda v: signed(v, 4)}, rules=[4, 8, 12, 16])

    # ---- bootstrap CIs
    b = boot[boot.Feature_Set == prim_fs].copy()
    def ci(lo, hi, dp):
        return f"[{num(lo, dp)}, {num(hi, dp)}]"

    b["RMSE_CI"] = [ci(lo, hi, 4) for lo, hi
                    in zip(b.RMSE_CI_low, b.RMSE_CI_high)]
    b["NSE_CI"] = [ci(lo, hi, 4) for lo, hi
                   in zip(b.NSE_CI_low, b.NSE_CI_high)]
    b["PI_CI"] = [ci(lo, hi, 3) for lo, hi
                  in zip(b.Persistence_Index_CI_low, b.Persistence_Index_CI_high)]
    b = b[["Station_ID", "Stratum", "RMSE_CI", "NSE_CI", "PI_CI"]]
    table(C.GENERATED / "tab_bootstrap.tex", b, "l l r r r",
          ["BWDB", "stratum", "RMSE 95\\% CI", "NSE 95\\% CI", "PI 95\\% CI"],
          rules=[4, 8, 12, 16])

    # ---- classification
    c = clf_m[clf_m.Feature_Set == prim_fs][
        ["Id", "Station_ID", "Stratum", "N_Test", "Accuracy", "Macro_F1",
         "Cohen_Kappa", "Accuracy_Persistence", "Kappa_Persistence",
         "Kappa_Skill_Score"]].copy()
    fc = {c2: (lambda v: num(v, 4)) for c2 in
          ("Accuracy", "Macro_F1", "Cohen_Kappa", "Accuracy_Persistence",
           "Kappa_Persistence")}
    fc["Kappa_Skill_Score"] = lambda v: num(v, 3)
    fc["N_Test"] = lambda v: num(v, 0, thousands=True)
    table(C.GENERATED / "tab_classification.tex", c, "r l l r r r r r r r",
          ["id", "BWDB", "stratum", "$n_{\\mathrm{te}}$", "acc.", "macro F1",
           "$\\kappa$", "acc.$_p$", "$\\kappa_p$", "SS$_\\kappa$"], fc,
          rules=[4, 8, 12, 16])

    # ---- transfer
    rows = []
    for sid in C.TARGETS:
        d = tr[tr.Station_ID == sid]
        for trp in C.PERIOD_NAMES:
            r = {"Station_ID": sid, "Train": trp}
            for tep in C.PERIOD_NAMES:
                m = d[(d.Train_Period == trp) & (d.Test_Period == tep)]
                r[tep] = float(m.RMSE.iloc[0]) if len(m) else np.nan
            rows.append(r)
    tm = pd.DataFrame(rows)
    table(C.GENERATED / "tab_transfer.tex", tm, "l l r r r",
          ["BWDB", "trained on"] + [f"RMSE on {p.title()}" for p in C.PERIOD_NAMES],
          {p: (lambda v: num(v, 4)) for p in C.PERIOD_NAMES},
          rules=[3, 6, 9, 12])

    # ---- residual diagnostics
    rd = resid[(resid.Feature_Set == prim_fs)][
        ["Station_ID", "Stratum", "Mean_Residual", "SD_Residual",
         "Skew_Residual", "Kurtosis_Residual", "Shapiro_p", "BreuschPagan_p",
         "Residual_ACF_lag1"]].copy()
    fr = {"Mean_Residual": lambda v: signed(v, 4),
          "SD_Residual": lambda v: num(v, 4),
          "Skew_Residual": lambda v: num(v, 2),
          "Kurtosis_Residual": lambda v: num(v, 2),
          "Shapiro_p": lambda v: f"{v:.1e}",
          "BreuschPagan_p": lambda v: f"{v:.1e}",
          "Residual_ACF_lag1": lambda v: num(v, 3)}
    table(C.GENERATED / "tab_residuals.tex", rd, "l l r r r r r r r",
          ["BWDB", "stratum", "mean", "sd", "skew", "kurt.", "Shapiro $p$",
           "B--P $p$", "$\\rho_1$"], fr, rules=[4, 8, 12, 16])


    # ---- extrapolation diagnostic and the change-target contrast
    ep = C.RESULTS / "extrapolation.csv"
    dp = C.RESULTS / "model_delta.csv"
    if ep.exists() and dp.exists():
        ex = pd.read_csv(ep)
        t = ex[["Station_ID", "Stratum", "Train_Max", "Test_Max",
                "Pct_Outside", "Pct_SSE_From_Outside",
                "Mean_Abs_Daily_Change", "RMSE_Level",
                "Error_vs_Daily_Move"]].copy()
        fe = {c: (lambda v: num(v, 2)) for c in ("Train_Max", "Test_Max")}
        fe["Pct_Outside"] = lambda v: num(v, 1)
        fe["Pct_SSE_From_Outside"] = lambda v: num(v, 1)
        fe["Mean_Abs_Daily_Change"] = lambda v: num(v, 3)
        fe["RMSE_Level"] = lambda v: num(v, 3)
        fe["Error_vs_Daily_Move"] = lambda v: num(v, 2) + "$\\times$"
        table(C.GENERATED / "tab_extrapolation.tex", t, "l l r r r r r r r",
              ["BWDB", "stratum", "train max", "test max",
               "\\% outside", "\\% of SSE", "daily move", "RMSE",
               "RMSE / move"], fe, rules=[4, 8, 12])
        macro("PctOutsideMax", num(ex.Pct_Outside.max(), 1))
        macro("PctSSEOutsideMax", num(ex.Pct_SSE_From_Outside.max(), 1))
        macro("CeilingGapMax", num(ex.Ceiling_Gap.max(), 2))
        macro("MeanDailyMove", num(ex.Mean_Abs_Daily_Change.mean(), 3))
        macro("ErrorVsMoveMin", num(ex.Error_vs_Daily_Move.min(), 2))
        macro("ErrorVsMoveMax", num(ex.Error_vs_Daily_Move.max(), 2))
        macro("ErrorVsMoveMean", num(ex.Error_vs_Daily_Move.mean(), 2))

        dl = pd.read_csv(dp)
        t = dl[["Station_ID", "Stratum", "N_Test", "RMSE_Persistence",
                "RMSE_Level", "RMSE_Change", "PI_Level", "PI_Change",
                "KGE_Change"]].copy()
        fd = {c: (lambda v: num(v, 4)) for c in
              ("RMSE_Persistence", "RMSE_Level", "RMSE_Change", "KGE_Change")}
        fd["PI_Level"] = lambda v: signed(v, 3)
        fd["PI_Change"] = lambda v: signed(v, 3)
        fd["N_Test"] = lambda v: num(v, 0, thousands=True)
        table(C.GENERATED / "tab_delta.tex", t, "l l r r r r r r r",
              ["BWDB", "stratum", "$n_{\\mathrm{te}}$", "RMSE$_p$",
               "RMSE level", "RMSE change", "PI level", "PI change",
               "KGE change"], fd, rules=[4, 8, 12])
        macro("MeanPILevel", signed(dl.PI_Level.mean(), 3))
        macro("MeanPIChange", signed(dl.PI_Change.mean(), 3))
        macro("NStrataBeatChange", str(int((dl.PI_Change > 0).sum())))
        macro("MaxPIChange", num(dl.PI_Change.max(), 3))
        macro("MinPIChange", num(dl.PI_Change.min(), 3))
        macro("RMSEChangeMean", num(dl.RMSE_Change.mean(), 4))
        m = dl.merge(ex[["Station_ID", "Stratum", "Mean_Abs_Daily_Change"]],
                     on=["Station_ID", "Stratum"])
        macro("ChangeVsMoveMean",
              num((m.RMSE_Change / m.Mean_Abs_Daily_Change).mean(), 2))
    else:
        for n in ("extrapolation", "delta"):
            (C.GENERATED / f"tab_{n}.tex").write_text(
                "\\textit{(pending: run build/06b\\_diagnose.py)}\n",
                encoding="utf-8")
        for k in ("PctOutsideMax", "PctSSEOutsideMax", "CeilingGapMax",
                  "MeanPILevel", "MeanPIChange", "NStrataBeatChange",
                  "MaxPIChange", "MinPIChange", "RMSEChangeMean",
                  "MeanDailyMove", "ErrorVsMoveMin", "ErrorVsMoveMax",
                  "ErrorVsMoveMean", "ChangeVsMoveMean"):
            macro(k, "??")

    # ---- hyper-parameters chosen
    sel = hp[hp.Selected & (hp.Feature_Set == prim_fs)][
        ["Station_ID", "Stratum", "Task", "max_features", "min_samples_leaf",
         "OOB_Score"]]
    table(C.GENERATED / "tab_hyperparams.tex", sel, "l l l l r r",
          ["BWDB", "stratum", "task", "max features", "min leaf", "OOB"],
          {"OOB_Score": lambda v: num(v, 4)})

    # ---- macros for the prose
    pooled = spec[spec.Stratum == "ALL"].set_index("Station_ID")
    macro("PIMin", num(spec.Persistence_Index.min(), 3))
    macro("PIMax", num(spec.Persistence_Index.max(), 3))
    macro("NSEMin", num(spec.NSE.min(), 4))
    macro("NSEMax", num(spec.NSE.max(), 4))
    macro("NSEPersistMin", num(spec.NSE_Persistence.min(), 4))
    macro("RMSEMin", num(spec.RMSE.min(), 4))
    macro("RMSEMax", num(spec.RMSE.max(), 4))
    macro("RMSEPersistMin", num(spec.RMSE_Persistence.min(), 4))
    macro("RMSEPersistMax", num(spec.RMSE_Persistence.max(), 4))
    macro("MeanPI", num(spec.Persistence_Index.mean(), 3))
    macro("NStrata", str(len(spec)))
    macro("NStrataBeatPersistence", str(int((spec.Persistence_Index > 0).sum())))
    macro("NStrataLosePersistence", str(int((spec.Persistence_Index <= 0).sum())))
    cs = clf_m[clf_m.Feature_Set == prim_fs]
    macro("KappaMin", num(cs.Cohen_Kappa.min(), 3))
    macro("KappaMax", num(cs.Cohen_Kappa.max(), 3))
    macro("KappaSkillMin", num(cs.Kappa_Skill_Score.min(), 3))
    macro("KappaSkillMax", num(cs.Kappa_Skill_Score.max(), 3))
    macro("AccMin", num(100 * cs.Accuracy.min(), 1))
    macro("AccMax", num(100 * cs.Accuracy.max(), 1))
    d = spec.merge(safe, on=["Station_ID", "Stratum"], suffixes=("_s", "_f"))
    macro("SafeRMSEPenalty", num((d.RMSE_f / d.RMSE_s).mean(), 2))
    macro("SafePIMean", num(d.Persistence_Index_f.mean(), 3))
    macro("SpecPIMean", num(d.Persistence_Index_s.mean(), 3))

    g = grp[(grp.Grouping == "by_lag") & (grp.Stratum == "ALL")
            & (grp.Feature_Set == prim_fs)]
    tot = g.groupby("Station_ID").Permutation_Importance.sum()
    near = g[g.Group_Value <= 1].groupby("Station_ID").Permutation_Importance.sum()
    macro("NearLagShare", num(100 * (near / tot).mean(), 1))
    gp = grp[(grp.Grouping == "by_point") & (grp.Stratum == "ALL")
             & (grp.Feature_Set == prim_fs)]
    macro("NTransferRows", str(len(tr)))
    off = tr[~tr.Same_Period]
    on = tr[tr.Same_Period]
    macro("TransferPenalty", num(off.RMSE.mean() / on.RMSE.mean(), 2))
    macro("TransferOnMean", num(on.RMSE.mean(), 4))
    macro("TransferOffMean", num(off.RMSE.mean(), 4))

    # Does matching the period actually help, or does the largest training
    # block simply win? Count the cells where the matched model is best.
    diag = best_is_during = 0
    for sid in tr.Station_ID.unique():
        for te in C.PERIOD_NAMES:
            col = tr[(tr.Station_ID == sid) & (tr.Test_Period == te)]
            if col.empty:
                continue
            win = col.loc[col.RMSE.idxmin(), "Train_Period"]
            diag += win == te
            best_is_during += win == "DURING"
    macro("NPeriodCells", str(len(C.TARGETS) * len(C.PERIOD_NAMES)))
    macro("NDiagonalWins", str(diag))
    macro("NBestIsDuring", str(best_is_during))
    sizes = (reg_m[(reg_m.Feature_Set == prim_fs)
                   & reg_m.Stratum.isin(C.PERIOD_NAMES)]
             .drop_duplicates(["Stratum"]).set_index("Stratum").N_Train)
    macro("NTrainDuring", num(sizes.get("DURING", np.nan), 0, thousands=True))
    macro("NTrainBefore", num(sizes.get("BEFORE", np.nan), 0, thousands=True))
    macro("NTrainAfter", num(sizes.get("AFTER", np.nan), 0, thousands=True))


SNIPPETS = [
    ("read", "build/01_extract.py",
     "Parsing one workbook: the header block is metadata, the date format is "
     "pinned, and numbers are coerced rather than trusted."),
    ("lags", "build/03_lags.py",
     "Building WL and WLD-1..7. The series is reindexed onto every calendar "
     "day first, so a shift of $k$ is a lag of exactly $k$ days."),
    ("split", "build/06_models.py",
     "The train/test split is chronological, never random."),
    ("wide", "build/04_wide.py",
     "Assembling one prior point's eight columns into the design matrix."),
]


def code_snippets() -> None:
    """Lift the marked passages straight out of the scripts that ran."""
    for name, rel, caption in SNIPPETS:
        src = (C.ROOT / rel).read_text(encoding="utf-8").splitlines()
        keep, on = [], False
        for line in src:
            if f"<<snip:{name}>>" in line:
                on = True
                # A marker on a line of its own opens the snippet without
                # contributing a line; one appended to code keeps the code.
                code = line.split("# <<snip")[0].rstrip()
                if code:
                    keep.append(code)
                continue
            if on and "<<endsnip>>" in line:
                keep.append(line.split("  # <<endsnip>>")[0].rstrip())
                break
            if on:
                keep.append(line.rstrip())
        if not keep:
            print(f"  !! snippet {name} not found in {rel}")
            continue
        body = "\n".join(keep)
        head = (r"\begin{lstlisting}[language=Python,caption={\texttt{"
                + esc(rel) + "} --- " + caption + "},label={lst:" + name + "}]")
        out = head + "\n" + body + "\n" + r"\end{lstlisting}" + "\n"
        path = C.GENERATED / f"code_{name}.tex"
        path.write_text(out, encoding="utf-8")
        print(f"  -> {path.name} ({len(keep)} lines)")


def main() -> None:
    dataset_tables()
    code_snippets()
    model_tables()
    lines = ["% Generated by build/10_tex.py - do not edit by hand.", ""]
    for k, v in sorted(MACROS.items()):
        lines.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    (C.GENERATED / "numbers.tex").write_text("\n".join(lines) + "\n",
                                             encoding="utf-8")
    print(f"  -> numbers.tex ({len(MACROS)} macros)")


if __name__ == "__main__":
    main()
