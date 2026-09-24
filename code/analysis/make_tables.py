"""LaTeX table rows generated from the study CSVs (so text and tables cannot drift)."""
from __future__ import annotations

import pandas as pd

import common as cm

END = r" \\ \hline"


def risk_composition() -> None:
    long = cm.load_long()
    reg = cm.load_registry()
    cls = cm.risk_class(long.WL - long.Id.map(reg.Danger_Level_mMSL))
    tab = pd.crosstab(long.Id, cls)

    def line(name: str, r: pd.Series) -> str:
        n = r.sum()
        cells = " & ".join(f"{int(v):,} ({100 * v / n:.1f}\\%)" for v in r)
        return f"{name} & {cells} & {int(n):,}{END}"

    rows = [line(f"{cm.SHORT_NAME[s]} ({s})", tab.loc[s]) for s in cm.TARGETS]
    rows.append(line("All 17 gauges", tab.sum()))
    (cm.GEN_DIR / "tab_risk_composition.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")


def upstream_alarms() -> None:
    a = pd.read_csv(cm.RES_DIR / "upstream_alarms.csv")
    rows = []
    for _, r in a.iterrows():
        if r.n_alarms == 0:
            continue
        lead = "--" if pd.isna(r.lead_median) else f"{r.lead_median:.1f} ({r.lead_q25:.0f}--{r.lead_q75:.0f})"
        up = int(r.upstream)
        rows.append(f"{cm.TARGETS[int(r.target)]} & {cm.SHORT_NAME[up]} ({up}) & "
                    f"{int(r.n_alarms)} & {int(r.n_followed)} ({100 * r.hit_rate:.0f}\\%) & {lead}{END}")
    (cm.GEN_DIR / "tab_upstream_alarms.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")


def transfer() -> None:
    t = pd.read_csv(cm.RES_DIR / "bridge_transfer.csv")
    rows = []
    for tgt in cm.TARGETS:
        for tr in ["Before", "During", "After"]:
            s = t[(t.target == tgt) & (t.train == tr)].set_index("test")
            cells = " & ".join((f"\\textbf{{{s.at[e, 'PI']:.2f}}}" if e == tr else f"{s.at[e, 'PI']:.2f}")
                               for e in ["Before", "During", "After"])
            name = cm.TARGETS[tgt] if tr == "Before" else ""
            rows.append(f"{name} & {tr} & {cells}{END if tr == 'After' else r' \\'}")
    (cm.GEN_DIR / "tab_transfer.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")


def risk_summary() -> None:
    """Warning-or-worse CSI / POD / FAR at 1, 3, 7 days: persistence vs. the pre-declared
    primary classifier (random forest) and logistic regression. No model is chosen on test data."""
    s = pd.read_csv(cm.RES_DIR / "risk_scores.csv")
    rows = []
    for tgt in cm.TARGETS:
        for h in (1, 3, 7):
            g = s[(s.target == tgt) & (s.h == h)].set_index("model")
            p, rf, lr = g.loc["Persistence"], g.loc["Random forest"], g.loc["Logistic regression"]
            name = f"{cm.TARGETS[tgt]} ({tgt})" if h == 1 else ""
            trip = lambda r: f"{r.CSI_W:.2f} / {r.POD_W:.2f} / {r.FAR_W:.2f}"  # noqa: E731
            rows.append(f"{name} & {h} & {trip(p)} & {trip(rf)} & {trip(lr)} & "
                        f"{rf.macroF1:.2f} ({p.macroF1:.2f}){END}")
    (cm.GEN_DIR / "tab_risk_summary.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    risk_composition()
    upstream_alarms()
    transfer()
    risk_summary()
    print("tables written")


if __name__ == "__main__":
    main()
