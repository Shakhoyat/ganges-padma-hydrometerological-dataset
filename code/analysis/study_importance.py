"""Study I4: which gauges does a 3-day flood warning depend on? (permutation importance)

Random forest, warning-or-worse at h = 3 days, trained 2011-2021, scored on
2022-2025. Features are permuted as whole gauge groups (margin + 1/3/7-day
changes together), 20 repeats, so the importance answers a maintenance
question: how much warning skill is lost if this gauge's feed goes dark?
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

import common as cm
from study_flood_risk import PREDICTORS, feature_table

H = 3
N_REPEATS = 20


def group_importance(tgt: int, stage: pd.DataFrame, danger: pd.Series) -> pd.DataFrame:
    feats = feature_table(stage, danger, PREDICTORS[tgt])
    margin = stage[tgt] - danger[tgt]
    y = pd.Series(cm.risk_class(margin.shift(-H)), index=stage.index) >= 1
    ok = margin.shift(-H).notna() & margin.notna()
    label_year = (stage.index + pd.Timedelta(days=H)).year
    tr, te = ok & (label_year <= 2021), ok & (label_year >= 2022)
    model = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample",
                                   n_jobs=-1, random_state=cm.SEED).fit(feats[tr], y[tr])
    x_te, y_te = feats[te].copy(), y[te].to_numpy()
    base = f1_score(y_te, model.predict(x_te))
    rng = np.random.default_rng(cm.SEED)
    rows = []
    groups = {g: [c for c in feats if c == f"m{g}" or c.endswith(f"_{g}")] for g in PREDICTORS[tgt]}
    groups["season"] = ["doy_sin", "doy_cos"]
    for g, cols in groups.items():
        drops = []
        for _ in range(N_REPEATS):
            xp = x_te.copy()
            perm = rng.permutation(len(xp))
            xp[cols] = xp[cols].to_numpy()[perm]
            drops.append(base - f1_score(y_te, model.predict(xp)))
        rows.append({"target": tgt, "group": str(g), "drop_mean": np.mean(drops),
                     "drop_sd": np.std(drops), "base_f1": base})
    return pd.DataFrame(rows)


def main() -> None:
    danger = cm.load_registry()["Danger_Level_mMSL"]
    stage = cm.stage_matrix(cm.load_long())
    out = pd.concat([group_importance(t, stage, danger) for t in cm.TARGETS])
    out.to_csv(cm.RES_DIR / "importance_groups.csv", index=False)
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()
