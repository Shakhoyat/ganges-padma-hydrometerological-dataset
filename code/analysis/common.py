"""Shared paths, data loading, figure style and output helpers for report-v4.

Every study in this folder reads only the released dataset files (long panel,
wide matrices, station registry) plus one external reference: the BWDB danger
level of each gauge. Nothing is trained on a GPU; all models are scikit-learn
on CPU with a fixed seed.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]  # repository root
DATASET = ROOT
LONG_CSV = DATASET / "processed" / "ganges_padma_hydromet_dataset_2011_2025.csv"
WIDE_DIR = DATASET / "processed" / "wide"
REGISTRY_CSV = DATASET / "metadata" / "station_registry.csv"
DANGER_CSV = DATASET / "metadata" / "danger_levels.csv"  # external reference, not a model input

OUT = ROOT / "results" / "analysis"
FIG_DIR = OUT / "figures"
GEN_DIR = OUT / "tables"
RES_DIR = OUT / "csv"
for _d in (FIG_DIR, GEN_DIR, RES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 20260913
TRAIN_END = pd.Timestamp("2021-12-31")
VAL_YEAR = 2022
TEST_START = pd.Timestamp("2023-01-01")

TARGETS = {11: "Baruria", 12: "Bhagyakul", 15: "Sureswar"}
WIDE_FILES = {11: "wide_sw91_9l_baruria_transit.csv",
              12: "wide_sw93_4l_bhagyakul.csv",
              15: "wide_sw95_sureswar.csv"}

SHORT_NAME = {1: "Panka", 2: "Rajshahi", 3: "Sardah", 4: "Hardinge Br.", 5: "Talbaria",
              6: "Sengram", 7: "Mohendrapur", 8: "Bahadurabad", 9: "Aricha",
              10: "Goalundo", 11: "Baruria", 12: "Bhagyakul", 13: "Mawa", 14: "Tarpasa",
              15: "Sureswar", 16: "Bhairab Bazar", 17: "Chandpur"}

# Risk classes relative to the BWDB danger level DL (backup_v2 definition).
RISK_LABELS = ["Normal", "Warning", "Danger", "Severe"]
RISK_EDGES = [-np.inf, -1.0, 0.0, 0.5, np.inf]  # WL - DL, metres

# Okabe-Ito, colour-blind safe; fixed meaning across every v4 figure.
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "pink": "#CC79A7",
     "sky": "#56B4E9", "red": "#D55E00", "yellow": "#F0E442", "grey": "#7f7f7f",
     "ink": "#1a1a1a", "light": "#d9d9d9"}
RISK_COLORS = ["#9ecae1", "#fdd49e", "#fc8d59", "#b30000"]
MODEL_COLORS = {"Persistence": C["grey"], "Logistic regression": C["blue"],
                "Random forest": C["green"], "Gradient boosting": C["orange"],
                "Ridge": C["blue"]}

CM = 1 / 2.54
FULL_W = 17.0 * CM  # \linewidth of the A4 report with 20 mm margins
LAND_W = 25.0 * CM  # \linewidth of a landscape A4 page (257 mm), minus a small safety margin


def use_style() -> None:
    """Journal style: Arial (template rule for figures), 9 pt at print size."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.5,
        "axes.axisbelow": True, "legend.frameon": False,
        "figure.dpi": 150, "savefig.dpi": 300, "pdf.fonttype": 42,
        "axes.titleweight": "bold", "axes.titlelocation": "left",
    })


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(FIG_DIR / f"{stem}.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  wrote {stem}")


def panel_label(ax: plt.Axes, text: str) -> None:
    ax.set_title(text, loc="left", fontweight="bold")


def load_registry() -> pd.DataFrame:
    reg = pd.read_csv(REGISTRY_CSV).set_index("Id")
    danger = pd.read_csv(DANGER_CSV).set_index("Id")[["Danger_Level_mMSL", "Danger_Level_Source"]]
    return reg.join(danger)


def load_long() -> pd.DataFrame:
    df = pd.read_csv(LONG_CSV, parse_dates=["Date"],
                     dtype={"Rainfall_Level": "Int64", "WL_Trend": "Int64"})
    return df.sort_values(["Id", "Date"]).reset_index(drop=True)


def stage_matrix(long: pd.DataFrame) -> pd.DataFrame:
    """Date x station daily stage on the full calendar (gaps stay NaN)."""
    wl = long.pivot(index="Date", columns="Id", values="WL")
    full = pd.date_range("2011-01-01", "2025-12-31", freq="D")
    return wl.reindex(full)


def load_wide(target: int) -> pd.DataFrame:
    w = pd.read_csv(WIDE_DIR / WIDE_FILES[target], parse_dates=["Date"])
    return w.sort_values("Date").reset_index(drop=True)  # chronological splits rely on order


def risk_class(margin: pd.Series | np.ndarray) -> np.ndarray:
    """0 Normal, 1 Warning, 2 Danger, 3 Severe; NaN where margin is NaN."""
    m = np.asarray(margin, dtype=float)
    out = np.digitize(m, RISK_EDGES[1:-1], right=False).astype(float)
    out[np.isnan(m)] = np.nan
    return out


def persistence_index(y: np.ndarray, yhat: np.ndarray, y_prev: np.ndarray) -> float:
    return float(1 - np.sum((y - yhat) ** 2) / np.sum((y - y_prev) ** 2))


class Macros:
    """Collects \\newcommand macros so every number in the text traces to code."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.items: dict[str, str] = {}

    def add(self, name: str, value: object) -> None:
        if not name.isalpha():
            raise ValueError(f"macro name must be letters only: {name}")
        self.items[name] = str(value)

    def write(self) -> None:
        lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in self.items.items()]
        self.path.write_text("% generated by report-v4 -- do not edit\n" + "\n".join(lines) + "\n",
                             encoding="utf-8")
        print(f"  wrote {self.path.name} ({len(lines)} macros)")


def fmt(x: float, d: int = 2) -> str:
    return f"{x:.{d}f}"
