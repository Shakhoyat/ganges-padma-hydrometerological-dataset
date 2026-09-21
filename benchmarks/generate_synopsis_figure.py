"""Generate high-resolution Black & White Figure 1 matching user reference layout exactly.
(a) Framework illustrating the practical application of the data (5 boxes)
(b) Technical pipeline of the ML method using the dataset (8 boxes in 2 rows snake-flow)
"""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral", "serif"],
    "mathtext.fontset": "stix",
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

ROOT = Path("e:/4-1/2k21/Labs/ML-Lab")
FIGURES_DIRS = [
    ROOT / "Sir-Task-1" / "figures",
    ROOT / "figures",
    ROOT / "padma-corridor" / "figures",
    ROOT / "report-final" / "figures",
]

def generate_synopsis_figure():
    fig = plt.figure(figsize=(7.6, 5.0), facecolor="white")
    
    # -------------------------------------------------------------------------
    # (a) Practical Application Framework
    # -------------------------------------------------------------------------
    ax1 = fig.add_axes([0.005, 0.58, 0.99, 0.39])
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 36)
    ax1.axis("off")
    
    # Section header
    ax1.text(0, 34.2, r"$\mathbf{(a)\; Framework\; illustrating\; the\; practical\; application\; of\; the\; data}$",
             fontsize=10.5, fontweight="bold", ha="left", va="top", color="black")
    
    boxes_a = [
        (0.2, 17.2, "Domain Problem", "Manual / legacy stage\nforecasting, backwater afflux\n& Padma scour risks"),
        (20.4, 17.2, "Data Acquisition", "15-year BWDB records\n(2011–2025; 17 stations,\n5,479 calendar days)"),
        (40.6, 17.2, "Curated Dataset", "31-var panel, 7-day stage\n& rain lags, 4-class flood\nrisk (90.7k cases)"),
        (60.8, 17.2, "ML / Analytics", r"$\Delta_t$" + " Delta formulation,\nLSTM, LightGBM,\nPhysics-Loss PI-STGNN"),
        (81.0, 18.8, "Practical Impact", "1–7 day stage forecasts,\nautomated gate control &\nFFWC flood alerts"),
    ]
    
    for x, w, header, text in boxes_a:
        patch = FancyBboxPatch((x, 9.5), w, 19.5, boxstyle="round,pad=0.2,rounding_size=0.8",
                               ec="black", fc="#ffffff", lw=0.95)
        ax1.add_patch(patch)
        ax1.text(x + w/2, 24.8, header, ha="center", va="center", fontweight="bold", fontsize=8.8, color="black")
        ax1.text(x + w/2, 16.0, text, ha="center", va="center", fontsize=7.8, color="black", linespacing=1.2)
    
    for (x1, w1, _, _), (x2, _, _, _) in zip(boxes_a[:-1], boxes_a[1:]):
        arr = FancyArrowPatch((x1 + w1, 19.25), (x2, 19.25),
                              arrowstyle="-|>,head_width=3.0,head_length=4.8", color="black", lw=0.95)
        ax1.add_patch(arr)
    
    caption_a_custom = "Figure 1(a). Dataset-specific practical application framework illustrating end-to-end impact from hydrometric collection\nto automated river corridor warning and bridge operations."
    ax1.text(0, 5.2, caption_a_custom, fontsize=8.6, ha="left", va="top", color="black", linespacing=1.2)
    
    # -------------------------------------------------------------------------
    # (b) Technical ML Pipeline
    # -------------------------------------------------------------------------
    ax2 = fig.add_axes([0.005, 0.01, 0.99, 0.54])
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 52)
    ax2.axis("off")
    
    ax2.text(0, 50.8, r"$\mathbf{(b)\; Technical\; pipeline\; of\; the\; ML\; method\; using\; the\; dataset}$",
             fontsize=10.5, fontweight="bold", ha="left", va="top", color="black")
    
    row1 = [
        (0.2, 22.0, "Raw Hydromet Records", "50 official BWDB sheets\n(WL, Rain, Q, Evap, GW)"),
        (25.4, 22.0, "Quality Checks", "Calendar reindexing, zero-\npad removal, outlier audit"),
        (50.6, 22.0, "Preprocessing", "7-day lag embedding,\n" + r"$\Delta_t$" + " delta, flood classes"),
        (75.8, 22.0, "Leakage-Safe Split", "Pre / During / Post Bridge\n80% Train / 20% Test"),
    ]
    
    row2 = [
        (75.8, 22.0, "Feature Stratification", "Wide matrices (" + r"$\leq 245$" + " feats),\ntributary Q, tidal indicators"),
        (50.6, 22.0, "Candidate Models", "Persistence, HBV, LSTM,\nGBDT, PI-STGNN"),
        (25.4, 22.0, "Evaluation Metrics", "RMSE, MAE, NSE, " + r"$R^2$" + ",\nPI > 0, Mass/Mom Error"),
        (0.2, 22.0, "Interpretation / App", "SHAP feature rankings,\noperational FFWC alerts"),
    ]
    
    for x, w, header, text in row1:
        patch = FancyBboxPatch((x, 30.2), w, 15.8, boxstyle="round,pad=0.2,rounding_size=0.8",
                               ec="black", fc="#ffffff", lw=0.95)
        ax2.add_patch(patch)
        ax2.text(x + w/2, 41.8, header, ha="center", va="center", fontweight="bold", fontsize=8.8, color="black")
        ax2.text(x + w/2, 35.2, text, ha="center", va="center", fontsize=7.8, color="black", linespacing=1.2)
    
    for (x1, w1, _, _), (x2, _, _, _) in zip(row1[:-1], row1[1:]):
        arr = FancyArrowPatch((x1 + w1, 38.1), (x2, 38.1),
                              arrowstyle="-|>,head_width=3.0,head_length=4.8", color="black", lw=0.95)
        ax2.add_patch(arr)
    
    arr_down = FancyArrowPatch((75.8 + 22.0/2, 30.2), (75.8 + 22.0/2, 23.8),
                               arrowstyle="-|>,head_width=3.0,head_length=4.8", color="black", lw=0.95)
    ax2.add_patch(arr_down)
    
    for x, w, header, text in row2:
        patch = FancyBboxPatch((x, 8.0), w, 15.8, boxstyle="round,pad=0.2,rounding_size=0.8",
                               ec="black", fc="#ffffff", lw=0.95)
        ax2.add_patch(patch)
        ax2.text(x + w/2, 19.6, header, ha="center", va="center", fontweight="bold", fontsize=8.8, color="black")
        ax2.text(x + w/2, 13.0, text, ha="center", va="center", fontsize=7.8, color="black", linespacing=1.2)
    
    for (x1, _, _, _), (x2, w2, _, _) in zip(row2[:-1], row2[1:]):
        arr = FancyArrowPatch((x1, 15.9), (x2 + w2, 15.9),
                              arrowstyle="-|>,head_width=3.0,head_length=4.8", color="black", lw=0.95)
        ax2.add_patch(arr)
    
    caption_b_custom = "Figure 1(b). Exact technical pipeline showing leakage-safe partitioning, multivariate lag matrix generation, physics-guided\narchitectures, and persistence-benchmarked evaluation."
    ax2.text(0, 4.0, caption_b_custom, fontsize=8.6, ha="left", va="top", color="black", linespacing=1.2)
    
    for d in FIGURES_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / "fig_synopsis_framework.pdf", bbox_inches="tight", pad_inches=0.02)
        fig.savefig(d / "fig_synopsis_framework.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("[OK] Exact reference-styled Figure 1 generated successfully.")

if __name__ == "__main__":
    generate_synopsis_figure()
