"""Step 07 - the station map: 17 corridor ids on Bangladesh, with their gauges.

Left panel   national outline and divisions, the four river branches drawn
             separately, every gauge marked with its corridor id, and an inset
             enlarging the confluence-to-bridge reach where eight of the
             seventeen gauges sit within 80 km of one another.
Right panel  the same ids resolved to BWDB station, name, river and chainage,
             so the numbering used throughout the report reads off the figure.

Boundaries are the geoBoundaries open release (ADM0 and ADM1), cached under
data/interim. Geometry is drawn straight from the GeoJSON coordinate arrays, so
no GIS stack is required.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
import corridor as C  # noqa: E402

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.linewidth": 0.6,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})

INK = "#1a1a1a"
MUTED = "#6b6558"
LAND = "#f2f0ea"
EDGE = "#b8b2a4"
WATER = "#3d6b99"
C_MAIN = "#4a6f8a"
C_BOUND = "#8a8f98"
C_TARGET = "#c1462f"
C_BRIDGE = "#1f7a4d"

# Four branches, drawn separately. A single polyline through id order would run
# Mohendrapur -> Bahadurabad -> Aricha and draw a river that does not exist:
# ids 8-9 sit on the Jamuna, some 200 km to the north.
BRANCHES = [
    (["SW88A", "SW88", "SW89", "SW90", "SW91", "SW91.1", "SW91.2", "SW91.9R"], 1.7),
    (["SW46.9L", "SW50.6", "SW91.9R"], 1.3),
    (["SW91.9R", "SW91.9L", "SW93.4L", "SW93.5L", "SW94", "SW95", "SW277"], 1.9),
    (["SW273", "SW277"], 1.3),
]

# Label offsets in points, for the gauges that would otherwise collide.
INSET_OFFSETS = {
    "SW50.6": (0, 9), "SW91.9R": (-9, -3), "SW91.9L": (9, 5),
    "SW93.4L": (-7, 7), "SW93.5L": (2, -12), "SW94": (11, 4),
    "SW95": (10, -3),
}
NATIONAL_LABELS = {
    "SW88A": (7, 0, "left"), "SW46.9L": (-8, 0, "right"),
    "SW273": (8, 3, "left"), "SW277": (5, -9, "left"),
}


def rings(geom: dict) -> list:
    """Every exterior ring in a GeoJSON geometry, whatever its type."""
    t = geom["type"]
    if t == "Polygon":
        return [geom["coordinates"][0]]
    if t == "MultiPolygon":
        return [poly[0] for poly in geom["coordinates"]]
    if t == "GeometryCollection":
        return [r for g in geom["geometries"] for r in rings(g)]
    return []


def draw_boundaries(ax, division_labels: bool = True) -> None:
    adm1 = json.loads((C.INTERIM / "bgd_adm1.geojson").read_text(encoding="utf-8"))
    adm0 = json.loads((C.INTERIM / "bgd_adm0.geojson").read_text(encoding="utf-8"))
    for f in adm1["features"]:
        for ring in rings(f["geometry"]):
            ax.add_patch(Polygon(ring, closed=True, facecolor=LAND,
                                 edgecolor=EDGE, linewidth=0.45, zorder=1))
    for f in adm0["features"]:
        for ring in rings(f["geometry"]):
            ax.add_patch(Polygon(ring, closed=True, facecolor="none",
                                 edgecolor="#5d5749", linewidth=0.9, zorder=4))
    if not division_labels:
        # Text is not clipped to the axes, so an inset that reused these would
        # scatter division names across the whole figure.
        return
    for f in adm1["features"]:
        pts = [p for r in rings(f["geometry"]) for p in r]
        if pts:
            ax.text(sum(p[0] for p in pts) / len(pts),
                    sum(p[1] for p in pts) / len(pts),
                    f["properties"].get("shapeName", "").upper(),
                    fontsize=5.2, color="#9a9384", ha="center", va="center",
                    zorder=2, style="italic")


def style_of(r) -> tuple[str, float, float]:
    if r.Is_Target:
        return C_TARGET, 62, 1.1
    if r.Station_ID == "SW93.5L":
        return C_BRIDGE, 62, 1.1
    if str(r.Reach) == "main_stem":
        return C_MAIN, 44, 0.7
    return C_BOUND, 44, 0.7


def main() -> None:
    reg = pd.read_csv(C.DATA / "station_registry.csv").sort_values("Id")
    P = reg.set_index("Station_ID")

    fig = plt.figure(figsize=(10.6, 6.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.30, 1.0], wspace=0.05)
    ax = fig.add_subplot(gs[0, 0])
    tx = fig.add_subplot(gs[0, 1])
    tx.axis("off")

    def draw_rivers(a, extra_lw: float = 0.0) -> None:
        for ids, lw in BRANCHES:
            sub = P.loc[ids]
            a.plot(sub.Longitude, sub.Latitude, "-", color=WATER,
                   lw=lw + extra_lw, alpha=0.5, zorder=5, solid_capstyle="round")

    def draw_stations(a, scale: float = 1.0, fs: float = 5.0,
                      offsets: dict | None = None, name_fs: float = 6.0) -> None:
        for _, r in reg.iterrows():
            col, size, edge = style_of(r)
            a.scatter(r.Longitude, r.Latitude, s=size * scale, c=col, marker="o",
                      edgecolors="white", linewidths=edge, zorder=7)
            a.annotate(str(int(r.Id)), (r.Longitude, r.Latitude), color="white",
                       fontsize=fs, fontweight="bold", ha="center", va="center",
                       zorder=8)
            if offsets and r.Station_ID in offsets:
                dx, dy = offsets[r.Station_ID][:2]
                ha = offsets[r.Station_ID][2] if len(offsets[r.Station_ID]) > 2 else (
                    "left" if dx > 0 else "right" if dx < 0 else "center")
                a.annotate(C.display_name(r.Station_Name),
                           (r.Longitude, r.Latitude), textcoords="offset points",
                           xytext=(dx, dy), fontsize=name_fs, color="#3f3f3f",
                           ha=ha, va="center", zorder=9)

    # ----------------------------------------------------------- national map
    draw_boundaries(ax)
    draw_rivers(ax)
    draw_stations(ax, offsets=NATIONAL_LABELS, name_fs=5.8)

    # ------------------------------------------------- inset over the reach
    ins = ax.inset_axes([0.530, 0.555, 0.470, 0.400])
    draw_boundaries(ins, division_labels=False)
    draw_rivers(ins, extra_lw=0.7)
    draw_stations(ins, scale=1.15, fs=5.2, offsets=INSET_OFFSETS, name_fs=5.6)
    ins.scatter(90.253, 23.475, s=190, facecolors="none", edgecolors=C_BRIDGE,
                linewidths=1.0, zorder=6)
    ins.annotate("Padma Bridge", (90.253, 23.475), textcoords="offset points",
                 xytext=(-6, -22), fontsize=6.2, color=C_BRIDGE,
                 fontweight="bold", ha="right", zorder=9)
    ins.set_xlim(89.60, 90.72)
    ins.set_ylim(23.16, 23.97)
    ins.set_aspect(1.0 / 0.92)
    ins.set_xticks([])
    ins.set_yticks([])
    ins.set_facecolor("#fbfaf7")
    for sp in ins.spines.values():
        sp.set_color("#8a8478")
        sp.set_linewidth(0.8)
    ins.text(0.022, 0.962, "Confluence to bridge reach", transform=ins.transAxes,
             fontsize=5.8, color=MUTED, va="top", style="italic")
    ax.indicate_inset_zoom(ins, edgecolor="#8a8478", linewidth=0.7, alpha=0.9)

    ax.set_xlim(87.9, 92.8)
    ax.set_ylim(20.5, 26.8)
    ax.set_aspect(1.0 / 0.92)
    ax.set_xlabel("Longitude (°E)", fontsize=7)
    ax.set_ylabel("Latitude (°N)", fontsize=7)
    ax.tick_params(labelsize=6, length=2.5, width=0.5)
    for s in ax.spines.values():
        s.set_color("#c8c2b4")
    ax.set_title("Gauge network, numbered in corridor order", fontsize=8.5,
                 color=INK, pad=6, loc="left")
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", mfc=C_MAIN, mec="white", ms=6,
               label="Ganges–Padma main stem"),
        Line2D([], [], marker="o", ls="", mfc=C_BOUND, mec="white", ms=6,
               label="Boundary inflow / outflow"),
        Line2D([], [], marker="o", ls="", mfc=C_TARGET, mec="white", ms=7,
               label="Prediction target"),
        Line2D([], [], marker="o", ls="", mfc=C_BRIDGE, mec="white", ms=7,
               label="Padma Bridge (Mawa)"),
    ], loc="lower left", fontsize=6, frameon=True, framealpha=0.94,
        edgecolor="#c8c2b4", borderpad=0.6)

    # ----------------------------------------------------------- the id table
    tx.set_xlim(0, 1)
    tx.set_ylim(0, 1)
    tx.text(0.0, 0.978, "Corridor id → gauge", fontsize=8.5, color=INK, va="top")

    cols = [(0.000, "id"), (0.072, "BWDB"), (0.250, "Station name"),
            (0.530, "River"), (0.752, "km"), (0.838, "Role")]
    y = 0.918
    for x, h in cols:
        tx.text(x, y, h, fontsize=6.4, color=MUTED, va="top", fontweight="bold")
    tx.plot([0, 1], [y - 0.020, y - 0.020], color="#c8c2b4", lw=0.6)

    y -= 0.045
    for _, r in reg.iterrows():
        if r.Is_Target:
            col, weight = C_TARGET, "bold"
        elif r.Station_ID == "SW93.5L":
            col, weight = C_BRIDGE, "bold"
        else:
            col, weight = INK, "normal"
        role = ("target" if r.Is_Target else
                "bridge" if r.Station_ID == "SW93.5L" else
                "held out" if r.Station_ID in C.EXCLUDED_PREDICTORS else
                "main stem" if str(r.Reach) == "main_stem" else "boundary")
        vals = [str(int(r.Id)), r.Station_ID, C.display_name(r.Station_Name),
                r.River, f"{r.Chainage_km:.0f}" if pd.notna(r.Chainage_km) else "–",
                role]
        for (x, _), v in zip(cols, vals):
            tx.text(x, y, v, fontsize=6.2, color=col, va="top", fontweight=weight)
        y -= 0.0495

    tx.plot([0, 1], [y + 0.030, y + 0.030], color="#c8c2b4", lw=0.6)
    tx.text(0.0, y + 0.010,
            "Ids run downstream. Ids 8–9 (Jamuna) and 16 (Meghna) are boundary\n"
            "inflows, inserted at the confluence through which their water enters;\n"
            "for those three alone the lower ids are not upstream. Id 14 keeps its\n"
            "number but is held out of every predictor chain — it is missing\n"
            "2016-11-01 to 2020-07-31, inside the construction period.",
            fontsize=5.6, color=MUTED, va="top", linespacing=1.55)

    fig.savefig(C.FIGURES / "fig_map.pdf")
    fig.savefig(C.FIGURES / "fig_map.png", dpi=200)
    plt.close(fig)
    print(f"  -> {C.FIGURES / 'fig_map.pdf'}")


if __name__ == "__main__":
    main()
