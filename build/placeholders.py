"""Stand-in figures for anything report.tex includes that has not been built yet.

Lets the document compile mid-build so LaTeX errors surface early. Real figures
overwrite these; nothing here survives a full build.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
import corridor as C  # noqa: E402


def main() -> None:
    tex = (C.ROOT / "report" / "report.tex").read_text(encoding="utf-8")
    wanted = set(re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", tex))
    made = []
    for name in sorted(wanted):
        p = C.FIGURES / name
        if p.exists():
            continue
        fig, ax = plt.subplots(figsize=(7.2, 2.0))
        ax.axis("off")
        ax.text(0.5, 0.5, f"[ {name} — pending ]", ha="center", va="center",
                fontsize=11, color="#aaaaaa", family="serif")
        fig.savefig(p)
        plt.close(fig)
        made.append(name)
    print(f"  placeholders: {', '.join(made) if made else 'none needed'}")


if __name__ == "__main__":
    main()
