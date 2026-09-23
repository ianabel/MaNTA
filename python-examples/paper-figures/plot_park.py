"""fig:park -- cost/accuracy frontier on Park's problem.

Reads park_frontier.json and park_frontier_ptc.json (written by park_fig.py and
park_fig_ptc.py, which drive python-examples/park-convergence/benchmark.py's own
solve()) and writes paper/figures/park-frontier.pdf.
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).parent
DEST = (pathlib.Path(__file__).resolve().parents[3]
        / "paper" / "figures" / "park-frontier.pdf")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.0,
    "lines.markersize": 4.2,
})

tm = [r for r in json.loads((HERE / "park_frontier.json").read_text())["manta"]
      if r.get("error") is not None]
ptc = [r for r in json.loads((HERE / "park_frontier_ptc.json").read_text())["manta_ptc"]
       if r.get("error") is not None]
fd = json.loads((HERE / "park_frontier.json").read_text())["fd"]


def frontier(rows):
    out, best = [], float("inf")
    for r in sorted(rows, key=lambda r: r["nflux"]):
        if r["error"] < best:
            best, _ = r["error"], out.append(r)
    return out


fig, ax = plt.subplots(figsize=(5.6, 3.8))

# --- reference schemes ----------------------------------------------------
ax.plot([r["N"] for r in fd], [r["error"] for r in fd],
        color="0.45", marker="s", markersize=3.2, linestyle="--",
        label="2nd-order FD, Park Eq. (17)")
ax.plot([11], [6.0e-5], color="0.1", marker="*", markersize=11,
        linestyle="none", label=r"Park IDO, $N = 11$ (proxy)")

# --- MaNTA, steady-solved, one line per mesh ------------------------------
cells = sorted({r["cells"] for r in ptc})
colours = plt.get_cmap("viridis")([0.04, 0.34, 0.58, 0.80])
markers = ["o", "^", "D", "v"]
for colour, marker, nc in zip(colours, markers, cells):
    sel = sorted((r for r in ptc if r["cells"] == nc), key=lambda r: r["nflux"])
    ax.plot([r["nflux"] for r in sel], [r["error"] for r in sel],
            color=colour, marker=marker, label=f"MaNTA, {nc} cells")
    for r in sel:
        ax.annotate(str(r["k"]), (r["nflux"], r["error"]),
                    textcoords="offset points", xytext=(4.0, 2.6),
                    fontsize=6.5, color=colour)

# --- the same runs, time-marched instead ----------------------------------
tmf = frontier(tm)
ax.plot([r["nflux"] for r in tmf], [r["error"] for r in tmf],
        color="0.55", marker="o", markersize=3.0, markerfacecolor="none",
        linestyle="-.", linewidth=0.9,
        label="MaNTA, time-marched (frontier)")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("transport-model evaluations")
ax.set_ylabel(r"relative $L^1$ error")
ax.grid(True, which="major", linewidth=0.3, color="0.86")
ax.grid(True, which="minor", linewidth=0.2, color="0.94")
ax.legend(frameon=False, loc="upper center", ncol=3, handlelength=1.7,
          columnspacing=1.0, borderaxespad=0.2, handletextpad=0.5)
ax.set_ylim(1e-14, 3.0e3)
fig.tight_layout(pad=0.3)

DEST.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(DEST)
print("wrote", DEST)
print("steady frontier:",
      [(r["cells"], r["k"], r["nflux"], f"{r['error']:.2e}") for r in frontier(ptc)])
print("time-marched frontier:",
      [(r["cells"], r["k"], r["nflux"], f"{r['error']:.2e}") for r in tmf])
