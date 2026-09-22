# Scripts behind the paper's tables and figure

The three `benchmark.py` files under `../` produce most of Section 6 of
`paper/main.tex` on their own. These four do the rest, and all of them import
the examples' own cases rather than restating them:

| script | what it produces |
|---|---|
| `steady_modes.py` | `tab:steady` -- visits per point by `SteadyStateSolver`, on all three benchmarks at Park 4 cells, Jardin and Shestakov 10, all `k = 3` |
| `reuse.py` | the `NewtonJacobianReuse` columns of the same table |
| `park_fig.py` / `park_fig_ptc.py` | the cost/accuracy sweep for `fig:park`, time-marched and by continuation, over 4/6/10/20 cells at `k = 2..6` |
| `plot_park.py` | draws `paper/figures/park-frontier.pdf` from the two JSON files |

Run them against a built tree:

    PYTHONPATH=../../python python3 steady_modes.py

`park_frontier.json` and `park_frontier_ptc.json` are the sweep output as of
2026-09-21, kept so the figure can be redrawn without a re-run.

**A rebuild relinks `python/manta/_manta*.so` in place**, so a benchmark that
is running when `cmake --build` reaches the link step dies with `ImportError:
file too short`. Build first, then measure.
