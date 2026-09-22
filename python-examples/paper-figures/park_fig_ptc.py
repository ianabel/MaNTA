"""The same sweep as park_fig.py, reached by pseudo-transient continuation."""
import json
import os
import pathlib
import sys
HERE = pathlib.Path(__file__).resolve().parent
EX = str(HERE.parent / "park-convergence")
sys.path.insert(0, EX); os.chdir(EX)
import benchmark as B

rows = []
for ncells in (4, 6, 10, 20):
    for k in (2, 3, 4, 5, 6):
        try:
            case, runner = B.solve(ncells, k, mode="PseudoTransient")
            rows.append({"cells": ncells, "k": k, "points": ncells * (k + 1),
                         "nflux": case.nFlux, "nderiv": case.nDeriv,
                         "visits": (case.nFlux + case.nDeriv) // (ncells * (k + 1)),
                         "error": B.park_error(runner)})
            print(f"done {ncells:3d} {k} -> {case.nFlux:6d} "
                  f"{rows[-1]['error']:.3e}", file=sys.stderr, flush=True)
        except RuntimeError as e:
            rows.append({"cells": ncells, "k": k, "error": None, "failure": str(e)})
            print(f"FAIL {ncells:3d} {k}: {e}", file=sys.stderr, flush=True)

dest = str(HERE / "park_frontier_ptc.json")
json.dump({"manta_ptc": rows}, open(dest, "w"), indent=1)
print("wrote", dest, file=sys.stderr)
