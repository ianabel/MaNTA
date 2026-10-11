"""Cost--accuracy sweep for the paper's fig:park, using park-convergence's own code.

Reuses `benchmark.solve`, `benchmark.park_error` and `benchmark.second_order_fd`
unchanged, over the wider grid the figure shows: N in {4,6,10,20} cells,
k = 2..6.  Writes park_frontier.json beside itself.
"""
import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
EX = str(HERE.parent / "park-convergence")
sys.path.insert(0, EX)
os.chdir(EX)

import benchmark as B

CELLS = (4, 6, 10, 20)
DEGREES = (2, 3, 4, 5, 6)

rows = []
for ncells in CELLS:
    for k in DEGREES:
        try:
            case, runner = B.solve(ncells, k)
            rows.append({"cells": ncells, "k": k,
                         "points": ncells * (k + 1),
                         "nflux": case.nFlux, "nderiv": case.nDeriv,
                         "visits": (case.nFlux + case.nDeriv) // (ncells * (k + 1)),
                         "error": B.park_error(runner)})
            print(f"done {ncells:3d} {k} -> {rows[-1]['nflux']:6d} "
                  f"{rows[-1]['error']:.3e}", file=sys.stderr, flush=True)
        except RuntimeError as e:
            rows.append({"cells": ncells, "k": k, "error": None,
                         "failure": str(e)})
            print(f"FAIL {ncells:3d} {k}: {e}", file=sys.stderr, flush=True)

fd = [{"N": N, "error": B.second_order_fd(N)} for N in (11, 21, 41, 101, 201)]

out = {"manta": rows, "fd": fd}
dest = str(HERE / "park_frontier.json")
with open(dest, "w") as f:
    json.dump(out, f, indent=1)
print("wrote", dest, file=sys.stderr)
