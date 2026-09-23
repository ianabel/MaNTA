"""What NewtonJacobianReuse costs on the two nonlinear benchmarks.

Same cases and resolutions as steady_modes.py, so the numbers sit alongside
tab:steady: Park on 4 cells, Jardin and Shestakov on 10, all at k = 3.
"""
import pathlib
import sys

import numpy as np
import manta
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from steady_modes import park, jardin, shestakov, run as _run


def run(case, ncells, k, extra, sample, exact, mode, reuse):
    e = dict(extra); e["NewtonJacobianReuse"] = reuse
    return _run(case, ncells, k, e, sample, exact, mode)


rows = []
for maker in (park, jardin, shestakov):
    name, ncells, k, mkcase, extra, sample, exact = maker()
    for mode in ("PseudoTransient", "Newton"):
        for reuse in (10, 1):
            try:
                nflux, nderiv, visits, err = run(mkcase(), ncells, k, extra,
                                                 sample, exact, mode, reuse)
                rows.append((name, mode, reuse, nflux, nderiv, visits, err))
            except RuntimeError as e:
                rows.append((name, mode, reuse, None, None, None, str(e)[:60]))

print()
print("NewtonJacobianReuse, at the resolutions of tab:steady")
print(f"  {'case':>10} {'mode':>16} {'reuse':>5} {'flux':>8} {'deriv':>7} "
      f"{'visits':>7} {'error':>12}")
for name, mode, reuse, nflux, nderiv, visits, err in rows:
    if nflux is None:
        print(f"  {name:>10} {mode:>16} {reuse:5d} {'FAILS':>8}   {err}")
    else:
        print(f"  {name:>10} {mode:>16} {reuse:5d} {nflux:8d} {nderiv:7d} "
              f"{visits:7d} {err:12.4e}")
