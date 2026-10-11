"""MaNTA against the published ASTRA scheme and TGYRO's iteration, on the four
benchmark problems, in one currency: model evaluations per point.

    python compare.py               # every problem, every method
    python compare.py --json out.json

One evaluation is the transport model at one point, every channel at once --
a TGYRO flux call at one radius, an ASTRA coefficient call at one face, a
MaNTA SigmaFn (or derivative) call at one node. Visits per point is that
total over the number of points the method resolves the profile with: radii,
cells, or N (k + 1) nodes. The error is the relative L1 error against the
closed form on 201 points, the worst channel's, each method's profile
interpolated the way the method itself defines it.

Every method stops on its own residual: MaNTA at SteadyStateTolerance = 1e-11,
ASTRA's march when its discrete steady residual is below 1e-10 relative to its
terms, TGYRO when its flux mismatch is below 1e-10 of the largest target.

The solver writes progress lines as it goes, so the tables are collected and
printed together at the end.
"""

import argparse
import ctypes
import json
import pathlib
import sys

import numpy as np

import manta

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import problems                                         # noqa: E402
import astra                                            # noqa: E402
import tgyro                                            # noqa: E402
from problems import park, jardin, pinch, twotemp       # noqa: E402


def manta_case(problem):
    if isinstance(problem, problems.Park):
        return park.ParkConvergence()
    if isinstance(problem, problems.Jardin):
        return jardin.JardinCriticalGradient()
    if isinstance(problem, problems.Pinch):
        return pinch.ThermodiffusivePinch(D=problem.D)
    return twotemp.TwoTemperatureCriticalGradient(exchange=problem.exchange)


def run_manta(problem, ncells, k, solver):
    case = manta_case(problem)
    runner = manta.Runner(case)
    runner.configure({
        "OutputFilename": "compare", "PolynomialDegree": k, "GridSize": ncells,
        "LowerBoundary": 0.0, "UpperBoundary": 1.0,
        "Relative_tolerance": 1.0e-6, "Absolute_tolerance": 1.0e-3,
        "SteadyStateTolerance": 1.0e-11, "delta_t": 1.0e4, "t_final": 1.0e4,
        "SteadyStateSolver": solver, "WriteOutput": False,
    })
    case.reset_counts()
    try:
        runner.run_ss()
    except RuntimeError as e:
        return dict(failed=str(e)[:60])

    def profile(x):
        return np.array([np.asarray(runner.getSolution(v, list(x))).reshape(-1)
                         for v in range(problem.nvars)])

    points = ncells * (k + 1)
    evals = case.nFlux + case.nDeriv
    return dict(points=points, evals=evals, visits=evals / points,
                error=problem.error(profile), detail=f"{ncells} cells, k = {k}")


def run_astra(problem, ncells, **kw):
    r = astra.solve(problem, ncells, **kw)
    if not r.converged:
        return dict(failed=f"no steady state in {r.accepted + r.rejected} steps",
                    points=ncells, evals=r.evals, visits=r.evals / ncells,
                    error=problem.error(r.profile))
    return dict(points=ncells, evals=r.evals, visits=r.evals / ncells,
                error=problem.error(r.profile),
                detail=f"{ncells} cells, {r.accepted} steps, {r.rejected} rejected")


def run_tgyro(problem, nradii, **kw):
    r = tgyro.solve(problem, nradii, **kw)
    out = dict(points=nradii, evals=r.evals, visits=r.evals / nradii,
               error=problem.error(r.profile),
               detail=f"{nradii} radii, {r.iterations} iterations")
    if not r.converged:
        out["failed"] = f"not converged in {r.iterations} iterations"
    return out


MANTA_LEVELS = [(4, 2), (4, 3), (8, 3), (4, 5), (8, 5)]
ASTRA_CELLS = [10, 20, 40, 80, 160]
TGYRO_RADII = [4, 8, 16, 32]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="also write every row to this file")
    args = ap.parse_args()

    rows = []
    for problem in problems.all_problems():
        for solver in ("Newton", "PseudoTransient"):
            for ncells, k in MANTA_LEVELS:
                rows.append(dict(problem=problem.name, method=f"MaNTA {solver}",
                                 **run_manta(problem, ncells, k, solver)))
        for n in ASTRA_CELLS:
            rows.append(dict(problem=problem.name, method=f"ASTRA (Dbar = {problem.dbar:g})",
                             **run_astra(problem, n)))
        if problem.dbar > 0:
            # And without the stabiliser, at one resolution, to show what it is for.
            rows.append(dict(problem=problem.name, method="ASTRA (Dbar = 0)",
                             **run_astra(problem, 40, dbar=0.0, max_steps=20000)))
        for dz, label in ((0.1, "dz = 0.1, the code's"), (0.01, "dz = 0.01")):
            for n in TGYRO_RADII:
                rows.append(dict(problem=problem.name, method=f"TGYRO ({label})",
                                 **run_tgyro(problem, n, dz=dz)))

    # The stiffness-disparity axis: the pinch problem's answer is independent of
    # its particle diffusivity, so only the cost should move.
    sweep = []
    for D in (0.01, 0.1, 1.0, 10.0, 100.0):
        problem = problems.Pinch(D)
        sweep.append(dict(problem=problem.name, method="MaNTA Newton", **run_manta(problem, 8, 3, "Newton")))
        sweep.append(dict(problem=problem.name, method="MaNTA PseudoTransient",
                          **run_manta(problem, 8, 3, "PseudoTransient")))
        sweep.append(dict(problem=problem.name, method="ASTRA", **run_astra(problem, 40)))
        sweep.append(dict(problem=problem.name, method="TGYRO (dz = 0.1)", **run_tgyro(problem, 16)))

    def table(rs):
        print(f"  {'problem':<28} {'method':<30} {'resolution':<36} {'evals':>8} {'visits':>8} {'error':>9}")
        last = None
        for r in rs:
            if last is not None and r["problem"] != last:
                print()
            last = r["problem"]
            err = f"{r['error']:9.2e}" if "error" in r else " " * 9
            evals = f"{r['evals']:8d}" if "evals" in r else " " * 8
            visits = f"{r['visits']:8.1f}" if "visits" in r else " " * 8
            note = f"  FAILED: {r['failed']}" if "failed" in r else ""
            print(f"  {r['problem']:<28} {r['method']:<30} {r.get('detail', ''):<36} {evals} {visits} {err}{note}")

    # The solver's own progress lines sit in the C library's buffer; flush them
    # first, so they come out ahead of the tables rather than after them.
    ctypes.CDLL(None).fflush(None)
    print()
    print("Every problem, every method")
    table(rows)
    print()
    print("Pinch at fixed resolution, sweeping the particle diffusivity D (the answer does not move)")
    table(sweep)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(rows=rows, sweep=sweep), f, indent=1)


if __name__ == "__main__":
    main()
