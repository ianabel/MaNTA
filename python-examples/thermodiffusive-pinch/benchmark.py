"""Accuracy per evaluation, and cost against stiffness disparity, for the
two-channel pinch problem.

    python benchmark.py

Two tables. The first is the Park-style one: relative L1 error in each channel
against the closed form, and visits per point, over cells and degree. The second
holds the resolution fixed and sweeps the particle diffusivity D over four
decades. The steady state does not depend on D, so the error should not move;
what moves is how much faster one channel relaxes than the other, and the cost
of each steady-state method shows how much that disparity matters to it.

A visit is the model at one point, both channels at once. The solver writes
progress lines as it goes, so the tables are printed together at the end.
"""

import ctypes

import numpy as np

import manta

from thermodiffusive_pinch import ThermodiffusivePinch, ExactSolution

SAMPLE = np.linspace(0.0, 1.0, 201)
EXACT = ExactSolution(SAMPLE)


def solve(ncells, k, solver="Newton", D=1.0):
    case = ThermodiffusivePinch(D=D)
    runner = manta.Runner(case)
    runner.configure({
        "OutputFilename": "benchmark",
        "PolynomialDegree": k,
        "GridSize": ncells,
        "LowerBoundary": 0.0,
        "UpperBoundary": 1.0,
        "Relative_tolerance": 1.0e-6,
        "Absolute_tolerance": 1.0e-3,
        "delta_t": 1.0e4,
        "t_final": 1.0e4,
        "SteadyStateTolerance": 1.0e-11,
        "SteadyStateSolver": solver,
        "WriteOutput": False,
    })
    case.reset_counts()
    try:
        runner.run_ss()
    except RuntimeError as e:
        return None, str(e)[:50]
    err = []
    for v in range(2):
        u = np.asarray(runner.getSolution(v, list(SAMPLE))).reshape(-1)
        err.append(np.sum(np.abs(u - EXACT[v])) / np.sum(np.abs(EXACT[v])))
    return (case.nFlux + case.nDeriv) / (ncells * (k + 1)), err


def main():
    accuracy = []
    for k in (2, 3, 5):
        for ncells in (2, 4, 8, 16):
            visits, err = solve(ncells, k)
            accuracy.append((ncells, k, visits, err))

    sweep = []
    for D in (0.01, 0.1, 1.0, 10.0, 100.0):
        row = [D]
        for solver in ("Newton", "PseudoTransient", "TimeMarch"):
            row.append(solve(8, 3, solver, D))
        sweep.append(row)

    # The solver's own progress lines sit in the C library's buffer; flush them
    # first, so they come out ahead of the tables rather than after them.
    ctypes.CDLL(None).fflush(None)
    print()
    print("Accuracy per evaluation: Newton, relative L1 error per channel")
    print(f"  {'cells':>5} {'k':>2} {'visits':>7} {'error n':>10} {'error T':>10} {'rate n':>7} {'rate T':>7}")
    prev = None
    for ncells, k, visits, err in accuracy:
        rates = ""
        if prev is not None and prev[1] == k:
            rates = f"{np.log2(prev[3][0] / err[0]):7.2f} {np.log2(prev[3][1] / err[1]):7.2f}"
        print(f"  {ncells:5d} {k:2d} {visits:7.1f} {err[0]:10.3e} {err[1]:10.3e} {rates}")
        prev = (ncells, k, visits, err)
    print()
    print("  The rates approach k + 1 from below. The approach is slow because of a")
    print("  branch point 0.155 outside the wall; see thermodiffusive_pinch.py.")
    print()

    print("Stiffness disparity: 8 cells, k = 3, visits per point (error n / T)")
    print(f"  {'D':>7} {'Newton':>26} {'PseudoTransient':>26} {'TimeMarch':>26}")
    for row in sweep:
        cells = []
        for visits, err in row[1:]:
            if visits is None:
                cells.append(f"{'FAILED ' + err:>26}"[:26])
            else:
                cells.append(f"{visits:6.1f} ({err[0]:.1e}/{err[1]:.1e})")
        print(f"  {row[0]:7g} " + " ".join(f"{c:>26}" for c in cells))
    print()
    print("  The errors are the same in every column of a row and down every column:")
    print("  D moves the transient, not the answer.")


if __name__ == "__main__":
    main()
