"""Visits per collocation point to reach the steady state, by SteadyStateSolver.

Regenerates the paper's tab:steady from the three python-examples benchmarks at
the resolutions the caption states: Park on 4 cells, Jardin and Shestakov on 10,
all at k = 3.  Each row's configuration is copied from that example's own
benchmark.py `solve()`, with only `SteadyStateSolver` and `NewtonJacobianReuse`
varied.
"""
import os
import pathlib
import sys

import numpy as np
import manta

EX = str(pathlib.Path(__file__).resolve().parent.parent)

# (SteadyStateSolver, NewtonJacobianReuse): the table's five columns. Time
# marching has no Jacobian reuse to vary.
COLUMNS = (("TimeMarch", None),
           ("PseudoTransient", 10), ("PseudoTransient", 1),
           ("Newton", 10), ("Newton", 1))

# The two steady methods stop on the Newton correction, SteadyStateTest =
# "NewtonStep": ss * ||J^-1 F||_WRMS <= STEP_TOLERANCE, in the error weights
# Relative_tolerance and Absolute_tolerance set. The tolerance is the default,
# 0.33, because that is the loosest one that buys an answer worth having: a
# relative error of 5e-3 is already better than a plasma transport run can
# expect, because the model producing sigma is not that good, and at 0.33 every
# row stops within 6.4e-4 of the converged discrete solution. At 1.0 Shestakov
# by Newton with a reuse of one stops 9.5e-3 from it. Newton's method converges
# too fast for anything between to buy much: most rows reach round-off on the
# iteration that passes. Set MANTA_STEP_TOL to measure a different regime.
STEP_TOLERANCE = float(os.environ.get("MANTA_STEP_TOL", "0.33"))

# Time marching reads neither of those. It stops when ||dY/dt|| falls below
# SteadyStateTolerance, and it is the one column the setting moves little: its
# absolute floor is fixed in the code, so it cannot be asked for a cheaper, less
# accurate answer. MANTA_STEADY_TEST=Residual measures the steady methods on
# KINSOL's ||F|| test instead, at this same tolerance.
TOLERANCE = float(os.environ.get("MANTA_STEADY_TOL", "1.0e-2"))
STEADY_TEST = os.environ.get("MANTA_STEADY_TEST", "NewtonStep")


def solve(case, ncells, k, extra, sample, mode, reuse,
          step_tolerance=STEP_TOLERANCE, test=STEADY_TEST):
    """Counts and the sampled u for one steady solve."""
    runner = manta.Runner(case)
    cfg = {
        "OutputFilename": "steady_modes",
        "PolynomialDegree": k,
        "GridSize": ncells,
        "LowerBoundary": 0.0,
        "Relative_tolerance": 1.0e-6,
        "Absolute_tolerance": 1.0e-3,
        "SteadyStateTolerance": TOLERANCE,
        "SteadyStateTest": test,
        "SteadyStateStepTolerance": step_tolerance,
        "SteadyStateSolver": mode,
        "WriteOutput": False,
        "WriteDatFile": False,
    }
    if reuse is not None:
        cfg["NewtonJacobianReuse"] = reuse
    cfg.update(extra)
    runner.configure(cfg)
    case.reset_counts()
    runner.run_ss()
    u = np.asarray(runner.getSolution(0, list(sample))).reshape(-1)
    return case.nFlux, case.nDeriv, u


def relative_l1(u, v):
    return np.sum(np.abs(u - v)) / np.sum(np.abs(v))


def park():
    sys.path.insert(0, f"{EX}/park-convergence")
    from park_convergence import ParkConvergence, ExactSolution
    s = np.linspace(0.0, 1.0, 201)
    return ("Park", 4, 3,
            lambda: ParkConvergence(),
            {"UpperBoundary": 1.0, "delta_t": 1.0e4, "t_final": 1.0e4},
            s, ExactSolution(s))


def jardin():
    sys.path.insert(0, f"{EX}/jardin-critical-gradient")
    from jardin_critical_gradient import JardinCriticalGradient, ExactSolution
    s = np.linspace(0.0, 1.0, 201)
    return ("Jardin", 10, 3,
            lambda: JardinCriticalGradient(),
            {"UpperBoundary": 1.0, "delta_t": 1.0e4, "t_final": 1.0e4},
            s, ExactSolution(s))


def shestakov():
    sys.path.insert(0, f"{EX}/shestakov-nonlinear")
    from shestakov_nonlinear import ShestakovNonlinear, ExactSolution, LX
    s = np.linspace(0.0, LX, 201)
    return ("Shestakov", 10, 3,
            lambda: ShestakovNonlinear(n_b=0.05),
            {"UpperBoundary": LX, "delta_t": 1.0e3, "t_final": 1.0e3,
             "SuppressAlgebraicError": True},
            s, ExactSolution(s, 0.05))


def main():
    rows = []
    for maker in (park, jardin, shestakov):
        name, ncells, k, mkcase, extra, sample, exact = maker()
        # The converged discrete solution, against which the algebraic error --
        # what the stopping test controls -- is measured. The error against the
        # closed form also carries the discretisation's, which on Shestakov is
        # 7.2e-3 and of either sign relative to the algebraic one.
        *_, converged = solve(mkcase(), ncells, k, extra, sample, "Newton", 1,
                              step_tolerance=1.0e-6, test="NewtonStep")
        for mode, reuse in COLUMNS:
            label = mode if reuse is None else f"{mode} {reuse}"
            try:
                nflux, nderiv, u = solve(mkcase(), ncells, k, extra, sample,
                                         mode, reuse)
                visits = (nflux + nderiv) // (ncells * (k + 1))
                rows.append((name, ncells, k, label, nflux, nderiv, visits,
                             relative_l1(u, exact), relative_l1(u, converged)))
            except RuntimeError as e:
                rows.append((name, ncells, k, label, None, None, None, str(e),
                             None))

    print()
    steady = (f"SteadyStateStepTolerance = {STEP_TOLERANCE:g}"
              if STEADY_TEST == "NewtonStep" else STEADY_TEST)
    print(f"Visits per collocation point to reach the steady state "
          f"({steady}; SteadyStateTolerance = {TOLERANCE:g})")
    print(f"  {'problem':>10} {'cells':>5} {'k':>2} {'mode, reuse':>18} "
          f"{'flux calls':>11} {'deriv pts':>10} {'visits':>7} {'error':>11} "
          f"{'algebraic':>11}")
    for name, ncells, k, label, nflux, nderiv, visits, err, alg in rows:
        if nflux is None:
            print(f"  {name:>10} {ncells:5d} {k:2d} {label:>18} {'FAILED':>11}  {err}")
        else:
            print(f"  {name:>10} {ncells:5d} {k:2d} {label:>18} {nflux:11d} "
                  f"{nderiv:10d} {visits:7d} {err:11.2e} {alg:11.2e}")


if __name__ == "__main__":
    main()
