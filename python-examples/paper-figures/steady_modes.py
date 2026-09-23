"""Visits per collocation point to reach the steady state, by SteadyStateSolver.

Regenerates the paper's tab:steady from the three python-examples benchmarks at
the resolutions the caption states: Park on 4 cells, Jardin and Shestakov on 10,
all at k = 3.  Each row's configuration is copied from that example's own
benchmark.py `solve()`, with only `SteadyStateSolver` varied.
"""
import pathlib
import sys

import numpy as np
import manta

EX = str(pathlib.Path(__file__).resolve().parent.parent)
MODES = ("TimeMarch", "PseudoTransient", "Newton")


def run(case, ncells, k, extra, sample, exact, mode):
    runner = manta.Runner(case)
    cfg = {
        "OutputFilename": "steady_modes",
        "Polynomial_degree": k,
        "Grid_size": ncells,
        "Lower_boundary": 0.0,
        "Relative_tolerance": 1.0e-6,
        "Absolute_tolerance": 1.0e-3,
        "SteadyStateTolerance": 1.0e-11,
        "SteadyStateSolver": mode,
        "WriteOutput": False,
        "WriteDatFile": False,
    }
    cfg.update(extra)
    runner.configure(cfg)
    case.reset_counts()
    runner.run_ss()
    u = np.asarray(runner.getSolution(0, list(sample))).reshape(-1)
    err = np.sum(np.abs(u - exact)) / np.sum(np.abs(exact))
    points = ncells * (k + 1)
    return (case.nFlux, case.nDeriv, (case.nFlux + case.nDeriv) // points, err)


def park():
    sys.path.insert(0, f"{EX}/park-convergence")
    from park_convergence import ParkConvergence, ExactSolution
    s = np.linspace(0.0, 1.0, 201)
    return ("Park", 4, 3,
            lambda: ParkConvergence(),
            {"Upper_boundary": 1.0, "delta_t": 1.0e4, "t_final": 1.0e4},
            s, ExactSolution(s))


def jardin():
    sys.path.insert(0, f"{EX}/jardin-critical-gradient")
    from jardin_critical_gradient import JardinCriticalGradient, ExactSolution
    s = np.linspace(0.0, 1.0, 201)
    return ("Jardin", 10, 3,
            lambda: JardinCriticalGradient(),
            {"Upper_boundary": 1.0, "delta_t": 1.0e4, "t_final": 1.0e4},
            s, ExactSolution(s))


def shestakov():
    sys.path.insert(0, f"{EX}/shestakov-nonlinear")
    from shestakov_nonlinear import ShestakovNonlinear, ExactSolution, LX
    s = np.linspace(0.0, LX, 201)
    return ("Shestakov", 10, 3,
            lambda: ShestakovNonlinear(n_b=0.05),
            {"Upper_boundary": LX, "delta_t": 1.0e3, "t_final": 1.0e3,
             "SuppressAlgebraicError": True},
            s, ExactSolution(s, 0.05))


def main():
    rows = []
    for maker in (park, jardin, shestakov):
        name, ncells, k, mkcase, extra, sample, exact = maker()
        for mode in MODES:
            try:
                nflux, nderiv, visits, err = run(mkcase(), ncells, k, extra,
                                                 sample, exact, mode)
                rows.append((name, ncells, k, mode, nflux, nderiv, visits, err))
            except RuntimeError as e:
                rows.append((name, ncells, k, mode, None, None, None, str(e)))

    print()
    print("Visits per collocation point to reach the steady state")
    print(f"  {'problem':>10} {'cells':>5} {'k':>2} {'mode':>16} "
          f"{'flux calls':>11} {'deriv pts':>10} {'visits':>7} {'error':>12}")
    for name, ncells, k, mode, nflux, nderiv, visits, err in rows:
        if nflux is None:
            print(f"  {name:>10} {ncells:5d} {k:2d} {mode:>16} {'FAILED':>11}  {err}")
        else:
            print(f"  {name:>10} {ncells:5d} {k:2d} {mode:>16} {nflux:11d} "
                  f"{nderiv:10d} {visits:7d} {err:12.4e}")


if __name__ == "__main__":
    main()
