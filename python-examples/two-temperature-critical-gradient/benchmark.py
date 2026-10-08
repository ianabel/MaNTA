"""The cost of the coupled stiff solve, in visits per point, for the
two-temperature critical-gradient problem.

    python benchmark.py

The steady state is linear in both channels, so every resolution should
reproduce it to round-off and the tables measure cost alone -- as for Jardin's
single-channel problem in ../jardin-critical-gradient/, which is the number to
hold these against. Three tables:

  1. cost by steady-state method, without exchange and with it;
  2. what happens as the steady state approaches the threshold -- the heating
     that sets the ion gradient 0.0054 above it, where the steady-state
     solvers pay 40-50 times their usual cost and time marching does not;
  3. the starting point: the constant-chi steady state Jardin also uses starts
     the exchange variant at gradients near 6, which IDA's first step does not
     survive.

A visit is the model at one point, both channels at once. The solver writes
progress lines as it goes, so the tables are printed together at the end.
"""

import ctypes

import numpy as np

import manta

import two_temperature_critical_gradient as tt
from two_temperature_critical_gradient import TwoTemperatureCriticalGradient, ExactSolution

SAMPLE = np.linspace(0.0, 1.0, 201)
SOLVERS = ("Newton", "PseudoTransient", "TimeMarch")


def solve(case, ncells, k, solver, exact):
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
        return f"FAILED: {str(e)[:40]}"
    err = max(np.sum(np.abs(np.asarray(runner.getSolution(v, list(SAMPLE))).reshape(-1) - exact[v]))
              / np.sum(np.abs(exact[v])) for v in range(2))
    return f"{(case.nFlux + case.nDeriv) / (ncells * (k + 1)):6.1f} ({err:.1e})"


def main():
    cost = []
    for exchange in (False, True):
        exact = ExactSolution(SAMPLE, exchange)
        for ncells, k in ((4, 2), (4, 3), (10, 3), (10, 5)):
            row = [exchange, ncells, k]
            for solver in SOLVERS:
                row.append(solve(TwoTemperatureCriticalGradient(exchange=exchange), ncells, k, solver, exact))
            cost.append(row)

    # Near the kink. H_PLAIN is read when the case and the exact solution are
    # built, so it is swapped around both.
    kink = []
    saved = tt.H_PLAIN
    for H in ((1.0, 2.0), (1.5, 2.5), (2.0, 3.0)):
        tt.H_PLAIN = np.array(H)
        g = tt.SteadyGradients()
        exact = ExactSolution(SAMPLE)
        row = [H, g - tt.QC]
        for solver in SOLVERS:
            row.append(solve(TwoTemperatureCriticalGradient(), 4, 2, solver, exact))
        kink.append(row)
    tt.H_PLAIN = saved

    # Jardin's other starting point, the constant-chi steady state.
    starts = []
    for exchange in (False, True):
        case = TwoTemperatureCriticalGradient(exchange=exchange)
        case.InitialValue = lambda i, x, c=case: c.Tb[i] + c.H[i] / tt.CHI0 * (1.0 - x)
        case.InitialDerivative = lambda i, x, c=case: -c.H[i] / tt.CHI0
        starts.append((exchange, case.H.copy(),
                       solve(case, 10, 3, "TimeMarch", ExactSolution(SAMPLE, exchange))))

    # The solver's own progress lines sit in the C library's buffer; flush them
    # first, so they come out ahead of the tables rather than after them.
    ctypes.CDLL(None).fflush(None)
    print()
    print("Cost to the stiff steady state: visits per point (worst-channel L1 error)")
    print(f"  {'exchange':<9} {'cells':>5} {'k':>2} " + " ".join(f"{s:>18}" for s in SOLVERS))
    for row in cost:
        print(f"  {str(row[0]):<9} {row[1]:5d} {row[2]:2d} " + " ".join(f"{c:>18}" for c in row[3:]))
    print()
    print("Approaching the threshold: 4 cells, k = 2, no exchange")
    print(f"  {'H':<12} {'g - qc':<20} " + " ".join(f"{s:>18}" for s in SOLVERS))
    for H, above, *cells in kink:
        print(f"  {str(H):<12} {np.array2string(above, precision=4):<20} " + " ".join(f"{c:>18}" for c in cells))
    print()
    print("  At H = (1, 2) the steady-state solvers still reach round-off, at 40-50")
    print("  times their cost further from the threshold; time marching's cost does")
    print("  not move.")
    print()
    print("Starting from the constant-chi steady state, 10 cells, k = 3, TimeMarch")
    for exchange, H, result in starts:
        print(f"  exchange {str(exchange):<6} initial gradients {np.array2string(H, precision=2):<14} {result}")


if __name__ == "__main__":
    main()
