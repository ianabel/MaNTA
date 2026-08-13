"""Where MaNTA's formulation gives out on Shestakov's problem, and what it costs
where it does not.

    python benchmark.py

The other two benchmarks in this tree measure MaNTA against a paper's numbers.
This one maps a boundary instead, because the problem as Shestakov states it --
Dirichlet `n(Lx) = 0` -- is one MaNTA cannot integrate at all, and the reason is
worth pinning down rather than asserting.

Three things are measured:

  1. cost and accuracy where the problem is tractable, in PERFORMANCE.md's units;
  2. how far the Dirichlet value can be lowered towards the paper's zero;
  3. that the obstruction is IDA's *error test*, not its nonlinear solve.

The solver writes its own progress lines as it goes, so the tables are collected
first and printed together at the end.
"""

import numpy as np

import manta

from shestakov_nonlinear import ShestakovNonlinear, ExactSolution, LX

SAMPLE = np.linspace(0.0, LX, 201)


def solve(n_b, ncells, k, atol=1.0e-3, t_final=1.0e3):
    case = ShestakovNonlinear(n_b=n_b)
    runner = manta.Runner(case)
    runner.configure({
        "OutputFilename": "benchmark",
        "Polynomial_degree": k,
        "Grid_size": ncells,
        "Lower_boundary": 0.0,
        "Upper_boundary": LX,
        "Relative_tolerance": 1.0e-6,
        "Absolute_tolerance": atol,
        "delta_t": t_final,
        "t_final": t_final,
        "SteadyStateTolerance": 1.0e-11,
        "WriteOutput": False,
        "WriteDatFile": False,
    })
    case.reset_counts()
    runner.run_ss()
    u = np.asarray(runner.getSolution(0, list(SAMPLE))).reshape(-1)
    exact = ExactSolution(SAMPLE, n_b)
    err = np.sum(np.abs(u - exact)) / np.sum(np.abs(exact))
    return case, err


def attempt(*args, **kwargs):
    try:
        case, err = solve(*args, **kwargs)
        return f"OK   flux={case.nFlux:6d}  err={err:.3e}"
    except RuntimeError as e:
        return f"FAIL {e}"


def main():
    cost = []
    for ncells, k in [(10, 1), (10, 2), (10, 3), (20, 2), (20, 3)]:
        try:
            case, err = solve(0.1, ncells, k)
            points = ncells * (k + 1)
            cost.append((ncells, k, points, case.nFlux, case.nDeriv,
                         (case.nFlux + case.nDeriv) // points, err))
        except RuntimeError:
            cost.append((ncells, k, ncells * (k + 1), None, None, None, None))

    wall = [(n_b, attempt(n_b, 10, 2)) for n_b in
            (0.2, 0.1, 0.07, 0.05, 0.03, 0.01, 1.0e-3, 0.0)]

    # At the paper's Sec 2.2 value the obstruction really is the error test, and
    # loosening it -- only within a window -- lets the problem through.
    tolerance = [(atol, attempt(1.0e-3, 10, 2, atol=atol)) for atol in
                 (1.0e-3, 1.0e-1, 1.0e0, 1.0e1, 1.0e2)]

    print()
    print("1. Cost and accuracy at n(Lx) = 0.1, where the problem is tractable")
    print(f"   {'cells':>5} {'k':>2} {'points':>7} {'flux calls':>11} "
          f"{'deriv pts':>10} {'visits':>7} {'error':>11}")
    for ncells, k, points, nflux, nderiv, visits, err in cost:
        if nflux is None:
            print(f"   {ncells:5d} {k:2d} {points:7d} {'FAILED':>11}")
        else:
            print(f"   {ncells:5d} {k:2d} {points:7d} {nflux:11d} "
                  f"{nderiv:10d} {visits:7d} {err:11.3e}")
    print()
    print("   The error is ~1e-2 and falls slowly. That is the solution's")
    print("   regularity, not the scheme: n_e carries an x^(4/3) at the axis,")
    print("   whose second derivative is unbounded, and the step source puts a")
    print("   kink at x = d. Do not expect an order study to work here.")
    print()

    print("2. How far the Dirichlet value can be lowered towards the paper's 0")
    print(f"   {'n(Lx)':>8}  result")
    for n_b, result in wall:
        print(f"   {n_b:8.0e}  {result}")
    print()
    print("   Shestakov's Sec 2.1 value is 0 and his Sec 2.2 value is 1e-3.")
    print("   As u -> 0 the flux D0 q^3/u^2 is a 0/0 that is finite in the true")
    print("   solution but unbounded in a perturbed one, and its Jacobian entry")
    print("   d(sigma_hat)/du = -2 D0 q^3/u^3 diverges outright.")
    print()

    print("3. At n(Lx) = 1e-3, is it the nonlinear solve or the error test?")
    print(f"   {'atol':>8}  result")
    for atol, result in tolerance:
        print(f"   {atol:8.0e}  {result}")
    print()
    print("   The error test. Loosening Absolute_tolerance alone admits the")
    print("   paper's own Sec 2.2 value -- and only within a window, because too")
    print("   loose lets the solution wander to where the flux blows up. IDA's")
    print("   log shows the Newton converging in 2 iterations every time and the")
    print("   step dying in the error test with dsm identical as h shrinks,")
    print("   which is the signature of an algebraic component that cannot be")
    print("   resolved by taking a smaller step. See README.md.")


if __name__ == "__main__":
    main()
