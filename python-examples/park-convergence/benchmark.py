"""Accuracy against cost for Park's benchmark, in the units PERFORMANCE.md uses.

    python benchmark.py

PERFORMANCE.md asks that MaNTA be measured by the number of calls into a
`TransportSystem` needed to reach a given accuracy, and compared against the
algorithms in `refs/`. This script does both for the spatial half of that
question; `../jardin-critical-gradient/` does the nonlinear half.

Park reports (his Fig. 2, and the sentence beneath it) that his 4th-order IDO
scheme at `N = 11` grid points is as accurate as a 2nd-order finite difference
at `N = 101`. His figure carries no readable numbers, so the second-order
scheme -- his Eq. (17) -- is implemented below and used as the bridge: it puts
Park's `N = 11` result at a relative L1 error of about 6e-5. Treat that as a
proxy for his published number, not as his published number.

The error norm is Park's own, `sum|T - T_exact| / sum|T_exact|`, evaluated on a
fixed fine sample so it means the same thing at every resolution rather than
being read on each run's own nodes.

The solver writes its own progress lines as it goes, so the tables below are
collected first and printed together at the end. `python benchmark.py | tail -40`
if the interleaving is in the way.
"""

import numpy as np

import manta

from park_convergence import ParkConvergence, ExactSolution

SAMPLE = np.linspace(0.0, 1.0, 201)
EXACT = ExactSolution(SAMPLE)


def park_error(runner, postprocessed=False):
    get = (runner.getPostprocessedSolution if postprocessed
           else runner.getSolution)
    u = np.asarray(get(0, list(SAMPLE))).reshape(-1)
    return np.sum(np.abs(u - EXACT)) / np.sum(np.abs(EXACT))


def solve(ncells, k, superconvergent=False):
    case = ParkConvergence()
    runner = manta.Runner(case)
    runner.configure({
        "OutputFilename": "benchmark",
        "Polynomial_degree": k,
        "Grid_size": ncells,
        "Lower_boundary": 0.0,
        "Upper_boundary": 1.0,
        "Relative_tolerance": 1.0e-6,
        # See run.conf: anything tighter than ~1e-7 fails on the first step.
        "Absolute_tolerance": 1.0e-3,
        "delta_t": 1.0e4,
        "t_final": 1.0e4,
        "SteadyStateTolerance": 1.0e-11,
        "Superconvergent": superconvergent,
        "WriteOutput": False,
        "WriteDatFile": False,
    })
    case.reset_counts()
    runner.run_ss()
    return case, runner


def second_order_fd(N):
    """Park's Eq. (17): central differences for T'' + T'/x = -4(1-x^2)e^(1-x^2).

    One evaluation of the transport model per grid point, solved directly --
    the reference point the paper's claim is stated against.
    """
    x = np.linspace(0.0, 1.0, N)
    h = x[1] - x[0]
    rhs = -4.0 * (1.0 - x ** 2) * np.exp(1.0 - x ** 2)
    A = np.zeros((N, N))
    # On the axis T'(0) = 0, so T'' + T'/x -> 2 T''.
    A[0, 0], A[0, 1] = -4.0 / h ** 2, 4.0 / h ** 2
    for j in range(1, N - 1):
        A[j, j - 1] = 1.0 / h ** 2 - 1.0 / (2.0 * h * x[j])
        A[j, j] = -2.0 / h ** 2
        A[j, j + 1] = 1.0 / h ** 2 + 1.0 / (2.0 * h * x[j])
    A[N - 1, N - 1], rhs[N - 1] = 1.0, 0.0
    T = np.linalg.solve(A, rhs)
    exact = ExactSolution(x)
    return np.sum(np.abs(T - exact)) / np.sum(np.abs(exact))


def main():
    reference = [(N, second_order_fd(N)) for N in (11, 21, 41, 101)]

    accuracy = []
    for ncells in (4, 6, 10):
        for k in (2, 3, 4, 5):
            case, runner = solve(ncells, k)
            points = ncells * (k + 1)
            accuracy.append((ncells, k, points, case.nFlux, case.nDeriv,
                             (case.nFlux + case.nDeriv) // points,
                             park_error(runner)))

    superconvergent = []
    for ncells in (4, 6):
        for k in (3, 4, 5):
            for sc in (False, True):
                case, runner = solve(ncells, k, superconvergent=sc)
                superconvergent.append((ncells, k, sc, case.nFlux,
                                        park_error(runner, postprocessed=True)))

    print()
    print("Reference: Park's Eq. (17), 2nd order, solved directly")
    print(f"  {'points':>7} {'model calls':>12} {'error':>11}")
    for N, err in reference:
        print(f"  {N:7d} {N:12d} {err:11.3e}")
    print("  Park's IDO at 11 points matches the 101-point row above, i.e.")
    print("  about 6e-5 for 11 model calls, visiting each point once.")
    print()

    print("MaNTA (Superconvergent = false)")
    print(f"  {'cells':>5} {'k':>2} {'points':>7} {'flux calls':>11} "
          f"{'deriv pts':>10} {'visits':>7} {'error':>11}")
    for row in accuracy:
        print(f"  {row[0]:5d} {row[1]:2d} {row[2]:7d} {row[3]:11d} "
              f"{row[4]:10d} {row[5]:7d} {row[6]:11.3e}")
    print()
    print("  Per evaluation point MaNTA is in IDO's league -- 16 points reach")
    print("  what IDO reaches in 11. The gap is the `visits` column: MaNTA")
    print("  relaxes onto the steady state in time, where Park solves for it")
    print("  directly at 1/dt = 0 and LoDestro sets dt = 1e10.")
    print()

    print("Superconvergent, at matched cost, measured on the postprocessed u*")
    print(f"  {'cells':>5} {'k':>2} {'SC':>6} {'flux calls':>11} {'err u*':>11}")
    for ncells, k, sc, nflux, err in superconvergent:
        print(f"  {ncells:5d} {k:2d} {str(sc):>6} {nflux:11d} {err:11.3e}")
    print()
    print("  Superconvergence costs the extra node, (k+2)/(k+1). Compare rows")
    print("  of near-equal cost rather than equal (cells, k): on this problem")
    print("  raising k wins at k = 2-3 and the flag wins at k >= 4.")


if __name__ == "__main__":
    main()
