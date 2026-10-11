"""Nonlinear robustness and cost for Jardin's benchmark, in PERFORMANCE.md's units.

    python benchmark.py

PERFORMANCE.md asks that MaNTA be measured by the number of calls into a
`TransportSystem` needed to reach a given accuracy, and compared against the
algorithms in `refs/`. Where `../park-convergence/` measures spatial accuracy
per call on a linear problem, this measures the nonlinear solve: the exact
steady state here is a degree-1 polynomial and so is representable exactly at
every order, which takes the discretisation out of the answer entirely.

What the algorithms in `refs/` cost on a problem of this kind, for comparison:

  Jardin, JCP 227 (2008)   1-3 Newton iterations per step, each needing chi and
                           dchi/dq; PTRANSP uses 3. Tridiagonal, no assembled
                           Jacobian.
  Park, CPC 214 (2017)     9-15 root-finding iterations for a steady state, one
                           transport-model call per grid point per iteration,
                           and no derivatives at all.
  TRINITY, PoP 17 (2010)   n_r x (1 + n_p) flux evaluations per transport step,
                           2 Newton iterations per step, 10-15 steps.

The number to compare against those is the `visits` column: how many times
MaNTA evaluates the physics at each point over the whole run.

The solver writes its own progress lines as it goes, so the tables below are
collected first and printed together at the end. `python benchmark.py | tail -45`
if the interleaving is in the way.
"""

import numpy as np

import manta

from jardin_critical_gradient import (JardinCriticalGradient, ExactSolution,
                                      CriticalGradient)

SAMPLE = np.linspace(0.0, 1.0, 201)
EXACT = ExactSolution(SAMPLE)


def measure(case, ncells, k, t_final=1.0e4):
    runner = manta.Runner(case)
    runner.configure({
        "OutputFilename": "benchmark",
        "Polynomial_degree": k,
        "Grid_size": ncells,
        "Lower_boundary": 0.0,
        "Upper_boundary": 1.0,
        "Relative_tolerance": 1.0e-6,
        "Absolute_tolerance": 1.0e-3,
        "delta_t": t_final,
        "t_final": t_final,
        "SteadyStateTolerance": 1.0e-11,
        # Pinned, so the tables below keep measuring one algorithm. The
        # three-mode comparison is a separate block at the end.
        "SteadyStateSolver": "TimeMarch",
        "WriteOutput": False,
        "WriteDatFile": False,
    })
    case.reset_counts()
    runner.run_ss()
    u = np.asarray(runner.getSolution(0, list(SAMPLE))).reshape(-1)
    return case, np.sum(np.abs(u - EXACT)) / np.sum(np.abs(EXACT))


def solve(ncells, k, t_final=1.0e4, exact_start=False):
    case = JardinCriticalGradient()
    if exact_start:
        g = CriticalGradient()
        case.InitialValue = lambda index, x: g * (1.0 - x)
        case.InitialDerivative = lambda index, x: -g
    return measure(case, ncells, k, t_final)


def main():
    cost = []
    for ncells, k in [(4, 2), (4, 3), (4, 5), (6, 3), (10, 3), (10, 5)]:
        case, err = solve(ncells, k)
        points = ncells * (k + 1)
        cost.append((ncells, k, points, case.nFlux, case.nDeriv,
                     (case.nFlux + case.nDeriv) // points, err))

    # The boundary condition is the whole story on this problem, and the case now
    # states it correctly -- zero *flux*, as Mixed(d=1). To keep the trap
    # measurable, this block reconstructs the plausible mistake explicitly: a
    # *Neumann* end, which fixes q rather than the flux, with the same zero value.
    # That imposes q(0) = 0, an extra condition Jardin's problem does not have and
    # a false one, since the true axis gradient is -g.
    wrong = []
    for ncells, k in [(4, 2), (10, 3), (20, 3), (40, 3)]:
        case = JardinCriticalGradient()
        case.spec.variables[0].lower = manta.Neumann   # fixes q, not the flux
        case.LowerBoundary = lambda index, t: 0.0
        wrong.append((ncells, k) + (measure(case, ncells, k)[1],))

    # And the middle ground the case used to ship: the right quantity is still q,
    # but supplied with the value only a closed form provides.
    told = []
    for ncells, k in [(10, 3), (10, 5)]:
        case = JardinCriticalGradient()
        case.spec.variables[0].lower = manta.Neumann
        g = CriticalGradient()
        case.LowerBoundary = lambda index, t, g=g: -g
        told.append((ncells, k) + (measure(case, ncells, k)[1],))

    starts = []
    for label, es in (("Jardin's u = 1 - x", False),
                      ("the exact steady state", True)):
        try:
            _, err = solve(10, 3, exact_start=es)
            starts.append((label, f"converged, error {err:.3e}"))
        except RuntimeError as e:
            starts.append((label, f"{type(e).__name__}: {e}"))

    print()
    print("Cost to relax onto the stiff steady state, zero flux on the axis")
    print(f"  {'cells':>5} {'k':>2} {'points':>7} {'flux calls':>11} "
          f"{'deriv pts':>10} {'visits':>7} {'error':>11}")
    for row in cost:
        print(f"  {row[0]:5d} {row[1]:2d} {row[2]:7d} {row[3]:11d} "
              f"{row[4]:10d} {row[5]:7d} {row[6]:11.3e}")
    print()
    print("  The error is at round-off at every resolution, which is the right")
    print("  answer: the exact steady state is degree 1 and lies in P_k for every")
    print("  k here, so a correct scheme must reproduce it exactly. What the")
    print("  table measures is therefore the cost alone. Note that `deriv pts`")
    print("  cost no flux evaluations at all -- the derivative is analytic.")
    print()

    print("The same value on a Neumann end instead -- i.e. imposed on q")
    print(f"  {'cells':>5} {'k':>2} {'error':>11}")
    for ncells, k, err in wrong:
        print(f"  {ncells:5d} {k:2d} {err:11.3e}")
    print()
    print("  First order in h, and independent of k -- 40 cells at k=3 is no")
    print("  better than the rate suggests. That is the signature of a wrong")
    print("  boundary condition, not of a discretisation limit: q(0) = 0 forces")
    print("  a one-cell layer on the axis where the true q is -0.509. A Neumann")
    print("  boundary fixes q, and Jardin's problem constrains only the flux --")
    print("  which vanishes on the axis for any q at all.")
    print()

    print("A Neumann end given the *correct* gradient, -g, as this case used to")
    print(f"  {'cells':>5} {'k':>2} {'error':>11}")
    for ncells, k, err in told:
        print(f"  {ncells:5d} {k:2d} {err:11.3e}")
    print()
    print("  Also at round-off, so accuracy is not what the mixed form buys. What")
    print("  it buys is not having to know -g: that value comes from the closed")
    print("  form, which a real problem does not supply. Zero flux is a statement")
    print("  about the equations, and the run finds the axis gradient itself.")
    print()

    print("Starting point (10 cells, k=3):")
    for label, result in starts:
        print(f"  from {label:24s} {result}")
    print()
    print("  Both work. On a Neumann end with the value at zero the exact steady")
    print("  state was not even a usable initial condition -- IDACalcIC failed on")
    print("  it, because q = -g everywhere contradicts the imposed q(0) = 0.")


if __name__ == "__main__":
    main()
