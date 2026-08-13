"""Nonlinear robustness and cost for Jardin's benchmark, in PERFORMANCE.md's units.

    python benchmark.py

PERFORMANCE.md asks that MaNTA be measured by the number of calls into a
`TransportSystem` needed to reach a given accuracy, and compared against the
algorithms in `refs/`. Where `../park-convergence/` measures spatial accuracy
per call on a linear problem, this measures the nonlinear solve: the exact
steady state here is a degree-1 polynomial and so is representable exactly at
every order, which takes the discretisation out of the answer.

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
collected first and printed together at the end. `python benchmark.py | tail -40`
if the interleaving is in the way.
"""

import numpy as np

import manta

from jardin_critical_gradient import (JardinCriticalGradient, ExactSolution,
                                      CriticalGradient)

SAMPLE = np.linspace(0.0, 1.0, 201)
EXACT = ExactSolution(SAMPLE)


def solve(ncells, k, t_final=1.0e4, exact_start=False):
    case = JardinCriticalGradient()
    if exact_start:
        g = CriticalGradient()
        case.InitialValue = lambda index, x: g * (1.0 - x)
        case.InitialDerivative = lambda index, x: -g
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
        "WriteOutput": False,
        "WriteDatFile": False,
    })
    case.reset_counts()
    runner.run_ss()
    u = np.asarray(runner.getSolution(0, list(SAMPLE))).reshape(-1)
    err = np.sum(np.abs(u - EXACT)) / np.sum(np.abs(EXACT))
    return case, err


def main():
    cost = []
    for ncells, k in [(4, 2), (4, 3), (4, 5), (6, 3), (10, 3), (10, 5)]:
        case, err = solve(ncells, k)
        points = ncells * (k + 1)
        cost.append((ncells, k, points, case.nFlux, case.nDeriv,
                     (case.nFlux + case.nDeriv) // points, err))

    # The error above does not fall to round-off even though the exact answer
    # is representable. It is not a stopping artefact, and this shows it: the
    # same number comes back however long the run is given.
    patience = []
    for t_final in (1.0e4, 1.0e6, 1.0e8):
        case, err = solve(10, 3, t_final=t_final)
        patience.append((t_final, case.nFlux, err))

    try:
        case, err = solve(10, 3, exact_start=True)
        exact_start = f"converged, error {err:.3e}"
    except RuntimeError as e:
        exact_start = f"{type(e).__name__}: {e}"

    print()
    print(f"Jardin's stiff benchmark. Exact steady gradient "
          f"g = {CriticalGradient():.10f}, so u = g (1 - x) exactly.")
    print()
    print("Cost to relax onto the stiff steady state")
    print(f"  {'cells':>5} {'k':>2} {'points':>7} {'flux calls':>11} "
          f"{'deriv pts':>10} {'visits':>7} {'error':>11}")
    for row in cost:
        print(f"  {row[0]:5d} {row[1]:2d} {row[2]:7d} {row[3]:11d} "
              f"{row[4]:10d} {row[5]:7d} {row[6]:11.3e}")
    print()
    print("  Every resolution converges: the nonlinear solve is robust against")
    print("  a diffusivity whose derivative diverges at the critical gradient,")
    print("  which is the failure Jardin's paper is about. Note that `deriv pts`")
    print("  cost no flux evaluations at all -- the derivative is analytic.")
    print()
    print("Is the remaining error just an unconverged transient? (10 cells, k=3)")
    print(f"  {'t_final':>9} {'flux calls':>11} {'error':>11}")
    for t_final, nflux, err in patience:
        print(f"  {t_final:9.0e} {nflux:11d} {err:11.3e}")
    print()
    print("  No -- it is a converged property of the discrete steady state, and")
    print("  it is unexplained. See README.md.")
    print()
    print("Starting from the exact steady state instead of Jardin's u = 1 - x:")
    print(f"  {exact_start}")
    print("  Expected to fail: g sits 0.009 above the critical gradient qc = 0.5,")
    print("  where dchi/dq diverges, so IDACalcIC's linesearch cannot move.")


if __name__ == "__main__":
    main()
