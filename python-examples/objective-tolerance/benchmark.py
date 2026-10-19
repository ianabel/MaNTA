"""Stopping a steady solve on its objective, SteadyStateTest = "Objective".

    python benchmark.py

The four deterministic benchmarks beside this directory, each given the
objective a transport study would ask for -- the stored energy, int V' u dx,
with V' = x on the three cylindrical problems and 1 on Shestakov's slab, and one
objective per channel on the two-temperature problem -- and solved at the
resolutions python-examples/paper-figures/steady_modes.py uses.

For each problem and each SteadyStateSolver, Newton and PseudoTransient, it
runs NewtonStep at its default and the objective test at ObjectiveTolerance
1e-2, 1e-3 and 1e-4, with the default ObjectiveBudget, so the stop is held to
half of each. Against a tightly converged reference at the same
discretisation -- NewtonStep at SteadyStateStepTolerance = 1e-6, a correction
of 1e-12 relative at Relative_tolerance = 1e-6 -- it reports the relative
algebraic error left in G at the stop, and the cost: Newton iterations,
residual evaluations, Jacobian builds and solves, adjoint solves, and visits
(physics point-evaluations per collocation point, PERFORMANCE.md's currency)
and wall-clock time. EstimateObjectiveOnFinish is off, so every column is the
stopping test's own.

The solver prints its own progress as it goes; the tables are collected and
printed at the end. `python benchmark.py | tail -60` if that is in the way.
"""

import pathlib
import sys
import time

import numpy as np

import manta

EX = pathlib.Path(__file__).resolve().parent.parent
TOLERANCES = (1e-2, 1e-3, 1e-4)
MODES = ("Newton", "PseudoTransient")


class StoredEnergy(manta.AdjointProblem):
    """G_v = int V' u_v dx, one objective per variable.

    `cylinder` makes V' = x. One parameter that nothing depends on: an adjoint
    problem has to have one for the end-of-run gradient solveAdjoint makes.
    """

    def __init__(self, nvars, cylinder):
        manta.AdjointProblem.__init__(self)
        self.ng = nvars
        self.np = 1
        self.np_boundary = 0
        self.spatialParameters = False
        self.cylinder = cylinder

    def weight(self, x):
        x = np.asarray(x)
        return x if self.cylinder else np.ones_like(x)

    def gFn(self, g, states, x):
        return self.weight(x) * np.asarray(states["Variable"])[:, g]

    def dgFndp(self, g, states, x):
        return np.zeros((1, len(x)))

    def dg(self, g, states, x):
        V = np.asarray(states["Variable"])
        dV = np.zeros_like(V)
        dV[:, g] = self.weight(x)
        zeros = np.zeros_like(V)
        return {"Variable": dV, "Derivative": zeros, "Flux": zeros,
                "Aux": np.zeros((len(x), 0)), "Scalars": np.zeros(0)}

    def dSigma(self, i, states, x):
        return np.zeros((1, len(x)))

    def dSources(self, i, states, x):
        return np.zeros((1, len(x)))

    def getName(self, p):
        return "none"


def with_objective(cls, nvars, cylinder, **kwargs):
    class Case(cls):
        def createAdjointProblem(self):
            return StoredEnergy(nvars, cylinder)
    return lambda: Case(**kwargs)


def problems():
    sys.path.insert(0, str(EX / "park-convergence"))
    sys.path.insert(0, str(EX / "jardin-critical-gradient"))
    sys.path.insert(0, str(EX / "shestakov-nonlinear"))
    sys.path.insert(0, str(EX / "two-temperature-critical-gradient"))
    from park_convergence import ParkConvergence
    from jardin_critical_gradient import JardinCriticalGradient
    from shestakov_nonlinear import ShestakovNonlinear, LX
    from two_temperature_critical_gradient import TwoTemperatureCriticalGradient

    cyl = {"UpperBoundary": 1.0, "delta_t": 1.0e4, "t_final": 1.0e4}
    return [
        ("Park", with_objective(ParkConvergence, 1, True), 4, 3, cyl),
        ("Jardin", with_objective(JardinCriticalGradient, 1, True), 10, 3, cyl),
        ("Shestakov", with_objective(ShestakovNonlinear, 1, False, n_b=0.05), 10, 3,
         {"UpperBoundary": LX, "delta_t": 1.0e3, "t_final": 1.0e3,
          "SuppressAlgebraicError": True}),
        ("Two-temp.", with_objective(TwoTemperatureCriticalGradient, 2, True), 10, 3, cyl),
    ]


def solve(make, ncells, k, extra, mode, **settings):
    case = make()
    runner = manta.Runner(case)
    cfg = {
        "OutputFilename": "objective_tolerance",
        "PolynomialDegree": k,
        "GridSize": ncells,
        "LowerBoundary": 0.0,
        "Relative_tolerance": 1.0e-6,
        "Absolute_tolerance": 1.0e-3,
        "SteadyStateSolve": True,
        "SteadyStateSolver": mode,
        "solveAdjoint": True,
        "EstimateObjectiveOnFinish": False,
        "WriteOutput": False,
        "WriteDatFile": False,
    }
    cfg.update(extra)
    cfg.update(settings)
    runner.configure(cfg)
    case.reset_counts()
    start = time.perf_counter()
    runner.run_ss()
    wall = time.perf_counter() - start
    stats = runner.steadyStats()
    visits = (case.nFlux + case.nDeriv) / (ncells * (k + 1))
    return runner, np.asarray(runner.G()), stats, visits, wall


def main():
    rows = []
    for name, make, ncells, k, extra in problems():
        _, G_ref, ref, _, _ = solve(make, ncells, k, extra, "Newton", SteadyStateStepTolerance=1e-6)
        assert ref["outcome"] == manta.SteadyOutcome.Converged
        for mode in MODES:
            settings = [("NewtonStep", None, {})] + [
                ("Objective", tol, {"SteadyStateTest": "Objective", "ObjectiveTolerance": tol})
                for tol in TOLERANCES]
            for test, tol, kw in settings:
                try:
                    _, G, s, visits, wall = solve(make, ncells, k, extra, mode, **kw)
                    err = np.max(np.abs(G - G_ref) / np.abs(G_ref))
                    rows.append((name, mode, test, tol, err, s, visits, wall))
                except RuntimeError as e:
                    rows.append((name, mode, test, tol, None, str(e)[:60], None, None))

    print()
    print("Relative algebraic error in G at the stop, against NewtonStep at "
          "SteadyStateStepTolerance = 1e-6, and the cost of getting there")
    print(f"  {'problem':>10} {'mode':>15} {'test':>10} {'tol':>6} {'share':>7} {'|dG|/|G|':>9} "
          f"{'iters':>5} {'res':>4} {'jac':>4} {'solves':>6} {'adj':>4} {'visits':>6} {'ms':>6}")
    for name, mode, test, tol, err, s, visits, wall in rows:
        tol_s = f"{tol:6.0e}" if tol else f"{'-':>6}"
        share_s = f"{0.5 * tol:7.0e}" if tol else f"{'-':>7}"
        if err is None:
            print(f"  {name:>10} {mode:>15} {test:>10} {tol_s} {share_s} FAILED: {s}")
            continue
        print(f"  {name:>10} {mode:>15} {test:>10} {tol_s} {share_s} {err:9.1e} "
              f"{s['newton_iterations']:5d} {s['residual_evaluations']:4d} "
              f"{s['jacobian_builds']:4d} {s['jacobian_solves']:6d} {s['adjoint_solves']:4d} "
              f"{visits:6.1f} {1e3 * wall:6.1f}")


if __name__ == "__main__":
    main()
