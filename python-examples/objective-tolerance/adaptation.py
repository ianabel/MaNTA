"""Adaptation keyed off the objective, AdaptationTarget = "Objective".

    python adaptation.py            # the comparison table, a minute or so
    python adaptation.py --sweep    # whether eps_G is met, over 210 runs
    python adaptation.py --starts   # ... and from 35 non-uniform meshes, 700 runs

Four problems with a closed-form objective, each given ObjectiveTolerance 1e-3
and 1e-5:

  * a manufactured nonlinear problem, -d/dx[(1 + u^2) u'] = S with u = sin(pi x),
    and G = int u^2 dx = 1/2, by DegreeAdaptation from k = 1 on 4 cells;
  * Park's benchmark (../park-convergence) with its stored energy
    W = int x u dx = (e - 2)/2, by DegreeAdaptation from k = 1 on 4 cells, and by
    MeshAdaptation from k = 3 on 10 as the case that wants no grading;
  * u = x - x^(4/3), the x^(4/3) singularity of Shestakov's axis in a linear
    problem, with G = int u dx = 1/14, by MeshAdaptation from k = 3 on 10 cells
    (and its mirror image, singular at x = 1, in the sweeps);
  * Shestakov's (../shestakov-nonlinear, n_b = 0.05) with its inventory
    N = int u dx, from the closed-form steady state, by MeshAdaptation from k = 3
    on 10 cells, and by DegreeAdaptation alone for the limit of raising k.

For each it runs the objective target, and the solution target with its
DegreeTolerance at the default 1e-6 and tuned: the loosest half-decade from 1e-1
down at which the error in G is within eps_G -- which needs the closed form, and
so is a number no study would have. Both targets stop each steady solve with the
same Objective test at the same ObjectiveTolerance, so only what a level is
measured by differs. Reported: the relative error in G against the closed form,
the degrees solved, the final mesh, and the physics the run asked for -- whole
sweeps over the grid, flux and derivative, and the point evaluations in them.
"""

import contextlib
import ctypes
import os
import pathlib
import re
import sys
import tempfile

import numpy as np
from scipy import integrate

import manta

EX = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EX / "park-convergence"))
sys.path.insert(0, str(EX / "shestakov-nonlinear"))
from park_convergence import ParkConvergence  # noqa: E402
from shestakov_nonlinear import ShestakovNonlinear, ExactSolution, LX  # noqa: E402

TOLERANCES = (1e-3, 1e-5)
SCAN = [10.0 ** (-e / 2) for e in range(2, 25)]   # DegreeTolerance 1e-1 .. 1e-12
N_B = 0.05


class Weighted(manta.AdjointProblem):
    """G = int w(x) u dx, or int u^2 dx; one parameter nothing depends on."""

    def __init__(self, weight=None, square=False):
        manta.AdjointProblem.__init__(self)
        self.ng = 1
        self.np = 1
        self.np_boundary = 0
        self.spatialParameters = False
        self.weight = weight or (lambda x: np.ones_like(x))
        self.square = square

    def gFn(self, g, states, x):
        u = np.asarray(states["Variable"])[:, 0]
        return self.weight(np.asarray(x)) * (u * u if self.square else u)

    def dgFndp(self, g, states, x):
        return np.zeros((1, len(x)))

    def dg(self, g, states, x):
        V = np.asarray(states["Variable"])
        dV = np.zeros_like(V)
        dV[:, 0] = self.weight(np.asarray(x)) * (2.0 * V[:, 0] if self.square else 1.0)
        zeros = np.zeros_like(V)
        return {"Variable": dV, "Derivative": zeros, "Flux": zeros,
                "Aux": np.zeros((len(x), 0)), "Scalars": np.zeros(0)}

    def dSigma(self, i, states, x):
        return np.zeros((1, len(x)))

    def dSources(self, i, states, x):
        return np.zeros((1, len(x)))

    def getName(self, p):
        return "none"


class Sweeps:
    """Counts the physics a run asks for, in points and in whole sweeps.

    The cases here are pointwise, so MaNTA's batched calls reach them one point
    at a time; a sweep visits its points in increasing x, so a call at an x no
    larger than the previous one starts a new sweep.
    """

    def reset_counts(self):
        self.points = {"flux": 0, "deriv": 0}
        self.sweeps = {"flux": 0, "deriv": 0}
        self._last = {"flux": None, "deriv": None}

    def tick(self, which, x):
        if self._last[which] is None or x <= self._last[which]:
            self.sweeps[which] += 1
        self._last[which] = x
        self.points[which] += 1


def counted(cls, objective, derivative="dSigmaFn_dq", **kwargs):
    """`cls` with an objective, adaptable in place, and its physics counted."""
    parent_derivative = getattr(cls, derivative)

    class Case(cls, Sweeps):
        regrid = manta.Regrid.InPlace

        def __init__(self):
            cls.__init__(self, **kwargs)
            self.reset_counts()

        # The benchmarks' own counters too, which their hooks increment.
        def reset_counts(self):
            if hasattr(cls, "reset_counts"):
                cls.reset_counts(self)
            Sweeps.reset_counts(self)

        def createAdjointProblem(self):
            return objective()

        def SigmaFn(self, index, state, x, t):
            self.tick("flux", x)
            return cls.SigmaFn(self, index, state, x, t)

    def counted_derivative(self, index, state, x, t):
        self.tick("deriv", x)
        return parent_derivative(self, index, state, x, t)

    setattr(Case, derivative, counted_derivative)
    return Case


class ManufacturedNonlinear(manta.TransportSystem):
    """-d/dx[(1 + u^2) u'] = S, u = sin(pi x) on [0, 1], zero Dirichlet ends."""

    def __init__(self):
        manta.TransportSystem.__init__(self, manta.numbered_spec(1))

    def SigmaFn(self, i, state, x, t):
        u = state.u[0]
        return (1.0 + u * u) * state.q[0]

    def Sources(self, i, state, x, t):
        s, c = np.sin(np.pi * x), np.cos(np.pi * x)
        return np.pi**2 * s * (1.0 + s * s) - 2.0 * np.pi**2 * s * c * c

    def dSigmaFn_du(self, i, state, x, t):
        return np.array([2.0 * state.u[0] * state.q[0]])

    def dSigmaFn_dq(self, i, state, x, t):
        return np.array([1.0 + state.u[0] ** 2])

    def LowerBoundary(self, i, t):
        return 0.0

    def UpperBoundary(self, i, t):
        return 0.0

    def InitialValue(self, i, x):
        return 0.5 * x * (1.0 - x)

    def InitialDerivative(self, i, x):
        return 0.5 - x


class AxisSingular(manta.TransportSystem):
    """-u'' = (4/9) x^(-2/3), u = x - x^(4/3) on [0, 1]: Shestakov's x^(4/3) at the
    axis, in a linear problem with a closed form, G = int u dx = 1/14."""

    def __init__(self):
        manta.TransportSystem.__init__(self, manta.numbered_spec(1))

    def SigmaFn(self, i, state, x, t):
        return state.q[0]

    def Sources(self, i, state, x, t):
        return (4.0 / 9.0) * x ** (-2.0 / 3.0)

    def dSigmaFn_dq(self, i, state, x, t):
        return np.ones(1)

    def LowerBoundary(self, i, t):
        return 0.0

    def UpperBoundary(self, i, t):
        return 0.0

    def InitialValue(self, i, x):
        return 0.0

    def InitialDerivative(self, i, x):
        return 0.0


class WallSingular(AxisSingular):
    """The same singularity at the other end: u = (1 - x) - (1 - x)^(4/3)."""

    def Sources(self, i, state, x, t):
        return (4.0 / 9.0) * (1.0 - x) ** (-2.0 / 3.0)


def shestakov_inventory():
    n = lambda x: ExactSolution(x, N_B)  # noqa: E731
    # Split at the source edge, where the closed form changes branch.
    return (integrate.quad(n, 0.0, 0.1, epsabs=1e-15, epsrel=1e-14)[0] +
            integrate.quad(n, 0.1, LX, epsabs=1e-15, epsrel=1e-14)[0])


PROBLEMS = {
    "manufactured": dict(
        make=counted(ManufacturedNonlinear, lambda: Weighted(square=True), "dSigmaFn_du"),
        exact=0.5, extra={"UpperBoundary": 1.0}),
    "Park": dict(
        make=counted(ParkConvergence, lambda: Weighted(lambda x: x)),
        exact=(np.e - 2.0) / 2.0, extra={"UpperBoundary": 1.0}),
    "axis": dict(
        make=counted(AxisSingular, lambda: Weighted()),
        exact=1.0 / 14.0, extra={"UpperBoundary": 1.0}),
    "wall": dict(
        make=counted(WallSingular, lambda: Weighted()),
        exact=1.0 / 14.0, extra={"UpperBoundary": 1.0}),
    "Shestakov": dict(
        make=counted(ShestakovNonlinear, lambda: Weighted(), n_b=N_B),
        exact=shestakov_inventory(),
        extra={"UpperBoundary": LX, "SuppressAlgebraicError": True}),
}

# problem, driver, cells, starting degree
SETUPS = [
    ("manufactured", "DegreeAdaptation", 4, 1),
    ("Park", "DegreeAdaptation", 4, 1),
    ("Park", "MeshAdaptation", 10, 3),
    ("axis", "MeshAdaptation", 10, 3),
    ("Shestakov", "MeshAdaptation", 10, 3),
    ("Shestakov", "DegreeAdaptation", 10, 3),
]


@contextlib.contextmanager
def captured(into):
    """The solver's own progress, which goes to C stdout, into a string."""
    sys.stdout.flush()
    saved = os.dup(1)
    with tempfile.TemporaryFile(mode="w+b") as tmp:
        os.dup2(tmp.fileno(), 1)
        try:
            yield
        finally:
            ctypes.CDLL(None).fflush(None)
            os.dup2(saved, 1)
            os.close(saved)
            tmp.seek(0)
            into.append(tmp.read().decode(errors="replace"))


def run(problem, driver, cells, k0, tol, points=None, **settings):
    p = PROBLEMS[problem]
    case = p["make"]()
    runner = manta.Runner(case)
    mesh = {"GridPoints": points} if points is not None else {"GridSize": cells}
    cfg = {
        "OutputFilename": "objective_adaptation",
        "PolynomialDegree": k0,
        **mesh,
        "LowerBoundary": 0.0,
        "delta_t": 1.0,
        "Relative_tolerance": 1e-8,
        "Absolute_tolerance": 1e-10,
        "SteadyStateSolve": True,
        "SteadyStateSolver": "Newton",
        "solveAdjoint": True,
        "MaxPolynomialDegree": 12,
        "ObjectiveTolerance": tol,
        "WriteOutput": False,
        driver: True,
    }
    cfg.update(p["extra"])
    cfg.update(settings)
    log = []
    with captured(log):
        runner.configure(cfg)
        case.reset_counts()
        runner.run_ss()
    text = log[0]
    G = float(np.asarray(runner.G())[0])
    widths = np.diff(np.asarray(runner.getCellBoundaries()))
    return {
        "err": abs(G - p["exact"]) / abs(p["exact"]),
        "ks": list(dict.fromkeys(int(k) for k in re.findall(r"^  k = (\d+):", text, re.M))),
        "ceiling": "stopped at the ceiling" in text,
        "cells": len(widths),
        "graded": widths.min() / widths.max() < 0.5,
        "hmin": widths.min() / widths.sum(),
        "sweeps": case.sweeps["flux"] + case.sweeps["deriv"],
        "points": case.points["flux"] + case.points["deriv"],
    }


def objective(problem, driver, cells, k0, tol, points=None):
    return run(problem, driver, cells, k0, tol, points, AdaptationTarget="Objective")


def solution(problem, driver, cells, k0, tol, degree_tolerance=None, points=None):
    # The same steady test as the objective target's, and no estimate: the
    # solution target reads none.
    kw = {"SteadyStateTest": "Objective", "EstimateObjectiveOnFinish": False}
    if degree_tolerance is not None:
        kw["DegreeTolerance"] = degree_tolerance
    return run(problem, driver, cells, k0, tol, points, **kw)


def describe(r):
    if r is None:
        return "not reachable at any DegreeTolerance down to 1e-12"
    mesh = (f"{r['cells']} cells, graded to {r['hmin']:.1e}" if r["graded"]
            else f"{r['cells']} uniform")
    met = "met" if r["err"] <= r["tol"] else ("ceiling" if r["ceiling"] else "MISSED")
    return (f"{r['err']:.1e} ({met:7s}) k {'/'.join(map(str, r['ks']))}, {mesh}, "
            f"{r['sweeps']} sweeps, {r['points']} points")


def table():
    for problem, driver, cells, k0 in SETUPS:
        for tol in TOLERANCES:
            print(f"\n{problem}, {driver} from k = {k0} on {cells} cells, eps_G = {tol:.0e}")
            o = objective(problem, driver, cells, k0, tol)
            o["tol"] = tol
            print(f"  {'Objective':>26}: {describe(o)}")
            d = solution(problem, driver, cells, k0, tol)
            d["tol"] = tol
            print(f"  {'Solution, 1e-6':>26}: {describe(d)}")
            tuned, at = None, None
            for dtol in SCAN:
                r = solution(problem, driver, cells, k0, tol, dtol)
                if r["err"] <= tol:
                    tuned, at = r, dtol
                    tuned["tol"] = tol
                    break
            label = f"Solution, tuned {at:.0e}" if at else "Solution, tuned"
            print(f"  {label:>26}: {describe(tuned)}", flush=True)


def sweep():
    """Over problems, drivers, meshes, starting degrees and tolerances."""
    cases = [(p, d) for p in ("manufactured", "Park") for d in ("DegreeAdaptation", "MeshAdaptation")]
    cases += [("axis", "MeshAdaptation"), ("wall", "MeshAdaptation"), ("Shestakov", "MeshAdaptation")]
    counts = {"met": 0, "ceiling, met anyway": 0, "ceiling": 0, "MISSED": 0}
    worst = 0.0
    for (problem, driver), cells, k0, tol in (
            (c, n, k, t) for c in cases for n in (5, 10, 20) for k in (3, 4)
            for t in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7)):
        r = objective(problem, driver, cells, k0, tol)
        ok = r["err"] <= tol
        key = ("ceiling, met anyway" if ok else "ceiling") if r["ceiling"] else ("met" if ok else "MISSED")
        counts[key] += 1
        if not r["ceiling"]:
            worst = max(worst, r["err"] / tol)
        print(f"  {problem:>12} {driver:>16} N={cells:2d} k0={k0} eps {tol:.0e}: {key:20s} "
              f"err {r['err']:.1e} k {r['ks']} {'graded' if r['graded'] else 'uniform'}", flush=True)
    print(counts, f"worst converged err / eps_G = {worst:.2f}")


def graded(end, fraction, ratio, cells, layer=3):
    """`layer` cells at `ratio` in `fraction` of the domain at `end`, the rest uniform."""
    inner = [fraction * ratio ** j for j in range(layer - 1, -1, -1)]
    points = np.array([0.0] + inner + list(np.linspace(fraction, 1.0, cells - layer + 1))[1:])
    return list(points if end == "lower" else 1.0 - points[::-1])


def start_meshes():
    """Graded at either end, graded deeply (the mesh a uniform start's grading
    leaves), and random."""
    meshes = {}
    for end in ("lower", "upper"):
        for fraction in (0.1, 0.2, 0.3):
            for ratio in (0.3, 0.5):
                for cells in (6, 10):
                    meshes[f"graded {end} {fraction} at {ratio}, {cells} cells"] = graded(
                        end, fraction, ratio, cells)
        meshes[f"graded {end} 0.1 at 0.3, 9 of 10 cells"] = graded(end, 0.1, 0.3, 10, 9)
    rng = np.random.default_rng(1)
    for cells in (4, 6, 10):
        for seed in range(3):
            meshes[f"random {cells} cells, {seed}"] = [0.0] + list(np.sort(rng.uniform(0, 1, cells - 1))) + [1.0]
    return meshes


def starts():
    """MeshAdaptation from non-uniform meshes, both targets, at 1e-5 and 1e-7."""
    counts = {target: {"met": 0, "ceiling, met anyway": 0, "ceiling": 0, "MISSED": 0, "failed": 0}
              for target in ("Objective", "Solution, 1e-6")}
    for problem in PROBLEMS:
        for label, points in start_meshes().items():
            for tol in (1e-5, 1e-7):
                for target in counts:
                    try:
                        if target == "Objective":
                            r = objective(problem, "MeshAdaptation", len(points) - 1, 3, tol, points)
                        else:
                            r = solution(problem, "MeshAdaptation", len(points) - 1, 3, tol, points=points)
                    except RuntimeError as e:
                        counts[target]["failed"] += 1
                        print(f"  {problem:>12} {label:>34} eps {tol:.0e} {target:>15}: failed, {e}"[:160])
                        continue
                    ok = r["err"] <= tol
                    key = (("ceiling, met anyway" if ok else "ceiling") if r["ceiling"]
                           else ("met" if ok else "MISSED"))
                    counts[target][key] += 1
                    print(f"  {problem:>12} {label:>34} eps {tol:.0e} {target:>15}: {key:20s} "
                          f"err {r['err']:.1e} {r['cells']} cells", flush=True)
    for target, c in counts.items():
        print(target, c)


if __name__ == "__main__":
    if "--sweep" in sys.argv:
        sweep()
    elif "--starts" in sys.argv:
        starts()
    else:
        table()
