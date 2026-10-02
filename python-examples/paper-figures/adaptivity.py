"""What MeshAdaptation does, and costs, for the paper's adaptivity section.

Two parts.

1. A Dirichlet wall layer with a closed-form steady state -- the case MaNTA's
   adaptive driver was measured on in MESH-REFINEMENT.md section 12:

       -d/dx[ 2 x D u^n u' ] = H exp(-x^2/W),   sigma(0) = 0,   u(1) = u_b,

   whose layer at x = 1 is about 2 D u_b^(n+1) / ((n+1) sigma(1)) wide -- 5e-4 at
   n = 2.5. Uniform against graded at the same cell count and degree, with and
   without the degree loop, and with tauScaling = "Diffusive".

2. The three benchmarks of the paper at the resolutions of tab:steady, as the
   negative (Park, Jardin) and positive (Shestakov) controls for the decision to
   grade.

Cost is physics point-evaluations -- flux plus derivative -- per collocation point
of the final discretisation, the unit of PERFORMANCE.md; every solve the adaptive
driver makes is in it. Run against a build's package directory:

    PYTHONPATH=$BUILD_DIR/python python3 adaptivity.py
"""
import ctypes
import os
import pathlib
import re
import sys
import tempfile

import numpy as np
from netCDF4 import Dataset
from scipy.integrate import quad
from scipy.special import erf

import manta

EX = str(pathlib.Path(__file__).resolve().parent.parent)

# ------------------------------------------------------------------ wall layer --

H, W, D, UB = 0.1, 0.2, 0.01, 0.2


def sigma_exact(x):
    return H * np.sqrt(np.pi * W) / 2 * erf(np.asarray(x) / np.sqrt(W))


def u_exact(x, n):
    f = lambda s: sigma_exact(s) / s if s > 0 else H  # noqa: E731
    integral = np.array([quad(f, xi, 1.0, epsabs=1e-14, epsrel=1e-13)[0] for xi in x])
    return (UB ** (n + 1) + (n + 1) / (2 * D) * integral) ** (1 / (n + 1))


class WallLayer(manta.TransportSystem):
    """sigma_hat = 2 x D u^n q, a Gaussian source, zero flux on the axis."""

    variables = [manta.Field("u", "", "", lower=manta.Mixed(d=1.0))]

    # InPlace: nothing here depends on where the case is evaluated -- the
    # counters count calls, whatever the points -- so MeshAdaptation and
    # DegreeAdaptation may move one instance between meshes and degrees.
    regrid = manta.Regrid.InPlace

    def __init__(self, n):
        super().__init__()
        self.n = n
        self.nFlux = self.nDeriv = 0

    # u^n in numpy floats: a Newton iterate below zero gives NaN, which the solver
    # rejects as a failed step, rather than a complex number or an |u|^n that lets
    # the iterate wander on into u < 0.
    def SigmaFn(self, i, s, x, t):
        self.nFlux += 1
        return 2 * x * D * np.float64(s.u[0]) ** self.n * s.q[0]

    def dSigmaFn_du(self, i, s, x, t):
        return np.array([2 * x * D * self.n * np.float64(s.u[0]) ** (self.n - 1) * s.q[0]])

    def dSigmaFn_dq(self, i, s, x, t):
        self.nDeriv += 1
        return np.array([2 * x * D * np.float64(s.u[0]) ** self.n])

    def Sources(self, i, s, x, t):
        return H * np.exp(-x * x / W)

    def LowerBoundary(self, i, t):
        return 0.0

    def UpperBoundary(self, i, t):
        return UB

    def InitialValue(self, i, x):
        return UB

    def InitialDerivative(self, i, x):
        return 0.0


# Dense evaluation points, packed geometrically into the layer and never on a face.
XE = np.unique(np.concatenate([np.linspace(0, 1, 2001)[1:-1] + 1 / 4000,
                               1 - np.geomspace(1e-7, 0.05, 400) * (1 + 1e-9)]))
XE = XE[(XE > 0) & (XE < 1)]


def captured(fn):
    """Run fn() with the solver's stdout captured; return (result, error, log)."""
    libc = ctypes.CDLL(None)
    libc.fflush(None)
    fd, path = tempfile.mkstemp()
    saved = os.dup(1)
    os.dup2(fd, 1)
    try:
        result, error = fn(), None
    except Exception as e:  # noqa: BLE001
        result, error = None, str(e)[:90]
    finally:
        libc.fflush(None)
        os.dup2(saved, 1)
        os.close(fd)
    log = pathlib.Path(path).read_text()
    os.remove(path)
    return result, error, log


def decision(log):
    m = re.search(r"roughness vs interior: lower ([\d.]+|inf)x, upper ([\d.]+|inf)x.*-> grade (\w+)", log)
    return (m.group(3), float(m.group(1)), float(m.group(2))) if m else ("-", None, None)


def final_degree(log, k0):
    ks = re.findall(r"converged at k = (\d+)|stopped at the ceiling k = (\d+)", log)
    return int(next(a or b for a, b in ks[-1:])) if ks else k0


def wall_layer(n, cells, k, extra):
    case = WallLayer(n)
    runner = manta.Runner(case)
    cfg = {"PolynomialDegree": k, "GridSize": cells, "LowerBoundary": 0.0, "UpperBoundary": 1.0,
           "Relative_tolerance": 1e-6, "Absolute_tolerance": [1e-6], "initialTimestep": 1e-3,
           "MinStepSize": 1e-12, "SteadyStateSolver": "PseudoTransient",
           "SteadyStateTolerance": 1e-9, "delta_t": 1.0, "OutputFilename": "adaptivity",
           "WriteOutput": True}
    cfg.update(extra)
    runner.configure(cfg)
    _, error, log = captured(runner.run_ss)
    if error:
        return dict(error=error)
    u = np.asarray(runner.getSolution(0, list(XE))).ravel()
    ue = u_exact(XE, n)
    with Dataset("adaptivity.nc") as nc:
        sigma_h = np.array(nc.groups["u"].variables["sigma"][-1, :])
    verdict, lo, hi = decision(log)
    faces = np.asarray(runner.getCellBoundaries())
    k_final = final_degree(log, k)
    return dict(
        verdict=verdict, ratios=(lo, hi), faces=faces, k=k_final,
        linf=float(np.max(np.abs(u - ue)) / np.max(ue)),
        pointwise=float(np.max(np.abs(u - ue) / ue)),
        wall_flux=float(abs(sigma_h[-1] - sigma_exact(1.0)) / sigma_exact(1.0)),
        visits=(case.nFlux + case.nDeriv) / (cells * (k_final + 1)),
        evals=case.nFlux + case.nDeriv,
    )


WALL_SETUPS = {
    "uniform":                {"Superconvergent": True},
    "graded":                 {"MeshAdaptation": True, "MaxPolynomialDegree": 4},
    "graded + p (1e-8)":      {"MeshAdaptation": True, "DegreeTolerance": 1e-8},
    "graded, 20% layer":      {"MeshAdaptation": True, "MaxPolynomialDegree": 4,
                               "UpperBoundaryFraction": 0.2},
    "uniform, Diffusive tau": {"Superconvergent": True, "tauScaling": "Diffusive",
                               "tauUpdate": "ContinuationStep"},
    "graded, Diffusive tau":  {"MeshAdaptation": True, "MaxPolynomialDegree": 4,
                               "tauScaling": "Diffusive", "tauUpdate": "ContinuationStep"},
}


def part1():
    print("Wall layer, k = 4. Errors relative to the exact steady state; cost per point")
    print("of the final discretisation, every solve included, and relative to the")
    print("uniform solve on the same cells.")
    print(f"  {'n':>4} {'cells':>5} {'setup':<24} {'decision':<16} {'k':>2} {'wall cell':>9} "
          f"{'L_inf':>8} {'pointwise':>9} {'wall flux':>9} {'visits':>7} {'cost':>6}")
    for n in (1.0, 1.75, 2.5):
        for cells in (4, 5, 6):
            base = None
            for name, extra in WALL_SETUPS.items():
                r = wall_layer(n, cells, 4, extra)
                if "error" in r:
                    print(f"  {n:4.2f} {cells:5d} {name:<24} FAILED {r['error']}")
                    continue
                base = base or r["evals"]
                lo, hi = r["ratios"]
                dec = f"{r['verdict']} ({hi:.2f}x)" if hi is not None else "-"
                print(f"  {n:4.2f} {cells:5d} {name:<24} {dec:<16} {r['k']:2d} "
                      f"{r['faces'][-1] - r['faces'][-2]:9.2e} {r['linf']:8.2e} "
                      f"{r['pointwise']:9.2e} {r['wall_flux']:9.2e} {r['visits']:7.0f} "
                      f"{r['evals'] / base:5.2f}x")
        print()


# ------------------------------------------------------------------ benchmarks --

def park():
    sys.path.insert(0, f"{EX}/park-convergence")
    from park_convergence import ParkConvergence, ExactSolution
    s = np.linspace(0.0, 1.0, 201)
    return ("Park", 4, 3, ParkConvergence, {"UpperBoundary": 1.0, "delta_t": 1.0e4},
            s, ExactSolution(s))


def jardin():
    sys.path.insert(0, f"{EX}/jardin-critical-gradient")
    from jardin_critical_gradient import JardinCriticalGradient, ExactSolution
    s = np.linspace(0.0, 1.0, 201)
    return ("Jardin", 10, 3, JardinCriticalGradient, {"UpperBoundary": 1.0, "delta_t": 1.0e4},
            s, ExactSolution(s))


def shestakov():
    sys.path.insert(0, f"{EX}/shestakov-nonlinear")
    from shestakov_nonlinear import ShestakovNonlinear, ExactSolution, LX
    s = np.linspace(0.0, LX, 201)
    return ("Shestakov", 10, 3, lambda: ShestakovNonlinear(n_b=0.05),
            {"UpperBoundary": LX, "delta_t": 1.0e3}, s, ExactSolution(s, 0.05))


def benchmark(mkcase, ncells, k, extra, sample, exact, adapt):
    case = mkcase()
    runner = manta.Runner(case)
    cfg = {"OutputFilename": "adaptivity_bench", "PolynomialDegree": k, "GridSize": ncells,
           "LowerBoundary": 0.0, "Relative_tolerance": 1.0e-6, "Absolute_tolerance": 1.0e-3,
           "SteadyStateTolerance": 1.0e-11, "SteadyStateSolver": "PseudoTransient",
           "WriteOutput": False, "Superconvergent": True}
    cfg.update(extra)
    cfg.update(adapt)
    runner.configure(cfg)
    case.reset_counts()
    _, error, log = captured(runner.run_ss)
    if error:
        return dict(error=error)
    u = np.asarray(runner.getSolution(0, list(sample))).reshape(-1)
    k_final = final_degree(log, k)
    verdict, lo, hi = decision(log)
    return dict(verdict=verdict, ratios=(lo, hi), k=k_final,
                error=float(np.sum(np.abs(u - exact)) / np.sum(np.abs(exact))),
                evals=case.nFlux + case.nDeriv,
                visits=(case.nFlux + case.nDeriv) / (ncells * (k_final + 1)))


def part2():
    print("The paper's benchmarks at tab:steady's resolutions, PseudoTransient. Error is")
    print("the relative L1 error of tab:steady; roughness is each end cell's decay-rate")
    print("ratio to the interior median, graded above 2.")
    print(f"  {'problem':>10} {'cells':>5} {'setup':<22} {'roughness lo / hi':<20} {'decision':<8} "
          f"{'k':>2} {'error':>10} {'visits':>7} {'cost':>6}")
    for maker in (park, jardin, shestakov):
        name, ncells, k, mkcase, extra, sample, exact = maker()
        base = None
        for setup, adapt in (("fixed k", {}),
                             ("graded", {"MeshAdaptation": True, "MaxPolynomialDegree": k}),
                             ("graded + p (1e-8)", {"MeshAdaptation": True, "DegreeTolerance": 1e-8}),
                             ("graded, 20% layer", {"MeshAdaptation": True, "MaxPolynomialDegree": k,
                                                    "LowerBoundaryFraction": 0.2,
                                                    "UpperBoundaryFraction": 0.2})):
            r = benchmark(mkcase, ncells, k, extra, sample, exact, adapt)
            if "error" in r and isinstance(r["error"], str):
                print(f"  {name:>10} {ncells:5d} {setup:<22} FAILED {r['error']}")
                continue
            base = base or r["evals"]
            lo, hi = r["ratios"]
            rough = f"{lo:.2f} / {hi:.2f}" if lo is not None else "-"
            print(f"  {name:>10} {ncells:5d} {setup:<22} {rough:<20} {r['verdict']:<8} {r['k']:2d} "
                  f"{r['error']:10.3e} {r['visits']:7.0f} {r['evals'] / base:5.2f}x")
    print()


if __name__ == "__main__":
    part1()
    part2()
