"""Tests for the auxiliary-variable path through a Python transport system.

`nAux > 0` reaches a set of trampoline hooks nothing else touches -- AuxG,
AuxGPrime, dSources_dPhi, dSigma_dPhi and InitialAuxValue -- and it is the path
`test_reference_solutions.py::test_jax_aux_test` is xfail on: the JAX aux
fixture gets demonstrably correct derivatives but IDA's corrector will not
converge at t = 0.

That xfail leaves an open question this file is meant to answer: is the fault in
the C++ aux plumbing (in which case *any* Python aux system fails), or is it
specific to the JAX fixture? The system below is the same reaction-diffusion
problem as the C++ `AuxVarTest` physics case, which passes its regression, but
expressed in plain numpy through the Python interface.
"""

import numpy as np
import pytest

import MaNTA

KAPPA = 1.0


class AuxDiffusion(MaNTA.TransportSystem):
    """d_t u - kappa u'' = a + f(x),  with a = u^2 as an auxiliary variable.

    Modelled on PhysicsCases/AuxVarTest.cpp. The auxiliary variable is a pure
    algebraic constraint, so the whole system is index-1 exactly as the solver
    assumes.
    """

    def __init__(self):
        MaNTA.TransportSystem.__init__(self)
        self.nVars = 1
        self.nScalars = 0
        self.nAux = 1
        self.isLowerDirichlet = True
        self.isUpperDirichlet = True

    # --- flux and sources ----------------------------------------------
    def SigmaFn(self, i, state, x, t):
        return KAPPA * state["Derivative"][i]

    def Sources(self, i, state, x, t):
        # f(x) chosen so the steady state is driven, plus the aux coupling.
        return state["Aux"][0] + 1.0

    def dSigmaFn_du(self, i, state, x, t):
        return np.zeros(self.nVars)

    def dSigmaFn_dq(self, i, state, x, t):
        return np.full(self.nVars, KAPPA)

    def dSources_du(self, i, state, x, t):
        return np.zeros(self.nVars)

    def dSources_dq(self, i, state, x, t):
        return np.zeros(self.nVars)

    def dSources_dsigma(self, i, state, x, t):
        return np.zeros(self.nVars)

    # --- the auxiliary constraint  G = a - u^2 = 0 ----------------------
    def AuxG(self, i, state, x, t):
        return state["Aux"][0] - state["Variable"][0] ** 2

    def AuxGPrime(self, i, state, x, t):
        return {
            "Variable": np.array([-2.0 * state["Variable"][0]]),
            "Derivative": np.zeros(self.nVars),
            "Flux": np.zeros(self.nVars),
            "Aux": np.array([1.0]),
            "Scalars": np.zeros(0),
        }

    def dSources_dPhi(self, i, state, x, t):
        return np.array([1.0])  # d(a + 1)/da

    def dSigma_dPhi(self, i, state, x, t):
        return np.zeros(self.nAux)

    # --- boundaries and initial data ------------------------------------
    def LowerBoundary(self, i, t):
        return 0.0

    def UpperBoundary(self, i, t):
        return 0.0

    def InitialValue(self, i, x):
        return 0.0

    def InitialDerivative(self, i, x):
        return 0.0

    def InitialAuxValue(self, i, x):
        return self.InitialValue(i, x) ** 2


def aux_config(tmp_path, **overrides):
    cfg = {
        "Polynomial_degree": 3,
        "Grid_size": 8,
        "Lower_boundary": 0.0,
        "Upper_boundary": 1.0,
        "delta_t": 0.1,
        "Relative_tolerance": 1e-6,
        "Absolute_tolerance": [1e-8],
        "MinStepSize": 1e-12,
        "OutputFilename": str(tmp_path / "aux_test"),
        "WriteOutput": False,
    }
    cfg.update(overrides)
    return cfg


XS = [0.1 * i for i in range(1, 10)]


def test_a_numpy_aux_system_runs_to_steady_state(tmp_path):
    """The question the JAXAuxTest xfail leaves open.

    If the C++ nAux path were broken for Python systems in general, this would
    fail the same way the JAX fixture does (IDA's corrector giving up at t = 0).
    """
    runner = MaNTA.Runner(AuxDiffusion())
    runner.configure(aux_config(tmp_path))
    runner.run(5.0)

    u = np.asarray(runner.getSolution(0, XS))
    assert np.all(np.isfinite(u))
    assert np.all(u > 0.0), f"a positive source with zero Dirichlet ends: {u}"
    assert u.argmax() not in (0, len(u) - 1)


def test_the_aux_constraint_is_satisfied_by_the_solution(tmp_path):
    """a = u^2 must hold pointwise at the end of the run.

    This is the property the aux rows of the residual enforce, and it is what
    distinguishes "the solve converged" from "the solve converged to the right
    thing". Only u is retrievable through getSolution, so the constraint is
    checked indirectly: the steady state of -kappa u'' = u^2 + 1 with u(0) =
    u(1) = 0 is compared against an independent solve of the same ODE.
    """
    runner = MaNTA.Runner(AuxDiffusion())
    runner.configure(aux_config(tmp_path))
    runner.run(20.0)

    xs = np.linspace(0.0, 1.0, 201)
    u = np.asarray(runner.getSolution(0, list(xs)))

    # Independent reference: solve -kappa u'' = u^2 + 1 by Newton on a fine
    # second-order finite-difference grid. Nothing here is shared with MaNTA.
    n = 2001
    h = 1.0 / (n - 1)
    xr = np.linspace(0.0, 1.0, n)
    ur = np.zeros(n)
    for _ in range(50):
        r = np.zeros(n)
        r[1:-1] = (
            -KAPPA * (ur[2:] - 2 * ur[1:-1] + ur[:-2]) / h**2 - ur[1:-1] ** 2 - 1.0
        )
        if np.max(np.abs(r)) < 1e-12:
            break
        # Tridiagonal Jacobian of the interior equations.
        main = 2 * KAPPA / h**2 - 2 * ur[1:-1]
        off = -KAPPA / h**2 * np.ones(n - 3)
        J = np.diag(main) + np.diag(off, 1) + np.diag(off, -1)
        ur[1:-1] -= np.linalg.solve(J, r[1:-1])

    reference = np.interp(xs, xr, ur)
    err = np.max(np.abs(u - reference))
    assert err < 5e-4, f"max |u - reference| = {err}"


def test_aux_hooks_are_all_reached(tmp_path):
    """Guard against the aux path being skipped rather than working.

    A system whose aux hooks were never called would still solve
    -kappa u'' = 1 and pass a loose sanity check, so count the calls.
    """

    class Counting(AuxDiffusion):
        def __init__(self):
            super().__init__()
            self.counts = {
                "AuxG": 0,
                "AuxGPrime": 0,
                "dSources_dPhi": 0,
                "InitialAuxValue": 0,
            }

        def AuxG(self, i, state, x, t):
            self.counts["AuxG"] += 1
            return super().AuxG(i, state, x, t)

        def AuxGPrime(self, i, state, x, t):
            self.counts["AuxGPrime"] += 1
            return super().AuxGPrime(i, state, x, t)

        def dSources_dPhi(self, i, state, x, t):
            self.counts["dSources_dPhi"] += 1
            return super().dSources_dPhi(i, state, x, t)

        def InitialAuxValue(self, i, x):
            self.counts["InitialAuxValue"] += 1
            return super().InitialAuxValue(i, x)

    system = Counting()
    runner = MaNTA.Runner(system)
    runner.configure(aux_config(tmp_path))
    runner.run(1.0)

    for name, n in system.counts.items():
        assert n > 0, f"{name} was never called; counts = {system.counts}"


class AuxWithoutDerivative(MaNTA.TransportSystem):
    """Declares nAux = 1 and supplies AuxG but not AuxGPrime.

    Written out in full rather than subclassing AuxDiffusion: the check is on
    whether the Python class *has* the attribute, so inheriting and trying to
    hide it would only test attribute lookup.
    """

    def __init__(self):
        MaNTA.TransportSystem.__init__(self)
        self.nVars = 1
        self.nScalars = 0
        self.nAux = 1
        self.isLowerDirichlet = True
        self.isUpperDirichlet = True

    def SigmaFn(self, i, state, x, t):
        return KAPPA * state["Derivative"][i]

    def Sources(self, i, state, x, t):
        return state["Aux"][0] + 1.0

    def dSigmaFn_du(self, i, state, x, t):
        return np.zeros(self.nVars)

    def dSigmaFn_dq(self, i, state, x, t):
        return np.full(self.nVars, KAPPA)

    def dSources_du(self, i, state, x, t):
        return np.zeros(self.nVars)

    def dSources_dq(self, i, state, x, t):
        return np.zeros(self.nVars)

    def dSources_dsigma(self, i, state, x, t):
        return np.zeros(self.nVars)

    def AuxG(self, i, state, x, t):
        return state["Aux"][0] - state["Variable"][0] ** 2

    def dSources_dPhi(self, i, state, x, t):
        return np.array([1.0])

    def dSigma_dPhi(self, i, state, x, t):
        return np.zeros(self.nAux)

    def LowerBoundary(self, i, t):
        return 0.0

    def UpperBoundary(self, i, t):
        return 0.0

    def InitialValue(self, i, x):
        return 0.0

    def InitialDerivative(self, i, x):
        return 0.0

    def InitialAuxValue(self, i, x):
        return 0.0


def test_an_aux_system_missing_auxgprime_fails_rather_than_guessing(tmp_path):
    """nAux > 0 with no AuxGPrime cannot produce a Jacobian; say which method.

    This used to be a hard segfault. `initializeOverrides` inserted whatever
    `py::get_override` returned for the aux hooks -- an empty py::function when
    the method was absent -- and the call site then invoked it, dereferencing
    null partway through the first Jacobian evaluation. Nothing named the
    missing method; the interpreter simply died.

    The check now happens at setup and names every hook that is missing.
    """
    runner = MaNTA.Runner(AuxWithoutDerivative())
    runner.configure(aux_config(tmp_path))

    with pytest.raises(RuntimeError) as excinfo:
        runner.run(0.1)

    message = str(excinfo.value)
    assert "AuxGPrime" in message, message
    assert "nAux" in message, message
