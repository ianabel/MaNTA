"""End-to-end tests for the adjoint sensitivity path.

This is the least-verified machinery in the project. `WriteAdjoints()` is
commented out at Solver.cpp:350 (commit 57d2652, "adjoint writing doesn't work
for spatial adjoints"), so no run emits the gradients and both the regression
and reference-solution suites skip their adjoint comparison entirely. Until now
nothing checked that `solveAdjointState` and `computeAdjointGradients` produce a
correct number -- only that they ran.

`PyRunner::getAdjointGradients` is a way in that does *not* depend on
WriteAdjoints: it reads `SystemSolver::G_p` directly. That makes the whole chain
testable from Python --

    PyAdjointProblem (gFn / dg / dgFndp / dSigma / dSources)
      -> initializeMatricesForAdjointSolve
      -> solveAdjointState
      -> computeAdjointGradients
      -> G_p

-- against the one reference that cannot be fooled: finite differences of the
objective with respect to the parameters, computed by re-running the solver.

The physics is chosen so the answer is also known in closed form. For

    -d_x( kappa d_x u ) = S   on [0, 1],  u(0) = u(1) = 0

the steady state is u = S x(1-x) / (2 kappa), and for the objective
G = int 0.5 u^2 dx,

    G        =  S^2 / (240 kappa^2)
    dG/dkappa = -S^2 / (120 kappa^3)
    dG/dS     =  S   / (120 kappa^2)

u is a quadratic and 0.5 u^2 a quartic, so at Polynomial_degree = 4 both the
solution and the objective's quadrature are exact and the closed form is a
legitimate reference rather than an approximation.
"""

import numpy as np
import pytest

import MaNTA

KAPPA0 = 1.5
SOURCE0 = 2.0

# Index into the parameter vector. `p` is deliberately a single array so the
# finite-difference driver can perturb one entry without knowing anything about
# the physics.
P_KAPPA, P_SOURCE = 0, 1
NP = 2


def exact_G(kappa, source):
    return source**2 / (240.0 * kappa**2)


def exact_dG(kappa, source):
    return np.array(
        [
            -(source**2) / (120.0 * kappa**3),  # d/d kappa
            source / (120.0 * kappa**2),  # d/d source
        ]
    )


class ParametricDiffusion(MaNTA.TransportSystem):
    """Linear diffusion whose diffusivity and source are the parameters."""

    def __init__(self, p):
        MaNTA.TransportSystem.__init__(self)
        self.nVars = 1
        self.nScalars = 0
        self.nAux = 0
        self.isLowerDirichlet = True
        self.isUpperDirichlet = True
        self.p = np.asarray(p, dtype=float)

    def SigmaFn(self, i, state, x, t):
        return self.p[P_KAPPA] * state["Derivative"][i]

    def Sources(self, i, state, x, t):
        return self.p[P_SOURCE]

    def dSigmaFn_du(self, i, state, x, t):
        return np.zeros(self.nVars)

    def dSigmaFn_dq(self, i, state, x, t):
        return np.full(self.nVars, self.p[P_KAPPA])

    def dSources_du(self, i, state, x, t):
        return np.zeros(self.nVars)

    def dSources_dq(self, i, state, x, t):
        return np.zeros(self.nVars)

    def dSources_dsigma(self, i, state, x, t):
        return np.zeros(self.nVars)

    def LowerBoundary(self, i, t):
        return 0.0

    def UpperBoundary(self, i, t):
        return 0.0

    def InitialValue(self, i, x):
        return 0.0

    def InitialDerivative(self, i, x):
        return 0.0

    def createAdjointProblem(self):
        # Returned as a unique_ptr<AdjointProblem>; PyAdjointProblem inherits
        # py::trampoline_self_life_support so ownership transfers to C++ without
        # the Python object being collected underneath it.
        return DiffusionAdjoint(self)


class DiffusionAdjoint(MaNTA.AdjointProblem):
    """G = int 0.5 u^2 dx, differentiated with respect to (kappa, source).

    Only the vectorised hooks are implemented. `ComputePhysicsDerivatives` is
    deliberately *not* overridden, so the C++ default dispatches to dSigma /
    dSources / dAux -- that fallback branch of PyAdjointProblem is on the path
    here too.

    Note `self.np` is the C++ member `AdjointProblem::np` exposed by
    def_readwrite, not the numpy module.
    """

    def __init__(self, transport_system):
        MaNTA.AdjointProblem.__init__(self)
        self.ts = transport_system
        self.ng = 1
        self.np = NP
        self.np_boundary = 0
        self.spatialParameters = False
        self.call_counts = {"gFn": 0, "dgFndp": 0, "dg": 0, "dSigma": 0, "dSources": 0}

    # --- the objective and its explicit parameter derivative -----------
    def gFn(self, gIndex, states, positions):
        self.call_counts["gFn"] += 1
        u = np.asarray(states["Variable"])[:, 0]
        return 0.5 * u * u

    def dgFndp(self, gIndex, states, positions):
        # g depends on p only through the state, so the *explicit* derivative
        # is zero and the entire gradient has to come out of the adjoint term.
        # That makes this a test of the adjoint solve, not of dGFndp.
        self.call_counts["dgFndp"] += 1
        return np.zeros((self.np, len(positions)))

    def dg(self, gIndex, states, positions):
        # dg/d(state): only dg/du = u is nonzero.
        self.call_counts["dg"] += 1
        V = np.asarray(states["Variable"])
        zeros = np.zeros_like(V)
        return {
            "Variable": V.copy(),
            "Derivative": zeros,
            "Flux": zeros,
            "Aux": np.zeros((len(positions), 0)),
            "Scalars": np.zeros(0),
        }

    # --- dF/dp, as (np x nPoints) blocks -------------------------------
    def dSigma(self, i, states, positions):
        self.call_counts["dSigma"] += 1
        out = np.zeros((self.np, len(positions)))
        out[P_KAPPA, :] = np.asarray(states["Derivative"])[:, i]  # d(kappa q)/d kappa
        return out

    def dSources(self, i, states, positions):
        self.call_counts["dSources"] += 1
        out = np.zeros((self.np, len(positions)))
        out[P_SOURCE, :] = 1.0
        return out

    def getName(self, pIndex):
        return ("kappa", "source")[pIndex]


def adjoint_config(tmp_path, **overrides):
    cfg = {
        # k = 4 makes both the quadratic solution and the quartic objective
        # integrand exact, so the closed-form reference is meaningful.
        "Polynomial_degree": 4,
        "Grid_size": 6,
        "Lower_boundary": 0.0,
        "Upper_boundary": 1.0,
        "delta_t": 0.5,
        # Tight, so the finite-difference reference is limited by the step size
        # rather than by integration error.
        "Relative_tolerance": 1e-8,
        "Absolute_tolerance": [1e-10],
        # The default MinStepSize is 1e-7, which at these tolerances IDA hits
        # while still at t = 0 -- it fails with "the error test failed
        # repeatedly or with |h| = hmin" rather than with anything that points
        # at the step floor. Anything below about 1e-8 works.
        "MinStepSize": 1e-12,
        "solveAdjoint": True,
        "OutputFilename": str(tmp_path / "adjoint_test"),
        "WriteOutput": False,
    }
    cfg.update(overrides)
    return cfg


# Long enough that the diffusive transient (t ~ 1/(kappa pi^2) ~ 0.07) is
# thoroughly dead: the adjoint state method assumes F(y, p) = 0, which only
# holds once du/dt has decayed.
T_FINAL = 15.0


def solve(p, tmp_path, **overrides):
    """Run to steady state and return (G, dG/dp, system)."""
    system = ParametricDiffusion(p)
    runner = MaNTA.Runner(system)
    runner.configure(adjoint_config(tmp_path, **overrides))
    runner.run(T_FINAL)
    G, gradients = runner.getAdjointGradients()
    return np.asarray(G), np.asarray(gradients["G_p"]), system, runner


def objective_only(p, tmp_path):
    G, _, _, _ = solve(p, tmp_path)
    return float(G[0])


# --------------------------------------------------------------- the tests --


def test_objective_matches_the_closed_form(tmp_path):
    """Before trusting a gradient, check the thing being differentiated.

    GFn integrates gFn with the basis's own quadrature weights
    (PyAdjointProblem::GFn). At k = 4 that is exact for the quartic 0.5 u^2, so
    this pins both the steady state and the weights.
    """
    p = np.array([KAPPA0, SOURCE0])
    G, _, _, _ = solve(p, tmp_path)

    assert G.shape == (1,), G.shape
    assert G[0] == pytest.approx(exact_G(*p), rel=1e-6), (
        f"G = {G[0]}, expected {exact_G(*p)}"
    )


def test_adjoint_gradient_matches_finite_differences(tmp_path):
    """The load-bearing test: dG/dp from the adjoint vs. re-running the solver.

    Central differences, so the truncation error is O(h^2). Nothing about the
    adjoint implementation is reused here -- the reference comes purely from
    evaluating the objective at perturbed parameters.
    """
    p0 = np.array([KAPPA0, SOURCE0])
    _, adjoint_grad, _, _ = solve(p0, tmp_path / "base")
    adjoint_grad = adjoint_grad.reshape(-1)

    assert adjoint_grad.shape == (NP,), adjoint_grad.shape

    fd = np.zeros(NP)
    for i in range(NP):
        h = 1e-4 * abs(p0[i])
        p_plus, p_minus = p0.copy(), p0.copy()
        p_plus[i] += h
        p_minus[i] -= h
        fd[i] = (
            objective_only(p_plus, tmp_path / f"p{i}")
            - objective_only(p_minus, tmp_path / f"m{i}")
        ) / (2.0 * h)

    rel = np.abs(adjoint_grad - fd) / np.maximum(np.abs(fd), 1e-300)
    assert np.all(rel < 1e-3), (
        f"adjoint={adjoint_grad}\nfinite-difference={fd}\nrelative error={rel}"
    )


def objective_only_superconvergent(p, tmp_path):
    G, _, _, _ = solve(p, tmp_path, Superconvergent=True)
    return float(G[0])


def test_the_superconvergent_objective_matches_the_closed_form(tmp_path):
    """With the flag on, G is a functional of u* rather than of u_h.

    At k = 4 the exact steady state u = S x(1-x)/(2 kappa) is a quadratic, so u_h
    represents it exactly and the reconstruction -- which is exact for anything of
    degree <= k+1 -- returns the same function. The closed form is therefore still
    the right reference, which makes this a clean check that the u*-based objective
    is wired up correctly rather than a check of the postprocessing's accuracy.
    """
    p = np.array([KAPPA0, SOURCE0])
    G, _, _, _ = solve(p, tmp_path, Superconvergent=True)

    assert G[0] == pytest.approx(exact_G(*p), rel=1e-6), (
        f"G = {G[0]}, expected {exact_G(*p)}"
    )


def test_the_superconvergent_adjoint_gradient_matches_finite_differences(tmp_path):
    """The load-bearing test for the superconvergent adjoint path.

    Both sides use the u*-based objective: the adjoint gradient because G_y now
    contracts dg/du with B12/B11 through the reconstruction, and the reference
    because objective_only_superconvergent re-runs with the flag on. Comparing a
    u*-based gradient against differences of a u_h-based objective would be
    comparing derivatives of two different functionals.
    """
    p0 = np.array([KAPPA0, SOURCE0])
    _, adjoint_grad, _, _ = solve(p0, tmp_path / "base", Superconvergent=True)
    adjoint_grad = adjoint_grad.reshape(-1)

    assert adjoint_grad.shape == (NP,), adjoint_grad.shape

    fd = np.zeros(NP)
    for i in range(NP):
        h = 1e-4 * abs(p0[i])
        p_plus, p_minus = p0.copy(), p0.copy()
        p_plus[i] += h
        p_minus[i] -= h
        fd[i] = (
            objective_only_superconvergent(p_plus, tmp_path / f"p{i}")
            - objective_only_superconvergent(p_minus, tmp_path / f"m{i}")
        ) / (2.0 * h)

    rel = np.abs(adjoint_grad - fd) / np.maximum(np.abs(fd), 1e-300)
    assert np.all(rel < 1e-3), (
        f"adjoint={adjoint_grad}\nfinite-difference={fd}\nrelative error={rel}"
    )


def test_adjoint_gradient_matches_the_closed_form(tmp_path):
    """Independent of the finite-difference check, and of the solver's own state.

    dG/dkappa = -S^2/(120 kappa^3), dG/dS = S/(120 kappa^2). These differ in
    sign and magnitude, so a gradient that had the parameters transposed or a
    sign flipped could not pass.
    """
    p = np.array([KAPPA0, SOURCE0])
    _, grad, _, _ = solve(p, tmp_path)
    grad = grad.reshape(-1)

    expected = exact_dG(*p)
    assert grad == pytest.approx(expected, rel=5e-3), (
        f"adjoint={grad}, closed form={expected}"
    )


def test_the_python_adjoint_hooks_are_all_exercised(tmp_path):
    """Guard against the gradient being right for the wrong reason.

    If, say, dSigma were never called, the gradient would still be correct for
    the source parameter -- so the agreement tests above would pass at a glance.
    Assert that every hook the C++ side is supposed to reach was reached.
    """
    # createAdjointProblem hands the object to C++ as a unique_ptr, so the
    # normal route gives no way to inspect it afterwards. Override the hook with
    # an instance attribute returning an adjoint we keep a reference to --
    # py::trampoline_self_life_support is what makes that safe.
    ts = ParametricDiffusion(np.array([KAPPA0, SOURCE0]))
    adjoint = DiffusionAdjoint(ts)
    ts.createAdjointProblem = lambda: adjoint

    runner = MaNTA.Runner(ts)
    runner.configure(adjoint_config(tmp_path / "counted"))
    runner.run(T_FINAL)
    runner.getAdjointGradients()

    for name, n in adjoint.call_counts.items():
        assert n > 0, f"{name} was never called; counts = {adjoint.call_counts}"


def test_parameter_names_come_from_the_python_subclass(tmp_path):
    """getName is what labels the adjoint output groups in netCDF."""
    adjoint = DiffusionAdjoint(ParametricDiffusion(np.array([KAPPA0, SOURCE0])))
    assert adjoint.getName(P_KAPPA) == "kappa"
    assert adjoint.getName(P_SOURCE) == "source"

    # The C++ defaults, which a subclass that does not override getName gets.
    plain = MaNTA.AdjointProblem()
    assert plain.getName(3) == "p3"
    assert plain.computeUpperBoundarySensitivity(0, 0) is False
    assert plain.computeLowerBoundarySensitivity(0, 0) is False


def test_gradients_are_refused_when_no_adjoint_was_configured(tmp_path):
    """solveAdjoint defaults to false; asking for gradients must say so."""
    runner = MaNTA.Runner(ParametricDiffusion(np.array([KAPPA0, SOURCE0])))
    runner.configure(adjoint_config(tmp_path, solveAdjoint=False))
    runner.run(0.5)

    with pytest.raises(RuntimeError, match="adjoint problem not set"):
        runner.getAdjointGradients()


def test_a_subclass_missing_the_vectorised_hooks_is_reported(tmp_path):
    """PyAdjointProblem::dg throws by name when the override is absent.

    The message matters: these are pure-virtual-in-practice methods with no C++
    fallback, and the failure otherwise surfaces deep inside the solve.
    """

    class NoVectorisedHooks(MaNTA.AdjointProblem):
        def __init__(self):
            MaNTA.AdjointProblem.__init__(self)
            self.ng = 1
            self.np = NP
            self.np_boundary = 0
            self.spatialParameters = False

        def gFn(self, gIndex, states, positions):
            return np.zeros(len(positions))

        def dgFndp(self, gIndex, states, positions):
            return np.zeros((self.np, len(positions)))

        # dg, dSigma, dSources deliberately absent.

    ts = ParametricDiffusion(np.array([KAPPA0, SOURCE0]))
    adjoint = NoVectorisedHooks()
    ts.createAdjointProblem = lambda: adjoint

    runner = MaNTA.Runner(ts)
    runner.configure(adjoint_config(tmp_path))

    with pytest.raises(RuntimeError) as excinfo:
        runner.run(0.5)
    assert "dg" in str(excinfo.value), str(excinfo.value)
