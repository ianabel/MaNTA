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

import manta as MaNTA
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
        MaNTA.TransportSystem.__init__(self, MaNTA.numbered_spec(1))
        self.p = np.asarray(p, dtype=float)

    def SigmaFn(self, i, state, x, t):
        return self.p[P_KAPPA] * state.q[i]

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


def test_G_returns_the_objective_without_an_adjoint_solve(tmp_path):
    """PyRunner::G -- the objective alone, for a driver that needs no gradient.

    The saving is in the *run*: SystemSolver::integrate calls runAdjointSolve()
    whenever solveAdjoint is set, so with solveAdjoint = True the gradients are
    already computed by the time run() returns. Configuring solveAdjoint = False
    skips that, and G() is then the way to get the objective out -- it builds an
    AdjointProblem on demand purely to evaluate GFn.

    The call counts are what prove no adjoint solve happened: dSigma and dSources
    are reached only by computeAdjointGradients.
    """
    ts = ParametricDiffusion(np.array([KAPPA0, SOURCE0]))
    adjoint = DiffusionAdjoint(ts)
    ts.createAdjointProblem = lambda: adjoint

    runner = MaNTA.Runner(ts)
    runner.configure(adjoint_config(tmp_path / "gonly", solveAdjoint=False))
    runner.run(T_FINAL)

    G_only = np.asarray(runner.G())
    assert G_only.shape == (1,), G_only.shape
    assert G_only[0] == pytest.approx(exact_G(KAPPA0, SOURCE0), rel=1e-6)

    assert adjoint.call_counts["dSigma"] == 0, (
        f"an adjoint solve happened; counts = {adjoint.call_counts}"
    )
    assert adjoint.call_counts["dSources"] == 0
    # But GFn was reached, so the number above is not a coincidence.
    assert adjoint.call_counts["gFn"] > 0


def test_G_agrees_with_the_G_from_getAdjointGradients(tmp_path):
    """Both read the same GFn at the same yJac, so they must agree exactly."""
    p = np.array([KAPPA0, SOURCE0])
    ts = ParametricDiffusion(p)
    runner = MaNTA.Runner(ts)
    runner.configure(adjoint_config(tmp_path / "gboth"))
    runner.run(T_FINAL)

    G_full, _ = runner.getAdjointGradients()
    assert float(np.asarray(runner.G())[0]) == float(np.asarray(G_full)[0])


def test_G_is_refused_before_configure():
    """G reads system->yJac, so there has to be a system."""
    runner = MaNTA.Runner(ParametricDiffusion(np.array([KAPPA0, SOURCE0])))

    with pytest.raises(RuntimeError, match="must be configured"):
        runner.G()


def test_aggressive_timesteps_is_accepted_and_does_not_change_the_answer(tmp_path):
    """IDASetEtaMax changes how fast IDA may grow the step, not the answer."""
    p = np.array([KAPPA0, SOURCE0])
    G_default, _, _, _ = solve(p, tmp_path / "eta_default")
    G_eta, _, _, _ = solve(p, tmp_path / "eta_max", aggressiveTimesteps=True)

    assert float(G_eta[0]) == pytest.approx(float(G_default[0]), rel=1e-5)
    assert float(G_eta[0]) == pytest.approx(exact_G(*p), rel=1e-6)


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


# ----------------------------------------------------- the dG/dt gate --
#
# ObjectiveDecreaseTolerance abandons a run after the initial condition is built
# if the objective is already getting worse, so an optimisation sweep pays
# initialisation instead of a whole transport solve for a bad step. The
# convention is that G is maximised, so "worse" means falling.
#
# int 0.5 u^2 dx cannot drive this, which is worth saying because it is the
# objective the rest of this file uses: its dg/du is u, and the initial condition
# here is u = 0, so dG/dt would be exactly zero and neither verdict reachable. The
# gate tests below use an objective linear in u instead.


class SignedIntegralAdjoint(DiffusionAdjoint):
    """G = sign * int u dx, so dG/dt at t = 0 is sign * int du/dt dx.

    Inherits dgFndp, dSigma and dSources unchanged -- the gate does not use them,
    but PyTransportSystem requires them to be present.
    """

    def __init__(self, transport_system, sign):
        super().__init__(transport_system)
        self.sign = sign

    def gFn(self, gIndex, states, positions):
        return self.sign * np.asarray(states["Variable"])[:, 0]

    def dg(self, gIndex, states, positions):
        V = np.asarray(states["Variable"])
        dgdu = np.zeros_like(V)
        dgdu[:, 0] = self.sign
        zeros = np.zeros_like(V)
        return {
            "Variable": dgdu,
            "Derivative": zeros,
            "Flux": zeros,
            "Aux": np.zeros((len(positions), 0)),
            "Scalars": np.zeros(0),
        }


class GateDiffusion(ParametricDiffusion):
    """ParametricDiffusion with a sign-selectable linear objective on it."""

    def __init__(self, p, sign):
        super().__init__(p)
        self.sign = sign

    def createAdjointProblem(self):
        return SignedIntegralAdjoint(self, self.sign)


# The steady state is u = S x(1-x) / (2 kappa), so int u dx = S / (12 kappa) --
# the value a *completed* run's objective must take, and hence the evidence that
# an accepted run really did integrate.
def exact_integral_of_u(kappa, source):
    return source / (12.0 * kappa)


def gate_run(tmp_path, sign, **overrides):
    system = GateDiffusion(np.array([KAPPA0, SOURCE0]), sign)
    runner = MaNTA.Runner(system)
    runner.configure(adjoint_config(tmp_path, **overrides))
    runner.run(T_FINAL)
    # Held so the trampoline's Python object outlives the call.
    return runner, system


def test_the_gate_rejects_a_worsening_objective(tmp_path):
    runner, _ = gate_run(tmp_path, -1.0, ObjectiveDecreaseTolerance=1e-12)

    assert runner.wasRejected()

    dGdt = np.asarray(runner.lastDGdt())
    assert dGdt.shape == (1,), dGdt.shape
    assert dGdt[0] < 0.0, dGdt

    # Rejection leaves the solver at the initial condition rather than
    # synthesising an objective value, so G stays readable and reports G(t0) --
    # here sign * int u dx with u = 0.
    G = np.asarray(runner.G())
    assert np.isfinite(G).all()
    assert G[0] == pytest.approx(0.0, abs=1e-12), G

    # getAdjointGradients() is deliberately not called: the adjoint solve happens
    # inside integrate(), which never ran, so G_p was never computed.


def test_the_gate_passes_an_improving_objective(tmp_path):
    runner, _ = gate_run(tmp_path, +1.0, ObjectiveDecreaseTolerance=1e-12)

    assert not runner.wasRejected()
    assert np.asarray(runner.lastDGdt())[0] > 0.0

    # And it genuinely integrated: G is the steady-state value, not G(t0) = 0.
    G = np.asarray(runner.G())
    assert G[0] == pytest.approx(exact_integral_of_u(KAPPA0, SOURCE0), rel=1e-4), G


def test_without_the_tolerance_nothing_is_rejected(tmp_path):
    """The same worsening objective, gate unarmed: the run must proceed."""
    runner, _ = gate_run(tmp_path, -1.0)

    assert not runner.wasRejected()

    G = np.asarray(runner.G())
    assert G[0] == pytest.approx(-exact_integral_of_u(KAPPA0, SOURCE0), rel=1e-4), G


def test_a_negative_tolerance_is_rejected(tmp_path):
    """Zero means absent, so reaching the setter with zero or less is an error."""
    system = GateDiffusion(np.array([KAPPA0, SOURCE0]), -1.0)
    runner = MaNTA.Runner(system)
    # pybind11 maps std::logic_error to RuntimeError -- only invalid_argument,
    # domain_error and the like become ValueError.
    with pytest.raises(RuntimeError, match="cannot be zero or negative"):
        runner.configure(adjoint_config(tmp_path, ObjectiveDecreaseTolerance=-1.0))


def test_the_gate_verdict_does_not_depend_on_the_output_cadence(tmp_path):
    """Defect 2 of origin/optimize-mode's version, kept out.

    That one compared dt * dG/dt against its threshold with dt the netCDF output
    cadence, so an I/O setting decided whether a step was rejected and
    setOutputCadence(0.0) disarmed the gate entirely. Here delta_t must not reach
    the decision.

    Only the verdict and the sign are compared, not the values: delta_t is also
    IDACalcIC's initial step guess, so the corrected initial condition -- and
    therefore dG/dt -- may differ in the last digits between the two.
    """
    verdicts = []
    for delta_t in (0.5, 50.0):
        runner, _ = gate_run(
            tmp_path, -1.0, delta_t=delta_t, ObjectiveDecreaseTolerance=1e-12
        )
        verdicts.append(runner.wasRejected())
        assert np.asarray(runner.lastDGdt())[0] < 0.0, delta_t

    assert verdicts[0] == verdicts[1], verdicts
    assert all(verdicts), "a worsening objective should be rejected at either cadence"


# --------------------------------------------------- spatial parameters --
#
# `spatialParameters = True` indexes the parameter vector by node: `G_p` becomes
# `(ng * nCells * (k + 1), np)`, one row per (objective, node) and one column per
# parameter *field*. Two places in computeAdjointGradients build that matrix and
# they have to agree on its orientation -- the explicit term assigns `dGFndp`
# into a `(nPoints, np)` block (SystemSolver.cpp:1675), and the adjoint term
# writes `G_p(gIndex * nPoints + node, pIndex)` (SystemSolver.cpp:1814).
#
# PyAdjointProblem::dGFndp returned `dgFndp` raw in the spatial branch. That is
# `(np, nPoints)` -- the orientation the non-spatial branch immediately below it
# indexes as `dgdp(p, ind)` -- so every run with spatial parameters aborted
# inside Eigen's assignment, with `checkShapeAndSet` reduced to a plain
# assignment in a release build and so contributing no message of its own.
#
# Nothing here covered it: every other adjoint fixture in this file sets
# `spatialParameters = False`, and python-examples/adjoints/spatial_adjoints.py,
# the only thing in the tree that sets it True, is collected by no suite. That
# example could not have caught the *orientation* in any case, only the shape --
# its `g` ignores its parameters, so its `dgFndp` is identically zero.

NP_SPATIAL = 3


def spatial_marker(np_, nPoints):
    """A dgFndp with every entry distinct, so orientation is observable.

    Deliberately neither symmetric nor square: a transposed G_p fails here on
    shape, and would still fail on values if nPoints ever equalled np_.
    """
    p = np.arange(np_).reshape(np_, 1)
    j = np.arange(nPoints).reshape(1, nPoints)
    return 1.0 + p + 100.0 * j


class SpatialObjectiveAdjoint(MaNTA.AdjointProblem):
    """dG/dp is a known matrix and nothing else contributes.

    `dSigma` and `dSources` are zero, so `F_p` is zero and the adjoint term adds
    nothing at all -- `G_p` is then exactly `dgFndp` transposed, which makes the
    assertion an equality rather than a tolerance.
    """

    def __init__(self):
        MaNTA.AdjointProblem.__init__(self)
        self.ng = 1
        self.np = NP_SPATIAL
        self.np_boundary = 0
        self.spatialParameters = True

    def gFn(self, gIndex, states, positions):
        u = np.asarray(states["Variable"])[:, 0]
        return 0.5 * u * u

    def dgFndp(self, gIndex, states, positions):
        return spatial_marker(self.np, len(positions))

    def dg(self, gIndex, states, positions):
        V = np.asarray(states["Variable"])
        zeros = np.zeros_like(V)
        return {
            "Variable": V.copy(),
            "Derivative": zeros,
            "Flux": zeros,
            "Aux": np.zeros((len(positions), 0)),
            "Scalars": np.zeros(0),
        }

    def dSigma(self, i, states, positions):
        return np.zeros((self.np, len(positions)))

    def dSources(self, i, states, positions):
        return np.zeros((self.np, len(positions)))

    def getName(self, pIndex):
        return f"field{pIndex}"


def test_spatial_gradients_keep_dgdp_in_the_layout_G_p_uses(tmp_path):
    """G_p is (nPoints, np), and dGFndp has to arrive that way round."""
    system = ParametricDiffusion(np.array([KAPPA0, SOURCE0]))
    cfg = adjoint_config(tmp_path, Polynomial_degree=3, Grid_size=4)
    nPoints = cfg["Grid_size"] * (cfg["Polynomial_degree"] + 1)

    adjoint = SpatialObjectiveAdjoint()
    system.createAdjointProblem = lambda: adjoint

    runner = MaNTA.Runner(system)
    runner.configure(cfg)
    runner.run(T_FINAL)
    _, gradients = runner.getAdjointGradients()
    G_p = np.asarray(gradients["G_p"])

    assert G_p.shape == (nPoints, NP_SPATIAL), G_p.shape
    np.testing.assert_allclose(
        G_p, spatial_marker(NP_SPATIAL, nPoints).T, rtol=0.0, atol=0.0
    )


# The value tests. A parameter field the *physics* depends on, so the gradient
# comes out of the adjoint solve rather than out of dgFndp.

SPATIAL_KAPPA = 1.0
SPATIAL_S0 = 2.0
SPATIAL_K = 4
SPATIAL_NCELLS = 6


class NodalSourceDiffusion(MaNTA.TransportSystem):
    """-d_x(kappa d_x u) = S, with S given by one value per node.

    The hooks are evaluated at the k+1 nodes of each cell, which is exactly the
    array `MaNTA.getNodes` returns, so `Sources` can look its value up by
    position. `argmin` rather than an equality test because the position makes a
    round trip through C++ as a double.
    """

    def __init__(self, points, source_nodes):
        MaNTA.TransportSystem.__init__(self, MaNTA.numbered_spec(1))
        self.points = np.asarray(points, dtype=float)
        self.source_nodes = np.asarray(source_nodes, dtype=float)

    def Sources(self, i, state, x, t):
        return float(self.source_nodes[np.argmin(np.abs(self.points - x))])

    def SigmaFn(self, i, state, x, t):
        return SPATIAL_KAPPA * state.q[i]

    def dSigmaFn_dq(self, i, state, x, t):
        return np.full(self.nVars, SPATIAL_KAPPA)

    def dSigmaFn_du(self, i, state, x, t):
        return np.zeros(self.nVars)

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
        return NodalSourceAdjoint()


class NodalSourceAdjoint(MaNTA.AdjointProblem):
    """G = int 0.5 u^2 dx against the one nodal source field.

    `np` is 1 -- one *field* -- and its per-node structure is what
    `spatialParameters` expresses. dS/dp is 1 at every node, because the value
    at node j is the parameter at node j.
    """

    def __init__(self):
        MaNTA.AdjointProblem.__init__(self)
        self.ng = 1
        self.np = 1
        self.np_boundary = 0
        self.spatialParameters = True

    def gFn(self, gIndex, states, positions):
        u = np.asarray(states["Variable"])[:, 0]
        return 0.5 * u * u

    def dgFndp(self, gIndex, states, positions):
        # g has no explicit parameter dependence: the whole gradient comes out
        # of the adjoint term.
        return np.zeros((self.np, len(positions)))

    def dg(self, gIndex, states, positions):
        V = np.asarray(states["Variable"])
        zeros = np.zeros_like(V)
        return {
            "Variable": V.copy(),
            "Derivative": zeros,
            "Flux": zeros,
            "Aux": np.zeros((len(positions), 0)),
            "Scalars": np.zeros(0),
        }

    def dSigma(self, i, states, positions):
        return np.zeros((self.np, len(positions)))

    def dSources(self, i, states, positions):
        return np.ones((self.np, len(positions)))

    def getName(self, pIndex):
        return "S"


def spatial_points():
    return np.asarray(
        MaNTA.getNodes(0.0, 1.0, SPATIAL_NCELLS, SPATIAL_K), dtype=float
    )


def spatial_solve(points, source_nodes, tmp_path):
    system = NodalSourceDiffusion(points, source_nodes)
    runner = MaNTA.Runner(system)
    runner.configure(
        adjoint_config(
            tmp_path, Polynomial_degree=SPATIAL_K, Grid_size=SPATIAL_NCELLS
        )
    )
    runner.run(T_FINAL)
    G, gradients = runner.getAdjointGradients()
    return float(np.asarray(G)[0]), np.asarray(gradients["G_p"])


def test_the_spatial_gradient_sums_to_the_closed_form(tmp_path):
    """Summed over nodes, the spatial gradient is the scalar one.

    With the field uniform at S0, perturbing every node together is exactly
    perturbing the scalar source, and G is quadratic in the parameters, so
    sum_j dG/dp_j is dG/dS = S / (120 kappa^2) with no discretisation error at
    all. This holds to machine precision and is the tightest statement
    available about the spatial path: it pins the total, the node ordering's
    completeness, and the quadrature weighting inside F_p all at once.
    """
    points = spatial_points()
    _, G_p = spatial_solve(points, SPATIAL_S0 * np.ones(len(points)), tmp_path)

    exact = SPATIAL_S0 / (120.0 * SPATIAL_KAPPA**2)
    assert G_p.shape == (len(points), 1), G_p.shape
    assert G_p[:, 0].sum() == pytest.approx(exact, rel=1e-12), (
        f"sum = {G_p[:, 0].sum()}, expected {exact}"
    )


def test_spatial_adjoint_gradient_matches_finite_differences(tmp_path):
    """dG/dp at a node, from the adjoint, against re-running the solver.

    Compared against the largest entry rather than each node's own, because the
    discrepancy is uniform in *absolute* terms -- at the resolutions measured it
    is the same number at every node to seven figures -- while the gradient
    itself spans two orders of magnitude, being ~1e-5 at the nodes next to the
    Dirichlet boundaries and ~5e-4 mid-domain. A per-node relative test would
    therefore be a test of the two boundary nodes and nothing else.

    That absolute offset is a discretisation error and converges: 1.549e-07 at
    nCells = 4 against 3.091e-08 at nCells = 6, a ratio of 5.01 for a mesh ratio
    of 1.5, i.e. O(h^4). It is not the non-spatial behaviour, where the same
    machinery agrees with finite differences to 2e-8 and with the closed form to
    7e-16 -- so the spatial gradient is consistent rather than exact, and the
    total is exact regardless (see the sum test above). Worth knowing before
    trusting a single node of it.

    Only a few nodes are differenced -- each costs two steady-state solves --
    chosen either side of a cell boundary, which is where a layout error in the
    per-node loop would show.
    """
    points = spatial_points()
    p0 = SPATIAL_S0 * np.ones(len(points))

    _, G_p = spatial_solve(points, p0, tmp_path / "base")
    adjoint_grad = G_p[:, 0]
    scale = np.abs(adjoint_grad).max()

    probe = [0, SPATIAL_K, SPATIAL_K + 1, len(points) // 2, len(points) - 1]
    h = 0.1 * SPATIAL_S0  # G is quadratic in p, so a central difference is exact
    for j in probe:
        plus, minus = p0.copy(), p0.copy()
        plus[j] += h
        minus[j] -= h
        G_plus, _ = spatial_solve(points, plus, tmp_path / f"p{j}")
        G_minus, _ = spatial_solve(points, minus, tmp_path / f"m{j}")
        fd = (G_plus - G_minus) / (2.0 * h)

        assert abs(adjoint_grad[j] - fd) < 1e-3 * scale, (
            f"node {j} (x = {points[j]:.4f}): adjoint = {adjoint_grad[j]}, "
            f"finite-difference = {fd}, largest gradient entry = {scale}"
        )
