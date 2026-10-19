"""End-to-end tests for the adjoint sensitivity path.

No reference file in the regression or reference-solution suites carries
adjoint output -- it is written only under WriteAdjoints = true -- so neither
compares gradients. `PyRunner::getAdjointGradients` reads `SystemSolver::G_p`
directly, which makes the whole chain testable from Python --

    PyAdjointProblem (gFn / dg / dgFndp / dSigma / dSources)
      -> assembleAdjointOperator, assembleObjectiveGradient
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

u is a quadratic and 0.5 u^2 a quartic, so at PolynomialDegree = 4 both the
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
        "PolynomialDegree": 4,
        "GridSize": 6,
        "LowerBoundary": 0.0,
        "UpperBoundary": 1.0,
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


def _solve_steady(p, tmp_path, mode):
    """As solve(), but reaching the final state through run_ss()."""
    system = ParametricDiffusion(p)
    runner = MaNTA.Runner(system)
    runner.configure(adjoint_config(
        tmp_path, SteadyStateSolver=mode, SteadyStateTolerance=1e-11
    ))
    runner.run_ss()
    G, gradients = runner.getAdjointGradients()
    return float(np.asarray(G)[0]), np.asarray(gradients["G_p"]).reshape(-1)


def _fd_gradient(tmp_path, mode):
    p0 = np.array([KAPPA0, SOURCE0])
    fd = np.zeros(NP)
    for i in range(NP):
        h = 1e-4 * abs(p0[i])
        p_plus, p_minus = p0.copy(), p0.copy()
        p_plus[i] += h
        p_minus[i] -= h
        fd[i] = (
            _solve_steady(p_plus, tmp_path / f"p{i}", mode)[0]
            - _solve_steady(p_minus, tmp_path / f"m{i}", mode)[0]
        ) / (2.0 * h)
    return fd


@pytest.mark.parametrize("mode", ["PseudoTransient", "Newton"])
def test_a_steady_solve_gives_the_adjoint_a_better_state_than_time_marching(tmp_path, mode):
    """The adjoint wants F(y, p) = 0; a steady solve enforces it.

    solve() above reaches its state by integrating to T_FINAL = 15, which its
    own comment explains is chosen so that "the diffusive transient is
    thoroughly dead: the adjoint state method assumes F(y, p) = 0, which only
    holds once du/dt has decayed". run_ss() with TimeMarch stops on a dY/dt
    threshold instead, which is a proxy for the same thing and a looser one --
    it lands the gradient about 2e-5 from the finite-difference reference where
    the long integration reaches 2e-8.

    The pseudo-transient and Newton solvers drive the residual to zero directly,
    so they recover the full 2e-8 while doing a fraction of the work. That makes
    this a correctness argument for the steady path and not only a cost one.
    """
    p0 = np.array([KAPPA0, SOURCE0])
    _, adjoint_grad = _solve_steady(p0, tmp_path / "base", mode)
    fd = _fd_gradient(tmp_path, mode)

    rel = np.abs(adjoint_grad - fd) / np.maximum(np.abs(fd), 1e-300)
    assert np.all(rel < 1e-6), (
        f"mode={mode}\nadjoint={adjoint_grad}\nfinite-difference={fd}\n"
        f"relative error={rel}"
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


@pytest.mark.parametrize(
    "scheme",
    [
        pytest.param({}, id="plain"),
        # k = 1 cannot hold the quadratic steady state and u* can, so the u_h-
        # and u*-based objectives differ by far more than round-off here.
        pytest.param({"Superconvergent": True, "PolynomialDegree": 1}, id="superconvergent"),
    ],
)
def test_G_agrees_with_the_G_from_getAdjointGradients(tmp_path, scheme):
    """Both read the same objective at the same yJac, so they must agree exactly.

    Under Superconvergent that objective is the u*-based one, which is also the
    one objectiveEstimate() reports and bounds.
    """
    p = np.array([KAPPA0, SOURCE0])
    ts = ParametricDiffusion(p)
    runner = MaNTA.Runner(ts)
    runner.configure(adjoint_config(tmp_path / "gboth", **scheme))
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
    """PyAdjointProblem throws, naming the hook, when an override is absent.

    The message matters: these are pure-virtual-in-practice methods with no C++
    fallback, and the failure otherwise surfaces deep inside the solve.

    Which of the three is named is the adjoint solve's business, not this test's
    -- all three are missing, so whichever it reaches first is the right answer.
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
    message = str(excinfo.value)
    assert any(hook in message for hook in ("dg", "dSigma", "dSources")), message


# --------------------------------------------------- spatial parameters --
#
# `spatialParameters = True` gives each parameter its own value at every physics
# node -- the k+1 basis nodes of each cell, or its k+2 star nodes with
# `Superconvergent = True`, which is what `MaNTA.getPhysicsNodes` returns. `G_p`
# is then `(ng * nPoints, np)`, one row per (objective, node) and one column per
# parameter *field*, and each entry is the discrete derivative dG/dp_m.
#
# Two places in accumulateAdjointGradients build that matrix and have to agree
# on it: the explicit term, w_m dg/dp_m from `dgFndp` -- which the hook reports
# as `(np, nPoints)`, so it is transposed on the way in -- and the adjoint term,
# which writes `G_p(gIndex * nPoints + node, pIndex)`. The layout test below
# pins the first with the second zeroed; the value tests pin the second with the
# first zeroed; the explicit-field tests pin the first by differencing G.

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


def test_a_transposed_dgfndp_is_reported_rather_than_aborting(tmp_path):
    """A wrong-way-round dgFndp names itself instead of killing the process.

    Neither wrong shape announces itself on its own. `checkShapeAndSet` is a plain
    assignment outside a DEBUG build, so a mismatched one reaches Eigen and aborts
    the process naming `Block<Matrix<double,-1,-1>,-1,-1,false>` and nothing about
    MaNTA -- and where `np` happens to equal the node count nothing aborts at all:
    the gradient is silently transposed and the run returns a plausible wrong
    answer. So the orientation is checked where it arrives.
    """

    class TransposedSpatialAdjoint(SpatialObjectiveAdjoint):
        def dgFndp(self, gIndex, states, positions):
            return np.asarray(
                super().dgFndp(gIndex, states, positions)
            ).T.copy()

    system = ParametricDiffusion(np.array([KAPPA0, SOURCE0]))
    cfg = adjoint_config(tmp_path, PolynomialDegree=3, GridSize=4)
    nPoints = cfg["GridSize"] * (cfg["PolynomialDegree"] + 1)
    assert nPoints != NP_SPATIAL, (
        "np and nPoints coincide here, so a transpose would be undetectable"
    )

    adjoint = TransposedSpatialAdjoint()
    system.createAdjointProblem = lambda: adjoint

    runner = MaNTA.Runner(system)
    runner.configure(cfg)

    with pytest.raises(RuntimeError) as excinfo:
        runner.run(T_FINAL)

    message = str(excinfo.value)
    assert "dgFndp" in message, message
    assert f"({NP_SPATIAL}, {nPoints})" in message, message
    assert "transpose" in message, message


def cell_weights(points, edges):
    """w with sum_m w_m f(x_m) = int I[f] dx over each cell: the quadrature GFn
    uses, for whichever nodes the physics is evaluated on.

    From the moments of the monomials on the reference cell, so it is built from
    the points alone and shares no code with the solver's weights.
    """
    nCells = len(edges) - 1
    n = len(points) // nCells
    j = np.arange(n)
    moments = (1.0 + (-1.0) ** j) / (j + 1.0)
    w = np.empty(len(points))
    for c in range(nCells):
        a, b = edges[c], edges[c + 1]
        s = (2.0 * points[c * n : (c + 1) * n] - (a + b)) / (b - a)
        w[c * n : (c + 1) * n] = 0.5 * (b - a) * np.linalg.solve(
            np.vander(s, n, increasing=True).T, moments
        )
    return w


@pytest.mark.parametrize("superconvergent", [False, True], ids=["basis", "star"])
def test_spatial_gradients_keep_dgdp_in_the_layout_G_p_uses(tmp_path, superconvergent):
    """G_p is (nPoints, np), with dGFndp weighted and the right way round.

    F_p is zero here, so G_p is the explicit term alone: w_m dg/dp_m in row m.
    The weights are the objective's own quadrature over the physics nodes, which
    is what makes this half of G_p the same derivative as the adjoint half.
    """
    system = ParametricDiffusion(np.array([KAPPA0, SOURCE0]))
    cfg = adjoint_config(
        tmp_path, PolynomialDegree=3, GridSize=4, Superconvergent=superconvergent
    )
    points = np.asarray(
        MaNTA.getPhysicsNodes(0.0, 1.0, cfg["GridSize"], cfg["PolynomialDegree"],
                              superconvergent),
        dtype=float,
    )
    nPoints = len(points)
    assert nPoints == cfg["GridSize"] * (cfg["PolynomialDegree"] + (2 if superconvergent else 1))

    adjoint = SpatialObjectiveAdjoint()
    system.createAdjointProblem = lambda: adjoint

    runner = MaNTA.Runner(system)
    runner.configure(cfg)
    runner.run(T_FINAL)
    _, gradients = runner.getAdjointGradients()
    G_p = np.asarray(gradients["G_p"])

    assert G_p.shape == (nPoints, NP_SPATIAL), G_p.shape
    w = cell_weights(points, np.linspace(0.0, 1.0, cfg["GridSize"] + 1))
    np.testing.assert_allclose(
        G_p, (spatial_marker(NP_SPATIAL, nPoints) * w).T, rtol=1e-12, atol=0.0
    )


# The value tests. Parameter fields the *physics* depends on, so the gradient
# comes out of the adjoint solve rather than out of dgFndp: a diffusivity
# kappa(x), reaching the residual through the flux, and a source S(x).

SPATIAL_KAPPA = 1.0
SPATIAL_S0 = 2.0
SPATIAL_K = 4
SPATIAL_NCELLS = 6
F_KAPPA, F_SOURCE = 0, 1

SCHEMES = pytest.mark.parametrize("superconvergent", [False, True], ids=["basis", "star"])


class NodalFieldDiffusion(MaNTA.TransportSystem):
    """-d_x(kappa(x) d_x u) = S(x), with kappa and S given by one value per node.

    The hooks are evaluated at the physics nodes, which is exactly the array
    `MaNTA.getPhysicsNodes` returns, so the case looks its values up by
    position. `argmin` rather than an equality test because the position makes a
    round trip through C++ as a double.
    """

    def __init__(self, points, kappa_nodes, source_nodes):
        MaNTA.TransportSystem.__init__(self, MaNTA.numbered_spec(1))
        self.points = np.asarray(points, dtype=float)
        self.kappa_nodes = np.asarray(kappa_nodes, dtype=float)
        self.source_nodes = np.asarray(source_nodes, dtype=float)

    def _node(self, x):
        return np.argmin(np.abs(self.points - x))

    def Sources(self, i, state, x, t):
        return float(self.source_nodes[self._node(x)])

    def SigmaFn(self, i, state, x, t):
        return float(self.kappa_nodes[self._node(x)]) * state.q[i]

    def dSigmaFn_dq(self, i, state, x, t):
        return np.full(self.nVars, float(self.kappa_nodes[self._node(x)]))

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
        return NodalFieldAdjoint()


class NodalFieldAdjoint(MaNTA.AdjointProblem):
    """G = int 0.5 u^2 dx against the two nodal fields.

    `np` is 2 -- two *fields* -- and their per-node structure is what
    `spatialParameters` expresses. The hooks report the pointwise derivative at
    every node, d(kappa q)/d kappa = q and dS/dS = 1, and the solver places each
    one at the node it belongs to.
    """

    def __init__(self):
        MaNTA.AdjointProblem.__init__(self)
        self.ng = 1
        self.np = 2
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
        out = np.zeros((self.np, len(positions)))
        out[F_KAPPA, :] = np.asarray(states["Derivative"])[:, i]
        return out

    def dSources(self, i, states, positions):
        out = np.zeros((self.np, len(positions)))
        out[F_SOURCE, :] = 1.0
        return out

    def getName(self, pIndex):
        return ("kappa", "S")[pIndex]


def spatial_points(superconvergent=False):
    return np.asarray(
        MaNTA.getPhysicsNodes(0.0, 1.0, SPATIAL_NCELLS, SPATIAL_K, superconvergent),
        dtype=float,
    )


def spatial_runner(points, kappa_nodes, source_nodes, tmp_path, superconvergent=False):
    system = NodalFieldDiffusion(points, kappa_nodes, source_nodes)
    runner = MaNTA.Runner(system)
    runner.configure(
        adjoint_config(
            tmp_path,
            PolynomialDegree=SPATIAL_K,
            GridSize=SPATIAL_NCELLS,
            Superconvergent=superconvergent,
        )
    )
    runner.run(T_FINAL)
    return runner


def spatial_solve(points, kappa_nodes, source_nodes, tmp_path, superconvergent=False):
    runner = spatial_runner(points, kappa_nodes, source_nodes, tmp_path, superconvergent)
    G, gradients = runner.getAdjointGradients()
    return float(np.asarray(G)[0]), np.asarray(gradients["G_p"])


def uniform_fields(points):
    return SPATIAL_KAPPA * np.ones(len(points)), SPATIAL_S0 * np.ones(len(points))


def test_physics_nodes_are_where_the_physics_is_evaluated(tmp_path):
    """getPhysicsNodes is the basis nodes, or the star nodes, cell by cell."""
    basis = spatial_points(False)
    star = spatial_points(True)
    np.testing.assert_array_equal(
        basis, np.asarray(MaNTA.getNodes(0.0, 1.0, SPATIAL_NCELLS, SPATIAL_K))
    )
    assert star.shape == (SPATIAL_NCELLS * (SPATIAL_K + 2),)
    edges = np.linspace(0.0, 1.0, SPATIAL_NCELLS + 1)
    cells = star.reshape(SPATIAL_NCELLS, SPATIAL_K + 2)
    assert np.all(cells > edges[:-1, None]) and np.all(cells < edges[1:, None])
    assert np.all(np.diff(star) > 0)


@SCHEMES
def test_the_spatial_gradient_sums_to_the_closed_form(tmp_path, superconvergent):
    """Summed over nodes, each spatial gradient is the scalar one.

    With both fields uniform, perturbing every node together is exactly
    perturbing the scalar parameter, and u = S x(1-x) / (2 kappa) is a quadratic
    every scheme here represents exactly. So G = S^2 / (240 kappa^2), and
    sum_j dG/dS_j = S / (120 kappa^2) and sum_j dG/dkappa_j = -S^2 / (120
    kappa^3), with no discretisation error at all. This holds to machine
    precision and is the tightest statement available about the spatial path: it
    pins the total, the node ordering's completeness, and the weighting inside
    F_p all at once -- over the k+2 star nodes with the superconvergent scheme,
    where F_p projects through A9 rather than the mass matrix.
    """
    points = spatial_points(superconvergent)
    _, G_p = spatial_solve(points, *uniform_fields(points), tmp_path, superconvergent)

    assert G_p.shape == (len(points), 2), G_p.shape
    dG_dS = SPATIAL_S0 / (120.0 * SPATIAL_KAPPA**2)
    dG_dkappa = -(SPATIAL_S0**2) / (120.0 * SPATIAL_KAPPA**3)
    assert G_p[:, F_SOURCE].sum() == pytest.approx(dG_dS, rel=1e-12)
    assert G_p[:, F_KAPPA].sum() == pytest.approx(dG_dkappa, rel=1e-12)


@SCHEMES
def test_spatial_adjoint_gradient_matches_finite_differences(tmp_path, superconvergent):
    """dG/dp at a node, from the adjoint, against re-running the solver.

    For the source, to machine precision, and the tolerance is the point. The
    assembled dG/dZ is diag(w) dg/dZ, the exact derivative of the
    sum_m w_m g(Z_m) that GFn reports, and the cell mass matrix M differs from
    diag(w) only by an operator that annihilates constants: an error of that
    kind sums to zero over every cell, leaves G_p.sum() and the scalar path
    intact, and is visible only per node, as an error set by the intra-cell node
    index. So a loose tolerance here does not degrade gracefully -- it stops
    testing the thing that matters. 1e-10 of the largest entry is ~250x the
    round-off this achieves on either set of nodes.

    G is quadratic in S, so a central difference in it is exact. It is not
    quadratic in kappa, and there the reference carries an O(h^2) truncation
    error -- 3.5e-9 of the largest entry at h = 1e-4, 3.5e-7 at 1e-3 -- which
    sets that field's tolerance.

    Compared against the largest entry rather than each node's own because the
    gradient spans two orders of magnitude across the domain, so a per-node
    relative test would weight the nodes next to the Dirichlet boundaries far
    above everything else.

    Only a few nodes are differenced -- each costs two steady-state solves --
    chosen either side of a cell boundary, which is where a layout error in the
    per-node loop would show.
    """
    points = spatial_points(superconvergent)
    kappa0, S0 = uniform_fields(points)
    perCell = len(points) // SPATIAL_NCELLS

    _, G_p = spatial_solve(points, kappa0, S0, tmp_path / "base", superconvergent)

    probe = [0, perCell - 1, perCell, len(points) // 2, len(points) - 1]
    for field, base, h, tol in (
        (F_SOURCE, S0, 0.1 * SPATIAL_S0, 1e-10),
        (F_KAPPA, kappa0, 1e-4 * SPATIAL_KAPPA, 1e-7),
    ):
        adjoint_grad = G_p[:, field]
        scale = np.abs(adjoint_grad).max()
        for j in probe:
            plus, minus = base.copy(), base.copy()
            plus[j] += h
            minus[j] -= h
            fields_plus = (plus, S0) if field == F_KAPPA else (kappa0, plus)
            fields_minus = (minus, S0) if field == F_KAPPA else (kappa0, minus)
            G_plus, _ = spatial_solve(points, *fields_plus, tmp_path / f"p{field}{j}", superconvergent)
            G_minus, _ = spatial_solve(points, *fields_minus, tmp_path / f"m{field}{j}", superconvergent)
            fd = (G_plus - G_minus) / (2.0 * h)

            assert abs(adjoint_grad[j] - fd) < tol * scale, (
                f"{('kappa', 'S')[field]}, node {j} (x = {points[j]:.4f}): "
                f"adjoint = {adjoint_grad[j]}, finite-difference = {fd}, "
                f"largest gradient entry = {scale}"
            )


# ----------------------------------------- a field in the objective itself --
#
# The explicit half of G_p, differenced. A field that enters g and nothing else
# leaves u untouched, so G(p) can be re-evaluated with Runner.G() on the one
# converged state rather than by re-running the solver -- and G is linear in p,
# so the central difference is exact and every node can be checked.


class ExplicitFieldAdjoint(MaNTA.AdjointProblem):
    """G = int (1/2 + p(x)) u^2 dx, with p one value per physics node."""

    def __init__(self, points, p):
        MaNTA.AdjointProblem.__init__(self)
        self.ng = 1
        self.np = 1
        self.np_boundary = 0
        self.spatialParameters = True
        self.points = np.asarray(points, dtype=float)
        self.p = np.asarray(p, dtype=float).copy()

    def _p(self, positions):
        x = np.asarray(positions, dtype=float)
        return self.p[np.argmin(np.abs(self.points[None, :] - x[:, None]), axis=1)]

    def gFn(self, gIndex, states, positions):
        u = np.asarray(states["Variable"])[:, 0]
        return (0.5 + self._p(positions)) * u * u

    def dgFndp(self, gIndex, states, positions):
        u = np.asarray(states["Variable"])[:, 0]
        return (u * u).reshape(1, -1)

    def dg(self, gIndex, states, positions):
        V = np.asarray(states["Variable"])
        zeros = np.zeros_like(V)
        return {
            "Variable": (1.0 + 2.0 * self._p(positions)).reshape(-1, 1) * V,
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
        return "p"


@SCHEMES
def test_a_spatial_field_in_the_objective_is_weighted_by_the_quadrature(tmp_path, superconvergent):
    """dG/dp_m = w_m dg/dp_m, at every node, with the scheme's own weights.

    This is the half of G_p the value tests above zero out. Reported without its
    weight the explicit term is a pointwise derivative beside a discrete one, off
    by 1/w_m -- a factor of 20 to 100 here, set by the intra-cell node index --
    and with the superconvergent scheme it has to be taken on the star nodes, at
    u*, with the star weights b1, or it is the derivative of an objective the
    run does not report. Runner.G() reports the same u*-based objective as
    getAdjointGradients(), which is what lets it serve as the reference.
    """
    points = spatial_points(superconvergent)
    adjoint = ExplicitFieldAdjoint(points, 0.3 * (1.0 + 0.5 * np.sin(7.0 * points)))
    system = NodalFieldDiffusion(points, *uniform_fields(points))
    system.createAdjointProblem = lambda: adjoint

    runner = MaNTA.Runner(system)
    runner.configure(
        adjoint_config(
            tmp_path,
            PolynomialDegree=SPATIAL_K,
            GridSize=SPATIAL_NCELLS,
            Superconvergent=superconvergent,
        )
    )
    runner.run(T_FINAL)
    G, gradients = runner.getAdjointGradients()
    G_p = np.asarray(gradients["G_p"])[:, 0]
    assert float(np.asarray(runner.G())[0]) == float(np.asarray(G)[0])

    h = 0.1
    fd = np.zeros(len(points))
    p0 = adjoint.p.copy()
    for j in range(len(points)):
        adjoint.p = p0.copy()
        adjoint.p[j] += h
        G_plus = float(np.asarray(runner.G())[0])
        adjoint.p[j] -= 2.0 * h
        G_minus = float(np.asarray(runner.G())[0])
        fd[j] = (G_plus - G_minus) / (2.0 * h)

    scale = np.abs(fd).max()
    np.testing.assert_allclose(G_p, fd, rtol=0.0, atol=1e-10 * scale)


class ExplicitScalarAdjoint(DiffusionAdjoint):
    """DiffusionAdjoint with a third parameter c, entering only g = (1/2 + c) u^2.

    Quadratic in u on purpose. u* has the same cell means as u_h, so a term
    linear in u integrates identically on either and could not tell whether the
    explicit derivative was taken at u* or at u_h.
    """

    def __init__(self, transport_system, c):
        super().__init__(transport_system)
        self.np = NP + 1
        self.c = c

    def gFn(self, gIndex, states, positions):
        u = np.asarray(states["Variable"])[:, 0]
        return (0.5 + self.c) * u * u

    def dgFndp(self, gIndex, states, positions):
        out = np.zeros((self.np, len(positions)))
        out[NP, :] = np.asarray(states["Variable"])[:, 0] ** 2
        return out

    def dg(self, gIndex, states, positions):
        d = super().dg(gIndex, states, positions)
        d["Variable"] = d["Variable"] * (1.0 + 2.0 * self.c)
        return d

    def getName(self, pIndex):
        return ("kappa", "source", "c")[pIndex]


def explicit_scalar_gradient(tmp_path, superconvergent):
    """(adjoint dG/dc, central difference of Runner.G() in c)."""
    system = ParametricDiffusion(np.array([KAPPA0, SOURCE0]))
    adjoint = ExplicitScalarAdjoint(system, 0.2)
    system.createAdjointProblem = lambda: adjoint
    runner = MaNTA.Runner(system)
    # k = 1, so u_h is not the quadratic solution and u* differs from it.
    runner.configure(
        adjoint_config(tmp_path, PolynomialDegree=1, GridSize=4, Superconvergent=superconvergent)
    )
    runner.run(T_FINAL)
    _, gradients = runner.getAdjointGradients()

    h = 0.1
    adjoint.c = 0.2 + h
    G_plus = float(np.asarray(runner.G())[0])
    adjoint.c = 0.2 - h
    G_minus = float(np.asarray(runner.G())[0])
    return float(np.asarray(gradients["G_p"])[0, NP]), (G_plus - G_minus) / (2.0 * h)


def test_the_superconvergent_explicit_derivative_is_taken_at_u_star(tmp_path):
    """A parameter in g is differentiated at u* on the star nodes, as G is.

    The guard is the comparison with the basis-node scheme: the two objectives
    differ by 2.3% here, so an explicit derivative taken at u_h with the k+1
    weights would miss by that much rather than by round-off.
    """
    sc_adjoint, sc_fd = explicit_scalar_gradient(tmp_path / "star", True)
    basis_adjoint, basis_fd = explicit_scalar_gradient(tmp_path / "basis", False)

    assert abs(sc_fd - basis_fd) > 1e-2 * abs(sc_fd), (
        "u* and u_h integrate g alike here, so the check below cannot tell them apart"
    )
    assert sc_adjoint == pytest.approx(sc_fd, rel=1e-12)
    assert basis_adjoint == pytest.approx(basis_fd, rel=1e-12)
