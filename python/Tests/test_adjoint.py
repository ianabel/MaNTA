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
    cfg = adjoint_config(tmp_path, Polynomial_degree=3, Grid_size=4)
    nPoints = cfg["Grid_size"] * (cfg["Polynomial_degree"] + 1)
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

    To machine precision, and the tolerance is the point. This used to be a
    thousand times looser, because the assembled dG/dZ applied the cell mass
    matrix to dg/dZ where the objective demands the integration weights: GFn
    reports sum_m w_m g(Z_m), whose exact derivative in the nodal coefficient
    Z_i is w_i dg/dZ|_i, and M is not diag(w). Since M 1 = w exactly, the two
    agree whenever dg/dZ is constant over a cell and differ by
    (M - diag(w)) dg/dZ otherwise -- an operator that annihilates constants, so
    the error summed to zero over every cell and left all of `G_p.sum()`, the
    scalar-parameter path and the closed-form checks intact. It was visible only
    per node, i.e. only through spatial parameters, as an error set purely by the
    intra-cell node index and decaying as O(h^4).

    So a loose tolerance here does not degrade gracefully -- it stops testing the
    thing that was wrong. 1e-10 of the largest entry is ~400x the round-off this
    actually achieves and ~3e6 times tighter than the defect it replaced.

    Compared against the largest entry rather than each node's own because the
    gradient spans two orders of magnitude across the domain -- ~1e-5 at the
    nodes next to the Dirichlet boundaries against ~1.3e-3 mid-domain -- so a
    per-node relative test would weight the two boundary nodes far above
    everything else.

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

        assert abs(adjoint_grad[j] - fd) < 1e-10 * scale, (
            f"node {j} (x = {points[j]:.4f}): adjoint = {adjoint_grad[j]}, "
            f"finite-difference = {fd}, largest gradient entry = {scale}"
        )
