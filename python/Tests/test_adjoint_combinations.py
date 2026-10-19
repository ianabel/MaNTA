"""The adjoint gradient against finite differences, over every combination of
auxiliary variables, global scalars, spatial parameters and the superconvergent
scheme.

Each feature has its own block of the adjoint -- aux rows in F_p and G_y, the
scalar border of the operator, a column per physics node, A9 and u* on the star
nodes -- and a block is easiest to get wrong where it meets another. So one
problem carries all four as switches, and every one of the sixteen settings is
checked the same way:

    -d_x( kappa(x) d_x u ) = S(x) + mu + x nu       on [0, 1],  u(0) = u(1) = 0

    aux:     S reaches the source through phi_S - S(x) = 0, beside a decoy
             phi_u - u = 0 that keeps nAux != nVars;
    scalar:  two algebraic global scalars feeding back into the source,

                 mu - int u dx - int c(x) dx = 0
                 nu - 2 mu - int x^2 u dx   = 0,

             with a parameter c that enters through the first constraint and
             nowhere else;

with three parameters -- kappa, S and c -- each either one number or one value
per physics node, and two objectives:

    G_0 = int u^2 / 2 dx,     G_1 = int (mu + 3 nu) u dx   (int u dx without).

Every route a parameter has into the gradient is then exercised: kappa through
dSigma, S through dSources or dAux, c through dScalarG_dp alone, and the scalars
through the border of the operator and, in G_1, through dg's Scalars entry.
Without that border the adjoint treats the scalars as constants and misses the
share of the gradient that comes through them.

Two scalars rather than one, and nothing symmetric between them. With a single
scalar every block of the border is 1 x 1, and a transposition in it -- N for
N^T, or the Schur complement's indices the wrong way round -- would be exact.
So dG_s/dmu_j is [[1, 0], [-2, 1]]; the source reads mu and nu with different
shapes, 1 and x, so the two columns of V are independent; and the second
constraint weights u by x^2 rather than x. Constant weights in the source, or x
in the constraint, make V^T A^-T W exactly symmetric on this problem -- u is
symmetric about x = 1/2, so int x u = int u / 2 -- and the swapped Schur
complement then passes.
"""

import itertools

import numpy as np
import pytest

import manta as MaNTA

from test_adjoint import cell_weights

KAPPA, S0, C0 = 1.5, 2.0, 0.3
K, NCELLS = 4, 6
F_KAPPA, F_S, F_C = 0, 1, 2
FIELDS = ("kappa", "S", "c")
A_S, A_U = 0, 1


class Combined(MaNTA.TransportSystem):
    """The problem above, with the field values looked up by position."""

    def __init__(self, aux, scalar, spatial, points, kappa_nodes, s_nodes, c_nodes):
        MaNTA.TransportSystem.__init__(
            self, MaNTA.numbered_spec(1, nAux=2 if aux else 0, nScalars=2 if scalar else 0)
        )
        self.aux, self.scalar, self.spatial = aux, scalar, spatial
        self.points = np.asarray(points, dtype=float)
        self.kn = np.asarray(kappa_nodes, dtype=float)
        self.sn = np.asarray(s_nodes, dtype=float)
        self.cn = np.asarray(c_nodes, dtype=float)
        edges = np.linspace(0.0, 1.0, NCELLS + 1)
        # int c dx over the physics nodes, with the quadrature that integrates a
        # field defined on them.
        self.c_integral = float(cell_weights(self.points, edges) @ self.cn)

    def _node(self, x):
        return np.argmin(np.abs(self.points - x))

    def _kappa(self, x):
        return float(self.kn[self._node(x)])

    def _S(self, x):
        return float(self.sn[self._node(x)])

    def SigmaFn(self, i, state, x, t):
        return self._kappa(x) * state.q[i]

    def dSigmaFn_dq(self, i, state, x, t):
        return np.full(self.nVars, self._kappa(x))

    def dSigmaFn_du(self, i, state, x, t):
        return np.zeros(self.nVars)

    def Sources(self, i, state, x, t):
        source = state.phi[A_S] if self.aux else self._S(x)
        if self.scalar:
            source += state.scalars[0] + x * state.scalars[1]
        return source

    def dSources_du(self, i, state, x, t):
        return np.zeros(self.nVars)

    def dSources_dq(self, i, state, x, t):
        return np.zeros(self.nVars)

    def dSources_dsigma(self, i, state, x, t):
        return np.zeros(self.nVars)

    # --- aux: phi_S - S(x) = 0 and phi_u - u = 0 --------------------------
    def dSigma_dPhi(self, i, state, x, t):
        return np.zeros(self.nAux)

    def dSources_dPhi(self, i, state, x, t):
        out = np.zeros(self.nAux)
        out[A_S] = 1.0
        return out

    def AuxG(self, i, state, x, t):
        if i == A_S:
            return state.phi[A_S] - self._S(x)
        return state.phi[A_U] - state.u[0]

    def AuxGPrime(self, i, out, state, x, t):
        out.phi[i] = 1.0
        if i != A_S:
            out.u[0] = -1.0

    def InitialAuxValue(self, i, x):
        return self._S(x) if i == A_S else 0.0

    # --- scalars: mu - int u - int c = 0,  nu - 2 mu - int x^2 u = 0 ------
    def dSources_dScalars(self, i, state, x, t):
        return np.array([1.0, x])

    def ScalarG(self, s, states, states_dot, abscissae, weights, phi_boundary, t):
        u = np.asarray(states["Variable"])[:, 0]
        w = np.asarray(weights)
        mu, nu = np.asarray(states["Scalars"])
        if s == 0:
            return float(mu - w @ u - self.c_integral)
        return float(nu - 2.0 * mu - w @ (np.asarray(abscissae) ** 2 * u))

    def ScalarGPrime(self, states, states_dot, abscissae, weights, phi_boundary, t):
        w = np.asarray(weights)
        n = len(w)

        def zeros():
            return {
                "Variable": np.zeros((n, self.nVars)),
                "Derivative": np.zeros((n, self.nVars)),
                "Flux": np.zeros((n, self.nVars)),
                "Aux": np.zeros((n, self.nAux)),
                "Scalars": np.zeros(self.nScalars),
            }

        dG0, dG1 = zeros(), zeros()
        dG0["Variable"][:, 0] = -w
        dG0["Scalars"][:] = [1.0, 0.0]
        dG1["Variable"][:, 0] = -w * np.asarray(abscissae) ** 2
        dG1["Scalars"][:] = [-2.0, 1.0]
        return ([dG0, dG1], [zeros(), zeros()])

    def InitialScalarValue(self, s):
        return 0.0

    def InitialScalarDerivative(self, s, states, states_dot, weights):
        return 0.0

    def LowerBoundary(self, i, t):
        return 0.0

    def UpperBoundary(self, i, t):
        return 0.0

    def InitialValue(self, i, x):
        return 0.0

    def InitialDerivative(self, i, x):
        return 0.0

    def createAdjointProblem(self):
        return CombinedAdjoint(self, self.spatial)


class ObjectiveAdjoint(MaNTA.AdjointProblem):
    """G_0 = int u^2/2 dx and G_1 = int (mu + 3 nu) u dx, against kappa, S
    and c -- everything but the scalar constraints' parameter derivative."""

    def __init__(self, ts, spatial):
        MaNTA.AdjointProblem.__init__(self)
        self.ts = ts
        self.ng = 2
        self.np = 3
        self.np_boundary = 0
        self.spatialParameters = spatial
        self.weights = cell_weights(ts.points, np.linspace(0.0, 1.0, NCELLS + 1))

    def _factor(self, states):
        if not self.ts.scalar:
            return 1.0
        mu, nu = np.asarray(states["Scalars"])
        return float(mu + 3.0 * nu)

    def gFn(self, gIndex, states, positions):
        u = np.asarray(states["Variable"])[:, 0]
        return 0.5 * u * u if gIndex == 0 else self._factor(states) * u

    def dgFndp(self, gIndex, states, positions):
        return np.zeros((self.np, len(positions)))

    def dg(self, gIndex, states, positions):
        V = np.asarray(states["Variable"])
        n = len(positions)
        out = {
            "Variable": V.copy() if gIndex == 0 else np.full_like(V, self._factor(states)),
            "Derivative": np.zeros_like(V),
            "Flux": np.zeros_like(V),
            "Aux": np.zeros((n, self.ts.nAux)),
            "Scalars": np.zeros(self.ts.nScalars),
        }
        if gIndex == 1 and self.ts.scalar:
            # dG_1/dmu = int u dx and dG_1/dnu = 3 int u dx, with the
            # quadrature G_1 is taken with: the physics nodes these states are
            # on, u* there under the superconvergent scheme.
            integral = self.weights @ V[:, 0]
            out["Scalars"][:] = [integral, 3.0 * integral]
        return out

    def dSigma(self, i, states, positions):
        out = np.zeros((self.np, len(positions)))
        out[F_KAPPA] = np.asarray(states["Derivative"])[:, i]
        return out

    def dSources(self, i, states, positions):
        out = np.zeros((self.np, len(positions)))
        if not self.ts.aux:
            out[F_S] = 1.0
        return out

    def dAux(self, i, states, positions):
        out = np.zeros((self.np, len(positions)))
        if i == A_S:
            out[F_S] = -1.0
        return out

    def getName(self, pIndex):
        return FIELDS[pIndex]


class CombinedAdjoint(ObjectiveAdjoint):
    def dScalarG_dp(self, s, states, states_dot, abscissae, weights, phi_boundary, t):
        # d/dc of -int c dx in the first constraint: the physics-node weight at
        # each node for a field, their sum -- the domain length -- for one
        # number. The second constraint does not read the parameters, and says
        # so with zeros.
        if self.spatialParameters:
            out = np.zeros((self.np, len(self.weights)))
            if s == 0:
                out[F_C] = -self.weights
            return out
        out = np.zeros(self.np)
        if s == 0:
            out[F_C] = -self.weights.sum()
        return out


def physics_points(superconvergent):
    return np.asarray(MaNTA.getPhysicsNodes(0.0, 1.0, NCELLS, K, superconvergent), dtype=float)


def config(tmp_path, superconvergent):
    return {
        "PolynomialDegree": K,
        "GridSize": NCELLS,
        "LowerBoundary": 0.0,
        "UpperBoundary": 1.0,
        "delta_t": 0.5,
        "Relative_tolerance": 1e-8,
        "Absolute_tolerance": [1e-10],
        "MinStepSize": 1e-12,
        # A steady solve, not a time march: the problem is linear, so Newton
        # lands on the same state to round-off -- G_p agrees to 7e-16 of its
        # largest entry -- at about a twentieth of the cost, and every gradient
        # here needs a handful of them.
        "SteadyStateTolerance": 1e-9,
        "solveAdjoint": True,
        "Superconvergent": superconvergent,
        "OutputFilename": str(tmp_path / "combined"),
        "WriteOutput": False,
    }


def solve(aux, scalar, spatial, superconvergent, fields, tmp_path):
    """(G, G_p) with G_p as (ng, nRows, np): nRows is 1, or one per node."""
    system = Combined(aux, scalar, spatial, physics_points(superconvergent), *fields)
    runner = MaNTA.Runner(system)
    runner.configure(config(tmp_path, superconvergent))
    runner.run_ss()
    G, gradients = runner.getAdjointGradients()
    G_p = np.asarray(gradients["G_p"])
    return np.asarray(G), G_p.reshape(2, -1, 3)


COMBINATIONS = list(itertools.product((False, True), repeat=4))


def combination_id(c):
    aux, scalar, spatial, sc = c
    return "-".join(
        name
        for name, on in (("aux", aux), ("scalar", scalar), ("spatial", spatial), ("star", sc))
        if on
    ) or "plain"


@pytest.mark.parametrize("combination", COMBINATIONS, ids=combination_id)
def test_the_gradient_matches_finite_differences(tmp_path, combination):
    """Both objectives, every parameter route, at a few nodes either side of a
    cell boundary when the parameters are spatial.

    G is quadratic in S and in c, so a central difference in either is exact:
    the worst disagreement over all sixteen settings is 8e-13 of the largest
    entry, against a tolerance of 1e-9. It is not quadratic in kappa, whose
    reference carries an O(h^2) truncation error -- 1.0e-9 of the largest entry
    at h = 2e-5 kappa, 2.6e-8 at 1e-4 -- against 1e-7. Without the scalars
    nothing reaches c, and its column must be exactly zero.
    """
    aux, scalar, spatial, sc = combination
    points = physics_points(sc)
    n = len(points)
    base = (KAPPA * np.ones(n), S0 * np.ones(n), C0 * np.ones(n))

    _, G_p = solve(aux, scalar, spatial, sc, base, tmp_path / "base")
    assert G_p.shape == (2, n if spatial else 1, 3)

    if not scalar:
        assert np.all(G_p[:, :, F_C] == 0.0)

    perCell = n // NCELLS
    probes = [perCell - 1, perCell, n // 2] if spatial else [None]
    steps = {F_KAPPA: (2e-5 * KAPPA, 1e-7), F_S: (1e-2 * S0, 1e-9), F_C: (1e-2, 1e-9)}
    for field in (F_KAPPA, F_S, F_C) if scalar else (F_KAPPA, F_S):
        h, tol = steps[field]
        scale = np.abs(G_p[:, :, field]).max(axis=1)
        for j in probes:
            G_pm = []
            for sign in (1.0, -1.0):
                fields = [f.copy() for f in base]
                if j is None:
                    fields[field] += sign * h
                else:
                    fields[field][j] += sign * h
                G_pm.append(solve(aux, scalar, spatial, sc, fields, tmp_path / f"{field}{j}{sign}")[0])
            fd = (G_pm[0] - G_pm[1]) / (2.0 * h)
            adjoint = G_p[:, 0 if j is None else j, field]
            for g in range(2):
                assert abs(adjoint[g] - fd[g]) < tol * scale[g], (
                    f"G_{g}, d/d{FIELDS[field]}"
                    + ("" if j is None else f" at node {j} (x = {points[j]:.4f})")
                    + f": adjoint = {adjoint[g]}, finite difference = {fd[g]}, "
                    f"largest entry = {scale[g]}"
                )


def test_a_scalar_system_requires_dScalarG_dp(tmp_path):
    """No hook is not taken to mean no dependence: the run refuses, by name."""

    points = physics_points(False)
    system = Combined(False, True, False, points, *(v * np.ones(len(points)) for v in (KAPPA, S0, C0)))
    adjoint = ObjectiveAdjoint(system, False)
    system.createAdjointProblem = lambda: adjoint
    runner = MaNTA.Runner(system)
    runner.configure(config(tmp_path, False))
    with pytest.raises(Exception, match="dScalarG_dp"):
        runner.run_ss()


@pytest.mark.parametrize("spatial", [False, True], ids=["number", "field"])
def test_a_wrongly_shaped_dScalarG_dp_is_reported(tmp_path, spatial):
    """The shape is checked against spatialParameters where it arrives."""

    class Transposed(CombinedAdjoint):
        def dScalarG_dp(self, *args):
            out = np.asarray(super().dScalarG_dp(*args))
            # (1, np) for one number each, (nPoints, np) for fields.
            return out.reshape(1, -1).copy() if out.ndim == 1 else out.T.copy()

    points = physics_points(False)
    system = Combined(False, True, spatial, points, *(v * np.ones(len(points)) for v in (KAPPA, S0, C0)))
    adjoint = Transposed(system, spatial)
    system.createAdjointProblem = lambda: adjoint
    runner = MaNTA.Runner(system)
    runner.configure(config(tmp_path, False))
    with pytest.raises(RuntimeError, match=r"dScalarG_dp.*must be \(np, "):
        runner.run_ss()


def test_the_corrected_objective_sees_the_scalars(tmp_path):
    """estimateObjective's dG/dy includes dG/dmu, so G_1 corrects to second order.

    One pseudo-transient step from u = 0 at dt = 10 leaves a state whose G is
    off at first order in its distance from the steady state; the Newton
    correction -(dG/dy) J^-1 F removes that, leaving second order. G_1 reads the
    scalars, so its correction needs G_mu . delta_mu: with it the corrected
    error is 2e-3 of the uncorrected one, without it 0.28. G_0 does not read
    them, and is the control.
    """
    points = physics_points(False)
    fields = tuple(v * np.ones(len(points)) for v in (KAPPA, S0, C0))

    def runner(name, **overrides):
        r = MaNTA.Runner(Combined(False, True, False, points, *fields))
        cfg = config(tmp_path / name, False)
        cfg.update(overrides)
        r.configure(cfg)
        return r

    converged = runner("converged")
    converged.run_ss()
    G_true = np.asarray(converged.G())

    partial = runner(
        "partial",
        SteadyStateSolver="PseudoTransient",
        PseudoTransientInitialStep=10.0,
        MaxContinuationSteps=1,
    )
    assert partial.start_steady(True) == MaNTA.SteadyOutcome.OutOfSteps
    estimate = partial.objectiveEstimate()
    partial.abandon_steady()

    value_error = np.abs(np.asarray(estimate["value"]) - G_true)
    corrected_error = np.abs(np.asarray(estimate["corrected"]) - G_true)
    assert np.all(value_error > 1e-4), "the state is too close to steady to tell"
    assert np.all(corrected_error < 1e-2 * value_error), (
        f"uncorrected error {value_error}, corrected {corrected_error}"
    )


def _counted(name):
    def hook(self, *args):
        self.calls += 1
        return getattr(Combined, name)(self, *args)

    return hook


class CountingCombined(Combined):
    """Combined with every physics hook it defines counted."""

    calls = 0


for _name in (
    "SigmaFn", "Sources", "dSigmaFn_dq", "dSigmaFn_du", "dSources_du", "dSources_dq",
    "dSources_dsigma", "dSigma_dPhi", "dSources_dPhi", "AuxG", "AuxGPrime",
    "dSources_dScalars", "ScalarG", "ScalarGPrime",
):
    setattr(CountingCombined, _name, _counted(_name))


@pytest.mark.parametrize("superconvergent", [False, True], ids=["basis", "star"])
def test_an_adjoint_on_demand_with_scalars_asks_the_case_for_nothing(tmp_path, superconvergent):
    """solve_adjoint takes the scalar border from the last Jacobian build.

    The reference is the end-of-run adjoint of a run with solveAdjoint, which
    evaluates its own derivatives at the final state; the on-demand one, on a
    run without, has only the build's -- so a border missing from what the build
    keeps, or not assembled from it, is a different z. The problem is linear,
    so the build's Jacobian is the final state's and the two agree to round-off.
    """
    points = physics_points(superconvergent)
    fields = tuple(v * np.ones(len(points)) for v in (KAPPA, S0, C0))

    reference = MaNTA.Runner(Combined(False, True, False, points, *fields))
    reference.configure(config(tmp_path / "reference", superconvergent))
    reference.run_ss()
    expected = [reference.adjoint_state(g) for g in range(2)]

    case = CountingCombined(False, True, False, points, *fields)
    runner = MaNTA.Runner(case)
    cfg = config(tmp_path / "on-demand", superconvergent)
    cfg["solveAdjoint"] = False
    runner.configure(cfg)
    runner.run_ss()

    case.calls = 0
    runner.solve_adjoint()
    assert case.calls == 0, f"{case.calls} physics calls from solve_adjoint"

    for g in range(2):
        z = runner.adjoint_state(g)
        assert np.abs(z["scalars"]).max() > 0.0
        for key in ("sigma", "q", "u", "lambda", "scalars"):
            scale = np.abs(expected[g][key]).max()
            np.testing.assert_allclose(
                z[key], expected[g][key], rtol=0.0, atol=1e-10 * scale, err_msg=f"G_{g} {key}"
            )
        mu_rhs = runner.objective_gradient(g)["scalars"]
        assert mu_rhs.shape == (2,)


def test_the_objective_steady_test_runs_with_scalars(tmp_path):
    """SteadyStateTest = "Objective" accepts scalars and stops at the right G."""
    points = physics_points(False)
    fields = tuple(v * np.ones(len(points)) for v in (KAPPA, S0, C0))

    converged = MaNTA.Runner(Combined(False, True, False, points, *fields))
    converged.configure(config(tmp_path / "converged", False))
    converged.run_ss()
    G_true = np.asarray(converged.G())

    runner = MaNTA.Runner(Combined(False, True, False, points, *fields))
    cfg = config(tmp_path / "objective", False)
    cfg.update({"SteadyStateTest": "Objective", "ObjectiveTolerance": 1e-6})
    runner.configure(cfg)
    runner.run_ss()

    stats = runner.steadyStats()
    assert stats["objective_test"] <= 1.0
    np.testing.assert_allclose(np.asarray(runner.G()), G_true, rtol=1e-6)
