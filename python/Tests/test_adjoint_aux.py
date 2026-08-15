"""Adjoint gradients for a system with auxiliary variables.

`test_adjoint.py` covers the adjoint path at `nAux = 0`, and `test_aux.py`
covers the forward path at `nAux > 0`. Nothing covered the intersection, and two
defects lived in it:

  * `initializeMatricesForAdjointSolve` never wrote the `dSigma/dPhi` block of
    the local matrix, so whenever the flux depended on an auxiliary variable the
    matrix it stored was not the transpose of the forward Jacobian that
    `updateMatricesForJacSolve` builds. On the forward side an inconsistent
    Jacobian only costs Newton iterations; here `M.transpose()` *is* the adjoint
    operator, so the gradient was silently wrong.
  * `dGdaux_Vec` asserted its output vector was `nVars * (k + 1)` long while its
    caller sizes it `nAux * (k + 1)`, so `nAux != nVars` aborted the process.
    Nothing defines NDEBUG in any build variant, so that was live in release
    builds too.

The system below is built to make both unmissable.

The physics is the same linear diffusion as `test_adjoint.py`, but routed
through the auxiliary variables:

    sigma_hat = kappa * phi_q,   phi_q - q   = 0
                                 phi_u - u   = 0
    source    = S

`dSigmaFn_dq` is therefore *identically zero* and the whole flux/derivative
coupling passes through the `dSigma/dPhi` block. With that block missing the
adjoint operator loses the coupling entirely and no gradient can be right.
`phi_u` is otherwise unused; it exists so that `nAux = 2 != nVars = 1` and the
`dGdaux_Vec` length contract is exercised.

Because `phi_q = q`, the continuous problem is still

    -d_x( kappa d_x u ) = S   on [0, 1],  u(0) = u(1) = 0

with steady state u = S x(1-x) / (2 kappa), so for G = int 0.5 u^2 dx the same
closed form as `test_adjoint.py` applies and is used here as a second, entirely
independent reference:

    G         =  S^2 / (240 kappa^2)
    dG/dkappa = -S^2 / (120 kappa^3)
    dG/dS     =  S   / (120 kappa^2)
"""

import numpy as np
import pytest

import manta as MaNTA
KAPPA0 = 1.5
SOURCE0 = 2.0

# Parameter vector layout.
P_KAPPA, P_SOURCE = 0, 1
NP = 2

# Auxiliary variable layout: phi_q = q carries the flux, phi_u = u is a decoy
# whose only job is to make nAux differ from nVars.
A_Q, A_U = 0, 1
N_AUX = 2


def exact_G(kappa, source):
    return source**2 / (240.0 * kappa**2)


def exact_dG(kappa, source):
    return np.array(
        [
            -(source**2) / (120.0 * kappa**3),  # d/d kappa
            source / (120.0 * kappa**2),  # d/d source
        ]
    )


class AuxParametricDiffusion(MaNTA.TransportSystem):
    """Linear diffusion whose flux is written in terms of an auxiliary variable."""

    def __init__(self, p):
        MaNTA.TransportSystem.__init__(self, MaNTA.numbered_spec(1, nAux=N_AUX))
        self.p = np.asarray(p, dtype=float)

    # --- flux and sources ------------------------------------------------
    def SigmaFn(self, i, state, x, t):
        # Note: phi_q, not Derivative. This is what puts the flux's dependence
        # on q behind the auxiliary constraint.
        return self.p[P_KAPPA] * state.phi[A_Q]

    def Sources(self, i, state, x, t):
        return self.p[P_SOURCE]

    def dSigmaFn_du(self, i, state, x, t):
        return np.zeros(self.nVars)

    def dSigmaFn_dq(self, i, state, x, t):
        # Identically zero -- see the module docstring.
        return np.zeros(self.nVars)

    def dSources_du(self, i, state, x, t):
        return np.zeros(self.nVars)

    def dSources_dq(self, i, state, x, t):
        return np.zeros(self.nVars)

    def dSources_dsigma(self, i, state, x, t):
        return np.zeros(self.nVars)

    def dSigma_dPhi(self, i, state, x, t):
        out = np.zeros(self.nAux)
        out[A_Q] = self.p[P_KAPPA]
        return out

    def dSources_dPhi(self, i, state, x, t):
        return np.zeros(self.nAux)

    # --- the auxiliary constraints:  phi_q - q = 0,  phi_u - u = 0 --------
    def AuxG(self, i, state, x, t):
        if i == A_Q:
            return state.phi[A_Q] - state.q[0]
        return state.phi[A_U] - state.u[0]

    def AuxGPrime(self, i, out, state, x, t):
        out.phi[i] = 1.0
        if i == A_Q:
            out.q[0] = -1.0
        else:
            out.u[0] = -1.0

    # --- boundaries and initial data -------------------------------------
    def LowerBoundary(self, i, t):
        return 0.0

    def UpperBoundary(self, i, t):
        return 0.0

    def InitialValue(self, i, x):
        return 0.0

    def InitialDerivative(self, i, x):
        return 0.0

    def InitialAuxValue(self, i, x):
        # Consistent with u = 0: both constraints are satisfied at t = 0.
        return 0.0

    def createAdjointProblem(self):
        return AuxDiffusionAdjoint(self)


class AuxDiffusionAdjoint(MaNTA.AdjointProblem):
    """G = int 0.5 u^2 dx, differentiated with respect to (kappa, source)."""

    def __init__(self, transport_system):
        MaNTA.AdjointProblem.__init__(self)
        self.ts = transport_system
        self.ng = 1
        self.np = NP
        self.np_boundary = 0
        self.spatialParameters = False
        self.call_counts = {
            "gFn": 0,
            "dgFndp": 0,
            "dg": 0,
            "dSigma": 0,
            "dSources": 0,
            "dAux": 0,
        }

    # --- the objective and its explicit parameter derivative -------------
    def gFn(self, gIndex, states, positions):
        self.call_counts["gFn"] += 1
        u = np.asarray(states["Variable"])[:, 0]
        return 0.5 * u * u

    def dgFndp(self, gIndex, states, positions):
        # g depends on p only through the state, so the whole gradient has to
        # come out of the adjoint term.
        self.call_counts["dgFndp"] += 1
        return np.zeros((self.np, len(positions)))

    def dg(self, gIndex, states, positions):
        self.call_counts["dg"] += 1
        V = np.asarray(states["Variable"])
        zeros = np.zeros_like(V)
        return {
            "Variable": V.copy(),
            "Derivative": zeros,
            "Flux": zeros,
            # dg/dphi = 0: the objective does not see the auxiliary variables,
            # which keeps G_y out of the comparison and leaves the local matrix
            # as the only place the aux coupling can enter.
            "Aux": np.zeros((len(positions), self.ts.nAux)),
            "Scalars": np.zeros(0),
        }

    # --- dF/dp, as (np x nPoints) blocks ---------------------------------
    def dSigma(self, i, states, positions):
        self.call_counts["dSigma"] += 1
        out = np.zeros((self.np, len(positions)))
        # d(kappa phi_q)/d kappa = phi_q
        out[P_KAPPA, :] = np.asarray(states["Aux"])[:, A_Q]
        return out

    def dSources(self, i, states, positions):
        self.call_counts["dSources"] += 1
        out = np.zeros((self.np, len(positions)))
        out[P_SOURCE, :] = 1.0
        return out

    def dAux(self, i, states, positions):
        # Neither constraint mentions kappa or S.
        self.call_counts["dAux"] += 1
        return np.zeros((self.np, len(positions)))

    # No dgFn_dphi and no dAux_dp. Both were pointwise hooks a Python adjoint
    # used to have to supply, and PyAdjointProblem now raises from each of them
    # rather than dispatching: dg/dphi reaches G_y through the batched `dg`
    # above, and dAux/dp through the batched `dAux`. Defining them here would be
    # dead code that reads as though it were part of the contract.

    def getName(self, pIndex):
        return ("kappa", "source")[pIndex]


def aux_adjoint_config(tmp_path, **overrides):
    cfg = {
        # k = 4: u is a quadratic and 0.5 u^2 a quartic, so both the solution
        # and the objective's quadrature are exact and the closed form is a
        # reference rather than an approximation.
        "Polynomial_degree": 4,
        "Grid_size": 6,
        "Lower_boundary": 0.0,
        "Upper_boundary": 1.0,
        "delta_t": 0.5,
        "Relative_tolerance": 1e-8,
        "Absolute_tolerance": [1e-10],
        "MinStepSize": 1e-12,
        "solveAdjoint": True,
        "OutputFilename": str(tmp_path / "adjoint_aux_test"),
        "WriteOutput": False,
    }
    cfg.update(overrides)
    return cfg


# Long enough for the diffusive transient to be thoroughly dead: the adjoint
# state method assumes F(y, p) = 0.
T_FINAL = 15.0


def solve(p, tmp_path, **overrides):
    """Run to steady state and return (G, dG/dp, system, runner)."""
    system = AuxParametricDiffusion(p)
    runner = MaNTA.Runner(system)
    runner.configure(aux_adjoint_config(tmp_path, **overrides))
    runner.run(T_FINAL)
    G, gradients = runner.getAdjointGradients()
    return np.asarray(G), np.asarray(gradients["G_p"]), system, runner


def objective_only(p, tmp_path, **overrides):
    G, _, _, _ = solve(p, tmp_path, **overrides)
    return float(G[0])


# --------------------------------------------------------------- the tests --


def test_the_flux_derivative_is_carried_entirely_by_the_aux_block():
    """The premise that makes the gradient tests below sharp.

    If a future edit gave the flux a direct dependence on q, the gradient could
    come out right with the dSigma/dPhi block still missing, and the tests below
    would stop testing what they are for. Assert the premise directly -- no
    solver run needed.
    """
    system = AuxParametricDiffusion(np.array([KAPPA0, SOURCE0]))
    state = {
        "Variable": np.array([0.3]),
        "Derivative": np.array([0.7]),
        "Flux": np.array([0.0]),
        "Aux": np.array([0.7, 0.3]),
        "Scalars": np.zeros(0),
    }

    assert np.all(system.dSigmaFn_dq(0, state, 0.5, 0.0) == 0.0)
    assert np.all(system.dSigmaFn_du(0, state, 0.5, 0.0) == 0.0)

    d_phi = system.dSigma_dPhi(0, state, 0.5, 0.0)
    assert d_phi.shape == (N_AUX,)
    assert d_phi[A_Q] == pytest.approx(KAPPA0)

    # And the length contract dGdaux_Vec cares about.
    assert system.nAux != system.nVars


def test_objective_matches_the_closed_form(tmp_path):
    """Check the forward solve through the aux route before trusting a gradient."""
    p = np.array([KAPPA0, SOURCE0])
    G, _, _, _ = solve(p, tmp_path)

    assert G.shape == (1,), G.shape
    assert G[0] == pytest.approx(exact_G(*p), rel=1e-6), (
        f"G = {G[0]}, expected {exact_G(*p)}"
    )


# Both discretisations. `Superconvergent = True` evaluates the physics at the k+2
# star nodes with u -> u*, so it takes a *different* route to the same aux blocks:
# the chain rule A9 diag(dX/dphi) V rather than the interpolatory dPhi_Mat. Both
# have to give the right gradient, and the flag-on aux path has no other coverage
# -- the flag-on tests in test_adjoint.py all have nAux = 0.
#
# The closed form survives the flag: u is a quadratic and k = 4, so u* = u_h = u
# exactly and the reconstruction changes nothing it could get wrong.
SCHEMES = [
    pytest.param({}, id="interpolatory"),
    pytest.param({"Superconvergent": True}, id="superconvergent"),
]


@pytest.mark.parametrize("scheme", SCHEMES)
def test_adjoint_gradient_matches_finite_differences(tmp_path, scheme):
    """The load-bearing test: dG/dp from the adjoint vs. re-running the solver.

    Central differences, so truncation error is O(h^2). Nothing about the
    adjoint implementation is reused -- the reference comes purely from
    evaluating the objective at perturbed parameters, with the same scheme.

    With the dSigma/dPhi block absent from the adjoint matrix this fails on both
    components by O(1).
    """
    p0 = np.array([KAPPA0, SOURCE0])
    _, adjoint_grad, _, _ = solve(p0, tmp_path / "base", **scheme)
    adjoint_grad = adjoint_grad.reshape(-1)

    assert adjoint_grad.shape == (NP,), adjoint_grad.shape

    fd = np.zeros(NP)
    for i in range(NP):
        h = 1e-4 * abs(p0[i])
        p_plus, p_minus = p0.copy(), p0.copy()
        p_plus[i] += h
        p_minus[i] -= h
        fd[i] = (
            objective_only(p_plus, tmp_path / f"p{i}", **scheme)
            - objective_only(p_minus, tmp_path / f"m{i}", **scheme)
        ) / (2.0 * h)

    rel = np.abs(adjoint_grad - fd) / np.maximum(np.abs(fd), 1e-300)
    assert np.all(rel < 1e-3), (
        f"adjoint={adjoint_grad}\nfinite-difference={fd}\nrelative error={rel}"
    )


@pytest.mark.parametrize("scheme", SCHEMES)
def test_adjoint_gradient_matches_the_closed_form(tmp_path, scheme):
    """Independent of the finite-difference check and of the solver's own state."""
    p = np.array([KAPPA0, SOURCE0])
    _, grad, _, _ = solve(p, tmp_path, **scheme)
    grad = grad.reshape(-1)

    expected = exact_dG(*p)
    assert grad == pytest.approx(expected, rel=5e-3), (
        f"adjoint={grad}, closed form={expected}"
    )


def test_the_aux_adjoint_hooks_are_all_exercised(tmp_path):
    """Guard against the gradient being right for the wrong reason.

    `dAux` is reached only when nAux > 0, so this is the only place it is known
    to be called at all. It used to check `dgFn_dphi` alongside it; that hook is
    unreachable from Python now -- the trampoline raises rather than dispatching,
    because dg/dphi arrives with the rest of `dg`.
    """
    ts = AuxParametricDiffusion(np.array([KAPPA0, SOURCE0]))
    adjoint = AuxDiffusionAdjoint(ts)
    ts.createAdjointProblem = lambda: adjoint

    runner = MaNTA.Runner(ts)
    runner.configure(aux_adjoint_config(tmp_path / "counted"))
    runner.run(T_FINAL)
    runner.getAdjointGradients()

    for name, n in adjoint.call_counts.items():
        assert n > 0, f"{name} was never called; counts = {adjoint.call_counts}"


# ------------------------------------------- F_p's auxiliary block, dAux/dp --
#
# Everything above leaves that block identically zero: `dAux` returns zeros,
# because neither constraint in AuxParametricDiffusion mentions kappa or S. So
# the fixtures cover the aux column of the adjoint *matrix* and say nothing at
# all about the aux rows of F_p -- the dF/dp side, which
# computeAdjointGradients fills from the batched dAux and contracts with the
# adjoint state. Dropping that block, or giving it the wrong sign, is invisible
# to every test above and to every test in test_adjoint.py, where nAux is zero.
#
# Both fixtures below move a parameter *out* of a hook that is already covered
# and into a constraint, leaving the continuous problem alone. The closed forms
# at the top of this file therefore still apply, and a wrong aux block moves the
# gradient while G stays exactly right -- the same asymmetry the module
# docstring relies on for dSigma/dPhi.


class ConstraintParametricDiffusion(MaNTA.TransportSystem):
    """kappa moved out of the flux and into the auxiliary constraint.

        sigma_hat = phi_q,   phi_q - kappa q = 0,   phi_u - u = 0,   source = S

    phi_q is still kappa q, so this is the same diffusion equation as
    AuxParametricDiffusion, with the same steady state and the same closed
    forms. What moves is where kappa reaches the residual: dSigma/dkappa is now
    identically zero and dAux/dkappa = -q, so the whole of dG/dkappa arrives
    through F_p's auxiliary block. dG/dS still comes through the source block,
    which is the control -- a dropped or mis-signed aux block moves one
    component of the gradient and leaves the other alone.
    """

    def __init__(self, p):
        MaNTA.TransportSystem.__init__(self, MaNTA.numbered_spec(1, nAux=N_AUX))
        self.p = np.asarray(p, dtype=float)

    def SigmaFn(self, i, state, x, t):
        return state.phi[A_Q]

    def Sources(self, i, state, x, t):
        return self.p[P_SOURCE]

    def dSigmaFn_du(self, i, state, x, t):
        return np.zeros(self.nVars)

    def dSigmaFn_dq(self, i, state, x, t):
        return np.zeros(self.nVars)

    def dSources_du(self, i, state, x, t):
        return np.zeros(self.nVars)

    def dSources_dq(self, i, state, x, t):
        return np.zeros(self.nVars)

    def dSources_dsigma(self, i, state, x, t):
        return np.zeros(self.nVars)

    def dSigma_dPhi(self, i, state, x, t):
        out = np.zeros(self.nAux)
        # One, not kappa: the flux is phi_q itself now.
        out[A_Q] = 1.0
        return out

    def dSources_dPhi(self, i, state, x, t):
        return np.zeros(self.nAux)

    def AuxG(self, i, state, x, t):
        if i == A_Q:
            return state.phi[A_Q] - self.p[P_KAPPA] * state.q[0]
        return state.phi[A_U] - state.u[0]

    def AuxGPrime(self, i, out, state, x, t):
        out.phi[i] = 1.0
        if i == A_Q:
            out.q[0] = -self.p[P_KAPPA]
        else:
            out.u[0] = -1.0

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

    def createAdjointProblem(self):
        return ConstraintParametricAdjoint(self)


class ConstraintParametricAdjoint(MaNTA.AdjointProblem):
    """The same objective, with kappa's dF/dp now living in dAux."""

    def __init__(self, transport_system):
        MaNTA.AdjointProblem.__init__(self)
        self.ts = transport_system
        self.ng = 1
        self.np = NP
        self.np_boundary = 0
        self.spatialParameters = False

    def gFn(self, gIndex, states, positions):
        u = np.asarray(states["Variable"])[:, 0]
        return 0.5 * u * u

    def dgFndp(self, gIndex, states, positions):
        return np.zeros((self.np, len(positions)))

    def dg(self, gIndex, states, positions):
        V = np.asarray(states["Variable"])
        zeros = np.zeros_like(V)
        return {
            "Variable": V.copy(),
            "Derivative": zeros,
            "Flux": zeros,
            "Aux": np.zeros((len(positions), self.ts.nAux)),
            "Scalars": np.zeros(0),
        }

    def dSigma(self, i, states, positions):
        # Identically zero: kappa has left the flux.
        return np.zeros((self.np, len(positions)))

    def dSources(self, i, states, positions):
        out = np.zeros((self.np, len(positions)))
        out[P_SOURCE, :] = 1.0
        return out

    def dAux(self, i, states, positions):
        out = np.zeros((self.np, len(positions)))
        if i == A_Q:
            # d(phi_q - kappa q)/d kappa = -q. Indexed by *aux*, so the decoy
            # constraint keeps its zero row and a block written at the wrong aux
            # offset lands on it.
            out[P_KAPPA, :] = -np.asarray(states["Derivative"])[:, 0]
        return out

    def getName(self, pIndex):
        return ("kappa", "source")[pIndex]


def constraint_solve(p, tmp_path, **overrides):
    system = ConstraintParametricDiffusion(p)
    runner = MaNTA.Runner(system)
    runner.configure(aux_adjoint_config(tmp_path, **overrides))
    runner.run(T_FINAL)
    G, gradients = runner.getAdjointGradients()
    return np.asarray(G), np.asarray(gradients["G_p"]).reshape(-1)


def test_the_constraint_route_puts_kappa_only_in_dAux():
    """The premise, asserted directly, the way the dSigma/dPhi one is above."""
    system = ConstraintParametricDiffusion(np.array([KAPPA0, SOURCE0]))
    adjoint = ConstraintParametricAdjoint(system)
    states = {
        "Variable": np.array([[0.3]]),
        "Derivative": np.array([[0.7]]),
        "Flux": np.array([[0.0]]),
        "Aux": np.array([[0.7 * KAPPA0, 0.3]]),
        "Scalars": np.zeros(0),
    }
    positions = [0.5]

    assert np.all(adjoint.dSigma(0, states, positions) == 0.0), (
        "kappa still reaches the flux, so the aux block is not the only route"
    )
    assert adjoint.dSources(0, states, positions)[P_KAPPA, 0] == 0.0

    dphi_q = adjoint.dAux(A_Q, states, positions)
    assert dphi_q[P_KAPPA, 0] == pytest.approx(-0.7)
    assert dphi_q[P_SOURCE, 0] == 0.0
    assert np.all(adjoint.dAux(A_U, states, positions) == 0.0)


def test_the_objective_is_unchanged_by_the_constraint_route(tmp_path):
    """Same problem, so a wrong aux block has to show in the gradient alone."""
    p = np.array([KAPPA0, SOURCE0])
    G, _ = constraint_solve(p, tmp_path)
    assert G[0] == pytest.approx(exact_G(*p), rel=1e-6)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_the_aux_parameter_block_matches_the_closed_form(tmp_path, scheme):
    """dG/dkappa now comes entirely out of F_p's auxiliary rows.

    With that block dropped, dG/dkappa collapses to zero -- nothing else in the
    problem depends on kappa -- while dG/dS and G itself stay exactly right.
    Both schemes, because the superconvergent branch takes A9 over the star
    nodes rather than the interpolatory projection and has its own line here.
    """
    p = np.array([KAPPA0, SOURCE0])
    _, grad = constraint_solve(p, tmp_path, **scheme)

    expected = exact_dG(*p)
    assert grad == pytest.approx(expected, rel=5e-3), (
        f"adjoint={grad}, closed form={expected}"
    )
    assert abs(grad[P_KAPPA]) > 1e-3, (
        "dG/dkappa is ~zero, which is what a missing aux block looks like"
    )


# ------------------------------------ ...and the same block, indexed by node --
#
# The spatial branch is a separate loop: it fills one F_p column per node rather
# than one vector per cell, and nothing reached it before -- test_adjoint.py's
# spatial fixtures have nAux = 0, and every fixture with nAux > 0 sets
# spatialParameters False. Superconvergent is refused with spatial parameters,
# so there is only the one path to cover here.

SPATIAL_KAPPA = 1.0
SPATIAL_S0 = 2.0
SPATIAL_K = 4
SPATIAL_NCELLS = 6

# Auxiliary layout for the spatial fixture: phi_S carries the source, phi_u is
# the decoy that keeps nAux != nVars.
A_S = 0


class NodalConstraintSource(MaNTA.TransportSystem):
    """-d_x(kappa d_x u) = S(x), with S delivered through a constraint.

        source = phi_S,   phi_S - S(x) = 0,   phi_u - u = 0

    The parameter is one field of nodal values, so dAux/dp is -1 at the node
    that owns the point and zero elsewhere -- which the solver expresses by
    handing the hook one column per node. Routing it through phi rather than
    returning it from Sources is the whole difference from
    test_adjoint.py::NodalSourceDiffusion, whose gradient is checked there
    against both a closed form and node-by-node finite differences.
    """

    def __init__(self, points, source_nodes):
        MaNTA.TransportSystem.__init__(self, MaNTA.numbered_spec(1, nAux=2))
        self.points = np.asarray(points, dtype=float)
        self.source_nodes = np.asarray(source_nodes, dtype=float)

    def _S(self, x):
        return float(self.source_nodes[np.argmin(np.abs(self.points - x))])

    def SigmaFn(self, i, state, x, t):
        return SPATIAL_KAPPA * state.q[i]

    def Sources(self, i, state, x, t):
        return state.phi[A_S]

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

    def dSigma_dPhi(self, i, state, x, t):
        return np.zeros(self.nAux)

    def dSources_dPhi(self, i, state, x, t):
        out = np.zeros(self.nAux)
        out[A_S] = 1.0
        return out

    def AuxG(self, i, state, x, t):
        if i == A_S:
            return state.phi[A_S] - self._S(x)
        return state.phi[1] - state.u[0]

    def AuxGPrime(self, i, out, state, x, t):
        out.phi[i] = 1.0
        if i != A_S:
            out.u[0] = -1.0

    def LowerBoundary(self, i, t):
        return 0.0

    def UpperBoundary(self, i, t):
        return 0.0

    def InitialValue(self, i, x):
        return 0.0

    def InitialDerivative(self, i, x):
        return 0.0

    def InitialAuxValue(self, i, x):
        # phi_S = S(x) and phi_u = u = 0: both constraints hold at t = 0.
        return self._S(x) if i == A_S else 0.0

    def createAdjointProblem(self):
        return NodalConstraintAdjoint(self)


class NodalConstraintAdjoint(MaNTA.AdjointProblem):
    """One parameter *field*, reaching the residual only through phi_S."""

    def __init__(self, transport_system):
        MaNTA.AdjointProblem.__init__(self)
        self.ts = transport_system
        self.ng = 1
        self.np = 1
        self.np_boundary = 0
        self.spatialParameters = True

    def gFn(self, gIndex, states, positions):
        u = np.asarray(states["Variable"])[:, 0]
        return 0.5 * u * u

    def dgFndp(self, gIndex, states, positions):
        return np.zeros((self.np, len(positions)))

    def dg(self, gIndex, states, positions):
        V = np.asarray(states["Variable"])
        zeros = np.zeros_like(V)
        return {
            "Variable": V.copy(),
            "Derivative": zeros,
            "Flux": zeros,
            "Aux": np.zeros((len(positions), self.ts.nAux)),
            "Scalars": np.zeros(0),
        }

    def dSigma(self, i, states, positions):
        return np.zeros((self.np, len(positions)))

    def dSources(self, i, states, positions):
        # Zero, unlike the direct route: the source is phi_S, not the parameter.
        return np.zeros((self.np, len(positions)))

    def dAux(self, i, states, positions):
        out = np.zeros((self.np, len(positions)))
        if i == A_S:
            # d(phi_S - S)/dS = -1 at the node this point *is*; the per-node
            # structure is what spatialParameters expresses, so the hook reports
            # -1 everywhere and the solver places it.
            out[0, :] = -1.0
        return out

    def getName(self, pIndex):
        return "S"


def spatial_aux_solve(source_nodes, tmp_path):
    points = np.asarray(
        MaNTA.getNodes(0.0, 1.0, SPATIAL_NCELLS, SPATIAL_K), dtype=float
    )
    system = NodalConstraintSource(points, source_nodes)
    runner = MaNTA.Runner(system)
    runner.configure(
        aux_adjoint_config(
            tmp_path, Polynomial_degree=SPATIAL_K, Grid_size=SPATIAL_NCELLS
        )
    )
    runner.run(T_FINAL)
    G, gradients = runner.getAdjointGradients()
    return float(np.asarray(G)[0]), np.asarray(gradients["G_p"]), points


def test_the_spatial_aux_gradient_sums_to_the_closed_form(tmp_path):
    """Perturbing every node together is perturbing the scalar source.

    G is quadratic in the parameters, so the sum is dG/dS = S / (120 kappa^2)
    with no discretisation error -- the same statement
    test_adjoint.py::test_the_spatial_gradient_sums_to_the_closed_form makes
    about the direct route, and it holds here only if the aux rows of F_p carry
    the right weighting and the right sign.
    """
    nodes = SPATIAL_S0 * np.ones(SPATIAL_NCELLS * (SPATIAL_K + 1))
    G, G_p, _ = spatial_aux_solve(nodes, tmp_path)

    assert G_p.shape == (len(nodes), 1), G_p.shape
    assert G == pytest.approx(SPATIAL_S0**2 / (240.0 * SPATIAL_KAPPA**2), rel=1e-6)
    assert G_p.sum() == pytest.approx(
        SPATIAL_S0 / (120.0 * SPATIAL_KAPPA**2), rel=1e-6
    )


def test_the_spatial_aux_gradient_is_right_node_by_node(tmp_path):
    """The sum is blind to node ordering; central differences are not.

    The aux branch places its result at F_p.col(j) inside a loop over the
    intra-cell node index, so a gradient whose entries are permuted or shifted
    within a cell still sums correctly. Difference the objective one node at a
    time instead. A non-uniform field, so the answer varies from node to node
    and a constant-per-cell result would be visible.
    """
    n = SPATIAL_NCELLS * (SPATIAL_K + 1)
    base = SPATIAL_S0 * (1.0 + 0.25 * np.cos(np.arange(n)))

    _, G_p, _ = spatial_aux_solve(base, tmp_path / "base")
    adjoint = G_p.reshape(-1)

    h = 1e-4 * SPATIAL_S0
    fd = np.zeros(n)
    for j in range(n):
        plus, minus = base.copy(), base.copy()
        plus[j] += h
        minus[j] -= h
        G_plus, _, _ = spatial_aux_solve(plus, tmp_path / f"p{j}")
        G_minus, _, _ = spatial_aux_solve(minus, tmp_path / f"m{j}")
        fd[j] = (G_plus - G_minus) / (2.0 * h)

    np.testing.assert_allclose(adjoint, fd, rtol=2e-3, atol=1e-12)
