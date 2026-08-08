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

import MaNTA

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
        MaNTA.TransportSystem.__init__(self)
        self.nVars = 1
        self.nScalars = 0
        self.nAux = N_AUX
        self.isLowerDirichlet = True
        self.isUpperDirichlet = True
        self.p = np.asarray(p, dtype=float)

    # --- flux and sources ------------------------------------------------
    def SigmaFn(self, i, state, x, t):
        # Note: phi_q, not Derivative. This is what puts the flux's dependence
        # on q behind the auxiliary constraint.
        return self.p[P_KAPPA] * state["Aux"][A_Q]

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
            return state["Aux"][A_Q] - state["Derivative"][0]
        return state["Aux"][A_U] - state["Variable"][0]

    def AuxGPrime(self, i, state, x, t):
        variable = np.zeros(self.nVars)
        derivative = np.zeros(self.nVars)
        aux = np.zeros(self.nAux)
        aux[i] = 1.0
        if i == A_Q:
            derivative[0] = -1.0
        else:
            variable[0] = -1.0
        return {
            "Variable": variable,
            "Derivative": derivative,
            "Flux": np.zeros(self.nVars),
            "Aux": aux,
            "Scalars": np.zeros(0),
        }

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
            "dgFn_dphi": 0,
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

    def dgFn_dphi(self, gIndex, state, x):
        # Pointwise, and the return value *is* the derivative vector -- unlike
        # the C++ signature (Index, VectorRef, State, Position), the trampoline
        # does not pass the output reference through to Python.
        self.call_counts["dgFn_dphi"] += 1
        return np.zeros(self.ts.nAux)

    def dAux_dp(self, i, pIndex, state, x):
        return 0.0

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


def objective_only(p, tmp_path):
    G, _, _, _ = solve(p, tmp_path)
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


def test_adjoint_gradient_matches_finite_differences(tmp_path):
    """The load-bearing test: dG/dp from the adjoint vs. re-running the solver.

    Central differences, so truncation error is O(h^2). Nothing about the
    adjoint implementation is reused -- the reference comes purely from
    evaluating the objective at perturbed parameters.

    With the dSigma/dPhi block absent from the adjoint matrix this fails on both
    components by O(1).
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


def test_adjoint_gradient_matches_the_closed_form(tmp_path):
    """Independent of the finite-difference check and of the solver's own state."""
    p = np.array([KAPPA0, SOURCE0])
    _, grad, _, _ = solve(p, tmp_path)
    grad = grad.reshape(-1)

    expected = exact_dG(*p)
    assert grad == pytest.approx(expected, rel=5e-3), (
        f"adjoint={grad}, closed form={expected}"
    )


def test_the_aux_adjoint_hooks_are_all_exercised(tmp_path):
    """Guard against the gradient being right for the wrong reason.

    `dAux` and `dgFn_dphi` are reached only when nAux > 0, so this is the only
    place they are known to be called at all.
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
