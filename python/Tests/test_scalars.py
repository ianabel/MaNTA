"""Global (non-spatial) scalars through the Python transport-system interface.

`nScalars > 0` reaches six trampoline hooks nothing else touches -- `ScalarG`,
`ScalarGPrime`, `InitialScalarValue`, `InitialScalarDerivative`,
`isScalarDifferential` and `dSources_dScalars` -- and until this file there was no
Python test with a scalar at all. `TODO` carried "Add Scalars to python
interface"; the plumbing turned out to be there, so what was missing was any
evidence that it works.

The scalar hooks are the most awkward part of the interface, because the C++
signatures take `DGSoln` and `Interval`, which have no Python representation.
`PyTransportSystem` bridges that: it evaluates the solution on the nodes and
passes a `GlobalState` plus the quadrature data the hook needs. The resulting
The Python signatures are now the C++ ones, argument for argument -- the two
used to differ, because the C++ hooks took DGSoln, a std::function test
function and an Interval, none of which have a Python representation:

    InitialScalarValue(s)                                     -> float
    InitialScalarDerivative(s, states, states_dot, weights)   -> float
    ScalarG(s, states, states_dot, abscissae, weights, phi_boundary, t) -> float
    ScalarGPrime(states, states_dot, abscissae, weights, phi_boundary, t)
        -> (list of nScalars GlobalState dicts,   d G_s / d state
            list of nScalars GlobalState dicts)   d G_s / d state_dot
    dSources_dScalars(s, state, x, t)  -> vector of length nScalars

Whether a scalar is differential is spec data (`MaNTA.Scalar(..., differential=True)`)
rather than an isScalarDifferential hook.

`weights` is the global quadrature weight per node, length nCells*(k+1), so an
integral over the domain is just `weights @ u`. `phi_boundary` is (k+1, 2): the
basis functions evaluated at the two ends of the domain.

The system
----------
Linear diffusion whose source is raised by the scalar, and whose scalar is the
integral of the solution -- so the two are genuinely coupled and neither can be
got right by accident:

    -d_x( kappa d_x u ) = S0 + mu     on [0, 1],  u(0) = u(1) = 0
    G_0 = mu - int_0^1 u dx = 0

Substituting u = (S0 + mu) x(1-x) / (2 kappa) and int_0^1 x(1-x) dx = 1/6,

    mu = (S0 + mu) / (12 kappa)   =>   mu = S0 / (12 kappa - 1)

which is a closed form for both the scalar and the field. With kappa = 1.5 the
denominator is 17, comfortably away from the 12*kappa = 1 resonance, so the fixed
point is well conditioned.

A scalar that fed nothing back into the physics would be a much weaker test: a
wrong `dSources_dScalars` or a wrong `ScalarGPrime` column would leave u
untouched. Here both change the field.
"""

import numpy as np
import pytest

import manta as MaNTA
KAPPA = 1.5
S0 = 2.0
N_SCALARS = 1


def exact_mu():
    return S0 / (12.0 * KAPPA - 1.0)


def exact_u(x):
    return (S0 + exact_mu()) * x * (1.0 - x) / (2.0 * KAPPA)


class ScalarDiffusion(MaNTA.TransportSystem):
    """Linear diffusion coupled to one algebraic global scalar."""

    def __init__(self, differential=False):
        MaNTA.TransportSystem.__init__(
            self, MaNTA.numbered_spec(1, nScalars=N_SCALARS, differential=differential))
        self.calls = {
            "ScalarG": 0,
            "ScalarGPrime": 0,
            "InitialScalarValue": 0,
            "InitialScalarDerivative": 0,
            "dSources_dScalars": 0,
        }

    # --- flux and sources ------------------------------------------------
    def SigmaFn(self, i, state, x, t):
        return KAPPA * state.q[i]

    def Sources(self, i, state, x, t):
        # The scalar raises the source uniformly.
        return S0 + state.scalars[0]

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

    def dSources_dScalars(self, i, state, x, t):
        # d S_i / d mu_j, so this is indexed by *scalar*, length nScalars --
        # not nVars. dS_0/dmu_0 = 1.
        self.calls["dSources_dScalars"] += 1
        return np.ones(self.nScalars)

    # --- the scalar constraint  G_0 = mu - int u dx = 0 -------------------
    def ScalarG(self, s, states, states_dot, abscissae, weights, phi_boundary, t):
        self.calls["ScalarG"] += 1
        u = np.asarray(states["Variable"])[:, 0]
        mu = np.asarray(states["Scalars"])[0]
        return float(mu - np.dot(np.asarray(weights), u))

    def ScalarGPrime(self, states, states_dot, abscissae, weights, phi_boundary, t):
        # Two lists of nScalars GlobalState dicts: dG/dstate and dG/dstate_dot.
        #
        # G_0 = mu - sum_j w_j u_j, so dG_0/du_j = -w_j -- the quadrature weight
        # belongs in the derivative because it is in G. Everything else is zero,
        # and dG_0/dstate_dot is identically zero because the scalar is algebraic.
        self.calls["ScalarGPrime"] += 1
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

        dG = zeros()
        dG["Variable"][:, 0] = -w
        dG["Scalars"][0] = 1.0

        return ([dG], [zeros()])

    # --- boundaries and initial data -------------------------------------
    def LowerBoundary(self, i, t):
        return 0.0

    def UpperBoundary(self, i, t):
        return 0.0

    def InitialValue(self, i, x):
        return 0.0

    def InitialDerivative(self, i, x):
        return 0.0

    def InitialScalarValue(self, s):
        self.calls["InitialScalarValue"] += 1
        return 0.0

    def InitialScalarDerivative(self, s, states, states_dot, weights):
        # Only consulted for *differential* scalars -- setInitialConditions calls
        # it under `if (problem->isScalarDifferential(s))`. For the algebraic
        # scalar above it is therefore never reached, which is why
        # DifferentialScalarDiffusion below exists. The trampoline still requires
        # the override to be present whenever nScalars > 0.
        self.calls["InitialScalarDerivative"] += 1
        return 0.0


class DifferentialScalarDiffusion(ScalarDiffusion):
    """The same steady state, reached by relaxation, with mu *differential*.

        G_0 = dmu/dt - ( int u dx - mu ) = 0

    At steady state dmu/dt = 0 and this reduces to mu = int u dx, so the closed
    form is unchanged -- but the scalar is now differential, which reaches two
    things the algebraic version cannot:

      * InitialScalarDerivative, called once per differential scalar by
        setInitialConditions;
      * the `states_dot` argument of ScalarG and the second (d/d state_dot) half
        of ScalarGPrime's return, both identically zero above.

    mu(0) is deliberately nonzero, so the consistent initial derivative
    dmu/dt(0) = int u(0) dx - mu(0) = -mu(0) is nonzero too and a hook that
    returned a constant 0 would be visibly wrong rather than accidentally right.
    """

    MU0 = 0.5

    # Differential, so the base's spec is built with differential=True. This
    # was an isScalarDifferential override; the flag is spec data now.
    def __init__(self):
        super().__init__(differential=True)

    def InitialScalarValue(self, s):
        self.calls["InitialScalarValue"] += 1
        return self.MU0

    def InitialScalarDerivative(self, s, states, states_dot, weights):
        # Computed from the arguments rather than hardcoded, so the values the
        # trampoline passes are themselves under test.
        self.calls["InitialScalarDerivative"] += 1
        u = np.asarray(states["Variable"])[:, 0]
        mu = np.asarray(states["Scalars"])[0]
        return float(np.dot(np.asarray(weights), u) - mu)

    def ScalarG(self, s, states, states_dot, abscissae, weights, phi_boundary, t):
        self.calls["ScalarG"] += 1
        u = np.asarray(states["Variable"])[:, 0]
        mu = np.asarray(states["Scalars"])[0]
        mu_dot = np.asarray(states_dot["Scalars"])[0]
        return float(mu_dot - (np.dot(np.asarray(weights), u) - mu))

    def ScalarGPrime(self, states, states_dot, abscissae, weights, phi_boundary, t):
        # G_0 = mu' - int u dx + mu, so d/du_j = -w_j, d/dmu = +1, and the
        # state_dot half is d/dmu' = 1.
        self.calls["ScalarGPrime"] += 1
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

        dG = zeros()
        dG["Variable"][:, 0] = -w
        dG["Scalars"][0] = 1.0

        dG_dt = zeros()
        dG_dt["Scalars"][0] = 1.0

        return ([dG], [dG_dt])


def scalar_config(tmp_path, **overrides):
    cfg = {
        # k = 4: the steady state is a quadratic, so it is represented exactly and
        # the closed form above is a reference rather than an approximation.
        "PolynomialDegree": 4,
        "GridSize": 6,
        "LowerBoundary": 0.0,
        "UpperBoundary": 1.0,
        "delta_t": 0.5,
        "Relative_tolerance": 1e-8,
        "Absolute_tolerance": [1e-10],
        "MinStepSize": 1e-12,
        "OutputFilename": str(tmp_path / "scalar_test"),
        "WriteOutput": False,
    }
    cfg.update(overrides)
    return cfg


# The diffusive transient is t ~ 1/(kappa pi^2) ~ 0.07; this is far past it.
T_FINAL = 15.0

XS = [0.1 * i for i in range(1, 10)]

# Both kinds of scalar. They share a steady state by construction, so the same
# closed form checks both -- but they reach different parts of the interface.
SYSTEMS = [
    pytest.param(ScalarDiffusion, id="algebraic"),
    pytest.param(DifferentialScalarDiffusion, id="differential"),
]


# --------------------------------------------------------------- the tests --


@pytest.mark.parametrize("system_cls", SYSTEMS)
def test_a_python_system_with_a_scalar_runs(tmp_path, system_cls):
    """The first question: does nScalars > 0 work from Python at all."""
    system = system_cls()
    runner = MaNTA.Runner(system)
    runner.configure(scalar_config(tmp_path))
    runner.run(T_FINAL)

    u = np.asarray(runner.getSolution(0, XS))
    assert np.all(np.isfinite(u))
    assert np.all(u > 0.0), f"positive source with zero Dirichlet ends: {u}"


@pytest.mark.parametrize("system_cls", SYSTEMS)
def test_the_field_matches_the_coupled_closed_form(tmp_path, system_cls):
    """u must solve the problem *including* the scalar's contribution.

    This is what distinguishes a working coupling from one where the scalar is
    computed but never fed back: with dSources_dScalars or the scalar's effect on
    the source dropped, u would come out as the mu = 0 solution, which is smaller
    by a factor (S0 + mu)/S0 = 1 + 1/(12 kappa - 1) -- about 6% here, far outside
    the tolerance below.
    """
    system = system_cls()
    runner = MaNTA.Runner(system)
    runner.configure(scalar_config(tmp_path))
    runner.run(T_FINAL)

    u = np.asarray(runner.getSolution(0, XS))
    expected = np.array([exact_u(x) for x in XS])

    rel = np.max(np.abs(u - expected) / np.abs(expected))
    assert rel < 1e-5, f"u = {u}\nexpected = {expected}\nmax rel err = {rel}"

    # And explicitly not the uncoupled solution, so this cannot pass by accident.
    uncoupled = np.array([S0 * x * (1.0 - x) / (2.0 * KAPPA) for x in XS])
    assert np.max(np.abs(u - uncoupled) / np.abs(uncoupled)) > 1e-3


@pytest.mark.parametrize("system_cls", SYSTEMS)
def test_every_scalar_hook_is_reached(tmp_path, system_cls):
    """A system whose scalar hooks were skipped would still solve the mu = 0
    problem and fail only the closed-form test above, with no hint as to why.

    InitialScalarDerivative is the one exception, and it is asserted in both
    directions: reached for a differential scalar, and *not* reached for an
    algebraic one, since setInitialConditions only consults it under
    `if (problem->isScalarDifferential(s))`. Asserting the negative case pins the
    branch rather than tolerating it.
    """
    system = system_cls()
    runner = MaNTA.Runner(system)
    runner.configure(scalar_config(tmp_path))
    runner.run(1.0)

    differential = system_cls is DifferentialScalarDiffusion
    for name, n in system.calls.items():
        if name == "InitialScalarDerivative" and not differential:
            assert n == 0, (
                "an algebraic scalar should not need an initial derivative; "
                f"counts = {system.calls}"
            )
            continue
        assert n > 0, f"{name} was never called; counts = {system.calls}"


@pytest.mark.parametrize("system_cls", SYSTEMS)
def test_the_scalar_constraint_holds_at_the_solution(tmp_path, system_cls):
    """G_0 = 0 means mu = int u dx. Check it against the field the solver returns.

    mu itself is not exposed through getSolution, so the constraint is checked the
    other way round: integrate the returned u on a fine grid and compare with the
    closed-form mu.
    """
    system = system_cls()
    runner = MaNTA.Runner(system)
    runner.configure(scalar_config(tmp_path))
    runner.run(T_FINAL)

    xs = np.linspace(0.0, 1.0, 2001)
    u = np.asarray(runner.getSolution(0, list(xs)))
    integral = np.trapezoid(u, xs) if hasattr(np, "trapezoid") else np.trapz(u, xs)

    assert integral == pytest.approx(exact_mu(), rel=1e-4), (
        f"int u dx = {integral}, mu (closed form) = {exact_mu()}"
    )


def test_a_scalar_system_missing_its_hooks_is_reported(tmp_path):
    """nScalars > 0 with no ScalarG cannot form the scalar residual; say so.

    The check happens at setup and names the missing hooks, rather than failing
    somewhere inside the bordered solve.
    """

    class NoScalarHooks(MaNTA.TransportSystem):
        def __init__(self):
            MaNTA.TransportSystem.__init__(self, MaNTA.numbered_spec(1, nScalars=1))

        def SigmaFn(self, i, state, x, t):
            return KAPPA * state.q[i]

        def Sources(self, i, state, x, t):
            return S0

        def dSigmaFn_du(self, i, state, x, t):
            return np.zeros(1)

        def dSigmaFn_dq(self, i, state, x, t):
            return np.full(1, KAPPA)

        def dSources_du(self, i, state, x, t):
            return np.zeros(1)

        def dSources_dq(self, i, state, x, t):
            return np.zeros(1)

        def dSources_dsigma(self, i, state, x, t):
            return np.zeros(1)

        def LowerBoundary(self, i, t):
            return 0.0

        def UpperBoundary(self, i, t):
            return 0.0

        def InitialValue(self, i, x):
            return 0.0

        def InitialDerivative(self, i, x):
            return 0.0

        # ScalarG, ScalarGPrime, InitialScalarValue, InitialScalarDerivative,
        # isScalarDifferential and dSources_dScalars all deliberately absent.

    runner = MaNTA.Runner(NoScalarHooks())

    with pytest.raises(RuntimeError) as excinfo:
        runner.configure(scalar_config(tmp_path))
        runner.run(0.1)

    message = str(excinfo.value)
    assert "Scalar" in message or "nScalars" in message, message
