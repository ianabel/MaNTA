"""Jardin's stiff critical-gradient problem in two channels: ion and electron
temperatures, each channel's diffusivity switched on by *both* gradients, with
optional collisional exchange between them.

`../jardin-critical-gradient/` is the single-channel original (Jardin et al.,
JCP 227 (2008) 8769). Real stiff transport is not single-channel: in a
critical-gradient model the ion heat flux responds to the electron temperature
gradient as well as to its own, and the two temperatures exchange energy
collisionally. In the conservative form MaNTA integrates,
`a d_t u - d_x[sigma_hat] = S`, for s in {i, e}:

    d_t T_s - d_x[ x chi_s(q_i, q_e) q_s ] = S_s

    chi_s = chi0 + sum_r kappa_sr (|q_r| - qc_r)_+^gamma

    S_i = H_i + nu (T_e - T_i),      S_e = H_e - nu (T_e - T_i)

    sigma_hat(0) = 0                 (mixed, d = 1: zero flux)
    T_s(1) = T_sb                    (Dirichlet)

with chi0 = 1, kappa_ii = kappa_ee = 10, kappa_ie = 2, kappa_ei = 3,
gamma = 1/2, qc_i = 0.5 and qc_e = 0.8 -- Jardin's numbers on the diagonal,
weaker cross-thresholds off it.

**The stiff steady state is exactly linear, as in Jardin's problem.** Without
exchange (nu = 0, the default), integrating once and requiring regularity on
the axis gives chi_s q_s = -H_s, so both gradients are constants -g_s solving

    chi_i(g_i, g_e) g_i = H_i,     chi_e(g_i, g_e) g_e = H_e,

and T_s = T_sb + g_s (1 - x). With H_i = 2 and H_e = 3 that is
g_i = 0.5506245..., g_e = 0.8365288..., 0.051 and 0.037 above their thresholds.
Being degree 1, both profiles lie in P_k for every k, so the benchmark measures
the coupled nonlinear solve on its own: any correct scheme reproduces the
answer to round-off, and the cost is the result. At that state the tangent
diffusivity matrix d(chi_s q_s)/d(q_r) is [[15.9, 2.9], [5.6, 25.5]]: the
Jacobian is genuinely coupled.

**The heating is chosen to keep the steady state off the kink.** chi has a
square-root threshold, so d chi/d q diverges just above it, and a Newton
iteration whose answer sits there struggles. With H = (1, 2) the ion gradient
is 0.0054 above its threshold, and the steady-state solvers need 40-50 times
the visits they need at H = (2, 3); started from the constant-chi steady state
instead of the initial condition below, they stall outright. Time marching is
indifferent to it. README.md has the numbers.

**With exchange** (`exchange=True`) a linear steady state needs T_e - T_i to be
a constant delta, which forces g_i = g_e = g. Choose H_i, nu and delta; then g
solves chi_i(g, g) g = H_i + nu delta, and H_e = chi_e(g, g) g + nu delta is
*set* so that the electron equation balances. The exchange is active -- it
carries nu delta from the electrons to the ions -- but H_e is rigged, and the
two gradients are equal by construction. A Ti-Te problem with exchange and
unequal, non-polynomial profiles has no closed form of this kind; it needs a
manufactured solution.
"""

import numpy as np

import manta

CHI0 = 1.0
KAPPA = np.array([[10.0, 2.0],      # kappa_ii, kappa_ie
                  [3.0, 10.0]])     # kappa_ei, kappa_ee
GAMMA = 0.5
QC = np.array([0.5, 0.8])           # critical gradients, ion then electron

# Without exchange.
H_PLAIN = np.array([2.0, 3.0])
TB_PLAIN = np.array([1.0, 1.0])

# With exchange: T_e - T_i = DELTA throughout, NU the exchange rate, H_i given
# and H_e derived (see ExchangeHeating).
NU = 2.0
DELTA = 0.5
HI_EXCHANGE = 6.0
TB_EXCHANGE = np.array([1.0, 1.0 + DELTA])


def excess(q, s):
    """(|q| - qc_s)_+ , elementwise."""
    return np.maximum(np.abs(q) - QC[s], 0.0)


def chi(qi, qe):
    """(chi_i, chi_e) at the gradients (qi, qe); works on scalars or arrays."""
    ei, ee = excess(qi, 0) ** GAMMA, excess(qe, 1) ** GAMMA
    return (CHI0 + KAPPA[0, 0] * ei + KAPPA[0, 1] * ee,
            CHI0 + KAPPA[1, 0] * ei + KAPPA[1, 1] * ee)


def dchi_dq(qi, qe):
    """d chi_s / d q_r as a 2x2 nested tuple. The threshold makes it
    discontinuous; above it, it diverges as (|q| - qc)^(gamma - 1)."""
    def one(q, s):
        e = excess(q, s)
        with np.errstate(divide="ignore", invalid="ignore"):
            d = np.where(e > 0.0, GAMMA * np.sign(q) * e ** (GAMMA - 1.0), 0.0)
        return d
    di, de = one(qi, 0), one(qe, 1)
    return ((KAPPA[0, 0] * di, KAPPA[0, 1] * de),
            (KAPPA[1, 0] * di, KAPPA[1, 1] * de))


def fluxes(x, qi, qe):
    """(sigma_i, sigma_e) = x chi_s q_s."""
    ci, ce = chi(qi, qe)
    return x * ci * qi, x * ce * qe


def _newton2(F, J, g0):
    """A damped Newton on two unknowns, keeping this example to numpy alone."""
    g = np.array(g0, dtype=float)
    for _ in range(200):
        r = F(g)
        if np.max(np.abs(r)) < 1e-15:
            break
        step = np.linalg.solve(J(g), -r)
        lam = 1.0
        while lam > 1e-6 and np.max(np.abs(F(g + lam * step))) >= np.max(np.abs(r)):
            lam *= 0.5
        g = g + lam * step
    return g


def SteadyGradients():
    """(g_i, g_e) without exchange: chi_s(-g_i, -g_e) g_s = H_s.

    Newton from the constant-chi gradients H_s / chi0, which are above both
    thresholds; a scan of Newton starts over (0, 4]^2 finds
    no other positive root."""
    def F(g):
        ci, ce = chi(-g[0], -g[1])
        return np.array([ci * g[0], ce * g[1]]) - H_PLAIN

    def J(g):
        ci, ce = chi(-g[0], -g[1])
        d = dchi_dq(-g[0], -g[1])
        # d/dg of chi_s(-g) g_s = delta_sr chi_s - g_s dchi_s/dq_r
        return np.array([[ci - g[0] * d[0][0], -g[0] * d[0][1]],
                         [-g[1] * d[1][0], ce - g[1] * d[1][1]]])

    return _newton2(F, J, H_PLAIN / CHI0)


def ExchangeGradient():
    """g with exchange: chi_i(-g, -g) g = H_i + nu delta. Bisection: the left
    side increases monotonically in g."""
    target = HI_EXCHANGE + NU * DELTA
    lo, hi = 0.0, target / CHI0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if chi(-mid, -mid)[0] * mid < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def ExchangeHeating():
    """H_e that makes T_e - T_i = delta a steady state."""
    g = ExchangeGradient()
    return chi(-g, -g)[1] * g + NU * DELTA


def ExactSolution(x, exchange=False):
    """(T_i, T_e) at x, stacked as a (2, len(x)) array."""
    x = np.asarray(x, dtype=float)
    if exchange:
        g = ExchangeGradient()
        return np.array([TB_EXCHANGE[0] + g * (1.0 - x), TB_EXCHANGE[1] + g * (1.0 - x)])
    gi, ge = SteadyGradients()
    return np.array([TB_PLAIN[0] + gi * (1.0 - x), TB_PLAIN[1] + ge * (1.0 - x)])


class TwoTemperatureCriticalGradient(manta.TransportSystem):
    # Zero flux on the axis, as a mixed condition with d = 1 -- the flux carries
    # a factor of x, so it vanishes there for any gradient. See
    # ../jardin-critical-gradient/README.md for why a Neumann end is wrong here.
    variables = [
        manta.Field("Ti", "ion temperature", "", lower=manta.Mixed(d=1.0), upper=manta.Dirichlet),
        manta.Field("Te", "electron temperature", "", lower=manta.Mixed(d=1.0), upper=manta.Dirichlet),
    ]

    # InPlace: nothing here depends on where the case is evaluated.
    regrid = manta.Regrid.InPlace

    def __init__(self, config=None, grid=None, exchange=False):
        super().__init__()
        self.exchange = exchange
        if exchange:
            self.nu = NU
            self.H = np.array([HI_EXCHANGE, ExchangeHeating()])
            self.Tb = TB_EXCHANGE
        else:
            self.nu = 0.0
            self.H = H_PLAIN
            self.Tb = TB_PLAIN
        self.reset_counts()

    # Counted per *point*: one visit evaluates the model at one point for both
    # channels, which is what a transport model call does. The solver asks for
    # each channel's flux separately, so only the ion channel's calls count.
    def reset_counts(self):
        self.nFlux = 0
        self.nDeriv = 0

    # --- boundaries --------------------------------------------------------
    def LowerBoundary(self, index, t):
        return 0.0

    def UpperBoundary(self, index, t):
        return self.Tb[index]

    # --- physics -----------------------------------------------------------
    def SigmaFn(self, index, state, x, t):
        if index == 0:
            self.nFlux += 1
        return fluxes(x, state.q[0], state.q[1])[index]

    def Sources(self, index, state, x, t):
        exch = self.nu * (state.u[1] - state.u[0])
        return self.H[index] + (exch if index == 0 else -exch)

    # --- derivatives -------------------------------------------------------
    # d(sigma_s)/d(q_r) = x [delta_sr chi_s + q_s dchi_s/dq_r]: the tangent
    # diffusivity matrix, coupled through the cross-thresholds. Supplied
    # analytically, so a Jacobian costs no flux evaluations.
    def dSigmaFn_dq(self, index, state, x, t):
        if index == 0:
            self.nDeriv += 1
        qi, qe = state.q[0], state.q[1]
        c = chi(qi, qe)[index]
        d = dchi_dq(qi, qe)[index]
        q = (qi, qe)[index]
        out = np.array([x * q * float(d[0]), x * q * float(d[1])])
        out[index] += x * c
        return out

    def dSources_du(self, index, state, x, t):
        return np.array([-self.nu, self.nu]) if index == 0 else np.array([self.nu, -self.nu])

    # --- initial condition -------------------------------------------------
    # Jardin's initial gradient, 1, in each channel: above every threshold, and
    # not the answer. (The constant-chi steady state, Jardin's other reading,
    # starts the exchange variant at gradients of about 6, from which IDA's
    # first step fails; see README.md.)
    def InitialValue(self, index, x):
        return self.Tb[index] + (1.0 - x)

    def InitialDerivative(self, index, x):
        return -1.0


def registerTransportSystems():
    manta.registerPhysicsCase("TwoTemperatureCriticalGradient", TwoTemperatureCriticalGradient)
