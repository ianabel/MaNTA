"""Density and temperature coupled through a thermodiffusive pinch: a two-channel
benchmark with a closed-form, non-polynomial steady state.

Every benchmark in the paper's comparison set -- Park, Jardin, Shestakov -- has
one channel. This one has two, coupled in both directions through the fluxes,
and its steady state is still known exactly. On the cylinder `V' = x`, in the
conservative form MaNTA integrates, `a d_t u - d_x[sigma_hat] = S`:

    d_t n - d_x[ x D (n' - C_T n T'/T) ]              = 0
    d_t T - d_x[ x n chi0 (T/T_b)^alpha T' ]           = x H

    sigma_hat(0) = 0 for both                  (mixed, d = 1: zero flux)
    n(1) = n_b,  T(1) = T_b                    (Dirichlet)

The particle flux is diffusion plus a pinch, Gamma = -D n' + V n with
V = C_T D T'/T: thermodiffusion, inward wherever T falls outward. The heat flux
is gyro-Bohm-like, its diffusivity rising as T^alpha, and carries the density.

**The steady state.** With no particle source and no flux on the axis the
particle flux vanishes identically, so n' / n = C_T T' / T and

    n = n_b (T/T_b)^C_T.

Putting that into the heat flux and integrating once, the Kirchhoff transform
gives

    (T/T_b)^beta = 1 + beta H (1 - x^2) / (4 n_b chi0 T_b),    beta = C_T + alpha + 1.

With the defaults below -- alpha = 3/2, C_T = 1/2, H = 4 and n_b = T_b =
chi0 = 1 -- that is

    T = (4 - 3 x^2)^(1/3),      n = (4 - 3 x^2)^(1/6),

peaked 1.59x in T and 1.26x in n. Neither is a polynomial, so this measures
spatial accuracy per evaluation, as Park's problem does, but on a coupled
system.

**H is set by the analytic radius, not by the physics.** Both profiles have a
branch point where 4 - 3x^2 vanishes, at x = 1.155 -- 0.155 outside the wall.
A larger H peaks the profiles more and moves it in: H = 16 gives
(13 - 12x^2)^(1/3), whose branch point is 0.042 from the wall, and there the
maximum error is still pre-asymptotic at 32 cells, converging at k = 3 at a
rate under 3. At H = 4 the L1 rate at k = 3 is 3.6 at 8 cells and 3.98 at 64.

**What it exercises that one channel cannot.** The Jacobian has off-diagonal
blocks in both directions: d(sigma_n)/d(q_T) = -x D C_T n/T and
d(sigma_n)/d(T) through the pinch, and d(sigma_T)/d(n) through the heat flux.
And **the steady state does not depend on D**: the particle diffusivity sets how
fast the density relaxes relative to the temperature, i.e. how much stiffer one
channel is than the other, without moving the answer. A sweep in D is therefore
a cost-against-stiffness-disparity axis with a fixed reference solution, which
no single-channel benchmark offers. See benchmark.py.
"""

import numpy as np

import manta

CHI0 = 1.0      # heat diffusivity at T = T_b
ALPHA = 1.5     # gyro-Bohm: chi ~ T^(3/2)
C_T = 0.5       # thermodiffusion coefficient
H = 4.0         # heating
N_B = 1.0       # edge density
T_B = 1.0       # edge temperature
BETA = C_T + ALPHA + 1.0


def ExactT(x):
    return T_B * (1.0 + BETA * H * (1.0 - x * x) / (4.0 * N_B * CHI0 * T_B)) ** (1.0 / BETA)


def ExactN(x):
    return N_B * (ExactT(x) / T_B) ** C_T


def ExactSolution(x):
    """(n, T) at the points x, stacked as a (2, len(x)) array."""
    x = np.asarray(x, dtype=float)
    return np.array([ExactN(x), ExactT(x)])


# The fluxes and their derivatives, written once on numpy arrays so that the
# finite-difference reference algorithms in ../reference-algorithms/ evaluate
# exactly the physics this case does.

def fluxes(x, n, T, qn, qT, D):
    """(sigma_n, sigma_T)."""
    chi = CHI0 * (T / T_B) ** ALPHA
    return x * D * (qn - C_T * n * qT / T), x * n * chi * qT


def pinch_coefficients(x, n, T, qn, qT, D):
    """The fluxes in the diffusivity-and-pinch form a transport code is handed:
    sigma_s = Dcoef_s q_s + V_s u_s for each channel, with the cross-coupling
    carried in the coefficients. The pinch V_n = -x D C_T q_T / T is the
    thermodiffusion; the temperature channel has none."""
    chi = CHI0 * (T / T_B) ** ALPHA
    Dn = x * D * np.ones_like(n)
    Vn = -x * D * C_T * qT / T
    DT = x * n * chi
    return np.array([Dn, DT]), np.array([Vn, np.zeros_like(T)])


class ThermodiffusivePinch(manta.TransportSystem):
    # Zero *flux* on the axis for both channels -- as a mixed condition with
    # d = 1, which says exactly that: both fluxes carry a factor of x and vanish
    # there for any gradient, so nothing more may be imposed. See
    # ../jardin-critical-gradient/README.md for what a Neumann end would do.
    variables = [
        manta.Field("n", "density", "", lower=manta.Mixed(d=1.0), upper=manta.Dirichlet),
        manta.Field("T", "temperature", "", lower=manta.Mixed(d=1.0), upper=manta.Dirichlet),
    ]

    # InPlace: nothing here depends on where the case is evaluated.
    regrid = manta.Regrid.InPlace

    # `D` is the particle diffusivity. It moves the transient and the relative
    # stiffness of the two channels, and not the steady state.
    def __init__(self, config=None, grid=None, D=1.0):
        super().__init__()
        self.D = D
        self.reset_counts()

    # Counted per *point*, not per channel: one visit evaluates the model at one
    # point for both channels, which is what a transport model call does and
    # what the reference algorithms count. The solver asks for each channel's
    # flux separately, so only the n channel's calls are counted.
    def reset_counts(self):
        self.nFlux = 0        # point-evaluations of the fluxes
        self.nDeriv = 0       # point-evaluations of their derivatives

    # --- boundaries --------------------------------------------------------
    def LowerBoundary(self, index, t):
        return 0.0

    def UpperBoundary(self, index, t):
        return N_B if index == 0 else T_B

    # --- physics -----------------------------------------------------------
    def SigmaFn(self, index, state, x, t):
        if index == 0:
            self.nFlux += 1
        sn, sT = fluxes(x, state.u[0], state.u[1], state.q[0], state.q[1], self.D)
        return sn if index == 0 else sT

    def Sources(self, index, state, x, t):
        return 0.0 if index == 0 else x * H

    # --- derivatives -------------------------------------------------------
    def dSigmaFn_dq(self, index, state, x, t):
        n, T = state.u[0], state.u[1]
        if index == 0:
            self.nDeriv += 1
            return np.array([x * self.D, -x * self.D * C_T * n / T])
        chi = CHI0 * (T / T_B) ** ALPHA
        return np.array([0.0, x * n * chi])

    def dSigmaFn_du(self, index, state, x, t):
        n, T, qT = state.u[0], state.u[1], state.q[1]
        if index == 0:
            return np.array([-x * self.D * C_T * qT / T, x * self.D * C_T * n * qT / T**2])
        chi = CHI0 * (T / T_B) ** ALPHA
        return np.array([x * chi * qT, x * n * ALPHA * chi / T * qT])

    # --- initial condition -------------------------------------------------
    # Flat density, and a temperature peaked twofold: the right boundary values,
    # the wrong shape for both.
    def InitialValue(self, index, x):
        return N_B if index == 0 else T_B * (2.0 - x * x)

    def InitialDerivative(self, index, x):
        return 0.0 if index == 0 else -2.0 * T_B * x


def registerTransportSystems():
    manta.registerPhysicsCase("ThermodiffusivePinch", ThermodiffusivePinch)
