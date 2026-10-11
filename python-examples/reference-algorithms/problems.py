"""The benchmark problems, in the form a finite-difference transport code is
handed them.

Each problem is the physics of one example directory beside this one --
imported from it, not restated -- with the pieces ASTRA's scheme and TGYRO's
iteration need:

  flux(x, U, Q)          sigma_hat for every channel at the points x, from the
                         profiles U and their gradients Q. What TGYRO calls.
  coefficients(x, U, Q)  the same fluxes as sigma_s = D_s q_s + V_s u_s: a
                         diffusivity and a pinch per channel, which is the form
                         ASTRA's transport equation takes (manual eq. (60)).
                         Whatever couples the channels sits in the coefficients
                         and is lagged with them.
  source(x, U)           S = S0 + S1 * U per channel: an explicit part and a
                         coefficient linear in the channel's own profile, which
                         ASTRA treats implicitly.

All arrays are (nvars, npoints). `flux` and `coefficients` each count one
evaluation per point -- the model at one point, every channel at once -- in
`evals`, which is the unit MaNTA's examples count in. Sources are cheap in all
four problems and are not counted, as TGYRO does not count its targets.

The sign convention is MaNTA's: d_t u - d_x[sigma_hat] = S, on [0, 1], zero
flux at x = 0 and a Dirichlet value at x = 1.
"""

import pathlib
import sys

import numpy as np

EXAMPLES = pathlib.Path(__file__).resolve().parent.parent
for d in ("park-convergence", "jardin-critical-gradient", "thermodiffusive-pinch",
          "two-temperature-critical-gradient"):
    sys.path.insert(0, str(EXAMPLES / d))

import park_convergence as park                       # noqa: E402
import jardin_critical_gradient as jardin              # noqa: E402
import thermodiffusive_pinch as pinch                  # noqa: E402
import two_temperature_critical_gradient as twotemp    # noqa: E402


class Problem:
    name = ""
    names = ()
    # A value added to every profile before TGYRO takes its logarithmic
    # gradient, for a problem whose Dirichlet value is zero. Only problems
    # whose physics is invariant under u -> u + c may set it.
    tgyro_offset = 0.0
    # Pereverzev-Corrigan's added diffusivity for the ASTRA scheme. It wants to
    # exceed the slope dq/d(eta) of the flux-gradient curve. It is a
    # *diffusivity*: ASTRA's equation carries the geometry V' outside it, so
    # the flux it adds is geometry(x) * dbar * u_x.
    dbar = 0.0

    def geometry(self, x):
        """V', the factor every flux here carries. All four problems are
        cylindrical."""
        return np.asarray(x, dtype=float)

    def __init__(self):
        self.evals = 0

    @property
    def nvars(self):
        return len(self.names)

    def flux(self, x, U, Q):
        self.evals += np.size(x)
        return self._flux(x, U, Q)

    def coefficients(self, x, U, Q):
        self.evals += np.size(x)
        return self._coefficients(x, U, Q)

    def error(self, profile, sample=np.linspace(0.0, 1.0, 201)):
        """Relative L1 error against the closed form, worst channel."""
        u, e = profile(sample), self.exact(sample)
        return float(max(np.sum(np.abs(u[s] - e[s])) / np.sum(np.abs(e[s]))
                         for s in range(self.nvars)))


class Park(Problem):
    """Linear: sigma_hat = x q. Park's spatial-accuracy problem."""
    name = "Park"
    names = ("T",)
    tgyro_offset = 1.0      # u(1) = 0; the problem only sees q
    ub = np.array([0.0])

    def exact(self, x):
        return park.ExactSolution(np.asarray(x))[None, :]

    def initial(self, x):
        return np.zeros((1, np.size(x)))

    def _flux(self, x, U, Q):
        return x * Q

    def _coefficients(self, x, U, Q):
        return np.array([x]), np.zeros_like(U)

    def source(self, x, U):
        return 4.0 * x * (1.0 - x * x) * np.exp(1.0 - x * x) * np.ones_like(U), np.zeros_like(U)


def _jardin_chi(q):
    return jardin.CHI0 + jardin.KAPPA * np.maximum(np.abs(q) - jardin.QC, 0.0) ** jardin.ALPHA


class Jardin(Problem):
    """sigma_hat = x chi(q) q, chi switched on above a critical gradient."""
    name = "Jardin"
    names = ("T",)
    tgyro_offset = 1.0      # u(1) = 0; chi depends on q alone
    dbar = 30.0             # the tangent diffusivity is about 28 at the answer
    ub = np.array([0.0])

    def exact(self, x):
        return jardin.ExactSolution(np.asarray(x))[None, :]

    def initial(self, x):
        return (1.0 - np.asarray(x))[None, :]

    def _flux(self, x, U, Q):
        return x * _jardin_chi(Q) * Q

    def _coefficients(self, x, U, Q):
        # What a critical-gradient model hands a transport code: the effective
        # diffusivity, flux over gradient. No pinch.
        return x * _jardin_chi(Q), np.zeros_like(U)

    def source(self, x, U):
        return np.ones_like(U), np.zeros_like(U)


class Pinch(Problem):
    """Density and temperature, coupled through a thermodiffusive pinch."""
    name = "Pinch"
    names = ("n", "T")
    ub = np.array([pinch.N_B, pinch.T_B])

    def __init__(self, D=1.0):
        super().__init__()
        self.D = D
        self.name = f"Pinch (D = {D:g})"

    def exact(self, x):
        return pinch.ExactSolution(x)

    def initial(self, x):
        x = np.asarray(x)
        return np.array([pinch.N_B * np.ones_like(x), pinch.T_B * (2.0 - x * x)])

    def _flux(self, x, U, Q):
        return np.array(pinch.fluxes(x, U[0], U[1], Q[0], Q[1], self.D))

    def _coefficients(self, x, U, Q):
        return pinch.pinch_coefficients(x, U[0], U[1], Q[0], Q[1], self.D)

    def source(self, x, U):
        S0 = np.array([np.zeros_like(x), x * pinch.H])
        return S0 * np.ones_like(U), np.zeros_like(U)


class TwoTemperature(Problem):
    """Ion and electron temperatures with cross-coupled critical gradients."""
    names = ("Ti", "Te")
    dbar = 30.0             # the tangent diffusivities are 16-26 at the answer

    def __init__(self, exchange=False):
        super().__init__()
        self.exchange = exchange
        self.name = "Two-temperature" + (" + exchange" if exchange else "")
        # The parameters exactly as the MaNTA case derives them.
        if exchange:
            self.nu = twotemp.NU
            self.H = np.array([twotemp.HI_EXCHANGE, twotemp.ExchangeHeating()])
            self.ub = twotemp.TB_EXCHANGE
        else:
            self.nu, self.H, self.ub = 0.0, twotemp.H_PLAIN, twotemp.TB_PLAIN

    def exact(self, x):
        return twotemp.ExactSolution(x, self.exchange)

    def initial(self, x):
        x = np.asarray(x)
        return np.array([self.ub[0] + (1.0 - x), self.ub[1] + (1.0 - x)])

    def _flux(self, x, U, Q):
        return np.array(twotemp.fluxes(x, Q[0], Q[1]))

    def _coefficients(self, x, U, Q):
        ci, ce = twotemp.chi(Q[0], Q[1])
        return np.array([x * ci, x * ce]), np.zeros_like(U)

    def source(self, x, U):
        # S_i = H_i + nu (Te - Ti): the Ti part implicit, the Te part explicit,
        # and the same the other way round -- ASTRA's split of a source into a
        # part linear in the unknown and a remainder.
        one = np.ones_like(U[0])
        S0 = np.array([self.H[0] + self.nu * U[1], self.H[1] + self.nu * U[0]])
        S1 = np.array([-self.nu * one, -self.nu * one])
        return S0, S1


def all_problems():
    return [Park(), Jardin(), Pinch(), TwoTemperature(), TwoTemperature(exchange=True)]
