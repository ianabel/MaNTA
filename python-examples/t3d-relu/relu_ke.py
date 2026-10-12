"""T3D's `test-w7x-relu-ke` benchmark, posed for MaNTA.

T3D (Trinity3D) evolves n_e, p_H and p_e in W7-X geometry with its analytic
`ReLU` flux model standing in for GX. Everything here is T3D's: the equations,
normalisations, geometry and sources come from `t3d_relu_ke.npz`, which
`extract_t3d.py` writes by running T3D's own initialisation on the benchmark's
input. In Trinity units, with V' = dV/drho / a^3 and A = <|grad rho|> dV/drho / a^2,

    V' d_t n   + d_rho[ A/B^2 p_H^1.5 n^-0.5 Gamma_e ]          = 0
    V' d_t p_s + d_rho[ (2/3) A/B^2 p_H^2.5 n^-1.5 Q_s ]       = V' ( (2/3) S_aux,s + C_s )

so sigma_hat is minus the bracket. n_H = n_e (`equal_to`), hydrogen is the
reference species, C_s is T3D's collisional exchange, and S_aux is a 5 MW
electron Gaussian at rho = 0.1, width 0.05, normalised over the continuous
volume. The axis carries zero flux and rho = 0.8 is held at the input's edge
values. The normalised fluxes are T3D's ReLU model, in kappa = -d ln/d rho:

    Gamma_e = D n kappa_n   + sign(kappa_n - a) |m (kappa_n - a)|^2 (1 + noise U)
    Q_s     = D p_s kappa_ps + m (kappa_Ts - a)^2 (1 + noise U)  if kappa_Ts > a, else 1e-16

with U uniform on [-1, 1], drawn afresh on every call. T3D draws it once per call
for the whole Gamma profile -- its `Linear` is not vectorised the way `ReLU` is --
so the particle-flux noise is fully correlated across radius and the heat-flux
noise independent at every point. That is reproduced here.

The black box is `ReluFluxes`: local (n, p_H, p_e, kappa_n, kappa_pH, kappa_pe)
in, (Gamma_e, Q_H, Q_e) out, as GX would be called. The prefactors, geometry and
sources around it are cheap and analytic. `ReluKeCase` builds MaNTA's Jacobian
from it one of three ways:

* "control" -- exact derivatives of the noiseless box: the reference;
* "full"    -- forward differences in all six box inputs;
* "t3d"     -- T3D's scheme: one perturbation per kappa, with T3D's step
               max(rel |kappa|, abs) (rel = abs = 0.5 by default) and its signs,
               the box's explicit dependence on n and p dropped, and -- as T3D's
               psi matrices do when the evolved density is not the reference
               species' -- the gyro-Bohm prefactor's density derivative too.

Every differenced build keeps each channel's d K_i / d kappa_i at or above the
known neoclassical part, D u_i. The kappa block is lower triangular here (Gamma_e
sees only kappa_n; Q_s sees kappa_n and kappa_ps), so that keeps the q block of
the Jacobian definite.
"""

import os

import numpy as np
from scipy.interpolate import CubicSpline

import manta

_HERE = os.path.dirname(os.path.abspath(__file__))
_D = np.load(os.path.join(_HERE, "t3d_relu_ke.npz"))

A_VMEC = float(_D["a_vmec"])
B_TOR = float(_D["Btor"])
T_REF = float(_D["t_ref"])
P_REF_MWM3 = float(_D["P_ref_MWm3"])
M_H, M_E = float(_D["m_H"]), float(_D["m_e"])
Z_H, Z_E = float(_D["Z_H"]), float(_D["Z_e"])
D_NEO = float(_D["D_neo"])
POWER = float(_D["power"])
RHO_EDGE = float(_D["rho_edge"])
# relu_s = [n critical gradient, n slope, p critical gradient, p slope]
RELU_H, RELU_E = _D["relu_H"], _D["relu_e"]

_r = _D["vmec_rho"]
VPRIME = CubicSpline(_r, _D["vmec_dVdrho"] / A_VMEC ** 3)
AREA = CubicSpline(_r, _D["vmec_area"] / A_VMEC ** 2)

_w, _c, _P = (float(v) for v in _D["Sp_aux"])


def _gauss(rho):
    return np.exp(-((rho - _c) / _w) ** 2 / 2)


_q = np.linspace(0.0, RHO_EDGE, 20001)
_SAUX = _P / (np.trapezoid(_gauss(_q) * VPRIME(_q), _q) * A_VMEC ** 3 * P_REF_MWM3)


def S_aux_e(rho):
    """The electron auxiliary pressure source, Trinity units, 5 MW in all."""
    return _SAUX * _gauss(rho)


# Initial profiles: T3D's, from its fine-grid initialisation.
_g = _D["rho"]
INITIAL = [CubicSpline(_g, _D[k]) for k in ("n_e", "p_H", "p_e")]
EDGE = [float(_D[k][-1]) for k in ("n_e", "p_H", "p_e")]


def _below(kT, a, one_sided):
    """Where the ReLU gives no turbulent flux.

    T3D's ReLU is mirror-symmetric, zero only for |kappa_T| < a. Below -a it gives
    m (kappa_T + a)^2 > 0 with dQ/dkappa_T < 0: an anti-diffusive branch, and a
    second kappa_T for every flux. Where the enclosed heating is small -- inside
    this benchmark's heating peak -- both branches carry it, and the steady state
    there is not unique: noiseless, T3D's own N = 17 and 65 solutions and MaNTA's
    at several resolutions sit on the negative branch near the axis, the others on
    the positive one, and the central T_e wanders by 25% under refinement in both
    codes. `one_sided` keeps the physical half, kappa_T > a, and is the default.
    """
    return kT < a if one_sided else np.abs(kT) < a


class ReluFluxes:
    """T3D's ReLU flux model as a counted, noisy black box.

    Inputs are (nPoints,) arrays; returns K, (nPoints, 3): Gamma_e, Q_H, Q_e.
    """

    def __init__(self, noise=0.0, seed=0, one_sided=True):
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        self.one_sided = one_sided
        self.calls = 0

    def _u(self, shape):
        if not self.noise:
            return 0.0
        return self.rng.uniform(-1.0, 1.0, shape)

    def __call__(self, n, pH, pe, kn, kpH, kpe):
        self.calls += 1
        a_n, m_n = RELU_E[0], RELU_E[1]
        lin = np.sign(kn - a_n) * np.abs(m_n * (kn - a_n)) ** POWER
        gamma = D_NEO * n * kn + lin * (1 + self.noise * self._u(()))
        Q = []
        for p, kp, relu in ((pH, kpH, RELU_H), (pe, kpe, RELU_E)):
            kT = kp - kn
            a, m = relu[2], relu[3]
            turb = np.where(_below(kT, a, self.one_sided), 1e-16, m * (kT - a * np.sign(kT)) ** POWER)
            Q.append(D_NEO * p * kp + turb * (1 + self.noise * self._u(kT.shape)))
        return np.stack([gamma, Q[0], Q[1]], axis=1)

    def exact_derivatives(self, n, pH, pe, kn, kpH, kpe):
        """Noiseless dK/dkappa (n,3,3) and dK/du at fixed kappa (n,3,3)."""
        npts = n.shape[0]
        dk = np.zeros((npts, 3, 3)); du = np.zeros((npts, 3, 3))
        a_n, m_n = RELU_E[0], RELU_E[1]
        dk[:, 0, 0] = D_NEO * n + POWER * m_n ** POWER * np.abs(kn - a_n) ** (POWER - 1)
        du[:, 0, 0] = D_NEO * kn
        for i, (p, kp, relu) in enumerate(((pH, kpH, RELU_H), (pe, kpe, RELU_E)), start=1):
            kT = kp - kn
            a, m = relu[2], relu[3]
            dturb = np.where(_below(kT, a, self.one_sided), 0.0,
                             POWER * m * (kT - a * np.sign(kT)) ** (POWER - 1))
            dk[:, i, i] = D_NEO * p + dturb
            dk[:, i, 0] = -dturb
            du[:, i, i] = D_NEO * kp
        return dk, du


def _nuT(ms, mu, Zs, Zu, Ts, Tu, nu, lnL):
    """T3D's nu_equil for species s on u, per unit Trinity time."""
    nu_su = 2.85e2 * (Zs * Zu) ** 2 / ms ** 0.5 * lnL * nu / Ts ** 1.5
    return T_REF * 8.0 / (3 * np.sqrt(np.pi)) * nu_su * (mu / ms) ** 0.5 * (mu / ms + Tu / Ts) ** -1.5


def collisional_exchange(n, pH, pe):
    """T3D's coll_eq for (H, e): p_s nu_su (T_u/T_s - 1), Trinity units."""
    TH, Te = pH / n, pe / n
    lnL = 24.0 - np.log(np.sqrt(n * 1e14) / (Te * 1e3))   # ie and ei alike
    CH = pH * _nuT(M_H, M_E, Z_H, Z_E, TH, Te, n, lnL) * (Te / TH - 1)
    Ce = pe * _nuT(M_E, M_H, Z_E, Z_H, Te, TH, n, lnL) * (TH / Te - 1)
    return CH, Ce


class ReluKeCase(manta.TransportSystem):
    variables = [
        manta.Field("n_e", "electron density", "1e20 m^-3", lower=manta.Mixed(d=1.0)),
        manta.Field("p_H", "hydrogen pressure", "1e20 m^-3 keV", lower=manta.Mixed(d=1.0)),
        manta.Field("p_e", "electron pressure", "1e20 m^-3 keV", lower=manta.Mixed(d=1.0)),
    ]
    regrid = manta.Regrid.InPlace

    def __init__(self, box=None, jacobian="control", rel_step=0.5, abs_step=0.5,
                 reuse_base=True):
        super().__init__()
        if jacobian not in ("control", "full", "t3d"):
            raise ValueError(f"unknown jacobian {jacobian!r}")
        self.box = box if box is not None else ReluFluxes()
        self.jacobian = jacobian
        self.rel_step, self.abs_step = rel_step, abs_step
        self.reuse_base = reuse_base
        self._last = None
        self.reset_counts()

    def reset_counts(self):
        self.value_calls = 0
        self.fd_calls = 0
        self.base_reused = 0
        self.builds = 0
        self.guarded = 0

    # --- boundaries and initial condition -----------------------------------
    def LowerBoundary(self, index, t):
        return 0.0                       # sigma = 0 on the axis

    def UpperBoundary(self, index, t):
        return EDGE[index]

    def InitialValue(self, index, x):
        return float(INITIAL[index](x))

    def InitialDerivative(self, index, x):
        return float(INITIAL[index](x, 1))

    def aFn(self, index, x):
        return float(VPRIME(x))

    # --- the composition around the box -------------------------------------
    @staticmethod
    def _inputs(states, positions):
        u = np.array(states["Variable"], dtype=float)
        q = np.array(states["Derivative"], dtype=float)
        x = np.array(positions, dtype=float)
        return u, q, x

    @staticmethod
    def _prefactor(u, x):
        """C (n,3) with sigma_hat = -C K, and dC/du (n,3,3)."""
        n, pH = u[:, 0], u[:, 1]
        g = AREA(x) / B_TOR ** 2
        C = np.empty_like(u)
        C[:, 0] = g * pH ** 1.5 * n ** -0.5
        C[:, 1] = C[:, 2] = (2.0 / 3.0) * g * pH ** 2.5 * n ** -1.5
        dC = np.zeros((u.shape[0], 3, 3))
        dC[:, 0, 0] = -0.5 * C[:, 0] / n
        dC[:, 0, 1] = 1.5 * C[:, 0] / pH
        for i in (1, 2):
            dC[:, i, 0] = -1.5 * C[:, i] / n
            dC[:, i, 1] = 2.5 * C[:, i] / pH
        return C, dC

    @staticmethod
    def _sources(u, x):
        n, pH, pe = u[:, 0], u[:, 1], u[:, 2]
        CH, Ce = collisional_exchange(n, pH, pe)
        Vp = VPRIME(x)
        return np.stack([np.zeros_like(n), Vp * CH, Vp * ((2.0 / 3.0) * S_aux_e(x) + Ce)], axis=1)

    def _source_jacobian(self, u, x):
        # Cheap, smooth and noiseless: central differences at a tiny step.
        dS = np.zeros((u.shape[0], 3, 3))
        for j in range(3):
            h = 1e-7 * np.abs(u[:, j])
            up, um = u.copy(), u.copy()
            up[:, j] += h; um[:, j] -= h
            dS[:, :, j] = (self._sources(up, x) - self._sources(um, x)) / (2 * h[:, None])
        return dS

    def ComputePhysics(self, states, positions, t):
        u, q, x = self._inputs(states, positions)
        kappa = -q / u
        K = self.box(u[:, 0], u[:, 1], u[:, 2], kappa[:, 0], kappa[:, 1], kappa[:, 2])
        self.value_calls += 1
        self._last = (t, u, q, K)
        C, _ = self._prefactor(u, x)
        return [list((-C * K).T), list(self._sources(u, x).T), []]

    def _base(self, u, q, kappa, t):
        last = self._last
        if (self.reuse_base and last is not None and last[0] == t
                and np.array_equal(last[1], u) and np.array_equal(last[2], q)):
            self.base_reused += 1
            return last[3]
        self.fd_calls += 1
        return self.box(*u.T, *kappa.T)

    def _differenced(self, u, kappa, t, q):
        """dK/dkappa and dK/du|kappa from the box, by forward differences."""
        npts = u.shape[0]
        dk = np.zeros((npts, 3, 3)); du = np.zeros((npts, 3, 3))
        K0 = self._base(u, q, kappa, t)
        for j in range(3):
            step = np.maximum(self.rel_step * np.abs(kappa[:, j]), self.abs_step)
            if j == 0:
                step = -step                 # T3D steps kappa_n downwards
            kp = kappa.copy(); kp[:, j] += step
            dk[:, :, j] = (self.box(*u.T, *kp.T) - K0) / step[:, None]
            self.fd_calls += 1
        if self.jacobian == "full":
            for j in range(3):
                h = self.rel_step * np.abs(u[:, j])
                up = u.copy(); up[:, j] += h
                du[:, :, j] = (self.box(*up.T, *kappa.T) - K0) / h[:, None]
                self.fd_calls += 1
        # Keep each channel's own diffusivity at or above the neoclassical part.
        for i in range(3):
            floor = D_NEO * u[:, i]
            low = dk[:, i, i] < floor
            self.guarded += int(np.count_nonzero(low))
            dk[low, i, i] = floor[low]
        return dk, du

    def ComputePhysicsDerivatives(self, states, positions, t):
        u, q, x = self._inputs(states, positions)
        npts = u.shape[0]
        kappa = -q / u
        self.builds += 1
        if self.jacobian == "control":
            clean = ReluFluxes(noise=0.0, one_sided=self.box.one_sided)
            K = clean(*u.T, *kappa.T)
            dk, du = clean.exact_derivatives(*u.T, *kappa.T)
        else:
            dk, du = self._differenced(u, kappa, t, q)
            K = self._last[3] if self._last is not None and np.array_equal(self._last[1], u) \
                else self.box(*u.T, *kappa.T)
        C, dC = self._prefactor(u, x)
        if self.jacobian == "t3d":
            # T3D's psi matrices carry the prefactor's density derivative only
            # for the reference species' own density (delta_sr). The evolved
            # density here is the electrons' and the reference is hydrogen,
            # so T3D drops it, though n_H is tied to n_e.
            dC[:, :, 0] = 0.0

        # sigma_hat_i = -C_i K_i(u, kappa(u, q)), kappa_j = -q_j / u_j
        dsdu = -dC * K[:, :, None] - C[:, :, None] * (du + dk * (kappa / u)[:, None, :] * -1.0)
        dsdq = C[:, :, None] * dk / u[:, None, :]
        dSdu = self._source_jacobian(u, x)

        def rows(dU, dQ):
            return [{
                "Variable": dU[:, i, :],
                "Derivative": dQ[:, i, :],
                "Flux": np.zeros((npts, 3)),
                "Aux": np.zeros((npts, 0)),
                "Scalars": np.zeros(0),
            } for i in range(3)]

        return [rows(dsdu, dsdq), rows(dSdu, np.zeros_like(dSdu)), []]
