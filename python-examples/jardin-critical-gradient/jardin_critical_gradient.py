"""Jardin's stiff critical-gradient benchmark, from Journal of Computational
Physics 227 (2008) 8769-8775 (`refs/PTRANSP.pdf` -- despite the file name, that
paper is not about PTRANSP).

The problem that paper opens with: 1-D diffusion whose diffusivity depends
strongly and non-analytically on the *gradient*, mimicking a critical-gradient
turbulent transport model such as GLF23. It is the problem every algorithm in
`refs/` exists to solve, and the one on which a plain implicit time step fails:
Jardin shows backward Euler oscillating and needing a step some four orders of
magnitude smaller than his linearised scheme to reach the right answer.

In the conservative form MaNTA integrates, `a d_t u - d_x[sigma_hat] = S`:

    d_t u - d_x[ x chi(q) q ] = 1,       q = du/dx

    chi(q) = chi0 + kappa (|q| - qc)^alpha    for |q| > qc
           = chi0                             otherwise

    q(0) = -g                            (Neumann -- see below, it fixes q)
    u(1) = 0                             (Dirichlet)

with Jardin's chi0 = 1, kappa = 10, alpha = 0.5, qc = 0.5. The initial
condition is the constant-chi steady state u = 1 - x, which is where Jardin
starts.

**The stiff steady state is exactly linear.** Integrating once and requiring
regularity on the axis gives chi(q) q = -1, so q is a constant -g solving

    [chi0 + kappa (g - qc)^alpha] g = 1,     g = 0.5092841043...

and u = g (1 - x). Being a degree-1 polynomial it lies in P_k for every order
this solver runs at, so the benchmark measures the *nonlinear* solve rather
than the spatial discretisation -- the complement of
`../park-convergence/`, which measures the other half.

**The boundary condition is the trap here, not the stiffness.** Jardin's problem
has *no* condition on the axis: sigma_hat = x chi(q) q vanishes there for any q,
so regularity alone picks the solution. MaNTA's Neumann boundary does not
express that -- it fixes `q` -- so asking for a zero Neumann value imposes
q(0) = 0, an extra condition, and a false one. That is a wrong problem, not a
hard one, and it shows up as a first-order error independent of polynomial
degree, from a one-cell layer on the axis. Supplying the true gradient -g
instead takes the error from 7e-4 to machine precision at every resolution. See
README.md; docs/physics_interface.rst carries the general warning.

Three more things worth knowing before using this as a test of anything.

`g` sits only 0.009 above the critical gradient `qc`, and dchi/dq diverges
there, so the steady state lives right against the kink. Starting a run *from*
the exact steady state makes `IDACalcIC` fail with IDA_CONV_FAIL (-4); starting
from Jardin's u = 1 - x, as this case does, converges at every resolution
tried.

Jardin's Section on implementing this in TSC and PTRANSP warns that the tangent
diffusivity `chi + (dchi/dq) q` -- which is what `dSigmaFn_dq` returns, divided
by x -- can go negative, "such as could happen at a transport barrier
bifurcation", and that they had to limit it to keep it positive. **MaNTA has no
such limiter**, and nothing here needs one: at this steady state the tangent
diffusivity is about 28x, comfortably positive. A case that wanders into the
negative region is on its own.
"""

import numpy as np

import manta

CHI0 = 1.0      # Jardin's chi_0
KAPPA = 10.0    # ... his k, the strength of the stiffness
ALPHA = 0.5     # ... his alpha
QC = 0.5        # ... his T'_c, the critical gradient


def chi(q):
    s = abs(q)
    return CHI0 + KAPPA * (s - QC) ** ALPHA if s > QC else CHI0


def dchi_dq(q):
    s = abs(q)
    if s <= QC:
        return 0.0
    return np.sign(q) * KAPPA * ALPHA * (s - QC) ** (ALPHA - 1.0)


def CriticalGradient():
    """The constant g with chi(-g) g = 1: the steady gradient.

    Bisection rather than a library root-find, to keep this example's
    dependencies down to numpy. The bracket is safe because
    f(g) = [chi0 + kappa (g-qc)^alpha] g increases monotonically for g > qc,
    with f(qc) = qc < 1 and f(1) = chi0 + kappa (1-qc)^alpha > 1.
    """
    lo, hi = QC, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if chi(mid) * mid < 1.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def ExactSolution(x):
    """The stiff steady state, u = g (1 - x)."""
    return CriticalGradient() * (1.0 - x)


class JardinCriticalGradient(manta.TransportSystem):
    variables = [manta.Field("T", "temperature", "", lower=manta.Neumann,
                             upper=manta.Dirichlet)]

    # A registered case is built by the factory as `(config, grid)`; both are
    # defaulted so `benchmark.py` can build one directly for a `manta.Runner`.
    def __init__(self, config=None, grid=None):
        super().__init__()
        self.reset_counts()

    # PERFORMANCE.md measures MaNTA by the number of calls into a
    # TransportSystem, so this case counts its own. `benchmark.py` reads them;
    # a `manta run.conf` run simply ignores them.
    def reset_counts(self):
        self.nFlux = 0        # SigmaFn point-evaluations
        self.nDeriv = 0       # derivative point-evaluations

    # --- boundaries --------------------------------------------------------
    # A Neumann boundary in MaNTA fixes `q`, the gradient -- *not* the flux.
    # That matters here more than anywhere: the physical condition on the axis
    # is that sigma_hat = x chi(q) q vanishes, which it does for *any* q, so the
    # original problem has no condition there at all. Asking for a zero Neumann
    # value does not express that; it imposes q(0) = 0, which is an extra
    # constraint and a false one -- the true gradient on the axis is -g. Getting
    # this wrong costs a factor of 1e12 in accuracy and looks like a
    # discretisation defect. See README.md.
    def LowerBoundary(self, index, t):
        return -CriticalGradient()

    def UpperBoundary(self, index, t):
        return 0.0

    # --- physics -----------------------------------------------------------
    def SigmaFn(self, index, state, x, t):
        self.nFlux += 1
        q = state.q[0]
        return x * chi(q) * q

    def Sources(self, index, state, x, t):
        return 1.0

    # --- derivatives -------------------------------------------------------
    # The one nonzero block. An absent derivative hook means that block is
    # identically zero, which is already what the zeroed out-parameter gives.
    #
    # d(sigma_hat)/dq = x d/dq[chi(q) q] = x [chi + (dchi/dq) q] -- the
    # *tangent* diffusivity, which is exactly the quantity Jardin's Eq. (10a)
    # substitutes for chi to keep his difference equations tridiagonal. Here it
    # is supplied analytically instead, so unlike Jardin -- who differences it
    # numerically -- and unlike TRINITY -- which pays (1 + n_p) extra flux
    # evaluations per point for it -- MaNTA's Jacobian costs no flux calls.
    def dSigmaFn_dq(self, index, state, x, t):
        self.nDeriv += 1
        q = state.q[0]
        return np.array([x * (chi(q) + q * dchi_dq(q))])

    # --- initial condition -------------------------------------------------
    # Jardin's: the steady state of the same problem with chi held at chi0.
    def InitialValue(self, index, x):
        return 1.0 - x

    def InitialDerivative(self, index, x):
        return -1.0


def registerTransportSystems():
    manta.registerPhysicsCase("JardinCriticalGradient", JardinCriticalGradient)
