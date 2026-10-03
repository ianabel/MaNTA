import manta
from manta.jax import VectorizedTransportSystem
import equinox as eqx
from jaxtyping import Float, ArrayLike
import jax.numpy as jnp
import jax
from netCDF4 import Dataset
import matplotlib.pyplot as plt


class ncconfig(eqx.Module):
    u_exponent: Float
    SourceHeight: Float
    SourceWidth: Float
    D: Float
    u_upper: Float

    def __init__(self, u_exponent, SourceHeight, SourceWidth, D, u_upper):
        self.u_exponent = u_exponent
        self.SourceHeight = SourceHeight
        self.SourceWidth = SourceWidth
        self.D = D
        self.u_upper = u_upper


class nc_test(VectorizedTransportSystem):
    # MeshAdaptation moves this one instance from the uniform sample to the
    # graded mesh and through the degree loop, each a new evaluation plan. It
    # keeps nothing that depends on the mesh, so it can follow them in place.
    regrid = manta.Regrid.InPlace

    def __init__(self, u_exponent, solver_config):
        # Zero flux at the axis, sigma(0) = 0, rather than u'(0) = 0. The flux
        # 2 x D u^n u' vanishes at x = 0 through its factor of x, and the steady
        # solution has u'(0) = -SourceHeight / (2 D u(0)^n) != 0, so imposing
        # u'(0) = 0 contradicts the equation and costs a first-order error there.
        super().__init__(manta.numbered_spec(1, lower=manta.Mixed(d=1.0)))

        self.params = ncconfig(
            u_exponent=u_exponent,
            SourceHeight=0.1,
            SourceWidth=0.2,
            D=0.01,
            u_upper=0.2,
        )

        self.runner = manta.Runner(self)

        self.runner.configure(solver_config)

    # Flux = 2 * x * D * (pressure) ^ u_exponent * ( d pressure / d x )
    def sigma(self, index, state, x, t, params: ncconfig):
        pi = state.Variable[index]
        pi_prime = state.Derivative[index]
        return 2 * x * params.D * pi ** (params.u_exponent) * pi_prime

    def source(self, index, state, x, t, params: ncconfig):
        return params.SourceHeight * jnp.exp(-x * x / params.SourceWidth)

    def LowerBoundary(self, index, t):
        return 0.0

    def InitialValue(self, index, x):
        return self.params.u_upper

    def UpperBoundary(self, index, t):
        return self.params.u_upper


# Two things fix the last cell, and they fix different halves of it.
#
# tauScaling = "Diffusive" scales the HDG stabilisation by kappa/h. A constant
# tau much larger than kappa/h at the wall -- which is what tau = 1 is here once
# u^n is small -- pushes tau * (u_h - u_upper) into sigma_h, and that is the
# oscillation in the last cell. "ContinuationStep" re-evaluates tau once per
# steady continuation step, which costs ~8% over a constant tau where updating it
# in every residual costs ~75%, for the same answer.
#
# MeshAdaptation then resolves the layer: it decides from one uniform solve
# whether an end needs grading (here the wall, from n ~ 1 up), regrades the same
# five cells towards it, and raises the degree only if the estimated relative L2
# error is still above DegreeTolerance. The graded layer is the sample's wall
# cell, [0.8, 1]: four cells shrinking by 0.3 towards the wall, the narrowest
# 5.4e-3 wide, and one cell over [0, 0.8]. So out.nc's x is that mesh, not a
# uniform one. DegreeTolerance is an estimated relative L2 error, not the L_inf
# target it stands in for; at 1e-2 it stays at k = 4. Measured at
# n = 2.5, 0.24% relative L_inf in u and 1.3% pointwise at worst, against 3.3% and
# 17% for five uniform cells at constant tau. The price is a second solve -- the
# uniform one that decides, then the graded one, warm-started from it -- which
# comes to 1.4x the physics evaluations of a single uniform solve at constant tau
# (34 Newton iterations in all, against 28).
#
# Once the mesh is graded the two remedies overlap, and the local tau is then
# the less accurate one: on these five graded cells a constant tau gives 0.09%
# L_inf in u against Diffusive's 0.24% (python-examples/paper-figures/
# adaptivity.py on main, n = 2.5). What Diffusive buys is the wall flux on a mesh
# that stays uniform -- 0.4% against 50% at constant tau on five uniform cells --
# and the sensor does not grade the smallest exponents here. Drop it for the
# best u on the graded runs.
solver_config = {
    "OutputFilename": "out",
    "PolynomialDegree": 4,
    "GridSize": 5,
    "tau": 1.0,
    "tauScaling": "Diffusive",
    "tauUpdate": "ContinuationStep",
    "MeshAdaptation": True,
    "DegreeTolerance": 1e-2,
    "LowerBoundary": 0.0,
    "UpperBoundary": 1.0,
    "Relative_tolerance": 1e-6,
    "Absolute_tolerance": [1e-6],
    "initialTimestep": 1e-3,
    "MinStepSize": 1e-12,
    "SteadyStateSolver": "PseudoTransient",
    "SteadyStateTolerance": 1e-5,
    "delta_t": 1.0,
    "restart": False,
}

# As the exponent increases the wall layer narrows -- about 5e-4 wide at n = 2.5 --
# which on a uniform mesh is what made the flux in the last cell oscillate.
u_exponent = jnp.linspace(0.25, 5.0 / 2.0, 10)
fig, ax = plt.subplots(1, 2)

plt.rcParams.update({"font.family": "serif", "font.size": 10})
for exponent in u_exponent:
    nc = nc_test(exponent, solver_config)
    nc.runner.run_ss()
    data = Dataset("out.nc")

    x = jnp.array(data.variables["x"][:])
    group = data.groups["Var0"]
    ui = jnp.array(group.variables["u"][:])
    sigma = jnp.array(group.variables["sigma"][:])
    data.close()

    ax[0].plot(x, ui[-1, :])
    ax[1].plot(x, sigma[-1, :])


ax[0].set_ylabel("u")
ax[1].set_ylabel(r"$\sigma$")

for a in ax:
    a.set_box_aspect(1)
    a.set_xlabel("x")

fig.tight_layout()
fig.savefig("nc_test.png")
plt.show()
