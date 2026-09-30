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
    def __init__(self, u_exponent, solver_config):
        super().__init__(manta.numbered_spec(1, lower=manta.Neumann))

        self.params = ncconfig(
            u_exponent=u_exponent,
            SourceHeight=0.1,
            SourceWidth=0.2,
            D=0.01,
            u_upper=0.2,
        )

        self.runner = manta.Runner(self)

        self.runner.configure(solver_config)

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


# lower tau seems to alleviate the problem somewhat
solver_config = {
    "OutputFilename": "out",
    "Polynomial_degree": 4,
    "Grid_size": 5,
    "tau": 1.0,
    "Lower_boundary": 0.0,
    "Upper_boundary": 1.0,
    "Relative_tolerance": 1e-6,
    "Absolute_tolerance": [1e-6],
    "initialTimestep": 1e-3,
    "MinStepSize": 1e-12,
    "SteadyStateSolver": "PseudoTransient",
    "SteadyStateTolerance": 1e-5,
    "delta_t": 1.0,
    "restart": False,
}

# as the exponent on Ti is increased the flux in the last cell gets more squiggly
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
