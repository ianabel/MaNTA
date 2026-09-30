import manta
from manta.jax import VectorizedTransportSystem
import equinox as eqx
from jaxtyping import Float, ArrayLike
import jax.numpy as jnp
import jax
from netCDF4 import Dataset
import matplotlib.pyplot as plt


class ncconfig(eqx.Module):
    Ti_exponent: Float
    epsilon: Float
    SourceHeight: Float
    SourceWidth: Float
    D: Float
    n0: Float
    Ti_upper: Float

    def __init__(
        self, Ti_exponent, epsilon, SourceHeight, SourceWidth, D, n0, Ti_upper
    ):
        self.Ti_exponent = Ti_exponent
        self.epsilon = epsilon
        self.SourceHeight = SourceHeight
        self.SourceWidth = SourceWidth
        self.D = D
        self.n0 = n0
        self.Ti_upper = Ti_upper


class nc_test(VectorizedTransportSystem):
    def __init__(self, Ti_exponent, solver_config):
        super().__init__(manta.numbered_spec(1, lower=manta.Neumann))

        self.params = ncconfig(
            Ti_exponent=Ti_exponent,
            epsilon=0.1,
            SourceHeight=0.1,
            SourceWidth=0.2,
            D=0.1,
            n0=0.5,
            Ti_upper=0.2,
        )

        self.runner = manta.Runner(self)

        self.runner.configure(solver_config)

    def density(self, x):
        return (self.params.n0 - 0.1) * (1 - x**4) + 0.1

    @staticmethod
    def uvp_to_u(uvp, x):
        return uvp / (2.0 * x)

    @staticmethod
    def uvpp_to_up(uvpp, uvp, x):
        return 1 / (2.0 * x) * (uvpp - uvp / x)

    def sigma(self, index, state, x, t, params: ncconfig):
        n, nprime = jax.value_and_grad(self.density)(x)
        pi = 2.0 / 3.0 * self.uvp_to_u(state.Variable[index], x)
        pi_prime = (
            2.0
            / 3.0
            * self.uvpp_to_up(state.Derivative[index], state.Variable[index], x)
        )
        Ti = pi / n

        # if the exponent one Ti is 1.0 or less, no squiggle
        return (
            2
            * x
            * params.D
            * params.epsilon ** (3.0 / 2.0)
            * n
            * Ti ** (params.Ti_exponent)
            * pi_prime
            / pi
        )

    def source(self, index, state, x, t, params: ncconfig):
        return params.SourceHeight * jnp.exp(-x * x / params.SourceWidth)

    def LowerBoundary(self, index, t):
        return 0.0

    def InitialValue(self, index, x):
        return self.params.Ti_upper

    def UpperBoundary(self, index, t):
        return 2 * 3.0 / 2.0 * self.density(1.0) * self.params.Ti_upper


solver_config = {
    "OutputFilename": "out",
    "Polynomial_degree": 4,
    "Grid_size": 5,
    "tau": 1.0,
    "Lower_boundary": 0.0,
    "Upper_boundary": 1.0,
    "Relative_tolerance": 1e-4,
    "Absolute_tolerance": [1e-6],
    "initialTimestep": 1e-3,
    "MinStepSize": 1e-12,
    "SteadyStateSolver": "PseudoTransient",
    "SteadyStateTolerance": 1e-5,
    "delta_t": 1.0,
    "restart": False,
    "zeroFlux": True,
}

# as the exponent on Ti is increased the flux in the last cell gets more squiggly
Ti_exponent = jnp.linspace(0.25, 5.0 / 2.0, 10)
fig, ax = plt.subplots(1, 2)

plt.rcParams.update({"font.family": "serif", "font.size": 10})
for exponent in Ti_exponent:
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
