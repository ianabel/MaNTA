from typing import NamedTuple

from functools import partial

import manta as MaNTA
from manta.jax import VectorizedTransportSystem, JAXAdjointProblem, FFIRunner
import jax.numpy as jnp
import numpy as np
import jax
from jaxtyping import Array, ArrayLike, Float, Int
import equinox as eqx
import jax.scipy.special as sci

import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.size": 12})
# Float64, before any array is built. JAX defaults to float32, which caps a
# gradient check at about six digits and makes the finite-difference reference
# the least accurate thing in the comparison rather than the most. The solver
# tolerances below are tightened to match -- there is no point differencing an
# objective to 1e-10 if the steady state it comes from is only converged to 1e-2.
# jax.config.update("jax_enable_x64", True)


class NonlinearDiffusionParams(NamedTuple):
    SourceCentre: float
    D: float
    T_s: float
    a: float
    SourceWidth: float

    @classmethod
    def make(cls, SourceCentre, D, a) -> "NonlinearDiffusionParams":
        return cls(
            SourceCentre=SourceCentre,
            D=D,
            T_s=50.0,
            a=a,
            SourceWidth=0.02,
        )


class JAXAuxTest(VectorizedTransportSystem):
    def __init__(self, params, config):
        super().__init__(MaNTA.numbered_spec(1, lower=MaNTA.Neumann))

        self.params = params
        self.points = MaNTA.getNodes(
            config["Lower_boundary"],
            config["Upper_boundary"],
            config["Grid_size"],
            config["Polynomial_degree"],
        )
        print(params)

        self.adjointProblem = JAXAdjointProblem(self, self.g)

        self.runner = FFIRunner(self, self.points, 1, self.adjointProblem.np)
        self.runner.configure(config)

    def run(self, tFinal=None):
        if tFinal is not None:
            self.runner.Run(tFinal)
        else:
            self.runner.Run_ss()

    def getAdjointGradients(self):
        G, G_p = self.runner.Get_adjoint_gradients()
        return G, G_p

    def g(self, state, x, params):
        u = state.Variable[0]
        return 0.5 * u * u

    def sigma(self, index, state, x, t, params):
        u = state.Variable[index]
        q = state.Derivative[index]
        return params.D * (u**params.a) * q

    def source(self, index, state, x, t, params):
        y = x - params.SourceCentre
        return params.T_s * jnp.exp(-y * y / params.SourceWidth)

    def LowerBoundary(self, index, t):
        return 0.0

    def UpperBoundary(self, index, t):
        return 0.3

    def InitialValue(self, index, x):
        return 0.3

    def InitialAuxValue(self, index, x):
        u0 = self.InitialValue(index, x)
        return self.params.D * u0 * u0

    def createAdjointProblem(self):
        return self.adjointProblem


solver_config = {
    "OutputFilename": "out",
    "Polynomial_degree": 4,
    "Grid_size": 20,
    "tau": 1.0,
    "Lower_boundary": 0.0,
    "Upper_boundary": 1.0,
    # Tight, so the finite-difference reference is limited by its step size
    # rather than by how well the steady state was resolved. MinStepSize has to
    # come down with them: the 1e-7 default is one IDA hits while still at
    # t = 0 at these tolerances, and it reports that as a repeated error-test
    # failure rather than as anything about the step floor. 1e-8 / 1e-10 is a
    # step too far on this 4-cell grid -- IDASolve gives up with IDA_ERR_FAIL.
    "Relative_tolerance": 1e-3,
    "Absolute_tolerance": [1e-4],
    "MinStepSize": 1e-12,
    # run_ss() arms steady-state termination whatever the config says, but
    # falls back to 1e-3, and that -- not Relative_tolerance -- is then the
    # floor on how well G is determined. Left at the default it swamps any
    # finite difference worth taking.
    "SteadyStateSolver": "TimeMarch",
    "SteadyStateTolerance": 1e-9,
    "delta_t": 1.0,
    "restart": False,
    "solveAdjoint": True,
}


def runMaNTA(params):
    transportSystem = JAXAuxTest(params, solver_config)

    transportSystem.run()
    G, G_p = transportSystem.getAdjointGradients()
    uout = transportSystem.runner.Get_profile(0)
    return G, G_p


@jax.custom_jvp
def fun(params):
    G, G_p = runMaNTA(params)
    return G[0]


@fun.defjvp
def fun_jvp(primals, tangents):

    (params,) = primals
    (params_dot,) = tangents

    G, G_p = runMaNTA(params)
    params_dot_flatten, _ = jax.flatten_util.ravel_pytree(params_dot)

    return G[0], jnp.dot(G_p[0], params_dot_flatten)


points = MaNTA.getNodes(
    solver_config["Lower_boundary"],
    solver_config["Upper_boundary"],
    solver_config["Grid_size"],
    solver_config["Polynomial_degree"],
)

# d = np.sqrt(c)


# S = d*np.sqrt(np.pi)*( d/np.sqrt(np.pi)*( np.exp(-(x-b)**2/c) - np.exp(-(1-b)**2/c) ) + (1 - b)*sci.erf((1-b)/d)
#                      + (1-x)*sci.erf(b/d) - (x-b)*sci.erf((x-b)/d) )


# uan = (u0**(1+a) + T_s/D*(1+a)*S)**(1/(1+a))
def uan(x, c, a, u1):
    b = 0.02
    d = 50.0

    exponent = 0.0

    y = (x - c) / jnp.sqrt(b)
    G = (b * d / (4 * a)) * (jnp.exp(-((1 - c) ** 2) / b) - jnp.exp(-(y**2))) + (
        d * jnp.sqrt(b * jnp.pi) / (4 * a)
    ) * (
        (c - 1) * sci.erf((c - 1) / jnp.sqrt(b))
        + (1 - x) * sci.erf(c / jnp.sqrt(b))
        - (x - c) * sci.erf(y)
    )
    u2 = u1 ** (1 + exponent) + 2 * (1 + exponent) * G
    return u2 ** (1.0 / (1 + exponent))


x = jnp.linspace(0, 1, 200)

u0 = 0.3


def g(c, D, u0):
    return jax.scipy.integrate.trapezoid(0.5 * uan(x, c, D, u0) ** 2, x)


D = 2.0
C = 0.3

fac = 0.5

fig, ax = plt.subplots(1, 2)

Ds = jnp.linspace(D, 2.0 * D, 10)

dgdD_adj = []
for d in Ds:
    params = NonlinearDiffusionParams.make(
        C,
        d,
        0.0,
    )
    gval = jax.grad(fun)(params)
    dgdD_adj.append(jax.grad(fun)(params).D)


Dan = jnp.linspace(D, 2.0 * D, 100)
dgdD_an = jax.vmap(jax.grad(g, argnums=1), in_axes=(None, 0, None))(C, Dan, u0)
ax[0].plot(Dan, dgdD_an, "r", label="Analytic")
ax[0].plot(Ds, jnp.array(dgdD_adj), "bx", label="Adjoints")
ax[0].set_xlabel(r"D")
ax[0].set_ylabel(r"$\partial G/\partial D$")


cs = jnp.linspace(0.5 * C, 1.5 * C, 10)


dgdc_adj = []
for c in cs:
    params = NonlinearDiffusionParams.make(
        c,
        D,
        0.0,
    )
    dgdc_adj.append(jax.grad(fun)(params).SourceCentre)


can = jnp.linspace(0.5 * C, 1.5 * C, 100)
dgdc_an = jax.vmap(jax.grad(g, argnums=0), in_axes=(0, None, None))(can, D, u0)
ax[1].plot(can, dgdc_an, "r", label="Analytic")
ax[1].plot(cs, jnp.array(dgdc_adj), "bx", label="Adjoints")

ax[1].set_xlabel(r"c")
ax[1].set_ylabel(r"$\partial G/\partial c$")
for a in ax:
    a.legend()
    a.set_box_aspect(1)
fig.tight_layout()
fig.set_figwidth(5.0)
fig.savefig("adjoints.eps", dpi=500)
plt.show()
