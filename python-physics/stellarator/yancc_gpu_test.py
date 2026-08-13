import jax
import jax.numpy as jnp
import equinox as eqx
from manta.jax import State
import desc
from desc import set_device
from typing import NamedTuple
from jax.sharding import Mesh, PartitionSpec, NamedSharding
import matplotlib.pyplot as plt

# explain cache misses
import yancc
from yancc.solve import solve_dke
from yancc.species import LocalMaxwellian

P = PartitionSpec
devices = jax.devices()
print(devices)
mesh = Mesh(devices, ("axis",), axis_types=(jax.sharding.AxisType.Auto,))
data_sharding = NamedSharding(
    mesh,
    P(
        "axis",
    ),
)
static_sharding = NamedSharding(mesh, P())


from yancc_wrapper2 import yancc_data

import numpy as np

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

vmap_axes = (State.vmap_axes(), 0)
vmap_axes_wfield = (None, State.vmap_axes(), 0, None, 0, 0, 0, None, None)
vmap_axes_sources = (None, State.vmap_axes(), 0, None, 0, None)
set_device("gpu")

st_config = {
    "SourceCenter": 0.2,
    "SourceHeight": 10.0,
    "SourceWidth": 0.4,
    "EdgeTemperature": 0.2,
    "EdgeDensity": 0.2,
    "n0": 1.0,
    "use_chunking": False,
}


class StellaratorParams(NamedTuple):
    SourceCenter: float
    SourceHeight: float
    SourceWidth: float
    EdgeTemperature: float
    EdgeDensity: float
    n0: float

    @classmethod
    def from_config(cls, config):
        return cls(
            SourceCenter=config["SourceCenter"],
            SourceHeight=config["SourceHeight"],
            SourceWidth=config["SourceWidth"],
            EdgeTemperature=config["EdgeTemperature"],
            EdgeDensity=config["EdgeDensity"],
            n0=config["n0"],
        )


params = StellaratorParams.from_config(st_config)
rho_upper = 1.0
rtol = 1e-2
atol = 1e-3
# nodes = [0.0,0.5, 0.75, 0.9, 1.0]
npoints = 20

points = jnp.linspace(0.1, 0.9, 20)

yancc_rho = jnp.array(points)
yancc_ntheta = 13
yancc_nzeta = 27

yancc_res = {"na": 43, "nx": 5}
## to allow maximum flexibility to match manta, we use a spline with the same control points as manta \
# + axis and lcfs
# initial pressure is all zeros, can change this if desired

eq = desc.examples.get("W7-X")

eq.change_resolution(M=4, N=4, L_grid=len(points))
eq = eq.solve(x_scale="ess")[0]
eq_init = eq.copy()
yancc_wrapper = yancc_data.from_eq(
    points, eq=eq_init, nt=yancc_ntheta, nz=yancc_nzeta, **yancc_res
)


def initial_profile(x, edge_value, peak_value):
    return (peak_value - edge_value) * (1 - x**4) + edge_value


def Density(x):
    return initial_profile(x, params.EdgeDensity, params.n0)


def InitialValue(x):
    return 1.5 * params.EdgeTemperature * Density(x) * yancc_wrapper.Vp


pi = InitialValue(points)
grad_pi = jax.grad(InitialValue, argnums=1)(points)

s = jax.device_put(State(pi, grad_pi, jnp.zeros(pi.shape), None, None), data_sharding)


def Vp_u_to_u(index, s, x, vp, vpp):
    return jax.lax.cond(
        jax.lax.eq(x, 0.0),
        lambda state: state.Derivative[index] / vpp,
        lambda state: state.Variable[index] / vp,
        s,
    )


def Vp_up_to_up(index, s, x, vp, vpp):
    return jax.lax.cond(
        jax.lax.eq(x, 0.0),
        lambda state: 0.0,
        lambda state: (
            (state.Derivative[index] * vp - vpp * state.Variable[index]) / vp**2
        ),
        s,
    )


(field_shard, vp_shard, vpp_shard) = eqx.filter_shard(
    yancc_wrapper.get_fields(), data_sharding
)


def SigmaFn_v(index, states: State, positions, t):

    sigma_vmap = eqx.filter_vmap(sigma, in_axes=vmap_axes_wfield)
    flux, _ = sigma_vmap(
        index,
        states,
        positions,
        t,
        field_shard,
        vp_shard,
        vpp_shard,
        params,
    )
    return flux


def dSigma(index, states: State, positions, t):
    fgrad = jax.grad(sigma, argnums=1, has_aux=True)
    fgrad_vmap = eqx.filter_vmap(fgrad, in_axes=vmap_axes_wfield)
    dflux = fgrad_vmap(
        index,
        states,
        positions,
        t,
        field_shard,
        vp_shard,
        vpp_shard,
        params,
    )[0]
    return dflux


"""
Sigma and source, and auxilliary functions

Parameters
----------
index : int
    Variable index
state : dict
    Dictionary containing "Variable", "Derivative, "Flux", "Aux", and "Scalar" arrays
x : float
    Spatial location
t : float
    Time
params : NamedTuple
    Transport system parameters, passed for JAX PyTree compatibility
Returns
-------
float
    Computed sigma or source term
"""


def sigma(index, state: State, x, t, field, vp, vpp, params, f1_prev=None):
    n, nprime = jax.value_and_grad(Density)(x)

    p_i = 2.0 / 3.0 * Vp_u_to_u(index, state, x, vp, vpp)
    p_i_prime = 2.0 / 3.0 * Vp_up_to_up(index, state, x, vp, vpp)

    dndrho = nprime
    Erho = jnp.array(0.0)
    Ti = jax.lax.cond(
        jax.lax.eq(x, 0.0),
        lambda x: params.EdgeTemperature,
        lambda x: x[0] / x[1],
        (p_i, n),
    )

    dTidrho = (p_i_prime - Ti * dndrho) / n

    species = [
        LocalMaxwellian(
            # can just give mass and charge in units of proton mass and elementary charge
            yancc.species.Species(1, 1),
            temperature=Ti * yancc_wrapper.Tnorm,
            density=n * yancc_wrapper.nNorm,
            dTdrho=dTidrho * yancc_wrapper.Tnorm,
            dndrho=dndrho * yancc_wrapper.nNorm,
        ),
    ]
    sol, info = jax.jit(solve_dke, static_argnames=["verbose"])(
        field,
        yancc_wrapper.pitchgrid,
        yancc_wrapper.speedgrid,
        species,
        Erho,
        f1=f1_prev,
    )
    # nmv
    # M is the size of krylov space
    fout = sol.get("<heat_flux>")[0] * vp / (yancc_wrapper.FluxNorm)
    return -jnp.nan_to_num(fout, nan=0.0, posinf=0.0, neginf=0.0), sol.f1


flux_out = SigmaFn_v(0, s, points, 0.0)
dflux_out = dSigma(0, s, points, 0.0)

fig, ax = plt.subplots()
ax.plot(points, flux_out)

fig, ax = plt.subplots()
ax.plot(points, dflux_out.Variable)
ax.plot(points, dflux_out.Derivative)
plt.show()
