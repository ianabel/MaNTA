import os

os.environ["JAX_ENABLE_PINNED_HOST_TRANSFER"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


import jax
import jax.numpy as jnp
import equinox as eqx
from manta.jax import State
import desc
from desc import set_device
from typing import NamedTuple
from jax.sharding import Mesh, PartitionSpec, NamedSharding
import matplotlib.pyplot as plt
from stellarator_multichannel import StellaratorTransport
from netCDF4 import Dataset
from interpax import Akima1DInterpolator
from desc.equilibrium import Equilibrium
from desc.plotting import plot_boozer_surface
from scipy.optimize import curve_fit

from functools import partial
from desc.profiles import SplineProfile

# explain cache misses
import yancc
from yancc.solve import solve_dke
from yancc.species import LocalMaxwellian
import manta as MaNTA

from yancc_wrapper import yancc_data

import numpy as np

plt.rcParams.update({"font.family": "serif", "font.size": 14})
rho_upper = 1.0
rtol = 1e-2
atol = 1e-5
# nodes = [0.0,0.5, 0.75, 0.9, 1.0]
npoints = 4
degree = 4
base = 1.6
tau = 10.0
nodes = np.linspace(0.2, 1.0, npoints + 1)
# nodes = np.concatenate(([0], nodes, [1]))
# # %%
solver_config = {
    "OutputFilename": "a",
    "Polynomial_degree": degree,
    "Grid_points": nodes,
    "Grid_size": len(nodes) - 1,
    "tau": tau,
    "Lower_boundary": 0.0,
    "Upper_boundary": rho_upper,
    "Relative_tolerance": rtol,
    "Absolute_tolerance": [atol],
    "delta_t": 1.0,
    "initialTimestep": 1e-2,
    "MinStepSize": 1e-9,
    "SteadyStateTolerance": 1e-3,
    "AggressiveTimesteps": False,
    "WriteDatFile": True,
    "zeroFlux": True,
    "solveAdjoint": False,
}


points = MaNTA.getNodes(
    nodes,
    solver_config["Polynomial_degree"],
)
yancc_rho = points
pressure_rho = jnp.concatenate([jnp.zeros(1), yancc_rho, jnp.ones(1)])
desc_pressure = SplineProfile(jnp.zeros_like(pressure_rho), pressure_rho)
eq = desc.io.load("eq_qa.h5")
fig, ax = plot_boozer_surface(eq, fieldlines=8)
plt.show()
# eq = desc.examples.get("precise_QA")
#
# eq_name = "eq_amb3"
# eq = desc.io.load(eq_name + "_all_equilibria.h5")[-1]
#
# eq = desc.compat.rescale(eq, L=("R0", 10), B=("B0", 5.0))
# eq.change_resolution(M=4, N=4, L_grid=len(points), M_grid=8, N_grid=8)
#
# # eq = Equilibrium(M=4, N=4, Psi=0.1, surface=surf, pressure=desc_pressure)
# eq = eq.solve(x_scale="ess")[0]
eq_init = eq.copy()


def make_test_state(rho, st):
    Variable = jnp.stack(
        [st.InitialValue(0, rho), st.InitialValue(1, rho), st.InitialValue(2, rho)]
    ).transpose()
    Derivative = jnp.stack(
        [
            jax.vmap(partial(st.InitialDerivative, 0))(rho),
            jax.vmap(partial(st.InitialDerivative, 1))(rho),
            jax.vmap(partial(st.InitialDerivative, 2))(rho),
        ]
    ).transpose()

    def Er(rho):
        return -6.0 * rho

    state = {
        "Variable": Variable,
        "Derivative": Derivative,
        "Flux": jnp.zeros(Variable.shape),
        "Aux": jnp.atleast_2d(Er(rho)).transpose(),
        "Scalars": [],
    }
    return state


def compute_physics(nt, nz, na, nx):
    print(f"running at resolution nt={nt}, nz={nz}, na={na}, nx={nx}")
    #
    # st_config = {
    #     "ParticleSourceCenter": 0.0,
    #     "ParticleSourceHeight": 1.25e-2,
    #     "ParticleSourceWidth": 0.5,
    #     "NBICenter": 0.0,
    #     "NBIPower": 0.6,
    #     "NBIWidth": 0.3,
    #     "ECHCenter": 0.0,
    #     "ECHPower": 0.1,
    #     "ECHWidth": 0.26,
    #     "EdgeTemperature": 0.2,
    #     "EdgeDensity": 0.2,
    #     "n0": 1.0,
    #     "T0": 1.0,
    #     "evolveDensity": True,
    #     "useBatching": False,
    # }
    #

    st_config = {
        "ParticleSourceCenter": 0.0,
        "ParticleSourceHeight": 0.0,
        "ParticleSourceWidth": 0.6,
        "NBICenter": 0.0,
        "NBIPower": 0.0,
        "NBIWidth": 0.4,
        "ECHCenter": 0.0,
        "ECHPower": 0.0,
        "ECHWidth": 0.4,
        "EdgeTemperature": 0.2,
        "EdgeDensity": 0.2,
        "n0": 1.0,
        "T0": 1.5,
        "FusionFactor": 0.01,
        "evolveDensity": True,
        "useBatching": False,
    }

    config = {
        "Stellarator": st_config,
        "Solver": solver_config,
    }

    yancc_res = {"na": na, "nx": nx}

    ## to allow maximum flexibility to match manta, we use a spline with the same control points as manta \
    # + axis and lcfs
    # initial pressure is all zeros, can change this if desired

    yancc_wrapper = yancc_data.from_eq(points, eq=eq_init, nt=nt, nz=nz, **yancc_res)

    st = StellaratorTransport(config, yancc_wrapper=yancc_wrapper)
    states = make_test_state(points, st)

    out = st.ComputePhysics(states, points, 0.0)
    return out, yancc_wrapper


def run():
    # res = (17, 33, 55, 7)

    high_res = (23, 43, 71, 7)
    out, yancc_wrapper = compute_physics(*high_res)

    flux = out[0]
    source = out[1]

    vp = yancc_wrapper.Vp

    Sout = []
    for i in range(0, 3):
        divFlux = jnp.gradient(-flux[i], points)
        Sout.append((divFlux - source[i]) / vp)

    fig, ax = plt.subplots(1, 3)

    ax[0].plot(points, flux[0])
    ax[1].plot(points, flux[1])
    ax[2].plot(points, flux[2])

    fig.suptitle("fluxes")
    fig, ax = plt.subplots(1, 3)

    ax[0].plot(points, source[0])
    ax[1].plot(points, source[1])
    ax[2].plot(points, source[2])
    fig.suptitle("sources")

    fig, ax = plt.subplots(1, 3)

    def gaussian(x, s0, s2):
        return s0 * np.exp(-(x**2) / (2 * s2))

    for i in range(0, 3):
        ind = Sout[i] > 0

        p, _ = curve_fit(gaussian, points[ind], Sout[i][ind], bounds=(0, np.inf))
        s0, s2 = p

        print(f"Fit parameters for i={i}: S₀={s0}, σ={np.sqrt(s2)}")
        ax[i].plot(points, Sout[i])
        ax[i].plot(
            points,
            gaussian(points, s0, s2),
            label=f"S₀={s0:0.2e}, σ={np.sqrt(s2):0.2f}",
        )
        ax[i].legend()

    fig.set_figwidth(10)
    fig.savefig("required_source.png", dpi=300)


run()
plt.show()
