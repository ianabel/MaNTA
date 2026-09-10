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

# explain cache misses
import yancc
from yancc.solve import solve_dke
from yancc.species import LocalMaxwellian
import manta as MaNTA

from yancc_wrapper2 import yancc_data

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
nodes = 1 - 1.0 / np.logspace(1, npoints - 1, base=base, num=npoints - 1)
nodes = np.concatenate(([0], nodes, [1]))
# # %%
solver_config = {
    "OutputFilename": "stellarator_w7x",
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
eq = desc.examples.get("W7-X")

# Reduce the number of modes (not sure if this is a good thing to do)
eq.change_resolution(M=4, N=4, L_grid=len(points), M_grid=8, N_grid=8)
eq = eq.solve(x_scale="ess")[0]
eq_init = eq.copy()


def make_test_state(rho, fname="stellarator_w7x"):
    data = Dataset(fname + ".nc", "r")
    x = jnp.array(data.variables["x"][:])
    n = Akima1DInterpolator(
        x, jnp.array(data.groups["Density"].variables["u"][:][-1, :])
    )(rho)
    ui = Akima1DInterpolator(
        x, jnp.array(data.groups["IonEnergy"].variables["u"][:][-1, :])
    )(rho)
    ue = Akima1DInterpolator(
        x, jnp.array(data.groups["ElectronEnergy"].variables["u"][:][-1, :])
    )(rho)
    dndx = Akima1DInterpolator(
        x, jnp.array(data.groups["Density"].variables["q"][:][-1, :])
    )(rho)
    duidx = Akima1DInterpolator(
        x, jnp.array(data.groups["IonEnergy"].variables["q"][:][-1, :])
    )(rho)
    duedx = Akima1DInterpolator(
        x, jnp.array(data.groups["ElectronEnergy"].variables["q"][:][-1, :])
    )(rho)
    Er = Akima1DInterpolator(x, jnp.array(data.variables["Er"][:][-1, :]))(rho)
    data.close()
    Variable = jnp.stack([n, ui, ue]).transpose()
    Derivative = jnp.stack([dndx, duidx, duedx]).transpose()
    state = {
        "Variable": Variable,
        "Derivative": Derivative,
        "Flux": jnp.zeros(Variable.shape),
        "Aux": Er,
        "Scalars": [],
    }
    return state


# test memory constraints for a single gpu
def test_single_gpu():
    # rho = jnp.linspace(0.1, 0.9, 4)
    # states = make_test_state(rho)
    # out = st.ComputePhysics(states, rho, 0.0)
    pass


def test_multi_gpu():
    pass


def run_yancc_at_res(nt, nz, na, nx):
    print(f"running at resolution nt={nt}, nz={nz}, na={na}, nx={nx}")
    st_config = {
        "ParticleSourceCenter": 0.1,
        "ParticleSourceHeight": 0.01,
        "ParticleSourceWidth": 0.4,
        "HeatSourceCenter": 0.1,
        "HeatSourceHeight": 0.1,
        "HeatSourceWidth": 0.2,
        "EdgeTemperature": 0.2,
        "EdgeDensity": 0.3,
        "n0": 0.5,
        "evolveDensity": True,
    }

    config = {
        "Stellarator": st_config,
        "Solver": solver_config,
    }

    yancc_res = {"na": na, "nx": nx}

    ## to allow maximum flexibility to match manta, we use a spline with the same control points as manta \
    # + axis and lcfs
    # initial pressure is all zeros, can change this if desired

    scale = 1.0
    yancc_wrapper = yancc_data.from_eq(
        points, scale=scale, eq=eq_init, nt=nt, nz=nz, **yancc_res
    )

    st = StellaratorTransport(config, yancc_wrapper=yancc_wrapper)
    states = make_test_state(points)
    return st.ComputePhysics(states, points, 0.0)


def test_yancc_res():
    low_res = (13, 23, 43, 5)
    mid_res = (17, 33, 55, 5)
    high_res = (23, 43, 71, 7)
    super_high_res = (27, 49, 99, 7)

    test_res = [low_res, mid_res, high_res, super_high_res]

    fig, ax = plt.subplots(1, 4)
    labels = ("low", "mid", "high", "super")
    colors = ("r", "g", "b", "y")
    for res, l, c in zip(test_res, labels, colors):
        physics = run_yancc_at_res(*res)
        flux = physics[0]
        aux = physics[2]
        ax[0].plot(points, -flux[0], c, label=l)
        ax[1].plot(points, -flux[1], c, label=l)
        ax[2].plot(points, -flux[2], c, label=l)
        ax[3].plot(points, aux[0], c, label=l)

    ax[0].set_ylabel(r"$V'\langle\Gamma \cdot \nabla \rho\rangle / \Gamma_{GB}$")
    ax[1].set_ylabel(r"$ V' \langle q_i \cdot \nabla \rho\rangle / q_{GB}$")
    ax[2].set_ylabel(r"$ V' \langle q_e \cdot \nabla \rho\rangle / q_{GB}$")
    ax[3].set_ylabel(r"$ V' \langle J_\rho \cdot \nabla \rho\rangle / J_{GB}$")

    for a in ax:
        a.set_box_aspect(1)
        a.set_xlabel(r"$\rho$")
        a.legend()
    fig.set_figwidth(14)
    fig.tight_layout()
    fig.savefig("figs/test_res.png", dpi=500)

    plt.show()


test_yancc_res()
