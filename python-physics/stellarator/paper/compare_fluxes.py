"""
Script for comparing itg fluxes from beginning and end of an optimization
"""

import sys

sys.path.append("..")

from interpax import Akima1DInterpolator
import jax.numpy as jnp
from netCDF4 import Dataset
import desc.io
import matplotlib.pyplot as plt
from config import (
    StellaratorParams,
    StellaratorConfig,
    field_norm_from_eq,
    normalization_args,
)
from yancc_wrapper import yancc_data
from stellarator_multichannel import vmap_axes_wfield

import jax
import equinox as eqx
from jax.sharding import PartitionSpec, NamedSharding
from itg_wrapper import itg_at, make_itg_builder
from dke_sol import compute_dke_sol
from manta.jax import State, vmap_batched

P = PartitionSpec
devices = jax.devices()
mesh = jax.make_mesh(
    (jax.device_count(),), ("axis",), axis_types=(jax.sharding.AxisType.Auto,)
)
data_sharding = NamedSharding(
    mesh,
    P(
        "axis",
    ),
)
eq_name = "eq_itg"
eqs = desc.io.load(eq_name + "_all_equilibria.h5")
eq_initial, eq_final = eqs[0], eqs[-1]


def make_state_from_output(rho, fname="stellarator_opt" + eq_name):
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
        "VariableDot": [],
        "Flux": jnp.zeros(Variable.shape),
        "Aux": Er,
        "Scalars": [],
    }

    return eqx.partition(State.from_manta(state), lambda x: x.size > 0)[0]


fac = 1.0
st_config = {
    "ParticleSourceCenter": 0.0,
    "ParticleSourceHeight": fac * 0.015,
    "ParticleSourceWidth": 0.6,
    "NBICenter": 0.0,
    "NBIPower": fac * 8.0,
    "NBIWidth": 0.15,
    "ECHCenter": 0.0,
    "ECHPower": fac * 0.0,
    "ECHWidth": 0.1,
    "EdgeTemperature": 0.5,
    "EdgeDensity": 0.4,
    "ADCoefficient": 0.0,
    "n0": 0.5,
    "T0": 1.0,
    "FusionFactor": 0.01,
    "evolveDensity": True,
    "useITGProxy": True,  # NN ITG heat flux (itg_wrapper.py); needs torch + ITG_MODEL_DIR
    # must match the run's: MaNTA transported ITGScale times the prediction,
    # and that is what is plotted
    "ITGScale": 1.0,
    "useBatching": True,
    "nchunks": 3,
}

config = StellaratorConfig(rhoUpper=1.0, **st_config)
# The run's normalisation. The ITG flux takes its gyro-Bohm value at the
# normalisation's a and B0, so this sets the flux itself, not just its units.
# paper/desc-optimize-stored-energy.py normalises to its initial equilibrium;
# a run made before that (2ce3f82) used the defaults, i.e. {}.
normalization = field_norm_from_eq(eq_initial)
params = StellaratorParams(config, **normalization_args(normalization))

yancc_ntheta = 17
yancc_nzeta = 25

yancc_res = {"na": 45, "nx": 7}


rho = jnp.linspace(0.01, 0.9, 64)

_eqs = [eq_initial, eq_final]

plt.rcParams.update({"font.family": "serif", "font.size": 10})

names = ["initial", "final"]
colors = ["r", "b"]
fig, ax = plt.subplots(1, 2, sharey=True)

state = make_state_from_output(rho)
# Built once, on the initial equilibrium, as the driver builds it: the builder
# fixes the flux tubes' toroidal extent there, and MaNTA evaluated the final
# equilibrium with that extent, not one re-solved on it.
itg_builder = make_itg_builder(st_config, eq_initial, solve_length_at_compute=False)
for eq, name, color in zip(_eqs, names, colors):
    yancc_wrapper = yancc_data.from_eq(
        rho, eq=eq, nt=yancc_ntheta, nz=yancc_nzeta, **yancc_res
    )
    itg_wrapper = itg_at(itg_builder, eq.params_dict)
    fields, vp, vpp = yancc_wrapper.get_fields()
    Qi_itg = jax.vmap(itg_wrapper, in_axes=vmap_axes_wfield)(
        state,
        rho,
        0.0,
        fields,
        vp,
        vpp,
        yancc_wrapper.pitchgrid,
        yancc_wrapper.speedgrid,
        params,
        True,
    )[0][1]

    Qi_nc = vmap_batched(
        compute_dke_sol,
        vmap_axes_wfield,
        chunk_size=16,
        nchunks=4,
        sharding=data_sharding,
    )(
        state,
        rho,
        0.0,
        fields,
        vp,
        vpp,
        yancc_wrapper.pitchgrid,
        yancc_wrapper.speedgrid,
        params,
        True,
    )[0][1]

    # Qi is MaNTA's normalised flux, -V' <q . grad rho> / HeatNorm; times
    # HeatNorm that is V' <q . grad rho> in W/m^3, and times a it is
    ax[0].semilogy(
        rho,
        -jnp.array(Qi_nc)
        * params.constants.HeatEquationNormalization()
        * params.constants.a
        * 1e-6,
        color,
        label=name,
    )

    ax[1].semilogy(
        rho,
        -jnp.array(Qi_itg)
        * params.constants.HeatEquationNormalization()
        * params.constants.a
        * 1e-6,
        color,
        label=name,
    )


for a in ax.flatten():
    a.set_xlabel(r"$\rho$")
    a.legend()
    a.set_box_aspect(1)
ax[0].set_ylabel(r"$V'Q$ (MW/$m^2$)")

ax[0].set_title("Neoclassical")
ax[1].set_title("ITG")
fig.tight_layout()
fig.savefig("figs/compare_fluxes.pdf", dpi=500)
plt.show()
