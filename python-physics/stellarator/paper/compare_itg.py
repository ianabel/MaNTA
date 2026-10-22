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
import os

os.environ["ITG_NN_DIR"] = "/home/eatocco/projects/itg-proxy/demo/neural_networks/"

from itg_wrapper import itg_at, make_itg_builder, make_itg_flux
from manta.jax import State
from stellarator_state import StellaratorState

eq_name = "eq_itg_flux"
eqs = desc.io.load(eq_name + "_all_equilibria.h5")
eq_initial = desc.examples.get("precise_QH")
# eq_final = eqs[0]
eq_initial, eq_final = eqs[0], eqs[-1]


st_config = {
    "ParticleSourceCenter": 0.0,
    "ParticleSourceHeight": 0.02,
    "ParticleSourceWidth": 0.6,
    "NBICenter": 0.0,
    "NBIPower": 16.0,  # doubled from the stored-energy script's 8.0
    "NBIWidth": 0.15,
    "ECHCenter": 0.0,
    "ECHPower": 0.0,
    "ECHWidth": 0.1,
    "EdgeTemperature": 0.5,
    "EdgeDensity": 0.4,
    "ADCoefficient": 0.0,
    "n0": 1.0,
    "T0": 1.0,
    "alpha": 4.0,
    "FusionFactor": 0.01,
    "evolveDensity": False,
    "useITGProxy": True,  # NN ITG heat flux (itg_wrapper.py); needs ITG_NN_DIR
    "ITGScale": 1.0,
    "useBatching": True,
    "nchunks": 3,
}


def make_state_from_output(rho, fname="stellarator_opt" + eq_name):
    data = Dataset(fname + ".nc", "r")
    x = jnp.array(data.variables["x"][:])
    ui = Akima1DInterpolator(
        x, jnp.array(data.groups["IonEnergy"].variables["u"][:][-1, :])
    )(rho)

    duidx = Akima1DInterpolator(
        x, jnp.array(data.groups["IonEnergy"].variables["q"][:][-1, :])
    )(rho)

    if st_config["evolveDensity"]:
        n = Akima1DInterpolator(
            x, jnp.array(data.groups["Density"].variables["u"][:][-1, :])
        )(rho)
        ue = Akima1DInterpolator(
            x, jnp.array(data.groups["ElectronEnergy"].variables["u"][:][-1, :])
        )(rho)
        dndx = Akima1DInterpolator(
            x, jnp.array(data.groups["Density"].variables["q"][:][-1, :])
        )(rho)
        duedx = Akima1DInterpolator(
            x, jnp.array(data.groups["ElectronEnergy"].variables["q"][:][-1, :])
        )(rho)
        Er = Akima1DInterpolator(x, jnp.array(data.variables["Er"][:][-1, :]))(rho)
        Variable = jnp.stack([n, ui, ue]).transpose()
        Derivative = jnp.stack([dndx, duidx, duedx]).transpose()
    else:
        Variable = jnp.asarray(ui)[:, None]
        Derivative = jnp.asarray(duidx)[:, None]

        Er = jnp.zeros(Variable.shape)

    data.close()

    state = {
        "Variable": Variable,
        "Derivative": Derivative,
        "VariableDot": jnp.zeros(Variable.shape),
        "Flux": jnp.zeros(Variable.shape),
        "Aux": Er,
        "Scalars": [],
    }
    return State.from_manta(state)


config = StellaratorConfig(rhoUpper=1.0, **st_config)
# The run's normalisation. The ITG flux takes its gyro-Bohm value at the
# normalisation's a and B0, so this sets the flux itself, not just its units.
# paper/desc-optimize-stored-energy.py normalises to its initial equilibrium;
# a run made before that (2ce3f82) used the defaults, i.e. {}.
normalization = field_norm_from_eq(eq_final)
params = StellaratorParams(config, **normalization_args(normalization))

yancc_ntheta = 17
yancc_nzeta = 25

yancc_res = {"na": 45, "nx": 7}


rho = jnp.linspace(0.05, 0.95, 37)

_eqs = [eq_initial, eq_final]

plt.rcParams.update({"font.family": "serif", "font.size": 10})

names = ["initial", "final"]
colors = ["r", "b"]
fig, ax = plt.subplots()
fig2, ax2 = plt.subplots()

state_ = make_state_from_output(rho)
# Built once, on the initial equilibrium, as the driver builds it: the builder
# fixes the flux tubes' toroidal extent there, and MaNTA evaluated the final
# equilibrium with that extent, not one re-solved on it.
# itg_builder = make_itg_builder(
#     st_config,
#     eq_initial,
#     model_dir="/home/eatocco/projects/itg-proxy/demo/neural_networks/",
#     solve_length_at_compute=False,
# )

builder = make_itg_builder(
    st_config,
    eq_initial,
    model_dir="/home/eatocco/projects/itg-proxy/demo/neural_networks/",
)

for eq, name, color in zip(_eqs, names, colors):
    yancc_wrapper = yancc_data.from_eq(
        rho, eq=eq, nt=yancc_ntheta, nz=yancc_nzeta, **yancc_res
    )
    itg_wrapper = builder(eq.params_dict)

    # itg_wrapper = make_itg_flux(
    #     eq, rho, model_path="/home/eatocco/projects/itg-proxy/demo/neural_networks/"
    # )
    # itg_at(itg_builder, eq.params_dict)
    fields, vp, vpp = yancc_wrapper.get_fields()
    state = jax.vmap(
        StellaratorState.from_state, in_axes=(State.vmap_axes(), 0, 0, 0, None)
    )(state_, rho, vp, vpp, params)

    gb = jax.vmap(itg_wrapper.gyro_bohm, in_axes=(0, 0, None, None, None))(
        state.n, state.Ti, params, params.constants.B0, params.constants.a
    )
    Qi_GB = jax.vmap(itg_wrapper.normalized_heat_flux)(
        rho, -state.dTidrho / state.Ti, -0.5 * state.dndrho / state.n
    )

    # Qi = vp * jax.vmap(itg_wrapper.heat_flux_SI, in_axes=(0, 0, None))(
    #     state, rho, params
    # )

    Qi = jax.vmap(itg_wrapper, in_axes=vmap_axes_wfield)(
        state_,
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
    # V' <q . grad r> in W/m^2, r = a rho with the a the gyro-Bohm value used
    ax.plot(
        rho,
        # Qi_GB,
        -jnp.array(Qi)
        * params.constants.HeatEquationNormalization()
        * params.constants.a
        * 1e-6,
        color,
        label=name,
    )

# ax2.plot(rho, gb)
# ax2.axhline(params.constants.HeatEquationNormalization())
#
ax2.plot(rho, state.n, label="n")

ax2.plot(rho, state.Ti, label="Ti")
ax2.legend()

ax.set_xlabel(r"$\rho$")
ax.set_ylabel(r"$V'Q_{ITG}$ (MW/$m^2$)")
ax.legend()
ax.set_box_aspect(1)
fig.tight_layout()
fig.savefig("figs/compare_itg.pdf", dpi=500)
plt.show()
