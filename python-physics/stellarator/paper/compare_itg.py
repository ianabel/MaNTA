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
from config import StellaratorParams, StellaratorConfig
from yancc_wrapper import yancc_data
from stellarator_multichannel import vmap_axes_wfield
import jax
from itg_wrapper import itg_at, make_itg_builder, make_itg_flux
from manta.jax import State

eq_name = "eq_itg"
eqs = desc.io.load(eq_name + "_all_equilibria.h5")


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
        "VariableDot": jnp.zeros(Variable.shape),
        "Flux": jnp.zeros(Variable.shape),
        "Aux": Er,
        "Scalars": [],
    }
    return State.from_manta(state)


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
    "ITGScale": 1e-2,  # gyro-Bohm is large in these units; scales the prediction
    "useBatching": True,
    "nchunks": 3,
}

config = StellaratorConfig(rhoUpper=1.0, **st_config)
params = StellaratorParams(config, _B0=5.0, _a=1.0)

yancc_ntheta = 17
yancc_nzeta = 25

yancc_res = {"na": 45, "nx": 7}


rho = jnp.linspace(0.01, 0.9, 50)

_eqs = [eqs[0], eqs[-1]]

plt.rcParams.update({"font.family": "serif", "font.size": 10})

names = ["initial", "final"]
colors = ["r", "b"]
fig, ax = plt.subplots()

state = make_state_from_output(rho)
for eq, name, color in zip(_eqs, names, colors):
    yancc_wrapper = yancc_data.from_eq(
        rho, eq=eq, nt=yancc_ntheta, nz=yancc_nzeta, **yancc_res
    )
    itg_builder = make_itg_builder(st_config, eq, solve_length_at_compute=False)
    itg_wrapper = itg_at(itg_builder, eq.params_dict)
    fields, vp, vpp = yancc_wrapper.get_fields()
    Qi = jax.vmap(itg_wrapper, in_axes=vmap_axes_wfield)(
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
    ax.plot(
        rho,
        -jnp.array(Qi)
        / params.config.ITGScale
        * 1e-6
        * params.constants.HeatEquationNormalization()
        / params.constants.a,
        color,
        label=name,
    )


ax.set_xlabel(r"$\rho$")
ax.set_ylabel(r"$V'Q_{ITG}$ (MW/$m^2$)")
ax.legend()
ax.set_box_aspect(1)
fig.tight_layout()
fig.savefig("figs/compare_itg.pdf", dpi=500)
plt.show()
