import sys

sys.path.append("..")
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx


from scipy.integrate import quad

from mirror_plasma.configs import CMFX, Fusion
from mirror_plasma.constants import PlasmaConstants
from mirror_plasma.magnetic_field import StraightMagneticField
from mirror_plasma.ion_species import Hydrogen
from mirror_plasma.mirror_plasma import MirrorPlasma
from mirror_plasma.config import MirrorPlasmaConfig
from mirror_plasma.plasma_state import (
    MirrorPlasmaConfig,
    MirrorPlasmaState,
    MirrorPlasmaParams,
)
from mirror_plasma.parallel_physics import (
    CentrifugalPotential,
    InitialPhiValue,
    ParallelCurrent,
    IonPastukhovLossRate,
    ElectronPastukhovLossRate,
    Xi_i,
)
import mirror_plasma.sources as S
from manta.jax import State

plt.rcParams.update({"font.family": "serif", "font.size": 10})
data = Dataset("mirror.nc")

t = jnp.array(data.variables["t"][:])
x = jnp.array(data.variables["x"][:])


n = jnp.array(data.groups["Density"].variables["u"][:])
dn = jnp.array(data.groups["Density"].variables["q"][:])
u_i = jnp.array(data.groups["IonEnergy"].variables["u"][:])
u_e = jnp.array(data.groups["ElectronEnergy"].variables["u"][:])
du_i = jnp.array(data.groups["IonEnergy"].variables["q"][:])
du_e = jnp.array(data.groups["ElectronEnergy"].variables["q"][:])
Gamma = jnp.array(data.groups["Density"].variables["sigma"][:])
qi = jnp.array(data.groups["IonEnergy"].variables["sigma"][:])
qe = jnp.array(data.groups["ElectronEnergy"].variables["sigma"][:])
Pi = jnp.array(data.groups["AngularMomentum"].variables["sigma"][:])
L = jnp.array(data.groups["AngularMomentum"].variables["u"][:])
dL = jnp.array(data.groups["AngularMomentum"].variables["q"][:])
phi = jnp.array(data.variables["AmbipolarPhi"][:])

nCells = 22


def cheb_nodes(nCells):
    nodes = np.ndarray((nCells + 1,))
    for i in range(0, len(nodes)):
        nodes[nCells - i] = 0.5 * (1 + np.cos(i * np.pi / nCells))
    return nodes


solver_config = {
    "restart": False,
    "OutputFilename": "mirror",
    "Grid_points": cheb_nodes(nCells),
    "Polynomial_degree": 7,
    "Grid_size": nCells,
    "tau": 10000.0,
    "Lower_boundary": 0.0,
    "Upper_boundary": 1.0,
    "Relative_tolerance": 1e-2,
    "Absolute_tolerance": [1e-3],
    "MinStepSize": 1e-10,
    "delta_t": 0.01,
}


MP = MirrorPlasma(CMFX, solver_config)

params = MP.params
if params.Config.useConstantVoltage:
    E = jnp.array(data.variables["VoltageError"][:])
    I = jnp.array(data.variables["VoltageErrorIntegral"][:])
    Current = jnp.array(data.variables["RadialCurrent"][:])

else:
    E = np.zeros((len(t),))
    I = np.zeros((len(t),))
    Current = np.zeros((len(t),))

_final_state = State(
    np.stack((n[-1, :], L[-1, :], u_i[-1, :], u_e[-1, :])).transpose(),
    np.stack((dn[-1, :], dL[-1, :], du_i[-1, :], du_e[-1, :])).transpose(),
    np.stack((Gamma[-1, :], Pi[-1, :], qi[-1, :], qe[-1, :])).transpose(),
    np.atleast_2d(phi[-1, :]).transpose(),
    np.stack((E[-1], I[-1], Current[-1])),
)

_initial_state = State(
    np.stack((n[0, :], L[0, :], u_i[0, :], u_e[0, :])).transpose(),
    np.stack((dn[0, :], dL[0, :], du_i[0, :], du_e[0, :])).transpose(),
    np.stack((Gamma[0, :], Pi[0, :], qi[0, :], qe[0, :])).transpose(),
    np.atleast_2d(phi[0, :]).transpose(),
    np.stack((E[0], I[0], Current[0])),
)

_all_states = State(
    np.stack((n, L, u_i, u_e)).transpose(),
    np.stack((dn, dL, du_i, du_e)).transpose(),
    np.stack((Gamma, Pi, qi, qe)).transpose(),
    np.atleast_3d(np.atleast_2d(phi).transpose()),
    np.stack((E, I, Current)),
)


final_state = jax.vmap(MirrorPlasmaState.from_state, (State.vmap_axes(), 0, None))(
    _final_state, x, params
)


params_no_AD = eqx.tree_at(lambda p: p.Config.ADCoefficient, params, 0.0)
MP.params = params_no_AD
fluxes_no_ad = MP.ComputePhysics(_final_state.to_manta(), x, t[-1])[0]
initial_state = jax.vmap(MirrorPlasmaState.from_state, (State.vmap_axes(), 0, None))(
    _initial_state, x, params
)
states = jax.vmap(
    jax.vmap(MirrorPlasmaState.from_state, (State.vmap_axes(), 0, None)),
    in_axes=(1, None, None),
)(_all_states, x, params)
r = final_state.R * params.Constants.a

fig, axs = plt.subplots(2, 2)
ax = axs[0, 0]
# ax.plot(r,n[0,:],label="t=0")
ax.plot(r, final_state.n, label=r"$\hat{n}$")
ax.plot(r, initial_state.n, label=r"$\hat{n}$ t=0")
ax.set_box_aspect(1)
# sol = np.array(data.groups["MMS"].variables["Var0"]);
# ax.plot(r,sol[-1,:],label="MMS solution")
ax.legend()
ax.set_title(r"Density ($10^{20}$  $1/m^3$)")
ax.set_xlabel(r"r $(m)$")
ax = axs[0, 1]
ax.plot(r, final_state.Te, label=r"$\hat{T}_e$")
ax.plot(r, final_state.Ti, label=r"$\hat{T}_i$")
ax.plot(r, initial_state.Te, label=r"$\hat{T}_e$ t=0")
ax.plot(r, initial_state.Ti, label=r"$\hat{T}_i$ t=0")
ax.set_box_aspect(1)
ax.set_xlabel(r"r $(m)$")
ax.legend()
ax.set_title(r"Ion and Electron Temperatures ($keV$)")
ax = axs[1, 0]
ax.plot(r, r * final_state.omega, label=r"$v_\theta$")
ax.plot(r, r * initial_state.omega, label=r"$v_\theta$ t=0")
ax.set_box_aspect(1)
ax.legend()
ax.set_xlabel(r"r $(m)$")
# ax.set_ylabel(r"$v_\theta (m/s)$")
ax.set_title(r"Azumuthal velocity ($km/s$)")

ax = axs[1, 1]
ax.plot(r, final_state.M, label=r"M")
ax.plot(r, initial_state.M, label=r"M t=0")
ax.legend()
ax.set_xlabel(r"r $(m)$")
ax.set_title("Mach Number")
height = 10
fig.set_size_inches(height, height)


fig, axs = plt.subplots(2, 2)
ax = axs[0, 0]
# ax.plot(r,n[0,:],label="t=0")
ax.plot(r, -final_state.gamma, label=r"$\hat{\Gamma}$")
ax.plot(r, fluxes_no_ad[0], label=r"$\hat{\Gamma}$ no AD")
ax.set_box_aspect(1)
# sol = np.array(data.groups["MMS"].variables["Var0"]);
# ax.plot(r,sol[-1,:],label="MMS solution")
ax.legend()
ax.set_xlabel(r"r $(m)$")
ax = axs[0, 1]
ax.plot(r, -final_state.Pi, label=r"$\hat{\pi}$")
ax.plot(r, fluxes_no_ad[1], label=r"$\hat{\pi}$ no AD")

ax.set_box_aspect(1)
ax.set_xlabel(r"r $(m)$")
ax.legend()
ax = axs[1, 0]

ax.plot(r, -final_state.qi, label=r"$\hat{q}_i$")
ax.plot(r, fluxes_no_ad[2], label=r"$\hat{q}_i$ no AD")

ax.set_box_aspect(1)
ax.legend()
ax.set_xlabel(r"r $(m)$")
# ax.set_ylabel(r"$v_\theta (m/s)$")
ax.set_title(r"Azumuthal velocity ($km/s$)")

ax = axs[1, 1]

ax.plot(r, -final_state.qe, label=r"$\hat{q}_e$")
ax.plot(r, fluxes_no_ad[3], label=r"$\hat{q}_e$ no AD")
ax.legend()
ax.set_xlabel(r"r $(m)$")
height = 10
fig.set_size_inches(height, height)


if params.Config.useConstantVoltage:
    fig, ax = plt.subplots()
    ax.plot(t * params.Constants.NormalizingTime(), Current * params.Constants.I0())
    ax.set_title("Current")
    fig, ax = plt.subplots()
    ax.plot(t * params.Constants.NormalizingTime(), E)
    ax.set_title("Error")
    fig, ax = plt.subplots()
    ax.plot(t * params.Constants.NormalizingTime(), I)
    ax.set_title("Integral")

Voltage = jax.vmap(jax.scipy.integrate.trapezoid, in_axes=(0, None))(
    states.omega * params.Constants.omega0 / states.VPrime, x
)

fig, ax = plt.subplots()
ax.plot(t * params.Constants.NormalizingTime(), Voltage)
ax.set_title("Voltage")
print(f"Final voltage: {Voltage[-1]}")
fig, ax = plt.subplots()
ax.plot(
    r,
    params.Constants.IonElectronEnergyExchange(
        final_state.n, final_state.pe, final_state.pi
    ),
)

viscous_heating = jax.vmap(
    S.ViscousHeating, in_axes=(MirrorPlasmaState.vmap_axes(), 0, None, None)
)(final_state, x, t, params)
potential_heating = jax.vmap(
    S.IonPotentialHeating, in_axes=(MirrorPlasmaState.vmap_axes(), 0, None, None)
)(final_state, x, t[-1], params)
energy_exchange = jax.vmap(
    params.Constants.IonElectronEnergyExchange,
    in_axes=(0),
)(final_state.n, final_state.pe, final_state.pi)
uniform_heat_source = jax.vmap(
    S.UniformHeatSource, in_axes=(MirrorPlasmaState.vmap_axes(), 0, None, None)
)(final_state, x, t, params)
ion_parallel_heat_loss = jax.vmap(
    S.IonParallelHeatLosses, in_axes=(MirrorPlasmaState.vmap_axes(), 0, None, None)
)(final_state, x, t, params)
charge_exchange_heat_loss = jax.vmap(
    S.ChargeExchangeHeatLosses, in_axes=(MirrorPlasmaState.vmap_axes(), 0, None, None)
)(final_state, x, t, params)
#
# total_ion_source = (
#     viscous_heating
#     + potential_heating
#     + energy_exchange
#     - ion_parallel_heat_loss
#     - charge_exchange_heat_loss
# )
#
fig, ax = plt.subplots()
ax.plot(r, viscous_heating, label="viscous heating")
ax.plot(r, potential_heating, label="potential_heating")
ax.plot(r, energy_exchange, label="energy_exchange")
ax.plot(r, -ion_parallel_heat_loss, label="ion_parallel_heat_loss")
ax.plot(r, -charge_exchange_heat_loss, label="charge_exchange_heat_loss")
# ax.plot(r, total_ion_source, label="total_ion_source")
ax.legend()

jxb_force = S.JxBForce(final_state, x, t, params)
parallel_angular_momentum_losses = jax.vmap(
    S.ParallelAngularMomentumLosses,
    in_axes=(MirrorPlasmaState.vmap_axes(), 0, None, None),
)(final_state, x, t, params)
charge_exhance_momentum_losses = jax.vmap(
    S.ChargeExchangeMomentumLosses,
    in_axes=(MirrorPlasmaState.vmap_axes(), 0, None, None),
)(final_state, x, t, params)

fig, ax = plt.subplots()
ax.plot(r, jxb_force, label="jxb_force")
ax.plot(r, -parallel_angular_momentum_losses, label="parallel_angular_momentum_losses")
ax.plot(r, -charge_exhance_momentum_losses, label="charge_exhance_momentum_losses")
ax.legend()

particle_source = jax.vmap(
    S.ParticleSource,
    in_axes=(MirrorPlasmaState.vmap_axes(), 0, None, None),
)(final_state, x, t[-1], params)
ionization_source = jax.vmap(
    S.IonizationSource,
    in_axes=(MirrorPlasmaState.vmap_axes(), 0, None, None),
)(final_state, x, t, params)
parallel_particle_losses = jax.vmap(
    S.ParallelParticleLosses,
    in_axes=(MirrorPlasmaState.vmap_axes(), 0, None, None),
)(final_state, x, t, params)

fig, ax = plt.subplots()
ax.plot(r, particle_source, label="particle_source")
ax.plot(r, ionization_source, label="ionization_source")
ax.plot(r, -parallel_particle_losses, label="parallel_particle_losses")
ax.legend()

alpha_heating = jax.vmap(
    S.AlphaHeating,
    in_axes=(MirrorPlasmaState.vmap_axes(), 0, None, None),
)(final_state, x, t, params)
radiation_heat_losses = jax.vmap(
    S.RadiationHeatLosses,
    in_axes=(MirrorPlasmaState.vmap_axes(), 0, None, None),
)(final_state, x, t, params)
electron_parallel_heat_losses = jax.vmap(
    S.ElectronParallelHeatLosses,
    in_axes=(MirrorPlasmaState.vmap_axes(), 0, None, None),
)(final_state, x, t, params)

fig, ax = plt.subplots()
ax.plot(r, alpha_heating, label="alpha_heating")
# ax.plot(r, -radiation_heat_losses, label="radiation_heat_losses")
# ax.plot(r, -energy_exchange, label="energy_exchange")
# ax.plot(r, -electron_parallel_heat_losses, label="electron_parallel_heat_losses")
ax.legend()

plt.show()
