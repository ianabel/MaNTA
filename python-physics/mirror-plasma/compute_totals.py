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
    Channel,
)
from mirror_plasma.parallel_physics import (
    CentrifugalPotential,
    InitialPhiValue,
    ParallelCurrent,
    IonPastukhovLossRate,
    ElectronPastukhovLossRate,
    Xi_i,
)


from mirror_plasma.sources import source_registry, sink_registry
from manta.jax import State
import re


def split_camel_case(text):
    # Inserts a space between a lowercase/number and an uppercase letter
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1 \2", text)
    # Inserts a space between a lowercase letter and an uppercase letter/number
    return re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s1)


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

E = jnp.array(data.variables["VoltageError"][:])
I = jnp.array(data.variables["VoltageErrorIntegral"][:])
Current = jnp.array(data.variables["RadialCurrent"][:])

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
points = MP.points

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
Voltage = jax.scipy.integrate.trapezoid(
    final_state.omega * params.Constants.omega0 / final_state.VPrime, x
)


print("\n")
print("MEAN VALUES")
print("-----------")
print(f"Density: {jnp.mean(final_state.n) * params.Constants.n0:.3e} particles")
print(f"Ion Temperature: {jnp.mean(final_state.Ti) * params.Constants.T0eV:.3f} eV")
print(
    f"Electron Temperature: {jnp.mean(final_state.Te) * params.Constants.T0eV:.3f} eV"
)
print(
    f"Angular Frequency: {jnp.mean(final_state.omega) * params.Constants.omega0:.3e} 1/s"
)
print(f"Mach Number: {jnp.mean(final_state.M):.3f}")

print(f"Current: {final_state.Current[0] * params.Constants.I0():.3f} A")
print(f"Voltage: {Voltage}")
print(f"Input Power: {Voltage * final_state.Current[0] * params.Constants.I0():.3f}")

print("\n")
print("DENSITY SOURCES")
print("---------------")
density_sources = 0.0
for source in source_registry[Channel.Density]:
    sval = jax.vmap(source, in_axes=(MirrorPlasmaState.vmap_axes(), 0, None, None))(
        final_state, x, t[-1], params
    )
    s_int = (
        jax.scipy.integrate.trapezoid(sval, x)
        * params.Constants.DensityEquationNormalization()
    )
    density_sources += s_int
    print(f"{split_camel_case(source.name)}: {s_int:.3e} #/s")

print("\n")
print("DENSITY SINKS")
print("-------------")
density_sinks = 0.0
for sink in sink_registry[Channel.Density]:
    sval = jax.vmap(sink, in_axes=(MirrorPlasmaState.vmap_axes(), 0, None, None))(
        final_state, x, t[-1], params
    )
    s_int = (
        jax.scipy.integrate.trapezoid(sval, x)
        * params.Constants.DensityEquationNormalization()
    )
    density_sinks += s_int
    print(f"{split_camel_case(sink.name)}: {s_int:.3e} #/s")

print("\n")
print("ANGULAR MOMENTUM SOURCES")
print("------------------------")
angular_momentum_sources = 0.0
for source in source_registry[Channel.AngularMomentum]:
    sval = source(final_state, x, t[-1], params)

    s_int = (
        jax.scipy.integrate.trapezoid(sval, x)
        * params.Constants.MomentumEquationNormalization()
    )
    angular_momentum_sources += s_int
    print(f"{split_camel_case(source.name)}: {s_int:.3e} N*m/s")

print("\n")
print("ANGULAR MOMENTUM SINKS")
print("----------------------")
angular_momentum_sinks = 0.0
for sink in sink_registry[Channel.AngularMomentum]:
    sval = jax.vmap(sink, in_axes=(MirrorPlasmaState.vmap_axes(), 0, None, None))(
        final_state, x, t[-1], params
    )

    s_int = (
        jax.scipy.integrate.trapezoid(sval, x)
        * params.Constants.MomentumEquationNormalization()
    )
    angular_momentum_sinks += s_int
    print(f"{split_camel_case(sink.name)}: {s_int:.3e} N*m/s")


print("\n")
print("ION HEAT SOURCES")
print("----------------")
ion_energy_sources = 0.0
for source in source_registry[Channel.IonEnergy]:
    sval = jax.vmap(source, in_axes=(MirrorPlasmaState.vmap_axes(), 0, None, None))(
        final_state, x, t[-1], params
    )
    s_int = (
        jax.scipy.integrate.trapezoid(sval, x)
        * params.Constants.HeatEquationNormalization()
    )
    ion_energy_sources += s_int
    print(f"{split_camel_case(source.name)}: {s_int:.3e} W")


print("\n")
print("ION HEAT SINKS")
print("--------------")
ion_energy_sinks = 0.0
for sink in sink_registry[Channel.IonEnergy]:
    sval = jax.vmap(sink, in_axes=(MirrorPlasmaState.vmap_axes(), 0, None, None))(
        final_state, x, t[-1], params
    )
    s_int = (
        jax.scipy.integrate.trapezoid(sval, x)
        * params.Constants.HeatEquationNormalization()
    )
    ion_energy_sinks += s_int
    print(f"{split_camel_case(sink.name)}: {s_int:.3e} W")


print("\n")
print("ELECTRON HEAT SOURCES")
print("---------------------")
electron_energy_sources = 0.0
for source in source_registry[Channel.ElectronEnergy]:
    sval = jax.vmap(source, in_axes=(MirrorPlasmaState.vmap_axes(), 0, None, None))(
        final_state, x, t[-1], params
    )
    s_int = (
        jax.scipy.integrate.trapezoid(sval, x)
        * params.Constants.HeatEquationNormalization()
    )
    electron_energy_sources += s_int
    print(f"{split_camel_case(source.name)}: {s_int:.3e} W")

print("\n")
print("ELECTRON HEAT SINKS")
print("-------------------")
electron_energy_sinks = 0.0
for sink in sink_registry[Channel.ElectronEnergy]:
    sval = jax.vmap(sink, in_axes=(MirrorPlasmaState.vmap_axes(), 0, None, None))(
        final_state, x, t[-1], params
    )
    s_int = (
        jax.scipy.integrate.trapezoid(sval, x)
        * params.Constants.HeatEquationNormalization()
    )
    electron_energy_sinks += s_int
    print(f"{split_camel_case(sink.name)}: {s_int:.3e} W")


print("\n")
print("PARTICLE CONSERVATION")
print("---------------------")

particle_flux_conservation = (
    final_state.gamma[-1] - final_state.gamma[0]
) * params.Constants.DensityEquationNormalization()

print("    |xR")
print(
    f"V'Γ | - (Sources - Sinks) = {particle_flux_conservation - (density_sources - density_sinks):.3e} #/s"
)
print("    |xL")


print("\n")
print("ANGULAR MOMENTUM CONSERVATION")
print("-----------------------------")

angular_momentum_flux_conservation = (
    final_state.Pi[-1] - final_state.Pi[0]
) * params.Constants.MomentumEquationNormalization()

print("    |xR")
print(
    f"V'π | - (Sources - Sinks) = {angular_momentum_flux_conservation - (angular_momentum_sources - angular_momentum_sinks):.3e} N*m/s"
)
print("    |xL")

print("\n")
print("ION ENERGY CONSERVATION")
print("-----------------------")

ion_energy_flux_conservation = (
    final_state.qi[-1] - final_state.qi[0]
) * params.Constants.HeatEquationNormalization()


print("    |xR")
print(
    f"V'q | - (Sources - Sinks) = {ion_energy_flux_conservation - (jnp.sum(ion_energy_sources) - jnp.sum(ion_energy_sinks)):.3e} W"
)
print("    |xL")


print("\n")
print("ELECTRON ENERGY CONSERVATION")
print("----------------------------")

electron_energy_flux_conservation = (
    final_state.qe[-1] - final_state.qe[0]
) * params.Constants.HeatEquationNormalization()

print("    |xR")
print(
    f"V'q | - (Sources - Sinks) = {electron_energy_flux_conservation - (electron_energy_sources - electron_energy_sinks):.3e} W"
)
print("    |xL")
