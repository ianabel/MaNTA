import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import matplotlib.pyplot as plt
from MirrorPlasma.Constants import PlasmaConstants
from MirrorPlasma.MagneticField import StraightMagneticField
from MirrorPlasma.IonSpecies import Hydrogen
from MirrorPlasma.MirrorPlasma import MirrorPlasma
from MirrorPlasma.PlasmaState import MirrorPlasmaConfig
from MirrorPlasma.PlasmaState import MirrorPlasmaState
from MirrorPlasma.PlasmaState import MirrorPlasmaParams
from MirrorPlasma.ParallelPhysics import CentrifugalPotential, InitialPhiValue
from MirrorPlasma.ParallelPhysics import (
    ParallelCurrent,
    IonPastukhovLossRate,
    ElectronPastukhovLossRate,
    Xi_i,
)
from desc.backend import tree_unstack
from scipy.special import gammaincc
import sys

sys.path.append("..")

from State import State
import MaNTA

# # B = StraightMagneticField(_m=0.5, _Rmin=0.01, _Rmax=0.2)
H = Hydrogen()
k = 4
n = 21
# x = MaNTA.getNodes(0.0, 1.0, n, k)
solver_config = {
    "OutputFilename": "mirror",
    "Polynomial_degree": k,
    "Grid_size": n,
    "tau": 0.1,
    "Lower_boundary": 0.0,
    "Upper_boundary": 1.0,
    "Relative_tolerance": 1e-3,
    "MinStepSize": 1e-10,
    "delta_t": 0.1,
}


#
# plt.plot(x, B.R_x(x))
# plt.figure()
# plt.plot(x, B.VPrime(x))
# psi = B.Psi_x(x)
# fd = jnp.gradient(x, psi)
# plt.plot(x, fd)
# plt.show()
#


def make_test_state(x, MP):

    n0 = MP.InitialValue(0, x)
    L0 = MP.InitialValue(1, x)
    ui0 = MP.InitialValue(2, x)
    ue0 = MP.InitialValue(3, x)

    dn0 = MP.InitialDerivative(0, x)
    dL0 = MP.InitialDerivative(1, x)
    dui0 = MP.InitialDerivative(2, x)
    due0 = MP.InitialDerivative(3, x)
    s0 = State(
        jnp.stack([n0, L0, ui0, ue0]).transpose(),
        jnp.stack([dn0, dL0, dui0, due0]).transpose(),
        Flux_=jnp.zeros((4,)),
        Aux_=jnp.zeros((1,)),
        Scalars_=jnp.zeros((3,)),
    )

    gamma0 = MP.sigma(0, s0, x, 0.0, MP.params)
    Pi0 = MP.sigma(1, s0, x, 0.0, MP.params)
    qi0 = MP.sigma(2, s0, x, 0.0, MP.params)
    qe0 = MP.sigma(3, s0, x, 0.0, MP.params)

    s0 = eqx.tree_at(
        lambda s: s.Flux, s0, jnp.stack([gamma0, Pi0, qi0, qe0]).transpose()
    )

    return MirrorPlasmaState.from_state(s0, x, MP.params), s0


def test_bfield():
    Rmin = 0.1
    Rmax = 0.3

    B = StraightMagneticField(_Rmin=Rmin, _Rmax=Rmax)
    psi_min = 0.5 * Rmin**2 * B.B_z
    psi_max = 0.5 * Rmax**2 * B.B_z
    psi_grid = jnp.linspace(psi_min, psi_max)
    r_grid = jnp.sqrt(2 * psi_grid / B.B_z)
    VPrime = 2 * jnp.pi * B.L_z / B.B_z

    x = (2 * jnp.pi * B.L_z * psi_grid / B.B_z - B.Vmin) / (B.dV)
    Vp = jax.vmap(B.VPrime)(x)
    Vp_test = 2 * jnp.pi * B.L_z / (B.B_z * B.dV)
    print(jnp.sum((Vp - Vp_test) ** 2))

    div = jax.vmap(jax.grad(lambda x: B.VPrime(x) * (B.B_z * B.R_x(x)) ** 2))(x)

    div_test = B.B_z**2 * (2 / B.B_z)

    print(jnp.sum((div_test - div) ** 2))


def test_neutrals():

    nn = 0.001
    v = 10.0
    n = 1.0
    pi = 10.0
    C = PlasmaConstants(H, B, _nIntPoints=100)

    def XS(Energy):
        return 1e4

    T = pi / n
    vtheta = v * C.cs0
    vth2 = 2 * T * C.T0 / H.IonMass
    Mth = vtheta / jnp.sqrt(vth2)

    # Rtest = (4 * Mth**2 + 6) / 8 * jnp.exp(2 * Mth**2)
    A = Mth
    Rtest = jnp.sqrt(vth2 / 2) * (
        -gammaincc(1 / 2, A**2) * A**2
        + jnp.sqrt(jnp.pi) * A**2
        + 2 * gammaincc(1, A**2) * A
        - gammaincc(3 / 2, A**2)
        + jnp.sqrt(jnp.pi) / 2
    )

    Rtest /= jnp.sqrt(jnp.pi) * Mth
    R = C.NeutralProcess(XS, v, T, H.IonMass, 0.0)

    print(Rtest)
    print(R)
    print(100 * (R - Rtest) / Rtest)

    print(C.IonizationRate(n, nn, v, pi / n, pi / n))


def test_current():
    s1, s0, MP = make_test_state()
    plt.plot(x, s1.Ti)
    # plt.plot(x, ParallelCurrent(s1, x, 0.0, MP.params))
    plt.show()
    plt.figure()
    s1_unstack = tree_unstack(s1)

    aux = np.zeros(x.shape)
    for i in range(0, len(x)):
        print(x[i])
        aux[i] = InitialPhiValue(s1_unstack[i], x[i], 0.0, MP.params)
    # plt.plot(x, s1.dndx)
    s0 = eqx.tree_at(lambda s: s.Aux, s0, aux)

    s2 = jax.vmap(MirrorPlasmaState.from_state, in_axes=(0, 0, None))(s0, x, MP.params)
    plt.plot(
        x,
        IonPastukhovLossRate(s2, x, 0.0, MP.params)
        / MP.params.Constants.DensityEquationNormalization(),
    )
    plt.plot(
        x,
        ElectronPastukhovLossRate(s2, x, 0.0, MP.params)
        / MP.params.Constants.DensityEquationNormalization(),
    )
    plt.show()


def make_profiles(psi, params):
    R = jnp.sqrt(2 * psi / params.MagneticField.B_z)
    Rmin = params.Config.Rmin
    Rmax = params.Config.Rmax
    Rmid = 0.5 * (Rmin + Rmax)
    v = jnp.cos(jnp.pi * (R - Rmid) / (Rmax - Rmin))

    n = (
        params.Config.EdgeDensity
        + (params.Config.InitialDensityHeight - params.Config.EdgeDensity) * v
    )

    Ti = (
        params.Config.EdgeIonTemperature
        + (params.Config.InitialIonTemperatureHeight - params.Config.EdgeIonTemperature)
        * v
        * v
    )

    Te = (
        params.Config.EdgeElectronTemperature
        + (
            params.Config.InitialElectronTemperatureHeight
            - params.Config.EdgeElectronTemperature
        )
        * v
        * v
    )

    M0 = (
        params.Config.EdgeMachNumber
        + (params.Config.InitialMachNumber - params.Config.EdgeMachNumber) * v
    )
    omega = jnp.sqrt(Te) * M0 / R

    return n, omega, Ti, Te


@jax.jit
def make_gradients(psi, params):
    return jax.jacobian(lambda psi: make_profiles(psi, params))(psi)


def test_gradients():
    Rmin = 0.1
    Rmax = 0.4
    B_z = 0.34
    args = {}
    # args = {"MagneticFieldStrength": B_z}

    config = MirrorPlasmaConfig(Rmin, Rmax, **args)
    params = MirrorPlasmaParams.make(config)
    B = params.MagneticField
    C = params.Constants

    psi_min = 0.5 * Rmin**2 * B.B_z
    psi_max = 0.5 * Rmax**2 * B.B_z
    psi_grid = jnp.linspace(psi_min, psi_max)
    r_grid = jnp.sqrt(2 * psi_grid / B.B_z)
    VPrime = 2 * jnp.pi * B.L_z / B.B_z

    n, omega, Ti, Te = make_profiles(psi_grid, params)
    dn, domega, dTi, dTe = jax.vmap(make_gradients, in_axes=(0, None))(psi_grid, params)
    """
    Compute values from MirrorPlasma
    """
    MP = MirrorPlasma(config=config, solver_config=solver_config)

    x = (2 * jnp.pi * B.L_z * psi_grid / B.B_z - B.Vmin) / (B.dV)
    s1, s0 = jax.vmap(make_test_state, in_axes=(0, None))(x, MP)

    fig, ax = plt.subplots(2, 2)
    ax[0, 0].plot(x, dn, label="test")
    ax[0, 0].plot(x, s1.dndx, label="MP")

    ax[0, 1].plot(x, domega, label="test")
    ax[0, 1].plot(x, s1.domegadx, label="MP")

    ax[1, 0].plot(x, dTi, label="test")
    ax[1, 0].plot(x, s1.dTidx, label="MP")

    ax[1, 1].plot(x, dTe, label="test")
    ax[1, 1].plot(x, s1.dTedx, label="MP")

    for a in ax.flatten():
        a.legend()

    plt.show()


def gamma(psi, params):
    B = params.MagneticField
    C = params.Constants
    R = jnp.sqrt(2 * psi / B.B_z)
    VPrime = 2 * jnp.pi * B.L_z / B.B_z
    (n, omega, Ti, Te) = make_profiles(psi, params)
    (dn, domega, dTi, dTe) = make_gradients(psi, params)

    pi = n * Ti
    pe = n * Te
    dpi = n * dTi + Ti * dn
    dpe = n * dTe + Te * dn

    Uei = dpe / pe + (Ti / (C.Z_eff * Te)) * dpi / pi + omega * R * R / Te * domega
    # Uei = 1.0

    gamma0 = (
        C.B0**2
        * C.n0
        * C.T0
        / (
            C.ElectronMass
            * C.ReferenceElectronGyrofrequency() ** 2
            * C.ReferenceElectronCollisionTime()
        )
    )
    print(f"ratio of gammas {gamma0 / C.Gamma0()}")

    D = gamma0 * VPrime * R**2 * pe / (C.ElectronCollisionTime(n, Te))

    return D * (Uei - 3.0 / 2.0 * dTe / Te)


def Pi(psi, params):
    B = params.MagneticField
    R = jnp.sqrt(2 * psi / B.B_z)
    VPrime = 2 * jnp.pi * B.L_z / B.B_z
    (n, omega, Ti, Te) = make_profiles(psi, params)
    (dn, domega, dTi, dTe) = make_gradients(psi, params)

    pi = n * Ti
    pe = n * Te
    dpi = n * dTi + Ti * dn
    dpe = n * dTe + Te * dn

    C = params.Constants
    Pi0 = (C.B0**2 * C.n0 * C.T0 * C.omega0) / (
        C.ReferenceIonGyrofrequency() ** 2 * C.ReferenceIonCollisionTime()
    )

    D = VPrime * 3.0 / 10.0 * Pi0 * R**4 * pi / C.IonCollisionTime(n, Ti)

    print(f"ratio of pis {Pi0 / C.Pi0()}")
    return D * domega + C.IonSpecies.IonMass * omega * C.omega0 * R**2 * gamma(
        psi, params
    )


def qi(psi, params):
    B = params.MagneticField
    R = jnp.sqrt(2 * psi / B.B_z)
    VPrime = 2 * jnp.pi * B.L_z / B.B_z
    (n, omega, Ti, Te) = make_profiles(psi, params)
    (dn, domega, dTi, dTe) = make_gradients(psi, params)

    pi = n * Ti
    pe = n * Te
    dpi = n * dTi + Ti * dn
    dpe = n * dTe + Te * dn

    C = params.Constants

    qi0 = (C.B0**2 * C.n0 * C.T0**2) / (
        C.IonSpecies.IonMass
        * C.ReferenceIonGyrofrequency() ** 2
        * C.ReferenceIonCollisionTime()
    )

    D = VPrime * 2 * qi0 * R**2 * pi * Ti / C.IonCollisionTime(n, Ti)

    return D * dTi / Ti - 0.5 * C.IonSpecies.IonMass * (
        R * omega * C.omega0
    ) ** 2 * gamma(psi, params)


def qe(psi, params):
    B = params.MagneticField
    R = jnp.sqrt(2 * psi / B.B_z)
    VPrime = 2 * jnp.pi * B.L_z / B.B_z
    (n, omega, Ti, Te) = make_profiles(psi, params)
    (dn, domega, dTi, dTe) = make_gradients(psi, params)

    pi = n * Ti
    pe = n * Te
    dpi = n * dTi + Ti * dn
    dpe = n * dTe + Te * dn

    C = params.Constants

    Uei = dpe / pe + (Ti / (C.Z_eff * Te)) * dpi / pi + omega * R * R / Te * domega

    qe0 = (C.B0**2 * C.n0 * C.T0**2) / (
        C.ElectronMass
        * C.ReferenceElectronGyrofrequency() ** 2
        * C.ReferenceElectronCollisionTime()
    )

    D = VPrime * R**2 * qe0 * pe * Te / C.ElectronCollisionTime(n, Te)

    return D * (4.66 * dTe / Te - 3.0 / 2.0 * Uei)


def test_fluxes():
    Rmin = 0.1
    Rmax = 0.4
    B_z = 0.34
    args = {}
    # args = {"MagneticFieldStrength": B_z}

    config = MirrorPlasmaConfig(Rmin, Rmax, **args)
    params = MirrorPlasmaParams.make(config)
    B = params.MagneticField
    C = params.Constants

    psi_min = 0.5 * Rmin**2 * B.B_z
    psi_max = 0.5 * Rmax**2 * B.B_z
    psi_grid = jnp.linspace(psi_min, psi_max)
    r_grid = jnp.sqrt(2 * psi_grid / B.B_z)
    VPrime = 2 * jnp.pi * B.L_z / B.B_z

    n = make_profiles(psi_grid, params)[0]
    dn = jax.vmap(make_gradients, in_axes=(0, None))(psi_grid, params)[0]

    """
    Compute test values
    """

    div_gamma_test = (
        1 / VPrime * jax.vmap(jax.grad(gamma), in_axes=(0, None))(psi_grid, params)
    )
    div_pi_test = (
        1 / VPrime * jax.vmap(jax.grad(Pi), in_axes=(0, None))(psi_grid, params)
    )
    div_qi_test = (
        1 / VPrime * jax.vmap(jax.grad(qi), in_axes=(0, None))(psi_grid, params)
    )
    div_qe_test = (
        1 / VPrime * jax.vmap(jax.grad(qe), in_axes=(0, None))(psi_grid, params)
    )

    """
    Compute values from MirrorPlasma
    """
    MP = MirrorPlasma(config=config, solver_config=solver_config)

    def flux(x, index):
        s1, s0 = make_test_state(x, MP)
        return MP.sigma(index, s0, x, 0.0, MP.params)

    @jax.jit
    def div_flux(x, index):
        return jax.vmap(jax.grad(flux), in_axes=(0, None))(x, index)

    x = (2 * jnp.pi * B.L_z * psi_grid / B.B_z - B.Vmin) / (B.dV)
    print(jnp.sum((r_grid - B.R_x(x)) ** 2))
    print(jnp.sum((psi_grid - B.Psi_x(x)) ** 2))

    div_gamma = div_flux(x, 0) * MP.params.Constants.DensityEquationNormalization()
    div_pi = div_flux(x, 1) * MP.params.Constants.MomentumEquationNormalization()
    div_qi = div_flux(x, 2) * MP.params.Constants.HeatEquationNormalization()
    div_qe = div_flux(x, 3) * MP.params.Constants.HeatEquationNormalization()

    print(
        f"Gamma error: {jnp.sum((div_gamma_test - div_gamma) ** 2) / jnp.mean(div_gamma_test)}"
    )
    print(jnp.mean(div_gamma_test / div_gamma))
    plt.figure()
    plt.plot(div_gamma_test / div_gamma)

    fig, ax = plt.subplots(2, 2)
    titles = [
        r"$\nabla \cdot \Gamma$",
        r"$\nabla\cdot\pi_i$",
        r"$\nabla\cdot q_i$",
        r"$\nabla \cdot q_e$",
    ]
    ax[0, 0].plot(r_grid, div_gamma_test, label="test")
    ax[0, 0].plot(r_grid, div_gamma, label="MP")

    ax[0, 1].plot(r_grid, div_pi_test, label="test")
    ax[0, 1].plot(r_grid, div_pi, label="MP")

    ax[1, 0].plot(r_grid, div_qi_test, label="test")
    ax[1, 0].plot(r_grid, div_qi, label="MP")

    ax[1, 1].plot(r_grid, div_qe_test, label="test")
    ax[1, 1].plot(r_grid, div_qe, label="MP")
    for a, t in zip(ax.flatten(), titles):
        a.legend()
        a.set_title(t)
    plt.show()


def test_sources():
    s1, s0, MP = make_test_state()
    fig, axs = plt.subplots(1, 4)
    s1_unstack = tree_unstack(s1)
    aux = np.zeros(x.shape)
    for i in range(0, len(x)):
        aux[i] = InitialPhiValue(s1_unstack[i], x[i], 0.0, MP.params)
    # plt.plot(x, s1.dndx)
    s0 = eqx.tree_at(lambda s: s.Aux, s0, aux)

    # s2 = jax.vmap(MirrorPlasmaState.from_state, in_axes=(0, 0, None))(s0, x, MP.params)

    for i in range(0, 4):
        axs[i].plot(x, MP.Sources_v(i, s0, x, 0.0))

    plt.show()


test_fluxes()
