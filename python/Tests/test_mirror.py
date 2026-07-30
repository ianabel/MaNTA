import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import pytest
import matplotlib.pyplot as plt
import sys

sys.path.append("..")

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

from State import State
import MaNTA

# # B = StraightMagneticField(_m=0.5, _Rmin=0.01, _Rmax=0.2)
H = Hydrogen()


@pytest.fixture
def solver_config():
    k = 4
    n = 21
    return {
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


@pytest.fixture
def atol():
    return 1e-6


@pytest.fixture
def config():
    Rmin = 0.1
    Rmax = 0.4
    B_z = 0.34
    args = {"MagneticFieldStrength": B_z, "ADCoefficient": 0.0}

    return MirrorPlasmaConfig(Rmin, Rmax, **args)


@pytest.fixture
def params(config):
    params = MirrorPlasmaParams.make(config)
    return params


@pytest.fixture
def B(params):
    return params.MagneticField


@pytest.fixture
def C(params):
    return params.Constants


@pytest.fixture
def psi(B, config):
    Rmin = config.Rmin
    Rmax = config.Rmax

    psi_min = 0.5 * Rmin**2 * B.B_z
    psi_max = 0.5 * Rmax**2 * B.B_z
    psi_grid = jnp.linspace(psi_min, psi_max)
    return psi_grid


@pytest.fixture
def VPrime(B):
    return 2 * jnp.pi * B.L_z / B.B_z


def make_x(psi, B):
    return (2 * jnp.pi * B.L_z * psi / B.B_z - B.Vmin) / (B.dV)


@pytest.fixture
def x(psi, B):
    return make_x(psi, B)


@pytest.fixture
def MP(config, solver_config):

    return MirrorPlasma(config=config, solver_config=solver_config)


def make_state(x, MP):
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

    return eqx.tree_at(
        lambda s: s.Flux, s0, jnp.stack([gamma0, Pi0, qi0, qe0]).transpose()
    )


@pytest.fixture
def state(x, MP):
    return jax.vmap(make_state, in_axes=(0, None))(x, MP)


def make_mirror_state(x, MP):
    if jnp.isscalar(x):
        return MirrorPlasmaState.from_state(make_state(x, MP), x, MP.params)
    else:
        return jax.vmap(
            lambda x, MP: MirrorPlasmaState.from_state(make_state(x, MP), x, MP.params),
            in_axes=(0, None),
        )(x, MP)


def profiles(psi, MP):
    s = make_mirror_state(make_x(psi, MP.params.MagneticField), MP)
    return s.n, s.omega / MP.params.Constants.a, s.Ti, s.Te


def gradients(psi, MP):

    s = make_mirror_state(make_x(psi, MP.params.MagneticField), MP)
    return (
        s.dndpsi,
        s.domegadpsi / MP.params.Constants.a,
        s.dTidpsi,
        s.dTedpsi,
    )


def test_norm(C):
    B = StraightMagneticField()
    H = Hydrogen()
    ElectronMass = 9.1094e-31
    ProtonMass = 1.6726e-27
    ElementaryCharge = 1.60217663e-19
    VacuumPermittivity = 8.8541878128e-12

    T0 = C.T0eV * ElementaryCharge
    n0 = C.n0

    def make_norm(a, B0):
        rho_star = jnp.sqrt(T0 * ProtonMass) / (ElementaryCharge * a * B0)
        C = PlasmaConstants(H, B, _a=a, _B0=B0)
        tau = C.ReferenceIonCollisionTime() / rho_star**2
        omega0 = 1 / a * jnp.sqrt(T0 / ProtonMass)

        DENSITY_EQ_NORM = n0 / tau
        MOMENTUM_EQ_NORM = ProtonMass * n0 * a**2 * omega0 / tau
        HEAT_EQ_NORM = n0 * T0 / tau
        return DENSITY_EQ_NORM, MOMENTUM_EQ_NORM, HEAT_EQ_NORM, C

    t1 = (1.0, 1.0)
    t2 = (0.2, 1.0)
    t3 = (1.0, 0.2)
    ts = (t1, t2, t3)

    for t in ts:
        (dndt0, dLdt0, dudt0, C) = make_norm(*t)

        print(dndt0 - C.DensityEquationNormalization())
        print(dLdt0 - C.MomentumEquationNormalization())
        print(dudt0 - C.HeatEquationNormalization())


def test_bfield(config):
    Rmax = config.Rmax
    Rmin = config.Rmin

    B = StraightMagneticField(_Rmin=Rmin, _Rmax=Rmax)
    psi_min = 0.5 * Rmin**2 * B.B_z
    psi_max = 0.5 * Rmax**2 * B.B_z
    psi_grid = jnp.linspace(psi_min, psi_max)

    x = (2 * jnp.pi * B.L_z * psi_grid / B.B_z - B.Vmin) / (B.dV)
    Vp = jax.vmap(B.VPrime)(x)
    Vp_test = 2 * jnp.pi * B.L_z / (B.B_z * B.dV)
    print(jnp.sum((Vp - Vp_test) ** 2))

    div = jax.vmap(jax.grad(lambda x: B.VPrime(x) * (B.B_z * B.R_x(x)) ** 2))(x)

    div_test = B.B_z**2 * (2 / B.B_z)

    assert jnp.sum((div_test - div) ** 2) == pytest.approx(0.0)


def test_neutrals(atol):

    H = Hydrogen()
    B = StraightMagneticField()
    v = 5.0
    n = 1.0
    pi = 1.0
    C = PlasmaConstants(H, B, _nIntPoints=200)
    T = pi / n

    vth2 = 2 * T * C.T0 / H.IonMass
    v *= C.cs0

    def XS(Energy):
        return 1e4

    def Ian(v):
        spi = np.sqrt(np.pi)
        v2 = v**2
        a1 = v2 * (2 * gammaincc(0.5, v2) - 2 * spi)
        a2 = 4 * gammaincc(1, v2) * v
        a3 = 2 * gammaincc(3.0 / 2.0, v2)

        return -0.25 * (a1 - a2 + a3 - spi)

    Rtest = (
        Ian(v / jnp.sqrt(vth2))
        * 2.0
        * jnp.sqrt(vth2)
        / jnp.sqrt(jnp.pi)
        / (v / jnp.sqrt(vth2))
    )
    R = C.NeutralProcess(XS, v, T, H.IonMass, 0.0)

    assert jnp.abs((R - Rtest) / Rtest) == pytest.approx(0.0, abs=atol)


def test_current(state, x, MP):

    mirror_state = make_mirror_state(x, MP)
    s1_unstack = tree_unstack(mirror_state)

    aux = np.zeros(x.shape)
    for i in range(0, len(x)):
        print(x[i])
        aux[i] = InitialPhiValue(s1_unstack[i], x[i], 0.0, MP.params)
    # plt.plot(x, s1.dndx)
    state = eqx.tree_at(lambda s: s.Aux, state, jnp.atleast_2d(aux).transpose())

    s2 = jax.vmap(MirrorPlasmaState.from_state, in_axes=(0, 0, None))(
        state, x, MP.params
    )

    dnidt = (
        IonPastukhovLossRate(s2, x, 0.0, MP.params)
        / MP.params.Constants.DensityEquationNormalization()
    )

    dnedt = (
        ElectronPastukhovLossRate(s2, x, 0.0, MP.params)
        / MP.params.Constants.DensityEquationNormalization()
    )

    assert (jnp.sum((dnidt - dnedt) ** 2 / dnidt)) == pytest.approx(0.0)


def gamma(psi, B, C, MP):
    R = jnp.sqrt(2 * psi / B.B_z)
    VPrime = 2 * jnp.pi * B.L_z / B.B_z
    (n, omega, Ti, Te) = profiles(psi, MP)
    (dn, domega, dTi, dTe) = gradients(psi, MP)

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

    D = gamma0 * VPrime * R**2 * pe / (C.ElectronCollisionTime(n, Te))

    return D * (Uei - 3.0 / 2.0 * dTe / Te)


def Pi(psi, B, C, MP):
    R = jnp.sqrt(2 * psi / B.B_z)
    VPrime = 2 * jnp.pi * B.L_z / B.B_z
    (n, omega, Ti, Te) = profiles(psi, MP)
    (dn, domega, dTi, dTe) = gradients(psi, MP)

    pi = n * Ti
    pe = n * Te
    dpi = n * dTi + Ti * dn
    dpe = n * dTe + Te * dn

    Pi0 = (C.B0**2 * C.n0 * C.T0 * C.omega0 * C.a) / (
        C.ReferenceIonGyrofrequency() ** 2 * C.ReferenceIonCollisionTime()
    )

    D = VPrime * 3.0 / 10.0 * Pi0 * R**4 * pi / C.IonCollisionTime(n, Ti)

    print(f"ratio of pis {Pi0 / C.Pi0()}")
    return D * domega + C.IonSpecies.IonMass * omega * C.omega0 * C.a * R**2 * gamma(
        psi, B, C, MP
    )


def qi(psi, B, C, MP):
    R = jnp.sqrt(2 * psi / B.B_z)
    VPrime = 2 * jnp.pi * B.L_z / B.B_z
    (n, omega, Ti, Te) = profiles(psi, MP)
    (dn, domega, dTi, dTe) = gradients(psi, MP)

    pi = n * Ti
    pe = n * Te
    dpi = n * dTi + Ti * dn
    dpe = n * dTe + Te * dn

    qi0 = (C.B0**2 * C.n0 * C.T0**2) / (
        C.IonSpecies.IonMass
        * C.ReferenceIonGyrofrequency() ** 2
        * C.ReferenceIonCollisionTime()
    )

    D = VPrime * 2 * qi0 * R**2 * pi * Ti / C.IonCollisionTime(n, Ti)

    return D * dTi / Ti - 0.5 * C.IonSpecies.IonMass * (
        R * omega * C.omega0 * C.a
    ) ** 2 * gamma(psi, B, C, MP)


def qe(psi, B, C, MP):
    R = jnp.sqrt(2 * psi / B.B_z)
    VPrime = 2 * jnp.pi * B.L_z / B.B_z
    (n, omega, Ti, Te) = profiles(psi, MP)
    (dn, domega, dTi, dTe) = gradients(psi, MP)

    pi = n * Ti
    pe = n * Te
    dpi = n * dTi + Ti * dn
    dpe = n * dTe + Te * dn

    Uei = dpe / pe + (Ti / (C.Z_eff * Te)) * dpi / pi + omega * R * R / Te * domega

    qe0 = (C.B0**2 * C.n0 * C.T0**2) / (
        C.ElectronMass
        * C.ReferenceElectronGyrofrequency() ** 2
        * C.ReferenceElectronCollisionTime()
    )

    D = VPrime * R**2 * qe0 * pe * Te / C.ElectronCollisionTime(n, Te)

    return D * (4.66 * dTe / Te - 3.0 / 2.0 * Uei)


def test_fluxes(psi, x, B, C, MP, VPrime, atol):
    """
    Compute test values
    """

    div_gamma_test = (
        1
        / VPrime
        * jax.vmap(jax.grad(gamma), in_axes=(0, None, None, None))(psi, B, C, MP)
    )
    div_pi_test = (
        1
        / VPrime
        * jax.vmap(jax.grad(Pi), in_axes=(0, None, None, None))(psi, B, C, MP)
    )
    div_qi_test = (
        1
        / VPrime
        * jax.vmap(jax.grad(qi), in_axes=(0, None, None, None))(psi, B, C, MP)
    )
    div_qe_test = (
        1
        / VPrime
        * jax.vmap(jax.grad(qe), in_axes=(0, None, None, None))(psi, B, C, MP)
    )
    """
    Compute values from MirrorPlasma
    """

    def flux(x, index):
        s0 = make_state(x, MP)
        return MP.sigma(index, s0, x, 0.0, MP.params)

    @jax.jit
    def div_flux(x, index):
        return jax.vmap(jax.grad(flux), in_axes=(0, None))(x, index)

    x = (2 * jnp.pi * B.L_z * psi / B.B_z - B.Vmin) / (B.dV)

    div_gamma = div_flux(x, 0) * MP.params.Constants.DensityEquationNormalization()
    div_pi = div_flux(x, 1) * MP.params.Constants.MomentumEquationNormalization()
    div_qi = div_flux(x, 2) * MP.params.Constants.HeatEquationNormalization()
    div_qe = div_flux(x, 3) * MP.params.Constants.HeatEquationNormalization()

    assert jnp.sum((div_gamma_test - div_gamma) ** 2) / jnp.mean(
        div_gamma_test
    ) == pytest.approx(0.0, abs=atol)
    assert jnp.sum((div_pi_test - div_pi) ** 2) / jnp.mean(
        div_pi_test
    ) == pytest.approx(0.0, abs=atol)
    assert jnp.sum((div_qi_test - div_qi) ** 2) / jnp.mean(
        div_qi_test
    ) == pytest.approx(0.0, abs=atol)
    assert jnp.sum((div_qe_test - div_qe) ** 2) / jnp.mean(
        div_qe_test
    ) == pytest.approx(0.0, abs=atol)


def test_sources(psi, x, MP, B, C, VPrime, atol):

    mirror_state = make_mirror_state(x, MP)
    n, omega, Ti, Te = profiles(psi, MP)
    dn, domega, dTi, dTe = gradients(psi, MP)

    pi_test = jax.vmap(Pi, in_axes=(0, None, None, None))(psi, B, C, MP)
    viscous_heating_test = -domega * C.a * C.omega0 * pi_test / VPrime

    """
    Compute values from MirrorPlasma
    """
    viscous_heating = C.HeatEquationNormalization() * jax.vmap(
        MP.ViscousHeating, in_axes=(MirrorPlasmaState.vmap_axes(), 0, None, None)
    )(mirror_state, x, 0.0, MP.params)

    """

    """
    assert jnp.sum((viscous_heating_test - viscous_heating) ** 2) / jnp.mean(
        viscous_heating_test
    ) == pytest.approx(0.0, abs=atol)

    JxBtest = state.Current / VPrime

    JxB = MP.JxBForce(mirror_state[0], x[0], 0.0, MP.params)

    assert jnp.abs(JxB - JxBtest) == pytest.approx(0.0, abs=atol)
