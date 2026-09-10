import jax
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import ArrayLike, Float, Int
import enum
import matplotlib.pyplot as plt
import numpy as np
import pathlib
from ion_species import _IonSpecies


class PlasmaConstants(eqx.Module):
    IonSpecies: _IonSpecies
    n0: Float = eqx.field(static=True)
    n0cgs: Float = eqx.field(static=True)
    T0: Float = eqx.field(static=True)
    T0eV: Float = eqx.field(static=True)
    a: Float = eqx.field(static=True)
    Z_eff: Float = eqx.field(static=True)
    B0: Float = eqx.field(static=True)
    cs0: Float = eqx.field(static=True)
    ElectronMass = 9.1094e-31
    ProtonMass = 1.6726e-27
    ElementaryCharge = 1.60217663e-19
    VacuumPermittivity = 8.8541878128e-12

    def __init__(
        self,
        _ionSpecies,
        _n0=1e20,
        _T0=1000.0,
        _a=1.0,
        _Z_eff=3.0,
        _B0=1.0,
    ):
        self.IonSpecies = _ionSpecies
        self.n0 = _n0
        self.n0cgs = self.n0 * 1.0e-6
        self.T0 = _T0 * self.ElementaryCharge
        self.T0eV = _T0
        self.a = _a
        self.Z_eff = _Z_eff
        self.B0 = _B0
        self.cs0 = jnp.sqrt(2 * self.T0 / self.IonSpecies.mass)

    def ReferenceElectronCollisionTime(self):
        LogLambdaRef = 24.0 - jnp.log(self.n0cgs) / 2.0 + jnp.log(self.T0eV)
        return (
            12.0
            * jnp.pow(jnp.pi, 1.5)
            * jnp.sqrt(self.ElectronMass)
            * jnp.pow(self.T0, 1.5)
            * self.VacuumPermittivity**2
            / (jnp.sqrt(2) * self.n0 * jnp.pow(self.ElementaryCharge, 4) * LogLambdaRef)
        )

    def ReferenceIonCollisionTime(self):
        LogLambdaRef = (
            23.0 - jnp.log(2.0) - jnp.log(self.n0cgs) / 2.0 + jnp.log(self.T0eV) * 1.5
        )  # 23 - ln( (2n)^1/2 T^-3/2 ) from NRL pg 34
        return (
            12.0
            * jnp.pow(jnp.pi, 1.5)
            * jnp.sqrt(self.IonSpecies.IonMass)
            * jnp.pow(self.T0, 1.5)
            * self.VacuumPermittivity
            * self.VacuumPermittivity
            / (self.n0 * jnp.pow(self.ElementaryCharge, 4) * LogLambdaRef)
        )

    def ReferenceElectronThermalVelocity(self):
        return jnp.sqrt(2 * self.T0 / self.ElectronMass)

    def ReferenceIonThermalVelocity(self):
        return jnp.sqrt(2 * self.T0 / self.IonSpecies.mass)

    def ReferenceIonGyrofrequency(self):
        return self.ElementaryCharge * self.B0 / self.IonSpecies.mass

    def ReferenceElectronGyrofrequency(self):
        return self.ElementaryCharge * self.B0 / self.ElectronMass

    """
    Normalisation:
    All lengths to a, densities to n0, temperatures to T0
    We normalise time to   [ n0 T0 R_ref B_ref^2 / ( m_e Omega_e(B_ref)^2
    tau_e(n0,T0) ) ]^-1 in effect we are normalising to the particle diffusion time
    across a distance 1
    """

    def RhoStarRef(self):
        return jnp.sqrt(self.T0 * self.IonSpecies.mass) / (
            self.ElementaryCharge * self.B0 * self.a
        )

    def mu(self):
        return self.IonSpecies.mass / self.ElectronMass

    def NormalizingTime(self):
        return self.a / (self.cs0 * self.RhoStarRef() ** 2)

    def LogLambda_ii(self, ni, Ti):
        LogLambdaRef = 24.0 - jnp.log(self.n0cgs) / 2.0 + jnp.log(self.T0eV)
        LogLambda = 24.0 - jnp.log(self.n0cgs * ni) / 2.0 + jnp.log(self.T0eV * Ti)
        return LogLambda / LogLambdaRef  #  really needs to know Ti as well

    def LogLambda_ei(self, ne, Te):
        LogLambdaRef = (
            23.0 - jnp.log(2.0) - jnp.log(self.n0cgs) / 2.0 + jnp.log(self.T0eV) * 1.5
        )
        LogLambda = (
            23.0
            - jnp.log(2.0)
            - jnp.log(ne * self.n0cgs) / 2.0
            + jnp.log(Te * self.T0eV) * 1.5
        )
        return LogLambda / LogLambdaRef  # really needs to know Ti as well

    # Return tau_ei (Helander & Sigmar notation ) normalised to tau_ei( n0, 0 )
    # This is equal to tau_e as used in Braginskii
    def ElectronCollisionTime(self, ne, Te):
        return Te**1.5 / (ne * self.LogLambda_ei(ne, Te))

    # Return sqrt(2) * tau_ii (Helander & Sigmar notation ) normalised to tau_ii(
    # n0, 0 ) his is equal to tau_i as used in Braginskii
    def IonCollisionTime(self, ni, Ti):
        return Ti**1.5 / (ni * self.LogLambda_ii(ni, Ti))

    def c_s(self, Te):
        return jnp.sqrt(2 * self.T0 * Te / self.IonSpecies.mass)

    def DensityEquationNormalization(self):
        return self.n0 / self.NormalizingTime()

    def HeatEquationNormalization(self):
        return self.n0 * self.T0 / self.NormalizingTime()

    def CurrentNormalization(self):
        return self.ElementaryCharge * self.DensityEquationNormalization()

    def IonElectronEnergyExchange(self, n, pe, pi):
        Te = pe / n
        pDiff = self.n0 * self.T0 * (pe - pi)
        taue = self.ElectronCollisionTime(n, Te) * self.ReferenceElectronCollisionTime()

        IonHeating = 3 * pDiff / taue * (1 / self.mu())

        return IonHeating / self.HeatEquationNormalization()

    def FusionRate(self, n, Ti):
        return self.IonSpecies.FusionRate(n * self.n0cgs, Ti * self.T0eV/ 1000.0) * 1e6

    def AlphaHeating(self, n, Ti):
        Factor = 3.5e6 * self.T0
        return Factor * self.FusionRate(n, Ti)

