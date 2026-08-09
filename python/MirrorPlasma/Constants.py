import jax

jax.config.update("jax_enable_x64", True)
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import ArrayLike, Float, Int
import enum
from MirrorPlasma.MagneticField import _MagneticField
from MirrorPlasma.IonSpecies import _IonSpecies
import matplotlib.pyplot as plt
import numpy as np
import pickle


class MomentType(enum.IntEnum):
    Density = 0
    Momentum = 1
    Energy = 2


class PlasmaConstants(eqx.Module):
    IonSpecies: _IonSpecies
    MagneticField: _MagneticField
    n0: Float = eqx.field(static=True)
    n0cgs: Float = eqx.field(static=True)
    T0: Float = eqx.field(static=True)
    T0eV: Float = eqx.field(static=True)
    omega0: Float = eqx.field(static=True)
    a: Float = eqx.field(static=True)
    Z_eff: Float = eqx.field(static=True)
    B0: Float = eqx.field(static=True)
    cs0: Float = eqx.field(static=True)
    nIntPoints: Int = eqx.field(static=True)
    land: tuple
    herm: tuple
    ElectronMass = 9.1094e-31
    ProtonMass = 1.6726e-27
    ElementaryCharge = 1.60217663e-19
    VacuumPermittivity = 8.8541878128e-12

    def __init__(
        self,
        _ionSpecies,
        _magneticField,
        _n0=1e20,
        _T0=1000.0,
        _a=1.0,
        _Z_eff=3.0,
        _B0=1.0,
        _nIntPoints=200,
    ):
        self.n0 = _n0
        self.n0cgs = self.n0 * 1.0e-6
        self.T0 = _T0 * self.ElementaryCharge
        self.T0eV = _T0
        self.a = _a
        self.Z_eff = _Z_eff
        self.B0 = _B0
        self.IonSpecies = _ionSpecies
        self.MagneticField = _magneticField
        self.cs0 = jnp.sqrt(self.T0 / self.IonSpecies.IonMass)
        self.nIntPoints = _nIntPoints
        self.omega0 = self.cs0 / self.a

        a, w = np.polynomial.hermite.hermgauss(19)
        self.herm = (a, w)

        with open("util/land.pkl", "rb") as file:
            a = pickle.load(file)
            w = jnp.array(pickle.load(file))
            self.land = (a, w)

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
        return jnp.sqrt(2 * self.T0 / self.IonSpecies.IonMass)

    def ReferenceIonGyrofrequency(self):
        return self.ElementaryCharge * self.B0 / self.IonSpecies.IonMass

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
        return jnp.sqrt(self.T0 * self.IonSpecies.IonMass) / (
            self.ElementaryCharge * self.B0 * self.a
        )

    def mu(self):
        return self.IonSpecies.IonMass / self.ElectronMass

    def NormalizingTime(self):
        return self.ReferenceIonCollisionTime() / self.RhoStarRef() ** 2

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

    def Om_i(self, x):
        return self.MagneticField.B(x)

    def Om_e(self, x):
        return self.MagneticField.B(x)

    def c_s(self, Te):
        return jnp.sqrt(self.T0 * Te / self.IonSpecies.IonMass)

    def FusionRate(self, n, pi):
        Ti_keV = pi / n * self.T0eV / 1000.0
        return self.IonSpecies.FusionRate(self.n0 * n, Ti_keV)

    def TotalAlphaPower(self, n, pi):
        Factor = 5.6e-13
        return Factor * self.FusionRate(n, pi) / self.HeatEquationNormalization()

    def BremsstrahlungLosses(self, n, pe):
        n20 = n * self.n0 / 1e20
        TkeV = pe / n * self.T0eV / 1000.0
        Pbrem = 5.34e3 * jnp.sqrt(TkeV) * self.Z_eff * n20**2

        return Pbrem / self.HeatEquationNormalization()

    def CyclotronLosses(self, x, n, Te):
        # NRL formulary with reference values factored out
        # Return units are W/m^3
        Te_eV = self.T0eV * Te
        n_e20 = n * self.n0 / 1e20
        B_z = self.MagneticField.B(x) * self.B0  # in Tesla
        P_vacuum = 6.21 * n_e20 * Te_eV * B_z * B_z

        # Characteristic absorption length
        # lambda_0 = (Electron Inertial Lenght) / ( Plasma Frequency / Cyclotron
        # Frequency )   Eq (4) of Tamor
        # = (5.31 * 10^-4 / (n_e20)^1/2) / ( 3.21 *
        # n_e20)^1/2 / B )  From NRL Formulary, converted to our units (Tesla for B
        # 10^20 /m^3 for n_e)

        PlasmaWidth = self.MagneticField.R_x(1.0) - self.MagneticField.R_x(0.0)
        LambdaZero = (5.31e-4 / 3.21) * (B_z / n_e20)
        WallReflectivity = 0.95
        OpticalThickness = (PlasmaWidth / (1.0 - WallReflectivity)) / LambdaZero
        # This is the Phi introduced by Trubnikov and later approximated by Tamor
        TransparencyFactor = pow(Te, 1.5) / (200.0 * jnp.sqrt(OpticalThickness))
        # Moderate the vacuum emission by the transparency factor

        P_cy = P_vacuum * TransparencyFactor / self.HeatEquationNormalization()
        return P_cy

    def DensityEquationNormalization(self):
        return self.n0 / self.NormalizingTime()

    # m_i * n * R^2 * omega
    def MomentumEquationNormalization(self):
        return (
            self.IonSpecies.IonMass
            * self.n0
            * self.a**2
            * self.omega0
            / self.NormalizingTime()
        )

    def HeatEquationNormalization(self):
        return self.n0 * self.T0 / self.NormalizingTime()

    def IonElectronEnergyExchange(self, n, pe, pi):
        Te = pe / n
        pDiff = self.n0 * self.T0 * (pe - pi)
        taue = self.ElectronCollisionTime(n, Te) * self.ReferenceElectronCollisionTime()

        IonHeating = 3 * pDiff / taue * (1 / self.mu())

        return IonHeating / self.HeatEquationNormalization()

    # Hold all dimensional values for fluxes here
    def Gamma0(self):
        return (
            self.B0**2
            * self.a**2
            * self.n0
            * self.T0
            / (
                self.ElectronMass
                * self.ReferenceElectronGyrofrequency() ** 2
                * self.ReferenceElectronCollisionTime()
            )
        )

    def Pi0(self):
        return (
            self.B0**2
            * self.a**4
            * self.n0
            * self.T0
            * self.omega0
            / (self.ReferenceIonGyrofrequency() ** 2 * self.ReferenceIonCollisionTime())
        )

    def qi0(self):
        return (
            self.B0**2
            * self.a**2
            * self.n0
            * self.T0**2
            / (
                self.IonSpecies.IonMass
                * self.ReferenceIonGyrofrequency() ** 2
                * self.ReferenceIonCollisionTime()
            )
        )

    def qe0(self):
        return self.T0 * self.Gamma0()

    # Current normalization (A)
    def I0(self):
        return self.ElementaryCharge * self.n0 * self.a**3 / self.NormalizingTime()

    def NeutralProcess(
        self, CrossSection, vtheta, T, Mass, minEnergy, Moment=MomentType.Density
    ):
        vth2 = 2 * T * self.T0 / Mass
        vtheta /= jnp.sqrt(vth2)

        def IntegrandHermite(x, XS):
            def mDensity(V):
                return jnp.ones(V.shape)

            def mMomentum(V):
                return Mass * (V - vtheta)

            def mEnergy(V):
                return 0.5 * Mass * (V * V - 2 * vtheta * V + vtheta * vtheta)

            # x = jax.lax.switch(Moment, [mDensity, mMomentum, mEnergy], v)

            x1 = x + vtheta
            return XS * (
                jnp.exp(-(vtheta**2))
                * x1**2
                * jnp.sinh(2 * vtheta * x1)
                * jnp.exp(-(vtheta**2 + 2 * vtheta * x))
            )

        def IntegrandLandremann(x, XS):
            def mDensity(V):
                return jnp.ones(V.shape)

            def mMomentum(V):
                return Mass * (V - vtheta)

            def mEnergy(V):
                return 0.5 * Mass * (V * V - 2 * vtheta * V + vtheta * vtheta)

            # x = jax.lax.switch(Moment, [mDensity, mMomentum, mEnergy], v)

            return XS * jnp.exp(-(vtheta**2)) * x**2 * jnp.sinh(2 * vtheta * x)

        x, w = jax.lax.cond(
            jax.lax.le(vtheta, 4.0), lambda: self.land, lambda: self.herm
        )
        Energy = jax.lax.cond(
            jax.lax.le(vtheta, 4.0),
            lambda _x: _x**2 * self.T0eV,
            lambda _x: (_x + vtheta) ** 2 * self.T0eV,
            x,
        )
        XS = CrossSection(Energy) * 1e-4

        integrand = jax.lax.cond(
            jax.lax.le(vtheta, 4.0), IntegrandLandremann, IntegrandHermite, x, XS
        )

        I = jnp.dot(integrand, w)
        integral = 2.0 * jnp.sqrt(vth2) / jnp.sqrt(jnp.pi) / vtheta * I
        return integral

    def IonizationRate(self, n, NeutralDensity, v, Te, Ti):
        n_m3 = n * self.n0
        n_neutrals = NeutralDensity
        IonIntegral = self.NeutralProcess(
            self.IonSpecies.protonImpactIonizationCrossSection,
            v,
            Ti,
            self.IonSpecies.IonMass,
            200.0,
        )

        ElectronIntegral = self.NeutralProcess(
            self.IonSpecies.electronImpactIonizationCrossSection,
            v,
            Te,
            self.ElectronMass,
            13.6,
        )
        R = n_m3 * n_neutrals * (IonIntegral + ElectronIntegral)
        return R

    def ChargeExchangeLossRate(self, n, NeutralDensity, v, Ti):
        n_m3 = n * self.n0
        n_neutrals = NeutralDensity

        R = (
            n_m3
            * n_neutrals
            * self.NeutralProcess(
                self.IonSpecies.hydrogenChargeExchangeCrossSection,
                v,
                Ti,
                self.IonSpecies.IonMass,
                0.1,
            )
        )
        return R
