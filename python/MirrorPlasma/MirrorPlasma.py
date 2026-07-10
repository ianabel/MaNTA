import jax
import jax.numpy as jnp
import equinox as eqx
from numpy.random import geometric
from PlasmaState import (
    MirrorPlasmaDecorator,
    MirrorPlasmaParams,
    MirrorPlasmaState,
    MirrorPlasmaConfig,
)
import sys

sys.path.append("..")
from VectorizedTransportSystem import VectorizedTransportSystem


class MirrorPlasma(VectorizedTransportSystem):
    def __init__(self, config: MirrorPlasmaConfig, solver_config=None):
        super().__init__()
        self.params = MirrorPlasmaParams.make(config)

    def InitialValue(self, index, x):
        Rmin = self.params.Config.Rmin
        Rmax = self.params.Config.Rmax
        Rmid = 0.5 * (Rmin + Rmax)
        R = self.params.MagneticField.R_x(x)
        v = jnp.cos(jnp.pi * (R - Rmid) / (Rmax - Rmin))

        def n0():
            return (
                self.params.Config.EdgeDensity
                + (
                    self.params.Config.InitialDensityHeight
                    - self.params.Config.EdgeDensity
                )
                * v**2
            )

        def ui0():
            return (
                3.0
                / 2.0
                * (
                    self.params.Config.EdgeIonTemperature
                    + (
                        self.params.Config.InitialIonTemperatureHeight
                        - self.params.Config.EdgeIonTemperature
                    )
                    * v
                )
                * n0()
            )

        def ue0():
            return (
                3.0
                / 2.0
                * (
                    self.params.Config.EdgeElectronTemperature
                    + (
                        self.params.Config.InitialElectronTemperatureHeight
                        - self.params.Config.EdgeElectronTemperature
                    )
                    * v
                )
                * n0()
            )

        def L0():
            Te = 2.0 / 3.0 * ue0() / n0()
            M0 = (
                self.params.Config.EdgeMachNumber
                + (
                    self.params.Config.InitialMachNumber
                    - self.params.Config.EdgeMachNumber
                )
                * v
            )
            omega = jnp.sqrt(Te) * M0 / R

            return omega * R * R * n0()

        return jax.lax.switch(index, [n0, L0, ui0, ue0])

    @MirrorPlasmaDecorator
    def sigma(self, index, state, x, t, params):
        return jax.lax.switch(
            index, [self.Gamma, self.Pi, self.qi, self.qe], state, x, t, params
        )

    @MirrorPlasmaDecorator
    def source(self, index, state, x, t, params):
        return jax.lax.switch(
            index, [self.Sn, self.Somega, self.Spi, self.Spe], state, x, t, params
        )

    @MirrorPlasmaDecorator
    def aux(self, index, state, x, t, params):
        pass

    def Gamma(self, state: MirrorPlasmaState, x, t, params: MirrorPlasmaParams):
        R = params.MagneticField.R_x(x)
        VPrime = params.MagneticField.VPrime(x)
        GeometricFactor = (R * VPrime) ** 2

        Uei = (
            state.dpedx / state.pe
            + state.Ti / (state.Te * params.Constants.Z_eff)
            + (state.omega * R * R) / state.Ti * state.domegadx
        )

        G = (
            GeometricFactor
            * state.pe
            / params.Constants.ElectronCollisionTime(state.n, state.Te)
            * (Uei - 3.0 / 2.0 * state.dTedx / state.Te)
        )
        return (
            G
            * params.Constants.Gamma0()
            / params.Constants.DensityEquationNormalization()
        )

    def Pi(self, state: MirrorPlasmaState, x, t, params: MirrorPlasmaParams):
        R = params.MagneticField.R_x(x)
        VPrime = params.MagneticField.VPrime(x)
        GeometricFactor = (R * VPrime**2) ** 2

        IonClassicalViscosity = (
            GeometricFactor
            * 3.0
            / 10.0
            * state.pi
            / params.Constants.IonCollisionTime(state.n, state.Ti)
            * state.domegadx
        )

        Pi_out = (
            params.Constants.Pi0() * IonClassicalViscosity
            + (
                params.IonSpecies.IonMass
                * params.Constants.omega0
                * params.Constants.a**2
            )
            * state.omega
            * R**2
            * self.Gamma(state, x, t, params)
            * params.Constants.DensityEquationNormalization()
        )
        return Pi_out / params.Constants.MomentumEquationNormalization()

    def qi(self, state: MirrorPlasmaState, x, t, params: MirrorPlasmaParams):
        R = params.MagneticField.R_x(x)
        VPrime = params.MagneticField.VPrime(x)
        GeometricFactor = (R * VPrime) ** 2

        HeatFlux = (
            2
            * GeometricFactor
            * state.pi
            * state.Ti
            / params.Constants.IonCollisionTime(state.n, state.Ti)
            * state.dTidx
            / state.Ti
        )
        qi_out = (
            params.Constants.qi0() * HeatFlux
            - (params.IonSpecies.IonMass * params.Constants.omega0**2)
            * 0.5
            * (state.omega * R) ** 2
            * self.Gamma(state, x, t, params)
            * params.Constants.DensityEquationNormalization()
        )
        return qi_out / params.Constants.HeatEquationNormalization()

    def qe(self, state: MirrorPlasmaState, x, t, params: MirrorPlasmaParams):
        R = params.MagneticField.R_x(x)
        VPrime = params.MagneticField.VPrime(x)
        GeometricFactor = (R * VPrime) ** 2
        Uei = (
            state.dpedx / state.pe
            + state.Ti / (state.Te * params.Constants.Z_eff)
            + (state.omega * R * R) / state.Ti * state.domegadx
        )

        HeatFlux = (
            GeometricFactor
            * state.pe
            * state.Te
            / params.Constants.ElectronCollisionTime(state.n, state.Te)
            * (4.66 * state.dTedx / state.Te - Uei)
        )
        return (
            params.Constants.qe0()
            / params.Constants.HeatEquationNormalization()
            * HeatFlux
        )

    def Sn(self, state: MirrorPlasmaState, x, t, params: MirrorPlasmaParams):
        pass

    def Somega(self, state: MirrorPlasmaState, x, t, params: MirrorPlasmaParams):
        pass

    def Spi(self, state: MirrorPlasmaState, x, t, params: MirrorPlasmaParams):
        pass

    def Spe(self, state: MirrorPlasmaState, x, t, params: MirrorPlasmaParams):
        pass
