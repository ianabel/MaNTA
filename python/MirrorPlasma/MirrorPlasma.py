import jax
import equinox as eqx
from numpy.random import geometric
from PlasmaState import (
    MirrorPlasmaDecorator,
    MirrorPlasmaParams,
    MirrorPlasmaState,
)
import sys

from python.MirrorPlasma import Constants

sys.path.append("..")
from VectorizedTransportSystem import VectorizedTransportSystem


class MirrorPlasma(VectorizedTransportSystem):
    def __init__(self):
        pass

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
        return G

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

        Pi_out = IonClassicalViscosity + state.omega * R**2 * self.Gamma(
            state, x, t, params
        )
        return Pi_out

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
        qi_out = HeatFlux - 0.5 * (state.omega * R) ** 2 * self.Gamma(
            state, x, t, params
        )
        return qi_out

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
        return HeatFlux

    def Sn(self, state: MirrorPlasmaState, x, t, params: MirrorPlasmaParams):
        pass

    def Somega(self, state: MirrorPlasmaState, x, t, params: MirrorPlasmaParams):
        pass

    def Spi(self, state: MirrorPlasmaState, x, t, params: MirrorPlasmaParams):
        pass

    def Spe(self, state: MirrorPlasmaState, x, t, params: MirrorPlasmaParams):
        pass
