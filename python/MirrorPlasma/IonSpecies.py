import equinox as eqx
import jax.numpy as jnp
from jax.typing import Float
from abc import abstractmethod
from typing import override

ElectronMass = 9.1094e-31
ProtonMass = 1.6726e-27
ElementaryCharge = 1.60217663e-19
VacuumPermittivity = 8.8541878128e-12


class _IonSpecies(eqx.Module):
    IonMass: Float

    @abstractmethod
    def FusionRate(self, n, pi):
        return jnp.zeros(n.shape)

    @abstractmethod
    def electronImpactIonizationCrossSection(self, Energy):
        raise NotImplementedError(
            "electronImpactIonizationCrossSection not implemented in IonSpecies"
        )

    @abstractmethod
    def protonImpactIonizationCrossSection(self, Energy):
        raise NotImplementedError(
            "protonImpactIonizationCrossSection not implemented in IonSpecies"
        )

    @abstractmethod
    def hydrogenChargeExchangeCrossSection(self, Energy):
        raise NotImplementedError(
            "hydrogenChargeExchangeCrossSection not implemented in IonSpecies"
        )


class Hydrogen(_IonSpecies):
    def __init__(self):
        self.IonMass = 1.0 * ProtonMass

    @override
    def electronImpactIonizationCrossSection(self, Energy):
        ionizationEnergy = 13.6
        minimumEnergySigma = ionizationEnergy
        # Contribution from ground state
        # Janev 1993, ATOMIC AND PLASMA-MATERIAL INTERACTION DATA FOR FUSION,
        # Volume 4 Equation 1.2.1 e + H(1s) --> e + H+ + e Accuracy is 10% or
        # better
        fittingParamA = 0.18450
        fittingParamB = jnp.array([-0.032226, -0.034539, 1.4003, -2.8115, 2.2986])

        if Energy < minimumEnergySigma:
            return 0.0
        else:
            sum = 0.0
            x = 1.0 - ionizationEnergy / Energy
            if x <= 0:
                return 0.0

            sum = x * (
                fittingParamB[0]
                + x
                * (
                    fittingParamB[1]
                    + x
                    * (fittingParamB[2] + x * (fittingParamB[3] + x * fittingParamB[4]))
                )
            )
            sigma = (1.0e-13 / (ionizationEnergy * Energy)) * (
                fittingParamA * jnp.log(Energy / ionizationEnergy) + sum
            )
            return sigma

    @override
    def protonImpactIonizationCrossSection(self, Energy):
        # Minimum energy of cross section in keV
        minimumEnergySigma = 0.5
        # Convert to keV
        EnergyKEV = Energy / 1000

        # Contribution from ground state
        # Janev 1993, ATOMIC AND PLASMA-MATERIAL INTERACTION DATA FOR FUSION,
        # Volume 4 Equation 2.2.1 H+ + H(1s) --> H+ + H+ + e Accuracy is 30% or
        # better
        A1 = 12.899
        A2 = 61.897
        A3 = 9.2731e3
        A4 = 4.9749e-4
        A5 = 3.9890e-2
        A6 = -1.5900
        A7 = 3.1834
        A8 = -3.7154

        if EnergyKEV < minimumEnergySigma:
            return 0.0
        else:
            # Energy is in units of keV
            sigma = (
                1e-16
                * A1
                * (
                    jnp.exp(-A2 / EnergyKEV) * jnp.log(1 + A3 * EnergyKEV) / EnergyKEV
                    + A4
                    * jnp.exp(-A5 * EnergyKEV)
                    / (jnp.pow(EnergyKEV, A6) + A7 * jnp.pow(EnergyKEV, A8))
                )
            )
            return sigma

    @override
    def hydrogenChargeExchangeCrossSection(self, Energy):

        # Minimum energy of cross section in eV
        minimumEnergySigma_n1 = 0.12
        # Contribution from ground -> ground state
        # Janev 1993 2.3.1
        # p + H(n=1) --> H + p
        if Energy < minimumEnergySigma_n1:
            return jnp.zeros(Energy.shape)
        else:
            EnergyKEV = Energy / 1000
            sigma_n1 = (
                1e-16
                * 3.2345
                * jnp.log(235.88 / EnergyKEV + 2.3713)
                / (
                    1
                    + 0.038371 * EnergyKEV
                    + 3.8068e-6 * jnp.pow(EnergyKEV, 3.5)
                    + 1.1832e-10 * jnp.pow(EnergyKEV, 5.4)
                )
            )
            return sigma_n1
