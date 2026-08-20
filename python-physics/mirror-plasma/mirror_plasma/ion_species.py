import equinox as eqx
import jax.numpy as jnp
import jax
from jaxtyping import Float
from abc import abstractmethod
from typing import override

ElectronMass = 9.1094e-31
ProtonMass = 1.6726e-27
ElementaryCharge = 1.60217663e-19
VacuumPermittivity = 8.8541878128e-12


class _IonSpecies(eqx.Module):
    IonMass: Float

    def FusionRate(self, n, Ti):
        return 0.0

    """
        Electron ionization cross section, energy in ev
    """

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

        x = 1.0 - ionizationEnergy / Energy

        def _compute(x):
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

        return jnp.where(jax.lax.le(Energy, minimumEnergySigma), 0.0, _compute(x))

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

        def _compute(E):
            # Energy is in units of keV
            sigma = (
                1e-16
                * A1
                * (
                    jnp.exp(-A2 / E) * jnp.log(1 + A3 * E) / E
                    + A4 * jnp.exp(-A5 * E) / (jnp.pow(E, A6) + A7 * jnp.pow(E, A8))
                )
            )
            return sigma

        return jnp.where(
            jax.lax.le(EnergyKEV, minimumEnergySigma), 0.0, _compute(EnergyKEV)
        )

    @override
    def hydrogenChargeExchangeCrossSection(self, Energy):

        # Minimum energy of cross section in eV
        minimumEnergySigma_n1 = 0.12

        # Contribution from ground -> ground state
        # Janev 1993 2.3.1
        # p + H(n=1) --> H + p
        def _compute(Energy):
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

        return jnp.where(
            jax.lax.le(Energy, minimumEnergySigma_n1), 0.0, _compute(Energy)
        )


class DeuteriumTritium(_IonSpecies):
    def __init__(self):
        self.IonMass = 2.5 * ProtonMass

    @override
    def FusionRate(self, n, Ti):

        # c.f. H.-S. Bosch and G.M. Hale 1992 Nucl. Fusion 32 611
        C1 = 5.65718e-12
        C2 = 3.41267e-3
        C3 = 1.99167e-3
        C4 = 0.0
        C5 = 1.05060e-5
        C6 = 0.0
        C7 = 0.0
        BG = 31.3970
        mrc2 = 937814

        theta = Ti / (
            1
            - (Ti * (C2 + Ti * (C4 + Ti * C6))) / (1 + Ti * (C3 + Ti * (C5 + Ti * C7)))
        )

        xi = (BG**2 / (4 * theta)) ** (1.0 / 3.0)

        sigmav = C1 * theta * jnp.sqrt(xi / (mrc2 * Ti**3)) * jnp.exp(-3 * xi)

        return 0.25 * n**2 * sigmav

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

        x = 1.0 - ionizationEnergy / Energy

        def _compute(x):
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

        return jnp.where(jax.lax.le(Energy, minimumEnergySigma), 0.0, _compute(x))

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

        def _compute(E):
            # Energy is in units of keV
            sigma = (
                1e-16
                * A1
                * (
                    jnp.exp(-A2 / E) * jnp.log(1 + A3 * E) / E
                    + A4 * jnp.exp(-A5 * E) / (jnp.pow(E, A6) + A7 * jnp.pow(E, A8))
                )
            )
            return sigma

        return jnp.where(
            jax.lax.le(EnergyKEV, minimumEnergySigma), 0.0, _compute(EnergyKEV)
        )

    @override
    def hydrogenChargeExchangeCrossSection(self, Energy):

        # Minimum energy of cross section in eV
        minimumEnergySigma_n1 = 0.12

        # Contribution from ground -> ground state
        # Janev 1993 2.3.1
        # p + H(n=1) --> H + p
        def _compute(Energy):
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

        return jnp.where(
            jax.lax.le(Energy, minimumEnergySigma_n1), 0.0, _compute(Energy)
        )
