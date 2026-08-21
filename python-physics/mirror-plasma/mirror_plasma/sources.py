import jax
import jax.numpy as jnp
from collections.abc import Callable

from .plasma_state import (
    MirrorPlasmaParams,
    MirrorPlasmaState,
    Channel,
)


from .parallel_physics import (
    ElectronPastukhovLossRate,
    IonPastukhovLossRate,
    Xi_i,
    Xi_e,
)


# simple class to hold a source, saving the name for analysis
class _source_base:
    method: Callable
    name: str

    def __init__(self, func, name):
        self.method = func
        self.name = name

    def __call__(self, state, x, t, params):
        return self.method(state, x, t, params)


# register sources on input as either sources or sinks, specifying for a given Channel
source_registry = [[], [], [], []]
sink_registry = [[], [], [], []]


def register_source(channel: Channel):
    def decorator(func):
        source_registry[channel].append(_source_base(func, func.__name__))
        return func

    return decorator


def register_sink(channel: Channel):
    def decorator(func):
        sink_registry[channel].append(_source_base(func, func.__name__))
        return func

    return decorator


# ======================================================================= #
# Particle Sources                                                        #
# ======================================================================= #


@register_source(Channel.Density)
def ParticleSource(state: MirrorPlasmaState, x, t, params: MirrorPlasmaParams):
    Center = params.Config.ParticleSourceCenter / params.Constants.a
    Width = params.Config.ParticleSourceWidth / params.Constants.a
    Height = params.Config.ParticleSourceHeight * params.Constants.a**2
    return Height * jnp.exp(-(((state.R - Center) / Width) ** 2)) * jnp.exp(-t / 0.01)


@register_source(Channel.Density)
def IonizationSource(state, x, t, params):
    return (
        params.Constants.IonizationRate(
            state.n,
            params.Config.NeutralDensity,
            state.R * params.Constants.a * state.omega * params.Constants.omega0,
            state.Te,
            state.Ti,
        )
        / params.Constants.DensityEquationNormalization()
    )


@register_sink(Channel.Density)
def ParallelParticleLosses(state, x, t, params):
    return (
        ElectronPastukhovLossRate(state, x, t, params)
        / params.Constants.DensityEquationNormalization()
    )


@register_sink(Channel.Density)
def FusionParticleLosses(state, x, t, params):
    return (
        params.Constants.FusionRate(state.n, state.pi)
        / params.Constants.DensityEquationNormalization()
    )


# ======================================================================= #
# Momentum Sources                                                        #
# ======================================================================= #


@register_sink(Channel.AngularMomentum)
def ParallelAngularMomentumLosses(state, x, t, params):
    return (
        state.omega
        * state.R**2
        * IonPastukhovLossRate(state, x, t, params)
        * (
            params.Constants.IonSpecies.IonMass
            * params.Constants.omega0
            * params.Constants.a**2
        )
        / params.Constants.MomentumEquationNormalization()
    )


@register_sink(Channel.AngularMomentum)
def ChargeExchangeMomentumLosses(state, x, t, params):

    def true_fun():
        return (
            state.omega
            * state.R**2
            * params.Constants.ChargeExchangeLossRate(
                state.n,
                params.Config.NeutralDensity,
                state.R * params.Constants.a * state.omega * params.Constants.omega0,
                state.Ti,
            )
            * (
                params.Constants.IonSpecies.IonMass
                * params.Constants.omega0
                * params.Constants.a**2
            )
            / params.Constants.MomentumEquationNormalization()
        )

    def false_fun():
        return 0.0

    return jax.lax.cond(params.Config.useNeutralsModel, true_fun, false_fun)


@register_source(Channel.AngularMomentum)
def JxBForce(state, x, t, params):
    return (
        state.Current * params.Constants.I0() / state.VPrime
    ) / params.Constants.MomentumEquationNormalization()


# ======================================================================= #
# Heat sources                                                            #
# ======================================================================= #


# Decaying uniform source to help solution along
@register_source(Channel.IonEnergy)
@register_source(Channel.ElectronEnergy)
def UniformHeatSource(state: MirrorPlasmaState, x, t, params: MirrorPlasmaParams):
    return params.Constants.a**2 * 200.0 * jnp.exp(-t / 0.01)


@register_source(Channel.IonEnergy)
@register_sink(Channel.ElectronEnergy)
def EnergyExchange(state: MirrorPlasmaState, x, t, params: MirrorPlasmaParams):
    return params.Constants.IonElectronEnergyExchange(state.n, state.pe, state.pi)


"""
Ion heat sources
"""


@register_source(Channel.IonEnergy)
def ViscousHeating(state: MirrorPlasmaState, x, t, params: MirrorPlasmaParams):
    return (
        -1
        * (state.domegadpsi * state.Pi / state.VPrime)
        * (params.Constants.omega0 * params.Constants.MomentumEquationNormalization())
        / params.Constants.HeatEquationNormalization()
    )


@register_source(Channel.IonEnergy)
def IonPotentialHeating(state: MirrorPlasmaState, x, t, params: MirrorPlasmaParams):
    return (
        -0.5
        * params.Constants.IonSpecies.IonMass
        * (params.Constants.a * params.Constants.omega0) ** 2
        * (state.R * state.omega) ** 2
        * (
            ParticleSource(state, x, t, params)
            + IonizationSource(state, x, t, params)
            - ParallelParticleLosses(state, x, t, params)
        )
        * params.Constants.DensityEquationNormalization()
    ) / params.Constants.HeatEquationNormalization()


@register_sink(Channel.IonEnergy)
def ChargeExchangeHeatLosses(
    state: MirrorPlasmaState, x, t, params: MirrorPlasmaParams
):
    def true_fun():
        return (
            state.Ti
            * params.Constants.T0
            * params.Constants.ChargeExchangeLossRate(
                state.n,
                params.Config.NeutralDensity,
                state.R * params.Constants.a * state.omega * params.Constants.omega0,
                state.Ti,
            )
            / params.Constants.HeatEquationNormalization()
        )

    def false_fun():
        return 0.0

    return jax.lax.cond(params.Config.useNeutralsModel, true_fun, false_fun)


@register_sink(Channel.IonEnergy)
def IonParallelHeatLosses(state: MirrorPlasmaState, x, t, params: MirrorPlasmaParams):
    ParticleEnergy = state.Ti * (1 + Xi_i(state, x, t, params))
    return (
        ParticleEnergy
        * params.Constants.T0
        * IonPastukhovLossRate(state, x, t, params)
        / params.Constants.HeatEquationNormalization()
    )


"""
Electron heat sources
"""


@register_sink(Channel.ElectronEnergy)
def RadiationHeatLosses(state: MirrorPlasmaState, x, t, params: MirrorPlasmaParams):
    return params.Constants.BremsstrahlungLosses(
        state.n, state.pe
    ) + params.Constants.CyclotronLosses(x, state.n, state.Te)


@register_source(Channel.ElectronEnergy)
def AlphaHeating(state: MirrorPlasmaState, x, t, params: MirrorPlasmaParams):
    return params.Constants.TotalAlphaPower(state.n, state.pi)


@register_sink(Channel.ElectronEnergy)
def ElectronParallelHeatLosses(
    state: MirrorPlasmaState, x, t, params: MirrorPlasmaParams
):
    ParticleEnergy = state.Te * (1 + Xi_e(state, x, t, params))
    return (
        ParticleEnergy
        * params.Constants.T0
        * ElectronPastukhovLossRate(state, x, t, params)
        / params.Constants.HeatEquationNormalization()
    )
