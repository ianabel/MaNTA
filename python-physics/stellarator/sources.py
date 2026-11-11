"""The sources of the multi-channel stellarator case, as pure functions.

Each term is a function `(state, x, t, vp, params)` of a `StellaratorState`,
registered against the channel it feeds as a source or as a sink.
`compute_sources` sums each channel's sources, subtracts its sinks and
multiplies by V', since the evolved variables are V' times a density.

Nothing here reads a `StellaratorTransport`, and that is the point of the
file: `compute_sources` is handed to `filter_jit`, which keys its cache on the
callable. A module-level function is the same object for every case, so it
compiles once per process; a bound method is a new key per case, which
compiled at every objective evaluation and kept each case alive in the cache.
A term that needs something a case holds takes it through `params`.
"""

from collections.abc import Callable

import jax.numpy as jnp

from config import StellaratorParams
from stellarator_state import Channel, StellaratorDecorator, StellaratorState


class _source_base:
    """A registered term, with its name kept for analysis."""

    method: Callable
    name: str

    def __init__(self, func, name):
        self.method = func
        self.name = name

    def __call__(self, state, x, t, vp, params):
        return self.method(state, x, t, vp, params)


_channels = (Channel.Density, Channel.IonEnergy, Channel.ElectronEnergy)
source_registry = {channel: [] for channel in _channels}
sink_registry = {channel: [] for channel in _channels}


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


def _gaussian(x, height, center, width):
    return height * jnp.exp(-((x - center) ** 2) / (2 * width**2))


# ======================================================================= #
# Particle sources                                                        #
# ======================================================================= #


@register_source(Channel.Density)
def ParticleSource(state: StellaratorState, x, t, vp, params: StellaratorParams):
    return _gaussian(
        x,
        params.config.ParticleSourceHeight,
        params.config.ParticleSourceCenter,
        params.config.ParticleSourceWidth,
    )


@register_sink(Channel.Density)
def FusionParticleLosses(state, x, t, vp, params):
    return (
        params.constants.FusionRate(state.n, state.Ti)
        / params.constants.DensityEquationNormalization()
    )


# ======================================================================= #
# Heat sources                                                            #
# ======================================================================= #


@register_source(Channel.IonEnergy)
def NBIHeating(state, x, t, vp, params):
    return _gaussian(
        x, params.config.NBIPower, params.config.NBICenter, params.config.NBIWidth
    )


@register_source(Channel.ElectronEnergy)
def ECHHeating(state, x, t, vp, params):
    return _gaussian(
        x, params.config.ECHPower, params.config.ECHCenter, params.config.ECHWidth
    )


# electrons to ions: a source for the ions and the same sink for the electrons
@register_source(Channel.IonEnergy)
@register_sink(Channel.ElectronEnergy)
def CollisionalEnergyExchange(state, x, t, vp, params):
    return params.constants.IonElectronEnergyExchange(state.n, state.pe, state.pi)


@register_source(Channel.ElectronEnergy)
def AlphaHeating(state, x, t, vp, params):
    return (
        params.config.FusionFactor
        * params.constants.AlphaHeating(state.n, state.Ti)
        / params.constants.HeatEquationNormalization()
    )


# ======================================================================= #
# The loop                                                                #
# ======================================================================= #


def channel_source(channel: Channel, state: StellaratorState, x, t, vp, params):
    """V' (sources - sinks) for one channel."""
    total = sum(s(state, x, t, vp, params) for s in source_registry[channel])
    total -= sum(s(state, x, t, vp, params) for s in sink_registry[channel])
    return vp * total


@StellaratorDecorator
def compute_sources(
    state: StellaratorState, x, t, field, vp, vpp, params: StellaratorParams
):
    """Every evolved channel's source, in variable order: density, ion energy
    and electron energy with `evolveDensity`, the ion energy alone without."""
    if params.config.evolveDensity:
        channels = _channels
    else:
        channels = (Channel.IonEnergy,)
    return [channel_source(c, state, x, t, vp, params) for c in channels]
