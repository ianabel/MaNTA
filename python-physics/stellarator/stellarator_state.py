"""Named accessors for a multi-channel stellarator state. Nothing imports this yet.

Brought over from `origin/optimize-mode`, where it is a dependency of that
branch's rewritten `Stellarator2.py` -- the version that evolves density, ion
energy and electron energy together with an ambipolar Er, rather than the single
channel `stellarator2.py` here carries. That rewrite is *not* merged: it is
written against the pre-`SystemSpec` interface (`TransportSystem.__init__(self)`
followed by assigning `self.nVars` and `self.isUpperDirichlet`), which main has
removed, so taking it wholesale would undo the migration.

So this module is here to keep the work from rotting on a stale branch, not
because anything calls it. Two things to know before wiring it up:

  * its `StellaratorParams` is a different class from the one in
    `stellarator2.py`, and deliberately so -- it splits the single
    `SourceCenter`/`SourceHeight`/`SourceWidth` into separate particle and heat
    sources and adds `evolveDensity`. `scan_eq_ambipolar.py` is configured
    against *this* one, which is why that driver cannot run against
    `stellarator2.py` as it stands.
  * `StellaratorDecorator` assumes the hook signature
    `(self, state, x, t, field, vp, vpp, params)`, which is the branch's
    `Stellarator2`, not this directory's.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Float, ArrayLike
import equinox as eqx
import enum
from manta.jax import State


class Channel(enum.IntEnum):
    Density = 0
    IonEnergy = 1
    ElectronEnergy = 2


def StellaratorDecorator(func):
    def wrapper(self, state, x, t, field, vp, vpp, params):
        _state = StellaratorState.from_state(state, x, vp, vpp, params)
        res = func(self, _state, x, t, field, vp, vpp, params)
        return res

    return wrapper


class StellaratorParams(eqx.Module):
    ParticleSourceCenter: float
    ParticleSourceWidth: float
    ParticleSourceHeight: float
    HeatSourceCenter: float
    HeatSourceHeight: float
    HeatSourceWidth: float
    EdgeTemperature: float
    EdgeDensity: float
    n0: float
    evolveDensity: bool

    def __init__(
        self,
        EdgeTemperature,
        EdgeDensity,
        n0,
        HeatSourceCenter,
        HeatSourceHeight,
        HeatSourceWidth,
        ParticleSourceCenter = 0.0,
        ParticleSourceWidth = 0.1,
        ParticleSourceHeight = 1.0,
        evolveDensity=False,
    ):
        self.ParticleSourceCenter = ParticleSourceCenter
        self.ParticleSourceWidth = ParticleSourceWidth
        self.ParticleSourceHeight = ParticleSourceHeight
        self.HeatSourceCenter = HeatSourceCenter
        self.HeatSourceHeight = HeatSourceHeight
        self.HeatSourceWidth = HeatSourceWidth
        self.EdgeTemperature = EdgeTemperature
        self.EdgeDensity = EdgeDensity
        self.n0 = n0
        self.evolveDensity = evolveDensity


"""
Wrapper class for State to make accessing variables easier 
"""


class StellaratorState(eqx.Module):
    n: Float[ArrayLike, "..."]  # Density
    pi: Float[ArrayLike, "..."]  # Ion pressure
    pe: Float[ArrayLike, "..."]  # Electron pressure
    Ti: Float[ArrayLike, "..."]  # Ion temperature
    Te: Float[ArrayLike, "..."]  # Electron temperature
    dndrho: Float[ArrayLike, "..."]
    dpidrho: Float[ArrayLike, "..."]
    dpedrho: Float[ArrayLike, "..."]
    dTidrho: Float[ArrayLike, "..."]
    dTedrho: Float[ArrayLike, "..."]
    gamma: Float[ArrayLike, "..."]
    qi: Float[ArrayLike, "..."]
    qe: Float[ArrayLike, "..."]
    Er: Float[ArrayLike, "..."]  # Ambipolar potential correction
    rho: Float[ArrayLike, "..."]
    vp: Float[ArrayLike, "..."]
    vpp: Float[ArrayLike, "..."]

    def __init__(
        self,
        n: Float[ArrayLike, "..."],
        pi: Float[ArrayLike, "..."],
        pe: Float[ArrayLike, "..."],
        Ti: Float[ArrayLike, "..."],
        Te: Float[ArrayLike, "..."],
        dndrho: Float[ArrayLike, "..."],
        dpidrho: Float[ArrayLike, "..."],
        dpedrho: Float[ArrayLike, "..."],
        dTidrho: Float[ArrayLike, "..."],
        dTedrho: Float[ArrayLike, "..."],
        gamma: Float[ArrayLike, "..."],
        qi: Float[ArrayLike, "..."],
        qe: Float[ArrayLike, "..."],
        Er: Float[ArrayLike, "..."],
        rho: Float[ArrayLike, "..."],
        vp: Float[ArrayLike, "..."],
        vpp: Float[ArrayLike, "..."],
    ):
        self.n = n
        self.pi = pi
        self.pe = pe
        self.Ti = Ti
        self.Te = Te
        self.dndrho = dndrho
        self.dpidrho = dpidrho
        self.dpedrho = dpedrho
        self.dTidrho = dTidrho
        self.dTedrho = dTedrho
        self.gamma = gamma
        self.qi = qi
        self.qe = qe
        self.Er = Er
        self.rho = rho
        self.vp = vp
        self.vpp = vpp

    @classmethod
    def from_state(cls, state: State, x, vp, vpp, params: StellaratorParams):

        def constant_density(state, x, vp, vpp, params):
            n, dndrho = jax.value_and_grad(
                lambda x: StellaratorState.initial_profile(
                    x, params.EdgeDensity, params.n0
                )
            )(x)

            pi = 2.0 / 3.0 * StellaratorState.Vp_u_to_u(0, state, x, vp, vpp)
            dpidrho = 2.0 / 3.0 * StellaratorState.Vp_up_to_up(0, state, x, vp, vpp)

            Ti = pi / n

            dTidrho = (dpidrho - dndrho * Ti) / n
            _zero = jnp.zeros(pi.shape)

            return cls(
                n=n,
                pi=pi,
                pe=pi,
                Ti=Ti,
                Te=Ti,
                dndrho=dndrho,
                dpidrho=dpidrho,
                dpedrho=dpidrho,
                dTidrho=dTidrho,
                dTedrho=dTidrho,
                gamma=_zero,
                qi=state.Flux[0],
                qe=_zero,
                Er=state.Aux,
                rho=x,
                vp=vp,
                vpp=vpp,
            )

        def ambipolar(state, x, vp, vpp, params):
            n = StellaratorState.Vp_u_to_u(Channel.Density, state, x, vp, vpp)
            dndrho = StellaratorState.Vp_up_to_up(Channel.Density, state, x, vp, vpp)

            pi = (
                2.0
                / 3.0
                * StellaratorState.Vp_u_to_u(Channel.IonEnergy, state, x, vp, vpp)
            )
            dpidrho = (
                2.0
                / 3.0
                * StellaratorState.Vp_up_to_up(Channel.IonEnergy, state, x, vp, vpp)
            )
            pe = (
                2.0
                / 3.0
                * StellaratorState.Vp_u_to_u(Channel.ElectronEnergy, state, x, vp, vpp)
            )
            dpedrho = (
                2.0
                / 3.0
                * StellaratorState.Vp_up_to_up(
                    Channel.ElectronEnergy, state, x, vp, vpp
                )
            )

            Ti = pi / n
            Te = pe / n

            dTidrho = (dpidrho - dndrho * Ti) / n
            dTedrho = (dpedrho - dndrho * Te) / n

            return cls(
                n=n,
                pi=pi,
                pe=pe,
                Ti=Ti,
                Te=Te,
                dndrho=dndrho,
                dpidrho=dpidrho,
                dpedrho=dpedrho,
                dTidrho=dTidrho,
                dTedrho=dTedrho,
                gamma=state.Flux[Channel.Density],
                qi=state.Flux[Channel.IonEnergy],
                qe=state.Flux[Channel.ElectronEnergy],
                Er=state.Aux,
                rho=x,
                vp=vp,
                vpp=vpp,
            )

        return jax.lax.cond(
            params.evolveDensity, ambipolar, constant_density, state, x, vp, vpp, params
        )

    @staticmethod
    def Vp_u_to_u(index, s, x, vp, vpp):
        return jax.lax.cond(
            jax.lax.eq(x, 0.0),
            lambda state: state.Derivative[index] / vpp,
            lambda state: state.Variable[index] / vp,
            s,
        )

    @staticmethod
    def Vp_up_to_up(index, s, x, vp, vpp):
        return jax.lax.cond(
            jax.lax.eq(x, 0.0),
            lambda state: 0.0,
            lambda state: (
                (state.Derivative[index] * vp - vpp * state.Variable[index]) / vp**2
            ),
            s,
        )

    @staticmethod
    def initial_profile(x, edge_value, peak_value):
        return (peak_value - edge_value) * (1 - x**4) + edge_value

    #
    # @staticmethod
    # def vmap_axes():
    #     return StellaratorState(
    #         n=0,
    #         pi=0,
    #         pe=0,
    #         Ti=0,
    #         Te=0,
    #         dndrho=0,
    #         dpidrho=0,
    #         dpedrho=0,
    #         dTidrho=0,
    #         dTedrho=0,
    #         gamma=0,
    #         qi=0,
    #         qe=0,
    #         Er=0,
    #         rho=0,
    #         vp=0,
    #         vpp=0,
    #     )
