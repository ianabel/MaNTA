import equinox as eqx
import jax.numpy as jnp
from Constants import PlasmaConstants
from MagneticField import StraightMagneticField
from IonSpecies import Hydrogen
import sys

from jaxtyping import Float, ArrayLike
import enum
import sys

sys.path.append("..")

from State import State


class MirrorPlasmaParams(eqx.Module):
    MagneticField: StraightMagneticField
    IonSpecies: Hydrogen
    Constants: PlasmaConstants


def MirrorPlasmaDecorator(func):
    def wrapper(self, index, state, x, t, params):
        _state = MirrorPlasmaState.from_state(state, x, params)
        res = func(self, index, _state, x, t, params)
        if isinstance(res, MirrorPlasmaState):
            return res.to_state()
        else:
            return res

    return wrapper


class Channel(enum.IntEnum):
    Density = 0
    AngularMomentum = 1
    IonEnergy = 2
    ElectronEnergy = 3


class MirrorPlasmaState(eqx.Module):
    n: Float[ArrayLike, "..."]
    pi: Float[ArrayLike, "..."]
    pe: Float[ArrayLike, "..."]
    L: Float[ArrayLike, "..."]
    omega: Float[ArrayLike, "..."]
    Ti: Float[ArrayLike, "..."]
    Te: Float[ArrayLike, "..."]
    dndx: Float[ArrayLike, "..."]
    dpidx: Float[ArrayLike, "..."]
    dpedx: Float[ArrayLike, "..."]
    dLdx: Float[ArrayLike, "..."]
    domegadx: Float[ArrayLike, "..."]
    dTidx: Float[ArrayLike, "..."]
    dTedx: Float[ArrayLike, "..."]
    gamma: Float[ArrayLike, "..."]
    Pi: Float[ArrayLike, "..."]
    qi: Float[ArrayLike, "..."]
    qe: Float[ArrayLike, "..."]
    phi: Float[ArrayLike, "..."]
    Scalars: Float[ArrayLike, "..."]

    def __init__(
        self,
        n: Float[ArrayLike, "..."],
        pi: Float[ArrayLike, "..."],
        pe: Float[ArrayLike, "..."],
        L: Float[ArrayLike, "..."],
        omega: Float[ArrayLike, "..."],
        Ti: Float[ArrayLike, "..."],
        Te: Float[ArrayLike, "..."],
        dndx: Float[ArrayLike, "..."],
        dpidx: Float[ArrayLike, "..."],
        dpedx: Float[ArrayLike, "..."],
        dLdx: Float[ArrayLike, "..."],
        domegadx: Float[ArrayLike, "..."],
        dTidx: Float[ArrayLike, "..."],
        dTedx: Float[ArrayLike, "..."],
        gamma: Float[ArrayLike, "..."],
        Pi: Float[ArrayLike, "..."],
        qi: Float[ArrayLike, "..."],
        qe: Float[ArrayLike, "..."],
        phi: Float[ArrayLike, "..."],
        Scalars: Float[ArrayLike, "..."],
    ):
        self.n = n
        self.pi = pi
        self.pe = pe
        self.L = L
        self.omega = omega
        self.Ti = Ti
        self.Te = Te
        self.dndx = dndx
        self.dpidx = dpidx
        self.dpedx = dpedx
        self.dLdx = dLdx
        self.domegadx = domegadx
        self.dTidx = dTidx
        self.dTedx = dTedx
        self.gamma = gamma
        self.Pi = Pi
        self.qi = qi
        self.qe = qe
        self.phi = phi
        self.Scalars = Scalars

    @classmethod
    def from_state(cls, state: State, x, params: MirrorPlasmaParams):
        n = state.Variable[Channel.Density]
        L = state.Variable[Channel.AngularMomentum]
        pi = 2.0 / 3.0 * state.Variable[Channel.IonEnergy]
        pe = 2.0 / 3.0 * state.Variable[Channel.ElectronEnergy]

        Ti = pi / n
        Te = pe / n

        R = params.MagneticField.B(x)
        J = n * R**2
        omega = L / J

        dndx = state.Derivative[Channel.Density]
        dLdx = state.Derivative[Channel.AngularMomentum]
        dpidx = 2.0 / 3.0 * state.Derivative[Channel.IonEnergy]
        dpedx = 2.0 / 3.0 * state.Derivative[Channel.ElectronEnergy]

        dTidx = (dpidx - dndx * Ti) / n
        dTedx = (dpedx - dndx * Te) / n

        dRdx = params.MagneticField.dRdx(x)
        dJdx = R * R * dndx + 2.0 * dRdx * R * n
        domegadx = dLdx / J - dJdx * L / (J * J)

        return cls(
            n,
            pi,
            pe,
            L,
            omega,
            Ti,
            Te,
            dndx,
            dpidx,
            dpedx,
            dLdx,
            domegadx,
            dTidx,
            dTedx,
            state.Flux[Channel.Density],
            state.Flux[Channel.AngularMomentum],
            state.Flux[Channel.IonEnergy],
            state.Flux[Channel.ElectronEnergy],
            state.Aux,
            state.Scalars,
        )

    def to_state(self):
        Variable = jnp.array([self.n, self.L, 3.0 / 2.0 * self.pi, 3.0 / 2.0 * self.pe])
        Derivative = jnp.array(
            [self.dndx, self.dLdx, 3.0 / 2.0 * self.dpidx, 3.0 / 2.0 * self.dpedx]
        )
        Flux = jnp.array([self.gamma, self.Pi, self.qi, self.qe])
        return State(Variable, Derivative, Flux, self.phi, self.Scalars)
