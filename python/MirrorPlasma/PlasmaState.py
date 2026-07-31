import equinox as eqx
import jax
import jax.numpy as jnp
from MirrorPlasma.Constants import PlasmaConstants
from MirrorPlasma.MagneticField import _MagneticField
from MirrorPlasma.IonSpecies import _IonSpecies
from MirrorPlasma.Config import MirrorPlasmaConfig

from jaxtyping import Float, ArrayLike, Bool
import enum
import sys


sys.path.append("..")

from State import State


class MirrorPlasmaParams(eqx.Module):
    MagneticField: _MagneticField
    IonSpecies: _IonSpecies
    Constants: PlasmaConstants
    Config: MirrorPlasmaConfig

    def __init__(self, MagneticField, IonSpecies, Constants, Config):
        self.MagneticField = MagneticField
        self.IonSpecies = IonSpecies
        self.Constants = Constants
        self.Config = Config

    @classmethod
    def make(cls, config: MirrorPlasmaConfig):
        if config.NormalizeToR:
            a = config.Rmax - config.Rmin
        else:
            a = 1.0
        B = config.MagneticField
        P = config.IonSpecies

        C = PlasmaConstants(P, B, _a=a)
        return cls(MagneticField=B, IonSpecies=P, Constants=C, Config=config)


def MirrorPlasmaDecorator(func):
    def wrapper(self, index, state, x, t, params):
        _state = MirrorPlasmaState.from_state(state, x, params)
        res = func(self, index, _state, x, t, params)
        return res

    return wrapper


class Channel(enum.IntEnum):
    Density = 0
    AngularMomentum = 1
    IonEnergy = 2
    ElectronEnergy = 3


class Scalar(enum.IntEnum):
    Error = 0
    Integral = 1
    Current = 2


"""
Wrapper class for State to make accessing variables easier 
All variables are normalized 
"""


class MirrorPlasmaState(eqx.Module):
    n: Float[ArrayLike, "..."]  # Density
    pi: Float[ArrayLike, "..."]  # Ion pressure
    pe: Float[ArrayLike, "..."]  # Electron pressure
    L: Float[ArrayLike, "..."]  # Angular momentum density
    omega: Float[ArrayLike, "..."]  # Angular frequency
    Ti: Float[ArrayLike, "..."]  # Ion temperature
    Te: Float[ArrayLike, "..."]  # Electron temperature
    M: Float[ArrayLike, "..."]  # Mach number
    dndpsi: Float[ArrayLike, "..."]
    dpidpsi: Float[ArrayLike, "..."]
    dpedpsi: Float[ArrayLike, "..."]
    dLdpsi: Float[ArrayLike, "..."]
    domegadpsi: Float[ArrayLike, "..."]
    dTidpsi: Float[ArrayLike, "..."]
    dTedpsi: Float[ArrayLike, "..."]
    gamma: Float[ArrayLike, "..."]  # Particle flux
    Pi: Float[ArrayLike, "..."]  # Viscous stress
    qi: Float[ArrayLike, "..."]  # Ion heat flux
    qe: Float[ArrayLike, "..."]  # Electron heat flux
    phi: Float[ArrayLike, "..."]  # Ambipolar potential correction
    Current: Float
    Scalars: Float[ArrayLike, "..."]
    R: Float[ArrayLike, "..."]
    VPrime: Float[ArrayLike, "..."]

    def __init__(
        self,
        n: Float[ArrayLike, "..."],
        pi: Float[ArrayLike, "..."],
        pe: Float[ArrayLike, "..."],
        L: Float[ArrayLike, "..."],
        omega: Float[ArrayLike, "..."],
        Ti: Float[ArrayLike, "..."],
        Te: Float[ArrayLike, "..."],
        M: Float[ArrayLike, "..."],
        dndpsi: Float[ArrayLike, "..."],
        dpidpsi: Float[ArrayLike, "..."],
        dpedpsi: Float[ArrayLike, "..."],
        dLdpsi: Float[ArrayLike, "..."],
        domegadpsi: Float[ArrayLike, "..."],
        dTidpsi: Float[ArrayLike, "..."],
        dTedpsi: Float[ArrayLike, "..."],
        gamma: Float[ArrayLike, "..."],
        Pi: Float[ArrayLike, "..."],
        qi: Float[ArrayLike, "..."],
        qe: Float[ArrayLike, "..."],
        phi: Float[ArrayLike, "..."],
        Current: Float,
        Scalars: Float[ArrayLike, "..."],
        R: Float[ArrayLike, "..."],
        VPrime: Float[ArrayLike, "..."],
    ):
        self.n = n
        self.pi = pi
        self.pe = pe
        self.L = L
        self.omega = omega
        self.Ti = Ti
        self.Te = Te
        self.M = M
        self.dndpsi = dndpsi
        self.dpidpsi = dpidpsi
        self.dpedpsi = dpedpsi
        self.dLdpsi = dLdpsi
        self.domegadpsi = domegadpsi
        self.dTidpsi = dTidpsi
        self.dTedpsi = dTedpsi
        self.gamma = gamma
        self.Pi = Pi
        self.qi = qi
        self.qe = qe
        self.phi = phi
        self.Current = Current
        self.Scalars = Scalars
        self.R = R
        self.VPrime = VPrime

    @classmethod
    def from_state(cls, state: State, x, params: MirrorPlasmaParams):

        a = params.Constants.a
        n = state.Variable[Channel.Density]
        L = state.Variable[Channel.AngularMomentum]
        pi = 2.0 / 3.0 * state.Variable[Channel.IonEnergy]
        pe = 2.0 / 3.0 * state.Variable[Channel.ElectronEnergy]

        Ti = pi / n
        Te = pe / n

        R = params.MagneticField.R_x(x) / a
        VPrime = params.MagneticField.VPrime(x)
        J = n * R**2
        omega = L / J
        M = R * omega / jnp.sqrt(Te)

        dndpsi = state.Derivative[Channel.Density] * VPrime
        dLdpsi = state.Derivative[Channel.AngularMomentum] * VPrime
        dpidpsi = 2.0 / 3.0 * state.Derivative[Channel.IonEnergy] * VPrime
        dpedpsi = 2.0 / 3.0 * state.Derivative[Channel.ElectronEnergy] * VPrime

        dTidpsi = (dpidpsi - dndpsi * Ti) / n
        dTedpsi = (dpedpsi - dndpsi * Te) / n

        dRdpsi = params.MagneticField.dRdx(x) / a * VPrime
        dJdpsi = R * R * dndpsi + 2.0 * dRdpsi * R * n
        domegadpsi = dLdpsi / J - dJdpsi * L / (J * J)

        if params.Config.useConstantVoltage:
            Current = state.Scalars[Scalar.Current]
        else:
            Current = params.Config.Current / params.Constants.I0()

        return cls(
            n=n,
            pi=pi,
            pe=pe,
            L=L,
            omega=omega,
            Ti=Ti,
            Te=Te,
            M=M,
            dndpsi=dndpsi,
            dpidpsi=dpidpsi,
            dpedpsi=dpedpsi,
            dLdpsi=dLdpsi,
            domegadpsi=domegadpsi,
            dTidpsi=dTidpsi,
            dTedpsi=dTedpsi,
            gamma=state.Flux[Channel.Density],
            Pi=state.Flux[Channel.AngularMomentum],
            qi=state.Flux[Channel.IonEnergy],
            qe=state.Flux[Channel.ElectronEnergy],
            phi=state.Aux[0],
            Current=Current,
            Scalars=state.Scalars,
            R=R,
            VPrime=VPrime,
        )

    def to_state(self):
        Variable = jnp.array([self.n, self.L, 3.0 / 2.0 * self.pi, 3.0 / 2.0 * self.pe])
        Derivative = jnp.array(
            [
                self.dndpsi,
                self.dLdpsi,
                3.0 / 2.0 * self.dpidpsi,
                3.0 / 2.0 * self.dpedpsi,
            ]
        )
        Flux = jnp.array([self.gamma, self.Pi, self.qi, self.qe])
        return State(Variable, Derivative, Flux, self.phi, self.Scalars)

    @staticmethod
    def vmap_axes():
        return MirrorPlasmaState(
            n=0,
            pi=0,
            pe=0,
            L=0,
            omega=0,
            Ti=0,
            Te=0,
            M=0,
            dndpsi=0,
            dpidpsi=0,
            dpedpsi=0,
            dLdpsi=0,
            domegadpsi=0,
            dTidpsi=0,
            dTedpsi=0,
            gamma=0,
            Pi=0,
            qi=0,
            qe=0,
            phi=0,
            Current=None,
            Scalars=None,
            R=0,
            VPrime=0,
        )
