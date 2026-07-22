import equinox as eqx
import jax
import jax.numpy as jnp
from MirrorPlasma.Constants import PlasmaConstants
from MirrorPlasma.MagneticField import StraightMagneticField
from MirrorPlasma.IonSpecies import ElementaryCharge, Hydrogen

from jaxtyping import Float, ArrayLike, Bool
import enum
import sys


sys.path.append("..")

from State import State


class MirrorPlasmaConfig(eqx.Module):
    Rmin: Float = eqx.field(static=True)
    Rmax: Float = eqx.field(static=True)
    MagneticFieldSlope: Float
    InitialDensityHeight: Float = eqx.field(static=True)
    EdgeDensity: Float = eqx.field(static=True)
    InitialIonTemperatureHeight: Float = eqx.field(static=True)
    EdgeIonTemperature: Float = eqx.field(static=True)
    InitialElectronTemperatureHeight: Float = eqx.field(static=True)
    EdgeElectronTemperature: Float = eqx.field(static=True)
    InitialMachNumber: Float = eqx.field(static=True)
    EdgeMachNumber: Float = eqx.field(static=True)
    gamma: Float
    gamma_d: Float
    gamma_h: Float
    PlasmaVoltage: Float
    useConstantVoltage: Bool = eqx.field(static=True)
    Current: Float
    CurrentDecay: Float
    ParticleSourceCenter: Float
    ParticleSourceWidth: Float
    ParticleSourceHeight: Float
    PlasmaLength: Float
    MagneticFieldStrength: Float
    MirrorRatio: Float
    NeutralDensity: Float
    useNeutralsModel: Bool = eqx.field(static=True)

    def __init__(
        self,
        Rmin: Float,
        Rmax: Float,
        MagneticFieldSlope: Float = 0.0,
        InitialDensityHeight: Float = 0.1,
        EdgeDensity: Float = 0.05,
        InitialIonTemperatureHeight: Float = 0.1,
        EdgeIonTemperature: Float = 0.05,
        InitialElectronTemperatureHeight: Float = 0.1,
        EdgeElectronTemperature: Float = 0.05,
        InitialMachNumber: Float = 6.0,
        EdgeMachNumber: Float = 3.0,
        gamma: Float = 10000.0,
        gamma_d: Float = 100.0,
        gamma_h: Float = 1000.0,
        PlasmaVoltage: Float = 100.0e3,
        useConstantVoltage: Bool = True,
        Current: Float = 0.2,
        CurrentDecay: Float = 1e-3,
        ParticleSourceCenter: Float = 0.1,
        ParticleSourceWidth: Float = 0.1,
        ParticleSourceHeight: Float = 50.0,
        PlasmaLength: Float = 0.6,
        MagneticFieldStrength: Float = 0.34,
        MirrorRatio: Float = 10.0,
        NeutralDensity: Float = 1e13,
        useNeutralsModel: Bool = False,
    ):
        self.Rmin = Rmin
        self.Rmax = Rmax
        self.MagneticFieldSlope = MagneticFieldSlope
        self.InitialDensityHeight = InitialDensityHeight
        self.EdgeDensity = EdgeDensity
        self.InitialIonTemperatureHeight = InitialIonTemperatureHeight
        self.EdgeIonTemperature = EdgeIonTemperature
        self.InitialElectronTemperatureHeight = InitialElectronTemperatureHeight
        self.EdgeElectronTemperature = EdgeElectronTemperature
        self.InitialMachNumber = InitialMachNumber
        self.EdgeMachNumber = EdgeMachNumber
        self.gamma = gamma
        self.gamma_d = gamma_d
        self.gamma_h = gamma_h
        self.PlasmaVoltage = PlasmaVoltage
        self.useConstantVoltage = useConstantVoltage
        self.Current = Current
        self.CurrentDecay = CurrentDecay
        self.ParticleSourceCenter = ParticleSourceCenter
        self.ParticleSourceWidth = ParticleSourceWidth
        self.ParticleSourceHeight = ParticleSourceHeight
        self.PlasmaLength = PlasmaLength
        self.MagneticFieldStrength = MagneticFieldStrength
        self.MirrorRatio = MirrorRatio
        self.NeutralDensity = NeutralDensity
        self.useNeutralsModel = useNeutralsModel


class MirrorPlasmaParams(eqx.Module):
    MagneticField: StraightMagneticField
    IonSpecies: Hydrogen
    Constants: PlasmaConstants
    Config: MirrorPlasmaConfig

    def __init__(self, MagneticField, IonSpecies, Constants, Config):
        self.MagneticField = MagneticField
        self.IonSpecies = IonSpecies
        self.Constants = Constants
        self.Config = Config

    @classmethod
    def make(cls, config: MirrorPlasmaConfig):
        a = config.Rmax - config.Rmin
        B = StraightMagneticField(
            _L_z=config.PlasmaLength,
            _B_z=config.MagneticFieldStrength,
            _Rm=config.MirrorRatio,
            _Rmin=config.Rmin,
            _Rmax=config.Rmax,
            _m=config.MagneticFieldSlope,
        )
        H = Hydrogen()

        C = PlasmaConstants(H, B, _a=a)
        return cls(MagneticField=B, IonSpecies=H, Constants=C, Config=config)


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
            Current = -state.Scalars[Scalar.Current]
        else:
            Current = -params.Config.Current / params.Constants.I0()

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
