import equinox as eqx
import jax.numpy as jnp
from Constants import PlasmaConstants
from MagneticField import StraightMagneticField
from IonSpecies import Hydrogen

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
    ParticleSourceCenter: Float
    ParticleSourceWidth: Float
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
        EdgeDensity: Float = 0.01,
        InitialIonTemperatureHeight: Float = 0.1,
        EdgeIonTemperature: Float = 0.05,
        InitialElectronTemperatureHeight: Float = 0.1,
        EdgeElectronTemperature: Float = 0.01,
        InitialMachNumber: Float = 6.0,
        EdgeMachNumber: Float = 3.0,
        gamma: Float = 10000.0,
        gamma_d: Float = 100.0,
        gamma_h: Float = 1000.0,
        PlasmaVoltage: Float = 100.0e3,
        useConstantVoltage: Bool = True,
        ParticleSourceCenter: Float = 0.5,
        ParticleSourceWidth: Float = 0.1,
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
        self.ParticleSourceCenter = ParticleSourceCenter
        self.ParticleSourceWidth = ParticleSourceWidth
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
        B = StraightMagneticField(
            config.PlasmaLength,
            config.MagneticFieldStrength,
            config.MirrorRatio,
            config.Rmin,
            config.Rmax,
            config.MagneticFieldSlope,
        )
        H = Hydrogen()

        C = PlasmaConstants(H, B)
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


"""
Wrapper class for State to make accessing variables easier 
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
    dndx: Float[ArrayLike, "..."]
    dpidx: Float[ArrayLike, "..."]
    dpedx: Float[ArrayLike, "..."]
    dLdx: Float[ArrayLike, "..."]
    domegadx: Float[ArrayLike, "..."]
    dTidx: Float[ArrayLike, "..."]
    dTedx: Float[ArrayLike, "..."]
    gamma: Float[ArrayLike, "..."]  # Particle flux
    Pi: Float[ArrayLike, "..."]  # Viscous stress
    qi: Float[ArrayLike, "..."]  # Ion heat flux
    qe: Float[ArrayLike, "..."]  # Electron heat flux
    phi: Float[ArrayLike, "..."]  # Ambipolar potential correction
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
        M: Float[ArrayLike, "..."],
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
        self.M = M
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

        R = params.MagneticField.R_x(x)
        J = n * R**2
        omega = L / J
        M = R * omega / jnp.sqrt(Te)

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
            n=n,
            pi=pi,
            pe=pe,
            L=L,
            omega=omega,
            Ti=Ti,
            Te=Te,
            M=M,
            dndx=dndx,
            dpidx=dpidx,
            dpedx=dpedx,
            dLdx=dLdx,
            domegadx=domegadx,
            dTidx=dTidx,
            dTedx=dTedx,
            gamma=state.Flux[Channel.Density],
            Pi=state.Flux[Channel.AngularMomentum],
            qi=state.Flux[Channel.IonEnergy],
            qe=state.Flux[Channel.ElectronEnergy],
            phi=state.Aux[0],
            Scalars=state.Scalars,
        )

    def to_state(self):
        Variable = jnp.array([self.n, self.L, 3.0 / 2.0 * self.pi, 3.0 / 2.0 * self.pe])
        Derivative = jnp.array(
            [self.dndx, self.dLdx, 3.0 / 2.0 * self.dpidx, 3.0 / 2.0 * self.dpedx]
        )
        Flux = jnp.array([self.gamma, self.Pi, self.qi, self.qe])
        return State(Variable, Derivative, Flux, self.phi, self.Scalars)
