import equinox as eqx
import jax
import jax.numpy as jnp
from MirrorPlasma.Constants import PlasmaConstants
from MirrorPlasma.MagneticField import _MagneticField, StraightMagneticField
from MirrorPlasma.IonSpecies import _IonSpecies, Hydrogen

from jaxtyping import Float, ArrayLike, Bool


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
    ADCoefficient: Float
    MagneticField: _MagneticField
    IonSpecies: _IonSpecies
    NormalizeToR: Bool

    def __init__(
        self,
        Rmin: Float,
        Rmax: Float,
        # Initial conditions
        InitialDensityHeight: Float = 0.1,
        EdgeDensity: Float = 0.05,
        InitialIonTemperatureHeight: Float = 0.1,
        EdgeIonTemperature: Float = 0.1,
        InitialElectronTemperatureHeight: Float = 0.1,
        EdgeElectronTemperature: Float = 0.1,
        InitialMachNumber: Float = 6.0,
        EdgeMachNumber: Float = 3.0,
        # Voltage controller
        gamma: Float = 10000.0,
        gamma_d: Float = 100.0,
        gamma_h: Float = 1000.0,
        PlasmaVoltage: Float = 100.0e3,
        useConstantVoltage: Bool = True,
        Current: Float = 0.2,
        CurrentDecay: Float = 1e-3,
        # Particle source
        ParticleSourceCenter: Float = 0.1,
        ParticleSourceWidth: Float = 0.1,
        ParticleSourceHeight: Float = 50.0,
        # Magnetic field
        PlasmaLength: Float = 0.6,
        MagneticFieldSlope: Float = 0.0,
        MagneticFieldStrength: Float = 0.34,
        MirrorRatio: Float = 10.0,
        # Neutrals
        NeutralDensity: Float = 1e13,
        useNeutralsModel: Bool = False,
        # Artificial diffusion
        ADCoefficient: Float = 0.0,
        # Normalizations
        NormalizeToR: Bool = True,
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
        self.ADCoefficient = ADCoefficient
        self.IonSpecies = Hydrogen()
        self.MagneticField = StraightMagneticField(
            _L_z=self.PlasmaLength,
            _B_z=self.MagneticFieldStrength,
            _Rm=self.MirrorRatio,
            _Rmin=self.Rmin,
            _Rmax=self.Rmax,
            _m=self.MagneticFieldSlope,
        )
        self.NormalizeToR = NormalizeToR
