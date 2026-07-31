from MirrorPlasma.Config import MirrorPlasmaConfig
from MirrorPlasma.IonSpecies import Hydrogen, DeuteriumTritium
from python.MirrorPlasma import IonSpecies

CMFX = MirrorPlasmaConfig(
    0.1,
    0.35,
    gamma=5e3,
    gamma_d=100.0,
    gamma_h=1e5,
    InitialIonTemperatureHeight=0.2,
    InitialElectronTemperatureHeight=0.2,
    InitialMachNumber=8.0,
    EdgeMachNumber=4.0,
    Current=15.0,
    useConstantVoltage=True,
    useNeutralsModel=True,
    NeutralDensity=1e14,
    ParticleSourceHeight=50.0,
    ParticleSourceWidth=0.1,
    ParticleSourceCenter=0.2,
    MagneticFieldSlope=0.0,
    PlasmaVoltage=80e3,
    ADCoefficient=0.5,
)

CMFX1keV = MirrorPlasmaConfig(
    0.05,
    0.25,
    gamma=5e4,
    gamma_d=100.0,
    gamma_h=1e5,
    InitialIonTemperatureHeight=0.2,
    InitialElectronTemperatureHeight=0.2,
    InitialMachNumber=8.0,
    EdgeMachNumber=4.0,
    Current=15.0,
    useConstantVoltage=True,
    useNeutralsModel=True,
    NeutralDensity=1e14,
    ParticleSourceHeight=100.0,
    ParticleSourceWidth=0.1,
    ParticleSourceCenter=0.2,
    PlasmaVoltage=100e3,
    ADCoefficient=1.0,
)

Fusion = MirrorPlasmaConfig(
    0.1,
    0.7,
    gamma=1e4,
    gamma_d=10.0,
    gamma_h=1000.0,
    InitialMachNumber=5.0,
    EdgeMachNumber=4.0,
    PlasmaVoltage=10e6,
    MagneticFieldStrength=3.0,
    ParticleSourceHeight=100.0,
    ParticleSourceWidth=0.1,
    ParticleSourceCenter=0.2,
    MirrorRatio=3.0,
    useNeutralsModel=True,
    IonSpecies=DeuteriumTritium(),
)
