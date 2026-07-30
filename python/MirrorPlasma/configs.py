from MirrorPlasma.PlasmaState import MirrorPlasmaConfig

CMFX = MirrorPlasmaConfig(
    0.05,
    0.2,
    gamma=5e5,
    gamma_d=10.0,
    gamma_h=1e4,
    InitialIonTemperatureHeight=0.2,
    InitialElectronTemperatureHeight=0.2,
    InitialMachNumber=8.0,
    EdgeMachNumber=5.0,
    Current=15.0,
    useConstantVoltage=True,
    useNeutralsModel=True,
    NeutralDensity=1e14,
    ParticleSourceHeight=100.0,
    ParticleSourceWidth=0.1,
    ParticleSourceCenter=0.1,
    PlasmaVoltage=50e3,
    ADCoefficient=1e-3,
)

Fusion = MirrorPlasmaConfig(
    0.1,
    0.7,
    gamma=1e2,
    gamma_d=1.0,
    gamma_h=100.0,
    InitialMachNumber=10.0,
    EdgeMachNumber=4.0,
    PlasmaVoltage=10e6,
    MagneticFieldStrength=3.0,
    MirrorRatio=3.0,
    useNeutralsModel=True,
)
