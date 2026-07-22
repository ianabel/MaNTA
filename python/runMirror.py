import jax

jax.config.update("jax_enable_x64", True)
from MirrorPlasma.MirrorPlasma import MirrorPlasma
from MirrorPlasma.PlasmaState import MirrorPlasmaConfig

solver_config = {
    "OutputFilename": "mirror",
    "High_Grid_Boundary": True,
    "Polynomial_degree": 6,
    "Grid_size": 43,
    "tau": 1000.0,
    "Lower_boundary": 0.0,
    "Upper_boundary": 1.0,
    "Relative_tolerance": 1e-3,
    "MinStepSize": 1e-12,
    "delta_t": 0.001,
}

config = MirrorPlasmaConfig(
    0.1,
    0.5,
    InitialMachNumber=6.0,
    EdgeMachNumber=2.5,
    Current=0.5,
    useConstantVoltage=False,
    useNeutralsModel=False,
    ParticleSourceHeight=5.0,
    ParticleSourceCenter=0.3,
)

MP = MirrorPlasma(config, solver_config=solver_config)
MP.run()
