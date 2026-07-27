import jax

jax.config.update("jax_enable_x64", True)
from MirrorPlasma.MirrorPlasma import MirrorPlasma
from MirrorPlasma.PlasmaState import MirrorPlasmaConfig

solver_config = {
    "OutputFilename": "mirror",
    "High_Grid_Boundary": False,
    "Lower_Boundary_Fraction": 0.05,
    "Upper_Boundary_Fraction": 0.05,
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
    gamma=0.0,
    gamma_d=0.0,
    gamma_h=0.0,
    InitialMachNumber=8.0,
    EdgeMachNumber=4.5,
    Current=0.5,
    useConstantVoltage=True,
    useNeutralsModel=False,
    ParticleSourceHeight=5.0,
    ParticleSourceCenter=0.2,
)

MP = MirrorPlasma(config, solver_config=solver_config)
MP.run()
