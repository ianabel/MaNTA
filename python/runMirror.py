import jax

jax.config.update("jax_enable_x64", True)
from MirrorPlasma.MirrorPlasma import MirrorPlasma
from MirrorPlasma.PlasmaState import MirrorPlasmaConfig

solver_config = {
    "OutputFilename": "mirror",
    "High_Grid_Boundary": True,
    "Polynomial_degree": 6,
    "Grid_size": 31,
    "tau": 10.0,
    "Lower_boundary": 0.0,
    "Upper_boundary": 1.0,
    "Relative_tolerance": 1e-3,
    "MinStepSize": 1e-10,
    "delta_t": 0.001,
}

config = MirrorPlasmaConfig(
    0.1,
    0.3,
    InitialMachNumber=6.0,
    EdgeMachNumber=4.0,
    Current=1.0,
    useConstantVoltage=False,
)

MP = MirrorPlasma(config, solver_config=solver_config)
MP.run()
