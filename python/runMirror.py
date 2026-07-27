import jax

jax.config.update("jax_enable_x64", True)
from MirrorPlasma.MirrorPlasma import MirrorPlasma
from MirrorPlasma.PlasmaState import MirrorPlasmaConfig
import numpy as np

nCells = 22


def cheb_nodes(nCells):
    nodes = np.ndarray((nCells + 1,))
    for i in range(0, len(nodes)):
        nodes[nCells - i] = 0.5 * (1 + np.cos(i * np.pi / nCells))
    return nodes


solver_config = {
    "OutputFilename": "mirror",
    "Grid_points": cheb_nodes(nCells),
    "Polynomial_degree": 8,
    "Grid_size": nCells,
    "tau": 1000.0,
    "Lower_boundary": 0.0,
    "Upper_boundary": 1.0,
    "Relative_tolerance": 1e-3,
    "MinStepSize": 1e-12,
    "delta_t": 0.001,
}

config = MirrorPlasmaConfig(
    0.05,
    0.2,
    gamma=10000.0,
    gamma_d=0.01,
    gamma_h=10000.0,
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
