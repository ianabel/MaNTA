import jax

jax.config.update("jax_enable_x64", True)
from MirrorPlasma.MirrorPlasma import MirrorPlasma
from MirrorPlasma.PlasmaState import MirrorPlasmaConfig
import MirrorPlasma.configs as config
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
    "Polynomial_degree": 7,
    "Grid_size": nCells,
    "tau": 10000.0,
    "Lower_boundary": 0.0,
    "Upper_boundary": 1.0,
    "Relative_tolerance": 1e-3,
    "Absolute_tolerance": [1e-3],
    "MinStepSize": 1e-10,
    "delta_t": 0.01,
}

MP = MirrorPlasma(config.CMFX, solver_config=solver_config)
MP.run(1.0)
