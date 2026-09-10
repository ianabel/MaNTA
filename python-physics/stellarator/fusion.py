import jax.numpy as jnp
from constants import PlasmaConstants

def DTFusionRate(n, Ti, constants: PlasmaConstants):

    Ti *= constants.T0eV / 1000.0
    n *= constants.n0cgs
    # c.f. H.-S. Bosch and G.M. Hale 1992 Nucl. Fusion 32 611
    C1 = 5.65718e-12
    C2 = 3.41267e-3
    C3 = 1.99167e-3
    C4 = 0.0
    C5 = 1.05060e-5
    C6 = 0.0
    C7 = 0.0
    BG = 31.3970
    mrc2 = 937814

    theta = Ti / (
        1 - (Ti * (C2 + Ti * (C4 + Ti * C6))) / (1 + Ti * (C3 + Ti * (C5 + Ti * C7)))
    )

    xi = (BG**2 / (4 * theta)) ** (1.0 / 3.0)

    sigmav = C1 * theta * jnp.sqrt(xi / (mrc2 * Ti**3)) * jnp.exp(-3 * xi)

    return 0.25 * n**2 * sigmav * 1e6


# assuming all energy is deposited into electrons
def DTAlphaHeating(n, Ti, constants: PlasmaConstants):
    Factor = 3.5e6 * constants.T0
    return Factor * DTFusionRate(n, Ti, constants)
