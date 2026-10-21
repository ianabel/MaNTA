import equinox as eqx
from jaxtyping import ArrayLike, Float, Int
import jax
import jax.numpy as jnp
import numpy as np


class Integrator(eqx.Module):
    weights: Float[ArrayLike, "..."]
    phi_boundary: Float[ArrayLike, "..."]
    k: Int = eqx.field(static=True)
    nCells: Int = eqx.field(static=True)

    def __init__(self, k_, nCells_, weights_, phi_boundary_):
        self.weights = weights_
        self.phi_boundary = phi_boundary_
        self.k = k_
        self.nCells = nCells_

    def __call__(self, f):
        # jax.debug.print("weights = {val}", val=self.weights)
        return jnp.dot(f, self.weights)

    def computeCellProducts(self, f):
        return f * self.weights

    def phiL(self):
        return self.phi_boundary[:, 0]

    def phiR(self):
        return self.phi_boundary[:, 1]


def scalar_integrator(case, weights, phi_boundary):
    """The Integrator ScalarG's arguments describe.

    `k` and `nCells` are static fields the quadrature itself never reads. A case
    that sets them as attributes keeps them; otherwise they come from the data,
    phi_boundary being (k+1, 2) and weights one per node.
    """
    k = getattr(case, "k", None)
    nCells = getattr(case, "nCells", None)
    if k is None or nCells is None:
        k = int(np.shape(phi_boundary)[0]) - 1
        nCells = len(weights) // (k + 1)
    return Integrator(k, nCells, weights, phi_boundary)
